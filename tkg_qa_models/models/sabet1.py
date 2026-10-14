import torch
import torch.nn.functional as F
from torch import nn

from tkg_qa_models.models.common import QAModel


class Sabet1(QAModel):
    """
    SABET-QA v2.

    Changes with respect to Sabet (the submitted model):
      * One code path for both variants, selected with --supervision:
          'hard' -> temporal hint (t1, t2) read from the raw TKG (SABET-QA-Hard)
          other  -> temporal hint predicted from the TKG embeddings (SABET-QA).
        In the standard variant (t1, t2) are never read at inference; they are only
        used as an optional auxiliary target during training (--interval_aux, 0 disables it).
      * Entity-aware question encoding: entity spans are [MASK]ed by the dataset, so the
        TKG embeddings of the mentions are written back at those positions (as in TempoQR).
      * Slot queries carry a learned role embedding (head / tail / time).
      * The hint is injected once into the initial state instead of being re-added at every hop,
        and is exposed to every hop as two memory slots.
      * One shared hop network conditioned on a hop embedding (instead of K separate MLPs).
      * Working memory over the top-k entity/time candidates of the previous hop, read with
        attention biased by their log-probabilities, followed by a GRU state update.
        (The previous single-key attention reduced to a linear map.)
      * Bidirectional gates conditioned on the hop relation, so the direction can change across hops.
      * Hop aggregation conditioned on each hop state, not only on the question summary.
      * Answer-type head (entity vs. time) mixed into the final scores.
      * Deep supervision on every hop; the loss uses every gold answer, not one sampled answer.
      * Ablations are command-line flags instead of commented code.
    """

    entity_aware = True

    def __init__(self, tkbc_model, args):
        super().__init__(tkbc_model, args)
        D = self.tkbc_embedding_dim
        dropout = getattr(args, "dropout", 0.15)
        self.dropout = nn.Dropout(dropout)
        self.num_hops = getattr(args, "num_hops", 4)

        # Loss weights and ablation switches
        self.hop_loss_w = getattr(args, "hop_loss", 0.3)
        self.type_loss_w = getattr(args, "type_loss", 0.5)
        self.interval_aux_w = getattr(args, "interval_aux", 0.5)
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)
        self.memory_topk = getattr(args, "memory_topk", 16)
        self.memory_temp = getattr(args, "memory_temp", 1.0)
        self.use_injection = not getattr(args, "no_injection", False)
        self.use_slot_ctx = not getattr(args, "no_slot_ctx", False)
        self.use_bidir = not getattr(args, "no_bidir", False)
        self.use_type_head = not getattr(args, "no_type_head", False)

        nhead = next(h for h in (8, 4, 2, 1) if D % h == 0)
        lm_dim = self.sentence_embedding_dim

        # ── 1. Entity-aware question encoding ─────────────────────────────
        self.text_proj = nn.Sequential(
            nn.Linear(lm_dim, D), nn.LayerNorm(D), nn.GELU(), nn.Dropout(dropout),
        )
        self.inject_proj = nn.Linear(D, D)

        # ── 2. Slot contextualization ─────────────────────────────────────
        self.slot_role = nn.Parameter(torch.zeros(3, D))
        self.no_time = nn.Parameter(torch.zeros(D))     # time slot of questions without a time mention
        self.slot_cross_attn = nn.MultiheadAttention(D, nhead, dropout=dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(D)
        self.slot_gate = nn.Sequential(nn.Linear(2 * D, D), nn.Sigmoid())

        # ── 3. Question summary and initial state ─────────────────────────
        self.summary_proj = nn.Sequential(
            nn.Linear(3 * D, D), nn.LayerNorm(D), nn.GELU(), nn.Dropout(dropout),
        )
        self.state_init = nn.Sequential(
            nn.Linear(4 * D, 2 * D), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * D, D),
        )

        # ── 4. Temporal hint (hard: read from TKG, soft: predicted) ───────
        self.interval_rel = nn.Linear(2 * D, 2 * D)       # [z0; s] -> r_start, r_end
        self.hint_gate = nn.Sequential(nn.Linear(3 * D, D), nn.Sigmoid())

        # ── 5. Shared hop network ─────────────────────────────────────────
        self.hop_emb = nn.Embedding(self.num_hops, D)
        self.hop_rel = nn.Sequential(
            nn.Linear(3 * D, 2 * D), nn.LayerNorm(2 * D), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * D, D),
        )
        self.ent_head = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))
        self.time_head = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))
        self.dir_gate = nn.Linear(2 * D, 2)               # [s; r] -> (alpha_ent, alpha_time)

        # ── 6. Working memory ─────────────────────────────────────────────
        self.slot_kind = nn.Embedding(3, D)               # 0 entity cand., 1 time cand., 2 hint
        self.mem_q = nn.Linear(D, D)
        self.mem_k = nn.Linear(D, D)
        self.mem_v = nn.Linear(D, D)
        self.mem_update = nn.GRUCell(D, D)

        # ── 7. Hop aggregation and answer type ────────────────────────────
        self.hop_score = nn.Sequential(nn.Linear(2 * D, D), nn.Tanh(), nn.Linear(D, 1))
        self.type_head = nn.Linear(D, 1)                  # logit of P(answer is a timestamp)

        self.loss_dict = {}

    # ── Helpers ─────────────────────────────────────────────────────────

    def _masked_mean(self, x, mask):
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def _masked_max(self, x, mask):
        m = mask.unsqueeze(-1).float()
        return x.masked_fill(m == 0, -1e9).max(dim=1)[0]

    def _encode_question(self, question_tokenized, attention_mask, entities_times, entity_mask):
        lm_out = self.lm_model(question_tokenized, attention_mask=attention_mask)[0]
        tok = self.text_proj(lm_out)                                  # [B, L, D]
        if self.use_injection:
            # entity_mask is 1 for ordinary tokens and 0 at entity/time positions
            keep = entity_mask.unsqueeze(-1)
            injected = self.inject_proj(self.entity_time_embedding(entities_times))
            tok = tok * keep + injected * (1 - keep)
        return tok

    def _contextualize_slots(self, tok, mask, h, t, time):
        slots = torch.stack([h, t, time], dim=1)                      # [B, 3, D]
        if not self.use_slot_ctx:
            return h, t, time
        ctx, _ = self.slot_cross_attn(
            slots + self.slot_role, tok, tok,
            key_padding_mask=~mask.bool(), need_weights=False,
        )
        ctx = self.slot_norm(ctx)
        g = self.slot_gate(torch.cat([slots, ctx], dim=-1))
        fused = g * slots + (1 - g) * ctx
        return fused[:, 0], fused[:, 1], fused[:, 2]

    def _build_summary(self, tok, mask):
        return self.summary_proj(torch.cat(
            [tok[:, 0], self._masked_mean(tok, mask), self._masked_max(tok, mask)], dim=-1
        ))

    def _candidate_slots(self, scores, table, kind):
        """Top-k candidates of a score vector as memory slots with their log-probabilities."""
        k = min(self.memory_topk, scores.shape[1])
        top_val, top_idx = (scores / self.memory_temp).topk(k, dim=1)
        log_p = F.log_softmax(top_val, dim=1)                         # renormalized over top-k
        slots = table[top_idx] + self.slot_kind.weight[kind]          # [B, k, D]
        return slots, log_p

    def _read_memory(self, state, slots, bias):
        q = self.mem_q(state).unsqueeze(1)                            # [B, 1, D]
        att = (q * self.mem_k(slots)).sum(-1) / slots.shape[-1] ** 0.5 + bias
        att = self.dropout(F.softmax(att, dim=1))
        return (att.unsqueeze(-1) * self.mem_v(slots)).sum(1)         # [B, D]

    def _mix_answer_type(self, sc_ent, sc_time, type_logit):
        if not self.use_type_head:
            return torch.cat([sc_ent, sc_time], dim=1)
        return torch.cat([
            sc_ent + F.logsigmoid(-type_logit),
            sc_time + F.logsigmoid(type_logit),
        ], dim=1)

    def _answer_loss(self, scores, answers):
        """Mean negative log-likelihood of all gold answers; answers: [B, A] padded with -1."""
        valid = answers >= 0
        rows = valid.any(1)                                           # unanswerable rows are skipped
        if not rows.any():
            return scores.sum() * 0
        logp = F.log_softmax(scores[rows], dim=1)
        ans, valid = answers[rows], valid[rows]
        nll = -(logp.gather(1, ans.clamp(min=0)) * valid).sum(1) / valid.sum(1)
        if self.label_smoothing > 0:
            nll = (1 - self.label_smoothing) * nll - self.label_smoothing * logp.mean(1)
        return nll.mean()

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, a, mode="train"):
        device = next(self.parameters()).device

        question_tokenized      = a["input_ids"].to(device)
        question_attention_mask = a["attention_mask"].to(device)
        entities_times_padded   = a["entity_time_ids"].to(device)
        entity_mask_padded      = a["entity_mask"].to(device)
        heads                   = a["head"].to(device)
        tails                   = a["tail"].to(device)
        times                   = a["time"].to(device)
        has_time                = a["has_time"].to(device).unsqueeze(-1)
        t1                      = a["t1"].to(device)
        t2                      = a["t2"].to(device)
        has_hint                = a["has_hint"].to(device)
        answers = a["answers"].to(device) if mode == "train" else None

        E_ent = self.tkbc_model.entity_table
        E_time = self.tkbc_model.time_table

        # ── Encoding ──
        tok = self._encode_question(
            question_tokenized, question_attention_mask, entities_times_padded, entity_mask_padded
        )
        h, t, tm = self._contextualize_slots(
            tok, question_attention_mask,
            self.entity_time_embedding(heads),
            self.entity_time_embedding(tails),
            torch.where(has_time, self.entity_time_embedding(times), self.no_time.expand(times.shape[0], -1)),
        )
        summary = self._build_summary(tok, question_attention_mask)
        state = self.state_init(torch.cat([summary, h, t, tm], dim=-1))
        type_logit = self.type_head(summary)                          # [B, 1]

        # ── Temporal hint ──
        interval_logits = None
        if self.hard:
            e_t1, e_t2 = E_time[t1], E_time[t2]
            hint_avail = has_hint.float().unsqueeze(-1)              # no fact retrieved -> no hint
        else:
            r_start, r_end = self.interval_rel(torch.cat([state, summary], dim=-1)).chunk(2, dim=-1)
            logit_t1 = self.score_timestamps(h, r_start, t)
            logit_t2 = self.score_timestamps(t, r_end, h)
            e_t1 = F.softmax(logit_t1, dim=1) @ E_time
            e_t2 = F.softmax(logit_t2, dim=1) @ E_time
            hint_avail = torch.ones_like(state[:, :1])
            interval_logits = (logit_t1, logit_t2)

        lam = self.hint_gate(torch.cat([state, e_t1, e_t2], dim=-1)) * hint_avail
        state = state + lam * (e_t1 + e_t2)
        hint_slots = torch.stack([e_t1, e_t2], dim=1) + self.slot_kind.weight[2]
        hint_bias = torch.log(hint_avail.clamp(min=1e-6)).expand(-1, 2)

        # ── Iterative reasoning ──
        hop_scores, hop_logits = [], []
        for hop in range(self.num_hops):
            hop_vec = self.hop_emb.weight[hop].expand_as(summary)
            r_hop = self.hop_rel(torch.cat([state, summary, hop_vec], dim=-1))
            r_ent = self.dropout(self.ent_head(r_hop))
            r_time = self.dropout(self.time_head(r_hop))

            if self.use_bidir:
                alpha = torch.sigmoid(self.dir_gate(torch.cat([summary, r_hop], dim=-1)))
                sc_ent = alpha[:, :1] * self.score_entities(h, r_ent, tm) \
                    + (1 - alpha[:, :1]) * self.score_entities(t, r_ent, tm)
                sc_time = alpha[:, 1:] * self.score_timestamps(h, r_time, t) \
                    + (1 - alpha[:, 1:]) * self.score_timestamps(t, r_time, h)
            else:
                sc_ent = self.score_entities(h, r_ent, tm)
                sc_time = self.score_timestamps(h, r_time, t)

            hop_scores.append(self._mix_answer_type(sc_ent, sc_time, type_logit))
            hop_logits.append(self.hop_score(torch.cat([summary, state], dim=-1)))

            # Working-memory update for the next hop
            if hop < self.num_hops - 1:
                ent_slots, ent_bias = self._candidate_slots(sc_ent, E_ent, 0)
                time_slots, time_bias = self._candidate_slots(sc_time, E_time, 1)
                slots = torch.cat([ent_slots, time_slots, hint_slots], dim=1)
                bias = torch.cat([ent_bias, time_bias, hint_bias], dim=1)
                state = self.mem_update(self._read_memory(state, slots, bias), state)

        # ── Hop aggregation ──
        hop_weights = F.softmax(torch.cat(hop_logits, dim=-1), dim=-1)    # [B, K]
        scores = sum(hop_weights[:, i:i + 1] * hop_scores[i] for i in range(self.num_hops))

        if mode != "train":
            return scores, hop_weights

        # ── Losses ──
        loss_main = self._answer_loss(scores, answers)
        loss_hops = sum(self._answer_loss(s, answers) for s in hop_scores) / self.num_hops
        loss = loss_main + self.hop_loss_w * loss_hops
        self.loss_dict = {"main": loss_main.detach(), "hops": loss_hops.detach()}

        if self.use_type_head:
            is_time = a["answer_is_time"].to(device).float().unsqueeze(-1)
            loss_type = F.binary_cross_entropy_with_logits(type_logit, is_time)
            loss = loss + self.type_loss_w * loss_type
            self.loss_dict["type"] = loss_type.detach()

        if interval_logits is not None and self.interval_aux_w > 0:
            # Training-only target: (t1, t2) are never read by the standard variant at inference
            known = has_hint
            if known.any():
                loss_int = 0.5 * (
                    F.cross_entropy(interval_logits[0][known], t1[known])
                    + F.cross_entropy(interval_logits[1][known], t2[known])
                )
                loss = loss + self.interval_aux_w * loss_int
                self.loss_dict["interval"] = loss_int.detach()

        self.loss_dict["total"] = loss.detach()
        return scores, loss
