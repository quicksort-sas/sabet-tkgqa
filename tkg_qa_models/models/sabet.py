import torch
import torch.nn.functional as F
from torch import nn

from tkg_qa_models.models.common import QAModel


class Sabet(QAModel):
    """
    SABET-QA as described in the submitted paper (kept to reproduce its results; see Sabet1 for v2).

    Slot-aware question encoding, K hops of bidirectional entity/time scoring with a working-memory
    update, and a question-conditioned hop selector. The (t1, t2) hint is added to the state at every
    hop with --supervision hard (SABET-QA-Hard) and not used otherwise (SABET-QA).
    """

    entity_aware = True

    def __init__(self, tkbc_model, args):
        super().__init__(tkbc_model, args)
        D = self.tkbc_embedding_dim
        dropout = getattr(args, "dropout", 0.15)
        self.dropout = nn.Dropout(dropout)
        self.num_hops = getattr(args, "num_hops", 4)
        nhead = next(h for h in (8, 4, 2, 1) if D % h == 0)

        self.text_proj = nn.Sequential(
            nn.Linear(self.sentence_embedding_dim, D), nn.LayerNorm(D), nn.GELU(), nn.Dropout(dropout))
        self.slot_cross_attn = nn.MultiheadAttention(D, nhead, dropout=dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(D)
        self.slot_gate = nn.Sequential(nn.Linear(2 * D, D), nn.Sigmoid())
        self.summary_proj = nn.Sequential(nn.Linear(3 * D, D), nn.LayerNorm(D), nn.GELU(), nn.Dropout(dropout))
        self.rel_head = nn.Sequential(nn.Linear(4 * D, 2 * D), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * D, D))
        self.hop_rel_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * D, 2 * D), nn.LayerNorm(2 * D), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(2 * D, D))
            for _ in range(self.num_hops)
        ])
        self.memory_proj = nn.Linear(D, D)
        self.hop_processors = nn.ModuleList([
            nn.MultiheadAttention(D, nhead, dropout=dropout, batch_first=True) for _ in range(self.num_hops - 1)])
        self.hop_norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(self.num_hops - 1)])
        self.hop_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * D, D), nn.Sigmoid()) for _ in range(self.num_hops - 1)])
        self.hop_score_gate = nn.Sequential(nn.Linear(D, self.num_hops), nn.Softmax(dim=-1))
        self.ent_lin1, self.ent_lin2 = nn.Linear(D, D), nn.Linear(D, D)
        self.ent_norm1, self.ent_norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.time_lin1, self.time_lin2 = nn.Linear(D, D), nn.Linear(D, D)
        self.time_norm1, self.time_norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.bi_gate_proj = nn.Linear(D, 1)
        self.bi_gate_proj2 = nn.Linear(D, 1)
        self.temporal_hop_gate = nn.Sequential(nn.Linear(3 * D, D), nn.Sigmoid())

    @staticmethod
    def _masked_mean(x, mask):
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    @staticmethod
    def _masked_max(x, mask):
        return x.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(dim=1)[0]

    def _contextualize_slots(self, tok, mask, h, t, time):
        ctx, _ = self.slot_cross_attn(torch.stack([h, t, time], dim=1), tok, tok,
                                      key_padding_mask=~mask.bool(), need_weights=False)
        ctx = self.slot_norm(ctx)
        out = []
        for i, slot in enumerate((h, t, time)):
            g = self.slot_gate(torch.cat([slot, ctx[:, i]], dim=-1))
            out.append(g * slot + (1 - g) * ctx[:, i])
        return out

    def forward(self, batch, mode="train"):
        input_ids, mask, heads, tails, times = self.inputs(batch, "input_ids", "attention_mask", "head", "tail", "time")
        tok = self.text_proj(self.lm_model(input_ids, attention_mask=mask).last_hidden_state)
        h, t, tm = self._contextualize_slots(tok, mask, self.entity_time_embedding(heads),
                                             self.entity_time_embedding(tails), self.entity_time_embedding(times))
        summary = self.summary_proj(torch.cat([tok[:, 0], self._masked_mean(tok, mask), self._masked_max(tok, mask)], dim=-1))
        state = self.rel_head(torch.cat([summary, h, t, tm], dim=-1))
        hint = self.hint_embeddings(batch) if self.hard else None

        hop_scores_entity, hop_scores_time = [], []
        for hop in range(self.num_hops):
            if hint is not None:
                tau = self.temporal_hop_gate(torch.cat([state, hint[0], hint[1]], dim=-1))
                state = state + tau * (hint[0] + hint[1])
            r_hop = self.hop_rel_proj[hop](torch.cat([state, summary], dim=-1))
            r_ent = self.dropout(self.ent_norm2(self.ent_lin2(self.dropout(self.ent_norm1(self.ent_lin1(r_hop))))))
            r_time = self.dropout(self.time_norm2(self.time_lin2(self.dropout(self.time_norm1(self.time_lin1(r_hop))))))

            g_ent = torch.sigmoid(self.bi_gate_proj(summary))
            sc_ent = g_ent * self.score_entities(h, r_ent, tm) + (1 - g_ent) * self.score_entities(t, r_ent, tm)
            g_time = torch.sigmoid(self.bi_gate_proj2(summary))
            sc_time = g_time * self.score_timestamps(h, r_time, t) + (1 - g_time) * self.score_timestamps(t, r_time, h)
            hop_scores_entity.append(sc_ent)
            hop_scores_time.append(sc_time)

            if hop < self.num_hops - 1:
                memory = self.memory_proj(F.softmax(sc_ent, dim=1) @ self.tkbc_model.entity_table
                                          + F.softmax(sc_time, dim=1) @ self.tkbc_model.time_table)
                mem_out, _ = self.hop_processors[hop](state.unsqueeze(1), memory.unsqueeze(1), memory.unsqueeze(1),
                                                      need_weights=False)
                mem_out = self.hop_norms[hop](mem_out.squeeze(1))
                gate = self.hop_gates[hop](torch.cat([state, mem_out], dim=-1))
                state = gate * state + (1 - gate) * mem_out

        hop_weights = self.hop_score_gate(summary)
        scores = torch.cat([
            sum(hop_weights[:, i:i + 1] * hop_scores_entity[i] for i in range(self.num_hops)),
            sum(hop_weights[:, i:i + 1] * hop_scores_time[i] for i in range(self.num_hops)),
        ], dim=1)
        if mode != "train":
            return scores, hop_weights
        return scores, self.single_answer_loss(scores, batch)
