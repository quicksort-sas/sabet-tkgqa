import torch
from torch import nn

from tkg_qa_models.kge import time_query
from tkg_qa_models.models.common import QAModel


class TempoQR(QAModel):
    """
    TempoQR (Mavromatis et al., AAAI 2022).

    Mention tokens are replaced by their TKG embeddings plus two time embeddings (t1, t2), the
    sequence is fused by a transformer, and the [CLS] output is scored with TComplEx in both
    head-tail orders (max). (t1, t2) are retrieved from the raw TKG with --supervision hard
    (TempoQR-Hard) or inferred from the embeddings otherwise (TempoQR-Soft).

    time_sensitivity=True adds the time-sensitivity layer of SubGTR (Chen et al., 2022): timestamp
    scores are the max of the TComplEx score and a dot product with the time embeddings. This is
    registered as `subgtr`, but it is NOT the full SubGTR (no subgraph reasoning).
    """

    entity_aware = True
    max_seq_length = 100

    def __init__(self, tkbc_model, args, time_sensitivity=False):
        super().__init__(tkbc_model, args)
        D = self.tkbc_embedding_dim
        self.time_sensitivity = time_sensitivity or getattr(args, "time_sensitivity", False)
        self.fuse = getattr(args, "fuse", "add")

        encoder_layer = nn.TransformerEncoderLayer(d_model=D, nhead=8, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(D),
                                                         enable_nested_tensor=False)
        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, D)
        self.project_entity = nn.Linear(D, D)
        self.position_embedding = nn.Embedding(self.max_seq_length, D)
        self.layer_norm = nn.LayerNorm(D)
        self.linearT = nn.Linear(self.sentence_embedding_dim, D)
        self.lin_cat = nn.Linear(3 * D, D)
        self.linear1 = nn.Linear(D, D)
        self.linear2 = nn.Linear(D, D)
        self.bn1 = nn.BatchNorm1d(D)
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch, mode="train"):
        input_ids, mask, ent_time_ids, ent_mask, heads, tails, times = self.inputs(
            batch, "input_ids", "attention_mask", "entity_time_ids", "entity_mask", "head", "tail", "time")
        head = self.entity_time_embedding(heads)
        tail = self.entity_time_embedding(tails)
        time = self.entity_time_embedding(times)

        hidden = self.lm_model(input_ids, attention_mask=mask).last_hidden_state
        if self.hard:
            t1, t2 = self.hint_embeddings(batch)
        else:
            cls = self.linearT(hidden[:, 0])
            t1, t2 = time_query(head, cls, tail, self.rank), time_query(tail, cls, head, self.rank)

        # entity-aware step: mention positions take their (time-aware) TKG embedding
        question = self.project_sentence_to_transformer_dim(hidden)
        keep = ent_mask.unsqueeze(-1)                                    # 1 = ordinary token
        mentions = self.project_entity(self.entity_time_embedding(ent_time_ids))
        t1, t2 = t1.unsqueeze(1).expand_as(mentions), t2.unsqueeze(1).expand_as(mentions)
        if self.fuse == "cat":
            mentions = self.lin_cat(torch.cat([mentions, t1, t2], dim=-1))
        else:
            mentions = mentions + t1 + t2
        combined = question * keep + mentions * (1 - keep)

        positions = torch.arange(combined.shape[1], device=self.device)
        combined = self.layer_norm(combined + self.position_embedding(positions).unsqueeze(0))
        output = self.transformer_encoder(combined.transpose(0, 1), src_key_padding_mask=~mask.bool())

        relation = output[0]                                             # [CLS]
        rel_time = self.dropout(self.bn1(self.linear1(relation)))
        rel_ent = self.dropout(self.bn1(self.linear2(relation)))       # bn1 twice, as in the original code

        scores_time = self.score_timestamps(head, rel_time, tail)
        if self.time_sensitivity:
            scores_time = torch.maximum(scores_time, rel_time @ self.tkbc_model.time_table.data.T)
        scores_entity = torch.maximum(self.score_entities(head, rel_ent, time),
                                      self.score_entities(tail, rel_ent, time))
        scores = torch.cat([scores_entity, scores_time], dim=1)
        if mode != "train":
            return scores
        return scores, self.single_answer_loss(scores, batch)
