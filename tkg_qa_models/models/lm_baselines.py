import torch
from torch import nn

from tkg_qa_models.kge import time_query
from tkg_qa_models.models.common import QAModel


class LMBaseline(QAModel):
    """BERT / RoBERTa baseline: [head; relation; tail; time] projected and matched against every entity and time."""

    def __init__(self, tkbc_model, args):
        super().__init__(tkbc_model, args, lm=args.model)
        D = self.tkbc_embedding_dim
        self.linear = nn.Linear(self.sentence_embedding_dim, D)
        self.final_linear = nn.Linear(4 * D, D)

    def forward(self, batch, mode="train"):
        input_ids, mask, heads, tails, times = self.inputs(batch, "input_ids", "attention_mask", "head", "tail", "time")
        relation = self.linear(self.cls_embedding(input_ids, mask))
        output = self.final_linear(torch.cat([
            self.entity_time_embedding(heads), relation,
            self.entity_time_embedding(tails), self.entity_time_embedding(times),
        ], dim=-1))
        scores = output @ self.entity_time_embedding.weight.data[:-1].T
        if mode != "train":
            return scores
        return scores, self.single_answer_loss(scores, batch)


class EmbedKGQA(QAModel):
    """EmbedKGQA (Saxena et al., 2020): ComplEx-style scoring of the head with a question relation, no time."""

    def __init__(self, tkbc_model, args):
        super().__init__(tkbc_model, args)
        D = self.tkbc_embedding_dim
        self.linear = nn.Linear(self.sentence_embedding_dim, D)
        self.linear1 = nn.Linear(D, D)
        self.bn1 = nn.BatchNorm1d(D)
        self.dropout = nn.Dropout(0.3)

    def score(self, lhs, rel):
        right = torch.cat([self.tkbc_model.entity_table, self.tkbc_model.time_table], dim=0)
        r = self.rank
        lhs, rel, right = (lhs[:, :r], lhs[:, r:]), (rel[:, :r], rel[:, r:]), (right[:, :r], right[:, r:])
        return (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].t() + (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].t()

    def forward(self, batch, mode="train"):
        input_ids, mask, heads = self.inputs(batch, "input_ids", "attention_mask", "head")
        relation = self.linear(self.cls_embedding(input_ids, mask))
        relation = self.dropout(self.bn1(self.linear1(relation)))
        scores = self.score(self.tkbc_model.embeddings[0](heads), relation)
        if mode != "train":
            return scores
        return scores, self.single_answer_loss(scores, batch)


class CronKGQA(QAModel):
    """
    CronKGQA (Saxena et al., 2021): the question is a virtual relation scored with TComplEx.
    The time slot is averaged with two question-specific times: inferred from the embeddings
    (soft, as in the original) or retrieved from the raw TKG (--supervision hard).
    """

    def __init__(self, tkbc_model, args):
        super().__init__(tkbc_model, args)
        D = self.tkbc_embedding_dim
        self.linear = nn.Linear(self.sentence_embedding_dim, D)
        self.linear1 = nn.Linear(D, D)
        self.linear2 = nn.Linear(D, D)
        self.bn1 = nn.BatchNorm1d(D)
        self.bn2 = nn.BatchNorm1d(D)
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch, mode="train"):
        input_ids, mask, heads, tails, times = self.inputs(batch, "input_ids", "attention_mask", "head", "tail", "time")
        head = self.entity_time_embedding(heads)
        tail = self.entity_time_embedding(tails)
        time = self.entity_time_embedding(times)
        relation = self.linear(self.cls_embedding(input_ids, mask))
        if self.hard:
            t1, t2 = self.hint_embeddings(batch)
        else:
            t1, t2 = time_query(head, relation, tail, self.rank), time_query(tail, relation, head, self.rank)
        time = (time + t1 + t2) / 3
        rel_time = self.dropout(self.bn1(self.linear1(relation)))
        rel_ent = self.dropout(self.bn2(self.linear2(relation)))
        scores = torch.cat([
            self.score_entities(head, rel_ent, time),
            self.score_timestamps(head, rel_time, tail),
        ], dim=1)
        if mode != "train":
            return scores
        return scores, self.single_answer_loss(scores, batch)
