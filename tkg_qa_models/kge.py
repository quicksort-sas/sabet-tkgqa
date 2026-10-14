"""
Pretrained TComplEx embeddings and the scoring functions shared by every QA model.

Complex vectors are stored as [real | imaginary] halves of size `rank` (embedding dim = 2 * rank).
TComplEx (Lacroix et al., 2020) scores a fact (s, r, o, t) as Re(<e_s, v_r, conj(e_o), w_t>).
"""

import torch
from torch import nn


class TComplExEmbeddings(nn.Module):
    """Entity, relation and timestamp tables of a pretrained TComplEx checkpoint."""

    def __init__(self, num_entities, num_relations, num_timestamps, rank):
        super().__init__()
        self.rank = rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, 2 * rank) for n in (num_entities, num_relations, num_timestamps)
        ])

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        sizes = [state[f"embeddings.{i}.weight"].shape for i in range(3)]
        model = cls(sizes[0][0], sizes[1][0], sizes[2][0], rank=sizes[0][1] // 2)
        model.load_state_dict({k: v for k, v in state.items() if k.startswith("embeddings.")})
        print(f"Loaded TComplEx from {path}: {sizes[0][0]} entities, {sizes[1][0]} relations, "
              f"{sizes[2][0]} timestamps, dim {sizes[0][1]}")
        return model

    @property
    def entity_table(self):
        return self.embeddings[0].weight

    @property
    def time_table(self):
        return self.embeddings[2].weight

    @property
    def num_entities(self):
        return self.embeddings[0].weight.shape[0]

    @property
    def num_timestamps(self):
        return self.embeddings[2].weight.shape[0]


def _split(x, rank):
    return x[..., :rank], x[..., rank:]


def score_entities(lhs, rel, time, entity_table, rank):
    """Re(<lhs, rel, conj(e_c), time>) for every entity c.   [B, D] x3 -> [B, num_entities]"""
    lhs, rel, time = _split(lhs, rank), _split(rel, rank), _split(time, rank)
    right = _split(entity_table, rank)
    full_rel = rel[0] * time[0] - rel[1] * time[1], rel[1] * time[0] + rel[0] * time[1]
    return (
        (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t()
        + (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
    )


def time_query(lhs, rel, rhs, rank):
    """The complex vector whose inner product with w_t gives Re(<lhs, rel, conj(rhs), w_t>).   -> [B, D]"""
    lhs, rel, rhs = _split(lhs, rank), _split(rel, rank), _split(rhs, rank)
    return torch.cat([
        lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] - lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1],
        lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] + lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1],
    ], dim=-1)


def score_timestamps(lhs, rel, rhs, time_table, rank):
    """Re(<lhs, rel, conj(rhs), w_c>) for every timestamp c.   [B, D] x3 -> [B, num_timestamps]"""
    q = _split(time_query(lhs, rel, rhs, rank), rank)
    time = _split(time_table, rank)
    return q[0] @ time[0].t() + q[1] @ time[1].t()
