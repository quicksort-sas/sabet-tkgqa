"""
Unified data loading for all temporal KGQA benchmarks.

The benchmark folders are produced by scripts/build_benchmarks.py (see that file for the format):

    Data/benchmarks/<name>/{kg/, questions/, embeddings/, meta.json}

    bench = TKGBenchmark("complex_cronquestions")
    train = TKGQADataset(bench, "train", tokenizer_name="distilbert-base-uncased")
    loader = DataLoader(train, batch_size=150, shuffle=True, collate_fn=train.collate)

Index spaces (identical for every benchmark):
  * entity ids      0 .. E-1          rows of the TComplEx entity table
  * timestamp ids   0 .. T-1          rows of the TComplEx time table
  * answer ids      0 .. E+T-1        entities first, then E + timestamp id (matches model score vectors)
  * padding id      E + T             zero row of `entity_time_embedding`

A batch is a dict of tensors and lists (see `TKGQADataset.collate`).
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from tkg_qa_models.paths import DATA_ROOT

BENCHMARK_ROOT = os.path.join(DATA_ROOT, "benchmarks")

# Legacy --dataset_name values -> benchmark folder
BENCHMARK_ALIASES = {
    "wikidata_big": "cronquestions",
    "wikidata_big_complex": "complex_cronquestions",
    "timequestions": "timequestions",
    "MultiTQ": "multitq",
}

LM_NAMES = {
    "distilbert": "distilbert-base-uncased",
    "distill_bert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}


def resolve_benchmark(name):
    return BENCHMARK_ALIASES.get(name, name)


class TKGBenchmark:
    """Vocabulary sizes, paths and (lazily) labels of one standardized benchmark."""

    def __init__(self, name, root=BENCHMARK_ROOT):
        self.name = resolve_benchmark(name)
        self.dir = os.path.join(root, self.name)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(
                f"{self.dir} not found; build it with `python scripts/build_benchmarks.py`"
            )
        with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.num_entities = self.meta["num_entities"]
        self.num_relations = self.meta["num_relations"]
        self.num_timestamps = self.meta["num_timestamps"]
        self.padding_idx = self.num_entities + self.num_timestamps
        self._entity_keys = self._timestamp_keys = None

    def embeddings_path(self, filename="tcomplex.ckpt"):
        return os.path.join(self.dir, "embeddings", filename)

    def split_path(self, split):
        return os.path.join(self.dir, "questions", f"{split}.jsonl")

    def load_split(self, split):
        with open(self.split_path(split), encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _read_keys(self, filename):
        keys = []
        with open(os.path.join(self.dir, "kg", filename), encoding="utf-8") as f:
            for line in f:
                keys.append(line.rstrip("\n").split("\t")[1])
        return keys

    def answer_key(self, answer_id):
        """Human-readable key (Wikidata id, MultiTQ name or timestamp) of an answer id."""
        if self._entity_keys is None:
            self._entity_keys = self._read_keys("entities.tsv")
            self._timestamp_keys = self._read_keys("timestamps.tsv")
        if answer_id < self.num_entities:
            return self._entity_keys[answer_id]
        return self._timestamp_keys[answer_id - self.num_entities]

    def check_embeddings(self, tkbc_model):
        n_ent = tkbc_model.embeddings[0].weight.shape[0]
        n_ts = tkbc_model.embeddings[2].weight.shape[0]
        if (n_ent, n_ts) != (self.num_entities, self.num_timestamps):
            raise ValueError(
                f"{self.name}: embeddings have {n_ent} entities / {n_ts} timestamps, "
                f"benchmark has {self.num_entities} / {self.num_timestamps}"
            )


class TKGQADataset(Dataset):
    """
    One dataset class for every benchmark and every model.

    Args:
        benchmark:           TKGBenchmark (or its name)
        split:               'train' | 'valid' | 'test'
        tokenizer_name:      Hugging Face name or a key of LM_NAMES; must match the model's LM
        mask_entities:       replace each entity/time mention by the tokenizer's mask token (TempoQR-style
                             models inject the TKG embedding there); otherwise keep the words and tag
                             their sub-tokens with the mention id
        sample_paraphrases:  draw a random paraphrase per access (default: only for 'train')
        drop_unanswerable:   drop questions whose gold answers are all outside the KG (default: only for
                             'train'; in evaluation they are kept and count as errors)
        max_questions:       keep only the first N questions (quick checks, training-set monitoring)
        max_length:          tokenizer truncation length
    """

    def __init__(self, benchmark, split, tokenizer_name="distilbert", mask_entities=True,
                 sample_paraphrases=None, drop_unanswerable=None, max_questions=None, max_length=128):
        self.benchmark = benchmark if isinstance(benchmark, TKGBenchmark) else TKGBenchmark(benchmark)
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(LM_NAMES.get(tokenizer_name, tokenizer_name), use_fast=True)
        self.mask_entities = mask_entities
        self.sample_paraphrases = (split == "train") if sample_paraphrases is None else sample_paraphrases
        self.max_length = max_length

        records = self.benchmark.load_split(split)
        if (split == "train") if drop_unanswerable is None else drop_unanswerable:
            records = [r for r in records if r["answers"]]
        if max_questions is not None:
            records = records[:max_questions]
        self.records = records

        self.num_entities = self.benchmark.num_entities
        self.num_timestamps = self.benchmark.num_timestamps
        self.padding_idx = self.benchmark.padding_idx

    def __len__(self):
        return len(self.records)

    # ── Encoding ────────────────────────────────────────────────────────

    def _mention_id(self, kind, idx):
        return idx if kind == "entity" else self.num_entities + idx

    def _tokenize(self, text, mentions):
        """Token ids plus, per token, the entity/time id it stands for (padding_idx if none)."""
        tok = self.tokenizer
        if self.mask_entities and mentions:
            pieces, spans, pos = [], [], 0
            for start, end, kind, idx in mentions:
                pieces.append(text[pos:start])
                s = sum(len(p) for p in pieces)
                pieces.append(tok.mask_token)
                spans.append((s, s + len(tok.mask_token), self._mention_id(kind, idx)))
                pos = end
            pieces.append(text[pos:])
            text = "".join(pieces)
        else:
            spans = [(s, e, self._mention_id(k, i)) for s, e, k, i in mentions]

        enc = tok(text, truncation=True, max_length=self.max_length, return_offsets_mapping=True)
        ent_ids = []
        for a, b in enc["offset_mapping"]:
            tag = self.padding_idx
            if b > a:
                for s, e, mid in spans:
                    if a < e and b > s:
                        tag = mid
                        break
            ent_ids.append(tag)
        return enc["input_ids"], ent_ids

    def __getitem__(self, index):
        r = self.records[index]
        if self.sample_paraphrases and len(r["paraphrases"]) > 1:
            para = random.choice(r["paraphrases"])
        else:
            para = r["paraphrases"][0]
        input_ids, ent_ids = self._tokenize(para["text"], para["mentions"])

        ents = r["entities"]
        head = ents[0] if ents else 0
        tail = ents[1] if len(ents) > 1 else head
        tail2 = ents[2] if len(ents) > 2 else tail
        has_time = bool(r["times"])
        time = self.num_entities + (r["times"][0] if has_time else 0)

        hint = r["hint"]
        has_hint = hint["t1"] is not None

        offset = self.num_entities if r["answer_type"] == "time" else 0
        answers = [offset + a for a in r["answers"]]
        if not answers:
            answer = -100                                     # ignored by cross-entropy
        elif self.split == "train":
            answer = random.choice(answers)
        else:
            answer = answers[0]

        return {
            "input_ids": input_ids,
            "entity_time_ids": ent_ids,
            "head": head, "tail": tail, "tail2": tail2,
            "num_entities_in_question": len(ents),
            "time": time, "has_time": has_time,
            "t1": hint["t1"] if has_hint else 0,
            "t2": hint["t2"] if has_hint else 0,
            "has_hint": has_hint,
            "answer": answer,
            "answers": answers,
            "answer_is_time": r["answer_type"] == "time",
            "meta": {k: r[k] for k in ("id", "qtype", "answer_type", "complexity", "time_level")},
        }

    # ── Batching ────────────────────────────────────────────────────────

    def collate(self, items):
        B = len(items)
        L = max(len(it["input_ids"]) for it in items)
        A = max(1, max(len(it["answers"]) for it in items))
        input_ids = torch.full((B, L), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        entity_time_ids = torch.full((B, L), self.padding_idx, dtype=torch.long)
        answers = torch.full((B, A), -1, dtype=torch.long)
        for i, it in enumerate(items):
            n = len(it["input_ids"])
            input_ids[i, :n] = torch.tensor(it["input_ids"])
            attention_mask[i, :n] = 1
            entity_time_ids[i, :n] = torch.tensor(it["entity_time_ids"])
            if it["answers"]:
                answers[i, :len(it["answers"])] = torch.tensor(it["answers"])

        def col(key, dtype=torch.long):
            return torch.tensor([it[key] for it in items], dtype=dtype)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_time_ids": entity_time_ids,
            # 1.0 for ordinary tokens, 0.0 where the token stands for an entity/time (TempoQR convention)
            "entity_mask": (entity_time_ids == self.padding_idx).float(),
            "head": col("head"), "tail": col("tail"), "tail2": col("tail2"),
            "num_entities_in_question": col("num_entities_in_question"),
            "time": col("time"), "has_time": col("has_time", torch.bool),
            "t1": col("t1"), "t2": col("t2"), "has_hint": col("has_hint", torch.bool),
            "answer": col("answer"),
            "answers": answers,                                # padded with -1
            "answer_is_time": col("answer_is_time", torch.bool),
            "meta": [it["meta"] for it in items],
        }


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
