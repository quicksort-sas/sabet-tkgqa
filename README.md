# Temporal Knowledge Graph Question Answering

This project combines pretrained temporal knowledge graph embeddings with transformer-based language representations to answer questions whose answers may be either **entities** or **timestamps**.

The code includes standard KGQA baselines, temporal reasoning models, entity-aware question encoding, hard/soft temporal supervision, multi-hop reasoning, training, checkpointing, and detailed Hits@K evaluation.

> **Research-code status:** this repository assumes a specific preprocessed data layout and pretrained TKBC checkpoint.

## Available models

| `--model` | Class | Description |
|---|---|---|
| `bert`, `roberta` | `LMBaseline` | Language-model baseline: [head; question; tail; time] matched against every entity and time. |
| `embedkgqa` | `EmbedKGQA` | EmbedKGQA-style ComplEx scoring of the head with a question relation (no time). |
| `cronkgqa` | `CronKGQA` | CronKGQA: the question as a TComplEx virtual relation. |
| `tempoqr` | `TempoQR` | TempoQR: entity- and time-aware transformer fusion. |
| `subgtr` | `TempoQR(time_sensitivity=True)` | TempoQR with SubGTR's time-sensitivity layer. Not the full SubGTR (no subgraph reasoning). |
| `sabet` | `Sabet` | SABET-QA as submitted (kept to reproduce the paper). |
| `sabet1` | `Sabet1` | SABET v2: predicted or retrieved temporal hint, top-k working memory, answer-type head, per-hop losses. |

Every model has a standard and a `-Hard` variant, selected with `--supervision soft|hard`. With `hard`, the model receives the `(t1, t2)` hint retrieved from the raw TKG; with `soft`, it only uses the embeddings (TempoQR/CronKGQA infer the times, SABET v2 predicts them, SABET does without).

## Repository structure

```text
.
├── requirements.txt
├── scripts/
│   ├── build_benchmarks.py      # raw data -> standardized benchmarks (run once)
│   ├── validate_benchmarks.py   # cross-checks the standardized data against the raw files
│   └── export_kg_splits.py      # disjoint KG train/valid/test splits for embedding training
└── tkg_qa_models/
    ├── train.py                 # command-line entry point (train / eval)
    ├── training.py              # training loop, evaluation, device selection, checkpoints
    ├── data.py                  # TKGBenchmark and TKGQADataset: one dataset class for every benchmark
    ├── kge.py                   # pretrained TComplEx tables and the shared scoring functions
    ├── kge_train.py             # TComplEx training and filtered link-prediction evaluation
    ├── paths.py                 # data root (SABET_DATA_DIR)
    └── models/
        ├── common.py            # QAModel base class (language model, embedding tables, losses)
        ├── lm_baselines.py      # LMBaseline, EmbedKGQA, CronKGQA
        ├── tempoqr.py           # TempoQR (+ SubGTR time sensitivity)
        ├── sabet.py             # SABET-QA (submitted version)
        └── sabet1.py            # SABET v2
```

## Requirements

Python 3.9+ and PyTorch 2.1+. Training runs on an NVIDIA GPU (CUDA), an Apple-silicon GPU (MPS) or the CPU; `--device auto` picks the first available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The data root is defined in `tkg_qa_models/paths.py`. It defaults to `Data/` at the repository root and can be overridden with `export SABET_DATA_DIR=/path/to/Data`.

1. Download `Data 2.zip` (about 2.7 GB) from https://osf.io/h39ma/overview?view_only=3f09fa70ccc04929923898a972830652, unzip it, and move its `data/` and `models/` folders into `Data/raw/` (the build only reads the files listed in `scripts/build_benchmarks.py` and `scripts/export_kg_splits.py`).
2. Build the standardized benchmarks, then check them:

```bash
python scripts/build_benchmarks.py --raw Data/raw --out Data/benchmarks
python scripts/validate_benchmarks.py --raw Data/raw --out Data/benchmarks
```

```text
Data/
├── raw/                                   # original release, only read by the build script
├── benchmarks/
│   └── <benchmark>/                       # cronquestions | complex_cronquestions | timequestions | multitq
│       ├── kg/
│       │   ├── entities.tsv               # id  key  label     (id = row of the entity embedding table)
│       │   ├── relations.tsv              # id  key  label
│       │   ├── timestamps.tsv             # id  key            (year, or ISO date/month/year for MultiTQ)
│       │   └── facts.tsv                  # head  rel  tail  start  end   (ids)
│       ├── questions/{train,valid,test}.jsonl
│       ├── embeddings/tcomplex.ckpt       # TComplEx checkpoints aligned with kg/
│       └── meta.json                      # sizes, checks and statistics of the build
└── qa_models/<benchmark>/<run>.ckpt       # checkpoints saved during training
```

Every question record has the same fields for the four benchmarks (`id`, `text`, `paraphrases` with mention spans, `entities`, `times`, `roles`, `answer_type`, `answers`, `qtype`, `complexity`, `time_level`, `hint`); all ids index the embedding tables directly. The full schema is at the top of `scripts/build_benchmarks.py`.

**Temporal hint `(t1, t2)`** (used by the `-Hard` variants). Following TempoQR-Hard, it is the earliest and latest timestamp of the TKG facts involving the question entities: facts linking two question entities when there are any (either direction), otherwise all facts of each entity; explicit question times are used directly. It is computed once and deterministically; placeholder timestamps (unknown start/end) are ignored, and only day-level facts are used for MultiTQ. It is read from the raw TKG, so `-Hard` models are not embedding-only. On CronQuestions, the gold answer of almost every time question is equal to `t1` or `t2` (see `hint_stats` in `meta.json`).

**Known data limitations.** MultiTQ has no gold entity annotation; its entities come from Flair NER + linking (preprocessed in the release) and some links are wrong. TimeQuestions has questions whose gold answers are not in the KG (48 train, 229 valid, 270 test); they are skipped in training and counted as errors in evaluation.

## Temporal KG embeddings

Each benchmark ships the released TComplEx checkpoint (`embeddings/tcomplex.ckpt`). New embeddings can be trained with the procedure of Lacroix et al. (ICLR 2020, `tkbc`): reciprocal relations, full cross-entropy over entities plus a time-prediction loss, N3 and Λ3 (temporal smoothness) regularisation, Adagrad (lr 0.1, batch 1000), a timestamp sampled uniformly in each fact's interval, and filtered MRR/Hits@k of object prediction for model selection.

```bash
python scripts/export_kg_splits.py
python -m tkg_qa_models.kge_train --dataset_name cronquestions --name tcomplex_v2
python -m tkg_qa_models.kge_train --dataset_name cronquestions --evaluate tcomplex.ckpt
```

`export_kg_splits.py` writes `kg/{train,valid,test}.tsv` from the data the released checkpoints were trained on, deduplicated and made disjoint (in the raw CronQuestions KG every validation fact is also a training fact, so the released checkpoint scores 0.999 MRR on it). The new checkpoint is saved as `embeddings/<name>.ckpt` with `<name>.json` (configuration, per-epoch history, valid/test metrics); CronQuestions and Complex-CronQuestions share it. Use it for QA with `--tkbc_model_file <name>.ckpt`. `--train_on all` trains final embeddings on every split for a fixed number of epochs.

## Training

```bash
python -m tkg_qa_models.train --model sabet1 --dataset_name complex_cronquestions --supervision soft --save_to sabet1_std
```

```bash
python -m tkg_qa_models.train --model sabet1 --dataset_name complex_cronquestions --supervision hard --save_to sabet1_hard
```

The script loads the benchmark and its TComplEx checkpoint, checks that their sizes agree, trains with Adam (linear warm-up, cosine decay), evaluates on the validation set and on 1,000 training questions every `--valid_freq` epochs (optional early stopping with `--patience`), keeps the best validation checkpoint and finally evaluates it on the test set. Training samples a random paraphrase of each question (disable with `--no_paraphrases`). Use `--max_train_questions` / `--max_eval_questions` for quick runs.

Outputs: `results/<benchmark>/<save_to>.log` (per-epoch metrics), `results/<benchmark>/<save_to>_test.json` (final test metrics) and `Data/qa_models/<benchmark>/<save_to>.ckpt` (best checkpoint).

## Evaluation

```bash
python -m tkg_qa_models.train --mode eval --model sabet1 --dataset_name complex_cronquestions --load_from sabet1_std --eval_split test
```

The evaluator reports:

- Loss.
- Hits@1 and Hits@10.
- Accuracy by individual question type.
- Accuracy for simple versus complex questions.
- Accuracy for entity-answer versus time-answer questions.

A prediction is counted as correct when at least one ground-truth answer appears in the top-*k* predictions.

## Experimental results

The following tables report test-set performance using Hits@1 and Hits@10. Results are grouped by overall performance, question complexity, and answer type where those annotations are available.

### CronQuestions

| Model | H@1 Overall | H@1 Complex | H@1 Simple | H@1 Entity | H@1 Time | H@10 Overall | H@10 Complex | H@10 Simple | H@10 Entity | H@10 Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT | 0.257 | 0.253 | 0.262 | 0.292 | 0.191 | 0.642 | 0.617 | 0.676 | 0.650 | 0.627 |
| CronKGQA | 0.646 | 0.391 | 0.987 | 0.698 | 0.550 | 0.886 | 0.806 | 0.993 | 0.901 | 0.858 |
| EmbedKGQA | 0.433 | 0.370 | 0.516 | 0.576 | 0.166 | 0.787 | 0.735 | 0.857 | 0.892 | 0.592 |
| TempoQR | 0.796 | 0.658 | 0.981 | 0.880 | 0.640 | 0.959 | 0.934 | 0.992 | 0.975 | 0.930 |
| TempoQR-Hard | 0.914 | 0.861 | 0.984 | 0.923 | 0.896 | 0.979 | 0.968 | 0.993 | 0.982 | 0.973 |
| SubGTR-Hard | 0.913 | 0.859 | 0.984 | 0.917 | 0.904 | 0.980 | 0.970 | 0.993 | 0.982 | 0.975 |
| **SABET-QA** | **0.843** | **0.733** | **0.989** | **0.882** | **0.770** | **0.969** | **0.953** | **0.994** | **0.979** | **0.954** |
| **SABET-QA-Hard** | **0.954** | **0.926** | **0.994** | **0.941** | **0.980** | **0.989** | **0.983** | **0.996** | **0.986** | **0.994** |

*Comparison on the CronQuestions test set. Metrics are reported for overall performance, question complexity, and answer type.*

### Complex-CronQuestions

| Model | H@1 Overall | H@1 Entity | H@1 Time | H@10 Overall | H@10 Entity | H@10 Time |
|---|---:|---:|---:|---:|---:|---:|
| BERT | 0.087 | 0.097 | 0.068 | 0.421 | 0.351 | 0.567 |
| CronKGQA | 0.288 | 0.365 | 0.129 | 0.736 | 0.758 | 0.689 |
| EmbedKGQA | 0.260 | 0.361 | 0.050 | 0.618 | 0.742 | 0.360 |
| TempoQR | 0.438 | 0.585 | 0.132 | 0.853 | 0.906 | 0.743 |
| TempoQR-Hard | 0.632 | 0.721 | 0.448 | 0.933 | 0.942 | 0.914 |
| SubGTR-Hard | 0.623 | 0.719 | 0.422 | 0.928 | 0.944 | 0.895 |
| SABET-QA | 0.524 | 0.590 | 0.384 | 0.896 | 0.925 | 0.836 |
| **SABET-QA-Hard** | **0.807** | **0.747** | **0.931** | **0.962** | **0.955** | **0.978** |

*Comparison on the Complex-CronQuestions test set. Metrics are reported for overall performance and answer type.*

### MultiTQ and TimeQuestions

| Model | MultiTQ H@1 Overall | MultiTQ H@1 Entity | MultiTQ H@1 Time | MultiTQ H@10 Overall | MultiTQ H@10 Entity | MultiTQ H@10 Time | TimeQuestions H@1 Overall | TimeQuestions H@1 Entity | TimeQuestions H@1 Time | TimeQuestions H@10 Overall | TimeQuestions H@10 Entity | TimeQuestions H@10 Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT | 0.103 | 0.117 | 0.069 | 0.516 | 0.632 | 0.234 | 0.450 | 0.409 | 0.556 | 0.569 | 0.520 | 0.698 |
| CronKGQA | 0.278 | 0.387 | 0.011 | 0.527 | 0.724 | 0.046 | 0.326 | 0.296 | 0.405 | 0.454 | 0.409 | 0.569 |
| EmbedKGQA | 0.243 | 0.342 | 0.002 | 0.489 | 0.685 | 0.012 | 0.288 | 0.266 | 0.344 | 0.468 | 0.426 | 0.577 |
| TempoQR | 0.327 | 0.456 | 0.014 | 0.571 | 0.783 | 0.055 | 0.409 | 0.406 | 0.416 | 0.530 | 0.513 | 0.574 |
| TempoQR-Hard | 0.335 | 0.465 | 0.018 | 0.579 | 0.788 | 0.068 | 0.410 | 0.411 | 0.407 | 0.528 | 0.511 | 0.572 |
| SubGTR-Hard | 0.337 | 0.469 | 0.015 | 0.576 | 0.789 | 0.056 | 0.419 | 0.415 | 0.427 | 0.532 | 0.512 | 0.583 |
| SABET-QA | 0.373 | **0.480** | 0.111 | 0.700 | 0.804 | 0.444 | 0.502 | 0.500 | **0.609** | **0.619** | **0.568** | **0.750** |
| **SABET-QA-Hard** | **0.403** | 0.479 | **0.219** | **0.715** | **0.810** | **0.485** | **0.504** | **0.513** | 0.604 | 0.615 | 0.565 | 0.747 |

*Comparison on the MultiTQ and TimeQuestions test sets.*

## Main command-line arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--model` | `sabet1` | QA architecture to use. |
| `--dataset_name` | `complex_cronquestions` | Benchmark: `cronquestions`, `complex_cronquestions`, `timequestions` or `multitq` (old names are accepted). |
| `--tkbc_model_file` | `tcomplex.ckpt` | TComplEx checkpoint in `Data/benchmarks/<benchmark>/embeddings/`. |
| `--mode` | `train` | `train` or `eval`. |
| `--supervision` | `soft` | `hard` gives the models the `(t1, t2)` hint retrieved from the raw TKG (`-Hard` variants). |
| `--load_from` | empty | Existing QA checkpoint name without `.ckpt`. |
| `--save_to` | empty | Run/checkpoint name. Empty values become `temp` inside training. |
| `--max_epochs` | `20` | Number of training epochs. |
| `--batch_size` | `150` | Training batch size. |
| `--valid_batch_size` | `150` | Validation/test batch size. |
| `--lr` | `2e-4` | Adam learning rate. |
| `--valid_freq` | `1` | Validation frequency in epochs. |
| `--eval_k` | `1` | Hits@k used for model selection (Hits@1 and Hits@10 are always reported). |
| `--frozen` | `1` | Freeze entity/time and TKBC parameters when set to `1`. |
| `--lm_frozen` | `1` | Freeze the pretrained language model when set to `1`. |
| `--fuse` | `add` | Combine temporal representations using `add` or `cat`. |
| `--time_sensitivity` | off | Enable the SubGTR time-sensitivity scoring branch. |
| `--eval_split` | `test` | Split evaluated with `--mode eval`. |
| `--lm` | `distilbert` | Language model of `sabet`/`sabet1`: `distilbert`, `bert` or `roberta` (the tokenizer follows). |
| `--no_paraphrases` | off | Train on paraphrase 0 only. |
| `--device` | `auto` | `cuda`, `mps`, `cpu` or `auto`. |
| `--seed` | `0` | Random seed. |
| `--patience` | `0` | Stop after this many validations without improvement (`0`: never). |
| `--weight_decay`, `--grad_clip` | `0` | Adam weight decay and gradient-norm clipping. |
| `--max_train_questions`, `--max_eval_questions` | all | Subsample the splits for quick runs. |
| `--num_hops`, `--hop_loss`, `--type_loss`, `--interval_aux`, `--memory_topk` | `4`, `0.3`, `0.5`, `0.5`, `16` | SABET v2 settings (`--interval_aux 0`: no raw-TKG target at all). |
| `--no_injection`, `--no_slot_ctx`, `--no_bidir`, `--no_type_head` | off | SABET v2 ablations. |
<!-- 
## Citation

Add the corresponding paper citation here when the project is released:

```bibtex
@inproceedings{your_key,
  title     = {Your Paper Title},
  author    = {Author One and Author Two},
  booktitle = {Conference or Journal},
  year      = {2026}
}
``` -->
