# Temporal Knowledge Graph Question Answering

This project combines pretrained temporal knowledge graph embeddings with transformer-based language representations to answer questions whose answers may be either **entities** or **timestamps**.

The code includes standard KGQA baselines, temporal reasoning models, entity-aware question encoding, hard/soft temporal supervision, multi-hop reasoning, training, checkpointing, and detailed Hits@K evaluation.

> **Research-code status:** this repository assumes a specific preprocessed data layout and pretrained TKBC checkpoint.

## Available models

| CLI value | Model | Description |
|---|---|---|
| `bert` | `QA_lm` | BERT language-model baseline over head, relation, tail, and time embeddings. |
| `roberta` | `QA_lm` | RoBERTa variant of the language-model baseline. |
| `embedkgqa` | `QA_embedkgqa` | EmbedKGQA-style complex-valued entity scoring. |
| `cronkgqa` | `QA_cronkgqa` | Temporal KGQA baseline with entity and timestamp prediction. |
| `tempo_qr` | `QA_TempoQR` | Entity- and time-aware transformer reasoning model. |
| `subgtr` | `QA_SubGTR` | Subgraph-aware temporal reasoning with optional time sensitivity. |
| `sabet` | `QA_Sabet` | Multi-hop, bidirectional temporal KGQA model with contextualized entity/time slots and learned hop fusion. |

### SABET (our model) overview

SABET uses:

1. DistilBERT token representations projected into the TKBC embedding space.
2. Cross-attention to contextualize head, tail, and time slots from the question.
3. CLS, mean, and max pooling to build a question summary.
4. Multi-hop relation states with temporal hints from retrieved start/end times.
5. Bidirectional entity and timestamp scoring using TComplEx-style operations.
6. Working-memory updates based on predicted entity and time distributions.
7. A learned gate that aggregates predictions from all reasoning hops.

## Repository structure

The  code is organized as follows:

```text
.
└── tkg_qa_models/
    ├── train_qa_model.py               # Training and evaluation entry point
    ├── qa_baselines.py                 # QA_lm, EmbedKGQA, CronKGQA, TempoQR,
    │                                   # SubGTR, and SABET implementations
    ├── qa_datasets.py                  # Dataset preparation and collate functions
    ├── hard_supervision_functions.py   # Temporal fact retrieval and supervision
    ├── tcomplex.py                     # Temporal KG embedding model (TComplEx)
    ├── ner_task.py                     # Named entity recognition and entity linking
    └── utils.py                        # Dictionary and TKBC-loading utilities
```

## Requirements

- Python 3.9 or newer.
- A CUDA-capable GPU.
- PyTorch with CUDA support.

Core Python dependencies:

```text
torch
transformers
numpy
tqdm
```

Install the basic dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch transformers numpy tqdm
```

## Data layout

The code currently uses the absolute root:

```python
data_dir = "/Data/data"
```

Under that root, it expects a layout similar to:

```text
/Data/data/
├── data/
│   └── <dataset_name>/
│       ├── questions/
│       │   ├── train.pickle
│       │   ├── valid.pickle           # or dev.pickle for MultiTQ
│       │   └── test.pickle
│       └── ...                         # mappings and TKG resources used by utils
├── models/
│   └── <dataset_name>/
│       ├── kg_embeddings/
│       │   └── tcomplex.ckpt
│       └── qa_models/
│           └── <checkpoint>.ckpt       # checkpoints loaded with --load_from
└── qa_models/
    └── <dataset_name>/
        └── <run_name>.ckpt             # checkpoints saved during training
```


```text
head_entity    relation    tail_entity    start_time    end_time
```

Example:

```text
Q76    P39    Q11696    2009    2017
```

Replace the hard-coded `data_dir` in both the training and dataset modules, or expose it as a command-line argument/environment variable.

The data can be downloaded from https://osf.io/h39ma/overview?view_only=3f09fa70ccc04929923898a972830652

## Training

A representative SABET run:

```bash
python train_qa.py \
  --model sabet \
  --dataset_name wikidata_big_complex \
  --tkbc_model_file tcomplex.ckpt \
  --tkg_file full.txt \
  --supervision hard \
  --save_to sabet_run_01 \
  --max_epochs 20 \
  --batch_size 150 \
  --valid_batch_size 150 \
  --lr 2e-4 \
  --valid_freq 1 \
  --eval_k 1
```

During training, the script:

1. Loads a pretrained TKBC model.
2. Instantiates the selected QA model and matching dataset class.
3. Trains with Adam.
4. Applies linear warm-up followed by cosine learning-rate decay.
5. Evaluates on the validation set at the configured frequency.
6. Saves the best validation checkpoint.
7. Evaluates a 1,000-example training subset to help diagnose overfitting.
8. Reloads the best checkpoint and evaluates it on the test set.

Logs are written to:

```text
results/<dataset_name>/<save_to>.log
```

Best checkpoints are written to:

```text
/Data/data/qa_models/<dataset_name>/<save_to>.ckpt
```

## Evaluation

Evaluate an existing checkpoint:

```bash
python train_qa.py \
  --mode eval \
  --model sabet \
  --dataset_name wikidata_big_complex \
  --tkbc_model_file tcomplex.ckpt \
  --load_from sabet_run_01 \
  --eval_split test \
  --valid_batch_size 150 \
  --eval_k 1
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
| `--model` | `sabet` | QA architecture to use. |
| `--dataset_name` | `wikidata_big_complex` | Dataset directory/name. |
| `--tkbc_model_file` | `tcomplex.ckpt` | Pretrained TKBC checkpoint filename. |
| `--tkg_file` | `full.txt` | Temporal KG facts file used for supervision/retrieval. |
| `--mode` | `train` | `train`, `eval`, or `test_kge`. |
| `--supervision` | `soft` | Temporal supervision mode; model support varies. |
| `--load_from` | empty | Existing QA checkpoint name without `.ckpt`. |
| `--save_to` | empty | Run/checkpoint name. Empty values become `temp` inside training. |
| `--max_epochs` | `20` | Number of training epochs. |
| `--batch_size` | `150` | Training batch size. |
| `--valid_batch_size` | `150` | Validation/test batch size. |
| `--lr` | `2e-4` | Adam learning rate. |
| `--valid_freq` | `1` | Validation frequency in epochs. |
| `--eval_k` | `1` | Requested evaluation cutoff. See the caveat below. |
| `--frozen` | `1` | Freeze entity/time and TKBC parameters when set to `1`. |
| `--lm_frozen` | `1` | Freeze the pretrained language model when set to `1`. |
| `--fuse` | `add` | Combine temporal representations using `add` or `cat`. |
| `--time_sensitivity` | off | Enable the SubGTR time-sensitivity scoring branch. |
| `--corrupt_hard` | `0.0` | Probability/amount of corruption applied to hard supervision. |
| `--eval_split` | `valid` | Label used in evaluation output and KGE checks. |
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
