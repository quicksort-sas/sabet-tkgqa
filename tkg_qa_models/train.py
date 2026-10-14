"""
Train or evaluate a temporal KGQA model.

    python -m tkg_qa_models.train --model sabet1 --dataset_name complex_cronquestions --save_to run1
    python -m tkg_qa_models.train --mode eval --model sabet1 --dataset_name complex_cronquestions --load_from run1
"""

import argparse
import os

from tkg_qa_models.data import TKGBenchmark, TKGQADataset
from tkg_qa_models.kge import TComplExEmbeddings
from tkg_qa_models.models import MODEL_ALIASES, MODELS, build_model
from tkg_qa_models.paths import DATA_ROOT
from tkg_qa_models.training import (Logger, evaluate, load_checkpoint, make_loader, select_device, set_seed,
                                    train, write_json)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Temporal KGQA")
    p.add_argument("--model", default="sabet1", choices=sorted(set(MODELS) | set(MODEL_ALIASES)))
    p.add_argument("--dataset_name", default="complex_cronquestions",
                   help="cronquestions | complex_cronquestions | timequestions | multitq (legacy names accepted)")
    p.add_argument("--tkbc_model_file", default="tcomplex.ckpt", help="file in <benchmark>/embeddings/")
    p.add_argument("--supervision", default="soft", choices=["soft", "hard"],
                   help="hard: the model receives the (t1, t2) hint retrieved from the raw TKG (-Hard variants)")
    p.add_argument("--lm", default="distilbert", help="distilbert | bert | roberta (ignored by the bert/roberta baselines)")
    p.add_argument("--mode", default="train", choices=["train", "eval"])
    p.add_argument("--load_from", default="")
    p.add_argument("--save_to", default="")
    p.add_argument("--eval_split", default="test")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--seed", type=int, default=0)
    # optimisation
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=150)
    p.add_argument("--valid_batch_size", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--valid_freq", type=int, default=1)
    p.add_argument("--patience", type=int, default=0, help="stop after N evaluations without gain (0: never)")
    p.add_argument("--eval_k", type=int, default=1, help="Hits@k used to select the best checkpoint")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--frozen", type=int, default=1, help="1: freeze the TComplEx tables")
    p.add_argument("--lm_frozen", type=int, default=1, help="1: freeze the language model")
    # data
    p.add_argument("--no_paraphrases", action="store_true", help="always train on paraphrase 0")
    p.add_argument("--max_train_questions", type=int, default=None)
    p.add_argument("--max_eval_questions", type=int, default=None)
    # TempoQR / SubGTR
    p.add_argument("--fuse", default="add", choices=["add", "cat"])
    p.add_argument("--time_sensitivity", action="store_true")
    # SABET / SABET v2
    p.add_argument("--num_hops", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--hop_loss", type=float, default=0.3, help="sabet1: weight of the per-hop losses")
    p.add_argument("--type_loss", type=float, default=0.5, help="sabet1: weight of the answer-type loss")
    p.add_argument("--interval_aux", type=float, default=0.5,
                   help="sabet1 (soft): weight of the training-only (t1, t2) target, 0 disables it")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--memory_topk", type=int, default=16)
    p.add_argument("--memory_temp", type=float, default=1.0)
    p.add_argument("--no_injection", action="store_true")
    p.add_argument("--no_slot_ctx", action="store_true")
    p.add_argument("--no_bidir", action="store_true")
    p.add_argument("--no_type_head", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    bench = TKGBenchmark(args.dataset_name)
    tkbc = TComplExEmbeddings.load(bench.embeddings_path(args.tkbc_model_file))
    bench.check_embeddings(tkbc)
    model = build_model(args.model, tkbc, args)

    data = dict(tokenizer_name=model.lm_name, mask_entities=model.entity_aware)
    run = args.load_from if args.mode == "eval" else (args.save_to or "temp")
    checkpoint_path = os.path.join(DATA_ROOT, "qa_models", bench.name, f"{run}.ckpt")
    results_dir = os.path.join("results", bench.name)

    if args.load_from:
        load_checkpoint(model, os.path.join(DATA_ROOT, "qa_models", bench.name, f"{args.load_from}.ckpt"))
    model = model.to(device)

    if args.mode == "eval":
        split = TKGQADataset(bench, args.eval_split, max_questions=args.max_eval_questions, **data)
        metrics, _ = evaluate(model, make_loader(split, args.valid_batch_size, False, args.num_workers),
                              args.eval_split,
                              hop_weights_file=os.path.join(results_dir, f"{run}_{args.eval_split}_hop_weights.pt"))
        write_json(os.path.join(results_dir, f"{run}_{args.eval_split}.json"), metrics)
        return

    train_set = TKGQADataset(bench, "train", sample_paraphrases=not args.no_paraphrases,
                             max_questions=args.max_train_questions, **data)
    valid_set = TKGQADataset(bench, "valid", max_questions=args.max_eval_questions, **data)
    test_set = TKGQADataset(bench, "test", max_questions=args.max_eval_questions, **data)
    monitor_set = TKGQADataset(bench, "train", sample_paraphrases=False, max_questions=1000, **data)
    print(f"Questions: train {len(train_set)}, valid {len(valid_set)}, test {len(test_set)}")

    logger = Logger(os.path.join(results_dir, f"{run}.log"))
    best = train(model,
                 make_loader(train_set, args.batch_size, True, args.num_workers),
                 make_loader(valid_set, args.valid_batch_size, False, args.num_workers),
                 make_loader(monitor_set, args.valid_batch_size, False, 0),
                 args, checkpoint_path, logger)

    load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    metrics, lines = evaluate(model, make_loader(test_set, args.valid_batch_size, False, args.num_workers), "test",
                              hop_weights_file=os.path.join(results_dir, f"{run}_test_hop_weights.pt"))
    logger.write(lines, header="Final test evaluation of the best validation checkpoint")
    write_json(os.path.join(results_dir, f"{run}_test.json"),
               dict(metrics, best_valid=best, args=vars(args), device=str(device)))


if __name__ == "__main__":
    main()
