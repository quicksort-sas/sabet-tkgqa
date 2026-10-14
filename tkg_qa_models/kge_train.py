"""
Train TComplEx embeddings on a benchmark KG, following Lacroix et al. (ICLR 2020) and their tkbc code:

  * reciprocal relations: every fact (s, r, o, [b, e]) is also trained as (o, r + R, s, [b, e])
  * full cross-entropy over all entities for the object, plus cross-entropy over all timestamps
    for the time of facts whose interval is not the whole timeline (tkbc's IKBCOptimizer)
  * a timestamp is sampled uniformly in [begin, end] at every step (interval facts)
  * N3 regularisation of the factors (|lhs|, |rel * time|, |rhs| moduli) and Λ3 temporal smoothness
    on consecutive timestamp embeddings
  * Adagrad, learning rate 0.1, batch size 1000
  * filtered MRR / Hits@k of object prediction; the query time is sampled in the fact interval and
    the filter removes every object valid at that time in any split

    python -m tkg_qa_models.kge_train --dataset_name cronquestions --name tcomplex_v2
    python -m tkg_qa_models.kge_train --dataset_name cronquestions --evaluate tcomplex.ckpt

The checkpoint has the same format as the released ones (embeddings.{0,1,2}.weight) and is written to
Data/benchmarks/<benchmark>/embeddings/<name>.ckpt with <name>.json (config and metrics). Use it in QA
training with --tkbc_model_file <name>.ckpt.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from tkg_qa_models.data import TKGBenchmark
from tkg_qa_models.kge import TComplExEmbeddings, score_entities, score_timestamps
from tkg_qa_models.training import select_device, set_seed

SHARED_KG = {"complex_cronquestions": "cronquestions"}      # benchmarks that share one KG


def load_facts(bench, split):
    path = os.path.join(bench.dir, "kg", f"{split}.tsv")
    return torch.from_numpy(np.loadtxt(path, dtype=np.int64, ndmin=2))


def n3(factors):
    return sum(torch.sum(torch.abs(f) ** 3) for f in factors) / factors[0].shape[0]


def lambda3(time_table, rank):
    diff = time_table[1:] - time_table[:-1]
    return torch.sum(torch.sqrt(diff[:, :rank] ** 2 + diff[:, rank:] ** 2) ** 3) / (time_table.shape[0] - 1)


def modulus(x, rank):
    return torch.sqrt(x[:, :rank] ** 2 + x[:, rank:] ** 2)


def sample_time(begin, end, generator=None):
    u = torch.rand(begin.shape, generator=generator)
    return torch.round(begin + u * (end - begin)).long()


class TComplExTrainer:
    def __init__(self, model, num_base_relations, args, device):
        self.model, self.args, self.device = model, args, device
        self.R = num_base_relations
        self.last_ts = model.num_timestamps - 1
        self.optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    def epoch(self, facts):
        """facts: [N, 5] (s, r, o, begin, end) without reciprocals."""
        inverse = facts[:, [2, 1, 0, 3, 4]].clone()
        inverse[:, 1] += self.R
        data = torch.cat([facts, inverse])[torch.randperm(2 * len(facts))]
        rank = self.model.rank
        E, Rel, T = self.model.embeddings
        totals, start = defaultdict(float), time.time()
        for i in range(0, len(data), self.args.batch_size):
            batch = data[i:i + self.args.batch_size]
            t = sample_time(batch[:, 3].float(), batch[:, 4].float())
            b = batch.to(self.device)
            t = t.to(self.device)
            lhs, rel, rhs, tim = E(b[:, 0]), Rel(b[:, 1]), E(b[:, 2]), T(t)
            full_rel = torch.cat([rel[:, :rank] * tim[:, :rank] - rel[:, rank:] * tim[:, rank:],
                                  rel[:, rank:] * tim[:, :rank] + rel[:, :rank] * tim[:, rank:]], dim=1)
            scores = score_entities(lhs, rel, tim, E.weight, rank)
            fit = F.cross_entropy(scores, b[:, 2])
            reg = self.args.emb_reg * n3((modulus(lhs, rank), modulus(full_rel, rank), modulus(rhs, rank)))
            treg = self.args.time_reg * lambda3(T.weight, rank)
            # time prediction, except for facts valid over the whole timeline
            has_time = ~((b[:, 3] == 0) & (b[:, 4] == self.last_ts))
            if has_time.any():
                tscores = score_timestamps(lhs[has_time], rel[has_time], rhs[has_time], T.weight, rank)
                time_loss = F.cross_entropy(tscores, t[has_time])
            else:
                time_loss = scores.new_zeros(())
            loss = fit + reg + treg + time_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            for k, v in (("fit", fit), ("reg", reg), ("time_reg", treg), ("time", time_loss), ("total", loss)):
                totals[k] += v.item() * len(batch)
        return {k: v / len(data) for k, v in totals.items()} | {"seconds": round(time.time() - start, 1)}


class FilteredEvaluator:
    """Filtered object ranking at a sampled time, against every fact of every split."""

    def __init__(self, all_facts, num_timestamps, seed=0):
        self.index = defaultdict(list)                     # (s, r) -> [(o, begin, end)]
        for s, r, o, b, e in all_facts.tolist():
            self.index[(s, r)].append((o, b, e))
        self.seed = seed

    @torch.no_grad()
    def evaluate(self, model, facts, device, batch_size=500, max_facts=None):
        g = torch.Generator().manual_seed(self.seed)
        if max_facts and len(facts) > max_facts:
            facts = facts[torch.randperm(len(facts), generator=g)[:max_facts]]
        times = sample_time(facts[:, 3].float(), facts[:, 4].float(), generator=g)
        E, Rel, T = model.embeddings
        ranks = []
        for i in range(0, len(facts), batch_size):
            f, t = facts[i:i + batch_size], times[i:i + batch_size]
            scores = score_entities(E(f[:, 0].to(device)), Rel(f[:, 1].to(device)), T(t.to(device)),
                                    E.weight, model.rank)
            target = scores.gather(1, f[:, 2:3].to(device))
            rows, cols = [], []
            for j, (s, r, o, _, _) in enumerate(f.tolist()):
                tj = int(t[j])
                for o2, b2, e2 in self.index[(s, r)]:
                    if o2 != o and b2 <= tj <= e2:
                        rows.append(j)
                        cols.append(o2)
            if rows:
                scores[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = -float("inf")
            ranks.append((scores > target).sum(1).cpu() + 1)
        ranks = torch.cat(ranks).float()
        return {"mrr": float((1 / ranks).mean()), "hits@1": float((ranks <= 1).float().mean()),
                "hits@3": float((ranks <= 3).float().mean()), "hits@10": float((ranks <= 10).float().mean()),
                "facts": len(ranks)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train TComplEx embeddings on a benchmark KG")
    p.add_argument("--dataset_name", default="cronquestions")
    p.add_argument("--name", default="tcomplex_v2", help="output checkpoint name (without .ckpt)")
    p.add_argument("--evaluate", default="", help="only evaluate this checkpoint file of <benchmark>/embeddings/")
    p.add_argument("--rank", type=int, default=256, help="complex rank; embedding dim = 2 * rank")
    p.add_argument("--emb_reg", type=float, default=1e-2)
    p.add_argument("--time_reg", type=float, default=1e-2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--init_size", type=float, default=1e-2)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--valid_freq", type=int, default=5)
    p.add_argument("--patience", type=int, default=3, help="validations without MRR gain before stopping")
    p.add_argument("--max_eval_facts", type=int, default=20000, help="subsample large valid/test splits")
    p.add_argument("--train_on", default="train", choices=["train", "all"],
                   help="all: also train on valid+test facts (final embeddings for QA, fixed --max_epochs)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = select_device(args.device)
    bench = TKGBenchmark(SHARED_KG.get(args.dataset_name, args.dataset_name))
    splits = {s: load_facts(bench, s) for s in ("train", "valid", "test")}
    evaluator = FilteredEvaluator(torch.cat(list(splits.values())), bench.num_timestamps, seed=args.seed)
    emb_dir = os.path.join(bench.dir, "embeddings")
    print(f"Device {device}; {bench.name}: train {len(splits['train'])}, valid {len(splits['valid'])}, "
          f"test {len(splits['test'])} facts")

    if args.evaluate:
        model = TComplExEmbeddings.load(os.path.join(emb_dir, args.evaluate)).to(device)
        report = {s: evaluator.evaluate(model, splits[s], device, max_facts=args.max_eval_facts)
                  for s in ("valid", "test")}
        print(json.dumps(report, indent=1))
        return

    model = TComplExEmbeddings(bench.num_entities, 2 * bench.num_relations, bench.num_timestamps, args.rank)
    for emb in model.embeddings:
        emb.weight.data *= args.init_size
    model = model.to(device)
    trainer = TComplExTrainer(model, bench.num_relations, args, device)
    train_facts = splits["train"] if args.train_on == "train" else torch.cat(list(splits.values()))

    ckpt_path = os.path.join(emb_dir, f"{args.name}.ckpt")
    history, best, best_epoch, bad = [], -1.0, 0, 0
    for epoch in range(1, args.max_epochs + 1):
        stats = trainer.epoch(train_facts)
        line = {"epoch": epoch, **stats}
        if args.train_on == "train" and epoch % args.valid_freq == 0:
            line["valid"] = evaluator.evaluate(model, splits["valid"], device, max_facts=args.max_eval_facts)
            if line["valid"]["mrr"] > best:
                best, best_epoch, bad = line["valid"]["mrr"], epoch, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                bad += 1
        history.append(line)
        print(json.dumps(line), flush=True)
        if args.train_on == "train" and args.patience and bad >= args.patience:
            print(f"No validation gain for {args.patience} evaluations, stopping")
            break
    if args.train_on == "all":
        torch.save(model.state_dict(), ckpt_path)
        best_epoch = args.max_epochs

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    final = {s: evaluator.evaluate(model, splits[s], device, max_facts=args.max_eval_facts) for s in ("valid", "test")}
    report = {"checkpoint": ckpt_path, "best_epoch": best_epoch, "final": final, "config": vars(args),
              "device": str(device), "history": history}
    with open(os.path.join(emb_dir, f"{args.name}.json"), "w") as f:
        json.dump(report, f, indent=1)
    for other, kg in SHARED_KG.items():                    # share the file with benchmarks on the same KG
        if kg == bench.name:
            dst = os.path.join(TKGBenchmark(other).dir, "embeddings")
            for ext in (".ckpt", ".json"):
                target = os.path.join(dst, args.name + ext)
                if os.path.exists(target):
                    os.remove(target)
                os.link(os.path.join(emb_dir, args.name + ext), target)
    print(json.dumps({"best_epoch": best_epoch, "final": final}, indent=1))


if __name__ == "__main__":
    main()
