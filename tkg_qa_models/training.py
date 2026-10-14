import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader


def select_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=dataset.collate, persistent_workers=num_workers > 0)


class Logger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, lines, header=None):
        with open(self.path, "a") as f:
            f.write(f"Log time: {datetime.now():%d/%m/%Y %H:%M:%S}\n")
            if header:
                f.write(header + "\n")
            for line in lines:
                f.write(f"{line}\n")
            f.write("\n")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, split, k_list=(1, 10), hop_weights_file=None):
    """
    Hits@k against the full gold answer set of each question. Questions whose answers are outside
    the KG (TimeQuestions) are kept and count as errors. Returns (metrics dict, log lines).
    """
    model.eval()
    num_answers = model.num_entities + model.num_times
    hits = {k: [] for k in k_list}
    metas, hop_weights_all = [], []
    total_loss, loss_batches, unanswerable = 0.0, 0, 0
    start = time.time()

    with torch.no_grad():
        for i_batch, batch in enumerate(loader):
            out = model(batch, mode="eval")
            scores, hop_weights = out if isinstance(out, tuple) else (out, None)
            if hop_weights is not None:
                hop_weights_all.append(hop_weights.cpu())
            scores = scores[:, :num_answers].float()

            answer = batch["answer"].to(scores.device)
            answerable = answer >= 0
            unanswerable += int((~answerable).sum())
            if answerable.any():
                total_loss += F.cross_entropy(scores[answerable], answer[answerable]).item()
                loss_batches += 1

            top = scores.topk(max(k_list), dim=1).indices.cpu()                     # [B, K]
            gold = batch["answers"]                                                  # [B, A], -1 padded
            match = ((top.unsqueeze(-1) == gold.unsqueeze(1)) & (gold.unsqueeze(1) >= 0)).any(-1)
            for k in k_list:
                hits[k] += match[:, :k].any(1).tolist()
            metas += batch["meta"]
            if (i_batch + 1) % 100 == 0:
                print(f"  eval {split}: {i_batch + 1}/{len(loader)} batches", flush=True)

    if hop_weights_file and hop_weights_all:
        os.makedirs(os.path.dirname(hop_weights_file), exist_ok=True)
        torch.save({"hop_weights": torch.cat(hop_weights_all),
                    "question_type": [m["qtype"] for m in metas],
                    "answer_type": [m["answer_type"] for m in metas]}, hop_weights_file)

    metrics = {"split": split, "questions": len(metas), "unanswerable": unanswerable,
               "loss": total_loss / max(1, loss_batches), "seconds": round(time.time() - start, 1)}
    lines = [f"Split {split}", f"Loss {metrics['loss']:.6f}",
             f"Questions {len(metas)} (unanswerable, counted as errors: {unanswerable})"]
    for k in k_list:
        metrics[f"hits@{k}"] = sum(hits[k]) / max(1, len(hits[k]))
        lines.append(f"Hits at {k}: {metrics[f'hits@{k}']:.3f}")
        for field in ("qtype", "complexity", "answer_type", "time_level"):
            groups = defaultdict(list)
            for m, h in zip(metas, hits[k]):
                if m[field] is not None:
                    groups[m[field]].append(h)
            for key in sorted(groups, key=lambda x: str(x).lower()):
                v = groups[key]
                metrics[f"hits@{k}/{field}/{key}"] = sum(v) / len(v)
                lines.append(f"{key} \t {sum(v) / len(v):.3f} \t total questions: {len(v)}")
            if groups:
                lines.append("")
    for line in lines:
        print(line)
    return metrics, lines


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, train_loader, valid_loader, monitor_loader, args, checkpoint_path, logger):
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_epochs * len(train_loader)
    warmup_steps = min(200, total_steps // 10)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_steps)),
                    CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.lr * 0.01)],
        milestones=[warmup_steps],
    )
    logger.write([f"{k}:\t{v}" for k, v in vars(args).items()], header="Config:")

    best, epochs_without_gain = -1.0, 0
    for epoch in range(args.max_epochs):
        model.train()
        start, running = time.time(), defaultdict(float)
        for i_batch, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(batch, mode="train")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            for name, value in model.loss_dict.items():
                running[name] += float(value)
            n = i_batch + 1
            if n % 50 == 0 or n == len(train_loader):
                parts = " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                print(f"Epoch {epoch + 1}/{args.max_epochs} step {n}/{len(train_loader)} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {parts} ({time.time() - start:.0f}s)", flush=True)
        epoch_line = f"Epoch {epoch + 1}: " + " ".join(f"{k} {v / len(train_loader):.4f}" for k, v in running.items()) \
            + f" ({time.time() - start:.0f}s)"
        print(epoch_line)

        if (epoch + 1) % args.valid_freq == 0:
            metrics, lines = evaluate(model, valid_loader, "valid")
            score = metrics[f"hits@{args.eval_k}"]
            if score > best:
                best, epochs_without_gain = score, 0
                save_checkpoint(model, checkpoint_path)
                lines.append(f"New best validation Hits@{args.eval_k}: {score:.3f} (saved)")
            else:
                epochs_without_gain += 1
            _, monitor_lines = evaluate(model, monitor_loader, "train_1000")
            logger.write([epoch_line] + lines + monitor_lines, header=f"Epoch {epoch}")
            if args.patience and epochs_without_gain >= args.patience:
                print(f"No validation gain for {args.patience} evaluations, stopping")
                break
    return best


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Saved model to", path)


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    print("Loaded model from", path)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
