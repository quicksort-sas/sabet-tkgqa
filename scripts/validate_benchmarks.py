"""
Cross-check the standardized benchmarks against the raw files.

    python scripts/validate_benchmarks.py --raw Data/raw --out Data/benchmarks
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "scripts")
from build_benchmarks import load_pickle  # noqa: E402

RAW = {"cronquestions": "wikidata_big", "complex_cronquestions": "wikidata_big_complex",
       "timequestions": "timequestions", "multitq": "MultiTQ"}


def keys(path):
    return [line.rstrip("\n").split("\t")[1] for line in open(path, encoding="utf-8")]


def raw_questions(raw, name, split):
    if name == "multitq":
        s = {"valid": "dev"}.get(split, split)
        return json.load(open(f"{raw}/data/MultiTQ/questions/processed_questions/{s}.json", encoding="utf-8"))
    return load_pickle(f"{raw}/data/{RAW[name]}/questions/{split}.pickle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="Data/raw")
    ap.add_argument("--out", default="Data/benchmarks")
    ap.add_argument("--sample", type=int, default=2000)
    a = ap.parse_args()
    rng = random.Random(0)
    ok = True
    for name in RAW:
        ent = keys(f"{a.out}/{name}/kg/entities.tsv")
        ts = keys(f"{a.out}/{name}/kg/timestamps.tsv")
        for split in ("train", "valid", "test"):
            recs = [json.loads(l) for l in open(f"{a.out}/{name}/questions/{split}.jsonl", encoding="utf-8")]
            raw = raw_questions(a.raw, name, split)
            errors = []
            if len(recs) != len(raw):
                errors.append(f"count {len(recs)} != raw {len(raw)}")
            for i in rng.sample(range(len(recs)), min(a.sample, len(recs))):
                r, q = recs[i], raw[i]
                rid = q.get("uniq_id", q.get("quid"))
                if r["id"] != rid:
                    errors.append(f"#{i} id {r['id']} != {rid}")
                vocab = ent if r["answer_type"] == "entity" else ts
                back = {vocab[x] for x in r["answers"]}
                if name == "multitq":
                    want = {x.replace(" ", "_") if r["answer_type"] == "entity" else x for x in q["answers"]}
                else:
                    want = {str(x) for x in q["answers"]}
                if back != {w for w in want if w in set(vocab)} or len(back) + r["num_unmapped_answers"] != len(want):
                    errors.append(f"#{i} answers differ")
                if name != "multitq":
                    if {ent[e] for e in r["entities"]} != {e for e in q["entities"] if e in set(ent)}:
                        errors.append(f"#{i} entities differ")
                    if {int(ts[t]) for t in r["times"]} != {t for t in q["times"] if str(t) in set(ts)}:
                        errors.append(f"#{i} times differ")
                for p in r["paraphrases"]:
                    for s, e, kind, idx in p["mentions"]:
                        if not (0 <= s < e <= len(p["text"])):
                            errors.append(f"#{i} span out of bounds")
                    spans = sorted((s, e) for s, e, _, _ in p["mentions"])
                    if any(spans[j][1] > spans[j + 1][0] for j in range(len(spans) - 1)):
                        errors.append(f"#{i} overlapping mentions")
                h = r["hint"]
                if (h["t1"] is None) != (h["t2"] is None) or (h["t1"] is not None and not h["t1"] <= h["t2"] and name != "multitq"):
                    errors.append(f"#{i} bad hint {h}")
            status = "OK" if not errors else f"{len(errors)} problems, e.g. {errors[:3]}"
            ok &= not errors
            print(f"{name:22s} {split:5s} n={len(recs):7d}  {status}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
