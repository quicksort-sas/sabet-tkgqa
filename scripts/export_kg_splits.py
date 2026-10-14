"""
Export the KG splits used to train TComplEx as id-aligned, disjoint files:

    python scripts/export_kg_splits.py --raw Data/raw --out Data/benchmarks

writes <out>/<benchmark>/kg/{train,valid,test}.tsv   (head rel tail start end, all ids)

Source: tkbc_processed_data/<dataset>/{train,valid,test}.pickle, i.e. the data the released
checkpoints were trained on. The raw splits overlap (CronQuestions: every validation fact is also a
training fact; MultiTQ: month/year copies cross splits), so rows are deduplicated and each fact is kept
in one split only, with priority test > valid > train. Point timestamps (MultiTQ) get start = end.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_benchmarks import load_pickle  # noqa: E402

SOURCES = {
    "cronquestions": "wikidata_big",
    "complex_cronquestions": "wikidata_big",       # same KG and checkpoint
    "timequestions": "timequestions",
    "multitq": "MultiTQ",
}


def read_split(raw, ds, split):
    arr = load_pickle(f"{raw}/data/{ds}/kg/tkbc_processed_data/{ds}/{split}.pickle", allow_numpy=True).tolist()
    rows = [tuple(int(x) for x in r) for r in arr]
    return [r if len(r) == 5 else (r[0], r[1], r[2], r[3], r[3]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="Data/raw")
    ap.add_argument("--out", default="Data/benchmarks")
    args = ap.parse_args()

    for name, ds in SOURCES.items():
        raw = {s: read_split(args.raw, ds, s) for s in ("train", "valid", "test")}
        test = list(dict.fromkeys(raw["test"]))
        test_set = set(test)
        valid = [f for f in dict.fromkeys(raw["valid"]) if f not in test_set]
        held_out = test_set | set(valid)
        train = [f for f in dict.fromkeys(raw["train"]) if f not in held_out]
        stats = {
            "source": f"{args.raw}/data/{ds}/kg/tkbc_processed_data/{ds}",
            "raw_rows": {s: len(v) for s, v in raw.items()},
            "exported": {"train": len(train), "valid": len(valid), "test": len(test)},
            "removed_from_train_as_held_out": len(set(raw["train"]) & held_out),
            "removed_from_valid_as_test": len(set(raw["valid"]) & test_set),
        }
        for split, rows in (("train", train), ("valid", valid), ("test", test)):
            with open(f"{args.out}/{name}/kg/{split}.tsv", "w") as f:
                for r in rows:
                    f.write("\t".join(map(str, r)) + "\n")
        meta_path = f"{args.out}/{name}/meta.json"
        meta = json.load(open(meta_path))
        meta["kg_splits"] = stats
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(name, stats)


if __name__ == "__main__":
    main()
