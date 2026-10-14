"""
Build the standardized benchmark folders used by tkg_qa_models.data.

    python scripts/build_benchmarks.py --raw Data/raw --out Data/benchmarks

Output, for each benchmark (cronquestions, complex_cronquestions, timequestions, multitq):

    <out>/<name>/
    ├── kg/
    │   ├── entities.tsv     id <TAB> key <TAB> label        (id = row of the entity embedding table)
    │   ├── relations.tsv    id <TAB> key <TAB> label        (id = row of the relation embedding table)
    │   ├── timestamps.tsv   id <TAB> key                    (id = row of the time embedding table)
    │   └── facts.tsv        head <TAB> rel <TAB> tail <TAB> start <TAB> end   (all ids)
    ├── questions/{train,valid,test}.jsonl
    ├── embeddings/*.ckpt    (hard links to the raw checkpoints)
    └── meta.json            sizes, sentinel timestamps, build statistics and checks

Every question record has the same schema (see `QUESTION_SCHEMA` below). All ids are final:
entities and timestamps index the TComplEx tables directly, timestamps are NOT offset by the
number of entities.

The temporal hint (t1, t2) follows TempoQR-Hard (Mavromatis et al., AAAI 2022, Sec. 4.3):
collect the timestamps of the TKG facts that involve the annotated question entities, sort them
and keep the first and the last. It is computed once, deterministically, with these rules:
  * explicit question times             -> t1 = min, t2 = max of those times       (source "question")
  * facts linking two question entities -> those facts, either direction            (source "pair")
    otherwise facts of each entity      -> their union                              (source "entity")
  * CronQuestions event questions       -> the event's occurrence facts plus the facts of the
                                           other entities that fall within +/- 10 years (source "event")
  * placeholder timestamps (unknown start/end of TimeQuestions, year 0 / 9620 in Wikidata) are
    ignored; for MultiTQ only day-level facts are used (the month/year copies are augmentation)
  * nothing retrieved                   -> t1 = t2 = null (the model sees "no hint")
"""

import argparse
import json
import os
import pickle
import re
import shutil
from collections import Counter, defaultdict

QUESTION_SCHEMA = {
    "id": "original question id",
    "text": "canonical question text (paraphrase 0, entity names, not Wikidata ids)",
    "paraphrases": "[{text, mentions}] with mentions = [[start, end, kind, id]] (kind: 'entity'|'time')",
    "entities": "entity ids, ordered by first mention in `text` (unlocated ones last, by id)",
    "times": "timestamp ids mentioned in the question",
    "roles": "{head, tail, tail2, event_head, time} ids from the dataset annotation (CronQuestions only)",
    "answer_type": "'entity' | 'time'",
    "answers": "gold answer ids (entity ids or timestamp ids, according to answer_type)",
    "num_unmapped_answers": "gold answers absent from the KG vocabulary",
    "qtype": "question type used for reporting",
    "qtype_all": "all type labels (TimeQuestions questions can have several)",
    "complexity": "'simple' | 'complex' (CronQuestions), 'single' | 'multiple' (MultiTQ), null otherwise",
    "time_level": "MultiTQ answer granularity ('day' | 'month' | 'year'), null otherwise",
    "hint": "{t1, t2, source, num_facts}",
}

KG_SOURCE = {"wikidata_big_complex": "wikidata_big"}   # Complex-CronQuestions uses the CronQuestions KG
WIKIDATA_EVENT = ("P793", "Q1190554")   # (significant event, occurrence) facts give event times
EVENT_WINDOW = 10                         # years around the event used by CronQuestions event questions


# ── Raw readers ─────────────────────────────────────────────────────────────────

class _SafeUnpickler(pickle.Unpickler):
    """Only builtins/collections (and numpy arrays when allowed); refuses arbitrary code."""

    def __init__(self, f, allow_numpy=False):
        super().__init__(f)
        self.allow_numpy = allow_numpy

    def find_class(self, module, name):
        if module in ("builtins", "collections", "copyreg", "_codecs"):
            return super().find_class(module, name)
        if self.allow_numpy and module.startswith("numpy"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def load_pickle(path, allow_numpy=False):
    with open(path, "rb") as f:
        return _SafeUnpickler(f, allow_numpy).load()


def read_tsv_dict(path, value_int=True):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            d[parts[0]] = int(parts[1]) if value_int else parts[1]
    return d


def read_facts(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                yield parts


# ── Mention location ────────────────────────────────────────────────────────────

_SLOT = re.compile(r"\{(\w+)\}")


def template_regex(template):
    """'Who was {head} after {tail}' -> regex with one named group per slot (None if unusable)."""
    pieces, pos, names = [], 0, []
    for m in _SLOT.finditer(template):
        pieces.append(re.escape(template[pos:m.start()]))
        if m.group(1) in names:
            return None
        names.append(m.group(1))
        pieces.append(f"(?P<{m.group(1)}>.+?)")
        pos = m.end()
    pieces.append(re.escape(template[pos:]))
    try:
        return re.compile(r"\s*" + "".join(pieces) + r"\s*[?.]?\s*", re.IGNORECASE)
    except re.error:
        return None


def find_surface(text, surfaces, taken):
    """Leftmost non-overlapping, word-bounded, case-insensitive match of the longest surface form."""
    for s in sorted({s for s in surfaces if s}, key=len, reverse=True):
        for m in re.finditer(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, re.IGNORECASE):
            if all(m.end() <= a or m.start() >= b for a, b in taken):
                return m.start(), m.end()
    return None


def locate_mentions(text, items, slot_spans=None):
    """
    items: [(kind, id, surfaces, slot_name)]. Uses template slot spans when available,
    otherwise surface-form search. Returns sorted [[start, end, kind, id]].
    """
    mentions, taken = [], []
    for kind, idx, surfaces, slot in items:
        span = slot_spans.get(slot) if (slot_spans and slot) else None
        if span is None:
            span = find_surface(text, surfaces, taken)
        if span is not None and all(span[1] <= a or span[0] >= b for a, b in taken):
            taken.append(span)
            mentions.append([span[0], span[1], kind, idx])
    mentions.sort()
    return mentions


# ── Temporal hint ───────────────────────────────────────────────────────────────

class HintIndex:
    def __init__(self, facts, ignore_start=(), ignore_end=(), usable_ts=None, events=None):
        """facts: iterable of (h, r, t, start, end) ids."""
        self.by_ent = defaultdict(list)
        self.by_pair = defaultdict(list)
        self.events = events or {}
        for h, r, t, s, e in facts:
            if usable_ts is not None and (s not in usable_ts or e not in usable_ts):
                continue
            s = None if s in ignore_start else s
            e = None if e in ignore_end else e
            if s is None and e is None:
                continue
            s, e = (s if s is not None else e), (e if e is not None else s)
            s, e = min(s, e), max(s, e)          # some raw facts have start > end
            self.by_ent[h].append((s, e))
            self.by_ent[t].append((s, e))
            self.by_pair[frozenset((h, t))].append((s, e))

    def hint(self, entities, times, event_head=None, year_of=None):
        if times:
            return {"t1": min(times), "t2": max(times), "source": "question", "num_facts": 0}
        ents = list(dict.fromkeys(entities))
        intervals, source = [], None
        if event_head is not None and event_head in self.events:
            ev = self.events[event_head]
            lo, hi = min(s for s, _ in ev), max(e for _, e in ev)
            intervals = list(ev)
            for e in ents:
                if e != event_head:
                    intervals += [(s, t) for s, t in self.by_ent.get(e, ())
                                  if year_of(s) >= year_of(lo) - EVENT_WINDOW and year_of(t) <= year_of(hi) + EVENT_WINDOW]
            source = "event"
        else:
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    intervals += self.by_pair.get(frozenset((ents[i], ents[j])), [])
            source = "pair"
            if not intervals:
                for e in ents:
                    intervals += self.by_ent.get(e, [])
                source = "entity"
        if not intervals:
            return {"t1": None, "t2": None, "source": "none", "num_facts": 0}
        return {"t1": min(s for s, _ in intervals), "t2": max(e for _, e in intervals),
                "source": source, "num_facts": len(intervals)}


# ── Benchmarks ──────────────────────────────────────────────────────────────────

def wikidata_vocab(raw, ds, label_dir):
    base = f"{raw}/data/{ds}/kg/tkbc_processed_data/{ds}/"
    ent = load_pickle(base + "ent_id")
    rel = load_pickle(base + "rel_id")
    ts = {k[0]: v for k, v in load_pickle(base + "ts_id").items()}           # year -> id
    ent_label = read_tsv_dict(f"{label_dir}/wd_id2entity_text.txt", value_int=False)
    rel_label = read_tsv_dict(f"{label_dir}/wd_id2relation_text.txt", value_int=False)
    return ent, rel, ts, ent_label, rel_label


def convert_cron(raw, ds, split, ent, ts, ent_label, aliases, hint_index, year_of):
    questions = load_pickle(f"{raw}/data/{ds}/questions/{split}.pickle")
    out, stats = [], Counter()
    for q in questions:
        ann = q.get("annotation", {})
        rx = template_regex(q["template"]) if q.get("template") else None
        roles = {k: (ent[v] if k != "time" else ts[int(v)]) for k, v in ann.items()
                 if k in ("head", "tail", "tail2", "event_head", "time")}
        # entity/time items in template-slot order, so slot spans can be used directly
        items = []
        for slot in ("head", "tail", "tail2", "event_head"):
            if slot in ann:
                qid = ann[slot]
                items.append(("entity", ent[qid], [ent_label.get(qid, "")] + sorted(aliases.get(qid, ())), slot))
        for qid in sorted(q["entities"]):
            if ent[qid] not in [i[1] for i in items]:
                items.append(("entity", ent[qid], [ent_label.get(qid, "")] + sorted(aliases.get(qid, ())), None))
        for year in sorted(q["times"]):
            items.append(("time", ts[year], [str(year)], "time" if ann.get("time") == year else None))

        paraphrases = []
        for text in q["paraphrases"]:
            m = rx.fullmatch(text) if rx else None
            slot_spans = {k: m.span(k) for k in m.groupdict()} if m else None
            stats["paraphrase_template_aligned"] += m is not None
            stats["paraphrases"] += 1
            mentions = locate_mentions(text, items, slot_spans)
            stats["mentions_found"] += len(mentions)
            stats["mentions_expected"] += len(items)
            paraphrases.append({"text": text, "mentions": mentions})

        answers, unmapped = [], 0
        for a in q["answers"]:
            idx = ent.get(a) if q["answer_type"] == "entity" else ts.get(int(a))
            if idx is None:
                unmapped += 1
            else:
                answers.append(idx)
        entities = order_entities(paraphrases[0]["mentions"], [ent[e] for e in q["entities"]])
        times = sorted(ts[y] for y in q["times"])
        hint = hint_index.hint(entities, times,
                               event_head=roles.get("event_head"), year_of=year_of)
        qtype = q["type"]
        out.append({
            "id": q.get("uniq_id"),
            "text": paraphrases[0]["text"],
            "paraphrases": paraphrases,
            "entities": entities,
            "times": times,
            "roles": roles,
            "answer_type": q["answer_type"],
            "answers": sorted(answers),
            "num_unmapped_answers": unmapped,
            "qtype": qtype,
            "qtype_all": [qtype],
            "complexity": "simple" if qtype in ("simple_entity", "simple_time") else "complex",
            "time_level": None,
            "hint": hint,
        })
    return out, stats


def convert_timequestions(raw, split, ent, ts, ent_label, aliases, hint_index):
    questions = load_pickle(f"{raw}/data/timequestions/questions/{split}.pickle")
    out, stats = [], Counter()
    for q in questions:
        items = [("entity", ent[e], [ent_label.get(e, "")] + sorted(aliases.get(e, ())), None)
                 for e in sorted(q["entities"]) if e in ent]
        items += [("time", ts[y], [str(y)], None) for y in sorted(q["times"]) if y in ts]
        text = q["paraphrases"][0] if q.get("paraphrases") else q["question"]
        mentions = locate_mentions(text, items)
        stats["paraphrases"] += 1
        stats["mentions_found"] += len(mentions)
        stats["mentions_expected"] += len(items)
        stats["entities_not_in_vocab"] += sum(e not in ent for e in q["entities"])

        answers, unmapped = [], 0
        for a in q["answers"]:
            # time answers are ints; strings such as '+37' are quantities (ages, counts), not years
            idx = ent.get(a) if q["answer_type"] == "entity" else (ts.get(a) if isinstance(a, int) else None)
            if idx is None:
                unmapped += 1
            else:
                answers.append(idx)
        entities = order_entities(mentions, [ent[e] for e in q["entities"] if e in ent])
        times = sorted(ts[y] for y in q["times"] if y in ts)
        types = list(q["type"]) if isinstance(q["type"], (list, tuple)) else [q["type"]]
        out.append({
            "id": q.get("uniq_id"),
            "text": text,
            "paraphrases": [{"text": text, "mentions": mentions}],
            "entities": entities,
            "times": times,
            "roles": {},
            "answer_type": q["answer_type"],
            "answers": sorted(set(answers)),
            "num_unmapped_answers": unmapped,
            "qtype": types[0],
            "qtype_all": types,
            "complexity": None,
            "time_level": None,
            "hint": hint_index.hint(entities, times),
        })
    return out, stats


def convert_multitq(raw, split, ent, ts, hint_index):
    raw_split = {"valid": "dev"}.get(split, split)
    questions = json.load(open(f"{raw}/data/MultiTQ/questions/processed_questions/{raw_split}.json", encoding="utf-8"))
    out, stats = [], Counter()
    for q in questions:
        text = q["question"]
        mentions, taken = [], []
        for ep in q.get("entity_positions", []):
            name = ep["entity"][0]
            idx = ent.get(name.replace(" ", "_"))
            s, e = ep["position"]
            stats["mentions_expected"] += 1
            if idx is None:
                stats["linked_entity_not_in_vocab"] += 1
                continue
            if all(e <= a or s >= b for a, b in taken):
                taken.append((s, e))
                mentions.append([s, e, "entity", idx])
        times = []
        for t in q.get("time", []):
            idx = ts.get(t)
            stats["mentions_expected"] += 1
            if idx is None:
                stats["question_time_not_in_vocab"] += 1
                continue
            times.append(idx)
            span = find_surface(text, [t], taken)
            if span:
                taken.append(span)
                mentions.append([span[0], span[1], "time", idx])
        mentions.sort()
        stats["mentions_found"] += len(mentions)
        stats["paraphrases"] += 1

        answers, unmapped = [], 0
        for a in q["answers"]:
            idx = ent.get(a.replace(" ", "_")) if q["answer_type"] == "entity" else ts.get(a)
            if idx is None:
                unmapped += 1
            else:
                answers.append(idx)
        linked = [ent[ep["entity"][0].replace(" ", "_")] for ep in q.get("entity_positions", [])
                  if ep["entity"][0].replace(" ", "_") in ent]
        entities = order_entities(mentions, linked)
        out.append({
            "id": q.get("quid"),
            "text": text,
            "paraphrases": [{"text": text, "mentions": mentions}],
            "entities": entities,
            "times": sorted(set(times)),
            "roles": {},
            "answer_type": q["answer_type"],
            "answers": sorted(set(answers)),
            "num_unmapped_answers": unmapped,
            "qtype": q["qtype"],
            "qtype_all": [q["qtype"]],
            "complexity": q["qlabel"].lower(),
            "time_level": q["time_level"],
            "hint": hint_index.hint(entities, sorted(set(times))),
        })
    return out, stats


def order_entities(mentions, all_entities):
    ordered = []
    for _, _, kind, idx in mentions:
        if kind == "entity" and idx not in ordered:
            ordered.append(idx)
    ordered += sorted(set(all_entities) - set(ordered))
    return ordered


def _int_or_none(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# ── Writing and checks ──────────────────────────────────────────────────────────

def write_vocab(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for idx, key, label in sorted(items):
            f.write(f"{idx}\t{key}\t{label}\n" if label is not None else f"{idx}\t{key}\n")


def check_ids(name, mapping):
    ids = sorted(mapping.values())
    assert ids == list(range(len(ids))), f"{name}: ids are not contiguous 0..{len(ids) - 1}"


def check_embeddings(ckpt_dir, n_ent, n_rel, n_ts):
    import torch
    report = {}
    for f in sorted(os.listdir(ckpt_dir)):
        if not f.endswith(".ckpt"):
            continue
        sd = torch.load(os.path.join(ckpt_dir, f), map_location="cpu", weights_only=True)
        shapes = [tuple(sd[f"embeddings.{i}.weight"].shape) for i in range(3)]
        report[f] = {"entities": shapes[0][0], "relations": shapes[1][0], "timestamps": shapes[2][0], "dim": shapes[0][1]}
        report[f]["matches_vocab"] = (shapes[0][0] == n_ent and shapes[1][0] in (n_rel, 2 * n_rel) and shapes[2][0] == n_ts)
    return report


def check_tkbc_alignment(tkbc_pickle, facts_ids, sample=20000):
    """Rows of the TComplEx training file (numpy ids) must be facts of our id-converted KG."""
    arr = load_pickle(tkbc_pickle, allow_numpy=True)
    ours = {tuple(f[:3]) for f in facts_ids}
    rows = [tuple(r[:3]) for r in arr[:sample].tolist()]
    return {"tkbc_rows_checked": len(rows), "found_in_converted_facts": sum(r in ours for r in rows),
            "tkbc_columns": int(arr.shape[1])}


def link_embeddings(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        if f.endswith(".ckpt"):
            dst = os.path.join(dst_dir, f)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(src_dir, f), dst)
                except OSError:
                    shutil.copy2(os.path.join(src_dir, f), dst)


def hint_stats(records, year_of):
    """How often the hint equals or brackets a gold time answer (relevant for the leakage discussion)."""
    s = defaultdict(Counter)
    for r in records:
        h = r["hint"]
        s["_all"][h["source"]] += 1
        if r["answer_type"] != "time" or not r["answers"] or h["t1"] is None:
            continue
        key = r["qtype"]
        s[key]["time_questions_with_hint"] += 1
        golds = set(r["answers"])
        s[key]["answer_is_t1_or_t2"] += bool(golds & {h["t1"], h["t2"]})
        s[key]["answer_within_t1_t2"] += any(year_of(h["t1"]) <= year_of(a) <= year_of(h["t2"]) for a in golds)
    return {k: dict(v) for k, v in s.items()}


def dump_split(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_wikidata(raw, out, ds, name, aliases):
    print(f"[{name}]")
    kg_ds = KG_SOURCE.get(ds, ds)
    label_dir = f"{raw}/data/{kg_ds}/kg"
    ent, rel, ts, ent_label, rel_label = wikidata_vocab(raw, kg_ds, label_dir)
    for n, m in (("entities", ent), ("relations", rel), ("timestamps", ts)):
        check_ids(n, m)
    os.makedirs(f"{out}/{name}/kg", exist_ok=True)
    os.makedirs(f"{out}/{name}/questions", exist_ok=True)
    write_vocab(f"{out}/{name}/kg/entities.tsv", [(v, k, ent_label.get(k, "")) for k, v in ent.items()])
    write_vocab(f"{out}/{name}/kg/relations.tsv", [(v, k, rel_label.get(k, "")) for k, v in rel.items()])
    write_vocab(f"{out}/{name}/kg/timestamps.tsv", [(v, str(k), None) for k, v in ts.items()])

    facts, skipped = [], Counter()
    for h, r, t, s, e in (p[:5] for p in read_facts(f"{raw}/data/{kg_ds}/kg/full.txt")):
        if h not in ent or t not in ent:
            skipped["entity_not_in_vocab"] += 1
            continue
        if r not in rel:
            skipped["relation_not_in_vocab"] += 1
            continue
        s, e = _int_or_none(s), _int_or_none(e)
        if s not in ts or e not in ts:
            skipped["timestamp_not_in_vocab"] += 1
            continue
        facts.append((ent[h], rel[r], ent[t], ts[s], ts[e]))
    with open(f"{out}/{name}/kg/facts.tsv", "w") as f:
        for fact in facts:
            f.write("\t".join(map(str, fact)) + "\n")

    if ds == "timequestions":
        placeholders_start, placeholders_end = {ts[min(ts)]}, {ts[max(ts)]}     # years 1 and 2916
    else:
        placeholders_start = placeholders_end = {ts[y] for y in (0, max(ts)) if y in ts}   # years 0 and 9620
    events = defaultdict(list)
    ev_rel = rel.get(WIKIDATA_EVENT[0])
    ev_obj = ent.get(WIKIDATA_EVENT[1])
    for h, r, t, s, e in facts:
        if r == ev_rel and t == ev_obj:
            events[h].append((s, e))
    year_of_id = {v: k for k, v in ts.items()}
    index = HintIndex(facts, placeholders_start, placeholders_end, events=events)

    tkbc_check = check_tkbc_alignment(f"{raw}/data/{kg_ds}/kg/tkbc_processed_data/{kg_ds}/train.pickle", facts)
    split_stats, hints = {}, {}
    for split in ("train", "valid", "test"):
        if ds == "timequestions":
            recs, st = convert_timequestions(raw, split, ent, ts, ent_label, aliases, index)
        else:
            recs, st = convert_cron(raw, ds, split, ent, ts, ent_label, aliases, index, lambda i: year_of_id[i])
        dump_split(f"{out}/{name}/questions/{split}.jsonl", recs)
        split_stats[split] = dict(st, questions=len(recs),
                                  unanswerable=sum(not r["answers"] for r in recs),
                                  answer_types=dict(Counter(r["answer_type"] for r in recs)))
        hints[split] = hint_stats(recs, lambda i: year_of_id[i])
        print(f"  {split}: {split_stats[split]}")

    link_embeddings(f"{raw}/models/{kg_ds}/kg_embeddings", f"{out}/{name}/embeddings")
    meta = {
        "name": name, "source": f"{raw}/data/{ds}",
        "num_entities": len(ent), "num_relations": len(rel), "num_timestamps": len(ts),
        "timestamp_format": "year",
        "placeholder_timestamps": {"start": sorted(placeholders_start), "end": sorted(placeholders_end)},
        "facts": len(facts), "facts_skipped": dict(skipped),
        "embeddings": check_embeddings(f"{out}/{name}/embeddings", len(ent), len(rel), len(ts)),
        "tkbc_alignment": tkbc_check,
        "splits": split_stats, "hint_stats": hints, "schema": QUESTION_SCHEMA,
    }
    json.dump(meta, open(f"{out}/{name}/meta.json", "w"), indent=1)
    print(f"  embeddings: {meta['embeddings']}\n  tkbc alignment: {tkbc_check}\n  facts skipped: {dict(skipped)}")


def build_multitq(raw, out):
    name = "multitq"
    print(f"[{name}]")
    kg = f"{raw}/data/MultiTQ/kg"
    # The TComplEx checkpoint was trained with entity2id.json (underscored names);
    # tkbc_processed_data/MultiTQ/ent_id assigns different ids to 357 entities and must not be used.
    ent = json.load(open(f"{kg}/entity2id.json", encoding="utf-8"))
    rel = json.load(open(f"{kg}/relation2id.json", encoding="utf-8"))
    ts = json.load(open(f"{kg}/ts2id.json", encoding="utf-8"))
    for n, m in (("entities", ent), ("relations", rel), ("timestamps", ts)):
        check_ids(n, m)
    os.makedirs(f"{out}/{name}/kg", exist_ok=True)
    os.makedirs(f"{out}/{name}/questions", exist_ok=True)
    write_vocab(f"{out}/{name}/kg/entities.tsv", [(v, k, k.replace("_", " ")) for k, v in ent.items()])
    write_vocab(f"{out}/{name}/kg/relations.tsv", [(v, k, k.replace("_", " ")) for k, v in rel.items()])
    write_vocab(f"{out}/{name}/kg/timestamps.tsv", [(v, k, None) for k, v in ts.items()])

    facts, skipped = [], Counter()
    for p in read_facts(f"{kg}/full.txt"):
        h, r, t, d = p[:4]
        if h not in ent or t not in ent or r not in rel or d not in ts:
            skipped["not_in_vocab"] += 1
            continue
        facts.append((ent[h], rel[r], ent[t], ts[d], ts[d]))
    with open(f"{out}/{name}/kg/facts.tsv", "w") as f:
        for fact in facts:
            f.write("\t".join(map(str, fact)) + "\n")

    day_ids = {v for k, v in ts.items() if len(k) == 10}
    index = HintIndex(facts, usable_ts=day_ids)
    tkbc_check = check_tkbc_alignment(f"{kg}/tkbc_processed_data/MultiTQ/train.pickle", facts)
    key_of = {v: k for k, v in ts.items()}
    split_stats, hints = {}, {}
    for split in ("train", "valid", "test"):
        recs, st = convert_multitq(raw, split, ent, ts, index)
        dump_split(f"{out}/{name}/questions/{split}.jsonl", recs)
        split_stats[split] = dict(st, questions=len(recs),
                                  unanswerable=sum(not r["answers"] for r in recs),
                                  answer_types=dict(Counter(r["answer_type"] for r in recs)))
        hints[split] = hint_stats(recs, lambda i: key_of[i][:10].ljust(10, "0"))
        print(f"  {split}: {split_stats[split]}")

    link_embeddings(f"{raw}/models/MultiTQ/kg_embeddings", f"{out}/{name}/embeddings")
    meta = {
        "name": name, "source": f"{raw}/data/MultiTQ",
        "num_entities": len(ent), "num_relations": len(rel), "num_timestamps": len(ts),
        "timestamp_format": "iso (day YYYY-MM-DD, month YYYY-MM, year YYYY)",
        "placeholder_timestamps": {"start": [], "end": []},
        "facts": len(facts), "facts_skipped": dict(skipped),
        "embeddings": check_embeddings(f"{out}/{name}/embeddings", len(ent), len(rel), len(ts)),
        "tkbc_alignment": tkbc_check,
        "entity_mentions": "predicted by Flair NER + linking (no gold annotation in MultiTQ)",
        "splits": split_stats, "hint_stats": hints, "schema": QUESTION_SCHEMA,
    }
    json.dump(meta, open(f"{out}/{name}/meta.json", "w"), indent=1)
    print(f"  embeddings: {meta['embeddings']}\n  tkbc alignment: {tkbc_check}\n  facts skipped: {dict(skipped)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="Data/raw")
    parser.add_argument("--out", default="Data/benchmarks")
    parser.add_argument("--only", nargs="*", default=None,
                        help="subset of: cronquestions complex_cronquestions timequestions multitq")
    args = parser.parse_args()
    todo = set(args.only or ["cronquestions", "complex_cronquestions", "timequestions", "multitq"])

    aliases = {k: set(v) for k, v in load_pickle(f"{args.raw}/data/wikidata_big/kg/wd_id_to_aliases.pickle").items()}
    if "cronquestions" in todo:
        build_wikidata(args.raw, args.out, "wikidata_big", "cronquestions", aliases)
    if "complex_cronquestions" in todo:
        build_wikidata(args.raw, args.out, "wikidata_big_complex", "complex_cronquestions", aliases)
    if "timequestions" in todo:
        build_wikidata(args.raw, args.out, "timequestions", "timequestions", aliases)
    if "multitq" in todo:
        build_multitq(args.raw, args.out)


if __name__ == "__main__":
    main()
