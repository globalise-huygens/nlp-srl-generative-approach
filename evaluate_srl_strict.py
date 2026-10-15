#!/usr/bin/env python3
"""
Evaluate predicted semantic role labels against gold annotations.

Both files use the pipeline's TSV format (one token per line):
    doc  reg  anno  tok  token  event  role  [entity]

MATCHING. Every block is identified by its (doc, reg, anno) key; a region with
several events has a different anno number per event. A gold block and a
predicted block are only compared if the key matches AND the event annotation
(the whole event column) is identical. Otherwise the block is reported but not
scored: key missing on one side, or same key with a different event annotation.
Matched blocks whose tokens differ are skipped and reported too.

Metrics (computed over all matched blocks):
  * strict span F1   -- role AND exact boundaries must match (seqeval-style: a
                        stray I-X after O/another role starts a new span, so an
                        I-X in place of B-X can still give a correct span)
  * lenient span P/R -- same role and any token overlap
  * token role F1    -- role of each non-O token, IGNORING B-/I-
  * TAG-LEVEL F1     -- token level in the strict sense: the full tag must match,
                        so predicting I-Patient where gold has B-Patient is an
                        error (one error for I-Patient, one miss for B-Patient).
                        Computed per tag class (B-Agent, I-Agent, ...); O is not
                        scored, so the many O tokens do not inflate the numbers.
  * block exact match, tag accuracy incl. O, plus a per-event-type breakdown

Usage:
    python evaluate_srl.py gold.tsv pred.tsv
        [--json report.json] [--errors errors.txt]
        [--rename Benefactive=Beneficiary] [--exclude doc_1139:reg_0:anno_1 ...]
        [--failed-json predictions.json]

--rename maps a role name in BOTH files before scoring (e.g. if the gold data
says "Benefactive" but the label set says "Beneficiary").
--exclude skips regions by key, e.g. the regions used as few-shot examples.
--failed-json takes the JSON written by srl_labelling.py. Blocks that contain an
event whose prediction FAILED (written as all-"O" in the TSV) are then left out
of the scoring and counted separately. Without this flag, failed events count
as "predicted nothing", which lowers recall.
"""

import sys
import json
import argparse
from collections import OrderedDict, defaultdict, Counter


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_blocks(path):
    """Group token lines into blocks keyed by (doc, reg, anno)."""
    blocks, skipped, dup_ids = OrderedDict(), 0, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                skipped += 1
                continue
            doc, reg, anno, tok, token, event, role = parts[:7]
            key = (doc, reg, anno)
            b = blocks.setdefault(key, {"key": key, "ids": set(), "tokens": [], "events": [], "roles": []})
            if tok in b["ids"]:
                dup_ids.add(key)
            b["ids"].add(tok)
            b["tokens"].append(token)
            b["events"].append(event)
            b["roles"].append(role)
    if skipped:
        print(f"[warn] {path}: skipped {skipped} lines with fewer than 7 columns "
              f"(is the role column present?)")
    if dup_ids:
        print(f"[warn] {path}: repeated token ids within {len(dup_ids)} key(s), "
              f"e.g. {':'.join(sorted(dup_ids)[0])}; is the anno number really unique per event?")
    return list(blocks.values())


def make_normalizer(rename):
    def norm(tag):
        if tag[:2] in ("B-", "I-"):
            return tag[:2] + rename.get(tag[2:], tag[2:])
        return tag
    return norm


def role_of(tag):
    return tag[2:] if tag[:2] in ("B-", "I-") else tag


def get_spans(tags):
    """BIO -> list of (role, start, end). A stray I-X starts a new span."""
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t in ("O", ""):
            if cur:
                spans.append(tuple(cur))
                cur = None
        elif t.startswith("B-") or cur is None or cur[0] != role_of(t):
            if cur:
                spans.append(tuple(cur))
            cur = [role_of(t), i, i]
        else:
            cur[2] = i
    if cur:
        spans.append(tuple(cur))
    return spans


def event_types(events):
    out = []
    for t in events:
        if t != "O":
            r = role_of(t)
            if r not in out:
                out.append(r)
    return "+".join(out) if out else "(none)"


def overlaps(a, b):
    return a[1] <= b[2] and b[1] <= a[2]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Matching gold blocks to predicted blocks
# ---------------------------------------------------------------------------

def match_blocks(gold, pred):
    """Pair blocks with the same (doc, reg, anno) key AND identical event column.
    Returns (pairs, only_gold, only_pred, event_mismatch)."""
    p_by_key = {b["key"]: b for b in pred}
    g_keys = {b["key"] for b in gold}
    pairs, only_gold, event_mismatch = [], [], []
    for g in gold:
        p = p_by_key.get(g["key"])
        if p is None:
            only_gold.append(g)
        elif p["events"] != g["events"]:
            event_mismatch.append(g["key"])
        else:
            pairs.append((g, p))
    only_pred = [b for b in pred if b["key"] not in g_keys]
    return pairs, only_gold, only_pred, event_mismatch


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class Counts:
    def __init__(self):
        self.s_tp = Counter(); self.s_fp = Counter(); self.s_fn = Counter()   # strict spans
        self.l_ptp = Counter(); self.n_pred = Counter()                      # lenient precision
        self.l_rtp = Counter(); self.n_gold = Counter()                      # lenient recall
        self.t_tp = Counter(); self.t_fp = Counter(); self.t_fn = Counter()  # token level (role only)
        self.tag_tp = Counter(); self.tag_fp = Counter(); self.tag_fn = Counter()  # tag level (B-/I- strict)


def score_block(g_tags, p_tags, c, ev_counts):
    g_sp, p_sp = get_spans(g_tags), get_spans(p_tags)
    g_set, p_set = set(g_sp), set(p_sp)
    for r, st, en in g_sp:
        c.n_gold[r] += 1
        if (r, st, en) in p_set:
            c.s_tp[r] += 1; ev_counts["tp"] += 1
        else:
            c.s_fn[r] += 1; ev_counts["fn"] += 1
        if any(pr == r and overlaps((r, st, en), (pr, ps, pe)) for pr, ps, pe in p_sp):
            c.l_rtp[r] += 1
    for r, st, en in p_sp:
        c.n_pred[r] += 1
        if (r, st, en) not in g_set:
            c.s_fp[r] += 1; ev_counts["fp"] += 1
        if any(gr == r and overlaps((r, st, en), (gr, gs, ge)) for gr, gs, ge in g_sp):
            c.l_ptp[r] += 1
    for g, p in zip(g_tags, p_tags):
        gr, pr = role_of(g), role_of(p)
        if gr == pr and gr != "O":
            c.t_tp[gr] += 1
        elif gr != pr:
            if pr != "O":
                c.t_fp[pr] += 1
            if gr != "O":
                c.t_fn[gr] += 1
        # strict tag level: the whole tag (prefix + role) must be identical
        if g == p:
            if g != "O":
                c.tag_tp[g] += 1
        else:
            if p != "O":
                c.tag_fp[p] += 1
            if g != "O":
                c.tag_fn[g] += 1


def macro(rows, key):
    vals = [r[key] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


COUNT_FIELDS = ("s_tp", "s_fp", "s_fn", "l_ptp", "n_pred", "l_rtp", "n_gold", "t_tp", "t_fp", "t_fn")
MACRO_KEYS = ("strict_p", "strict_r", "strict_f1", "lenient_p", "lenient_r", "token_p", "token_r", "token_f1")


def counts_to_dict(c):
    """Counts -> {role: {field: int}} (plain numbers, so counts can be pooled across files)."""
    roles = sorted(set().union(*[set(getattr(c, f)) for f in COUNT_FIELDS]))
    return {r: {f: getattr(c, f)[r] for f in COUNT_FIELDS} for r in roles}


def summarize(cd):
    """Per-role rows, micro and macro scores from {role: counts}. Works for one
    file or for counts summed over several files (pooling)."""
    rows = []
    for r in sorted(cd):
        c = cd[r]
        sp, sr, sf = prf(c["s_tp"], c["s_fp"], c["s_fn"])
        lp = c["l_ptp"] / c["n_pred"] if c["n_pred"] else 0.0
        lr = c["l_rtp"] / c["n_gold"] if c["n_gold"] else 0.0
        tp_, tr_, tf_ = prf(c["t_tp"], c["t_fp"], c["t_fn"])
        rows.append({"role": r, "gold_spans": c["n_gold"], "pred_spans": c["n_pred"],
                     "strict_p": sp, "strict_r": sr, "strict_f1": sf,
                     "lenient_p": lp, "lenient_r": lr,
                     "token_p": tp_, "token_r": tr_, "token_f1": tf_})
    tot = {f: sum(c[f] for c in cd.values()) for f in COUNT_FIELDS}
    micro = {"strict": dict(zip("prf", prf(tot["s_tp"], tot["s_fp"], tot["s_fn"]))),
             "lenient": {"p": tot["l_ptp"] / tot["n_pred"] if tot["n_pred"] else 0.0,
                         "r": tot["l_rtp"] / tot["n_gold"] if tot["n_gold"] else 0.0},
             "token": dict(zip("prf", prf(tot["t_tp"], tot["t_fp"], tot["t_fn"])))}
    macro_d = {k: macro(rows, k) for k in MACRO_KEYS}
    return rows, micro, macro_d, tot


def tag_counts_to_dict(c):
    tags = sorted(set(c.tag_tp) | set(c.tag_fp) | set(c.tag_fn),
                  key=lambda t: (role_of(t), t[:2] != "B-", t))   # B- before I- within a role
    return {t: {"tp": c.tag_tp[t], "fp": c.tag_fp[t], "fn": c.tag_fn[t]} for t in tags}


def summarize_tags(tc):
    """Strict tag-level scores from {tag: {tp, fp, fn}} (also usable for pooled counts)."""
    rows = []
    for t, c in tc.items():
        p_, r_, f_ = prf(c["tp"], c["fp"], c["fn"])
        rows.append({"tag": t, "gold": c["tp"] + c["fn"], "pred": c["tp"] + c["fp"],
                     "p": p_, "r": r_, "f1": f_})
    tp = sum(c["tp"] for c in tc.values()); fp = sum(c["fp"] for c in tc.values())
    fn = sum(c["fn"] for c in tc.values())
    micro = dict(zip("prf", prf(tp, fp, fn)))
    macro_d = {"p": macro(rows, "p"), "r": macro(rows, "r"), "f": macro(rows, "f1")}
    return {"per_tag": rows, "micro": micro, "macro": macro_d, "gold": tp + fn, "pred": tp + fp}


def evaluate(gold_path, pred_path, rename=None, exclude=None, failed_json=None):
    """Score one predicted TSV against one gold TSV. Returns (report, errors)."""
    norm = make_normalizer(rename or {})
    excluded = {tuple(x.split(":")) for x in (exclude or [])}

    failed_keys = set()
    if failed_json:
        with open(failed_json, encoding="utf-8") as f:
            failed_keys = {tuple(j["key"]) for j in json.load(f)
                           if any(ev.get("failed") for ev in j["events"])}
    skip = excluded | failed_keys
    gold_all = read_blocks(gold_path)
    n_failed_gold = sum(1 for b in gold_all if b["key"] in failed_keys)
    gold = [b for b in gold_all if b["key"] not in skip]
    pred = [b for b in read_blocks(pred_path) if b["key"] not in skip]
    for b in gold + pred:
        b["roles"] = [norm(t) for t in b["roles"]]

    pairs, only_gold, only_pred, event_mismatch = match_blocks(gold, pred)

    counts, by_event = Counts(), defaultdict(lambda: {"blocks": 0, "tp": 0, "fp": 0, "fn": 0})
    confusion, tag_confusion = defaultdict(Counter), defaultdict(Counter)
    scored, tok_mismatch, exact, n_tokens, tok_correct = 0, [], 0, 0, 0
    errors = []
    for g, p in pairs:
        if g["tokens"] != p["tokens"]:
            tok_mismatch.append(g["key"])
            continue
        scored += 1
        et = event_types(g["events"])
        by_event[et]["blocks"] += 1
        score_block(g["roles"], p["roles"], counts, by_event[et])
        for gt, pt in zip(g["roles"], p["roles"]):
            confusion[role_of(gt)][role_of(pt)] += 1
            tag_confusion[gt][pt] += 1
            n_tokens += 1
            tok_correct += gt == pt   # exact tag incl. B-/I-
        if g["roles"] == p["roles"]:
            exact += 1
        else:
            errors.append((g, p, et))

    cd = counts_to_dict(counts)
    rows, micro, macro_d, tot = summarize(cd)
    tc = tag_counts_to_dict(counts)
    tag_level = summarize_tags(tc)

    ev_rows = []
    for et, d in sorted(by_event.items(), key=lambda kv: -kv[1]["blocks"]):
        p_, r_, f_ = prf(d["tp"], d["fp"], d["fn"])
        ev_rows.append({"event": et, "blocks": d["blocks"], "tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
                        "p": p_, "r": r_, "f1": f_})

    report = {"blocks": {"gold": len(gold), "pred": len(pred), "matched": len(pairs), "scored": scored,
                         "only_gold": [list(b["key"]) for b in only_gold],
                         "only_pred": [list(b["key"]) for b in only_pred],
                         "event_mismatch": [list(k) for k in event_mismatch],
                         "token_mismatch": [list(k) for k in tok_mismatch],
                         "exact_match": exact,
                         "tag_accuracy": tok_correct / n_tokens if n_tokens else 0.0,
                         "n_excluded_failed": n_failed_gold,
                         "excluded_failed": sorted(":".join(k) for k in failed_keys)},
              "micro": micro, "macro": macro_d,
              "totals": {"gold_spans": tot["n_gold"], "pred_spans": tot["n_pred"]},
              "per_role": rows, "per_event": ev_rows, "counts": cd,
              "tag_level": tag_level, "tag_counts": tc,
              "token_confusion_gold_to_pred": {g: dict(c) for g, c in confusion.items()},
              "tag_confusion_gold_to_pred": {g: dict(c) for g, c in tag_confusion.items()}}
    return report, errors


def print_report(rep, show_failed=False):
    b, mi, ma = rep["blocks"], rep["micro"], rep["macro"]
    print(f"\nBlocks: gold {b['gold']}, predicted {b['pred']}, matched {b['matched']}, scored {b['scored']}")
    if show_failed:
        print(f"  excluded because a prediction FAILED:                  {b['n_excluded_failed']}")
    print(f"  only in gold (key missing in pred):                    {len(b['only_gold'])}")
    print(f"  only in pred (key missing in gold):                    {len(b['only_pred'])}")
    print(f"  same key, different event annotation (skipped):        {len(b['event_mismatch'])}")
    print(f"  token mismatch between matched blocks (skipped):       {len(b['token_mismatch'])}")
    if b["scored"]:
        print(f"  block exact match: {b['exact_match']}/{b['scored']} = {b['exact_match'] / b['scored']:.3f}")
        print(f"  tag accuracy incl. O (exact B-/I-): {b['tag_accuracy']:.3f}")

    hdr = f"{'role':<14}{'gold':>6}{'pred':>6} | {'P':>6}{'R':>6}{'F1':>6} | {'lenP':>6}{'lenR':>6} | {'tokP':>6}{'tokR':>6}{'tokF1':>6}"
    print("\nPer role   (strict spans | lenient overlap | token level, ignoring B/I)")
    print(hdr); print("-" * len(hdr))
    for r in rep["per_role"]:
        print(f"{r['role']:<14}{r['gold_spans']:>6}{r['pred_spans']:>6} | "
              f"{r['strict_p']:>6.3f}{r['strict_r']:>6.3f}{r['strict_f1']:>6.3f} | "
              f"{r['lenient_p']:>6.3f}{r['lenient_r']:>6.3f} | "
              f"{r['token_p']:>6.3f}{r['token_r']:>6.3f}{r['token_f1']:>6.3f}")
    print("-" * len(hdr))
    print(f"{'micro':<14}{rep['totals']['gold_spans']:>6}{rep['totals']['pred_spans']:>6} | "
          f"{mi['strict']['p']:>6.3f}{mi['strict']['r']:>6.3f}{mi['strict']['f']:>6.3f} | "
          f"{mi['lenient']['p']:>6.3f}{mi['lenient']['r']:>6.3f} | "
          f"{mi['token']['p']:>6.3f}{mi['token']['r']:>6.3f}{mi['token']['f']:>6.3f}")
    print(f"{'macro':<14}{'':>6}{'':>6} | "
          f"{ma['strict_p']:>6.3f}{ma['strict_r']:>6.3f}{ma['strict_f1']:>6.3f} | "
          f"{ma['lenient_p']:>6.3f}{ma['lenient_r']:>6.3f} | "
          f"{ma['token_p']:>6.3f}{ma['token_r']:>6.3f}{ma['token_f1']:>6.3f}")

    tl = rep["tag_level"]
    hdr2 = f"{'tag':<22}{'gold':>6}{'pred':>6} | {'P':>6}{'R':>6}{'F1':>6}"
    print("\nPer tag   (token level, STRICT: full tag incl. B-/I- must match; O not scored)")
    print(hdr2); print("-" * len(hdr2))
    for r in tl["per_tag"]:
        print(f"{r['tag']:<22}{r['gold']:>6}{r['pred']:>6} | {r['p']:>6.3f}{r['r']:>6.3f}{r['f1']:>6.3f}")
    print("-" * len(hdr2))
    print(f"{'micro':<22}{tl['gold']:>6}{tl['pred']:>6} | "
          f"{tl['micro']['p']:>6.3f}{tl['micro']['r']:>6.3f}{tl['micro']['f']:>6.3f}")
    print(f"{'macro (over tags)':<22}{'':>6}{'':>6} | "
          f"{tl['macro']['p']:>6.3f}{tl['macro']['r']:>6.3f}{tl['macro']['f']:>6.3f}")

    print("\nPer event type (strict span micro)")
    print(f"{'event':<28}{'blocks':>7}{'P':>7}{'R':>7}{'F1':>7}")
    for d in rep["per_event"]:
        if d["tp"] + d["fp"] + d["fn"] == 0:   # no gold and no predicted spans at all
            print(f"{d['event']:<28}{d['blocks']:>7}{'n/a':>7}{'n/a':>7}{'n/a':>7}  (no roles in gold or prediction)")
        else:
            print(f"{d['event']:<28}{d['blocks']:>7}{d['p']:>7.3f}{d['r']:>7.3f}{d['f1']:>7.3f}")


def write_errors(path, errors):
    with open(path, "w", encoding="utf-8") as f:
        for g, p, et in errors:
            f.write(f"== {':'.join(g['key'])}  event={et}\n")
            gs, ps = set(get_spans(g["roles"])), set(get_spans(p["roles"]))
            for r, st, en in sorted(gs - ps, key=lambda x: x[1]):
                f.write(f"  MISSED   {r:<12} [{st}-{en}] {' '.join(g['tokens'][st:en + 1])}\n")
            for r, st, en in sorted(ps - gs, key=lambda x: x[1]):
                f.write(f"  SPURIOUS {r:<12} [{st}-{en}] {' '.join(p['tokens'][st:en + 1])}\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold"); ap.add_argument("pred")
    ap.add_argument("--json", help="write the full report as JSON")
    ap.add_argument("--errors", help="write mismatching blocks as text for error analysis")
    ap.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="doc:reg:anno")
    ap.add_argument("--failed-json", help="predictions JSON from srl_labelling.py; skip blocks with failed events")
    a = ap.parse_args()

    report, errors = evaluate(a.gold, a.pred, rename=dict(x.split("=", 1) for x in a.rename),
                              exclude=a.exclude, failed_json=a.failed_json)
    print_report(report, show_failed=bool(a.failed_json))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"\n[done] wrote {a.json}")
    if a.errors:
        write_errors(a.errors, errors)
        print(f"[done] wrote {a.errors} ({len(errors)} blocks with differences)")


if __name__ == "__main__":
    main()
