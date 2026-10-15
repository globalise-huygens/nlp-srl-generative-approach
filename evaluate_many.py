#!/usr/bin/env python3
"""
Run the SRL evaluation (evaluate_srl.py) over several files and compare them.

Three ways to say which files belong together:

  1. One gold file, several prediction files (different models, prompts, runs):
         python evaluate_many.py gold.tsv run1.tsv run2.tsv "runs/*.tsv"

  2. Several (gold, prediction) pairs, e.g. one pair per document; after the
     flag, the files alternate gold, pred, gold, pred, ...:
         python evaluate_many.py --pairs gold_a.tsv pred_a.tsv gold_b.tsv pred_b.tsv

  3. Two directories whose files have the same names (only *.tsv is used):
         python evaluate_many.py --dirs gold_dir/ pred_dir/

Output:
  * a comparison table with one row per prediction file (strict span micro
    P/R/F1, macro F1, lenient P/R, token-role F1 (ignores B-/I-), tag F1 (strict
    token level: B-/I- must match), block exact match, skipped blocks);
  * a second table for the STRICT TOKEN (tag) LEVEL, where B-/I- must match:
    micro and MACRO precision/recall/F1 per file (macro = unweighted mean over
    the tag classes B-Agent, I-Agent, ... that occur in gold or prediction);
  * for several files: mean and standard deviation over the files; in modes 2
    and 3 also a POOLED row (counts summed over all pairs, then scored), which
    is the right overall number when the gold files differ;
  * a per-role table of strict F1 per file.
Matching of blocks, metrics, --rename and --exclude work exactly as in
evaluate_srl.py (see there).

Options:
  --auto-failed-json   for every prediction x.tsv use x.json (written by
                       srl_labelling.py) if it exists, and leave blocks with
                       failed events out of the scoring
  --rename OLD=NEW ... --exclude doc:reg:anno ...
  --csv summary.csv    one row per file, incl. strict F1 per role
  --json reports.json  the full evaluate_srl report of every file
  --errors-dir DIR     one error-analysis text file per prediction file
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

from evaluate_srl_strict import evaluate, write_errors, summarize, summarize_tags, COUNT_FIELDS


# ---------------------------------------------------------------------------
# Collecting jobs
# ---------------------------------------------------------------------------

def expand(patterns):
    """Expand wildcards ourselves (the Windows shell does not)."""
    out = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            hits = sorted(glob.glob(pat))
            if not hits:
                print(f"[warn] no files match {pat!r}")
            out.extend(hits)
        else:
            out.append(pat)
    return out


def make_labels(preds):
    """File stems as labels; fall back to the full path if stems collide."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in preds]
    return [st if stems.count(st) == 1 else os.path.splitext(p)[0] for st, p in zip(stems, preds)]


def collect_jobs(a):
    """-> (jobs [(gold, pred)], mode) where mode is 'runs' or 'pairs'."""
    modes = sum([bool(a.pairs), bool(a.dirs), bool(a.files)])
    if modes != 1:
        sys.exit("Choose exactly one input mode: GOLD PRED [PRED ...]  |  --pairs g1 p1 g2 p2 ...  |  --dirs GOLD_DIR PRED_DIR")
    if a.pairs:
        if len(a.pairs) % 2:
            sys.exit("--pairs needs an even number of files: gold1 pred1 gold2 pred2 ...")
        return [(a.pairs[i], a.pairs[i + 1]) for i in range(0, len(a.pairs), 2)], "pairs"
    if a.dirs:
        gdir, pdir = a.dirs
        jobs = []
        for pred in sorted(glob.glob(os.path.join(pdir, "*.tsv"))):
            gold = os.path.join(gdir, os.path.basename(pred))
            if os.path.exists(gold):
                jobs.append((gold, pred))
            else:
                print(f"[warn] no gold file named {os.path.basename(pred)} in {gdir}; skipped")
        return jobs, "pairs"
    files = expand(a.files)
    if len(files) < 2:
        sys.exit("Need one gold file followed by at least one prediction file.")
    return [(files[0], p) for p in files[1:]], "runs"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def row_from_report(label, rep):
    b, mi, ma = rep["blocks"], rep["micro"], rep["macro"]
    skipped = (len(b["only_gold"]) + len(b["only_pred"]) + len(b["event_mismatch"])
               + len(b["token_mismatch"]))
    row = {"name": label, "scored": b["scored"],
           "exact": b["exact_match"] / b["scored"] if b["scored"] else None,
           "strict_p": mi["strict"]["p"], "strict_r": mi["strict"]["r"], "strict_f1": mi["strict"]["f"],
           "macro_f1": ma["strict_f1"],
           "lenient_p": mi["lenient"]["p"], "lenient_r": mi["lenient"]["r"],
           "token_f1": mi["token"]["f"],
           "tag_p": rep["tag_level"]["micro"]["p"], "tag_r": rep["tag_level"]["micro"]["r"],
           "tag_f1": rep["tag_level"]["micro"]["f"],
           "macro_tag_p": rep["tag_level"]["macro"]["p"], "macro_tag_r": rep["tag_level"]["macro"]["r"],
           "macro_tag_f1": rep["tag_level"]["macro"]["f"],
           "failed": b["n_excluded_failed"], "skipped": skipped}
    if rep["totals"]["gold_spans"] == 0 and rep["totals"]["pred_spans"] == 0:
        # no roles in gold and none predicted: scores are undefined, not 0
        for k in ("strict_p", "strict_r", "strict_f1", "macro_f1", "lenient_p", "lenient_r", "token_f1",
                  "tag_p", "tag_r", "tag_f1", "macro_tag_p", "macro_tag_r", "macro_tag_f1"):
            row[k] = None
    return row


METRICS = [("exact", "exact"), ("strict_p", "P"), ("strict_r", "R"), ("strict_f1", "F1"),
           ("macro_f1", "macF1"), ("lenient_p", "lenP"), ("lenient_r", "lenR"), ("token_f1", "tokF1"), ("tag_f1", "tagF1")]


def fmt(v, width=7):
    return f"{v:>{width}.3f}" if v is not None else f"{'n/a':>{width}}"


TAG_METRICS = [("tag_p", "P"), ("tag_r", "R"), ("tag_f1", "F1"),
               ("macro_tag_p", "P"), ("macro_tag_r", "R"), ("macro_tag_f1", "F1")]


def print_table(rows, pooled=None):
    w = max([len(r["name"]) for r in rows] + [6, len("pooled")]) + 2
    head = f"{'file':<{w}}{'scored':>7}" + "".join(f"{h:>7}" for _, h in METRICS) + f"{'failed':>7}{'skip':>6}"
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['name']:<{w}}{r['scored']:>7}" + "".join(fmt(r[k]) for k, _ in METRICS)
              + f"{r['failed']:>7}{r['skipped']:>6}")
    print("-" * len(head))
    print_aggregates(rows, pooled, METRICS, w, lead=f"{'':>7}")
    print("failed = blocks left out because a prediction failed (--auto-failed-json); "
          "skip = unmatched / event-mismatch / token-mismatch blocks;\n"
          "n/a = no roles in gold or prediction (left out of mean and std);\n"
          "tokF1 ignores B-/I-, tagF1 needs the full tag to match (strict token level)")


def print_aggregates(rows, pooled, metrics, w, lead):
    """mean / std over files (n/a files left out) and, if given, the pooled row."""
    if len(rows) > 1:
        vals = {k: [r[k] for r in rows if r[k] is not None] for k, _ in metrics}
        means = {k: statistics.mean(v) if v else None for k, v in vals.items()}
        stds = {k: statistics.stdev(v) if len(v) > 1 else None for k, v in vals.items()}
        print(f"{'mean':<{w}}{lead}" + "".join(fmt(means[k]) for k, _ in metrics))
        print(f"{'std':<{w}}{lead}" + "".join(fmt(stds[k]) for k, _ in metrics))
    if pooled:
        print(f"{'pooled':<{w}}{pooled['scored']:>7}" + "".join(fmt(pooled[k]) for k, _ in metrics))


def print_tag_table(rows, pooled=None):
    """Strict token level: full tag (B-/I- + role) must match. Micro and macro P/R/F1."""
    w = max([len(r["name"]) for r in rows] + [6, len("pooled")]) + 2
    print("\nSTRICT TOKEN LEVEL (tag level: B-/I- must match; O not scored)")
    print(f"{'':<{w}}{'':>7}{'micro':^21}{'macro (over tag classes)':^21}")
    head = f"{'file':<{w}}{'scored':>7}" + "".join(f"{h:>7}" for _, h in TAG_METRICS)
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['name']:<{w}}{r['scored']:>7}" + "".join(fmt(r[k]) for k, _ in TAG_METRICS))
    print("-" * len(head))
    print_aggregates(rows, pooled, TAG_METRICS, w, lead=f"{'':>7}")


def print_role_table(labels, reports):
    roles = sorted({r["role"] for rep in reports for r in rep["per_role"]})
    if not roles:
        return
    print("\nStrict F1 per role   (- = role absent in gold and prediction)")
    for i, lab in enumerate(labels, 1):
        print(f"  #{i} = {lab}")
    head = f"{'role':<14}" + "".join(f"{'#' + str(i):>7}" for i in range(1, len(labels) + 1))
    if len(labels) > 1:
        head += f"{'mean':>7}"
    print(head)
    print("-" * len(head))
    for role in roles:
        vals = []
        for rep in reports:
            hit = next((r for r in rep["per_role"] if r["role"] == role), None)
            vals.append(hit["strict_f1"] if hit and (hit["gold_spans"] or hit["pred_spans"]) else None)
        line = f"{role:<14}" + "".join(f"{v:>7.3f}" if v is not None else f"{'-':>7}" for v in vals)
        present = [v for v in vals if v is not None]
        if len(labels) > 1:
            line += f"{statistics.mean(present):>7.3f}" if present else f"{'-':>7}"
        print(line)


def pool(reports):
    """Sum the raw counts of all reports, then score once."""
    cd = {}
    for rep in reports:
        for role, c in rep["counts"].items():
            tgt = cd.setdefault(role, {f: 0 for f in COUNT_FIELDS})
            for f in COUNT_FIELDS:
                tgt[f] += c[f]
    rows, micro, macro_d, _ = summarize(cd)
    tc = {}
    for rep_ in reports:
        for tag, c in rep_["tag_counts"].items():
            tgt = tc.setdefault(tag, {"tp": 0, "fp": 0, "fn": 0})
            for f in tgt:
                tgt[f] += c[f]
    tl = summarize_tags(tc)
    scored = sum(r["blocks"]["scored"] for r in reports)
    exact = sum(r["blocks"]["exact_match"] for r in reports)
    return {"scored": scored, "exact": exact / scored if scored else 0.0,
            "strict_p": micro["strict"]["p"], "strict_r": micro["strict"]["r"], "strict_f1": micro["strict"]["f"],
            "macro_f1": macro_d["strict_f1"], "lenient_p": micro["lenient"]["p"],
            "lenient_r": micro["lenient"]["r"], "token_f1": micro["token"]["f"],
            "tag_p": tl["micro"]["p"], "tag_r": tl["micro"]["r"], "tag_f1": tl["micro"]["f"],
            "macro_tag_p": tl["macro"]["p"], "macro_tag_r": tl["macro"]["r"],
            "macro_tag_f1": tl["macro"]["f"]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="GOLD PRED [PRED ...]")
    ap.add_argument("--pairs", nargs="+", metavar="FILE")
    ap.add_argument("--dirs", nargs=2, metavar=("GOLD_DIR", "PRED_DIR"))
    ap.add_argument("--auto-failed-json", action="store_true")
    ap.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="doc:reg:anno")
    ap.add_argument("--csv"); ap.add_argument("--json"); ap.add_argument("--errors-dir")
    a = ap.parse_args()

    jobs, mode = collect_jobs(a)
    if not jobs:
        sys.exit("No files to evaluate.")
    labels = make_labels([p for _, p in jobs])
    rename = dict(x.split("=", 1) for x in a.rename)

    reports, used_labels, rows = [], [], []
    for (gold, pred), label in zip(jobs, labels):
        fj = None
        if a.auto_failed_json:
            cand = os.path.splitext(pred)[0] + ".json"
            if os.path.exists(cand):
                fj = cand
            else:
                print(f"[warn] {label}: no {os.path.basename(cand)} next to the prediction; failed events count as empty")
        print(f"[run] {label}: {gold} vs {pred}" + (f" (failed events from {fj})" if fj else ""))
        try:
            rep, errors = evaluate(gold, pred, rename=rename, exclude=a.exclude, failed_json=fj)
        except Exception as err:   # keep going if one file is broken
            print(f"[error] {label}: {type(err).__name__}: {err}")
            continue
        reports.append(rep); used_labels.append(label); rows.append(row_from_report(label, rep))
        if a.errors_dir:
            os.makedirs(a.errors_dir, exist_ok=True)
            write_errors(os.path.join(a.errors_dir, f"{label.replace(os.sep, '_')}.errors.txt"), errors)

    if not reports:
        sys.exit("No file could be evaluated.")

    pooled = pool(reports) if mode == "pairs" and len(reports) > 1 else None
    print_table(rows, pooled)
    print_tag_table(rows, pooled)
    print_role_table(used_labels, reports)

    if a.csv:
        roles = sorted({r["role"] for rep in reports for r in rep["per_role"]})
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "scored", "exact_match", "strict_p", "strict_r", "strict_f1", "macro_strict_f1",
                        "lenient_p", "lenient_r", "token_f1", "tag_p", "tag_r", "tag_f1", "macro_tag_p", "macro_tag_r", "macro_tag_f1",
                        "excluded_failed", "skipped"]
                       + [f"f1_{r}" for r in roles])
            for row, rep in zip(rows, reports):
                f1 = {r["role"]: r["strict_f1"] for r in rep["per_role"] if r["gold_spans"] or r["pred_spans"]}
                w.writerow([row["name"], row["scored"]] + [f"{row[k]:.4f}" if row[k] is not None else "" for k in
                           ("exact", "strict_p", "strict_r", "strict_f1", "macro_f1", "lenient_p", "lenient_r", "token_f1",
                            "tag_p", "tag_r", "tag_f1", "macro_tag_p", "macro_tag_r", "macro_tag_f1")]
                           + [row["failed"], row["skipped"]]
                           + [f"{f1[r]:.4f}" if r in f1 else "" for r in roles])
        print(f"\n[done] wrote {a.csv}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": rows, "pooled": pooled,
                       "files": {lab: rep for lab, rep in zip(used_labels, reports)}}, f, ensure_ascii=False, indent=1)
        print(f"[done] wrote {a.json}")
    if a.errors_dir:
        print(f"[done] error files in {a.errors_dir}/")


if __name__ == "__main__":
    main()
