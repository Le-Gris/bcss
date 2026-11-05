"""Print the descriptive counts reported in the Results and Methods.

Covers the human-data numbers that come from counting rather than from a
fitted model: trivial-error filtering, dataset coverage, reproduction rates
(per error and attempt-weighted), participants and attempts modelled, programs
per error, effective edits, exact-match capture, alternative-rule programs,
sampling and labelling effort, and the exhaustive depth-1 baseline.

An error counts as reproduced when at least one of its programs is kept by
``filter_candidates.py``.  Evaluation-split errors are restricted to tasks
with a synthesized reference program.

Reads OSF files only; prints to stdout.

Usage:
    uv run python scripts/paper_counts.py --osf-dir OSF
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.rule_types import ALTERNATIVE, rule_type  # noqa: E402
from src.utils import grid_hash, record_id  # noqa: E402

SPLITS = ("training", "evaluation")
STEP_BUDGETS = (2, 4, 8)

# Prefilter reason -> display bucket (mutually exclusive; sums to total dropped).
PREFILTER_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("Near-miss (≈ correct output)", ("ground_truth",)),
    ("Test-input copy (identity)", ("identity",)),
    ("Training-example copy", ("train_input_copy", "train_output_copy")),
    ("Empty 3×3 grid", ("blank_grid",)),
]


# ── Loading ──────────────────────────────────────────────────────────────────
def _iter_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _participants(descriptions: list[dict]) -> set[str]:
    """Respondent ids behind an error (``response_id = <hashed_id>_attempt<N>``)."""
    return {
        str(desc.get("response_id", "")).rsplit("_attempt", 1)[0]
        for desc in descriptions
        if desc.get("response_id")
    }


def load_errors(path: Path, keep_tasks: set[str] | None = None) -> dict[str, dict]:
    """error_id -> {task_id, attempt_count, participants, grid_key}, optionally gated."""
    out: dict[str, dict] = {}
    for d in _iter_jsonl(path):
        if keep_tasks is not None and d["task_id"] not in keep_tasks:
            continue
        out[d["error_id"]] = {
            "task_id": d["task_id"],
            "attempt_count": int(d["attempt_count"]),
            "participants": _participants(d.get("descriptions", [])),
            "grid_key": (d["task_id"], grid_hash(d["output_grid"])),
        }
    return out


def load_rows(path: Path) -> list[dict]:
    return list(_iter_jsonl(path))


def count_rows(paths: list[Path]) -> int:
    return sum(sum(1 for _ in _iter_jsonl(p)) for p in paths)


# ── Reproduction and program distribution ───────────────────────────────────
def summarize(errors: dict[str, dict], rows: list[dict]) -> dict:
    """Coverage, reproduction and program-distribution stats for one split."""
    by_err: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["error_id"] in errors:
            by_err[r["error_id"]].append(r)

    rep = set(by_err)
    exact = {e for e, rs in by_err.items() if any(r.get("exact_match") for r in rs)}

    base = {
        "tasks": len({v["task_id"] for v in errors.values()}),
        "rep_tasks": len({errors[e]["task_id"] for e in rep}),
        "errors": len(errors),
        "reproduced": len(rep),
        "exact": len(exact),
        "total_attempts": sum(v["attempt_count"] for v in errors.values()),
        "rep_attempts": sum(errors[e]["attempt_count"] for e in rep),
        "exact_attempts": sum(errors[e]["attempt_count"] for e in exact),
        "_all_participants": set().union(*(v["participants"] for v in errors.values()))
        if errors else set(),
        "_rep_participants": set().union(*(errors[e]["participants"] for e in rep))
        if rep else set(),
        "_ppe": [len(rs) for rs in by_err.values()],
        "_emc": [r["effective_mod_count"] for rs in by_err.values() for r in rs],
        "_emc_min": [min(r["effective_mod_count"] for r in rs) for rs in by_err.values()],
    }
    return _finalize(base)


def _finalize(d: dict) -> dict:
    n_err, total_att = d["errors"], d["total_attempts"]
    ppe, emc, emc_min = d["_ppe"], d["_emc"], d["_emc_min"]
    d |= {
        "total_programs": len(emc),
        "total_participants": len(d["_all_participants"]),
        "rep_participants": len(d["_rep_participants"]),
        "unweighted": d["reproduced"] / n_err if n_err else 0.0,
        "attempt_weighted": d["rep_attempts"] / total_att if total_att else 0.0,
        "exact_unweighted": d["exact"] / n_err if n_err else 0.0,
        "exact_weighted": d["exact_attempts"] / total_att if total_att else 0.0,
        "ppe_mean": statistics.mean(ppe) if ppe else 0.0,
        "ppe_sd": statistics.stdev(ppe) if len(ppe) > 1 else 0.0,
        "ppe_median": statistics.median(ppe) if ppe else 0.0,
        "ppe_q1": float(np.percentile(ppe, 25)) if ppe else 0.0,
        "ppe_q3": float(np.percentile(ppe, 75)) if ppe else 0.0,
        "ppe_max": max(ppe) if ppe else 0,
        "emc_mean": statistics.mean(emc) if emc else 0.0,
        "emc_sd": statistics.stdev(emc) if len(emc) > 1 else 0.0,
        "emc_min_mean": statistics.mean(emc_min) if emc_min else 0.0,
        "emc_min_sd": statistics.stdev(emc_min) if len(emc_min) > 1 else 0.0,
    }
    return d


def combine(a: dict, b: dict) -> dict:
    """Pool two disjoint splits: sum counts, concatenate raw distributions."""
    base = {k: a[k] + b[k] for k in (
        "tasks", "rep_tasks", "errors", "reproduced", "exact",
        "total_attempts", "rep_attempts", "exact_attempts", "_ppe", "_emc", "_emc_min",
    )}
    base["_all_participants"] = a["_all_participants"] | b["_all_participants"]
    base["_rep_participants"] = a["_rep_participants"] | b["_rep_participants"]
    return _finalize(base)


def alt_rule_counts(errors: dict[str, dict], rows: list[dict]) -> dict:
    """Kept programs that also reproduce every training output."""
    by_err: dict[str, list[dict]] = defaultdict(list)
    programs = 0
    for r in rows:
        if r["error_id"] not in errors:
            continue
        if rule_type(r.get("training_matches"))[0] != ALTERNATIVE:
            continue
        programs += 1
        by_err[r["error_id"]].append(r)
    alt = set(by_err)
    return {
        "tasks": {errors[e]["task_id"] for e in alt},
        "errors": alt,
        "programs": programs,
        "attempts": sum(errors[e]["attempt_count"] for e in alt),
        "participants": set().union(*(errors[e]["participants"] for e in alt)) if alt else set(),
    }


def combine_alt(parts: list[dict]) -> dict:
    return {
        "tasks": set().union(*(p["tasks"] for p in parts)),
        "errors": set().union(*(p["errors"] for p in parts)),
        "programs": sum(p["programs"] for p in parts),
        "attempts": sum(p["attempts"] for p in parts),
        "participants": set().union(*(p["participants"] for p in parts)),
    }


def baseline_capture(path: Path, errors: dict[str, dict]) -> dict:
    """Exhaustive depth-1 capture, matched to the errors by (task_id, grid_hash)."""
    denom = {v["grid_key"]: v["attempt_count"] for v in errors.values()}
    captured: set[tuple] = set()
    for r in _iter_jsonl(path):
        key = (r["task_id"], r["grid_hash"])
        if key in denom:
            captured.add(key)
    cap_att = sum(denom[k] for k in captured)
    total_att = sum(denom.values())
    return {
        "captured": len(captured),
        "errors": len(denom),
        "cap_attempts": cap_att,
        "total_attempts": total_att,
        "unweighted": len(captured) / len(denom) if denom else 0.0,
        "attempt_weighted": cap_att / total_att if total_att else 0.0,
    }


# ── Stage-1 trivial-error filtering ──────────────────────────────────────────
def prefilter_stats(raw: Path, filtered: Path, dropped: Path) -> dict:
    """Dropped errors/attempts by reason over survivors + dropped, plus the
    near-miss mismatch-cell histogram."""
    size_by_eid: dict[str, int] = {}
    for r in _iter_jsonl(raw):
        g = r["output_grid"]
        size_by_eid[record_id(r["task_id"], g)] = len(g) * len(g[0]) if g else 0

    surv_err = surv_att = 0
    for r in _iter_jsonl(filtered):
        surv_err += 1
        surv_att += int(r["attempt_count"])

    d_err: dict[str, int] = defaultdict(int)
    d_att: dict[str, int] = defaultdict(int)
    nearmiss_cells: dict[int, int] = defaultdict(int)
    for r in _iter_jsonl(dropped):
        reason = r["reason"]
        d_err[reason] += 1
        d_att[reason] += int(r["attempt_count"])
        if reason == "ground_truth":
            n = size_by_eid.get(r["error_id"])
            prop = (r.get("diagnostic") or {}).get("mismatch_proportion_vs_test_output")
            if n and prop is not None:
                nearmiss_cells[round(prop * n)] += 1

    return {
        "tot_err": surv_err + sum(d_err.values()),
        "tot_att": surv_att + sum(d_att.values()),
        "drop_err": sum(d_err.values()),
        "drop_att": sum(d_att.values()),
        "d_err": dict(d_err),
        "d_att": dict(d_att),
        "nearmiss_cells": dict(nearmiss_cells),
    }


def combine_prefilter(parts: list[dict]) -> dict:
    out: dict = {"tot_err": 0, "tot_att": 0, "drop_err": 0, "drop_att": 0,
                 "d_err": defaultdict(int), "d_att": defaultdict(int),
                 "nearmiss_cells": defaultdict(int)}
    for p in parts:
        for k in ("tot_err", "tot_att", "drop_err", "drop_att"):
            out[k] += p[k]
        for k in ("d_err", "d_att", "nearmiss_cells"):
            for key, v in p[k].items():
                out[k][key] += v
    for k in ("d_err", "d_att", "nearmiss_cells"):
        out[k] = dict(out[k])
    return out


# ── Output ───────────────────────────────────────────────────────────────────
def pct(x: float) -> str:
    return f"{x:.1%}"


def _table(title: str, cols: list[str], rows: list[tuple[str, list[str]]]) -> None:
    print(f"\n## {title}\n")
    w = max(len(r[0]) for r in rows)
    cw = [max(len(c), *(len(r[1][i]) for r in rows)) for i, c in enumerate(cols)]
    print(" " * w + "  " + "  ".join(c.rjust(cw[i]) for i, c in enumerate(cols)))
    for label, cells in rows:
        print(label.ljust(w) + "  " + "  ".join(v.rjust(cw[i]) for i, v in enumerate(cells)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--osf-dir", type=Path, default=Path("OSF"))
    p.add_argument("--baseline", type=Path, default=None,
                   help="Exhaustive depth-1 candidates (training), filtered under the "
                        "same tolerance; needs task_id and grid_hash (default: "
                        "<osf-dir>/03_candidates/baseline/exhaustive_depth1_training.jsonl).")
    args = p.parse_args()
    osf = args.osf_dir
    if args.baseline is None:
        args.baseline = osf / "03_candidates" / "baseline" / "exhaustive_depth1_training.jsonl"
    inp = osf / "00_inputs" / "human"

    # Filtering (stage 1), over all errors before solver gating.
    pre = {s: prefilter_stats(inp / f"harc_errors_{s}.all.jsonl",
                              inp / f"harc_errors_{s}.filtered.jsonl",
                              inp / f"harc_errors_{s}.dropped.jsonl") for s in SPLITS}
    pre["combined"] = combine_prefilter([pre[s] for s in SPLITS])
    cols = ["Training", "Evaluation", "Combined"]
    keys = [*SPLITS, "combined"]

    def cell(n: int, tot: int) -> str:
        return f"{n:,} ({n / tot:.1%})" if tot else "0"

    for level, top in (("att", "Incorrect responses (attempts)"), ("err", "Unique error grids")):
        tot, d, drop = f"tot_{level}", f"d_{level}", f"drop_{level}"
        rows = [(top, [f"{pre[k][tot]:,}" for k in keys]),
                ("Filtered (all trivial)", [cell(pre[k][drop], pre[k][tot]) for k in keys])]
        rows += [(label, [cell(sum(pre[k][d].get(r, 0) for r in reasons), pre[k][tot])
                          for k in keys]) for label, reasons in PREFILTER_BUCKETS]
        _table(f"Trivial-error filtering, by {'attempts' if level == 'att' else 'unique grids'}",
               cols, rows)
    nm = pre["combined"]["nearmiss_cells"]
    nm_tot = sum(nm.values())
    print(f"\nNear-miss grids: {nm_tot:,}; differ by 1 cell {nm.get(1, 0) / nm_tot:.1%}, "
          f"by 2 cells {nm.get(2, 0) / nm_tot:.1%}, by 0 cells {nm.get(0, 0) / nm_tot:.1%}")

    # Coverage and reproduction.
    solver_tasks = set(json.loads((osf / "01_reference_programs" / "evaluation_solvers.json")
                                  .read_text()))
    train_errors = load_errors(inp / "harc_errors_training.filtered.jsonl")
    eval_all = load_errors(inp / "harc_errors_evaluation.filtered.jsonl")
    eval_errors = load_errors(inp / "harc_errors_evaluation.filtered.jsonl",
                              keep_tasks=solver_tasks)
    rows_by = {s: load_rows(osf / "03_candidates" / "human" / f"{s}.jsonl") for s in SPLITS}
    tr = summarize(train_errors, rows_by["training"])
    ev = summarize(eval_errors, rows_by["evaluation"])
    co = combine(tr, ev)
    S = [tr, ev, co]

    eval_tasks_all = {v["task_id"] for v in eval_all.values()}
    print("\n## Dataset coverage\n")
    print(f"Training: hand-written reference programs for all {tr['tasks']} tasks with errors.")
    print(f"Evaluation: synthesized reference programs for {len(solver_tasks)} tasks; "
          f"{len(eval_tasks_all)} tasks have non-trivial errors, {ev['tasks']} of them are modelled.")
    print(f"Evaluation not modelled: {len(eval_tasks_all) - ev['tasks']} tasks, "
          f"{len(eval_all) - len(eval_errors):,} of {len(eval_all):,} non-trivial unique errors.")

    _table("Reproduction", cols, [
        ("Tasks modelled", [f"{s['tasks']:,}" for s in S]),
        ("Unique non-trivial errors", [f"{s['errors']:,}" for s in S]),
        ("Errors reproduced", [f"{s['reproduced']:,} ({pct(s['unweighted'])})" for s in S]),
        ("Attempts", [f"{s['total_attempts']:,}" for s in S]),
        ("Attempts reproduced", [f"{s['rep_attempts']:,} ({pct(s['attempt_weighted'])})" for s in S]),
        ("Tasks with >=1 error reproduced", [f"{s['rep_tasks']:,}" for s in S]),
        ("Participants (all errors)", [f"{s['total_participants']:,}" for s in S]),
        ("Participants modelled", [f"{s['rep_participants']:,}" for s in S]),
        ("Exact match, errors", [f"{s['exact']:,} ({pct(s['exact_unweighted'])})" for s in S]),
        ("Exact match, attempt-weighted", [pct(s["exact_weighted"]) for s in S]),
    ])

    _table("Programs", cols, [
        ("Kept programs", [f"{s['total_programs']:,}" for s in S]),
        ("Programs per error, M ± SD", [f"{s['ppe_mean']:.2f} ± {s['ppe_sd']:.2f}" for s in S]),
        ("Programs per error, median [IQR]",
         [f"{s['ppe_median']:.0f} [{s['ppe_q1']:.0f}, {s['ppe_q3']:.0f}]" for s in S]),
        ("Programs per error, max", [f"{s['ppe_max']}" for s in S]),
        ("Effective edits per program, M ± SD", [f"{s['emc_mean']:.2f} ± {s['emc_sd']:.2f}" for s in S]),
        ("Effective edits, per-error min, M ± SD",
         [f"{s['emc_min_mean']:.2f} ± {s['emc_min_sd']:.2f}" for s in S]),
    ])

    alt = [alt_rule_counts(train_errors, rows_by["training"]),
           alt_rule_counts(eval_errors, rows_by["evaluation"])]
    alt.append(combine_alt(alt))
    _table("Alternative-rule programs (also reproduce every training output)", cols, [
        ("Tasks", [f"{len(a['tasks'])}" for a in alt]),
        ("Errors", [f"{len(a['errors'])} / {s['reproduced']:,}" for a, s in zip(alt, S)]),
        ("Programs", [f"{a['programs']} / {s['total_programs']:,}" for a, s in zip(alt, S)]),
        ("Attempts", [f"{a['attempts']:,}" for a in alt]),
        ("Participants", [f"{len(a['participants']):,}" for a in alt]),
    ])

    samples = [count_rows([osf / "02_samples" / "human" / f"{s}_k{k}.jsonl" for k in STEP_BUDGETS])
               for s in SPLITS]
    labels = osf / "04_classifications" / "human" / "training.jsonl"
    print("\n## Sampling effort\n")
    print(f"Synthesized programs sampled: training {samples[0]:,}, evaluation {samples[1]:,}, "
          f"total {sum(samples):,}")
    print(f"Error-category labels (training): {count_rows([labels]):,}")

    print("\n## Exhaustive depth-1 baseline (training)\n")
    if not args.baseline.exists():
        print(f"Skipped: {args.baseline} not found.")
    else:
        b = baseline_capture(args.baseline, train_errors)
        print(f"Attempts captured: {b['cap_attempts']:,} / {b['total_attempts']:,} "
              f"({pct(b['attempt_weighted'])})")
        print(f"Errors captured:   {b['captured']:,} / {b['errors']:,} ({pct(b['unweighted'])})")


if __name__ == "__main__":
    main()
