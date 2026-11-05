"""Per-task relative length change, treatment vs control (Figure 2C data).

For every task with both treatment (error-fitting) and control (correct-output)
programs, takes the mean relative length change, (rewritten - reference) /
reference, over that task's programs in each arm.  Done separately for the
training and evaluation splits.

Reads the filtered candidate files from OSF; writes
``<output-root>/humans_<split>/per_task_relative_length_change.csv`` with
columns ``task_id, treatment, control``.  No model is fit.

Usage:
    uv run python analyses/compute_simplicity_per_task.py --osf-dir OSF
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# split -> (treatment, control), relative to the OSF directory
SPLITS = {
    "training": ("03_candidates/human/training.jsonl",
                 "03_candidates/control/human_training.jsonl"),
    "evaluation": ("03_candidates/human/evaluation.jsonl",
                   "03_candidates/control/human_evaluation.jsonl"),
}


def _load(path: Path) -> dict[str, dict[str, list[dict]]]:
    """task_id -> error_id -> candidate rows."""
    t: dict = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            t[str(r["task_id"])][r["error_id"]].append(r)
    return t


def _rel_changes(d: dict, tid: str) -> list[float]:
    """Relative length change of every candidate program for one task."""
    out = []
    for e in d[tid].values():
        for c in e:
            o, dl = c.get("original_line_count"), c.get("line_count_delta")
            if o and dl is not None:
                out.append(dl / o)
    return out


def compute(treat_path: Path, ctrl_path: Path) -> list[tuple[str, float, float]]:
    """(task_id, mean treatment change, mean control change) per shared task."""
    tre, ctl = _load(treat_path), _load(ctrl_path)
    rows = []
    for tid in sorted(set(tre) & set(ctl)):
        tr, cr = _rel_changes(tre, tid), _rel_changes(ctl, tid)
        if tr and cr:
            rows.append((tid, round(float(np.mean(tr)), 4), round(float(np.mean(cr)), 4)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--osf-dir", type=Path, default=Path("OSF"))
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "results" / "analyses" / "program_simplicity")
    args = p.parse_args()

    for split, (treat, ctrl) in SPLITS.items():
        rows = compute(args.osf_dir / treat, args.osf_dir / ctrl)
        out_dir = args.output_root / f"humans_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "per_task_relative_length_change.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["task_id", "treatment", "control"])
            w.writerows(rows)
        t = [r[1] for r in rows]
        c = [r[2] for r in rows]
        print(f"wrote {path}  ({len(rows)} tasks; median treatment={np.median(t):+.3f}, "
              f"control={np.median(c):+.3f})")


if __name__ == "__main__":
    main()
