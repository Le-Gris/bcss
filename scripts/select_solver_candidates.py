"""Collect the valid synthesized solvers for each task into one file.

Keeps the programs that solve their task (``strict_solve``) and groups them by
task: either only the shortest (``--select shortest``) or every distinct
program, shortest first (``--select all-unique``). The result replaces the
human-written solvers when synthesizing error programs (``--solvers-file``);
OSF's ``01_reference_programs/evaluation_solvers.json`` has this format.

Reads the output of ``evaluate_synthesized_solver_programs.py``. Writes
``synthesized_solvers_<select>.json``, shaped ``{task_id: [program, ...]}``.

Usage:
    uv run python scripts/select_solver_candidates.py \\
        --eval-files <solver_evaluation_output>.jsonl --select all-unique
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

DEFAULT_OUTPUT_DIR = "results/solver_synthesis_selection"


def output_filename(select_mode: str) -> str:
    """Stamp the selection mode into the filename so the two modes don't clobber
    each other under the same ``--run-tag``."""
    return f"synthesized_solvers_{select_mode}.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_eval_records(paths: List[Path]) -> List[Dict[str, Any]]:
    """Read all solver-evaluation JSONLs into a flat list of records."""
    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _sort_key(record: Dict[str, Any]) -> tuple[int, int]:
    """Order candidates shortest-first, ties broken by sample_index (stable).

    ``line_count`` / ``sample_index`` are guaranteed non-``None`` for
    strict-solving records (they imply a successful rewrite), but we coerce
    defensively so a malformed row sorts last rather than raising.
    """
    line_count = record.get("line_count")
    sample_index = record.get("sample_index")
    return (
        line_count if line_count is not None else sys.maxsize,
        sample_index if sample_index is not None else sys.maxsize,
    )


def select_solvers(
    records: List[Dict[str, Any]],
    *,
    mode: str,
) -> tuple[Dict[str, List[str]], Dict[str, int]]:
    """Gate on ``strict_solve`` and select per-task solver bundles.

    Returns ``(bundles, counters)`` where *bundles* maps ``task_id`` to a list
    of full solver sources (empty-bundle tasks are omitted entirely).
    """
    counters = {
        "input_records": len(records),
        "strict_solving": 0,
        "missing_program": 0,
    }

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if not rec.get("strict_solve"):
            continue
        program = rec.get("rewritten_program")
        if not program:
            # strict_solve should guarantee a rewritten program; skip defensively.
            counters["missing_program"] += 1
            continue
        counters["strict_solving"] += 1
        by_task[rec["task_id"]].append(rec)

    bundles: Dict[str, List[str]] = {}
    for task_id, group in by_task.items():
        group_sorted = sorted(group, key=_sort_key)
        if mode == "shortest":
            bundles[task_id] = [group_sorted[0]["rewritten_program"]]
        else:  # all-unique
            seen: set[str] = set()
            unique: List[str] = []
            for rec in group_sorted:
                program = rec["rewritten_program"]
                if program not in seen:
                    seen.add(program)
                    unique.append(program)
            bundles[task_id] = unique

    counters["tasks_with_solver"] = len(bundles)
    counters["bodies_kept"] = sum(len(v) for v in bundles.values())
    return bundles, counters


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(counters: Dict[str, int], bundles: Dict[str, List[str]]) -> None:
    print("\n--- select_solver_candidates summary ---")
    for label in (
        "input_records",
        "strict_solving",
        "missing_program",
        "tasks_with_solver",
        "bodies_kept",
    ):
        print(f"  {label:<24} {counters.get(label, 0)}")

    bundle_sizes = [len(v) for v in bundles.values()]
    if bundle_sizes:
        arr = np.array(bundle_sizes)
        print(
            f"  bodies_per_task          mean={arr.mean():.2f} "
            f"median={np.median(arr):.1f} max={arr.max()}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select strict-solving synthesized solver programs per task and export "
            "them as a JSON map for error synthesis (--solvers-file)."
        )
    )
    parser.add_argument(
        "--eval-files",
        type=str,
        nargs="+",
        required=True,
        help="One or more solver-evaluation JSONL files (stage C output).",
    )
    parser.add_argument(
        "--select",
        choices=["shortest", "all-unique"],
        required=True,
        help=(
            "shortest: one shortest valid solver per task. "
            "all-unique: all unique valid solvers per task, ordered shortest-first."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the output map (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag; nests output under <output-dir>/<run-tag>/.",
    )
    args = parser.parse_args()

    eval_paths = [Path(p) for p in args.eval_files]
    for p in eval_paths:
        if not p.exists():
            print(f"Error: {p} not found.")
            return

    output_dir = Path(args.output_dir)
    if args.run_tag:
        output_dir = output_dir / args.run_tag

    print(f"Loading {len(eval_paths)} eval file(s)...")
    records = load_eval_records(eval_paths)
    print(f"  loaded {len(records)} records.")
    print(f"Selection mode: {args.select}")

    bundles, counters = select_solvers(records, mode=args.select)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(args.select)
    with open(output_path, "w") as f:
        json.dump(bundles, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"\nWrote {counters['bodies_kept']} solver(s) across "
        f"{counters['tasks_with_solver']} task(s) to {output_path}"
    )
    print_summary(counters, bundles)


if __name__ == "__main__":
    main()
