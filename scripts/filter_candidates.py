"""Keep the programs that reproduce each error (stage 4).

For each error, keeps the programs whose output is within the tolerance of
``prefilter_errors.py`` and, among those, only the ones tied at the fewest wrong
cells. Programs with nested function calls are dropped unless
``--keep-nested`` is given. Each kept row gains ``n_programs`` (distinct
programs kept for its error) and ``n_edit_paths`` (edit sets that led to the
same program).

Reads one or more outputs of ``evaluate_synthesized_error_programs.py`` (e.g.
all edit budgets). Writes one JSONL of accepted programs, the format of OSF's
``03_candidates/``.

Usage:
    uv run python scripts/filter_candidates.py \\
        --eval-files OSF/02_samples/human/training_k{2,4,8}.jsonl \\
        --k-cells 2 --min-grid-size 10 \\
        --output candidates_training.jsonl
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.filters import passes_threshold  # noqa: E402
from src.utils import grid_array  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_eval_records(paths: List[Path]) -> List[Dict[str, Any]]:
    """Read all eval JSONLs into a flat list of records."""
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
# Candidate deduplication
# ---------------------------------------------------------------------------


def _candidate_key(record: Dict[str, Any]) -> str:
    """Hash a candidate so duplicates (same edits) collapse to one entry.

    Modifications-based records hash a sorted tuple of
    ``(variable_name, original, replacement)`` triples.  Free-form records
    fall back to hashing the rewritten program text.
    """
    mods = record.get("modifications")
    if mods:
        triples = sorted(
            (
                m.get("variable_name", ""),
                m.get("original", "") or "",
                m.get("replacement", "") or "",
            )
            for m in mods
        )
        payload = json.dumps(triples)
    else:
        payload = record.get("rewritten_program") or ""
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-group selection
# ---------------------------------------------------------------------------


def select_best_per_group(
    records: List[Dict[str, Any]],
    *,
    tau: Optional[float],
    k_cells: Optional[int],
    min_grid_size: Optional[int],
    keep_nested: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Group by ``error_id`` and pick the best per group.

    For each group: drop records that fail to execute, fail the threshold, or
    (unless *keep_nested*) carry ``has_nested_violation=True``.  Among the
    survivors, keep all candidates tied at the minimum mismatch proportion;
    deduplicate in two passes.  Pass 1 collapses identical modification
    triples via :func:`_candidate_key`.  Pass 2 collapses distinct mod-sets
    that produce the same ``rewritten_program`` text, keeping the candidate
    with the fewest modifications and stamping ``n_edit_paths`` (size of the
    collapsed group) and ``n_programs`` (distinct programs in the error_id
    group after pass 2) on every kept record.

    Returns the kept candidates plus a small counters dict for the summary.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    counters = {
        "input_records": 0,
        "missing_target_grid": 0,
        "execution_failed": 0,
        "nested_violation": 0,
        "shape_mismatch": 0,
        "above_threshold": 0,
    }

    # Track every unique error_id in the eval input (the stage-1 denominator)
    # and its attempt_count, so we can compute coverage rates and the
    # attempt-weighted variant after grouping. attempt_count is constant per
    # error_id by construction (set in build_error_lookup), so first-seen wins.
    attempts_by_error: Dict[str, int] = {}

    for rec in records:
        counters["input_records"] += 1
        eid = rec.get("error_id")
        if eid is not None and eid not in attempts_by_error:
            ac = rec.get("attempt_count")
            attempts_by_error[eid] = int(ac) if ac is not None else 0
        target = rec.get("target_error_grid")
        if target is None:
            counters["missing_target_grid"] += 1
            continue
        if not rec.get("execution_success"):
            counters["execution_failed"] += 1
            continue
        if rec.get("has_nested_violation") and not keep_nested:
            counters["nested_violation"] += 1
            continue
        output = rec.get("program_output_grid")
        if output is None:
            counters["execution_failed"] += 1
            continue

        out_arr = grid_array(output)
        tgt_arr = grid_array(target)
        if out_arr is None or tgt_arr is None or out_arr.shape != tgt_arr.shape:
            counters["shape_mismatch"] += 1
            continue

        if not passes_threshold(
            output, target, tau=tau, k_cells=k_cells, min_grid_size=min_grid_size
        ):
            counters["above_threshold"] += 1
            continue

        diff_count = int((out_arr != tgt_arr).sum())
        mismatch_proportion = diff_count / tgt_arr.size if tgt_arr.size else 0.0
        rec = {
            **rec,
            "mismatch_proportion": mismatch_proportion,
            "n_mismatch": diff_count,
        }

        groups[rec["error_id"]].append(rec)

    kept: List[Dict[str, Any]] = []
    for _error_id, group_records in groups.items():
        best_mp = min(r["mismatch_proportion"] for r in group_records)
        tied = [r for r in group_records if r["mismatch_proportion"] == best_mp]

        # Pass 1: dedup by normalized modification triples.
        seen: Dict[str, Dict[str, Any]] = {}
        for r in tied:
            seen.setdefault(_candidate_key(r), r)

        # Pass 2: distinct mod-sets can yield the same rewritten program. Group
        # by rewritten_program and keep the candidate with the fewest mods
        # (stable sort -> first-seen wins on ties). Record n_edit_paths as the
        # convergence count.
        program_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in seen.values():
            program_groups[r.get("rewritten_program") or ""].append(r)

        final: List[Dict[str, Any]] = []
        for entries in program_groups.values():
            entries.sort(key=lambda r: len(r.get("modifications") or []))
            final.append({**entries[0], "n_edit_paths": len(entries)})

        n_programs = len(final)
        for r in final:
            r["n_programs"] = n_programs

        kept.extend(final)

    counters["unique_errors_total"] = len(attempts_by_error)
    counters["unique_errors_with_candidate"] = len(groups)
    counters["candidates_kept"] = len(kept)

    attempts_total = sum(attempts_by_error.values())
    attempts_explained = sum(
        attempts_by_error.get(eid, 0) for eid in groups.keys()
    )
    counters["attempts_total"] = attempts_total
    counters["attempts_explained"] = attempts_explained
    return kept, counters


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(counters: Dict[str, int], kept: List[Dict[str, Any]]) -> None:
    print("\n--- filter_candidates summary ---")
    for label in (
        "input_records",
        "missing_target_grid",
        "execution_failed",
        "nested_violation",
        "shape_mismatch",
        "above_threshold",
        "unique_errors_total",
        "unique_errors_with_candidate",
        "candidates_kept",
    ):
        print(f"  {label:<32} {counters.get(label, 0)}")

    total_errors = counters.get("unique_errors_total", 0)
    explained_errors = counters.get("unique_errors_with_candidate", 0)
    if total_errors:
        print(
            f"  unique_error_coverage            {explained_errors}/{total_errors} "
            f"({explained_errors / total_errors * 100:.1f}%)"
        )

    attempts_total = counters.get("attempts_total", 0)
    attempts_explained = counters.get("attempts_explained", 0)
    if attempts_total:
        print(
            f"  attempt_weighted_coverage        {attempts_explained}/{attempts_total} "
            f"({attempts_explained / attempts_total * 100:.1f}%)"
        )

    if not kept:
        return

    mps = np.array([r["mismatch_proportion"] for r in kept])
    n_exact = int(np.sum(mps == 0.0))
    print(
        f"  exact_match candidates           {n_exact} ({n_exact/len(kept)*100:.1f}%)"
    )
    print(
        f"  mismatch_proportion              mean={mps.mean():.4f} "
        f"median={np.median(mps):.4f} max={mps.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter synthesized program candidates against the participant's error grid. "
            "Keeps tied-at-minimum candidates per error_id under the given threshold."
        )
    )
    parser.add_argument(
        "--eval-files",
        type=str,
        nargs="+",
        required=True,
        help="One or more eval JSONL files (output of evaluate_synthesized_error_programs.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSONL.",
    )
    group_thresh = parser.add_mutually_exclusive_group(required=True)
    group_thresh.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Threshold on mismatch proportion (e.g. 0.1 keeps candidates within 10%% wrong cells).",
    )
    group_thresh.add_argument(
        "--k-cells",
        type=int,
        default=None,
        help="Threshold on absolute mismatched cell count. Pair with --min-grid-size when desired.",
    )
    parser.add_argument(
        "--min-grid-size",
        type=int,
        default=None,
        help=(
            "Only meaningful with --k-cells: grids with fewer cells than this require exact match. "
            "Guards against accepting '3 of 9 wrong' on tiny grids."
        ),
    )
    parser.add_argument(
        "--keep-nested",
        action="store_true",
        help="Retain records flagged with has_nested_violation (default: drop them).",
    )

    args = parser.parse_args()

    if args.min_grid_size is not None and args.k_cells is None:
        parser.error("--min-grid-size requires --k-cells")

    eval_paths = [Path(p) for p in args.eval_files]
    for p in eval_paths:
        if not p.exists():
            print(f"Error: {p} not found.")
            return

    print(f"Loading {len(eval_paths)} eval file(s)...")
    records = load_eval_records(eval_paths)
    print(f"  loaded {len(records)} records.")

    if args.tau is not None:
        print(f"Threshold: mismatch_proportion <= {args.tau:g}")
    else:
        ms = (
            f" (with min-grid-size guard {args.min_grid_size})"
            if args.min_grid_size is not None
            else ""
        )
        print(f"Threshold: mismatched cells <= {args.k_cells}{ms}")
    print(f"Nested violations: {'kept' if args.keep_nested else 'dropped'}")

    kept, counters = select_best_per_group(
        records,
        tau=args.tau,
        k_cells=args.k_cells,
        min_grid_size=args.min_grid_size,
        keep_nested=args.keep_nested,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(kept)} kept candidates to {output_path}")
    print_summary(counters, kept)


if __name__ == "__main__":
    main()
