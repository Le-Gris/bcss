"""Remove errors that need no program to explain them (stage 1).

An error is dropped if it is (nearly) the test input, a copy of a training
input or output, a blank grid (``--filter-blanks``), or (nearly) the correct
output. "Nearly" uses the same tolerance as ``filter_candidates.py``: at most
``--k-cells`` wrong cells (grids of at least ``--min-grid-size`` cells), or a
mismatch proportion of at most ``--tau``.

Reads a JSONL of unique errors and the ARC task JSONs. Writes the kept errors
and the dropped ones (with the reason); by default next to the input, as
``<input>.<settings>.filtered.jsonl`` and ``<input>.<settings>.dropped.jsonl``.
OSF's ``00_inputs/`` holds the outputs for the paper's settings, shown below.

Usage:
    uv run python scripts/prefilter_errors.py \\
        --input-file OSF/00_inputs/human/harc_errors_training.all.jsonl \\
        --task-dir external/arc-agi-1/training \\
        --k-cells 2 --min-grid-size 10 --filter-blanks --blank-shape 3x3 \\
        --output-filtered results/errors/training.filtered.jsonl \\
        --output-dropped results/errors/training.dropped.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.filters import (  # noqa: E402
    is_blank_grid,
    is_identity,
    is_training_copy,
    matches_ground_truth,
    passes_word_count,
)
from src.utils import (  # noqa: E402
    calculate_mismatch,
    load_grid,
    load_unique_errors,
    record_id,
)


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


_BLANK_SHAPE_RE = re.compile(r"^(\d+)\s*[xX]\s*(\d+)$")


def parse_blank_shape(spec: str) -> Tuple[int, int]:
    """Parse a 'HxW' string into a ``(rows, cols)`` tuple."""
    match = _BLANK_SHAPE_RE.match(spec.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid --blank-shape value '{spec}': expected 'HxW' (e.g. '3x3')."
        )
    rows, cols = int(match.group(1)), int(match.group(2))
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError(
            f"--blank-shape values must be positive: got '{spec}'."
        )
    return (rows, cols)


def build_filter_slug(args: argparse.Namespace) -> str:
    """Short filename-safe identifier of the filter settings.

    Threshold first (``tau{X}`` or ``k{K}[_minN{M}]``), then ``minw{N}`` for
    the optional description word-count filter, then suffixes for any
    non-default toggle (``_blanks[-HxW...]``, ``_noid``, ``_notrain``,
    ``_nogt``), e.g. ``k2_minN10``.
    """
    parts: List[str] = []
    if args.tau is not None:
        parts.append(f"tau{args.tau:g}")
    else:
        thresh = f"k{args.k_cells}"
        if args.min_grid_size is not None:
            thresh += f"_minN{args.min_grid_size}"
        parts.append(thresh)
    if args.min_description_words is not None:
        parts.append(f"minw{args.min_description_words}")
    if args.filter_blanks:
        if args.blank_shape:
            shape_str = "-".join(f"{r}x{c}" for r, c in args.blank_shape)
            parts.append(f"blanks-{shape_str}")
        else:
            parts.append("blanks")
    if args.no_identity:
        parts.append("noid")
    if args.no_train_copies:
        parts.append("notrain")
    if args.no_ground_truth:
        parts.append("nogt")
    return "_".join(parts)


def build_prefilter_stamp(args: argparse.Namespace) -> Dict[str, Any]:
    """Serialisable snapshot of the filter settings for in-record provenance."""
    checks = [
        name
        for name, on in (
            ("identity", not args.no_identity),
            ("train_copies", not args.no_train_copies),
            ("ground_truth", not args.no_ground_truth),
        )
        if on
    ]
    return {
        "tau": args.tau,
        "k_cells": args.k_cells,
        "min_grid_size": args.min_grid_size,
        "min_description_words": args.min_description_words,
        "checks": checks,
        "filter_blanks": bool(args.filter_blanks),
        "blank_shapes": (
            [list(s) for s in args.blank_shape] if args.blank_shape else None
        ),
    }


# ---------------------------------------------------------------------------
# Per-error classification
# ---------------------------------------------------------------------------


def classify_error(
    error: Dict[str, Any],
    *,
    task_dir: Path,
    tau: Optional[float],
    k_cells: Optional[int],
    min_grid_size: Optional[int],
    check_identity: bool,
    check_train_copies: bool,
    check_ground_truth: bool,
    filter_blanks: bool,
    blank_shapes: Optional[List[Tuple[int, int]]],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Apply trivial-error filters to one record.

    Returns ``(reason, diagnostic)`` where *reason* is ``None`` when the record
    should be kept.  *diagnostic* carries check-specific evidence (mismatch
    proportions, train-example index, matched shape).
    """
    task_id = error["task_id"]
    output_grid = error["output_grid"]
    threshold_kwargs = {"tau": tau, "k_cells": k_cells, "min_grid_size": min_grid_size}

    grid_data = load_grid(task_id, [task_dir])
    if not grid_data or "test" not in grid_data or not grid_data["test"]:
        return "no_task_data", {"task_id": task_id}

    test_input = grid_data["test"][0]["input"]

    if check_identity and is_identity(output_grid, test_input, **threshold_kwargs):
        return "identity", {
            "mismatch_proportion": calculate_mismatch(output_grid, test_input)
        }

    if check_train_copies:
        match = is_training_copy(output_grid, grid_data.get("train", []))
        if match is not None:
            which, idx = match
            return f"train_{which}_copy", {"train_index": idx}

    if filter_blanks and is_blank_grid(output_grid, blank_shapes):
        shape = (len(output_grid), len(output_grid[0]) if output_grid else 0)
        return "blank_grid", {"shape": list(shape)}

    if check_ground_truth:
        test_output = grid_data["test"][0].get("output")
        if test_output is not None and matches_ground_truth(
            output_grid, test_output, **threshold_kwargs
        ):
            return "ground_truth", {
                "mismatch_proportion_vs_test_output": calculate_mismatch(
                    output_grid, test_output
                )
            }

    return None, {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Drop unique errors that are trivially explained (identity, training-grid copy, "
            "blank grid, ground-truth) under the chosen threshold."
        )
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the raw unique-errors JSONL (one record per unique error).",
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        required=True,
        help="Directory containing ARC-format task JSONs (one <task_id>.json per task).",
    )
    parser.add_argument(
        "--output-filtered",
        type=str,
        default=None,
        help="Path for the kept-errors JSONL. Defaults to <input_stem>.filtered.jsonl.",
    )
    parser.add_argument(
        "--output-dropped",
        type=str,
        default=None,
        help="Path for the dropped-errors log JSONL. Defaults to <input_stem>.dropped.jsonl.",
    )

    grp_thresh = parser.add_mutually_exclusive_group(required=True)
    grp_thresh.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Threshold on mismatch proportion (used by identity and ground-truth checks).",
    )
    grp_thresh.add_argument(
        "--k-cells",
        type=int,
        default=None,
        help="Threshold on absolute mismatched cell count (used by identity and ground-truth checks).",
    )
    parser.add_argument(
        "--min-grid-size",
        type=int,
        default=None,
        help="With --k-cells: require exact match on grids smaller than this many cells.",
    )

    parser.add_argument(
        "--no-identity",
        action="store_true",
        help="Disable the identity (error ≈ test input) check.",
    )
    parser.add_argument(
        "--no-train-copies",
        action="store_true",
        help="Disable the training-grid copy check.",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Disable the ground-truth (error ≈ task JSON test output) check.",
    )
    parser.add_argument(
        "--filter-blanks",
        action="store_true",
        help="Enable filtering of all-zeros (blank) grids. Off by default.",
    )
    parser.add_argument(
        "--blank-shape",
        type=parse_blank_shape,
        action="append",
        default=None,
        help=(
            "Restrict blank-grid filtering to a specific 'HxW' shape (repeatable). "
            "If --filter-blanks is set but no shapes are listed, blanks of any shape are filtered."
        ),
    )
    parser.add_argument(
        "--min-description-words",
        type=int,
        default=None,
        help=(
            "If set, drop descriptions whose 'text' has fewer than N "
            "whitespace-separated words. Errors that originally had "
            "descriptions but lose all of them are dropped with reason "
            "'no_usable_descriptions'. Errors that arrived with no "
            "descriptions are unaffected. Off by default."
        ),
    )

    args = parser.parse_args()

    if args.min_grid_size is not None and args.k_cells is None:
        parser.error("--min-grid-size requires --k-cells")
    if args.min_description_words is not None and args.min_description_words < 1:
        parser.error("--min-description-words must be >= 1")

    input_path = Path(args.input_file)
    task_dir = Path(args.task_dir)

    if not input_path.exists():
        print(f"Error: input file {input_path} not found.")
        return
    if not task_dir.exists():
        print(f"Error: task directory {task_dir} not found.")
        return

    slug = build_filter_slug(args)
    stamp = build_prefilter_stamp(args)

    if args.output_filtered is None:
        filtered_path = input_path.with_suffix(f".{slug}.filtered.jsonl")
    else:
        filtered_path = Path(args.output_filtered)
    if args.output_dropped is None:
        dropped_path = input_path.with_suffix(f".{slug}.dropped.jsonl")
    else:
        dropped_path = Path(args.output_dropped)

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_path.parent.mkdir(parents=True, exist_ok=True)

    if args.tau is not None:
        print(f"Threshold: mismatch_proportion <= {args.tau:g}")
    else:
        ms = (
            f" (min-grid-size guard {args.min_grid_size})"
            if args.min_grid_size is not None
            else ""
        )
        print(f"Threshold: mismatched cells <= {args.k_cells}{ms}")
    enabled = []
    if not args.no_identity:
        enabled.append("identity")
    if not args.no_train_copies:
        enabled.append("train_copies")
    if args.filter_blanks:
        shapes_repr = (
            ",".join(f"{r}x{c}" for r, c in args.blank_shape)
            if args.blank_shape
            else "any"
        )
        enabled.append(f"blanks({shapes_repr})")
    if not args.no_ground_truth:
        enabled.append("ground_truth")
    print(f"Enabled filters: {', '.join(enabled) if enabled else '(none)'}")

    counters: Dict[str, int] = {
        "total": 0,
        "kept": 0,
        "descriptions_kept": 0,
        "descriptions_dropped": 0,
    }
    reasons: Dict[str, int] = {}

    with open(filtered_path, "w") as f_keep, open(dropped_path, "w") as f_drop:
        for error in tqdm(list(load_unique_errors(input_path)), desc="Prefiltering"):
            counters["total"] += 1
            reason, diag = classify_error(
                error,
                task_dir=task_dir,
                tau=args.tau,
                k_cells=args.k_cells,
                min_grid_size=args.min_grid_size,
                check_identity=not args.no_identity,
                check_train_copies=not args.no_train_copies,
                check_ground_truth=not args.no_ground_truth,
                filter_blanks=args.filter_blanks,
                blank_shapes=args.blank_shape,
            )
            error_id = record_id(error["task_id"], error["output_grid"])

            # Per-description word-count filter. Only engages when the grid
            # filters passed; this is purely about the NL payload.
            filtered_descriptions: Optional[List[Dict[str, Any]]] = None
            if reason is None and args.min_description_words is not None:
                originals = error.get("descriptions") or []
                if originals:
                    kept_descs = [
                        d for d in originals
                        if passes_word_count(d.get("text"), args.min_description_words)
                    ]
                    n_dropped = len(originals) - len(kept_descs)
                    counters["descriptions_kept"] += len(kept_descs)
                    counters["descriptions_dropped"] += n_dropped
                    if not kept_descs:
                        reason = "no_usable_descriptions"
                        diag = {
                            "original_count": len(originals),
                            "min_description_words": args.min_description_words,
                        }
                    else:
                        filtered_descriptions = kept_descs

            if reason is None:
                counters["kept"] += 1
                error_out = {"error_id": error_id, **error}
                if filtered_descriptions is not None:
                    error_out["descriptions"] = filtered_descriptions
                metadata = dict(error_out.get("metadata") or {})
                metadata["prefilter"] = stamp
                error_out["metadata"] = metadata
                f_keep.write(json.dumps(error_out) + "\n")
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
                response_ids = [
                    d.get("response_id")
                    for d in (error.get("descriptions") or [])
                    if d.get("response_id")
                ]
                drop_record = {
                    "error_id": error_id,
                    "task_id": error["task_id"],
                    "response_ids": response_ids,
                    "reason": reason,
                    "diagnostic": diag,
                    "dataset": error.get("dataset"),
                    "source": error.get("source"),
                    "attempt_count": error.get("attempt_count"),
                    "prefilter": stamp,
                }
                f_drop.write(json.dumps(drop_record) + "\n")

    print(f"\nWrote kept errors to:   {filtered_path}")
    print(f"Wrote dropped log to:   {dropped_path}")
    print(f"\nKept {counters['kept']} / {counters['total']} errors.")
    if args.min_description_words is not None:
        d_total = counters["descriptions_kept"] + counters["descriptions_dropped"]
        print(
            f"Description word-count filter (>= {args.min_description_words} words): "
            f"kept {counters['descriptions_kept']} / {d_total} descriptions."
        )
    if reasons:
        print("Dropped breakdown:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<24} {count}")


if __name__ == "__main__":
    main()
