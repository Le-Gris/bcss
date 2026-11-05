"""Apply, run and score each synthesized edit set (stage 3).

For each response from ``synthesize_error_programs.py``, applies the line edits
to the reference program, removes code that no longer affects the output, and
runs the result on the test input. The output is compared with the
participant's error (exact match, Hamming and Levenshtein distance), and the
program is also run on the training inputs to record which training examples it
reproduces.

Reads the synthesis output, the filtered errors and the ARC task JSONs. Writes
one row per sample to ``eval_<input>.jsonl`` (the format of OSF's
``02_samples/``) and prints summary statistics.

Usage:
    uv run python scripts/evaluate_synthesized_error_programs.py \\
        --input-file <error_synthesis_output>.jsonl \\
        --errors-file OSF/00_inputs/human/harc_errors_training.filtered.jsonl \\
        --task-dir external/arc-agi-1/training
"""

import argparse
import ast
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Ensure project root and src are on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from hodel_dsl.dsl_parser import ProgramRewriter  # noqa: E402

from src.filters import build_dsl_namespace  # noqa: E402
from src.pipeline import build_control_error_records  # noqa: E402
from src.program_edits import ModifyOnlyLineModification, apply_modifications  # noqa: E402
from src.utils import (  # noqa: E402
    DEFAULT_PROGRAM_TIMEOUT_S,
    calculate_mismatch,
    collapse_solver_bundles,
    flatten_grid,
    grid_dimensions,
    grid_to_tuple,
    load_grid,
    load_solvers,
    load_synthesized_solvers,
    load_unique_errors,
    record_id,
    time_limit,
)

DEFAULT_OUTPUT_DIR = "results/error_synthesis_evaluation"


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


def levenshtein_distance(seq1: tuple, seq2: tuple) -> int:
    """Edit distance between two sequences (memory-efficient two-row DP)."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


# ---------------------------------------------------------------------------
# Program-source helpers
# ---------------------------------------------------------------------------


def extract_original_rhs(program_source: str, variable_names: List[str]) -> Dict[str, str]:
    """Return the original RHS expression for each variable in *program_source*."""
    result: Dict[str, str] = {}
    remaining = set(variable_names)
    pat = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$")
    for line in program_source.splitlines():
        if not remaining:
            break
        match = pat.match(line.strip())
        if match:
            var, rhs = match.groups()
            if var in remaining:
                result[var] = rhs
                remaining.discard(var)
    return result


def count_program_body_lines(program_source: str) -> int:
    """Count assignment lines in a rewritten program body (excludes def / return)."""
    count = 0
    for line in program_source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("def ") or stripped.startswith("return "):
            continue
        count += 1
    return count


def count_calls(expression: str) -> int:
    """Count function calls in *expression*. Used to surface nested-call inflation."""
    if not expression:
        return 0
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return 1
    count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))
    return max(1, count)


def has_nested_calls(expression: str) -> bool:
    """True if *expression* contains a function call as an argument to another call.

    The prompt explicitly forbids nested function calls (e.g. ``rotate(rotate(I))``),
    so this is a contract check against the model's output, not a runtime check.
    """
    if not expression:
        return False
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    return True
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Call):
                    return True
            if isinstance(node.func, ast.Call):
                return True
    return False


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert tuples / sets / frozensets to JSON-friendly types."""
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [make_json_serializable(x) for x in sorted(list(obj), key=str)]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Build the (error_id, response_id) → error-record lookup
# ---------------------------------------------------------------------------


def build_error_lookup(
    errors_file: Path,
) -> Dict[Tuple[str, Optional[str]], Dict[str, Any]]:
    """Index the unique-errors JSONL by ``(error_id, response_id)``.

    Each unique error contributes up to ``1 + len(descriptions)`` lookup keys:

    * One key per description, using that description's ``response_id``.  The
      stored value's ``description`` is the description text.
    * One ``(error_id, None)`` entry covering synth records produced in the
      no-NL mode (where ``response_id`` is ``None``).

    ``error_id`` is read from the input record when present; otherwise it is
    derived on the fly via :func:`record_id` for back-compat with older
    prefilter outputs.
    """
    lookup: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for error in load_unique_errors(errors_file):
        task_id = error["task_id"]
        output_grid = error["output_grid"]
        error_id = error.get("error_id") or record_id(task_id, output_grid)
        common = {
            "error_id": error_id,
            "task_id": task_id,
            "output_grid": output_grid,
            "attempt_count": error.get("attempt_count"),
            "dataset": error.get("dataset"),
            "source": error.get("source"),
        }

        descriptions = error.get("descriptions") or []
        for desc in descriptions:
            rid = desc.get("response_id")
            if not rid:
                continue
            lookup[(error_id, rid)] = {**common, "description": desc.get("text", "")}

        # Bucket for the no-NL synthesis path (response_id is None there).
        lookup.setdefault((error_id, None), {**common, "description": ""})

    return lookup


def build_control_error_lookup(
    task_ids: List[str],
    task_dir: Path,
) -> Dict[Tuple[str, Optional[str]], Dict[str, Any]]:
    """Build the ``(error_id, response_id)`` lookup for a control-mode run.

    Sources synthetic records from :func:`build_control_error_records` and
    keys them by ``(error_id, None)`` — the same key that
    ``build_prompt_records`` produces for control records (empty
    ``descriptions`` → no-NL → ``response_id=None``).
    """
    lookup: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for record in build_control_error_records(task_ids, task_dir):
        error_id = record_id(record["task_id"], record["output_grid"])
        lookup[(error_id, None)] = {
            "error_id": error_id,
            "task_id": record["task_id"],
            "output_grid": record["output_grid"],
            "attempt_count": record["attempt_count"],
            "dataset": record["dataset"],
            "source": record["source"],
            "description": "",
        }
    return lookup


# ---------------------------------------------------------------------------
# Core evaluation for a single record
# ---------------------------------------------------------------------------


def evaluate_record(
    record: Dict[str, Any],
    solvers: Dict[str, str],
    error_lookup: Dict[Tuple[str, Optional[str]], Dict[str, Any]],
    task_dir: Path,
    rewriter: ProgramRewriter,
    dsl_namespace: dict,
    timeout_s: float = DEFAULT_PROGRAM_TIMEOUT_S,
) -> Dict[str, Any]:
    """Evaluate one synthesis result record and return a flat dict of metrics."""
    task_id = record["task_id"]
    response_id = record["response_id"]
    error_id = record["error_id"]

    entry = error_lookup.get((error_id, response_id))
    participant_grid = entry["output_grid"] if entry else None
    participant_description = entry["description"] if entry else ""
    attempt_count = entry["attempt_count"] if entry else None
    dataset = entry["dataset"] if entry else None
    source = entry["source"] if entry else None

    result: Dict[str, Any] = {
        "error_id": error_id,
        "task_id": task_id,
        "response_id": response_id,
        "description": participant_description,
        "dataset": dataset,
        "source": source,
        "attempt_count": attempt_count,
        "entry_index": record.get("entry_index"),
        "sample_index": record.get("sample_index", 0),
        "is_syntactically_valid": False,
        "parse_error": None,
        "execution_success": False,
        "runtime_error": None,
        "exact_match": None,
        "levenshtein_distance": None,
        "hamming_distance": None,
        "grid_size_match": None,
        "rewritten_program": None,
        "original_canonical": None,
        "modifications": None,
        "original_line_count": None,
        "rewritten_line_count": None,
        "line_count_delta": None,
        "mod_count": 0,
        "effective_mod_count": 0,
        "has_nested_violation": False,
        "identity_count": 0,
        "is_identity_output": False,
        "is_gt_output": False,
        "mismatch_proportion": None,
        "is_meaningful_error": False,
        "target_error_grid": participant_grid,
        "training_matches": None,
        "alternative_rule": False,
        "partial_match": False,
        "explains_in_order": None,
    }

    solver_source = solvers.get(task_id)
    if not solver_source:
        result["parse_error"] = f"No solver found for task {task_id}"
        return result

    try:
        rewriter.parse(solver_source)
        original_canonical = rewriter.clean().rewrite()
        result["original_canonical"] = original_canonical
        result["original_program"] = solver_source
        result["original_line_count"] = count_program_body_lines(original_canonical)
    except Exception as e:
        print(f"Parse error for task {task_id}: {e}")

    mods_data = record.get("modifications")
    if not mods_data:
        result["parse_error"] = "No modifications in record"
        return result

    if isinstance(mods_data, dict):
        if "modifications" not in mods_data:
            # e.g. {"error": "Invalid JSON", "raw": ...} from the response parser;
            # defaulting to [] would score it as the unedited ground truth.
            result["parse_error"] = mods_data.get("error") or "Missing 'modifications' key"
            return result
        mod_list = mods_data["modifications"]
    elif isinstance(mods_data, list):
        mod_list = mods_data
    else:
        result["parse_error"] = f"Unexpected modifications format: {type(mods_data)}"
        return result

    if not isinstance(mod_list, list):
        result["parse_error"] = f"Modifications list is not a list: {type(mod_list)}"
        return result

    # Only RHS replacements exist; an explicit other action cannot come from
    # the response schema, so reject it rather than guess.
    unsupported = sorted(
        {m.get("action") for m in mod_list if isinstance(m, dict)} - {None, "modify"}
    )
    if unsupported:
        result["parse_error"] = f"Unsupported edit action(s): {unsupported}"
        return result

    try:
        # model_construct skips validation, so a missing replacement flows
        # through as None and fails at execution.
        line_mods = [
            ModifyOnlyLineModification.model_construct(
                variable_name=m["variable_name"],
                replacement=m.get("replacement"),
            )
            for m in mod_list
        ]
    except (KeyError, TypeError) as e:
        result["parse_error"] = f"Invalid modification entry: {e}"
        return result

    result["mod_count"] = len(line_mods)
    effective_count = 0
    for m in line_mods:
        if m.replacement:
            effective_count += count_calls(m.replacement)
        else:
            effective_count += 1

    original_rhs = extract_original_rhs(solver_source, [m.variable_name for m in line_mods])

    for m in line_mods:
        if m.replacement and has_nested_calls(m.replacement):
            result["has_nested_violation"] = True

    identity_count = 0
    for m in line_mods:
        if m.replacement and re.match(r"\s*identity\s*\(", m.replacement):
            identity_count += 1
    result["identity_count"] = identity_count
    result["effective_mod_count"] = effective_count

    result["modifications"] = [
        {
            "action": "modify",
            "variable_name": m.variable_name,
            "original": original_rhs.get(m.variable_name),
            "replacement": m.replacement,
        }
        for m in line_mods
    ]

    modified_source = apply_modifications(solver_source, line_mods)

    try:
        rewriter.parse(modified_source)
        result["is_syntactically_valid"] = True
    except Exception as e:
        result["parse_error"] = str(e)
        return result

    try:
        rewritten = rewriter.clean().rewrite()
        result["rewritten_program"] = rewritten
    except Exception as e:
        result["is_syntactically_valid"] = False
        result["parse_error"] = f"Rewrite failed: {e}"
        return result

    result["rewritten_line_count"] = count_program_body_lines(rewritten)
    if result["original_line_count"] is not None:
        result["line_count_delta"] = result["rewritten_line_count"] - result["original_line_count"]

    grid_data = load_grid(task_id, [task_dir])
    if not grid_data or "test" not in grid_data or not grid_data["test"]:
        result["runtime_error"] = f"No test grid data for task {task_id}"
        return result

    test_input = grid_data["test"][0]["input"]
    test_input_tuple = grid_to_tuple(test_input)

    try:
        exec_ns = copy.copy(dsl_namespace)
        exec(rewritten, exec_ns)
        solver_fn = exec_ns.get("solver")
        if solver_fn is None:
            result["runtime_error"] = "Rewritten program did not define 'solver'"
            return result
        with time_limit(timeout_s):
            program_output = solver_fn(test_input_tuple)
        result["execution_success"] = True
    except Exception as e:
        result["runtime_error"] = f"{type(e).__name__}: {e}"
        return result

    if not isinstance(program_output, tuple) or not program_output:
        result["runtime_error"] = f"Program output is not a grid: {type(program_output).__name__}"
        result["execution_success"] = True
        return result
    if not all(isinstance(row, tuple) for row in program_output):
        result["runtime_error"] = "Program output rows are not tuples"
        result["execution_success"] = True
        return result

    result["program_output_grid"] = make_json_serializable(program_output)

    test_output = grid_data["test"][0].get("output")
    gt_output_tuple = grid_to_tuple(test_output) if test_output is not None else None

    is_identity = program_output == test_input_tuple
    is_gt = program_output == gt_output_tuple if gt_output_tuple is not None else False
    result["is_identity_output"] = is_identity
    result["is_gt_output"] = is_gt
    result["is_meaningful_error"] = result["execution_success"] and not is_identity and not is_gt

    if participant_grid is None:
        result["runtime_error"] = "Participant output_grid not found in errors file"
        return result

    participant_tuple = grid_to_tuple(participant_grid)
    result["exact_match"] = program_output == participant_tuple

    prog_dims = grid_dimensions(program_output)
    part_dims = grid_dimensions(participant_tuple)
    result["grid_size_match"] = prog_dims == part_dims

    flat_prog = flatten_grid(program_output)
    flat_part = flatten_grid(participant_tuple)
    result["levenshtein_distance"] = levenshtein_distance(flat_prog, flat_part)

    if prog_dims == part_dims and prog_dims[0] > 0 and prog_dims[1] > 0:
        total_cells = prog_dims[0] * prog_dims[1]
        diff_cells = sum(1 for a, b in zip(flat_prog, flat_part) if a != b)
        result["hamming_distance"] = diff_cells / total_cells
    else:
        result["hamming_distance"] = None

    result["mismatch_proportion"] = calculate_mismatch(program_output, participant_tuple)

    train_examples = grid_data.get("train", [])
    if train_examples and solver_fn is not None:
        training_matches = []
        for train_pair in train_examples:
            train_input = train_pair.get("input")
            train_expected = train_pair.get("output")
            if train_input is None or train_expected is None:
                training_matches.append(False)
                continue
            try:
                train_input_tuple = grid_to_tuple(train_input)
                with time_limit(timeout_s):
                    train_output = solver_fn(train_input_tuple)
                train_expected_tuple = grid_to_tuple(train_expected)
                training_matches.append(train_output == train_expected_tuple)
            except Exception:
                training_matches.append(False)

        result["training_matches"] = training_matches
        if all(training_matches):
            result["alternative_rule"] = True
        if any(training_matches):
            result["partial_match"] = True
            k = sum(training_matches)
            result["explains_in_order"] = training_matches == [True] * k + [False] * (
                len(training_matches) - k
            )

    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def print_summary(eval_results: List[Dict[str, Any]]) -> None:
    """Print multi-level summary statistics to stdout."""
    total = len(eval_results)
    if total == 0:
        print("No results to summarize.")
        return

    valid = [r for r in eval_results if r["is_syntactically_valid"]]
    executed = [r for r in eval_results if r["execution_success"]]
    matched = [r for r in executed if r["exact_match"]]

    lev_distances = [r["levenshtein_distance"] for r in executed if r["levenshtein_distance"] is not None]
    ham_distances = [r["hamming_distance"] for r in executed if r["hamming_distance"] is not None]

    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)

    print("\n--- 1. Global (per-sample) ---")
    print(f"  Total records:            {total}")
    print(f"  Syntactically valid:      {len(valid)} ({len(valid)/total*100:.1f}%)")
    print(f"  Execution success:        {len(executed)} ({len(executed)/total*100:.1f}%)")
    if executed:
        print(f"  Exact match (of executed):{len(matched)} ({len(matched)/len(executed)*100:.1f}%)")
        print(f"  Exact match (of total):   {len(matched)} ({len(matched)/total*100:.1f}%)")
    if lev_distances:
        arr = np.array(lev_distances)
        print(
            f"  Levenshtein distance:     mean={np.mean(arr):.2f}  median={np.median(arr):.1f}  "
            f"min={np.min(arr)}  max={np.max(arr)}"
        )
    if ham_distances:
        arr = np.array(ham_distances)
        print(
            f"  Hamming distance (norm):  mean={np.mean(arr):.4f}  median={np.median(arr):.4f}  "
            f"min={np.min(arr):.4f}  max={np.max(arr):.4f}"
        )

    deltas = [r["line_count_delta"] for r in valid if r["line_count_delta"] is not None]
    if deltas:
        darr = np.array(deltas)
        shorter = int(np.sum(darr < 0))
        same = int(np.sum(darr == 0))
        longer = int(np.sum(darr > 0))
        n = len(deltas)
        print(
            f"  Program length change:    shorter={shorter} ({shorter/n*100:.1f}%)  "
            f"same={same} ({same/n*100:.1f}%)  longer={longer} ({longer/n*100:.1f}%)"
        )
        print(
            f"  Line count delta:         mean={np.mean(darr):.2f}  median={np.median(darr):.1f}  "
            f"min={np.min(darr)}  max={np.max(darr)}"
        )

    mod_counts = [r["mod_count"] for r in eval_results]
    eff_counts = [r["effective_mod_count"] for r in eval_results]
    if mod_counts:
        m_arr = np.array(mod_counts)
        e_arr = np.array(eff_counts)
        print("\n  Modification Counts (model-claimed):")
        print(
            f"    mean={np.mean(m_arr):.2f}  median={np.median(m_arr):.1f}  "
            f"min={np.min(m_arr)}  max={np.max(m_arr)}"
        )
        print("  Effective Counts (unnested):")
        print(
            f"    mean={np.mean(e_arr):.2f}  median={np.median(e_arr):.1f}  "
            f"min={np.min(e_arr)}  max={np.max(e_arr)}"
        )
        diff = e_arr - m_arr
        if np.any(diff > 0):
            nested_samples = int(np.sum(diff > 0))
            print(
                f"  Samples with nested calls: {nested_samples} ({nested_samples/total*100:.1f}%)"
            )
            print(f"    Avg increase when nested: +{np.mean(diff[diff > 0]):.2f} lines")

    violations = [r for r in eval_results if r.get("has_nested_violation")]
    num_violations = len(violations)
    print(f"\n  Nested function violations: {num_violations} ({num_violations/total*100:.1f}%)")

    entries: Dict[Tuple[str, Optional[str]], List[Dict]] = defaultdict(list)
    for r in eval_results:
        entries[(r["error_id"], r["response_id"])].append(r)

    num_entries = len(entries)
    entry_avg_match_rates: List[float] = []
    entry_any_match = 0
    entry_best_lev: List[int] = []
    entry_best_ham: List[float] = []
    entry_any_valid = 0
    entry_any_executed = 0

    for key, samples in entries.items():
        executed_samples = [s for s in samples if s["execution_success"]]
        matched_samples = [s for s in executed_samples if s["exact_match"]]
        valid_samples = [s for s in samples if s["is_syntactically_valid"]]

        if valid_samples:
            entry_any_valid += 1
        if executed_samples:
            entry_any_executed += 1
        if matched_samples:
            entry_any_match += 1

        entry_avg_match_rates.append(len(matched_samples) / len(samples) if executed_samples else 0.0)

        lev_vals = [s["levenshtein_distance"] for s in executed_samples if s["levenshtein_distance"] is not None]
        ham_vals = [s["hamming_distance"] for s in executed_samples if s["hamming_distance"] is not None]
        if lev_vals:
            entry_best_lev.append(min(lev_vals))
        if ham_vals:
            entry_best_ham.append(min(ham_vals))

    print(f"\n--- 2. Per-entry (n={num_entries} unique entries) ---")
    print(
        f"  Entries with any valid parse:     {entry_any_valid} ({entry_any_valid/num_entries*100:.1f}%)"
    )
    print(
        f"  Entries with any execution:       {entry_any_executed} ({entry_any_executed/num_entries*100:.1f}%)"
    )
    print(f"  Average exact-match rate:         {np.mean(entry_avg_match_rates):.4f}")
    print(
        f"  Any-match (hit) rate:             {entry_any_match}/{num_entries} ({entry_any_match/num_entries*100:.1f}%)"
    )
    if entry_best_lev:
        arr = np.array(entry_best_lev)
        print(
            f"  Best Levenshtein per entry:       mean={np.mean(arr):.2f}  median={np.median(arr):.1f}  "
            f"min={np.min(arr)}  max={np.max(arr)}"
        )
    if entry_best_ham:
        arr = np.array(entry_best_ham)
        print(
            f"  Best Hamming per entry:           mean={np.mean(arr):.4f}  median={np.median(arr):.4f}"
        )

    # Per-task grouping: walk eval_results so we can map error_id -> task_id
    # (entries are keyed on error_id, which doesn't carry task back on its own).
    error_to_task: Dict[str, str] = {r["error_id"]: r["task_id"] for r in eval_results}
    tasks: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
    for key in entries:
        tasks[error_to_task[key[0]]].append(key)

    num_tasks = len(tasks)
    task_any_match = 0
    task_entry_match_rates: List[float] = []
    for _task_id, entry_keys in tasks.items():
        matched_entries = 0
        for ek in entry_keys:
            samples = entries[ek]
            if any(s["exact_match"] for s in samples if s["execution_success"]):
                matched_entries += 1
        rate = matched_entries / len(entry_keys)
        task_entry_match_rates.append(rate)
        if matched_entries > 0:
            task_any_match += 1

    print(f"\n--- 3. Per-task (n={num_tasks} unique tasks) ---")
    print(
        f"  Task-level any-match:             {task_any_match}/{num_tasks} ({task_any_match/num_tasks*100:.1f}%)"
    )
    print(f"  Mean entry-match rate per task:    {np.mean(task_entry_match_rates):.4f}")

    # Per-unique-error grouping by error_id (already unique per
    # (task_id, output_grid) by construction).
    unique_errors: Dict[str, List[Dict]] = defaultdict(list)
    for r in eval_results:
        if r.get("target_error_grid") is None:
            continue
        unique_errors[r["error_id"]].append(r)

    num_unique_errors = len(unique_errors)
    unique_error_any_match = sum(
        1 for recs in unique_errors.values()
        if any(r["exact_match"] for r in recs if r.get("exact_match") is not None)
    )

    print(f"\n--- 4. Per-unique-error (n={num_unique_errors} unique error grids) ---")
    print(f"  Unique error grids:               {num_unique_errors}")
    if num_unique_errors > 0:
        print(
            f"  Unique error any-match rate:      {unique_error_any_match}/{num_unique_errors} "
            f"({unique_error_any_match/num_unique_errors*100:.1f}%)"
        )

    programs_with_training = [r for r in eval_results if r.get("training_matches") is not None]
    alternative_rules = [r for r in programs_with_training if r["alternative_rule"]]
    partial_matches = [r for r in programs_with_training if r["partial_match"]]
    print("\n--- 5. Training-example evaluation (per-sample) ---")
    print(f"  Programs evaluated on training:   {len(programs_with_training)}")
    if programs_with_training:
        print(
            f"  Alternative rules (all train):    {len(alternative_rules)} "
            f"({len(alternative_rules)/len(programs_with_training)*100:.1f}%)"
        )
        print(
            f"  Partial matches (any train):      {len(partial_matches)} "
            f"({len(partial_matches)/len(programs_with_training)*100:.1f}%)"
        )

    invalid = [r for r in eval_results if not r["is_syntactically_valid"]]
    runtime_failed = [
        r for r in eval_results if r["is_syntactically_valid"] and not r["execution_success"]
    ]
    if invalid or runtime_failed:
        print("\n--- Failure breakdown ---")
        if invalid:
            parse_errors: Dict[str, int] = defaultdict(int)
            for r in invalid:
                err = r.get("parse_error", "unknown")
                short_err = err[:80] if err else "unknown"
                parse_errors[short_err] += 1
            print(f"  Parse failures ({len(invalid)}):")
            for err, count in sorted(parse_errors.items(), key=lambda x: -x[1])[:10]:
                print(f"    [{count:>4}] {err}")
        if runtime_failed:
            runtime_errors: Dict[str, int] = defaultdict(int)
            for r in runtime_failed:
                err = r.get("runtime_error", "unknown")
                short_err = err[:80] if err else "unknown"
                runtime_errors[short_err] += 1
            print(f"  Runtime failures ({len(runtime_failed)}):")
            for err, count in sorted(runtime_errors.items(), key=lambda x: -x[1])[:10]:
                print(f"    [{count:>4}] {err}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model-generated program modifications against participant error grids."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to a synthesis results JSONL (e.g. async_results_*.jsonl).",
    )
    parser.add_argument(
        "--errors-file",
        type=str,
        default=None,
        help=(
            "Path to the unique-errors JSONL used to drive synthesis. "
            "Required unless --control is set; in control mode, optional and "
            "acts as a task-id filter (mirrors synthesize_error_programs.py)."
        ),
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help=(
            "Evaluate a control-mode synthesis run. The participant 'error' "
            "grid for each task is the task's correct test output, and the "
            "lookup is built on the fly from solvers ∩ task-dir (optionally "
            "filtered by --errors-file's task_ids)."
        ),
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        required=True,
        help="Directory containing ARC-format task JSONs (one per task_id).",
    )
    parser.add_argument(
        "--solvers-file",
        type=str,
        default=None,
        help=(
            "Path to a {task_id: [solver_source,...]} JSON map (output of "
            "scripts/select_solver_candidates.py). When given, replaces the "
            "hand-written hodel_dsl.solvers as the reference-program source. "
            "Must be the SAME file (and selection) passed to "
            "synthesize_error_programs.py, so the recorded edits re-apply to the "
            "identical body. Works in both default and --control modes."
        ),
    )
    parser.add_argument(
        "--solver-selection",
        choices=["first", "all"],
        default="first",
        help=(
            "With --solvers-file and multiple solvers per task: 'first' uses the "
            "first body in each bundle (the shortest, given an all-unique export). "
            "'all' (per-task fan-out) is not yet implemented."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one status line per record (useful for debugging).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_PROGRAM_TIMEOUT_S,
        help=(
            "Per-program execution time budget in seconds (default: "
            f"{DEFAULT_PROGRAM_TIMEOUT_S:g}). A program exceeding it on the test "
            "input is recorded as a runtime error; on a train pair it counts as "
            "a mismatch. Set <= 0 to disable (NOT recommended: a pathological "
            "synthesized program can hang the whole run)."
        ),
    )

    args = parser.parse_args()

    if args.solver_selection == "all":
        parser.error("--solver-selection all is not yet implemented; use first")

    if not args.control and args.errors_file is None:
        parser.error("--errors-file is required unless --control is set")

    input_path = Path(args.input_file)
    errors_path = Path(args.errors_file) if args.errors_file else None
    task_dir = Path(args.task_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return
    if errors_path is not None and not errors_path.exists():
        print(f"Error: Errors file {errors_path} not found.")
        return
    if not task_dir.exists():
        print(f"Error: Task directory {task_dir} not found.")
        return

    print("Loading resources...")
    if args.solvers_file:
        solvers_path = Path(args.solvers_file)
        if not solvers_path.exists():
            print(f"Error: Solvers file {solvers_path} not found.")
            return
        bundles = load_synthesized_solvers(solvers_path)
        solvers = collapse_solver_bundles(bundles, args.solver_selection)
        print(
            f"  Loaded {len(solvers)} synthesized solvers "
            f"(from {sum(len(v) for v in bundles.values())} bodies in {solvers_path.name})."
        )
    else:
        solvers = load_solvers()
        print(f"  Loaded {len(solvers)} solvers.")

    if args.control:
        print("Mode: CONTROL (synthetic correct-output records)")
        base = set(solvers) & {p.stem for p in task_dir.glob("*.json")}
        if errors_path is not None:
            filter_ids = {rec["task_id"] for rec in load_unique_errors(errors_path)}
            task_ids = sorted(base & filter_ids)
            print(
                f"  Control scope: {len(task_ids)} tasks "
                f"(solvers∩task-dir={len(base)} ∩ errors-file={len(filter_ids)})."
            )
        else:
            task_ids = sorted(base)
            print(
                f"  Control scope: {len(task_ids)} tasks "
                f"(solvers∩task-dir; no errors-file filter)."
            )
        error_lookup = build_control_error_lookup(task_ids, task_dir)
        print(f"  Built control lookup with {len(error_lookup)} keys.")
    else:
        error_lookup = build_error_lookup(errors_path)
        print(f"  Built error lookup with {len(error_lookup)} (error_id, response_id) keys.")

    dsl_path = str(_project_root / "hodel_dsl" / "dsl.py")
    primitives_path = str(_project_root / "hodel_dsl" / "primitives.py")
    rewriter = ProgramRewriter(
        dsl_symbols_path=dsl_path,
        dsl_constants_path=primitives_path,
    )
    print("  ProgramRewriter initialized.")

    dsl_namespace = build_dsl_namespace()
    print(f"  DSL namespace built ({len(dsl_namespace)} symbols).")

    records: List[Dict[str, Any]] = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} result records from {input_path.name}.")

    missing = sum(
        1 for r in records if (r["error_id"], r["response_id"]) not in error_lookup
    )
    if missing:
        print(
            f"  Warning: {missing}/{len(records)} result rows have no matching "
            f"lookup entry; they will be flagged with 'Participant output_grid "
            f"not found' and skipped at the grid-comparison stage."
        )

    print(f"\nEvaluating {len(records)} records...")
    eval_results: List[Dict[str, Any]] = []
    for record in tqdm(records, desc="Evaluating"):
        result = evaluate_record(
            record=record,
            solvers=solvers,
            error_lookup=error_lookup,
            task_dir=task_dir,
            rewriter=rewriter,
            dsl_namespace=dsl_namespace,
            timeout_s=args.timeout,
        )
        eval_results.append(result)

        if args.verbose:
            if result["exact_match"]:
                status = "MATCH"
            elif result["execution_success"]:
                status = "EXEC_OK"
            elif result["is_syntactically_valid"]:
                status = "VALID"
            else:
                status = "INVALID"
            lev = result["levenshtein_distance"]
            lev_str = f"lev={lev}" if lev is not None else ""
            print(
                f"  [{status:>7}] task={result['task_id']}  "
                f"resp={str(result['response_id'])[:8]}...  "
                f"sample={result['sample_index']}  {lev_str}"
            )

    output_file = output_dir / f"eval_{input_path.stem}.jsonl"
    with open(output_file, "w") as f:
        for result in eval_results:
            f.write(json.dumps(result) + "\n")
    print(f"\nDetailed results saved to {output_file}")

    print_summary(eval_results)


if __name__ == "__main__":
    main()
