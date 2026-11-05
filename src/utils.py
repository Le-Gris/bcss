"""Helpers shared across the pipeline.

Loading the DSL, the reference programs and JSONL files; formatting grids and
the DSL for prompts; grid conversion, hashing (``grid_hash``) and mismatch
computation; and a per-program execution time limit.
"""

import ast
import functools
import hashlib
import json
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOLVERS_PATH = PROJECT_ROOT / "hodel_dsl/solvers.py"
DSL_PATH = PROJECT_ROOT / "hodel_dsl/dsl.py"
ARC_TYPES_PATH = PROJECT_ROOT / "hodel_dsl/arc_types.py"
PRIMITIVES_PATH = PROJECT_ROOT / "hodel_dsl/primitives.py"

# Default per-program execution budget (seconds) for the evaluators.
DEFAULT_PROGRAM_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Execution timeout
# ---------------------------------------------------------------------------


class ProgramTimeout(Exception):
    """Raised when an executed program exceeds its time budget."""


@contextmanager
def time_limit(seconds: float):
    """Abort the wrapped block with :class:`ProgramTimeout` if it runs longer
    than ``seconds`` wall-clock.

    Both evaluators ``exec`` arbitrary synthesized DSL programs and call the
    resulting ``solver(...)``. A pathological program (iterated higher-order
    combinators, unbounded object construction, huge-grid sweeps) can run
    effectively forever and stall the whole serial eval; wrapping each call
    records it as a runtime error instead.

    Implementation note: uses ``SIGALRM``, so it is Unix-only and fires only in
    the main thread — which is where the serial eval loops run. CPython checks
    pending signals between bytecode instructions, so pure-Python DSL loops are
    interruptible. ``seconds <= 0`` (or ``None``) disables the limit.
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise ProgramTimeout(f"execution exceeded {seconds:g}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_unique_errors(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream the unique-errors JSONL, yielding one record per non-blank line.

    Each record is expected to contain at minimum::

        {
          "task_id": str,
          "dataset": str,
          "source": str,
          "output_grid": list[list[int]],
          "attempt_count": int,
          "descriptions": list[{"response_id": str, "text": str, ...}],
          "metadata": dict (optional)
        }

    Field validation is intentionally light — the loader streams records and
    leaves enforcement of required fields to the downstream caller.
    """
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Load all records from one or more JSONL files, skipping blank lines.

    Generic plumbing for analysis scripts; by project convention these are
    pointed at stage-4 ``filter_candidates.py`` output (filtered candidates).
    """
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def load_descriptions(path: Path, min_words: int) -> Dict[str, str]:
    """Load a generic ``{task_id, text}`` JSONL.

    Multiple records for the same task_id are concatenated with newlines.
    Records whose ``text`` has fewer than ``min_words`` whitespace-separated
    tokens are dropped.
    """
    desc: Dict[str, List[str]] = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            tid = rec.get("task_id")
            text = (rec.get("text") or "").strip()
            if not tid or not text:
                continue
            if len(text.split()) < min_words:
                continue
            desc.setdefault(str(tid), []).append(text)
    return {tid: "\n".join(parts) for tid, parts in desc.items()}


# ---------------------------------------------------------------------------
# DSL loading
# ---------------------------------------------------------------------------


def load_solvers(solvers_path: Path = SOLVERS_PATH) -> Dict[str, str]:
    """Parse solvers.py and return a mapping of task_id to its source code."""
    with open(solvers_path, "r") as f:
        source = f.read()
    tree = ast.parse(source)

    solvers: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("solve_"):
            task_id = node.name.replace("solve_", "")
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                solvers[task_id] = segment
    return solvers


def load_synthesized_solvers(path: Path) -> Dict[str, List[str]]:
    """Load a ``{task_id: [full_solver_source, ...]}`` JSON map.

    This is the export produced by ``scripts/select_solver_candidates.py``.
    Each value is a list of **full** solver sources (a complete
    ``def solver(I): ... return O`` per entry, the same shape
    :func:`load_solvers` returns), so downstream consumers can call
    :func:`get_solver_body` on any entry interchangeably with the hand-written
    Hodel solvers.

    Validation is deliberately strict on structure (so a malformed file fails
    loudly here rather than silently mis-rendering prompts) but lenient on
    coverage: an empty list is allowed and simply means "no synthesized solver
    for this task" — such tasks are skipped gracefully downstream.

    Raises:
        ValueError: if the top level is not an object, a value is not a list,
            or any list element is not a string.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a top-level JSON object, got {type(data).__name__}."
        )
    for task_id, bodies in data.items():
        if not isinstance(bodies, list) or any(not isinstance(b, str) for b in bodies):
            raise ValueError(
                f"{path}: value for task {task_id!r} must be a list of strings."
            )
    return data


def collapse_solver_bundles(
    bundles: Dict[str, List[str]], selection: str = "first"
) -> Dict[str, str]:
    """Collapse a ``{task_id: [solver_source, ...]}`` bundle map to one body per task.

    Single source of truth for the reference-solver selection rule shared by
    ``synthesize_error_programs.py`` (stage 2) and
    ``evaluate_synthesized_error_programs.py`` (stage 3).  Both stages **must**
    pick the identical body: stage 2 records line-edits against it and stage 3
    re-applies them, so any divergence makes ``apply_modifications`` target the
    wrong lines.

    ``selection="first"`` keeps ``bundle[0]`` — the shortest, given an
    ``all-unique`` export (stage D pre-sorts shortest-first).  Tasks with an
    empty bundle are dropped (no synthesized solver for that task).

    Raises:
        ValueError: for any ``selection`` other than ``"first"`` (``"all"``,
            per-``(task, solver)`` fan-out, is not yet implemented).
    """
    if selection != "first":
        raise ValueError(
            f"solver selection {selection!r} is not implemented; use 'first'."
        )
    return {task_id: bodies[0] for task_id, bodies in bundles.items() if bodies}


def load_dsl_api(dsl_path: Path = DSL_PATH) -> Dict[str, Dict[str, str]]:
    """Parse dsl.py and return ``name -> {"signature", "docstring"}``."""
    with open(dsl_path, "r") as f:
        source_code = f.read()
    tree = ast.parse(source_code)

    api: Dict[str, Dict[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            end_lineno = node.body[0].lineno - 1 if node.body else node.lineno
            signature_lines = source_code.splitlines()[node.lineno - 1 : end_lineno]
            signature = "\n".join(signature_lines).strip()
            if signature.endswith(":"):
                signature = signature[:-1]
            docstring = ast.get_docstring(node)
            api[node.name] = {"signature": signature, "docstring": docstring or ""}
    return api


def load_dsl_source(dsl_path: Path = DSL_PATH) -> Dict[str, str]:
    """Parse dsl.py and return ``name -> full source code``."""
    with open(dsl_path, "r") as f:
        source_code = f.read()
    tree = ast.parse(source_code)

    sources: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            segment = ast.get_source_segment(source_code, node)
            if segment:
                sources[node.name] = segment
    return sources


def load_dsl_preamble(
    arc_types_path: Path = ARC_TYPES_PATH,
    primitives_path: Path = PRIMITIVES_PATH,
) -> str:
    """Load type aliases from arc_types.py and primitive constants from primitives.py."""
    sections: List[str] = []

    sections.append("# Type aliases")
    with open(arc_types_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(("class ", "def ", "CUSTOM_TYPE_NAMES")):
                break
            if "=" in stripped and stripped[0].isupper():
                sections.append(line.rstrip())

    sections.append("")
    sections.append("# Primitive constants")
    with open(primitives_path, "r") as f:
        in_constants = False
        for line in f:
            stripped = line.strip()
            if not in_constants:
                if stripped == "F = False":
                    in_constants = True
                else:
                    continue
            if stripped.startswith(("PRIMITIVE_CONSTANTS", "PRIMITIVE_FUNCTIONS", "def ")):
                break
            sections.append(line.rstrip())

    return "\n".join(sections).strip()


def get_solver_body(solver_source: str) -> str:
    """Strip the ``def`` line, dedent the body, and drop the trailing ``return``."""
    lines = solver_source.splitlines()
    body_lines: List[str] = []
    in_body = False
    for line in lines:
        if in_body:
            body_lines.append(line)
        elif line.strip().startswith("def "):
            in_body = True

    dedented: List[str] = []
    for line in body_lines:
        if line.startswith("    "):
            dedented.append(line[4:])
        else:
            dedented.append(line)

    result: List[str] = []
    for line in dedented:
        if line.strip().startswith("return "):
            continue
        result.append(line)
    return "\n".join(result).strip()


def get_referenced_dsl_functions(
    source_code: str, dsl_api: Dict[str, Any]
) -> List[str]:
    """Identify which DSL functions are referenced in a given piece of source code."""
    referenced: List[str] = []
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in dsl_api:
                referenced.append(node.func.id)
    return list(set(referenced))


def format_dsl_api_for_prompt(
    api: Dict[str, Dict[str, str]], function_names: List[str]
) -> str:
    """Format the signatures + docstrings of specific DSL functions for a prompt."""
    lines: List[str] = []
    for name in function_names:
        if name in api:
            entry = api[name]
            lines.append(f"{entry['signature']}")
            if entry["docstring"]:
                lines.append(f'    """{entry["docstring"]}"""')
            lines.append("")
    return "\n".join(lines)


def format_dsl_source_for_prompt(
    dsl_sources: Dict[str, str], function_names: List[str]
) -> str:
    """Format the full source of specific DSL functions for a prompt."""
    blocks = [dsl_sources[name] for name in function_names if name in dsl_sources]
    return "\n\n".join(blocks)


def format_full_dsl_for_prompt(
    dsl_api: Dict[str, Dict[str, str]],
    dsl_sources: Dict[str, str],
    preamble: str,
    api_only: bool = False,
) -> str:
    """Format the complete DSL for a prompt, with types and primitives prepended."""
    parts: List[str] = [preamble, ""]
    func_names = list(dsl_sources.keys())  # stable file order
    if api_only:
        parts.append(format_dsl_api_for_prompt(dsl_api, func_names))
    else:
        parts.append(format_dsl_source_for_prompt(dsl_sources, func_names))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Grid I/O
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1000)
def _load_grid_cached(task_id: str, grid_dirs: Tuple[Path, ...]) -> Optional[Dict[str, Any]]:
    for grid_dir in grid_dirs:
        path = grid_dir / f"{task_id}.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None


def load_grid(task_id: str, grid_dirs: Tuple[Path, ...] | List[Path]) -> Optional[Dict[str, Any]]:
    """Load the ARC-format task JSON for *task_id* from one of *grid_dirs*.

    *grid_dirs* is required: callers must thread it through from the
    ``--task-dir`` CLI flag.  Each entry is a directory containing
    ``<task_id>.json`` files in the standard ARC schema (train / test lists
    of {input, output} grids).
    """
    return _load_grid_cached(task_id, tuple(grid_dirs))


def format_grid(grid: List[List[int]]) -> str:
    """Format a grid as space-separated rows."""
    return "\n".join(" ".join(map(str, row)) for row in grid)


def format_task_grids_for_prompt(grid_data: Dict[str, Any]) -> str:
    """Format all training and test grids for a prompt."""
    lines: List[str] = []
    for i, example in enumerate(grid_data.get("train", [])):
        lines.append(f"Example {i+1} Input:")
        lines.append(format_grid(example["input"]))
        lines.append(f"Example {i+1} Output:")
        lines.append(format_grid(example["output"]))
        lines.append("")

    test_examples = grid_data.get("test", [])
    if test_examples:
        example = test_examples[0]
        lines.append("Test Example Input:")
        lines.append(format_grid(example["input"]))
        if "output" in example:
            lines.append("Test Example Output:")
            lines.append(format_grid(example["output"]))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grid conversion helpers
# ---------------------------------------------------------------------------


def grid_to_tuple(grid: Any) -> tuple:
    """Convert a list-of-lists grid to a tuple-of-tuples (idempotent)."""
    if isinstance(grid, tuple) and all(isinstance(row, tuple) for row in grid):
        return grid
    return tuple(tuple(row) for row in grid)


def grid_hash(grid: Any) -> str:
    """Stable 16-char hash for a grid."""
    return hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest()[:16]


def record_id(task_id: str, grid: Any) -> str:
    """Deterministic 32-char id for a ``(task_id, grid)`` pair.

    Composed via :func:`grid_hash` so a record's id stays consistent with any
    other keying that already uses grid_hash.  Stages 2-4 all carry this id
    end-to-end as ``error_id``.
    """
    payload = f"{task_id}:{grid_hash(grid)}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def grid_to_pipe(grid: list) -> str:
    """Convert a list-of-lists grid to a pipe-separated string (CSV encoding)."""
    return "|" + "|".join("".join(str(c) for c in row) for row in grid) + "|"


def pipe_to_grid(pipe: str) -> list:
    """Inverse of grid_to_pipe."""
    rows = pipe.strip("|").split("|")
    return [[int(c) for c in row] for row in rows]


def flatten_grid(grid: tuple) -> tuple:
    """Flatten a 2D grid (tuple-of-tuples) into a 1D tuple of cell values."""
    return tuple(cell for row in grid for cell in row)


def grid_dimensions(grid: tuple) -> Tuple[int, int]:
    """Return ``(rows, cols)`` for a grid."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    return rows, cols


@functools.lru_cache(maxsize=8192)
def _cached_grid_array(grid_key: tuple) -> Optional[np.ndarray]:
    if not grid_key or not grid_key[0]:
        return None
    ncols = len(grid_key[0])
    if any(len(row) != ncols for row in grid_key):
        return None
    return np.array(grid_key, dtype=np.int8)


def grid_array(grid: Any) -> Optional[np.ndarray]:
    """Return a numpy view of *grid*, cached by value. ``None`` for ragged grids."""
    try:
        return _cached_grid_array(grid_to_tuple(grid))
    except (TypeError, ValueError):
        return None


def calculate_mismatch(grid1: Any, grid2: Any) -> float:
    """Cell-level mismatch rate between two grids (0.0 to 1.0). Returns 1.0 for shape mismatch or invalid grids."""
    a1 = grid_array(grid1)
    a2 = grid_array(grid2)
    if a1 is None or a2 is None:
        return 1.0
    if a1.shape != a2.shape:
        return 1.0
    if a1.size == 0:
        return 0.0
    return float((a1 != a2).mean())


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def unique_path(path: Path) -> Path:
    """Return *path* unchanged if it doesn't exist; otherwise append _run2, _run3, ..."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    run = 2
    while True:
        candidate = parent / f"{stem}_run{run}{suffix}"
        if not candidate.exists():
            return candidate
        run += 1
