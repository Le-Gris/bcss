"""Build a static HTML site for browsing the accepted error programs.

The index lists every task with its error and program counts and rule types.
Each task page shows the task's grids, the reference program, and one card per
error with the participant's output and every program that reproduces it,
with its relative length change, rule type and (with
``--classification-file``) error category.

Reads accepted programs (the output of ``filter_candidates.py``) and the ARC
task JSONs. Writes ``index.html``, ``styles.css`` and ``tasks/<task_id>.html``
to ``--output-dir``.

Usage:
    uv run python scripts/visualize_error_pages.py \\
        --candidates-file OSF/03_candidates/human/training.jsonl \\
        --classification-file OSF/04_classifications/human/training.jsonl \\
        --task-dir external/arc-agi-1/training --output-dir error_pages
"""

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.rule_types import (  # noqa: E402
    ALTERNATIVE,
    NEITHER,
    PARTIAL,
    UNKNOWN,
    rule_type,
)
from src.utils import (  # noqa: E402
    collapse_solver_bundles,
    grid_hash,
    load_grid,
    load_jsonl_records,
    load_synthesized_solvers,
)

# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/test_visualize_error_pages.py)
# --------------------------------------------------------------------------- #

# ARC colour palette (index -> hex).
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 cyan
    "#870C25",  # 9 maroon
]

def relative_length_change(rec: dict[str, Any]) -> float | None:
    """RLC = ``line_count_delta / original_line_count`` (None if undefined).

    Matches the definition in ``analyses/analyze_program_simplicity.py``.
    Returns ``None`` when either field is missing or the original program has
    zero body lines (avoids division by zero).
    """
    delta = rec.get("line_count_delta")
    original = rec.get("original_line_count")
    if delta is None or not original:
        return None
    return delta / original


# Every grid is fit into a fixed ~GRID_BOX_PX bounding box: cells stay square
# and their pixel size scales down as the grid grows, so a 30x30 grid occupies
# roughly the same footprint as a 3x3 one (cell sizes just differ).
GRID_BOX_PX = 240
CELL_MAX_PX = 24
CELL_MIN_PX = 2


def cell_size_px(rows: int, cols: int) -> int:
    """Square cell size (px) so the grid's largest side fits ``GRID_BOX_PX``.

    Clamped to ``[CELL_MIN_PX, CELL_MAX_PX]`` so tiny grids don't balloon and
    huge grids stay legible.
    """
    max_dim = max(rows, cols, 1)
    return max(CELL_MIN_PX, min(CELL_MAX_PX, GRID_BOX_PX // max_dim))


def render_grid(grid: Any, title: str = "") -> str:
    """Render an ARC grid as an HTML ``<table>`` with palette-classed cells.

    Cells in 0-9 get class ``c{v}``; anything else falls back to ``c-other``.
    Cells are square and sized (via a per-table ``--cell`` CSS variable) so the
    whole grid fits a fixed bounding box — large grids scale down rather than
    sprawling.  Applies uniformly to task, behavioral, and program-output grids.
    """
    if not grid:
        label = html.escape(title) if title else "Grid"
        return f'<div class="grid-wrap"><span class="grid-label">{label}: empty</span></div>'

    rows = len(grid)
    cols = max((len(r) for r in grid), default=0)
    cell = cell_size_px(rows, cols)

    parts: list[str] = ['<div class="grid-wrap">']
    if title:
        parts.append(f'<span class="grid-label">{html.escape(title)}</span>')
    parts.append(f'<table class="grid" style="--cell:{cell}px"><tbody>')
    for row in grid:
        parts.append("<tr>")
        for cell in row:
            cls = f"c{cell}" if isinstance(cell, int) and 0 <= cell <= 9 else "c-other"
            parts.append(f'<td class="{cls}"></td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def render_program(source: str | None) -> str:
    """Render program source as an escaped ``<pre><code>`` block."""
    text = source or ""
    return f'<pre class="program"><code>{html.escape(text)}</code></pre>'


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_classification(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load an optional classification lookup keyed by ``program_id``.

    Accepts a ``.jsonl`` (one object per line) or ``.csv`` file.  Each entry is
    keyed by ``program_id`` when present (per-program badges), else falls back to
    ``error_id`` (legacy per-error badges).  The remaining fields are the
    classification payload (a ``category`` / ``error_category`` plus arbitrary
    extras); both id keys are stripped from the payload.  Returns an empty dict
    when ``path`` is None.
    """
    if path is None:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".csv":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = list(load_jsonl_records([path]))
    for row in rows:
        key = row.get("program_id") or row.get("error_id")
        if key:
            lookup[key] = {
                k: v for k, v in row.items() if k not in ("program_id", "error_id")
            }
    return lookup


def classification_category(payload: dict[str, Any] | None) -> str | None:
    """Pull the primary category label from a classification payload."""
    if not payload:
        return None
    cat = payload.get("category") or payload.get("error_category")
    return str(cat) if cat else None


def reference_program(
    task_id: str,
    records: list[dict[str, Any]],
    solver_override: dict[str, str] | None,
) -> tuple[str, str]:
    """Return ``(source, origin_label)`` for a task's reference program.

    Prefers a ``--solvers-file`` override; otherwise the record's embedded
    ``original_program`` (falling back to ``original_canonical``).
    """
    if solver_override and task_id in solver_override:
        return solver_override[task_id], "synthesized (solver file)"
    src = records[0].get("original_program") or records[0].get("original_canonical") or ""
    label = "synthesized" if "def solver(" in src else "ground truth"
    return src, label


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

STYLES = """\
:root { --cell: 20px; }
* { box-sizing: border-box; }
body {
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0; padding: 1.5rem 2rem; color: #1a1a1a; background: #fafafa;
}
h1 { font-size: 1.4rem; margin: 0 0 1rem; }
h2 { font-size: 1.1rem; margin: 1.5rem 0 .5rem; }
a { color: #0066cc; text-decoration: none; }
a:hover { text-decoration: underline; }
.back { display: inline-block; margin-bottom: 1rem; }

/* index table */
table.index { border-collapse: collapse; width: 100%; background: #fff; }
table.index th, table.index td {
  border: 1px solid #e2e2e2; padding: .35rem .6rem; text-align: left;
}
table.index th { background: #f0f0f0; }
table.index td.num { text-align: right; font-variant-numeric: tabular-nums; }

/* grids */
.grid-row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
/* align-items: flex-start is load-bearing.  A column flex container stretches
   its children to the cross size by default, so a grid narrower than its own
   label (e.g. an 18x2 grid under "perturbed out (target_error_grid)") had its
   table stretched to the label's width -- and with table-layout: fixed the
   cells absorb that width, rendering as wide rectangles instead of squares. */
.grid-wrap { display: inline-flex; flex-direction: column; gap: .2rem;
  align-items: flex-start; }
.grid-label { font-size: .75rem; color: #666; }
/* Fixed square cells (ARC-viewer style): every cell is var(--cell) px square,
   regardless of grid dimensions; larger grids simply occupy more space. */
table.grid { border-collapse: collapse; table-layout: fixed; background: #fff;
  width: auto; align-self: flex-start; }
table.grid td {
  width: var(--cell); height: var(--cell);
  min-width: var(--cell); max-width: var(--cell);
  padding: 0; border: 1px solid #555;
}
"""

# Append the ARC palette to the stylesheet (one rule per colour, defined once).
STYLES += "".join(
    f"table.grid td.c{i} {{ background: {c}; }}\n" for i, c in enumerate(ARC_COLORS)
)
STYLES += "table.grid td.c-other { background: #ffcdd2; }\n"

STYLES += """\
pre.program {
  background: #1e1e1e; color: #e6e6e6; padding: .75rem 1rem; border-radius: 6px;
  overflow-x: auto; font: 12px/1.45 "SFMono-Regular", Menlo, Consolas, monospace;
}
.reference pre.program { background: #f4f4f4; color: #1a1a1a; border: 1px solid #ddd; }

/* error cards (collapsible dropdowns) */
details.error-card {
  background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; margin: .6rem 0;
}
summary { list-style: none; }
summary::-webkit-details-marker { display: none; }
details.error-card > summary, details.program-drop > summary {
  cursor: pointer; display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
}
details.error-card > summary { padding: .6rem 1rem; }
details.error-card > summary:hover { background: #f7f9fb; }
details.error-card > summary::before { content: "\\25B6"; font-size: .7rem; color: #999; }
details.error-card[open] > summary::before { content: "\\25BC"; }
details.error-card[open] > summary { border-bottom: 1px solid #eee; }
.error-body { padding: .8rem 1rem 1rem; }
.eid { font-family: monospace; font-size: .85rem; color: #444; }
.desc { margin: .2rem 0 .7rem; color: #333; font-style: italic; }
.muted { color: #888; font-size: .8rem; }

details.program-drop { border-top: 1px dashed #e3e3e3; padding: .55rem 0 .2rem; }
details.program-drop:first-of-type { border-top: none; }
details.program-drop > summary::before { content: "\\25B6"; font-size: .65rem; color: #bbb; }
details.program-drop[open] > summary::before { content: "\\25BC"; }
.mods { font-family: monospace; font-size: .8rem; color: #555; margin: .3rem 0 0; }

/* badges */
.badge { display: inline-block; padding: .1rem .5rem; border-radius: 10px;
  font-size: .75rem; font-weight: 600; white-space: nowrap; }
.b-alternative { background: #d4edda; color: #155724; }
.b-partial { background: #fff3cd; color: #856404; }
.b-neither { background: #f8d7da; color: #721c24; }
.b-unknown { background: #e2e3e5; color: #383d41; }
.b-order { background: #e7e7ff; color: #2a2a7a; }
.b-class { background: #cfe2ff; color: #084298; }
.b-unclassified { background: #e2e3e5; color: #6c757d; }
.b-rlc { background: #eee; color: #333; font-family: monospace; }
.b-mods { background: #ede7f6; color: #4527a0; }
.b-exact { background: #d1ecf1; color: #0c5460; }
"""


def _page(title: str, body: str, css_href: str) -> str:
    """Wrap a body fragment in a minimal HTML document linking the shared CSS."""
    return (
        "<!doctype html>\n<html lang=\"en\"><head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        f'<link rel="stylesheet" href="{css_href}">'
        f"</head><body>\n{body}\n</body></html>\n"
    )


def _rule_badge(label: str, order: str | None) -> str:
    out = f'<span class="badge b-{label}">{label}</span>'
    if order:
        out += f' <span class="badge b-order">{order}</span>'
    return out


def _class_chip(payload: dict[str, Any] | None) -> str:
    """Compact classification badge for an error summary line (no extras)."""
    cat = classification_category(payload)
    if not cat:
        return '<span class="badge b-unclassified">unclassified</span>'
    return f'<span class="badge b-class">{html.escape(cat)}</span>'


def _class_extras(payload: dict[str, Any] | None) -> str:
    """Render the non-category classification fields as a muted detail line."""
    extras = {
        k: v
        for k, v in (payload or {}).items()
        if k not in ("category", "error_category") and v not in (None, "")
    }
    if not extras:
        return ""
    detail = "; ".join(f"{html.escape(str(k))}={html.escape(str(v))}" for k, v in extras.items())
    return f'<div class="mods">{detail}</div>'


def _mods_summary(modifications: list[dict[str, Any]] | None) -> str:
    if not modifications:
        return ""
    bits: list[str] = []
    for m in modifications:
        var = m.get("variable_name", "?")
        bits.append(f"modify {var}: {m.get('original')} → {m.get('replacement')}")
    return '<div class="mods">' + html.escape("  |  ".join(bits)) + "</div>"


def render_task_examples(task_id: str, grid_dirs: list[Path]) -> str:
    """Render train input/output pairs and the test input for a task."""
    data = load_grid(task_id, grid_dirs)
    if not data:
        return '<p class="grid-label">task grids not found in --task-dir</p>'
    parts: list[str] = ['<div class="grid-row">']
    for i, ex in enumerate(data.get("train", []), 1):
        parts.append(render_grid(ex.get("input"), f"train {i} in"))
        parts.append(render_grid(ex.get("output"), f"train {i} out"))
    for ex in data.get("test", []):
        parts.append(render_grid(ex.get("input"), "test in"))
        if ex.get("output") is not None:
            parts.append(render_grid(ex.get("output"), "test out (solution)"))
        break
    parts.append("</div>")
    return "".join(parts)


def render_error_card(
    error_id: str,
    group: list[dict[str, Any]],
    classification: dict[str, dict[str, Any]],
) -> str:
    """Render one collapsible error dropdown.

    The ``<summary>`` carries scannable facts (error id, respondent count, the
    distinct program-level classifications, program count, distinct rule types)
    so a task page reads as a list of one-line errors; expanding reveals the
    behavioral grid and a nested dropdown per synthesized error program, each
    carrying its own classification badge (keyed by ``program_id``).
    """
    head = group[0]
    n_prog = len(group)

    # per-program classification payload, keyed by program_id = grid_hash(program)
    prog_payloads: list[dict[str, Any] | None] = [
        classification.get(grid_hash(rec.get("rewritten_program"))) for rec in group
    ]

    # distinct rule-type labels + classification categories (for the summary)
    rule_labels: list[str] = []
    for rec in group:
        lab, _ = rule_type(rec.get("training_matches"))
        if lab not in rule_labels:
            rule_labels.append(lab)
    seen_cats: list[str] = []
    for payload in prog_payloads:
        cat = classification_category(payload)
        if cat and cat not in seen_cats:
            seen_cats.append(cat)

    parts: list[str] = ['<details class="error-card"><summary>']
    parts.append(f'<span class="eid">{html.escape(error_id[:12])}</span>')
    attempts = head.get("attempt_count")
    if attempts is not None:
        parts.append(f'<span class="muted">{attempts} resp.</span>')
    for cat in seen_cats:
        parts.append(f'<span class="badge b-class">{html.escape(cat)}</span>')
    plural = "s" if n_prog != 1 else ""
    parts.append(f'<span class="muted">{n_prog} program{plural}</span>')
    for lab in rule_labels:
        parts.append(f'<span class="badge b-{lab}">{lab}</span>')
    parts.append("</summary>")

    # --- expanded body ---
    parts.append('<div class="error-body">')
    desc = (head.get("description") or "").strip()
    if desc:
        parts.append(f'<p class="desc">{html.escape(desc)}</p>')
    parts.append('<div class="grid-row">')
    parts.append(render_grid(head.get("target_error_grid"), "behavioral output"))
    parts.append("</div>")

    # one nested dropdown per synthesized error program (auto-open if unique)
    for rec, payload in zip(group, prog_payloads):
        label, order = rule_type(rec.get("training_matches"))
        rlc = relative_length_change(rec)
        open_attr = " open" if n_prog == 1 else ""
        parts.append(f'<details class="program-drop"{open_attr}><summary>')
        parts.append(_class_chip(payload))
        parts.append(_rule_badge(label, order))
        if rlc is not None:
            parts.append(f'<span class="badge b-rlc">RLC {rlc:+.2f}</span>')
        eff_mods = rec.get("effective_mod_count")
        if eff_mods is not None:
            parts.append(f'<span class="badge b-mods">eff. mods {eff_mods}</span>')
        delta = rec.get("line_count_delta")
        if delta is not None:
            parts.append(f'<span class="muted">Δlines {delta:+d}</span>')
        if rec.get("exact_match"):
            parts.append('<span class="badge b-exact">reproduces behavioral output</span>')
        parts.append("</summary>")
        class_extras = _class_extras(payload)
        if class_extras:
            parts.append(class_extras)
        parts.append(render_program(rec.get("rewritten_program")))
        parts.append(_mods_summary(rec.get("modifications")))
        n_mismatch = rec.get("n_mismatch")
        if not rec.get("exact_match") and rec.get("program_output_grid"):
            title = f"program output ({n_mismatch} cells off)" if n_mismatch else "program output"
            parts.append('<div class="grid-row">')
            parts.append(render_grid(rec.get("program_output_grid"), title))
            parts.append("</div>")
        parts.append("</details>")

    parts.append("</div></details>")
    return "".join(parts)


def render_task_page(
    task_id: str,
    records: list[dict[str, Any]],
    grid_dirs: list[Path],
    classification: dict[str, dict[str, Any]],
    solver_override: dict[str, str] | None,
) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["error_id"]].append(rec)

    ref_src, ref_origin = reference_program(task_id, records, solver_override)

    body: list[str] = []
    body.append('<a class="back" href="../index.html">← all tasks</a>')
    body.append(f"<h1>Task {html.escape(task_id)}</h1>")
    body.append("<h2>Task examples</h2>")
    body.append(render_task_examples(task_id, grid_dirs))
    body.append(f"<h2>Reference program <span class='grid-label'>({ref_origin})</span></h2>")
    body.append(f'<div class="reference">{render_program(ref_src)}</div>')
    body.append(f"<h2>Modeled errors ({len(groups)})</h2>")
    for error_id in sorted(groups):
        body.append(render_error_card(error_id, groups[error_id], classification))

    return _page(f"Task {task_id}", "\n".join(body), "../styles.css")


def render_index(records_by_task: dict[str, list[dict[str, Any]]]) -> str:
    body: list[str] = ["<h1>Error-synthesis report</h1>"]
    n_tasks = len(records_by_task)
    n_records = sum(len(r) for r in records_by_task.values())
    body.append(
        f'<p class="grid-label">{n_tasks} tasks · {n_records} error programs</p>'
    )
    body.append("<table class='index'><thead><tr>")
    body.append(
        "<th>task</th><th># errors</th><th># programs</th>"
        "<th>alternative</th><th>partial</th><th>neither</th>"
    )
    body.append("</tr></thead><tbody>")
    for task_id in sorted(records_by_task):
        recs = records_by_task[task_id]
        n_errors = len({r["error_id"] for r in recs})
        counts = {ALTERNATIVE: 0, PARTIAL: 0, NEITHER: 0, UNKNOWN: 0}
        for r in recs:
            counts[rule_type(r.get("training_matches"))[0]] += 1
        body.append("<tr>")
        body.append(f'<td><a href="tasks/{html.escape(task_id)}.html">{html.escape(task_id)}</a></td>')
        body.append(f'<td class="num">{n_errors}</td>')
        body.append(f'<td class="num">{len(recs)}</td>')
        body.append(f'<td class="num">{counts[ALTERNATIVE]}</td>')
        body.append(f'<td class="num">{counts[PARTIAL]}</td>')
        body.append(f'<td class="num">{counts[NEITHER]}</td>')
        body.append("</tr>")
    body.append("</tbody></table>")
    return _page("Error-synthesis report", "\n".join(body), "styles.css")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_site(
    records: list[dict[str, Any]],
    output_dir: Path,
    grid_dirs: list[Path],
    classification: dict[str, dict[str, Any]],
    solver_override: dict[str, str] | None,
) -> int:
    records_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        records_by_task[rec["task_id"]].append(rec)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "styles.css").write_text(STYLES)
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    (output_dir / "index.html").write_text(render_index(records_by_task))
    for task_id, recs in records_by_task.items():
        page = render_task_page(task_id, recs, grid_dirs, classification, solver_override)
        (tasks_dir / f"{task_id}.html").write_text(page)
    return len(records_by_task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-file", type=Path, required=True,
                        help="Stage-4 filtered candidates JSONL (filter_candidates.py output).")
    parser.add_argument("--task-dir", type=Path, action="append", required=True,
                        help="Directory of <task_id>.json files (repeatable).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for index.html, styles.css, tasks/.")
    parser.add_argument("--classification-file", type=Path, default=None,
                        help="Optional error_id -> classification map (.jsonl or .csv).")
    parser.add_argument("--solvers-file", type=Path, default=None,
                        help="Optional synthesized-solver bundle to override the reference program.")
    args = parser.parse_args()

    records = load_jsonl_records([args.candidates_file])
    classification = load_classification(args.classification_file)
    solver_override: dict[str, str] | None = None
    if args.solvers_file:
        solver_override = collapse_solver_bundles(
            load_synthesized_solvers(args.solvers_file), "first"
        )

    n_tasks = build_site(records, args.output_dir, args.task_dir, classification, solver_override)
    print(f"Wrote {n_tasks} task pages + index to {args.output_dir}")


if __name__ == "__main__":
    main()
