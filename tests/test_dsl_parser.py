"""Smoke tests for ``hodel_dsl.dsl_parser.ProgramRewriter``.

Verifies that representative solvers from ``hodel_dsl/solvers.py`` can be
parsed and rewritten to canonical (unnested) form, and that the rewritten
form executes against the DSL namespace.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

from hodel_dsl.dsl_parser import ProgramRewriter
from src.filters import build_dsl_namespace


_DSL_PATH = str(_project_root / "hodel_dsl" / "dsl.py")
_PRIMITIVES_PATH = str(_project_root / "hodel_dsl" / "primitives.py")


def _make_rewriter() -> ProgramRewriter:
    return ProgramRewriter(
        dsl_symbols_path=_DSL_PATH,
        dsl_constants_path=_PRIMITIVES_PATH,
    )


def test_rewriter_handles_simple_one_line_solver():
    """``vmirror(I)`` is already canonical — should round-trip unchanged in spirit."""
    source = (
        "def solve_test(I):\n"
        "    O = vmirror(I)\n"
        "    return O\n"
    )
    rewriter = _make_rewriter()
    rewriter.parse(source)
    rewritten = rewriter.clean().rewrite()
    # Execute the rewritten form and check correctness against vmirror semantics.
    ns = build_dsl_namespace()
    exec(rewritten, ns)
    out = ns["solver"](((1, 2, 3), (4, 5, 6)))
    assert out == ((3, 2, 1), (6, 5, 4))


def test_rewriter_canonicalises_nested_form():
    """A nested call should be flattened into multiple assignment lines."""
    source = (
        "def solve_test(I):\n"
        "    O = hconcat(I, vmirror(I))\n"
        "    return O\n"
    )
    rewriter = _make_rewriter()
    rewriter.parse(source)
    rewritten = rewriter.clean().rewrite()
    body_lines = [
        line.strip()
        for line in rewritten.splitlines()
        if line.strip() and not line.strip().startswith(("def ", "return "))
    ]
    # The body should contain at least two assignment lines (no nested calls).
    assert len(body_lines) >= 2
    # And the canonical form should still execute to the original semantics.
    ns = build_dsl_namespace()
    exec(rewritten, ns)
    out = ns["solver"](((1, 2), (3, 4)))
    # hconcat(I, vmirror(I)): rows of I concatenated with vmirror(I) row-wise
    assert out == ((1, 2, 2, 1), (3, 4, 4, 3))


def test_rewriter_handles_multi_step_solver():
    """A two-step solver: vmirror then hconcat with the original input."""
    source = (
        "def solve_test(I):\n"
        "    x1 = vmirror(I)\n"
        "    O = hconcat(I, x1)\n"
        "    return O\n"
    )
    rewriter = _make_rewriter()
    rewriter.parse(source)
    rewritten = rewriter.clean().rewrite()
    ns = build_dsl_namespace()
    exec(rewritten, ns)
    out = ns["solver"](((1, 2), (3, 4)))
    assert out == ((1, 2, 2, 1), (3, 4, 4, 3))
