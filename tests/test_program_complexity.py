"""Tests for ``src.program_complexity.safe_node_count``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.program_complexity import safe_node_count

# A canonical-form program where x1 is referenced twice, so the executed-trace
# count (every reference expanded) strictly exceeds the DAG count (shared
# subtrees deduplicated by computation_id).
PROGRAM = (
    "def solver(I):\n"
    "    x1 = ofcolor(I, EIGHT)\n"
    "    x2 = fill(I, THREE, x1)\n"
    "    O = fill(x2, FOUR, x1)\n"
    "    return O"
)


def test_safe_node_count_returns_positive_int():
    count = safe_node_count(PROGRAM)
    assert isinstance(count, int)
    assert count > 0


def test_dedup_count_at_most_executed_trace_count():
    dag = safe_node_count(PROGRAM, dedup_subtrees=True)
    trace = safe_node_count(PROGRAM, dedup_subtrees=False)
    assert dag is not None and trace is not None
    assert dag <= trace


def test_none_and_garbage_return_none():
    assert safe_node_count(None) is None
    assert safe_node_count("") is None
    assert safe_node_count("not a ( valid program") is None
