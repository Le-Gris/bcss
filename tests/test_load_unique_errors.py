"""Tests for ``src.utils.load_unique_errors``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_unique_errors

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "errors_sample.jsonl"


def test_load_unique_errors_streams_all_records():
    records = list(load_unique_errors(_FIXTURE))
    assert len(records) == 3


def test_load_unique_errors_preserves_required_fields():
    records = list(load_unique_errors(_FIXTURE))
    first = records[0]
    for key in ("task_id", "dataset", "source", "output_grid", "attempt_count", "descriptions"):
        assert key in first, f"required field '{key}' missing from loaded record"
    assert first["task_id"] == "00000001"
    assert first["attempt_count"] == 3
    assert len(first["descriptions"]) == 2
    assert first["descriptions"][0]["response_id"] == "r-a"


def test_load_unique_errors_preserves_optional_metadata():
    records = list(load_unique_errors(_FIXTURE))
    third = records[2]
    assert third.get("metadata") == {"note": "carried through unchanged"}


def test_load_unique_errors_handles_empty_descriptions():
    records = list(load_unique_errors(_FIXTURE))
    second = records[1]
    assert second["descriptions"] == []
