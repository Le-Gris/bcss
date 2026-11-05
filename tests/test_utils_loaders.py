"""Tests for the ``src.utils`` input loaders."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_descriptions, load_jsonl_records


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_jsonl_records_concatenates_files_in_order(tmp_path):
    first = _write_jsonl(
        tmp_path / "a.jsonl",
        ['{"task_id": "t1", "n_mismatch": 0}', '{"task_id": "t2", "n_mismatch": 1}'],
    )
    second = _write_jsonl(tmp_path / "b.jsonl", ['{"task_id": "t3", "n_mismatch": 2}'])

    records = load_jsonl_records([first, second])

    assert [r["task_id"] for r in records] == ["t1", "t2", "t3"]
    assert records[2]["n_mismatch"] == 2


def test_load_jsonl_records_skips_blank_lines(tmp_path):
    path = _write_jsonl(
        tmp_path / "gaps.jsonl",
        ['{"task_id": "t1"}', "", "   ", '{"task_id": "t2"}'],
    )

    records = load_jsonl_records([path])

    assert len(records) == 2
    assert records[1]["task_id"] == "t2"


def test_load_jsonl_records_empty_input_list():
    assert load_jsonl_records([]) == []


# ---------------------------------------------------------------------------
# load_descriptions
# ---------------------------------------------------------------------------


def test_load_descriptions_concatenates_per_task(tmp_path):
    path = _write_jsonl(
        tmp_path / "desc.jsonl",
        [
            '{"task_id": "t1", "text": "flip the grid along its vertical axis"}',
            '{"task_id": "t2", "text": "copy the input grid twice side by side"}',
            '{"task_id": "t1", "text": "mirror every row of the input grid"}',
        ],
    )

    desc = load_descriptions(path, min_words=5)

    assert desc["t1"] == (
        "flip the grid along its vertical axis\n"
        "mirror every row of the input grid"
    )
    assert desc["t2"] == "copy the input grid twice side by side"


def test_load_descriptions_drops_short_texts(tmp_path):
    path = _write_jsonl(
        tmp_path / "desc.jsonl",
        [
            '{"task_id": "t1", "text": "too short"}',
            '{"task_id": "t2", "text": "this one is long enough to keep"}',
        ],
    )

    desc = load_descriptions(path, min_words=5)

    assert "t1" not in desc
    assert "t2" in desc


def test_load_descriptions_skips_blank_lines_and_missing_fields(tmp_path):
    path = _write_jsonl(
        tmp_path / "desc.jsonl",
        [
            "",
            '{"text": "a record with no task id at all"}',
            '{"task_id": "t1", "text": "   "}',
            '{"task_id": "t1", "text": "rotate the grid ninety degrees clockwise"}',
        ],
    )

    desc = load_descriptions(path, min_words=5)

    assert desc == {"t1": "rotate the grid ninety degrees clockwise"}


def test_load_descriptions_invalid_json_raises(tmp_path):
    path = _write_jsonl(
        tmp_path / "desc.jsonl",
        ['{"task_id": "t1", "text": "a perfectly valid first record"}', "{not json"],
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        load_descriptions(path, min_words=1)
