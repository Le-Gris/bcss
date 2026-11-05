"""Tests for the stage-2 ``--control`` record synthesizer.

Covers :func:`src.pipeline.build_control_error_records` and the integration
with :func:`src.pipeline.build_prompt_records` — specifically that a control
record with ``descriptions=[]`` flows through prompt construction with
``response_id=None`` and gets identified by ``error_id``.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_control_error_records, build_prompt_records
from src.utils import record_id


def _write_task_json(dir_path: Path, task_id: str, output_grid):
    """Write a minimal ARC-format task JSON to *dir_path/<task_id>.json*."""
    task = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[0]], "output": output_grid}],
    }
    (dir_path / f"{task_id}.json").write_text(json.dumps(task))


def test_build_control_records_uses_test0_output(tmp_path):
    _write_task_json(tmp_path, "t1", [[1, 2], [3, 4]])
    _write_task_json(tmp_path, "t2", [[5, 6]])

    records = build_control_error_records(["t1", "t2"], tmp_path)
    assert len(records) == 2
    assert records[0] == {
        "task_id": "t1",
        "output_grid": [[1, 2], [3, 4]],
        "dataset": "control_correct_output",
        "source": "control",
        "attempt_count": 1,
        "descriptions": [],
    }
    assert records[1]["output_grid"] == [[5, 6]]


def test_build_control_records_skips_missing_task(tmp_path, capsys):
    _write_task_json(tmp_path, "present", [[7]])

    records = build_control_error_records(["present", "missing"], tmp_path)
    assert [r["task_id"] for r in records] == ["present"]
    assert "skipped 1 control tasks" in capsys.readouterr().out


def test_control_records_get_none_response_id_and_error_id(tmp_path):
    """A control record with descriptions=[] should produce a prompt record
    whose response_id is None and whose error_id is record_id(task, grid)."""
    grid = [[0, 1], [1, 0]]
    _write_task_json(tmp_path, "t1", grid)

    errors = build_control_error_records(["t1"], tmp_path)

    # Minimal template with all placeholders the constructor substitutes.
    template = (
        "DSL={{dsl_code}}\nPROG={{ref_program}}\nEX={{ref_examples}}\n"
        "DESC={{participant_description}}\nOUT={{participant_output}}\n"
        "STEPS={{num_modified_lines}}\n"
    )
    # Fake "solver" — only the key needs to exist; body is opaque to this test.
    solvers = {"t1": "def solve_t1(I):\n    x1 = identity(I)\n    O = x1\n    return O\n"}

    records = build_prompt_records(
        errors=errors,
        template=template,
        solvers=solvers,
        dsl_code="<dsl>",
        task_dir=tmp_path,
        num_steps=2,
        include_nl=False,
    )

    assert len(records) == 1
    assert records[0]["task_id"] == "t1"
    assert records[0]["response_id"] is None
    assert records[0]["error_id"] == record_id("t1", grid)
    assert records[0]["description"] is None
