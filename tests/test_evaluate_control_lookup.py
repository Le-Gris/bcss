"""Pin the contract between control-mode synthesis and control-mode evaluation.

In a control run, ``build_prompt_records`` assigns each prompt
``response_id=None`` (since ``descriptions=[]``) and an ``error_id`` derived
from ``(task_id, output_grid)``.  The evaluator's
``build_control_error_lookup`` must key its lookup on the same
``(error_id, response_id)`` tuple, otherwise the result rows produced by synth
won't find their participant grid at eval time.

This test exercises both paths against the same fixture and asserts key
equality.
"""

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

from evaluate_synthesized_error_programs import build_control_error_lookup  # noqa: E402

from src.pipeline import build_control_error_records, build_prompt_records  # noqa: E402


def _write_task_json(dir_path: Path, task_id: str, output_grid):
    task = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[0]], "output": output_grid}],
    }
    (dir_path / f"{task_id}.json").write_text(json.dumps(task))


def test_control_lookup_keys_match_prompt_record_response_ids(tmp_path):
    _write_task_json(tmp_path, "t1", [[1, 2], [3, 4]])
    _write_task_json(tmp_path, "t2", [[5, 6]])
    task_ids = ["t1", "t2"]

    errors = build_control_error_records(task_ids, tmp_path)
    template = (
        "DSL={{dsl_code}}\nPROG={{ref_program}}\nEX={{ref_examples}}\n"
        "DESC={{participant_description}}\nOUT={{participant_output}}\n"
        "STEPS={{num_modified_lines}}\n"
    )
    solvers = {
        tid: f"def solve_{tid}(I):\n    x1 = identity(I)\n    O = x1\n    return O\n"
        for tid in task_ids
    }
    prompt_records = build_prompt_records(
        errors=errors,
        template=template,
        solvers=solvers,
        dsl_code="<dsl>",
        task_dir=tmp_path,
        num_steps=2,
        include_nl=False,
    )

    lookup = build_control_error_lookup(task_ids, tmp_path)

    prompt_keys = {(r["error_id"], r["response_id"]) for r in prompt_records}
    assert set(lookup.keys()) == prompt_keys
    # Every key has response_id=None in control mode (no descriptions).
    assert all(rid is None for _eid, rid in lookup.keys())

    for value in lookup.values():
        assert value["source"] == "control"
        assert value["dataset"] == "control_correct_output"
        assert value["attempt_count"] == 1
        assert value["description"] == ""
        assert isinstance(value["output_grid"], list)


def test_control_lookup_skips_missing_task(tmp_path):
    _write_task_json(tmp_path, "present", [[7, 8]])

    lookup = build_control_error_lookup(["present", "missing"], tmp_path)

    assert len(lookup) == 1
    (_eid, rid), value = next(iter(lookup.items()))
    assert rid is None
    assert value["task_id"] == "present"
    assert value["output_grid"] == [[7, 8]]
