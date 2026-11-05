"""Pin how Gemini responses are decoded into ``response_data``.

Gemini can split one JSON answer across several non-thought text parts (seen
with gemini-3.5-flash batch output: ``'{"modifications":'`` then ``' [...]}'``).
The parsers must join the parts before decoding, keep undecodable text instead
of dropping it, and stage 3 must not score such an error record as the
unedited ground truth.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

import evaluate_synthesized_error_programs as ev  # noqa: E402

from src.pipeline import _parse_async_response, _parse_batch_response  # noqa: E402

_MODEL = "gemini-3.5-flash"
_SPLIT = ['{\n  "modifications":', ' [\n    {"variable_name": "x1", "replacement": "rot', '180(I)"}\n  ]\n}']
_EXPECTED = {"modifications": [{"variable_name": "x1", "replacement": "rot180(I)"}]}


def _batch_entry(parts: list[dict]) -> dict:
    return {"key": "k", "response": {"candidates": [{"content": {"parts": parts}}]}}


def _async_result(parts: list[dict]) -> SimpleNamespace:
    ns_parts = [SimpleNamespace(text=p.get("text"), thought=p.get("thought")) for p in parts]
    content = SimpleNamespace(parts=ns_parts)
    text = "".join(p["text"] for p in parts if not p.get("thought")) or None
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)], text=text, usage_metadata=None
    )


def _parts(texts: list[str], thoughts: list[str] = ()) -> list[dict]:
    return [{"text": t, "thought": True} for t in thoughts] + [{"text": t} for t in texts]


def test_batch_joins_split_json_parts():
    parsed = _parse_batch_response(_batch_entry(_parts(_SPLIT, ["plan", "check"])), _MODEL)
    assert parsed["response_data"] == _EXPECTED
    assert parsed["thoughts"] == "plan\ncheck"


def test_batch_single_part_unchanged():
    parsed = _parse_batch_response(_batch_entry(_parts([json.dumps(_EXPECTED)])), _MODEL)
    assert parsed["response_data"] == _EXPECTED


def test_batch_keeps_raw_text_when_undecodable():
    parsed = _parse_batch_response(_batch_entry(_parts(['{"modifications": [', "oops"])), _MODEL)
    assert parsed["response_data"] == {"error": "Invalid JSON", "raw": '{"modifications": [oops'}


def test_batch_without_text_is_none():
    parsed = _parse_batch_response(_batch_entry(_parts([], ["only thinking"])), _MODEL)
    assert parsed["response_data"] is None
    assert parsed["thoughts"] == "only thinking"


def test_async_joins_split_json_parts():
    parsed = _parse_async_response(_async_result(_parts(_SPLIT, ["plan"])), _MODEL)
    assert parsed["response_data"] == _EXPECTED
    assert parsed["thoughts"] == "plan"


def test_async_keeps_raw_text_when_undecodable():
    parsed = _parse_async_response(_async_result(_parts(["not json"])), _MODEL)
    assert parsed["response_data"] == {"error": "Invalid JSON", "raw": "not json"}


@pytest.fixture
def ctx():
    rewriter = ev.ProgramRewriter(
        dsl_symbols_path=str(_root / "hodel_dsl" / "dsl.py"),
        dsl_constants_path=str(_root / "hodel_dsl" / "primitives.py"),
    )
    return rewriter, ev.build_dsl_namespace()


def test_stage3_rejects_error_record_instead_of_scoring_ground_truth(ctx, tmp_path):
    rewriter, dsl_namespace = ctx
    (tmp_path / "t1.json").write_text(
        json.dumps(
            {
                "train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[2]], "output": [[2]]}],
            }
        )
    )
    record = {
        "task_id": "t1",
        "error_id": "e-1",
        "response_id": None,
        "modifications": {"error": "Invalid JSON", "raw": '{"modifications": ['},
    }
    result = ev.evaluate_record(
        record=record,
        solvers={"t1": "def solver(I):\n    O = identity(I)\n    return O"},
        error_lookup={},
        task_dir=tmp_path,
        rewriter=rewriter,
        dsl_namespace=dsl_namespace,
    )
    assert result["parse_error"] == "Invalid JSON"
    assert result["is_syntactically_valid"] is False
    assert result["execution_success"] is False
