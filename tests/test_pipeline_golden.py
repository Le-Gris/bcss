"""Golden-output tests for error synthesis, error classification and the rule-type analysis.

These tests pin the inputs that ``classify_errors.py`` and
``analyze_rule_type_distribution.py`` produce on a small self-contained
fixture, so that a refactor can be shown to preserve behaviour byte for byte:

* **Classification** — drives ``classify_errors.main()`` end to end in
  ``--mode prepare`` with the mode runner stubbed out, and hashes exactly what
  would be sent to the API (the expanded per-sample request items), the output
  stem, and the response schema.  Covers full-source and ``--api-only`` DSL,
  multi-sample expansion, the eval-split ``original_program`` fallback, an
  execution failure, and the skip paths (analytical / missing grid).
* **Error synthesis** — drives ``synthesize_error_programs.main()`` in
  ``--mode prepare`` with the runner stubbed, and hashes the request items,
  stem, response schema and batch config.  Covers the default and
  ``--include-nl`` prompts, multi-sample expansion, ``--control``, and
  ``--solvers-file`` with ``--api-only``.
* **Rule-type analysis** — hashes everything the NUTS fit consumes:
  ``load_classified`` → (``restrict_to_partial``) → ``build_index_maps`` +
  ``build_attempt_counts``, for both the rule-type and order-effect analyses.
  The fit itself is deterministic given these inputs and a seed, and too slow
  for the suite.

If one of these hashes changes after an edit, the edit changed results — do
not just re-freeze.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from src.categorical_model import build_index_maps  # noqa: E402


def _load_by_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _root / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ce = _load_by_path("classify_errors_golden", "scripts/classify_errors.py")
artd = _load_by_path(
    "analyze_rule_type_distribution_golden",
    "analyses/analyze_rule_type_distribution.py",
)


def _sha(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fixture: two tiny tasks + a stage-4-shaped candidates file
# ---------------------------------------------------------------------------

# 67a3c6ac has a hand-written GT solver (vmirror).  "evalfix01" has none, so
# classification must fall back to the record's ``original_program``.
_TASKS = {
    "67a3c6ac": {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
            {"input": [[5, 0, 6], [0, 7, 0]], "output": [[6, 0, 5], [0, 7, 0]]},
        ],
        "test": [{"input": [[8, 9], [1, 0]], "output": [[9, 8], [0, 1]]}],
    },
    "evalfix01": {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            {"input": [[0, 5], [6, 0]], "output": [[6, 0], [0, 5]]},
        ],
        "test": [{"input": [[7, 8], [9, 1]], "output": [[9, 7], [1, 8]]}],
    },
}


def _prog(name: str, body: str) -> str:
    return f"def solve_{name}(I):\n    {body}\n    return O"


_CANDIDATES = [
    # Training task, two candidate programs for one error.
    {
        "task_id": "67a3c6ac",
        "error_id": "e_train_1",
        "candidate_type": "model",
        "rewritten_program": _prog("67a3c6ac", "O = hmirror(I)"),
        "target_error_grid": [[0, 1], [9, 8]],
        "program_output_grid": [[1, 0], [9, 8]],
        "attempt_count": 3,
    },
    {
        "task_id": "67a3c6ac",
        "error_id": "e_train_1",
        "candidate_type": "model",
        "rewritten_program": _prog("67a3c6ac", "O = rot180(I)"),
        "target_error_grid": [[0, 1], [9, 8]],
        "program_output_grid": None,  # → falls back to target_error_grid
        "attempt_count": 3,
    },
    # Program that fails at execution → "[EXECUTION FAILED]" in the comparison.
    {
        "task_id": "67a3c6ac",
        "error_id": "e_train_2",
        "candidate_type": "model",
        "rewritten_program": _prog("67a3c6ac", "O = not_a_dsl_function(I)"),
        "target_error_grid": [[1, 1], [1, 1]],
        "program_output_grid": [[1, 1], [1, 1]],
        "attempt_count": 1,
    },
    # Eval-split task: no GT solver, reference comes from original_program.
    {
        "task_id": "evalfix01",
        "error_id": "e_eval_1",
        "candidate_type": "model",
        "original_program": _prog("evalfix01", "O = rot90(I)"),
        "rewritten_program": _prog("evalfix01", "O = rot270(I)"),
        "target_error_grid": [[8, 1], [7, 9]],
        "program_output_grid": [[8, 1], [7, 9]],
        "attempt_count": 2,
    },
    # Skip paths.
    {
        "task_id": "67a3c6ac",
        "error_id": "e_analytical",
        "candidate_type": "analytical",
        "rewritten_program": _prog("67a3c6ac", "O = vmirror(I)"),
        "target_error_grid": [[9, 8], [0, 1]],
    },
    {
        "task_id": "missingtask",
        "error_id": "e_missing",
        "candidate_type": "model",
        "original_program": _prog("missingtask", "O = I"),
        "rewritten_program": _prog("missingtask", "O = hmirror(I)"),
        "target_error_grid": [[1]],
    },
]


@pytest.fixture
def classify_fixture(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    for tid, data in _TASKS.items():
        (task_dir / f"{tid}.json").write_text(json.dumps(data))
    cand = tmp_path / "candidates_fixture.jsonl"
    cand.write_text("".join(json.dumps(r) + "\n" for r in _CANDIDATES))
    return tmp_path, task_dir, cand


def _run_classify_prepare(monkeypatch, tmp_path, task_dir, cand, extra_args):
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ce, "run_prepare_submit_mode", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_errors.py",
            "--mode", "prepare",
            "--model", "gemini-fixture-model",
            "--input-file", str(cand),
            "--task-dir", str(task_dir),
            "--output-dir", str(tmp_path / "out"),
            "--template-file", str(_root / "prompts" / "classify_errors.md"),
            *extra_args,
        ],
    )
    ce.main()
    assert captured, "run_prepare_submit_mode was never reached"
    return captured


# Frozen hashes; see the module docstring before changing them.
CLASSIFY_GOLDEN = {
    "full_n1": "e898f11b63d0d77afe5391f69185a04c3682e38464b8469b0df087751c840a20",
    "api_only_n2": "25f0d2561caa45cdbf5be042d7dfdb2924cd46121333eabab42d3b47293733ee",
}


@pytest.mark.parametrize(
    "case,extra_args",
    [
        ("full_n1", []),
        ("api_only_n2", ["--api-only", "--num-samples", "2"]),
    ],
)
def test_classify_prompts_golden(monkeypatch, classify_fixture, case, extra_args):
    tmp_path, task_dir, cand = classify_fixture
    cap = _run_classify_prepare(monkeypatch, tmp_path, task_dir, cand, extra_args)

    # Sanity on the fixture itself, so a vacuous hash can't pass.
    expanded = cap["expanded"]
    n_samples = 2 if "--num-samples" in extra_args else 1
    assert len(expanded) == 4 * n_samples  # 2 skipped of 6
    prompts = [e["prompt"] for e in expanded]
    assert any("[EXECUTION FAILED]" in p for p in prompts)
    assert any("rot90" in p for p in prompts)  # eval fallback reference

    digest = _sha(
        {
            "expanded": expanded,
            "stem": cap["stem"],
            "response_schema": cap["response_schema"],
            "metadata_keys": cap["metadata_keys"],
        }
    )
    print(f"\n[golden] classify {case}: {digest}")
    assert digest == CLASSIFY_GOLDEN[case]


# ---------------------------------------------------------------------------
# Rule-type analysis inputs
# ---------------------------------------------------------------------------

# Mostly 3 train pairs; e3/p1 uses 4, since "contiguous" needs a middle run of
# >= 2 (a lone middle match is "other").  Every rule type and tier is reached.
_RULE_ROWS = [
    # e1: alternative + partial(first-k)
    ("e1", "p1", [True, True, True], 4),
    ("e1", "p2", [True, False, False], 4),
    # e2: neither + partial(last-k)
    ("e2", "p1", [False, False, False], 1),
    ("e2", "p2", [False, False, True], 1),
    # e3: partial(contiguous middle) + partial(other)
    ("e3", "p1", [False, True, True, False], 2),
    ("e3", "p2", [True, False, True], 2),
    # e4: empty training_matches → unclassifiable, dropped
    ("e4", "p1", [], 5),
    # e5: alternative only
    ("e5", "p1", [True, True, True], 7),
]


@pytest.fixture
def rule_fixture(tmp_path):
    p = tmp_path / "candidates_rule_fixture.jsonl"
    with p.open("w") as f:
        for eid, pid, tm, ac in _RULE_ROWS:
            f.write(
                json.dumps(
                    {
                        "task_id": f"t_{eid}",
                        "error_id": eid,
                        "program_id": pid,
                        "training_matches": tm,
                        "attempt_count": ac,
                        "rewritten_program": "def solve(I):\n    O = I\n    return O",
                    }
                )
                + "\n"
            )
    return p


def _index_payload(df, category_col):
    idx = build_index_maps(df, category_col=category_col, three_level=False)
    att = artd.build_attempt_counts(df, idx["error_ids"])
    return {
        "rows": df.sort(["error_id", "program_id"]).to_dicts(),
        "categories": idx["categories"],
        "error_ids": idx["error_ids"],
        "y": np.asarray(idx["y"]).tolist(),
        "error_idx": np.asarray(idx["error_idx"]).tolist(),
        "n_errors": idx["n_errors"],
        "n_obs": idx["n_obs"],
        "K": idx["K"],
        "attempt_counts": att.tolist(),
    }


RULE_GOLDEN = {
    "rule_type": "ed10940f438b09bfa0e2fcebddb90b951d3fbf08f47e47a4a2abbf73a3c38489",
    "order_effects": "cc895bac81128b219d9ef25b23a3a876fc71f8a71f39b8a93c4054093ee18527",
}


def test_rule_type_inputs_golden(rule_fixture):
    df = artd.load_classified(rule_fixture)
    assert df.height == 7  # e4 dropped
    payload = _index_payload(df, "program_rule_type")
    assert payload["K"] == 3
    digest = _sha(payload)
    print(f"\n[golden] rule_type: {digest}")
    assert digest == RULE_GOLDEN["rule_type"]


def test_order_effects_inputs_golden(rule_fixture):
    df = artd.restrict_to_partial(artd.load_classified(rule_fixture))
    payload = _index_payload(df, "program_order_tier")
    assert payload["K"] == 4
    digest = _sha(payload)
    print(f"\n[golden] order_effects: {digest}")
    assert digest == RULE_GOLDEN["order_effects"]


# ---------------------------------------------------------------------------
# Error-program synthesis prompts
# ---------------------------------------------------------------------------

se = _load_by_path(
    "synthesize_error_programs_golden", "scripts/synthesize_error_programs.py"
)

# 67a3c6ac uses the hand-written Hodel solver; evalfix01 only has a solver
# through --solvers-file.  e_nl has two descriptions, e_plain has none.
_SYNTH_ERRORS = [
    {
        "task_id": "67a3c6ac",
        "error_id": "e_nl",
        "dataset": "training",
        "source": "human",
        "output_grid": [[0, 1], [9, 8]],
        "attempt_count": 3,
        "descriptions": [
            {"response_id": "r1", "text": "I flipped it upside down."},
            {"response_id": "r2", "text": "Mirror, I think."},
        ],
    },
    {
        "task_id": "67a3c6ac",
        "error_id": "e_plain",
        "dataset": "training",
        "source": "human",
        "output_grid": [[1, 1], [1, 1]],
        "attempt_count": 1,
        "descriptions": [],
    },
    {
        "task_id": "evalfix01",
        "error_id": "e_eval",
        "dataset": "evaluation",
        "source": "human",
        "output_grid": [[8, 1], [7, 9]],
        "attempt_count": 2,
        "descriptions": [],
    },
]

_SYNTH_SOLVERS_FILE = {"evalfix01": [_prog("evalfix01", "O = rot90(I)")]}

@pytest.fixture
def synth_fixture(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    for tid, data in _TASKS.items():
        (task_dir / f"{tid}.json").write_text(json.dumps(data))
    errors = tmp_path / "errors_fixture.filtered.jsonl"
    errors.write_text("".join(json.dumps(r) + "\n" for r in _SYNTH_ERRORS))
    solvers = tmp_path / "solvers_fixture.json"
    solvers.write_text(json.dumps(_SYNTH_SOLVERS_FILE))
    return tmp_path, task_dir, errors, solvers


def _run_synth_prepare(monkeypatch, tmp_path, task_dir, errors, extra_args):
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(se, "run_prepare_submit_mode", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthesize_error_programs.py",
            "--mode", "prepare",
            "--model", "gemini-fixture-model",
            "--input-file", str(errors),
            "--task-dir", str(task_dir),
            "--output-dir", str(tmp_path / "out"),
            *extra_args,
        ],
    )
    se.main()
    assert captured, "run_prepare_submit_mode was never reached"
    return captured


# Frozen hashes; see the module docstring before changing them.
SYNTH_GOLDEN = {
    "default": "b4ed73046134331427ba9afb7b234c66b8181f4224f23f6d92dd5613ff2433da",
    "include_nl_n2": "f45e608e5cc509a72a8e8d9e67de0b3e43fdb85c83e0b4c365a3e869a75c459c",
    "control": "c586c98b62731aeeea50fb9c5f5175bdad63ab1f75d55371d0f1bc5d4bd42404",
    "solvers_file_api_steps3": "820ca3794fd1457d1ba9f82858bee8fa0fcacd49b745f9e5774216c0eac8fd57",
}


@pytest.mark.parametrize(
    "case,extra_args,n_expected",
    [
        # e_eval skipped (no Hodel solver for evalfix01).
        ("default", [], 2),
        # e_nl x 2 descriptions + e_plain with no NL, x 2 samples.
        ("include_nl_n2", ["--include-nl", "--num-samples", "2"], 6),
        # One correct-output record per test pair of 67a3c6ac.
        ("control", ["--control"], 1),
        # --solvers-file replaces Hodel: only evalfix01 has a solver.
        ("solvers_file_api_steps3", ["--solvers-file", "{solvers}", "--api-only",
                                     "--num-steps", "3"], 1),
    ],
)
def test_synthesis_prompts_golden(
    monkeypatch, synth_fixture, case, extra_args, n_expected
):
    tmp_path, task_dir, errors, solvers = synth_fixture
    args = [a.replace("{solvers}", str(solvers)) for a in extra_args]
    cap = _run_synth_prepare(monkeypatch, tmp_path, task_dir, errors, args)

    expanded = cap["expanded"]
    assert len(expanded) == n_expected

    digest = _sha(
        {
            "expanded": expanded,
            "stem": cap["stem"],
            "response_schema": cap["response_schema"],
            "batch_config": cap["batch_config"],
        }
    )
    print(f"\n[golden] synthesis {case}: {digest}")
    assert digest == SYNTH_GOLDEN[case]


# ---------------------------------------------------------------------------
# Edit evaluation (stage 3)
# ---------------------------------------------------------------------------

ev = _load_by_path(
    "evaluate_synthesized_error_programs_golden",
    "scripts/evaluate_synthesized_error_programs.py",
)

_EVAL_TASK_ID = "67a3c6ac"
_EVAL_SOLVER = (
    "def solver(I):\n"
    "    x1 = vmirror(I)\n"
    "    x2 = identity(x1)\n"
    "    O = identity(x2)\n"
    "    return O"
)


def _mods(*mods, wrap=False):
    return {"modifications": list(mods)} if wrap else list(mods)


# Every case is a response the response schema can produce, plus the
# malformed payloads the response parser can hand over.
_EVAL_CASES = {
    "modify_with_action": _mods(
        {"action": "modify", "variable_name": "x1", "replacement": "hmirror(I)"}
    ),
    "modify_no_action": _mods({"variable_name": "x1", "replacement": "rot180(I)"}),
    "wrapped_dict": _mods(
        {"variable_name": "x2", "replacement": "rot90(x1)"}, wrap=True
    ),
    "duplicate_last_wins": _mods(
        {"variable_name": "x1", "replacement": "hmirror(I)"},
        {"variable_name": "x1", "replacement": "rot270(I)"},
    ),
    "nested_call": _mods({"variable_name": "x1", "replacement": "hmirror(rot90(I))"}),
    "identity_replacement": _mods(
        {"variable_name": "x1", "replacement": "identity(I)"},
        {"variable_name": "O", "replacement": "identity(x1)"},
    ),
    "missing_replacement": _mods({"variable_name": "x1"}),
    "unknown_variable": _mods({"variable_name": "x9", "replacement": "hmirror(I)"}),
    "runtime_error": _mods({"variable_name": "x1", "replacement": "crop(I, ORIGIN, ORIGIN)"}),
    "output_not_grid": _mods({"variable_name": "O", "replacement": "height(x2)"}),
    "missing_variable_name": _mods({"replacement": "hmirror(I)"}),
    "parser_error_payload": {"error": "Invalid JSON", "raw": "{not json"},
    "not_a_list": {"modifications": "x1 = hmirror(I)"},
    "empty_list": [],
}


@pytest.fixture
def eval_fixture(tmp_path):
    (tmp_path / f"{_EVAL_TASK_ID}.json").write_text(json.dumps(_TASKS[_EVAL_TASK_ID]))
    rewriter = ev.ProgramRewriter(
        dsl_symbols_path=str(_root / "hodel_dsl" / "dsl.py"),
        dsl_constants_path=str(_root / "hodel_dsl" / "primitives.py"),
    )
    lookup = {
        ("e_eval", None): {
            "output_grid": [[0, 1], [9, 8]],
            "description": "",
            "attempt_count": 2,
            "dataset": "training",
            "source": "human",
        }
    }
    return tmp_path, rewriter, ev.build_dsl_namespace(), lookup


EVAL_GOLDEN = "2eb4a21c9befceaa778441da0c6967f93da4ad4b1bc357a0e42d9c42a908d635"


def test_evaluate_record_golden(eval_fixture):
    task_dir, rewriter, ns, lookup = eval_fixture
    results = {}
    for name, mods in _EVAL_CASES.items():
        record = {
            "task_id": _EVAL_TASK_ID,
            "error_id": "e_eval",
            "response_id": None,
            "modifications": mods,
        }
        results[name] = ev.evaluate_record(
            record, {_EVAL_TASK_ID: _EVAL_SOLVER}, lookup, task_dir, rewriter, ns
        )

    # Sanity on the fixture, so a vacuous hash can't pass.
    assert results["modify_with_action"]["execution_success"]
    assert results["duplicate_last_wins"]["modifications"][-1]["replacement"] == "rot270(I)"
    assert results["nested_call"]["has_nested_violation"]
    assert results["identity_replacement"]["identity_count"] == 2
    assert results["parser_error_payload"]["parse_error"] == "Invalid JSON"

    digest = _sha(results)
    print(f"\n[golden] evaluate_record: {digest}")
    assert digest == EVAL_GOLDEN


@pytest.mark.parametrize("action", ["delete", "add"])
def test_evaluate_record_rejects_non_modify_actions(eval_fixture, action):
    task_dir, rewriter, ns, lookup = eval_fixture
    record = {
        "task_id": _EVAL_TASK_ID,
        "error_id": "e_eval",
        "response_id": None,
        "modifications": [{"action": action, "variable_name": "x1", "replacement": "hmirror(I)"}],
    }
    result = ev.evaluate_record(
        record, {_EVAL_TASK_ID: _EVAL_SOLVER}, lookup, task_dir, rewriter, ns
    )
    assert result["parse_error"] == f"Unsupported edit action(s): ['{action}']"
    assert not result["is_syntactically_valid"]
