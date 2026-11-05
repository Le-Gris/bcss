"""Unit tests for the ``src.utils.time_limit`` execution-timeout guard.

Both evaluators (``evaluate_synthesized_solver_programs`` and
``evaluate_synthesized_error_programs``) wrap each ``solver(...)`` call in this
context manager so a pathological synthesized program is recorded as a runtime
error instead of stalling the whole serial eval.
"""

import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import DEFAULT_PROGRAM_TIMEOUT_S, ProgramTimeout, time_limit


def test_slow_block_raises_program_timeout():
    """A block that runs past its budget is aborted with ProgramTimeout."""
    with pytest.raises(ProgramTimeout):
        with time_limit(0.2):
            while True:
                pass


def test_program_timeout_is_an_exception():
    """ProgramTimeout must subclass Exception so the evaluators' bare
    ``except Exception`` catches it and records a runtime error / mismatch."""
    assert issubclass(ProgramTimeout, Exception)


def test_fast_block_completes():
    """A block that finishes within budget returns its value untouched."""
    with time_limit(5):
        total = sum(range(1000))
    assert total == 499500


@pytest.mark.parametrize("disabled", [0, 0.0, -1, None])
def test_non_positive_timeout_disables_guard(disabled):
    """``seconds <= 0`` (or None) is an explicit opt-out: no alarm is armed."""
    with time_limit(disabled):
        pass
    # No itimer should be left running after an opt-out block.
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    assert remaining == 0


def test_handler_and_timer_restored_after_success():
    """On clean exit the SIGALRM handler is restored and the timer disarmed,
    so a later block does not inherit a stale alarm."""
    sentinel = signal.getsignal(signal.SIGALRM)
    with time_limit(5):
        pass
    assert signal.getsignal(signal.SIGALRM) == sentinel
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    assert remaining == 0


def test_handler_restored_after_timeout():
    """Even when the timeout fires, the previous handler is restored on exit."""
    sentinel = signal.getsignal(signal.SIGALRM)
    with pytest.raises(ProgramTimeout):
        with time_limit(0.2):
            while True:
                pass
    assert signal.getsignal(signal.SIGALRM) == sentinel
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    assert remaining == 0


def test_default_budget_is_positive():
    assert DEFAULT_PROGRAM_TIMEOUT_S > 0
