"""Unit tests for the shared rule-type classifier in ``src/rule_types.py``.

The analysis and the HTML report (``scripts/visualize_error_pages.py``) both
import this taxonomy; these tests pin it directly so they share one verified
implementation.
"""

from src.rule_types import (
    ORDER_TIER_ORDER,
    RULE_TYPE_DISPLAY,
    RULE_TYPE_ORDER,
    rule_type,
)


def test_alternative_all_match():
    assert rule_type([True, True, True]) == ("alternative", None)


def test_neither_no_match():
    assert rule_type([False, False, False]) == ("neither", None)


def test_unknown_empty_or_none():
    assert rule_type([]) == ("unknown", None)
    assert rule_type(None) == ("unknown", None)


def test_partial_first_k():
    # Prefix run touching pair 0 (any length, incl. a lone first pair).
    assert rule_type([True, False, False]) == ("partial", "first-k")
    assert rule_type([True, True, False, False]) == ("partial", "first-k")


def test_partial_last_k():
    # Suffix run touching the last pair (any length, incl. a lone last pair).
    assert rule_type([False, False, True]) == ("partial", "last-k")
    assert rule_type([False, False, True, True]) == ("partial", "last-k")


def test_partial_contiguous_interior_run():
    # Interior consecutive run of length >= 2 -> "contiguous".
    assert rule_type([False, True, True, False]) == ("partial", "contiguous")
    assert rule_type([False, True, True, True, False]) == ("partial", "contiguous")


def test_partial_other_single_interior():
    # A lone interior match (length 1, not first/last) -> "other"
    # (this moved out of "contiguous" under the recoding).
    assert rule_type([False, True, False]) == ("partial", "other")
    assert rule_type([False, False, True, False, False]) == ("partial", "other")


def test_partial_other_scattered():
    # Non-contiguous (gapped) matches fold into "other".
    assert rule_type([True, False, True]) == ("partial", "other")
    assert rule_type([True, False, True, False]) == ("partial", "other")
    assert rule_type([False, True, False, True, False]) == ("partial", "other")


def test_coerces_truthy_values():
    assert rule_type([1, 1, 1]) == ("alternative", None)
    assert rule_type([1, 0, 0]) == ("partial", "first-k")


def test_display_map_covers_classifiable_labels():
    # Every label rule_type can emit (except "unknown") must have a display name.
    for label in ("alternative", "partial", "neither"):
        assert label in RULE_TYPE_DISPLAY
    assert set(RULE_TYPE_DISPLAY.values()) == set(RULE_TYPE_ORDER)


def test_order_tier_order_matches_emitted_tags():
    # The plot order must list exactly the order tags rule_type produces.
    emitted = set()
    for mask in (
        [True, False, False],         # first-k
        [False, False, True],         # last-k
        [False, True, True, False],   # contiguous (interior run >= 2)
        [False, True, False],         # other (single interior)
        [True, False, True],          # other (scattered)
    ):
        _, tag = rule_type(mask)
        emitted.add(tag)
    assert emitted == set(ORDER_TIER_ORDER)
