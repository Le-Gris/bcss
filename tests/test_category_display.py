"""Guards for the display-name rename of the ``attribute_substitution`` category.

The category is *displayed* as "Attribute Misselection" while its internal slug
stays ``attribute_substitution`` (frozen: it is the response-schema enum value,
the ``count_<slug>`` tally column, and the value stored in every shipped result
artifact).  These tests pin both sides of that boundary.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.category_definitions import (
    CATEGORIES,
    CATEGORY_NAMES,
    DISPLAY_LABEL_OVERRIDES,
)


# slug → display name for the "…Substitution" → "…Misselection" relabels.
_RENAMED = {
    "attribute_substitution": "Attribute Misselection",
    "transformation_substitution": "Transformation Misselection",
}


def test_slug_is_frozen():
    # The internal identifiers must remain unchanged for schema/result compat.
    for slug in _RENAMED:
        assert slug in CATEGORY_NAMES
        assert CATEGORIES[slug].name == slug


def test_display_name_renamed():
    for slug, display in _RENAMED.items():
        assert CATEGORIES[slug].display_name == display
        assert DISPLAY_LABEL_OVERRIDES[slug] == display


def test_full_section_prompt_text_stays_frozen():
    # full_section is a verbatim copy of the (frozen) classifier prompt.
    assert "## Attribute Substitution" in CATEGORIES["attribute_substitution"].full_section
    assert (
        "## Transformation Substitution"
        in CATEGORIES["transformation_substitution"].full_section
    )


def test_label_helpers_honour_override():
    from analyses.analyze_program_length_by_difficulty import _format_category_label

    # Overridden categories → display name; others → plain title-case fallback.
    for slug, display in _RENAMED.items():
        assert _format_category_label(slug) == display
    assert _format_category_label("scope_restriction") == "Scope Restriction"


def test_classifier_schema_enum_stays_frozen():
    # The classifier's response-schema enum is what the model actually sees. It
    # must keep the original "…_substitution" slugs (the pipeline ran with them)
    # and must never pick up a display "…Misselection" string, regardless of the
    # display rename. Guards reproducibility of the shipped classification run.
    classify_errors = pytest.importorskip("scripts.classify_errors")
    enum = classify_errors.get_characterization_response_schema()[
        "properties"
    ]["error_category"]["enum"]

    assert set(CATEGORY_NAMES) | {"unclassifiable"} == set(enum)
    assert "attribute_substitution" in enum
    assert "transformation_substitution" in enum
    assert not any("missel" in e.lower() for e in enum)
