"""The error-category taxonomy: the seven categories and their definitions.

Each category has a display name, a one-line summary, the full definition used
in the classifier prompt (``prompts/classify_errors.md``), and the
category-specific fields of the classifier's structured response.
``CATEGORY_NAMES`` lists the seven categories; ``unclassifiable`` is not among
them because it is the label for errors that fit none, and is added only to the
classifier's response options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryDefinition:
    """Immutable descriptor for a single error category."""

    name: str
    """Internal key, e.g. ``"attribute_substitution"``."""

    display_name: str
    """Human-facing name for figures/tables/docs, e.g. ``"Attribute Misselection"``.

    Distinct from :attr:`name` (the frozen internal slug) and from
    :attr:`full_section` (the verbatim, frozen prompt text): a category may be
    *displayed* under a different name than the slug/prompt use.
    """

    one_line_summary: str
    """Contrastive one-liner distinguishing this category from the others."""

    full_section: str
    """Full definition block (definition + key diagnostic + distinguishing criteria).

    Extracted verbatim from ``prompts/classify_errors.md``.
    """

    specific_fields: List[str]
    """Names of the category-specific response fields (nullable on a non-match)."""

    schema_properties: Dict[str, Dict[str, Any]] = field(repr=False)
    """Gemini-format JSON schema properties for the category-specific fields."""


# ---------------------------------------------------------------------------
# Per-category definitions
# ---------------------------------------------------------------------------

# ── Attribute Misselection ────────────────────────────────────────────────
# Displayed as "Attribute Misselection" (the cognizer latched onto an easier /
# more salient feature, not necessarily knowing the correct one).  The internal
# slug ``attribute_substitution`` and the ``full_section`` prompt text are frozen
# by the completed classification run and MUST NOT change.

_ATTRIBUTE_SUBSTITUTION = CategoryDefinition(
    name="attribute_substitution",
    display_name="Attribute Misselection",
    one_line_summary=(
        "Selects a simpler or more perceptually salient attribute or feature "
        "(e.g., color, dimension, direction) in place of the one that must be "
        "computed; the operation itself is correct."
    ),
    full_section="""\
## Attribute Substitution

**Definition:** The participant replaces a target quantity, property, or feature that must be computed or inferred with a different one. Often, this is a simpler strategy: the participant may anchor to a specific training example's attribute (e.g., hardcoding a color or dimension) rather than computing the correct general rule.

**Key diagnostic:** The error lies in the *value, parameter, or feature* used by an operation, rather than the operation itself. The substituted attribute is often a hardcoded constant, a perceptually salient feature, or one that requires fewer inferential steps. These substituted attributes can often be easily inferred from a subset (or more rarely, all) of the training examples.

**Distinguishing criteria:** Unlike operator confusion, the error is in the *attribute* or *input value* (e.g., color, size, grid dimensions) rather than the transformation applied. Unlike scope restriction, the participant applies a different computation rather than applying the correct one incompletely.

**Sub-check for boundary precision:** If the difference between the correct and substituted attribute is only ±1, check whether the training examples discriminate between the two values. If no (the off-by-one value also fits the training data), classify as attribute substitution (hardcoded constant), not boundary precision.""",
    specific_fields=[
        "target_parameter",
        "correct_attribute",
        "substituted_attribute",
        "substitution_type",
        "hardcoded_value",
    ],
    schema_properties={
        "target_parameter": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The quantity or parameter in the transformation that "
                "the correct and substituted attributes determine "
                "(e.g., 'width of the drawn patterns')."
            ),
        },
        "correct_attribute": {
            "type": "STRING",
            "nullable": True,
            "description": "The attribute the GT program computes.",
        },
        "substituted_attribute": {
            "type": "STRING",
            "nullable": True,
            "description": "The attribute the error program uses instead.",
        },
        "substitution_type": {
            "type": "STRING",
            "nullable": True,
            "enum": [
                "computed_property",
                "hardcoded_constant",
                "surface_correlation",
            ],
            "description": "Type of attribute substitution.",
        },
        "hardcoded_value": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The literal value substituted (only when "
                "substitution_type = hardcoded_constant)."
            ),
        },
    },
)

# ── Scope Restriction ─────────────────────────────────────────────────────

_SCOPE_RESTRICTION = CategoryDefinition(
    name="scope_restriction",
    display_name="Scope Restriction",
    one_line_summary=(
        "Applies the correct operation but only to a subset of the "
        "targets, regions, or conditions where it should apply."
    ),
    full_section="""\
## Scope Restriction

**Definition:** The participant infers the correct computation and applies it correctly, but only to a subset of the entities, regions, or conditions where it should apply.

**Key diagnostic:** The output is consistent with the ground-truth rule within the restricted scope. The error is not in what is computed but in where or how broadly it is applied.

**Distinguishing criteria:** Unlike attribute substitution, the core computation itself is correct — the error is purely in the extent of its application. Unlike operator confusion, the participant has identified the right high-level operation and applies it faithfully where they do apply it.""",
    specific_fields=[
        "restricted_operation",
        "correct_scope",
        "actual_scope",
        "fraction_covered",
    ],
    schema_properties={
        "restricted_operation": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The transformation that is correctly applied but "
                "to a limited scope, in perceptual terms "
                "(e.g., 'color fill', 'shape copying')."
            ),
        },
        "correct_scope": {
            "type": "STRING",
            "nullable": True,
            "description": "What the GT program applies the operation to.",
        },
        "actual_scope": {
            "type": "STRING",
            "nullable": True,
            "description": "What the error program applies the operation to.",
        },
        "fraction_covered": {
            "type": "NUMBER",
            "nullable": True,
            "description": (
                "Approximate fraction of targets covered (0.0–1.0)."
            ),
        },
    },
)

# ── Operator Confusion ────────────────────────────────────────────────────

_OPERATOR_CONFUSION = CategoryDefinition(
    name="operator_confusion",
    display_name="Operator Confusion",
    one_line_summary=(
        "Selects the wrong member of the correct operation class "
        "(e.g., rotate left instead of right, mirror horizontally instead of vertically)."
    ),
    full_section="""\
## Operator Confusion

**Definition:** The participant correctly identifies the class of operation needed but selects the wrong member of that class (e.g., rotating left instead of right, mirroring horizontally instead of vertically). Note that this "operation" may be a composite subprogram, and does not have to correspond to a single DSL function.

**Key diagnostic:** The error reflects a spatial or perceptual discrimination failure between similar, alternative transformations within the same family.

**Distinguishing criteria:** Unlike attribute substitution, the error is in the *transformation itself* (the operation) rather than an input attribute or parameter (like a color or size). Unlike scope restriction, the operation is applied to all the right targets — the error is in the choice of operation.""",
    specific_fields=[
        "operator_target",
        "operation_class",
        "correct_operator",
        "confused_operator",
        "confusion_axis",
    ],
    schema_properties={
        "operator_target": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "What aspect of the transformation the operator "
                "determines, in perceptual terms (e.g., "
                "'orientation of the copied shape')."
            ),
        },
        "operation_class": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The class of operation (e.g., rotation, mirroring)."
            ),
        },
        "correct_operator": {
            "type": "STRING",
            "nullable": True,
            "description": "The GT operator.",
        },
        "confused_operator": {
            "type": "STRING",
            "nullable": True,
            "description": "The operator the error program uses.",
        },
        "confusion_axis": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "What dimension the confusion is along "
                "(e.g., CW vs CCW, horizontal vs vertical)."
            ),
        },
    },
)

# ── Transformation Misselection ───────────────────────────────────────────
# Displayed as "Transformation Misselection" (the cognizer applied a
# transformation from the wrong computational family, not knowingly substituting
# a known-correct one).  The internal slug ``transformation_substitution`` and
# the ``full_section`` prompt text are frozen by the completed run and MUST NOT
# change.

_TRANSFORMATION_SUBSTITUTION = CategoryDefinition(
    name="transformation_substitution",
    display_name="Transformation Misselection",
    one_line_summary=(
        "Applies a transformation from an entirely different computational "
        "family than the one required (e.g., recoloring where mirroring was "
        "needed), while correctly identifying the target objects."
    ),
    full_section="""\
## Transformation Substitution

**Definition:** The participant correctly identifies the target objects, regions, or spatial structure of the task, but applies a transformation from an entirely different computational family than the one required — e.g., recoloring where mirroring was needed, spatial shifting where concentric expansion was needed, cropping where tiling was needed. The scaffolding (object identification, grid decomposition) is often correct; the operation itself is from the wrong conceptual domain.

**Key diagnostic:** The participant's program operates on the right targets but performs a categorically different kind of computation. The correct and applied operations are not alternative members of the same class (that would be operator confusion) — they belong to different transformation families altogether.

**Distinguishing criteria:** Unlike **operator confusion**, the two operations are not from the same class (e.g., mirror vs. rotate are both spatial transformations but from different families; CW vs. CCW rotation are same-family alternatives). Unlike **attribute substitution**, the error is in the *operation applied*, not in the *input feature* used to parameterize it. Unlike **partial program**, the participant is not missing steps — they are performing *different* steps. Unlike **unclassifiable**, the participant has engaged with the task structure (correct targets, correct spatial layout) and the applied operation follows a coherent, non-ad-hoc logic.""",
    specific_fields=[
        "correct_operation_domain",
        "applied_operation_domain",
        "shared_structure",
        "domain_distance",
    ],
    schema_properties={
        "correct_operation_domain": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The family of transformation the GT program uses "
                "(e.g., 'rotation-based tiling', 'concentric expansion')."
            ),
        },
        "applied_operation_domain": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The family of transformation the error program uses "
                "instead (e.g., 'mirror-based tiling', 'spatial shift')."
            ),
        },
        "shared_structure": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "What aspects of the task structure the participant "
                "correctly identified (e.g., 'correct target objects "
                "and 2x2 grid layout')."
            ),
        },
        "domain_distance": {
            "type": "STRING",
            "nullable": True,
            "enum": ["adjacent", "distant"],
            "description": (
                "Whether the confused operation families are from "
                "neighboring domains (adjacent) or very different "
                "domains (distant)."
            ),
        },
    },
)

# ── Compositional Sequencing ──────────────────────────────────────────────

_COMPOSITIONAL_SEQUENCING = CategoryDefinition(
    name="compositional_sequencing",
    display_name="Compositional Sequencing Error",
    one_line_summary=(
        "Identifies all correct operations but assembles them in the wrong "
        "order or binds the right operations to the wrong targets."
    ),
    full_section="""\
## Compositional Sequencing Error

**Definition:** The participant correctly identifies all the individual operations required but assembles them in the wrong order, or binds the right operations to the wrong targets when multiple parallel branches must be composed.

**Key diagnostic:** Each individual operation is recognizable as part of the ground-truth solution. The error emerges only at the level of composition. The participant's output contains all the right pieces but in the wrong spatial arrangement or assigned to the wrong targets.

**Distinguishing criteria:** Unlike attribute substitution, the correct operations and attributes are identified. Unlike scope restriction, all operations are applied to all relevant targets. Unlike operator confusion, the individual operations are correctly selected — the error is in how they are assembled.

**Sub-types:**

- **sequential_ordering**: Correct operations applied in the wrong order (where order matters for the result).
- **parallel_binding**: Correct set of operations and targets, but incorrect mapping of which operation goes with which target.""",
    specific_fields=[
        "sequencing_sub_type",
        "operations_involved",
        "sequencing_description",
    ],
    schema_properties={
        "sequencing_sub_type": {
            "type": "STRING",
            "nullable": True,
            "enum": ["sequential_ordering", "parallel_binding"],
            "description": "Sub-type of compositional sequencing error.",
        },
        "operations_involved": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "A natural-language description of the operations "
                "that are mis-sequenced (e.g., 'recoloring the "
                "border and filling the interior')."
            ),
        },
        "sequencing_description": {
            "type": "STRING",
            "nullable": True,
            "description": "How the sequencing differs from GT.",
        },
    },
)

# ── Partial Program ───────────────────────────────────────────────────────

_PARTIAL_PROGRAM = CategoryDefinition(
    name="partial_program",
    display_name="Partial Program",
    one_line_summary=(
        "Executes a proper subset of the ground-truth pipeline steps; the "
        "steps present are correct but some computational phases are missing entirely."
    ),
    full_section="""\
## Partial Program

**Definition:** The participant has correctly identified part of the transformation pipeline but has not completed it. The steps that are present are correct — the error is that some steps are missing entirely.

**Key diagnostic:** The participant's output is consistent with executing a proper subset of the ground-truth program's steps. In the DSL, this manifests as step deletions where the deleted steps constitute a coherent phase of the computation.

**Distinguishing criteria:** Unlike scope restriction (where the full pipeline is applied to a subset of targets), here the pipeline itself is incomplete — some computational steps are missing. Unlike attribute substitution, the steps that are present use the correct features and attributes.""",
    specific_fields=[
        "steps_present",
        "steps_missing",
        "missing_phase_function",
        "missing_phase_position",
    ],
    schema_properties={
        "steps_present": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "A natural-language description of the GT steps "
                "that are preserved (e.g., 'identifies the colored "
                "objects and extracts their bounding boxes')."
            ),
        },
        "steps_missing": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "A natural-language description of the GT steps "
                "that are absent (e.g., 'does not recolor the "
                "extracted shapes according to their size rank')."
            ),
        },
        "missing_phase_function": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "What the missing steps accomplish in perceptual "
                "terms (e.g., 'recoloring the output objects', "
                "'extending the pattern to fill the grid')."
            ),
        },
        "missing_phase_position": {
            "type": "STRING",
            "nullable": True,
            "enum": ["early", "middle", "late"],
            "description": (
                "Where in the pipeline the missing steps fall."
            ),
        },
    },
)

# ── Boundary Precision ────────────────────────────────────────────────────

_BOUNDARY_PRECISION = CategoryDefinition(
    name="boundary_precision",
    display_name="Boundary Precision Error",
    one_line_summary=(
        "Computes the correct type of quantity from the correct source but "
        "is off by ±1 (or a small constant) in counting, indexing, or "
        "boundary placement — and training examples discriminate the two values."
    ),
    full_section="""\
## Boundary Precision Error

**Definition:** The participant computes the correct type of quantity from the correct source but is off by one (or a small constant) in counting, indexing, or boundary placement — and this discrepancy is not explainable by overfitting to training examples or a subset.

**Key diagnostic:** Two conditions must both hold: (1) the computation targets the correct feature and uses the correct operation class, differing only by a small additive offset (typically ±1), and (2) the training examples vary sufficiently that the off-by-one error produces visible mistakes on training data (i.e., the error is not attributable to hardcoding a value that happens to fit some training examples).

**Distinguishing criteria:** Unlike attribute substitution, there is no salience or complexity asymmetry — `height` and `height - 1` are equally accessible. The critical test: if training examples do not discriminate between the two values, classify as attribute substitution (hardcoded constant) instead.""",
    specific_fields=[
        "target_quantity",
        "correct_value",
        "participant_value",
        "offset",
        "training_discriminates",
    ],
    schema_properties={
        "target_quantity": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The perceptual quantity that is off by a small "
                "constant, in intuitive terms (e.g., 'height of "
                "the filled region', 'number of repetitions')."
            ),
        },
        "correct_value": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The GT boundary/count value, described with its "
                "semantic type (e.g., 'a height of 3 rows', "
                "'repeating 5 times')."
            ),
        },
        "participant_value": {
            "type": "STRING",
            "nullable": True,
            "description": (
                "The participant's boundary/count value, in the "
                "same format (e.g., 'a height of 2 rows', "
                "'repeating 4 times')."
            ),
        },
        "offset": {
            "type": "INTEGER",
            "nullable": True,
            "description": (
                "The signed difference (participant - correct)."
            ),
        },
        "training_discriminates": {
            "type": "BOOLEAN",
            "nullable": True,
            "description": (
                "Whether training examples distinguish the two values."
            ),
        },
    },
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CATEGORIES: Dict[str, CategoryDefinition] = {
    cat.name: cat
    for cat in [
        _ATTRIBUTE_SUBSTITUTION,
        _SCOPE_RESTRICTION,
        _OPERATOR_CONFUSION,
        _TRANSFORMATION_SUBSTITUTION,
        _COMPOSITIONAL_SEQUENCING,
        _PARTIAL_PROGRAM,
        _BOUNDARY_PRECISION,
    ]
}
"""All 7 classifiable error categories (excludes ``unclassifiable``)."""

CATEGORY_NAMES: List[str] = list(CATEGORIES.keys())
"""Ordered list of category names."""


# ---------------------------------------------------------------------------
# Display-label overrides
# ---------------------------------------------------------------------------

DISPLAY_LABEL_OVERRIDES: Dict[str, str] = {
    "attribute_substitution": "Attribute Misselection",
    "transformation_substitution": "Transformation Misselection",
}
"""Full-title display overrides keyed by internal slug.

Consumers that otherwise derive a human-readable label from the slug via
``slug.replace("_", " ").title()`` (e.g. the length-by-difficulty forest table)
should look here first, so a category can be *displayed* under a name that
differs from its frozen slug.  Scoped to only the
renamed categories: any slug absent here keeps its title-case rendering.
"""
