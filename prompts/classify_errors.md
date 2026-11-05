You are an expert computational cognitive scientist studying human errors on visual program synthesis tasks (ARC — Abstraction and Reasoning Corpus). For each participant error, we have synthesized a program that reproduces the participant's incorrect output. Your task is to characterize the cognitive error that this program represents using a structured taxonomy.

# DSL Reference

The following DSL functions are used in the programs below (signatures + docstrings):

```python
{{dsl_code}}
```

# Task Data

**Ground-truth program** (solves the task correctly):

```python
{{ref_program}}
```

**Training Examples (Input → Output):**
{{ref_examples}}

# Error Program

The following program was synthesized to reproduce a human participant's incorrect output on this task:

```python
{{error_program}}
```

## Behavioral Comparison on Training Examples

Below is a side-by-side comparison of expected outputs (from the ground-truth program) vs the inferred error program's (unobserved) outputs on each training example. This reveals how the synthesized program behaves on the training examples:

{{training_comparison}}

## Error Program Test Output

The error program produces the following output on the test input:

{{participant_output}}

{{participant_description}}

# Error Category Taxonomy

Classify the error into exactly one of the following 8 categories. Read all definitions before deciding.

## 1. Attribute Substitution

**Definition:** The participant replaces a target quantity, property, or feature that must be computed or inferred with a different one. Often, this is a simpler strategy: the participant may anchor to a specific training example's attribute (e.g., hardcoding a color or dimension) rather than computing the correct general rule.

**Key diagnostic:** The error lies in the _value, parameter, or feature_ used by an operation, rather than the operation itself. The substituted attribute is often a hardcoded constant, a perceptually salient feature, or one that requires fewer inferential steps. These substituted attributes can often be easily inferred from a subset (or more rarely, all) of the training examples.

**Distinguishing criteria:** Unlike operator confusion, the error is in the _attribute_ or _input value_ (e.g., color, size, grid dimensions) rather than the transformation applied. Unlike scope restriction, the participant applies a different computation rather than applying the correct one incompletely.

## 2. Scope Restriction

**Definition:** The participant infers the correct computation and applies it correctly, but only to a subset of the entities, regions, or conditions where it should apply.

**Key diagnostic:** The output is consistent with the ground-truth rule within the restricted scope. The error is not in what is computed but in where or how broadly it is applied.

**Distinguishing criteria:** Unlike attribute substitution, the core computation itself is correct — the error is purely in the extent of its application. Unlike operator confusion, the participant has identified the right high-level operation and applies it faithfully where they do apply it.

## 3. Operator Confusion

**Definition:** The participant correctly identifies the class of operation needed but selects the wrong member of that class (e.g., rotating left instead of right, mirroring horizontally instead of vertically). Note that this "operation" may be a composite subprogram, and does not have to correspond to a single DSL function.

**Key diagnostic:** The error reflects a spatial or perceptual discrimination failure between similar, alternative transformations within the same family.

**Distinguishing criteria:** Unlike attribute substitution, the error is in the _transformation itself_ (the operation) rather than an input attribute or parameter (like a color or size). Unlike scope restriction, the operation is applied to all the right targets — the error is in the choice of operation.

## 4. Transformation Substitution

**Definition:** The participant correctly identifies the target objects, regions, or spatial structure of the task, but applies a transformation from an entirely different computational family than the one required — e.g., recoloring where mirroring was needed, spatial shifting where concentric expansion was needed, cropping where tiling was needed. The scaffolding (object identification, grid decomposition) is often correct; the operation itself is from the wrong conceptual domain.

**Key diagnostic:** The participant's program operates on the right targets but performs a categorically different kind of computation. The correct and applied operations are not alternative members of the same class (that would be operator confusion) — they belong to different transformation families altogether.

**Distinguishing criteria:** Unlike **operator confusion**, the two operations are not from the same class (e.g., mirror vs. rotate are both spatial transformations but from different families; CW vs. CCW rotation are same-family alternatives). Unlike **attribute substitution**, the error is in the _operation applied_, not in the _input feature_ used to parameterize it. Unlike **partial program**, the participant is not missing steps — they are performing _different_ steps. Unlike **unclassifiable**, the participant has engaged with the task structure (correct targets, correct spatial layout) and the applied operation follows a coherent, non-ad-hoc logic.

## 5. Compositional Sequencing Error

**Definition:** The participant correctly identifies all the individual operations required but assembles them in the wrong order, or binds the right operations to the wrong targets when multiple parallel branches must be composed.

**Key diagnostic:** Each individual operation is recognizable as part of the ground-truth solution. The error emerges only at the level of composition. The participant's output contains all the right pieces but in the wrong spatial arrangement or assigned to the wrong targets.

**Distinguishing criteria:** Unlike attribute substitution, the correct operations and attributes are identified. Unlike scope restriction, all operations are applied to all relevant targets. Unlike operator confusion, the individual operations are correctly selected — the error is in how they are assembled.

**Sub-types:**

- **sequential_ordering**: Correct operations applied in the wrong order (where order matters for the result).
- **parallel_binding**: Correct set of operations and targets, but incorrect mapping of which operation goes with which target.

## 6. Partial Program

**Definition:** The participant has correctly identified part of the transformation pipeline but has not completed it. The steps that are present are correct — the error is that some steps are missing entirely.

**Key diagnostic:** The participant's output is consistent with executing a proper subset of the ground-truth program's steps. In the DSL, this manifests as step deletions where the deleted steps constitute a coherent phase of the computation.

**Distinguishing criteria:** Unlike scope restriction (where the full pipeline is applied to a subset of targets), here the pipeline itself is incomplete — some computational steps are missing. Unlike attribute substitution, the steps that are present use the correct features and attributes.

## 7. Boundary Precision Error

**Definition:** The participant computes the correct type of quantity from the correct source but is off by one (or a small constant) in counting, indexing, or boundary placement — and this discrepancy is not explainable by overfitting to training examples or a subset.

**Key diagnostic:** Two conditions must both hold: (1) the computation targets the correct feature and uses the correct operation class, differing only by a small additive offset (typically ±1), and (2) the training examples vary sufficiently that the off-by-one error produces visible mistakes on training data (i.e., the error is not attributable to hardcoding a value that happens to fit some training examples).

**Distinguishing criteria:** Unlike attribute substitution, there is no salience or complexity asymmetry — `height` and `height - 1` are equally accessible. The critical test: if training examples do not discriminate between the two values, classify as attribute substitution (hardcoded constant) instead.

## 8. Unclassifiable / Ad-hoc Rule

**Definition:** The participant's response does not reflect a localized error within the correct task structure. Instead, the participant appears to have constructed an idiosyncratic rule that bears little or no structural relationship to the ground-truth transformation.

**Key diagnostic:** The synthesized program requires rewrites so extensive that the original program's computational skeleton is no longer recognizable. The surviving lines, if any, are generic structural scaffolding rather than task-specific computation.

**Distinguishing criteria:** In all other categories, the participant has latched to some extent onto the correct task structure and made a localized error within it. Here, the participant has not identified the correct structure.

# Quick Decision Guide

Ask these questions in order. Stop at the first "yes":

1. Is the participant's program a proper subset of the ground-truth pipeline, with missing steps representing unperformed computation? → **partial_program**
2. Is the correct operation applied, but not everywhere it should be? → **scope_restriction**
3. Is a different attribute/feature (e.g., color, dimension) used in place of the computed/inferred one, perhaps resorting to a simpler strategy or hardcoded value? → **attribute_substitution**

- Sub-check: If the difference is only ±1, do the training examples discriminate between the two values? If no → attribute_substitution (hardcoded_constant). If yes → **boundary_precision** (skip to 7).

4. Is the right operation class identified, but the wrong member of that class selected (e.g., wrong rotation direction, wrong mirroring axis)? → **operator_confusion**
5. Are the correct targets identified, but is a transformation from an entirely different computational family applied? → **transformation_substitution**
6. Are all the right operations identified and applied to all targets, but assembled in the wrong order or bound to the wrong targets? → **compositional_sequencing**
7. Is the computation off by exactly ±1 from the correct value, and do the training examples rule out overfitting? → **boundary_precision**
8. Does the participant's program bear little structural resemblance to the ground truth? → **unclassifiable**

# Response Instructions

Follow these three steps in order:

**Step 1 — Rule descriptions.** Write two separate descriptions:

- `**correct_rule_description`\*\*: State what the ground-truth program is doing as a rule a human could articulate — what visual feature it attends to, what transformation it performs, and on what targets.
- `**error_rule_description`\*\*: State what the error program is doing as a rule — the same format, but describing the participant's (incorrect) approach.

Both descriptions should be precise but written in cognitive/perceptual terms, not DSL or mathematical terms. Describe _behavior_ (what changes in the output and why), not _syntax_ (which DSL line changed). This is a process-level description, but in cognitive terms.

**Vocabulary guidance:** Avoid phrases like "normalized direction vector", "upper-left corner coordinate", or "position function output". Instead use natural perceptual language: "which direction the colored dot points from the central shape", "how far apart the objects are", "the color of the largest region". The programs use numeric color codes (0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=light blue, 9=maroon) — always translate these to color names in your descriptions; never write a bare number when you mean a color. This guidance applies to all fields, including category-specific fields.

**Step 1b — Operation domain.** In a short sentence, name the kind of mental operation at stake in this error — the cognitive faculty or perceptual domain the participant had to engage with (e.g., "spatial placement", "object counting", "color attribution", "shape matching"). Do not use DSL function names. This label should describe what type of reasoning the task demands. Many tasks involve muiltiple types of reasoning, focus on the kind of computation that the participant failed at.

**Step 2 — Category selection.** Walk through the Quick Decision Guide above. Select the single best-fitting `error_category` from the 8 options.

**Step 3 — Fill category-specific fields.** Based on the selected `error_category`, fill in ONLY the fields for that category (see table below). All fields for other categories must be null.

## Field-to-Category Mapping

| Category                      | Fields to fill (non-null)                                                                                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attribute_substitution`      | `target_parameter`, `correct_attribute`, `substituted_attribute`, `substitution_type`, `hardcoded_value` (only when substitution_type = hardcoded_constant) |
| `scope_restriction`           | `restricted_operation`, `correct_scope`, `actual_scope`, `fraction_covered`                                                                                 |
| `operator_confusion`          | `operator_target`, `operation_class`, `correct_operator`, `confused_operator`, `confusion_axis`                                                             |
| `transformation_substitution` | `correct_operation_domain`, `applied_operation_domain`, `shared_structure`, `domain_distance`                                                               |
| `compositional_sequencing`    | `sequencing_sub_type`, `operations_involved`, `sequencing_description`                                                                                      |
| `partial_program`             | `steps_present`, `steps_missing`, `missing_phase_function`, `missing_phase_position`                                                                        |
| `boundary_precision`          | `target_quantity`, `correct_value`, `participant_value`, `offset`, `training_discriminates`                                                                 |
| `unclassifiable`              | `apparent_strategy`, `structural_relationship_to_gt`                                                                                                        |

# Response Format

Respond **only** with a JSON object containing all fields below. Fields not belonging to your selected category must be null.

```json
{
  "correct_rule_description": "string — what the ground-truth program does, as a rule a human could articulate, in perceptual/cognitive terms (e.g., 'copy the small shape in the direction it points from the central pattern, at intervals of 4')",
  "error_rule_description": "string — what the error program does instead, as a rule in perceptual/cognitive terms (e.g., 'copy the small shape by stepping the same distance as the gap between the objects, rather than one unit in that direction')",
  "operation_domain": "string — one to three words naming the cognitive/perceptual faculty at stake (e.g., 'spatial placement', 'object counting', 'color attribution', 'shape matching') — not a DSL term, not an error category",
  "error_category": "attribute_substitution | scope_restriction | operator_confusion | transformation_substitution | compositional_sequencing | partial_program | boundary_precision | unclassifiable",

  "target_parameter": "string or null — the quantity or parameter in the transformation that the correct and substituted attributes are used to determine, in perceptual terms (e.g., 'width of the drawn patterns', 'placement direction of the copied shape')",
  "correct_attribute": "string or null — the perceptual or cognitive feature the correct approach relies on, described in intuitive terms (e.g., 'the direction the small dot points relative to the central shape', NOT 'normalized position vector')",
  "substituted_attribute": "string or null — the simpler or more salient feature the participant used instead, in intuitive terms (e.g., 'the raw spatial gap between the objects', NOT 'upper-left corner coordinate difference')",
  "substitution_type": "computed_property | hardcoded_constant | surface_correlation | null — 'hardcoded_constant' means a fixed literal value (e.g. 'the color blue', 'a height of 5'); 'surface_correlation' means using a spurious correlation from the training examples (e.g. using object size when it should be position, because they happen to correlate in training); 'computed_property' means calculating a completely different logical property",
  "hardcoded_value": "string or null — a natural-language description of the literal value substituted (e.g., 'a height of 5', 'the color blue', 'the top-left corner of the second subgrid'). Use color names, not numeric codes. Only fill when substitution_type = hardcoded_constant.",

  "restricted_operation": "string or null — the transformation that is correctly applied but to a limited scope, in perceptual terms (e.g., 'filling the shapes with color', 'copying the large objects', 'extending the pattern in each cardinal direction')",
  "correct_scope": "string or null — what the GT program applies the operation to.",
  "actual_scope": "string or null — what the error program applies the operation to",
  "fraction_covered": "number or null — approximate fraction of targets covered (0.0–1.0)",

  "operator_target": "string or null — what aspect of the transformation the operator determines, in perceptual terms (e.g., 'orientation of the copied shape', 'direction of the fill pattern')",
  "operation_class": "string or null — the class of operation (e.g., rotation, mirroring, selection)",
  "correct_operator": "string or null — the GT operator in cognitive terms",
  "confused_operator": "string or null — the operator the error program uses in cognitive terms",
  "confusion_axis": "string or null — what dimension the confusion is along, if applicable (e.g., CW vs CCW, horizontal vs vertical)",

  "correct_operation_domain": "string or null — the family of transformation the ground-truth program uses, in perceptual terms (e.g., 'rotation-based tiling', 'concentric expansion', 'pattern intersection')",
  "applied_operation_domain": "string or null — the family of transformation the error program uses instead (e.g., 'mirror-based tiling', 'spatial shift', 'color substitution')",
  "shared_structure": "string or null — what aspects of the task structure the participant correctly identified (e.g., 'correct target objects and 2x2 grid layout', 'correct object identification and bounding')",
  "domain_distance": "adjacent | distant | null — whether the confused operation families are from neighboring domains (e.g., both spatial but different families) or very different domains (e.g., geometric vs. recoloring)",

  "sequencing_sub_type": "sequential_ordering | parallel_binding | null",
  "operations_involved": "string or null — a natural-language description of the operations that are mis-sequenced (e.g., 'recoloring the border and filling the interior')",
  "sequencing_description": "string or null — how the sequencing differs from GT",

  "steps_present": "string or null — a natural-language description of the GT steps that are preserved (e.g., 'identifies the colored objects and extracts their bounding boxes')",
  "steps_missing": "string or null — a natural-language description of the GT steps that are absent (e.g., 'does not recolor the extracted shapes according to their size rank')",
  "missing_phase_function": "string or null — what the missing steps accomplish in perceptual terms (e.g., 'recoloring the output objects', 'extending the pattern to fill the grid')",
  "missing_phase_position": "early | middle | late | null — where in the pipeline the missing steps fall",

  "target_quantity": "string or null — the perceptual quantity that is off by a small constant, in intuitive terms (e.g., 'height of the filled region', 'number of repetitions', 'gap between copied shapes')",
  "correct_value": "string or null — the GT boundary/count value, described with its semantic type (e.g., 'a height of 3 rows', 'repeating 5 times', '2 cells from the edge')",
  "participant_value": "string or null — the participant's boundary/count value, in the same format (e.g., 'a height of 2 rows', 'repeating 4 times', '1 cell from the edge')",
  "offset": "integer or null — the signed difference (participant - correct)",
  "training_discriminates": "boolean or null — whether training examples distinguish the two values",

  "apparent_strategy": "string or null — a one- to two-sentence description of the rule the participant appears to have followed, written in the same perceptual/cognitive vocabulary as the rule descriptions above (e.g., 'the participant colored each cell based on its row index rather than responding to any feature of the input objects', 'the participant appears to have copied the input grid verbatim and overlaid a fixed pattern unrelated to the task structure')",
  "structural_relationship_to_gt": "none | minimal_scaffolding | null"
}
```
