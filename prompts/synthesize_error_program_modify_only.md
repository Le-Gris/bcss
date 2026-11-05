You are an expert computational cognitive scientist and program synthesis researcher. Participants were given visual program synthesis tasks, and you are trying to model their reasoning. The participant saw a small set of training examples before producing their test output. Your task is to modify a few lines of an existing program to reverse-engineer approximate rule representations.

## Task Overview

You will be given:

1. The full DSL for writing programs that solve visual program synthesis tasks
2. A ground-truth program written in the DSL that solves the specific task a human participant tried
3. The training and test examples for the visual program synthesis task
4. Participant data including the human's test output grid

Your goal is to modify the ground-truth program such that when executed, it produces an output that closely matches the human's response. Since you are limited in how much you can modify the program, the resulting modified program should capture the core conceptual aspects of the human's thinking as much as possible. A program capturing the participant's general strategy is preferable to one that perfectly reconstructs every cell of their output. The question is what alternative or partial rule the participant may have inferred from the training examples, and how that rule diverges from the ground-truth program.

## DSL Reference

The following DSL is available for program modification:

```python
{{dsl_code}}
```

## Task data

**Formal DSL Program:**

```python
{{ref_program}}
```

**Training Examples (Input -> Output):**
{{ref_examples}}

## Participant data

{{participant_description}}

**Participant's Output:**
{{participant_output}}

## Instructions

Produce an alternative program to the ground-truth program that can reproduce the error made by the human participant. You are only allowed to modify {{num_modified_lines}} lines of code in the program, so choose them carefully. The lines of code you choose to modify can be rewritten in the following ways:

1. Modifying an argument to a function (e.g. `replace(I, SIX, TWO)` -> `replace(I, SIX, THREE)`)
2. Replacing an existing function with another function (e.g. `vmirror(x1)` -> `hmirror(x1)`)
3. Regenerating a line of code completely using the DSL (e.g. `upscale(I, THREE)` -> `rot90(I)`)

Here are some rules for writing the each line of code:

- Each line of code you modify or regenerate must follow the same syntax as the ground-truth program:
  - x[k] = function_name(arg1, ..., argL) for line k and L arguments. Arguments can only be: variable names from the existing program, constants from the DSL, or function names from the DSL passed as references (e.g., `mapply(outbox, x2)` where outbox is passed by name, not called). Do not nest function calls — that is, do not use a function call as an argument to another function (e.g., f(g(x)) is not allowed).
- You cannot use any other operations than the ones in the DSL.
  - Do not use nested functions like a `def` inside `verifier` or nested function calls like func(func(x)). Instead, you can compose functions in various ways with existing DSL functions if needed.
  - Do not use lambda expressions
  - Do not use Python primitive operations like `*`, `//`, `+`, `-` and others. For those operations, use the corresponding DSL functions, when available.
  - Do not directly create native Python data structures like lists, tuples, dictionaries, sets.
  - Do not use double variable assignments or use of `*` operator for arguments.
  - Do not perform direct assignment of numbers. Instead, use constants where appropriate.

Your outputs will be used to replace each of the lines you choose to modify in the original program. The resulting program should produce an output that is consistent with the participant's response on the test input.

## Response Format

Respond **only** with a JSON object of this form:

```json
{
  "modifications": [
    { "variable_name": "<var>", "replacement": "<rhs>" }
  ]
}
```

- **variable_name**: Exact variable from the ground-truth program whose assignment line to change (e.g. `"x1"`, `"x2"`, `"O"`).
- **replacement**: The new right-hand side only (e.g. `"rotate(I)"`, `"objects(I, T, F, T)"`). Same DSL syntax: `function_name(arg1, arg2, ...)`. No `var = ` prefix.
- **modifications**: At most {{num_modified_lines}} entries. Each entry replaces one line: the line `variable_name = ...` becomes `variable_name = replacement`.

Example: to change `x3 = hmirror(x2)` to `x3 = vmirror(x2)`, output `{"variable_name": "x3", "replacement": "vmirror(x2)"}` inside `modifications`.
