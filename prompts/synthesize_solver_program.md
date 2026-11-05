You are an expert program synthesis researcher. Your task is to write a program in a domain-specific language (DSL) that solves a visual pattern transformation task. The task consists of input-output grid pairs that demonstrate a consistent transformation rule. You must discover this rule and express it as a DSL program.

## DSL Reference

The following DSL is available for writing programs:

```python
{{dsl_code}}
```

{{in_context_examples}}

## Target Task

Write a solver program for the following task. Study the training examples to discover the transformation rule, then express it as a DSL program.

{{target_examples}}

{{task_description}}

## Instructions

Write a DSL program that transforms the input grid to the output grid for the target task above. The program should generalize the transformation rule demonstrated by the training examples.

**Syntax rules:**
- Each line of code must follow this form: `x[k] = function_name(arg1, ..., argL)` for line k and L arguments.
- The final output variable must be named `O`.
- Arguments can only be: variable names from previous lines in the program, constants from the DSL, or function names from the DSL passed as references (e.g., `mapply(outbox, x2)` where outbox is passed by name, not called).
- Do not nest function calls — that is, do not use a function call as an argument to another function (e.g., `f(g(x))` is not allowed).
- You cannot use any other operations than the ones in the DSL.
  - Do not use nested functions like a `def` or nested function calls like `func(func(x))`. Instead, compose functions using existing DSL functions.
  - Do not use lambda expressions.
  - Do not use Python primitive operations like `*`, `//`, `+`, `-` and others. For those operations, use the corresponding DSL functions, when available.
  - Do not directly create native Python data structures like lists, tuples, dictionaries, sets.
  - Do not use double variable assignments or use of `*` operator for arguments.
  - Do not perform direct assignment of numbers. Instead, use constants where appropriate.
- The input grid is always available as the variable `I`.

## Response Format

Respond **only** with a JSON object of this form:

```json
{
  "program": "x1 = ...\nx2 = ...\nO = ..."
}
```

- **program**: The complete solver program body, with lines separated by `\n`. Each line assigns to a variable. The last meaningful assignment should produce the output grid, assigned to `O`.
