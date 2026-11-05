"""Rule type of an error program, from which training examples it reproduces.

``rule_type`` labels a program ``alternative`` (reproduces every training
output), ``partial`` (some but not all), ``neither`` (none) or ``unknown`` (no
data). For partial matches it also returns an order tag: whether the
reproduced examples are the first k, the last k, a contiguous run, or other.
The display labels and plotting order are defined here too.
"""

from __future__ import annotations

# Rule-type labels (lowercase; used as HTML badge keys downstream).
ALTERNATIVE = "alternative"
PARTIAL = "partial"
NEITHER = "neither"
UNKNOWN = "unknown"


def rule_type(training_matches: list[bool] | None) -> tuple[str, str | None]:
    """Classify a program's rule type from its per-train-pair match mask.

    ``training_matches`` is the stage-3 ``list[bool]`` (one entry per training
    pair).  With ``k = sum`` matches over ``N`` pairs:

    * ``k == N``      -> ``("alternative", None)`` — reproduces *every* train pair
    * ``0 < k < N``   -> ``("partial", order_tag)`` — see below
    * ``k == 0``      -> ``("neither", None)`` — a genuine error
    * empty / ``None``-> ``("unknown", None)``

    For partial matches the *order tag* describes where the matched pairs fall,
    derived from their positions (not from the stage-3 ``explains_in_order``
    field, which only detects the prefix case).  Let the matched positions form
    a single unbroken run iff ``idx[-1] - idx[0] + 1 == k``:

    * run touching the start (``idx[0] == 0``)   -> ``"first-k"`` (prefix run)
    * run touching the end (``idx[-1] == n-1``)  -> ``"last-k"`` (suffix run)
    * interior run of length >= 2                 -> ``"contiguous"``
    * single interior match OR non-contiguous set -> ``"other"``

    Note the run tags absorb the length-1 boundary cases (a lone match at
    pair 0 is ``"first-k"``; at the last pair is ``"last-k"``).  ``"other"`` is
    the catch-all for a lone interior match *and* a scattered (gapped) set.
    """
    if not training_matches:
        return (UNKNOWN, None)
    mask = [bool(m) for m in training_matches]
    n = len(mask)
    k = sum(mask)
    if k == 0:
        return (NEITHER, None)
    if k == n:
        return (ALTERNATIVE, None)
    idx = [i for i, m in enumerate(mask) if m]
    is_run = (idx[-1] - idx[0] + 1) == k        # matched positions form one block
    if is_run and idx[0] == 0:
        return (PARTIAL, "first-k")             # prefix run (incl. lone pair 0)
    if is_run and idx[-1] == n - 1:
        return (PARTIAL, "last-k")              # suffix run (incl. lone last pair)
    if is_run and k > 1:
        return (PARTIAL, "contiguous")          # interior run, length >= 2
    return (PARTIAL, "other")                   # single interior match OR scatter


# ── Analysis-facing constants ─────────────────────────────────────────────
#
# The hierarchical categorical model needs display labels and a fixed
# left-to-right plot ordering.  The reference (softmax-zero) category is the
# alphabetically-first display label — "Alt. rule" — which the model picks up
# automatically via ``sorted(categories)`` in ``build_index_maps``.

# Map the lowercase ``rule_type`` labels to publication display labels.
RULE_TYPE_DISPLAY: dict[str, str] = {
    ALTERNATIVE: "Alt. rule",
    PARTIAL: "Partial match",
    NEITHER: "Neither",
}

# Left-to-right plot order for the rule-type distribution.
RULE_TYPE_ORDER: list[str] = ["Neither", "Partial match", "Alt. rule"]

# Left-to-right plot order for the order-effect distribution (partial matches
# only).  These match the order tags returned by :func:`rule_type` directly.
ORDER_TIER_ORDER: list[str] = ["first-k", "last-k", "contiguous", "other"]
