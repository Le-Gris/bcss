"""Shared figure saving for the analysis scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

FIGURE_FORMATS = ("png", "pdf")


def save_figure(
    fig: Figure,
    stem: str | Path,
    *,
    transparent: bool,
    dpi: int = 300,
    close: bool = True,
) -> None:
    """Save *fig* as ``<stem>.png`` and ``<stem>.pdf``.

    *stem* is the output path without an extension.  With ``transparent=True``
    every format has a transparent background; otherwise the background is
    opaque white.  The figure is closed afterwards unless ``close=False``.
    """
    kw: dict = dict(dpi=dpi, bbox_inches="tight", transparent=transparent)
    if not transparent:
        kw["facecolor"] = "white"
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        path = stem.parent / f"{stem.name}.{ext}"
        fig.savefig(path, **kw)
        print(f"Saved {path}")
    if close:
        plt.close(fig)
