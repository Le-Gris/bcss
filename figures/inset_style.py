"""Plotting helpers for the result insets in Figures 2C, 3C and 4C.

Two chart types: a stacked treatment-vs-control histogram (simplicity) and
horizontal bar charts of posterior proportions with 94% HDI whiskers (rule
types, order effects, error categories).  Figures are saved as PDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rule_types import ORDER_TIER_ORDER, RULE_TYPE_ORDER  # noqa: E402,F401

FORMATS = ("pdf",)

INK, MUTED = "#1a1a1a", "#6b6b6b"
BLUE = "#4878cf"
GROUP_COLORS = ["#6f6f6f", "#bdbdbd"]  # first series darker

_SHORT = {"Partial match": "Partial", "Alt. rule": "Alt.", "Neither": "Neither",
          "first-k": "First-k", "last-k": "Last-k",
          "contiguous": "Contiguous", "other": "Other"}


def _save(fig, out_stem: str | Path, transparent: bool) -> None:
    kw = dict(dpi=300, bbox_inches="tight", transparent=transparent)
    if not transparent:
        kw["facecolor"] = "white"
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(f"{out_stem}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {out_stem}.{{{','.join(FORMATS)}}}")


# ── simplicity: treatment vs control per-task histograms ─────────────────────
def save_condition_dist_grid(panels, out_stem, *, figsize=None, transparent=False,
                             bins=28, share_x=True, xlabel=None, legend=False):
    """One treatment-vs-control histogram per split, stacked on a shared x-axis.

    *panels* is a list of ``(subtitle, treatment, control)``.  The y-axis (% of
    tasks) is broken so the treatment shape (0-20%) and the control spike at 0
    (70-80%) are both visible.
    """
    n = len(panels)
    if figsize is None:
        figsize = (4.0, 1.45 * n + 0.3)
    edges = None
    if share_x:
        allv = np.concatenate(
            [np.asarray(t) for _, t, _ in panels] + [np.asarray(c) for _, _, c in panels]
        )
        edges = np.linspace(float(allv.min()), float(allv.max()), bins + 1)

    fig = plt.figure(figsize=figsize, dpi=300)
    left, width = 0.14, 0.82
    top_m, bot_m, gap_panel = 0.05, 0.12, 0.14
    ph = (1.0 - top_m - bot_m - gap_panel * (n - 1)) / n
    gap_break = 0.028
    h_top = ph * 0.26
    h_bot = ph - h_top - gap_break

    for i, (subtitle, treat, ctrl) in enumerate(panels):
        y_bot = bot_m + (n - 1 - i) * (ph + gap_panel)
        y_top = y_bot + h_bot + gap_break

        ax_top = fig.add_axes([left, y_top, width, h_top])
        ax_bot = fig.add_axes([left, y_bot, width, h_bot])

        pct_t = np.histogram(treat, bins=edges)[0] / len(treat) * 100
        pct_c = np.histogram(ctrl, bins=edges)[0] / len(ctrl) * 100

        for ax in (ax_top, ax_bot):
            ax.bar(edges[:-1], pct_t, width=np.diff(edges), align="edge",
                   color=BLUE, alpha=0.65, ec="white", lw=0.2)
            ax.stairs(pct_c, edges, color=MUTED, lw=1.3)
            ax.set_xlim(edges[0] - 0.02, 0.02)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color("#ccc")
            ax.tick_params(length=0, labelsize=7.0, colors=MUTED)

        ax_bot.set_ylim(0, 20.8)
        ax_bot.set_yticks([0, 10, 20])
        ax_bot.set_yticklabels(["0%", "10%", "20%"], fontsize=7.0)
        ax_bot.spines["bottom"].set_color("#ccc")
        ax_bot.spines["top"].set_visible(False)
        ax_bot.tick_params(axis="x", labelsize=8.5)
        if i != n - 1:
            ax_bot.set_xticklabels([])
        elif xlabel:
            ax_bot.set_xlabel(xlabel, fontsize=9, color=MUTED)

        ax_top.set_ylim(65, 83)
        ax_top.set_yticks([80])
        ax_top.set_yticklabels(["80%"], fontsize=7.0)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.set_xticklabels([])
        ax_top.tick_params(bottom=False)

        # break marks on the left spine
        y_mid = y_bot + h_bot + gap_break / 2
        slash_dx, slash_dy = 0.010, 0.009
        for offset_y in (-0.005, 0.005):
            fig.lines.append(plt.Line2D([left - slash_dx, left + slash_dx],
                                        [y_mid + offset_y - slash_dy, y_mid + offset_y + slash_dy],
                                        color="#888", lw=1.1, transform=fig.transFigure,
                                        figure=fig))

        if subtitle:
            ax_top.text(0.04, 0.95, subtitle, transform=ax_top.transAxes, color=INK,
                        fontsize=9.0, fontweight="bold", va="top")
        if i == 0 and legend:
            ax_bot.text(0.84, 0.90, "treatment", transform=ax_bot.transAxes, color=BLUE,
                        fontsize=6.8, fontweight="bold", va="top", ha="right")
            ax_bot.text(0.84, 0.72, "control", transform=ax_bot.transAxes, color=MUTED,
                        fontsize=6.8, fontweight="bold", va="top", ha="right")

    _save(fig, out_stem, transparent)
    return out_stem


# ── posterior proportions: horizontal bars with 94% HDI whiskers ─────────────
def _estimand_barh(fig, rect, probs, order, *, title=None, highlight=None,
                   fs_ylab=6.6, fs_val=6.6, fs_xtick=6.0, fs_title=7.3, bar_h=0.64):
    """probs: {category: [median, hdi_lo, hdi_hi]}; order: top-to-bottom."""
    ax = fig.add_axes(rect)
    cats = [c for c in order if c in probs]
    med = [probs[c][0] for c in cats]
    lo = [probs[c][1] for c in cats]
    hi = [probs[c][2] for c in cats]
    yp = np.arange(len(cats))[::-1]
    cols = [BLUE if (highlight and c == highlight) else "#c4c4c4" for c in cats]
    xerr = np.array([[m - l for m, l in zip(med, lo)], [h - m for h, m in zip(hi, med)]])
    ax.barh(yp, med, color=cols, ec="white", height=bar_h)
    ax.errorbar(med, yp, xerr=xerr, fmt="none", ecolor="#555", elinewidth=0.8, capsize=1.5)
    for y, m in zip(yp, med):
        ax.text(1.03, y, f"{m:.1%}", va="center", ha="left", fontsize=fs_val, color=INK, clip_on=False)
    ax.set_yticks(yp)
    ax.set_yticklabels([_SHORT.get(c, c.capitalize()) for c in cats], fontsize=fs_ylab)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "50", "100%"], fontsize=fs_xtick)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#ccc"); ax.spines["bottom"].set_color("#ccc")
    if title:
        ax.set_title(title, fontsize=fs_title, color=MUTED, loc="left", pad=4)
    return ax


def save_estimand_figure(probs, order, out_stem, *, title=None, highlight=None,
                         figsize=(3.7, 1.5), transparent=False):
    """One bar chart of posterior proportions."""
    fig = plt.figure(figsize=figsize, dpi=300)
    _estimand_barh(fig, [0.30, 0.24, 0.60, 0.60], probs, order,
                   title=title, highlight=highlight,
                   fs_ylab=9.5, fs_val=9.0, fs_xtick=8.0, fs_title=10.5, bar_h=0.62)
    _save(fig, out_stem, transparent)
    return out_stem


def _estimand_grouped_barh(fig, rect, series, order, *, colors=None,
                           fs_ylab=9.0, fs_val=8.0, fs_xtick=8.0, fs_legend=8.0,
                           group_h=0.82, legend_loc="lower right"):
    """series: [(label, probs)], one shaded bar per series per category."""
    ax = fig.add_axes(rect)
    colors = colors or GROUP_COLORS
    cats = [c for c in order if all(c in p for _, p in series)]
    m = len(series)
    base = np.arange(len(cats))[::-1].astype(float)
    bar_h = group_h / m
    for si, (label, probs) in enumerate(series):
        med = np.array([probs[c][0] for c in cats])
        lo = np.array([probs[c][1] for c in cats])
        hi = np.array([probs[c][2] for c in cats])
        yp = base + ((m - 1) / 2 - si) * bar_h
        xerr = np.array([med - lo, hi - med])
        ax.barh(yp, med, height=bar_h, color=colors[si % len(colors)],
                ec="white", label=label, zorder=2)
        ax.errorbar(med, yp, xerr=xerr, fmt="none", ecolor="#555",
                    elinewidth=0.7, capsize=1.2, zorder=3)
        for y, mv, h in zip(yp, med, hi):
            ax.text(h + 0.02, y, f"{mv:.1%}", va="center", ha="left",
                    fontsize=fs_val, color=INK, clip_on=False)
    ax.set_yticks(base)
    ax.set_yticklabels([_SHORT.get(c, c.capitalize()) for c in cats], fontsize=fs_ylab)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100%"], fontsize=fs_xtick)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#ccc"); ax.spines["bottom"].set_color("#ccc")
    ax.legend(loc=legend_loc, frameon=False, fontsize=fs_legend,
              handlelength=1.0, handletextpad=0.5, borderaxespad=0.3, labelspacing=0.3)
    return ax


def save_estimand_grouped(series, order, out_stem, *, colors=None, figsize=None,
                          transparent=False, legend_loc="lower right"):
    """Several splits' posterior proportions as one grouped bar chart."""
    ncat = len([c for c in order if all(c in p for _, p in series)])
    if figsize is None:
        figsize = (3.9, 0.45 * ncat + 0.6)
    fig = plt.figure(figsize=figsize, dpi=300)
    _estimand_grouped_barh(fig, [0.30, 0.20, 0.60, 0.72], series, order, colors=colors,
                           legend_loc=legend_loc)
    _save(fig, out_stem, transparent)
    return out_stem
