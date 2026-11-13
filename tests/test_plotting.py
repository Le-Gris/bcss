"""Tests for src.plotting.save_figure."""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src.plotting import FIGURE_FORMATS, save_figure  # noqa: E402


def _figure():
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    return fig


def test_writes_every_format(tmp_path):
    stem = tmp_path / "nested" / "plot"
    save_figure(_figure(), stem, transparent=False)
    for ext in FIGURE_FORMATS:
        assert (tmp_path / "nested" / f"plot.{ext}").stat().st_size > 0


def test_keeps_dots_in_stem(tmp_path):
    save_figure(_figure(), tmp_path / "alpha0.1_forest", transparent=False)
    assert (tmp_path / "alpha0.1_forest.png").exists()
    assert (tmp_path / "alpha0.1_forest.pdf").exists()


def test_transparent_png_has_clear_corner(tmp_path):
    save_figure(_figure(), tmp_path / "t", transparent=True)
    img = mpimg.imread(tmp_path / "t.png")
    assert img.shape[-1] == 4
    assert img[0, 0, 3] == 0


def test_opaque_png_has_white_corner(tmp_path):
    save_figure(_figure(), tmp_path / "o", transparent=False)
    img = mpimg.imread(tmp_path / "o.png")
    corner = img[0, 0]
    assert corner[:3] == pytest.approx([1.0, 1.0, 1.0])
    if img.shape[-1] == 4:
        assert corner[3] == 1.0


# matplotlib writes white as the PDF gray operator "1 g" (set fill to white).
WHITE_FILL = re.compile(rb"(?<![\d.])1 g\b")


def test_transparent_pdf_has_no_white_background(tmp_path):
    # The opaque PDF fills white figure and axes backgrounds; the transparent
    # one fills nothing white.  Compression is off so the content stream is
    # readable.
    with matplotlib.rc_context({"pdf.compression": 0}):
        save_figure(_figure(), tmp_path / "t", transparent=True)
        save_figure(_figure(), tmp_path / "o", transparent=False)
    assert WHITE_FILL.search((tmp_path / "o.pdf").read_bytes())
    assert not WHITE_FILL.search((tmp_path / "t.pdf").read_bytes())


def test_close_flag(tmp_path):
    fig = _figure()
    save_figure(fig, tmp_path / "keep", transparent=False, close=False)
    assert plt.fignum_exists(fig.number)
    save_figure(fig, tmp_path / "drop", transparent=False)
    assert not plt.fignum_exists(fig.number)
