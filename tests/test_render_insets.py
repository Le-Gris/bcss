"""Renders the Figure 2C, 3C and 4C insets from the committed summaries.

``figures/render_insets.py`` reads the CSVs under ``results/analyses/human/``
by path, so this fails if the committed layout and the paths the renderer
builds drift apart.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEMS = ("fig2c_simplicity", "fig3c_rule_type", "fig3c_order_effects",
         "fig4c_error_category")


def _load_render_insets():
    spec = importlib.util.spec_from_file_location(
        "render_insets", ROOT / "figures" / "render_insets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_insets_render_from_committed_summaries(tmp_path, monkeypatch):
    render_insets = _load_render_insets()
    monkeypatch.setattr(sys, "argv", [
        "render_insets.py",
        "--analyses-dir", str(ROOT / "results" / "analyses"),
        "--output-dir", str(tmp_path),
    ])
    render_insets.main()

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == sorted(f"{s}.{ext}" for s in STEMS for ext in ("pdf", "png"))
    assert all((tmp_path / name).stat().st_size > 0 for name in written)
