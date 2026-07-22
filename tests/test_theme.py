"""Theme palette, persistence, and plotting guardrails."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from seriesxrd.core.uiprefs import load_prefs, save_prefs
from seriesxrd.guikit import theme


COLOR_ROLES = frozenset({
    "BG", "BG2", "FG", "ACCENT", "ACCENT2", "WARN", "BORDER",
    "ENTRY_BG", "BTN_BG", "BTN_ACT", "MUTED", "CLR_RAW",
    "CLR_MSKD", "CLR_DIFF", "CLR_SMTH", "CLR_REF",
})


def _luminance(color: str) -> float:
    channels = [int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    lighter, darker = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_palettes_have_the_same_semantic_roles_and_aa_text_contrast():
    assert set(theme.PALETTES) == {"mocha", "latte"}
    assert all(set(palette) == COLOR_ROLES
               for palette in theme.PALETTES.values())
    for name, palette in theme.PALETTES.items():
        assert _contrast(palette["FG"], palette["BG"]) >= 4.5, name
        assert _contrast(palette["FG"], palette["BG2"]) >= 4.5, name
        assert _contrast(palette["MUTED"], palette["BG"]) >= 4.5, name


def test_theme_switch_mutates_singleton_and_notifies_once():
    original = theme.active_theme()
    target = "latte" if original == "mocha" else "mocha"
    palette_id = id(theme.C)
    calls = []

    def _callback():
        calls.append(theme.active_theme())

    theme.register_restyle(_callback)
    try:
        assert theme.set_theme(target) == target
        assert id(theme.C) == palette_id
        assert theme.C.as_dict() == theme.PALETTES[target]
        theme.set_theme(target)
        assert calls == [target]
        with pytest.raises(ValueError, match="Unknown theme"):
            theme.set_theme("sepia")
    finally:
        theme.unregister_restyle(_callback)
        theme.set_theme(original)


def test_ui_preferences_round_trip_without_workspace_state(tmp_path):
    path = tmp_path / "config" / "seriesxrd" / "ui.json"
    assert load_prefs(path) == {"theme": "mocha"}
    assert save_prefs({"theme": "latte", "future_setting": True}, path) == path
    assert load_prefs(path) == {"theme": "latte", "future_setting": True}


def test_figure_restyle_changes_surfaces_but_preserves_data_colors():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_hex

    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], [1, 0], color="#123456", label="data")
    ax.legend()
    theme.style_figure(fig, palette="latte", background="#ffffff")
    try:
        assert to_hex(fig.get_facecolor()) == "#ffffff"
        assert to_hex(ax.get_facecolor()) == "#ffffff"
        assert to_hex(ax.xaxis.label.get_color()) == theme.PALETTES["latte"]["FG"]
        assert to_hex(line.get_color()) == "#123456"
    finally:
        plt.close(fig)


def test_gui_code_does_not_snapshot_palette_colors():
    package = Path(__file__).resolve().parents[1] / "seriesxrd"
    offenders = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not (node.module or "").endswith("guikit.theme"):
                continue
            imported = COLOR_ROLES.intersection(alias.name for alias in node.names)
            if imported:
                offenders.append((path.relative_to(package), sorted(imported)))
    assert offenders == []

