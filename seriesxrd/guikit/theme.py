"""Live Catppuccin palettes shared by Tk widgets and Matplotlib figures.

``C`` is intentionally mutated in place when :func:`set_theme` is called.
Callers must read colors through ``theme.C`` at widget/figure creation time so
the same imported object remains valid for the lifetime of the process.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any
import weakref


PALETTES: dict[str, dict[str, str]] = {
    "mocha": {
        "BG": "#1e1e2e",
        "BG2": "#2a2a3e",
        "FG": "#cdd6f4",
        "ACCENT": "#89b4fa",
        "ACCENT2": "#a6e3a1",
        "WARN": "#f38ba8",
        "BORDER": "#45475a",
        "ENTRY_BG": "#313244",
        "BTN_BG": "#363650",
        "BTN_ACT": "#45475a",
        "MUTED": "#9399b2",
        "CLR_RAW": "#89b4fa",
        "CLR_MSKD": "#a6e3a1",
        "CLR_DIFF": "#f38ba8",
        "CLR_SMTH": "#fab387",
        "CLR_REF": "#f5c2e7",
    },
    "latte": {
        "BG": "#eff1f5",
        "BG2": "#e6e9ef",
        "FG": "#4c4f69",
        "ACCENT": "#1e66f5",
        "ACCENT2": "#40a02b",
        "WARN": "#d20f39",
        "BORDER": "#bcc0cc",
        "ENTRY_BG": "#dce0e8",
        "BTN_BG": "#ccd0da",
        "BTN_ACT": "#bcc0cc",
        "MUTED": "#5c5f77",
        "CLR_RAW": "#1e66f5",
        "CLR_MSKD": "#40a02b",
        "CLR_DIFF": "#d20f39",
        "CLR_SMTH": "#fe640b",
        "CLR_REF": "#ea76cb",
    },
}


class _Palette:
    """Mutable palette identity with attribute access for semantic roles."""

    def __init__(self, name: str, values: Mapping[str, str]):
        self._replace(name, values)

    def _replace(self, name: str, values: Mapping[str, str]) -> None:
        self.name = name
        for role, value in values.items():
            setattr(self, role, value)

    def as_dict(self) -> dict[str, str]:
        return {role: getattr(self, role) for role in PALETTES[self.name]}


C = _Palette("mocha", PALETTES["mocha"])

# Weak registrations prevent a closed standalone window (or a transient
# widget) from being retained for the remainder of the Python process.
_restyle_callbacks: list[Any] = []
_widgets: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()
_WIDGET_ROLES = frozenset({"window", "panel", "text", "canvas"})


def active_theme() -> str:
    """Return the active palette name (``mocha`` or ``latte``)."""
    return C.name


def _callback_ref(fn: Callable[[], None]):
    if getattr(fn, "__self__", None) is not None:
        return weakref.WeakMethod(fn)
    try:
        return weakref.ref(fn)
    except TypeError:
        # Unusual callable objects may not support weak references. Keeping one
        # strongly is preferable to silently dropping a requested restyle.
        return lambda: fn


def register_restyle(fn: Callable[[], None]) -> None:
    """Register a no-argument callback invoked after a palette change."""
    for ref in tuple(_restyle_callbacks):
        callback = ref()
        if callback is None:
            _restyle_callbacks.remove(ref)
        elif callback == fn:
            return
    _restyle_callbacks.append(_callback_ref(fn))


def unregister_restyle(fn: Callable[[], None]) -> None:
    """Remove a callback previously registered with :func:`register_restyle`."""
    for ref in tuple(_restyle_callbacks):
        callback = ref()
        if callback is None or callback == fn:
            _restyle_callbacks.remove(ref)


def set_theme(name: str) -> str:
    """Activate ``name`` in place and notify live GUI registrations."""
    normalized = str(name or "").strip().lower()
    if normalized not in PALETTES:
        choices = ", ".join(sorted(PALETTES))
        raise ValueError(f"Unknown theme {name!r}; choose one of: {choices}")
    if normalized == C.name:
        return normalized
    C._replace(normalized, PALETTES[normalized])
    for ref in tuple(_restyle_callbacks):
        callback = ref()
        if callback is None:
            _restyle_callbacks.remove(ref)
            continue
        try:
            callback()
        except Exception:
            # A closing Tk window can disappear between the weak-reference
            # check and callback. One stale window must not block the others.
            continue
    return normalized


def register_widget(widget: Any, role: str) -> Any:
    """Register a persistent raw Tk widget for palette-based restyling."""
    if role not in _WIDGET_ROLES:
        raise ValueError(f"Unknown widget theme role {role!r}")
    _widgets[widget] = role
    _style_widget(widget, role)
    return widget


def register_widget_tree(parent: Any) -> None:
    """Enroll existing raw Tk descendants while leaving ttk to its Style."""
    try:
        widget_class = parent.winfo_class()
    except Exception:
        return
    role = {"Toplevel": "window", "Frame": "window",
            "Text": "text", "Canvas": "canvas",
            "Button": "panel", "Label": "panel"}.get(widget_class)
    if role is not None:
        try:
            register_widget(parent, role)
        except Exception:
            pass
    try:
        children = parent.winfo_children()
    except Exception:
        children = ()
    for child in children:
        register_widget_tree(child)


def _style_widget(widget: Any, role: str) -> None:
    if role == "window":
        widget.configure(bg=C.BG)
    elif role == "panel":
        options = set(widget.keys())
        config = {"bg": C.BTN_BG if widget.winfo_class() == "Button" else C.BG2}
        if "foreground" in options:
            config["fg"] = C.FG
        if "activebackground" in options:
            config["activebackground"] = C.BTN_ACT
        if "activeforeground" in options:
            config["activeforeground"] = C.FG
        widget.configure(**config)
    elif role == "text":
        widget.configure(
            bg=C.BG2,
            fg=C.FG,
            insertbackground=C.FG,
            selectbackground=C.ACCENT,
            selectforeground=C.BG,
        )
    elif role == "canvas":
        widget.configure(bg=C.BG)


def restyle_widgets() -> None:
    """Apply the active palette to all live registered raw Tk widgets."""
    for widget, role in tuple(_widgets.items()):
        try:
            if not hasattr(widget, "winfo_exists") or widget.winfo_exists():
                _style_widget(widget, role)
        except Exception:
            continue


def matplotlib_palette(name: "str | None" = None) -> dict[str, str]:
    """Return a detached palette mapping suitable for Matplotlib code."""
    if name is None:
        return C.as_dict()
    normalized = str(name).strip().lower()
    if normalized not in PALETTES:
        raise ValueError(f"Unknown theme {name!r}")
    return dict(PALETTES[normalized])


def _palette_view(palette: "str | Mapping[str, str] | _Palette | None"):
    if palette is None:
        return C
    if isinstance(palette, str):
        return SimpleNamespace(**matplotlib_palette(palette))
    if isinstance(palette, Mapping):
        return SimpleNamespace(**palette)
    return palette


def style_figure(
    fig: Any,
    *,
    palette: "str | Mapping[str, str] | _Palette | None" = None,
    background: "str | None" = None,
) -> Any:
    """Apply a palette to an existing Matplotlib figure in place.

    Data-series colors are intentionally preserved. The helper styles figure
    and axes surfaces, structural text, ticks, spines, legends, and gridlines.
    """
    p = _palette_view(palette)
    figure_bg = background or p.BG
    axes_bg = background or p.BG2
    fig.patch.set_facecolor(figure_bg)
    for text in getattr(fig, "texts", ()):  # suptitles and fig.text labels
        text.set_color(p.FG)
    for ax in getattr(fig, "axes", ()):
        ax.set_facecolor(axes_bg)
        ax.tick_params(colors=p.FG, which="both")
        ax.xaxis.label.set_color(p.FG)
        ax.yaxis.label.set_color(p.FG)
        ax.title.set_color(p.FG)
        ax.xaxis.get_offset_text().set_color(p.FG)
        ax.yaxis.get_offset_text().set_color(p.FG)
        for spine in ax.spines.values():
            spine.set_color(p.BORDER)
        for gridline in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            gridline.set_color(p.BORDER)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(axes_bg)
            legend.get_frame().set_edgecolor(p.BORDER)
            for text in legend.get_texts():
                text.set_color(p.FG)
    return fig


def restyle_owner_figures(owner: Any) -> None:
    """Restyle and redraw Matplotlib figures currently retained by a GUI."""
    figures: dict[int, Any] = {}
    canvases: list[Any] = []
    for value in vars(owner).values():
        if hasattr(value, "axes") and hasattr(value, "patch"):
            figures[id(value)] = value
        figure = getattr(value, "figure", None)
        if figure is not None and hasattr(value, "draw_idle"):
            figures[id(figure)] = figure
            canvases.append(value)
    for figure in figures.values():
        try:
            style_figure(figure)
        except Exception:
            continue
    for canvas in canvases:
        try:
            widget = canvas.get_tk_widget()
            if widget.winfo_exists() and widget.winfo_ismapped():
                canvas.draw_idle()
        except Exception:
            continue
