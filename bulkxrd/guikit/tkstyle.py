"""Shared ttk styling for all bulkxrd GUIs (dark Catppuccin, see theme.py).

calib/gui.py predates this module and still carries its own copy of the same
style block; new GUIs should call ``apply_dark_theme`` instead so the look
stays consistent from one place.
"""
from __future__ import annotations

from .theme import BG, BG2, FG, ACCENT, BORDER, ENTRY_BG, BTN_BG, BTN_ACT


def apply_dark_theme(root, ttk) -> None:
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".",             background=BG,       foreground=FG,      fieldbackground=ENTRY_BG,
                     bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.configure("TFrame",        background=BG)
    style.configure("TLabel",        background=BG,       foreground=FG)
    style.configure("TButton",       background=BTN_BG,   foreground=FG,      bordercolor=BORDER, relief="flat", padding=4)
    style.map(       "TButton",      background=[("active", BTN_ACT), ("pressed", BTN_ACT)])
    style.configure("TEntry",        fieldbackground=ENTRY_BG, foreground=FG,  insertcolor=FG, bordercolor=BORDER)
    style.configure("TCombobox",     fieldbackground=ENTRY_BG, foreground=FG,  background=BTN_BG, arrowcolor=FG)
    style.map(       "TCombobox",    fieldbackground=[("readonly", ENTRY_BG)])
    style.configure("TCheckbutton",  background=BG,       foreground=FG,      indicatorcolor=ACCENT)
    style.map(       "TCheckbutton", background=[("active", BG2)])
    style.configure("TRadiobutton",  background=BG,       foreground=FG,      indicatorcolor=ACCENT)
    style.map(       "TRadiobutton", background=[("active", BG2)])
    style.configure("TScrollbar",    background=BG2,      troughcolor=BG,     arrowcolor=FG)
    style.configure("TNotebook",     background=BG,       tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", background=BG2,      foreground=FG,      padding=[10, 4])
    style.map(       "TNotebook.Tab",background=[("selected", BG)], foreground=[("selected", ACCENT)])
    style.configure("TSeparator",    background=BORDER)
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BG2, bordercolor=BORDER)
    root.configure(bg=BG)
    try:
        root.option_add("*TCombobox*Listbox.background", ENTRY_BG)
        root.option_add("*TCombobox*Listbox.foreground", FG)
        root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    except Exception:
        pass
