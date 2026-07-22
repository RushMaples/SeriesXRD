"""Shared live ttk styling for all SeriesXRD GUIs."""
from __future__ import annotations

from . import theme


def apply_theme(root, ttk, palette=None) -> None:
    """Apply ``palette`` (or the active palette) to a Tk/ttk root in place."""
    c = palette or theme.C
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        ".", background=c.BG, foreground=c.FG, fieldbackground=c.ENTRY_BG,
        bordercolor=c.BORDER, lightcolor=c.BORDER, darkcolor=c.BORDER,
    )
    style.configure("TFrame", background=c.BG)
    style.configure("TLabel", background=c.BG, foreground=c.FG)
    style.configure("Muted.TLabel", background=c.BG, foreground=c.MUTED)
    style.configure("Ok.TLabel", background=c.BG, foreground=c.ACCENT2)
    style.configure("Warn.TLabel", background=c.BG, foreground=c.WARN)
    style.configure("Accent.TLabel", background=c.BG, foreground=c.ACCENT)
    style.configure(
        "TButton", background=c.BTN_BG, foreground=c.FG,
        bordercolor=c.BORDER, relief="flat", padding=4,
    )
    style.map("TButton", background=[("active", c.BTN_ACT),
                                     ("pressed", c.BTN_ACT)])
    style.configure(
        "Accent.TButton", background=c.ACCENT, foreground=c.BG,
        bordercolor=c.ACCENT, relief="flat", padding=4,
    )
    style.map("Accent.TButton", background=[("active", c.ACCENT2),
                                            ("pressed", c.ACCENT2)])
    style.configure(
        "TEntry", fieldbackground=c.ENTRY_BG, foreground=c.FG,
        insertcolor=c.FG, bordercolor=c.BORDER,
    )
    style.configure(
        "TCombobox", fieldbackground=c.ENTRY_BG, foreground=c.FG,
        background=c.BTN_BG, arrowcolor=c.FG,
    )
    style.map("TCombobox", fieldbackground=[("readonly", c.ENTRY_BG)],
              foreground=[("readonly", c.FG)])
    style.configure("TCheckbutton", background=c.BG, foreground=c.FG,
                    indicatorcolor=c.ACCENT)
    style.map("TCheckbutton", background=[("active", c.BG2)])
    style.configure("TRadiobutton", background=c.BG, foreground=c.FG,
                    indicatorcolor=c.ACCENT)
    style.map("TRadiobutton", background=[("active", c.BG2)])
    style.configure("TScrollbar", background=c.BG2, troughcolor=c.BG,
                    arrowcolor=c.FG)
    style.configure("TNotebook", background=c.BG,
                    tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", background=c.BG2, foreground=c.FG,
                    padding=[10, 4])
    style.map("TNotebook.Tab", background=[("selected", c.BG)],
              foreground=[("selected", c.ACCENT)])
    style.configure("TSeparator", background=c.BORDER)
    style.configure("Horizontal.TProgressbar", background=c.ACCENT,
                    troughcolor=c.BG2, bordercolor=c.BORDER)
    style.configure("Treeview", background=c.BG2, fieldbackground=c.BG2,
                    foreground=c.FG, bordercolor=c.BORDER, relief="flat")
    style.map("Treeview", background=[("selected", c.ACCENT)],
              foreground=[("selected", c.BG)])
    style.configure("Treeview.Heading", background=c.BTN_BG,
                    foreground=c.FG, relief="flat", bordercolor=c.BORDER)
    style.map("Treeview.Heading", background=[("active", c.BTN_ACT)])
    root.configure(bg=c.BG)
    try:
        root.option_add("*TCombobox*Listbox.background", c.ENTRY_BG)
        root.option_add("*TCombobox*Listbox.foreground", c.FG)
        root.option_add("*TCombobox*Listbox.selectBackground", c.ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", c.BG)
    except Exception:
        pass
