"""Shared lightweight Tk tooltip used by every bulkxrd stage GUI.

Schedules a hover popup on <Enter>, cancels/destroys it on <Leave> or any
button press. No dependency on the App classes, so it lives here and is
imported by calib/gui.py and reduce/gui.py instead of being copy-pasted.
"""
from __future__ import annotations


class ToolTip:
    """Lightweight Tk tooltip: schedule on <Enter>, destroy on <Leave>/<ButtonPress>."""
    def __init__(self, widget, text: str, delay_ms: int = 500):
        self._widget = widget
        self._text = text
        self._delay = delay_ms
        self._id = None
        self._tip_win = None
        widget.bind("<Enter>",       self._on_enter, add="+")
        widget.bind("<Leave>",       self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None):
        if self._id is not None:
            return
        self._id = self._widget.after(self._delay, self._show)

    def _on_leave(self, _event=None):
        if self._id is not None:
            try:
                self._widget.after_cancel(self._id)
            except Exception:
                pass
            self._id = None
        self._destroy()

    def _show(self):
        self._id = None
        if self._tip_win or not self._text:
            return
        try:
            import tkinter as _tk
            x = self._widget.winfo_rootx() + 24
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
            tw = _tk.Toplevel(self._widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            lbl = _tk.Label(tw, text=self._text, justify="left",
                            background="#ffffe0", relief="solid", borderwidth=1,
                            wraplength=340, padx=4, pady=2)
            lbl.pack()
            self._tip_win = tw
        except Exception:
            pass

    def _destroy(self):
        if self._tip_win:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None
