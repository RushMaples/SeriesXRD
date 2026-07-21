"""Shared lightweight Tk tooltip used by every seriesxrd stage GUI.

Shows on mouse hover after a short delay, and — for keyboard users — on
focus (WCAG keyboard access: the same help must be reachable without a
pointer). Dismissed on leave/blur, any button press, or Escape. No
dependency on the App classes, so it lives here and is imported by the
stage GUIs instead of being copy-pasted.
"""
from __future__ import annotations


class ToolTip:
    """Lightweight Tk tooltip.

    Mouse: schedule on <Enter>, destroy on <Leave>/<ButtonPress>.
    Keyboard: show on <FocusIn>, destroy on <FocusOut>/<Escape>.
    """
    def __init__(self, widget, text: str, delay_ms: int = 500):
        self._widget = widget
        self._text = text
        self._delay = delay_ms
        self._id = None
        self._tip_win = None
        widget.bind("<Enter>",       self._on_enter, add="+")
        widget.bind("<Leave>",       self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")
        widget.bind("<FocusIn>",     self._on_focus_in, add="+")
        widget.bind("<FocusOut>",    self._on_leave, add="+")
        widget.bind("<Escape>",      self._on_leave, add="+")

    def _on_enter(self, _event=None):
        if self._id is not None:
            return
        self._id = self._widget.after(self._delay, self._show)

    def _on_focus_in(self, _event=None):
        # Keyboard focus shows the tip on the same delay as hover; a widget
        # that is focused AND hovered must not double-schedule.
        self._on_enter()

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
                            background="#ffffe0", foreground="#1a1a1a",
                            relief="solid", borderwidth=1,
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
