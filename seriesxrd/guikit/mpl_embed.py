"""Map-aware embedding for Matplotlib figures in Tk widgets.

Tk notebook pages commonly exist at a 1x1 placeholder size until they are
selected. Drawing a Matplotlib canvas before that page is mapped creates a
large backing buffer that Tk clips to the eventual pane. A reload appears to
fix the plot only because the second draw happens after layout.

The helpers here make visibility the lifecycle boundary: the first draw waits
for ``<Map>`` and every later draw is fitted to the allocated widget size.
"""

from __future__ import annotations

from typing import Any, Callable


MIN_CANVAS_SIZE = 20


def make_canvas_responsive(
    canvas: Any,
    root: Any,
    *,
    fixed_height_px: int | None = None,
) -> Any:
    """Fit and draw ``canvas`` only after its Tk widget has a real size.

    ``fixed_height_px`` preserves a tall figure inside a scrolling container;
    otherwise both figure dimensions track the allocated widget.
    """
    widget = canvas.get_tk_widget()
    state: dict[str, Any] = {"after_id": None, "last_size": None}

    def _fit_and_draw() -> bool:
        state["after_id"] = None
        try:
            if not widget.winfo_exists() or not widget.winfo_ismapped():
                return False
            width = int(widget.winfo_width())
            height = int(fixed_height_px or widget.winfo_height())
            if width < MIN_CANVAS_SIZE or height < MIN_CANVAS_SIZE:
                return False
        except Exception:
            return False

        size = (width, height)
        dpi = canvas.figure.get_dpi() or 100
        canvas.figure.set_size_inches(width / dpi, height / dpi, forward=False)
        # An explicit draw here is intentional: at this point mapping and size
        # are known, so the first visible frame cannot be a stale clipped buffer.
        canvas.draw()
        state["last_size"] = size
        return True

    def _schedule(_event: Any = None) -> None:
        try:
            if state["after_id"] is not None:
                root.after_cancel(state["after_id"])
            state["after_id"] = root.after_idle(_fit_and_draw)
        except Exception:
            state["after_id"] = None

    widget.bind("<Map>", _schedule, add="+")
    widget.bind("<Configure>", _schedule, add="+")
    # Covers a canvas attached to an already-mapped parent. If it is still
    # hidden, the persistent <Map> binding will perform the first draw later.
    _schedule()

    # Retain callbacks for Tk and provide a safe redraw entry point to callers
    # that update artists while their notebook page may be hidden.
    canvas._seriesxrd_layout_state = state
    canvas._seriesxrd_schedule_draw = _schedule
    return canvas


def request_canvas_draw(canvas: Any) -> None:
    """Request a fitted draw, deferring it if the canvas is currently hidden."""
    schedule = getattr(canvas, "_seriesxrd_schedule_draw", None)
    if callable(schedule):
        schedule()
    else:
        canvas.draw_idle()


def embed_figure(
    parent: Any,
    fig: Any,
    root: Any,
    *,
    toolbar_factory: Callable[[Any, Any], Any] | None = None,
    fixed_height_px: int | None = None,
) -> Any:
    """Create, pack, and map-safely draw a ``FigureCanvasTkAgg``."""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    canvas = FigureCanvasTkAgg(fig, master=parent)
    widget = canvas.get_tk_widget()
    widget.configure(width=10, height=fixed_height_px or 10)
    if toolbar_factory is not None:
        toolbar_factory(canvas, parent)
    widget.pack(side="top", fill="both", expand=True)
    return make_canvas_responsive(
        canvas,
        root,
        fixed_height_px=fixed_height_px,
    )

