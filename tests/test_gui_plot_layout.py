"""Regression tests for first-render sizing of embedded Matplotlib plots."""

from __future__ import annotations

import time

import pytest


def _pump_events(root, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)


def test_plot_fits_when_notebook_tab_first_becomes_visible(monkeypatch):
    """A plot created on a hidden tab must fit on its first visible draw."""
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        draw_states = []
        original_draw = FigureCanvasTkAgg.draw

        def recording_draw(canvas):
            widget = canvas.get_tk_widget()
            draw_states.append(
                {
                    "mapped": bool(widget.winfo_ismapped()),
                    "widget": (widget.winfo_width(), widget.winfo_height()),
                    "figure": (
                        round(canvas.figure.get_figwidth() * canvas.figure.get_dpi()),
                        round(canvas.figure.get_figheight() * canvas.figure.get_dpi()),
                    ),
                }
            )
            return original_draw(canvas)

        monkeypatch.setattr(FigureCanvasTkAgg, "draw", recording_draw)

        root.geometry("900x650")
        for stage in ("analysis", "reduction", "shared"):
            draw_states.clear()
            notebook = ttk.Notebook(root)
            first_tab = ttk.Frame(notebook)
            plot_tab = ttk.Frame(notebook)
            notebook.add(first_tab, text="Run")
            notebook.add(plot_tab, text="Plot")
            notebook.pack(fill="both", expand=True)
            notebook.select(first_tab)
            root.update()

            fig = Figure(figsize=(12, 9), dpi=100, layout="constrained")
            ax = fig.add_subplot(111)
            ax.plot([0, 1, 2], [0, 1, 0])
            ax.set_xlabel("A deliberately long radial-axis label for crop detection")
            ax.set_ylabel("Intensity")
            ax.set_title("First render")
            if stage == "analysis":
                from seriesxrd.analysis.gui import AnalysisApp

                app = AnalysisApp.__new__(AnalysisApp)
                app.root = root
                app.tk = tkinter
                app._add_nav_toolbar = lambda *_args, **_kwargs: None
                canvas = AnalysisApp._embed_figure(app, plot_tab, fig, toolbar=False)
            elif stage == "reduction":
                from seriesxrd.reduce.gui import ReductionApp

                app = ReductionApp.__new__(ReductionApp)
                app.root = root
                app.tk = tkinter
                app.ttk = ttk
                app.review_plot_frame = plot_tab
                app._add_review_toolbar = lambda *_args, **_kwargs: None
                ReductionApp._embed_review_figure(app, fig, scroll=False)
                canvas = app._review_canvas
            else:
                from seriesxrd.guikit.mpl_embed import embed_figure, request_canvas_draw

                canvas = embed_figure(plot_tab, fig, root)
                request_canvas_draw(canvas)

            # A draw against an unmapped 10 px placeholder is the cropped first
            # render seen in the GUI. Rendering must wait for the tab's real size.
            _pump_events(root, 0.2)
            assert draw_states == [], stage

            notebook.select(plot_tab)
            _pump_events(root, 0.15)

            assert draw_states, stage
            assert draw_states[0]["mapped"] is True, stage
            assert draw_states[0]["widget"] == draw_states[0]["figure"], stage

            widget = canvas.get_tk_widget()
            canvas.draw()
            figure_width = round(fig.get_figwidth() * fig.get_dpi())
            figure_height = round(fig.get_figheight() * fig.get_dpi())
            assert abs(figure_width - widget.winfo_width()) <= 2, stage
            assert abs(figure_height - widget.winfo_height()) <= 2, stage

            renderer = canvas.get_renderer()
            tight = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
            assert tight.x0 >= -1, stage
            assert tight.y0 >= -1, stage
            assert tight.x1 <= fig.bbox.width + 1, stage
            assert tight.y1 <= fig.bbox.height + 1, stage
            notebook.destroy()
            root.update()
    finally:
        root.destroy()
