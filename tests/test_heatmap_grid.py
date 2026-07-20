"""Scan-grid mapping + series-axis resolution (heatmap.py)."""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from seriesxrd.analysis.heatmap import (
    frame_grid, grid_map, series_axis, frame_values, pattern_image,
)


def test_frame_grid_horizontal():
    # 7 frames, 3 per row, unidirectional: rows read left->right.
    g = frame_grid(7, n_cols=3, serpentine=False)
    assert g.shape == (3, 3)
    assert g.tolist() == [[0, 1, 2], [3, 4, 5], [6, -1, -1]]
    # Boustrophedon: every second row reversed (incl. the padded one — the
    # stage turned around, so the last frames sit at the far end).
    g = frame_grid(7, n_cols=3, serpentine=True)
    assert g.tolist() == [[0, 1, 2], [5, 4, 3], [6, -1, -1]]
    # Giving n_rows instead derives the line length.
    g2 = frame_grid(6, n_rows=2, order="horizontal", serpentine=False)
    assert g2.shape == (2, 3) and g2[1].tolist() == [3, 4, 5]


def test_frame_grid_vertical():
    # Scan lines are columns of length 3; frame 3 starts the second column.
    g = frame_grid(6, n_rows=3, order="vertical", serpentine=False)
    assert g.shape == (3, 2)
    assert g[:, 0].tolist() == [0, 1, 2]
    assert g[:, 1].tolist() == [3, 4, 5]
    g = frame_grid(6, n_rows=3, order="vertical", serpentine=True)
    assert g[:, 1].tolist() == [5, 4, 3]


def test_grid_map_values():
    vals = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    m = grid_map(vals, n_cols=2, serpentine=True)
    assert m.shape == (3, 2)
    assert m[0].tolist() == [10.0, 11.0]
    assert m[1].tolist() == [13.0, 12.0]          # reversed line
    assert m[2, 0] == 14.0 and np.isnan(m[2, 1])  # padding is NaN


def _series_file(path, n=4):
    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.create_dataset("radial", data=np.linspace(1.0, 8.0, 16))
        h.create_group("background").create_dataset(
            "clean", data=np.arange(n * 16, dtype="f8").reshape(n, 16))
        gf = h.create_group("frames")
        gf.create_dataset("pressure", data=np.array([0.0, 1.0, 2.0, 3.0]))
        gf.create_dataset("temperature", data=np.array([300.0, 350.0, 400.0, 450.0]))
        ts = ["2026-01-01T00:00:00", "2026-01-01T00:01:00",
              "2026-01-01T00:02:00", "bad-stamp"]
        gf.create_dataset("timestamp", data=np.array(ts, dtype=object),
                          dtype=h5py.string_dtype())
        gf.create_dataset("contamination", data=np.array([5.0, 6.0, 7.0, 8.0]))


def test_series_axis_kinds():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _series_file(p)
        sx = series_axis(p, "frame")
        assert sx["ok"] and sx["x"].tolist() == [0.0, 1.0, 2.0, 3.0]
        sx = series_axis(p, "pressure")
        assert sx["ok"] and sx["label"] == "pressure (GPa)"
        sx = series_axis(p, "temperature")
        assert sx["ok"] and sx["x"][1] == 350.0
        sx = series_axis(p, "time")
        assert sx["ok"] and sx["x"][2] == 120.0 and np.isnan(sx["x"][3])
        sx = series_axis(p, "nonsense")
        assert not sx["ok"] and "Unknown series axis" in sx["error"]


def test_pattern_image_series_axes():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _series_file(p)
        img = pattern_image(p, x_axis="temperature")
        assert img["ok"] and img["x_label"] == "temperature (K)"
        assert img["x"].tolist() == [300.0, 350.0, 400.0, 450.0]
        img = pattern_image(p, x_axis="time")
        assert img["ok"] and img["x_label"] == "elapsed time (s)"


def test_frame_values():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _series_file(p)
        fv = frame_values(p, "total")
        assert fv["ok"] and fv["values"].size == 4
        # frame 1 has larger clean values than frame 0 (arange fixture)
        assert fv["values"][1] > fv["values"][0]
        # ROI restriction changes the sum
        roi = frame_values(p, "total", radial_min=4.0)
        assert roi["ok"] and roi["values"][0] < fv["values"][0]
        assert "ROI" in roi["label"]
        fv = frame_values(p, "contamination")
        assert fv["ok"] and fv["values"].tolist() == [5.0, 6.0, 7.0, 8.0]
        fv = frame_values(p, "pressure")
        assert fv["ok"] and fv["values"][3] == 3.0
        fv = frame_values(p, "n_peaks")
        assert not fv["ok"]              # no /peaks in fixture
        fv = frame_values(p, "bogus")
        assert not fv["ok"] and "Unknown value" in fv["error"]


def main() -> None:
    test_frame_grid_horizontal()
    test_frame_grid_vertical()
    test_grid_map_values()
    test_series_axis_kinds()
    test_pattern_image_series_axes()
    test_frame_values()
    print("HEATMAP GRID TEST OK")


if __name__ == "__main__":
    main()
