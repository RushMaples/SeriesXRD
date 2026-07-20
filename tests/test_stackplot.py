"""Tests for analysis/stackplot.py (stacked-panel pattern figure)."""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("matplotlib")

from seriesxrd.analysis.stackplot import stack_figure


@pytest.fixture()
def analysis_file(tmp_path):
    """Minimal analysis HDF5: 5 frames, 2 pressures + 1 unknown-P frame,
    one saturated frame, a gaussian peak that drifts with pressure."""
    path = tmp_path / "mini_analysis.h5"
    q = np.linspace(0.5, 6.0, 400)
    n = 5
    pressure = np.array([1.0, 1.0, 5.0, 5.0, np.nan])
    clean = np.zeros((n, q.size))
    rng = np.random.default_rng(7)
    for i in range(n):
        p = pressure[i] if np.isfinite(pressure[i]) else 3.0
        c = 2.8 + 0.02 * p
        clean[i] = 100.0 * np.exp(-0.5 * ((q - c) / 0.03) ** 2)
        clean[i] += rng.normal(0, 1.0, q.size)
    clean[1, 200] = 5.0e6          # saturated partner at 1 GPa
    with h5py.File(path, "w") as h:
        h.attrs["unit"] = "q_A^-1"
        h.attrs["wavelength"] = 0.4133
        h["radial"] = q
        g = h.create_group("background")
        g["clean"] = clean
        g["baseline"] = np.zeros_like(clean)
        g["spot_residual"] = np.zeros_like(clean)
        fr = h.create_group("frames")
        fr["filename"] = np.array([f"scan_{i:02d}.tif".encode() for i in range(n)])
        fr["pressure"] = pressure
        fr["excluded"] = np.zeros(n, bool)
    return path


def test_auto_per_pressure(analysis_file, tmp_path):
    out = tmp_path / "stack.png"
    man = stack_figure(analysis_file, out, source="clean")
    assert out.is_file() and out.stat().st_size > 0
    # 2 pressure groups + 1 unknown-P frame = 3 panels
    assert man["n_panels"] == 3
    # the saturated 1-GPa partner (frame 1) must have been vetoed
    assert 1 not in man["frames"]
    assert man["axis"] == "2th_deg"
    assert man["labels"][0] == "1 GPa"


def test_explicit_frames_and_exclude_d(analysis_file, tmp_path):
    out = tmp_path / "stack2.png"
    man = stack_figure(analysis_file, out, source="clean",
                       frames=[3, 0], exclude_d=[2.231])
    assert man["n_panels"] == 2
    # explicit frames are kept (even order re-sorted by pressure: 0 then 3)
    assert man["frames"] == [0, 3]
    assert len(man["excluded_windows"]) == 1
    d0, lo, hi = man["excluded_windows"][0]
    assert lo < 2 * np.pi / d0 < hi


def test_all_saturated_raises(analysis_file, tmp_path):
    with pytest.raises(ValueError):
        stack_figure(analysis_file, tmp_path / "x.png", source="clean",
                     saturation_cutoff=0.5)


def test_waterfall_style(analysis_file, tmp_path):
    out = tmp_path / "wf.png"
    man = stack_figure(analysis_file, out, source="clean", style="waterfall",
                       frames=[0, 2, 3])
    assert out.is_file() and out.stat().st_size > 0
    assert man["style"] == "waterfall"
    assert man["n_panels"] == 3


def test_unknown_style_raises(analysis_file, tmp_path):
    with pytest.raises(ValueError):
        stack_figure(analysis_file, tmp_path / "y.png", source="clean",
                     style="mountain")


def test_export_frames_exclude_d(analysis_file, tmp_path):
    """export_frames zeroes the excluded windows in the written .xy data."""
    from seriesxrd.analysis.refine_export import export_frames
    out = tmp_path / "xy"
    d0 = 2.231   # ~ the synthetic peak position at 1 GPa (q ~ 2.82)
    man = export_frames(analysis_file, out, frames=[0], source="clean",
                        peaks=False, residual_peaks=False, unknowns=False,
                        exclude_d=[d0])
    assert len(man["excluded_windows"]) == 1
    q, y = np.loadtxt(out / "patterns" / "frame_0000_q.xy", unpack=True)
    qc = 2 * np.pi / d0
    win = (q > qc * 0.98) & (q < qc * 1.02)
    assert np.all(y[win] == 0.0)          # window zeroed
    assert np.any(y[~win] != 0.0)         # everything else intact
    # header carries the exclusion flag
    head = (out / "patterns" / "frame_0000_q.xy").read_text().splitlines()[4]
    assert "excluded_d" in head
