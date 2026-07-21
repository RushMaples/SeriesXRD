"""Step-1 background separation correctness (numpy-only; no pyFAI/h5py needed)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seriesxrd.analysis.background import (
    snip_baseline, spot_residual, contamination_score, separate_background)


def _gauss(x, c, a, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def main() -> None:
    q = np.linspace(1, 8, 1500)
    true_bg = 80 + 40 * np.exp(-(q - 1) / 6.0) + _gauss(q, 4.5, 60, 2.0)  # smooth bg + broad hump
    peaks = (_gauss(q, 2.5, 400, 0.012) + _gauss(q, 3.6, 260, 0.013)
             + _gauss(q, 5.1, 520, 0.011) + _gauss(q, 6.4, 180, 0.013))
    robust = true_bg + peaks
    mean = robust + _gauss(q, 4.0, 3000, 0.012)  # diamond single-crystal spike, MEAN only

    # SNIP: under the data, tracks smooth bg, preserves sharp peaks.
    base = snip_baseline(robust, max_half_window=60)
    assert base.shape == robust.shape and np.all(base <= robust + 1e-6)
    clean = robust - base
    interior = (np.arange(q.size) > 30) & (np.arange(q.size) < q.size - 30)
    off = (peaks < 1) & interior
    assert np.median(np.abs(base[off] - true_bg[off]) / true_bg[off]) < 0.05
    for c, a in [(2.5, 400), (3.6, 260), (5.1, 520), (6.4, 180)]:
        assert clean[np.argmin(np.abs(q - c))] > 0.88 * a, f"peak {c} eroded"
    assert np.median(np.abs(clean[off])) < 0.05 * np.median(true_bg[off])

    # NaN regions preserved, rest finite.
    rn = robust.copy(); rn[:20] = np.nan
    b2 = snip_baseline(rn, max_half_window=60)
    assert np.all(np.isnan(b2[:20])) and np.all(np.isfinite(b2[20:]))

    # Spot residual is exact; ~0 away from the spike; contamination > 0.
    res = spot_residual(mean, robust)
    assert res[np.argmin(np.abs(q - 4.0))] > 2000
    assert np.max(np.abs(res[np.abs(q - 4.0) > 0.3])) < 1.0
    assert contamination_score(res) > 0

    out = separate_background(mean, robust, max_half_window=60)
    assert set(out) == {"spot_residual", "baseline", "clean", "contamination"}
    print("BACKGROUND TEST OK")


def test_main():
    """Pytest entry point — this file predates the test_* function convention."""
    main()


if __name__ == "__main__":
    main()
