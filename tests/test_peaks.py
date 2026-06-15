"""Step-2 peak/profile fitting correctness (numpy + scipy; no pyFAI/h5py)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.analysis.peaks import (
    pseudo_voigt, pseudo_voigt_area, mad_sigma, detect_peaks,
    fit_pattern, fit_dataset, FLAG_OK)


def main() -> None:
    rng = np.random.default_rng(0)
    q = np.linspace(1.0, 8.0, 2000)

    # Ground-truth peaks: (center, amplitude, fwhm, eta).
    truth = [(2.50, 400.0, 0.040, 0.3),
             (3.62, 260.0, 0.050, 0.6),
             (5.10, 520.0, 0.035, 0.0),   # pure gaussian
             (6.40, 180.0, 0.060, 1.0)]   # pure lorentzian
    clean = np.zeros_like(q)
    for c, a, w, e in truth:
        clean += pseudo_voigt(q, c, a, w, e)
    noisy = clean + rng.normal(0.0, 2.0, q.size)

    # analytic area sanity: numeric integral matches closed form within 1%.
    a_num = np.trapezoid(pseudo_voigt(q, 5.10, 520.0, 0.035, 0.0), q)
    a_ana = float(pseudo_voigt_area(520.0, 0.035, 0.0))
    assert abs(a_num - a_ana) / a_ana < 0.01, (a_num, a_ana)

    # MAD noise floor recovers the injected sigma reasonably.
    sig = mad_sigma(noisy - clean)
    assert 1.0 < sig < 3.5, sig

    # detection finds exactly the four peaks.
    cands = detect_peaks(q, noisy, min_snr=5.0)
    assert len(cands) == 4, [round(c["center"], 2) for c in cands]

    # full fit recovers centers, widths and areas.
    peaks = fit_pattern(q, noisy, min_snr=5.0)
    assert len(peaks) == 4, len(peaks)
    for (c, a, w, e), p in zip(truth, peaks):
        assert p["flag"] == FLAG_OK, (c, p["flag"])
        assert abs(p["center"] - c) < 0.01, (c, p["center"])
        assert abs(p["fwhm"] - w) < 0.2 * w, (c, w, p["fwhm"])
        a_true = float(pseudo_voigt_area(a, w, e))
        assert abs(p["area"] - a_true) < 0.08 * a_true, (c, a_true, p["area"])

    # overlapping doublet fit jointly: two close peaks resolved.
    q2 = np.linspace(1.0, 4.0, 1500)
    over = pseudo_voigt(q2, 2.50, 300.0, 0.05, 0.4) + pseudo_voigt(q2, 2.62, 300.0, 0.05, 0.4)
    over += rng.normal(0.0, 1.5, q2.size)
    pk = fit_pattern(q2, over, min_snr=5.0)
    cen = sorted(p["center"] for p in pk)
    assert len(pk) == 2, [round(c, 3) for c in cen]
    assert abs(cen[0] - 2.50) < 0.02 and abs(cen[1] - 2.62) < 0.02, cen

    # dataset driver with seed propagation across a drifting series.
    series = []
    for k in range(6):
        shift = 0.01 * k                       # lattice compresses, peaks drift
        y = np.zeros_like(q)
        for c, a, w, e in truth:
            y += pseudo_voigt(q, c - shift, a, w, e)
        series.append(y + rng.normal(0.0, 2.0, q.size))
    res = fit_dataset(q, np.array(series), min_snr=5.0, propagate_seeds=True)
    assert len(res) == 6 and all(len(r) == 4 for r in res), [len(r) for r in res]
    # the first reflection tracks its drift downward across frames.
    first = [sorted(p["center"] for p in r)[0] for r in res]
    assert first[0] > first[-1], first
    assert abs((first[0] - first[-1]) - 0.05) < 0.02, first

    print("PEAKS TEST OK")


if __name__ == "__main__":
    main()
