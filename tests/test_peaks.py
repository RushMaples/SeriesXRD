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

    _test_prominence_decoupling()
    _test_edge_window_and_min_width()
    print("PEAKS TEST OK")


def _test_edge_window_and_min_width():
    """Fit window excludes out-of-range/edge artefacts; min-FWHM floor flags
    single-bin spikes."""
    from bulkxrd.analysis.peaks import fit_pattern, FLAG_WIDTH_BOUND
    rng = np.random.default_rng(1)
    x = np.linspace(2.0, 22.0, 2000)            # 2θ degrees
    dx = x[1] - x[0]
    real = pseudo_voigt(x, 9.3, 12.0, 0.08, 0.5) + pseudo_voigt(x, 10.7, 14.0, 0.08, 0.5)
    onset = pseudo_voigt(x, 3.2, 30.0, 0.10, 0.5)        # beamstop-onset artefact
    y = onset + real + rng.normal(0, 1.0, x.size)

    full = fit_pattern(x, y, min_snr=4.0, window_factor=2.5, max_chi2=1e9)
    assert any(abs(p["center"] - 3.2) < 0.2 for p in full)   # artefact detected w/o window
    win = fit_pattern(x, y, min_snr=4.0, window_factor=2.5, max_chi2=1e9,
                      fit_min=6.5, fit_max=21.0, edge_bins=5)
    assert all(6.5 <= p["center"] <= 21.0 for p in win)      # window excludes it
    assert sum(any(abs(p["center"] - c) < 0.2 for p in win)
               for c in (9.3, 10.7)) == 2                    # real peaks kept

    # Min-FWHM floor flags a one-bin spike (WIDTH_BOUND) but not the real peaks.
    y2 = real + rng.normal(0, 1.0, x.size)
    k = int(np.argmin(np.abs(x - 15.0))); y2[k] += 60.0
    pk = fit_pattern(x, y2, min_snr=4.0, window_factor=2.5, max_chi2=1e9,
                     min_fwhm_bins=2.0, fit_min=6.5, fit_max=21.0)
    spikes = [p for p in pk if abs(p["center"] - 15.0) < 3 * dx]
    assert spikes and all(p["flag"] & FLAG_WIDTH_BOUND for p in spikes), spikes
    reals = [p for p in pk if abs(p["center"] - 9.3) < 0.2]
    assert reals and all(not (p["flag"] & FLAG_WIDTH_BOUND) for p in reals)


def _test_prominence_decoupling():
    """A peak on the shoulder of a taller one (shallow saddle) has low prominence
    but adequate height: the coupled threshold drops it, a lower decoupled
    prominence keeps it."""
    # idx2 is a local max (8) whose saddle toward the taller idx4 (9) sits at 6,
    # so its prominence is 8-6=2; idx4's prominence is 9.
    x = np.arange(7, dtype=float)
    y = np.array([0, 1, 8, 6, 9, 1, 0], dtype=float)
    coupled = detect_peaks(x, y, min_snr=3.0, sigma=1.0)            # prom thresh 3
    decoupled = detect_peaks(x, y, min_snr=3.0, min_prominence_snr=1.0, sigma=1.0)
    cc = {round(c["center"]) for c in coupled}
    dd = {round(c["center"]) for c in decoupled}
    assert 4 in cc and 2 not in cc, f"coupled should drop the shoulder peak: {cc}"
    assert 2 in dd and 4 in dd, f"decoupled should keep both: {dd}"


if __name__ == "__main__":
    main()
