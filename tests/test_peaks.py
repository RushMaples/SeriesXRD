"""Step-2 peak/profile fitting correctness (numpy + scipy; no pyFAI/h5py)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seriesxrd.analysis.peaks as peaks_mod
from seriesxrd.analysis.peaks import (
    pseudo_voigt, pseudo_voigt_area, pseudo_voigt_jac, mad_sigma, detect_peaks,
    fit_pattern, fit_dataset, FLAG_OK,
    resolve_sensitivity, winsorize_excess, auto_fit_range, build_fit_source,
    SENSITIVITY_PRESETS, _seed_frame_orders, _predict_seed_centers,
    _fit_ordered_rows)


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
    _test_local_detrend_detection()
    _test_sloped_baseline_fit()
    _test_sensitivity_presets()
    _test_winsorize_and_sources()
    _test_auto_fit_range()
    _test_esd_columns()
    _test_group_size_cap()
    _test_seed_tracking_order()
    _test_pseudo_voigt_jac()
    _test_seeds_dont_cross_scans()
    print("PEAKS TEST OK")


def _test_seeds_dont_cross_scans():
    """The seed_group_by='scan' setting must keep propagation inside a scan:
    the first frame of scan B gets NO seed carried from scan A. With grouping
    off ('none') the same border DOES carry seeds — the behaviour the setting
    exists to prevent. Exercises the exact seam the driver uses (per-scan orders
    from _seed_frame_orders, each an independent _fit_ordered_rows pass)."""
    n_bins = 400
    radial = np.linspace(1.0, 8.0, n_bins)
    # 8 frames, 2 scans (A=frames 0-3, B=frames 4-7). Encode each frame's peak
    # center in its data so the recording stub can report what it was seeded with.
    centers = np.array([10., 11., 12., 13., 20., 21., 22., 23.])
    clean = np.repeat(centers[:, None], n_bins, axis=1)      # row i is constant = center_i
    excluded = np.zeros(8, dtype=bool)
    scan_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])            # what _tracking_groups('scan') yields
    axis = np.arange(8, dtype=float)

    recorded: dict = {}

    def _stub(radial_, y, **kw):                              # stand in for a real fit
        seeds = kw.get("seed_centers")
        c = float(y[0])
        recorded[c] = None if seeds is None else [round(s, 3) for s in seeds]
        return [{"center": c, "amplitude": 100.0, "fwhm": 0.05, "eta": 0.5,
                 "area": 1.0, "chi2": 1.0, "flag": FLAG_OK,
                 "center_err": 0.01, "amplitude_err": 1.0, "fwhm_err": 0.01}]

    def _run(orders):
        recorded.clear()
        for order in orders:
            _fit_ordered_rows(
                radial, clean[order], excluded[order], np.arange(len(order)),
                axis[order], min_snr=5.0, window_factor=3.0, max_chi2=25.0,
                propagate=True, min_prominence_snr=None, edge_bins=0,
                fit_min=None, fit_max=None, min_fwhm_bins=0.0,
                local_baseline_bins=0, seed_max_axis_gap=None,
                seed_axis_predictor=False)
        return dict(recorded)

    orig = peaks_mod.fit_pattern
    peaks_mod.fit_pattern = _stub
    try:
        scan_orders = _seed_frame_orders(8, axis, "frame", scan_ids)
        assert [o.tolist() for o in scan_orders] == [[0, 1, 2, 3], [4, 5, 6, 7]]
        scan_seen = _run(scan_orders)
        none_seen = _run([np.arange(8)])
    finally:
        peaks_mod.fit_pattern = orig

    # scan mode: frame 4 (scan B, center 20) starts a fresh path -> no seed.
    assert scan_seen[20.0] is None, scan_seen[20.0]
    assert scan_seen[10.0] is None                         # scan A also starts fresh
    assert scan_seen[13.0] == [12.0]                       # but propagates WITHIN scan A
    # grouping off: frame 4 inherits scan A's last good center (13) — cross-border.
    assert none_seen[20.0] == [13.0], none_seen[20.0]


def _test_pseudo_voigt_jac():
    """The closed-form Jacobian (used to accelerate the least-squares fit) must
    match central finite differences of :func:`pseudo_voigt` — a wrong sign or
    factor would silently slow convergence or corrupt the covariance esd's."""
    x = np.linspace(1.0, 8.0, 400)
    for c, a, w, e in [(4.0, 300.0, 0.06, 0.4), (2.5, 120.0, 0.03, 0.0),
                       (6.1, 80.0, 0.09, 1.0)]:
        d_c, d_a, d_w, d_e = pseudo_voigt_jac(x, c, a, w, e)
        f = lambda cc, aa, ww, ee: pseudo_voigt(x, cc, aa, ww, ee)
        num = {
            "c": (f(c + 1e-6, a, w, e) - f(c - 1e-6, a, w, e)) / 2e-6,
            "a": (f(c, a + 1e-3, w, e) - f(c, a - 1e-3, w, e)) / 2e-3,
            "w": (f(c, a, w + 1e-7, e) - f(c, a, w - 1e-7, e)) / 2e-7,
            "e": (f(c, a, w, e + 1e-6) - f(c, a, w, e - 1e-6)) / 2e-6,
        }
        for key, ana in (("c", d_c), ("a", d_a), ("w", d_w), ("e", d_e)):
            scale = max(np.abs(num[key]).max(), 1e-9)
            assert np.allclose(ana, num[key], atol=1e-4 * scale, rtol=1e-4), \
                (c, a, w, e, key, np.abs(ana - num[key]).max(), scale)


def _test_group_size_cap():
    """A chain of overlapping candidates must be split at its widest gaps
    instead of forming one unbounded joint fit (which effectively hangs —
    observed on noisy detections with sensitive thresholds)."""
    from seriesxrd.analysis.peaks import _group_peaks, MAX_GROUP_SIZE
    cands = [{"center": 1.0 + 0.05 * i, "amplitude": 10.0, "fwhm": 0.05}
             for i in range(40)]                        # windows all chain
    groups = _group_peaks(cands, window_factor=3.0)
    assert sum(len(g) for g in groups) == 40            # nothing lost
    assert max(len(g) for g in groups) <= MAX_GROUP_SIZE
    flat = [c["center"] for g in groups for c in g]
    assert flat == sorted(flat)                         # order preserved
    # Small groups pass through untouched.
    g2 = _group_peaks(cands[:3], window_factor=3.0)
    assert len(g2) == 1 and len(g2[0]) == 3
    # Splits prefer the widest internal gap.
    uneven = ([{"center": 1.0 + 0.02 * i, "amplitude": 1, "fwhm": 0.05} for i in range(8)]
              + [{"center": 1.5 + 0.02 * i, "amplitude": 1, "fwhm": 0.05} for i in range(8)])
    gs = _group_peaks(uneven, window_factor=3.0, max_group_size=10)
    assert len(gs) == 2 and len(gs[0]) == 8 and len(gs[1]) == 8


def _test_seed_tracking_order():
    """Peak seed propagation follows independent scan paths and pressure order."""
    axis = np.array([2.0, 1.0, 2.0, 1.0, np.nan])
    groups = np.array([0, 0, 1, 1, 1])
    pressure_orders = [o.tolist() for o in _seed_frame_orders(5, axis, "pressure", groups)]
    assert pressure_orders == [[1, 0], [3, 2], [4]], pressure_orders
    frame_orders = [o.tolist() for o in _seed_frame_orders(5, axis, "frame", groups)]
    assert frame_orders == [[0, 1], [2, 3, 4]], frame_orders

    pred = _predict_seed_centers((2.0, [2.8, 3.8]), (1.0, [2.9, 3.9]), 3.0, True)
    assert np.allclose(pred, [2.7, 3.7])


def _test_esd_columns():
    """1σ fit uncertainties from the covariance: present, positive, and sane
    (the center is pinned far better than a FWHM on a clean strong peak)."""
    rng = np.random.default_rng(3)
    x = np.linspace(1.0, 8.0, 1600)
    y = pseudo_voigt(x, 4.0, 300.0, 0.06, 0.4) + rng.normal(0.0, 2.0, x.size)
    pk = [p for p in fit_pattern(x, y, min_snr=5.0) if p["flag"] == FLAG_OK]
    assert len(pk) == 1, len(pk)
    p = pk[0]
    for k in ("center_err", "amplitude_err", "fwhm_err"):
        assert k in p and np.isfinite(p[k]) and p[k] > 0, (k, p.get(k))
    assert p["center_err"] < 0.5 * p["fwhm"]              # well-localised
    assert abs(p["center"] - 4.0) < 5 * p["center_err"] + 1e-3
    # A weaker, noisier peak carries a LARGER center esd than a strong one.
    y2 = pseudo_voigt(x, 4.0, 25.0, 0.06, 0.4) + rng.normal(0.0, 2.0, x.size)
    pk2 = [p2 for p2 in fit_pattern(x, y2, min_snr=4.0) if p2["flag"] == FLAG_OK]
    if pk2:                                               # may drown at this SNR
        assert pk2[0]["center_err"] > p["center_err"]


def _test_sensitivity_presets():
    """A preset fills unset knobs; an explicit value overrides it; an unknown
    name falls back to 'normal'."""
    n = resolve_sensitivity("normal")
    assert (n["min_snr"], n["min_prominence_snr"], n["min_fwhm_bins"], n["edge_bins"]) \
        == (5.0, 2.0, 2.0, 5), n
    # sensitive is looser than conservative on every knob.
    s, c = resolve_sensitivity("sensitive"), resolve_sensitivity("conservative")
    assert s["min_snr"] < c["min_snr"] and s["min_prominence_snr"] < c["min_prominence_snr"]
    # explicit overrides win; others still come from the preset.
    o = resolve_sensitivity("normal", min_snr=7.5, edge_bins=0)
    assert o["min_snr"] == 7.5 and o["edge_bins"] == 0 and o["min_prominence_snr"] == 2.0
    assert resolve_sensitivity("bogus")["preset"] == "normal"
    assert set(SENSITIVITY_PRESETS) == {"conservative", "normal", "sensitive"}


def _test_winsorize_and_sources():
    """Hybrid keeps a broad azimuthally-sparse real peak but drops a narrow
    diamond spike; build_fit_source composes the right channel and 'auto' prefers
    sigmaclip when present."""
    q = np.linspace(0.0, 6.0, 1500)
    clean = pseudo_voigt(q, 2.5, 100.0, 0.05, 0.4)                 # already in the median
    texture = pseudo_voigt(q, 4.0, 30.0, 0.08, 0.5)               # broad real mean-excess
    spike = np.zeros_like(q); spike[np.argmin(np.abs(q - 5.0))] += 4000.0  # diamond spot
    spot_residual = texture + spike

    add = winsorize_excess(spot_residual, spike_bins=5)
    ktex, ksp = np.argmin(np.abs(q - 4.0)), np.argmin(np.abs(q - 5.0))
    assert add[ktex] > 0.8 * texture[ktex], (add[ktex], texture[ktex])   # texture core kept
    assert add[ksp] < 1.0, add[ksp]                                       # spike removed

    sigres = texture                                              # principled trimmed-mean excess
    # mean keeps the diamond; hybrid removes it; both keep the real texture.
    mean_src, sm = build_fit_source("mean", clean, spot_residual=spot_residual)
    hyb_src, sh = build_fit_source("hybrid", clean, spot_residual=spot_residual)
    auto_src, sa = build_fit_source("auto", clean, spot_residual=spot_residual,
                                    sigmaclip_residual=sigres)
    assert (sm, sh, sa) == ("mean", "hybrid", "sigmaclip")
    assert mean_src[ksp] > 1000 and abs(hyb_src[ksp] - clean[ksp]) < 50
    assert hyb_src[ktex] > clean[ktex] + 0.8 * texture[ktex]
    # auto falls back to hybrid with no sigmaclip channel.
    _, sb = build_fit_source("auto", clean, spot_residual=spot_residual)
    assert sb == "hybrid"
    try:
        build_fit_source("sigmaclip", clean, spot_residual=spot_residual)
    except ValueError:
        pass
    else:
        raise AssertionError("sigmaclip source without its channel should raise")
    # spots = the spot_residual channel itself (single-crystal sample mode):
    # keeps the narrow spike the hybrid source rejects, ignores clean entirely.
    spots_src, ss = build_fit_source("spots", clean, spot_residual=spot_residual)
    assert ss == "spots"
    assert np.allclose(spots_src, spot_residual, equal_nan=True)
    assert spots_src[ksp] > 1000                       # crystal spike survives
    assert abs(spots_src[np.argmin(np.abs(q - 2.5))]) < 1.0   # powder peak absent
    try:
        build_fit_source("spots", clean)
    except ValueError:
        pass
    else:
        raise AssertionError("spots source without spot_residual should raise")


def _test_auto_fit_range():
    """Conservative range inference trims the beamstop ramp and the dead tail,
    never an interior peak."""
    x = np.linspace(0.2, 6.0, 1400)
    y = 60.0 / np.maximum(x - 0.1, 0.05)            # steep beamstop falloff at low q
    y[x < 0.4] = np.nan                             # masked beamstop
    y = np.where(x < 2.4, y, 0.0)
    y = (y + pseudo_voigt(x, 3.0, 200.0, 0.05, 0.5)
         + pseudo_voigt(x, 4.2, 150.0, 0.05, 0.5)
         + pseudo_voigt(x, 5.3, 120.0, 0.05, 0.5))
    y = y + np.random.default_rng(0).normal(0.0, 0.3, x.size)  # noisy throughout + dead tail >5.4
    lo, hi = auto_fit_range(x, y)
    assert lo is not None and 0.4 <= lo < 3.0, lo   # past beamstop, before first peak
    assert hi is not None and 5.3 < hi < 5.8, hi    # trims dead tail, stops at the last peak
    # All three interior peaks survive the inferred window.
    assert lo < 3.0 and hi > 5.3
    # A pattern that rises into both ends needs no trimming → (None, None).
    xs = np.linspace(1.0, 8.0, 1000)
    ys = 1.0 + (xs - 1.0)                            # gently rising, no ramp/dead tail
    assert auto_fit_range(xs, ys) == (None, None)


def _test_sloped_baseline_fit():
    """A peak on a strongly sloped local background fits cleanly with the local
    LINEAR baseline (a constant baseline would leave a sloped residual → high
    chi-square → bad_chi2 rejection)."""
    from seriesxrd.analysis.peaks import fit_pattern
    x = np.linspace(5, 12, 1400)
    y = (8.0 - 0.6 * (x - 5)) + pseudo_voigt(x, 9.0, 10.0, 0.06, 0.5) \
        + np.random.default_rng(0).normal(0, 0.3, x.size)
    pk = [p for p in fit_pattern(x, y, min_snr=4.0, window_factor=3.0,
                                 max_chi2=15.0, local_baseline_bins=81)
          if abs(p["center"] - 9.0) < 0.2]
    assert pk, "peak on a slope not found"
    p = pk[0]
    assert p["flag"] == 0 and p["chi2"] < 5.0, (p["flag"], p["chi2"])
    assert abs(p["amplitude"] - 10.0) < 1.5 and abs(p["fwhm"] - 0.06) < 0.02


def _test_local_detrend_detection():
    """When `clean` isn't background-flat, the global MAD σ is inflated and small
    real peaks fall under the height threshold. Local detrending fixes it."""
    from seriesxrd.analysis.peaks import fit_pattern
    x = np.linspace(0, 24, 1500)
    broad = (16 * np.exp(-0.5 * ((x - 3.5) / 2.0) ** 2)
             + 6 * np.exp(-0.5 * ((x - 15.5) / 3.0) ** 2))
    broad[x < 2] = 0.0
    real = (pseudo_voigt(x, 9.3, 3.5, 0.10, 0.5)
            + pseudo_voigt(x, 10.7, 4.0, 0.08, 0.5))   # small, on top of `broad`
    clean = np.clip(broad + real
                    + np.random.default_rng(0).normal(0, 0.6, x.size), 0, None)
    found = lambda peaks, c: any(abs(p["center"] - c) < 0.2 for p in peaks)

    off = fit_pattern(x, clean, min_snr=4.0, local_baseline_bins=0)
    on = fit_pattern(x, clean, min_snr=4.0, local_baseline_bins=81)
    # Without detrend the inflated σ hides the real peaks; with it, both appear.
    assert not (found(off, 9.3) and found(off, 10.7)), "expected misses without detrend"
    assert found(on, 9.3) and found(on, 10.7), "detrend should recover the real peaks"


def _test_edge_window_and_min_width():
    """Fit window excludes out-of-range/edge artefacts; min-FWHM floor flags
    single-bin spikes."""
    from seriesxrd.analysis.peaks import fit_pattern, FLAG_WIDTH_BOUND
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


def test_main():
    """Pytest entry point — this file predates the test_* function convention."""
    main()


if __name__ == "__main__":
    main()
