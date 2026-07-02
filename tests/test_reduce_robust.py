"""Reduce-stage robust-channel estimator + npt resolver (no pyFAI needed).

The robust channel must prefer a narrow azimuthal quantile-band mean over the
pure median: a median of integer photon counts is quantized (staircase-looking
patterns at low intensity, inherited by clean = robust − baseline). Stub
integrators emulate the three pyFAI generations the fallback chain spans.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bulkxrd.reduce.processing import _robust_integrate, _resolve_npt_1d


class _Res:
    def __init__(self, tag):
        self.intensity = np.zeros(4)
        self.tag = tag


class _ModernAI:
    """pyFAI with medfilt1d_ng(quant_min, quant_max)."""
    def medfilt1d_ng(self, image, npt, mask=None, unit=None,
                     quant_min=0.5, quant_max=0.5):
        r = _Res("ng"); r.quant = (quant_min, quant_max); return r

    def medfilt1d(self, image, npt, mask=None, unit=None, percentile=50):
        raise AssertionError("modern path must use medfilt1d_ng")


class _QuantileTupleAI:
    """pyFAI whose medfilt1d_ng takes a quantile=(lo, hi) tuple instead."""
    def medfilt1d_ng(self, image, npt, mask=None, unit=None, quantile=0.5):
        r = _Res("ng_tuple"); r.quant = quantile; return r


class _SwallowKwargsAI:
    """The observed real-world failure: medfilt1d_ng LOGS AND IGNORES unknown
    kwargs ('Got unknown argument quant_min ...') instead of raising TypeError —
    a naive try/except chain silently returns a pure median. The signature-based
    dispatch must skip _ng and use the legacy percentile-tuple path."""
    def medfilt1d_ng(self, image, npt, mask=None, unit=None, **kwargs):
        assert not any(k.startswith("quant") for k in kwargs), \
            "quant kwargs must not be passed to a **kwargs-only _ng"
        return _Res("ng_median")                 # silently a pure median

    def medfilt1d(self, image, npt, mask=None, unit=None, percentile=50):
        if isinstance(percentile, tuple):
            r = _Res("legacy_band"); r.percentile = percentile; return r
        return _Res("legacy_median")


class _LegacyBandAI:
    """Older pyFAI: no _ng, but medfilt1d accepts a percentile tuple."""
    def medfilt1d(self, image, npt, mask=None, unit=None, percentile=50):
        if isinstance(percentile, tuple):
            r = _Res("legacy_band"); r.percentile = percentile; return r
        return _Res("legacy_median")


class _MedianOnlyAI:
    """Oldest: medfilt1d rejects tuple percentiles entirely."""
    def medfilt1d(self, image, npt, mask=None, unit=None, percentile=50):
        if isinstance(percentile, tuple):
            raise TypeError("percentile must be a scalar")
        return _Res("median_only")


def test_quantile_band_preferred():
    res, est = _robust_integrate(_ModernAI(), None, 100, None, "q_A^-1", 0.05)
    assert res.tag == "ng" and res.quant == (0.45, 0.55)
    assert est.startswith("quantile_band")

    res, est = _robust_integrate(_QuantileTupleAI(), None, 100, None, "q_A^-1", 0.05)
    assert res.tag == "ng_tuple" and res.quant == (0.45, 0.55)
    assert est.startswith("quantile_band")

    # kwargs-swallowing pyFAI: never a silent median — route to percentile band.
    res, est = _robust_integrate(_SwallowKwargsAI(), None, 100, None, "q_A^-1", 0.05)
    assert res.tag == "legacy_band" and res.percentile == (45.0, 55.0)
    assert est.startswith("percentile_band")

    res, est = _robust_integrate(_LegacyBandAI(), None, 100, None, "q_A^-1", 0.05)
    assert res.tag == "legacy_band" and res.percentile == (45.0, 55.0)
    assert est.startswith("percentile_band")

    # Truly unsupported: fall back to the median but SAY SO in the estimator
    # (reduce_dataset warns on this string).
    res, est = _robust_integrate(_MedianOnlyAI(), None, 100, None, "q_A^-1", 0.05)
    assert res.tag == "median_only" and est == "median(band_unsupported)"


def test_pure_median_when_requested():
    """halfwidth 0 = the old behaviour, explicitly."""
    res, est = _robust_integrate(_ModernAI(), None, 100, None, "q_A^-1", 0.0)
    assert res.tag == "ng" and est == "median"   # _ng without quant kwargs
    res, est = _robust_integrate(_LegacyBandAI(), None, 100, None, "q_A^-1", 0.0)
    assert res.tag == "legacy_median" and est == "median"


def test_band_mean_dequantizes():
    """The numerical point of the change: on integer counts, a pure median is
    quantized to half-integers; a 45–55% band mean is continuous-valued."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(30.0, size=400)          # one radial bin's azimuthal pixels
    med = float(np.median(counts))
    assert (2 * med) == int(2 * med)              # quantized to 0.5 steps
    lo, hi = np.quantile(counts, [0.45, 0.55])
    band = counts[(counts >= lo) & (counts <= hi)]
    assert band.size >= 0.08 * counts.size        # a real band, not a point
    # band mean varies continuously as the sample changes; median jumps in 0.5s
    means, medians = [], []
    for s in range(30):
        c = rng.poisson(30.0, size=400)
        l, h = np.quantile(c, [0.45, 0.55])
        means.append(float(np.mean(c[(c >= l) & (c <= h)])))
        medians.append(float(np.median(c)))
    assert len(set(medians)) < len(set(means)), "band mean should take more distinct values"


def test_npt_resolver_fallbacks():
    """Auto without readable geometry -> 1500 fallback; explicit value honoured."""
    npt, sug, mode = _resolve_npt_1d("", "/nonexistent.poni", "/nonexistent.tif")
    assert (npt, sug, mode) == (1500, None, "fallback")
    npt, sug, mode = _resolve_npt_1d("2000", "/nonexistent.poni", "/nonexistent.tif")
    assert (npt, mode) == (2000, "explicit")
    npt, sug, mode = _resolve_npt_1d("auto", "/nonexistent.poni", "/nonexistent.tif")
    assert mode == "fallback"


def main() -> None:
    test_quantile_band_preferred()
    test_pure_median_when_requested()
    test_band_mean_dequantizes()
    test_npt_resolver_fallbacks()
    print("REDUCE ROBUST TEST OK")


if __name__ == "__main__":
    main()
