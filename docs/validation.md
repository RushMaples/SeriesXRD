# Validation and limitations

What SeriesXRD's outputs have been checked against, the tolerances you
should expect, and the assumptions baked into each stage. Read this before
using any output in a publication.

## What is validated, and how

**Azimuthal integration is pyFAI's, not ours.** SeriesXRD does not
reimplement integration: the reduce stage drives
`pyFAI.AzimuthalIntegrator` (`integrate1d`/`integrate2d`,
`medfilt1d_ng`/`sigma_clip_ng` for the robust channels). Integration
accuracy is therefore pyFAI's, and pyFAI's own validation applies. What
SeriesXRD adds on top — channel arithmetic (`spot_residual = mean −
robust`, `clean = robust − baseline`), quantile-band behavior, and NaN/edge
handling — is covered by the test suite on synthetic cakes with known
answers (`tests/test_reduce_robust.py`).

**Peak fitting is verified on synthetic patterns with known truth.** The
pseudo-Voigt analytical area matches numerical integration to better than
1% (`tests/test_peaks.py`), and the fitting pipeline is required to recover
planted peak centers, widths, and amplitudes on noisy synthetic series,
including overlapping groups and frame-to-frame drift. Fit uncertainties
(`center_err` etc.) are 1σ estimated standard deviations from the
least-squares covariance — they describe statistical error under the
model, and understate the true error where the model is wrong (overlapped
unresolved peaks, non-Voigt shapes, imperfect background).

**Identification is verified two ways.** Synthetic: `tests/test_identify.py`
plants phases at known pressures and requires the matcher to find them
(and to *not* find absent ones — one-to-one matching, evidence gates,
pressure-prior penalties are all exercised). Real data:
`seriesxrd-benchmark` runs labelled measured patterns (RRUFF, opXRD, or
your own) through the actual Step-1/2 preprocessing and scores
identification (hit@1 / hit@K / MRR). A learned Step-3b scorer is only
trusted if it beats the deterministic cosine baseline on that harness.

**Calibration quality is measured, not assumed.** The calibration stage is
a review/QA workflow around pyFAI's geometry refinement; the reduction
stage additionally fits ring waviness on the first cakes as it finishes
and warns — in physical units (implied transverse sample offset in mm, 1D
doublet splitting in bins) — when the geometry error is large enough to
corrupt peak fitting.

## Expected tolerances

- **Calibration/geometry:** limited by pyFAI refinement on your calibrant
  image. The waviness check warns when residual geometry error implies
  peak splitting at the bin level; below that, d-spacing accuracy is
  dominated by the calibration, not by SeriesXRD processing.
- **Peak positions:** on clean synthetic data, recovered centers are
  accurate to a small fraction of a bin; on real data, expect the fitted
  `center_err` to be a lower bound on the true uncertainty.
- **Phase matching:** the default relative d-tolerance (`rel_tol`) is a few
  ×10⁻³, widened per-peak by the fitted center uncertainty. A phase's
  `confidence` is deliberately conservative (F1 × evidence ×
  pressure-prior penalty); treat "seen" phases below your own confidence
  threshold as candidates, not conclusions.
- **Pressures from markers:** only as good as the marker's EOS
  parameterization — see the flagged entries in
  [`docs/phase-sources.md`](phase-sources.md). Different published scales
  disagree at the few-percent level at megabar pressures.

## Diagnostic and semi-quantitative outputs

These outputs are for guidance and screening, not for quantitative claims:

- **Contamination score** (`/frames/contamination`): the integrated
  positive spot residual per frame. Unitless, detector- and
  exposure-dependent; compare within a series, not across experiments.
- **Screening phase fractions** (`/fractions`, method `intensity_share` or
  `rir`): intensity shares of attributed peak areas. Texture, absorption,
  and structure-factor differences are **not** corrected. For refined
  weight fractions, complete the GSAS-II round trip below. Those are stored
  separately as `/refinement/fractions` with propagated esds, leaving this
  screening result available for comparison.
- **Texture metrics** (`/texture`): a texture index, spot fraction, and
  preferred-orientation second harmonic per ring. Interpreting the 2-fold
  modulation as texture vs differential stress requires the experiment
  geometry, which stays with you.
- **Williamson–Hall size/strain** (`/microstructure`): without an
  instrument-resolution profile the output is flagged `uncorrected` and
  sizes are lower bounds / strains upper bounds. W-H itself assumes
  size and strain broadening add in a particular way; treat trends, not
  absolute values, as the signal.
- **Unknown clusters** (`/unknowns`): coherent residual-peak tracks are
  *candidates* for real phases; a cluster is a hypothesis with a
  d-fingerprint, not an identification.

## Why the refinement export is not Rietveld refinement

`seriesxrd-export-refinement` writes patterns, phase CIFs, and a GSAS-II
instrument file — a *bridge*, deliberately not a Rietveld engine.
SeriesXRD's identification matches peak positions (with soft intensity
checks); it does not refine structure parameters, site occupancies,
thermal factors, profile convolutions, or weighted whole-pattern
residuals. Quantitative phase fractions, refined lattice parameters with
realistic uncertainties, and structure validation belong to the Rietveld
package you refine the bundle in.

`seriesxrd-import-gsas` closes that hand-off loop without reimplementing the
refinement. It imports GSAS-II's calculated `WgtFrac` values—not raw scale
factors—along with their reported uncertainties, unit cells, Rwp/GOF, and
convergence state. These are Rietveld results, but their validity still
depends on the user's structural model, absorption correction, preferred
orientation treatment, background/profile choices, and refinement stability.

## Known assumptions

- **EOS validity:** every bundled EOS is a room-temperature isotherm with
  an experimental range (see `docs/phase-sources.md`); `p_max` ceilings
  stop extrapolation past known transitions, but *within* the range the
  parameterization is still an extrapolating model.
- **Compression model:** cubic phases scale isotropically; anisotropic
  phases use per-axis moduli (or a signed axial expansivity) — a trigonal
  or monoclinic phase's angles are held fixed, which is approximate at
  high compression.
- **Temperature:** the optional thermal correction is a constant
  volumetric CTE around a reference temperature — adequate for modest
  excursions, not for melts or strongly anharmonic regimes.
- **Texture and intensities:** DAC texture legitimately scrambles relative
  intensities, so intensity agreement is a soft factor in matching
  (weight configurable down to position-only). Absence of a predicted weak
  line is weak evidence; presence of all strong lines is the requirement.
- **Pressure metadata is a prior, not a measurement:** `/frames/pressure`
  narrows the search window; the per-phase fitted pressure is the output.
  If your metadata is wrong by more than the window, identification will
  fail honestly rather than silently fit elsewhere.
- **Spot suppression can suppress the sample:** on coarse-grained or
  near-single-crystal loads the azimuthal median rejects Bragg signal
  itself. Step 1 measures this (`spotty_sample`) and Step 2's `auto`
  source falls back to the mean channel — but a heavily spotty dataset is
  fundamentally harder, and the cake-space spot tracker
  (`seriesxrd-spots`) is the right tool for its single-crystal content.
