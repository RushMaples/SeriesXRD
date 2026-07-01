# bulkxrd — Claude Code Project Brief

## What this is

Open-source Python package (`bulkxrd`) for automated high-pressure powder X-ray diffraction analysis from diamond-anvil cell (DAC) experiments. Processes thousands of raw 2D detector frames → calibrate → reduce to 1D patterns → **iteratively isolate and identify signal sources** → per-substance heatmaps across a pressure/frame series.

Target user: physics grad student running synchrotron DAC experiments. Final deliverable: filterable heatmaps of individual phases (known + unknown) vs. pressure/frame index.

---

## Repository layout

```
bulkxrd/
  core/          config, env, naming, io, masks, handoff, inspect
  guikit/        theme, tkstyle, tooltip, dpi — shared Tkinter helpers
  calib/         calibration stage (pyFAI geometry refinement, Dioptas-style GUI)
  reduce/        reduction stage (azimuthal integration → HDF5)
    processing.py   pure logic (pyFAI integrate1d / integrate2d)
    worker.py       crash-isolated subprocess
    gui.py          embeddable Tkinter pane
    run_gui.py      CLI entry
    session.py      config seeding
    review.py       read-only reduced-HDF5 inspector + gallery frame metadata
  analysis/      analysis stage (THIS IS THE ACTIVE WORK)
    background.py   Step 1 — DONE (also carries /frames pressure/temp/timestamp)
    peaks.py        Step 2 — DONE
    frame_metadata.py  pressure-prior seam — parse filenames / import CSV → /frames/pressure
    identify.py     Step 3a — pressure-aware EOS matching (consumes the prior)
    residual.py     Step 3a removal — evidence-gated subtraction + residual re-fit
    heatmap.py      waterfall + reflection tracks + per-phase layers
    mldata.py       d-grid resample + ML export + clean pressure-shift simulation
    ml_features.py  Step 3b — analysis HDF5 → model-ready frame features
    ml_simulate.py  Step 3b — pressure-conditioned simulator + DAC augmentations
    ml_rank.py      Step 3b — candidate ranker (ML proposes, physics verifies)
    ml_scorer.py    Step 3b — scorer seam: CosineScorer default; TorchScorer adapter (bulkxrd[ml])
    ml_train.py     Step 3b — learned-scorer training (bulkxrd-ml-train CLI; torch lazy)
    categorization.py  user's workflow spec (read-only notes)
  app.py         top-level launcher that embeds all stages
tests/
  test_imports.py
  test_background.py
  test_peaks.py
  smoke_test.py
```

---

## Stage architecture convention

Every stage follows the same pattern:
- `processing.py` — pure logic, no GUI, no subprocess
- `worker.py` — crash-isolated subprocess wrapper
- `gui.py` — embeddable Tkinter pane (`parent=None` → owns root; else embeds)
- `run_gui.py` — CLI entry point
- `session.py` — config seeding / handoff reading

GUI convention: `make_X_pane()` factory functions, `_owns_root` guard, `shutdown(confirm)` returns False to veto close.

---

## HDF5 schemas

### Reduced HDF5 (output of `reduce/processing.py`)

```
/  attrs: schema_version, unit, poni_text, radial_written
/patterns/intensity          (N_frames, N_bins)  azimuthal MEAN
/patterns/intensity_robust   (N_frames, N_bins)  azimuthal MEDIAN (diamond-spot suppressed)
/patterns/intensity_sigmaclip (N_frames, N_bins) azimuthal SIGMA-CLIPPED trimmed mean
                                                  (optional; keeps textured-ring peaks the
                                                  median drops while rejecting diamond spots)
/patterns/radial             (N_bins,)            q or 2θ axis
/cakes/intensity             (N_cakes, N_radial, N_azimuthal)  optional
/cakes/radial, /cakes/azimuthal, /cakes/frame_index
/frames/filename, ok, seconds, excluded, frame_index, thumb
/frames/pressure, temperature, timestamp   placeholders (pressure seeded NaN; populated
                                            downstream by analysis/frame_metadata.py)
```

### Analysis HDF5 (output of `analysis/background.py` Step 1)

```
/  attrs: schema_version="1", source_reduced, unit, max_half_window, n_passes, use_lls, has_sigmaclip
/radial                      (N_bins,)
/frames/filename             (N,)   copied from reduced
/frames/contamination        (N,)   integrated positive spot residual per frame
/frames/flagged              (N,)   bool, contamination > threshold (optional)
/frames/pressure             (N,)   GPa; carried from reduced, else parsed from filenames
                                     (frame_metadata.py). NaN where unknown. Step-3 prior.
/frames/pressure_sigma       (N,)   GPa per-frame uncertainty (only if a CSV supplied it)
/frames/temperature, timestamp (N,) carried from reduced when present
/background/clean            (N, N_bins)  = robust − baseline
/background/baseline         (N, N_bins)  SNIP estimate
/background/spot_residual    (N, N_bins)  = mean − robust
/background/sigmaclip_residual (N, N_bins) = sigmaclip − robust (only if the reduced file
                                            had intensity_sigmaclip)
```

Step 2 picks the **fit source** from these channels (every source is `clean` plus a
baseline-subtracted residual, since `clean = robust − baseline` and the smooth background
is azimuthally uniform): `clean` (median, conservative), `mean` (`clean + spot_residual`),
`hybrid` (`clean + winsorized(spot_residual)` — narrow diamond spikes removed by a
morphological opening, broad textured-ring excess kept), `sigmaclip`
(`clean + sigmaclip_residual`, the principled trimmed mean), `auto` (sigmaclip if present,
else hybrid).

### Peaks appended by `analysis/peaks.py` Step 2

```
/peaks/counts      (N_frames,)   number of peaks per frame
/peaks/frame       (P,)          frame index (0..N-1) for each peak
/peaks/center      (P,)          peak position on radial axis
/peaks/amplitude   (P,)          peak height
/peaks/fwhm        (P,)          full width at half maximum
/peaks/eta         (P,)          Lorentzian fraction ∈ [0,1]
/peaks/area        (P,)          integrated intensity
/peaks/chi2        (P,)          reduced chi-square of fit
/peaks/flag        (P,) int      0=good; bitmask of FLAG_* constants
```

P = sum(counts). Ragged layout — peak count varies per frame.

### Identify + residual appended by `analysis/identify.py` + `analysis/residual.py` (Step 3a)

```
/identify  attrs: ... p_min, p_max, rel_tol, pressure_window, pressure_sigma_k,
                  min_matched, n_pressure_prior
/identify/<phase>/pressure,score,confidence,recall,precision,n_matched,prior_penalty (N,)
/identify/<phase>  attrs: pressure_model (eos|axial_eos|no_eos), pressure_assumption
                  (eos_based|ambient_reference|eos_missing|ignore_prior), prior_penalized
/identify/<phase>/refl_d, refl_w, refl_hkl   cached ambient reflections (no pymatgen in GUI)
/peaks/phase                 (P,) str   phase attributed to each fitted peak ("" = unexplained)
/residual/clean              (N, N_bins) clean minus the reconstructed peaks of present phases
/residual/explained_counts   (N,) int   good peaks attributed to a known phase
/residual/unexplained_counts (N,) int   good peaks left over
/residual/peaks/counts,frame,center,amplitude,fwhm   peaks RE-FIT on the residual (→ Step 3c)
```

**Pressure prior (the DAC accuracy seam).** Identification reads `/frames/pressure`
(+ optional `/frames/pressure_sigma`) and confines each phase's fit to that frame's
pressure ± window (`pressure_sigma_k·σ` if known, else `pressure_window` GPa) instead of
searching all of `[p_min, p_max]`. This stops a wrong phase sliding along pressure until
a few lines coincide. `marker_prior=True` (no metadata) first fits the marker-category
phases, then reuses the best marker's per-frame pressure as the prior. `confidence` is
now conservative: F1(recall, precision) × evidence(min_matched) × Gaussian pressure-prior
penalty — **not** the old `max(recall, precision)`. Matching is **one-to-one** (an
observed peak can't satisfy several predicted lines).

`run_residual` runs automatically after `run_identification` in the worker. It reuses
the cached `/identify/<phase>/refl_d`+`refl_hkl` and `predicted_d` (same compression
model as 3a) so it needs **no pymatgen**. A phase is only subtracted when it clears
`seen_conf` AND has ≥ `min_matched` one-to-one matched reflections (`allow_sparse`
relaxes this for markers); explained peaks are subtracted (pseudo-Voigt reconstruction)
and the residual is **re-fit with the Step-2 pipeline** (not raw detection) to surface
weaker/unknown features. **Open-set ID**: `identify_all_phases=True` scores the *whole*
library per frame (no candidate pre-selection); "library" = bundled + user phases, not
all of ICSD/MP.

### Step 3b proposer appended by `analysis/ml_rank.py`

```
/ml/candidates  attrs: requested_source (auto|fit|residual|...), source (residual|fit),
                       resolved_source (actual channel, e.g. fit→sigmaclip), top_k,
                       method (scorer name, default cosine), fwhm_d, phases,
                       clip_negative, normalize, n_points (ML preprocessing provenance)
/ml/candidates/<phase>/score    (N,)  per-frame cosine similarity to the phase
/ml/candidates/<phase>/pressure (N,)  pressure the best score used
/ml/candidates/topk_names  (N, top_k) str   ranked candidate names per frame
/ml/candidates/topk_score  (N, top_k)       their scores
```

**ML proposes, physics verifies** (DARA/RADAR-PD). `ml_rank.rank_candidates` ranks the
*whole* library against each frame — cosine of the measured pattern (the `residual` by
default, RADAR-PD-style; else the Step-2 fit source) vs each phase simulated at that
frame's pressure (the metadata prior = the lattice-nudge analog). The union of per-frame
top-K is fed to `run_identification` as the candidate set (worker/`batch --ml-rank`), so
the deterministic matcher only *verifies* a shortlist. **Candidate-free**: with ML rank on,
no Phases-tab pre-selection is needed — it ranks the whole library. Simulation uses the
**same anisotropic `predicted_d`** as Step 3a (an axial-only phase shifts correctly instead
of staying at ambient), and the residual is clipped non-negative before cosine. The v1
ranker is pure-numpy (no torch). The similarity function lives behind the
**`ml_scorer` seam**: `rank_candidates(scorer=...)` takes a `PhaseScorer` — default
`CosineScorer`; `TorchScorer` (a TorchScript model on (measured, candidate) fingerprint
pairs, `bulkxrd[ml]`) raises instructive errors when torch/model are missing. Scorers
have a per-phase `score()` plus an overridable batched `score_frame()`. Whatever the
scorer proposes, Step 3a still verifies.
`ml_features.frame_features` builds the model input (d-grid resample of a chosen source +
pressure/contamination/peaks/excluded); `ml_simulate` builds the DAC-augmented training set
(mixtures, EOS shift, texture, broadening, drift, diamond spikes, background humps,
truncation, noise) on the same grid.

All HDF5 writes are atomic: `.tmp` file + `os.replace`.

---

## Analysis pipeline (categorization.py plan)

```
Step 1 DONE  background.py   diamond spot removal + SNIP baseline
Step 2 DONE  peaks.py        pseudo-Voigt peak/profile fitting
Step 3 compound ID:
    3a  DONE  Deterministic EOS matching, pressure-aware (frame_metadata prior),
              one-to-one match, evidence gate, residual removal
    3b  IN PROGRESS  ML proposes → physics verifies (DARA/RADAR-PD seam):
        ml_features (frame→d-grid features), ml_simulate (DAC-augmented training
        set), ml_rank (deterministic cosine ranker → /ml/candidates top-K →
        Step-3a verifier), ml_scorer (scorer seam), ml_train (DONE: RADAR-PD-style
        pair scorer — strided conv + self-attention on (measured, candidate)
        fingerprints; pairs = augmented mixture + candidate at true P (pos) /
        wrong P / absent (neg); `bulkxrd-ml-train` CLI → TorchScript → TorchScorer;
        train on WashU RIS with pip install -e .[phases,ml]). Untrained-on-real-data;
        deterministic cosine stays the default until a trained model is validated.
    3c  TODO  Unknown clustering (co-occurrence of unmatched peak tracks)
→ per-substance heatmaps (pressure vs frame, filterable by phase)
```

### Step 1 — background.py

Key math:
- `spot_residual = mean − robust` (azimuthal mean minus azimuthal median — diamond single-crystal spots average into mean but not median, so the difference isolates them)
- `baseline = SNIP(robust)` with LLS (Log-Log-Sqrt) transform for dynamic-range compression
- `clean = robust − baseline` — what goes to peak fitting
- `contamination_score` = Σ max(spot_residual, 0) per frame — flags diamond-dominated frames

SNIP recurrence: for m = 1..max_half_window: `work[i] = min(work[i], (work[i-m]+work[i+m])/2)`

LLS: `z = log(log(sqrt(y+1)+1)+1)`, inverse baked in `_lls_inv`.

### Step 2 — peaks.py

Key design:
- **Selectable fit source** (`run_peak_fitting(source=...)`): the azimuthal **median** (`clean`) suppresses diamond spots but *also* drops real peaks on spotty/textured/incomplete rings. Default `source="auto"` fits the reduce-side `sigmaclip` trimmed-mean channel when present, else the analysis-side `hybrid`; `clean`/`mean` remain available. `spot_residual` is kept as a diamond-contamination diagnostic, not the only thing thrown away. The whole-pipeline rationale: don't let the background step quietly eat real sample peaks.
- **Sensitivity presets** (`conservative`/`normal`/`sensitive`) set min_snr / min_prominence_snr / min_fwhm_bins / edge_bins; explicit knobs override. `normal` default = (5, 2, 2, 5). Collapses the per-knob tuning into one physical control.
- **Auto valid-range** (`auto_fit_range`): blank `fit_min`/`fit_max` → conservatively inferred (skip the beamstop-onset ramp + dead/noisy detector tail, capped to the outer ~15 %, decisions on a smoothed copy so noise can't trim interior peaks). Overridable.
- **Pseudo-Voigt**: `A*(η·L + (1−η)·G)`, both normalized to peak height A. L = Lorentzian (size broadening), G = Gaussian (strain/instrument). η fitted free.
- **Detection**: `scipy.signal.find_peaks` + MAD noise floor (`1.4826·median|x−median|`) + SNR threshold (preset default 5σ). Seeds FWHM from half-max crossings.
- **Seed propagation**: good centers from frame k seed detection for frame k+1, so a reflection keeps its identity as the lattice compresses. Merge tolerance scales with peak width (not bin size — this was a bug that caused false duplicates at 0.05 Å drift per frame).
- **Overlapping peaks**: grouped by window overlap, fitted jointly (sum of pseudo-Voigts + constant baseline), still one scipy least_squares call.
- **Rejection flags**: `FLAG_LOW_AMP=1, FLAG_BAD_CHI2=2, FLAG_CENTER_DRIFT=4, FLAG_WIDTH_BOUND=8, FLAG_NO_CONVERGE=16`
- **No JAX**: scipy + vectorized model (numpy broadcasting) handles ~10³ frames in seconds. JAX deferred — needs fixed peak count per batch, heavy dep, rarely the bottleneck.

Analytical area: `A·(η·π·Γ/2 + (1−η)·√(π/4ln2)·Γ)` — matches numeric integral to <1%.

Williamson–Hall: Γ_size = 2πK/D (size), Γ_strain = 4εq (microstrain). Both readable from FWHM across frames.

---

## Step 3 design (next to build)

### 3a — Deterministic EOS matching

For each frame: take the `clean` peak list (good flags only). For each candidate phase (known to be in the cell):
1. From the EOS (Birch–Murnaghan: P = 3K₀/2·[(V₀/V)^7/3−(V₀/V)^5/3]·[1+3/4·(K₀'−4)·((V₀/V)^2/3−1)]) and an initial pressure guess, compute lattice parameters.
2. Simulate the peak positions (d-spacings from lattice + Miller indices).
3. Score the match against observed peaks (sum of reciprocal |Δd| for matched peaks).
4. Iterate: pressure is the fitting parameter → converge on best-fit pressure per frame.

Output: per-frame pressure estimate + phase assignment confidence per peak.

### 3b — ML on simulated patterns

Reference: **SimXRD-4M** (Cao et al., ICLR 2025, arxiv 2406.15469). 4.07M simulated patterns from 119,569 MP structures × 33 conditions (grain size, stress, orientation, noise, background, drift). Open-sourced simulator: PysimXRD.

Key findings from paper relevant to us:
- Models trained on simulation generalize to real data (89% crystal-system accuracy on RRUFF experimental data, Table 5)
- **No pooling layers** — best model (CNN11) has none; pooling destroys peak-position and relative-intensity info
- Bidirectional reading helps (XRD patterns carry information in both directions)
- PatchTST (subsequence patching) better than raw transformer for peak identification
- **Out-of-library (unknown) performance collapses** — ~F1 0.15 near random (Table 4). Known closed-set works, unknowns need a different approach.
- Long-tailed space-group distribution: label smoothing + focal loss help, weighted classification does not
- Format: d–I pattern, 3501 points, d from 1.199–8.853 Å, wavelength-independent

**DAC-specific gap SimXRD doesn't cover**: lattice compression under pressure. Fix: augment simulated patterns by scaling lattice parameters along Birch–Murnaghan EOS → model sees the peak-shift manifold across pressure range.

**What we feed the ML**: the `clean` pattern resampled onto a fixed d-grid, background already subtracted. Important: **train the model on equivalently preprocessed simulated patterns** (apply the same SNIP step to training data), otherwise you create a new sim-to-real gap.

Multi-phase in a frame: after Step 3a removes matched-phase peaks, the residual is fed through again. Or: train a multi-label classifier that scores each candidate phase independently.

### 3c — Unknown detection

After 3a and 3b, peaks that match no known candidate are candidates for unknowns. Strategy:
- Track these unmatched peaks across the pressure series (they drift coherently if they belong to a real phase).
- Cluster by co-occurrence: two peaks that appear/disappear/drift together in the same frames likely belong to the same phase.
- Flag frames where a track appears, disappears, or merges as phase-transition candidates.
- Enough coherent tracks → fingerprint → search MP/ICSD or flag as genuinely unknown.

### Heatmap output

Per substance (known phase label or "unknown cluster N"):
- x-axis: frame index (or pressure if Step 3a converged)
- y-axis: q (or d)
- color: matched-peak amplitude or integrated area
- Filter UI: toggle per-substance layers on/off

---

## Active branch

`claude/tender-dirac-cn5dsj` — all Steps 1 and 2 committed here.

Notable earlier branches (not merged, potentially useful):
- `claude/reduce-gallery` / `claude/reduce-gallery2` — cake matrix viewer (click-to-flag thumbnails). Backend verified, not merged. Adds ~350 lines to reduce stage (gui.py +209, processing.py +63, review.py +75, __init__.py +4).

---

## Key design decisions (don't relitigate)

- **Fit in q, not 2θ**: peak width roughly constant in q → uniform window sizing across the pattern.
- **Robust integration (median)**: pyFAI `medfilt1d` gives the azimuthal median. It has 50% breakdown point so diamond spots (which affect <50% of azimuthal bins) are suppressed. The median is the *baseline reference* (most outlier-free), but it over-suppresses azimuthally-sparse real signal, so it is **not** forced to be the peak-fitting source — see Step 2 "Selectable fit source". The reduce-side `sigma_clip_ng` (`error_model="azimuthal"`, `sigmaclip_1d`) gives a trimmed mean that rejects spots like the median while keeping textured-ring intensity.
- **SNIP window conservative**: set to ~1.5–2× the broadest Bragg peak half-width. Over-aggressive window erodes real broad peaks — true information loss, not reversible. Step 1 records the baseline so the original data is always recoverable.
- **HDF5 atomic writes**: `.tmp` + `os.replace` throughout. Never partially-written files.
- **No JAX yet**: scipy handles the scale; JAX needs fixed peak count per batch (incompatible with variable peak count), heavy dependency, and rarely the bottleneck. Interface is clean enough to add a JAX backend behind it later.
- **Calibration handoff**: `handoff_for_next_notebook.json` (name kept for back-compat even though it's no longer a notebook workflow). Accepted PONI has geometry overrides baked in (`_apply_geometry_overrides`), not verbatim copy.
- **`intensity_robust` required for analysis**: if missing from reduced HDF5, `run_background_separation` raises with an instructive error. Re-run reduction with `robust_1d=True`.

---

## Environment

```
Python 3.x
numpy, scipy 1.17.1, h5py, tifffile
pyFAI 2026.5.0  (for reduce stage)
No display in CI — GUI verified by import/compile only
```

No JAX installed. No GPU required.

---

## Operational rules (for this project's Claude sessions)

- Develop on `claude/*` branches, push with `git push -u origin <branch>`
- Do NOT create pull requests unless explicitly asked
- Never push to main/master without permission
- Do NOT include model identifiers in commit messages, PR bodies, code, or any pushed artifact
- Commit messages end with `https://claude.ai/code/session_01CsEwBEbW7wy99urMF9b6nf`
- Model preference: Opus for planning/review, Sonnet for implementation, Haiku for prose/mechanical

---

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (`rmaples3/BulkXRD`) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical label strings (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
