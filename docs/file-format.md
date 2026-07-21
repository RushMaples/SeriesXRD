# File formats

The HDF5 layouts SeriesXRD writes and reads. Two files carry the pipeline:
the **reduced** file (output of the Reduction stage) and the **analysis**
file (created by analysis Step 1; later steps append groups to it). All
writers are atomic (`.tmp` + `os.replace`) except the live watch-mode file,
which trades atomicity for append speed and is superseded by a normal full
reduction afterwards.

Axis convention: the radial axis is q (Å⁻¹) by default; 2θ (°) is
selectable at reduce time and every consumer handles both (`unit` attr).

## Reduced HDF5 (`reduce/processing.py`)

```
/  attrs: schema_version, seriesxrd_version, created_at, unit, poni_text,
          poni_sha256, mask_sha256, npt_1d, npt_1d_mode, npt_1d_suggested, ...
/patterns/intensity            (N_frames, N_bins)  azimuthal MEAN
/patterns/intensity_robust     (N_frames, N_bins)  spot-suppressed: mean of a
                               narrow azimuthal quantile band around the median
/patterns/intensity_sigmaclip  (N_frames, N_bins)  optional sigma-clipped
                               trimmed mean (keeps textured-ring peaks the
                               median drops while rejecting diamond spots)
/patterns/intensity_straightened         (N_frames, N_bins)  optional; cake-
                               de-waved azimuthal mean (reduce/straighten.py)
/patterns/intensity_straightened_robust  (N_frames, N_bins)  optional; cake-
                               de-waved spot-suppressed median (NaN for frames
                               without a saved cake)
/patterns/radial               (N_bins,)  q or 2θ axis
/cakes/intensity               (N_cakes, N_radial, N_azimuthal)  optional
/cakes/radial, /cakes/azimuthal, /cakes/frame_index
/frames/filename, ok, seconds, excluded, frame_index, thumb
/frames/pressure, temperature, timestamp   placeholders (pressure seeded NaN;
                               populated downstream by frame metadata import).
                               For HDF5/NeXus stack inputs, timestamp,
                               temperature, and /frames/pos_x, pos_y are
                               harvested from the container at reduce time.
/texture/frame, ring_r0, texture_index, po_amplitude, po_phase_deg,
         spotty_frac, coverage   optional; written by reduce/texture.py
```

The live watch-mode variant (`seriesxrd-watch` → `*_live.h5`) uses the same
schema with `live_mode=True`, resizable datasets appended in arrival order,
and no cakes or thumbnails.

## Analysis HDF5

### Created by Step 1 (`analysis/background.py`)

```
/  attrs: schema_version, seriesxrd_version, created_at, source_reduced,
          unit, wavelength, max_half_window, n_passes, use_lls, has_sigmaclip,
          robust_source, n_straightened, signal_frac_clean, spotty_sample,
          npt_1d, npt_1d_mode, npt_1d_suggested
/provenance  attrs: seriesxrd_version, schema_version, tool, created_at,
          python_version, platform, config_json, dependencies_json, and
          per-input identity: input_<name>_path/_bytes/_mtime/_sha256/
          _hash_kind (full SHA-256 up to 64 MiB, head/tail sample above)
/provenance/steps/<step>  attrs: tool, seriesxrd_version, schema_version,
          created_at — one record per appending analysis step below
/radial                        (N_bins,)
/frames/filename               (N,)  copied from the reduced file
/frames/contamination          (N,)  integrated positive spot residual
/frames/flagged                (N,)  bool, contamination > threshold (optional)
/frames/excluded               (N,)  bool, carried from the reduce stage
/frames/pressure               (N,)  GPa; carried from reduced, else parsed
                               from filenames. NaN where unknown.
/frames/pressure_sigma         (N,)  GPa per-frame uncertainty (CSV import)
/frames/temperature, timestamp (N,)  carried when present
/frames/pos_x, pos_y           (N,)  stage positions (mapping scans)
/frames/user_edited            (N,)  bool; values a human set survive
                               re-parsing and Step-1 rebuilds
/background/clean              (N, N_bins)  = robust − baseline
/background/baseline           (N, N_bins)  SNIP estimate
/background/spot_residual      (N, N_bins)  = mean − robust
/background/sigmaclip_residual (N, N_bins)  = sigmaclip − robust (optional)
```

### Appended by Step 2 (`analysis/peaks.py`)

```
/peaks  attrs: schema_version, seriesxrd_version, source, sensitivity, ...
/peaks/counts      (N_frames,)  peaks per frame
/peaks/frame       (P,)  frame index for each peak      (P = sum(counts);
/peaks/center      (P,)  position on the radial axis     ragged layout —
/peaks/amplitude   (P,)  height                          peak count varies
/peaks/fwhm        (P,)  full width at half maximum      per frame)
/peaks/eta         (P,)  Lorentzian fraction ∈ [0, 1]
/peaks/area        (P,)  integrated intensity
/peaks/chi2        (P,)  reduced chi-square of the fit
/peaks/flag        (P,)  int; 0 = good, else bitmask (low amplitude, bad
                         chi², center drift, width at bound, no convergence)
/peaks/center_err, amplitude_err, fwhm_err   (P,)  1σ estimated standard
                         deviations from the least-squares covariance
```

### Appended by Step 3a (`analysis/identify.py` + `analysis/residual.py`)

```
/identify  attrs: p_min, p_max, rel_tol, pressure_window, pressure_sigma_k,
                  min_matched, intensity_k, ...
/identify/<phase>/pressure, score, confidence, recall, precision, n_matched,
                  prior_penalty, intensity_corr   (N,) per frame
/identify/<phase>  attrs: pressure_model, pressure_assumption, prior_penalized
/identify/<phase>/refl_d, refl_w, refl_hkl   cached ambient reflections
/peaks/phase                  (P,) str  phase attributed to each fitted peak
                              ("" = unexplained)
/residual/clean               (N, N_bins)  clean minus reconstructed peaks of
                              phases that cleared the evidence gate
/residual/explained_counts    (N,) int
/residual/unexplained_counts  (N,) int
/residual/peaks/counts, frame, center, amplitude, fwhm   peaks re-fitted on
                              the residual (input to Step 3c)
```

### Appended by Step 3b (`analysis/ml_rank.py`)

```
/ml/candidates  attrs: requested_source, source, resolved_source, top_k,
                method, fwhm_d, fwhm_q, fwhm_q_poly, phases, clip_negative,
                normalize, n_points
/ml/candidates/<phase>/score     (N,)  per-frame similarity to the phase
/ml/candidates/<phase>/pressure  (N,)  pressure the best score used
/ml/candidates/topk_names        (N, top_k) str  ranked candidates per frame
/ml/candidates/topk_score        (N, top_k)
```

### Later analysis groups

```
/unknowns        Step 3c (unknowns.py): obs/, tracks/, clusters/,
                 fingerprint/ — residual peaks linked into gap-tolerant
                 tracks, co-occurrence clusters, per-cluster d-fingerprints
/fractions       fractions.py: names (P,), fractions (N, P) intensity
                 shares; attrs method (intensity_share | rir).
                 Semi-quantitative by design.
/microstructure  microstructure.py: Williamson–Hall size_A, strain, r2 per
                 frame (flagged uncorrected without an instrument profile)
/spots           spots.py: single-crystal reflections tracked in cake space
                 (written to the analysis file, or <reduced>_spots.h5 when
                 no analysis file is given). obs/ per-frame blob detections
                 with pressure and d; scans/ per-scan groups; tracks/
                 pressure-ordered (azimuth, q) links with d0 and dd_dp.
```

## JSON manifests

Every stage run also returns/writes a JSON manifest whose header is
standardized (`core/provenance.manifest_provenance`):

```json
{
  "tool": "seriesxrd.analysis.peaks",
  "seriesxrd_version": "0.2.0",
  "schema_version": "1",
  "created_at": "...",
  "...": "per-stage fields"
}
```

`seriesxrd_version` is the package version that wrote the artifact;
`schema_version` only changes when the file layout changes.
