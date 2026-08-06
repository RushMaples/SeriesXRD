# UOTe Log² correlation mapping

> **Research reference only.** The supported, facility-neutral implementation
> now lives in `seriesxrd/correlations` and is available through the fourth
> Correlations tab or `seriesxrd-correlate`. It consumes an Analysis HDF5 and
> does not import this prototype's UOTe-specific paths, manifests, or frontend.
> This directory remains as algorithm history and validation context.

The supported MVP writes `correlations_powder.h5`/`manifest_powder.json` and
`correlations_single_crystal.h5`/`manifest_single_crystal.json`, so both
sample types can share one result directory; review images remain under
`heatmaps/<sample_type>`. In single-crystal mode, the GUI and CLI
select/recommend `spots`, and the supported ROI feature is explicitly a 1D
radial approximation. It does not reproduce this prototype's raw-TIFF pixel
ROI.

This prototype computes, validates, visualizes, and serves the current UOTe
XRD correlation results. The retained workflow is **Log²-only**; historical
experimental dashboards are not part of this deliverable.

The repository contains the reproducible analysis code. Generated result
images and numerical artifacts are deliberately kept outside Git:

- [`correlation_scripts/`](correlation_scripts/) — numerical pipeline,
  validators, tests, frozen configs, and provenance tools;
- [`correlation-explorer/`](correlation-explorer/) — read-only React/Vite
  browser for a complete local result tree;
- [`manifests/`](manifests/) — retained input and cross-check documentation.

Set `CORRELATION_RESULTS_ROOT` to the local result tree when running the
validators or correlation explorer. The generated heatmaps are not committed
to this repository.

## What the system does

The pipeline performs five scientific tasks for powder and single-crystal
measurements:

1. **ROI-area correlation** — compares peak intensity profiles or transformed
   ROI-area features between pressures.
2. **Peak-location correlation** — compares the positions of matched peaks;
   it is intentionally independent of the intensity transform.
3. **Original-XY shaded waterfalls** — draws original-positive powder or
   single-crystal XY profiles while using Log² ROI correlation as peak color.
4. **Window-to-window correlation across frames** — compares the same angular
   window between pressure frames using direct and ACF-based fingerprints.
5. **Window-to-window correlation within a frame** — compares different
   angular windows inside the same frame.

The current formal configuration is
[`correlation_scripts/configs/uote-formal-qwidth075.json`](correlation_scripts/configs/uote-formal-qwidth075.json).
It freezes the transform, q-width, output-count, entrypoint, and validation
contracts described below.

## Scientific workflow

```text
raw powder XY / raw single-crystal TIFF
                   │
                   ▼
       masks, baseline, normalization,
       peak registry and physical support
                   │
                   ▼
        fixed-scale bounded Log² transform
                   │
          ┌────────┼─────────┐
          ▼        ▼         ▼
       ROI area  location   fixed windows
          │        │         │
          └────────┼─────────┘
                   ▼
       validated matrices and heatmaps
                   │
          ┌────────┴────────┐
          ▼                 ▼
 original-XY waterfalls   read-only explorer
```

### 1. Stable Log² intensity preprocessing

The literal expression `log(f²)` is singular at zero and is not used. For a
physically normalized non-negative intensity `f`, the code first defines one
fixed pooled scale `a` for the complete comparison family:

```text
z = clip(max(f, 0) / a, 0, 1)
epsilon = max((noise_floor / a)², epsilon_floor)
Log²(z) = log1p(z² / epsilon) / log1p(1 / epsilon)
```

Important properties:

- `a` is shared across frames, so per-frame scaling cannot erase real
  amplitude differences;
- the formal scale is the pooled positive Q99.5 value;
- output is exactly bounded to `[0, 1]`;
- zero remains zero and negative residuals are clipped to zero for ROI work;
- masks remain masks and unmasked NaNs remain NaNs;
- for signed window residuals, input is clipped to `[-1, 1]` before squaring;
- the supported stage uses the same pooled scale and epsilon for its positive
  ROI transform and signed-residual window transform;
- this is a nonlinear dynamic-range/noise-suppression transform, not a spatial
  blur or a temporal smoothing operation.

The frozen powder parameters are:

| Parameter | Value |
|---|---:|
| Pooled scale quantile | `0.995` |
| Fixed pooled scale | `178.0838325514805` |
| Physical noise floor | `0.060889165620339095` |
| Epsilon | `1.1690445418573338e-07` |
| Epsilon floor | `1e-12` |

The frozen single-crystal raw-TIFF ROI parameters are pooled independently:

| Parameter | Value |
|---|---:|
| Curated ROI observations | `275` |
| ROI pixel instances | `90,398` |
| Fixed positive Q99.5 scale | `333.7071235707903` |
| Sideband noise floor | `0.1976865895529851` |

Implementation: [`nonlinear_intensity_preprocessing.py`](correlation_scripts/nonlinear_intensity_preprocessing.py).

### 2. Powder ROI-area correlation

The powder analysis uses 280 registered pressure-level peaks across 19
pressures. The formal registry is built from 519 observations and 360
spots-channel XY source files.

For each peak centered at `qᵢ` with detected width `q_widthᵢ`, the physical ROI
support is:

```text
[qᵢ - 0.75 × q_widthᵢ, qᵢ + 0.75 × q_widthᵢ]
```

The q bounds are converted to absolute 2θ because integration is performed in
`d(2θ)`. Profiles are never recentered or width-normalized. After positive
clipping, measurement normalization, and Log² preprocessing, observations in
the same physical frame are summed and distinct frames are averaged.

For anchor profile `A` and target profile `B`, the directional score is an
integrated intersection-over-union evaluated only on the anchor support:

```text
S(A → B) = ∫support(A) min(JA, JB) d(2θ)
           ─────────────────────────────
           ∫support(A) max(JA, JB) d(2θ)
```

`B` is exactly zero outside its physical support. Because the domain belongs
to the anchor, `S(A → B)` need not equal `S(B → A)`. Disjoint supports, no
positive overlap, and a zero denominator are represented by the finite value
`0`; `NaN` is reserved for structurally missing cells and the omitted
anchor-pressure row.

Entrypoint:
[`pressure_level_peak_spots_absolute_anchor_iou_correlations_v8.py`](correlation_scripts/pressure_level_peak_spots_absolute_anchor_iou_correlations_v8.py).

### 3. Peak-location correlation

Powder and single-crystal all-peak maps use the frozen 2θ location similarity:

```text
location_similarity = clip(1 - |2θᵢ - 2θⱼ| / 0.06°, 0, 1)
```

Location is not changed by Log² preprocessing. Powder location matrices are
checked against the previous formal baseline. Single-crystal location maps
have a new 275-anchor rectangular layout, so their independent all-Cartesian-
cell contract validates the formula, masks, shapes, counts, and score range.

### 4. Single-crystal ROI-area and location correlation

The single-crystal runner starts from raw TIFF pixels for 275 curated spots.
It applies the formal detector mask, frame mask, geometric peak ROI, and radial
sideband. The sideband median is subtracted, positive excess is divided by the
TIFF exposure, and all 90,398 valid ROI-pixel instances share one pooled Q99.5
scale. Log² is applied per pixel, and the mean transformed pixel value becomes
the observation's ROI-area feature.

All 275 observations remain independent peaks. Within each of the 12 curated
frames, peaks are sorted deterministically by 2θ, azimuth, and source row and
assigned local IDs such as `p15,9`. The source `track` value is retained only
for provenance; it is never used to match, group, filter, or score peaks.

For every pair of different frames, the runner evaluates the complete
Cartesian product of their local peaks. For two finite non-negative Log² ROI
features, the symmetric score is:

```text
area_similarity = min(Aᵢ, Aⱼ) / max(Aᵢ, Aⱼ)
```

Each peak is then used as an anchor. Its map contains 12 registered frame rows
and 35 local-peak columns (the maximum peak count in any frame). The anchor's
own frame row and nonexistent local slots are blank; every peak in every other
frame is scored. This produces 275 ROI-area and 275 location heatmaps.

Entrypoint:
[`run_single_crystal_transformed_roi_correlations.py`](correlation_scripts/run_single_crystal_transformed_roi_correlations.py).

### 5. Window correlations

This historical prototype applies the unchanged asymmetric-least-squares
baseline, forms bounded residuals, applies Log², and evaluates fixed integer
windows:

```text
0–5°, 1–6°, 2–7°, ..., 27–32°
```

Three across-frame products are retained:

| Method | Meaning |
|---|---|
| `acf_strict` | Pearson correlation between positive-lag FFT-ACF fingerprints for the same window |
| `direct_strict` | Pearson correlation between standardized transformed residual vectors for the same window |
| `shift_tolerant_secondary` | Symmetric maximum ACF score over the same window and its immediate ±1 neighbors; a secondary diagnostic |

Across-frame comparisons use same-scan pressure pairs. Within-frame maps use
Pearson correlation between the ACF fingerprints of different windows in the
same frame and then apply the frozen scan-support aggregation rule. Powder
contains `spots` and `fit_control` channels; single crystal contains the
`spots` channel.

These are prototype-only details. The supported SeriesXRD MVP retains direct
and strict ACF products, standardizes positive-lag FFT-ACF fingerprints, and
excludes lag zero. It does **not** migrate `shift_tolerant_secondary` or
same-scan aggregation because the public Analysis HDF5 has no stable
`scan_id`. In the supported stage, window width and step use the Analysis HDF5
native radial unit, and width cannot exceed the selected radial span.

Only the strict lower triangle is presented for square window matrices. The
single-crystal peak maps are rectangular anchor-to-frame-slot maps and instead
blank the anchor's whole frame row. A missing value is not the same as a
computed zero.

Entrypoint:
[`run_transformed_integer_window_correlations.py`](correlation_scripts/run_transformed_integer_window_correlations.py).

### 6. Original-XY shaded waterfalls

The latest validated local waterfall suite contains 280 powder plots and 275
single-crystal plots, one for every anchor. Each plot deliberately separates
two domains:

- **vertical profile height:** the measurement-normalized, positive-clipped
  spots-channel signal before the nonlinear Log² transform;
- **peak color:** the formal directional Log² ROI correlation for the selected
  anchor.

The two domains are joined exactly by frame-local peak identity. For powder,
the profile reconstruction uses the same 519 formal observation components
and q-width supports as the ROI calculation. Components are summed within a
frame, averaged across distinct frames for each pressure-level peak, and the
12–22 peak profiles at one pressure are summed into a common XY trace. For
single crystal,
the displayed trace is read directly from each original
`frame_XXXX_masked.xy`, positive-clipped, divided by its TIFF exposure, and
placed on one shared display scale. Log² changes the correlation color only;
it is not applied to the displayed single-crystal curve.

Because azimuthally distinct spots can overlap after projection onto 1D 2θ,
the colored fill alone is not lossless. A non-overlapping ribbon below each
trace is the authoritative peak-to-correlation encoding.

Entrypoints:
[`generate_denoised_peak_correlation_waterfall.py`](correlation_scripts/generate_denoised_peak_correlation_waterfall.py)
and
[`generate_single_crystal_all_peak_correlation_waterfalls.py`](correlation_scripts/generate_single_crystal_all_peak_correlation_waterfalls.py).

## Local generated PNG result folder

The curated export is generated locally and is not tracked by Git. It contains
only non-empty PNG files: no matrices, raw data, JSON, CSV, symlinks, or
machine-specific paths. A typical local export has this layout:

```text
latest_log_squared_heatmaps_organized_20260805/
├── powder/
│   ├── roi_area_correlation/{pressure}_GPa/
│   ├── location_correlation/{pressure}_GPa/
│   ├── waterfall_original_xy_shaded/{pressure}_GPa/
│   ├── window_to_window_across_frames/{channel}/{method}/
│   └── window_to_window_within_frames/{channel}/{method}/
└── single_crystal/
    ├── roi_area_correlation/{pressure}_GPa/
    ├── location_correlation/{pressure}_GPa/
    ├── waterfall_original_xy_shaded/{pressure}_GPa/
    ├── window_to_window_across_frames/{channel}/{method}/
    └── window_to_window_within_frames/{channel}/{method}/
```

| Sample / product | PNG files in organized folders | Unique scientific source images |
|---|---:|---:|
| Powder ROI area | 280 | 280 |
| Powder location | 280 | 280 |
| Powder original-XY waterfall | 280 | 280 |
| Powder across frames | 168 | 168 |
| Powder within frames | 40 | 40 |
| Single-crystal ROI area | 275 | 275 |
| Single-crystal location | 275 | 275 |
| Single-crystal original-XY waterfall | 275 | 275 |
| Single-crystal across frames | 57 | 57 |
| Single-crystal within frames | 12 | 12 |
| **Total** | **1,942** | **1,942** |

In the latest validated local export, each scientific image occurs once.
Powder ROI, location, and waterfall products cover 19 pressure folders;
single-crystal ROI, location, and waterfall products cover 12 pressure
folders. Single-crystal placement uses the anchor peak's pressure, not a
multi-pressure track identity.

## Installation

Run commands from `prototypes/correlation_mapping/` unless stated otherwise.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r correlation_scripts/requirements-dev.txt
```

For the full formal UOTe workflow, `requirements-dev.txt` includes the core
NumPy/SciPy/Matplotlib/pandas/h5py/Pillow stack plus pyFAI, tifffile, and
pytest. The raw experimental files are intentionally not committed.

Before a run, inspect the active contract and verify the retained source tree:

```bash
CORRELATION_RESULTS_ROOT=/path/to/correlation/results \
  python3 correlation_scripts/correlation_workspace.py status
python3 correlation_scripts/correlation_workspace.py catalog
python3 correlation_scripts/correlation_workspace.py check-code
```

## Running the pipeline

All output directories below must be new or empty. Replace `/path/to/...` with
the experimental and manifest locations on the machine performing the run.

### A. Generate powder ROI and location maps

```bash
python3 correlation_scripts/pressure_level_peak_spots_absolute_anchor_iou_correlations_v8.py \
  --out-dir /path/to/work/powder_log_squared \
  --spots-root /path/to/powder/spots_channel_xy \
  --observations /path/to/spot_observations.csv \
  --track-points /path/to/spot_track_points.csv \
  --untracked-points /path/to/spot_untracked_points.csv \
  --manifest /path/to/powder_manifest.csv \
  --fit-root /path/to/powder/fit_channel_xy \
  --half-width-factor 0.75 \
  --intensity-transform log_squared \
  --transform-scale-quantile 0.995 \
  --transform-noise-floor 0.060889165620339095
```

### B. Generate single-crystal ROI and location maps

```bash
python3 correlation_scripts/run_single_crystal_transformed_roi_correlations.py \
  --mode log_squared \
  --out-dir /path/to/work/single_crystal_log_squared \
  --data-root /path/to/curated/single_crystal_tables \
  --single-manifest /path/to/single_crystal_manifest.csv \
  --single-raw-root /path/to/single_crystal/raw_tiff \
  --scale-quantile 0.995
```

### C. Generate across-frame and within-frame window maps

```bash
python3 correlation_scripts/run_transformed_integer_window_correlations.py \
  --out-dir /path/to/work/log_squared_windows \
  --transform-mode log_squared \
  --workers 8 \
  --transform-scale-quantile 0.995 \
  --single-root /path/to/single_crystal/xy \
  --single-manifest /path/to/single_crystal_manifest.csv \
  --single-profile correlation_scripts/configs/uniform-correlation-v2.1.json \
  --powder-root /path/to/powder/xy \
  --powder-manifest /path/to/powder_manifest.csv \
  --powder-profile correlation_scripts/configs/uniform-correlation-v2.1.json
```

Use `--max-scans` only for a development smoke test; it is not a formal run.

### D. Assemble the compact formal package

The assembler creates a validator-visible tree containing only primary
ROI/location and window products. It uses hardlinks on the same filesystem and
refuses unsafe source/destination layouts.

```bash
python3 correlation_scripts/assemble_denoised_core_science_root.py \
  --output-root /path/to/results/log_squared \
  --transform-label log_squared \
  --powder-roi-source /path/to/work/powder_log_squared \
  --single-roi-source /path/to/work/single_crystal_log_squared \
  --window-root /path/to/work/log_squared_windows \
  --baseline-root /path/to/previous/formal_baseline
```

Add `--dry-run` first to inspect the planned assembly without writing it.

### E. Validate the formal package

```bash
python3 correlation_scripts/validate_package_denoised_correlation_suites.py \
  --log-root /path/to/results/log_squared \
  --baseline-root /path/to/previous/formal_baseline \
  --output-dir /path/to/results/log_validation \
  --dry-run
```

The validator checks hierarchy, exact counts, one-to-one PNG/CSV pairs,
`[0,1]` score ranges, strict-lower matrix structure, absence of supplementary
`1-r` diagnostics, and exact powder location equivalence to the baseline.
The single-crystal runner separately fails closed unless all 275 anchors,
cross-frame Cartesian cells, rectangular masks, and finite scores validate.
Remove `--dry-run` only when validation files should be written.

### F. Generate the original-XY shaded waterfalls

```bash
python3 correlation_scripts/generate_denoised_peak_correlation_waterfall.py \
  --comparison-root /path/to/results/comparison_root \
  --mode log_squared \
  --all-anchors \
  --trace-source formal_composite \
  --display-profile-domain original_positive \
  --out-dir /path/to/results/waterfall_original_xy \
  --compact-batch
```

Validate a newly generated waterfall suite with:

```bash
python3 correlation_scripts/validate_complete_formal_composite_waterfalls.py \
  --comparison-root /path/to/results/comparison_root \
  --suite-root /path/to/results/waterfall_original_xy \
  --powder-only \
  --modes log_squared
```

This validator writes index and validation files into the suite root; do not
run it against a frozen result directory unless that write is intended.

Generate the 275 single-crystal original-XY waterfalls from the all-peak run:

```bash
python3 correlation_scripts/generate_single_crystal_all_peak_correlation_waterfalls.py \
  --analysis-root /path/to/work/single_crystal_log_squared/single_crystal/all_peak_log_squared \
  --xy-root /path/to/single_crystal/Masked \
  --out-dir /path/to/results/single_crystal_waterfall_original_xy
```

The script writes `WATERFALL_INDEX.csv` and `SUITE_VALIDATION.json` and refuses
to finish if any cross-frame peak lacks a correlation value.

## Running the correlation explorer

Generated PNG folders are not committed or browsable directly on GitHub. The
explorer requires the complete local result tree because it audits companion
matrices and provenance metadata before exposing an image.

```bash
cd correlation-explorer
npm install
CORRELATION_RESULTS_ROOT=/path/to/correlation/results npm run index
CORRELATION_RESULTS_ROOT=/path/to/correlation/results npm run dev
```

Open `http://127.0.0.1:4311`. For a production build:

```bash
npm run build
CORRELATION_RESULTS_ROOT=/path/to/correlation/results npm start
```

Open `http://127.0.0.1:4312`. The server is read-only, serves only indexed
assets, rejects arbitrary paths and non-GET methods, and stores favorites in
browser local storage.

## Tests and validation gates

Code integrity and correlation unit tests:

```bash
python3 correlation_scripts/correlation_workspace.py check-code
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s correlation_scripts -p 'test_*.py'
python3 correlation_scripts/validate_package_denoised_correlation_suites.py \
  --self-test
```

Frontend index, API tests, and production build:

```bash
cd correlation-explorer
CORRELATION_RESULTS_ROOT=/path/to/correlation/results npm run index
CORRELATION_RESULTS_ROOT=/path/to/correlation/results npm test
npm run build
```

From the SeriesXRD repository root, run the full regression suite with:

```bash
python -m pytest
```

The latest validated local result set was prepared after these gates passed:

- correlation code-integrity check: PASS;
- correlation unit tests: 119 passed;
- package-validator self-test: PASS;
- real Log² package validation: PASS;
- frontend index: 1,942 records, 0 errors, 0 warnings;
- frontend tests: 15 passed;
- frontend production build: PASS;
- full SeriesXRD suite: 168 passed, 2 skipped.

## Interpretation limits

- A high similarity is evidence of signal/profile similarity under the stated
  support and preprocessing rules; by itself it is not proof of a phase
  transition.
- Missing cells, structurally omitted cells, and computed zeros have different
  meanings and must not be merged.
- Powder ROI IoU is directional. Do not silently symmetrize it.
- `fit_control` is a control channel and can be dominated by
  tungsten/background behavior; it is not direct evidence for UOTe.
- The original-XY waterfall changes only the displayed curve height. Its colors
  remain the formal Log² ROI correlation values.

## Data and provenance boundary

Raw TIFF/XY data, full numerical matrices, generated PNG result folders, and
generated local indexes remain outside Git. They are selected by explicit
command-line paths or `CORRELATION_RESULTS_ROOT`. Machine-specific explorer
indexes are ignored because they contain absolute paths.

The exact retained Python inventory is recorded in
[`correlation_scripts/CODE_CATALOG.json`](correlation_scripts/CODE_CATALOG.json)
and [`correlation_scripts/CODE_INVENTORY.csv`](correlation_scripts/CODE_INVENTORY.csv).
`correlation_workspace.py check-code` verifies catalog coverage, syntax,
imports, and the recorded SHA-256 hashes.
