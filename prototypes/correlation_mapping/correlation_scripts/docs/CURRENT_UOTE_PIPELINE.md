# Current formal UOTe correlation pipeline

## Frozen definition

The active workflow is described by
[`../configs/uote-formal-qwidth075.json`](../configs/uote-formal-qwidth075.json).
The current validated result root is
`correlations/results/uote_nonlinear_squared_qwidth075_comparison_20260803`.

For a powder peak centered at (q_i), the formal support is

\[
[q_i-0.75q_{\rm width},\;q_i+0.75q_{\rm width}].
\]

For anchor (A) and target (B), the directional ROI score is

\[
S(A\rightarrow B)=
\frac{\int_{\Omega_A}\min[J_A(2\theta),J_B(2\theta)]\,d(2\theta)}
{\int_{\Omega_A}\max[J_A(2\theta),J_B(2\theta)]\,d(2\theta)}.
\]

The target is zero outside its own physical support. Disjoint supports and
zero denominators therefore produce the finite correlation value 0.

## Data flow

```text
519 powder observations / 360 spots-channel XY source frames
  ├─ positive piecewise-linear clipping
  ├─ frame measurement normalization
  ├─ fixed pooled Q99.5 scaling
  ├─ Log² or Exp² transform
  ├─ same-frame observation sum
  └─ distinct-frame mean per pressure-level peak
       └─ 280 anchor maps over 19 pressure levels

curated single-crystal observations
  └─ 2D ellipse extraction + exposure normalization + Log²/Exp²

1,060 accepted powder frames + single-crystal frame set
  └─ fixed windows 0–5, 1–6, ..., 27–32 degrees
       ├─ across-frame correlation
       └─ within-frame window-to-window correlation
```

## Stages and ownership

### 1. Powder ROI-area

Entrypoint:
`pressure_level_peak_spots_absolute_anchor_iou_correlations_v8.py`.

Required dependencies:

```text
v8 → pressure_level_peak_spots_qwidth_correlations_v7
   → pressure_level_peak_correlations_v6
v8 → nonlinear_intensity_preprocessing
```

The current run must explicitly pass `--half-width-factor 0.75`; the historical
CLI default remains 0.6 for backward reproducibility.

### 2. Single-crystal ROI-area

Entrypoint: `run_single_crystal_transformed_roi_correlations.py`.

This uses curated 2D ellipse observations and is scientifically independent of
the powder q-width factor.

### 3. Window correlations

Entrypoint: `run_transformed_integer_window_correlations.py`.

It delegates numerical generation to `integer_window_correlations.py` and the
strict-lower delivery layer to `all_peak_frame_correlations.py`. Window bounds
are absolute integer-degree intervals, so changing powder q-width does not
change window results.

### 4. Assembly and validation

`assemble_denoised_core_science_root.py` assembles one compact tree per
transform. Location, unaffected window products, and unaffected single-crystal
products may be hardlinked from validated sources. Every reuse is recorded by
SHA-256/inode provenance.

`validate_package_denoised_correlation_suites.py` is the independent package
gate. It checks hierarchy, counts, score ranges, CSV/PNG pairing, strict-lower
matrices, location invariance, and numerical Log/Exp differences.

### 5. Waterfalls

Powder: `generate_denoised_peak_correlation_waterfall.py`.

- `correlation_transform`: curve/fill height is the transformed formal profile.
- `original_positive`: colors remain Log/Exp correlation values, while curve
  and fill height use measurement-normalized positive spots signal before the
  nonlinear transform.

Both modes preserve the same peak registry and support mapping. Same-frame
components are summed, distinct frames are averaged per peak, and the 12–22
formal peaks at one pressure are summed into one pressure trace.

Single crystal: `generate_single_crystal_denoised_correlation_waterfall.py`.

## Expected output counts per transform

| Sample/category | CSV | PNG |
|---|---:|---:|
| Powder ROI area | 280 | 280 |
| Powder location | 280 | 280 |
| Powder across frames | 168 | 168 |
| Powder within frame | 40 | 40 |
| Single ROI area | 75 | 75 |
| Single location | 75 | 75 |
| Single across frames | 57 | 57 |
| Single within frame | 12 | 12 |

## Resolved validation commands

Print resolved commands from the workspace front door:

```bash
python3 correlation_scripts/correlation_workspace.py commands
```

The front door only prints these commands. Running the full waterfall
validators refreshes their index and validation-report files inside the
selected result suite; use `status` for a read-only completion-marker check.

Check current completion markers and code integrity:

```bash
python3 correlation_scripts/correlation_workspace.py status
python3 correlation_scripts/correlation_workspace.py check-code
```

The examples above assume the working directory is `correlations/`.

## Operational boundaries

- Generate a waterfall suite into a new `--out-dir`. The formal waterfall
  generator writes anchors incrementally and is not an all-or-nothing commit.
- The `original_positive` display audit in the current validator includes
  constants frozen for this UOTe c=0.75 suite. It verifies this formal package;
  it is not a generic validator for a different dataset or q-width factor.
