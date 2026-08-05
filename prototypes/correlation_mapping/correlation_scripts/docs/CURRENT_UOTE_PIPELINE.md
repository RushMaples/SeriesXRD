# Current UOTe correlation pipeline

## Frozen definition

The active result mapping is
`../configs/uote-formal-qwidth075.json`. The frontend currently indexes:

- `results/uote_nonlinear_squared_qwidth075_comparison_20260803`;
- its transformed-profile powder waterfall suite;
- its Log-correlation/original-positive-profile powder waterfall suite;
- `results/uote_robust_peak_tracks_transition_analysis_20260804`.

For a powder peak centered at (q_i), the formal support is

[
[q_i-0.75q_{\rm width},\;q_i+0.75q_{\rm width}].
]

For anchor (A) and target (B), the directional ROI score is

[
S(A\rightarrow B)=
\frac{\int_{\Omega_A}\min[J_A(2\theta),J_B(2\theta)]\,d(2\theta)}
{\int_{\Omega_A}\max[J_A(2\theta),J_B(2\theta)]\,d(2\theta)}.
]

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

## Stages

### Powder ROI-area

Entrypoint:
`pressure_level_peak_spots_absolute_anchor_iou_correlations_v8.py`.

Dependency chain:

```text
v8 → pressure_level_peak_spots_qwidth_correlations_v7
   → pressure_level_peak_correlations_v6
v8 → nonlinear_intensity_preprocessing
```

The current run passes `--half-width-factor 0.75`; the historical CLI default
remains 0.6 for reproducibility.

### Single-crystal ROI-area

Entrypoint: `run_single_crystal_transformed_roi_correlations.py`.

This uses curated 2D ellipse observations and is scientifically independent of
the powder q-width factor.

### Window correlations

Entrypoint: `run_transformed_integer_window_correlations.py`.

It delegates numerical generation to `integer_window_correlations.py` and
strict-lower presentation to `all_peak_frame_correlations.py`. Window bounds
are fixed integer-degree intervals, so changing powder q-width does not alter
window results.

### Assembly and validation

`assemble_denoised_core_science_root.py` assembles one compact tree per
transform. Location, unaffected window products, and unaffected single-crystal
products may be hardlinked from validated sources; reuse is recorded by
SHA-256 and inode provenance.

`validate_package_denoised_correlation_suites.py` checks hierarchy, counts,
score ranges, CSV/PNG pairing, strict-lower matrices, location invariance, and
numerical Log/Exp differences.

### Powder waterfalls

Entrypoint: `generate_denoised_peak_correlation_waterfall.py`.

- `correlation_transform`: curve and fill height use the transformed profile.
- `original_positive`: color remains the Log/Exp correlation value, while
  height uses measurement-normalized positive spots signal before transform.

`validate_complete_formal_composite_waterfalls.py` audits plot, score,
support, mapping, and master-index contracts. The current indexed waterfall
deliverables are powder-only.

### Robust transition analysis

Entrypoint: `build_robust_tracks_transition_suite.py`.

It compares retained peak tracks across the c=0.60/c=0.75 and Log²/Exp² result
families, produces transition candidates and four summary plots, and writes an
independent validation report.

## Expected formal output counts per transform

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

## Checks

From `correlation_mapping/`:

```bash
python3 correlation_scripts/correlation_workspace.py status
python3 correlation_scripts/correlation_workspace.py check-code
python3 correlation_scripts/correlation_workspace.py commands
```

Write new result suites to new directories. Do not edit assembled result trees
in place because multiple payloads may be hardlinked.
