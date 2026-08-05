# Correlation code

This directory is the canonical source location for UOTe correlation work in
this workspace. It currently contains 104 top-level Python files: active
science entrypoints, required historical dependencies, generic BulkXRD/four-map
adapters, visualization branches, validators, exploratory analyses, and 20
contract/regression tests.

Files remain physically flat for reproducibility. Moving them now would break
bare sibling imports, frozen v2 hash guards, and historical result provenance.
Logical organization is provided by:

- [`CODE_CATALOG.json`](CODE_CATALOG.json): exact one-group-per-file ownership;
- [`CODE_INVENTORY.csv`](CODE_INVENTORY.csv): SHA-256 snapshot of code/configs;
- [`docs/CODE_CATALOG.md`](docs/CODE_CATALOG.md): readable category guide;
- [`correlation_workspace.py`](correlation_workspace.py): read-only navigator
  and integrity checker.

## Active formal entrypoints

| Stage | Entrypoint | Notes |
|---|---|---|
| Intensity transform | `nonlinear_intensity_preprocessing.py` | Bounded Log²/Exp² preprocessing |
| Powder ROI | `pressure_level_peak_spots_absolute_anchor_iou_correlations_v8.py` | Absolute-q directional integrated IoU; pass `--half-width-factor 0.75` |
| Single-crystal ROI | `run_single_crystal_transformed_roi_correlations.py` | Curated 2D ellipse observations |
| Across/within windows | `run_transformed_integer_window_correlations.py` | Fixed integer windows; strict-lower presentation |
| Package assembly | `assemble_denoised_core_science_root.py` | Hardlink assembly with provenance |
| Package validation | `validate_package_denoised_correlation_suites.py` | Independent delivery gate |
| Powder waterfall | `generate_denoised_peak_correlation_waterfall.py` | Transformed or `original_positive` height domain |
| Single waterfall | `generate_single_crystal_denoised_correlation_waterfall.py` | Formal single-crystal composite |
| Waterfall validation | `validate_complete_formal_composite_waterfalls.py` | PNG, score, support, and mapping audit |

The full dependency and aggregation semantics are in
[`docs/CURRENT_UOTE_PIPELINE.md`](docs/CURRENT_UOTE_PIPELINE.md).

## Read-only workspace commands

Run these from `correlations/`:

```bash
python3 correlation_scripts/correlation_workspace.py status
python3 correlation_scripts/correlation_workspace.py catalog --group active_formal
python3 correlation_scripts/correlation_workspace.py check-code
python3 correlation_scripts/correlation_workspace.py commands
```

`status` checks the current completion/validation markers against the configured
roots, c=0.75 scope, transform modes, sample scope, and expected counts. It also
checks that active entrypoints and required compatibility dependencies exist.
`check-code` checks catalog coverage, Python syntax, and every frozen SHA in
`CODE_INVENTORY.csv`. Neither command writes scientific results. `commands`
only prints resolved validator invocations; its core-package command uses
`--dry-run`, while the printed waterfall validators refresh validation index
and report files if you execute them.

## Version relationships that must remain visible

```text
powder v8 (ACTIVE)
  └─ imports v7 helpers
       └─ imports v6 helpers

transformed integer windows (ACTIVE)
  ├─ integer_window_correlations
  ├─ all_peak_frame_correlations
  └─ frozen uniform-v2 + additive v2.1 modules
```

`v6`, `v7`, and uniform-v2 are therefore `REQUIRED_DEPENDENCY` or
`FROZEN_COMPATIBILITY`, not deletion candidates.

## Requirements

- `requirements-core.txt`: numerical/plotting/data basics plus Pillow.
- `requirements-uote.txt`: current formal UOTe workflow, including pyFAI.
- `requirements-optional.txt`: Plotly legacy 3D and Streamlit dashboard.
- `requirements-dev.txt`: development/test environment.

The four `.mjs` workbook builders additionally require the external
`@oai/artifact-tool` runtime; they are not needed for numerical correlations.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s correlation_scripts -p 'test_*.py'
```

The legacy 3D module now imports without Plotly; only inline HTML generation
requires the optional package. Real-data integration checks remain local, while
unit tests use deterministic fixtures.
