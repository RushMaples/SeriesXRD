# Correlation mapping prototype

This directory collects the complete standalone correlation-mapping research
code and the XRD Correlation Atlas frontend while the feature is being designed
for integration with `seriesxrd.analysis`.

The prototype is intentionally isolated from the installable `seriesxrd`
package. Its current workflows include UOTe-specific inputs and historical
compatibility layers that need to be separated from the generic algorithms
before they can become a supported SeriesXRD analysis stage.

## Layout

| Path | Purpose |
|---|---|
| `correlation_scripts/` | Canonical Python/MJS correlation code, validators, tests, configs, and workflow documentation |
| `correlation-explorer/` | Read-only React/Vite plot search and comparison interface |
| `manifests/` | Small provenance and input manifests retained for reproducibility |

The backend files remain physically flat because active modules still use
sibling imports and the frozen code inventory records exact file hashes. Use
`correlation_scripts/CODE_CATALOG.json` or
`correlation_scripts/docs/CODE_CATALOG.md` for the logical grouping.

## Current formal workflow

The validated research pipeline provides four complementary correlation
families for powder and single-crystal observations:

1. directional peak ROI-area overlap;
2. peak-location similarity;
3. same-window correlation across frames;
4. window-to-window correlation within a frame.

It also contains correlation-coloured waterfall generation, robust adjacent
condition peak tracking, exploratory transition summaries, and independent
validation gates. The frozen UOTe configuration is
`correlation_scripts/configs/uote-formal-qwidth075.json`.

## Code navigation and checks

From this directory:

```bash
python3 correlation_scripts/correlation_workspace.py catalog
python3 correlation_scripts/correlation_workspace.py check-code
python3 -m unittest discover -s correlation_scripts -p 'test_*.py'
```

`check-code` validates source coverage, syntax, and the frozen SHA-256
inventory. Commands that inspect completed research runs require a local
`results/` directory and are not expected to pass in a clean checkout.

## Data boundary

Raw experimental data, generated results, workspaces, and machine-specific
indexes are deliberately excluded from Git. To use the prototype with local
results, either create an ignored `results/` directory here or point the
frontend at an external results directory with `CORRELATION_RESULTS_ROOT`.

The included manifests document provenance but are not a substitute for the
original experimental data. Do not commit local data or generated result
trees to this branch.

## Known portability debt

The canonical backend was copied byte-for-byte so its frozen inventory remains
valid. A small number of historical packaging, workbook, and transition-suite
entrypoints still contain workstation-specific default paths. They are listed
by:

```bash
rg -n '/Users/stanley|/home/|C:\\' correlation_scripts
```

These defaults should be replaced with explicit CLI arguments or environment
configuration when each script is promoted into the SeriesXRD package. The
browser frontend has already been made portable through
`CORRELATION_RESULTS_ROOT`.

## Integration direction

The next production step is to extract dataset-neutral numerical kernels and
HDF5 adapters into `seriesxrd.analysis`, then add focused tests against
SeriesXRD analysis files. UOTe dataset binding, historical result assembly,
and exploratory transition interpretation should remain optional layers.
