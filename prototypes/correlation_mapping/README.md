# Correlation mapping prototype

This directory contains the scripts and browser used by the latest indexed
UOTe correlation results. Historical experiments, obsolete dashboards,
workbook builders, unused plot branches, and unrelated BulkXRD adapters were
removed so the branch has one clear research workflow.

## Layout

| Path | Purpose |
|---|---|
| `correlation_scripts/` | Latest result entrypoints, recursive runtime dependencies, validators, tests, configs, and workflow notes |
| `correlation-explorer/` | Read-only React/Vite search and comparison interface for the indexed plots |
| `manifests/` | Small retained input/provenance manifests |

## Results represented here

The retained code reproduces and validates:

1. the c=0.75 Log² formal correlation package;
2. the Log-denoised and original-positive-profile powder waterfalls.

The frozen result/config mapping is in
`correlation_scripts/configs/uote-formal-qwidth075.json`.

## Integrity checks

From this directory:

```bash
python3 correlation_scripts/correlation_workspace.py catalog
python3 correlation_scripts/correlation_workspace.py check-code
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s correlation_scripts -p 'test_*.py'
```

`check-code` verifies exact catalog coverage, Python syntax, and every SHA-256
entry in `CODE_INVENTORY.csv`.

## Data boundary

Raw experimental data and generated results are excluded from Git. Put a local
result tree in the ignored `results/` directory, or point the explorer to an
external directory:

```bash
CORRELATION_RESULTS_ROOT=/absolute/path/to/results npm run index
```

Do not commit experimental data, generated plots, or machine-specific indexes.

## Integration direction

The next production step is to extract dataset-neutral numerical kernels and
HDF5 adapters into `seriesxrd.analysis`. UOTe dataset binding and historical
result assembly should remain optional layers.
