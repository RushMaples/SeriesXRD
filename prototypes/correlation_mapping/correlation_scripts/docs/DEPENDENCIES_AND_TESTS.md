# Dependencies and tests

## Dependency layers

### Core

`requirements-core.txt` contains NumPy, SciPy, Matplotlib, pandas, h5py, and
Pillow. This supports the common numerical and rendering layers.

### Current formal UOTe workflow

`requirements-uote.txt` extends core with pyFAI and tifffile. pyFAI is required
for the formal single-crystal geometry/ROI extraction.

### Optional presentation tools

`requirements-optional.txt` contains:

- Plotly for inline legacy 3D HTML;
- Streamlit for `xrd_results_dashboard.py`.

Neither is required for numerical ROI/window computation.

### Development

`requirements-dev.txt` adds pytest. The tests themselves are standard
`unittest` modules and can also run without pytest.

The `.mjs` workbook builders require an external `@oai/artifact-tool` runtime.
They are deliberately outside the numerical Python dependency set.

## Test commands

From `correlations/`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s correlation_scripts -p 'test_*.py'
```

Code/catalog integrity:

```bash
python3 correlation_scripts/correlation_workspace.py check-code
```

Current result completion markers and configured scope:

```bash
python3 correlation_scripts/correlation_workspace.py status
```

Full real-data result validators are intentionally separate from unit tests.
Use `correlation_workspace.py commands` to print resolved invocations. The
core package validator is emitted with `--dry-run`; the waterfall validators
refresh their master index and validation report when executed.

## Test policy

- Synthetic/unit tests may run in CI.
- The 519-observation and 1,060-frame integrations remain local because they
  depend on experimental data not intended for source control.
- Optional visualization tests must not make Plotly or Streamlit mandatory for
  numerical core imports.
- Any physical source move must keep catalog coverage exact, preserve frozen
  hashes, and pass the complete test suite before old paths are removed.
