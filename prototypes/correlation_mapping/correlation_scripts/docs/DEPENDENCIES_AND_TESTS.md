# Dependencies and tests

## Dependency layers

`requirements-core.txt` provides NumPy, SciPy, Matplotlib, pandas, h5py, and
Pillow. `requirements-uote.txt` adds pyFAI and tifffile for the current UOTe
single-crystal and raw-image workflow. `requirements-dev.txt` adds pytest.

Historical Plotly, Streamlit, and workbook-builder dependencies were removed
with their unused scripts. The React explorer is managed separately by its
`package.json`.

## Checks

From `correlation_mapping/`:

```bash
python3 correlation_scripts/correlation_workspace.py check-code

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s correlation_scripts -p 'test_*.py'
```

To inspect local result completion markers:

```bash
python3 correlation_scripts/correlation_workspace.py status
python3 correlation_scripts/correlation_workspace.py commands
```

Real-data generation and validation remain local because experimental inputs
are not committed. Unit tests use deterministic fixtures and are suitable for
CI.
