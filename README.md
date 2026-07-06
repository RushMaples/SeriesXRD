# bulkxrd

GUI-driven workflow for powder X-ray diffraction: detector calibration review,
dataset reduction, and pattern analysis. A single unified desktop application
(`bulkxrd`) hosts all pipeline stages in one window. Heavy pyFAI work still
runs in crash-isolated `worker.py` subprocesses so a pyFAI or matplotlib crash
never takes down the GUI.

## Pipeline

The workflow is one subpackage per stage, communicating only through artifacts
on disk plus a shared workspace folder:

1. **`bulkxrd.calib`** — calibration review (implemented): standard image →
   accepted `.poni` + mask + QA record, ending in a
   `handoff_for_next_notebook.json` (internal artifact passed automatically to
   the Reduction tab — not a user-facing step).
2. **`bulkxrd.reduce`** — dataset reduction (implemented): apply the accepted
   geometry/mask to a sample dataset, parallel batch azimuthal integration →
   1D patterns (mean + azimuthal-median "robust") and optional 2D cakes in
   one HDF5 file + JSON manifest.
3. **`bulkxrd.analysis`** — pattern analysis: SNIP background + diamond-spot
   separation (Step 1), pseudo-Voigt peak fitting (Step 2), pressure-aware
   EOS phase identification + residual removal (Step 3a), and the ML
   candidate-ranking seam (Step 3b: deterministic cosine ranker by default,
   optional learned scorer — see `docs/ml-training.md`).

The calib→reduce handoff JSON is an internal artifact written to the workspace
and automatically loaded by the Reduction tab — users do not need to manage it
manually.

## Repository layout

```
├── bulkxrd/             The installable package
│   ├── core/            Shared by all stages (stdlib/numpy only)
│   │   ├── config.py        SessionConfig, JSON/hash/file helpers
│   │   ├── env.py           dependency / conda environment checks
│   │   ├── naming.py        output folder/file naming conventions
│   │   ├── io.py            detector image readers (fabio/tifffile/PIL)
│   │   ├── masks.py         automatic + polygon detector masks
│   │   ├── handoff.py       the calib→reduce handoff contract (load/validate)
│   │   └── inspect.py       detector-image diagnostic CLI (bulkxrd-inspect)
│   ├── guikit/          Shared GUI/plot theming
│   │   ├── theme.py         dark Catppuccin palette (Tk + matplotlib)
│   │   ├── tkstyle.py       shared ttk style (apply_dark_theme)
│   │   └── dpi.py           HiDPI / Windows DPI-awareness helpers
│   ├── calib/           Calibration review stage
│   │   ├── processing.py    pyFAI integration + QA figure generation
│   │   ├── worker.py        crash-isolated worker subprocess
│   │   ├── gui.py           tabbed Tkinter GUI (embeddable pane)
│   │   ├── dioptas.py       optional Dioptas hand-off
│   │   └── run_gui.py       CLI entry point (bulkxrd-calib-gui)
│   ├── reduce/          Batch reduction stage
│   │   ├── processing.py    batch azimuthal integration logic
│   │   ├── worker.py        crash-isolated worker subprocess
│   │   ├── session.py       workspace config seeding (seed_reduction_config)
│   │   ├── review.py        read-only HDF5 checkpoint review
│   │   ├── gui.py           tabbed Tkinter GUI (embeddable pane)
│   │   └── run_gui.py       CLI entry point (bulkxrd-reduce-gui)
│   ├── app.py           unified application (bulkxrd entry point)
│   └── analysis/        analysis stage (background, peaks, identify, residual,
│                        heatmap, ML ranking/training — see CLAUDE.md for the
│                        full module map and HDF5 schemas)
├── tests/               import test + headless smoke test
├── examples/            example calibration_session_config.json (schema reference)
├── environment.yml      conda environment (recommended install route)
└── pyproject.toml       package metadata + pip dependencies
```

Stage convention: pure logic modules + a crash-isolated `worker.py` + an
embeddable `gui.py` pane. Logic stays importable and headless so stages can
also run as batch jobs without any GUI.

## Installation

Recommended (conda; pyFAI installs most reliably from conda-forge):

```bash
conda env create -f environment.yml
conda activate bulkxrd
```

Or with pip (pyFAI wheels are available for most platforms):

```bash
pip install -e .          # core dependencies
pip install -e ".[io]"    # + optional tifffile/imageio/h5py readers
```

`tkinter` must be available in your Python (it ships with python.org and
conda-forge Python; some Linux distros need `python3-tk`).

## Usage

### Unified application (primary)

```bash
bulkxrd --workspace <dir>          # after pip install -e .
# or without installing:
python -m bulkxrd.app --workspace <dir>
```

Opens one window with **Calibration** and **Reduction** tabs (Analysis
planned). The workspace folder holds the stage configs and all outputs. On
first launch the configs are auto-created with sensible defaults.

The GUI embeds both stage panes in one process; heavy pyFAI work still runs in
`worker.py` subprocesses (one per stage) so a worker crash never affects the
host window or the other stage.

### Per-stage standalone GUIs

Each stage also has a standalone entry point for advanced use:

```bash
bulkxrd-calib-gui  --config <path/to/calibration_session_config.json>
bulkxrd-reduce-gui --config <path/to/reduction_session_config.json>
```

### Detector-image diagnostic

```bash
bulkxrd-inspect <image_file>
# or:
python -m bulkxrd.core.inspect <image_file>
```

### Headless analysis + ML training

```bash
bulkxrd-analyze reduced.h5 --phases Au,Re          # Steps 1-3a, no GUI
bulkxrd-analyze reduced.h5 --ml-rank               # candidate-free: rank whole library
bulkxrd-ml-train --workspace <dir> --out scorer.pt # train the learned scorer
```

Training the Step-3b learned scorer (data collection, environment setup,
corpus building, validation gates, deployment) is documented in
[`docs/ml-training.md`](docs/ml-training.md).

## Tests

```bash
python tests/test_imports.py   # all modules import cleanly
python tests/smoke_test.py     # headless config round-trip (no pyFAI/display needed)
```

## Documentation

- [`docs/workflow.md`](docs/workflow.md) — end-to-end analysis workflow.
- [`docs/ml-training.md`](docs/ml-training.md) — training, validating, and
  deploying the Step-3b learned scorer (cluster-agnostic).
- [`docs/ml-training-ris.md`](docs/ml-training-ris.md) — WashU RIS-specific
  addendum (LSF jobs, storage paths).

## Roadmap

- [x] Implement `bulkxrd.reduce`: batch integration of sample datasets using
      the accepted calibration handoff.
- [x] Unified `bulkxrd` application: single window hosting Calibration and
      Reduction panes; calib→reduce handoff wired automatically between tabs.
- [ ] Implement `bulkxrd.analysis`: signal attribution on reduced data
      (diamond-spot rejection via robust-vs-mean pattern difference, known
      phase / pressure-marker d-spacing tracking, residual = unknown
      signatures), then peak fitting, pressure markers, EOS.
- [ ] Remove remaining machine-specific default paths (`C:\Research\...`)
      from the config defaults.
- [ ] Move tests to `pytest` so they can run in CI (GitHub Actions).
- [ ] Choose a license (check university/group policy) and add `LICENSE`.
- [ ] Add a `CITATION.cff` so the software can be cited.
