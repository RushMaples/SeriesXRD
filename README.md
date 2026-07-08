# bulkxrd

GUI-driven workflow for powder X-ray diffraction: detector calibration review,
dataset reduction, and pattern analysis. Facility-neutral by design — it works
the same way for a synchrotron beamline or a lab (in-house) diffractometer,
any calibrant, any detector pyFAI supports, and any beamline-specific frame
naming or metadata convention (see "Site adoption" in
[`docs/roadmap.md`](docs/roadmap.md) for exactly what a new site needs to
supply). A single unified desktop application (`bulkxrd`) hosts all pipeline
stages in one window. Heavy pyFAI work still runs in crash-isolated
`worker.py` subprocesses so a pyFAI or matplotlib crash never takes down the
GUI.

## Pipeline

The workflow is one subpackage per stage, communicating only through artifacts
on disk plus a shared workspace folder:

1. **`bulkxrd.calib`** — calibration review (implemented): standard image →
   accepted `.poni` + mask + QA record, ending in a
   `handoff_for_next_notebook.json` (internal artifact passed automatically to
   the Reduction tab — not a user-facing step).
2. **`bulkxrd.reduce`** — dataset reduction (implemented): apply the accepted
   geometry/mask to a sample dataset, parallel batch azimuthal integration →
   1D patterns (mean, azimuthal-quantile-band "robust", and optional
   sigma-clipped trimmed mean) and optional 2D cakes in one HDF5 file + JSON
   manifest. Frame sources include plain images and HDF5/NeXus stack
   containers (Eiger-style master files), with per-frame metadata (timestamp,
   stage position, temperature) harvested automatically. `bulkxrd-watch`
   adds a live mode that reduces and periodically re-analyzes a dataset
   folder while frames are still being collected.
3. **`bulkxrd.analysis`** — pattern analysis: SNIP background + diamond-spot
   separation (Step 1), pseudo-Voigt peak fitting (Step 2), pressure-aware
   EOS phase identification + residual removal (Step 3a), the ML
   candidate-ranking seam (Step 3b: deterministic cosine ranker by default,
   optional learned scorer — see `docs/ml-training.md`), and unknown-phase
   clustering of the leftover residual (Step 3c). Semi-quantitative phase
   fractions, azimuthal texture metrics, and a Rietveld hand-off export round
   out the tooling.

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
│   │   ├── io.py            detector image readers (fabio/tifffile/PIL) and
│   │   │                    HDF5/NeXus frame-stack ingestion (Eiger master files)
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
│   │   ├── straighten.py    cake-waviness diagnosis + straightened-1D rescue channel
│   │   ├── texture.py       azimuthal texture metrics per saved cake (bulkxrd-texture)
│   │   ├── watch.py         live (during-beamtime) reduction + rolling analysis (bulkxrd-watch)
│   │   ├── gui.py           tabbed Tkinter GUI (embeddable pane)
│   │   └── run_gui.py       CLI entry point (bulkxrd-reduce-gui)
│   ├── app.py           unified application (bulkxrd entry point)
│   └── analysis/        analysis stage (background, peaks, identify, residual,
│                        heatmap, ML ranking/training — see CLAUDE.md for the
│                        full module map and HDF5 schemas)
├── tests/               25 standalone test modules (each runnable directly, see "Tests")
├── examples/            calibration_session_config.example.json (schema reference),
│                        fetch_benchmark_example.sh (downloads a real-data
│                        bulkxrd-benchmark example set — see docs/ml-training.md)
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
pip install -e .              # core dependencies, including h5py
pip install -e ".[io]"        # + optional tifffile reader
pip install -e ".[stacks]"    # + hdf5plugin for compressed Eiger/HDF5 stacks
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

Opens one window with **1 Calibration**, **2 Reduction**, and **3 Analysis**
tabs. Accepting a calibration hands its PONI + mask to the Reduction tab
automatically; a finished reduction hands its output HDF5 to the Analysis
tab automatically — the handoffs are automatic in both directions, not a
manual file-picking step. The workspace folder holds the stage configs and
all outputs. On first launch the configs are auto-created with sensible
defaults.

The GUI embeds all three stage panes in one process; heavy pyFAI work still
runs in `worker.py` subprocesses (one per stage) so a worker crash never
affects the host window or another stage.

### Per-stage standalone GUIs

Each stage also has a standalone entry point for advanced use:

```bash
bulkxrd-calib-gui    --config <path/to/calibration_session_config.json>
bulkxrd-reduce-gui   --config <path/to/reduction_session_config.json>
bulkxrd-analysis-gui --config <path/to/analysis_session_config.json>   # optional; auto-found if omitted
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

### All console scripts

| Command | Purpose |
|---|---|
| `bulkxrd` | Unified GUI: Calibration + Reduction + Analysis tabs in one window. |
| `bulkxrd-calib-gui` | Calibration stage standalone GUI. |
| `bulkxrd-reduce-gui` | Reduction stage standalone GUI (includes live watch-mode controls). |
| `bulkxrd-analysis-gui` | Analysis stage standalone GUI. |
| `bulkxrd-analyze` | Headless analysis CLI (Steps 1-3, ML ranking, exports). |
| `bulkxrd-watch` | Live reduction + rolling analysis while frames are still being collected. |
| `bulkxrd-ml-train` | Train the Step-3b learned candidate scorer. |
| `bulkxrd-benchmark` | Score a scorer against labelled XY patterns (RRUFF/opXRD-style known-truth harness). |
| `bulkxrd-corpus` | Fetch/screen a training-only CIF corpus for `bulkxrd-ml-train --cif-dir`. |
| `bulkxrd-texture` | Azimuthal texture metrics (`/texture`) from a cakes-enabled reduction. |
| `bulkxrd-export-refinement` | Rietveld hand-off bundle (`.xy` patterns + phase CIFs + GSAS-II instprm) from an analysis HDF5. |
| `bulkxrd-inspect` | Detector-image diagnostic: true format from magic bytes, per-reader interpretation, intensity statistics, and a verdict. |

See [`docs/workflow.md`](docs/workflow.md) for how each of these fits into
the end-to-end pipeline and [`docs/ml-training.md`](docs/ml-training.md) for
the ML-specific ones.

## Tests

25 test modules under `tests/`. Most run standalone with plain `python`
(top-level script execution or an `if __name__ == "__main__":` guard — no
`pytest` needed, and it isn't a dependency of this project today, see the
Roadmap):

```bash
python tests/test_imports.py   # all modules import cleanly
python tests/smoke_test.py     # headless config round-trip (no pyFAI/display needed)
# and similarly for the other 23: test_background.py, test_peaks.py,
# test_identify.py, test_residual.py, test_watch.py, test_ml.py, ...
```

One module, `test_worker_script_bootstrap.py`, uses a `pytest` fixture
(`monkeypatch`) and needs `pytest` installed to actually exercise its test —
running it with plain `python` does nothing silently. This is one of the
reasons the "move tests to pytest" roadmap item exists: standardizing the
suite would also fix this one file's plain-`python` no-op.

## Documentation

- [`docs/workflow.md`](docs/workflow.md) — end-to-end analysis workflow.
- [`docs/group-meeting-workflow.md`](docs/group-meeting-workflow.md) —
  compact presenter notes, live-demo path, and result slots for a group
  meeting workflow presentation.
- [`docs/ml-training.md`](docs/ml-training.md) — training, validating, and
  deploying the Step-3b learned scorer (cluster-agnostic — works on any
  cluster or workstation).
- [`docs/ml-training-ris.md`](docs/ml-training-ris.md) — a worked example of
  a site-specific addendum to the guide above (WashU RIS: LSF job syntax,
  storage paths); write an equivalent short page for your own cluster if it
  needs one.
- [`docs/roadmap.md`](docs/roadmap.md) — implemented vs. planned features,
  and what a new facility needs to provide to adopt bulkxrd.
- [`docs/test-data.md`](docs/test-data.md) — open datasets you can download to
  exercise each stage (calibration frames, measured patterns, CIFs, simulated
  patterns) and which command each one feeds.

## Roadmap

Feature status (implemented vs. planned, and what a new facility needs to
provide to adopt bulkxrd) lives in [`docs/roadmap.md`](docs/roadmap.md), kept
current against the code — this section is only the open engineering chores,
not a feature list:

- [ ] Move tests to `pytest` so they can run in CI (GitHub Actions).
- [ ] Choose a license (check university/group policy) and add `LICENSE`.
- [ ] Add a `CITATION.cff` so the software can be cited.
