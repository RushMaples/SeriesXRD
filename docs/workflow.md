# bulkxrd workflow guide

End-to-end guide to running a powder-XRD series through bulkxrd: calibrate,
reduce, analyze. Written for a pressure series (diamond-anvil cell), but the
same stages and knobs apply to a temperature series, a time series, or a
spatial mapping scan — the series axis is just a different column of
per-frame metadata. Where something is pressure-specific, it says so.

Covers the GUI (`bulkxrd`) and the CLI (`bulkxrd-analyze` and friends), with
emphasis on parameter tuning and file management. For the ML/benchmark
tooling (training a learned candidate scorer, benchmarking against labelled
data), see [`docs/ml-training.md`](ml-training.md) — this guide only
references it.

## Contents

1. [Overview](#1-overview)
2. [Quick start](#2-quick-start)
3. [Stage guides](#3-stage-guides)
4. [Parameter tuning](#4-parameter-tuning)
5. [Series metadata](#5-series-metadata-pressure-temperature-time)
6. [Visualization](#6-visualization)
7. [File management](#7-file-management)
8. [Links](#8-links)

---

## 1. Overview

Three stages, run in order. Each stage reads the previous stage's output and
writes one file that hands off to the next stage.

```
raw detector frames + a calibrant image
        |
        v
  [1. CALIBRATE]  calib/gui.py  (bulkxrd-calib-gui)
        - refine geometry against a calibrant (CeO2, LaB6, Si, ...)
        - build a detector mask
        - accept a generation -> writes handoff_for_next_notebook.json
        |  (PONI + mask)
        v
  [2. REDUCE]     reduce/gui.py  (bulkxrd-reduce-gui)
        - apply the accepted calibration to every frame in the dataset
        - azimuthal integration: mean, robust (quantile-band), sigma-clip
        - optional 2D cakes, per-frame thumbnails
        - writes reduced_<session>_<timestamp>.h5
        |  (reduced HDF5: /patterns, /frames, /cakes)
        v
  [3. ANALYZE]    analysis/gui.py  (bulkxrd-analysis-gui / bulkxrd-analyze)
        Step 1  background separation (SNIP + spot residual)
        Step 2  pseudo-Voigt peak fitting
        Step 3a deterministic EOS phase matching (pressure-aware)
        Step 3a-removal  subtract identified phases -> residual
        Step 3b (optional) ML candidate ranking (proposes; 3a still verifies)
        Step 3c unknown-phase clustering of the residual
        - writes <reduced_stem>_analysis.h5
        |
        v
  per-substance heatmaps, pressure/temperature/frame series, ML export
```

| Stage | Reads | Writes | GUI | CLI |
|---|---|---|---|---|
| Calibrate | calibrant image + PONI | `handoff_for_next_notebook.json` (PONI + mask) | `bulkxrd-calib-gui` | none (worker only) |
| Reduce | handoff JSON + dataset folder | `reduced_<session>_<ts>.h5` | `bulkxrd-reduce-gui` | none (worker only) |
| Analyze | reduced `.h5` | `<stem>_analysis.h5` | `bulkxrd-analysis-gui` | `bulkxrd-analyze` |

Calibration and reduction have no polished argument-per-flag CLI the way
analysis does — see [§3.4](#34-headless-driving-of-calibrationreduction) for
what running them without the GUI actually looks like.

The unified launcher `bulkxrd` embeds all three stages as tabs in one window
and wires the handoffs automatically: accepting a calibration fills in the
Reduction tab, and a finished reduction fills in the Analysis tab. The
per-stage entry points (`bulkxrd-calib-gui`, `bulkxrd-reduce-gui`,
`bulkxrd-analysis-gui`) still work standalone if you only need one stage.

## 2. Quick start

### GUI route

```bash
bulkxrd --workspace ~/my_experiment
```

This creates `~/my_experiment/` (default workspace is `~/bulkxrd_workspace`
if `--workspace` is omitted) with a `calibration_session_config.json`,
`reduction_session_config.json`, and `analysis_session_config.json` seeded
inside it, and opens a window with three tabs: **1 Calibration**, **2
Reduction**, **3 Analysis**.

1. **Calibration tab**: point "Calibration image" at a CeO2/LaB6/Si frame and
   "Input PONI" at its geometry file, build a mask, click "Generate QA run",
   inspect the fit, then "Save selected items" on the Final Export tab. This
   writes the accepted-calibration folder and hands the PONI + mask straight
   to the Reduction tab.
2. **Reduction tab**: point "Dataset folder" at the folder of sample frames
   (the whole pressure/temperature/time/mapping series), leave the
   integration settings at their defaults (q-axis, robust + sigma-clip
   channels on), click "Run reduction". The finished `reduced_*.h5` flows
   straight to the Analysis tab.
3. **Analysis tab**: on the **1 Input** tab confirm the reduced file and
   output path, tick "Run Step 1"/"Run Step 2" (on by default), optionally
   enable Step 3a on the **6 Identify** tab with candidate phases picked on
   the **4 Phases** tab, then **7 Run** → "Run analysis". Inspect results on
   **8 Review** / **9 Peak map** / **10 Pattern map** / **11 Grid map**.

### CLI route

The CLI only covers the analysis stage — calibration and reduction still need
a run through their GUIs (or their config JSON + worker script, see
[§3.4](#34-headless-driving-of-calibrationreduction)) at least once to
produce the reduced HDF5. Once you have `reduced_myrun.h5`:

```bash
# Background + peaks only, autodetect settings, all CPUs but one.
bulkxrd-analyze reduced_myrun.h5 --steps 12

# Full pipeline with phase identification for a pressure series.
bulkxrd-analyze reduced_myrun.h5 \
    --steps 123 \
    --phases Au,Re,NaCl-B1 --workspace ~/my_experiment \
    --pressure-csv pressures.csv \
    -o myrun_analysis.h5
```

`--workspace` points at the folder holding your user phase library
(`reference_phases/`) — the same workspace the GUI uses, or any folder you
have added phases to.

## 3. Stage guides

### 3.1 Calibration (`bulkxrd-calib-gui`)

Five tabs, in order:

1. **1 Session & Inputs** — workspace/session paths, calibration image, input
   PONI, calibrant (auto-detected from the image filename when it contains a
   recognized name like `CeO2`/`LaB6`/`Si`), energy/wavelength (kept in
   sync), detector geometry fields (auto-filled by "Inspect PONI"). "Preview
   cake orientation" runs a fast dual-orientation cake so you can pick the
   flip that makes calibrant rings straight vertical lines before committing
   to a full generation.
2. **2 Masking & Inspection** — load the calibration image, build a mask:
   automatic rules (negative/zero/non-finite/saturated-threshold pixels) plus
   manual add/erase polygons, then "Accept Final Mask". The bottom pane
   embeds the Dioptas launcher for a manual geometry refinement round-trip.
3. **3 pyFAI Generate** — 1D bin count, cake radial/azimuthal bin counts,
   integration unit (2θ default here, independent of the reduction stage's
   default), pyFAI method, optional 2θ min/max, coverage threshold. "Auto-set
   bins from image/PONI" derives them from the detector geometry. "Generate
   QA run" runs the actual integration in a worker subprocess.
4. **4 QA Dashboard** — the viewer for each generation: raw/masked detector,
   mask-only, intensity+difference, normalized intensity, cake, and coverage
   diagnostic panels, each independently toggleable, with pan/zoom/replot
   controls. Step through generations with the arrow buttons.
5. **5 Final Export** — pick a generation, check which items to keep (source
   image/PONI, figures, data CSVs, session config), "Save selected items"
   writes the accepted-calibration folder and `handoff_for_next_notebook.json`
   that the Reduction stage consumes.

### 3.2 Reduction (`bulkxrd-reduce-gui`)

Six tabs, in order:

1. **1 Calibration** — the accepted handoff (auto-filled from the
   Calibration stage, or "Use latest calibration" / "Import a previous run…"
   to pull one from an earlier session).
2. **2 Dataset** — the folder of sample frames, file-glob patterns
   (default `*.tif;*.tiff;*.edf;*.cbf;*.mar3450;*.h5`), recursive search
   toggle, "Scan dataset" to preview the file list and count. HDF5/NeXus
   stack containers (Eiger-style master files holding many frames in one
   `.h5`) are expanded automatically — the scan preview shows the true frame
   count, and "HDF5 data path" pins the frame dataset when the auto-detection
   (NeXus `entry/data/data` first, else the largest 3D image dataset) doesn't
   match an unusual layout.
3. **3 Settings** — integration parameters (binning, unit, channels, cakes,
   thumbnails, workers). See [§4.2](#42-reduction) for the ones that matter.
4. **4 Run** — launch the crash-isolated worker subprocess, watch a progress
   bar and log.
5. **5 Review** — inspector for the reduced HDF5: structure summary plus
   overlaid/separated plots of a few sample patterns and (if saved) a cake.
   Two cake-waviness tools live here (both need `save_cakes`):
   "Diagnose waviness" fits the ring wobble r(φ) and reports the amplitude,
   the 1D doublet splitting it causes, and the implied transverse sample
   offset in mm; "Write straightened 1D" writes the rescue channel
   `/patterns/intensity_straightened` for data you can't re-reduce (the
   proper fix is re-refining the geometry on a sample ring).
6. **6 Gallery** — a per-frame cake+1D thumbnail grid; click a frame to
   toggle its `excluded` flag (written straight to the `.h5`, no re-reduction
   needed). Needs `make_thumbnails=True` from the run.

### 3.3 Analysis (`bulkxrd-analysis-gui`)

Eleven tabs. 1–6 configure a run, 7 runs it, 8–11 inspect the results.

1. **1 Input** — reduced HDF5 path, output analysis HDF5 path (auto-derived
   as `<stem>_analysis.h5`, editable), "Inspect input" (structure report +
   warns if `intensity_robust` is missing — Step 1 needs it).
2. **2 Background** — Step 1 on/off, SNIP `max_half_window`, `n_passes`, LLS
   toggle, contamination threshold.
3. **3 Peaks** — Step 2 on/off, peak source, sensitivity preset, auto-range
   toggle, and the advanced knobs (blank = follow the preset). See
   [§4.3](#43-peak-fitting-step-2).
4. **4 Phases** — the reference-phase library table (bundled + your workspace's
   user phases). Click the ✓ column to toggle a phase as a candidate for
   Step 3a; "Add phase…"/"Edit…"/"Remove" manage user phases (name, formula,
   category, space group, lattice, isotropic EOS, optional per-axis EOS for
   anisotropic compression); "Import CIF…" parses a CIF via pymatgen (or
   stores it for manual completion if pymatgen is absent).
5. **5 Frame meta** — extract pressures from filenames, import a CSV,
   preview pressure vs. frame, and hand-edit P/σ/T per selected frame in the
   table. See [§5](#5-series-metadata-pressure-temperature-time).
6. **6 Identify** — Step 3a/3b settings: pressure range, match tolerance,
   evidence gates, pressure-prior window, ML candidate ranking. See
   [§4.4](#44-phase-identification-step-3a).
7. **7 Run** — launch the worker, with a worker-count field and a live
   progress bar per phase (Background/Peaks/Identify).
8. **8 Review** — single-frame QC: overlay mean/robust/baseline/clean/
   spot_residual traces, fitted-peak markers (good vs. flagged), the 2D cake
   for that frame (read from the reduced file), and a contamination-vs-frame
   strip. Scrub frames with the slider or spinbox.
9. **9 Peak map** — scatter of every fitted peak's center vs. frame/pressure/
   temperature/time, colored by area/amplitude/FWHM. "Good peaks only"
   filters out flagged fits.
10. **10 Pattern map** — the full pattern waterfall (radial axis × series
    axis), any background-derived source, optional reflection-track overlays
    for enabled phases and per-phase intensity layers (needs Step 3a). Also
    the launch point for "Export ML dataset…" and "Export simulated set…".
11. **11 Grid map** — for mapping runs: refolds the frame series onto its 2D
    scan grid. See [§6.4](#64-grid-map).

Step 3c (unknown-phase clustering, writing `/unknowns`) and the
Williamson-Hall microstructure module (`analysis/microstructure.py`) both run
without a dedicated GUI tab: Step 3c fires automatically in the worker/CLI
after the residual step (as long as it left peaks behind) and its output is
only inspectable via `bulkxrd-inspect` or the HDF5 directly; microstructure
analysis is a Python-API-only module you call yourself on the peak-fit
output (size/strain per frame from FWHM vs. q). Neither is wired into
Pattern map or Grid map.

### 3.4 Headless driving of calibration/reduction

There's no `bulkxrd-calibrate`/`bulkxrd-reduce` console script. What the
GUI's "Generate QA run" / "Run reduction" buttons actually do is launch:

```bash
python -m bulkxrd.calib.worker  --config calibration_session_config.json --generation 1 --output-json out.json
python -m bulkxrd.reduce.worker --config reduction_session_config.json   --output-json out.json
```

against the same JSON the GUI edits. You can hand-edit that JSON (it's a
flat key/value dict — see `examples/calibration_session_config.example.json`
for the schema) and invoke the worker directly to script a run without
opening a window, but this path is not a stable, documented CLI the way
`bulkxrd-analyze` is — flags aren't validated, and the config's exact keys
can change between versions. For repeatable batch pressure-series work,
prefer configuring once in the GUI (it validates as you go) and re-running
the same config.

`bulkxrd-analyze` has no such caveat — its argparse flags are the supported
contract. Note it does **not** read `analysis_session_config.json`; it is a
fully independent entry point with its own flag set, covering every Step-2
detection knob the GUI has (`--min-snr`, `--min-prominence-snr`,
`--edge-bins`, `--fit-min`/`--fit-max`, `--min-fwhm-bins`,
`--detrend-bins`). For anything beyond that, call
`bulkxrd.analysis.peaks.run_peak_fitting(...)` /
`bulkxrd.analysis.worker.run_analysis(config_dict)` from your own script.

### 3.5 Live mode during a beamtime (`bulkxrd-watch`)

```bash
bulkxrd-watch --workspace ~/my_experiment            # needs an accepted calibration
bulkxrd-watch --workspace ~/my_experiment --steps 123  # live phase ID too
bulkxrd-watch --workspace ~/my_experiment --steps ''   # reduce only
```

Polls the configured dataset folder (`--poll 5` seconds) while frames are
still being collected, integrates each new frame once it settles, and
appends it to a growing `reduced_<session>_<ts>_live.h5`. After every batch
(`--analyze-every N` to thin this out) the analysis worker re-runs the
chosen steps against the live file, so opening the analysis GUI on it shows
current Review/Peak map/Pattern map views mid-run. Ctrl-C (or `--idle-exit
30` minutes without a new frame) ends the watch with a final analysis pass.

What it handles: plain image files (processed only after their size/mtime
is stable across two polls), growing HDF5 stacks (new frames picked up per
poll; the newest frame of a still-growing stack waits one poll so a
half-written chunk is never read), NeXus metadata (timestamps/positions/
temperature harvested per batch), and transient read failures (3 retries,
then the frame is marked failed and skipped).

What it deliberately does not do: cakes and gallery thumbnails are skipped
for speed, and the live file is appended in place rather than
written-tmp-and-replaced — a hard kill mid-append can corrupt the live file
(never an archival one). Frame order is arrival order. When the run is
over, do a normal full reduction for the archival file; the live file is a
working view.

**Other command-line tools.** `bulkxrd-texture reduced.h5` writes per-ring
azimuthal texture metrics (`/texture`: texture index, spot fraction,
preferred-orientation harmonic) from a cakes-enabled reduction.
`bulkxrd-export-refinement analysis.h5 out_dir` writes a Rietveld hand-off
bundle (patterns as `.xy`, phase CIFs, GSAS-II `instrument.instprm`, README
with a GSASIIscriptable snippet). `bulkxrd-analyze --fractions` adds
semi-quantitative intensity-share phase fractions (`/fractions`) after the
residual step — see `analysis/fractions.py`'s docstring for what those
fractions do and do not correct.

## 4. Parameter tuning

This is the centerpiece. For each stage: the parameters that matter, their
defaults, when to change them, and in which direction. A troubleshooting
table follows each stage's knobs.

### 4.1 Calibration

| Parameter | Default | Change it when |
|---|---|---|
| `npt_1d` (1D bins) | auto (~1 bin/pixel of radial extent) | Rarely — "Auto-set bins from image/PONI" derives it from geometry. |
| `npt_radial` / `npt_azimuthal` (cake bins) | auto / 360 | Increase azimuthal bins for a finer waviness diagnosis; rarely needed otherwise. |
| `coverage_threshold_pct` | 10 | Raise it if the coverage diagnostic shows too many low-statistics radial bins passing through as real signal; lower it if real high-angle data is getting zeroed. |
| `saturated_threshold` | blank (off) | Set it to your detector's saturation counts if hot/saturated pixels are visible in the raw detector panel and not caught by the negative/zero/non-finite rules. |
| cake orientation (flip) | ON (Dioptas alignment) | Use "Preview cake orientation" whenever calibrant rings look wavy in the QA cake — pick whichever flip renders them as straight vertical lines. |

### 4.2 Reduction

| Parameter | Default | Change it when |
|---|---|---|
| `npt_1d` | blank = auto (~1 bin/pixel of radial extent, pyFAI's rule of thumb) | Leave blank normally. If Step 2 later warns "peaks are UNDERSAMPLED" with a `median FWHM < 4 bins`, re-reduce with the `npt_recommended` value the warning prints. An explicit value well below the geometric suggestion (< 0.7×) triggers its own warning at reduce time — too few bins makes patterns look stepped/blocky and degrades every downstream fit. |
| `unit` | `q_A^-1` | Keep in q for series work — peak widths are roughly constant in q so window sizing is uniform across the pattern, and d-conversion downstream needs no wavelength. Switch to `2th_deg` only for parity with tools that expect 2θ. |
| `robust_1d` / `robust_quant_halfwidth` | on / `0.05` (45–55% azimuthal quantile band) | Keep on — Step 1 requires `intensity_robust`. Only touch the half-width if you understand the tradeoff: smaller keeps more spot-rejection power but pushes toward the pure-median quantization staircase; `0` explicitly requests the pure median (do this only if you've confirmed your pyFAI build actually honors the quantile-band kwargs — see the troubleshooting row below). |
| `sigmaclip_1d` / `sigmaclip_thresh` / `sigmaclip_maxiter` | on / `3.0` / `5` | Keep on — it's the recommended Step-2 fit source (`source="sigmaclip"`/`"auto"`) for spotty/textured rings. Lower `sigmaclip_thresh` (more aggressive rejection) only if diamond spots are still leaking into it; raise `sigmaclip_maxiter` if convergence looks incomplete on very spotty data. |
| `azimuth_range` | blank (full azimuth) | **Config-file only** — not exposed in the Reduction Settings tab. A `"min,max"` degree sector applied to all three 1D channels alike (mean/robust/sigmaclip must all see the same pixels, or `spot_residual = mean − robust` stops meaning anything). Stopgap for a wavy/tilted ring when re-calibration on a sample-position ring isn't possible; needs a pyFAI whose robust integrators accept `azimuth_range`. Cakes stay full-azimuth regardless (they're the waviness diagnostic). |
| `save_cakes` / `cake_every` | off / `1` | Turn `save_cakes` on if you plan to use the Review tab's "Diagnose waviness" / "Write straightened 1D" buttons (ring-waviness diagnosis + rescue) or want per-frame 2D cakes in Review/the Analysis Review tab. `cake_every > 1` samples every Nth frame to bound file size on a long series. |
| `make_thumbnails` | on | Turn off for very large datasets to save time/disk — you lose the Gallery tab's per-frame previews (results are unaffected). |
| `num_workers` | `0` (auto = CPU count − 1) | Set to `1` for deterministic serial runs (easier debugging) or lower than auto if the workstation is shared. |

**Troubleshooting — reduction**

| Problem | Likely cause | Knob |
|---|---|---|
| Low-intensity patterns look like clean staircases | Robust channel fell back to a pure median (your pyFAI ignores the quantile-band kwargs — check the reduce log for "median(band_unsupported)") | Upgrade pyFAI, or fit downstream on `sigmaclip`/`mean` instead of the median-derived channels. |
| Every peak in the pattern is a constant-splitting double-horned doublet | Ring waviness — sample measured off the calibrant's position (routine in a DAC where the calibrant sits outside the cell) | `save_cakes=True`, then Review tab → "Diagnose waviness" (reports the amplitude, doublet splitting, and implied sample offset in mm); re-refine the PONI on a sample-position ring and re-reduce, or Review tab → "Write straightened 1D" as a rescue path on already-collected data. |
| Analysis Step 1 refuses to run: "lacks patterns/intensity_robust" | `robust_1d` was off for this reduction | Re-run reduction with `robust_1d=True`. |
| Step-2 log warns peaks are undersampled | `npt_1d` too low for the peak widths actually present | Re-reduce with the printed `npt_recommended` value (or leave `npt_1d` blank next time). |

### 4.3 Peak fitting (Step 2)

| Parameter | Default | Change it when |
|---|---|---|
| `peak_source` | `auto` (reduce-side `sigmaclip` if present, else analysis-side `hybrid`) | If peaks you can clearly see in the pattern are missing from the fit, try `hybrid` or `mean` — the azimuthal-median-based channels (`clean`, and by extension the median foundation under `hybrid`/`sigmaclip`) suppress diamond spots but also drop real intensity on spotty/textured/incomplete Debye rings. `sigmaclip` is the most principled recovery (needs `sigmaclip_1d` from reduction); `mean` keeps everything including diamond spots (diagnostic, not recommended as a final source). `auto` already falls through to `mean` when Step 1 diagnosed a **spotty/coarse-grained sample** (`signal_frac_clean` low, `spotty_sample=True` in the Step-1 log) — a near-single-crystal DAC load where the median channels reject the sample itself. |
| `sensitivity` | `normal` (min_snr=5, min_prominence_snr=2, min_fwhm_bins=2, edge_bins=5) | `sensitive` (min_snr=3.5, min_prominence_snr=1.5, min_fwhm_bins=2, edge_bins=4) catches weak shoulders at the cost of more noise hits. `conservative` (min_snr=6, min_prominence_snr=3, min_fwhm_bins=3, edge_bins=6) gives fewer, cleaner peaks. Any of the four knobs below left blank follows the chosen preset; an explicit value always overrides it. |
| `min_snr` | preset (5.0 for normal) | Peak height threshold in noise-floor (MAD) units. Lower it if real, visibly-present peaks aren't being detected at all; raise it if noise bins are showing up as peaks. |
| `min_prominence_snr` | preset (2.0 for normal) | Prominence threshold, decoupled from height because prominence is measured against the taller neighbor — a real peak on the shoulder of a stronger one has low prominence even with fine height. Lower it to keep shoulder/adjacent peaks. |
| `min_fwhm_bins` | preset (2.0 for normal) | Rejects sub-resolution single-bin spikes as noise. If *real* peaks trip this, the pattern is under-sampled — re-reduce with more `npt_1d` bins rather than lowering this. |
| `edge_bins` | preset (5 for normal) | Bins excluded from detection at either end of the pattern (kills beamstop-onset and detector-truncation artifacts). Raise it if edge artifacts are still leaking through as spurious peaks. |
| `window_factor` | `3.0` | Fit-window half-width as a multiple of the estimated FWHM. Structural, not part of the sensitivity preset — rarely needs changing. |
| `max_chi2` | `25.0` | Reduced χ² above which a fit is flagged `FLAG_BAD_CHI2`. Tighten (lower) for a cleaner peak map at the cost of more rejected fits; loosen if visibly good fits are being flagged bad. |
| `auto_range` / `fit_min` / `fit_max` | on / blank / blank | Leave `auto_range` on and both bounds blank for the conservative automatic trim (skips the beamstop ramp and dead/noisy tail, capped at ~15% of the axis per end). Set `fit_min` above the beamstop onset explicitly if the low-angle ramp is still inflating the noise floor and hiding weak peaks; set `fit_max` below a noisy detector tail similarly. |
| `hybrid_spike_bins` | `5` | Only matters with `peak_source=hybrid`. Radial width (bins) below which the azimuthal-mean excess is treated as a diamond spike and removed; above it, kept as real ring texture. Lower it if diamond spikes are still bleeding through into `hybrid`; raise it if it's clipping genuinely broad real texture. |
| `detrend_bins` | `81` (GUI and `bulkxrd-analyze --detrend-bins`; the bare `run_peak_fitting()` function default is `0`=off) | Detection-only local-baseline window: removes residual broad background SNIP left behind so weak peaks clear the noise threshold (fitting still uses the un-detrended signal). Size it to a few peak widths; `0` disables it. |
| `propagate_seeds` | on | Keep on for series data — seeds each frame's detection with the previous frame's good peak centers so a reflection keeps its identity as the lattice compresses/expands. Turn off only if you suspect seed leakage is masking a genuine peak disappearance/transition. |

**Troubleshooting — peak fitting**

| Problem | Likely cause | Knob |
|---|---|---|
| Visible peaks missing from the fit | Median-based source dropped a spotty/textured/incomplete ring | `peak_source = hybrid`, `sigmaclip`, or `mean` |
| Weak shoulders not detected | Sensitivity too conservative | `sensitivity = sensitive`, or lower `min_snr`/`min_prominence_snr` explicitly |
| Noise fitted as peaks | Sensitivity too loose | `sensitivity = conservative`, or raise `min_snr`/`min_prominence_snr`/`min_fwhm_bins`/`edge_bins` |
| Stepped/blocky patterns, poor fits everywhere | Too few `npt_1d` bins at reduction time | Re-reduce with more bins (see the run log's `npt_recommended`) |
| Run log: "peaks are UNDERSAMPLED — median FWHM is only N bins" | Same as above, quantified | Re-reduce with the printed `npt_recommended` |
| Broad real peaks lose height after Step 1 | SNIP `max_half_window` too wide, flattening broad peaks into the baseline | Lower `max_half_window` (Background tab) |
| A cluster of nearby peaks never converges / joint fit is very slow | An oversized chain of marginal detections linked into one group | Handled automatically (`MAX_GROUP_SIZE=12` auto-splits at the widest internal gap); if it's still bad, tighten `sensitivity` so fewer marginal candidates exist to chain together |
| A reflection's identity seems to jump between adjacent frames | `propagate_seeds` off, or unrelated peaks merging | Turn `propagate_seeds` on (default) |

### 4.4 Phase identification (Step 3a)

| Parameter | Default | Change it when |
|---|---|---|
| `identify_all_phases` | off | Turn on ("Search entire library") when you don't know what's in the cell and want every bundled+user phase scored per frame instead of only the Phases-tab selection. Slower, more prone to spurious matches on a couple of coincidental lines. |
| `p_min` / `p_max` | `0` / `100` GPa | Widen if your experiment genuinely spans a higher pressure, or if the log warns it auto-widened the range to cover the metadata pressures + prior window (a sign your bounds were too narrow for the data you actually collected). |
| `rel_tol` | `0.01` (1% of d) | Raise it if real lines are visibly present but just miss their match (recall stays low despite an obviously-correct phase); too loose lets wrong phases claim matches. This is the match tolerance as a *fraction of d-spacing*, before esd-widening. |
| `seen_conf` | `0.5` | The confidence bar above which a phase counts as "present in this frame" for the residual-subtraction step. Raise it to subtract only very confident matches (leaves more in the residual); lower it to subtract more aggressively. Distinct from the Identify tab's plot-only "Min confidence" filter, which only controls what's drawn and defaults to the same `0.5` coincidentally. |
| `use_pressure_prior` | on | Keep on for any pressure series with populated `/frames/pressure` — without it, a wrong phase can slide freely along the whole `[p_min, p_max]` range until a few lines happen to coincide. Needs frame pressures from the Frame meta tab. |
| `pressure_window` | `2.0` GPa | Half-width used when a frame has no per-frame σ. Narrow it (e.g. to `0.5`–`1` GPa) if you trust your pressure marker tightly and want to reject phases that only match by drifting off the true pressure; widen it if a correct phase is being penalized because its fitted pressure legitimately sits a bit off the nominal value (e.g. non-hydrostatic stress, gauge offset). |
| `pressure_sigma_k` | `2.0` | When a frame carries a per-frame pressure σ (from a CSV column), the window becomes `k·σ` instead of the fixed `pressure_window`. Adjust `k` to make the window looser/tighter relative to your quoted uncertainty. |
| `marker_prior` | off | Turn on when you have **no** metadata pressure at all but do have marker-category phases (ruby, a pressure standard) in the library — a first pass fits the markers, then reuses the best marker's per-frame pressure as the prior for everything else. |
| `min_matched` | `3` | Minimum one-to-one matched reflections for a phase to count as present (guards against a 1–2 line coincidence). Raise it for stricter identification on dense/busy patterns; lower it (with care) for phases that only ever show a couple of strong lines. |
| `allow_sparse` | off | Turn on to let phases below `min_matched` still be subtracted in the residual step — appropriate for marker/sparse phases you're confident about but that only ever show 1–2 lines. |
| `intensity_k` | `0.3` | Weight of the soft intensity-agreement factor folded into confidence. `0` = position-only (recommended if DAC texture/preferred orientation is scrambling relative intensities, which is common). Raise toward `1` only if you trust measured intensities in this dataset. |
| `use_frame_temperature` | on | Applies `/frames/temperature` through each phase's thermal-expansion coefficient (`Phase.thermal`), the ambient-pressure analog of the pressure prior. Turn off to treat every frame as ambient temperature (e.g. if your temperature column is unreliable). |
| `run_ml_rank` / `ml_rank_top_k` / `ml_rank_source` / `ml_scorer` | off / `5` / `auto` / blank (cosine) | Turn on for candidate-free identification: ranks the *whole* library per frame by similarity to a simulated pattern at that frame's pressure, and only the top-K get verified by the deterministic matcher above. `ml_rank_source=auto` picks the residual if present else the fit source. `ml_scorer` stays blank/`cosine` unless you've trained and validated a scorer per `docs/ml-training.md` — the deterministic cosine baseline is the default everywhere, and whatever the scorer proposes, Step 3a still verifies it against the physics. |

**Troubleshooting — identification**

| Problem | Likely cause | Knob |
|---|---|---|
| A phase is confidently "present" off one or two coincidental lines | `min_matched` too low, or `seen_conf` too low | Raise `min_matched`, raise `seen_conf` |
| A wrong phase matches by sliding to an implausible pressure | No pressure prior in effect | Turn on `use_pressure_prior`, populate `/frames/pressure` (Frame meta tab), consider narrowing `pressure_window` |
| A phase you're sure is present shows low recall/confidence | `rel_tol` too tight | Raise `rel_tol` |
| A correct phase gets penalized on a frame where you know conditions are non-ideal | `pressure_window` too narrow for real non-hydrostatic spread | Widen `pressure_window` (or `pressure_sigma_k` if using per-frame σ) |
| Log warning: "no Birch-Murnaghan EOS for [...]" | Phase has no EOS entered — it's evaluated at ambient only | Add `V0`/`K0`/`K0'` on the Phases tab if you need it to track pressure |
| Log warning: pressure range auto-widened | `p_min`/`p_max` too narrow for the metadata pressures actually present | Widen `p_min`/`p_max` yourself, or accept the auto-widened range |
| ML rank / phase layers / reflection tracks are grayed out or refuse to run | pymatgen not installed | `pip install pymatgen` (or `pip install -e .[phases]`) |

## 5. Series metadata (pressure, temperature, time)

Every phase-identification and series-plot feature reads from these
`/frames` channels: `pressure` (GPa), `pressure_sigma` (GPa), `temperature`
(K), and — for mapping scans — the stage positions `pos_x`/`pos_y` (any
consistent unit, typically mm). There's also `timestamp` (ISO 8601 strings),
carried through from the reduce stage when the raw frame metadata had it,
used only for the "time" series axis. All of this lives on the **5 Frame
meta** analysis tab.

**Filename parsing** ("Extract from filenames"). This is a generic mechanism,
not tied to any one beamline's naming scheme: it recognizes **any**
`<number><unit>` token in the frame's basename, falling back to the nearest
parent folder if the basename has none. For example, a DAC session named
`UOTe-1GPa-001.tif` parses to 1.0 GPa — that is just one worked example, not
an assumed convention; the same parser handles `sample-1p5GPa` → 1.5 GPa (the
`p`-as-decimal convention), `3p9GPa` → 3.9 GPa, `500MPa` → 0.5 GPa, `10kbar` →
1.0 GPa, or any other prefix your facility's file-naming convention happens to
put around the number. Units GPa/MPa/kPa/Pa/kbar/bar convert as expected;
`Mbar` is read as *megabar* (100 GPa) rather than millibar — a DAC filename
token is essentially never millibar. This runs automatically at Step 1
already (populating `/frames/pressure` before you ever open this tab), so
"Extract from filenames" is mainly there to re-run it after a `replace`-style
override, or to check what got parsed. If your facility encodes pressure,
temperature, or other conditions somewhere other than the filename (a log
file, a beamline database export, a separate scan record), skip filename
parsing and use the CSV import below instead — it is the general seam for
"metadata lives somewhere else."

**CSV import** ("Import CSV…", also `--pressure-csv` on the CLI). Column
headers are matched case-insensitively against these aliases:

| Canonical field | Accepted header names |
|---|---|
| `frame` (0-based index) | `frame`, `frame_index`, `index`, `idx`, `i`, `n` |
| `filename` | `filename`, `file`, `name`, `fname`, `path` |
| `pressure` (GPa) | `pressure_gpa`, `pressure`, `p_gpa`, `p`, `gpa` |
| `pressure_sigma` (GPa) | `pressure_sigma_gpa`, `pressure_sigma`, `sigma_gpa`, `sigma`, `p_sigma`, `dp`, `p_err` |
| `temperature` (K) | `temperature_k`, `temperature`, `temp_k`, `temp`, `t_k`, `t` |
| `pos_x` (stage x) | `pos_x_mm`, `pos_x`, `x_mm`, `x`, `sam_x`, `sample_x`, `motor_x`, `samx` |
| `pos_y` (stage y) | `pos_y_mm`, `pos_y`, `y_mm`, `y`, `sam_y`, `sample_y`, `motor_y`, `samy` |

Each row needs at least one value column plus either `frame` or `filename`
to key it (`frame` wins if a row somehow has both) — a positions-only or
temperature-only sheet is fine. Rows key by exact match, then
basename, then stem, so a CSV built from your raw filenames (with or without
extension, with or without a leading path) will map cleanly. Import
**merges** by default: only frames the CSV actually provides get overwritten,
so a partial correction sheet for a handful of frames won't erase every other
frame's pressure. `import_csv_to_analysis(..., replace=True)` (Python API
only — not exposed in the GUI or CLI) wipes the whole channel and writes the
CSV verbatim, leaving un-listed frames at NaN.

**Stage positions from the frame headers** ("Read X/Y from headers…"). For
mapping scans whose raw frames carry motor positions in their image headers
(EDF/CBF and similar, read via fabio): give the two header key names —
matched case-insensitively, and the `motor_mne`/`motor_pos` paired-list
convention is resolved too — and every frame's position is written to
`/frames/pos_x`/`pos_y`. "List keys" shows what the first frame's header
actually contains, and a failed import lists the available keys in its error
message. Point "Frames folder" at the dataset directory when the stored
filenames are bare names rather than full paths. The Python API is
`frame_metadata.import_positions_from_headers(analysis_h5, key_x, key_y,
search_dir=...)`. Once positions exist, the Grid map's `coordinates` layout
(§6.4) places frames automatically.

**Manual editing.** The per-frame table on the Frame meta tab lets you
select one or more rows and type a P/σ/T value into the editor row above
"Apply to selected" — useful for a handful of frames a ruby-fluorescence or
membrane-gauge reading corrected by hand. Blank fields in the editor are left
unchanged on the selected frames.

**Edits persist.** Values you set by hand (and frames a CSV provided) are
marked `user` in the table's Src column (`/frames/user_edited` on disk).
Marked frames are skipped by "Extract from filenames", and a Step-1 re-run
carries them into the rebuilt analysis file (matched by filename) — so a
correction to a mistyped filename pressure (e.g. `50p7GPa` that should have
been 5.27) stays fixed no matter how often you re-run. The explicit reset is
`extract_to_analysis(..., replace=True)` (Python API), which re-parses every
frame and clears the marks. If identification ever widens its pressure range
to cover the metadata (the `[IDENTIFY] WARNING: widening pressure range`
log line), the frames responsible are now listed by name right below it,
with an outlier hint when one value sits far off the series median.

**How it feeds downstream.** `pressure`/`pressure_sigma` drive the Step-3a
pressure prior (§4.4); `temperature` drives the thermal-expansion seam
(`use_frame_temperature`); any of `frame`/`pressure`/`temperature`/`time` can
be picked as the x-axis on the Peak map, Pattern map, and (for `pressure`/
`temperature` as a per-frame *value*, not axis) Grid map tabs (§6).

## 6. Visualization

### 6.1 Review (analysis tab 8)

Single-frame QC. Scrub through frames with the slider/spinbox; toggle which
traces are drawn (mean, robust, baseline, clean, spot_residual, fitted
peaks, and the 2D cake for that frame — pulled from the *reduced* file, since
cakes don't live in the analysis HDF5). Fitted peaks are marked as vertical
lines, colored by their flag (good vs. rejected). A contamination-vs-frame
strip along the bottom always shows where the current frame sits relative to
the whole series' diamond-spot contamination.

### 6.2 Peak map (analysis tab 9)

A scatter of every fitted peak's center vs. an independent variable
(frame/pressure/temperature/time), colored by area, amplitude, or FWHM. "Good
peaks only" hides flagged fits. This is the rawest series view — no phase
attribution, just what Step 2 found.

### 6.3 Pattern map (analysis tab 10)

The Hrubiak/XDI-style full pattern waterfall: radial axis on one axis, the
chosen independent variable on the other, intensity as color. Source is any
of `clean`/`hybrid`/`robust`/`mean`/`sigmaclip`/`baseline`/`spot_residual`
(all reconstructed on demand from the stored `clean`+residual channels, no
extra disk cost). "Overlay reflection tracks" draws each enabled phase's
predicted line positions (using its Step-3a pressure track, or the raw frame
metadata if 3a hasn't run) across the frame axis only — tracks don't draw on
a physical (pressure/temperature/time) x-axis because a non-uniform axis
requires sorting frames, which breaks the frame-index curve. "Show phase
layers" adds a second panel: per-phase ROI-integrated intensity vs. the
chosen x-variable, the filterable per-substance signal this whole pipeline
exists to produce. Both need Step 3a to have run and pymatgen to be
installed; the waterfall itself needs neither.

"Export ML dataset…" and "Export simulated set…" on this tab are entry
points into the Step 3b tooling documented in
[`docs/ml-training.md`](ml-training.md).

### 6.4 Grid map (analysis tab 11)

For mapping runs: refolds the linear frame series back onto the 2D grid it
was physically collected on, colored by a per-frame scalar.

- **Value**: `total`/`max` (integrated or peak intensity of the Step-2 fit
  source, optionally restricted to an ROI via **ROI min/max** on the radial
  axis), `contamination` (the Step-1 spot score), `n_peaks` (fitted peak
  count), `pressure`, `temperature`, or (once phases are enabled and Step 3a
  has run) `phase: <name>` for that phase's matched-reflection intensity.
- **Layout**: how frames are placed on the grid.
  - `coordinates` — automatic. Each frame is placed by its recorded stage
    position (`/frames/pos_x`, `pos_y`; see §5 for how to import them from a
    CSV or the frame headers). Positions are clustered per axis with a
    jitter-tolerant snap, so collection order, serpentine vs raster, and
    missing frames are all irrelevant; axes are in real stage units. Use
    this whenever positions exist.
  - `scan lines` — manual, for series without recorded positions. Set:
    **Frames per line** (how many frames the stage collected before turning
    — must match your raster width/height, no auto-detection), **Scan
    lines** (`horizontal` = rows, `vertical` = columns), and
    **Boustrophedon** (checked = the stage reversed direction every line;
    unchecked = unidirectional. Get this wrong and every other line is
    mirrored). Frame 0 is the top-left of the first scan line.

Hovering the rendered grid shows the underlying frame index and value.

## 7. File management

### 7.1 Workspace layout

A workspace (the folder you point `--workspace` at, or the folder holding
your session config JSONs when running a stage standalone) has this shape
once you've run all three stages once:

```
<workspace>/
  calibration_session_config.json     stage-1 config  (gitignored, local paths)
  reduction_session_config.json       stage-2 config  (gitignored, local paths)
  analysis_session_config.json        stage-3 config  (gitignored, local paths)
  reference_phases/                   user phase library (gitignored)
    user_phases.json
    <imported CIFs>
  data/
    raw/                              (unused by default; frames stay where you put them)
    processed/reduction_<session>/
      reduced_<session>_<ts>.h5
      reduced_<session>_<ts>.manifest.json
      reduced_<session>_<ts>_previews/       (per-frame gallery thumbnails, if enabled)
      <reduced_stem>_analysis.h5             (default analysis output, beside the reduced file)
  figures/                            calibration QA figures (per generation)
  metadata/
    <workflow_name>/genNNN/*_metadata_*.json      calibration QA generation records
  accepted_calibrations/
    accepted_calibration_<session>_<ts>/
      source_raw/                     verbatim copies of the source image + input PONI
      accepted_calibration/
        accepted_calibration.poni     tuned geometry (overrides baked in, not a verbatim copy)
        accepted_mask.npz
        accepted_mask_preview.png
      data/                           intensity/difference/coverage CSVs, cake NPZ, master CSV
      figures/                        the QA figure set for the accepted generation
      metadata/
        report_txt, metadata_json
        calibration_session_config.json      (snapshot at accept time)
        master_metadata.json
        handoff_for_next_notebook.json       <-- this is what the Reduction stage reads
  logs/bulkxrd/
    reduce_<ts>.json                  reduction worker manifest
    analysis_<ts>.json                analysis worker manifest
    worker_preview_<ts>.json          cake-orientation preview output
```

The session config JSONs *are* your saved parameter tuning — every field you
set in a GUI tab lives in one of these three files, keyed by the names used
throughout §4 (e.g. `max_half_window`, `peak_source`, `pressure_window`).
Re-opening the same workspace restores every knob exactly where you left it.

### 7.2 What's safe to delete / regenerate

| Path | Safe to delete? | Notes |
|---|---|---|
| `logs/bulkxrd/*.json` | Yes | Worker manifests only — the actual results live in the `.h5` files, not here. |
| `*_previews/` (reduction gallery thumbnails) | Yes | Loses the Gallery tab's per-frame images; re-run reduction with `make_thumbnails=True` to regenerate, or just live without them — analysis results are unaffected. |
| `metadata/<workflow>/genNNN/...` (calibration QA generations you haven't accepted) | Yes, once you've accepted the generation you want | Costs pyFAI compute time to regenerate if you need to go back to an earlier trial. |
| `*.tmp` files | Yes, if any are ever left behind | Every HDF5/JSON write in this pipeline is atomic (`.tmp` file + `os.replace`) — a `.tmp` should never persist after a normal run or a caught exception. A stray one can only survive a hard kill mid-write (e.g. `kill -9`); it's an incomplete write and safe to remove. |
| `reduced_*.h5` | No — this is the reduction stage's entire output | Re-generating it re-runs the full integration over every frame. Keep it; it's the input every analysis run starts from. |
| `<stem>_analysis.h5` | Regenerable from the reduced file, but keep it if you've spent time tuning Step 2/3 parameters or have run Step 3a/ML export | Re-running the analysis worker rebuilds it from scratch (each step is not free — Step 3a in particular can be slow with many candidate phases). |
| `accepted_calibrations/.../handoff_for_next_notebook.json` + its `accepted_calibration/` folder | No | This is the calibration stage's entire deliverable and the reduction stage's only input. Small (a PONI + a mask + a couple of PNGs); back it up with your data. |
| `reference_phases/user_phases.json` + imported CIFs | No | Your hand-entered/imported phase library. Not regenerable without re-entering every phase. |

### 7.3 HDF5 atomic-write behavior

Every stage writes its `.h5` output as `<name>.h5.tmp`, then calls
`os.replace()` to the final name only after the write completes without
error; any exception mid-write deletes the `.tmp` instead of leaving a
truncated file. So a `reduced_*.h5` or `<stem>_analysis.h5` on disk is always
either complete or absent — never partially written. Session config JSONs
follow the same pattern.

For the analysis HDF5 specifically: Step 1 creates the file from scratch;
Steps 2, 3a, the residual step, and Step 3c each **copy the current file to a
new `.tmp`, add/replace their one group** (`/peaks`, `/identify`,
`/residual`, `/unknowns` respectively), **and atomically replace** — so
interrupting, say, Step 3a leaves the file exactly as Step 2 left it (peaks
intact, no `/identify` group), never a half-written `/identify`. This does
mean the file is rewritten in full at every step, so a very large series with
saved cakes can take a moment per step even when only one small group is
being added — cakes live in the *reduced* file, not the analysis file, so
they aren't part of this copy.

### 7.4 Gitignored runtime files

If your workspace happens to sit inside this repository (not the normal
case — a workspace is ordinarily an arbitrary folder outside the repo), note
that `.gitignore` excludes essentially everything a run produces: all three
session config JSONs, `handoff_for_next_notebook.json`, `master_metadata.json`,
`last_preflight.json`, `data/`, `figures/`, `metadata/`, `logs/`, `previews/`,
`reference_phases/`, and raw data extensions (`*.tif`, `*.tiff`, `*.edf`,
`*.npy`, `*.npz`, `*.csv`). The schema for a session config is documented by
example at `examples/calibration_session_config.example.json`. None of this
is committed on purpose — these files hold local absolute paths and, in the
case of raw data, can be arbitrarily large.

## 8. Links

- [`docs/ml-training.md`](ml-training.md) — training, validating, and
  deploying the Step-3b learned candidate scorer (`bulkxrd-ml-train`), the
  `bulkxrd-benchmark` known-truth harness (RRUFF/opXRD labelled patterns vs.
  the cosine baseline), and `bulkxrd-corpus` (training-only CIF corpus
  tooling). Covers the data-quality gate you should run before training on
  any dataset (cake waviness, sampling, channel diagnosis, robust-channel
  provenance) — the same diagnostics referenced in §4.2's troubleshooting
  table.
- [`docs/ml-training-ris.md`](ml-training-ris.md) — a worked example of a
  site-specific cluster addendum (WashU RIS: LSF job syntax, storage paths)
  to the same training pipeline; use it as a template for your own cluster's
  notes if you need one, not as a requirement.
- [`docs/roadmap.md`](roadmap.md) — implemented vs. planned features, and the
  "Site adoption" section covering what a new facility needs to provide.
