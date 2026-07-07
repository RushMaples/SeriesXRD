# Feature roadmap

What bulkxrd does today ("Implemented") and where it's headed ("Planned").
This is a features roadmap — for stage-by-stage usage see
[`docs/workflow.md`](workflow.md); for the ML training pipeline specifically
see [`docs/ml-training.md`](ml-training.md). Every implemented item below was
verified against the current code, not just described from memory.

---

## Implemented

**Scan-grid map with boustrophedon/unidirectional refolding.**
`analysis/heatmap.py` (`frame_grid`, `grid_map`) refolds a linear frame series
back onto the 2D grid it was physically collected on — horizontal or vertical
scan lines, boustrophedon (serpentine, direction reverses every line) or
unidirectional (every line scans the same way). This matters because most
mapping data is collected as one long frame sequence with no inherent 2D
structure recorded; without the refold you can only look at frames in
acquisition order, not as the physical map they came from. The Grid map tab
(analysis tab 11) also supports per-phase intensity as the plotted value
(`phase: <name>`, reading the Step-3a matched-reflection intensity layer for
an enabled phase) alongside total/max intensity, contamination, peak count,
pressure, and temperature — so a finished identification run can be viewed as
a spatial map of where each substance appears, not just a pressure or frame
heatmap.

**Coordinate-based spatial mapping (new in this release).** When per-frame
stage positions exist, the manual raster description above becomes
unnecessary: `/frames/pos_x`/`pos_y` are populated from a positions CSV
(`pos_x_mm`/`pos_y_mm` columns and aliases) or read directly from the raw
frames' image headers ("Read X/Y from headers…" on the Frame meta tab —
case-insensitive keys, ESRF-style `motor_mne`/`motor_pos` pairs resolved,
with a key-listing helper), and the Grid map's `coordinates` layout
(`analysis/heatmap.py` `coordinate_grid`) places every frame by snapping the
jittered motor read-backs onto the regular grid they were commanded on.
Collection order, serpentine vs raster, and missing frames become
irrelevant, and the map axes are real stage units. Positions are marked
user-edited, so they survive Step-1 rebuilds like every other deliberate
metadata input.

**Series x-axis across the map/plot tabs.** `analysis/heatmap.py`'s
`_series_x` and the Peak map / Pattern map / Grid map tabs let you pick
frame index, pressure, temperature, or elapsed time as the independent
variable. This matters because not every series is pressure-driven — a
temperature ramp, a time-resolved reaction, or a spatial scan all want a
different x-axis, and the underlying data model (`/frames/pressure`,
`temperature`, `timestamp`) is series-axis-agnostic by construction.

**Cake-waviness diagnosis + straightened-1D rescue.**
`reduce/straighten.py`, exposed as "Diagnose waviness" and "Write straightened
1D" on the Reduction stage's Review tab. A Debye ring should integrate to a
single sharp peak; if the sample sits off the calibrant's position (routine
in a DAC, common whenever the calibrant and sample aren't measured at the
same physical point), the ring wobbles in azimuth and every peak becomes a
constant-splitting double-horned doublet that no downstream fit or
identification step can explain. `diagnose_reduced` fits the wobble and
reports it in physical units (implied sample offset in mm); `straighten_cake`
recovers single sharp peaks from already-collected data when re-refining the
geometry and re-reducing isn't an option. This matters for identification
accuracy on any dataset where the calibrant and sample were not coincident.

**User-edit provenance for frame metadata.** `/frames/user_edited` (bool per
frame) marks pressure/temperature/sigma values a human set — by hand-editing
the Frame meta table or by CSV import. Marked frames are skipped by
"Extract from filenames" re-parsing, and a Step-1 re-run carries the marks
into the rebuilt analysis file (matched by filename). This matters because
without it, a manual correction to a mistyped or ambiguous filename token
would silently get overwritten the next time the analysis file is rebuilt —
a data-integrity trap for exactly the kind of hand-fixed value you can least
afford to lose track of.

**Known-truth benchmark harness, CIF corpus tooling, and learned-scorer
training.** Three CLIs: `bulkxrd-benchmark` (`analysis/benchmark.py`) ingests
labelled XY patterns (RRUFF, opXRD, or your own) through the real Step-1/2
preprocessing and scores any scorer against the labels (hit@1/hit@K/MRR, plus
Step-3a verify metrics); `bulkxrd-corpus` (`analysis/corpus.py`) fetches CIFs
from COD by ID and screens a CIF directory (parse/dedupe/size-screen) into a
training-only corpus; `bulkxrd-ml-train` (`analysis/ml_train.py`) trains the
Step-3b learned pair scorer against that corpus and exports a TorchScript
model. Together these are the validation gate a trained scorer must clear
before it's trusted over the deterministic cosine baseline — this matters
because an unvalidated learned model in an identification pipeline is a
silent-failure risk, and the benchmark harness is what makes "does this
model actually help" a measurable question instead of a guess.

**Step 3c unknown-phase clustering and Williamson-Hall microstructure.**
`analysis/unknowns.py` links residual (unexplained) peaks into coherent
tracks across the series and clusters tracks that appear/disappear/drift
together (Jaccard co-occurrence) into `/unknowns`, each with a d-spacing
fingerprint and candidate transition frames — this is what turns "leftover
peaks nothing matched" into "a candidate unidentified phase with its own
signature," which is the whole point of a pipeline that has to handle
genuinely unknown substances, not just a closed set of expected phases.
`analysis/microstructure.py` fits per-frame Williamson-Hall size/strain from
the Step-2 peak widths (esd-weighted, with an optional instrument-resolution
correction — uncorrected output is flagged as such), giving crystallite size
and microstrain trends across the series from data the pipeline already
collects.

---

## Planned

**1. HDF5/NeXus master-file ingestion for the reduce stage.** APS/ESRF
Eiger-style detectors write one HDF5 stack (a NeXus master file plus linked
data files) rather than file-per-frame images, and `reduce/processing.py`
currently reads frames one file at a time via `core/io.py`
(fabio/tifffile/PIL). Design: add an input-adapter seam in `core/io` that
yields `(frame, header)` pairs from either a file glob (today's path) or an
HDF5 dataset path/group, so the reduction loop itself doesn't need to know
which kind of source it's iterating. The main design question is how much of
the per-frame header (exposure time, timestamp, motor positions) NeXus
exposes for free versus what still has to come from a sidecar CSV.

**2. Live watch-folder / during-beamtime mode.** Reduce and analyze frames as
they land during an active beamtime, appending to the HDF5 and refreshing the
maps incrementally instead of requiring a complete dataset up front. Design:
the current convention is atomic rebuild — every analysis step copies the
whole file to a `.tmp` and replaces it — which is safe but means a live mode
needs either a genuinely incremental append path (extend datasets in place,
which loses the "always complete or absent" atomicity guarantee for the
duration of a run) or a cheap way to re-run the atomic rebuild often enough
that it feels live on a growing but still-modest frame count. The tradeoff
between these two is the crux of the design.

**3. Quantitative phase fractions.** Start with RIR (reference intensity
ratio) / normalized-intensity fractions using the per-phase intensity layers
that already exist (`analysis/heatmap.py` phase layers, Step 3a matched
reflections); Rietveld-quality quantification comes later, likely via the
refinement export below rather than an in-house Rietveld engine. This
matters because "which phases are present" (today's output) is a different
and often less useful question than "how much of each phase," which is what
most published DAC/phase-transition work actually needs to report.

**4. Refinement hand-off.** Export identified phases plus their patterns as a
GSAS-II project or a `.cif` + `.xy` bundle, so a user can continue with
Rietveld refinement in a dedicated tool rather than trying to get
publication-quality lattice parameters out of bulkxrd's own deterministic
fit. The design consideration is scope discipline: this is an export/bridge
feature, not a Rietveld reimplementation — the bundle needs to carry enough
(phase, space group, approximate lattice parameters, the actual measured
pattern) that GSAS-II or similar can pick up refinement cleanly, without
bulkxrd trying to own that step itself.

**5. Azimuthal texture analysis.** The `/cakes` data (2D intensity vs.
radial × azimuthal angle) already exists whenever `save_cakes=True`; per-ring
intensity vs. azimuth would give texture/preferred-orientation and
differential-stress indicators, building on the same ring-fitting machinery
`reduce/straighten.py` already uses for waviness. This matters because DAC
samples routinely develop texture under compression (which is also why
Step 3a's intensity-agreement weight is deliberately soft — see
`intensity_k` in CLAUDE.md) and texture itself is diagnostic information the
pipeline currently discards rather than surfaces.

**6. Open-set structure search for unknowns.** Take a Step 3c cluster's
d-fingerprint and search it against a COD-derived candidate set using the
same `ml_scorer` seam Step 3b already defines, instead of stopping at "here
is an unidentified cluster of coherent peaks." The design need is a
candidate-simulation cache: naively simulating reflections for a large COD
subset per query is too slow to be interactive, so this needs the corpus
tooling (`analysis/corpus.py`) plus a precomputed/cached simulation layer
searched by approximate d-fingerprint match before any scorer runs.

**7. Multi-detector / multi-geometry sessions.** Support one series measured
across two (or more) detector positions or geometries within a single
analysis — e.g. a wide-angle and a high-angle detector, or a mid-run
detector-distance change. Design: needs a per-frame PONI association (today
one accepted calibration applies to an entire reduced file) threaded through
reduce and into the analysis HDF5's frame metadata, so azimuthal integration
and downstream d-spacing conversion pick the right geometry per frame rather
than assuming one geometry for the whole series.

**8. Automatic calibrant detection + geometry health check on reduction
start.** Detect the calibrant from the accepted calibration's fit residuals
or the image itself, and flag likely geometry problems (poor fit, stale
calibration reused from a different session, detector-distance drift)
automatically when a reduction run starts, rather than relying on the user to
notice a bad PONI downstream in ring waviness or peak quality. The design
consideration is keeping this a warning/health-check layer, not a gate — a
false positive here shouldn't block a run the user knows is fine.

---

## Site adoption

bulkxrd is facility-neutral: nothing in the pipeline assumes a particular
beamline, cluster, or lab. A new facility adopting it needs to provide:

- **A calibration image + PONI** for a standard calibrant (CeO2, LaB6, Si, or
  any calibrant pyFAI recognizes) — the input to the Calibration stage. Any
  detector geometry pyFAI can describe works; nothing is hard-coded to a
  specific detector model.
- **File patterns for the raw frames** — the Reduction stage's dataset glob
  already covers common detector formats (`*.tif;*.tiff;*.edf;*.cbf;
  *.mar3450;*.h5`) and is user-editable, so a facility-specific extension or
  naming scheme just needs the right glob, not code changes.
- **A metadata source for pressure/temperature** — whichever mechanism fits
  how your facility records conditions: filename tokens (any `<number><unit>`
  token parses, not a fixed convention — see §5 of `docs/workflow.md`), or
  CSV import (for a beamline log, ruby-fluorescence spreadsheet, or scan
  record kept separately from the files). Use whichever one matches where
  your facility actually keeps this information; the analysis stage only
  cares that `/frames/pressure`/`temperature` end up populated, not how they
  got there.
- **For mapping runs, the frame positions** — either recorded per frame (the
  automatic path: a positions CSV with `pos_x`/`pos_y` columns, or the raw
  frames' header motor keys via "Read X/Y from headers…" on the Frame meta
  tab; the Grid map's `coordinates` layout then places every frame by its
  stage position with no further input), or, when no positions were
  recorded, the manual raster description (frames-per-line, scan direction,
  serpentine vs unidirectional; Grid map tab, §6.4 of `docs/workflow.md`).

Everything else — background separation, peak fitting, phase identification,
residual/unknown handling, heatmaps, and the ML tooling — is facility-neutral
and needs no site-specific configuration beyond the workspace's own phase
library (the compounds *your* experiment expects to see).
