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

**HDF5/NeXus stack ingestion for the reduce stage (new in this release).**
APS/ESRF Eiger-style detectors write one HDF5 stack (a master file plus
linked data files) rather than file-per-frame images. `core/io.py` now
expands any matched `.h5`/`.hdf5`/`.nxs` container into per-frame sources
(`"file.h5::entry/data/data#000123"`), auto-detecting the frame dataset
(NeXus convention first, else the largest 3D image dataset; `h5_data_path`
in the reduction config pins an unusual layout), and `read_detector_image`
reads either kind of source — so the whole downstream pipeline is unchanged.
bulkxrd's own output files are refused as frame sources, `hdf5plugin` is
loaded when present (Eiger bitshuffle/LZ4 compression), and the Dataset
tab's scan preview shows the true expanded frame count.

**Geometry health check on reduction completion (new in this release).**
When cakes were saved, the reduction fits ring waviness on the first few
cakes as it finishes and warns — with the implied 1D doublet splitting in
bins and the transverse sample offset in mm — when the geometry error is
large enough to corrupt peak fitting. Catching this at reduce time is far
cheaper than discovering doubled peaks after a full analysis run. (The
calibrant auto-detection half of the original roadmap item remains planned
below.)

**Semi-quantitative phase fractions (new in this release).**
`analysis/fractions.py` (`bulkxrd-analyze --fractions`) apportions each
frame's attributed peak areas (`/peaks/phase`, from the Step-3a removal)
into per-phase intensity shares, with optional per-phase RIR (I/Icor)
weighting, written to `/fractions`. The module says plainly what it is:
texture, absorption, and structure-factor differences are not corrected —
Rietveld refinement (the export below) is the quantitative path.

**Refinement hand-off bundle (new in this release).**
`analysis/refine_export.py` (`bulkxrd-export-refinement`) writes a
Rietveld-ready bundle: per-frame patterns as two-column `.xy` (native q axis
always, 2θ additionally when the wavelength is known), phase CIFs (copied
from the library entry, or synthesized via pymatgen from lattice+atoms), a
minimal GSAS-II `instrument.instprm`, and a README with a runnable
GSASIIscriptable snippet. An export/bridge, deliberately not a Rietveld
reimplementation.

**Azimuthal texture analysis (new in this release).**
`reduce/texture.py` (`bulkxrd-texture`) measures each saved cake's strongest
rings: intensity vs azimuth per ring, with a texture index (std/mean), a
spot fraction (coarse-grain indicator), and a preferred-orientation second
harmonic (amplitude + phase), written to `/texture` in the reduced file.
Interpreting the 2-fold modulation as texture vs differential stress needs
the experiment geometry, which stays with the user.

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

**1. NeXus per-frame metadata harvesting.** The stack ingestion above reads
the frame images; a NeXus master file often ALSO carries per-frame exposure
times, timestamps, and motor positions that today still need a sidecar CSV
or header keys. Design: a metadata sibling to the image auto-detection that
maps common NeXus locations onto `/frames/timestamp` and `pos_x`/`pos_y`
during reduction.

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

**3. Rietveld-quality phase fractions.** The intensity-share/RIR fractions
above are implemented; publication-grade weight fractions come from refining
the exported bundle in GSAS-II (or similar) and, if wanted later, importing
the refined scale factors back into `/fractions` — an import bridge, not an
in-house Rietveld engine.

**4. Open-set structure search for unknowns.** Take a Step 3c cluster's
d-fingerprint and search it against a COD-derived candidate set using the
same `ml_scorer` seam Step 3b already defines, instead of stopping at "here
is an unidentified cluster of coherent peaks." The design need is a
candidate-simulation cache: naively simulating reflections for a large COD
subset per query is too slow to be interactive, so this needs the corpus
tooling (`analysis/corpus.py`) plus a precomputed/cached simulation layer
searched by approximate d-fingerprint match before any scorer runs.

**5. Multi-detector / multi-geometry sessions.** Support one series measured
across two (or more) detector positions or geometries within a single
analysis — e.g. a wide-angle and a high-angle detector, or a mid-run
detector-distance change. Design: needs a per-frame PONI association (today
one accepted calibration applies to an entire reduced file) threaded through
reduce and into the analysis HDF5's frame metadata, so azimuthal integration
and downstream d-spacing conversion pick the right geometry per frame rather
than assuming one geometry for the whole series.

**6. Automatic calibrant detection.** The geometry health check on
reduction completion is implemented (see above); the remaining half is
detecting the calibrant from the accepted calibration's fit residuals or the
image itself, and flagging a stale calibration reused from a different
session BEFORE the run rather than after it. Same design constraint: a
warning layer, never a gate — a false positive must not block a run the
user knows is fine.

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
