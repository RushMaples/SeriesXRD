# Test data you can download

Open datasets for exercising seriesxrd end to end — calibration frames, real
measured 1D patterns, CIF structures for the phase library, and simulated
patterns for ML. Each entry says which stage or command it feeds and how to
point seriesxrd at it.

None of this data ships with the package (it is large and separately
licensed). Download what you need and cite the source; the license and
credit for each set are on its own page.

## Pick by what you want to test

| You want to test… | Use | Feeds |
|---|---|---|
| Complete GUI calibration + reduction workflow | SeriesXRD Ti-6Al-4V demo | unified `seriesxrd` GUI |
| Calibration + reduction on real detector frames | pyFAI test images | `seriesxrd-calib-gui` → `seriesxrd-reduce-gui` |
| HDF5/NeXus stack ingestion (Eiger-style master files) | NeXus exampledata | reduce stage frame sources |
| Phase identification / ranking against known truth | XRD-AutoAnalyzer example set, RRUFF, opXRD | `seriesxrd-benchmark` |
| A bigger phase library or ML training corpus | COD, Materials Project | `seriesxrd-corpus`, `seriesxrd-ml-train --cif-dir` |
| ML pretraining on simulated patterns | SimXRD-4M | your own training loop (reference set) |

## A note on downloading these

Most of these hosts (silx.org, rruff.info, zenodo.org, crystallography.net,
huggingface.co) are reachable from a normal workstation but are blocked by the
sandboxed proxy an automated agent runs behind — so download them on your own
machine, not from inside an agent session. The two GitHub-hosted sets
(XRD-AutoAnalyzer, NeXus exampledata) are served from
`raw.githubusercontent.com` and work from either.

---

## Real measured 1D patterns (identification / benchmarking)

These are the ground-truth sets `seriesxrd-benchmark` scores a scorer against:
each `.xy`/`.txt` pattern goes through the real Step-1/2 preprocessing, then
the ranker's hit@1 / hit@K / MRR is measured against a `filename,phases`
labels CSV. See `docs/ml-training.md` "Gate A" for the harness.

### XRD-AutoAnalyzer example set (ready to run)

Already wired up. 8 measured lab patterns + 18 reference CIFs in the
Li-Mn-Ti-O-F space (three of them multi-phase mixtures, plus polymorph
decoys), from the open [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer)
repository. Reachable from an agent sandbox.

```bash
bash examples/fetch_benchmark_example.sh ./benchdata
# import ./benchdata/cifs into a workspace library (Phases tab or
# reference_phases/user_phases.json), then:
seriesxrd-benchmark ./benchdata/spectra --labels ./benchdata/labels.csv \
    --workspace <ws> --out bench_cosine
```

Pinned cosine baseline on this set: hit@1 = 1.000, MRR = 1.000. This is the
regression gate a trained scorer must not fall below.

### RRUFF (mineral powder patterns + structures)

<https://rruff.info> — measured powder XRD patterns and matching CIFs for
thousands of mineral samples. Export a pattern as two-column XY and label it
by the RRUFF mineral name. RRUFF publishes a bulk download of all powder
patterns; the per-mineral pages also have single-file downloads.

```bash
# XY files exported as 2theta (Cu Ka by default on RRUFF); tell the harness:
seriesxrd-benchmark ./rruff_xy --labels rruff_labels.csv \
    --unit 2th_deg --wavelength 1.5406 --workspace <ws> --out bench_rruff
```

The label names in the CSV must already exist in your workspace library, so
import the corresponding RRUFF CIFs first (the benchmark measures the scorer,
not library coverage).

### opXRD (large open experimental database)

The Open Experimental Powder X-ray Diffraction database — ~90k real,
labelled patterns contributed across several labs (Riesel et al., 2025;
arXiv:2503.05577). Published on Zenodo; search Zenodo for "opXRD" for the
current record and DOI. Much larger and messier than RRUFF (varied
instruments, backgrounds, partial labels) — good for stress-testing the
preprocessing and the scorer on realistic noise. Same `seriesxrd-benchmark`
ingestion once you have a `filename,phases` CSV.

---

## Detector frames (calibration + reduction)

### SeriesXRD Ti-6Al-4V demo (recommended)

The repository's [`examples/ti64_demo`](../examples/ti64_demo/README.md)
example downloads and organizes 12 exposure-time measurements plus their
matching CeO2 calibration frame. It verifies the checksums published by
Zenodo, preserves source metadata and attribution, and provides a refined
PONI file and exact GUI walkthrough. The downloaded data remain gitignored
and are not included in the package.

```bash
python examples/ti64_demo/fetch_demo_data.py
seriesxrd --workspace examples/ti64_demo/workspace
```

Source: Daniel et al. (2022), *Synchrotron X-ray Diffraction Dataset -
Measuring Bulk Crystallographic Texture from Differently-Orientated
Ti-6Al-4V Samples*, Zenodo,
<https://doi.org/10.5281/zenodo.7270710>, CC BY 4.0. See the example's
`ATTRIBUTION.md` for the full citation and calibration provenance.

### pyFAI test images

<http://www.silx.org/pub/pyFAI/testimages/> — the calibrant frames and
detector images pyFAI's own test suite uses (LaB6, CeO2, AgBh rings on real
detector geometries). Use one as the calibration standard image in
`seriesxrd-calib-gui`, accept the geometry, then reduce a small stack to verify
the calib→reduce handoff and the 1D channels end to end without needing your
own beamtime data. pyFAI can also fetch these itself via
`pyFAI.test.utilstest.UtilsTest.getimage(<name>)` if pyFAI is installed.

---

## HDF5 / NeXus stacks (reduce frame-source ingestion)

### NeXus exampledata

<https://github.com/nexusformat/exampledata> — real `.nxs`/`.h5` files in
many NeXus layouts. Use them to exercise the HDF5/NeXus stack ingestion path
(`core/io.expand_frame_sources`, metadata harvesting) with a frame spec:

```bash
# point the reducer at a stack container; one entry per detector frame
#   file.h5::entry/data/data          (whole stack)
#   file.h5::entry/data/data#000123   (a single frame)
```

Reachable from an agent sandbox (GitHub-hosted). Layouts vary, so this is the
right set for checking that `h5_*_path` config keys pin an unusual detector
path and that timestamp/position/temperature harvesting finds the NeXus
locations.

---

## CIF structures (phase library + training corpus)

### Crystallography Open Database (COD)

<https://www.crystallography.net/cod/> — open CIF database. Two ways in:

```bash
# curated ID list (one COD id per line) -> CIFs, then screen them:
seriesxrd-corpus fetch cod_ids.txt ./training_cifs
seriesxrd-corpus screen ./training_cifs        # parse / dedupe / size-screen
seriesxrd-ml-train --workspace <ws> --cif-dir ./training_cifs --out scorer.pt
```

For bulk (10^4+) corpora use COD's rsync mirror rather than the ID fetcher.
Corpus CIFs feed the ML simulator only — they are never added to your
analysis library.

### Materials Project

<https://materialsproject.org> via the `mp-api` client (needs a free API
key). Query by chemistry/space group, export CIFs, and use them either as
real library phases (with EOS parameters you supply) or as an ML training
corpus (`--cif-dir`, synthetic BM3 EOS auto-assigned). See
`docs/ml-training.md` §3.

---

## Simulated patterns (ML pretraining reference)

### SimXRD-4M

<https://huggingface.co/datasets/caobin/SimXRD> — 4.07M simulated patterns
from 119,569 Materials Project structures across 33 measurement conditions
(Cao et al., ICLR 2025; arXiv:2406.15469). The reference design seriesxrd's
Step-3b simulator follows, and a pretraining set if you build your own model.
Note the gap seriesxrd's simulator fills that SimXRD does not: lattice
compression under pressure (the EOS peak-shift manifold) — see
`docs/ml-training.md` and CLAUDE.md "Step 3 design". The open-source
simulator behind it is PysimXRD.

---

## Exporting your own reference patterns

Any seriesxrd analysis run can emit selected frames as clean two-column XY plus
a peaks table — useful for building your own labelled benchmark set from data
you have already reduced and identified:

```bash
seriesxrd-export-refinement analysis.h5 ./out --frames 0,5,10 --peaks
```

or the "Export selected…" / "Export frame…" buttons in the Analysis GUI. See
`docs/workflow.md` for the flags.
