# Training the Step-3b learned scorer

This is the end-to-end guide for training, validating, and deploying the
RADAR-PD-style learned pair scorer (`seriesxrd-ml-train` → `scorer.pt` →
`TorchScorer`). It applies to workstations and compute clusters. Scheduler
examples below use placeholders so deployments can supply their own queue,
storage, environment, and allocation settings.

The pipeline analyzes pressure series, temperature series, and other in-situ
series. Examples below use diamond-anvil-cell (DAC) pressure series because
that is the primary use case today, but nothing here is pressure-specific
except where stated.

**What the model is.** A ~1M-parameter TorchScript module that scores one
`(measured pattern, candidate fingerprint)` pair on the shared d-grid
(SimXRD-4M format, 3501 points, d = 1.199–8.853 Å) to a similarity in [0, 1].
It replaces only the cosine similarity inside the Step-3b ranker. Candidate
simulation, pressure handling, and the deterministic Step-3a verifier do not
change. ML proposes, physics verifies: a trained scorer can shortlist
candidates. It never accepts them — Step 3a still verifies every candidate
the scorer proposes.

**What the model is not.** It is not a phase classifier and does not need
your exact sample phases in its training set. It learns a similarity
function: does this candidate fingerprint, at this pressure, plausibly sit
inside this measured mixture? What it needs from training data is therefore
diversity — crystal systems, peak densities, line multiplicities, widths,
overlaps — not a matching label set.

**Default stays cosine.** `CosineScorer` is the default scorer everywhere in
the pipeline. Train and validate a learned scorer before switching a real
analysis run to it (see §7, Validation gates).

---

## 1. The data-quality gate

Run this before anything below. The model's input is whatever the
deterministic pipeline produces, and the weak labels come from Step 3a, so
training inherits every upstream data pathology. All four checks are
automated and recorded in the output files. Train only when they pass on the
dataset(s) you will validate against.

1. **Geometry.** Wavy cake rings (a sample measured off the calibrant
   position) turn every peak into a constant-splitting doublet the simulator
   does not model. A cakes-enabled reduction already checks this
   automatically at completion: it fits the ring wobble on a few saved cakes
   and prints a `[REDUCE] geometry check OK` line, or a `WARNING` giving the
   implied 1D doublet splitting (in bins) and the sample offset in mm, and
   records the summary in the reduction manifest (`geometry_check`). Confirm
   that check passed before collecting training data from a dataset. For a
   manual look, or on a reduction from before this check existed, run
   `reduce.straighten.diagnose_reduced()` on a cakes-enabled reduction
   directly (same underlying fit: the first-harmonic amplitude `A1` should be
   much smaller than the peak FWHM), or use the Reduction stage's Review tab
   ("Diagnose waviness" / "Write straightened 1D" buttons — both need
   `save_cakes`). Fix by re-refining the PONI on a sample-position ring and
   re-reducing; `straighten_reduced()` (Python API) / "Write straightened 1D"
   (GUI) is the rescue path for already-collected data.
2. **Sampling.** Step 2 measures the median fitted FWHM in bins and warns
   with a concrete `npt_1d` recommendation when peaks are undersampled
   (< 4 bins). Check `median_fwhm_bins` / `npt_recommended` in the peaks
   manifest. Undersampled peaks also break the `fwhm_q="auto"` resolution
   estimate the ranker uses to render candidates at your instrument's width.
3. **Channel.** Step 1 records `signal_frac_clean` / `spotty_sample` (where
   the Bragg signal actually lives). On a coarse-grained sample the
   median-based channels contain background only. `source="auto"` handles
   this at fit time, but a model must be trained and run on the resolved
   channel recorded in `/ml/candidates` attrs (`resolved_source`), not on an
   assumed one.
4. **Robust channel.** The reduce manifest's `robust_estimator` must read
   `quantile_band(...)`. `median(band_unsupported)` means your pyFAI version
   ignored the band kwargs (it logs "Got unknown argument ...") and the
   channel is a quantized pure median.

If any check fails, fix it and re-run the affected stage before collecting
training data from that dataset.

---

## 2. Environment setup

Any machine with Python >= 3.10 works: a laptop, a lab workstation, or a
cluster node. GPU is optional — training runs on CPU, just slower.

```bash
python -m venv .venv-ml && source .venv-ml/bin/activate
pip install -e .[phases,ml]     # pymatgen (reflection simulation) + torch (training)
```

Notes:

* `[phases]` (pymatgen) is required. Reflections are simulated from CIFs or
  library structures once at startup.
* `[ml]` (torch) is required for training only. The rest of seriesxrd,
  including the deterministic ranker, never imports torch.
* If you use a container with its own PyTorch, `pip install -e .[phases]` is
  enough — skip `[ml]` — as long as `python -c "import torch"` works inside
  the image.
* No GUI, no pyFAI runtime work, and no display are needed for training.

### Smoke test (CPU, minutes)

Always run this before queueing a long or GPU job, on any machine:

```bash
seriesxrd-ml-train --workspace /path/to/workspace --out /tmp/scorer_smoke.pt \
    --epochs 2 --mixtures-per-epoch 32 --device cpu
python tests/test_ml.py          # train->export->rank roundtrip when torch is present
```

---

## 3. Collect phases before training

The bundled baseline is about 20 mostly-cubic high-pressure standards. That
is fine for the deterministic ranker — it only needs the phases you might
verify — but it is not enough pattern diversity to train a general
similarity model. A scorer trained only on sparse FCC/BCC patterns overfits
to "few-strong-lines" fingerprints and degrades on low-symmetry samples.

Collect two different things, for two different reasons:

1. **Workspace library phases** (for identification, small and curated).
   Everything you expect in your experiment: your sample compounds, likely
   decomposition/reaction products, your pressure or temperature marker,
   gasket, and medium. Add them on the Phases tab (or
   `reference_phases/user_phases.json`) with real EOS parameters where the
   literature has them, and an `eos["p_max"]` validity ceiling wherever a
   phase transition ends the entry's stability field. These are the phases
   the ranker ranks and Step 3a verifies at analysis time.

2. **Training-only CIF corpus** (for the scorer, large and uncurated). A
   folder of CIFs spanning crystal systems — a few hundred to a few thousand
   is a good start. Sources: COD (Crystallography Open Database, bulk
   download by ID or rsync mirror), Materials Project (`mp-api`, needs an API
   key), or your group's own CIF collection. Point `seriesxrd-ml-train
   --cif-dir` at the folder. These phases are never added to your workspace
   library — they only feed the simulator.

Corpus entries have no EOS by default, so each gets a synthetic plausible BM3
bulk modulus (random K0 in [30, 300] GPa, K0'=4) so the model sees their
compression manifold too. The model learns pattern similarity under
compression, not the fake K0. Pass `--no-synthetic-eos` to pin corpus phases
at ambient instead.

Training without a corpus (i.e. on the ~20 bundled standards) is a smoke test
only. Do not deploy a scorer trained that way — the deterministic cosine
will beat it on anything low-symmetry.

---

## 4. Build the training corpus

Two commands:

```bash
seriesxrd-corpus fetch cod_ids.txt ./training_cifs   # COD IDs, one per line
seriesxrd-corpus screen ./training_cifs              # parse / dedupe / size-screen
```

`seriesxrd-corpus fetch <ids> <out_dir> [--base-url URL]` downloads COD entries
by ID to `<out_dir>/<id>.cif`. Existing files are skipped, so it is
re-runnable. Failures are collected and reported, not fatal. For bulk (10^4+)
corpora, use COD's rsync mirror instead — this fetcher is for curated ID
lists.

`seriesxrd-corpus screen <cif_dir> [--max-sites N] [--no-dedupe]
[--keep-rejects-in-place]` parses every `*.cif` under `cif_dir` and rejects:

* files that fail to parse (an unreadable CIF would otherwise be silently
  skipped at training startup);
* cells with more than `--max-sites` sites (default 200) — large cells
  dominate reflection-simulation time for little diversity gain;
* duplicates — same reduced formula + space group + cell volume within
  0.5%, unless `--no-dedupe` is passed.

Rejects move to `cif_dir/rejected/<reason>/` unless
`--keep-rejects-in-place` is set. The command writes
`cif_dir/corpus_manifest.json` recording what was kept and why anything was
dropped. Run `screen` before pointing `seriesxrd-ml-train --cif-dir` at the
folder — it is not optional hygiene.

---

## 5. Train

```bash
seriesxrd-ml-train --workspace /path/to/workspace --out scorer.pt \
    --cif-dir /path/to/training_cifs \
    --epochs 20 --mixtures-per-epoch 512 --device cuda
```

Full flag list (`seriesxrd-ml-train --help`, verified against
`seriesxrd/analysis/ml_train.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--workspace` | `""` | Workspace with the phase library. |
| `--phases` | `""` | Comma-separated subset of library phases. Default: all simulatable library phases. |
| `--cif-dir` | `""` | Training-only CIF corpus folder (see §3-4). Never added to the workspace library. |
| `--no-synthetic-eos` | off | Do not assign a random plausible EOS to corpus CIFs lacking one — they simulate at ambient only. |
| `--out` | `scorer.pt` | Output TorchScript path. |
| `--epochs` | `20` | Training epochs. |
| `--mixtures-per-epoch` | `256` | Fresh simulated mixtures generated per epoch. |
| `--max-phases` | `2` | Max phases per simulated mixture. |
| `--batch-size` | `64` | Training batch size. |
| `--lr` | `3e-4` | Learning rate. |
| `--p-max` | `100.0` | Top of the training pressure grid (GPa). |
| `--p-step` | `5.0` | Pressure grid step (GPa). |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |
| `--seed` | `0` | RNG seed. |

Guidance beyond the defaults:

* `--cif-dir` is the main diversity lever — see §3.
* `--mixtures-per-epoch`: 512-2048 once you have a real corpus.
* `--max-phases`: 2-3 covers sample + marker + gasket/medium, the common case
  for a DAC frame; raise it if your experiment routinely has more phases per
  frame.
* `--p-max` / `--p-step`: cover your experimental pressure range. If your
  series is not pressure-driven (e.g. a pure temperature series with no
  compression), see the "Pressure-only simulation" note in §6 — the ML
  simulator does not have a temperature axis yet, so keep using the
  deterministic cosine ranker for that case.
* `--epochs`: 20-50. Every epoch is fresh simulated data, so overfitting
  pressure is low.

Training conventions that are baked in and have no flag — see §6.

### Pathfinder run

Before committing to a full training run on a new machine, environment, or
corpus, run a small disposable pathfinder job: ~100-300 CIFs,
`--epochs 10`, a modest `--mixtures-per-epoch` (e.g. 128). It should finish
in minutes to low tens of minutes and confirms the environment, corpus, and
training loop all work end to end before you spend a full-scale job's worth
of compute (and queue time, on a shared cluster) discovering a broken path
or a bad flag.

---

## 6. Simulation physics conventions

These are implementation details, not flags — see `CLAUDE.md` ("Simulation
physics conventions") for the authoritative description. Kept short here on
purpose:

* **Peak widths are q-constant, not d-constant.** Resolution is
  approximately constant in q, so each simulated peak gets `Δd = d²·Δq/2π`.
  At inference, `rank_candidates(fwhm_q="auto")` fits the measured resolution
  curve from the Step-2 peaks; training samples `fwhm_q` from
  `AugmentConfig.fwhm_q` per mixture.
* **One pressure per simulated mixture.** All phases in a training mixture
  share a single pressure, as in one real frame. Independent per-phase
  pressures would teach the scorer an unphysical manifold.
* **EOS validity ceilings.** `eos["p_max"]` caps where a phase is simulated
  or fit. A stability-limited entry (e.g. NaCl-B1 <= 30 GPa in the bundled
  baseline) is never trained or verified past its transition.
* **Pressure-only simulation.** The temperature seam (`Phase.thermal`) and
  signed axial expansivity (`axial_eos` `beta`) already move predictions in
  the deterministic identify/residual/heatmap paths, but the ML simulator is
  pressure-only for now. A temperature-series or axial-expansion-heavy
  problem should be ranked with the deterministic cosine scorer until the
  simulator grows those axes — training a pressure-only model on such data
  will not help and may mislead.
* **Fixed validation set.** Validation pairs come from mixtures generated
  once with a disjoint seed. No mixture appears on both sides of the split.

---

## 7. Validate against the deterministic baseline

The deterministic cosine scorer stays the default until a trained model is
validated against known truth. There are two gates. A trained scorer is
promoted only if it clears both.

### Gate A — external labelled patterns (`seriesxrd-benchmark`)

Ingest any labelled XY pattern set (RRUFF exports, opXRD dumps, or your own)
through the pipeline's own preprocessing and score the ranker against the
labels:

```bash
# pin the baseline once
seriesxrd-benchmark ./patterns --labels labels.csv --workspace ws --out bench_cosine
# same command, trained scorer
seriesxrd-benchmark ./patterns --labels labels.csv --workspace ws \
    --out bench_torch --ml-scorer torch:/path/scorer.pt
```

`seriesxrd-benchmark <patterns> --labels FILE [--workspace WS] [--out DIR]
[--unit UNIT] [--wavelength A] [--top-k K] [--ml-scorer SCORER]
[--no-identify]` — flags verified against
`seriesxrd/analysis/benchmark.py`:

| Flag | Default | Meaning |
|---|---|---|
| `patterns` | required | Directory of XY `.txt`/`.xy` files, or one file. |
| `--labels` | required | CSV: `filename,phases` (`;`-separated library names). |
| `--workspace` | `""` | Workspace with the phase library. |
| `--out` | `benchmark_out` | Output directory. |
| `--unit` | `2th_deg` | Axis unit of the XY files. |
| `--wavelength` | Cu Ka1 | Å, used when the axis unit needs one. |
| `--top-k` | `5` | Top-K for hit@K. |
| `--ml-scorer` | `""` (cosine) | `cosine` or `torch:<model.pt>`. |
| `--no-identify` | off | Skip the Step-3a verify metrics (rank-only, no pymatgen). |

Compare `hit@1` / `hit@K` / `MRR` in the two `benchmark_report.json` files.
Promote the trained scorer only when it at least matches the cosine
baseline. The labels CSV is `filename,phases` (`;`-separated for
multi-phase rows); the label names must already exist in the workspace
library (import the corresponding structures first — the benchmark measures
the scorer, not library coverage).

A ready-made real-data example: `examples/fetch_benchmark_example.sh`
downloads 8 measured lab patterns + 18 reference CIFs (Li-Mn-Ti-O-F space,
including three multi-phase mixtures and polymorph decoys) from the open
XRD-AutoAnalyzer repository, with a pre-built labels CSV:

```bash
bash examples/fetch_benchmark_example.sh ./benchdata
# import the CIFs into a workspace library, then:
seriesxrd-benchmark ./benchdata/spectra --labels ./benchdata/labels.csv \
    --workspace <ws> --out bench_cosine
```

Pinned cosine baseline on this set: **hit@1 = 1.000, MRR = 1.000, identify
hit rate = 1.000**. A trained scorer must not fall below this.

### Gate B — your own known-truth run

Run both scorers over an analysis file whose ground truth you know (your
marker + gasket run, or any frame set with known phases):

```bash
# baseline
seriesxrd-analyze reduced.h5 --steps 3 --ml-rank --ml-rank-top-k 5
# learned
seriesxrd-analyze reduced.h5 --steps 3 --ml-rank --ml-rank-top-k 5 \
    --ml-scorer torch:/path/to/scorer.pt
```

Compare `/ml/candidates/topk_names` and the Step-3a confidences that follow:

* Phases you know are present (marker, gasket, sample) must appear in the
  learned top-K at least as reliably as with cosine.
* The learned shortlist should be sharper (fewer spurious candidates
  surviving to verification), not merely different.
* Since Step 3a verifies everything, a bad scorer costs recall (missing
  proposals), not false identifications. That is why the check above focuses
  on known-present phases still being proposed.

Both runs record full provenance in `/ml/candidates` attrs (`method`,
`resolved_source`, `fwhm_q`, `clip_negative`, `normalize`), so the comparison
is reproducible.

---

## 8. Deploy

Copy `scorer.pt` somewhere stable and reference it wherever ranking runs
(requires `pip install seriesxrd[ml]` on that machine):

* **Batch CLI:** `seriesxrd-analyze ... --ml-rank --ml-scorer
  torch:/path/scorer.pt`
* **Worker/GUI config:** set `"ml_scorer": "torch:/path/scorer.pt"` in the
  analysis session config (the Identify tab's ML ranking then uses it).
* **Python:** `rank_candidates(h5, phases, scorer="torch:/path/scorer.pt")`

If torch or the model file is missing, ranking fails with an instructive
error telling you to fall back to `cosine`. It never crashes mid-run.

Retrain when you add a substantially different class of phases to your
work, change the d-grid/preprocessing conventions, or see the learned
shortlist underperform cosine on a known-truth dataset.

---

## 9. Running on any cluster

Training needs: a Python environment (venv or container) with `seriesxrd[phases,ml]`
installed, a workspace with a phase library, and optionally a CIF corpus
directory. Nothing about `seriesxrd-ml-train` is scheduler-specific — it is a
plain CLI process. GPU is optional (`--device cuda` vs `--device cpu`).

Any scheduler works the same way: request CPUs/GPU/memory, activate the
environment, run `seriesxrd-ml-train`. A generic Slurm example:

```bash
#!/bin/bash
#SBATCH --job-name=seriesxrd-ml-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=train_%j.log

source /path/to/.venv-ml/bin/activate
cd /path/to/seriesxrd
seriesxrd-ml-train \
    --workspace /path/to/xrd_workspace \
    --cif-dir   /path/to/training_cifs \
    --out       /path/to/models/scorer.pt \
    --epochs 30 --mixtures-per-epoch 512 --max-phases 3 \
    --p-max 100 --p-step 5 --device cuda --seed 0
```

To adapt this to another scheduler, translate the resource request and keep
the body:

* **LSF** (`bsub`): use `-q`, `-n`, `-M`, and `-gpu "num=1"` in place of the
  `#SBATCH` lines.
* **PBS/Torque** (`qsub`): `#PBS -l select=1:ncpus=4:ngpus=1`,
  `#PBS -l walltime=04:00:00`, `#PBS -q <queue>` in place of the `#SBATCH`
  lines; the `seriesxrd-ml-train` invocation is unchanged.

Reflection simulation for a large corpus happens once at startup and is
cached across epochs, so the first minutes of any job are pymatgen-bound
before the GPU (if any) does anything. Run the pathfinder job (§5) locally
or as a short interactive/queued job before committing to a full-scale
allocation on a new cluster.

### Reading the training log

```
[ML-TRAIN] device=cuda params=1,04x,xxx phases=812 epochs=30 mixtures/epoch=512
[ML-TRAIN] epoch 1/30 loss=0.61 val_AUC=0.71
...
[ML-TRAIN] done -> .../scorer.pt (best val AUC 0.97)
```

`val_AUC` is ROC-AUC on the fixed held-out pair set: 0.5 is random, and with
a healthy corpus it should clear about 0.9. A model that plateaus near 0.8
usually means the corpus is too small or too homogeneous, or `--max-phases
1` was used (no mixtures means no occlusion, so the model learns nothing
about it).

The export step traces the model to TorchScript, verifies numeric
equivalence of the traced graph, and round-trips the artifact through
`TorchScorer`. If the job printed `done ->`, the `.pt` file is loadable by
the consumer contract.

---

## 10. Training strategy: keep this a workflow, not a product

Do not maintain a pre-trained production model as the default deliverable
of this project. Keep a validated general training workflow (this
document) instead, and train at full scale only when a concrete need
appears:

* a large open-set library where the deterministic cosine's candidate
  shortlist is measurably too coarse, or
* cosine demonstrably failing on labelled data (Gate A or Gate B above
  showing a real gap, not a hypothetical one).

This is a deliberate choice, not a placeholder. Instrument-specific
properties — detector resolution, pressure or temperature priors, which
channel (`fit`/`residual`/etc.) the pipeline is reading — enter at
inference time (via `fwhm_q`, the metadata prior, `resolved_source`), not
baked into the trained weights. That is what makes one training workflow
(this document) reusable across labs and instruments, instead of needing a
bespoke model per beamline. Re-validate (§7) whenever you retrain, and treat
a trained scorer as an optimization on top of the deterministic pipeline,
never a replacement for it — Step 3a verification is what keeps a bad
scorer from producing a false identification.
