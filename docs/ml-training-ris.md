# Training the Step-3b learned scorer on WashU RIS

This is the end-to-end guide for training, validating, and deploying the
RADAR-PD-style learned pair scorer (`bulkxrd-ml-train` → `scorer.pt` →
`TorchScorer`). It replaces the short recipe that used to live only in the
`ml_train.py` docstring.

**What the model is.** A ~1M-parameter TorchScript module that scores one
`(measured pattern, candidate fingerprint)` pair on the shared d-grid
(SimXRD-4M format, 3501 points, d = 1.199–8.853 Å) to a similarity in [0, 1].
It replaces *only* the cosine similarity inside the Step-3b ranker; candidate
simulation, pressure handling, and the deterministic Step-3a verifier are
unchanged. **ML proposes, physics verifies** — a trained scorer can shortlist
candidates, never accept them.

**What the model is not.** It is not a phase classifier and it does not need
your exact sample phases in its training set. It learns a *similarity
function*: "does this candidate fingerprint, at this pressure, plausibly sit
inside this measured mixture?" What it needs from training data is therefore
*diversity* — crystal systems, peak densities, line multiplicities, widths,
overlaps — not a matching label set.

---

## 0. Before you train: collect phases

The bundled baseline is ~20 mostly-cubic high-pressure standards. That is fine
for the deterministic ranker (it just needs the phases you might verify), but
it is **not enough pattern diversity to train a general similarity model** —
a scorer trained only on sparse FCC/BCC patterns will overfit to
"few-strong-lines" fingerprints and degrade on low-symmetry samples.

Collect two different things, for two different reasons:

1. **Workspace library phases (for identification, small and curated).**
   Everything you expect in *your* cell: your sample compounds, likely
   decomposition/reaction products, your marker, gasket, and medium. Add them
   on the Phases tab (or `reference_phases/user_phases.json`) with real EOS
   parameters where the literature has them, and an `eos["p_max"]` validity
   ceiling wherever a phase transition ends the entry's stability field.
   These are the phases the ranker ranks and Step 3a verifies at analysis
   time.

2. **Training-only CIF corpus (for the scorer, large and uncurated).**
   A folder of CIFs spanning crystal systems — a few hundred to a few thousand
   is a good start. Sources:
   * **COD** (Crystallography Open Database) — bulk download, no license
     friction: `rsync` mirrors or per-file fetch by COD ID.
   * **Materials Project** — export CIFs via `mp-api` (needs an API key);
     filter to experimentally-observed structures if you want to stay
     conservative.
   * Your group's own CIF collection.

   Point `bulkxrd-ml-train --cif-dir` at the folder. These phases are **never
   added to your library** — they only feed the simulator. Corpus entries
   have no EOS, so by default each gets a *synthetic* plausible BM3 bulk
   modulus (random K0 ∈ [30, 300] GPa, K0′ = 4) so the model sees their
   compression manifold too; the model learns pattern similarity under
   compression, not the fake K0. Disable with `--no-synthetic-eos` if you
   want corpus phases pinned at ambient.

**Should you collect more phases before training? Yes — the corpus.** The
library only needs what you'll actually verify, but training without a corpus
(i.e., on the ~20 bundled standards) is only useful as a smoke test. Do not
deploy a scorer trained that way; the deterministic cosine will beat it on
anything low-symmetry.

---

## 1. RIS environment setup

RIS compute nodes run jobs under IBM LSF inside Docker images. Any image with
Python ≥ 3.10 and CUDA-enabled PyTorch works — e.g. `pytorch/pytorch` (CUDA
runtime included) — with the repo and outputs on your lab's `/storage1`
allocation.

```bash
# one-time, from a login/interactive node with your storage mounted
cd /storage1/fs1/<lab>/Active/<you>
git clone <your bulkxrd remote> bulkxrd
cd bulkxrd
python -m venv .venv-ml && source .venv-ml/bin/activate
pip install -e .[phases,ml]     # pymatgen (reflection simulation) + torch (training)
```

Notes:

* `[phases]` (pymatgen) is required — reflections are simulated from the
  CIFs/structures once at startup.
* `[ml]` (torch) is required for training only. The rest of bulkxrd, including
  the deterministic ranker, never imports torch.
* If you use the container's own PyTorch instead of a venv, `pip install -e
  .[phases]` is enough (skip `[ml]`) — just make sure `python -c "import
  torch"` works inside the image.
* No GUI, no pyFAI, and no display are needed for training.

### Smoke test (CPU, minutes)

Always run this before queueing a GPU job:

```bash
bulkxrd-ml-train --workspace /path/to/workspace --out /tmp/scorer_smoke.pt \
    --epochs 2 --mixtures-per-epoch 32 --device cpu
python tests/test_ml.py          # includes a train→export→rank roundtrip when torch is present
```

### LSF job (GPU)

```bash
bsub -q general -G compute-<lab> -n 4 -M 32GB -R 'rusage[mem=32GB]' \
     -gpu "num=1" \
     -a 'docker(pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime)' \
     -o train_%J.log \
     bash -lc '
       cd /storage1/fs1/<lab>/Active/<you>/bulkxrd
       pip install --user -e .[phases]
       bulkxrd-ml-train \
         --workspace /storage1/fs1/<lab>/Active/<you>/xrd_workspace \
         --cif-dir   /storage1/fs1/<lab>/Active/<you>/training_cifs \
         --out       /storage1/fs1/<lab>/Active/<you>/models/scorer.pt \
         --epochs 30 --mixtures-per-epoch 512 --max-phases 3 \
         --p-max 100 --p-step 5 --device cuda --seed 0
     '
```

Adjust queue/group/image to your lab's allocation. Reflection simulation for a
large corpus happens **once** at startup (it is cached across epochs), so the
first minutes are pymatgen-bound before the GPU does anything.

### Knobs that matter

| Flag | Meaning | Guidance |
|---|---|---|
| `--cif-dir` | training-only CIF corpus | see §0; the main diversity lever |
| `--mixtures-per-epoch` | fresh simulated mixtures per epoch | 512–2048 with a corpus |
| `--max-phases` | max phases per mixture | 2–3 (sample + marker + gasket is the DAC norm) |
| `--p-max`, `--p-step` | training pressure grid (GPa) | cover your experimental range |
| `--epochs` | training epochs | 20–50; every epoch is fresh data, so overfitting pressure is low |
| `--no-synthetic-eos` | corpus CIFs stay at ambient | only if you specifically want that |

Training conventions baked in (do not need flags):

* **One pressure per mixture** — all phases of a simulated mixture share a
  single pressure, like a real DAC frame, clamped per phase to its
  `eos["p_max"]` validity ceiling.
* **q-constant peak widths** — each pattern samples an instrument resolution
  Δq and every peak gets `Δd = d²·Δq/2π`, matching what a q-uniform detector
  axis produces on the d-grid. Candidate fingerprints are rendered at the same
  width as their mixture, exactly as inference does with the measured
  resolution.
* **Fixed validation set** — validation pairs come from mixtures generated
  once with a disjoint seed; no mixture appears on both sides of the split,
  and best-model selection compares epochs on the same yardstick.

---

## 2. Reading the training log

```
[ML-TRAIN] device=cuda params=1,04x,xxx phases=812 epochs=30 mixtures/epoch=512
[ML-TRAIN] epoch 1/30 loss=0.61 val_AUC=0.71
...
[ML-TRAIN] done -> .../scorer.pt (best val AUC 0.97)
```

* `val_AUC` is ROC-AUC on the fixed held-out pair set: 0.5 = random, and with
  a healthy corpus you should see it clear ~0.9. A model that plateaus near
  0.8 usually means the corpus is too small/homogeneous or `--max-phases 1`
  (no mixtures = too easy, learns nothing about occlusion).
* The export step traces to TorchScript, verifies numeric equivalence of the
  traced graph, and round-trips the artifact through `TorchScorer` — if the
  job printed `done ->`, the `.pt` is loadable by the consumer contract.

---

## 3. Validate against the deterministic baseline before trusting it

The deterministic cosine stays the default until a trained model is validated
**on your real data**. Run both scorers over an analysis file whose ground
truth you know (e.g. your marker + gasket run):

```bash
# baseline
bulkxrd-analyze reduced.h5 --steps 3 --ml-rank --ml-rank-top-k 5
# learned
bulkxrd-analyze reduced.h5 --steps 3 --ml-rank --ml-rank-top-k 5 \
    --ml-scorer torch:/path/to/scorer.pt
```

Compare `/ml/candidates/topk_names` (and the Step-3a confidences that follow):

* Phases you *know* are present (marker, gasket, sample) must appear in the
  learned top-K at least as reliably as with cosine.
* The learned shortlist should be *sharper* (fewer spurious candidates
  surviving to verification), not merely different.
* Since Step 3a verifies everything, a bad scorer costs recall (missing
  proposals), not false identifications — which is exactly why the check
  above focuses on "known-present phases still proposed".

Both runs record full provenance in `/ml/candidates` attrs (`method`,
`resolved_source`, `fwhm_q`, `clip_negative`, `normalize`), so the comparison
is reproducible.

---

## 4. Deploy

Copy `scorer.pt` somewhere stable and reference it wherever ranking runs
(requires `pip install bulkxrd[ml]` on that machine):

* **Batch CLI:** `bulkxrd-analyze ... --ml-rank --ml-scorer torch:/path/scorer.pt`
* **Worker/GUI config:** set `"ml_scorer": "torch:/path/scorer.pt"` in the
  analysis session config (the Identify tab's ML ranking then uses it).
* **Python:** `rank_candidates(h5, phases, scorer="torch:/path/scorer.pt")`

If torch or the model file is missing, ranking fails with an instructive
error telling you to fall back to `cosine` — it never crashes mid-run.

Retrain when you: add a substantially different class of phases to your work,
change the d-grid/preprocessing conventions, or see the learned shortlist
underperform cosine on a known-truth dataset.
