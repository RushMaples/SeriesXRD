# Training the Step-3b learned scorer on WashU RIS

Read [`docs/ml-training.md`](ml-training.md) first — it is the full,
cluster-agnostic guide (data-quality gate, corpus building, training flags,
validation gates, deployment). This page only covers the RIS-specific parts:
environment layout, storage paths, and the LSF job.

---

## RIS environment setup

RIS compute nodes run jobs under IBM LSF inside Docker images. Any image with
Python >= 3.10 and CUDA-enabled PyTorch works — e.g. `pytorch/pytorch` (CUDA
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

If you use the container's own PyTorch instead of a venv, `pip install -e
.[phases]` is enough (skip `[ml]`) — just make sure `python -c "import
torch"` works inside the image.

Run the pathfinder job (`docs/ml-training.md` §5) and the CPU smoke test
(`docs/ml-training.md` §2) from a login/interactive node before queueing a
GPU job — they catch a broken venv, missing corpus path, or bad flag without
burning GPU allocation.

## LSF job (GPU)

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

Adjust queue/group/image to your lab's allocation. Reflection simulation for
a large corpus happens once at startup (cached across epochs), so the first
minutes are pymatgen-bound before the GPU does anything.

## Storage notes

* Keep the repo, venv, workspace, CIF corpus, and output models all under
  your lab's `/storage1/fs1/<lab>/Active/<you>` allocation — home directories
  on RIS are small and not meant for this.
* The CIF corpus and any trained `.pt` models are reusable across jobs; do
  not re-fetch or re-screen the corpus per run.

For everything else — what the model is, the data-quality gate, corpus
commands, training flags, validation gates (A/B), and deployment — see
[`docs/ml-training.md`](ml-training.md).
