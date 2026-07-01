"""Step 3b training pipeline — the learned scorer behind ``bulkxrd[ml]``.

Trains the RADAR-PD-style similarity model :class:`ml_scorer.TorchScorer`
consumes: given a ``(2, P)`` stack of (measured pattern, candidate fingerprint)
on the shared d-grid, output a similarity in [0, 1]. Everything the model sees
comes through the SAME preprocessing the experimental pipeline uses (the
SimXRD-4M lesson: no sim-to-real preprocessing gap), and everything it learns is
a *proposal* — the deterministic Step-3a matcher still verifies every candidate.

Training pairs (pure numpy, testable without torch):
  * measured side — a DAC-augmented multi-phase mixture from
    :func:`ml_simulate.build_augmented_dataset` (EOS pressure shift, texture,
    broadening, drift, diamond spikes, background humps, truncation, noise);
  * positive — a phase present in the mixture, simulated *clean* at its true
    pressure (what :mod:`ml_rank` computes at inference time);
  * pressure negative — the same present phase simulated at a wrong pressure
    (teaches the model that pressure-consistency matters — the lattice-nudge
    discrimination the deterministic prior already enforces);
  * absent negative — a phase not in the mixture, simulated at the mixture's
    pressure (hard negative) or a random one.

Model (RADAR-PD-inspired, compact): strided 1-D conv stem over the paired
patterns (learned downsampling — strides, not max-pooling, so peak-position
information survives; cf. SimXRD-4M's no-pooling finding), a small multi-head
self-attention encoder, mean+max pooled head → sigmoid. ~1 M parameters —
minutes per epoch on a laptop CPU for small libraries, fast on a GPU.

Run on WashU RIS (or any GPU box)::

    pip install -e .[phases,ml]          # pymatgen for reflections, torch for training
    bulkxrd-ml-train --workspace /path/to/workspace --out scorer.pt \
        --epochs 20 --mixtures-per-epoch 512 --device cuda
    # use it: rank_candidates(..., scorer="torch:scorer.pt")

Importing this module never imports torch (safe for the worker/CLI); only
:func:`train` / :func:`main` do, and a missing torch raises the same instructive
error :class:`ml_scorer.TorchScorer` gives.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phases import Phase, has_pressure_dof, pymatgen_available
from .identify import phase_reflections
from .mldata import make_d_grid, simulate_training_pattern
from .ml_simulate import AugmentConfig, simulate_augmented_pattern

Reflections = Tuple[np.ndarray, np.ndarray, list]


# ---------------------------------------------------------------------------
# Pair generation (numpy — no torch)
# ---------------------------------------------------------------------------

def generate_pairs(
    phases: "Sequence[Phase]",
    *,
    n_mixtures: int = 256,
    max_phases_per_pattern: int = 2,
    pressures: "Optional[Sequence[float]]" = None,
    d_grid: "Optional[np.ndarray]" = None,
    cfg: "Optional[AugmentConfig]" = None,
    reflections: "Optional[Dict[str, Reflections]]" = None,
    wrong_pressure_offset: float = 10.0,
    fwhm_d: float = 0.03,
    seed: "Optional[int]" = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(X (M, 2, P) float32, y (M,) float32)`` training pairs.

    Per augmented mixture: one positive per present phase (candidate at its true
    pressure), one pressure-negative per pressure-capable present phase (same
    phase at ``±wrong_pressure_offset`` GPa), and one absent-negative (a phase
    not in the mixture at the mixture's pressure). Candidate fingerprints are
    rendered with the same ``simulate_training_pattern`` the inference-time
    scorers use. ``reflections`` bypasses pymatgen (name -> (d0, w, hkl)).
    """
    grid = make_d_grid() if d_grid is None else np.asarray(d_grid, dtype=float)
    cfg = cfg or AugmentConfig()
    rng = np.random.default_rng(seed)
    phases = list(phases)
    if len(phases) < 2:
        raise ValueError("Pair generation needs at least 2 phases (for absent negatives).")
    if reflections is None:
        if not pymatgen_available():
            raise RuntimeError("pymatgen is required to simulate reflections "
                               "(pip install bulkxrd[phases]), or pass reflections=.")
        reflections = {p.name: phase_reflections(p) for p in phases}
    if pressures is None:
        pressures = np.arange(0.0, 101.0, 5.0)
    pressures = np.asarray(pressures, float)
    p_lo, p_hi = float(pressures.min()), float(pressures.max())
    k_max = max(1, min(int(max_phases_per_pattern), len(phases)))

    def _cand(ph: Phase, P: float) -> np.ndarray:
        return simulate_training_pattern(ph, float(P), grid, refl=reflections[ph.name],
                                         fwhm_d=fwhm_d).astype("f4")

    Xs: List[np.ndarray] = []
    ys: List[float] = []
    for _ in range(int(n_mixtures)):
        k = 1 if k_max == 1 else int(rng.integers(1, k_max + 1))
        chosen = list(rng.choice(len(phases), size=k, replace=False))
        present = [phases[j] for j in chosen]
        ps = [float(rng.choice(pressures)) if has_pressure_dof(p) else 0.0 for p in present]
        meas = simulate_augmented_pattern(present, ps, reflections, grid, cfg, rng)

        for ph, P in zip(present, ps):
            # positive: the phase at its true pressure
            Xs.append(np.stack([meas, _cand(ph, P)]))
            ys.append(1.0)
            # pressure negative: same phase, wrong pressure
            if has_pressure_dof(ph):
                off = wrong_pressure_offset * (1.0 if rng.random() < 0.5 else -1.0)
                P_wrong = min(max(P + off, p_lo), p_hi)
                if abs(P_wrong - P) >= 0.5 * wrong_pressure_offset:
                    Xs.append(np.stack([meas, _cand(ph, P_wrong)]))
                    ys.append(0.0)
        # absent negative: a phase not in the mixture, at the mixture's pressure
        absent = [j for j in range(len(phases)) if j not in chosen]
        if absent:
            j = int(rng.choice(absent))
            ph_a = phases[j]
            P_a = ps[0] if has_pressure_dof(ph_a) else 0.0
            Xs.append(np.stack([meas, _cand(ph_a, P_a)]))
            ys.append(0.0)

    X = np.asarray(Xs, dtype="f4")
    y = np.asarray(ys, dtype="f4")
    order = rng.permutation(len(y))
    return X[order], y[order]


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the rank statistic (no sklearn)."""
    y_true = np.asarray(y_true, float).ravel()
    y_score = np.asarray(y_score, float).ravel()
    pos = y_score[y_true > 0.5]
    neg = y_score[y_true <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


# ---------------------------------------------------------------------------
# Model + training (torch — imported lazily)
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as e:
        raise RuntimeError(
            "Training the learned scorer needs PyTorch. Install the optional "
            "extra with `pip install bulkxrd[ml]` (on RIS: load a CUDA module "
            "first for GPU wheels).") from e


def build_model(n_points: int = 3501, *, channels: "Sequence[int]" = (32, 64, 96, 128),
                n_heads: int = 4, n_attn_layers: int = 2):
    """The pair scorer: strided conv stem → self-attention → pooled sigmoid head.

    Input ``(B, 2, n_points)`` — measured + candidate fingerprints — output
    ``(B, 1)`` in [0, 1]. Strided convs (not pooling) downsample so peak-position
    information is learned rather than discarded; the attention block relates
    measured peaks to candidate peaks across the pattern (RADAR-PD's recipe).
    Traceable (no data-dependent control flow) for TorchScript export.
    """
    torch = _require_torch()
    import torch.nn as nn

    class _AttnBlock(nn.Module):
        """Pre-norm self-attention + FFN. Deliberately explicit rather than
        nn.TransformerEncoder: the built-in's fast-path/nested-tensor branching
        makes torch.jit.trace graphs non-deterministic, and an exported artefact
        must trace cleanly."""

        def __init__(self, d: int, heads: int):
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

        def forward(self, z):
            h = self.norm1(z)
            a, _ = self.attn(h, h, h, need_weights=False)
            z = z + a
            return z + self.ff(self.norm2(z))

    class PairScorer(nn.Module):
        def __init__(self):
            super().__init__()
            convs: "List[nn.Module]" = []
            c_in = 2
            for c_out in channels:
                convs += [nn.Conv1d(c_in, c_out, kernel_size=5, stride=2, padding=2),
                          nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
                          nn.SiLU()]
                c_in = c_out
            self.stem = nn.Sequential(*convs)
            d = channels[-1]
            self.encoder = nn.Sequential(*[_AttnBlock(d, n_heads)
                                           for _ in range(n_attn_layers)])
            self.head = nn.Sequential(
                nn.Linear(2 * d, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

        def forward(self, x):
            z = self.stem(x)                    # (B, C, L)
            z = z.transpose(1, 2)               # (B, L, C)
            z = self.encoder(z)
            pooled = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=1)
            return self.head(pooled)            # (B, 1) in [0, 1]

    return PairScorer()


def train(
    phases: "Sequence[Phase]",
    out_path: "str | Path",
    *,
    reflections: "Optional[Dict[str, Reflections]]" = None,
    epochs: int = 20,
    mixtures_per_epoch: int = 256,
    max_phases_per_pattern: int = 2,
    pressures: "Optional[Sequence[float]]" = None,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "auto",
    seed: int = 0,
    channels: "Sequence[int]" = (32, 64, 96, 128),
    val_fraction: float = 0.15,
    log=print,
) -> Dict[str, Any]:
    """Train the pair scorer and export TorchScript for :class:`TorchScorer`.

    Every epoch regenerates its pairs with a fresh seed (infinite augmentation —
    the simulator is the dataset), holding out ``val_fraction`` for loss/AUC.
    Saves the best-validation-AUC weights, exports TorchScript to ``out_path``
    and verifies the export by scoring through ``ml_scorer.TorchScorer``.
    Returns a manifest (losses, AUCs, n_params, out path).
    """
    torch = _require_torch()
    import torch.nn as nn

    grid = make_d_grid()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = build_model(grid.size, channels=channels).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.BCELoss()
    log(f"[ML-TRAIN] device={dev} params={n_params:,} phases={len(phases)} "
        f"epochs={epochs} mixtures/epoch={mixtures_per_epoch}")

    history: List[Dict[str, float]] = []
    best_auc, best_state = -1.0, None
    for epoch in range(int(epochs)):
        X, y = generate_pairs(phases, n_mixtures=mixtures_per_epoch,
                              max_phases_per_pattern=max_phases_per_pattern,
                              pressures=pressures, d_grid=grid,
                              reflections=reflections, seed=seed + epoch)
        n_val = max(1, int(val_fraction * len(y)))
        Xt, yt = X[:-n_val], y[:-n_val]
        Xv, yv = X[-n_val:], y[-n_val:]

        model.train()
        perm = np.random.default_rng(seed + epoch).permutation(len(yt))
        tot, nb = 0.0, 0
        for a in range(0, len(perm), batch_size):
            idx = perm[a:a + batch_size]
            xb = torch.from_numpy(Xt[idx]).to(dev)
            yb = torch.from_numpy(yt[idx]).to(dev).unsqueeze(1)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xv).to(dev)).cpu().numpy().ravel()
        auc = roc_auc(yv, pv)
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1), "val_auc": auc})
        log(f"[ML-TRAIN] epoch {epoch + 1}/{epochs} loss={tot / max(nb, 1):.4f} "
            f"val_AUC={auc:.4f}")
        if auc == auc and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    out = export_torchscript(model.cpu(), out_path, grid.size)
    manifest = {"out": str(out), "n_params": int(n_params), "device": dev,
                "best_val_auc": float(best_auc), "history": history,
                "n_points": int(grid.size),
                "phases": [p.name for p in phases]}
    log(f"[ML-TRAIN] done -> {out} (best val AUC {best_auc:.4f})")
    return manifest


def export_torchscript(model, out_path: "str | Path", n_points: int) -> Path:
    """Trace + save the model, then verify the artefact loads and scores through
    :class:`ml_scorer.TorchScorer` (the exact consumer contract)."""
    torch = _require_torch()
    import os
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(1, 2, int(n_points))
    # check_trace compares graph structure, which attention fast-paths can make
    # non-deterministic; the meaningful invariant is numeric equivalence, which
    # we assert explicitly on random inputs below.
    traced = torch.jit.trace(model, example, check_trace=False)
    with torch.no_grad():
        for seed in (0, 1):
            g = torch.Generator().manual_seed(seed)
            x = torch.rand(2, 2, int(n_points), generator=g)
            a, b = model(x), traced(x)
            if not torch.allclose(a, b, atol=1e-5):
                raise RuntimeError("Traced model diverges from eager model "
                                   f"(max diff {float((a - b).abs().max()):.2e}).")
    tmp = out.with_name(out.name + ".tmp")
    traced.save(str(tmp))
    os.replace(tmp, out)
    # Contract check: the deployed consumer must accept the artefact.
    from .ml_scorer import TorchScorer
    scorer = TorchScorer(out)
    grid = make_d_grid(n_points=int(n_points))
    probe = Phase(name="_probe", eos={"type": "BM3", "K0": 160, "K0p": 4.0})
    refl = (np.array([2.0, 3.0]), np.array([1.0, 0.5]), ["", ""])
    s, p = scorer.score(np.zeros(int(n_points), "f4"), probe, refl, grid, 10.0)
    if not (0.0 <= s <= 1.0):
        raise RuntimeError(f"Exported model returned {s!r}, outside [0, 1].")
    return out


# ---------------------------------------------------------------------------
# CLI  (bulkxrd-ml-train)
# ---------------------------------------------------------------------------

def main(argv: "Optional[List[str]]" = None) -> int:
    p = argparse.ArgumentParser(
        prog="bulkxrd-ml-train",
        description="Train the Step-3b learned pair scorer (bulkxrd[ml]) on "
                    "DAC-augmented simulated mixtures of the reference library.")
    p.add_argument("--workspace", default="", help="Workspace with the phase library.")
    p.add_argument("--phases", default="", help="Comma-separated subset (default: all "
                                                "simulatable library phases).")
    p.add_argument("--out", default="scorer.pt", help="Output TorchScript path.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--mixtures-per-epoch", type=int, default=256)
    p.add_argument("--max-phases", type=int, default=2,
                   help="Max phases per simulated mixture. Default 2.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--p-max", type=float, default=100.0)
    p.add_argument("--p-step", type=float, default=5.0)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda. Default auto.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from .phases import load_library
    lib = load_library(args.workspace or Path.cwd())
    names = [s.strip() for s in (args.phases or "").split(",") if s.strip()]
    pool = ([lib[n] for n in names if n in lib] if names
            else [ph for ph in lib.values() if ph.has_structure()])
    if len(pool) < 2:
        print("[ERROR] need at least 2 simulatable phases (library empty or "
              "--phases too narrow).", flush=True)
        return 1
    try:
        train(pool, args.out,
              epochs=args.epochs, mixtures_per_epoch=args.mixtures_per_epoch,
              max_phases_per_pattern=args.max_phases,
              pressures=np.arange(0.0, args.p_max + 1e-9, args.p_step),
              batch_size=args.batch_size, lr=args.lr, device=args.device,
              seed=args.seed)
    except RuntimeError as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
