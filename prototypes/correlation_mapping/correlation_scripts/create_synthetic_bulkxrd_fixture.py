#!/usr/bin/env python3
"""Create a deterministic BulkXRD-format fixture with known frame relationships."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_h5", type=Path)
    return parser.parse_args()


def gaussian(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def make_pattern(two_theta: np.ndarray, frame: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.array([
        3.6, 4.3, 5.1, 6.2, 7.1, 8.4, 9.2, 10.4, 11.6, 12.7,
        13.8, 15.0, 16.2, 17.3, 18.6, 19.8, 21.0, 22.1, 23.4, 24.6, 26.0, 27.1,
    ])
    base_amp = 18.0 + 9.0 * (1.0 + np.sin(np.arange(centers.size) * 1.7))
    group = frame // 2
    shift = [0.0, 0.08, -0.04, 0.16][group]
    modulation = np.ones_like(base_amp)
    if group == 1:
        modulation *= 0.92 + 0.16 * np.cos(np.arange(centers.size) * 0.6)
    elif group == 2:
        modulation *= np.where(np.arange(centers.size) % 3 == 0, 0.25, 1.35)
    elif group == 3:
        modulation *= np.where(np.arange(centers.size) % 2 == 0, 1.55, 0.35)
    frame_scale = 1.0 + (0.015 if frame % 2 else 0.0)
    signal = 35.0 + 0.25 * two_theta + 1.5 * np.sin(two_theta * 0.75)
    for index, center in enumerate(centers):
        signal += gaussian(
            two_theta,
            center + shift,
            0.035 + 0.008 * (index % 4),
            base_amp[index] * modulation[index] * frame_scale,
        )
    if group == 3:
        signal += gaussian(two_theta, 14.35 + shift, 0.055, 55.0)
    return signal + rng.normal(0.0, 1e-4, two_theta.size)


def main() -> None:
    args = parse_args()
    output = args.output_h5.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_path = output.with_name("synthetic_accepted_mask.npz")
    mask = np.zeros((64, 64), dtype=bool)
    mask[:3, :] = True
    mask[:, :2] = True
    mask[28:34, 30:36] = True
    np.savez_compressed(
        mask_path,
        mask=mask,
        metadata={
            "origin": "upper",
            "convention": "pyFAI (origin upper-left, True = masked pixel)",
            "shape": list(mask.shape),
            "fixture": "known-pair correlation validation",
        },
    )
    mask_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()

    wavelength_angstrom = 0.3344
    two_theta = np.linspace(3.0, 28.0, 2501)
    q = 4.0 * np.pi * np.sin(np.radians(two_theta / 2.0)) / wavelength_angstrom
    rng = np.random.default_rng(4310)
    robust = np.vstack([make_pattern(two_theta, frame, rng) for frame in range(8)]).astype("f4")
    sigmaclip = robust + rng.normal(0.0, 5e-5, robust.shape).astype("f4")
    mean = sigmaclip.copy()
    # Narrow radial peaks present only in the azimuthal mean mimic sparse
    # single-crystal reflections rejected by the azimuthal median. Their
    # pressure-dependent shifts and areas exercise the dedicated spots channel.
    for frame in range(mean.shape[0]):
        group = frame // 2
        shift = [0.0, 0.06, -0.03, 0.12][group]
        repeat_scale = 1.0 + (0.02 if frame % 2 else 0.0)
        mean[frame] += gaussian(two_theta, 6.75 + shift, 0.022, 150.0 * repeat_scale)
        mean[frame] += gaussian(two_theta, 14.20 + shift, 0.028, (90.0 + 18.0 * group) * repeat_scale)
        if group >= 2:
            mean[frame] += gaussian(two_theta, 22.65 + shift, 0.025, 125.0 * repeat_scale)
    mean += rng.normal(0.0, 0.03, mean.shape).astype("f4")

    pressures = np.repeat([0.0, 5.0, 10.0, 15.0], 2)
    names = np.array(
        [f"synthetic_{pressure:g}GPa_repeat{frame % 2 + 1}.tif" for frame, pressure in enumerate(pressures)],
        dtype=object,
    )
    with h5py.File(output, "w") as h5:
        h5.attrs.update({
            "schema_version": "1",
            "tool": "bulkxrd.reduce",
            "tool_version": "synthetic-validation-fixture",
            "unit": "q_A^-1",
            "mask_file": str(mask_path),
            "mask_sha256": mask_hash,
            "poni_text": f"# Synthetic PONI\nWavelength: {wavelength_angstrom * 1e-10:.12g}\n",
            "radial_written": True,
        })
        patterns = h5.create_group("patterns")
        patterns.create_dataset("radial", data=q)
        patterns.create_dataset("intensity", data=mean)
        patterns.create_dataset("intensity_robust", data=robust)
        patterns.create_dataset("intensity_sigmaclip", data=sigmaclip)
        frames = h5.create_group("frames")
        frames.create_dataset("filename", data=names, dtype=h5py.string_dtype("utf-8"))
        frames.create_dataset("frame_index", data=np.arange(8, dtype="i8"))
        frames.create_dataset("pressure", data=pressures)
        frames.create_dataset("ok", data=np.ones(8, dtype=bool))
        frames.create_dataset("excluded", data=np.zeros(8, dtype=bool))
    print(f"Wrote synthetic BulkXRD fixture: {output}")


if __name__ == "__main__":
    main()
