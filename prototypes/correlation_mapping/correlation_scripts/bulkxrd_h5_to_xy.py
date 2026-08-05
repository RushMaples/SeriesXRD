#!/usr/bin/env python3
"""Export mask-applied BulkXRD HDF5 patterns for the correlation suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np


SOURCE_PATHS = {
    "mean": "patterns/intensity",
    "robust": "patterns/intensity_robust",
    "sigmaclip": "patterns/intensity_sigmaclip",
    "straightened": "patterns/intensity_straightened",
    "straightened-robust": "patterns/intensity_straightened_robust",
    "clean": "background/clean",
    "spots": "background/spot_residual",
    "residual": "residual/clean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a BulkXRD reduced/analysis HDF5 into one two-column .xy file per frame."
    )
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=["auto", *SOURCE_PATHS],
        default="auto",
        help="Pattern channel. auto prefers sigma-clipped mask-applied data.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Explicit HDF5 dataset path; overrides --source.",
    )
    parser.add_argument(
        "--axis",
        choices=["two-theta", "native"],
        default="two-theta",
        help="The correlation suite is calibrated in two-theta degrees.",
    )
    parser.add_argument(
        "--wavelength-angstrom",
        type=float,
        default=None,
        help="Required for q-to-two-theta conversion only when the PONI metadata lacks wavelength.",
    )
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--include-failed", action="store_true")
    return parser.parse_args()


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalized_unit(value: object) -> str:
    return decode(value).strip().lower().replace(" ", "")


def wavelength_from_poni_text(text: str) -> float | None:
    match = re.search(r"^\s*Wavelength\s*:\s*([0-9.eE+-]+)", text, re.MULTILINE | re.IGNORECASE)
    if not match:
        return None
    value_m = float(match.group(1))
    return value_m * 1e10 if value_m > 0 else None


def wavelength_from_h5(h5: h5py.File, input_path: Path) -> tuple[float | None, str]:
    for key in ("wavelength_angstrom", "wavelength_A"):
        if key in h5.attrs:
            value = float(h5.attrs[key])
            if value > 0:
                return value, f"attribute:{key}"
    if "wavelength_m" in h5.attrs:
        value = float(h5.attrs["wavelength_m"]) * 1e10
        if value > 0:
            return value, "attribute:wavelength_m"
    if "wavelength" in h5.attrs:
        value = float(h5.attrs["wavelength"])
        if value > 0:
            # BulkXRD analysis files store this generic attribute in angstrom;
            # accept metres as well for compatibility with other producers.
            return (value * 1e10 if value < 1e-6 else value), "attribute:wavelength"
    if "poni_text" in h5.attrs:
        value = wavelength_from_poni_text(decode(h5.attrs["poni_text"]))
        if value:
            return value, "attribute:poni_text"

    source = decode(h5.attrs.get("source_reduced", "")).strip()
    if source:
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = input_path.parent / source_path
        if source_path.is_file() and source_path.resolve() != input_path.resolve():
            with h5py.File(source_path, "r") as reduced:
                if "poni_text" in reduced.attrs:
                    value = wavelength_from_poni_text(decode(reduced.attrs["poni_text"]))
                    if value:
                        return value, f"source_reduced:{source_path}"
    return None, "unavailable"


def radial_dataset(h5: h5py.File) -> tuple[np.ndarray, str]:
    for path in ("patterns/radial", "radial"):
        if path in h5:
            return np.asarray(h5[path][:], dtype=float), path
    raise KeyError("BulkXRD HDF5 has neither /patterns/radial nor /radial")


def source_array(h5: h5py.File, requested: str, explicit: str) -> tuple[np.ndarray, str]:
    if explicit:
        path = explicit.strip("/")
        if path not in h5:
            raise KeyError(f"Requested dataset /{path} is absent")
        return np.asarray(h5[path][:], dtype=float), path

    if requested != "auto":
        path = SOURCE_PATHS[requested]
        if path in h5:
            return np.asarray(h5[path][:], dtype=float), path
        if requested == "sigmaclip" and "background/clean" in h5 and "background/sigmaclip_residual" in h5:
            data = np.asarray(h5["background/clean"][:], dtype=float)
            data += np.asarray(h5["background/sigmaclip_residual"][:], dtype=float)
            return data, "background/clean+background/sigmaclip_residual"
        if (
            requested == "spots"
            and "patterns/intensity" in h5
            and "patterns/intensity_robust" in h5
        ):
            data = np.asarray(h5["patterns/intensity"][:], dtype=float)
            data -= np.asarray(h5["patterns/intensity_robust"][:], dtype=float)
            return data, "patterns/intensity-patterns/intensity_robust"
        raise KeyError(f"Requested source {requested!r} is absent at /{path}")

    if "background/clean" in h5 and "background/sigmaclip_residual" in h5:
        data = np.asarray(h5["background/clean"][:], dtype=float)
        data += np.asarray(h5["background/sigmaclip_residual"][:], dtype=float)
        return data, "background/clean+background/sigmaclip_residual"
    preference = (
        "patterns/intensity_sigmaclip",
        "patterns/intensity_straightened_robust",
        "patterns/intensity_robust",
        "background/clean",
        "patterns/intensity",
    )
    for path in preference:
        if path in h5:
            return np.asarray(h5[path][:], dtype=float), path
    raise KeyError("No supported BulkXRD pattern channel found")


def frame_vector(h5: h5py.File, path: str, n: int, default: object) -> np.ndarray:
    if path not in h5:
        return np.full(n, default)
    values = np.asarray(h5[path][:])
    if len(values) != n:
        raise ValueError(f"/{path} has {len(values)} rows, expected {n}")
    return values


def convert_axis(
    radial: np.ndarray,
    unit: str,
    axis: str,
    wavelength_angstrom: float | None,
) -> tuple[np.ndarray, str]:
    if axis == "native":
        return radial, unit or "unknown"
    if unit in {"2th_deg", "2theta_deg", "2theta", "2th"}:
        return radial, "2th_deg"
    if unit in {"2th_rad", "2theta_rad"}:
        return np.degrees(radial), "2th_deg"
    if unit in {"q_a^-1", "q_a-1", "q_angstrom^-1", "q_å^-1"}:
        if not wavelength_angstrom:
            raise ValueError("q_A^-1 input needs wavelength; pass --wavelength-angstrom")
        argument = radial * wavelength_angstrom / (4.0 * math.pi)
        if np.nanmax(np.abs(argument)) > 1.0 + 1e-9:
            raise ValueError("q range is incompatible with the supplied wavelength")
        return np.degrees(2.0 * np.arcsin(np.clip(argument, -1.0, 1.0))), "2th_deg"
    if unit in {"q_nm^-1", "q_nm-1"}:
        if not wavelength_angstrom:
            raise ValueError("q_nm^-1 input needs wavelength; pass --wavelength-angstrom")
        q_angstrom = radial / 10.0
        argument = q_angstrom * wavelength_angstrom / (4.0 * math.pi)
        return np.degrees(2.0 * np.arcsin(np.clip(argument, -1.0, 1.0))), "2th_deg"
    raise ValueError(f"Cannot convert BulkXRD unit {unit!r} to two-theta")


def safe_stem(text: str, fallback: str) -> str:
    stem = Path(text.split("::", 1)[0]).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.")
    return cleaned or fallback


def main() -> None:
    args = parse_args()
    input_path = args.input_h5.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"BulkXRD HDF5 not found: {input_path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        radial, radial_path = radial_dataset(h5)
        patterns, source_name = source_array(h5, args.source, args.dataset)
        if patterns.ndim != 2 or patterns.shape[1] != radial.size:
            raise ValueError(
                f"/{source_name} shape {patterns.shape} does not match radial length {radial.size}"
            )
        n_frames = patterns.shape[0]
        names = frame_vector(h5, "frames/filename", n_frames, "")
        pressure = frame_vector(h5, "frames/pressure", n_frames, np.nan).astype(float)
        excluded = frame_vector(h5, "frames/excluded", n_frames, False).astype(bool)
        ok = frame_vector(h5, "frames/ok", n_frames, True).astype(bool)
        unit = normalized_unit(h5.attrs.get("unit", ""))
        wavelength, wavelength_source = wavelength_from_h5(h5, input_path)
        if args.wavelength_angstrom is not None:
            wavelength = float(args.wavelength_angstrom)
            wavelength_source = "command-line"
        mask_file = decode(h5.attrs.get("mask_file", ""))
        mask_sha256 = decode(h5.attrs.get("mask_sha256", ""))
        tool = decode(h5.attrs.get("tool", ""))

    exported_axis, exported_unit = convert_axis(radial, unit, args.axis, wavelength)
    manifest_rows: list[dict[str, object]] = []
    for index in range(n_frames):
        if excluded[index] and not args.include_excluded:
            continue
        if not ok[index] and not args.include_failed:
            continue
        original_name = decode(names[index]) if decode(names[index]).strip() else f"frame_{index:04d}"
        pressure_token = f"{pressure[index]:g}GPa_" if np.isfinite(pressure[index]) else ""
        output_name = f"{index:04d}_{pressure_token}{safe_stem(original_name, f'frame_{index:04d}')}.xy"
        output_path = args.out_dir / output_name
        intensity = np.asarray(patterns[index], dtype=float)
        keep = np.isfinite(exported_axis) & np.isfinite(intensity)
        if np.count_nonzero(keep) < 8:
            continue
        header = (
            f"BulkXRD source={source_name}; frame={index}; original={original_name}; "
            f"axis={exported_unit}; mask_file={mask_file}"
        )
        np.savetxt(output_path, np.column_stack([exported_axis[keep], intensity[keep]]), header=header)
        manifest_rows.append(
            {
                "frame_index": index,
                "output_xy": output_name,
                "original_filename": original_name,
                "pressure_gpa": "" if not np.isfinite(pressure[index]) else f"{pressure[index]:.12g}",
                "excluded": int(excluded[index]),
                "ok": int(ok[index]),
            }
        )

    manifest_path = args.out_dir / "conversion_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]) if manifest_rows else [
            "frame_index", "output_xy", "original_filename", "pressure_gpa", "excluded", "ok"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "input_h5": str(input_path),
        "bulkxrd_tool": tool,
        "source": source_name,
        "radial_dataset": radial_path,
        "input_unit": unit,
        "output_unit": exported_unit,
        "wavelength_angstrom": wavelength,
        "wavelength_source": wavelength_source,
        "mask_file": mask_file,
        "mask_sha256": mask_sha256,
        "frames_total": n_frames,
        "frames_exported": len(manifest_rows),
    }
    (args.out_dir / "conversion_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if not manifest_rows:
        raise SystemExit("No usable frames were exported")
    print(f"Exported {len(manifest_rows)} BulkXRD frames to {args.out_dir}")
    print(f"Source channel: {source_name}; axis: {unit} -> {exported_unit}")


if __name__ == "__main__":
    main()
