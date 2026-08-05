#!/usr/bin/env python3
"""Dataset adapters for uniform-correlation-v2.1.

The scientific runner consumes one explicit frame/scan/pressure grid and a
path for every requested channel.  Two layout adapters produce that same
contract:

* ``handoff`` keeps compatibility with the existing ``spots_channel`` and
  ``fit_channel`` directory layout; and
* ``direct`` accepts a fully generic manifest containing ``channel`` and
  ``file_path`` columns.

Neither adapter contains material-specific q ranges, peak positions, or
tracking expectations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from uniform_xy_input import FrameInput, read_handoff_manifest, resolve_channel_paths


DIRECT_REQUIRED = {"frame", "scan", "pressure_GPa", "channel", "file_path"}
HANDOFF_REQUIRED = {"frame", "scan", "pressure_GPa", "cover_excluded", "filename"}


@dataclass(frozen=True)
class InputDataset:
    """One channel-aligned analysis grid returned by either input adapter."""

    frames: tuple[FrameInput, ...]
    pressures: tuple[float, ...]
    scans: tuple[str, ...]
    paths_by_channel: Mapping[str, tuple[Path, ...]]
    selected_manifest_rows: tuple[Mapping[str, str], ...]
    input_mode: str
    excluded_rows: int


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def detect_manifest_mode(path: Path) -> str:
    """Return ``direct`` or ``handoff`` from the manifest header."""

    fields, _rows = _read_rows(path)
    field_set = set(fields)
    if DIRECT_REQUIRED.issubset(field_set):
        return "direct"
    if HANDOFF_REQUIRED.issubset(field_set):
        return "handoff"
    raise ValueError(
        "manifest is neither direct nor handoff format; direct requires "
        f"{sorted(DIRECT_REQUIRED)}, handoff requires {sorted(HANDOFF_REQUIRED)}"
    )


def _excluded(row: Mapping[str, str]) -> bool:
    raw = row.get("excluded", row.get("cover_excluded", "0"))
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_direct_path(manifest: Path, input_root: Path, raw: str) -> Path:
    candidate = Path(str(raw).strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    manifest_relative = (manifest.parent / candidate).resolve()
    root_relative = (input_root / candidate).resolve()
    if manifest_relative.is_file() or not root_relative.is_file():
        return manifest_relative
    return root_relative


def read_direct_manifest(
    manifest: Path,
    input_root: Path,
    channels: Sequence[str],
    max_scans: int | None = None,
) -> InputDataset:
    """Read the generic row-per-channel manifest.

    Required columns are ``frame,scan,pressure_GPa,channel,file_path``.
    ``excluded`` (or ``cover_excluded``) is optional and defaults to false.
    Relative file paths are resolved against the manifest directory first,
    then the supplied input root.  Every requested channel must contain the
    exact same frame/scan/pressure keys, preventing silent cross-channel or
    cross-scan pairing.
    """

    fields, rows = _read_rows(manifest)
    missing = DIRECT_REQUIRED.difference(fields)
    if missing:
        raise ValueError(f"direct manifest is missing columns: {sorted(missing)}")
    requested = tuple(dict.fromkeys(str(item).strip() for item in channels if str(item).strip()))
    if not requested:
        raise ValueError("at least one channel must be requested")

    all_scans = sorted({str(row["scan"]).strip() for row in rows})
    if max_scans is not None:
        all_scans = all_scans[: max(1, int(max_scans))]
    selected = [row for row in rows if str(row["scan"]).strip() in all_scans]
    included = [
        row
        for row in selected
        if not _excluded(row) and str(row["channel"]).strip() in requested
    ]
    if not included:
        raise ValueError("direct manifest has no included rows for the requested channels")

    rows_by_channel: dict[str, dict[tuple[int, str, float], dict[str, str]]] = {
        channel: {} for channel in requested
    }
    for row in included:
        channel = str(row["channel"]).strip()
        try:
            key = (
                int(row["frame"]),
                str(row["scan"]).strip(),
                float(row["pressure_GPa"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid direct manifest frame/pressure row: {row}") from error
        if key in rows_by_channel[channel]:
            raise ValueError(f"duplicate direct manifest row for channel={channel}, key={key}")
        rows_by_channel[channel][key] = row

    reference_keys = set(rows_by_channel[requested[0]])
    if not reference_keys:
        raise ValueError(f"requested channel {requested[0]!r} has no included rows")
    for channel in requested[1:]:
        keys = set(rows_by_channel[channel])
        if keys != reference_keys:
            missing_keys = sorted(reference_keys - keys)[:8]
            extra_keys = sorted(keys - reference_keys)[:8]
            raise ValueError(
                f"channel grids differ for {channel!r}; missing={missing_keys}, extra={extra_keys}"
            )

    ordered_keys = sorted(reference_keys, key=lambda item: (item[1], item[2], item[0]))
    frame_owners: dict[int, tuple[str, float]] = {}
    for frame, scan, pressure in ordered_keys:
        owner = (scan, pressure)
        previous = frame_owners.setdefault(frame, owner)
        if previous != owner:
            raise ValueError(
                f"direct manifest frame {frame} is reused for {previous} and {owner}; "
                "frame identifiers must be globally unique"
            )
    pressure_values = sorted({key[2] for key in ordered_keys})
    pressure_index = {value: index for index, value in enumerate(pressure_values)}
    frames: list[FrameInput] = []
    for frame, scan, pressure in ordered_keys:
        source = rows_by_channel[requested[0]][(frame, scan, pressure)]
        original_filename = str(source.get("original_filename", "")).strip()
        if not original_filename:
            original_filename = Path(str(source["file_path"])).name
        frames.append(
            FrameInput(
                frame=frame,
                scan=scan,
                pressure=pressure,
                pressure_index=pressure_index[pressure],
                original_filename=original_filename,
            )
        )

    paths_by_channel: dict[str, tuple[Path, ...]] = {}
    missing_files: list[str] = []
    for channel in requested:
        channel_paths: list[Path] = []
        for key in ordered_keys:
            row = rows_by_channel[channel][key]
            path = _resolve_direct_path(manifest, input_root, row["file_path"])
            if not path.is_file():
                missing_files.append(f"channel={channel}, key={key}, path={path}")
            channel_paths.append(path)
        paths_by_channel[channel] = tuple(channel_paths)
    if missing_files:
        raise FileNotFoundError("; ".join(missing_files[:12]))

    selected_scans = tuple(sorted({frame.scan for frame in frames}))
    return InputDataset(
        frames=tuple(frames),
        pressures=tuple(pressure_values),
        scans=selected_scans,
        paths_by_channel=paths_by_channel,
        selected_manifest_rows=tuple(dict(row) for row in selected),
        input_mode="direct",
        excluded_rows=sum(_excluded(row) for row in selected),
    )


def read_input_dataset(
    input_root: Path,
    manifest: Path,
    channels: Sequence[str],
    *,
    input_mode: str = "auto",
    max_scans: int | None = None,
) -> InputDataset:
    """Normalize either supported manifest layout to :class:`InputDataset`."""

    root = input_root.expanduser().resolve()
    manifest_path = manifest.expanduser().resolve()
    requested = tuple(dict.fromkeys(str(item).strip() for item in channels if str(item).strip()))
    mode = detect_manifest_mode(manifest_path) if input_mode == "auto" else input_mode
    if mode not in {"direct", "handoff"}:
        raise ValueError("input_mode must be auto, direct, or handoff")
    if mode == "direct":
        return read_direct_manifest(manifest_path, root, requested, max_scans)

    unsupported = sorted(set(requested) - {"spots", "fit"})
    if unsupported:
        raise ValueError(
            "handoff adapter supports only spots/fit; use a direct manifest for "
            f"channels {unsupported}"
        )
    frames, pressures, scans, selected = read_handoff_manifest(manifest_path, max_scans)
    return InputDataset(
        frames=tuple(frames),
        pressures=tuple(pressures),
        scans=tuple(scans),
        paths_by_channel={
            channel: tuple(resolve_channel_paths(root, frames, channel)) for channel in requested
        },
        selected_manifest_rows=tuple(dict(row) for row in selected),
        input_mode="handoff",
        excluded_rows=sum(
            str(row.get("cover_excluded", "")).strip().lower() in {"1", "true", "yes", "y"}
            for row in selected
        ),
    )


__all__ = [
    "DIRECT_REQUIRED",
    "HANDOFF_REQUIRED",
    "InputDataset",
    "detect_manifest_mode",
    "read_direct_manifest",
    "read_input_dataset",
]
