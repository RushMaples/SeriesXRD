#!/usr/bin/env python3
"""I/O, provenance, and plotting helpers for ``uniform-correlation-v2``.

This module deliberately contains no scientific decisions.  It serializes the
arrays returned by the frozen algorithms and gives missing/low-support cells a
visual encoding that cannot be confused with a numerical zero.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MISSING_COLOR = "#BDBDBD"


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.") or "item"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    # bool is a subclass of int in Python, so it must be handled first or the
    # serialized execution semantics change True/False into 1/0 and invalidate
    # their recorded SHA256.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return len(files), digest.hexdigest()


def write_rows_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return "" if not np.isfinite(float(value)) else f"{float(value):.12g}"
    return value


def write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray, row_header: str = "pressure") -> None:
    matrix = np.asarray(matrix)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"matrix shape {matrix.shape} does not match {len(labels)} labels")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([row_header, *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[_csv_value(value) for value in row]])


def plot_matrix(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
    title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
    insufficient_mask: np.ndarray | None = None,
    integer_annotations: bool = False,
) -> None:
    """Plot a full symmetric matrix; missing values are gray, not color-scale zero."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"matrix shape {matrix.shape} does not match {len(labels)} labels")
    size = max(7.2, min(16.0, 0.42 * len(labels) + 3.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad(MISSING_COLOR)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=color_map, vmin=vmin, vmax=vmax, interpolation="nearest")
    if insufficient_mask is not None:
        mask = np.asarray(insufficient_mask, dtype=bool)
        if mask.shape != matrix.shape:
            raise ValueError("insufficient-support mask shape differs from matrix")
        if np.any(mask):
            ax.contourf(
                np.arange(mask.shape[1]),
                np.arange(mask.shape[0]),
                mask.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
            )
    if integer_annotations and len(labels) <= 20:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{int(round(value))}", ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Pressure / window")
    ax.set_ylabel("Pressure / window")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def build_artifact_index(root: Path, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    exclude = exclude or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        row: dict[str, Any] = {
            "relative_path": relative,
            "extension": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "shape": "",
            "finite_count": "",
            "nan_count": "",
        }
        if path.suffix.lower() == ".npz":
            try:
                with np.load(path, allow_pickle=False) as archive:
                    shapes = []
                    finite = 0
                    nan_count = 0
                    for key in archive.files:
                        value = archive[key]
                        shapes.append(f"{key}:{'x'.join(str(n) for n in value.shape)}")
                        if np.issubdtype(value.dtype, np.number):
                            finite += int(np.count_nonzero(np.isfinite(value)))
                            if np.issubdtype(value.dtype, np.floating):
                                nan_count += int(np.count_nonzero(~np.isfinite(value)))
                    row["shape"] = ";".join(shapes)
                    row["finite_count"] = finite
                    row["nan_count"] = nan_count
            except Exception as error:  # artifact index must not hide an unreadable array
                row["shape"] = f"ERROR:{error}"
        rows.append(row)
    return rows
