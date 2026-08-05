#!/usr/bin/env python3
"""Focused tests for the v2.1 generic direct-manifest adapter."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from uniform_xy_input_v21 import detect_manifest_mode, read_input_dataset


def _write_xy(path: Path) -> None:
    x = np.linspace(1.0, 2.0, 128)
    y = np.sin(x) + 2.0
    path.write_text("\n".join(f"{a:.8f} {b:.8f}" for a, b in zip(x, y, strict=True)) + "\n")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["frame", "scan", "pressure_GPa", "channel", "file_path", "excluded"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class DirectManifestTests(unittest.TestCase):
    def test_aligns_channels_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            rows: list[dict[str, object]] = []
            for frame, pressure in ((1, 3.5), (2, 4.0)):
                for channel in ("spots", "fit"):
                    relative = f"data/{channel}_{frame}.xy"
                    target = tmp_path / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _write_xy(target)
                    rows.append(
                        {
                            "frame": frame,
                            "scan": "scan_a",
                            "pressure_GPa": pressure,
                            "channel": channel,
                            "file_path": relative,
                            "excluded": 0,
                        }
                    )
            manifest = tmp_path / "manifest.csv"
            _write_manifest(manifest, rows)
            self.assertEqual(detect_manifest_mode(manifest), "direct")
            dataset = read_input_dataset(
                tmp_path, manifest, ("spots", "fit"), input_mode="auto"
            )
            self.assertEqual(dataset.input_mode, "direct")
            self.assertEqual(dataset.pressures, (3.5, 4.0))
            self.assertEqual([frame.frame for frame in dataset.frames], [1, 2])
            self.assertTrue(
                all(
                    path.is_file()
                    for paths in dataset.paths_by_channel.values()
                    for path in paths
                )
            )

    def test_rejects_different_channel_grids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            for name in ("spots_1.xy", "fit_1.xy", "spots_2.xy"):
                _write_xy(tmp_path / name)
            rows = [
                {"frame": 1, "scan": "a", "pressure_GPa": 1, "channel": "spots", "file_path": "spots_1.xy", "excluded": 0},
                {"frame": 1, "scan": "a", "pressure_GPa": 1, "channel": "fit", "file_path": "fit_1.xy", "excluded": 0},
                {"frame": 2, "scan": "a", "pressure_GPa": 2, "channel": "spots", "file_path": "spots_2.xy", "excluded": 0},
            ]
            manifest = tmp_path / "manifest.csv"
            _write_manifest(manifest, rows)
            with self.assertRaisesRegex(ValueError, "channel grids differ"):
                read_input_dataset(
                    tmp_path, manifest, ("spots", "fit"), input_mode="direct"
                )

    def test_excluded_rows_do_not_enter_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            for name in ("a.xy", "b.xy"):
                _write_xy(tmp_path / name)
            rows = [
                {"frame": 1, "scan": "a", "pressure_GPa": 1, "channel": "spots", "file_path": "a.xy", "excluded": 0},
                {"frame": 2, "scan": "a", "pressure_GPa": 2, "channel": "spots", "file_path": "b.xy", "excluded": 1},
            ]
            manifest = tmp_path / "manifest.csv"
            _write_manifest(manifest, rows)
            dataset = read_input_dataset(
                tmp_path, manifest, ("spots",), input_mode="direct"
            )
            self.assertEqual([frame.frame for frame in dataset.frames], [1])
            self.assertEqual(dataset.excluded_rows, 1)


if __name__ == "__main__":
    unittest.main()
