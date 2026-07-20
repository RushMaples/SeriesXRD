"""Download and organize the pinned SeriesXRD Ti-6Al-4V demo subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "source_manifest.json"
DEFAULT_OUTPUT = HERE / "workspace"
CHUNK_SIZE = 1024 * 1024


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _digest(destination, "md5") == expected_md5:
        print(f"Using verified cache: {destination.name}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "SeriesXRD-demo/1"})
    print(f"Downloading {destination.name} ...")
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=CHUNK_SIZE)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    actual_md5 = _digest(partial, "md5")
    if actual_md5 != expected_md5:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: expected {expected_md5}, "
            f"received {actual_md5}"
        )
    partial.replace(destination)


def _copy_zip_member(archive: Path, member: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source_zip:
        try:
            source = source_zip.open(member)
        except KeyError as exc:
            raise RuntimeError(f"Required member not found in {archive.name}: {member}") from exc
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=CHUNK_SIZE)
    return _digest(destination, "sha256")


def fetch(output: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cache = output / ".cache"
    for filename, source in manifest["archives"].items():
        _download(source["url"], cache / filename, source["md5"])

    sample_dir = output / "data" / "raw" / "sample"
    calibration_dir = output / "data" / "raw" / "calibration"
    source_metadata_dir = output / "metadata" / "source"
    sample_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    source_metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_archive = cache / "rawdata_1.zip"
    frame_rows = []
    for frame in manifest["frames"]:
        run_number = frame["run_number"]
        member = f"rawdata_1/{run_number}-pilatus2M-files/00001.cbf"
        destination = sample_dir / frame["output_name"]
        sha256 = _copy_zip_member(raw_archive, member, destination)
        frame_rows.append(
            {
                "filename": destination.name,
                "run_number": run_number,
                "exposure_s": frame["exposure_s"],
                "source_archive": "rawdata_1.zip",
                "source_member": member,
                "sha256": sha256,
            }
        )

    calibration = manifest["calibration"]
    calibration_path = calibration_dir / calibration["output_name"]
    calibration_sha256 = _copy_zip_member(
        cache / "rawdata_calibration.zip",
        calibration["source_member"],
        calibration_path,
    )
    shutil.copy2(HERE / "geometry" / calibration["poni_name"], calibration_dir)
    shutil.copy2(cache / "sxrd_metadata_1.yaml", source_metadata_dir)
    shutil.copy2(MANIFEST_PATH, source_metadata_dir)
    shutil.copy2(HERE / "ATTRIBUTION.md", source_metadata_dir)

    with (output / "metadata" / "frames.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(frame_rows)

    checksums = [
        f"{row['sha256']}  data/raw/sample/{row['filename']}" for row in frame_rows
    ]
    checksums.extend(
        [
            f"{calibration_sha256}  data/raw/calibration/{calibration_path.name}",
            f"{_digest(calibration_dir / calibration['poni_name'], 'sha256')}  "
            f"data/raw/calibration/{calibration['poni_name']}",
        ]
    )
    (output / "metadata" / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )

    print("\nDemo workspace is ready:")
    print(f"  Workspace:        {output.resolve()}")
    print(f"  Calibration CBF:  {calibration_path.resolve()}")
    print(f"  Input PONI:       {(calibration_dir / calibration['poni_name']).resolve()}")
    print(f"  Sample folder:    {sample_dir.resolve()}")
    print(f"  Sample frames:    {len(frame_rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and organize the pinned Ti-6Al-4V SeriesXRD demo."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"workspace to populate (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    try:
        fetch(args.output.expanduser().resolve())
    except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
