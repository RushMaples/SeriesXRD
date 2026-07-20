"""I/O helpers for detector images and text data.

Frames come from two kinds of sources, both addressed by a plain string:

  * a single-image file (TIFF/EDF/CBF/...), read via fabio/tifffile/PIL;
  * one slice of an HDF5/NeXus stack (Eiger-style master files: one ``.h5``
    holding thousands of frames), addressed as
    ``"<file.h5>::<dataset_path>#<index>"``.

``expand_frame_sources`` turns a scanned file list into per-frame sources
(expanding stack containers), and ``read_detector_image`` accepts either
form — so the rest of the pipeline never needs to know which kind of file a
frame came from.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import csv
import numpy as np

# File extensions treated as potential HDF5 frame containers.
H5_CONTAINER_EXTS = {".h5", ".hdf5", ".nxs", ".nx"}

# Dataset paths probed (in order) when no explicit data path is given —
# the NeXus convention and common beamline layouts.
_NEXUS_DATA_PATHS = (
    "entry/data/data",
    "entry/instrument/detector/data",
    "entry_0000/measurement/data",
    "entry/data/data_000001",
)


def _import_hdf5_plugins() -> None:
    """Best-effort load of hdf5plugin: Eiger stacks are usually bitshuffle/
    LZ4-compressed, and reading them fails without the plugin filters. The
    import registers the filters as a side effect; absence is only a problem
    if a dataset actually uses them (h5py then raises on read)."""
    try:
        import hdf5plugin  # type: ignore  # noqa: F401
    except ImportError:
        pass


def is_h5_frame_spec(source) -> bool:
    """True when ``source`` addresses one slice of an HDF5 stack."""
    s = str(source)
    return "::" in s and "#" in s.rsplit("::", 1)[-1]


def parse_h5_frame_spec(source) -> "Tuple[Path, str, int]":
    """Split ``"<file>::<dataset>#<index>"`` into its parts."""
    s = str(source)
    file_part, rest = s.rsplit("::", 1)
    dset, idx = rest.rsplit("#", 1)
    return Path(file_part), dset, int(idx)


def h5_stack_info(path: str | Path, data_path: str = "") -> dict:
    """Inspect an HDF5 file as a frame container.

    Returns ``{ok, error, data_path, n_frames, frame_shape, ndim}``. With no
    explicit ``data_path`` the NeXus-convention locations are probed first,
    then the file is walked for the largest frame-stack-shaped dataset
    (ndim == 3 with image-sized trailing dims; a single 2D image dataset
    counts as a one-frame container). seriesxrd's own output files (reduced /
    analysis HDF5s carry ``tool``/``schema_version`` root attrs) are refused
    so a dataset folder that also holds results is safe to scan.
    """
    out = {"ok": False, "error": "", "data_path": "", "n_frames": 0,
           "frame_shape": None, "ndim": 0}
    p = Path(path).expanduser()
    try:
        import h5py  # type: ignore
    except ImportError:
        out["error"] = "h5py is not installed."
        return out
    _import_hdf5_plugins()

    def _usable(node) -> bool:
        return (isinstance(node, h5py.Dataset) and node.ndim in (2, 3)
                and node.shape[-1] >= 16 and node.shape[-2] >= 16)

    try:
        with h5py.File(str(p), "r") as h5:
            if not data_path and ("tool" in h5.attrs or "schema_version" in h5.attrs):
                out["error"] = ("a seriesxrd output file, not a detector stack "
                                "(pass an explicit HDF5 data path to override)")
                return out
            node = None
            if data_path:
                node = h5.get(data_path)
                if node is None or not isinstance(node, h5py.Dataset):
                    out["error"] = f"no dataset at {data_path!r}"
                    return out
                if node.ndim not in (2, 3):
                    out["error"] = (f"dataset {data_path!r} is {node.ndim}D — "
                                    "expected a 2D image or a 3D frame stack")
                    return out
            else:
                for cand in _NEXUS_DATA_PATHS:
                    n = h5.get(cand)
                    if n is not None and _usable(n):
                        node = n
                        break
                if node is None:
                    best = {"node": None, "frames": 0}

                    def _visit(name, obj):
                        if _usable(obj):
                            frames = obj.shape[0] if obj.ndim == 3 else 1
                            if frames > best["frames"]:
                                best.update(node=obj, frames=frames)
                    h5.visititems(_visit)
                    node = best["node"]
                if node is None:
                    out["error"] = ("no image-stack dataset found (pass an "
                                    "explicit HDF5 data path if the layout "
                                    "is unusual)")
                    return out
            out.update(ok=True, data_path=node.name.lstrip("/"),
                       ndim=int(node.ndim),
                       n_frames=int(node.shape[0]) if node.ndim == 3 else 1,
                       frame_shape=tuple(int(x) for x in node.shape[-2:]))
    except Exception as e:
        out["error"] = f"could not open as HDF5: {e!r}"
    return out


def expand_frame_sources(files: "Sequence[str | Path]", data_path: str = ""
                         ) -> "Tuple[List[str], int]":
    """Turn a scanned file list into per-frame sources.

    Plain image files pass through unchanged; HDF5 containers are expanded
    into one ``"<file>::<dataset>#<index>"`` spec per stored frame (indices
    zero-padded so lexical order is frame order). Unusable HDF5 files —
    seriesxrd's own outputs, files with no image dataset — are skipped with a
    log line rather than failing the run. Returns ``(sources, n_stacks)``.
    """
    sources: List[str] = []
    n_stacks = 0
    for f in files:
        p = Path(f)
        if p.suffix.lower() not in H5_CONTAINER_EXTS:
            sources.append(str(p))
            continue
        info = h5_stack_info(p, data_path)
        if not info["ok"]:
            print(f"[io] skipping {p.name}: {info['error']}", flush=True)
            continue
        n_stacks += 1
        dp = info["data_path"]
        for i in range(info["n_frames"]):
            sources.append(f"{p}::{dp}#{i:06d}")
    return sources, n_stacks


def frame_display_name(source, root: "Optional[str | Path]" = None) -> str:
    """Stored/display name for a frame source: plain files relative to
    ``root`` (basename if not under it); stack slices keep the
    ``file::dataset#index`` form with the file made relative the same way."""
    if is_h5_frame_spec(source):
        file_part, dset, idx = parse_h5_frame_spec(source)
        name = file_part.name
        if root is not None:
            try:
                name = str(file_part.relative_to(Path(root)))
            except ValueError:
                pass
        return f"{name}::{dset}#{idx:06d}"
    p = Path(str(source))
    if root is not None:
        try:
            return str(p.relative_to(Path(root)))
        except ValueError:
            return p.name
    return p.name


# Per-frame metadata locations probed inside a stack container when no
# explicit path is given. Beamlines are far less consistent here than with
# the image data itself, so explicit *_path overrides always win.
_NEXUS_TIMESTAMP_PATHS = (
    "entry/data/timestamp",
    "entry/data/time",
    "entry/instrument/NDAttributes/NDArrayTimeStamp",   # APS areaDetector
)
_NEXUS_POSITION_GROUPS = (
    "entry/sample", "entry/sample/positioners",
    "entry/instrument/positioners", "entry/data",
)
_NEXUS_POS_X_NAMES = ("pos_x", "sample_x", "sam_x", "samx", "x")
_NEXUS_POS_Y_NAMES = ("pos_y", "sample_y", "sam_y", "samy", "y")
_NEXUS_TEMPERATURE_PATHS = ("entry/sample/temperature",)


def _seconds_to_iso(values: np.ndarray) -> "List[str]":
    """Float seconds → ISO strings the analysis 'time' axis can parse. Only
    DIFFERENCES matter downstream (elapsed time), so any epoch base (UNIX,
    EPICS, run-relative) gives the same series axis."""
    from datetime import datetime, timedelta, timezone
    base = datetime(1970, 1, 1, tzinfo=timezone.utc)
    out = []
    for v in np.asarray(values, dtype=float).ravel():
        if not np.isfinite(v):
            out.append("")
            continue
        try:
            out.append((base + timedelta(seconds=float(v)))
                       .replace(tzinfo=None).isoformat())
        except (OverflowError, ValueError):
            out.append("")
    return out


def h5_stack_metadata(path: str | Path, n_frames: int, *,
                      timestamp_path: str = "", pos_x_path: str = "",
                      pos_y_path: str = "", temperature_path: str = "") -> dict:
    """Harvest per-frame metadata a NeXus/HDF5 stack container carries.

    Returns ``{timestamp, pos_x, pos_y, temperature}`` — ``timestamp`` a list
    of ISO strings (or None), the rest float arrays of length ``n_frames``
    (or None when not found). Explicit ``*_path`` arguments pin a dataset;
    otherwise common NeXus/areaDetector locations are probed. A per-frame
    array must match ``n_frames``; a scalar broadcasts (one value for the
    whole stack — normal for a still batch). Numeric timestamps are treated
    as seconds and converted (only elapsed time is used downstream).
    """
    out = {"timestamp": None, "pos_x": None, "pos_y": None, "temperature": None}
    p = Path(path).expanduser()
    try:
        import h5py  # type: ignore
    except ImportError:
        return out
    _import_hdf5_plugins()

    def _series(h5, dspath) -> "Optional[np.ndarray]":
        node = h5.get(dspath)
        if node is None or not isinstance(node, h5py.Dataset):
            return None
        try:
            raw = node[()]
        except Exception:
            return None
        arr = np.atleast_1d(np.asarray(raw))
        if arr.ndim != 1:
            return None
        if arr.size == 1:
            return np.repeat(arr, n_frames)
        if arr.size >= n_frames:
            return arr[:n_frames]
        return None

    def _float_series(h5, dspath) -> "Optional[np.ndarray]":
        arr = _series(h5, dspath)
        if arr is None or arr.dtype.kind not in "fiu":
            return None
        return np.asarray(arr, dtype="f8")

    try:
        with h5py.File(str(p), "r") as h5:
            # Timestamps: explicit path, else probe. String datasets pass
            # through; numeric ones convert as seconds.
            ts_paths = ([timestamp_path] if timestamp_path
                        else list(_NEXUS_TIMESTAMP_PATHS))
            for cand in ts_paths:
                arr = _series(h5, cand)
                if arr is None:
                    continue
                if arr.dtype.kind in "SUO":
                    out["timestamp"] = [
                        x.decode("utf-8", "replace")
                        if isinstance(x, (bytes, bytearray)) else str(x)
                        for x in arr]
                elif arr.dtype.kind in "fiu":
                    out["timestamp"] = _seconds_to_iso(arr)
                if out["timestamp"] is not None:
                    break

            def _positions(explicit, names):
                if explicit:
                    return _float_series(h5, explicit)
                for grp in _NEXUS_POSITION_GROUPS:
                    for nm in names:
                        arr = _float_series(h5, f"{grp}/{nm}")
                        if arr is not None:
                            return arr
                return None

            out["pos_x"] = _positions(pos_x_path, _NEXUS_POS_X_NAMES)
            out["pos_y"] = _positions(pos_y_path, _NEXUS_POS_Y_NAMES)

            t_paths = ([temperature_path] if temperature_path
                       else list(_NEXUS_TEMPERATURE_PATHS))
            for cand in t_paths:
                arr = _float_series(h5, cand)
                if arr is not None:
                    out["temperature"] = arr
                    break
    except Exception:
        return {"timestamp": None, "pos_x": None, "pos_y": None,
                "temperature": None}
    return out


def harvest_stack_metadata(sources: "Sequence[str]", *,
                           timestamp_path: str = "", pos_x_path: str = "",
                           pos_y_path: str = "", temperature_path: str = ""
                           ) -> dict:
    """Map container metadata onto an expanded frame-source list.

    ``sources`` is the :func:`expand_frame_sources` output (mixed plain files
    and stack specs). Container metadata is read once per container and
    aligned by each spec's frame index. Returns ``{timestamp (list[str]),
    pos_x, pos_y, temperature (float arrays)}`` of length ``len(sources)``
    — all-empty/NaN entries for plain files — plus ``n_frames_with_meta``.
    """
    n = len(sources)
    timestamp = [""] * n
    pos_x = np.full(n, np.nan, "f8")
    pos_y = np.full(n, np.nan, "f8")
    temperature = np.full(n, np.nan, "f8")
    # One pass to learn each container's frame count (max index asked for).
    parsed: "List[Optional[Tuple[Path, str, int]]]" = []
    counts: dict = {}
    for src in sources:
        if is_h5_frame_spec(src):
            file_part, dset, idx = parse_h5_frame_spec(src)
            parsed.append((file_part, dset, idx))
            key = str(file_part)
            counts[key] = max(counts.get(key, 0), idx + 1)
        else:
            parsed.append(None)
    per_container: dict = {}
    touched = 0
    for k, rec in enumerate(parsed):
        if rec is None:
            continue
        file_part, dset, idx = rec
        key = str(file_part)
        if key not in per_container:
            per_container[key] = h5_stack_metadata(
                file_part, counts[key],
                timestamp_path=timestamp_path, pos_x_path=pos_x_path,
                pos_y_path=pos_y_path, temperature_path=temperature_path)
        meta = per_container[key]
        got = False
        if meta["timestamp"] is not None and idx < len(meta["timestamp"]):
            timestamp[k] = meta["timestamp"][idx]
            got = True
        for name, arr in (("pos_x", pos_x), ("pos_y", pos_y),
                          ("temperature", temperature)):
            m = meta[name]
            if m is not None and idx < m.size:
                arr[k] = m[idx]
                got = True
        touched += int(got)
    return {"timestamp": timestamp, "pos_x": pos_x, "pos_y": pos_y,
            "temperature": temperature, "n_frames_with_meta": touched}


def _read_h5_frame(source) -> np.ndarray:
    import h5py  # type: ignore
    _import_hdf5_plugins()
    file_part, dset, idx = parse_h5_frame_spec(source)
    with h5py.File(str(file_part), "r") as h5:
        node = h5.get(dset)
        if node is None:
            raise FileNotFoundError(f"No dataset {dset!r} in {file_part}")
        if node.ndim == 3:
            if not 0 <= idx < node.shape[0]:
                raise IndexError(f"frame {idx} out of range 0..{node.shape[0]-1} "
                                 f"in {file_part}::{dset}")
            return np.asarray(node[idx])
        return np.asarray(node[()])


def _resolve_float_dtype(arr: np.ndarray) -> np.ndarray:
    """Recover float32 data from detector TIFFs that store IEEE floats but omit
    the TIFF SampleFormat tag (e.g. Pilatus CdTe and other corrected /
    flat-fielded frames). Generic readers then default to integer and hand back
    int32/uint32 whose bytes are actually floats.

    The bytes are reinterpreted as float32, and that reading is kept only when
    BOTH hold: the integer reading contains values too large to be photon counts
    (>2**28), and the reinterpreted floats land in a physical magnitude range.
    Genuine integer counts reinterpreted as float collapse to denormals
    (~1e-40), and real per-pixel counts never approach 2**28, so neither
    condition fires on true integer images — this never corrupts real count data.
    """
    if arr.dtype not in (np.dtype("int32"), np.dtype("uint32")):
        return arr
    as_float = np.ascontiguousarray(arr).view(np.float32)
    finite = as_float[np.isfinite(as_float)]
    nonzero = finite[finite != 0]
    big = 1 << 28
    int_implausible = bool((arr > big).any() or (arr < -big).any())
    float_physical = nonzero.size > 0 and 1e-4 < float(np.median(np.abs(nonzero))) < 1e9
    if int_implausible and float_physical:
        print("[io] integer-typed TIFF holds float32 data (no SampleFormat tag) "
              "— reinterpreting as float32", flush=True)
        return as_float
    return arr


def read_detector_image(path: str | Path, flip_up_down: bool = False) -> np.ndarray:
    if is_h5_frame_spec(path):
        arr = np.asarray(_resolve_float_dtype(_read_h5_frame(path)), dtype=np.float32)
        return np.flipud(arr) if flip_up_down else arr
    p = Path(path)
    errors = []
    raw = None
    try:
        import fabio  # type: ignore
        raw = np.asarray(fabio.open(str(p)).data)
    except Exception as e:
        errors.append(f"fabio: {e}")
    if raw is None:
        try:
            import tifffile  # type: ignore
            raw = np.asarray(tifffile.imread(str(p)))
        except Exception as e:
            errors.append(f"tifffile: {e}")
    if raw is None:
        try:
            from PIL import Image  # type: ignore
            raw = np.asarray(Image.open(p))
        except Exception as e:
            errors.append(f"PIL: {e}")
    if raw is None:
        raise RuntimeError("Could not read detector image. Tried fabio, tifffile, PIL. " + " | ".join(errors))
    
    arr = np.asarray(_resolve_float_dtype(raw), dtype=np.float32)
    if flip_up_down:
        arr = np.flipud(arr)

    return arr


def write_xy_csv(path: str | Path, x, y, x_name: str = "two_theta_deg", y_name: str = "intensity") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([x_name, y_name])
        for a, b in zip(np.asarray(x).ravel(), np.asarray(y).ravel()):
            w.writerow([float(a), float(b)])
    return p


def write_table_csv(path: str | Path, rows: Sequence[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()}) if rows else []
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return p
