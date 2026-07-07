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
    counts as a one-frame container). bulkxrd's own output files (reduced /
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
                out["error"] = ("a bulkxrd output file, not a detector stack "
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
    bulkxrd's own outputs, files with no image dataset — are skipped with a
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
