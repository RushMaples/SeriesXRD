"""Frame metadata seam: pressure (and temperature/timestamp) per frame.

The whole reason Step 3 exists is to answer "given this frame's conditions,
which phases explain the data?". The single most important per-frame condition
is **pressure**: in a diamond-anvil-cell series the lattice compresses frame to
frame, and if every candidate phase is free to pick its own best pressure, a
wrong phase can slide along P until a couple of lines coincide. Pinning pressure
turns phase matching from "find any pressure where this phase looks plausible"
into "at this frame's actual pressure, which phases fit?".

This module is the *source* of that pressure channel. It has one job — populate
``/frames/pressure`` (and friends) on the analysis HDF5 — from three inputs:

  1. **Filename / folder parsing** (:func:`extract_pressures`): the synchrotron
     naming convention usually encodes the load, e.g. ``UOTe-1GPa-001.tif`` →
     1.0 GPa, ``UOTe-1p5GPa`` → 1.5 GPa, ``3p9GPa`` → 3.9 GPa, or a parent folder
     ``"1 GPa/UOTe-..."`` → 1.0 GPa. Units GPa / MPa / kPa / Pa / Mbar / kbar /
     bar are recognised and normalised to GPa.
  2. **CSV import / override** (:func:`read_pressure_csv`, :func:`map_csv_to_frames`):
     a sheet keyed by ``frame`` *or* ``filename`` with a ``pressure_gpa`` column
     (and optional ``pressure_sigma_gpa`` / ``temperature_K``) — the escape hatch
     for membrane-gauge or ruby-fluorescence pressures the filename never carried.
  3. **Carry-over** from the reduced file (handled in ``background.py`` Step 1),
     which created the placeholder ``/frames/pressure`` datasets.

Downstream, ``identify.py`` reads ``/frames/pressure`` + ``/frames/pressure_sigma``
and scores each phase only within that window (see its ``pressure_by_frame`` arg).

**User-edit provenance** (``/frames/user_edited``, bool per frame): values a
human set deliberately — the GUI table editor or a CSV import — are marked so
that (a) a filename re-parse never overwrites them (a mistyped filename token
is exactly what the manual edit fixed) and (b) a Step-1 re-run carries them
forward into the rebuilt analysis file (``background.py`` matches by
filename). ``extract_to_analysis(replace=True)`` is the explicit reset: it
wipes the pressures AND the marks.

Pure stdlib + numpy + h5py (h5py lazy). HDF5 writes are atomic (``.tmp`` +
``os.replace``), matching the rest of the analysis stage.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Pressure-unit parsing
# ---------------------------------------------------------------------------

# Conversion to GPa, keyed by the lowercased unit token. ``mbar`` is read as
# *mega*bar (100 GPa): millibar is physically implausible in a DAC filename, so
# the DAC-friendly reading wins (documented; override via CSV if ever wrong).
_UNIT_TO_GPA: Dict[str, float] = {
    "gpa": 1.0,
    "mpa": 1e-3,
    "kpa": 1e-6,
    "pa": 1e-9,
    "mbar": 100.0,
    "kbar": 0.1,
    "bar": 1e-4,
}

# Number (allowing 'p' or ',' as the decimal point, the common "1p5" convention)
# immediately followed by a pressure unit, with no trailing letter so we don't
# clip a unit out of a longer word. Longest / most-specific units first.
_PRESSURE_RE = re.compile(
    r"(?P<num>\d+(?:[.,p]\d+)?)\s*[_\-]?\s*"
    r"(?P<unit>GPa|MPa|kPa|Mbar|kbar|bar|Pa)(?![A-Za-z])",
    re.IGNORECASE,
)


def parse_pressure(text: "Optional[str]") -> "Optional[float]":
    """Parse the first ``<number><unit>`` pressure token in ``text`` to GPa.

    Recognises ``1GPa``, ``1.5 GPa``, ``1p5GPa``, ``3p9GPa``, ``500MPa``,
    ``10kbar``, ``2Mbar`` … Returns the value in GPa, or ``None`` if no pressure
    token is present.
    """
    if not text:
        return None
    m = _PRESSURE_RE.search(str(text))
    if not m:
        return None
    num = m.group("num").replace("p", ".").replace(",", ".")
    try:
        val = float(num)
    except ValueError:
        return None
    return val * _UNIT_TO_GPA[m.group("unit").lower()]


def parse_pressure_from_path(name: "Optional[str]") -> "Optional[float]":
    """Pressure (GPa) for one frame name that may include folders.

    Prefers a token in the *basename* (closest to the frame), falling back to the
    parent-folder portion (e.g. ``"1 GPa/UOTe-001.tif"`` → 1.0). Accepts both
    ``/`` and ``\\`` separators.
    """
    if not name:
        return None
    s = str(name).replace("\\", "/")
    base = s.rsplit("/", 1)[-1]
    p = parse_pressure(base)
    if p is not None:
        return p
    # Search the folder portion, nearest folder first.
    parts = s.rsplit("/", 1)
    if len(parts) == 2 and parts[0]:
        for frag in reversed(parts[0].split("/")):
            p = parse_pressure(frag)
            if p is not None:
                return p
    return None


def extract_pressures(names: "Sequence[str]") -> np.ndarray:
    """Per-frame pressure (GPa) parsed from each frame name; NaN where absent."""
    out = np.full(len(names), np.nan, dtype="f8")
    for i, nm in enumerate(names):
        p = parse_pressure_from_path(nm)
        if p is not None:
            out[i] = p
    return out


def summarize_pressures(pressures: np.ndarray) -> Dict[str, Any]:
    """Quick stats for a parsed pressure array (for a preview / status line)."""
    p = np.asarray(pressures, float)
    finite = p[np.isfinite(p)]
    n = int(p.size)
    return {
        "n_frames": n,
        "n_parsed": int(finite.size),
        "frac_parsed": (float(finite.size) / n) if n else 0.0,
        "p_min": float(finite.min()) if finite.size else float("nan"),
        "p_max": float(finite.max()) if finite.size else float("nan"),
        "monotonic": bool(finite.size >= 2 and (
            np.all(np.diff(finite) >= -1e-9) or np.all(np.diff(finite) <= 1e-9))),
    }


# ---------------------------------------------------------------------------
# CSV import / override
# ---------------------------------------------------------------------------

# Accepted header spellings (lowercased, stripped) → canonical key.
_CSV_ALIASES = {
    "frame": "frame", "frame_index": "frame", "index": "frame", "idx": "frame",
    "i": "frame", "n": "frame",
    "filename": "filename", "file": "filename", "name": "filename",
    "fname": "filename", "path": "filename",
    "pressure_gpa": "pressure", "pressure": "pressure", "p_gpa": "pressure",
    "p": "pressure", "gpa": "pressure",
    "pressure_sigma_gpa": "pressure_sigma", "pressure_sigma": "pressure_sigma",
    "sigma_gpa": "pressure_sigma", "sigma": "pressure_sigma", "p_sigma": "pressure_sigma",
    "dp": "pressure_sigma", "p_err": "pressure_sigma",
    "temperature_k": "temperature", "temperature": "temperature",
    "temp_k": "temperature", "temp": "temperature", "t_k": "temperature", "t": "temperature",
}


def _to_float(s) -> "Optional[float]":
    try:
        v = float(str(s).strip())
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def read_pressure_csv(path: "str | Path") -> Dict[str, Any]:
    """Parse a pressure CSV into canonical per-row records.

    The header may use any of the aliases in :data:`_CSV_ALIASES`. Each row is
    keyed by ``frame`` (int) and/or ``filename`` (str); values are ``pressure``
    (GPa), ``pressure_sigma`` (GPa) and ``temperature`` (K). Returns
    ``{ok, error, rows, columns}`` — ``rows`` is a list of dicts. A pressure
    column is required.
    """
    p = Path(path).expanduser()
    out: Dict[str, Any] = {"ok": False, "error": "", "rows": [], "columns": []}
    if not p.is_file():
        out["error"] = f"CSV not found: {p}"
        return out
    try:
        with p.open("r", newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(fh, dialect)
            header = next(reader, None)
            if not header:
                out["error"] = "CSV is empty."
                return out
            colmap: Dict[int, str] = {}
            for j, h in enumerate(header):
                key = _CSV_ALIASES.get(str(h).strip().lower())
                if key:
                    colmap[j] = key
            cols = set(colmap.values())
            out["columns"] = sorted(cols)
            if "pressure" not in cols:
                out["error"] = ("CSV needs a pressure column (e.g. 'pressure_gpa'). "
                                f"Recognised columns: {sorted(cols) or 'none'}.")
                return out
            if "frame" not in cols and "filename" not in cols:
                out["error"] = "CSV needs a 'frame' or 'filename' column to key rows."
                return out
            rows: List[Dict[str, Any]] = []
            for raw in reader:
                if not any(str(c).strip() for c in raw):
                    continue
                rec: Dict[str, Any] = {}
                for j, key in colmap.items():
                    if j >= len(raw):
                        continue
                    val = raw[j]
                    if key == "frame":
                        f = _to_float(val)
                        if f is not None:
                            rec["frame"] = int(round(f))
                    elif key == "filename":
                        rec["filename"] = str(val).strip()
                    else:
                        f = _to_float(val)
                        if f is not None:
                            rec[key] = f
                if "pressure" in rec and ("frame" in rec or rec.get("filename")):
                    rows.append(rec)
            out["rows"] = rows
            out["ok"] = bool(rows)
            if not rows:
                out["error"] = "No usable rows (each needs a pressure and a frame/filename)."
    except Exception as e:  # pragma: no cover - defensive
        out["error"] = f"Failed to read CSV: {e!r}"
    return out


def _name_keys(name: str) -> List[str]:
    """Match keys for a frame name: full (slash-normalised), basename, stem."""
    s = str(name).replace("\\", "/")
    base = s.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    keys = {s, base, stem}
    return [k for k in keys if k]


def map_csv_to_frames(rows: "Sequence[Dict[str, Any]]", names: "Sequence[str]",
                      n: "Optional[int]" = None) -> Dict[str, np.ndarray]:
    """Project CSV rows onto per-frame arrays.

    Rows with a ``frame`` index map directly; rows with a ``filename`` map by
    basename / stem / full-path match against ``names``. Returns
    ``{pressure, pressure_sigma, temperature}`` arrays of length
    ``n`` (defaults to ``len(names)``), NaN where unmapped. ``frame`` keying
    takes precedence over ``filename`` when a row carries both.
    """
    n = int(n if n is not None else len(names))
    pressure = np.full(n, np.nan, "f8")
    sigma = np.full(n, np.nan, "f8")
    temperature = np.full(n, np.nan, "f8")

    # filename → frame index lookup (every alias key points to its frame).
    name_to_idx: Dict[str, int] = {}
    for i, nm in enumerate(names):
        for k in _name_keys(nm):
            name_to_idx.setdefault(k, i)

    def _assign(i: int, rec: Dict[str, Any]) -> None:
        if not (0 <= i < n):
            return
        pressure[i] = rec["pressure"]
        if "pressure_sigma" in rec:
            sigma[i] = rec["pressure_sigma"]
        if "temperature" in rec:
            temperature[i] = rec["temperature"]

    for rec in rows:
        if "frame" in rec:
            _assign(int(rec["frame"]), rec)
        elif rec.get("filename"):
            for k in _name_keys(rec["filename"]):
                if k in name_to_idx:
                    _assign(name_to_idx[k], rec)
                    break
    return {"pressure": pressure, "pressure_sigma": sigma, "temperature": temperature}


# ---------------------------------------------------------------------------
# Read / write the analysis HDF5 /frames metadata
# ---------------------------------------------------------------------------

_META_KEYS = ("pressure", "pressure_sigma", "temperature")


def read_frame_metadata(analysis_h5: "str | Path") -> Dict[str, Any]:
    """Read ``/frames`` metadata from an analysis (or reduced) HDF5.

    Returns ``{ok, error, n_frames, filename, pressure, pressure_sigma,
    temperature, timestamp, user_edited}``. Missing numeric channels come back
    as all-NaN arrays (length n_frames) so callers can use them
    unconditionally; ``user_edited`` is an all-False bool array when absent.
    """
    import h5py  # type: ignore

    p = Path(analysis_h5).expanduser()
    out: Dict[str, Any] = {"ok": False, "error": "", "n_frames": 0,
                           "filename": [], "pressure": None, "pressure_sigma": None,
                           "temperature": None, "timestamp": [], "user_edited": None}
    if not p.is_file():
        out["error"] = f"File not found: {p}"
        return out
    try:
        with h5py.File(str(p), "r") as h5:
            fr = h5.get("frames")
            names: List[str] = []
            if fr is not None and "filename" in fr:
                names = [x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray))
                         else str(x) for x in fr["filename"][:]]
            # Determine n from filenames, a background array, or a metadata channel.
            n = len(names)
            if not n:
                bg = h5.get("background")
                if bg is not None and "clean" in bg:
                    n = int(bg["clean"].shape[0])
                elif fr is not None and "pressure" in fr:
                    n = int(fr["pressure"].shape[0])
            out["filename"] = names
            out["n_frames"] = n
            for key in _META_KEYS:
                arr = (np.asarray(fr[key][:], dtype="f8")
                       if fr is not None and key in fr else np.full(n, np.nan, "f8"))
                out[key] = arr
            if fr is not None and "timestamp" in fr:
                out["timestamp"] = [x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray))
                                    else str(x) for x in fr["timestamp"][:]]
            out["user_edited"] = (np.asarray(fr["user_edited"][:], dtype=bool)
                                  if fr is not None and "user_edited" in fr
                                  else np.zeros(n, dtype=bool))
            out["ok"] = True
    except Exception as e:  # pragma: no cover - defensive
        out["error"] = f"Failed to read metadata: {e!r}"
    return out


def _write_frames_dataset(group, name: str, data, *, str_dtype=None) -> None:
    """Create-or-overwrite a 1D dataset under the /frames group."""
    if name in group:
        del group[name]
    if str_dtype is not None:
        group.create_dataset(name, data=np.asarray(data, dtype=object), dtype=str_dtype)
    else:
        group.create_dataset(name, data=np.asarray(data, dtype="f8"))


def apply_to_analysis(analysis_h5: "str | Path", *,
                      pressure: "Optional[Sequence[float]]" = None,
                      pressure_sigma: "Optional[Sequence[float]]" = None,
                      temperature: "Optional[Sequence[float]]" = None,
                      timestamp: "Optional[Sequence[str]]" = None,
                      user_frames: "Optional[Sequence[int]]" = None,
                      clear_user_marks: bool = False,
                      ) -> Dict[str, Any]:
    """Atomically write the supplied metadata channels into ``/frames``.

    Only the channels passed are touched (others on disk are preserved). Length
    is validated against the existing frame count. ``user_frames`` marks those
    frame indices in ``/frames/user_edited`` — deliberate human values that a
    filename re-parse must not overwrite and a Step-1 re-run must carry
    forward. ``clear_user_marks`` resets the whole mask first (the explicit
    "start over" path). Returns a small manifest with the per-channel parsed
    counts. Atomic via ``.tmp`` + ``os.replace``.
    """
    import h5py  # type: ignore

    src = Path(analysis_h5).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {src}")

    # Resolve frame count from the file.
    meta = read_frame_metadata(src)
    n = int(meta["n_frames"])
    if n <= 0:
        raise ValueError("Could not determine the frame count of the analysis file.")

    def _check(arr, label):
        if arr is None:
            return None
        a = np.asarray(arr, dtype="f8")
        if a.size != n:
            raise ValueError(f"{label} has {a.size} values but the file has {n} frames.")
        return a

    pres = _check(pressure, "pressure")
    psig = _check(pressure_sigma, "pressure_sigma")
    temp = _check(temperature, "temperature")
    ts = None
    if timestamp is not None:
        ts = [str(x) for x in timestamp]
        if len(ts) != n:
            raise ValueError(f"timestamp has {len(ts)} values but the file has {n} frames.")

    marks = None
    if user_frames is not None:
        marks = [int(i) for i in user_frames]
        bad = [i for i in marks if i < 0 or i >= n]
        if bad:
            raise ValueError(f"user_frames out of range 0..{n - 1}: {bad}")

    tmp = src.with_name(src.name + ".tmp")
    import shutil
    shutil.copy2(src, tmp)
    try:
        with h5py.File(str(tmp), "r+") as o:
            gf = o.require_group("frames")
            if pres is not None:
                _write_frames_dataset(gf, "pressure", pres)
            if psig is not None:
                _write_frames_dataset(gf, "pressure_sigma", psig)
            if temp is not None:
                _write_frames_dataset(gf, "temperature", temp)
            if ts is not None:
                _write_frames_dataset(gf, "timestamp", ts,
                                      str_dtype=h5py.string_dtype(encoding="utf-8"))
            if clear_user_marks or marks is not None:
                mask = (np.asarray(gf["user_edited"][:], dtype=bool)
                        if "user_edited" in gf and not clear_user_marks
                        else np.zeros(n, dtype=bool))
                if mask.size != n:
                    mask = np.zeros(n, dtype=bool)
                if marks is not None:
                    mask[marks] = True
                if "user_edited" in gf:
                    del gf["user_edited"]
                gf.create_dataset("user_edited", data=mask)
        os.replace(tmp, src)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    def _count(a):
        return int(np.sum(np.isfinite(a))) if a is not None else None

    return {"out_h5": str(src), "n_frames": n,
            "n_pressure": _count(pres), "n_pressure_sigma": _count(psig),
            "n_temperature": _count(temp)}


def _merge(new: np.ndarray, existing: "Optional[np.ndarray]") -> np.ndarray:
    """Overwrite only where ``new`` is finite; keep ``existing`` elsewhere."""
    new = np.asarray(new, dtype="f8")
    if existing is None:
        return new
    existing = np.asarray(existing, dtype="f8")
    return np.where(np.isfinite(new), new, existing)


def extract_to_analysis(analysis_h5: "str | Path", *, replace: bool = False
                        ) -> Dict[str, Any]:
    """Convenience: parse pressures from the file's own ``/frames/filename`` and
    write them to ``/frames/pressure``.

    ``replace=False`` (default) **merges**: only frames whose filename actually
    carries a pressure are overwritten, so a value already imported for a frame
    whose filename has no pressure token is preserved — and frames marked
    ``user_edited`` are never overwritten (a mistyped filename token is exactly
    what the manual edit fixed). ``replace=True`` wipes the whole channel AND
    the user-edit marks (frames without a parsed pressure become NaN). Returns
    the apply manifest plus a ``summary`` of the resulting pressures.
    """
    meta = read_frame_metadata(analysis_h5)
    if not meta["ok"]:
        raise ValueError(meta["error"] or "Could not read frame metadata.")
    if not meta["filename"]:
        raise ValueError("Analysis file has no /frames/filename to parse pressures from.")
    parsed = extract_pressures(meta["filename"])
    if replace:
        pressures = parsed
        man = apply_to_analysis(analysis_h5, pressure=pressures,
                                clear_user_marks=True)
    else:
        locked = meta.get("user_edited")
        if locked is not None and locked.size == parsed.size and locked.any():
            parsed = np.where(locked, np.nan, parsed)   # keep the human's values
        pressures = _merge(parsed, meta["pressure"])
        man = apply_to_analysis(analysis_h5, pressure=pressures)
    man["summary"] = summarize_pressures(pressures)
    man["n_parsed_from_names"] = int(np.sum(np.isfinite(parsed)))
    return man


def import_csv_to_analysis(analysis_h5: "str | Path", csv_path: "str | Path", *,
                           replace: bool = False) -> Dict[str, Any]:
    """Convenience: read a pressure CSV and write its channels onto the file's
    frames (matched by frame index or filename).

    ``replace=False`` (default) **merges**: only frames the CSV actually provides
    are overwritten — a partial correction sheet for a few frames will not erase
    the pressures of every other frame. ``replace=True`` writes the mapped array
    verbatim (frames absent from the CSV become NaN). Either way the mapped
    frames are marked ``user_edited``: a CSV is deliberate human input, so a
    later filename re-parse must not overwrite it (an explicit CSV import DOES
    override earlier manual edits — it is the same kind of input, newer).
    Returns the apply manifest plus the CSV parse result under ``csv``.
    """
    meta = read_frame_metadata(analysis_h5)
    if not meta["ok"]:
        raise ValueError(meta["error"] or "Could not read frame metadata.")
    parsed = read_pressure_csv(csv_path)
    if not parsed["ok"]:
        raise ValueError(parsed["error"] or "Could not read CSV.")
    mapped = map_csv_to_frames(parsed["rows"], meta["filename"], meta["n_frames"])
    pressure = mapped["pressure"]
    # Only write sigma/temperature if the CSV actually carried them.
    psig = mapped["pressure_sigma"] if np.any(np.isfinite(mapped["pressure_sigma"])) else None
    temp = mapped["temperature"] if np.any(np.isfinite(mapped["temperature"])) else None
    mapped_idx = np.nonzero(np.isfinite(mapped["pressure"]))[0]
    if not replace:
        pressure = _merge(pressure, meta["pressure"])
        if psig is not None:
            psig = _merge(psig, meta["pressure_sigma"])
        if temp is not None:
            temp = _merge(temp, meta["temperature"])
    man = apply_to_analysis(analysis_h5, pressure=pressure,
                            pressure_sigma=psig, temperature=temp,
                            user_frames=mapped_idx.tolist())
    man["csv"] = {"columns": parsed["columns"], "n_rows": len(parsed["rows"])}
    man["n_mapped"] = int(np.sum(np.isfinite(mapped["pressure"])))
    man["summary"] = summarize_pressures(pressure)
    return man
