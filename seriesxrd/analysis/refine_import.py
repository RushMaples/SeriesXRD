"""Import GSAS-II sequential-refinement results into an analysis HDF5.

The external interface is intentionally independent of any DAC collection
protocol::

    import_gsasii_results(analysis_h5, results, export_manifest=...)

``results`` is either a GSAS-II ``.gpx`` project (when GSASIIscriptable is
available in the current Python) or the portable
``seriesxrd_refinement.json`` written by the standalone helper included in a
SeriesXRD refinement export.  The export manifest maps GSAS histogram names
to one or more analysis-frame indices. Pressure, temperature, and time are
therefore ordinary frame metadata rather than importer requirements.

The importer writes the refined weight fractions, uncertainties, cells, and
fit-quality results together under ``/refinement`` atomically. The existing
``/fractions`` screening estimates are deliberately preserved. It never
substitutes a GSAS scale factor for ``WgtFrac``: GSAS-II computes weight
fractions using phase mass and the refined histogram/phase scale, and those
are the values that belong in a quantitative result.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ..core.config import VERSION
from ..core.provenance import manifest_provenance, write_step_provenance


RESULT_SCHEMA = "seriesxrd-gsasii-sequential"
RESULT_SCHEMA_VERSION = "1"
HDF5_SCHEMA_VERSION = "1"
_CELL_LABELS = ("a", "b", "c", "alpha", "beta", "gamma", "volume")


def _finite_float(value: Any) -> "float | None":
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_float(value: Any) -> "float | None":
    out = _finite_float(value)
    return out if out is not None else None


def _vector7(value: Any) -> "list[float] | None":
    try:
        vals = list(value)
    except TypeError:
        return None
    if len(vals) != 7:
        return None
    return [float(v) if _finite_float(v) is not None else float("nan")
            for v in vals]


def _quality_value(values: Mapping[str, Any], *names: str) -> "float | None":
    lookup = {str(k).casefold(): v for k, v in values.items()}
    for name in names:
        val = _finite_float(lookup.get(name.casefold()))
        if val is not None:
            return val
    return None


def _load_gsasii_module():
    try:
        from GSASII import GSASIIscriptable as g2  # type: ignore
        return g2
    except ImportError:
        try:
            import GSASIIscriptable as g2  # type: ignore
            return g2
        except ImportError as exc:
            raise RuntimeError(
                "GSASIIscriptable is not available in this Python. Run the "
                "export_seriesxrd_results.py helper included in the refinement "
                "bundle with GSAS-II's Python, then import the resulting "
                "seriesxrd_refinement.json instead."
            ) from exc


def _phase_result(seq, seq_data: Mapping[str, Any], phase, hist: str,
                  hist_id: int) -> "dict[str, Any] | None":
    phase_data = getattr(phase, "data", {})
    phase_id = int(phase_data.get("pId", getattr(phase, "id", -1)))
    var = f"{phase_id}:{hist_id}:WgtFrac"
    parm = seq_data.get("parmDict", {})
    value = _finite_float(parm.get(var))

    dep = seq_data.get("depParmDict", {}).get(var)
    if value is None and isinstance(dep, (list, tuple)) and dep:
        value = _finite_float(dep[0])
    dep_sig = seq_data.get("depSigDict", {}).get(var)
    if value is None and isinstance(dep_sig, (list, tuple)) and dep_sig:
        value = _finite_float(dep_sig[0])
    got = None
    if value is None:
        got = seq.get_Variable(hist, var)
        if isinstance(got, (list, tuple)) and got:
            value = _finite_float(got[0])
    if value is None:
        return None

    esd = None
    if isinstance(dep_sig, (list, tuple)):
        if len(dep_sig) > 1:
            esd = _finite_float(dep_sig[1])
    else:
        # Older projects/helpers may expose the dependent-variable esd alone.
        esd = _finite_float(dep_sig)
    if esd is None and isinstance(dep, (list, tuple)) and len(dep) > 1:
        esd = _finite_float(dep[1])
    if esd is None:
        if got is None:
            got = seq.get_Variable(hist, var)
        if isinstance(got, (list, tuple)) and len(got) > 1:
            esd = _finite_float(got[1])

    cell = cell_esd = None
    try:
        cell_raw, esd_raw, _unique = seq.get_cell_and_esd(phase, hist)
        cell = _vector7(cell_raw)
        cell_esd = _vector7(esd_raw)
    except Exception:
        pass
    return {
        "weight_fraction": value,
        "weight_fraction_esd": esd,
        "cell": cell,
        "cell_esd": cell_esd,
    }


def _read_gpx(path: Path, *, g2_module=None) -> Dict[str, Any]:
    """Read one GPX through GSAS-II's documented scripting interface."""
    g2 = g2_module or _load_gsasii_module()
    project = g2.G2Project(str(path))
    seq = project.seqref()
    if seq is None:
        raise ValueError(f"GSAS-II project has no sequential results: {path}")

    phases = list(project.phases())
    rows = []
    for hist in seq.histograms():
        seq_data, hist_data = seq.RefData(hist)
        try:
            hist_id = int(hist_data["data"][0]["hId"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot determine GSAS-II histogram id for {hist!r}") from exc
        rvals = seq_data.get("Rvals", {})
        converged = seq_data.get("converged")
        if converged is None:
            converged = rvals.get("converged")
        row: Dict[str, Any] = {
            "name": str(hist),
            "rwp": _quality_value(rvals, "Rwp", "wR"),
            "gof": _quality_value(rvals, "GOF"),
            "converged": bool(converged) if converged is not None else None,
            "phases": {},
        }
        for phase in phases:
            result = _phase_result(seq, seq_data, phase, str(hist), hist_id)
            if result is not None:
                name = str(getattr(phase, "name", "") or
                           getattr(phase, "data", {}).get("General", {}).get(
                               "Name", f"phase_{len(row['phases'])}"))
                row["phases"][name] = result
        rows.append(row)
    if not any(row["phases"] for row in rows):
        raise ValueError(
            "Sequential results contain no GSAS-II WgtFrac values. Refine "
            "phase fractions in a multiphase Rietveld model before importing."
        )
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "source_gpx": str(path),
        "histograms": rows,
    }


def _read_result_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read refinement result JSON {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Refinement JSON root must be an object.")
    if data.get("schema") != RESULT_SCHEMA:
        raise ValueError(
            f"Unsupported refinement JSON schema {data.get('schema')!r}; "
            f"expected {RESULT_SCHEMA!r}."
        )
    if str(data.get("schema_version")) != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported refinement JSON schema version "
            f"{data.get('schema_version')!r}."
        )
    if not isinstance(data.get("histograms"), list):
        raise ValueError("Refinement JSON must contain a histograms list.")
    return data


def read_sequential_results(path: "str | Path") -> Dict[str, Any]:
    """Normalize a SeriesXRD result JSON or a GSAS-II GPX project."""
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Refinement results not found: {src}")
    suffix = src.suffix.casefold()
    if suffix == ".json":
        return _read_result_json(src)
    if suffix == ".gpx":
        return _read_gpx(src)
    raise ValueError("Refinement results must be .gpx or seriesxrd_refinement.json.")


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _group_aliases(group: Mapping[str, Any]) -> set[str]:
    raw = [str(group.get("label", ""))]
    for key in ("pattern", "pattern_2theta"):
        if group.get(key):
            p = Path(str(group[key]))
            raw.extend((p.name, p.stem))
    for value in group.get("patterns", []) or []:
        p = Path(str(value))
        if p.stem.casefold().endswith("_q"):
            continue
        raw.extend((p.name, p.stem))
    return {key for key in (_name_key(v) for v in raw) if len(key) >= 4}


def _load_export_manifest(path: "str | Path | None", results: Path,
                          analysis: Path) -> "tuple[dict[str, Any] | None, Path | None]":
    if path is not None:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Export manifest not found: {candidate}")
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid export manifest JSON: {candidate}") from exc
        if not isinstance(data, Mapping):
            raise ValueError(f"Export manifest root must be an object: {candidate}")
        return dict(data), candidate
    for parent in (results.parent, analysis.parent):
        for name in ("refinement_manifest.json", "gsas_export_manifest.json"):
            candidate = parent / name
            if candidate.is_file():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid export manifest JSON: {candidate}") from exc
                if not isinstance(data, Mapping):
                    raise ValueError(
                        f"Export manifest root must be an object: {candidate}")
                return dict(data), candidate
    return None, None


def _frame_mapping(histograms: Sequence[Mapping[str, Any]], n_frames: int,
                   export_manifest: "Mapping[str, Any] | None") -> tuple[dict[int, list[int]], list[str]]:
    groups = list((export_manifest or {}).get("groups", []) or [])
    prepared = []
    for group in groups:
        frames_raw = group.get("frames", [])
        if not frames_raw and group.get("frame") is not None:
            frames_raw = [group["frame"]]
        frames = sorted({int(i) for i in frames_raw
                         if 0 <= int(i) < n_frames})
        if frames:
            prepared.append((_group_aliases(group), frames))

    mapping: dict[int, list[int]] = {}
    unmapped: list[str] = []
    for row_index, row in enumerate(histograms):
        name = str(row.get("name", ""))
        key = _name_key(name)
        candidates = []
        for aliases, frames in prepared:
            matches = [alias for alias in aliases if alias in key]
            if matches:
                candidates.append((max(len(alias) for alias in matches), frames))
        if candidates:
            best = max(length for length, _frames in candidates)
            choices = {tuple(frames) for length, frames in candidates if length == best}
            if len(choices) != 1:
                raise ValueError(f"Ambiguous export-manifest mapping for {name!r}.")
            mapping[row_index] = list(next(iter(choices)))
            continue
        match = re.search(r"frame[^0-9]*(\d+)", name, flags=re.IGNORECASE)
        if match and 0 <= int(match.group(1)) < n_frames:
            mapping[row_index] = [int(match.group(1))]
        else:
            unmapped.append(name)
    return mapping, unmapped


def _normalize_rows(data: Mapping[str, Any], phase_map: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = []
    for raw in data.get("histograms", []):
        if not isinstance(raw, Mapping):
            raise ValueError("Each refinement histogram must be an object.")
        phases: dict[str, dict[str, Any]] = {}
        raw_phases = raw.get("phases", {})
        if not isinstance(raw_phases, Mapping):
            raise ValueError(f"Histogram {raw.get('name')!r} has invalid phases data.")
        for external_name, result in raw_phases.items():
            if not isinstance(result, Mapping):
                continue
            name = str(phase_map.get(str(external_name), str(external_name))).strip()
            if not name:
                raise ValueError(f"Phase mapping for {external_name!r} is empty.")
            if name in phases:
                raise ValueError(
                    f"Several GSAS-II phases map to the same SeriesXRD name {name!r}."
                )
            fraction = _finite_float(result.get("weight_fraction"))
            if fraction is None:
                continue
            phases[name] = {
                "weight_fraction": fraction,
                "weight_fraction_esd": _finite_float(
                    result.get("weight_fraction_esd")),
                "cell": _vector7(result.get("cell")),
                "cell_esd": _vector7(result.get("cell_esd")),
            }
        rows.append({
            "name": str(raw.get("name", "")),
            "rwp": _finite_float(raw.get("rwp")),
            "gof": _finite_float(raw.get("gof")),
            "converged": (bool(raw["converged"])
                          if raw.get("converged") is not None else None),
            "phases": phases,
        })
    return rows


def _build_arrays(analysis_h5: Path, data: Mapping[str, Any], *,
                  export_manifest: "Mapping[str, Any] | None",
                  phase_map: Mapping[str, str]) -> Dict[str, Any]:
    import h5py  # type: ignore

    with h5py.File(str(analysis_h5), "r") as h5:
        if "frames" not in h5 or "filename" not in h5["frames"]:
            raise ValueError("Analysis file lacks /frames/filename.")
        n_frames = int(h5["frames/filename"].shape[0])
    rows = _normalize_rows(data, phase_map)
    mapping, unmapped = _frame_mapping(rows, n_frames, export_manifest)
    if not mapping:
        raise ValueError(
            "No GSAS-II histogram could be mapped to an analysis frame. Use "
            "the refinement_manifest.json/gsas_export_manifest.json written "
            "by SeriesXRD, or retain frame_#### in histogram names."
        )

    phase_names = sorted(
        {name for row in rows for name in row["phases"]}, key=str.casefold)
    if not phase_names:
        raise ValueError("Mapped sequential results contain no weight fractions.")
    phase_index = {name: j for j, name in enumerate(phase_names)}
    shape = (n_frames, len(phase_names))
    fractions = np.full(shape, np.nan, dtype=float)
    fraction_esd = np.full(shape, np.nan, dtype=float)
    cell = np.full(shape + (7,), np.nan, dtype=float)
    cell_esd = np.full(shape + (7,), np.nan, dtype=float)
    rwp = np.full(n_frames, np.nan, dtype=float)
    gof = np.full(n_frames, np.nan, dtype=float)
    converged = np.full(n_frames, -1, dtype=np.int8)
    source_histogram = np.full(n_frames, "", dtype=object)
    group_size = np.zeros(n_frames, dtype=np.int32)
    owner: dict[int, str] = {}

    for row_index, frames in mapping.items():
        row = rows[row_index]
        for frame in frames:
            if frame in owner:
                raise ValueError(
                    f"Analysis frame {frame} maps to both {owner[frame]!r} and "
                    f"{row['name']!r}."
                )
            owner[frame] = row["name"]
            source_histogram[frame] = row["name"]
            group_size[frame] = len(frames)
            if row["rwp"] is not None:
                rwp[frame] = row["rwp"]
            if row["gof"] is not None:
                gof[frame] = row["gof"]
            if row["converged"] is not None:
                converged[frame] = 1 if row["converged"] else 0
            for name, result in row["phases"].items():
                j = phase_index[name]
                fractions[frame, j] = result["weight_fraction"]
                if result["weight_fraction_esd"] is not None:
                    fraction_esd[frame, j] = result["weight_fraction_esd"]
                if result["cell"] is not None:
                    cell[frame, j] = result["cell"]
                if result["cell_esd"] is not None:
                    cell_esd[frame, j] = result["cell_esd"]

    warnings = []
    for frame in sorted(owner):
        values = fractions[frame]
        finite = values[np.isfinite(values)]
        if finite.size and (np.any(finite < -1e-6) or np.any(finite > 1.000001)):
            warnings.append(
                f"frame {frame}: a GSAS-II weight fraction lies outside [0, 1]"
            )
        if finite.size and abs(float(finite.sum()) - 1.0) > 0.02:
            warnings.append(
                f"frame {frame}: GSAS-II weight fractions sum to "
                f"{float(finite.sum()):.6g}, not 1"
            )
    return {
        "n_frames": n_frames,
        "phase_names": phase_names,
        "fractions": fractions,
        "fraction_esd": fraction_esd,
        "cell": cell,
        "cell_esd": cell_esd,
        "rwp": rwp,
        "gof": gof,
        "converged": converged,
        "source_histogram": source_histogram,
        "group_size": group_size,
        "mapped_frames": sorted(owner),
        "mapped_histograms": len(mapping),
        "unmapped_histograms": unmapped,
        "warnings": warnings,
    }


def _write_import(analysis_h5: Path, arrays: Mapping[str, Any], *,
                  results_path: Path, manifest_path: "Path | None") -> None:
    import h5py  # type: ignore

    tmp = analysis_h5.with_name(analysis_h5.name + ".tmp")
    shutil.copy2(analysis_h5, tmp)
    try:
        with h5py.File(str(tmp), "r+") as h5:
            if "refinement" in h5:
                del h5["refinement"]
            str_dtype = h5py.string_dtype(encoding="utf-8")
            names = np.asarray(arrays["phase_names"], dtype=object)

            rg = h5.create_group("refinement")
            rg.attrs.update({
                "schema_version": HDF5_SCHEMA_VERSION,
                "seriesxrd_version": VERSION,
                "engine": "GSAS-II",
                "fraction_method": "rietveld_gsasii",
                "source_results": str(results_path),
                "source_manifest": str(manifest_path or ""),
                "cell_columns": ",".join(_CELL_LABELS),
                "converged_encoding": "-1=unknown, 0=false, 1=true",
                "schema": (
                    "names (P,) phase names; fractions (N,P) GSAS-II WgtFrac; "
                    "fraction_esd (N,P) propagated standard uncertainties; "
                    "cell/cell_esd (N,P,7); fit quality is per frame. NaN "
                    "means the histogram/frame or phase was not imported."
                ),
                "group_mapping": (
                    "A histogram refined from several exported frames is "
                    "replicated to those frames; group_size records that "
                    "the values are one shared refinement result."
                ),
            })
            rg.create_dataset("names", data=names, dtype=str_dtype)
            rg.create_dataset("fractions", data=np.asarray(
                arrays["fractions"], "f8"))
            rg.create_dataset("fraction_esd", data=np.asarray(
                arrays["fraction_esd"], "f8"))
            rg.create_dataset("source_histogram", data=np.asarray(
                arrays["source_histogram"], dtype=object), dtype=str_dtype)
            for name, dtype in (("group_size", "i4"), ("converged", "i1"),
                                ("rwp", "f8"), ("gof", "f8"),
                                ("cell", "f8"), ("cell_esd", "f8")):
                rg.create_dataset(name, data=np.asarray(arrays[name], dtype=dtype))
            write_step_provenance(
                h5, "refinement_import", tool="seriesxrd.analysis.refine_import",
                schema_version=HDF5_SCHEMA_VERSION,
            )
        os.replace(tmp, analysis_h5)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def import_gsasii_results(
    analysis_h5: "str | Path",
    results: "str | Path",
    *,
    export_manifest: "str | Path | None" = None,
    phase_map: "Optional[Mapping[str, str]]" = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Import sequential GSAS-II weight fractions, cells, and fit quality.

    ``export_manifest`` is optional for histograms whose names retain a
    ``frame_####`` token. It is required for summed/grouped patterns because
    it is the non-guessing map from one GSAS histogram to several frames.
    ``phase_map`` optionally renames GSAS-II phases on import.
    """
    analysis = Path(analysis_h5).expanduser().resolve()
    result_path = Path(results).expanduser().resolve()
    if not analysis.is_file():
        raise FileNotFoundError(f"Analysis HDF5 not found: {analysis}")
    data = read_sequential_results(result_path)
    export_data, manifest_path = _load_export_manifest(
        export_manifest, result_path, analysis)
    arrays = _build_arrays(
        analysis, data, export_manifest=export_data,
        phase_map=dict(phase_map or {}),
    )
    if write:
        _write_import(
            analysis, arrays, results_path=result_path,
            manifest_path=manifest_path,
        )

    per_phase = {}
    for j, name in enumerate(arrays["phase_names"]):
        col = np.asarray(arrays["fractions"][:, j], dtype=float)
        finite = col[np.isfinite(col)]
        per_phase[name] = {
            "mean_fraction": float(np.mean(finite)) if finite.size else float("nan"),
            "max_fraction": float(np.max(finite)) if finite.size else float("nan"),
            "n_frames": int(finite.size),
        }
    return {
        **manifest_provenance("seriesxrd.analysis.refine_import",
                              HDF5_SCHEMA_VERSION),
        "source": str(result_path),
        "analysis_h5": str(analysis),
        "export_manifest": str(manifest_path or ""),
        "method": "rietveld_gsasii",
        "n_frames": int(arrays["n_frames"]),
        "n_frames_mapped": len(arrays["mapped_frames"]),
        "mapped_frames": arrays["mapped_frames"],
        "mapped_histograms": int(arrays["mapped_histograms"]),
        "unmapped_histograms": arrays["unmapped_histograms"],
        "phases": arrays["phase_names"],
        "per_phase": per_phase,
        "warnings": arrays["warnings"],
        "written": bool(write),
    }


GSASII_EXPORT_HELPER = r'''#!/usr/bin/env python
"""Export a GSAS-II sequential project to SeriesXRD's portable JSON."""
import json
import math
from pathlib import Path
import sys

try:
    from GSASII import GSASIIscriptable as G2sc
except ImportError:
    import GSASIIscriptable as G2sc


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def vector7(value):
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) != 7:
        return None
    return [finite(v) for v in values]


def quality(values, *names):
    lookup = {str(k).casefold(): v for k, v in values.items()}
    for name in names:
        value = finite(lookup.get(name.casefold()))
        if value is not None:
            return value
    return None


def extract(gpx_path):
    project = G2sc.G2Project(str(gpx_path))
    seq = project.seqref()
    if seq is None:
        raise ValueError("Project has no sequential refinement results")
    phases = list(project.phases())
    rows = []
    for hist in seq.histograms():
        seq_data, hist_data = seq.RefData(hist)
        hist_id = int(hist_data["data"][0]["hId"])
        rvals = seq_data.get("Rvals", {})
        converged = seq_data.get("converged")
        if converged is None:
            converged = rvals.get("converged")
        row = {
            "name": str(hist),
            "rwp": quality(rvals, "Rwp", "wR"),
            "gof": quality(rvals, "GOF"),
            "converged": bool(converged) if converged is not None else None,
            "phases": {},
        }
        for phase in phases:
            phase_data = getattr(phase, "data", {})
            phase_id = int(phase_data.get("pId", getattr(phase, "id", -1)))
            var = f"{phase_id}:{hist_id}:WgtFrac"
            parm = seq_data.get("parmDict", {})
            value = finite(parm.get(var))
            dep = seq_data.get("depParmDict", {}).get(var)
            if value is None and isinstance(dep, (list, tuple)) and dep:
                value = finite(dep[0])
            dep_sig = seq_data.get("depSigDict", {}).get(var)
            if value is None and isinstance(dep_sig, (list, tuple)) and dep_sig:
                value = finite(dep_sig[0])
            got = None
            if value is None:
                got = seq.get_Variable(hist, var)
                if isinstance(got, (list, tuple)) and got:
                    value = finite(got[0])
            if value is None:
                continue
            esd = None
            if isinstance(dep_sig, (list, tuple)):
                if len(dep_sig) > 1:
                    esd = finite(dep_sig[1])
            else:
                esd = finite(dep_sig)
            if esd is None and isinstance(dep, (list, tuple)) and len(dep) > 1:
                esd = finite(dep[1])
            if esd is None:
                if got is None:
                    got = seq.get_Variable(hist, var)
                if isinstance(got, (list, tuple)) and len(got) > 1:
                    esd = finite(got[1])
            cell = cell_esd = None
            try:
                c, ce, _unique = seq.get_cell_and_esd(phase, hist)
                cell, cell_esd = vector7(c), vector7(ce)
            except Exception:
                pass
            name = str(getattr(phase, "name", "") or
                       phase_data.get("General", {}).get("Name", "phase"))
            row["phases"][name] = {
                "weight_fraction": value,
                "weight_fraction_esd": esd,
                "cell": cell,
                "cell_esd": cell_esd,
            }
        rows.append(row)
    return {
        "schema": "seriesxrd-gsasii-sequential",
        "schema_version": "1",
        "source_gpx": str(Path(gpx_path).resolve()),
        "histograms": rows,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or len(argv) > 2:
        print("usage: export_seriesxrd_results.py PROJECT.gpx [OUTPUT.json]",
              file=sys.stderr)
        return 2
    source = Path(argv[0])
    output = Path(argv[1]) if len(argv) == 2 else Path("seriesxrd_refinement.json")
    output.write_text(json.dumps(extract(source), indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_gsasii_export_helper(path: "str | Path") -> Path:
    """Write the standalone GPX-to-SeriesXRD JSON helper."""
    target = Path(path)
    target.write_text(GSASII_EXPORT_HELPER, encoding="utf-8")
    return target


def _parse_phase_map(items: Sequence[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--phase-map needs GSAS_NAME=SERIESXRD_NAME, got {item!r}")
        external, internal = (part.strip() for part in item.split("=", 1))
        if not external or not internal:
            raise ValueError(f"Invalid --phase-map value {item!r}")
        out[external] = internal
    return out


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="seriesxrd-import-gsas",
        description=(
            "Import GSAS-II sequential WgtFrac, uncertainties, cell "
            "parameters, and fit quality into an analysis HDF5."
        ),
    )
    parser.add_argument("analysis", help="Analysis HDF5 to update atomically.")
    parser.add_argument("results", help="GSAS-II .gpx or seriesxrd_refinement.json.")
    parser.add_argument(
        "--manifest", default="",
        help="SeriesXRD refinement/GSAS export manifest (auto-found by default).",
    )
    parser.add_argument(
        "--phase-map", action="append", default=[], metavar="GSAS=SERIESXRD",
        help="Rename a GSAS-II phase on import; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and summarize without changing the HDF5.")
    args = parser.parse_args(argv)
    try:
        manifest = import_gsasii_results(
            args.analysis, args.results,
            export_manifest=(args.manifest or None),
            phase_map=_parse_phase_map(args.phase_map),
            write=not args.dry_run,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1
    action = "validated" if args.dry_run else "imported"
    print(
        f"[REFINEMENT] {action}: {manifest['mapped_histograms']} histogram(s) "
        f"-> {manifest['n_frames_mapped']} frame(s), "
        f"{len(manifest['phases'])} phase(s)", flush=True,
    )
    for warning in manifest["warnings"]:
        print(f"[WARN] {warning}", flush=True)
    for name in manifest["unmapped_histograms"]:
        print(f"[WARN] unmapped GSAS-II histogram: {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
