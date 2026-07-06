"""Training-only CIF corpus tooling (``bulkxrd-corpus``).

The learned scorer's generalisation comes from pattern DIVERSITY, and the
diversity lever is a large training-only CIF corpus (see
``docs/ml-training.md``). This module makes building one a two-command
job:

    bulkxrd-corpus fetch  ids.txt  ./training_cifs      # COD IDs, one per line
    bulkxrd-corpus screen ./training_cifs               # parse/dedupe/size-screen

``screen`` is the important half: a raw dump contains unparseable files,
near-duplicates (same structure deposited many times), and giant cells that
would dominate simulation time. It writes a ``corpus_manifest.json`` and moves
rejects into ``rejected/`` so ``bulkxrd-ml-train --cif-dir`` sees only usable,
distinct structures. Screening needs pymatgen (the same dependency training
needs anyway); the parser is injectable for tests.
"""
from __future__ import annotations

import json
import math
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

COD_BASE = "https://www.crystallography.net/cod"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_cod_cifs(out_dir: "str | Path", cod_ids: "Sequence[str | int]", *,
                   base_url: str = COD_BASE, timeout: float = 30.0,
                   log=print) -> Dict[str, Any]:
    """Download COD entries by ID (``<base_url>/<id>.cif``) into ``out_dir``.

    Existing files are skipped (re-runnable); failures are collected, not
    fatal. For bulk (>10⁴) corpora prefer COD's rsync mirror — this fetcher is
    for curated ID lists. Returns ``{n_requested, n_fetched, n_skipped,
    failures}``.
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    fetched = skipped = 0
    failures: List[Dict[str, str]] = []
    for raw in cod_ids:
        cid = str(raw).strip()
        if not cid:
            continue
        dst = out / f"{cid}.cif"
        if dst.is_file() and dst.stat().st_size > 0:
            skipped += 1
            continue
        url = f"{base_url.rstrip('/')}/{cid}.cif"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            if not data:
                raise ValueError("empty response")
            tmp = dst.with_name(dst.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dst)
            fetched += 1
        except Exception as e:
            failures.append({"id": cid, "error": repr(e)})
    log(f"[CORPUS] fetched {fetched}, skipped {skipped} existing, "
        f"{len(failures)} failed -> {out}")
    return {"n_requested": len(list(cod_ids)), "n_fetched": fetched,
            "n_skipped": skipped, "failures": failures}


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

def _default_parser(cif_path: Path) -> Dict[str, Any]:
    """pymatgen-backed summary used for screening/dedupe (injectable in tests)."""
    from .phases import pymatgen_available
    if not pymatgen_available():
        raise RuntimeError(
            "Screening needs pymatgen (pip install bulkxrd[phases]) — the same "
            "dependency bulkxrd-ml-train needs anyway.")
    from pymatgen.core import Structure  # type: ignore
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore
    s = Structure.from_file(str(cif_path))
    try:
        sg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
    except Exception:
        sg = 0
    return {"formula": s.composition.reduced_formula,
            "spacegroup": int(sg),
            "n_sites": len(s),
            "volume": float(s.volume)}


def screen_cifs(cif_dir: "str | Path", *, max_sites: int = 200,
                dedupe: bool = True, move_rejects: bool = True,
                parser: "Optional[Callable[[Path], Dict[str, Any]]]" = None,
                log=print) -> Dict[str, Any]:
    """Parse/size-screen/dedupe every ``*.cif`` under ``cif_dir``.

    * parse failure → reject (an unreadable CIF would be silently skipped at
      training startup — better to know now);
    * ``n_sites > max_sites`` → reject (huge cells dominate reflection
      simulation time for marginal diversity gain);
    * duplicate (same reduced formula + space group + volume within 0.5%) →
      reject when ``dedupe`` (repeat depositions add no diversity).

    Rejects move to ``cif_dir/rejected/<reason>/`` (or stay put with
    ``move_rejects=False``); a ``corpus_manifest.json`` records everything.
    """
    root = Path(cif_dir).expanduser().resolve()
    files = sorted(f for f in root.rglob("*.cif")
                   if "rejected" not in f.parts)
    if not files:
        raise ValueError(f"No .cif files under {root}")
    parse = parser or _default_parser

    kept: List[Dict[str, Any]] = []
    rejects: Dict[str, List[str]] = {"parse_failed": [], "too_big": [],
                                     "duplicate": []}
    seen: Dict[tuple, List[float]] = {}
    for f in files:
        try:
            info = parse(f)
        except RuntimeError:
            raise                              # missing pymatgen: instructive, fatal
        except Exception:
            rejects["parse_failed"].append(f.name)
            continue
        if int(info.get("n_sites", 0)) > int(max_sites):
            rejects["too_big"].append(f.name)
            continue
        if dedupe:
            # Same reduced formula + space group + cell volume within 0.5%
            # relative (an explicit pairwise check — volume BINNING would split
            # near-identical volumes across a bin edge).
            vol = float(info.get("volume", 0.0))
            key = (str(info.get("formula")), int(info.get("spacegroup", 0)))
            prior = seen.setdefault(key, [])
            if any(abs(vol - v) <= 0.005 * max(vol, v, 1e-9) for v in prior):
                rejects["duplicate"].append(f.name)
                continue
            prior.append(vol)
        kept.append({"file": f.name, **info})

    if move_rejects:
        for reason, names in rejects.items():
            if not names:
                continue
            d = root / "rejected" / reason
            d.mkdir(parents=True, exist_ok=True)
            for nm in names:
                src = root / nm
                if src.is_file():
                    shutil.move(str(src), str(d / nm))

    manifest = {"n_total": len(files), "n_kept": len(kept),
                "n_parse_failed": len(rejects["parse_failed"]),
                "n_too_big": len(rejects["too_big"]),
                "n_duplicates": len(rejects["duplicate"]),
                "max_sites": int(max_sites), "kept": kept}
    (root / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"[CORPUS] kept {len(kept)}/{len(files)} "
        f"(parse_failed={len(rejects['parse_failed'])}, "
        f"too_big={len(rejects['too_big'])}, "
        f"dupes={len(rejects['duplicate'])}) -> {root / 'corpus_manifest.json'}")
    return manifest


# ---------------------------------------------------------------------------
# CLI  (bulkxrd-corpus)
# ---------------------------------------------------------------------------

def main(argv: "Optional[List[str]]" = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="bulkxrd-corpus",
        description="Build/screen a training-only CIF corpus for bulkxrd-ml-train.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fetch", help="Download COD entries by ID.")
    pf.add_argument("ids", help="Text file of COD IDs, one per line.")
    pf.add_argument("out_dir", help="Destination folder for the CIFs.")
    pf.add_argument("--base-url", default=COD_BASE)
    ps = sub.add_parser("screen", help="Parse/dedupe/size-screen a CIF folder.")
    ps.add_argument("cif_dir")
    ps.add_argument("--max-sites", type=int, default=200)
    ps.add_argument("--no-dedupe", action="store_true")
    ps.add_argument("--keep-rejects-in-place", action="store_true")
    args = p.parse_args(argv)
    try:
        if args.cmd == "fetch":
            ids = [s.strip() for s in
                   Path(args.ids).read_text(encoding="utf-8").splitlines()
                   if s.strip() and not s.startswith("#")]
            fetch_cod_cifs(args.out_dir, ids, base_url=args.base_url)
        else:
            screen_cifs(args.cif_dir, max_sites=args.max_sites,
                        dedupe=not args.no_dedupe,
                        move_rejects=not args.keep_rejects_in_place)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
