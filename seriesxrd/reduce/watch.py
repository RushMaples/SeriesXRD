"""Watch-folder (live / during-beamtime) reduction: ``seriesxrd-watch``.

Polls a dataset folder while frames are still being collected, integrates
each new frame as it settles, appends it to a growing "live" reduced HDF5,
and (optionally) re-runs the analysis pipeline so Review / Peak map / Pattern
map stay current during the experiment.

Design decisions (see docs/roadmap.md, "Live watch-folder mode"):

* **Batch-closed appends, not a held-open writer.** The live file uses
  resizable datasets, but it is opened only for the few milliseconds of each
  batch append and closed again — between batches it is a complete, ordinary
  reduced HDF5 any reader (the analysis worker, the GUI) can open safely.
  This deliberately trades the pipeline's write-a-tmp-and-replace atomicity
  for append speed: a hard kill during an append can corrupt at most the
  live file (never a finished archival one). The live file is a working
  view — when the run is over, a normal full reduction remains the archival
  path (it also gets you cakes and gallery thumbnails, which live mode
  skips for speed).
* **Arrival order is frame order.** Frames are appended as they settle, so
  ``frame_index`` is arrival order — during a beamtime that is collection
  order. HDF5 stack containers that GROW mid-run (Eiger writing into a
  master file) are supported: new stack indices are picked up per poll, and
  the newest index of a still-growing stack is held back one poll so a
  half-written compressed chunk is never read.
* **Settling.** A plain image file is processed only after its (size,
  mtime) is unchanged between two consecutive polls; a frame whose
  integration fails is retried on later polls a few times before being
  marked failed.
* **Analysis re-runs are the normal atomic pipeline.** Every
  ``--analyze-every`` batches the crash-isolated analysis worker re-runs the
  configured steps (default 1-2; ``--steps 123`` adds phase ID, ``--steps
  ''`` disables) against the live file, rebuilding the analysis HDF5 the
  standard atomic way. Live mode adds no new analysis code path.

Integration itself reuses the batch reducer's per-frame task function
(:func:`processing._integrate_one`) in-process; the pyFAI engine caches its
integrator after the first frame, and beamtime frame rates are far below
serial integration throughput. The ``integrate``/``pool_init``/``analyze``
seams exist so tests (and exotic deployments) can inject replacements.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Works both as a package module (python -m seriesxrd.reduce.watch) and as a
# directly-launched script (the GUI runs this file by path in a subprocess),
# mirroring reduce/worker.py.
if __package__ in (None, ""):
    _pkg_parent = str(Path(__file__).resolve().parents[2])
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from seriesxrd.core.config import (ensure_dir, make_stdio_robust, now_iso,
                                     now_timestamp, output_base, read_json,
                                     safe_stem, write_json)
    from seriesxrd.core.handoff import load_handoff
    from seriesxrd.core.io import (expand_frame_sources, frame_display_name,
                                 harvest_stack_metadata, is_h5_frame_spec,
                                 parse_h5_frame_spec)
    from seriesxrd.reduce import processing as _proc
else:
    from ..core.config import (ensure_dir, make_stdio_robust, now_iso,
                               now_timestamp, output_base, read_json,
                               safe_stem, write_json)
    from ..core.handoff import load_handoff
    from ..core.io import (expand_frame_sources, frame_display_name,
                           harvest_stack_metadata, is_h5_frame_spec,
                           parse_h5_frame_spec)
    from . import processing as _proc

_MAX_RETRIES = 3


class _LiveWriter:
    """Appendable reduced-HDF5 writer (resizable datasets, closed between
    batches). Mirrors the batch reducer's schema minus cakes/thumbnails."""

    def __init__(self, h5_path: Path, npt_1d: int, settings: Dict[str, Any],
                 poni_text: str, npt_suggested: "Optional[int]", npt_mode: str):
        import h5py  # type: ignore
        self.path = Path(h5_path)
        self.npt = int(npt_1d)
        self.has_robust = bool(settings.get("robust_1d"))
        self.has_sigmaclip = bool(settings.get("sigmaclip_1d"))
        self._radial_written = False
        str_kw = {}
        with h5py.File(str(self.path), "w") as h5:
            h5.attrs.update({
                "tool": "seriesxrd.reduce", "tool_version": _proc.VERSION,
                "seriesxrd_version": _proc.VERSION,
                "created_at": now_iso(), "unit": settings["unit"],
                "npt_1d": self.npt,
                "npt_1d_suggested": int(npt_suggested or 0),
                "npt_1d_mode": npt_mode,
                "robust_quant_halfwidth": float(
                    settings.get("robust_quant_halfwidth", 0.0)),
                "live_mode": True,
            })
            h5.attrs["poni_text"] = poni_text
            g_pat = h5.create_group("patterns")
            g_pat.create_dataset("radial", shape=(self.npt,), dtype="f8")

            def _pat(name):
                g_pat.create_dataset(name, shape=(0, self.npt), dtype="f4",
                                     maxshape=(None, self.npt),
                                     chunks=(1, self.npt), fillvalue=np.nan)
            _pat("intensity")
            if self.has_robust:
                _pat("intensity_robust")
            if self.has_sigmaclip:
                _pat("intensity_sigmaclip")
            g_fr = h5.create_group("frames")
            str_dt = h5py.string_dtype(encoding="utf-8")

            def _fr(name, dtype, fill=None):
                kw = {"fillvalue": fill} if fill is not None else {}
                g_fr.create_dataset(name, shape=(0,), dtype=dtype,
                                    maxshape=(None,), **kw)
            _fr("filename", str_dt)
            _fr("ok", "?")
            _fr("seconds", "f4")
            _fr("excluded", "?")
            _fr("frame_index", "i8")
            _fr("pressure", "f8", np.nan)
            _fr("temperature", "f8", np.nan)
            _fr("timestamp", str_dt)
            _fr("pos_x", "f8", np.nan)
            _fr("pos_y", "f8", np.nan)
        self.n = 0

    @classmethod
    def resume(cls, h5_path: "str | Path") -> "_LiveWriter":
        """Reopen an existing live file for appending. Shape/channel facts
        come from the FILE (they must match what append writes), not from the
        session config."""
        import h5py  # type: ignore
        self = object.__new__(cls)
        self.path = Path(h5_path)
        with h5py.File(str(self.path), "r") as h5:
            if not h5.attrs.get("live_mode"):
                raise ValueError(f"{self.path} is not a live-mode reduced file "
                                 "(only *_live.h5 outputs of seriesxrd-watch can "
                                 "be resumed).")
            pat = h5["patterns"]
            self.npt = int(pat["intensity"].shape[1])
            self.n = int(pat["intensity"].shape[0])
            self.has_robust = "intensity_robust" in pat
            self.has_sigmaclip = "intensity_sigmaclip" in pat
            radial = np.asarray(pat["radial"][:], dtype=float)
            self._radial_written = bool(np.any(np.isfinite(radial))
                                        and np.any(radial != 0))
        return self

    def stored_names(self) -> "List[str]":
        """Display names of already-appended frames (the resume seen-set)."""
        import h5py  # type: ignore
        with h5py.File(str(self.path), "r") as h5:
            return [x.decode("utf-8", "replace")
                    if isinstance(x, (bytes, bytearray)) else str(x)
                    for x in h5["frames/filename"][:]]

    def append_batch(self, rows: "List[Dict[str, Any]]") -> None:
        """Append integrated results (with display name + metadata already
        attached under 'display'/'timestamp'/'pos_x'/'pos_y'/'temperature')."""
        if not rows:
            return
        import h5py  # type: ignore
        with h5py.File(str(self.path), "r+") as h5:
            g_pat, g_fr = h5["patterns"], h5["frames"]
            n0, n1 = self.n, self.n + len(rows)
            for name in g_pat:
                if name != "radial":
                    g_pat[name].resize(n1, axis=0)
            for name in g_fr:
                g_fr[name].resize(n1, axis=0)
            for j, r in enumerate(rows):
                i = n0 + j
                g_fr["filename"][i] = r["display"]
                g_fr["ok"][i] = bool(r.get("ok"))
                g_fr["seconds"][i] = float(r.get("seconds", 0.0))
                g_fr["excluded"][i] = False
                g_fr["frame_index"][i] = i
                g_fr["timestamp"][i] = r.get("timestamp", "")
                for key in ("pos_x", "pos_y", "temperature"):
                    v = r.get(key)
                    if v is not None and np.isfinite(v):
                        g_fr[key][i] = float(v)
                if not r.get("ok"):
                    continue
                g_pat["intensity"][i] = r["intensity"]
                if not self._radial_written and "radial" in r:
                    g_pat["radial"][:] = r["radial"]
                    self._radial_written = True
                if self.has_robust and "intensity_robust" in r:
                    g_pat["intensity_robust"][i] = r["intensity_robust"]
                if self.has_sigmaclip and "intensity_sigmaclip" in r:
                    g_pat["intensity_sigmaclip"][i] = r["intensity_sigmaclip"]
            self.n = n1


def _default_analyze(workspace_root: "Optional[Path]", logs_root: Path,
                     steps: str) -> Callable[[str], None]:
    """Build the default analysis trigger: run the crash-isolated analysis
    worker over the live file with the workspace's analysis config (when one
    exists) as the base settings."""
    worker_script = Path(__file__).resolve().parents[1] / "analysis" / "worker.py"

    def _run(reduced_h5: str) -> None:
        base: Dict[str, Any] = {}
        if workspace_root:
            cfg_path = Path(workspace_root) / "analysis_session_config.json"
            if cfg_path.is_file():
                base = read_json(cfg_path)
        base.update({
            "reduced_h5_file": str(reduced_h5),
            "run_step1": True,
            "run_step2": "2" in steps,
            "run_step3": "3" in steps,
        })
        base.setdefault("analysis_h5_file", "")
        cfg_file = logs_root / f"watch_analysis_cfg_{now_timestamp()}.json"
        out_json = logs_root / f"watch_analysis_{now_timestamp()}.json"
        write_json(cfg_file, base)
        proc = subprocess.run(
            [sys.executable, str(worker_script), "--config", str(cfg_file),
             "--output-json", str(out_json)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = "\n".join((proc.stdout or "").splitlines()[-3:])
        if proc.returncode != 0:
            print(f"[WATCH] analysis run failed (rc={proc.returncode}):\n{tail}",
                  flush=True)
        else:
            print(f"[WATCH] analysis refreshed: {tail.splitlines()[-1] if tail else 'ok'}",
                  flush=True)
    return _run


class WatchSession:
    """One live-reduction session. Drive it with :meth:`cycle` (one poll) or
    :func:`run_watch` (the sleep loop around it)."""

    def __init__(self, config: Dict[str, Any], *,
                 steps: str = "12", analyze_every: int = 1,
                 out_path: "str | Path" = "", resume: bool = False,
                 integrate: "Optional[Callable]" = None,
                 pool_init: "Optional[Callable]" = None,
                 analyze: "Optional[Callable[[str], None]]" = None):
        self.config = dict(config)
        self.steps = str(steps or "")
        self.analyze_every = max(1, int(analyze_every))
        self._integrate = integrate
        self._pool_init = pool_init
        self._analyze = analyze
        self._initialized = False
        self.writer: "Optional[_LiveWriter]" = None
        # seen/failed/retries are keyed by DISPLAY name (what the live file
        # stores in /frames/filename) so a resumed session and the original
        # one agree on identity regardless of absolute vs relative paths.
        self.seen: "set[str]" = set()
        self.retries: Dict[str, int] = {}
        self.failed: Dict[str, str] = {}
        self._settle: Dict[str, Tuple[int, float]] = {}
        self._stack_counts: Dict[str, int] = {}
        self._batches_since_analysis = 0
        self.n_appended_total = 0
        self.cycles = 0
        self.dataset_root = Path(self.config.get("dataset_dir", "") or ".")

        self.handoff = load_handoff(self.config.get("handoff_file", ""))
        if self._integrate is None and not self.handoff.ok:
            raise ValueError("Invalid handoff: " + "; ".join(self.handoff.problems))

        session = safe_stem(self.config.get("session_name", "reduction"),
                            default="reduction")
        out_root = ensure_dir(
            Path(self.config.get("processed_root")
                 or output_base(self.config) / "data" / "processed")
            / f"reduction_{session}")
        self.h5_path = (Path(out_path).expanduser() if out_path
                        else out_root / f"reduced_{session}_{now_timestamp()}_live.h5")
        self.logs_root = ensure_dir(
            Path(self.config.get("logs_root", "")
                 or output_base(self.config) / "logs"))
        self.workspace_root = self.config.get("workspace_root") or None

        if resume:
            if not self.h5_path.is_file():
                raise FileNotFoundError(
                    f"resume: live file not found: {self.h5_path}")
            self.writer = _LiveWriter.resume(self.h5_path)
            self.seen.update(self.writer.stored_names())
            self.n_appended_total = 0     # counts THIS session's appends
            print(f"[WATCH] resuming {self.h5_path.name}: "
                  f"{self.writer.n} frame(s) already present", flush=True)
        elif out_path and self.h5_path.is_file():
            raise ValueError(
                f"{self.h5_path} already exists — pass resume/--resume to "
                f"continue it, or choose a different output path.")

    # -- source discovery -------------------------------------------------

    def _scan_sources(self) -> "List[str]":
        files = _proc.scan_dataset(
            self.config.get("dataset_dir", ""),
            self.config.get("file_patterns", _proc.DEFAULT_PATTERNS),
            bool(self.config.get("recursive", False)))
        # Never re-ingest our own live output if it lives near the data.
        files = [f for f in files if Path(f).resolve() != self.h5_path.resolve()]
        sources, _ = expand_frame_sources(
            files, str(self.config.get("h5_data_path", "") or ""))
        return sources

    def _ready(self, sources: "List[str]") -> "List[str]":
        """Filter to sources safe to process this poll (settled files; stack
        frames except a growing stack's newest index)."""
        # Per-container counts to detect growth.
        counts: Dict[str, int] = {}
        for s in sources:
            if is_h5_frame_spec(s):
                f, d, i = parse_h5_frame_spec(s)
                key = str(f)
                counts[key] = max(counts.get(key, 0), i + 1)
        growing = {k for k, n in counts.items()
                   if n > self._stack_counts.get(k, 0)}
        ready: "List[str]" = []
        for s in sources:
            disp = frame_display_name(s, self.dataset_root)
            if disp in self.seen or disp in self.failed:
                continue
            if is_h5_frame_spec(s):
                f, d, i = parse_h5_frame_spec(s)
                key = str(f)
                if key in growing and i == counts[key] - 1:
                    continue      # newest frame of a growing stack: next poll
                ready.append(s)
            else:
                try:
                    st = Path(s).stat()
                except OSError:
                    continue
                sig = (st.st_size, st.st_mtime)
                if self._settle.get(s) == sig:
                    ready.append(s)
                else:
                    self._settle[s] = sig   # (re)arm; process when unchanged
        self._stack_counts = counts
        return ready

    # -- lifecycle ----------------------------------------------------------

    def _initialize(self, first_source: str) -> None:
        npt_raw = self.config.get("npt_1d", "")
        cfg = dict(self.config)
        cfg["save_cakes"] = False          # live mode: no cakes/thumbnails
        cfg["make_thumbnails"] = False
        if self.writer is not None:
            # Resuming: shapes and channels are dictated by the existing file.
            npt, suggested, mode = self.writer.npt, None, "resume"
            cfg["robust_1d"] = self.writer.has_robust
            cfg["sigmaclip_1d"] = self.writer.has_sigmaclip
            poni_text = ""
        elif self._integrate is None:
            npt, suggested, mode = _proc._resolve_npt_1d(
                npt_raw, self.handoff.accepted_poni, first_source)
            poni_text = Path(self.handoff.accepted_poni).read_text(
                encoding="utf-8", errors="replace")
        else:  # injected integrator (tests): geometry-free
            npt = int(npt_raw or 0) or 100
            suggested, mode = None, "explicit"
            poni_text = ""
        self.settings = _proc.build_settings(cfg, npt)
        self.settings.pop("previews_dir", None)
        if self._pool_init is not None:
            self._pool_init(str(self.handoff.accepted_poni or ""),
                            str(self.handoff.accepted_mask_npz or ""),
                            self.settings)
        elif self._integrate is None:
            _proc._pool_init(str(self.handoff.accepted_poni),
                             str(self.handoff.accepted_mask_npz or ""),
                             self.settings)
        if self.writer is None:
            self.writer = _LiveWriter(self.h5_path, npt, self.settings,
                                      poni_text, suggested, mode)
        if self._analyze is None and self.steps:
            self._analyze = _default_analyze(self.workspace_root,
                                             self.logs_root, self.steps)
        print(f"[WATCH] live file: {self.h5_path} (npt_1d={npt}; cakes and "
              f"thumbnails are skipped in live mode — run a full reduction "
              f"afterwards for the archival file)", flush=True)
        self._initialized = True

    def cycle(self) -> int:
        """One poll: discover, integrate, append, maybe analyze. Returns the
        number of frames appended."""
        self.cycles += 1
        sources = self._scan_sources()
        ready = self._ready(sources)
        if not ready:
            return 0
        if not self._initialized:
            try:
                self._initialize(ready[0])
            except Exception as e:
                # e.g. the very first frame is unreadable — retry next poll
                # rather than killing a beamtime watcher.
                print(f"[WATCH] init failed ({e!r}) — retrying next poll.",
                      flush=True)
                return 0
        integrate = self._integrate or _proc._integrate_one

        # Per-frame metadata for the batch's stack frames (one container read).
        meta = harvest_stack_metadata(
            ready,
            timestamp_path=str(self.config.get("h5_timestamp_path", "") or ""),
            pos_x_path=str(self.config.get("h5_pos_x_path", "") or ""),
            pos_y_path=str(self.config.get("h5_pos_y_path", "") or ""),
            temperature_path=str(self.config.get("h5_temperature_path", "") or ""),
        ) if any(is_h5_frame_spec(s) for s in ready) else None

        rows: "List[Dict[str, Any]]" = []
        for k, src in enumerate(ready):
            disp = frame_display_name(src, self.dataset_root)
            r = integrate((self.writer.n + len(rows), src, False))
            if not r.get("ok"):
                self.retries[disp] = self.retries.get(disp, 0) + 1
                if self.retries[disp] >= _MAX_RETRIES:
                    self.failed[disp] = str(r.get("error", "integration failed"))
                    print(f"[WATCH] FAILED (gave up after {_MAX_RETRIES} tries) "
                          f"{disp}: {self.failed[disp]}", flush=True)
                continue
            r["display"] = disp
            if meta is not None:
                r["timestamp"] = meta["timestamp"][k]
                r["pos_x"] = meta["pos_x"][k]
                r["pos_y"] = meta["pos_y"][k]
                r["temperature"] = meta["temperature"][k]
            rows.append(r)
            self.seen.add(disp)
            self._settle.pop(src, None)
            self.retries.pop(disp, None)
        if rows:
            self.writer.append_batch(rows)
            self.n_appended_total += len(rows)
            print(f"[WATCH] +{len(rows)} frame(s) -> {self.writer.n} total",
                  flush=True)
            self._batches_since_analysis += 1
            if (self.steps and self._analyze is not None
                    and self._batches_since_analysis >= self.analyze_every):
                self._batches_since_analysis = 0
                self._analyze(str(self.h5_path))
        return len(rows)

    def finish(self) -> Dict[str, Any]:
        """Final analysis pass (if frames arrived since the last one) and a
        summary manifest."""
        if (self.steps and self._analyze is not None and self._initialized
                and self._batches_since_analysis > 0):
            self._analyze(str(self.h5_path))
            self._batches_since_analysis = 0
        man = {
            "h5_file": str(self.h5_path) if self._initialized else "",
            "n_frames": self.n_appended_total,
            "n_failed": len(self.failed),
            "failures": dict(list(self.failed.items())[:20]),
            "cycles": self.cycles,
        }
        print(f"[WATCH] done: {man['n_frames']} frame(s) in {man['cycles']} "
              f"poll(s), {man['n_failed']} failed"
              + (f" -> {man['h5_file']}" if man["h5_file"] else ""), flush=True)
        return man


def run_watch(config: Dict[str, Any], *, poll: float = 5.0, steps: str = "12",
              analyze_every: int = 1, idle_exit_min: float = 0.0,
              out_path: "str | Path" = "", resume: bool = False,
              max_cycles: "Optional[int]" = None,
              integrate: "Optional[Callable]" = None,
              pool_init: "Optional[Callable]" = None,
              analyze: "Optional[Callable[[str], None]]" = None
              ) -> Dict[str, Any]:
    """Run the watch loop until Ctrl-C/SIGTERM, ``idle_exit_min`` minutes
    without a new frame (0 = run forever), or ``max_cycles`` polls (tests)."""
    ws = WatchSession(config, steps=steps, analyze_every=analyze_every,
                      out_path=out_path, resume=resume, integrate=integrate,
                      pool_init=pool_init, analyze=analyze)
    last_new = time.time()
    try:
        while True:
            n = ws.cycle()
            if n:
                last_new = time.time()
            elif idle_exit_min > 0 and (time.time() - last_new) > idle_exit_min * 60:
                print(f"[WATCH] idle for {idle_exit_min:g} min — exiting.", flush=True)
                break
            if max_cycles is not None and ws.cycles >= max_cycles:
                break
            time.sleep(max(0.0, float(poll)))
    except KeyboardInterrupt:
        print("[WATCH] interrupted — finishing up.", flush=True)
    return ws.finish()


def main(argv: "list[str] | None" = None) -> int:
    """CLI: ``seriesxrd-watch --workspace DIR [options]``."""
    import argparse
    make_stdio_robust()
    p = argparse.ArgumentParser(
        prog="seriesxrd-watch",
        description="Live (during-beamtime) reduction: watch the dataset "
                    "folder, append new frames to a growing reduced HDF5, and "
                    "periodically re-run the analysis pipeline. The live file "
                    "is a working view — run a normal full reduction for the "
                    "archival file (cakes, thumbnails, atomic writes).")
    p.add_argument("--workspace", default="",
                   help="Workspace folder holding reduction_session_config.json "
                        "(and optionally analysis_session_config.json for the "
                        "analysis knobs).")
    p.add_argument("--config", default="",
                   help="Explicit reduction config JSON (overrides --workspace).")
    p.add_argument("--poll", type=float, default=5.0,
                   help="Seconds between folder polls. Default 5.")
    p.add_argument("--steps", default="12",
                   help="Analysis steps to re-run as frames arrive: '12' "
                        "(default: background + peaks), '123' (adds phase ID "
                        "using the workspace's configured candidates), or '' "
                        "to only reduce.")
    p.add_argument("--analyze-every", type=int, default=1,
                   help="Re-run the analysis every N batches of new frames. "
                        "Default 1 (every batch).")
    p.add_argument("--idle-exit", type=float, default=0.0,
                   help="Exit after this many minutes with no new frames. "
                        "Default 0 = run until Ctrl-C.")
    p.add_argument("--out", default="",
                   help="Live reduced HDF5 path (default: "
                        "<processed>/reduction_<session>/reduced_<session>_<ts>_live.h5). "
                        "Refuses an existing file — use --resume for that.")
    p.add_argument("--resume", default="",
                   help="Continue an interrupted watch into this existing "
                        "*_live.h5: already-appended frames are skipped "
                        "(matched by stored name) and new ones append. "
                        "Overrides --out.")
    args = p.parse_args(argv)

    # A supervisor's terminate() (the GUI's Stop button, a scheduler kill)
    # should finish the current batch and flush a final analysis pass, same
    # as Ctrl-C.
    import signal

    def _graceful(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _graceful)
    except (ValueError, OSError):   # non-main thread / exotic platform
        pass
    if args.config:
        cfg_path = Path(args.config).expanduser()
    elif args.workspace:
        cfg_path = Path(args.workspace).expanduser() / "reduction_session_config.json"
    else:
        cfg_path = Path.cwd() / "reduction_session_config.json"
    if not cfg_path.is_file():
        print(f"[ERROR] reduction config not found: {cfg_path} — run the "
              f"Reduction GUI once (or seriesxrd --workspace ...) to create it.",
              flush=True)
        return 1
    config = read_json(cfg_path)
    if args.workspace and not config.get("workspace_root"):
        config["workspace_root"] = str(Path(args.workspace).expanduser())
    try:
        run_watch(config, poll=args.poll, steps=args.steps,
                  analyze_every=args.analyze_every,
                  idle_exit_min=args.idle_exit,
                  out_path=(args.resume or args.out),
                  resume=bool(args.resume))
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
