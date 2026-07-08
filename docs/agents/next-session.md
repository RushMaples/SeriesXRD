# Next-session worklist (agent handoff)

Prioritized fixes and features for the next AI coding session (any frontier
model), written 2026-07-07 at the end of the `claude/ml-training-readiness-
review-136udm` series (PR #30). Read `CLAUDE.md` first (architecture, schemas,
decisions not to relitigate), then this file. The user-facing feature roadmap
is `docs/roadmap.md`; this file is the actionable slice with code pointers.

## Verification gates (run before AND after any change)

```bash
for t in tests/test_*.py tests/smoke_test.py; do python "$t" || break; done
# GUI changes: also build each GUI under Xvfb and drive the changed paths
# (this session's smoke pattern: construct the App with a temp config,
# root.update() loops, assert widget/status state — see the git history of
# commits a4da6d2 / 1f54825 for what such tests must cover).
# ML/identification changes: the pinned open-data benchmark must not regress:
bash examples/fetch_benchmark_example.sh ./benchdata   # cosine hit@1 = 1.000
```

Note: `tests/test_worker_script_bootstrap.py` silently no-ops under plain
`python` (pytest `monkeypatch` fixture) — run it with pytest if touching
worker bootstrap code.

## First: land PR #30

The branch is ~20 commits of reviewed, green work. Get it merged (or split if
the maintainer prefers) before stacking more. After merge, restart the branch
from `main` per the repo's operational rules.

## High-value fixes (small, do early)

1. **CSV `filename` matching vs stack specs.** `frame_metadata._name_keys`
   derives basename/stem keys designed for plain files; for stack frames the
   stored name is `run.h5::entry/data/data#000123`, so only an exact-string
   CSV match works. Add spec-aware keys (`run.h5#000123`, index-only within
   one container) + tests in `tests/test_frame_metadata.py`.
2. **Live watch analysis cost grows linearly.** Each `--analyze-every` cycle
   re-runs Steps 1-2 over the WHOLE live file (atomic rebuild). Fine to ~1k
   frames; beyond that either thin the cadence automatically (batches since
   last run × file size heuristic) or implement incremental Step-1/2 (append
   to /background & /peaks for new frames only — breaks the atomicity
   convention, so gate it behind the live file's `live_mode` attr only).
3. **Pytest/CI migration** (README chore): the 25 test modules are
   main()-runnable; add a thin pytest collector + GitHub Actions workflow
   (no display: skip Xvfb smokes or apt-install xvfb + python3-tk).
   LICENSE and CITATION.cff need the user's input — ask, don't guess.

## User-blocked items (ask, don't do)

- UOTe: re-refine geometry on a sample-position ring and re-reduce (waviness
  implied a ~0.24-0.48 mm transverse offset); the `50p7GPa` filename; real
  UOTe structures into the library once the single-crystal solve exists;
  re-run the benchmark gate on the re-reduced data.
- RIS pathfinder training run (docs/ml-training.md §5) — environment
  validation, not science; full-scale training only when the open-set need
  appears (see "Training strategy", §10).

## Features (in rough priority order — see docs/roadmap.md "Planned")

1. **Rietveld import-back bridge** (roadmap #1). After a user refines the
   `bulkxrd-export-refinement` bundle in GSAS-II, import the refined phase
   scale/weight fractions back into `/fractions` (method="rietveld").
   Seam: `analysis/fractions.py` already versions `method` in the attrs;
   parse GSAS-II's `.lst`/project export — start with the `.lst` text
   (stable format) and keep GSASIIscriptable optional.
2. **Open-set COD search for unknowns** (roadmap #2). Step 3c writes
   per-cluster d-fingerprints under `/unknowns/fingerprint`. Search them
   against a corpus: simulate reflections once per corpus CIF
   (`analysis/corpus.py` + `identify.phase_reflections`), cache to an .npz
   (d-grid fingerprints), coarse-match by d-lines, re-rank survivors with the
   `ml_scorer` seam. New module `analysis/unknown_search.py` + CLI; the cache
   build belongs beside `bulkxrd-corpus screen`.
3. **GUI auto-refresh during live watch.** The analysis GUI already reloads
   tabs after a worker run; during a watch, poll the analysis file's mtime
   (main-thread `after` loop) and re-run the staggered loaders. Small, high
   perceived value at a beamline.
4. **Multi-detector sessions** (roadmap #3) and **calibrant auto-detection**
   (roadmap #4) — larger; re-read their design notes in docs/roadmap.md
   before starting. Multi-detector needs per-frame PONI association threaded
   through reduce; don't attempt it as a side effect of something else.
5. **Unmerged gallery branch**: `claude/reduce-gallery(2)` holds a
   click-to-flag cake-matrix viewer (backend verified, never merged. ~350
   lines). Evaluate against current main before reusing; may be stale.

## Conventions that bite (learned this session)

- Tk is not thread-safe: worker threads must never touch widgets OR
  `root.after` — push events through a queue drained by the main-thread
  poller (see `reduce/gui.py` `_watch_queue`, `analysis/gui.py`
  `_event_queue`).
- Every HDF5 write is tmp+`os.replace` EXCEPT the live watch file
  (documented exception). Don't add more exceptions.
- pyFAI logs-and-ignores unknown kwargs instead of raising — never feature-
  detect it with try/except TypeError; use signature inspection
  (`reduce/processing._named_params`).
- The GUIs must fit 1024x700: measure `winfo_reqwidth/reqheight` vs
  allocation under Xvfb after layout changes (this session's overflow fix).
- Fixed-`wraplength` labels clip; use `AnalysisApp.autowrap`.
- Docs claims are checked against argparse blocks / session `_DEFAULTS` —
  quoting a flag or default that doesn't exist in code is a review failure.
