from pathlib import Path
import sys, importlib
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
mods = [
    'bulkxrd',
    'bulkxrd.core', 'bulkxrd.core.config', 'bulkxrd.core.env', 'bulkxrd.core.naming',
    'bulkxrd.core.io', 'bulkxrd.core.masks', 'bulkxrd.core.handoff', 'bulkxrd.core.inspect',
    'bulkxrd.guikit', 'bulkxrd.guikit.theme', 'bulkxrd.guikit.tkstyle',
    'bulkxrd.guikit.tooltip', 'bulkxrd.guikit.dpi', 'bulkxrd.guikit.dpi',
    'bulkxrd.calib', 'bulkxrd.calib.processing', 'bulkxrd.calib.dioptas',
    'bulkxrd.calib.worker', 'bulkxrd.calib.gui',
    'bulkxrd.calib.run_gui',
    'bulkxrd.reduce', 'bulkxrd.reduce.processing',
    'bulkxrd.reduce.session', 'bulkxrd.reduce.review',
    'bulkxrd.reduce.worker', 'bulkxrd.reduce.gui', 'bulkxrd.reduce.run_gui',
    'bulkxrd.app',
    'bulkxrd.analysis', 'bulkxrd.analysis.background', 'bulkxrd.analysis.peaks',
    'bulkxrd.analysis.review', 'bulkxrd.analysis.session',
    'bulkxrd.analysis.worker', 'bulkxrd.analysis.gui', 'bulkxrd.analysis.run_gui',
    'bulkxrd.analysis.phases', 'bulkxrd.analysis.refdata', 'bulkxrd.analysis.identify',
    'bulkxrd.analysis.heatmap', 'bulkxrd.analysis.mldata',
    'bulkxrd.analysis.residual', 'bulkxrd.analysis.frame_metadata',
    'bulkxrd.analysis.ml_features', 'bulkxrd.analysis.ml_simulate',
    'bulkxrd.analysis.ml_rank',
    'bulkxrd.analysis.parallel', 'bulkxrd.analysis.batch',
]
for m in mods:
    importlib.import_module(m)
    print('IMPORT OK:', m)


def test_cp1252_stdio_does_not_crash():
    """A non-ASCII log line must not abort on a legacy (cp1252) console once
    make_stdio_robust() has run — Windows would otherwise raise UnicodeEncodeError."""
    import io
    from bulkxrd.core.config import make_stdio_robust, print_status
    saved = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")  # strict
        make_stdio_robust()
        print("[IDENTIFY] window 2.0*sigma σ → ok", flush=True)  # σ, →
        print_status("pressure prior σ set", "INFO")
        sys.stdout.flush()
    finally:
        sys.stdout = saved
    # print_status must also survive even without reconfigure (its own fallback).
    saved = sys.stdout
    try:
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        print_status("raw σ fallback", "INFO")   # no make_stdio_robust here
        sys.stdout.flush()
    finally:
        sys.stdout = saved


test_cp1252_stdio_does_not_crash()
print('ALL IMPORTS OK')
