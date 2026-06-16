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
    'bulkxrd.analysis.parallel', 'bulkxrd.analysis.batch',
]
for m in mods:
    importlib.import_module(m)
    print('IMPORT OK:', m)
print('ALL IMPORTS OK')
