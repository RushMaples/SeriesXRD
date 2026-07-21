"""Headless smoke test. Does not require pyFAI or a display.

Runnable directly (``python tests/test_smoke.py``) and collected by pytest.
"""
from pathlib import Path
import sys, tempfile, json

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "seriesxrd"
sys.path.insert(0, str(REPO_ROOT))

from seriesxrd import (SessionConfig, save_session_config, load_session_config,
                       validate_session_config, check_dependencies)


def test_smoke():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = SessionConfig.default(td, PKG_DIR)
        assert cfg.dioptas_image_flip is False
        cfg.python_exe = sys.executable
        cfg.raw_data_dir = str(td)
        cfg.processed_root = str(td / 'processed')
        cfg.figures_root = str(td / 'figures')
        cfg.metadata_root = str(td / 'metadata')
        cfg.accepted_output_root = str(td / 'accepted')
        cfg.logs_root = str(td / 'logs')
        p = save_session_config(cfg, td / 'calibration_session_config.json')
        cfg2 = load_session_config(td)
        assert cfg2.dioptas_image_flip is False
        problems = validate_session_config(cfg2)
        dep = check_dependencies(sys.executable)
        print('CONFIG:', p)
        print('PROBLEMS:', problems)
        print('DEPENDENCIES:', json.dumps(dep.to_dict(), indent=2))
        assert p.exists()
        assert not problems, f"unexpected config problems: {problems}"


if __name__ == "__main__":
    test_smoke()
    print('SMOKE TEST OK')
