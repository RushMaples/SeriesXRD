"""Shared infrastructure for all seriesxrd pipeline stages.

Everything here is importable before pyFAI/matplotlib/Tk are installed so the
notebook preflight can report missing packages instead of crashing.
`io` and `masks` additionally require numpy.
"""
from __future__ import annotations

from .config import (
    VERSION,
    TOOL_NAME,
    SessionConfig,
    config_path_for_notebook,
    copy_file,
    default_backend_dir_from_notebook,
    default_python_exe,
    default_workspace_paths,
    ensure_dir,
    json_default,
    load_session_config,
    now_iso,
    now_timestamp,
    print_status,
    read_json,
    safe_stem,
    save_session_config,
    sha256_file,
    validate_session_config,
    write_json,
)
from .env import (
    OPTIONAL_IMPORTS,
    REQUIRED_IMPORTS,
    DependencyStatus,
    check_dependencies,
    find_conda_exe,
    package_install_command,
    run_install_command,
)
from .handoff import (
    HANDOFF_FILENAME,
    Handoff,
    find_latest_handoff,
    load_handoff,
)
from .naming import (
    gen_label,
    generation_paths,
    generation_stem,
    next_available_path,
)
from .io import (
    read_detector_image,
    write_table_csv,
    write_xy_csv,
)
from .masks import (
    automatic_mask,
    load_mask_npz,
    load_mask_npz_with_metadata,
    polygon_to_mask,
    save_mask_npz,
    save_mask_preview_png,
)
