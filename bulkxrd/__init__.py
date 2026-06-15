"""bulkxrd — notebook-controlled workflow for powder XRD calibration,
reduction, and analysis.

The workflow is organized as one subpackage per pipeline stage, with shared
infrastructure in `core` (session config, IO, naming, masks, env checks) and
`guikit` (shared Tk/matplotlib theming):

    bulkxrd.core      shared utilities (stdlib + numpy only)
    bulkxrd.guikit    shared GUI/plot theme
    bulkxrd.calib     calibration review (pyFAI QA GUI + worker)
    bulkxrd.reduce    dataset reduction / batch integration (planned)
    bulkxrd.analysis  pattern analysis: peaks, EOS, figures (planned)

The light, dependency-free core API is re-exported here; stage APIs are
imported from their subpackage, e.g. `from bulkxrd.calib import run_app`.
"""
from __future__ import annotations

from .core import (
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
    OPTIONAL_IMPORTS,
    REQUIRED_IMPORTS,
    DependencyStatus,
    check_dependencies,
    find_conda_exe,
    package_install_command,
    run_install_command,
    gen_label,
    generation_paths,
    generation_stem,
    next_available_path,
)

__version__ = VERSION
