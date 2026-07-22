# Changelog

All notable changes to SeriesXRD will be documented here. The project follows
semantic versioning once a stable public API is declared.

## [Unreleased]

### Added

- A GSAS-II sequential-refinement round trip: exports now include explicit
  frame/group manifests and a standalone GPX-to-JSON helper, while
  `seriesxrd-import-gsas` (also available in the Analysis GUI) atomically
  imports refined weight fractions with uncertainties, unit cells, and fit
  quality under `/refinement` without replacing the earlier `/fractions`
  screening estimates.

### Changed

- The Analysis GUI now presents export, external refinement, and result import
  together on a dedicated **Refinement → GSAS-II round trip** page.
- Live Mocha/Latte theme switching from the unified application's View menu,
  with a per-user preference and `--theme` overrides for standalone stage
  launchers.

### Changed

- Shared ttk, raw-Tk, and embedded-Matplotlib styling now follows one mutable
  semantic palette without restarting GUI panes or their worker processes.
- Publication figure exports use a predictable light palette on a white
  background independently of the active UI theme.

## [0.2.0] - 2026-07-21

### Changed

- Adopted SeriesXRD as the project, Python package, application, and
  command-line tool name before the first public release.
- Clarified the calibration → reduction → analysis workflow and GUI labels.
- Analysis manifests now record the real SeriesXRD version
  (`seriesxrd_version`) separately from the file-layout `schema_version`;
  the analysis HDF5 carries root version attrs and a `/provenance` group
  (effective configuration, dependency versions, platform, input-file
  fingerprints), and each appending step records itself under
  `/provenance/steps/<step>`.
- Dependency declarations are truthful and tested: the environment check
  covers scipy and h5py as core requirements, `pyproject.toml` declares
  minimum versions validated by a lowest-supported-dependencies CI job,
  and `environment.yml` lists scipy/h5py explicitly.
- The Analysis stage navigates through a hierarchical left rail
  (Configure / Run / Review / Export) instead of a single row of 12 tabs.
- Plot axes use standard scientific notation — q (Å⁻¹), 2θ (°),
  Azimuth (°), Intensity (counts) — everywhere a person reads them;
  internal unit codes are unchanged.
- Successful saves and exports notify through the status bar instead of
  modal dialogs; errors remain modal.
- HDF5 inspection shows a human-readable summary by default, with the raw
  tree and full attributes behind an "Advanced details" toggle.
- Corrected phase-library source attributions (Pt and Si author lists,
  Re citation) and pinned the marker EOS parameters to named literature
  scales: Au to Anderson et al. 1989 as recommended (167 GPa / 5.5), Pt
  re-cited to the Fei et al. 2007 Vinet scale (273 GPa / 5.20), Re to
  Anzellini et al. 2014's Vinet fit unrounded (352.6 GPa / 4.56) — see
  `docs/phase-sources.md`.

### Added

- Unified desktop navigation and automatic stage handoffs.
- GUI access to texture, spot tracking, phase-fraction, microstructure, and
  refinement export tools.
- Continuous integration, distribution checks, citation metadata, and
  community contribution guidance.
- Tag-triggered release pipeline: build once, install-test wheel and sdist,
  TestPyPI, manual approval, PyPI Trusted Publishing with attestations, and
  an automatic GitHub release (`.github/workflows/release.yml`;
  `docs/releasing.md` documents the one-time setup).
- CI matrix: Python 3.10–3.14 on Ubuntu plus the newest Python on Windows
  and macOS, a dependency-floor job, wheel/sdist install smoke tests, a
  headless GUI startup test under xvfb, and a weekly ML-extras run.
- Documentation set for publication: `docs/architecture.md`,
  `docs/file-format.md`, `docs/validation.md` (validation and limitations),
  and `docs/phase-sources.md` (DOI-verified bibliography for every bundled
  phase-library value).
- Run page preflight (input, frame count, steps, output, warnings) and a
  completion summary with "Review results" / "Open output folder" actions.
- Help menu: user guide, demonstration, validation/limitations, citation,
  Report a problem, and Copy diagnostics (a support-ready provenance
  report); About states license and repository.
- Tools → Model development dialog: GUI access to corpus screening,
  benchmarking, and learned-scorer training with live output.
- Figure-export presets (screen / presentation / publication with
  PNG/SVG/PDF) and an `export_provenance.txt` sidecar on frame exports.
- Keyboard-accessible tooltips (focus shows, Escape dismisses) and an
  ellipsized workspace path in the header (full path in tooltip,
  click to copy).
- Governance/maintainer policy (`GOVERNANCE.md`), pull-request template,
  and the NSF REU funding acknowledgment (Award No. 2547979) in
  `CREDITS.md`.
- Six test files that pytest previously never collected (background,
  peaks, fit-source, analysis-review, spots, smoke) now run in CI.
