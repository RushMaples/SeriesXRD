# Governance and maintenance

SeriesXRD is maintained by a single maintainer, **Rush Maples**
([@RushMaples](https://github.com/RushMaples)), who has final say on
releases, scope, and merges. This document says how that works in practice
so contributors know what to expect.

## Decision making

- Small fixes and clearly-scoped improvements: open a pull request directly
  (see [`CONTRIBUTING.md`](CONTRIBUTING.md)).
- Behavior changes, new dependencies, or anything touching the scientific
  core (background separation, peak fitting, identification, EOS handling):
  open an issue first so the approach can be agreed before the work.
- Scientific-validity concerns are treated as defects, not opinions — if a
  result is wrong or a documented claim overstates what the code does,
  file an issue with the data to reproduce it.

## Releases

Releases follow [`docs/releasing.md`](docs/releasing.md): tagged from
`main`, published to PyPI through Trusted Publishing, and archived on
Zenodo. Versioning follows semantic versioning once a stable public API is
declared; while the project is pre-1.0, minor versions may change behavior
(noted in [`CHANGELOG.md`](CHANGELOG.md)).

## Maintainer changes

If the project gains regular contributors, commit access and this document
will be revisited. If the maintainer becomes unavailable for an extended
period, the project supervisor (see [`CREDITS.md`](CREDITS.md)) may
designate a successor or archive the repository with its Zenodo record as
the citable artifact.

## Development transparency

Parts of this codebase were developed with AI coding assistants under the
maintainer's direction and review. Repository files used to configure that
tooling (`CLAUDE.md`, `docs/agents/`) are development-automation
configuration, not user documentation — user documentation lives in
[`README.md`](README.md) and [`docs/`](docs/).
