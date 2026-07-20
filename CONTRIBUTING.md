# Contributing to SeriesXRD

Contributions that improve the reliability, clarity, or general applicability
of SeriesXRD are welcome. Open an issue before beginning a large change so its
scope and scientific assumptions can be discussed.

## Development setup

Create an isolated Python 3.10 or newer environment, then install the project:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Optional scientific integrations can be installed with:

```bash
python -m pip install -e ".[dev,io,stacks,phases]"
```

## Pull requests

- Keep changes focused and explain their user-facing effect.
- Add regression tests for defects and tests for new scientific behavior.
- Document assumptions, units, uncertainty, and validity ranges.
- Cite the source of bundled constants, structures, or equations of state.
- Do not commit raw experimental data, generated workspaces, credentials, or
  confidential sample information.
- Run the full test suite before requesting review.

Use GitHub Issues for defects, feature proposals, and support questions that
may benefit other users. Follow `SECURITY.md` for security-sensitive reports.
