# Releasing SeriesXRD

Releases are driven by the tag-triggered workflow in
`.github/workflows/release.yml`: it builds the distributions once, verifies
the tag matches the package version, install-tests the wheel and sdist,
publishes to TestPyPI, waits for manual approval, publishes the same
artifacts to PyPI through Trusted Publishing (with automatic attestations),
and creates the GitHub release.

## One-time setup (before the first release)

These are account-level actions the workflow depends on. Do them once, in
this order:

1. **Reserve the name with Pending Trusted Publishers.** On both
   [test.pypi.org](https://test.pypi.org/manage/account/publishing/) and
   [pypi.org](https://pypi.org/manage/account/publishing/), add a *pending*
   Trusted Publisher for the project name `seriesxrd` with:
   - Owner: `RushMaples`, Repository: `SeriesXRD`
   - Workflow name: `release.yml`
   - Environment: `testpypi` (on TestPyPI) / `pypi` (on PyPI)

   No API tokens are created or stored anywhere. See the
   [PyPI Trusted Publishers docs](https://docs.pypi.org/trusted-publishers/).
2. **Create the GitHub environments.** Repository → Settings →
   Environments: create `testpypi` and `pypi`. On `pypi`, add yourself as a
   **required reviewer** — this is the manual approval gate between TestPyPI
   and production PyPI.
3. **Protect `main`.** Settings → Branches → add a ruleset/protection rule
   for `main` requiring the CI checks to pass before merging.
4. **Enable Zenodo archiving.** Log in to [zenodo.org](https://zenodo.org)
   with GitHub, flip the toggle for `RushMaples/SeriesXRD` under GitHub
   integration *before* creating the first release. Zenodo reads
   `CITATION.cff` for metadata and archives every GitHub release
   automatically.
5. **Confirm licensing and attribution.** MIT release confirmed with the
   host lab/PI; funding acknowledgment present in `CREDITS.md`; bundled
   phase-library data provenance documented in `docs/phase-sources.md`.

## Per-release checklist

1. Update `CHANGELOG.md` (move Unreleased → the new version with a date).
2. Set the version in `seriesxrd/core/config.py` (`VERSION`) and the same
   `version:` in `CITATION.cff`.
3. Make sure CI is green on `main`, including the floor-versions and
   GUI-startup jobs.
4. Sanity-check locally if anything packaging-related changed:

   ```bash
   python -m build && python -m twine check dist/*
   ```

5. Tag the reviewed commit and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

6. The workflow publishes to **TestPyPI** automatically. Verify the release
   candidate installs from there before approving production:

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ seriesxrd
   seriesxrd --help
   ```

7. Approve the `pypi` environment deployment in the workflow run. The same
   artifacts go to PyPI and the GitHub release is created with generated
   notes.
8. After Zenodo mints the DOI for the release, add `doi` and
   `date-released` to `CITATION.cff` (this can ride in the next commit —
   Zenodo versions each release separately, and the concept DOI stays
   stable).

## Yanking / fixing a bad release

Never delete and re-upload the same version — PyPI forbids re-using a file
name. Fix forward: bump the patch version, tag again, and yank the bad
release on PyPI (yanked releases stay installable for pinned users but stop
being the default).
