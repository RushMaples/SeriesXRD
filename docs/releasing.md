# Release checklist

1. Confirm the package name, repository URLs, author metadata, license, and
   bundled-data provenance.
2. Update `CHANGELOG.md`, `CITATION.cff`, and the version in
   `seriesxrd/core/config.py`.
3. Run `python -m pytest` in every supported environment.
4. Build and inspect both distributions:

   ```bash
   python -m build
   python -m twine check dist/*
   ```

5. Install the wheel and source distribution in clean environments and test
   `seriesxrd --help`, the unified GUI, and a small end-to-end workflow.
6. Upload a release candidate to TestPyPI and repeat the installation check.
7. Tag the reviewed commit, publish through PyPI Trusted Publishing, and create
   the matching GitHub release.
8. After the repository is public, archive the release with Zenodo and add its
   DOI and release date to `CITATION.cff`.
