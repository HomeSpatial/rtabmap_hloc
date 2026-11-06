# Contributing

Use Linux and Python 3.12. Start with a branch and public, synthetic fixtures:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest
uv run --locked --extra dev python scripts/check_release.py
```

Keep fixes focused and describe the triggering input, resulting behavior and
validation. Include a regression test for parsing, geometry, coordinate-frame,
failure-status or publication changes. Tests generate temporary assets and do
not require private datasets, model weights, a GPU or external services.
They test real public pycolmap APIs while mocking expensive model inference
and Colab transport. They do not establish reconstruction accuracy or live
service compatibility.

When dependencies change, run `uv lock` deliberately, review the diff and rerun
tests. Keep the HLOC revision explicit. Use a host workspace's lockfile for
custom forks; do not hard-code host paths in the package.

`src/rtabmap_hloc_reprocess/_exchange/SOURCE.json` records the shared reference
implementation hashes. Coordinate changes with the reference implementation,
then synchronize files and hashes together. Do not silently patch only one copy.

## Public source and release checks

`release-files.json` is the explicit source allowlist. Add new public source
files to it. Keep scans, photos, database files, checkpoints, credentials,
runtime caches and experimental outputs out of the repository and distributions.

```bash
uv run --locked --extra dev python scripts/check_release.py
uv run --locked --extra dev python -m build
uv run --locked --extra dev python -m twine check dist/*
uv run --locked --extra dev python scripts/check_release.py --dist dist
```

The checker verifies the allowlist, vendored hashes, local Markdown links,
common credential/private-path patterns, license metadata, and package contents.
Pattern checks are not proof that every sensitive string has been detected;
review new data and Git history before publication. CI installs the wheel into
a separate environment and invokes it outside the checkout to detect missing
package resources and accidental source-tree imports.

Before a release, document validation limits in the changelog, check both the
wheel and source archive, and inspect the final diff. Publishing a Git tag,
GitHub release or package registry upload is a separate maintainer action;
the CI workflow does not publish automatically. The current Git-sourced HLOC
requirement is intended for source/GitHub installs; a PyPI release needs a
separate dependency-distribution decision because index uploads restrict
direct URL dependencies.

Contributions are accepted under this repository's Apache-2.0 license. Preserve
third-party notices and ensure you have permission to share contributed code,
documentation and fixtures. Report security issues according to [SECURITY.md](SECURITY.md).
