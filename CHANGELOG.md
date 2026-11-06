# Changelog

## 0.1.0 — Unreleased

- Standalone RTAB-Map database and export-directory ingestion, HLOC sparse
  reconstruction, existing-pose triangulation and Sim3 alignment.
- Portable reconstruction descriptions, immutable bundle stages, checksums
  and explicit execution/alignment/quality status.
- Apache-2.0 license, bilingual introduction, public installation and output
  documentation, contribution guidance and distribution validation.
- Linux/Python 3.12 CI using locked public HLOC and pycolmap dependencies,
  source/wheel checks and installed entry-point/resource smoke tests.
- Read-only database access by default; explicit Python opt-in for writes.
- Copy exported-directory photos into portable outputs and resume JPEG images.
- Respect the selected image list in the standard SfM branch.
- Add `--version`, module execution, positive pair-count validation and an
  actionable error when the optional Colab adapter is unavailable.

Validation covers synthetic fixtures and package integration. Real-scene
reconstruction quality, GPU performance and live Colab execution remain separate
acceptance work; they are not established by the test suite.
