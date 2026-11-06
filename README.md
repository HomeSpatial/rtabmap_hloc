# RTABMap-HLOC

[中文](README.zh-CN.md) · [Installation and advanced options](docs/advanced.md) · [Output format](docs/outputs.md)

Rebuild a **sparse COLMAP model** from RTAB-Map images, then align it to the
original metric trajectory. Use the result with downstream reconstruction or
localization tools without depending on a running RTAB-Map service.

The package reads an RTAB-Map SQLite database or an `rtabmap-export` directory.
It exports photos and calibration, runs [HLOC](https://github.com/cvg/Hierarchical-Localization),
and publishes a portable `reconstruction.json` with native COLMAP files and
checksums. It does not generate a dense mesh or a structured building model.

**Status:** alpha, version 0.1.0. Linux and Python 3.12 are the supported baseline.
No RTAB-Map C++ installation is needed to read an existing database.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then:

```bash
git clone https://github.com/HomeSpatial/rtabmap_hloc.git
cd rtabmap_hloc
uv sync --locked
uv run --locked rtabmap-hloc --version
uv run --locked rtabmap-hloc ./input.db -o ./output --scene-id house
```

The install includes PyTorch and can be large. Model weights are downloaded on
first use and are not distributed here. A CUDA-capable GPU is recommended for
neural feature inference; actual CUDA availability depends on your PyTorch
installation and driver. Unit tests and CLI checks do not need a GPU or weights.

The default pipeline uses **ALIKED + MegaLoc + LightGlue**, reconstructs camera
poses using full SfM, and estimates a Sim3 transform to the RTAB-Map trajectory.
RTAB-Map per-frame intrinsics are fixed by default. Alignment recovers the input
trajectory's scale; it does not independently establish ground-truth accuracy.

```text
RTAB-Map database / exported photos + calibration + poses
    → ALIKED features → MegaLoc retrieval → LightGlue matching
    → COLMAP sparse reconstruction → Sim3 alignment
    → sfm_aligned/ + images/ + reconstruction.json
```

## Common commands

```bash
# Use an existing rtabmap-export directory (rgb/, calib/, poses.txt).
uv run --locked rtabmap-hloc ./exported -o ./output-exported

# Triangulate from the input trajectory instead of estimating poses with SfM.
uv run --locked rtabmap-hloc ./input.db -o ./output-poses --use-existing-poses

# Allow refinement of the imported camera intrinsics.
uv run --locked rtabmap-hloc ./input.db -o ./output-refined --refine-intrinsics

# Store a successful result as a new immutable scene-bundle stage.
uv run --locked rtabmap-hloc ./input.db --bundle ./house --stage sfm

uv run --locked rtabmap-hloc --help
```

Choose a fresh output directory for each completed reconstruction. A directory
containing `reconstruction.json` cannot be overwritten. The source database is
opened read-only, and imported photos are copied into the output directory.

Full SfM uses RTAB-Map poses for final alignment; `--use-existing-poses` uses them
throughout triangulation. Keep this distinction when comparing results.
`--up-axis` declares the world up axis (default `+Z`); it does not rotate geometry.

## Results and failure handling

| Path | Purpose |
| --- | --- |
| `images/` | Exported input photos |
| `hloc/` | Feature descriptors, matches and intermediate SfM |
| `sfm_aligned/` | COLMAP binary sparse model in the RTAB-Map metric frame |
| `metadata.json` | Configuration, camera policy and reconstruction metrics |
| `alignment-report.json` | Alignment method and residuals |
| `reconstruction.json` | Portable description of registered photos and native assets |
| `run-status.json` | Execution, alignment and quality status |

The CLI exits nonzero on reconstruction, alignment, export or bundle-publication
failure. It publishes no new description if metric alignment fails. Successful
execution still records quality as `not_evaluated`. See [output semantics](docs/outputs.md)
for interpretation, resume constraints and downstream consumption.

## Development

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest
uv run --locked --extra dev python scripts/check_release.py
uv run --locked --extra dev python -m build
uv run --locked --extra dev python -m twine check dist/*
uv run --locked --extra dev python scripts/check_release.py --dist dist
```

CI runs tests with public dependencies, checks both distribution formats,
and installs the built wheel outside the source tree. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the release workflow and test boundaries.

## Optional integrations and limitations

The standalone path uses local PyTorch inference. The `colab` backend requires
a separately installed compatible `remap.colab.model_backend` adapter; it is
not included in this package. The `onnx` extra installs a runtime only and does
not add ONNX model configurations to official HLOC. [Advanced setup](docs/advanced.md)
explains these host-specific integrations and HLOC source overrides.

The lockfile pins Python packages and HLOC source, but upstream Torch Hub model
loaders may fetch additional code and weights at runtime. This is not a fully
offline or completely frozen model environment. Real-scene accuracy, GPU memory
use and runtime must be evaluated on your own data. No private scans, photos,
model checkpoints or reconstruction results are included.

## License

Original code and documentation are licensed under [Apache-2.0](LICENSE).
See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the
vendored exchange reference and external dependencies. Third-party model code,
weights and input data retain their own terms.
