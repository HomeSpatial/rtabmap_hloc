# Installation and host integration

## Supported baseline

Use Linux, Python 3.12 and the committed `uv.lock`. Git is required to fetch the
pinned HLOC source and its Git dependencies. `uv sync --locked` refuses to
silently update the lock. Windows and macOS are not validated release targets.
On Windows, use a Linux environment such as WSL2 for the supported path.

For ordinary pip installation into an active Python 3.12 virtual environment:

```bash
python -m pip install .
rtabmap-hloc --help
```

`./setup_env.sh` is an equivalent convenience script. Pip does not read
`uv.lock`; its resolved versions can differ. A local wheel can be installed
with `python -m pip install path/to/rtabmap_hloc_reprocess-0.1.0-py3-none-any.whl`.
Neither command implies that a release is available on PyPI.

## Input requirements

Database input needs the RTAB-Map `Node`, `Data` and `Admin` tables, usable
images, poses and calibration. Parsing supports the camera serialization
variants covered by the tests; arbitrary RTAB-Map schema versions or
multi-camera rigs are not guaranteed compatible. Databases open in SQLite
read-only mode. Python callers can explicitly request `read_only=False` when
they intentionally need a writable connection.

An exported directory contains:

```text
exported/
  poses.txt       # timestamp x y z qx qy qz qw node_id
  rgb/           # numeric node IDs, e.g. 1.jpg or 1.png
  calib/         # matching node IDs, e.g. 1.yaml
```

Calibration uses RTAB-Map's OpenCV YAML layout (`image_width`, `image_height`,
`camera_matrix`, optional `local_transform`). The pipeline exports PINHOLE
intrinsics and does not apply a general image-undistortion stage. Use suitably
rectified inputs and verify calibration for your sensor. Missing calibration
fails the default fixed-intrinsics path; `--no-rtabmap-intrinsics` deliberately
switches full SfM to COLMAP's image-reader intrinsics.

## GPU and model downloads

Check the actual environment with:

```bash
uv run --locked python -c 'import torch, pycolmap; print(torch.__version__, torch.cuda.is_available(), pycolmap.__version__)'
```

Neural inference and COLMAP geometry are separate components. A working CUDA
PyTorch install does not guarantee a GPU-enabled pycolmap build. The default
local path uses HLOC's own device selection and model loaders.

HLOC is pinned to `c13273bd0ecc2917a35910fd843712a1c6243193`. This pins the adapter
code, not every runtime model download: for example, its MegaLoc loader uses
Torch Hub. Review upstream downloads and cache them in a controlled environment
if you need offline execution. The repository does not prefetch or redistribute
weights automatically during installation or tests.

`--colmap-profile high_accuracy` applies stronger reconstruction settings and
checks them against the installed pycolmap API. It is not a guaranteed accuracy
improvement; compare registration, residuals and independent geometry evidence.

## Host HLOC overrides

A uv workspace can supply its own HLOC checkout at the host root:

```toml
[tool.uv.sources]
hloc = { path = "3rdparty/hloc", editable = true }
```

The host lockfile controls that environment. This package does not search host
directories or inject a private HLOC checkout into `sys.path`. Pip ignores uv
source overrides. The host is also responsible for custom pycolmap builds.
Re-run the tests after changing either source.

The optional `onnx` extra installs ONNX Runtime GPU dependencies for existing
host integrations. It does not replace PyTorch inference, add a CLI ONNX backend,
or provide ONNX model files/configurations for official HLOC.

## Optional Colab adapter

`--inference-backend colab` requires a separately installed compatible
`remap.colab.model_backend.ColabModelBackend`. There is no bundled transport,
authentication or provisioning in this distribution. Without the adapter the
command explains the missing dependency and the local `torch` alternative.
Do not install an unrelated package with a similar name to satisfy this import.

The adapter dispatches only feature extraction, retrieval and matching;
COLMAP reconstruction and camera geometry remain on the host, with CPU bundle
adjustment. Supported execution profiles are `colab-models-l4-v1` and
`colab-models-l4-v2`. Host-side source/weight bundling, result validation and
provenance are implemented here. Tests simulate the transport; they do not
provision Colab or establish live-service compatibility.
