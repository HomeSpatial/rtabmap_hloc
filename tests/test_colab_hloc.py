"""Exercise the model/geometry boundary without allocating a Colab runtime."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
from PIL import Image
import pytest

from rtabmap_hloc_reprocess import colab_hloc, hloc_pipeline


@pytest.fixture
def options():
    return {
        "feature_conf": "aliked-n16",
        "retrieval_conf": "megaloc",
        "matcher_conf": "aliked+lightglue",
        "num_matched": 20,
        "feature_config": {
            "model": {"name": "aliked", "max_num_keypoints": 4096},
            "preprocessing": {"resize_max": 1024},
        },
        "retrieval_config": {"model": {"name": "megaloc"}},
        "matcher_config": {"model": {"name": "lightglue", "features": "aliked"}},
    }


@pytest.fixture
def artifact(tmp_path, options):
    images = {}
    for name in ["nested/0001.png", "0002.png"]:
        path = tmp_path / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3)).save(path)
        images[name] = path
    output = tmp_path / "result"
    output.mkdir()
    (output.parent / "committed.json").write_text('{"request_id": "test"}')
    (output.parent / "request.json").write_text('{"inputs": {"hash": "test"}}')
    files = colab_hloc.output_names(options)
    with h5py.File(output / files["features"], "w") as h5:
        for name in images:
            group = h5.create_group(name)
            group["keypoints"] = np.array([[1.0, 1.0], [2.0, 2.0]])
            group["descriptors"] = np.ones((128, 2))
            group["image_size"] = [4, 3]
    with h5py.File(output / files["retrieval"], "w") as h5:
        for name in images:
            h5.create_group(name)["global_descriptor"] = np.ones(4)
    (output / files["pairs"]).write_text(
        "nested/0001.png 0002.png\n0002.png nested/0001.png\n"
    )
    with h5py.File(output / files["matches"], "w") as h5:
        group = h5.create_group("nested-0001.png/0002.png")
        group["matches0"] = np.array([1, -1], dtype=np.int16)
        group["matching_scores0"] = [0.75, 0.0]
    return output, images


def test_result_accepts_nested_names_and_reverse_deduplicated_pairs(artifact, options):
    output, images = artifact
    assert set(colab_hloc.validate_result(output, images, options)) == {
        "features",
        "retrieval",
        "pairs",
        "matches",
    }


@pytest.mark.parametrize("fault", ["shape", "index", "nan", "image_size", "pair"])
def test_result_rejects_corrupt_or_mismatched_payload(artifact, options, fault):
    output, images = artifact
    files = colab_hloc.output_names(options)
    if fault == "pair":
        (output / files["pairs"]).write_text("missing.png 0002.png\n")
    elif fault in {"shape", "image_size"}:
        with h5py.File(output / files["features"], "r+") as h5:
            group = h5["nested/0001.png"]
            if fault == "shape":
                del group["descriptors"]
                group["descriptors"] = np.ones((128, 3))
            else:
                group["image_size"][:] = [40, 30]
    else:
        with h5py.File(output / files["matches"], "r+") as h5:
            group = h5["nested-0001.png/0002.png"]
            if fault == "index":
                group["matches0"][0] = 2
            else:
                group["matching_scores0"][0] = np.nan
    with pytest.raises(ValueError):
        colab_hloc.validate_result(output, images, options)


@pytest.mark.parametrize(
    "names", [["../escape.png"], ["/escape.png"], ["a b.png"], ["a.png", "a.png"]]
)
def test_image_names_cannot_escape_or_alias_inputs(tmp_path, names):
    with pytest.raises(ValueError):
        colab_hloc.image_inputs(tmp_path, names)


def test_adapter_passes_exact_configs_and_validates_before_materializing(
    monkeypatch, artifact, options, tmp_path
):
    output, images = artifact
    captured = {}

    class Backend:
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            captured["closed"] = True

        def run(self, task, **kwargs):
            captured["task"] = task
            captured.update(kwargs)
            return output

    module = ModuleType("remap.colab_model_backend")
    module.ColabModelBackend = Backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(colab_hloc, "resolve_options", lambda *args: deepcopy(options))
    monkeypatch.setattr(
        colab_hloc, "bundle_hloc_sources", lambda opts: {"source.py": Path(__file__)}
    )
    target = tmp_path / "materialized"
    result = colab_hloc.run_colab_features(
        tmp_path / "images",
        target,
        "aliked-n16",
        "megaloc",
        "aliked+lightglue",
        20,
        list(images),
    )
    assert captured["task"] == "hloc"
    assert captured["options"] == options
    assert captured["images"] == images
    assert captured["closed"]
    assert all(path.is_file() for path in result.values())
    assert (
        json.loads((target / "colab_hloc_provenance.json").read_text())[
            "local_geometry_device"
        ]
        == "cpu"
    )


@pytest.mark.parametrize("triangulate", [False, True])
def test_colab_boundary_never_executes_local_models_and_forces_cpu_ba(
    monkeypatch, tmp_path, triangulate
):
    captured = {}

    def fail(*args, **kwargs):
        raise AssertionError("Local model execution is forbidden")

    def geometry(*args, **kwargs):
        captured["geometry"] = kwargs
        return SimpleNamespace(num_reg_images=lambda: 2, num_points3D=lambda: 100)

    fake_hloc = SimpleNamespace(
        extract_features=SimpleNamespace(main=fail),
        match_features=SimpleNamespace(main=fail),
        pairs_from_retrieval=SimpleNamespace(main=fail),
        reconstruction=SimpleNamespace(main=geometry),
        triangulation=SimpleNamespace(main=geometry),
    )
    monkeypatch.setitem(sys.modules, "hloc", fake_hloc)
    monkeypatch.setitem(
        sys.modules, "pycolmap", SimpleNamespace(CameraMode=SimpleNamespace(AUTO=1))
    )
    monkeypatch.setattr(
        colab_hloc,
        "run_colab_features",
        lambda *args, **kwargs: captured.update(remote=(args, kwargs)),
    )
    kwargs = dict(
        images_dir=tmp_path,
        output_dir=tmp_path / "output",
        inference_backend="colab",
        image_list=["a.png", "b.png"],
        colmap_options={"min_num_matches": 30, "ba_use_gpu": True},
    )
    if triangulate:
        hloc_pipeline.run_hloc_triangulation(reference_model_dir=tmp_path, **kwargs)
    else:
        hloc_pipeline.run_hloc_reconstruction(**kwargs)
    assert captured["remote"][0][6] == ["a.png", "b.png"]
    assert captured["geometry"]["mapper_options"] == {
        "min_num_matches": 30,
        "ba_use_gpu": False,
    }


def test_remote_uses_frozen_full_configs(monkeypatch, tmp_path, options):
    captured = []

    def extract(config, images, **kwargs):
        captured.append((deepcopy(config), kwargs))
        config["model"].pop("name")  # HLOC models mutate their own config copy.

    fake = SimpleNamespace(
        extract_features=SimpleNamespace(main=extract),
        pairs_from_retrieval=SimpleNamespace(
            main=lambda *args, **kwargs: captured.append(kwargs)
        ),
        match_features=SimpleNamespace(
            main=lambda conf, *args, **kwargs: captured.append(conf)
        ),
    )
    monkeypatch.setitem(sys.modules, "hloc", fake)
    import contextlib

    monkeypatch.setattr(
        colab_hloc, "_offline_hub", lambda *args: contextlib.nullcontext()
    )
    before = deepcopy(options)
    colab_hloc.run_remote_hloc(
        tmp_path, tmp_path / "out", options, ["a.png"], input_root=tmp_path
    )
    assert options == before
    assert captured[0][0] == before["feature_config"]
    assert captured[1][0] == before["retrieval_config"]
    assert captured[2] == {"num_matched": 20}
    assert captured[3] == before["matcher_config"]


def test_missing_cached_weight_fails_without_network(tmp_path):
    with pytest.raises(RuntimeError, match="prefetched cache file"):
        colab_hloc._cached_file([tmp_path], "checkpoints/missing.pth")


def test_incompatible_feature_matcher_is_rejected(options):
    options["matcher_config"]["model"]["features"] = "superpoint"
    with pytest.raises(ValueError, match="feature family"):
        colab_hloc._validate_options(options)
