"""HLOC model-only Colab boundary; camera geometry always stays on the host.

The source and already-prefetched weights are inputs, so the transport hashes
them with the images and configurations. Remote jobs never resolve a moving
Torch Hub branch or silently use a different feature configuration.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Mapping

PROFILE = "colab-models-l4-v1"
_FEATURE_MODELS = {"superpoint", "aliked", "disk"}
_RETRIEVAL_MODELS = {"megaloc", "netvlad"}


def output_names(options: Mapping[str, Any]) -> dict[str, str]:
    for field in ("feature_conf", "retrieval_conf", "matcher_conf"):
        value = options[field]
        if (
            not isinstance(value, str)
            or not value
            or any(
                ch
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+."
                for ch in value
            )
        ):
            raise ValueError(f"Unsafe HLOC configuration name: {field}")
    count = options["num_matched"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("HLOC num_matched must be a positive integer")
    return {
        "features": f"{options['feature_conf']}.h5",
        "retrieval": f"{options['retrieval_conf']}.h5",
        "pairs": f"pairs-{options['retrieval_conf']}{count}.txt",
        "matches": f"{options['matcher_conf']}.h5",
    }


def resolve_options(
    feature_conf: str, retrieval_conf: str, matcher_conf: str, num_matched: int
) -> dict[str, Any]:
    from hloc import extract_features, match_features

    options = {
        "feature_conf": feature_conf,
        "retrieval_conf": retrieval_conf,
        "matcher_conf": matcher_conf,
        "num_matched": num_matched,
        "feature_config": deepcopy(extract_features.confs[feature_conf]),
        "retrieval_config": deepcopy(extract_features.confs[retrieval_conf]),
        "matcher_config": deepcopy(match_features.confs[matcher_conf]),
    }
    _validate_options(options)
    return options


def _validate_options(options: Mapping[str, Any]) -> None:
    output_names(options)
    feature = options["feature_config"]["model"]["name"]
    retrieval = options["retrieval_config"]["model"]["name"]
    matcher = options["matcher_config"]["model"]
    if feature not in _FEATURE_MODELS or retrieval not in _RETRIEVAL_MODELS:
        raise ValueError(
            "Colab HLOC supports Torch SuperPoint/ALIKED/DISK and MegaLoc/NetVLAD"
        )
    if matcher.get("name") != "lightglue" or matcher.get("features") != feature:
        raise ValueError("Colab HLOC requires the matching LightGlue feature family")


def image_inputs(images_dir: Path, image_list: list[str] | None) -> dict[str, Path]:
    images_dir = Path(images_dir).resolve()
    names = image_list
    if names is None:
        names = sorted(
            p.relative_to(images_dir).as_posix()
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix in {".jpg", ".png", ".jpeg", ".JPG", ".PNG"}
        )
    if not names or len(names) != len(set(names)):
        raise ValueError("HLOC images must be nonempty and unique")
    result = {}
    for name in names:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or "\\" in name
            or any(c.isspace() for c in name)
        ):
            raise ValueError(f"Unsafe HLOC image name: {name!r}")
        source = (images_dir / name).resolve()
        if not source.is_relative_to(images_dir) or not source.is_file():
            raise ValueError(f"HLOC image is missing or outside its root: {name!r}")
        result[name] = source
    return result


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Colab HLOC requires local source package {name}")
    return Path(spec.origin).parent


def _add_tree(result: dict[str, Path], prefix: str, source: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"Missing Colab HLOC source directory: {source}")
    for file in sorted(source.rglob("*")):
        if not file.is_file() or any(p in {".git", "__pycache__"} for p in file.parts):
            continue
        if file.suffix == ".py" or file.name.startswith("LICENSE"):
            result[f"{prefix}/{file.relative_to(source).as_posix()}"] = file


def _cache_roots() -> tuple[list[Path], list[Path]]:
    # Respect configured deployment mounts, then conventional user caches.
    home_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    torch = [Path(os.environ.get("TORCH_HOME", home_cache / "torch")) / "hub"]
    hf = [
        Path(
            os.environ.get(
                "HF_HUB_CACHE",
                Path(os.environ.get("HF_HOME", home_cache / "huggingface")) / "hub",
            )
        )
    ]
    model_root = os.environ.get("HOMESPATIAL_MODEL_CACHE")
    if model_root:
        torch.append(Path(model_root) / "torch/hub")
        hf.append(Path(model_root) / "huggingface/hub")
    # Repository development cache, when available; absent in deployment images.
    for parent in Path(__file__).resolve().parents:
        if (parent / "models/torch/hub").is_dir():
            torch.append(parent / "models/torch/hub")
            hf.append(parent / "models/huggingface/hub")
    return torch, hf


def _cached_file(roots: list[Path], relative: str) -> Path:
    for root in roots:
        candidate = root / relative
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise RuntimeError(
        f"Colab HLOC needs prefetched cache file {relative}; configure TORCH_HOME, "
        "HF_HUB_CACHE or HOMESPATIAL_MODEL_CACHE before allocating a session"
    )


def bundle_hloc_sources(options: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Select exact installed HLOC/LightGlue source and requested cached weights.

    The caller hashes every returned file. Passing no options returns source only.
    No model is instantiated and no GPU/network operation happens here.
    """
    result: dict[str, Path] = {}
    hloc = _package_dir("hloc")
    _add_tree(result, "src/hloc", hloc)
    _add_tree(
        result,
        "src/third_party/SuperGluePretrainedNetwork",
        hloc.parent / "third_party/SuperGluePretrainedNetwork",
    )
    _add_tree(result, "src/lightglue", _package_dir("lightglue"))
    if options is None:
        return result
    _validate_options(options)
    torch_roots, hf_roots = _cache_roots()
    model = options["feature_config"]["model"]
    feature = model["name"]
    weights = [f"{feature}_lightglue_v0-1_arxiv.pth"]
    if feature == "superpoint":
        relative = (
            "third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth"
        )
        checkpoint = hloc.parent / relative
        if not checkpoint.is_file():
            raise RuntimeError("Missing HLOC SuperPoint weights in third_party")
        result[f"src/{relative}"] = checkpoint
    elif feature == "aliked":
        weights.append(f"{model.get('model_name', 'aliked-n16')}.pth")
    elif feature == "disk":
        weights.append(f"{model.get('weights', 'depth')}-save.pth")
    for name in weights:
        result[f"cache/torch/hub/checkpoints/{name}"] = _cached_file(
            torch_roots, f"checkpoints/{name}"
        )
    if options["retrieval_config"]["model"]["name"] == "netvlad":
        name = options["retrieval_config"]["model"].get(
            "model_name", "VGG16-NetVLAD-Pitts30K"
        )
        relative = f"netvlad/{name}.mat"
        result[f"cache/torch/hub/{relative}"] = _cached_file(torch_roots, relative)
    else:
        hubconf = _cached_file(torch_roots, "gmberton_MegaLoc_main/hubconf.py")
        _add_tree(result, "cache/torch/hub/gmberton_MegaLoc_main", hubconf.parent)
        repo = "models--gberton--MegaLoc"
        ref = _cached_file(hf_roots, f"{repo}/refs/main")
        revision = ref.read_text().strip()
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Invalid cached MegaLoc revision")
        snapshot = ref.parent.parent / "snapshots" / revision / "model.safetensors"
        if not snapshot.is_file():
            raise RuntimeError("Missing cached MegaLoc snapshot weights")
        result[f"cache/huggingface/hub/{repo}/refs/main"] = ref
        # Dereference cached symlink; remote archives contain ordinary files only.
        result[
            f"cache/huggingface/hub/{repo}/snapshots/{revision}/model.safetensors"
        ] = snapshot.resolve()
    return result


@contextmanager
def _offline_hub(input_root: Path):
    import torch
    import huggingface_hub.constants as hf_constants

    original_load = torch.hub.load
    original_download = torch.hub.download_url_to_file
    old_dir = torch.hub.get_dir()
    old_hf = (hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE)
    hf_constants.HF_HUB_OFFLINE = True
    hf_constants.HF_HUB_CACHE = str(input_root / "cache/huggingface/hub")
    old_env = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "HF_HUB_CACHE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = hf_constants.HF_HUB_CACHE
    torch.hub.set_dir(str(input_root / "cache/torch/hub"))

    def load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != "gmberton/MegaLoc":
            raise RuntimeError(f"Unpinned Torch Hub source: {repo_or_dir}")
        kwargs.pop("trust_repo", None)
        kwargs["source"] = "local"
        return original_load(
            str(input_root / "cache/torch/hub/gmberton_MegaLoc_main"),
            model,
            *args,
            **kwargs,
        )

    def download(*args, **kwargs):
        raise RuntimeError(
            "Remote HLOC weights must be present in the hashed input bundle"
        )

    torch.hub.load = load
    torch.hub.download_url_to_file = download
    try:
        yield
    finally:
        torch.hub.load = original_load
        torch.hub.download_url_to_file = original_download
        torch.hub.set_dir(old_dir)
        hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE = old_hf
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_remote_hloc(
    image_dir: Path,
    output_dir: Path,
    options: dict[str, Any],
    image_list: list[str],
    *,
    input_root: Path,
) -> dict[str, str]:
    """Remote handler: unmodified extraction/retrieval/matching, no geometry."""
    import sys

    _validate_options(options)
    input_root = Path(input_root)
    sys.path.insert(0, str(input_root / "src"))
    from hloc import extract_features, match_features, pairs_from_retrieval

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(options)
    with _offline_hub(input_root):
        extract_features.main(
            deepcopy(options["feature_config"]),
            image_dir,
            image_list=image_list,
            feature_path=output_dir / names["features"],
        )
        extract_features.main(
            deepcopy(options["retrieval_config"]),
            image_dir,
            image_list=image_list,
            feature_path=output_dir / names["retrieval"],
        )
        pairs_from_retrieval.main(
            output_dir / names["retrieval"],
            output_dir / names["pairs"],
            num_matched=options["num_matched"],
        )
        match_features.main(
            deepcopy(options["matcher_config"]),
            output_dir / names["pairs"],
            output_dir / names["features"],
            matches=output_dir / names["matches"],
        )
    return names


def validate_result(
    result_dir: Path, images: Mapping[str, Path], options: Mapping[str, Any]
) -> dict[str, Path]:
    """Reject truncated/mismatched HDF5 before importing anything into COLMAP."""
    import h5py
    import numpy as np
    from PIL import Image

    paths = {k: Path(result_dir) / v for k, v in output_names(options).items()}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Incomplete Colab HLOC result")
    counts = {}
    with (
        h5py.File(paths["features"], "r") as features,
        h5py.File(paths["retrieval"], "r") as retrieval,
    ):
        for name, image in images.items():
            feature = features[name]
            keypoints, descriptors = feature["keypoints"][:], feature["descriptors"][:]
            count = len(keypoints)
            if (
                keypoints.shape != (count, 2)
                or descriptors.ndim != 2
                or descriptors.shape[1] != count
            ):
                raise ValueError(f"Invalid HLOC feature shapes for {name}")
            if not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all():
                raise ValueError(f"Nonfinite HLOC features for {name}")
            with Image.open(image) as source:
                size = source.size
            if tuple(feature["image_size"][:]) != size:
                raise ValueError(f"HLOC image-size mismatch for {name}")
            descriptor = retrieval[name]["global_descriptor"][:]
            if (
                descriptor.ndim != 1
                or descriptor.size == 0
                or not np.isfinite(descriptor).all()
            ):
                raise ValueError(f"Invalid HLOC retrieval descriptor for {name}")
            counts[name] = count
    pairs = [
        line.split() for line in paths["pairs"].read_text().splitlines() if line.strip()
    ]
    if not pairs:
        raise ValueError("Colab HLOC produced no image pairs")
    with h5py.File(paths["matches"], "r") as matches:
        for pair in pairs:
            if (
                len(pair) != 2
                or any(name not in images for name in pair)
                or pair[0] == pair[1]
            ):
                raise ValueError("Invalid HLOC retrieval pair")
            a, b = pair
            key = "/".join(name.replace("/", "-") for name in (a, b))
            if key not in matches:
                a, b = b, a
                key = "/".join(name.replace("/", "-") for name in (a, b))
            group = matches[key]
            values, scores = group["matches0"][:], group["matching_scores0"][:]
            if (
                values.shape != (counts[a],)
                or scores.shape != values.shape
                or not np.issubdtype(values.dtype, np.integer)
                or (values < -1).any()
                or (values >= counts[b]).any()
                or not np.isfinite(scores).all()
            ):
                raise ValueError(f"Invalid HLOC matches for {a}, {b}")
    return paths


def run_colab_features(
    images_dir: Path,
    output_dir: Path,
    feature_conf: str,
    retrieval_conf: str,
    matcher_conf: str,
    num_matched: int,
    image_list: list[str] | None = None,
    profile: str = PROFILE,
) -> dict[str, Path]:
    try:
        from remap.colab.model_backend import ColabModelBackend
    except ModuleNotFoundError as exc:
        if exc.name not in {"remap", "remap.colab", "remap.colab.model_backend"}:
            raise
        raise RuntimeError(
            "The Colab backend requires a separately installed, compatible remap "
            "Colab adapter. Use --inference-backend torch for the standalone "
            "package, or see docs/advanced.md for host integration."
        ) from exc

    options = resolve_options(feature_conf, retrieval_conf, matcher_conf, num_matched)
    images = image_inputs(images_dir, image_list)
    inputs = bundle_hloc_sources(options)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ColabModelBackend(profile=profile, hloc_handler=Path(__file__)) as backend:
        result_dir = backend.run("hloc", images=images, options=options, inputs=inputs)
        paths = validate_result(result_dir, images, options)
        manifests = {
            name: Path(result_dir).parent / name
            for name in ("committed.json", "request.json")
        }
        if any(not path.is_file() for path in manifests.values()):
            raise ValueError("Colab HLOC result is missing committed provenance")
        for path in paths.values():
            target = output_dir / path.name
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(path, temporary)
            temporary.replace(target)
        for name, path in manifests.items():
            target = output_dir / f"colab_hloc_{name}"
            temporary = target.with_suffix(".json.partial")
            shutil.copyfile(path, temporary)
            temporary.replace(target)
        (output_dir / "colab_hloc_provenance.json").write_text(
            json.dumps(
                {
                    "execution_backend": "colab",
                    "execution_profile": profile,
                    "options": options,
                    "images": list(images),
                    "local_geometry_device": "cpu",
                    "request_manifest": "colab_hloc_request.json",
                    "commit_manifest": "colab_hloc_committed.json",
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
    return {key: output_dir / path.name for key, path in paths.items()}
