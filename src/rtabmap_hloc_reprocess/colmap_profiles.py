"""Named COLMAP option profiles for HLOC reconstruction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


DEFAULT_COLMAP_PROFILE = "default"


_PROFILE_OPTIONS: dict[str, dict[str, Any]] = {
    DEFAULT_COLMAP_PROFILE: {},
    "high_accuracy": {
        "min_num_matches": 30,
        "ba_refine_focal_length": True,
        "ba_refine_principal_point": False,
        "ba_refine_extra_params": True,
        "ba_local_max_num_iterations": 50,
        "ba_local_max_refinements": 3,
        "ba_global_frames_ratio": 1.05,
        "ba_global_points_ratio": 1.05,
        "ba_global_frames_freq": 200,
        "ba_global_points_freq": 100000,
        "ba_global_max_num_iterations": 150,
        "ba_global_max_refinements": 10,
        "ba_global_max_refinement_change": 1e-6,
        "mapper": {
            "init_min_num_inliers": 200,
            "init_max_error": 3.0,
            "init_min_tri_angle": 8.0,
            "init_max_forward_motion": 0.95,
            "abs_pose_max_error": 5.0,
            "abs_pose_min_num_inliers": 80,
            "abs_pose_min_inlier_ratio": 0.35,
            "filter_max_reproj_error": 2.5,
            "filter_min_tri_angle": 2.0,
            "ba_local_num_images": 12,
        },
        "triangulation": {
            "min_angle": 2.0,
            "merge_max_reproj_error": 2.5,
            "complete_max_reproj_error": 2.5,
            "ignore_two_view_tracks": True,
        },
    },
}


def available_colmap_profiles() -> tuple[str, ...]:
    """Return registered COLMAP profile names."""
    return tuple(sorted(_PROFILE_OPTIONS))


def normalize_colmap_profile_name(profile_name: str | None) -> str:
    """Normalize and validate a COLMAP profile name."""
    profile = (profile_name or DEFAULT_COLMAP_PROFILE).strip().lower()
    if profile not in _PROFILE_OPTIONS:
        valid = ", ".join(available_colmap_profiles())
        raise ValueError(f"Unknown COLMAP profile {profile_name!r}. Valid: {valid}")
    return profile


def resolve_colmap_profile(profile_name: str | None) -> dict[str, Any]:
    """Resolve a named profile into pycolmap-compatible option overrides."""
    profile = normalize_colmap_profile_name(profile_name)
    options = deepcopy(_PROFILE_OPTIONS[profile])
    _validate_option_keys(options, _incremental_pipeline_option_shape(), profile)
    return options


def apply_fixed_intrinsics_options(
    options: Mapping[str, Any],
    camera_ids: Any,
) -> dict[str, Any]:
    """Layer fixed-camera intrinsics policy over resolved COLMAP options."""
    resolved = deepcopy(dict(options))
    camera_set = {int(camera_id) for camera_id in camera_ids}

    resolved["constant_cameras"] = set(camera_set)
    mapper = resolved.setdefault("mapper", {})
    mapper["constant_cameras"] = set(camera_set)

    resolved["ba_refine_focal_length"] = False
    resolved["ba_refine_principal_point"] = False
    resolved["ba_refine_extra_params"] = False
    mapper["abs_pose_refine_focal_length"] = False
    mapper["abs_pose_refine_extra_params"] = False

    _validate_option_keys(
        resolved,
        _incremental_pipeline_option_shape(),
        "fixed_intrinsics",
    )
    return resolved


def json_safe_colmap_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of COLMAP options."""
    return _json_safe(dict(options))


def _incremental_pipeline_option_shape() -> dict[str, Any]:
    import pycolmap

    return pycolmap.IncrementalPipelineOptions().todict()


def _validate_option_keys(
    options: Mapping[str, Any],
    shape: Mapping[str, Any],
    profile_name: str,
    path: str = "",
) -> None:
    for key, value in options.items():
        if key not in shape:
            full_key = f"{path}.{key}" if path else key
            raise ValueError(
                f"COLMAP profile {profile_name!r} uses unsupported option {full_key!r}"
            )
        shape_value = shape[key]
        if isinstance(value, Mapping):
            if not isinstance(shape_value, Mapping):
                full_key = f"{path}.{key}" if path else key
                raise ValueError(
                    f"COLMAP profile {profile_name!r} uses nested option "
                    f"{full_key!r}, but pycolmap expects a scalar"
                )
            next_path = f"{path}.{key}" if path else key
            _validate_option_keys(value, shape_value, profile_name, next_path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_safe(v) for v in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and value.__class__.__module__.startswith("pycolmap"):
        return value.name
    return value
