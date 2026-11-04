from pathlib import Path

import pytest

from rtabmap_hloc_reprocess.colmap_profiles import (
    apply_fixed_intrinsics_options,
    available_colmap_profiles,
    json_safe_colmap_options,
    resolve_colmap_profile,
)


def test_default_colmap_profile_preserves_no_option_overrides():
    assert "default" in available_colmap_profiles()
    assert resolve_colmap_profile("default") == {}


def test_high_accuracy_colmap_profile_uses_pycolmap_option_shape():
    options = resolve_colmap_profile("high_accuracy")

    assert options["min_num_matches"] == 30
    assert options["ba_global_max_num_iterations"] == 150
    assert options["mapper"]["init_min_num_inliers"] == 200
    assert options["mapper"]["abs_pose_max_error"] == pytest.approx(5.0)
    assert options["triangulation"]["min_angle"] == pytest.approx(2.0)
    assert "BundleAdjustmentCeres.max_num_iterations" not in options
    assert "TwoViewGeometry.max_error" not in options


def test_unknown_colmap_profile_lists_valid_names():
    with pytest.raises(ValueError, match="Valid: .*default.*high_accuracy"):
        resolve_colmap_profile("extreme")


def test_fixed_intrinsics_options_are_json_safe():
    options = apply_fixed_intrinsics_options(
        {
            "ba_refine_focal_length": True,
            "ba_refine_extra_params": True,
            "mapper": {"abs_pose_refine_focal_length": True},
            "snapshot_path": Path("."),
        },
        camera_ids={7, 3},
    )

    safe = json_safe_colmap_options(options)

    assert safe["constant_cameras"] == [3, 7]
    assert safe["mapper"]["constant_cameras"] == [3, 7]
    assert safe["ba_refine_focal_length"] is False
    assert safe["ba_refine_extra_params"] is False
    assert safe["ba_refine_principal_point"] is False
    assert safe["mapper"]["abs_pose_refine_focal_length"] is False
    assert safe["mapper"]["abs_pose_refine_extra_params"] is False
    assert safe["snapshot_path"] == "."
