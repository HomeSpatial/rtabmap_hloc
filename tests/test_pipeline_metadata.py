import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from rtabmap_hloc_reprocess.pipeline import (
    PipelineConfig,
    PipelineResult,
    RTABMapHLOCPipeline,
)


class FakeReconstruction:
    cameras = {7: object()}

    def num_reg_images(self):
        return 1

    def num_points3D(self):
        return 2

    def compute_mean_reprojection_error(self):
        return 1.25

    def compute_mean_track_length(self):
        return 6.5

    def compute_mean_observations_per_reg_image(self):
        return 508.75

    def compute_num_observations(self):
        return 2035


def test_export_metadata_includes_final_ba_metrics(tmp_path: Path):
    db_path = tmp_path / "source.db"
    db_path.write_bytes(b"fake db")

    output_dir = tmp_path / "hloc_output"
    output_dir.mkdir()

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(
            rtabmap_db=str(db_path),
            output_dir=str(output_dir),
            feature_conf="aliked-n16",
            retrieval_conf="megaloc",
            matcher_conf="aliked+lightglue",
        )
    )
    result = PipelineResult(
        success=True,
        reconstruction=FakeReconstruction(),
        num_images_exported=5,
        num_images_registered=4,
        num_points3d=123,
        scale=1.0,
    )

    pipeline._export_metadata(result)

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert isinstance(metadata["ba_metrics"], dict)
    assert metadata["ba_metrics"]["mean_reprojection_error_px"] == pytest.approx(
        1.25
    )
    assert metadata["ba_metrics"]["mean_track_length"] == pytest.approx(6.5)
    assert metadata["ba_metrics"]["mean_observations_per_registered_image"] == (
        pytest.approx(508.75)
    )
    assert metadata["ba_metrics"]["num_observations"] == 2035
    assert metadata["ba_metrics"]["registered_images"] == 4
    assert metadata["ba_metrics"]["total_images"] == 5
    assert metadata["ba_metrics"]["points3d"] == 123


def test_export_metadata_records_full_sfm_camera_intrinsics_policy(tmp_path: Path):
    db_path = tmp_path / "source.db"
    db_path.write_bytes(b"fake db")

    output_dir = tmp_path / "hloc_output"
    output_dir.mkdir()

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(rtabmap_db=str(db_path), output_dir=str(output_dir))
    )
    result = PipelineResult(
        success=True,
        reconstruction=FakeReconstruction(),
        num_images_exported=5,
        num_images_registered=4,
        num_points3d=123,
        scale=1.0,
    )

    pipeline._export_metadata(result)

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["use_existing_poses"] is False
    assert metadata["camera_mode"] == "RTABMAP_INTRINSICS"
    assert metadata["colmap_profile"] == "default"
    assert metadata["colmap_options"]["constant_cameras"] == [7]
    assert metadata["colmap_options"]["mapper"]["constant_cameras"] == [7]
    assert metadata["colmap_options"]["ba_refine_focal_length"] is False
    assert metadata["colmap_options"]["ba_refine_extra_params"] is False
    assert metadata["colmap_options"]["ba_refine_principal_point"] is False
    assert metadata["camera_intrinsics"] == {
        "reconstruction_mode": "full_sfm",
        "camera_mode": "RTABMAP_INTRINSICS",
        "source": "rtabmap_database_calibration",
        "colmap_camera_model": "PINHOLE",
        "rtabmap_intrinsics_used": True,
        "rtabmap_poses_used": False,
        "intrinsics_refinement": {
            "refine_intrinsics": False,
            "refine_focal_length": False,
            "refine_extra_params": False,
            "refine_principal_point": False,
        },
    }


def test_export_metadata_records_existing_pose_camera_intrinsics_policy(
    tmp_path: Path,
):
    db_path = tmp_path / "source.db"
    db_path.write_bytes(b"fake db")

    output_dir = tmp_path / "hloc_output"
    output_dir.mkdir()

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(
            rtabmap_db=str(db_path),
            output_dir=str(output_dir),
            use_existing_poses=True,
        )
    )
    result = PipelineResult(
        success=True,
        reconstruction=FakeReconstruction(),
        num_images_exported=5,
        num_images_registered=4,
        num_points3d=123,
        scale=1.0,
    )

    pipeline._export_metadata(result)

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["use_existing_poses"] is True
    assert metadata["camera_mode"] == "REFERENCE_MODEL"
    assert metadata["colmap_profile"] == "default"
    assert metadata["colmap_options"]["constant_cameras"] == [7]
    assert metadata["colmap_options"]["mapper"]["constant_cameras"] == [7]
    assert metadata["colmap_options"]["ba_refine_focal_length"] is False
    assert metadata["colmap_options"]["ba_refine_extra_params"] is False
    assert metadata["colmap_options"]["ba_refine_principal_point"] is False
    assert metadata["camera_intrinsics"] == {
        "reconstruction_mode": "rtabmap_pose_triangulation",
        "camera_mode": "REFERENCE_MODEL",
        "source": "rtabmap_database_calibration",
        "colmap_camera_model": "PINHOLE",
        "rtabmap_intrinsics_used": True,
        "rtabmap_poses_used": True,
        "intrinsics_refinement": {
            "refine_intrinsics": False,
            "refine_focal_length": False,
            "refine_extra_params": False,
            "refine_principal_point": False,
        },
    }


def test_export_metadata_records_high_accuracy_colmap_profile(tmp_path: Path):
    db_path = tmp_path / "source.db"
    db_path.write_bytes(b"fake db")

    output_dir = tmp_path / "hloc_output"
    output_dir.mkdir()

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(
            rtabmap_db=str(db_path),
            output_dir=str(output_dir),
            colmap_profile="high_accuracy",
        )
    )
    result = PipelineResult(
        success=True,
        reconstruction=FakeReconstruction(),
        num_images_exported=5,
        num_images_registered=4,
        num_points3d=123,
        scale=1.0,
    )

    pipeline._export_metadata(result)

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["colmap_profile"] == "high_accuracy"
    assert metadata["colmap_options"]["min_num_matches"] == 30
    assert metadata["colmap_options"]["mapper"]["init_min_num_inliers"] == 200
    assert metadata["colmap_options"]["triangulation"]["min_angle"] == pytest.approx(
        2.0
    )
    assert metadata["colmap_options"]["constant_cameras"] == [7]
    assert metadata["colmap_options"]["ba_refine_focal_length"] is False


def test_export_metadata_layers_fixed_intrinsics_on_existing_pose_profile(
    tmp_path: Path,
):
    db_path = tmp_path / "source.db"
    db_path.write_bytes(b"fake db")

    output_dir = tmp_path / "hloc_output"
    output_dir.mkdir()

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(
            rtabmap_db=str(db_path),
            output_dir=str(output_dir),
            use_existing_poses=True,
            colmap_profile="high_accuracy",
        )
    )
    result = PipelineResult(
        success=True,
        reconstruction=FakeReconstruction(),
        num_images_exported=5,
        num_images_registered=4,
        num_points3d=123,
        scale=1.0,
    )

    pipeline._export_metadata(result)

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["camera_mode"] == "REFERENCE_MODEL"
    assert metadata["colmap_profile"] == "high_accuracy"
    assert metadata["colmap_options"]["min_num_matches"] == 30
    assert metadata["colmap_options"]["constant_cameras"] == [7]
    assert metadata["colmap_options"]["mapper"]["constant_cameras"] == [7]
    assert metadata["colmap_options"]["ba_refine_focal_length"] is False
    assert metadata["colmap_options"]["ba_refine_extra_params"] is False
    assert metadata["colmap_options"]["mapper"]["abs_pose_refine_focal_length"] is False
    assert metadata["colmap_options"]["mapper"]["abs_pose_refine_extra_params"] is False


def test_run_hloc_reconstruction_passes_rtabmap_intrinsics_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from rtabmap_hloc_reprocess import hloc_pipeline

    captured = {}

    def fake_run_hloc_reconstruction(**kwargs):
        captured.update(kwargs)
        return FakeReconstruction()

    monkeypatch.setattr(
        hloc_pipeline, "run_hloc_reconstruction", fake_run_hloc_reconstruction
    )

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(output_dir=str(tmp_path), camera_mode="PER_IMAGE")
    )
    pipeline.image_to_node_id = {"000001.png": 1}
    pipeline.rtabmap_camera_models = {
        "000001.png": SimpleNamespace(
            fx=500.0,
            fy=501.0,
            cx=320.0,
            cy=241.0,
            width=640,
            height=480,
        )
    }

    assert pipeline._run_hloc_reconstruction() is not None
    assert captured["camera_mode"] == "PER_IMAGE"
    assert captured["rtabmap_intrinsics"] == pipeline.rtabmap_camera_models
    assert captured["fix_intrinsics"] is True
    assert captured["colmap_options"]["constant_cameras"] == set()
    assert captured["colmap_options"]["ba_refine_focal_length"] is False
    assert captured["colmap_options"]["ba_refine_extra_params"] is False


def test_run_hloc_reconstruction_passes_high_accuracy_profile_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from rtabmap_hloc_reprocess import hloc_pipeline

    captured = {}

    def fake_run_hloc_reconstruction(**kwargs):
        captured.update(kwargs)
        return FakeReconstruction()

    monkeypatch.setattr(
        hloc_pipeline, "run_hloc_reconstruction", fake_run_hloc_reconstruction
    )

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(output_dir=str(tmp_path), colmap_profile="high_accuracy")
    )
    pipeline.image_to_node_id = {"000001.png": 1}
    pipeline.rtabmap_camera_models = {
        "000001.png": SimpleNamespace(
            fx=500.0,
            fy=501.0,
            cx=320.0,
            cy=241.0,
            width=640,
            height=480,
        )
    }

    assert pipeline._run_hloc_reconstruction() is not None
    assert captured["colmap_options"]["min_num_matches"] == 30
    assert captured["colmap_options"]["mapper"]["init_min_num_inliers"] == 200
    assert captured["colmap_options"]["triangulation"]["min_angle"] == pytest.approx(
        2.0
    )
    assert captured["colmap_options"]["ba_refine_focal_length"] is False


def test_run_hloc_reconstruction_can_disable_rtabmap_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from rtabmap_hloc_reprocess import hloc_pipeline

    captured = {}

    def fake_run_hloc_reconstruction(**kwargs):
        captured.update(kwargs)
        return FakeReconstruction()

    monkeypatch.setattr(
        hloc_pipeline, "run_hloc_reconstruction", fake_run_hloc_reconstruction
    )

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(
            output_dir=str(tmp_path),
            camera_mode="PER_IMAGE",
            use_rtabmap_intrinsics=False,
        )
    )
    pipeline.image_to_node_id = {"000001.png": 1}
    pipeline.rtabmap_camera_models = {
        "000001.png": SimpleNamespace(
            fx=500.0,
            fy=501.0,
            cx=320.0,
            cy=241.0,
            width=640,
            height=480,
        )
    }

    assert pipeline._run_hloc_reconstruction() is not None
    assert captured["rtabmap_intrinsics"] is None
    assert captured["fix_intrinsics"] is False
    assert captured["colmap_options"] == {}


def test_run_hloc_triangulation_passes_high_accuracy_profile_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from rtabmap_hloc_reprocess import hloc_pipeline

    captured = {}

    def fake_run_hloc_triangulation(**kwargs):
        captured.update(kwargs)
        return FakeReconstruction()

    monkeypatch.setattr(
        hloc_pipeline, "run_hloc_triangulation", fake_run_hloc_triangulation
    )

    pipeline = RTABMapHLOCPipeline(
        PipelineConfig(
            output_dir=str(tmp_path),
            use_existing_poses=True,
            colmap_profile="high_accuracy",
        )
    )
    pipeline.image_to_node_id = {"000001.png": 1}
    pipeline._camera_ids_from_reference_model = lambda: {7}

    assert pipeline._run_hloc_triangulation() is not None
    assert captured["colmap_options"]["min_num_matches"] == 30
    assert captured["colmap_options"]["mapper"]["abs_pose_min_num_inliers"] == 80
    assert captured["colmap_options"]["ba_refine_focal_length"] is False
    assert captured["colmap_options"]["ba_refine_extra_params"] is False
    assert captured["colmap_options"]["constant_cameras"] == {7}
    assert captured["colmap_options"]["mapper"]["constant_cameras"] == {7}


def test_import_images_with_rtabmap_intrinsics_writes_pinhole_cameras(
    tmp_path: Path,
):
    from PIL import Image
    import pycolmap

    from rtabmap_hloc_reprocess.hloc_pipeline import (
        import_images_with_rtabmap_intrinsics,
    )

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (640, 480), (0, 0, 0)).save(image_dir / "000001.png")
    database_path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path):
        pass

    image_ids, camera_ids = import_images_with_rtabmap_intrinsics(
        image_dir=image_dir,
        database_path=database_path,
        image_list=["000001.png"],
        rtabmap_intrinsics={
            "000001.png": SimpleNamespace(
                fx=500.0,
                fy=501.0,
                cx=320.0,
                cy=241.0,
                width=640,
                height=480,
            )
        },
    )

    assert image_ids == {"000001.png": 1}
    assert camera_ids == {1}
    with pycolmap.Database.open(database_path) as db:
        image = db.read_image(1)
        camera = db.read_camera(image.camera_id)
    assert camera.model.name == "PINHOLE"
    assert camera.has_prior_focal_length is True
    assert list(camera.params) == pytest.approx([500.0, 501.0, 320.0, 241.0])


def test_run_hloc_reconstruction_creates_sfm_dir_for_rtabmap_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from rtabmap_hloc_reprocess import hloc_pipeline

    captured = {}

    class FakeHlocReconstruction:
        @staticmethod
        def create_empty_db(database_path: Path):
            assert database_path.parent.exists()
            database_path.touch()

        @staticmethod
        def import_features(image_ids, db, features_path):
            captured["image_ids"] = image_ids

        @staticmethod
        def import_matches(image_ids, db, pairs_path, matches_path):
            captured["matches_image_ids"] = image_ids

        @staticmethod
        def estimation_and_geometric_verification(database_path, pairs_path, verbose):
            captured["verification_database"] = database_path

        @staticmethod
        def run_reconstruction(sfm_dir, database_path, image_dir, verbose, options):
            captured["mapper_options"] = options
            return FakeReconstruction()

    fake_hloc = SimpleNamespace(
        extract_features=SimpleNamespace(
            confs={"aliked-n16": {}, "megaloc": {}},
            main=lambda conf, images_dir, image_list, feature_path: feature_path,
        ),
        pairs_from_retrieval=SimpleNamespace(main=lambda *args, **kwargs: None),
        match_features=SimpleNamespace(
            confs={"aliked+lightglue": {}},
            main=lambda conf, pairs_path, features_path, matches: matches,
        ),
        reconstruction=FakeHlocReconstruction,
    )
    monkeypatch.setitem(sys.modules, "hloc", fake_hloc)
    monkeypatch.setattr(
        hloc_pipeline,
        "import_images_with_rtabmap_intrinsics",
        lambda *args, **kwargs: ({"000001.png": 1}, {7}),
    )

    result = hloc_pipeline.run_hloc_reconstruction(
        images_dir=tmp_path / "images",
        output_dir=tmp_path / "hloc",
        image_list=["000001.png"],
        rtabmap_intrinsics={"000001.png": object()},
        fix_intrinsics=True,
    )

    assert result is not None
    assert (tmp_path / "hloc" / "sfm").is_dir()
    assert captured["mapper_options"]["constant_cameras"] == {7}
    assert captured["mapper_options"]["mapper"]["constant_cameras"] == {7}
    assert captured["mapper_options"]["ba_refine_focal_length"] is False
    assert captured["mapper_options"]["ba_refine_extra_params"] is False
    assert captured["mapper_options"]["ba_refine_principal_point"] is False


def test_run_hloc_reconstruction_passes_colmap_options_to_standard_sfm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from rtabmap_hloc_reprocess import hloc_pipeline

    captured = {}

    class FakeHlocReconstruction:
        @staticmethod
        def main(*args, **kwargs):
            captured.update(kwargs)
            return FakeReconstruction()

    fake_hloc = SimpleNamespace(
        extract_features=SimpleNamespace(
            confs={"aliked-n16": {}, "megaloc": {}},
            main=lambda conf, images_dir, image_list, feature_path: feature_path,
        ),
        pairs_from_retrieval=SimpleNamespace(main=lambda *args, **kwargs: None),
        match_features=SimpleNamespace(
            confs={"aliked+lightglue": {}},
            main=lambda conf, pairs_path, features_path, matches: matches,
        ),
        reconstruction=FakeHlocReconstruction,
    )
    monkeypatch.setitem(sys.modules, "hloc", fake_hloc)

    result = hloc_pipeline.run_hloc_reconstruction(
        images_dir=tmp_path / "images",
        output_dir=tmp_path / "hloc",
        image_list=["000001.png"],
        rtabmap_intrinsics=None,
        colmap_options={"min_num_matches": 30},
    )

    assert result is not None
    assert captured["mapper_options"] == {"min_num_matches": 30}
