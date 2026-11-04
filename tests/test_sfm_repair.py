"""Observable pose recovery, no-op behavior and preservation of input artifacts."""

import json
import subprocess
import sys

import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

from rtabmap_hloc_reprocess.sfm_repair import inspect_sfm, relocalize


def scene(tmp_path, *, frames=5):
    database = tmp_path / "database.db"
    options = pycolmap.SyntheticDatasetOptions(
        {
            "num_rigs": 1,
            "num_frames_per_rig": frames,
            "num_points3D": 80,
            "camera_model_id": pycolmap.CameraModelId.PINHOLE,
            "camera_params": [800.0, 800.0, 512.0, 384.0],
        }
    )
    pycolmap.set_random_seed(42)
    with pycolmap.Database.open(database) as db:
        model = pycolmap.synthesize_dataset(options, db)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model.write(model_dir)
    return model, model_dir, database, list(model.reg_image_ids())


def perturb(model, image_id):
    image = model.images[image_id]
    original = image.cam_from_world().matrix().copy()
    rotation = Rotation.from_euler("y", 30, degrees=True).as_matrix() @ original[:, :3]
    pose = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), original[:, 3] + [0.5, 0, 0])
    model.frames[image.frame_id].set_cam_from_world(image.camera_id, pose)
    return original


def test_recovers_pose_without_changing_inputs_or_other_cameras(tmp_path):
    model, model_dir, database, order = scene(tmp_path)
    target = order[0]
    truth = perturb(model, target)
    model.write(model_dir)
    inputs = {p: p.read_bytes() for p in [*model_dir.glob("*.bin"), database]}
    inspection = inspect_sfm(model_dir, database)
    report = relocalize(
        model_dir,
        database,
        [target],
        tmp_path / "candidate",
        image_order=order,
        expected_input_id=inspection["input_id"],
    )
    candidate = pycolmap.Reconstruction(report["candidate_model"])
    np.testing.assert_allclose(
        candidate.images[target].cam_from_world().matrix(), truth, atol=1e-5
    )
    assert candidate.num_reg_images() == model.num_reg_images()
    for i, image in model.images.items():
        if i != target:
            np.testing.assert_array_equal(
                candidate.images[i].cam_from_world().matrix(),
                image.cam_from_world().matrix(),
            )
    for i, camera in model.cameras.items():
        np.testing.assert_array_equal(candidate.cameras[i].params, camera.params)
    assert all(p.read_bytes() == data for p, data in inputs.items())
    assert report["after"]["components"] == report["before"]["components"] == 1
    assert json.loads((tmp_path / "candidate/result.json").read_text()) == report


def test_empty_request_preserves_healthy_model_and_cli_inspects_it(tmp_path):
    _, model_dir, database, order = scene(tmp_path)
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "rtabmap_hloc_reprocess.sfm_repair",
            "inspect",
            str(model_dir),
            str(database),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inspection = json.loads(cli.stdout)
    report = relocalize(
        model_dir,
        database,
        [],
        tmp_path / "no_change",
        image_order=order,
        expected_input_id=inspection["input_id"],
    )
    assert report["candidate_model"] is None
    assert report["before"] == report["after"] == inspection["metrics"]
    assert not (tmp_path / "no_change/sfm").exists()
    assert inspect_sfm(model_dir, database)["input_id"] == inspection["input_id"]


def test_insufficient_views_produces_diagnostics_without_candidate(tmp_path):
    _, model_dir, database, order = scene(tmp_path, frames=2)
    inspection = inspect_sfm(model_dir, database)
    report = relocalize(
        model_dir,
        database,
        [order[0]],
        tmp_path / "unsupported",
        image_order=order,
        expected_input_id=inspection["input_id"],
    )
    assert report["status"] == "insufficient_support"
    assert report["candidate_model"] is None
    assert report["before"] == report["after"]
    assert not (tmp_path / "unsupported/sfm").exists()
    assert inspect_sfm(model_dir, database)["input_id"] == inspection["input_id"]


def test_old_inspection_cannot_modify_a_new_model(tmp_path):
    model, model_dir, database, order = scene(tmp_path)
    old = inspect_sfm(model_dir, database)
    perturb(model, order[0])
    model.write(model_dir)
    with pytest.raises(ValueError, match="Input changed"):
        relocalize(
            model_dir,
            database,
            [order[0]],
            tmp_path / "stale",
            image_order=order,
            expected_input_id=old["input_id"],
        )
    assert not (tmp_path / "stale").exists()
