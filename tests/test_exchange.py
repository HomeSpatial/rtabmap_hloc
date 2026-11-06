import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rtabmap_hloc_reprocess.pipeline import PipelineConfig, PipelineResult, RTABMapHLOCPipeline
from rtabmap_hloc_reprocess.exchange import export_description
from rtabmap_hloc_reprocess._exchange.verify import check


def native_fixture(root):
    """Real COLMAP binaries and photos; includes an exported, unregistered photo."""
    import pycolmap
    sparse = root / "sfm_aligned"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(struct.pack("<QiiQQdddd", 1, 1, 1, 4, 3, 2, 2, 2, 1.5))
    (sparse / "images.bin").write_bytes(struct.pack("<Qidddddddi", 1, 950, 1, 0, 0, 0, 0, 0, 0, 1)
                                        + b"nested/photo.jpg\0" + struct.pack("<Q", 0))
    (sparse / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    (root / "images/nested").mkdir(parents=True)
    Image.new("RGB", (4, 3)).save(root / "images/nested/photo.jpg")
    Image.new("RGB", (4, 3)).save(root / "images/not-registered.jpg")
    return pycolmap.Reconstruction(sparse)


def pipeline_fixture(tmp_path, monkeypatch, *, existing=False):
    model = native_fixture(tmp_path)
    pipeline = RTABMapHLOCPipeline(PipelineConfig(output_dir=str(tmp_path), use_existing_poses=existing, scene_id="house"))
    pipeline.image_to_node_id = {"nested/photo.jpg": 950, "not-registered.jpg": 951}
    monkeypatch.setattr(pipeline, "_export_images", lambda: None)
    monkeypatch.setattr(pipeline, "_run_hloc_reconstruction", lambda: model)
    monkeypatch.setattr(pipeline, "_run_hloc_triangulation", lambda: model)
    monkeypatch.setattr(pipeline, "_export_metadata", lambda result: (tmp_path / "metadata.json").write_text("{}"))
    return pipeline


def estimators(monkeypatch, robust, fallback=None):
    import rtabmap_hloc_reprocess.sim3_estimation as sim
    import rtabmap_hloc_reprocess.hloc_pipeline as hloc
    monkeypatch.setattr(sim, "estimate_sim3_robust", lambda *a, **k: robust)
    monkeypatch.setattr(sim, "estimate_sim3_from_poses", lambda *a, **k: fallback)
    monkeypatch.setattr(hloc, "get_reconstruction_poses", lambda *a: {})
    monkeypatch.setattr(sim, "compute_alignment_error", lambda *a: (0., 0., []))
    monkeypatch.setattr(sim, "compute_rotation_error", lambda *a: (0., 0., []))
    monkeypatch.setattr(sim, "transform_reconstruction", lambda *a: None)


def test_both_estimators_fail_pipeline_and_no_metric_description(tmp_path, monkeypatch):
    p = pipeline_fixture(tmp_path, monkeypatch)
    estimators(monkeypatch, (None, None, None, []), (None, None, None, []))
    result = p.run()
    assert not result.success and "both RANSAC" in result.error
    assert not (tmp_path / "reconstruction.json").exists()
    status = json.loads((tmp_path / "run-status.json").read_text())
    assert status["execution"] == "failed" and status["alignment"] == "failed"


@pytest.mark.parametrize("scale,rotation,translation", [
    (float("nan"), np.eye(3), np.zeros(3)), (0, np.eye(3), np.zeros(3)),
    (-1, np.eye(3), np.zeros(3)), (1, np.ones((3, 3)), np.zeros(3)),
    (1, np.diag([-1, 1, 1]), np.zeros(3)), (1, np.eye(3), [0, 0, float("inf")]),
])
def test_invalid_sim3_is_failure(tmp_path, monkeypatch, scale, rotation, translation):
    p = pipeline_fixture(tmp_path, monkeypatch)
    estimators(monkeypatch, (scale, rotation, translation, [1, 2, 3]))
    assert not p.run().success
    assert not (tmp_path / "reconstruction.json").exists()


@pytest.mark.parametrize("existing,fallback", [(True, False), (False, False), (False, True)])
def test_success_publishes_registered_subset_and_aligned_path(tmp_path, monkeypatch, existing, fallback):
    p = pipeline_fixture(tmp_path, monkeypatch, existing=existing)
    good = (1., np.eye(3), np.zeros(3), [1, 2, 3])
    estimators(monkeypatch, (None, None, None, []) if fallback else good, good)
    result = p.run()
    assert result.success and result.sfm_dir == str(tmp_path / "sfm_aligned")
    doc = json.loads(Path(result.description_path).read_text())
    assert doc["scene_id"] == "house"
    assert len(doc["assets"][1]["files"]) == 1
    assert doc["assets"][1]["files"][0]["ref"]["path"] == "images/nested/photo.jpg"
    assert doc["extensions"]["org.rtabmap_hloc.provenance"]["frame_ids"] == ["950"]
    check(Path(result.description_path))
    assert json.loads((tmp_path / "run-status.json").read_text())["quality"]["status"] == "not_evaluated"
    if fallback:
        assert result.alignment_method == "least_squares_sim3_fallback"
    before = {str(f.relative_to(tmp_path)): f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()}
    assert not p.run().success
    assert before == {str(f.relative_to(tmp_path)): f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()}


def test_missing_photo_cannot_publish_success(tmp_path, monkeypatch):
    p = pipeline_fixture(tmp_path, monkeypatch, existing=True)
    (tmp_path / "images/nested/photo.jpg").unlink()
    result = p.run()
    assert not result.success and "Missing output asset" in result.error
    assert not (tmp_path / "reconstruction.json").exists()


def test_metadata_failure_resets_success(tmp_path, monkeypatch):
    p = pipeline_fixture(tmp_path, monkeypatch, existing=True)
    def fail(result):
        raise OSError("disk failure")
    monkeypatch.setattr(p, "_export_metadata", fail)
    result = p.run()
    assert not result.success and result.error == "disk failure"
    assert not (tmp_path / "reconstruction.json").exists()


def test_cli_returns_nonzero_on_failure(tmp_path, monkeypatch):
    from rtabmap_hloc_reprocess.cli import main
    db = tmp_path / "source.db"
    db.touch()
    monkeypatch.setattr("sys.argv", ["rtabmap-hloc", str(db), "-o", str(tmp_path / "out")])
    monkeypatch.setattr(RTABMapHLOCPipeline, "run", lambda self: PipelineResult(error="alignment failed"))
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1


def test_cli_bundle_publishes_and_duplicate_never_runs(tmp_path, monkeypatch):
    from rtabmap_hloc_reprocess.cli import main
    import rtabmap_hloc_reprocess.pipeline as module
    from rtabmap_hloc_reprocess._exchange.verify import check
    p = pipeline_fixture(tmp_path / "work", monkeypatch, existing=True)
    db = tmp_path / "source.db"; db.touch()
    bundle = tmp_path / "scene/bundle.json"
    monkeypatch.setattr(module, "RTABMapHLOCPipeline", lambda config: p)
    monkeypatch.setattr("sys.argv", ["rtabmap-hloc", str(db), "--bundle", str(bundle), "--scene-id", "house"])
    with pytest.raises(SystemExit) as code:
        main()
    assert code.value.code == 0
    assert check(bundle)["lock_validation"] == "passed"
    before = bundle.read_bytes()
    monkeypatch.setattr(p, "run", lambda: pytest.fail("Duplicate stage must fail before inference"))
    with pytest.raises(SystemExit) as code:
        main()
    assert code.value.code != 0 and bundle.read_bytes() == before
