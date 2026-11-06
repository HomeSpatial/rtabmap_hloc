from pathlib import Path

from PIL import Image
import pytest

from rtabmap_hloc_reprocess._exchange.verify import resolve_asset
from rtabmap_hloc_reprocess.exported_data_reader import ExportedDataReader
from rtabmap_hloc_reprocess.pipeline import PipelineConfig, RTABMapHLOCPipeline


def exported_fixture(root):
    (root / "rgb").mkdir(parents=True)
    (root / "calib").mkdir()
    (root / "poses.txt").write_text("0 0 0 0 0 0 0 1 1\n")
    Image.new("RGB", (4, 3), color="red").save(root / "rgb/1.jpg")
    return root


def test_exported_photos_are_portable_and_refresh_without_changing_source(tmp_path):
    source = exported_fixture(tmp_path / "source")
    output = tmp_path / "output"
    original = (source / "rgb/1.jpg").read_bytes()
    reader = ExportedDataReader(str(source))
    assert reader.export_images(output) == {1: "000001.jpg"}
    copied = resolve_asset({"root": "local", "path": "000001.jpg"}, {"local": output})
    assert not copied.is_symlink()
    assert copied.read_bytes() == original
    copied.write_bytes(b"stale unpublished output")
    reader.export_images(output)
    assert copied.read_bytes() == original
    assert (source / "rgb/1.jpg").read_bytes() == original


def test_old_export_symlink_is_replaced_without_modifying_source(tmp_path):
    source = exported_fixture(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    image = source / "rgb/1.jpg"
    target = output / "000001.jpg"
    try:
        target.symlink_to(image)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this platform")
    original = image.read_bytes()
    ExportedDataReader(str(source)).export_images(output)
    assert not target.is_symlink()
    assert image.read_bytes() == target.read_bytes() == original


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".JPG", ".png"])
def test_resume_recovers_all_supported_image_formats(tmp_path, suffix):
    source = exported_fixture(tmp_path / "source")
    output = tmp_path / "output"
    (output / "images").mkdir(parents=True)
    Image.new("RGB", (4, 3)).save(output / "images" / ("000001" + suffix))
    pipeline = RTABMapHLOCPipeline(PipelineConfig(
        rtabmap_db=str(source), output_dir=str(output), skip_image_export=True,
    ))
    pipeline._export_images()
    assert pipeline.image_to_node_id == {"000001" + suffix: 1}
