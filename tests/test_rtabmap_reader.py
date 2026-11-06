import struct
import sqlite3

import pytest

from rtabmap_hloc_reprocess.rtabmap_reader import parse_camera_model


@pytest.mark.parametrize("options", [{}, {"read_only": True}])
def test_read_only_database_rejects_writes(tmp_path, options):
    from rtabmap_hloc_reprocess.rtabmap_reader import RTABMapDatabase

    path = tmp_path / "readonly.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE Node (id INTEGER)")
    with RTABMapDatabase(str(path), **options) as reader:
        assert reader.get_node_ids() == []
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.conn.execute("INSERT INTO Node VALUES (1)")


def test_database_write_access_requires_explicit_opt_in(tmp_path):
    from rtabmap_hloc_reprocess.rtabmap_reader import RTABMapDatabase

    path = tmp_path / "writable.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE Node (id INTEGER)")
    with RTABMapDatabase(str(path), read_only=False) as reader:
        reader.conn.execute("INSERT INTO Node VALUES (7)")
        reader.conn.commit()
    with RTABMapDatabase(str(path)) as reader:
        assert reader.get_node_ids() == [7]


IDENTITY_LOCAL_TRANSFORM = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
)


def _arcore_kmatrix_calibration_blob() -> bytes:
    """Builds the Android ARCore K-matrix RTABMap layout seen in real DBs."""
    header = struct.pack(
        "I10i",
        0,  # empty camera name
        23,
        5,
        0,
        640,
        480,
        9,
        0,
        0,
        0,
        12,
    )
    k_matrix = struct.pack(
        "9d",
        453.03582763671875,
        0.0,
        319.2976379394531,
        0.0,
        455.1301574707031,
        242.1616668701172,
        0.0,
        0.0,
        1.0,
    )
    local_transform = struct.pack(
        "12f",
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
    )
    return header + k_matrix + local_transform


def _serialized_mono_camera(
    *,
    width: int = 640,
    height: int = 480,
    k_matrix: tuple[float, ...] = (),
    projection: tuple[float, ...] = (),
) -> bytes:
    return struct.pack(
        "11i",
        0,
        23,
        5,
        0,
        width,
        height,
        len(k_matrix),
        0,
        0,
        len(projection),
        len(IDENTITY_LOCAL_TRANSFORM),
    ) + struct.pack(
        f"{len(k_matrix)}d{len(projection)}d12f",
        *k_matrix,
        *projection,
        *IDENTITY_LOCAL_TRANSFORM,
    )


def _serialized_stereo_camera(left: bytes, right: bytes) -> bytes:
    return struct.pack(
        "10i",
        0,
        23,
        5,
        1,
        0,
        0,
        0,
        0,
        len(left),
        len(right),
    ) + left + right


def test_parse_camera_model_kmatrix_prefers_decoded_image_size():
    camera = parse_camera_model(
        _arcore_kmatrix_calibration_blob(),
        image_width=640,
        image_height=480,
    )

    assert camera is not None
    assert camera.width == 640
    assert camera.height == 480


def test_parse_camera_model_serialized_mono_uses_projection_matrix():
    camera = parse_camera_model(
        _serialized_mono_camera(
            projection=(
                510.0,
                0.0,
                321.0,
                0.0,
                0.0,
                511.0,
                242.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            )
        )
    )

    assert camera is not None
    assert camera.width == 640
    assert camera.height == 480
    assert camera.fx == pytest.approx(510.0)
    assert camera.fy == pytest.approx(511.0)
    assert camera.cx == pytest.approx(321.0)
    assert camera.cy == pytest.approx(242.0)


def test_parse_camera_model_serialized_stereo_returns_left_camera():
    left = _serialized_mono_camera(
        width=640,
        height=480,
        projection=(
            510.0,
            0.0,
            321.0,
            0.0,
            0.0,
            511.0,
            242.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ),
    )
    right = _serialized_mono_camera(
        width=640,
        height=480,
        projection=(
            520.0,
            0.0,
            331.0,
            -52.0,
            0.0,
            521.0,
            252.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ),
    )

    camera = parse_camera_model(_serialized_stereo_camera(left, right))

    assert camera is not None
    assert camera.fx == pytest.approx(510.0)
    assert camera.fy == pytest.approx(511.0)
    assert camera.cx == pytest.approx(321.0)
    assert camera.cy == pytest.approx(242.0)


def test_parse_camera_model_legacy_float_mono_with_image_size():
    camera = parse_camera_model(
        struct.pack(
            "18f",
            500.0,
            501.0,
            320.0,
            241.0,
            640.0,
            480.0,
            *IDENTITY_LOCAL_TRANSFORM,
        )
    )

    assert camera is not None
    assert camera.width == 640
    assert camera.height == 480
    assert camera.fx == pytest.approx(500.0)
    assert camera.fy == pytest.approx(501.0)
    assert camera.cx == pytest.approx(320.0)
    assert camera.cy == pytest.approx(241.0)


def test_parse_camera_model_legacy_float_stereo_with_baseline_uses_left_camera():
    camera = parse_camera_model(
        struct.pack(
            "19f",
            500.0,
            501.0,
            320.0,
            241.0,
            0.12,
            640.0,
            480.0,
            *IDENTITY_LOCAL_TRANSFORM,
        )
    )

    assert camera is not None
    assert camera.width == 640
    assert camera.height == 480
    assert camera.fx == pytest.approx(500.0)
    assert camera.fy == pytest.approx(501.0)
    assert camera.cx == pytest.approx(320.0)
    assert camera.cy == pytest.approx(241.0)


def test_parse_camera_model_kmatrix_scans_size_on_aligned_offsets():
    camera = parse_camera_model(_arcore_kmatrix_calibration_blob())

    assert camera is not None
    assert camera.width == 640
    assert camera.height == 480
