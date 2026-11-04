import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rtabmap_hloc_reprocess.rotation_consistency import RotationEdge, audit_rotation_cycles


def yaw(degrees):
    return Rotation.from_euler("y", degrees, degrees=True).as_matrix()


def test_cycle_composition_rejects_false_loop_without_pose_reference():
    # IDs deliberately do not encode capture order. Rotation order matters.
    rx = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    ry = yaw(30)
    rz = Rotation.from_euler("z", 10, degrees=True).as_matrix()
    edges = [RotationEdge(99, 2, rx, 120), RotationEdge(2, 71, ry, 110),
             RotationEdge(71, 4, rz, 100),
             RotationEdge(99, 4, rz @ ry @ rx, 90),
             RotationEdge(2, 4, yaw(90) @ rz @ ry, 80)]
    result = audit_rotation_cycles([99, 2, 71, 4], edges)
    bypair = {(e["source"], e["target"]): e for e in result["edges"]}
    assert len(result["components"]) == 1
    assert not bypair[99, 4]["rejected"]
    assert bypair[99, 4]["rotation_error_deg"] < 1e-8
    assert bypair[2, 4]["rejected"]
    assert bypair[2, 4]["rotation_error_deg"] == pytest.approx(90)


def test_disconnected_backbone_does_not_certify_cross_component_loop():
    result = audit_rotation_cycles(
        [1, 2, 3, 4, 5],
        [RotationEdge(1, 2, yaw(3), 80), RotationEdge(4, 5, yaw(2), 90),
         RotationEdge(1, 5, yaw(80), 200)], temporal_window=1,
    )
    assert len(result["components"]) == 3
    assert result["unverifiable_count"] == 1
    assert result["edges"][-1]["verifiable"] is False


def test_invalid_rotation_cannot_silently_enter_filter():
    with pytest.raises(ValueError, match="SO"):
        audit_rotation_cycles([1, 2], [RotationEdge(1, 2, np.eye(3)*2, 80)])
