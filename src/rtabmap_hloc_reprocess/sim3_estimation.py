"""
Sim3 Transform Estimation

Estimates the Sim3 (similarity) transformation between HLOC reconstruction
and RTABMap original poses to recover absolute scale.

Coordinate System Notes:
- COLMAP/HLOC: Uses computer vision / optical convention
  - cam_from_world is world-to-camera transform
  - Camera center = -R^T @ t
  - Axes: X-right, Y-down, Z-forward (looking into the scene)

- RTABMap: Uses a modified ROS convention for camera poses
  - Poses stored as camera-to-world transform (pose of camera in world frame)
  - Camera center is the translation component directly
  - Camera axes in RTABMap convention:
    - X = camera forward (optical Z)
    - Y = camera left (-optical X)
    - Z = camera up (-optical Y)
  - This is similar to ROS REP-103 base_link convention

- Transformation between conventions:
  - To convert a rotation matrix from RTABMap to optical:
    R_optical = R_rtabmap @ T_RTABMAP_TO_OPTICAL (which equals T_BASE_TO_OPTICAL)
  - To convert from optical to RTABMap:
    R_rtabmap = R_optical @ T_BASE_TO_OPTICAL

  - T_BASE_TO_OPTICAL = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    Maps: rtabmap_X -> optical_Z, rtabmap_Y -> -optical_X, rtabmap_Z -> -optical_Y
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import pycolmap


# Transformation matrix from optical frame (X-right, Y-down, Z-forward)
# to RTABMap/ROS base_link frame (X-forward, Y-left, Z-up)
T_OPTICAL_TO_BASE = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

# Transformation from RTABMap/base_link to optical frame
# This is the transpose (inverse) of T_OPTICAL_TO_BASE
T_BASE_TO_OPTICAL = T_OPTICAL_TO_BASE.T


def estimate_sim3_umeyama(
    source_points: np.ndarray, target_points: np.ndarray, with_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate Sim3 transformation using Umeyama's method.

    Finds scale s, rotation R, and translation t such that:
    target = s * R @ source + t

    Args:
        source_points: Nx3 array of source points
        target_points: Nx3 array of target points
        with_scale: If True, estimate scale; otherwise assume scale=1

    Returns:
        Tuple of (scale, rotation_matrix, translation_vector)
    """
    assert source_points.shape == target_points.shape
    assert source_points.shape[1] == 3
    n = source_points.shape[0]

    # Compute centroids
    mu_source = np.mean(source_points, axis=0)
    mu_target = np.mean(target_points, axis=0)

    # Center the points
    source_centered = source_points - mu_source
    target_centered = target_points - mu_target

    # Compute covariance matrix
    cov = target_centered.T @ source_centered / n

    # SVD
    U, S, Vt = np.linalg.svd(cov)

    # Compute rotation
    d = np.linalg.det(U @ Vt)
    D = np.diag([1, 1, d])
    R = U @ D @ Vt

    # Compute scale
    if with_scale:
        var_source = np.sum(source_centered**2) / n
        scale = np.trace(np.diag(S) @ D) / var_source
    else:
        scale = 1.0

    # Compute translation
    t = mu_target - scale * R @ mu_source

    return scale, R, t


def extract_camera_centers(poses: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Extract camera centers from poses.

    Args:
        poses: Dictionary mapping image names to 4x4 transformation matrices

    Returns:
        Dictionary mapping image names to 3D camera centers
    """
    centers = {}
    for name, pose in poses.items():
        # For world-to-camera transform T, camera center is -R^T @ t
        R = pose[:3, :3]
        t = pose[:3, 3]
        center = -R.T @ t
        centers[name] = center
    return centers


def get_camera_center_from_world_to_cam(pose: np.ndarray) -> np.ndarray:
    """
    Extract camera center from a world-to-camera transformation matrix.

    For world-to-camera transform T_wc where:
        p_cam = R @ p_world + t

    Camera center in world frame:
        center = -R^T @ t

    Args:
        pose: 4x4 world-to-camera transformation matrix

    Returns:
        3D camera center in world coordinates
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    return -R.T @ t


def get_camera_center_from_cam_to_world(pose: np.ndarray) -> np.ndarray:
    """
    Extract camera center from a camera-to-world transformation matrix.

    For camera-to-world transform T_cw where:
        p_world = R @ p_cam + t

    Camera center in world frame is simply the translation component.

    Args:
        pose: 4x4 camera-to-world transformation matrix

    Returns:
        3D camera center in world coordinates
    """
    return pose[:3, 3].copy()


def rtabmap_pose_to_optical(pose: np.ndarray) -> np.ndarray:
    """
    Convert RTABMap pose (base_link convention) to optical frame convention.

    RTABMap stores camera poses in ROS base_link convention:
        - X-forward, Y-left, Z-up

    This function converts to optical frame convention (same as COLMAP):
        - X-right, Y-down, Z-forward

    The input and output are both camera-to-world transforms.

    Args:
        pose: 4x4 camera-to-world transform in base_link convention

    Returns:
        4x4 camera-to-world transform in optical convention
    """
    result = np.eye(4)
    # R_world_optical = R_world_base @ T_base_to_optical
    result[:3, :3] = pose[:3, :3] @ T_BASE_TO_OPTICAL
    result[:3, 3] = pose[:3, 3]  # Translation is the same (camera center in world)
    return result


def optical_pose_to_rtabmap(pose: np.ndarray) -> np.ndarray:
    """
    Convert optical frame pose to RTABMap (base_link convention) pose.

    This is the inverse of rtabmap_pose_to_optical.

    Args:
        pose: 4x4 camera-to-world transform in optical convention

    Returns:
        4x4 camera-to-world transform in RTABMap convention
    """
    result = np.eye(4)
    # RTABMap stores poses where the camera axes are:
    #   X = camera forward (optical Z)
    #   Y = camera left (-optical X)
    #   Z = camera up (-optical Y)
    # To convert from optical to RTABMap: R_rtabmap = R_optical @ T_BASE_TO_OPTICAL
    result[:3, :3] = pose[:3, :3] @ T_BASE_TO_OPTICAL
    result[:3, 3] = pose[:3, 3]  # Translation is the same
    return result


def hloc_to_rtabmap_pose(
    hloc_pose: np.ndarray, scale: float, R_sim3: np.ndarray, t_sim3: np.ndarray
) -> np.ndarray:
    """
    Convert HLOC pose (world-to-camera, optical frame) to RTABMap format.

    This applies the Sim3 transformation and coordinate system conversion:
    1. Apply Sim3 to transform from HLOC world to RTABMap world
    2. Convert from world-to-camera to camera-to-world
    3. Convert from optical frame to base_link frame

    Args:
        hloc_pose: 4x4 world-to-camera transform in HLOC frame (optical convention)
        scale: Sim3 scale factor
        R_sim3: 3x3 Sim3 rotation matrix
        t_sim3: 3D Sim3 translation vector

    Returns:
        4x4 camera-to-world transform in RTABMap frame (base_link convention)
    """
    R_wc_hloc = hloc_pose[:3, :3]
    t_wc_hloc = hloc_pose[:3, 3]

    # Camera center in HLOC world frame
    center_hloc = -R_wc_hloc.T @ t_wc_hloc

    # Transform center to RTABMap world frame
    center_rtab = scale * R_sim3 @ center_hloc + t_sim3

    # Transform rotation: R_wc_new = R_wc_hloc @ R_sim3^T
    R_wc_new = R_wc_hloc @ R_sim3.T

    # Convert to camera-to-world (optical frame)
    R_cw_optical = R_wc_new.T

    # Convert to RTABMap base_link frame
    # RTABMap stores poses in a coordinate system where:
    #   X = camera forward (optical Z)
    #   Y = camera left (-optical X)
    #   Z = camera up (-optical Y)
    # The transformation from optical to this convention is T_BASE_TO_OPTICAL
    # applied as a right multiplication on the rotation matrix
    R_cw_base = R_cw_optical @ T_BASE_TO_OPTICAL

    # Construct the result
    result = np.eye(4)
    result[:3, :3] = R_cw_base
    result[:3, 3] = center_rtab

    return result


def estimate_sim3_robust(
    model: "pycolmap.Reconstruction",
    rtabmap_poses: Dict[int, np.ndarray],
    image_to_node_id: Dict[str, int],
    max_error: float = 1.0,
    min_common_images: int = 5,
    verbose: bool = True,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Estimate Sim3 transformation using pycolmap's RANSAC-based aligner.

    Uses ``pycolmap.align_reconstruction_to_locations`` internally, which
    wraps COLMAP's LO-RANSAC + Umeyama for robust outlier rejection.

    Args:
        model: pycolmap.Reconstruction from HLOC (in arbitrary HLOC scale).
        rtabmap_poses: RTABMap optimized poses (node_id -> 4x4 camera-to-world).
        image_to_node_id: Mapping from image filename to RTABMap node ID.
        max_error: RANSAC max error in meters (metric scale expected).
        min_common_images: Minimum number of matched images for alignment.
        verbose: Print debug information.

    Returns:
        Tuple of (scale, rotation, translation, matched_image_names).
        Returns (None, None, None, []) if alignment fails.
    """
    import pycolmap

    tgt_names: List[str] = []
    tgt_locs: List[np.ndarray] = []

    for image_id, image in model.images.items():
        image_name = image.name
        if image_name not in image_to_node_id:
            continue
        node_id = image_to_node_id[image_name]
        if node_id not in rtabmap_poses:
            continue
        tgt_names.append(image_name)
        tgt_locs.append(rtabmap_poses[node_id][:3, 3])

    if len(tgt_names) < min_common_images:
        print(
            f"[estimate_sim3_robust] Insufficient common images: "
            f"{len(tgt_names)} < {min_common_images}"
        )
        return None, None, None, []

    if verbose:
        print(
            f"[estimate_sim3_robust] Found {len(tgt_names)} common images "
            f"for RANSAC-based Sim3 estimation"
        )

    tgt_locations = np.array(tgt_locs, dtype=np.float64)

    ransac_options = pycolmap.RANSACOptions()
    ransac_options.max_error = max_error
    ransac_options.min_inlier_ratio = 0.3
    ransac_options.confidence = 0.999

    sim3 = pycolmap.align_reconstruction_to_locations(
        model,
        tgt_names,
        tgt_locations,
        min_common_images=min_common_images,
        ransac_options=ransac_options,
    )

    if sim3 is None:
        print("[estimate_sim3_robust] RANSAC-based Sim3 estimation failed (no inliers)")
        return None, None, None, []

    scale = float(sim3.scale)
    R = np.array(sim3.rotation.matrix(), dtype=np.float64)
    t = np.array(sim3.translation, dtype=np.float64)

    matched_names = tgt_names[:]

    if verbose:
        print(f"[estimate_sim3_robust] Sim3 estimated successfully:")
        print(f"  Scale (HLOC -> RTABMap): {scale:.6f}")
        num_inliers = len(matched_names)
        print(f"  Inlier images: {num_inliers} / {len(tgt_names)} common")

    return scale, R, t, matched_names


def estimate_sim3_from_poses(
    hloc_poses: Dict[str, np.ndarray],
    rtabmap_poses: Dict[int, np.ndarray],
    image_to_node_id: Dict[str, int],
    min_correspondences: int = 3,
    verbose: bool = True,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Estimate Sim3 transformation from HLOC to RTABMap coordinate frame.

    DEPRECATED: Use ``estimate_sim3_robust()`` for RANSAC-based estimation.
    This function remains for backwards compatibility.

    Coordinate System Notes:
    - HLOC poses are world-to-camera transforms (from COLMAP's cam_from_world)
    - RTABMap poses are camera-to-world transforms (pose of camera in map frame)

    Args:
        hloc_poses: HLOC poses (image_name -> 4x4 world-to-camera transform)
        rtabmap_poses: RTABMap optimized poses (node_id -> 4x4 camera-to-world transform)
        image_to_node_id: Mapping from image filename to RTABMap node ID
        min_correspondences: Minimum number of correspondences required
        verbose: Print debug information

    Returns:
        Tuple of (scale, rotation, translation, matched_image_names)
        Returns (None, None, None, []) if insufficient correspondences
    """
    # Find common images
    source_points = []  # HLOC camera centers (in HLOC world frame)
    target_points = []  # RTABMap camera centers (in RTABMap map frame)
    matched_names = []

    for image_name, hloc_pose in hloc_poses.items():
        if image_name not in image_to_node_id:
            continue

        node_id = image_to_node_id[image_name]
        if node_id not in rtabmap_poses:
            continue

        rtabmap_pose = rtabmap_poses[node_id]

        # HLOC: world-to-camera transform, extract camera center
        center_hloc = get_camera_center_from_world_to_cam(hloc_pose)

        # RTABMap: camera-to-world transform (pose of camera in map frame)
        # Camera center is the translation part
        center_rtab = get_camera_center_from_cam_to_world(rtabmap_pose)

        source_points.append(center_hloc)
        target_points.append(center_rtab)
        matched_names.append(image_name)

    if len(source_points) < min_correspondences:
        print(
            f"Insufficient correspondences: {len(source_points)} < {min_correspondences}"
        )
        return None, None, None, []

    source_points = np.array(source_points)
    target_points = np.array(target_points)

    if verbose:
        print(f"\nCamera center statistics:")
        print(
            f"  HLOC centers - mean: {np.mean(source_points, axis=0)}, std: {np.std(source_points, axis=0)}"
        )
        print(
            f"  RTABMap centers - mean: {np.mean(target_points, axis=0)}, std: {np.std(target_points, axis=0)}"
        )

    # Estimate Sim3: target = scale * R @ source + t
    scale, R, t = estimate_sim3_umeyama(source_points, target_points, with_scale=True)

    return scale, R, t, matched_names


def transform_reconstruction(
    model: "pycolmap.Reconstruction",
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    """
    Apply Sim3 transformation to a COLMAP reconstruction in-place.

    The Sim3 transform maps points from the old (HLOC) world frame to the
    new (RTABMap) world frame:
        p_new = scale * rotation @ p_old + translation

    For camera poses (world-to-camera transforms), we need to update them
    so that they correctly relate the new world frame to camera frames.

    Args:
        model: pycolmap.Reconstruction object
        scale: Scale factor (HLOC to RTABMap scale)
        rotation: 3x3 rotation matrix (R_rtab_hloc)
        translation: 3D translation vector
    """
    import pycolmap

    # Transform 3D points: p_new = s * R @ p_old + t
    for point3d_id in list(model.points3D.keys()):
        point3d = model.points3D[point3d_id]
        new_xyz = scale * rotation @ point3d.xyz + translation
        point3d.xyz = new_xyz

    # Transform camera poses
    # Original: T_wc_old transforms from old world to camera
    # New: T_wc_new transforms from new world to camera
    #
    # Let S be the Sim3 transform: p_new = s * R_s @ p_old + t_s
    # Then: T_wc_new = T_wc_old @ S^(-1)
    #
    # For camera center:
    #   center_old = -R_wc^T @ t_wc  (in old world frame)
    #   center_new = s * R_s @ center_old + t_s  (in new world frame)
    #
    # For rotation:
    #   R_wc_new = R_wc_old @ R_s^T  (compose with inverse of Sim3 rotation)
    #
    # For translation:
    #   t_wc_new = -R_wc_new @ center_new

    for image_id in list(model.images.keys()):
        image = model.images[image_id]

        # Get current pose (world-to-camera)
        cam_from_world = (
            image.cam_from_world()
            if callable(image.cam_from_world)
            else image.cam_from_world
        )
        R_wc_old = cam_from_world.rotation.matrix()
        t_wc_old = cam_from_world.translation

        # Camera center in old world frame
        center_old = -R_wc_old.T @ t_wc_old

        # Transform center to new world frame
        center_new = scale * rotation @ center_old + translation

        # New rotation: R_wc_new = R_wc_old @ R_s^T
        # This rotates from new world to camera:
        # p_cam = R_wc_new @ p_new = R_wc_old @ R_s^T @ p_new = R_wc_old @ p_old
        new_R = R_wc_old @ rotation.T

        # New translation: ensures camera center is at center_new
        # center_new = -R_wc_new^T @ t_wc_new
        # => t_wc_new = -R_wc_new @ center_new
        new_t = -new_R @ center_new

        # Update pose using pycolmap API
        # Both scipy and pycolmap use [x, y, z, w] quaternion format
        quat = Rotation.from_matrix(new_R).as_quat()  # [x, y, z, w] scipy format
        new_rigid3d = pycolmap.Rigid3d(
            pycolmap.Rotation3d(quat),  # [x, y, z, w] pycolmap format
            new_t,
        )

        # Handle different pycolmap API versions
        if hasattr(image, "frame_id"):
            # pycolmap 3.14+ API - update via frame
            frame = model.frames[image.frame_id]
            frame.set_cam_from_world(image.camera_id, new_rigid3d)
        elif hasattr(image, "set_cam_from_world"):
            image.set_cam_from_world(new_rigid3d)
        else:
            image.cam_from_world = new_rigid3d


def get_rtabmap_format_poses(
    model: "pycolmap.Reconstruction",
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Get poses from COLMAP reconstruction in RTABMap format.

    This function:
    1. Applies Sim3 transformation to convert from HLOC world to RTABMap world
    2. Converts from world-to-camera to camera-to-world
    3. Converts from optical frame to ROS base_link frame

    The resulting poses can be directly written to an RTABMap database.

    Args:
        model: pycolmap.Reconstruction object (original, not transformed)
        scale: Sim3 scale factor
        rotation: 3x3 Sim3 rotation matrix
        translation: 3D Sim3 translation vector

    Returns:
        Dictionary mapping image names to 4x4 camera-to-world transforms (base_link convention)
    """
    poses = {}

    for image_id in model.images.keys():
        image = model.images[image_id]

        # Get current pose (world-to-camera, optical convention)
        cam_from_world = (
            image.cam_from_world()
            if callable(image.cam_from_world)
            else image.cam_from_world
        )
        hloc_pose = np.eye(4)
        hloc_pose[:3, :3] = cam_from_world.rotation.matrix()
        hloc_pose[:3, 3] = cam_from_world.translation

        # Convert to RTABMap format
        rtabmap_pose = hloc_to_rtabmap_pose(hloc_pose, scale, rotation, translation)
        poses[image.name] = rtabmap_pose

    return poses


def compute_alignment_error(
    hloc_poses: Dict[str, np.ndarray],
    rtabmap_poses: Dict[int, np.ndarray],
    image_to_node_id: Dict[str, int],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute alignment error after Sim3 transformation.

    Compares the transformed HLOC camera centers with RTABMap camera centers.

    Args:
        hloc_poses: HLOC poses (world-to-camera transforms)
        rtabmap_poses: RTABMap poses (camera-to-world transforms)
        image_to_node_id: Image name to node ID mapping
        scale: Estimated scale
        rotation: Estimated rotation
        translation: Estimated translation

    Returns:
        Tuple of (mean_error, max_error, per_image_errors)
    """
    errors = {}

    for image_name, hloc_pose in hloc_poses.items():
        if image_name not in image_to_node_id:
            continue

        node_id = image_to_node_id[image_name]
        if node_id not in rtabmap_poses:
            continue

        rtabmap_pose = rtabmap_poses[node_id]

        # HLOC camera center (from world-to-camera transform)
        center_hloc = get_camera_center_from_world_to_cam(hloc_pose)

        # Transform to RTABMap frame
        center_transformed = scale * rotation @ center_hloc + translation

        # RTABMap camera center (from camera-to-world transform)
        center_rtab = get_camera_center_from_cam_to_world(rtabmap_pose)

        # Compute error (Euclidean distance)
        error = np.linalg.norm(center_transformed - center_rtab)
        errors[image_name] = error

    if len(errors) == 0:
        return 0.0, 0.0, {}

    error_values = list(errors.values())
    return np.mean(error_values), np.max(error_values), errors


def compute_rotation_error(
    hloc_poses: Dict[str, np.ndarray],
    rtabmap_poses: Dict[int, np.ndarray],
    image_to_node_id: Dict[str, int],
    rotation: np.ndarray,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute rotation alignment error after Sim3 transformation.

    Takes into account the coordinate system difference between HLOC (optical)
    and RTABMap (base_link) conventions.

    Args:
        hloc_poses: HLOC poses (world-to-camera transforms, optical convention)
        rtabmap_poses: RTABMap poses (camera-to-world transforms, RTABMap convention)
        image_to_node_id: Image name to node ID mapping
        rotation: Estimated Sim3 rotation

    Returns:
        Tuple of (mean_error_deg, max_error_deg, per_image_errors)
    """
    errors = {}

    for image_name, hloc_pose in hloc_poses.items():
        if image_name not in image_to_node_id:
            continue

        node_id = image_to_node_id[image_name]
        if node_id not in rtabmap_poses:
            continue

        rtabmap_pose = rtabmap_poses[node_id]

        # HLOC: world-to-camera rotation R_wc (optical convention)
        R_wc_hloc = hloc_pose[:3, :3]

        # Transform HLOC rotation to RTABMap world frame
        # R_wc_new = R_wc_hloc @ R_sim3^T
        R_wc_transformed = R_wc_hloc @ rotation.T

        # Convert HLOC to camera-to-world (optical)
        R_cw_hloc_optical = R_wc_transformed.T

        # Convert to RTABMap convention for comparison
        # RTABMap uses: X=forward, Y=left, Z=up (different from standard optical)
        R_cw_hloc_rtabmap = R_cw_hloc_optical @ T_BASE_TO_OPTICAL

        # RTABMap: camera-to-world rotation (RTABMap convention)
        R_cw_rtab = rtabmap_pose[:3, :3]

        # Compute rotation difference
        R_diff = R_cw_hloc_rtabmap @ R_cw_rtab.T

        # Convert to angle (rotation angle of the difference)
        angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        angle_deg = np.degrees(angle_rad)

        errors[image_name] = angle_deg

    if len(errors) == 0:
        return 0.0, 0.0, {}

    error_values = list(errors.values())
    return np.mean(error_values), np.max(error_values), errors
