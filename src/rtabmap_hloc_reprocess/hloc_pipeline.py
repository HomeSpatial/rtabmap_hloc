"""
HLOC Reconstruction Pipeline

Provides sparse reconstruction using Hierarchical Localization (HLOC) with
ALIKED features and MegaLoc global descriptors.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def run_hloc_reconstruction(
    images_dir: Path,
    output_dir: Path,
    feature_conf: str = "aliked-n16",
    retrieval_conf: str = "megaloc",
    matcher_conf: str = "aliked+lightglue",
    num_matched: int = 20,
    image_list: Optional[List[str]] = None,
    camera_mode: str = "AUTO",
    rtabmap_intrinsics: Optional[Dict[str, Any]] = None,
    fix_intrinsics: bool = True,
    colmap_options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    inference_backend: str = "torch",
    hloc_execution_profile: str = "colab-models-l4-v1",
) -> Optional[Any]:
    """
    Run HLOC sparse reconstruction pipeline.

    Args:
        images_dir: Directory containing images
        output_dir: Output directory for reconstruction results
        feature_conf: Feature extractor configuration name
        retrieval_conf: Global descriptor configuration for image retrieval
        matcher_conf: Feature matcher configuration name
        num_matched: Number of image pairs to match per image
        image_list: Optional list of image names to process
        camera_mode: pycolmap CameraMode name used when importing images
        rtabmap_intrinsics: Optional image name -> camera intrinsics mapping
        fix_intrinsics: Keep RTABMap intrinsics constant during bundle adjustment
        colmap_options: Optional pycolmap IncrementalPipelineOptions overrides
        verbose: Enable verbose output

    Returns:
        pycolmap.Reconstruction object or None if failed
    """
    import pycolmap
    from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction

    from .colmap_profiles import apply_fixed_intrinsics_options

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    sfm_dir = output_dir / "sfm"
    features_path = output_dir / f"{feature_conf}.h5"
    global_descriptors_path = output_dir / f"{retrieval_conf}.h5"
    pairs_path = output_dir / f"pairs-{retrieval_conf}{num_matched}.txt"
    matches_path = output_dir / f"{matcher_conf}.h5"

    if inference_backend == "colab":
        from .colab_hloc import run_colab_features

        run_colab_features(
            images_dir, output_dir, feature_conf, retrieval_conf, matcher_conf,
            num_matched, image_list, profile=hloc_execution_profile,
        )
        colmap_options = {**(colmap_options or {}), "ba_use_gpu": False}
    else:
        # Step 1: Extract local features
        print(f"[1/4] Extracting local features using {feature_conf}...")
        feature_conf_dict = extract_features.confs[feature_conf]
        features_path = extract_features.main(
            feature_conf_dict,
            images_dir,
            image_list=image_list,
            feature_path=features_path,
        )

        # Step 2: Extract global descriptors for retrieval
        print(f"[2/4] Extracting global descriptors using {retrieval_conf}...")
        retrieval_conf_dict = extract_features.confs[retrieval_conf]
        global_descriptors_path = extract_features.main(
            retrieval_conf_dict,
            images_dir,
            image_list=image_list,
            feature_path=global_descriptors_path,
        )

        # Step 3: Find image pairs using retrieval
        print(f"[3/4] Finding image pairs (top {num_matched} matches per image)...")
        pairs_from_retrieval.main(
            global_descriptors_path,
            pairs_path,
            num_matched=num_matched,
        )

        # Step 4: Match features between pairs
        print(f"[4/4] Matching features using {matcher_conf}...")
        matcher_conf_dict = match_features.confs[matcher_conf]
        matches_path = match_features.main(
            matcher_conf_dict,
            pairs_path,
            features_path,
            matches=matches_path,
        )

    mode_name = (camera_mode or "AUTO").strip().upper()
    try:
        camera_mode_enum = getattr(pycolmap.CameraMode, mode_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown pycolmap CameraMode: {camera_mode!r}") from exc

    if rtabmap_intrinsics:
        print("[5/5] Running sparse reconstruction with RTABMap intrinsics...")
        sfm_dir.mkdir(parents=True, exist_ok=True)
        database = sfm_dir / "database.db"
        reconstruction.create_empty_db(database)
        image_ids, camera_ids = import_images_with_rtabmap_intrinsics(
            images_dir,
            database,
            image_list,
            rtabmap_intrinsics,
        )
        with pycolmap.Database.open(database) as db:
            reconstruction.import_features(image_ids, db, features_path)
            reconstruction.import_matches(image_ids, db, pairs_path, matches_path)
        reconstruction.estimation_and_geometric_verification(
            database, pairs_path, verbose
        )

        mapper_options: Dict[str, Any] = dict(colmap_options or {})
        if fix_intrinsics:
            mapper_options = apply_fixed_intrinsics_options(
                mapper_options,
                camera_ids,
            )

        model = reconstruction.run_reconstruction(
            sfm_dir,
            database,
            images_dir,
            verbose,
            mapper_options,
        )
        if model is not None:
            print(f"\nReconstruction successful!")
            print(f"  - Registered images: {model.num_reg_images()}")
            print(f"  - 3D points: {model.num_points3D()}")
            print(f"  - Output: {sfm_dir}")
        else:
            print("Reconstruction failed!")

        return model

    # Step 5: Run reconstruction
    print("[5/5] Running sparse reconstruction...")
    model = reconstruction.main(
        sfm_dir,
        images_dir,
        pairs_path,
        features_path,
        matches_path,
        camera_mode=camera_mode_enum,
        image_list=image_list,
        mapper_options=dict(colmap_options or {}),
        verbose=verbose,
    )

    if model is not None:
        print(f"\nReconstruction successful!")
        print(f"  - Registered images: {model.num_reg_images()}")
        print(f"  - 3D points: {model.num_points3D()}")
        print(f"  - Output: {sfm_dir}")
    else:
        print("Reconstruction failed!")

    return model


def import_images_with_rtabmap_intrinsics(
    image_dir: Path,
    database_path: Path,
    image_list: Optional[List[str]],
    rtabmap_intrinsics: Dict[str, Any],
) -> Tuple[Dict[str, int], set[int]]:
    """Import images, then replace COLMAP cameras with RTABMap PINHOLE intrinsics."""
    import pycolmap
    from hloc import reconstruction

    reconstruction.import_images(
        image_dir,
        database_path,
        pycolmap.CameraMode.PER_IMAGE,
        image_list,
    )

    image_ids: Dict[str, int] = {}
    camera_ids: set[int] = set()
    with pycolmap.Database.open(database_path) as db:
        for image in db.read_all_images():
            image_ids[image.name] = image.image_id
            camera_model = rtabmap_intrinsics.get(image.name)
            if camera_model is None:
                raise ValueError(
                    f"Missing RTABMap camera intrinsics for image {image.name!r}"
                )

            camera = pycolmap.Camera(
                camera_id=image.camera_id,
                model="PINHOLE",
                has_prior_focal_length=True,
                width=int(camera_model.width),
                height=int(camera_model.height),
                params=[
                    float(camera_model.fx),
                    float(camera_model.fy),
                    float(camera_model.cx),
                    float(camera_model.cy),
                ],
            )
            db.update_camera(camera)
            camera_ids.add(image.camera_id)

    return image_ids, camera_ids


def run_hloc_triangulation(
    images_dir: Path,
    output_dir: Path,
    reference_model_dir: Path,
    feature_conf: str = "aliked-n16",
    retrieval_conf: str = "megaloc",
    matcher_conf: str = "aliked+lightglue",
    num_matched: int = 20,
    image_list: Optional[List[str]] = None,
    colmap_options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    inference_backend: str = "torch",
    hloc_execution_profile: str = "colab-models-l4-v1",
) -> Optional[Any]:
    """
    Run HLOC triangulation pipeline using existing camera poses.

    Instead of running incremental SfM, this uses a reference model with
    known poses and only triangulates 3D points from feature matches.

    Args:
        images_dir: Directory containing images
        output_dir: Output directory for reconstruction results
        reference_model_dir: Directory containing reference COLMAP model
                            (cameras.txt, images.txt, points3D.txt)
        feature_conf: Feature extractor configuration name
        retrieval_conf: Global descriptor configuration for image retrieval
        matcher_conf: Feature matcher configuration name
        num_matched: Number of image pairs to match per image
        image_list: Optional list of image names to process
        colmap_options: Optional pycolmap IncrementalPipelineOptions overrides
        verbose: Enable verbose output

    Returns:
        pycolmap.Reconstruction object or None if failed
    """
    from hloc import extract_features, match_features, pairs_from_retrieval, triangulation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    sfm_dir = output_dir / "sfm"
    features_path = output_dir / f"{feature_conf}.h5"
    global_descriptors_path = output_dir / f"{retrieval_conf}.h5"
    pairs_path = output_dir / f"pairs-{retrieval_conf}{num_matched}.txt"
    matches_path = output_dir / f"{matcher_conf}.h5"

    if inference_backend == "colab":
        from .colab_hloc import run_colab_features

        run_colab_features(
            images_dir, output_dir, feature_conf, retrieval_conf, matcher_conf,
            num_matched, image_list, profile=hloc_execution_profile,
        )
        colmap_options = {**(colmap_options or {}), "ba_use_gpu": False}
    else:
        # Step 1: Extract local features
        print(f"[1/4] Extracting local features using {feature_conf}...")
        feature_conf_dict = extract_features.confs[feature_conf]
        features_path = extract_features.main(
            feature_conf_dict,
            images_dir,
            image_list=image_list,
            feature_path=features_path,
        )

        # Step 2: Extract global descriptors for retrieval
        print(f"[2/4] Extracting global descriptors using {retrieval_conf}...")
        retrieval_conf_dict = extract_features.confs[retrieval_conf]
        global_descriptors_path = extract_features.main(
            retrieval_conf_dict,
            images_dir,
            image_list=image_list,
            feature_path=global_descriptors_path,
        )

        # Step 3: Find image pairs using retrieval
        print(f"[3/4] Finding image pairs (top {num_matched} matches per image)...")
        pairs_from_retrieval.main(
            global_descriptors_path,
            pairs_path,
            num_matched=num_matched,
        )

        # Step 4: Match features between pairs
        print(f"[4/4] Matching features using {matcher_conf}...")
        matcher_conf_dict = match_features.confs[matcher_conf]
        matches_path = match_features.main(
            matcher_conf_dict,
            pairs_path,
            features_path,
            matches=matches_path,
        )

    # Step 5: Triangulate using existing poses
    print("[5/5] Triangulating 3D points using existing poses...")
    model = triangulation.main(
        sfm_dir,
        reference_model_dir,
        images_dir,
        pairs_path,
        features_path,
        matches_path,
        mapper_options=dict(colmap_options or {}),
        verbose=verbose,
    )

    if model is not None:
        print(f"\nTriangulation successful!")
        print(f"  - Registered images: {model.num_reg_images()}")
        print(f"  - 3D points: {model.num_points3D()}")
        print(f"  - Output: {sfm_dir}")
    else:
        print("Triangulation failed!")

    return model


def load_colmap_reconstruction(sfm_dir: Path) -> Optional[Any]:
    """Load a COLMAP reconstruction from disk"""
    import pycolmap

    sfm_dir = Path(sfm_dir)

    # Try to load binary format first
    if (sfm_dir / "cameras.bin").exists():
        return pycolmap.Reconstruction(sfm_dir)

    # Try text format
    if (sfm_dir / "cameras.txt").exists():
        rec = pycolmap.Reconstruction()
        rec.read_text(sfm_dir)
        return rec

    return None


def get_reconstruction_poses(model: Any) -> Dict[str, np.ndarray]:
    """
    Extract camera poses from a COLMAP reconstruction.

    Args:
        model: pycolmap.Reconstruction object

    Returns:
        Dictionary mapping image names to 4x4 transformation matrices (world to camera)
    """
    poses = {}

    for image_id, image in model.images.items():
        # COLMAP stores camera-to-world rotation and camera position
        # We want world-to-camera transform
        # Note: In newer pycolmap, cam_from_world is a method
        cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t

        poses[image.name] = pose

    return poses


def get_reconstruction_points3d(model: Any) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extract 3D points from a COLMAP reconstruction.

    Args:
        model: pycolmap.Reconstruction object

    Returns:
        Tuple of (xyz array, rgb array, point3d_ids list)
    """
    points = []
    colors = []
    point_ids = []

    for point3d_id, point3d in model.points3D.items():
        points.append(point3d.xyz)
        colors.append(point3d.color)
        point_ids.append(point3d_id)

    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), []

    return np.array(points), np.array(colors), point_ids
