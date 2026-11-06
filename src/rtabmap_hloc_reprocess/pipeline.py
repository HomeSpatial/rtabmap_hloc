"""
Main Pipeline

End-to-end pipeline for converting RTABMap database to HLOC format,
performing sparse reconstruction, and aligning to original scale.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PipelineConfig:
    """Configuration for the RTABMap-HLOC pipeline"""

    # Input/Output paths
    rtabmap_db: str = ""
    output_dir: str = ""

    # HLOC settings
    feature_conf: str = "aliked-n16"
    retrieval_conf: str = "megaloc"
    matcher_conf: str = "aliked+lightglue"
    num_matched: int = 20
    colmap_profile: str = "default"
    inference_backend: str = "torch"
    hloc_execution_profile: str = "colab-models-l4-v1"

    # Processing options
    skip_image_export: bool = False
    skip_hloc: bool = False  # Skip HLOC reconstruction, use existing sfm result
    use_existing_poses: bool = False  # Use RTABMap poses for triangulation only
    use_rtabmap_intrinsics: bool = True
    fix_intrinsics: bool = True
    export_depth: bool = False
    verbose: bool = False

    # Provenance metadata (schema v2, consumed by build_map_pack)
    map_id: str = ""
    camera_mode: str = "AUTO"  # pycolmap.CameraMode name used by full SfM import_images.
    scene_id: str = ""
    up_axis: str = "+Z"


@dataclass
class PipelineResult:
    """Results from the pipeline"""

    success: bool = False
    reconstruction: Any = None

    # Statistics
    num_images_exported: int = 0
    num_images_registered: int = 0
    num_points3d: int = 0

    # Sim3 alignment
    scale: Optional[float] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    alignment_error_mean: float = 0.0
    alignment_error_max: float = 0.0

    # Paths
    images_dir: str = ""
    sfm_dir: str = ""
    description_path: str = ""
    alignment_status: str = "not_estimated"
    alignment_method: str = ""
    error: Optional[str] = None


class RTABMapHLOCPipeline:
    """
    Pipeline for converting RTABMap database to HLOC reconstruction
    with scale recovery.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.images_dir = self.output_dir / "images"
        self.hloc_output_dir = self.output_dir / "hloc"
        self.sfm_dir = self.hloc_output_dir / "sfm"
        self.reference_model_dir = self.output_dir / "reference_model"

        # Mappings
        self.image_to_node_id: Dict[str, int] = {}
        self.rtabmap_poses: Dict[int, np.ndarray] = {}
        self.rtabmap_camera_models: Dict[str, Any] = {}

    def run(self) -> PipelineResult:
        """Run the complete pipeline"""
        result = PipelineResult()

        # A published description is an immutable snapshot; do not mutate its native files.
        if (self.output_dir / "reconstruction.json").exists():
            result.error = "Output already contains a reconstruction snapshot; choose a new output directory"
            print(result.error)
            return result

        try:
            # Step 1: Export images from RTABMap
            print("=" * 60)
            print("Step 1: Exporting images from RTABMap database")
            print("=" * 60)
            self._export_images()
            result.num_images_exported = len(self.image_to_node_id)
            print(f"Exported {result.num_images_exported} images")

            if result.num_images_exported == 0:
                raise RuntimeError("No images exported")

            # Step 2: Run HLOC reconstruction (or load existing)
            print("\n" + "=" * 60)
            if self.config.use_existing_poses:
                print("Step 2: Running HLOC triangulation with existing RTABMap poses")
                print("=" * 60)
                result.reconstruction = self._run_hloc_triangulation()
            elif self.config.skip_hloc and self.sfm_dir.exists():
                print("Step 2: Loading existing HLOC reconstruction (--skip-hloc)")
                print("=" * 60)
                result.reconstruction = self._load_existing_reconstruction()
            else:
                print("Step 2: Running HLOC sparse reconstruction")
                print("=" * 60)
                result.reconstruction = self._run_hloc_reconstruction()

            if result.reconstruction is None:
                raise RuntimeError("HLOC reconstruction failed")

            result.num_images_registered = result.reconstruction.num_reg_images()
            result.num_points3d = result.reconstruction.num_points3D()
            if result.num_images_registered == 0:
                raise RuntimeError("SfM has no registered images")

            # Step 3: Sim3 alignment (skip if using existing poses)
            if self.config.use_existing_poses:
                print("\n" + "=" * 60)
                print("Step 3: Skipping Sim3 alignment (using existing RTABMap poses)")
                print("=" * 60)
                # Save directly as aligned output since poses are already metric
                aligned_sfm_dir = self.output_dir / "sfm_aligned"
                aligned_sfm_dir.mkdir(parents=True, exist_ok=True)
                result.reconstruction.write(aligned_sfm_dir)
                print(f"Saved reconstruction to: {aligned_sfm_dir}")
                result.scale = 1.0
                result.alignment_error_mean = 0.0
                result.alignment_error_max = 0.0
                result.alignment_status = "input_poses"
                result.alignment_method = "triangulation_with_existing_metric_poses"
            else:
                print("\n" + "=" * 60)
                print("Step 3: Estimating Sim3 alignment to recover scale")
                print("=" * 60)
                result.alignment_status = "failed"
                self._estimate_sim3_alignment(result)
                if result.alignment_status != "aligned":
                    raise RuntimeError("Sim3 alignment did not complete; metric output is unavailable")

            result.images_dir = str(self.images_dir)
            result.sfm_dir = str(self.output_dir / "sfm_aligned")

            # Print summary
            self._print_summary(result)

            # Export metadata JSON
            self._export_metadata(result)
            from .exchange import export_description
            result.description_path = str(export_description(self, result))
            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)
            print(f"Pipeline failed with error: {e}")
            import traceback

            traceback.print_exc()

        from .exchange import write_run_status
        write_run_status(self, result)

        return result

    # Mapping from HLOC feature_conf to local descriptor dimension.
    # Used by build_map_pack to fail-fast when assumptions break.
    _FEATURE_DIM_BY_CONF: Dict[str, int] = {  # type: ignore[assignment]
        "superpoint": 256,
        "superpoint_aachen": 256,
        "superpoint_inloc": 256,
        "superpoint_max": 256,
        # Triton-backed variants share the same descriptor schema as their
        # in-process counterparts (SuperPoint outputs are 256-D regardless of
        # whether inference runs on Triton or locally).
        "superpoint_aachen_triton": 256,
        "superpoint_inloc_triton": 256,
        "superpoint_max_triton": 256,
        "superpoint_aachen_onnx": 256,
        "superpoint_inloc_onnx": 256,
        "superpoint_max_onnx": 256,
        "superpoint-trt": 256,
        "aliked-n16": 128,
        "aliked-n16-rot": 128,
        "disk": 128,
        "d2net-ss": 512,
        "r2d2": 128,
    }

    _VALID_CAMERA_MODES = {"AUTO_SINGLE", "AUTO", "PER_FOLDER", "PER_IMAGE", "SINGLE"}

    def _normalized_camera_mode(self) -> str:
        camera_mode = (self.config.camera_mode or "AUTO").strip().upper()
        if camera_mode not in self._VALID_CAMERA_MODES:
            raise ValueError(
                f"camera_mode={self.config.camera_mode!r} is not one of "
                f"{sorted(self._VALID_CAMERA_MODES)}"
            )
        return camera_mode

    def _normalized_colmap_profile(self) -> str:
        from .colmap_profiles import normalize_colmap_profile_name

        return normalize_colmap_profile_name(self.config.colmap_profile)

    def _resolve_colmap_options(self) -> Dict[str, Any]:
        from .colmap_profiles import resolve_colmap_profile

        return resolve_colmap_profile(self.config.colmap_profile)

    def _resolve_effective_colmap_options(
        self,
        *,
        camera_ids: Optional[set[int]] = None,
    ) -> Dict[str, Any]:
        from .colmap_profiles import apply_fixed_intrinsics_options

        options = self._resolve_colmap_options()
        if self.config.inference_backend == "colab":
            options["ba_use_gpu"] = False
        if (
            self.config.use_rtabmap_intrinsics
            and self.config.fix_intrinsics
            and not self.config.skip_hloc
        ):
            options = apply_fixed_intrinsics_options(options, camera_ids or set())
        return options

    @staticmethod
    def _camera_ids_from_reconstruction(reconstruction: Any) -> set[int]:
        cameras = getattr(reconstruction, "cameras", None)
        if not cameras:
            return set()
        try:
            return {int(camera_id) for camera_id in cameras.keys()}
        except Exception:
            return set()

    def _camera_ids_from_reference_model(self) -> set[int]:
        if not self.reference_model_dir.exists():
            return set()
        try:
            import pycolmap

            return self._camera_ids_from_reconstruction(
                pycolmap.Reconstruction(str(self.reference_model_dir))
            )
        except Exception as e:
            print(f"WARN: failed to load reference model camera ids: {e}")
            return set()

    def _build_colmap_options_metadata(self, result: PipelineResult) -> Dict[str, Any]:
        from .colmap_profiles import json_safe_colmap_options

        options = self._resolve_effective_colmap_options(
            camera_ids=self._camera_ids_from_reconstruction(result.reconstruction)
        )
        return json_safe_colmap_options(options)

    @staticmethod
    def _compute_db_sha256(path: Path) -> str:
        """SHA256 of source rtabmap.db (or directory: hash of sorted file list)."""
        import hashlib

        h = hashlib.sha256()
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
        elif path.is_dir():
            # Stable hash of (relpath, size) tuples; cheap and reproducible.
            for p in sorted(path.rglob("*")):
                if p.is_file():
                    h.update(str(p.relative_to(path)).encode())
                    h.update(str(p.stat().st_size).encode())
        return h.hexdigest()

    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _finite_int(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _call_reconstruction_metric(cls, reconstruction: Any, method_name: str, converter):
        method = getattr(reconstruction, method_name, None)
        if not callable(method):
            return None
        try:
            return converter(method())
        except Exception as e:
            print(f"WARN: failed to compute reconstruction metric {method_name}: {e}")
            return None

    @classmethod
    def _build_ba_metrics(cls, result: PipelineResult) -> Dict[str, Any]:
        """Collect final pycolmap reconstruction metrics for metadata.json."""
        metrics: Dict[str, Any] = {
            "registered_images": int(result.num_images_registered),
            "total_images": int(result.num_images_exported),
            "points3d": int(result.num_points3d),
        }

        reconstruction = result.reconstruction
        if reconstruction is None:
            return metrics

        optional_metrics = {
            "mean_reprojection_error_px": (
                "compute_mean_reprojection_error",
                cls._finite_float,
            ),
            "mean_track_length": ("compute_mean_track_length", cls._finite_float),
            "mean_observations_per_registered_image": (
                "compute_mean_observations_per_reg_image",
                cls._finite_float,
            ),
            "num_observations": ("compute_num_observations", cls._finite_int),
        }
        for key, (method_name, converter) in optional_metrics.items():
            value = cls._call_reconstruction_metric(
                reconstruction, method_name, converter
            )
            if value is not None:
                metrics[key] = value

        return metrics

    def _build_camera_intrinsics_metadata(self) -> Dict[str, Any]:
        """Record how COLMAP camera intrinsics were sourced and optimized."""
        if self.config.use_existing_poses:
            return {
                "reconstruction_mode": "rtabmap_pose_triangulation",
                "camera_mode": "REFERENCE_MODEL",
                "source": "rtabmap_database_calibration",
                "colmap_camera_model": "PINHOLE",
                "rtabmap_intrinsics_used": True,
                "rtabmap_poses_used": True,
                # pycolmap.triangulate_points defaults refine_intrinsics=False.
                "intrinsics_refinement": {
                    "refine_intrinsics": False,
                    "refine_focal_length": False,
                    "refine_extra_params": False,
                    "refine_principal_point": False,
                },
            }

        if self.config.skip_hloc:
            return {
                "reconstruction_mode": "loaded_existing_sfm",
                "camera_mode": "LOADED_MODEL",
                "source": "loaded_colmap_model",
                "colmap_camera_model": "loaded_model",
                "rtabmap_intrinsics_used": None,
                "rtabmap_poses_used": None,
                "intrinsics_refinement": {
                    "refine_intrinsics": None,
                    "refine_focal_length": None,
                    "refine_extra_params": None,
                    "refine_principal_point": None,
                },
            }

        if self.config.use_rtabmap_intrinsics:
            refine_intrinsics = not self.config.fix_intrinsics
            return {
                "reconstruction_mode": "full_sfm",
                "camera_mode": "RTABMAP_INTRINSICS",
                "source": "rtabmap_database_calibration",
                "colmap_camera_model": "PINHOLE",
                "rtabmap_intrinsics_used": True,
                "rtabmap_poses_used": False,
                "intrinsics_refinement": {
                    "refine_intrinsics": refine_intrinsics,
                    "refine_focal_length": refine_intrinsics,
                    "refine_extra_params": False,
                    "refine_principal_point": False,
                },
            }

        return {
            "reconstruction_mode": "full_sfm",
            "camera_mode": self._normalized_camera_mode(),
            "source": "colmap_image_reader",
            "colmap_camera_model": "SIMPLE_RADIAL",
            "rtabmap_intrinsics_used": False,
            "rtabmap_poses_used": False,
            # pycolmap incremental mapping default BundleAdjustmentOptions.
            "intrinsics_refinement": {
                "refine_intrinsics": True,
                "refine_focal_length": True,
                "refine_extra_params": True,
                "refine_principal_point": False,
            },
        }

    def _export_metadata(self, result: PipelineResult):
        """Export metadata.json (schema v2) consumed by build_map_pack and localization_service.

        See openspec/changes/map-pack-pipeline/specs/map-pack-pipeline/spec.md for
        the authoritative schema definition.
        """
        import json

        feature_conf = self.config.feature_conf
        feature_dim = self._FEATURE_DIM_BY_CONF.get(feature_conf, 0)
        if feature_dim == 0:
            print(
                f"WARN: feature_conf={feature_conf!r} has no known feature_dim; "
                "writing 0 (downstream baker will fail-fast)"
            )

        # World frame depends on whether we ran Sim3 alignment.
        world_frame = "rtabmap_local" if self.config.use_existing_poses else "rtabmap_local"
        # Note: when not use_existing_poses we still align to rtabmap via Sim3,
        # so the final world frame is still rtabmap_local. "sfm_local" would only
        # apply if alignment was skipped entirely.

        sim3_scale = float(result.scale) if result.scale is not None else 1.0

        try:
            source_sha = self._compute_db_sha256(Path(self.config.rtabmap_db))
        except Exception as e:
            print(f"WARN: failed to compute source_db_sha256: {e}")
            source_sha = ""

        camera_intrinsics = self._build_camera_intrinsics_metadata()

        metadata = {
            # Schema v2 provenance fields (new)
            "schema_version": 2,
            "map_id": self.config.map_id,
            "world_frame": world_frame,
            "sim3_scale": sim3_scale,
            "alignment_status": result.alignment_status,
            "alignment_method": result.alignment_method,
            "alignment_error_mean_m": result.alignment_error_mean,
            "alignment_error_max_m": result.alignment_error_max,
            "registered_images": int(result.num_images_registered),
            "total_images": int(result.num_images_exported),
            "feature_dim": feature_dim,
            "camera_mode": camera_intrinsics["camera_mode"],
            "use_existing_poses": bool(self.config.use_existing_poses),
            "camera_intrinsics": camera_intrinsics,
            "colmap_profile": self._normalized_colmap_profile(),
            "inference_backend": self.config.inference_backend,
            "hloc_execution_profile": (self.config.hloc_execution_profile
                                       if self.config.inference_backend == "colab" else None),
            "colmap_options": self._build_colmap_options_metadata(result),
            "source_db_sha256": source_sha,
            "ba_metrics": self._build_ba_metrics(result),
            # Path fields (relative to hloc_dir == output_dir)
            "sfm_model_dir": "sfm_aligned",
            "features_path": f"hloc/{feature_conf}.h5",
            "global_descriptors_path": f"hloc/{self.config.retrieval_conf}.h5",
            "matches_path": f"hloc/{self.config.matcher_conf}.h5",
            # HLOC config (kept for localization_service backward compat)
            "feature_conf": feature_conf,
            "retrieval_conf": self.config.retrieval_conf,
            "matcher_conf": self.config.matcher_conf,
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"\nExported metadata to: {metadata_file}")

    def _export_images(self):
        """Export images from RTABMap database or exported directory"""
        from .rtabmap_reader import RTABMapDatabase
        from .exported_data_reader import ExportedDataReader

        self.images_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(self.config.rtabmap_db)
        ReaderClass = ExportedDataReader if input_path.is_dir() else RTABMapDatabase

        with ReaderClass(str(input_path)) as db:
            self.rtabmap_camera_models = {}

            # Get optimized poses
            self.rtabmap_poses = db.get_optimized_poses()
            print(f"Found {len(self.rtabmap_poses)} poses in database")

            # Get node IDs with valid poses
            node_ids = list(self.rtabmap_poses.keys())

            # Export images
            if not self.config.skip_image_export:
                image_names = db.export_images(self.images_dir, node_ids)

                # Build mapping
                for node_id, filename in image_names.items():
                    self.image_to_node_id[filename] = node_id
            else:
                # Load existing images
                for img_path in self.images_dir.glob("*.png"):
                    try:
                        node_id = int(img_path.stem)
                        if node_id in self.rtabmap_poses:
                            self.image_to_node_id[img_path.name] = node_id
                    except ValueError:
                        continue

            # Export COLMAP reference model if using existing poses
            if self.config.use_existing_poses:
                # Build reverse mapping: node_id -> filename
                node_id_to_filename = {v: k for k, v in self.image_to_node_id.items()}
                db.export_colmap_reference_model(
                    self.reference_model_dir,
                    node_ids=list(self.image_to_node_id.values()),
                    image_names=node_id_to_filename,
                )

            for image_name, node_id in self.image_to_node_id.items():
                node = db.get_node_data(node_id, load_images=False, load_depth=False)
                if node and node.camera_model:
                    self.rtabmap_camera_models[image_name] = node.camera_model

    def _load_existing_reconstruction(self):
        """Load existing HLOC reconstruction"""
        import pycolmap

        if not self.sfm_dir.exists():
            print(f"SFM directory not found: {self.sfm_dir}")
            return None

        print(f"Loading reconstruction from {self.sfm_dir}")
        reconstruction = pycolmap.Reconstruction(str(self.sfm_dir))
        print(
            f"Loaded {reconstruction.num_reg_images()} registered images, {reconstruction.num_points3D()} 3D points"
        )
        return reconstruction

    def _run_hloc_reconstruction(self):
        """Run HLOC sparse reconstruction"""
        from .hloc_pipeline import run_hloc_reconstruction

        image_list = list(self.image_to_node_id.keys())
        colmap_options = self._resolve_effective_colmap_options()
        rtabmap_intrinsics = None
        fix_intrinsics = False
        if self.config.use_rtabmap_intrinsics:
            missing = [
                image_name
                for image_name in image_list
                if image_name not in self.rtabmap_camera_models
            ]
            if missing:
                examples = ", ".join(missing[:5])
                raise RuntimeError(
                    "RTABMap intrinsics are enabled but missing for "
                    f"{len(missing)} image(s): {examples}"
                )
            rtabmap_intrinsics = {
                image_name: self.rtabmap_camera_models[image_name]
                for image_name in image_list
            }
            fix_intrinsics = self.config.fix_intrinsics

        return run_hloc_reconstruction(
            images_dir=self.images_dir,
            output_dir=self.hloc_output_dir,
            feature_conf=self.config.feature_conf,
            retrieval_conf=self.config.retrieval_conf,
            matcher_conf=self.config.matcher_conf,
            num_matched=self.config.num_matched,
            image_list=image_list,
            camera_mode=self._normalized_camera_mode(),
            rtabmap_intrinsics=rtabmap_intrinsics,
            fix_intrinsics=fix_intrinsics,
            colmap_options=colmap_options,
            verbose=self.config.verbose,
            inference_backend=self.config.inference_backend,
            hloc_execution_profile=self.config.hloc_execution_profile,
        )

    def _run_hloc_triangulation(self):
        """Run HLOC triangulation with existing RTABMap poses"""
        from .hloc_pipeline import run_hloc_triangulation

        image_list = list(self.image_to_node_id.keys())
        colmap_options = self._resolve_effective_colmap_options(
            camera_ids=self._camera_ids_from_reference_model()
        )

        return run_hloc_triangulation(
            images_dir=self.images_dir,
            output_dir=self.hloc_output_dir,
            reference_model_dir=self.reference_model_dir,
            feature_conf=self.config.feature_conf,
            retrieval_conf=self.config.retrieval_conf,
            matcher_conf=self.config.matcher_conf,
            num_matched=self.config.num_matched,
            image_list=image_list,
            colmap_options=colmap_options,
            verbose=self.config.verbose,
            inference_backend=self.config.inference_backend,
            hloc_execution_profile=self.config.hloc_execution_profile,
        )

    def _estimate_sim3_alignment(self, result: PipelineResult):
        """Estimate Sim3 transformation to align with RTABMap poses"""
        from .hloc_pipeline import get_reconstruction_poses
        from .sim3_estimation import (
            compute_alignment_error,
            compute_rotation_error,
            estimate_sim3_from_poses,
            estimate_sim3_robust,
            transform_reconstruction,
        )

        scale, R, t, matched = estimate_sim3_robust(
            result.reconstruction,
            self.rtabmap_poses,
            self.image_to_node_id,
            max_error=1.0,
            min_common_images=5,
            verbose=self.config.verbose,
        )
        method = "ransac_sim3"

        if scale is None:
            method = "least_squares_sim3_fallback"
            hloc_poses = get_reconstruction_poses(result.reconstruction)
            print(f"  RANSAC failed, falling back to direct least-squares")
            print(
                f"  HLOC poses: {len(hloc_poses)}, RTABMap poses: {len(self.rtabmap_poses)}"
            )
            scale, R, t, matched = estimate_sim3_from_poses(
                hloc_poses,
                self.rtabmap_poses,
                self.image_to_node_id,
                min_correspondences=3,
                verbose=self.config.verbose,
            )
            if scale is None:
                raise RuntimeError("Sim3 alignment failed in both RANSAC and least-squares fallback")

        R, t = np.asarray(R, dtype=float), np.asarray(t, dtype=float)
        if (not np.isfinite(scale) or scale <= 0 or R.shape != (3, 3) or t.shape != (3,)
                or not np.isfinite(R).all() or not np.isfinite(t).all()
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5)):
            raise RuntimeError("Sim3 estimator returned a nonfinite or invalid transform")

        print(f"Estimated Sim3 transformation from {len(matched)} correspondences:")
        print(f"  Scale: {scale:.6f}")
        print(f"  Rotation:\n{R}")
        print(f"  Translation: {t}")

        hloc_poses = get_reconstruction_poses(result.reconstruction)
        mean_err, max_err, _ = compute_alignment_error(
            hloc_poses, self.rtabmap_poses, self.image_to_node_id, scale, R, t
        )

        mean_rot_err, max_rot_err, _ = compute_rotation_error(
            hloc_poses, self.rtabmap_poses, self.image_to_node_id, R
        )
        if not np.isfinite([mean_err, max_err, mean_rot_err, max_rot_err]).all():
            raise RuntimeError("Sim3 alignment diagnostics are nonfinite")

        print(f"\nAlignment error after Sim3:")
        print(f"  Position - Mean: {mean_err:.4f} m, Max: {max_err:.4f} m")
        print(f"  Rotation - Mean: {mean_rot_err:.2f}°, Max: {max_rot_err:.2f}°")

        # Apply transformation to reconstruction
        transform_reconstruction(result.reconstruction, scale, R, t)
        print("\nApplied Sim3 transformation to reconstruction")

        # Save transformed reconstruction
        aligned_sfm_dir = self.output_dir / "sfm_aligned"
        aligned_sfm_dir.mkdir(parents=True, exist_ok=True)
        result.reconstruction.write(aligned_sfm_dir)
        print(f"Saved aligned reconstruction to: {aligned_sfm_dir}")

        # Store results
        result.scale = scale
        result.rotation = R
        result.translation = t
        result.alignment_error_mean = mean_err
        result.alignment_error_max = max_err
        result.alignment_status = "aligned"
        result.alignment_method = method

    def _print_summary(self, result: PipelineResult):
        """Print pipeline summary"""
        print("\n" + "=" * 60)
        print("Pipeline Summary")
        print("=" * 60)
        print(f"Input database: {self.config.rtabmap_db}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nImages:")
        print(f"  Exported: {result.num_images_exported}")
        print(f"  Registered: {result.num_images_registered}")
        print(
            f"  Registration rate: {100 * result.num_images_registered / max(1, result.num_images_exported):.1f}%"
        )
        print(f"\n3D Points: {result.num_points3d}")

        if result.scale is not None:
            print(f"\nSim3 Alignment:")
            print(f"  Scale (HLOC -> RTABMap): {result.scale:.6f}")
            print(f"  Mean alignment error: {result.alignment_error_mean:.4f} m")
            print(f"  Max alignment error: {result.alignment_error_max:.4f} m")


def run_pipeline(
    rtabmap_db: str,
    output_dir: str,
    feature_conf: str = "aliked-n16",
    retrieval_conf: str = "megaloc",
    matcher_conf: str = "aliked+lightglue",
    num_matched: int = 20,
    verbose: bool = False,
) -> PipelineResult:
    """
    Convenience function to run the pipeline.

    Args:
        rtabmap_db: Path to RTABMap database
        output_dir: Output directory
        feature_conf: HLOC feature configuration
        retrieval_conf: HLOC retrieval configuration
        matcher_conf: HLOC matcher configuration
        num_matched: Number of image pairs to match
        verbose: Enable verbose output

    Returns:
        PipelineResult with reconstruction and alignment data
    """
    config = PipelineConfig(
        rtabmap_db=rtabmap_db,
        output_dir=output_dir,
        feature_conf=feature_conf,
        retrieval_conf=retrieval_conf,
        matcher_conf=matcher_conf,
        num_matched=num_matched,
        verbose=verbose,
    )

    pipeline = RTABMapHLOCPipeline(config)
    return pipeline.run()
