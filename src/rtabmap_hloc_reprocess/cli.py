"""
Command Line Interface for RTABMap-HLOC Reprocessing Pipeline
"""

import argparse
import sys
from pathlib import Path
from jsonschema import ValidationError


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="rtabmap-hloc",
        description="Convert RTABMap database to HLOC format and perform sparse reconstruction"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument(
        "rtabmap_db",
        type=str,
        help="Path to RTABMap database file (.db) or rtabmap-export directory"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output directory for reconstruction results"
    )

    parser.add_argument(
        "--inference-backend", choices=["torch", "colab"], default="torch",
        help="Run HLOC locally or with a separately installed remap Colab adapter; COLMAP stays local",
    )
    parser.add_argument(
        "--hloc-execution-profile", choices=["colab-models-l4-v1", "colab-models-l4-v2"],
        default="colab-models-l4-v1",
    )
    parser.add_argument(
        "--feature-conf",
        type=str,
        default="aliked-n16",
        choices=["superpoint_aachen", "superpoint_max", "disk", "aliked-n16", "sift"],
        help="Feature extractor configuration (default: aliked-n16)"
    )

    parser.add_argument(
        "--retrieval-conf",
        type=str,
        default="megaloc",
        choices=["netvlad", "openibl", "dir", "megaloc"],
        help="Global descriptor configuration for retrieval (default: megaloc)"
    )

    parser.add_argument(
        "--matcher-conf",
        type=str,
        default="aliked+lightglue",
        choices=["superpoint+lightglue", "disk+lightglue", "aliked+lightglue",
                 "superglue", "NN-superpoint", "NN-ratio"],
        help="Feature matcher configuration (default: aliked+lightglue)"
    )

    parser.add_argument(
        "--num-matched",
        type=_positive_int,
        default=20,
        help="Number of image pairs to match per image (default: 20)"
    )

    parser.add_argument(
        "--colmap-profile",
        default="default",
        choices=["default", "high_accuracy"],
        help="Named COLMAP option profile for HLOC reconstruction",
    )

    parser.add_argument(
        "--skip-image-export",
        action="store_true",
        help="Skip image export (use existing images in output/images)"
    )

    parser.add_argument(
        "--skip-hloc",
        action="store_true",
        help="Skip HLOC reconstruction (use existing sfm result in output/hloc/sfm)"
    )

    parser.add_argument(
        "--use-existing-poses",
        action="store_true",
        help="Use existing RTABMap poses for triangulation instead of full SfM reconstruction"
    )

    parser.add_argument(
        "--no-rtabmap-intrinsics",
        action="store_true",
        help=(
            "Disable RTABMap per-frame intrinsics in full SfM and let COLMAP "
            "read/estimate camera intrinsics from images"
        ),
    )

    parser.add_argument(
        "--refine-intrinsics",
        action="store_true",
        help="Allow COLMAP bundle adjustment to refine RTABMap intrinsics in full SfM",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--map-id",
        default="",
        help="Map identifier written to metadata.json (provenance for build_map_pack)",
    )

    parser.add_argument(
        "--camera-mode",
        default="AUTO",
        choices=["AUTO_SINGLE", "AUTO", "PER_FOLDER", "PER_IMAGE", "SINGLE"],
        help="pycolmap CameraMode used during SfM (recorded in metadata.json)",
    )

    parser.add_argument("--scene-id", default="", help="Portable scene identity; defaults to output directory name")
    parser.add_argument("--bundle", type=Path, help="Append successful SfM to this bundle; derives output and scene ID")
    parser.add_argument("--stage", default="sfm", help="New immutable bundle stage (default: sfm)")
    parser.add_argument("--up-axis", default="+Z", choices=["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
                        help="Declared RTAB-Map world up axis; default +Z. Does not rotate geometry.")
    args = parser.parse_args()
    stage_plan = None
    if args.bundle:
        from ._exchange.bundle import plan
        try:
            stage_plan = plan(args.bundle, args.stage, "rtabmap-hloc", scene_id=args.scene_id, out=args.output)
            args.output, args.scene_id = stage_plan["output"], stage_plan["scene_id"]
        except (ValueError, OSError, ValidationError) as exc:
            parser.error(str(exc))
    elif not args.output:
        parser.error("--output or --bundle is required")

    # Validate input
    db_path = Path(args.rtabmap_db)
    if not db_path.exists():
        print(f"Error: Target path not found: {db_path}")
        sys.exit(1)

    # Run pipeline
    from .pipeline import PipelineConfig, RTABMapHLOCPipeline

    config = PipelineConfig(
        rtabmap_db=str(db_path),
        output_dir=args.output,
        feature_conf=args.feature_conf,
        inference_backend=args.inference_backend,
        hloc_execution_profile=args.hloc_execution_profile,
        retrieval_conf=args.retrieval_conf,
        matcher_conf=args.matcher_conf,
        num_matched=args.num_matched,
        colmap_profile=args.colmap_profile,
        skip_image_export=args.skip_image_export,
        skip_hloc=args.skip_hloc,
        use_existing_poses=args.use_existing_poses,
        use_rtabmap_intrinsics=not args.no_rtabmap_intrinsics,
        fix_intrinsics=not args.refine_intrinsics,
        verbose=args.verbose,
        map_id=args.map_id,
        camera_mode=args.camera_mode,
        scene_id=args.scene_id,
        up_axis=args.up_axis,
    )

    pipeline = RTABMapHLOCPipeline(config)
    result = pipeline.run()

    if result.success and stage_plan:
        from ._exchange.bundle import publish
        from .exchange import write_run_status
        try:
            publication = publish(stage_plan, result.description_path)
            print(f"Bundle stage published: {publication['stage']} -> {publication['bundle']}")
        except (ValueError, OSError, ValidationError) as exc:
            result.success = False
            result.error = f"Bundle publication failed; standalone result retained: {exc}"
            write_run_status(pipeline, result)
            print(result.error)

    if result.success:
        print("\n✓ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
