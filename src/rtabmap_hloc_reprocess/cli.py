"""
Command Line Interface for RTABMap-HLOC Reprocessing Pipeline
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert RTABMap database to HLOC format and perform sparse reconstruction"
    )

    parser.add_argument(
        "rtabmap_db",
        type=str,
        help="Path to RTABMap database file (.db) or rtabmap-export directory"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for reconstruction results"
    )

    parser.add_argument(
        "--inference-backend", choices=["torch", "colab"], default="torch",
        help="Run HLOC models locally or in Colab; COLMAP remains on this host",
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
        type=int,
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

    args = parser.parse_args()

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
    )

    pipeline = RTABMapHLOCPipeline(config)
    result = pipeline.run()

    if result.success:
        print("\n✓ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
