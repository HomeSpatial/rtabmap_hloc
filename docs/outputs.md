# Outputs and coordinate semantics

## Native assets

`sfm_aligned/` contains COLMAP `cameras.bin`, `images.bin`, `points3D.bin`, and
`rigs.bin` / `frames.bin` when written by the installed pycolmap. These can be
loaded directly by COLMAP-compatible tools. `images/` contains copied photos.
`hloc/` retains features, retrieval descriptors, pairs, matches and the raw SfM
model. `reference_model/` is used by the existing-pose triangulation path.

`reconstruction.json` is a reconstruction-description v1 document validated
against the [included schema](../src/rtabmap_hloc_reprocess/_exchange/reconstruction.schema.json).
It references the aligned sparse model, **registered** photos, metadata and
alignment report with file hashes. Unregistered exported photos can remain in
`images/` but are not declared as registered coverage. Keep the relative file
layout intact when transferring the output; input database access is not needed
to read the published assets.

The output coordinate system is right-handed, uses metres and identifies the
RTAB-Map local world frame. The default world up declaration is `+Z`.
`--up-axis` changes the declaration only. COLMAP camera transforms use the
optical camera convention; the reader accounts for camera-local transforms
when exporting the RTAB-Map reference model.

## Metadata and quality

`metadata.json` currently has `schema_version: 2`:

| Field/group | Meaning |
| --- | --- |
| `world_frame`, `sim3_scale` | Output frame and estimated similarity scale |
| `alignment_status`, `alignment_method` | Whether/how alignment completed |
| `alignment_error_mean_m`, `alignment_error_max_m` | Residuals against input poses |
| `registered_images`, `total_images` | Registered and exported photo counts |
| `camera_intrinsics` | Source, reconstruction mode and refinement policy |
| `colmap_profile`, `colmap_options` | Requested and effective geometry settings |
| `ba_metrics` | Available pycolmap reprojection/track metrics |
| `feature_conf`, `retrieval_conf`, `matcher_conf` | HLOC configuration names |
| `source_db_sha256` | Database content SHA-256; for directory input, a digest of relative paths and sizes, **not** a content checksum |
| `sfm_model_dir`, feature/match paths | Paths relative to the result root |

`alignment-report.json` and `run-status.json` explicitly record quality as
`not_evaluated`. Alignment residuals compare against the input trajectory,
which may itself drift. They are not independent ground truth.

For full SfM, alignment tries RANSAC Sim3 and then a least-squares fallback.
The reported method distinguishes these paths. Both failing, invalid
transforms, no registered images, or missing published assets make the run
fail. Existing-pose triangulation reports `input_poses`, scale 1, and zero
alignment residuals by construction; those zeros are not measured accuracy.

## Exit status and retry

The CLI returns 0 after successful result publication, 1 on a pipeline or
bundle-publication failure, and 2 for invalid CLI arguments. Failure after
pipeline execution starts writes `run-status.json` when the output is writable.
Input/argument errors can happen before a status file exists.

A result containing `reconstruction.json` is immutable: re-running into it
fails without rewriting it. Use another output directory or bundle stage.
An unfinished full-SfM run can reuse exported photos and an existing raw model:

```bash
rtabmap-hloc input.db -o output --skip-image-export --skip-hloc
```

This requires compatible input/configuration and `output/hloc/sfm`; it is not a
checkpoint system for interrupted neural inference. `--use-existing-poses`
selects triangulation and takes precedence over `--skip-hloc`. Preserve original
inputs while resuming and use a fresh run if their content has changed.

## Scene bundles and downstream tools

```bash
rtabmap-hloc input.db --bundle house --stage sfm
rtabmap-hloc input.db --bundle house --stage sfm-r2
```

Default work paths are `house/work/<stage>`; immutable snapshots are stored at
`house/stages/<stage>`. Duplicate stage names cannot be overwritten. The first
run derives `scene_id` from the bundle directory unless provided, and later
stages inherit it. The bundle has its own hash-checked asset lock.

Consumers such as Restruct or a compatible remap installation can select the
native photo and camera assets from a description/bundle. No downstream tool is
required to create these outputs. If bundle publication fails, the CLI returns
nonzero and retains the standalone result. With a compatible Restruct install,
publication can be retried without reconstructing:

```bash
restruct bundle append house --result house/work/sfm/reconstruction.json --stage sfm --operation rtabmap-hloc
```

Select the intended stage/camera asset explicitly when a bundle contains
multiple reconstructions. Do not assume the last model is the preferred one.
