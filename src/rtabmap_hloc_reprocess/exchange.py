"""Publish portable native-file descriptions after successful metric reconstruction."""
import json
import os
from pathlib import Path
from uuid import uuid4

from ._exchange.verify import check, resolve_asset, sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_run_status(pipeline, result):
    atomic_json(pipeline.output_dir / "run-status.json", dict(
        execution="completed" if result.success else "failed", success=result.success,
        error=result.error, alignment=result.alignment_status,
        description=result.description_path or None,
        quality={"status": "not_evaluated"}))


def export_description(pipeline, result):
    root = pipeline.output_dir.resolve()
    if (root / "reconstruction.json").exists():
        raise ValueError("Refusing to replace an existing reconstruction snapshot")
    if result.alignment_status not in ("aligned", "input_poses"):
        raise ValueError("Cannot publish a metric description without completed alignment")
    model = result.reconstruction
    ids = set(model.reg_image_ids())
    rows = sorted((row for key, row in model.images.items() if key in ids), key=lambda row: row.name)
    if not rows:
        raise ValueError("No registered photos to publish")
    def file(relative, role):
        path = resolve_asset({"root": "local", "path": relative}, {"local": root})
        if not path.is_file():
            raise ValueError(f"Missing output asset: {relative}")
        return {"role": role, "ref": {"root": "local", "path": relative, "sha256": sha256(path)}}
    native = [file(f"sfm_aligned/{name}.bin", name) for name in ("cameras", "images", "points3D")]
    for role in ("rigs", "frames"):
        if (root / "sfm_aligned" / (role + ".bin")).is_file():
            native.append(file(f"sfm_aligned/{role}.bin", role))
    photos = [file("images/" + row.name, "image") for row in rows]
    quality = dict(status="not_evaluated", scope="Alignment completion is not geometric quality acceptance",
                   method=result.alignment_method, alignment_status=result.alignment_status,
                   sim3_scale=result.scale, position_error_mean_m=result.alignment_error_mean,
                   position_error_max_m=result.alignment_error_max,
                   registered_images=len(rows), exported_images=result.num_images_exported)
    atomic_json(root / "alignment-report.json", quality)
    scene_id = pipeline.config.scene_id or root.name
    document = dict(kind="reconstruction_description", schema_version=1, scene_id=scene_id,
        packaging="self_contained",
        coordinate_systems=[dict(id="rtabmap_local", handedness="right", up_axis=pipeline.config.up_axis, metres_per_unit=1)],
        assets=[dict(id="cameras", type="sparse_reconstruction", format="colmap.sparse.binary",
                     coordinate_system="rtabmap_local", files=native),
                dict(id="photos", type="image_collection", format="image.files", files=photos,
                     coverage=dict(scope="registered COLMAP image names", status="full", reference_asset="cameras")),
                dict(id="validation", type="validation_report", format="json",
                     files=[file("metadata.json", "metadata"), file("alignment-report.json", "alignment")])],
        runs=[dict(id="sfm", producer={"name": "rtabmap-hloc", "version": "0.1.0"}, inputs=["photos"], outputs=["cameras"])],
        extensions={"org.rtabmap_hloc.provenance": dict(quality=quality,
            frame_names=[row.name for row in rows], frame_ids=[str(row.image_id) for row in rows],
            source_up="declared RTAB-Map world axis; physical orientation not independently measured")})
    destination = root / "reconstruction.json"
    candidate = root / f".reconstruction-{uuid4().hex}.json"
    try:
        atomic_json(candidate, document)
        check(candidate)
        if destination.exists():
            raise ValueError("Refusing to replace an existing reconstruction snapshot")
        os.replace(candidate, destination)
    finally:
        candidate.unlink(missing_ok=True)
    print(f"Portable reconstruction description: {destination}")
    return destination
