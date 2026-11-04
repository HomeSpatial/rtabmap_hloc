"""Explicit, fixed-intrinsics SfM relocalization; outputs candidates, not acceptance.

The geometry uses fixed-camera, pose-only repair. No frame/edge
removal, BA, Sim3, inference service, or automatic pipeline promotion is used.
Requires binary COLMAP models with undistorted PINHOLE/SIMPLE_PINHOLE cameras
and a matching database. ``image_order`` is acquisition order, not image-ID order.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
from itertools import product
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import numpy as np
import pycolmap

MAX_ID = 2147483647

# Unchanged geometry settings from the measured pose-only experiment.
_POLICY = {
    "point_support": {
        "minimum_distinct_external_views": 3,
        "maximum_reprojection_px": 2.0,
        "minimum_parallax_deg": 2.0,
    },
    "correspondence_support": {
        "minimum_matching_external_views": 2,
        "local_capture_window": 12,
    },
}


def _sampson(matches, points_a, points_b, fundamental):
    if not len(matches):
        return np.empty(0)
    x = np.c_[points_a[matches[:, 0]], np.ones(len(matches))]
    y = np.c_[points_b[matches[:, 1]], np.ones(len(matches))]
    fx, fty = x @ fundamental.T, y @ fundamental
    return np.abs(np.sum(y * fx, axis=1)) / np.sqrt(
        np.maximum(
            np.sum(fx[:, :2] ** 2, axis=1) + np.sum(fty[:, :2] ** 2, axis=1), 1e-30
        )
    )


def _skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def _independent_points(model, targets, policy):
    """Retriangulate external observations at fixed poses; exclude suspect views."""
    from scipy.linalg import svd

    eligible = {}
    rejected = Counter()
    external_counts = Counter()
    for pid, pt in model.points3D.items():
        by_view = defaultdict(list)
        for obs in pt.track.elements:
            if obs.image_id not in targets:
                by_view[obs.image_id].append(obs.point2D_idx)
        # Multiple keypoints of one image are not independent support votes.
        observations = [(i, v[0]) for i, v in by_view.items() if len(v) == 1]
        external_counts[len(observations)] += 1
        if len(observations) < policy["minimum_distinct_external_views"]:
            rejected["too_few_distinct_external_views"] += 1
            continue
        rows = []
        centers = []
        for i, idx in observations:
            im = model.images[i]
            camera = model.cameras[im.camera_id]
            xy = im.points2D[idx].xy
            uv = camera.cam_from_img(xy)
            pose = im.cam_from_world().matrix()
            rows.extend((uv[0] * pose[2] - pose[0], uv[1] * pose[2] - pose[1]))
            centers.append(im.projection_center())
        _, _, vt = svd(np.array(rows), full_matrices=False)
        if abs(vt[-1, 3]) < 1e-12:
            rejected["point_at_infinity"] += 1
            continue
        xyz = vt[-1, :3] / vt[-1, 3]
        rays = xyz - np.array(centers)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        angle = float(np.rad2deg(np.arccos(np.clip(rays @ rays.T, -1, 1))).max())
        errors = []
        positive = True
        for i, idx in observations:
            im = model.images[i]
            cam = im.cam_from_world() * xyz
            if cam[2] <= 0:
                positive = False
                break
            xy = model.cameras[im.camera_id].img_from_cam(cam)
            errors.append(float(np.linalg.norm(xy - im.points2D[idx].xy)))
        if (
            not positive
            or not errors
            or max(errors) > policy["maximum_reprojection_px"]
        ):
            rejected["external_projection_or_cheirality"] += 1
            continue
        if angle < policy["minimum_parallax_deg"]:
            rejected["low_parallax"] += 1
            continue
        eligible[pid] = {"xyz": xyz, "views": set(by_view), "angle": angle}
    return eligible, {
        "eligible": len(eligible),
        "rejected": dict(rejected),
        "distinct_external_support_histogram": {
            str(count): n for count, n in external_counts.items()
        },
    }


def _collect_correspondences(fid, family, model, eligible, ordered, points, geos, p):
    rank = {i: j for j, i in enumerate(ordered)}
    votes = defaultdict(lambda: defaultdict(set))
    target_set = set(p["targets"])
    for (a, b), geometry in geos.items():
        if fid not in (a, b):
            continue
        other = b if fid == a else a
        if other in target_set:
            continue
        if (
            family == "local"
            and abs(rank[fid] - rank[other])
            > p["correspondence_support"]["local_capture_window"]
        ):
            continue
        matches = geometry.inlier_matches
        if fid == b:
            matches = matches[:, ::-1]
        im = model.images[other]
        for q_idx, ref_idx in matches:
            pt = im.points2D[int(ref_idx)]
            if pt.has_point3D() and pt.point3D_id in eligible:
                votes[int(q_idx)][int(pt.point3D_id)].add(other)
    correspondences = []
    conflicts = 0
    for idx, options in sorted(votes.items()):
        ranked = sorted(options.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
            conflicts += 1
            continue
        pid, views = ranked[0]
        if len(views) >= p["correspondence_support"]["minimum_matching_external_views"]:
            correspondences.append((idx, pid, sorted(views)))
    # One physical 3D point supplies only one image observation to PnP.
    by_point = defaultdict(list)
    for item in correspondences:
        by_point[item[1]].append(item)
    selected = [
        sorted(v, key=lambda x: (-len(x[2]), x[0]))[0] for v in by_point.values()
    ]
    selected.sort()
    return selected, {
        "query_keypoints_with_candidates": len(votes),
        "ambiguous_keypoints_removed": conflicts,
        "unique_correspondences": len(selected),
    }


def _track_components(model):
    parent = {i: i for i in model.reg_image_ids()}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pt in model.points3D.values():
        views = sorted({obs.image_id for obs in pt.track.elements})
        for other in views[1:]:
            parent[find(other)] = find(views[0])
    groups = defaultdict(list)
    for i in parent:
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _load(model_dir, database_path):
    model_dir, database_path = Path(model_dir).resolve(), Path(database_path).resolve()
    files = [model_dir / (name + ".bin") for name in ("cameras", "images", "points3D")]
    files += [
        p for name in ("rigs", "frames") if (p := model_dir / (name + ".bin")).exists()
    ]
    identity = hashlib.sha256()
    for path in [*files, database_path]:
        with path.open("rb") as source:
            identity.update(hashlib.file_digest(source, "sha256").digest())
    model = pycolmap.Reconstruction(model_dir)
    if any(
        c.model_name not in ("PINHOLE", "SIMPLE_PINHOLE")
        for c in model.cameras.values()
    ):
        raise ValueError("Relocalization requires undistorted pinhole cameras")
    active = set(model.reg_image_ids())
    with closing(sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        points = {
            i: np.frombuffer(data, np.float32).reshape(rows, cols)[:, :2].astype(float)
            for i, rows, cols, data in db.execute(
                "SELECT image_id, rows, cols, data FROM keypoints"
            )
            if i in active
        }
        geos = {
            (pair // MAX_ID, pair % MAX_ID): SimpleNamespace(
                inlier_matches=np.frombuffer(data, np.uint32).reshape(rows, cols)
            )
            for pair, rows, cols, data in db.execute(
                "SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows>0"
            )
            if pair // MAX_ID in active and pair % MAX_ID in active
        }
    for i in active:
        im = model.images[i]
        if (
            names.get(i) != im.name
            or i not in points
            or len(points[i]) != len(im.points2D)
        ):
            raise ValueError(f"Database and model image/keypoint identity differ: {i}")
    return model, points, geos, identity.hexdigest()


def _pair_metrics(model, points, geos, poses=None):
    poses = poses or {}
    result = []
    for (a, b), geometry in geos.items():
        relative = (
            poses.get(b, model.images[b].cam_from_world())
            * poses.get(a, model.images[a].cam_from_world()).inverse()
        )
        ka = model.cameras[model.images[a].camera_id].calibration_matrix()
        kb = model.cameras[model.images[b].camera_id].calibration_matrix()
        fundamental = (
            np.linalg.inv(kb).T
            @ _skew(relative.translation)
            @ relative.rotation.matrix()
            @ np.linalg.inv(ka)
        )
        errors = _sampson(geometry.inlier_matches, points[a], points[b], fundamental)
        result.append(
            {
                "image_a": a,
                "image_b": b,
                "inliers": len(errors),
                "median_sampson_px": float(np.median(errors)),
                "p95_sampson_px": float(np.percentile(errors, 95)),
            }
        )
    return result


def _metrics(model, points, geos):
    return {
        "registered_frames": model.num_reg_images(),
        "points3d": model.num_points3D(),
        "components": len(_track_components(model)),
        "training_pairs": _pair_metrics(model, points, geos),
    }


def _evidence(model, images_dir):
    return [
        {
            "image_id": i,
            "name": im.name,
            "rgb_path": (
                str((Path(images_dir) / im.name).resolve()) if images_dir else None
            ),
        }
        for i, im in sorted(model.images.items())
    ]


def inspect_sfm(model_dir, database_path, *, images_dir=None) -> dict:
    """Return JSON-safe diagnostics and an input ID for a subsequent repair.

    Pair residuals use database training inliers; they are not held-out accuracy
    or sufficient evidence to delete a relation. RGB files are linked on demand.
    """
    model, points, geos, identity = _load(model_dir, database_path)
    return {
        "schema": 1,
        "input_id": identity,
        "status": "diagnostic_only",
        "model_dir": str(Path(model_dir).resolve()),
        "database_path": str(Path(database_path).resolve()),
        "images": _evidence(model, images_dir),
        "metrics": _metrics(model, points, geos),
    }


def _candidates(model, points, geos, targets, image_order):
    policy = {**_POLICY, "targets": targets}
    eligible, support = _independent_points(
        model, set(targets), policy["point_support"]
    )
    candidates = {}
    for fid in targets:
        candidates[fid] = []
        for family in ("local", "all"):
            matches, stats = _collect_correspondences(
                fid, family, model, eligible, image_order, points, geos, policy
            )
            row = {"family": family, "status": "insufficient_support", **stats}
            candidates[fid].append(row)
            if len(matches) < 20:
                continue
            camera = pycolmap.Camera(
                model.cameras[model.images[fid].camera_id].todict()
            )
            xy = np.array([points[fid][i] for i, _, _ in matches])
            xyz = np.array([eligible[pid]["xyz"] for _, pid, _ in matches])
            fit = pycolmap.estimate_and_refine_absolute_pose(
                xy,
                xyz,
                camera,
                estimation_options={
                    "estimate_focal_length": False,
                    "ransac": {
                        "max_error": 4.0,
                        "min_inlier_ratio": 0.1,
                        "max_num_trials": 20000,
                        "random_seed": 42,
                    },
                },
                refinement_options={
                    "refine_focal_length": False,
                    "refine_extra_params": False,
                },
            )
            if fit is None or fit["num_inliers"] < 20:
                row["status"] = "pnp_failed"
                continue
            row.update(
                status="estimated",
                inliers=int(fit["num_inliers"]),
                pose=fit["cam_from_world"],
                matches=matches,
                mask=np.asarray(fit["inlier_mask"], bool),
            )
    valid = [[r for r in candidates[f] if r["status"] == "estimated"] for f in targets]
    chosen, best_score = None, None
    # Preserve the experiment's ranking: training pair votes, then PnP inliers.
    training = {
        pair: geo
        for pair, geo in geos.items()
        if set(pair) & set(targets) and len(geo.inlier_matches) >= 50
    }
    for combo in product(*valid):
        poses = {f: row["pose"] for f, row in zip(targets, combo)}
        checks = _pair_metrics(model, points, training, poses)
        score = (
            sum(r["median_sampson_px"] <= 3 for r in checks),
            sum(r["inliers"] for r in combo),
        )
        if best_score is None or score > best_score:
            chosen, best_score = combo, score
    attempts = {
        str(fid): [
            {k: v for k, v in row.items() if k not in ("pose", "matches", "mask")}
            for row in rows
        ]
        for fid, rows in candidates.items()
    }
    return chosen, eligible, {"point_support": support, "attempts": attempts}


def relocalize(
    model_dir,
    database_path,
    frame_ids,
    output_dir,
    *,
    image_order,
    expected_input_id,
    images_dir=None,
) -> dict:
    """Write one pose-only candidate plus result.json; never replace the input.

    IDs refer to COLMAP images. ``image_order`` must list all registered images
    in acquisition order. ``expected_input_id`` comes from inspect_sfm, avoiding
    stale AI/human requests. Empty targets are a no-op. Unsupported evidence
    leaves candidate_model null; neither outcome constitutes global acceptance.
    """
    started = time.perf_counter()
    model, points, geos, identity = _load(model_dir, database_path)
    if identity != expected_input_id:
        raise ValueError("Input changed since inspection; inspect the current model")
    targets, image_order = list(frame_ids), list(image_order)
    active = set(model.reg_image_ids())
    if len(targets) != len(set(targets)) or not set(targets) <= active:
        raise ValueError("Targets must be unique registered image IDs")
    if len(image_order) != len(active) or set(image_order) != active:
        raise ValueError("image_order must contain every registered image exactly once")
    frame_sizes = Counter(im.frame_id for im in model.images.values())
    if any(frame_sizes[model.images[i].frame_id] != 1 for i in targets):
        raise ValueError("Relocalization currently supports one image per target frame")
    output = Path(output_dir).resolve()
    # A separate new directory also prevents overwriting another attempt.
    output.mkdir(parents=True, exist_ok=False)
    before = _metrics(model, points, geos)
    poses = {i: im.cam_from_world().matrix().copy() for i, im in model.images.items()}
    params = {i: c.params.copy() for i, c in model.cameras.items()}
    result = {
        "schema": 1,
        "input_id": identity,
        "status": "no_change",
        "reason": "No target frames requested",
        "diagnostic_only": True,
        "candidate_model": None,
        "target_frames": targets,
        "image_order": image_order,
        "model_dir": str(Path(model_dir).resolve()),
        "database_path": str(Path(database_path).resolve()),
        "images": _evidence(model, images_dir),
        "before": before,
    }
    if targets:
        chosen, eligible, evidence = _candidates(
            model, points, geos, targets, image_order
        )
        result.update(evidence)
        if chosen is None:
            result.update(
                status="insufficient_support",
                reason="Could not estimate every requested pose from external views",
            )
        else:
            for i in targets:
                model.deregister_frame(model.images[i].frame_id)
            for i, candidate in zip(targets, chosen):
                im = model.images[i]
                model.frames[im.frame_id].set_cam_from_world(
                    im.camera_id, candidate["pose"]
                )
                model.register_frame(im.frame_id)
                for (idx, pid, _), keep in zip(candidate["matches"], candidate["mask"]):
                    if keep and model.exists_point3D(pid):
                        model.points3D[pid].xyz = eligible[pid]["xyz"]
                        model.add_observation(pid, pycolmap.TrackElement(i, idx))
            model.update_point_3d_errors()
            native = output / "sfm"
            native.mkdir()
            model.write(native)
            result.update(
                status="candidate_created",
                reason="Pose-only relocalization completed",
                candidate_model=str(native),
                selected_families={
                    str(i): c["family"] for i, c in zip(targets, chosen)
                },
            )
    after = _metrics(model, points, geos)
    result.update(
        after=after,
        delta={
            **{
                k: after[k] - before[k]
                for k in ("registered_frames", "points3d", "components")
            },
            "training_pairs": [
                {
                    "image_a": a["image_a"],
                    "image_b": a["image_b"],
                    **{k: a[k] - b[k] for k in ("median_sampson_px", "p95_sampson_px")},
                }
                for b, a in zip(before["training_pairs"], after["training_pairs"])
            ],
        },
        max_intrinsics_change=max(
            (
                float(np.abs(c.params - params[i]).max())
                for i, c in model.cameras.items()
            ),
            default=0.0,
        ),
        max_untargeted_pose_change=max(
            (
                float(np.abs(im.cam_from_world().matrix() - poses[i]).max())
                for i, im in model.images.items()
                if i not in targets
            ),
            default=0.0,
        ),
        elapsed_s=time.perf_counter() - started,
    )
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "relocalize"))
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("database_path", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--frames", type=int, nargs="*", default=[])
    parser.add_argument(
        "--image-order", type=Path, help="JSON array of image IDs in acquisition order"
    )
    parser.add_argument("--input-id", help="input_id from inspect")
    parser.add_argument("--output", type=Path, help="New candidate output directory")
    args = parser.parse_args()
    if args.action == "inspect":
        result = inspect_sfm(
            args.model_dir, args.database_path, images_dir=args.images_dir
        )
    else:
        if args.image_order is None or args.input_id is None or args.output is None:
            parser.error("relocalize requires --image-order, --input-id and --output")
        result = relocalize(
            args.model_dir,
            args.database_path,
            args.frames,
            args.output,
            image_order=json.loads(args.image_order.read_text()),
            expected_input_id=args.input_id,
            images_dir=args.images_dir,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
