"""Offline v1 reconstruction validation; core dependency: jsonschema only."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from jsonschema import Draft202012Validator, ValidationError

SPATIAL_TYPES = {
    "sparse_reconstruction", "point_cloud", "mesh", "parametric_scene",
    "gaussian_splat", "voxel_grid", "radiance_field", "editable_scene", "scene",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite JSON number")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8-sig") as stream:
        result = json.load(stream, object_pairs_hook=pairs)
    finite(result)
    return result


def schema_check(data, name):
    schema = read_json(Path(__file__).with_name(name))
    Draft202012Validator(schema).validate(data)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_asset(ref, roots):
    root, path = ref.get("root"), ref.get("path")
    require(isinstance(root, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", root),
            "Invalid logical root")
    require(isinstance(path, str) and path and not path.startswith("/")
            and not any(c in path for c in ("\\", ":", "\0"))
            and all(p not in ("", ".", "..") for p in path.split("/")),
            "Invalid relative asset path")
    require(root in roots, f"Unbound logical root: {root}")
    base = Path(roots[root]).resolve()
    result = base.joinpath(*path.split("/")).resolve()
    require(result.is_relative_to(base), "Asset escapes its logical root")
    return result


def asset_refs(node):
    if isinstance(node, dict):
        if "root" in node and "path" in node:
            yield node
        else:
            for key, value in node.items():
                if key != "extensions":
                    yield from asset_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from asset_refs(value)


def unique_index(rows, label):
    result = {row["id"]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {label} ID")
    return result


def similarity_scale(matrix):
    if (not isinstance(matrix, list) or len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)):
        return None
    values = [v for row in matrix for v in row]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in values):
        return None
    if any(abs(matrix[3][i] - v) > 1e-8 for i, v in enumerate([0, 0, 0, 1])):
        return None
    r = [row[:3] for row in matrix[:3]]
    scale2 = sum(r[j][0] ** 2 for j in range(3))
    if scale2 <= 0 or not math.isfinite(scale2):
        return None
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot / scale2 - (1 if i == j else 0)) > 1e-5:
                return None
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    return math.sqrt(scale2) if det > 0 and math.isfinite(det) else None


def rigid(matrix):
    scale = similarity_scale(matrix)
    return scale is not None and abs(scale - 1) <= 1e-5


def normalize_refs(node, document_ref):
    if isinstance(node, list):
        return [normalize_refs(item, document_ref) for item in node]
    if not isinstance(node, dict):
        return node
    if "root" in node and "path" in node:
        result = copy.deepcopy(node)
        resolve_asset(result, {result["root"]: Path.cwd()})
        if result["root"] == "local":
            result["root"] = document_ref["root"]
            result["path"] = str(
                PurePosixPath(document_ref["path"]).parent / result["path"])
        return result
    return {key: (copy.deepcopy(value) if key == "extensions" else
                  normalize_refs(value, document_ref)) for key, value in node.items()}


def optional_validate(document, adapters):
    if "indoor-recon" not in adapters:
        return "unsupported"
    try:
        from indoor_recon.contracts import validate
    except ImportError:
        return "unsupported"
    validate(document)
    return "passed"


def check_view(data, read_document, refs, roots, adapters):
    if "scene" not in data:
        require(not data.get("depth_sources"), "Depth lineage requires scene frames")
        return dict(frames=0, content_validation="not_requested",
                    input_eligibility=None)
    refs.append(data["scene"])
    scene = read_document(data["scene"])
    schema_check(scene, "input-view.schema.json")
    require(scene["kind"] == "scene" and scene["scene_id"] == data["scene_id"],
            "Scene kind or scene_id mismatch")
    spaces = unique_index(data["coordinate_systems"], "coordinate system")
    require("source" in spaces, "Scene view requires source coordinates")
    require(all(spaces["source"][k] == v
                for k, v in scene["source_coordinates"].items()),
            "Scene source coordinates disagree with registry")
    for geometry in scene["geometry"]:
        ref = geometry["asset"]
        require(any(a["type"] == geometry["type"]
                    and a.get("coordinate_system") == geometry["coordinate_frame"]
                    and any(f["ref"]["root"] == ref["root"]
                            and f["ref"]["path"] == ref["path"] for f in a["files"])
                    for a in data["assets"]), "Scene geometry missing matching asset")
        if geometry["coordinate_frame"] == "work":
            expected = dict(handedness="right", up_axis="+Z", metres_per_unit=1)
            require(all(spaces.get("work", {}).get(k) == v
                        for k, v in expected.items()), "Invalid work coordinates")
    refs.extend(asset_refs(scene))
    content = optional_validate(scene, adapters)
    frames = []
    for field in ("frames", "alignment"):
        if field not in scene:
            continue
        child = read_document(scene[field])
        schema_check(child, "input-view.schema.json")
        require(child["kind"] == field and child["scene_id"] == data["scene_id"],
                "Scene child kind or scene_id mismatch")
        refs.extend(asset_refs(child))
        optional_validate(child, adapters)
        if field == "frames":
            frames = child["frames"]
            unique_index(frames, "frame")
            for frame in frames:
                if "camera" in frame:
                    require(rigid(frame["camera"]["T_source_from_camera"]),
                            "Invalid camera pose")
        else:
            matrix = child["T_work_from_source_m"]
            require(rigid(matrix), "Invalid input alignment")
            up = scene["source_coordinates"]["up_axis"]
            if up != "unknown":
                axis, sign = "XYZ".index(up[1]), (1 if up[0] == "+" else -1)
                require(all(abs(matrix[i][axis] * sign - v) <= 1e-5
                            for i, v in enumerate([0, 0, 1])), "Invalid up alignment")
    depth = {f["id"] for f in frames if "depth" in f}
    declared = [i for g in data.get("depth_sources", []) for i in g["frame_ids"]]
    require(len(declared) == len(set(declared)) and set(declared) == depth,
            "Depth lineage must cover each depth frame exactly once")
    image_check = "unsupported"
    if "images" in adapters:
        try:
            from PIL import Image
        except ImportError:
            pass
        else:
            for frame in frames:
                for field in ("image", "depth"):
                    if field in frame:
                        ref = (frame[field] if field == "image"
                               else frame[field]["asset"])
                        with Image.open(resolve_asset(ref, roots)) as image:
                            require(list(image.size) == frame["size"],
                                    "Image dimensions disagree with frame")
            image_check = "passed"
    return dict(frames=len(frames), content_validation=content,
                reference_validation="passed", image_dimensions=image_check,
                input_eligibility=None)


def check(manifest, roots=None, *, adapters=(), profiles=(), require_content=()):
    manifest = Path(manifest).resolve()
    roots = dict(roots or {})
    require("local" not in roots, "local is reserved for document-relative references")
    roots.setdefault("manifest", manifest.parent)
    require(Path(roots["manifest"]).resolve() == manifest.parent,
            "manifest root must be the entry document directory")
    entry_ref = dict(root="manifest", path=manifest.name)
    raw = read_json(manifest)
    standalone = raw.get("kind") == "reconstruction_description"
    data = normalize_refs(raw, entry_ref) if standalone else raw
    schema_check(data, "reconstruction.schema.json" if standalone
                 else "bundle.schema.json")
    entry_digest = sha256(manifest)
    records, used_refs, locks = [], [], []

    def pinned(ref):
        path = resolve_asset(ref, roots)
        require(path.is_file(), f"Missing file: {ref['root']}:{ref['path']}")
        if "sha256" in ref:
            require(sha256(path) == ref["sha256"], f"Hash mismatch: {ref['path']}")
        return path

    def read_document(ref):
        return normalize_refs(read_json(pinned(ref)), ref)

    def load_lock(document):
        if "asset_lock" not in document:
            return None
        ref = document["asset_lock"]
        lock = read_document(ref)
        schema_check(lock, "asset-lock.schema.json")
        require(lock["scene_id"] == document["scene_id"], "Lock scene_id mismatch")
        table = {}
        for item in lock["files"]:
            require(item["root"] != "local", "Lock paths must be normalized")
            key = (item["root"], item["path"])
            require(key not in table, "Duplicate lock entry")
            path = pinned(item)
            require(path.stat().st_size == item["bytes"], "Locked byte count mismatch")
            table[key] = item
            used_refs.append(item)
        return table

    def document_record(document, docref, identity):
        references = [f["ref"] for a in document["assets"] for f in a["files"]]
        references.extend(r for t in document.get("coordinate_transforms", [])
                          for r in t["evidence"])
        view = check_view(document, read_document, references, roots, adapters)
        table = load_lock(document)
        records.append(dict(id=identity, data=document, ref=docref, refs=references,
                            lock=table, view=view))
        if table is not None:
            locks.append(table)
        used_refs.extend(references)
        if "asset_lock" in document:
            used_refs.append(document["asset_lock"])
        return records[-1]

    # Preserve the direct registry for its view, before importing namespaced assets.
    document_record(copy.deepcopy(data), entry_ref, "entry")
    description_statuses = []
    for rid, entry in unique_index(data.get("reconstructions", []),
                                   "reconstruction").items():
        require("/" not in rid, "Reconstruction ID must not contain slash")
        docref = entry["document"]
        pinned(docref)
        used_refs.append(docref)
        dependencies = normalize_refs(entry.get("dependencies", []), docref)
        used_refs.extend(dependencies)
        status = dict(id=rid, integrity="passed", content_validation="unsupported")
        if (entry["format"], entry["format_version"]) != (
                "reconstruction.description", "1"):
            require("dependencies" in entry,
                    "Unknown description format requires declared dependencies")
            description_statuses.append(status)
            continue
        document = read_document(docref)
        schema_check(document, "reconstruction.schema.json")
        require(document["scene_id"] == data["scene_id"],
                "Reconstruction scene_id mismatch")
        document_record(document, docref, rid)

        def scoped(value):
            require("/" not in value, "Description-local ID must not contain slash")
            return f"{rid}/{value}"

        for field in ("assets", "coordinate_systems", "runs"):
            unique_index(document.get(field, []), field)
            for row in document.get(field, []):
                item = copy.deepcopy(row)
                item["id"] = scoped(item["id"])
                if field == "assets":
                    if "coordinate_system" in item:
                        item["coordinate_system"] = scoped(item["coordinate_system"])
                    if "reference_asset" in item.get("coverage", {}):
                        c = item["coverage"]
                        c["reference_asset"] = scoped(c["reference_asset"])
                if field == "runs":
                    for role in ("inputs", "outputs"):
                        item[role] = [scoped(i) for i in item[role]]
                data.setdefault(field, []).append(item)
        for transform in document.get("coordinate_transforms", []):
            item = copy.deepcopy(transform)
            item["from"], item["to"] = scoped(item["from"]), scoped(item["to"])
            data.setdefault("coordinate_transforms", []).append(item)
        status["content_validation"] = "passed"
        description_statuses.append(status)

    spaces = unique_index(data["coordinate_systems"], "coordinate system")
    assets = unique_index(data["assets"], "asset")
    runs = unique_index(data.get("runs", []), "run")
    for record in records:
        coverage = record["data"].get("coverage", {})
        scope_assets = (assets if record["id"] == "entry" else
                        unique_index(record["data"]["assets"], "coverage asset"))
        if "reference_asset" in coverage:
            require(coverage["reference_asset"] in scope_assets,
                    "Unknown document coverage reference")
    if standalone:
        require(all("/" not in key for key in list(spaces) + list(assets) + list(runs)),
                "Description-local ID must not contain slash")
    pairs = set()
    for transform in data.get("coordinate_transforms", []):
        src, dst = transform["from"], transform["to"]
        require(src in spaces and dst in spaces and src != dst,
                "Invalid transform coordinate reference")
        pair = tuple(sorted((src, dst)))
        require(pair not in pairs, "Duplicate coordinate transform pair")
        pairs.add(pair)
        require(all(spaces[c]["handedness"] == "right" for c in (src, dst)),
                "Transform requires right-handed coordinates")
        if transform["units"] == "metres":
            require(all(spaces[c]["metres_per_unit"] is not None for c in (src, dst)),
                    "Metre transform requires known units")
        scale = similarity_scale(transform["T_to_from"])
        require(scale is not None, "Invalid similarity transform")
        if transform["kind"] == "rigid":
            require(abs(scale - 1) <= 1e-5, "Rigid transform contains scale")
    produced, graph = set(), {key: set() for key in assets}
    for run in runs.values():
        inputs, outputs = set(run["inputs"]), set(run["outputs"])
        require(inputs | outputs <= assets.keys(), "Run references unknown asset")
        require(not inputs & outputs, "Run must use new output asset IDs")
        require(not outputs & produced, "Asset has multiple producing runs")
        produced.update(outputs)
        for output in outputs:
            graph[output].update(inputs)
    remaining = set(graph)
    while remaining:
        ready = {key for key in remaining if not graph[key] & remaining}
        require(ready, "Cycle in run provenance")
        remaining -= ready
    statuses = []
    for asset in assets.values():
        if asset["type"] in SPATIAL_TYPES:
            require("coordinate_system" in asset, "Spatial asset needs coordinates")
        if "coordinate_system" in asset:
            require(asset["coordinate_system"] in spaces,
                    "Asset references unknown coordinate system")
        coverage = asset.get("coverage", {})
        if "reference_asset" in coverage:
            require(coverage["reference_asset"] in assets, "Unknown coverage reference")
        refs = [f["ref"] for f in asset["files"]]
        require(len({(r["root"], r["path"]) for r in refs}) == len(refs),
                "Duplicate file in asset")
        status = dict(id=asset["id"], integrity="passed",
                      file_roles_validation="unsupported",
                      content_validation="unsupported",
                      coverage_validation="declaration_only",
                      coverage=copy.deepcopy(coverage) or None)
        roles = [f["role"] for f in asset["files"]]
        if asset["format"] in {"colmap.sparse.binary", "colmap.sparse.text"}:
            require(asset["type"] == "sparse_reconstruction",
                    "COLMAP sparse format requires sparse_reconstruction type")
            for role in ("cameras", "images", "points3D"):
                require(roles.count(role) == 1, f"COLMAP requires one {role} file")
            for role in ("rigs", "frames"):
                require(roles.count(role) <= 1, f"Duplicate COLMAP {role} file")
            status["file_roles_validation"] = "passed"
        if (asset["format"] == "indoor-recon.model"
                and asset.get("format_version") == "2"):
            require(asset["type"] == "parametric_scene",
                    "Parameter model requires parametric_scene type")
            require(roles.count("primary") == 1, "Model requires one primary file")
            primary = next(f["ref"] for f in asset["files"] if f["role"] == "primary")
            model = read_document(primary)
            require(model.get("kind") == "model" and model.get("schema_version") == 2
                    and model.get("scene_id") == data["scene_id"],
                    "Wrong parameter model")
            dependencies = list(asset_refs(model))
            used_refs.extend(dependencies)
            for record in records:
                if primary in record["refs"]:
                    record["refs"].extend(dependencies)
            expected = dict(handedness="right", up_axis="+Z", metres_per_unit=1)
            require(all(spaces[asset["coordinate_system"]][k] == v
                        for k, v in expected.items()), "Invalid model coordinates")
            status.update(file_roles_validation="passed",
                          content_validation=optional_validate(model, adapters))
        statuses.append(status)

    # Check each directory's local lock, and the bundle lock across the whole closure.
    entry_lock = records[0]["lock"]
    if not standalone:
        require(entry_lock is not None, "Bundle requires an asset lock")
        records[0]["refs"] = [r for r in used_refs if r != data["asset_lock"]]
    for record in records:
        if record["lock"] is not None:
            for ref in record["refs"]:
                item = record["lock"].get((ref["root"], ref["path"]))
                require(item is not None,
                        "A referenced file is missing from asset lock")
                if "sha256" in ref:
                    require(ref["sha256"] == item["sha256"],
                            "Reference and lock hash disagree")
    declared_roots = set(data.get("external_roots", {}))
    for record in records:
        declared_roots.update(record["data"].get("external_roots", {}))
    for ref in used_refs:
        require(ref["root"] == "manifest" or ref["root"] in declared_roots,
                f"Undeclared external root: {ref['root']}")
        if "sha256" not in ref:
            require(any((ref["root"], ref["path"]) in table for table in locks),
                    "Unpinned dependency requires a lock")
        pinned(ref)
    for record in records:
        if record["data"].get("packaging") == "self_contained":
            base = resolve_asset(record["ref"], roots).parent
            references = (used_refs if record["id"] == "entry" else
                          record["refs"] + list((record["lock"] or {}).values()))
            if "asset_lock" in record["data"]:
                references = references + [record["data"]["asset_lock"]]
            for ref in references:
                require(resolve_asset(ref, roots).is_relative_to(base),
                        "Self-contained declaration references outside its directory")

    profile_results = []
    if "homespatial" in profiles:
        for record in records:
            extensions = record["data"].get("extensions", {})
            extension = extensions.get("org.homespatial.catalog")
            if extension is None:
                continue
            schema_check(extension, "homespatial.schema.json")
            if "map_id" in extension:
                require(int(extension["map_id"]) < 2**63,
                        "map_id exceeds signed BigInt")
            local_assets = unique_index(record["data"]["assets"], "profile asset")
            local_spaces = unique_index(record["data"]["coordinate_systems"],
                                        "profile space")
            if record["id"] == "entry":
                local_assets, local_spaces = assets, spaces
            binding = extension.get("map_binding")
            if binding:
                require(int(binding["map_id"]) < 2**63, "map_id exceeds signed BigInt")
                require("map_id" not in extension
                        or extension["map_id"] == binding["map_id"],
                        "Map association disagrees with frame binding")
                require(binding["coordinate_system"] in local_spaces,
                        "Map binding references unknown coordinate system")
            for item in extension.get("artifact_bindings", []) + extension.get(
                    "resource_selections", []):
                require(item["asset"] in local_assets,
                        "Profile references unknown asset")
            profile_results.append(dict(id=record["id"], profile="homespatial",
                                        structure_validation="passed",
                                        service_validation="not_performed"))
    by_id = {s["id"]: s for s in statuses}
    for identity in require_content:
        require(identity in by_id and by_id[identity]["content_validation"] == "passed",
                f"Required content validation unavailable: {identity}")
    return dict(kind=data["kind"], schema_version=1, scene_id=data["scene_id"],
                entry_sha256=entry_digest, integrity="passed",
                lock_validation="passed" if entry_lock is not None else "not_provided",
                files=len({(r["root"], r["path"]) for r in used_refs}),
                assets=statuses, reconstructions=description_statuses,
                reconstruction_views=[dict(id=r["id"], **r["view"]) for r in records],
                profiles=profile_results,
                packaging=data.get("packaging", "not_declared"),
                coverage=data.get("coverage"), coverage_validation="declaration_only",
                input_eligibility=None,
                frames=records[0]["view"]["frames"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--adapter", action="append", default=[],
                        choices=["indoor-recon", "images"])
    parser.add_argument("--profile", action="append", default=[],
                        choices=["homespatial"])
    parser.add_argument("--require-content", action="append", default=[],
                        metavar="ASSET")
    parser.add_argument("--list-assets", action="store_true",
                        help="List qualified asset IDs and coordinate declarations for consumer selection")
    args = parser.parse_args(argv)
    try:
        roots = {}
        for item in args.root:
            name, sep, path = item.partition("=")
            require(sep and name not in ("local", "manifest") and name and path
                    and name not in roots, f"Invalid root binding: {item}")
            roots[name] = Path(path).resolve()
        result = check(args.document, roots, adapters=args.adapter, profiles=args.profile,
                       require_content=args.require_content)
        result["summary"] = dict(execution="completed", data=dict(integrity="passed",
            content="partial" if any(a["content_validation"] != "passed" for a in result["assets"]) else "passed"),
            quality=dict(status="not_evaluated"),
            next_steps=["Run the consuming tool's preflight to check native content and input eligibility"])
        if args.list_assets:
            try:
                from .exchange import Catalog
            except ImportError:
                from exchange import Catalog
            result["catalog"] = Catalog(args.document, roots).summary()
    except (ValueError, OSError, KeyError, ValidationError) as exc:
        print(json.dumps(dict(execution="failed", error=getattr(exc, "message", str(exc))), ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
