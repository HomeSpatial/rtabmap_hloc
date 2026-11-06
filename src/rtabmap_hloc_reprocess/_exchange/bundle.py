"""Transactional stage append and selection; portable, no producer dependencies.

The manifest is the commit point. Immutable snapshots and versioned locks are
written first. Failed runs never enter the manifest. Concurrent publishers merge
under an exclusive short-lived lock; inference never holds that lock.
"""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

try:
    from .exchange import Catalog
    from .verify import asset_refs, check, normalize_refs, read_json, require, resolve_asset, sha256
except ImportError:
    from exchange import Catalog
    from verify import asset_refs, check, normalize_refs, read_json, require, resolve_asset, sha256

EXTENSION = "org.reconstruction.stages"
ROLE_TYPES = {"sparse": {"sparse_reconstruction"}, "images": {"image_collection"},
              "geometry": {"mesh", "point_cloud"}, "prepared": {"prepared_task"},
              "proposals": {"perception_proposals"}, "model": {"parametric_scene"}}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def bundle_path(path):
    path = Path(path).resolve()
    return path if path.suffix.lower() == ".json" else path / "bundle.json"


def stage_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
            and value not in (".", ".."), "Stage ID must be a simple name, without directories")
    return value


def roles_for(catalog):
    roles = {}
    for role, types in ROLE_TYPES.items():
        rows = [a["id"] for a in catalog.assets.values() if a["type"] in types]
        if len(rows) == 1:
            roles[role] = rows[0]
    return roles


def stage_inputs(path, required, identity=None, preferred=None, roots=None):
    """Select a unique compatible stage; never infer chronology from array order."""
    catalog = Catalog(path, roots)
    require(catalog.document["kind"] == "reconstruction_bundle", "--bundle requires a bundle")
    metadata = catalog.document.get("extensions", {}).get(EXTENSION, {})
    candidates = []
    for row in catalog.document.get("reconstructions", []):
        if (row["format"], row["format_version"]) != ("reconstruction.description", "1"):
            continue
        child_path = catalog.resolve(row["document"])
        child = Catalog(child_path, roots)
        declared = metadata.get(row["id"], {})
        roles = declared.get("roles", roles_for(child))
        for role, asset in roles.items():
            require(role in ROLE_TYPES and asset in child.assets
                    and child.assets[asset]["type"] in ROLE_TYPES[role], "Invalid stage role declaration")
        if set(required) <= roles.keys():
            candidates.append(dict(id=row["id"], document=str(child_path), sha256=sha256(child_path),
                                   roles=roles, operation=declared.get("operation")))
    if identity:
        candidates = [r for r in candidates if r["id"] == identity]
    elif preferred and any(r["operation"] == preferred for r in candidates):
        candidates = [r for r in candidates if r["operation"] == preferred]
    require(len(candidates) == 1, f"Select --from STAGE explicitly; compatible stages: {[r['id'] for r in candidates]}")
    return catalog, candidates[0]


def plan(path, stage, operation, *, required=(), source=None, preferred=None, scene_id=None, out=None, roots=None):
    """Read-only plan. Output creation and computation are owned by the producer."""
    path = bundle_path(path)
    stage_id(stage)
    input_stage = None
    if required or source:
        require(path.is_file(), "Input bundle does not exist")
        catalog, input_stage = stage_inputs(path, required, source, preferred, roots)
    elif path.exists():
        catalog = Catalog(path, roots)
    else:
        catalog = None
    if catalog:
        require(catalog.document["kind"] == "reconstruction_bundle", "--bundle requires a bundle")
        require(stage not in {r["id"] for r in catalog.document.get("reconstructions", [])},
                f"Stage {stage} already exists; choose a new --stage")
        require(not scene_id or scene_id == catalog.scene_id, "Scene identity disagrees with bundle")
        scene_id = catalog.scene_id
    else:
        require(not source, "--from requires an existing input bundle")
        scene_id = scene_id or path.parent.name
    output = Path(out).resolve() if out else path.parent / "work" / stage
    # No producer may overwrite committed snapshots, locks or the manifest.
    protected = [path.parent / "stages", path.parent / ".bundle"]
    require(not path.is_relative_to(output) and not any(output.is_relative_to(p.resolve()) for p in protected),
            "Working output must be outside committed stages and .bundle metadata")
    require(all(p.resolve().is_relative_to(path.parent) for p in protected), "Bundle metadata/stage directory escapes bundle root")
    require(not (path.parent / "stages" / stage).exists(), f"Stage snapshot already exists: {stage}")
    return dict(bundle=str(path), stage=stage, operation=operation, scene_id=scene_id,
                output=str(output), input=input_stage)


def _closure(document, roots=None):
    """The same closure used by validation, including child locks and model refs."""
    path = Path(document).resolve()
    raw = read_json(path)
    roots = {**(roots or {}), "manifest": path.parent}
    entry = copy.deepcopy(raw)
    entry.pop("reconstructions", None)
    refs = list(asset_refs(normalize_refs(entry, dict(root="manifest", path=path.name))))
    docs = [(raw, dict(root="manifest", path=path.name))]
    for row in raw.get("reconstructions", []):
        ref = normalize_refs(row["document"], dict(root="manifest", path=path.name))
        refs.append(ref)
        if (row["format"], row["format_version"]) == ("reconstruction.description", "1"):
            child = read_json(resolve_asset(ref, roots))
            docs.append((child, ref))
            refs.extend(asset_refs(normalize_refs(child, ref)))
        else:
            refs.extend(normalize_refs(row.get("dependencies", []), ref))
    for doc, doc_ref in docs:
        normalized = normalize_refs(doc, doc_ref)
        if "asset_lock" in normalized:
            lock_ref = normalized["asset_lock"]
            lock = read_json(resolve_asset(lock_ref, roots))
            refs.extend(normalize_refs(lock, lock_ref)["files"])
        for asset in normalized["assets"]:
            if asset["format"] == "indoor-recon.model" and asset.get("format_version") == "2":
                ref = next(f["ref"] for f in asset["files"] if f["role"] == "primary")
                refs.extend(asset_refs(normalize_refs(read_json(resolve_asset(ref, roots)), ref)))
    return roots, list({(r["root"], r["path"]): r for r in refs}.values())


def copy_snapshot(document, destination):
    """Freeze a self-contained descriptor and its declared dependency closure."""
    document, destination = Path(document).resolve(), Path(destination).resolve()
    check(document)
    raw = read_json(document)
    require(raw["kind"] == "reconstruction_description" and raw.get("packaging") == "self_contained",
            "Publication requires a self-contained reconstruction description")
    require(not raw.get("external_roots"), "Stage snapshots must use local/manifest references")
    roots, refs = _closure(document)
    destination.mkdir(parents=True, exist_ok=False)
    for ref in refs:
        source = resolve_asset(ref, roots)
        require(source.is_relative_to(document.parent), "Snapshot dependency escapes result directory")
        target = destination / source.relative_to(document.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(document, destination / document.name)
    check(destination / document.name)
    return destination / document.name


@contextmanager
def _writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / ("." + path.name + ".writer.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"Another publisher holds {lock}; retry when it finishes. If interrupted, inspect its PID before removing this lock.") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(dict(pid=os.getpid()), stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


def publish(stage_plan, document, *, roles=None, quality=None, execution="completed", coordinate_link=None):
    """Merge one successful result, validate complete closure, then atomically commit."""
    path = Path(stage_plan["bundle"])
    stage = stage_id(stage_plan["stage"])
    source = Path(document).resolve()
    child = Catalog(source)
    require(execution in ("completed", "partial"), "Only completed or explicitly partial results can be published")
    require(child.scene_id == stage_plan["scene_id"], "Result scene identity disagrees with bundle")
    roles = roles if roles is not None else roles_for(child)
    for role, identity in roles.items():
        require(role in ROLE_TYPES and identity in child.assets and child.assets[identity]["type"] in ROLE_TYPES[role],
                "Invalid output role declaration")
    with _writer(path):
        require(all((path.parent / name).resolve().is_relative_to(path.parent) for name in ("stages", ".bundle")),
                "Bundle metadata/stage directory escapes bundle root")
        if path.exists():
            current = Catalog(path)
            require(current.scene_id == stage_plan["scene_id"], "Bundle scene changed while running")
            data = copy.deepcopy(current.document)
        else:
            require(not stage_plan["input"], "Input bundle disappeared")
            data = dict(kind="reconstruction_bundle", schema_version=1, scene_id=child.scene_id,
                        packaging="self_contained", assets=[], coordinate_systems=[], reconstructions=[])
        require(stage not in {r["id"] for r in data.get("reconstructions", [])}, f"Stage already published: {stage}")
        input_stage = stage_plan["input"]
        if input_stage:
            rows = [r for r in data.get("reconstructions", []) if r["id"] == input_stage["id"]]
            require(len(rows) == 1 and sha256(resolve_asset(rows[0]["document"], {"manifest": path.parent})) == input_stage["sha256"],
                    "Selected input stage changed while running")
        destination = path.parent / "stages" / stage
        require(not destination.exists(), f"Stage directory already exists: {stage}")
        temp = path.parent / "stages" / (".pending-" + uuid4().hex)
        candidate = path.with_name("." + path.name + "." + uuid4().hex + ".json")
        committed = moved = False
        commit_digest = None
        try:
            snap = copy_snapshot(source, temp)
            os.rename(temp, destination)
            moved = True
            snap = destination / snap.name
            ref = dict(root="manifest", path=snap.relative_to(path.parent).as_posix(), sha256=sha256(snap))
            data.setdefault("reconstructions", []).append(dict(id=stage, format="reconstruction.description", format_version="1", document=ref))
            if coordinate_link is not None:
                require(input_stage is not None, "Coordinate link requires an input stage")
                before, after = coordinate_link
                source_space = current.spaces.get(input_stage["id"] + "/" + before)
                target_space = child.spaces.get(after)
                require(source_space is not None and target_space is not None, "Unknown coordinate link endpoint")
                require(source_space["handedness"] == target_space["handedness"] == "right"
                        and source_space["up_axis"] == target_space["up_axis"],
                        "Identity handoff requires matching declared axes")
                data.setdefault("coordinate_transforms", []).append(dict(
                    **{"from": input_stage["id"] + "/" + before, "to": stage + "/" + after},
                    kind="rigid", units="metres", T_to_from=[[int(i == j) for j in range(4)] for i in range(4)],
                    method=stage_plan["operation"] + ":preserved-frame", evidence=[ref]))
            data.setdefault("extensions", {}).setdefault(EXTENSION, {})[stage] = dict(
                operation=stage_plan["operation"], roles=roles, execution=execution,
                inputs=[dict(stage=input_stage["id"], document_sha256=input_stage["sha256"])] if input_stage else [],
                quality=quality or {"status": "not_evaluated"})
            # Build a new lock; old generations remain valid for concurrent readers.
            data.pop("asset_lock", None)
            atomic_json(candidate, data)
            roots, refs = _closure(candidate)
            files = []
            for item in refs:
                native = resolve_asset(item, roots)
                files.append(dict(root=item["root"], path=item["path"], sha256=sha256(native), bytes=native.stat().st_size))
            lock = path.parent / ".bundle/locks" / (uuid4().hex + ".json")
            atomic_json(lock, dict(kind="asset_lock", schema_version=1, scene_id=child.scene_id, files=files))
            data["asset_lock"] = dict(root="manifest", path=lock.relative_to(path.parent).as_posix(), sha256=sha256(lock))
            atomic_json(candidate, data)
            check(candidate)
            commit_digest = sha256(candidate)
            os.replace(candidate, path)
            committed = True
        finally:
            candidate.unlink(missing_ok=True)
            if temp.exists():
                require(temp.resolve().is_relative_to(path.parent / "stages"), "Unsafe temporary directory")
                shutil.rmtree(temp)
            if moved and not committed:
                # An interruption immediately after replace must not erase a now-live snapshot.
                committed = bool(commit_digest and path.is_file() and sha256(path) == commit_digest)
                if not committed:
                    require(destination.resolve().is_relative_to(path.parent / "stages"), "Unsafe snapshot directory")
                    shutil.rmtree(destination)
        return dict(bundle=str(path), stage=stage, description=str(snap), sha256=sha256(snap),
                    input_stage=input_stage["id"] if input_stage else None)
