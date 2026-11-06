"""Portable v1 asset selection and packaging. Standard library + jsonschema only.

This reference module and verify.py are vendored unchanged by consumer tools.
It resolves declarations; native content eligibility belongs to each consumer.
"""
import copy
import json
from pathlib import Path, PurePosixPath
import shutil

try:
    from .verify import check, normalize_refs, read_json, require, resolve_asset, sha256
except ImportError:  # standalone reference distribution
    from verify import check, normalize_refs, read_json, require, resolve_asset, sha256


def root_bindings(items):
    roots = {}
    for item in items or ():
        name, sep, path = item.partition("=")
        require(sep and name and path and name not in ("local", "manifest")
                and name not in roots, f"Invalid or duplicate root binding: {item}")
        roots[name] = Path(path).resolve()
    return roots


class Catalog:
    """Validated standalone or bundle registry, with qualified child asset IDs."""
    def __init__(self, document, roots=None):
        self.path = Path(document).resolve()
        self.roots = dict(roots or {})
        self.validation = check(self.path, self.roots)
        self.roots["manifest"] = self.path.parent
        self.document = read_json(self.path)
        self.scene_id = self.document["scene_id"]
        self.assets, self.spaces, self.transforms, self.extensions = {}, {}, [], {}
        entry = dict(root="manifest", path=self.path.name)
        self._add(self.document, entry, "")
        for row in self.document.get("reconstructions", []):
            if (row["format"], row["format_version"]) != ("reconstruction.description", "1"):
                continue
            ref = row["document"]
            self._add(read_json(self.resolve(ref)), ref, row["id"] + "/")

    def _add(self, raw, ref, prefix):
        data = normalize_refs(raw, ref)
        self.extensions[prefix.rstrip("/") or "entry"] = copy.deepcopy(data.get("extensions", {}))
        for space in data["coordinate_systems"]:
            self.spaces[prefix + space["id"]] = dict(space, id=prefix + space["id"])
        for asset in data["assets"]:
            row = copy.deepcopy(asset)
            row["id"] = prefix + row["id"]
            if "coordinate_system" in row:
                row["coordinate_system"] = prefix + row["coordinate_system"]
            coverage = row.get("coverage")
            if coverage and "reference_asset" in coverage:
                coverage["reference_asset"] = prefix + coverage["reference_asset"]
            self.assets[row["id"]] = row
        for transform in data.get("coordinate_transforms", []):
            row = copy.deepcopy(transform)
            row["from"], row["to"] = prefix + row["from"], prefix + row["to"]
            self.transforms.append(row)

    def resolve(self, ref):
        return resolve_asset(ref, self.roots)

    def select(self, types, identity=None, optional=False):
        candidates = [a for a in self.assets.values() if a["type"] in types]
        if identity is not None:
            require(identity in self.assets, f"Unknown asset {identity}; available: {list(self.assets)}")
            row = self.assets[identity]
            require(row["type"] in types, f"Asset {identity} must have type in {sorted(types)}")
            return row
        if not candidates and optional:
            return None
        require(len(candidates) == 1,
                f"Select an asset of type {sorted(types)} explicitly; candidates: {[a['id'] for a in candidates]}")
        return candidates[0]

    def file(self, asset, role="primary"):
        rows = [f for f in asset["files"] if f["role"] == role]
        require(len(rows) == 1, f"Asset {asset['id']} requires exactly one {role} file")
        return self.resolve(rows[0]["ref"])

    def image_map(self, asset, names):
        """Bind native COLMAP names by path suffix, never by array position."""
        paths = [self.resolve(f["ref"]) for f in asset["files"] if f["role"] == "image"]
        require(paths, f"Image asset {asset['id']} has no role=image files")
        result = {}
        for name in names:
            parts = PurePosixPath(name).parts
            require(parts and not PurePosixPath(name).is_absolute()
                    and not any(p in (".", "..") for p in parts)
                    and "\\" not in name and ":" not in name, f"Unsafe COLMAP image name: {name}")
            matches = [p for p in paths if p.parts[-len(parts):] == parts]
            require(len(matches) <= 1, f"Ambiguous photo for {name} in asset {asset['id']}")
            if matches:
                result[name] = matches[0]
        require(result, "No declared photos match the selected COLMAP model")
        require(len(set(result.values())) == len(paths),
                "Some declared photos do not match registered COLMAP names; choose the matching image asset")
        coverage = asset.get("coverage", {})
        if coverage.get("status") == "full":
            require(set(result) == set(names), "Full image coverage is declared but registered photos are missing")
        return result

    def transform_path(self, source, target):
        """Return explicit transforms and direction; consumers apply native/metre units."""
        queue, seen = [(source, [])], {source}
        while queue:
            node, path = queue.pop(0)
            if node == target:
                return path
            for row in self.transforms:
                for start, end, reverse in ((row["from"], row["to"], False), (row["to"], row["from"], True)):
                    if node == start and end not in seen:
                        seen.add(end)
                        queue.append((end, path + [(row, reverse)]))
        raise ValueError(f"No declared coordinate transform from {source} to {target}")

    def summary(self):
        return {"document": str(self.path), "scene_id": self.scene_id,
                "integrity": self.validation["integrity"],
                "assets": [{k: a[k] for k in ("id", "type", "format", "coordinate_system", "coverage") if k in a}
                           for a in self.assets.values()],
                "coordinate_systems": list(self.spaces.values()),
                "quality": {"status": "not_evaluated", "producer_extensions": self.extensions}}


def result_status(execution="completed", integrity="passed", content="not_evaluated",
                  quality="not_evaluated", next_steps=(), **details):
    return dict(execution=execution, data=dict(integrity=integrity, content=content),
                quality=dict(status=quality), next_steps=list(next_steps), **details)


def write_description(directory, scene_id, coordinates, sparse_files, images,
                      geometry=None, producer=None, quality_files=(), quality=None,
                      source_document=None):
    """Create a self-contained native-file exchange snapshot in a NEW directory.

    images is {COLMAP image name: file}. Geometry is (type, format, path).
    Deliberately copy instead of linking mutable producer files into a snapshot.
    """
    root = Path(directory).resolve()
    require(not root.exists(), f"Output exists; choose a new directory: {root}")
    # Validate names and sources before creating any output.
    for name in images:
        resolve_asset(dict(root="local", path="images/" + name), {"local": root})
    sources = list(sparse_files.values()) + list(images.values()) + list(quality_files)
    if geometry:
        sources.append(geometry[2])
    require(all(Path(p).is_file() for p in sources), "Missing snapshot source file")
    root.mkdir(parents=True)
    def retain(source, relative, role):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return dict(role=role, ref=dict(root="local", path=relative, sha256=sha256(target)))
    assets = []
    if sparse_files:
        assets.append(dict(id="cameras", type="sparse_reconstruction", format="colmap.sparse.binary",
                           coordinate_system="source", files=[retain(p, "sparse/" + role + ".bin", role)
                                                                for role, p in sparse_files.items()]))
    if images:
        coverage = dict(scope="COLMAP image names", status="subset", ids=list(images))
        if sparse_files:
            coverage["reference_asset"] = "cameras"
        assets.append(dict(id="photos", type="image_collection", format="image.files", coverage=coverage,
                           files=[retain(p, "images/" + name, "image") for name, p in images.items()]))
    if geometry:
        kind, fmt, path = geometry
        assets.append(dict(id="geometry", type=kind, format=fmt, coordinate_system="source",
                           files=[retain(path, "geometry" + Path(path).suffix, "primary")]))
    if quality_files:
        assets.append(dict(id="validation", type="validation_report", format="json",
                           files=[retain(p, f"reports/{i}-{Path(p).name}", "report")
                                  for i, p in enumerate(quality_files)]))
    doc = dict(kind="reconstruction_description", schema_version=1, scene_id=scene_id,
               packaging="self_contained", coordinate_systems=[dict(coordinates, id="source")], assets=assets)
    if producer and geometry:
        doc["runs"] = [dict(id="reconstruction", producer=producer,
                            inputs=[a["id"] for a in assets if a["id"] in ("cameras", "photos")], outputs=["geometry"])]
    if quality is not None or source_document is not None:
        doc["extensions"] = {"org.reconstruction.workflow": dict(
            quality=quality or {"status": "not_evaluated"},
            source_document_sha256=sha256(source_document) if source_document else None)}
    path = root / "reconstruction.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check(path)
    return path
