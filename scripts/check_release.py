"""Check public source and built distributions using only Python's standard library."""

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tomllib
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRS = {".git", ".venv", "artifacts", "build", "dist", "__pycache__",
              ".pytest_cache", ".ruff_cache", ".mypy_cache", "htmlcov"}
FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".glb", ".ply", ".h5", ".npz",
                      ".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key",
                      ".jpg", ".jpeg", ".png"}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "credential": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
        r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}|hf_[A-Za-z0-9]{30,}|"
        r"AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{35}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
        r"|https?://[^\s/:<>\"']+:[^\s/@<>\"']+@"),
    "private machine path": re.compile(
        r"[A-Z]:[/\\](?:Users|HomeSpatial" + r"Unity)[/\\]|/home/ywh" + r"1357|"
        r"wsl_" + r"pc1|gitlab\.home\.", re.I),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and ".." not in path.parts
            and "\\" not in name and ":" not in name and str(path) == name,
            f"Unsafe source/archive path: {name}")
    require(not any(part in LOCAL_DIRS for part in path.parts), f"Local files included: {name}")
    require(not path.name.startswith(".env") and path.suffix.lower() not in FORBIDDEN_SUFFIXES,
            f"Data or credential file requires separate review: {name}")


def scan(name, raw):
    safe_name(name)
    require(len(raw) <= 5 * 1024 * 1024, f"Oversized public source: {name}")
    content = raw.decode("utf-8")
    require("\0" not in content, f"Binary public source: {name}")
    for label, pattern in PATTERNS.items():
        require(not pattern.search(content), f"Possible {label}: {name}")
    return content


def source_files(root):
    manifest = json.loads((root / "release-files.json").read_text(encoding="utf-8"))
    names = manifest["files"]
    require(manifest.get("schema_version") == 1 and names
            and len(names) == len(set(names)), "Invalid public source manifest")
    sources = {}
    for name in names:
        safe_name(name)
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                f"Missing or unsafe public source: {name}")
        require(not any(p.is_symlink() for p in path.parents if p.is_relative_to(root)),
                f"Symlinked source directory: {name}")
        raw = path.read_bytes()
        scan(name, raw)
        sources[name] = raw
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in LOCAL_DIRS and not d.endswith(".egg-info")]
        for filename in files:
            if filename.endswith((".pyc", ".pyo")):
                continue
            name = (Path(directory) / filename).relative_to(root).as_posix()
            require(name in sources, f"Unlisted public source: {name}")
    for name, raw in sources.items():
        if not name.endswith(".md"):
            continue
        for href in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", raw.decode("utf-8")):
            if href.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = (root / name).parent / unquote(href.split("#", 1)[0])
            target = target.resolve()
            require(target.is_relative_to(root) and target.is_file()
                    and target.relative_to(root).as_posix() in sources,
                    f"Broken/nonpublic documentation link: {name} -> {href}")
    vendor = root / "src/rtabmap_hloc_reprocess/_exchange"
    for name, digest in json.loads((vendor / "SOURCE.json").read_text())["files"].items():
        require(hashlib.sha256((vendor / name).read_bytes()).hexdigest() == digest,
                f"Vendored reference hash mismatch: {name}")
    return sources


def check_metadata(raw, project):
    meta = BytesParser().parsebytes(raw)
    require(meta["License-Expression"] == "Apache-2.0", "Missing Apache-2.0 package metadata")
    require(meta["Version"] == project["version"], "Package/source version mismatch")
    require(set(meta.get_all("License-File", [])) == set(project["license-files"]),
            "Missing package license/notice files")


def check_archives(directory, sources, project):
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    require(len(wheels) == len(sdists) == 1, "Expected exactly one wheel and one sdist")
    reports = []
    package_files = {n.removeprefix("src/"): b for n, b in sources.items() if n.startswith("src/")}
    for path in wheels + sdists:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                entries = [i.filename for i in archive.infolist() if not i.is_dir()]
                require(len(entries) == len(set(entries)), "Duplicate wheel members")
                files = {n: archive.read(n) for n in entries}
            infos = {n.split("/", 1)[0] for n in files if ".dist-info/" in n}
            require(len(infos) == 1, "Expected one wheel metadata directory")
            info = infos.pop()
            metadata_names = {f"{info}/{n}" for n in ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD")}
            license_files = {f"{info}/licenses/{n}": sources[n] for n in project["license-files"]}
            expected = {**package_files, **license_files}
            require(set(files) == set(expected) | metadata_names, "Unexpected/missing wheel files")
            check_metadata(files[f"{info}/METADATA"], project)
        else:
            with tarfile.open(path) as archive:
                members = archive.getmembers()
                require(all(m.isfile() or m.isdir() for m in members), "Links/special files in sdist")
                entries = [m for m in members if m.isfile()]
                require(len(entries) == len({m.name for m in entries}), "Duplicate sdist members")
                require(len({m.name.split("/", 1)[0] for m in entries}) == 1, "Invalid sdist root")
                files = {m.name.split("/", 1)[1]: archive.extractfile(m).read() for m in entries}
            egg = "src/rtabmap_hloc_reprocess.egg-info/"
            generated = {"PKG-INFO", "setup.cfg"} | {egg + n for n in (
                "PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt", "top_level.txt")}
            require(set(files) == set(sources) | generated, "Unexpected/missing sdist files: "
                    + str(sorted(set(files) ^ (set(sources) | generated))))
            expected = sources
            check_metadata(files["PKG-INFO"], project)
        for name, raw in files.items():
            scan(name, raw)
        for name, raw in expected.items():
            require(files[name] == raw, f"Packaged file differs from source: {name}")
        reports.append({"file": path.name, "members": len(files),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sources = source_files(root)
    project = tomllib.loads(sources["pyproject.toml"].decode())["project"]
    require(project["license"] == "Apache-2.0", "Missing source license metadata")
    require(project["scripts"] == {"rtabmap-hloc": "rtabmap_hloc_reprocess.cli:main"},
            "Unexpected console scripts")
    version = re.search(r'__version__ = "([^"]+)"', sources["src/rtabmap_hloc_reprocess/__init__.py"].decode())
    require(version and version[1] == project["version"], "Runtime/package version mismatch")
    archives = check_archives(args.dist.resolve(), sources, project) if args.dist else []
    print(json.dumps({"passed": True, "public_files": len(sources), "distributions": archives}))


if __name__ == "__main__":
    main()
