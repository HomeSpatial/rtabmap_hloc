"""Run outside the checkout after installing the wheel into a fresh environment."""

from importlib import metadata, resources
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import rtabmap_hloc_reprocess

distribution = metadata.distribution("rtabmap-hloc-reprocess")
assert distribution.version == rtabmap_hloc_reprocess.__version__
assert distribution.metadata["License-Expression"] == "Apache-2.0"
assert Path(rtabmap_hloc_reprocess.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
scripts = {e.name: e.value for e in distribution.entry_points if e.group == "console_scripts"}
assert scripts == {"rtabmap-hloc": "rtabmap_hloc_reprocess.cli:main"}
data = resources.files("rtabmap_hloc_reprocess._exchange")
for name in ("reconstruction", "bundle", "asset-lock", "input-view", "homespatial"):
    schema = json.loads(data.joinpath(name + ".schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
cli = Path(sys.executable).with_name("rtabmap-hloc.exe" if sys.platform == "win32" else "rtabmap-hloc")
for args in ([str(cli), "--help"], [str(cli), "--version"],
             [sys.executable, "-m", "rtabmap_hloc_reprocess", "--help"]):
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "rtabmap-hloc" in result.stdout
print(json.dumps({"passed": True, "version": distribution.version,
                  "installed_import": rtabmap_hloc_reprocess.__file__, "schemas": 5}))
