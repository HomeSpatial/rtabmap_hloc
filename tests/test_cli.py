"""Public entry-point checks that need no images, models or GPU."""

import subprocess
import sys

import pytest

from rtabmap_hloc_reprocess import __version__


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "rtabmap_hloc_reprocess", *args],
        capture_output=True, text=True,
    )


def test_version_needs_no_input_or_output():
    result = invoke("--version")
    assert result.returncode == 0
    assert result.stdout.strip() == f"rtabmap-hloc {__version__}"


def test_help_needs_no_model_runtime():
    result = invoke("--help")
    assert result.returncode == 0
    assert "--use-existing-poses" in result.stdout
    assert "--bundle" in result.stdout


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_invalid_pair_count_fails_before_creating_output(tmp_path, value):
    output = tmp_path / "output"
    result = invoke("missing.db", "-o", str(output), "--num-matched", value)
    assert result.returncode == 2
    assert "--num-matched" in result.stderr
    assert not output.exists()
