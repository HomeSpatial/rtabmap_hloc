#!/usr/bin/env bash
# Install into the active Python environment; no workspace-specific paths.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON:-python3}" -m pip install -e "$SCRIPT_DIR"
"${PYTHON:-python3}" -c 'import hloc, pycolmap; print("HLOC:", hloc.__file__); print("pycolmap:", pycolmap.__version__)'
