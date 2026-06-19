#!/bin/bash
# Download paper checkpoints from Hugging Face (manual or via ensure_checkpoints.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../config/paths.sh"
"$PAPER_PYTHON" "$SCRIPT_DIR/download_checkpoints.py"
