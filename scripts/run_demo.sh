#!/bin/bash
# Run the bundled demo (same speaker, English + Lithuanian).
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
GPU="${1:-0}"
bash "$(dirname "$0")/run_inference.sh" "$GPU" "${REPO_ROOT}/demo_pair"
