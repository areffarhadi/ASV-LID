#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
bash "$(dirname "$0")/ensure_checkpoints.sh"

GPU="${1:-0}"
WAV_DIR="${2:-$DEMO_WAV_DIR}"
mkdir -p "$INFERENCE_OUT"

echo "=== ASV-LID end-to-end inference ==="
echo "Wav dir:  $WAV_DIR"
echo "GPU:      $GPU"
echo "Python:   $PAPER_PYTHON"
echo

cd "$REPO_ROOT/src/inference"
"$PAPER_PYTHON" end_to_end.py \
  --wav_dir "$WAV_DIR" \
  --gpu "$GPU" \
  --output_json "$INFERENCE_OUT/results.json"
