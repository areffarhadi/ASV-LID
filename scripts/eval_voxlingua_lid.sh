#!/bin/bash
# Evaluate VoxLingua LID checkpoint on dev_vox.
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
bash "$(dirname "$0")/ensure_checkpoints.sh"

if [ -z "${VOXLINGUA_EVAL_ROOT:-}" ]; then
  echo "Set VOXLINGUA_EVAL_ROOT to your VoxLingua dev_vox audio folder." >&2
  exit 1
fi

GPU="${1:-0}"
mkdir -p "$OUT_EVAL/voxlingua_lid"

echo "=== VoxLingua LID evaluation ==="
echo "Checkpoint: $CKPT_VOXLINGUA_LID_DIR"
echo "Eval data:  $VOXLINGUA_EVAL_ROOT"
echo "GPU:        $GPU"
echo

cd "$REPO_ROOT/src/voxlingua_lid"
"$PAPER_PYTHON" eval_voxlingua.py \
  --checkpoint_dir "$CKPT_VOXLINGUA_LID_DIR" \
  --eval_audio_root "$VOXLINGUA_EVAL_ROOT" \
  --gpu "$GPU" \
  --batch_size 16 \
  --num_workers 4 \
  --output_json "$OUT_EVAL/voxlingua_lid/eval_summary.json"
