#!/bin/bash
# Verify joint WPT embeddings match cached stage1_asv / stage2_lid extractions.
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
bash "$(dirname "$0")/ensure_checkpoints.sh"

GPU="${1:-0}"
NUM_SAMPLES="${2:-3}"
WAV_DIR="${3:-${TIDYVOICEX_DEV}}"

EMB_ASV_DEV="${EMB_TIDYVOICE}/stage1_asv/dev"
EMB_LID_DEV="${EMB_TIDYVOICE}/stage2_lid/dev"

echo "=== Joint embedding parity test ==="
echo "Wav dir:     $WAV_DIR"
echo "ASV cache:   $EMB_ASV_DEV"
echo "LID cache:   $EMB_LID_DEV"
echo "Samples:     $NUM_SAMPLES"
echo "GPU:         $GPU"
echo

cd "$REPO_ROOT/src/inference"
"$PAPER_PYTHON" compare_embeddings.py \
  --asv_ckpt "$CKPT_ASV" \
  --lid_ckpt "$CKPT_LID" \
  --asv_code "$ASV_CODE" \
  --lid_code "$LID_CODE" \
  --emb_asv_root "$EMB_ASV_DEV" \
  --emb_lid_root "$EMB_LID_DEV" \
  --ssl_model "$W2VBERT_MODEL" \
  --wav_dir "$WAV_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --gpu "$GPU"
