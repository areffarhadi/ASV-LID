#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
GPU=${1:-0}

if [ -z "${TIDYVOICEX_TRAIN:-}" ] || [ -z "${TIDYVOICEX_DEV:-}" ]; then
  echo "Set TIDYVOICEX_TRAIN and TIDYVOICEX_DEV to your TidyVoiceX audio roots." >&2
  exit 1
fi
if [ ! -f "$ASV_TRIAL_DEV2" ]; then
  echo "ASV trial file not found: $ASV_TRIAL_DEV2" >&2
  echo "Download trial lists into data/trials/asv/ (see data/README.md)." >&2
  exit 1
fi

mkdir -p "$OUT_ASV"
cd "$REPO_ROOT/src/wpt"
"$PAPER_PYTHON" main_train_asv_dynstats.py \
  --train_audio "$TIDYVOICEX_TRAIN" \
  --eval_audio "$TIDYVOICEX_DEV" \
  --trial_file "$ASV_TRIAL_DEV2" \
  --output_dir "$OUT_ASV" \
  --xlsr "$W2VBERT_MODEL" \
  --gpu "$GPU" \
  --batch_size 8 \
  --num_epochs 20 \
  --num_prompt_tokens 15 \
  --num_wavelet_tokens 8 \
  --num_heads 8 \
  --compression_dim 128 \
  --embedding_dim 256 \
  --head_type dynstats_ecapa \
  --use_arcface
# Best checkpoint: copy to $CKPT_ASV after training
