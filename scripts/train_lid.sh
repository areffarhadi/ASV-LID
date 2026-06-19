#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
GPU=${1:-0}
EPOCHS=${2:-15}
mkdir -p "$OUT_LID"
cd "$REPO_ROOT/src/wpt"
"$PAPER_PYTHON" main_train_lid.py \
  --manifest_file "$LID_MANIFEST" \
  --dataset_roots $TIDYVOICEX_DATA_ROOTS \
  --output_dir "$OUT_LID" \
  --model_dir "$W2VBERT_MODEL" \
  --gpu "$GPU" \
  --epochs "$EPOCHS" \
  --batch_size 4 \
  --lr 1e-4 \
  --num_heads 8 \
  --compression_dim 128 \
  --hidden_dim 512 \
  --arcface_margin 0.3 \
  --arcface_scale 30.0
