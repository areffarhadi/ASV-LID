#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
GPU=${1:-0}
EPOCHS=${2:-30}

if [ ! -f "$EVAL_ASV_NPZ" ] || [ ! -f "$EVAL_LID_NPZ" ]; then
  echo "Pre-extracted embeddings not found under $EMB_TIDYVOICE" >&2
  echo "See docs/REPLICATION.md for embedding extraction before fusion training." >&2
  exit 1
fi

mkdir -p "$OUT_FUSION_ASV"
"$PAPER_PYTHON" "$REPO_ROOT/src/fusion_asv/train_fusion_dual_path_v12.py" \
  --train_asv_emb "$EMB_TIDYVOICE/stage1_asv" \
  --train_lang_emb "$EMB_TIDYVOICE/stage2_lid" \
  --eval_asv_emb "$EVAL_ASV_NPZ" \
  --eval_lang_emb "$EVAL_LID_NPZ" \
  --trial_file "$ASV_TRIAL_DEV2" \
  --output_dir "$OUT_FUSION_ASV" \
  --gpu "$GPU" \
  --epochs "$EPOCHS"
# Best checkpoint: copy to $CKPT_FUSION_ASV after training
