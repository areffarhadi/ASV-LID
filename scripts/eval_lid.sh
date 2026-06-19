#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
bash "$(dirname "$0")/ensure_checkpoints.sh"
GPU=${1:-0}
mkdir -p "$OUT_EVAL/lid_tl26"
"$PAPER_PYTHON" "$REPO_ROOT/src/fusion_lid/evaluate_lid_fusion_dual_path_manifest_tl26.py" \
  --emb_base "$EMB_TIDYVOICEX2" \
  --tl26_lid "$TL26_LID" \
  --tl26_pairs "$TL26_PAIRS" \
  --tl26_enroll "$TL26_ENROLL" \
  --output_dir "$OUT_EVAL/lid_tl26" \
  --checkpoint "$CKPT_FUSION_LID" \
  --manifest_file "$LID_MANIFEST" \
  --gpu "$GPU" \
  --score_batch_size 20000 \
  --save_manifest_val_scores \
  --emb_stage1_base "$EMB_TIDYVOICE/stage1_asv" \
  --emb_stage2_base "$EMB_TIDYVOICE/stage2_lid" \
  --val_trial_file "$LID_TRIALS_DEV" \
  --val_enroll_manifest "$LID_ENROLL_MANIFEST"
