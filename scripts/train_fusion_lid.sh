#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
GPU=${1:-0}
EPOCHS=${2:-20}
mkdir -p "$OUT_FUSION_LID"
"$PAPER_PYTHON" "$REPO_ROOT/src/fusion_lid/train_lid_fusion_dual_path_manifest.py" \
  --manifest_file "$LID_MANIFEST" \
  --emb_stage1_base "$EMB_TIDYVOICE/stage1_asv" \
  --emb_stage2_base "$EMB_TIDYVOICE/stage2_lid" \
  --trial_file "$LID_TRIALS_DEV" \
  --enrollment_manifest "$LID_ENROLL_MANIFEST" \
  --output_dir "$OUT_FUSION_LID" \
  --gpu "$GPU" \
  --epochs "$EPOCHS"
