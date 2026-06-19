#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config/paths.sh"
bash "$(dirname "$0")/ensure_checkpoints.sh"
GPU=${1:-0}
mkdir -p "$OUT_EVAL/asv_mindcf"
export REPO_ROOT CKPT_FUSION_ASV EMB_TIDYVOICE EMB_TIDYVOICEX2 EVAL_ASV_NPZ EVAL_LID_NPZ ASV_TRIAL_DEV2 ASV_TASK1 ASV_TASK2
"$PAPER_PYTHON" "$REPO_ROOT/src/fusion_asv/evaluate_both_models_mindcf.py" --gpu "$GPU" --output_dir "$OUT_EVAL/asv_mindcf"
