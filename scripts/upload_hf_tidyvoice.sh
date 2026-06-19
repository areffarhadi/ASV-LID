#!/bin/bash
# Upload TidyVoice checkpoints to HuggingFace (maintainer only).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../config/paths.sh"
HF_REPO="${HF_REPO_TIDYVOICE:-areffarhadi/w2v-bert-TidyVoice}"
STAGING="${REPO_ROOT}/.hf_upload/tidyvoice"
mkdir -p "$STAGING"
cp "$REPO_ROOT/hf/tidyvoice/README.md" "$STAGING/README.md"
cp "$REPO_ROOT/checkpoints/asv_wpt.pt" "$STAGING/"
cp "$REPO_ROOT/checkpoints/lid_wpt.pt" "$STAGING/"
cp "$REPO_ROOT/checkpoints/fusion_asv.pt" "$STAGING/"
cp "$REPO_ROOT/checkpoints/fusion_lid.pth" "$STAGING/"
echo "Uploading to https://huggingface.co/${HF_REPO}"
hf upload "$HF_REPO" "$STAGING" . --repo-type model
echo "Done: https://huggingface.co/${HF_REPO}"
