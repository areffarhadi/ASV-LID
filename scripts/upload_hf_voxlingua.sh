#!/bin/bash
# Upload VoxLingua LID checkpoint to HuggingFace (maintainer only).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../config/paths.sh"
HF_REPO="${HF_REPO_VOXLINGUA:-areffarhadi/w2v-bert-VoxLingua}"
STAGING="${REPO_ROOT}/.hf_upload/voxlingua_lid"
mkdir -p "$STAGING"
cp "$REPO_ROOT/hf/voxlingua_lid/README.md" "$STAGING/README.md"
cp "$REPO_ROOT/checkpoints/voxlingua_lid/model.pt" "$STAGING/"
cp "$REPO_ROOT/checkpoints/voxlingua_lid/args.json" "$STAGING/"
echo "Uploading to https://huggingface.co/${HF_REPO}"
hf upload "$HF_REPO" "$STAGING" . --repo-type model
echo "Done: https://huggingface.co/${HF_REPO}"
