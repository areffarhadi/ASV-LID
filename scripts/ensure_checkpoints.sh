#!/bin/bash
# Download paper checkpoints from Hugging Face when any expected file is missing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../config/paths.sh"

_need=0
for f in "$CKPT_ASV" "$CKPT_LID" "$CKPT_FUSION_ASV" "$CKPT_FUSION_LID"; do
  if [ ! -f "$f" ]; then
    _need=1
    break
  fi
done
if [ ! -f "$CKPT_VOXLINGUA_LID_DIR/model.pt" ] || [ ! -f "$CKPT_VOXLINGUA_LID_DIR/args.json" ]; then
  _need=1
fi

if [ "$_need" -eq 1 ]; then
  echo "Paper checkpoints not found locally; downloading from Hugging Face..."
  "$PAPER_PYTHON" "$SCRIPT_DIR/download_checkpoints.py"
fi
