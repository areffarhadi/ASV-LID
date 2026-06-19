#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m venv "$REPO_ROOT/.venv"
source "$REPO_ROOT/.venv/bin/activate"
pip install -U pip
pip install -r "$REPO_ROOT/requirements.txt"
echo ""
echo "Done. Activate with:"
echo "  source $REPO_ROOT/.venv/bin/activate"
echo "  source $REPO_ROOT/config/paths.sh"
echo ""
echo "Note: transformers>=4.40 is required for Wav2Vec2Bert (unified inference)."
