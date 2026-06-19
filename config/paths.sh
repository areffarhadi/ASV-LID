#!/bin/bash
# Repository paths — source from repo root: source config/paths.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# User-provided lists (download from TidyLang2026-baseline and place under data/)
export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
export LID_MANIFEST="${LID_MANIFEST:-${DATA_DIR}/manifests/training_manifest.txt}"
export LID_TRIALS_DEV="${LID_TRIALS_DEV:-${DATA_DIR}/trials/lid/trials_Dev.txt}"
export LID_ENROLL_MANIFEST="${LID_ENROLL_MANIFEST:-${DATA_DIR}/trials/lid/enrollment_manifest.tsv}"
export TL26_LID="${TL26_LID:-${DATA_DIR}/trials/tl26/tl26_lid.txt}"
export TL26_PAIRS="${TL26_PAIRS:-${DATA_DIR}/trials/tl26/tl26_pairs.txt}"
export TL26_ENROLL="${TL26_ENROLL:-${DATA_DIR}/trials/tl26/tl26_enroll.tsv}"
export ASV_TRIAL_DEV2="${ASV_TRIAL_DEV2:-${DATA_DIR}/trials/asv/test_4types_trials_dev2.txt}"
export ASV_TASK1="${ASV_TASK1:-${DATA_DIR}/trials/asv/task1_labels.txt}"
export ASV_TASK2="${ASV_TASK2:-${DATA_DIR}/trials/asv/task2_labels.txt}"

# Audio corpora (set these env vars to your extracted dataset roots)
export TIDYVOICEX_TRAIN="${TIDYVOICEX_TRAIN:-}"
export TIDYVOICEX_DEV="${TIDYVOICEX_DEV:-}"
export TIDYVOICEX2_ASV="${TIDYVOICEX2_ASV:-}"
export TIDYVOICEX_DATA_ROOTS="${TIDYVOICEX_DATA_ROOTS:-${TIDYVOICEX_TRAIN} ${TIDYVOICEX_DEV}}"

# Optional augmentation data for training (MUSAN + RIR)
export RIR_FOLDER="${RIR_FOLDER:-}"
export NOISE_FOLDER="${NOISE_FOLDER:-}"

# Pre-extracted embeddings for benchmark eval (optional; see docs/REPLICATION.md)
export EMB_TIDYVOICE="${EMB_TIDYVOICE:-${DATA_DIR}/embeddings/tidyvoice}"
export EMB_TIDYVOICEX2="${EMB_TIDYVOICEX2:-${DATA_DIR}/embeddings/tidyvoicex2_asv}"
export EVAL_ASV_NPZ="${EVAL_ASV_NPZ:-${EMB_TIDYVOICE}/converted_npz/dev_asv.npz}"
export EVAL_LID_NPZ="${EVAL_LID_NPZ:-${EMB_TIDYVOICE}/converted_npz/dev_lid.npz}"

# VoxLingua audio roots (set before eval_voxlingua_lid.sh)
export VOXLINGUA_EVAL_ROOT="${VOXLINGUA_EVAL_ROOT:-}"
export VOXLINGUA_TRAIN_ROOT="${VOXLINGUA_TRAIN_ROOT:-}"

# Hugging Face cache (W2V-BERT-2.0 backbone downloads here on first run)
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export W2VBERT_MODEL="${W2VBERT_MODEL:-facebook/w2v-bert-2.0}"

# Paper checkpoints (auto-downloaded from Hugging Face when missing)
export CKPT_ASV="${REPO_ROOT}/checkpoints/asv_wpt.pt"
export CKPT_LID="${REPO_ROOT}/checkpoints/lid_wpt.pt"
export CKPT_FUSION_ASV="${REPO_ROOT}/checkpoints/fusion_asv.pt"
export CKPT_FUSION_LID="${REPO_ROOT}/checkpoints/fusion_lid.pth"
export CKPT_VOXLINGUA_LID_DIR="${REPO_ROOT}/checkpoints/voxlingua_lid"
export HF_REPO_TIDYVOICE="${HF_REPO_TIDYVOICE:-areffarhadi/w2v-bert-TidyVoice}"
export HF_REPO_VOXLINGUA="${HF_REPO_VOXLINGUA:-areffarhadi/w2v-bert-VoxLingua}"

# Training / inference code locations
export ASV_CODE="${ASV_CODE:-${REPO_ROOT}/src/wpt}"
export LID_CODE="${LID_CODE:-${REPO_ROOT}/src/wpt}"
export FUSION_ASV_CODE="${FUSION_ASV_CODE:-${REPO_ROOT}/src/fusion_asv}"
export FUSION_LID_CODE="${FUSION_LID_CODE:-${REPO_ROOT}/src/fusion_lid}"

# Outputs
export OUT_ASV="${OUT_ASV:-${REPO_ROOT}/outputs/asv_wpt}"
export OUT_LID="${OUT_LID:-${REPO_ROOT}/outputs/lid_wpt}"
export OUT_FUSION_ASV="${OUT_FUSION_ASV:-${REPO_ROOT}/outputs/fusion_asv_v12}"
export OUT_FUSION_LID="${OUT_FUSION_LID:-${REPO_ROOT}/outputs/fusion_lid_manifest}"
export OUT_EVAL="${REPO_ROOT}/results/runs"
export DEMO_WAV_DIR="${REPO_ROOT}/demo_pair"
export INFERENCE_OUT="${REPO_ROOT}/results/inference"

export PYTHONPATH="${REPO_ROOT}/src/voxlingua_lid:${REPO_ROOT}/src/inference:${REPO_ROOT}/src/wpt:${REPO_ROOT}/src/fusion_asv:${REPO_ROOT}/src/fusion_lid:${PYTHONPATH:-}"

if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  export PAPER_PYTHON="${PAPER_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
else
  export PAPER_PYTHON="${PAPER_PYTHON:-python3}"
fi
