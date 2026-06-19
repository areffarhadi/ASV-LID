# ASV-LID: Dual-Path WPT for Speaker and Language Recognition

Code and checkpoints for the Odyssey 2026 paper:

> **Subtract to Clean, Add to Enrich: Dual-Path Disentanglement for Speaker and Language Recognition**  
> Aref Farhadipour — Odyssey 2026

**Code:** [github.com/areffarhadi/ASV-LID](https://github.com/areffarhadi/ASV-LID)

This repository provides:

- **ASV** — WPT + W2V-BERT-2.0 + MHFA (speaker verification)
- **LID** — WPT + W2V-BERT-2.0 + MHFA (TidyVoice + VoxLingua)
- **Dual-path fusion** — subtractive + additive heads for ASV and LID
- **Unified inference** — one shared backbone, wav in → language labels + pair scores

## Citation

```bibtex
@inproceedings{farhadipour2026subtract,
  title={Subtract to Clean, Add to Enrich: Dual-Path Disentanglement for Speaker and Language Recognition},
  author={Farhadipour, Aref},
  booktitle={Odyssey 2026},
  year={2026}
}
```

## Quick start

```bash
git clone https://github.com/areffarhadi/ASV-LID.git
cd ASV-LID

bash scripts/setup_env.sh
source .venv/bin/activate
source config/paths.sh

# Checkpoints download automatically on first inference/eval run
bash scripts/run_demo.sh 0
```

### What you need to provide

| Item | Action |
|------|--------|
| **Checkpoints** | Auto-downloaded from Hugging Face on first run |
| **Manifests & trial lists** | Download from [TidyLang2026-baseline/data](https://github.com/areffarhadi/TidyLang2026-baseline/tree/main/data) into `data/` — see [data/README.md](data/README.md) |
| **Audio corpora** | Set `TIDYVOICEX_TRAIN`, `TIDYVOICEX_DEV`, etc. (training / raw-audio eval) |
| **VoxLingua eval** | Set `VOXLINGUA_EVAL_ROOT` to your `dev_vox` folder |

Manual checkpoint download:

```bash
bash scripts/download_checkpoints.sh
```

| Model | Hugging Face |
|-------|----------------|
| TidyVoice (ASV + LID + fusion) | [areffarhadi/w2v-bert-TidyVoice](https://huggingface.co/areffarhadi/w2v-bert-TidyVoice) |
| VoxLingua LID | [areffarhadi/w2v-bert-VoxLingua](https://huggingface.co/areffarhadi/w2v-bert-VoxLingua) |

## Inference (wav → results)

Put **at least two** `.wav` files in a folder and run:

```bash
bash scripts/run_inference.sh 0 /path/to/wavs
```

The bundled demo uses `demo_pair/` (English + Lithuanian, same speaker). Output includes:

| Output | Meaning |
|--------|---------|
| **LID-only** | Language from pure WPT LID head |
| **LID-fused** | Language from dual-path LID fusion |
| **ASV-only** | Cosine similarity of speaker embeddings |
| **ASV-fused** | Dual-path ASV fusion score |
| **LID-fused** (pair) | Cosine similarity of fused LID embeddings |

Results print to the terminal and save to `results/inference/results.json`.

Full guide: **[docs/INFERENCE.md](docs/INFERENCE.md)**

### How it works

`src/inference/` loads paper checkpoints once and runs a **joint WPT** pass: shared frozen W2V-BERT, separate ASV/LID prompt-conditioned encoder passes, then fusion heads.

```
wav → JointWPTMHFAInference → ASV emb + LID emb
                                    ↓
              fusion heads → languages + pair scores
```

## Datasets

### LID manifests and trials

Official files: **[TidyLang2026-baseline/data](https://github.com/areffarhadi/TidyLang2026-baseline/tree/main/data)**

Place under `data/manifests/` and `data/trials/` (default paths in `config/paths.sh`).

### ASV audio corpora

| Split | Dataset | Link |
|-------|---------|------|
| Train + Dev | **TidyVoiceX_ASV** | [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmihtsewu023so207xot1iqqw) |
| Eval (tv26) | **TidyVoiceX2_ASV** | [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmkv32i5e02tumg07j79d3c35) |

## Evaluation

```bash
# Paper ASV benchmarks (needs pre-extracted embeddings — see docs/REPLICATION.md)
bash scripts/eval_asv.sh 0

# Paper LID / TL26 benchmarks
bash scripts/eval_lid.sh 0

# VoxLingua LID (set VOXLINGUA_EVAL_ROOT first)
export VOXLINGUA_EVAL_ROOT=/path/to/VoxLingua90/dev_vox
bash scripts/eval_voxlingua_lid.sh 0
```

## Training pipeline

1. Place manifests and trials in `data/` — [data/README.md](data/README.md), [docs/MANIFEST.md](docs/MANIFEST.md)
2. Set `TIDYVOICEX_TRAIN` and `TIDYVOICEX_DEV`
3. `bash scripts/train_asv.sh [GPU]`
4. `bash scripts/train_lid.sh [GPU] [EPOCHS]`
5. Extract embeddings (see `docs/REPLICATION.md`)
6. `bash scripts/train_fusion_asv.sh [GPU]`
7. `bash scripts/train_fusion_lid.sh [GPU]`

## Layout

```
config/paths.sh          # paths + env vars
data/                    # manifests and trial lists (user-provided)
src/wpt/                 # ASV + LID WPT trainers
src/fusion_asv/          # dual-path ASV fusion + eval
src/fusion_lid/          # dual-path LID fusion + TL26 eval
src/inference/           # unified joint WPT + end-to-end pipeline
src/voxlingua_lid/       # VoxLingua LID eval
scripts/                 # train, eval, inference wrappers
demo_pair/               # bundled inference demo wavs
checkpoints/             # paper weights (from Hugging Face)
docs/                    # guides
```

## Requirements

`requirements.txt` — torch 2.1.2, **transformers 4.40.0** (Wav2Vec2Bert), torchaudio, huggingface-hub, etc.

W2V-BERT-2.0 downloads to `HF_HOME` (default: `.cache/huggingface`) on first run.
