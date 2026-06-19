# Replication guide

For **wav-file inference** (no embedding cache), see [INFERENCE.md](INFERENCE.md).

## Prerequisites

- TidyVoiceX train/dev audio on disk
- LID manifest + TL26 / ASV trial files in `data/` (see [data/README.md](../data/README.md))
- Pre-extracted embeddings under `EMB_TIDYVOICE` and `EMB_TIDYVOICEX2` (or re-extract from WPT checkpoints)
- GPU + CUDA; `pip install -r requirements.txt`

## Step 1 — Manifest

Create `training_manifest.txt` (tab-separated). See `docs/MANIFEST.md` and `protocols/manifest.example.txt`.

## Step 2 — Evaluate paper checkpoints (fastest)

```bash
source config/paths.sh
bash scripts/eval_asv.sh 0    # -> results/runs/asv_mindcf/
bash scripts/eval_lid.sh 0    # -> results/runs/lid_tl26/tl26_eval_summary.json
```

Compare your results to the **paper**.

## Step 3 — Full re-training

```bash
bash scripts/train_asv.sh 0
bash scripts/train_lid.sh 0 15
bash scripts/train_fusion_asv.sh 0 30
bash scripts/train_fusion_lid.sh 0 20
```

Fusion training expects cached embeddings at:
- `$EMB_TIDYVOICE/stage1_asv/` and `stage2_lid/`
- `$EMB_TIDYVOICEX2/` for TL26 / Task eval

## Architecture summary

**ASV fusion (`DualPathModel`):**
- Sub: remove language from speaker embedding
- Add: enrich speaker using language (cross-attention)

**LID fusion (`DualPathLanguageExtractor`):**
- Sub: remove speaker from language embedding
- Add: enrich language using speaker

Eval score fusion strategies include `sub`, `add`, `sub+add-lid`, `full`, `lid+sub+add`.
