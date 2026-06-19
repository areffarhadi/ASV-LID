# ASV-LID — Unified inference guide

End-to-end inference runs **raw wav files** through:

1. **Joint WPT backbone** — one shared frozen W2V-BERT, separate ASV/LID prompts and heads
2. **LID heads** — pure WPT LID + dual-path LID fusion (language names)
3. **ASV fusion head** — dual-path subtract+enrich for speaker pair scoring

No embedding cache or trial files required for inference.

## Prerequisites

```bash
git clone https://github.com/areffarhadi/ASV-LID.git
cd ASV-LID
bash scripts/setup_env.sh          # creates .venv with transformers 4.40
source .venv/bin/activate
source config/paths.sh
```

Checkpoints download automatically on first run (from Hugging Face).

`ASV_CODE` defaults to `src/wpt/` (`main_train_asv_dynstats.py` — dynstats_ecapa head). LID loads from `src/wpt/main_train_lid.py`.

## Quick demo

Bundled example: same speaker, English + Lithuanian (`demo_pair/`).

```bash
bash scripts/run_demo.sh 0
```

### Output

**Per file**

| Column | Description |
|--------|-------------|
| LID-only | Language from pure WPT LID ArcFace head |
| LID-fused | Language from dual-path LID fusion (0.5×sub + 0.5×add logits) |

**Per pair** (all wav pairs in the folder)

| Score | Description |
|-------|-------------|
| ASV-only | Cosine similarity of speaker embeddings |
| LID-only | Cosine similarity of pure language embeddings |
| ASV-fused | `α×sub + β×add − δ×lid_cos` (recipe stored in fusion ASV checkpoint) |
| LID-fused | Cosine similarity of fused LID embeddings |

Results are printed and saved to `results/inference/results.json`.

## Custom wav folder

Put **at least two** `.wav` files (16 kHz mono preferred; resampled automatically) in a folder:

```bash
bash scripts/run_inference.sh 0 /path/to/your/wavs
```

## Architecture

```
wav
 └─> Joint WPT (1× W2V-BERT, 2× prompt passes)
       ├─> ASV emb ──┬─> LID-only language (ArcFace on LID emb)
       │             ├─> ASV-only pair cosine
       │             └─> ASV fusion (sub + add paths) ──> ASV-fused score
       └─> LID emb ──┬─> LID fusion classifier ──> LID-fused language
                     ├─> LID-only pair cosine
                     └─> LID fusion embeddings ──> LID-fused pair cosine
```

## Paper benchmark eval (separate from inference)

Uses pre-extracted embeddings + trial lists:

```bash
bash scripts/eval_asv.sh 0
bash scripts/eval_lid.sh 0
```

See `docs/REPLICATION.md` for full training and benchmark replication.

## Verify joint embeddings (optional)

Confirms the unified backbone reproduces cached `stage1_asv` / `stage2_lid` extractions:

```bash
bash scripts/verify_joint_embeddings.sh 0 3
```

Requires `EMB_TIDYVOICE` caches and `TIDYVOICEX_DEV` wavs (defaults in `config/paths.sh`).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `cannot import Wav2Vec2BertModel` | Use repo `.venv` (`transformers>=4.40`), not older add-env |
| CUDA OOM | Use a free GPU index; one W2V-BERT (~4 GB) + fusion heads |
| Wrong language labels | Ensure `LID_MANIFEST` points to the training manifest (35 languages) |
