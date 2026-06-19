# VoxLingua LID evaluation

Paper checkpoint for **VoxLingua** language identification on `dev_vox`.

This is separate from the TidyVoice/TidyLang LID track (`CKPT_LID` → `lid_wpt.pt`). It uses a different architecture head (standard MHFA classifier, not ArcFace) and longer clips (`audio_len=96000`, 6 s at 16 kHz).

Reported performance: see the **paper**.

## Checkpoint

| File | Role |
|------|------|
| `checkpoints/voxlingua_lid/model.pt` | Paper VoxLingua LID weights |
| `checkpoints/voxlingua_lid/args.json` | Model hyperparameters |

Source: paper VoxLingua LID training run (see paper for details).

### Model config

| Parameter | Value |
|-----------|-------|
| Backbone | `facebook/w2v-bert-2.0` (frozen) |
| Prompt tokens | 15 regular + 8 wavelet per layer |
| MHFA | 8 heads, compression 128 → 256-D embedding |
| Classes | 107 languages (label map stored in checkpoint) |
| Augmentation | None (`use_augmentation: false`) |

## Dataset layout

VoxLingua eval data must follow folder-per-language layout:

```
dev_vox/
  en/
    utterance1.wav
  fr/
    utterance2.flac
  ...
```

Set before evaluation:

```bash
export VOXLINGUA_EVAL_ROOT=/path/to/VoxLingua90/dev_vox
```

## Run evaluation

```bash
source config/paths.sh
bash scripts/eval_voxlingua_lid.sh 0
```

Output:

- Terminal: overall accuracy + per-language breakdown
- `results/runs/voxlingua_lid/eval_summary.json`

Compare your run to the numbers in the **paper**.

### Custom paths

```bash
VOXLINGUA_EVAL_ROOT=/path/to/dev_vox \
CKPT_VOXLINGUA_LID_DIR=/path/to/checkpoint_dir \
bash scripts/eval_voxlingua_lid.sh 0
```

Or directly:

```bash
cd src/voxlingua_lid
python eval_voxlingua.py \
  --checkpoint_dir ../../checkpoints/voxlingua_lid \
  --eval_audio_root "$VOXLINGUA_EVAL_ROOT" \
  --gpu 0
```

## Code layout

```
src/voxlingua_lid/
  model.py           # WPT + W2V-BERT + MHFA + linear classifier
  dataset.py         # Folder-structure VoxLingua dataset
  eval_voxlingua.py  # Accuracy evaluation (inference only)
scripts/eval_voxlingua_lid.sh
```

Evaluation uses argmax on classifier logits (same protocol as the original VoxLingua training script).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Eval audio root not found` | Download VoxLingua and set `VOXLINGUA_EVAL_ROOT` |
| `cannot import Wav2Vec2BertModel` | Use repo `.venv` (`transformers>=4.40`) |
| Accuracy mismatch | Ensure `label_map` from checkpoint is used (handled automatically); do not shuffle eval data |
