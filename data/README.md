# Data layout

The only manual setup step for inference and evaluation is to download **manifests and trial lists** and place them here. Checkpoints are downloaded automatically on first run.

## Required for LID training / fusion / TL26 eval

Copy from [TidyLang2026-baseline/data](https://github.com/areffarhadi/TidyLang2026-baseline/tree/main/data):

```
data/
  manifests/
    training_manifest.txt
  trials/
    lid/
      trials_Dev.txt
      enrollment_manifest.tsv
    tl26/
      tl26_lid.txt
      tl26_pairs.txt
      tl26_enroll.tsv
```

## Required for ASV benchmark eval

Copy ASV trial lists into:

```
data/trials/asv/
  test_4types_trials_dev2.txt
  task1_labels.txt
  task2_labels.txt
```

Official lists are linked from the [TidyVoiceX2 dataset page](https://mozilladatacollective.com/datasets/cmkv32i5e02tumg07j79d3c35).

## Audio corpora (not stored in this repo)

Set environment variables before training or raw-audio eval:

| Variable | Dataset |
|----------|---------|
| `TIDYVOICEX_TRAIN` | [TidyVoiceX_ASV](https://mozilladatacollective.com/datasets/cmihtsewu023so207xot1iqqw) train split |
| `TIDYVOICEX_DEV` | TidyVoiceX_ASV dev split |
| `TIDYVOICEX2_ASV` | [TidyVoiceX2_ASV](https://mozilladatacollective.com/datasets/cmkv32i5e02tumg07j79d3c35) eval audio |
| `VOXLINGUA_EVAL_ROOT` | VoxLingua `dev_vox` folder layout |

Example:

```bash
export TIDYVOICEX_TRAIN=/path/to/TidyVoiceX_Train
export TIDYVOICEX_DEV=/path/to/TidyVoiceX_Dev
export VOXLINGUA_EVAL_ROOT=/path/to/VoxLingua90/dev_vox
```

## Optional: pre-extracted embeddings

Full ASV/LID benchmark reproduction needs stage embeddings under `data/embeddings/`. See `docs/REPLICATION.md`.
