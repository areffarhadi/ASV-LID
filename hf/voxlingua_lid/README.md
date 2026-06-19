---
license: mit
tags:
  - language-identification
  - wav2vec2-bert
  - voxlingua
  - odyssey-2026
library_name: pytorch
---

# VoxLingua LID — WPT + MHFA (Odyssey 2026)

Language identification checkpoint trained on **VoxLingua107**, evaluated on `dev_vox`.

Related paper: **Subtract to Clean, Add to Enrich: Dual-Path Disentanglement for Speaker and Language Recognition** (Odyssey 2026).

**Author:** [Aref Farhadipour](https://huggingface.co/areffarhadi)

## Files

| File | Description |
|------|-------------|
| `model.pt` | WPT + W2V-BERT-2.0 + MHFA classifier (107 languages) |
| `args.json` | Training hyperparameters |

- Backbone: [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)
- Prompt tokens: 15 regular + 8 wavelet per layer
- Clip length: 96000 samples (6 s @ 16 kHz)

Reported performance: see the **paper**.

## Download

```bash
huggingface-cli download areffarhadi/w2v-bert-VoxLingua --local-dir ./checkpoints/voxlingua_lid
```

## Code

GitHub: [areffarhadi/ASV-LID](https://github.com/areffarhadi/ASV-LID)

## Eval

Dataset layout: `dev_vox/<lang>/*.wav`

```bash
bash scripts/eval_voxlingua_lid.sh 0
```

## Citation

```bibtex
@inproceedings{farhadipour2026subtract,
  title={Subtract to Clean, Add to Enrich: Dual-Path Disentanglement for Speaker and Language Recognition},
  author={Farhadipour, Aref},
  booktitle={Proc. Odyssey},
  year={2026}
}
```
