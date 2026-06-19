---
license: mit
tags:
  - speaker-verification
  - language-identification
  - wav2vec2-bert
  - tidyvoice
  - odyssey-2026
library_name: pytorch
---

# TidyVoice WPT + Dual-Path Fusion (Odyssey 2026)

Checkpoints for **Subtract to Clean, Add to Enrich: Dual-Path Disentanglement for Speaker and Language Recognition** (Odyssey 2026).

**Author:** [Aref Farhadipour](https://huggingface.co/areffarhadi)

## Files

| File | Description |
|------|-------------|
| `asv_wpt.pt` | ASV — WPT + W2V-BERT-2.0 + MHFA (TidyVoiceX) |
| `lid_wpt.pt` | LID — WPT + W2V-BERT-2.0 + MHFA (TidyLang / TidyVoice) |
| `fusion_asv.pt` | Dual-path ASV fusion (subtractive + additive) |
| `fusion_lid.pth` | Dual-path LID fusion |

Backbone: [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) (frozen, prompt-tuned).

## Download

```bash
huggingface-cli download areffarhadi/w2v-bert-TidyVoice --local-dir ./checkpoints
```

## ASV fusion results (TidyVoice)

Validation set and evaluation tasks **tv26_eval-A** / **tv26_eval-U**. EER (%) and min-DCF.

| Fusion Strategy | Val EER | Val min-DCF | tv26_eval-A EER | tv26_eval-A min-DCF | tv26_eval-U EER | tv26_eval-U min-DCF |
|-----------------|--------:|------------:|------------------:|--------------------:|------------------:|---------------------:|
| ASV-only | 3.01 | 0.762 | 8.27 | 0.520 | 9.54 | 0.570 |
| ASV − LID (Logit Penalty) | 2.20 | 0.664 | 6.07 | 0.380 | 7.48 | 0.526 |
| Subtractive-only (Sub) | 2.21 | 0.700 | 7.15 | 0.467 | 8.91 | 0.537 |
| Additive-only (Add) | 2.25 | 0.709 | 7.14 | 0.466 | 9.82 | 0.599 |
| ASV + Sub | 2.25 | 0.703 | 7.18 | 0.469 | 8.92 | 0.536 |
| ASV + Add | 2.29 | 0.712 | 7.17 | 0.469 | 9.49 | 0.568 |
| Sub + Add | 2.21 | 0.700 | 7.08 | 0.462 | 8.97 | 0.541 |
| Sub − LID | 1.89 | 0.639 | 5.70 | 0.372 | 7.20 | 0.513 |
| Add − LID | 1.89 | 0.641 | 5.66 | 0.364 | 8.08 | 0.554 |
| Sub + Add − LID | **1.87** | 0.637 | 5.62 | 0.363 | 7.41 | 0.521 |
| ASV + Sub + Add | 2.29 | 0.706 | 7.13 | 0.465 | 9.11 | 0.545 |
| Full (αASV + βSub + γAdd − δLID) | 1.88 | 0.628 | 5.61 | 0.360 | 7.22 | 0.511 |
| Full + VoxCeleb ASV Head | 1.64 | 0.648 | 5.13 | 0.344 | 6.92 | 0.491 |
| Full + VC Head + AS-Norm & Calib. | **1.09** | **0.126** | **3.16** | **0.130** | **4.35** | **0.272** |

## Code

GitHub: [areffarhadi/ASV-LID](https://github.com/areffarhadi/ASV-LID)

```bash
bash scripts/download_checkpoints.sh   # TidyVoice bundle
bash scripts/eval_asv.sh 0
bash scripts/eval_lid.sh 0
bash scripts/run_demo.sh 0
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
