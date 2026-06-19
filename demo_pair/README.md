# Demo: same speaker, two languages

Speaker **id013915** from TidyVoiceX Dev.

| File | Language |
|------|----------|
| `speaker_en.wav` | English (`en`) |
| `speaker_lt.wav` | Lithuanian (`lt`) |

## Run

From repo root:

```bash
source config/paths.sh
bash scripts/run_demo.sh 0
```

## Expected output

**Languages** (per file): `en` and `lt` for both LID-only and LID-fused heads.

**Pair scores** (cross-language, same speaker):

| Metric | Interpretation |
|--------|----------------|
| ASV-only | High — same speaker |
| LID-only | Low — different languages |
| ASV-fused | High — fusion speaker match |
| LID-fused | Low — fused language embeddings differ |

Replace these wavs with your own (≥2 files per folder) and run `bash scripts/run_inference.sh GPU /path/to/wavs`.
