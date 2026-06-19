# Building the LID training manifest

Official train/dev manifests and trials: [TidyLang2026-baseline/data](https://github.com/areffarhadi/TidyLang2026-baseline/tree/main/data)

Tab-separated format:

```
<flag>	<relative_wav_path>	<language>
```

| Flag | Split | Paper name |
|------|-------|------------|
| 1 | Training | — |
| 2 | Validation (new speakers) | valid1 |
| 3 | Cross-lingual validation | val2 |

Paths are relative to dataset roots passed as `--dataset_roots` (see `config/paths.sh`).

Enrollment trials for LID verification EER:
- `trials_Dev.txt` + `enrollment_manifest.tsv`
