Download (automatic on first run): `bash scripts/ensure_checkpoints.sh`

Or manually: `bash scripts/download_checkpoints.sh`

| Hugging Face repo | Contents |
|-------------------|----------|
| [areffarhadi/w2v-bert-TidyVoice](https://huggingface.co/areffarhadi/w2v-bert-TidyVoice) | `asv_wpt.pt`, `lid_wpt.pt`, `fusion_asv.pt`, `fusion_lid.pth` |
| [areffarhadi/w2v-bert-VoxLingua](https://huggingface.co/areffarhadi/w2v-bert-VoxLingua) | `voxlingua_lid/model.pt`, `args.json` |

Environment variables (set in `config/paths.sh`):

- `CKPT_ASV` → `asv_wpt.pt`
- `CKPT_LID` → `lid_wpt.pt`
- `CKPT_FUSION_ASV` → `fusion_asv.pt`
- `CKPT_FUSION_LID` → `fusion_lid.pth`
- `CKPT_VOXLINGUA_LID_DIR` → `voxlingua_lid/`
