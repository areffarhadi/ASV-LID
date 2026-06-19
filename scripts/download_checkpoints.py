#!/usr/bin/env python3
"""Download paper checkpoints from Hugging Face into checkpoints/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "checkpoints"
VOX_DIR = CKPT_DIR / "voxlingua_lid"

TIDYVOICE_FILES = ("asv_wpt.pt", "lid_wpt.pt", "fusion_asv.pt", "fusion_lid.pth")
VOXLINGUA_FILES = ("model.pt", "args.json")


def _missing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not p.is_file()]


def _download(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading https://huggingface.co/{repo_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )


def main() -> int:
    hf_tidy = os.environ.get("HF_REPO_TIDYVOICE", "areffarhadi/w2v-bert-TidyVoice")
    hf_vox = os.environ.get("HF_REPO_VOXLINGUA", "areffarhadi/w2v-bert-VoxLingua")

    tidy_missing = _missing([CKPT_DIR / name for name in TIDYVOICE_FILES])
    vox_missing = _missing([VOX_DIR / name for name in VOXLINGUA_FILES])

    if not tidy_missing and not vox_missing:
        print("All checkpoints present.")
        return 0

    if tidy_missing:
        print(f"Missing TidyVoice files: {[p.name for p in tidy_missing]}")
        _download(hf_tidy, CKPT_DIR)

    if vox_missing:
        print(f"Missing VoxLingua files: {[p.name for p in vox_missing]}")
        _download(hf_vox, VOX_DIR)

    still_missing = _missing(
        [CKPT_DIR / name for name in TIDYVOICE_FILES]
        + [VOX_DIR / name for name in VOXLINGUA_FILES]
    )
    if still_missing:
        print("Download finished but some files are still missing:", file=sys.stderr)
        for path in still_missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    print("Checkpoint download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
