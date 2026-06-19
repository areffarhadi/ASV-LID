"""Audio I/O and scoring helpers for inference."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchaudio


def load_lang_map(manifest_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    langs = set()
    with open(manifest_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                langs.add(parts[2])
    sorted_langs = sorted(langs)
    lang_to_id = {lang: i for i, lang in enumerate(sorted_langs)}
    id_to_lang = {i: lang for lang, i in lang_to_id.items()}
    return lang_to_id, id_to_lang


def load_waveform(path: str, audio_len: int = 64600) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if len(waveform) < audio_len:
        waveform = F.pad(waveform, (0, audio_len - len(waveform)))
    else:
        waveform = waveform[:audio_len]
    return waveform


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(F.cosine_similarity(a, b).item())
