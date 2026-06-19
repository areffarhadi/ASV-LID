"""VoxLingua folder-structure LID dataset (lang subdirs with wav/flac)."""

from __future__ import annotations

import os

import torch
import torchaudio
from torch.utils.data import Dataset


def pad_waveform(wav: torch.Tensor, audio_length: int = 96000) -> torch.Tensor:
    waveform = wav.squeeze(0)
    if waveform.shape[0] >= audio_length:
        waveform = waveform[:audio_length]
    else:
        repeats = int(audio_length / waveform.shape[0]) + 1
        waveform = waveform.repeat(repeats)[:audio_length]
    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    return waveform


class VoxLinguaLangIDDataset(Dataset):
    """Each subfolder name is a language label; files are utterances."""

    def __init__(
        self,
        root_dir: str,
        audio_length: int = 96000,
        label_map: dict[str, int] | None = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.audio_length = audio_length
        self.files: list[str] = []
        languages: set[str] = set()

        for lang in sorted(os.listdir(root_dir)):
            lang_path = os.path.join(root_dir, lang)
            if not os.path.isdir(lang_path):
                continue
            for fname in sorted(os.listdir(lang_path)):
                if not fname.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                    continue
                self.files.append(os.path.join(lang, fname))
                languages.add(lang)

        if label_map is None:
            self.label_map = {lang: idx for idx, lang in enumerate(sorted(languages))}
        else:
            self.label_map = label_map

        self.id_to_lang = {v: k for k, v in self.label_map.items()}
        self.num_languages = len(self.label_map)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        rel_path = self.files[idx]
        lang = rel_path.split(os.sep)[0]
        label = self.label_map[lang]
        full_path = os.path.join(self.root_dir, rel_path)

        waveform, sr = torchaudio.load(full_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = pad_waveform(waveform, self.audio_length)

        filename = os.path.splitext(os.path.basename(rel_path))[0]
        return waveform, filename, label
