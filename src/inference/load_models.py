"""Load paper ASV/LID WPT checkpoints for joint inference."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from types import ModuleType

import torch
import torch.nn as nn
from transformers import AutoConfig


def _load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _infer_prompt_shape(state_dict: dict, key: str) -> int:
    return state_dict[f"wpt_w2vbert.{key}"].shape[1]


def _import_from_dir(module_name: str, filename: str, code_dir: str) -> ModuleType:
    path = os.path.join(code_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training module not found: {path}")
    if code_dir in sys.path:
        sys.path.remove(code_dir)
    sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WPTPromptsOnly(nn.Module):
    """LID/ASV prompt + wavelet tokens only — no W2V-BERT weights (shared at inference)."""

    def __init__(
        self,
        num_layers: int,
        num_prompt_tokens: int,
        num_wavelet_tokens: int,
        wavelet_block_cls,
        prompt_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.num_wavelet_tokens = num_wavelet_tokens
        self.prompt_dim = prompt_dim
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers, num_prompt_tokens, prompt_dim)
        )
        self.wavelet_prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers, num_wavelet_tokens, prompt_dim)
        )
        self.wavelet_block = wavelet_block_cls(
            wave="haar", J=1, input_dim=prompt_dim, output_dim=prompt_dim
        )
        val = math.sqrt(6.0 / float(2 * prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        nn.init.uniform_(self.wavelet_prompt_embeddings.data, -val, val)
        self.prompt_dropout = nn.Dropout(p=dropout)


class LIDBranchOnly(nn.Module):
    """LID prompts + MHFA head + ArcFace — backbone supplied separately."""

    def __init__(
        self,
        wpt_prompts: WPTPromptsOnly,
        mhfa_head: nn.Module,
        arcface: nn.Module,
        num_languages: int,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.wpt_w2vbert = wpt_prompts
        self.mhfa_head = mhfa_head
        self.arcface = arcface
        self.num_languages = num_languages
        self.embedding_dim = embedding_dim


def load_asv_model(asv_code: str, ckpt_path: str, ssl_model: str, device: torch.device):
    """Paper ASV ckpt uses dynstats_ecapa head (src/wpt/main_train_asv_dynstats.py)."""
    mod = _import_from_dir(
        "asv_train_dynstats",
        "main_train_asv_dynstats.py",
        asv_code,
    )
    SimpleSVModelWPTW2VBERTMHFA = mod.SimpleSVModelWPTW2VBERTMHFA

    sd = _load_state_dict(ckpt_path)
    num_speakers = sd["arcface_loss.weight"].shape[0]
    num_prompt_tokens = _infer_prompt_shape(sd, "prompt_embeddings")
    num_wavelet_tokens = _infer_prompt_shape(sd, "wavelet_prompt_embeddings")
    head_type = "dynstats_ecapa" if any("temporal_block" in k for k in sd) else "dynstats"

    model = SimpleSVModelWPTW2VBERTMHFA(
        model_dir=ssl_model,
        num_speakers=num_speakers,
        embedding_dim=256,
        num_prompt_tokens=num_prompt_tokens,
        num_wavelet_tokens=num_wavelet_tokens,
        head_type=head_type,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def load_lid_branch_only(
    lid_code: str,
    ckpt_path: str,
    ssl_model: str,
    device: torch.device,
    num_layers: int | None = None,
):
    """LID prompts + head only. Does not load a second W2V-BERT (use with shared ASV backbone)."""
    lid_mod = _import_from_dir("lid_train", "main_train_lid.py", lid_code)
    MHFAHeadImproved = lid_mod.MHFAHeadImproved
    ArcFaceLoss = lid_mod.ArcFaceLoss
    from main_train_asv import WaveletBlock

    sd = _load_state_dict(ckpt_path)
    num_languages = sd["arcface.weight"].shape[0]
    num_prompt_tokens = _infer_prompt_shape(sd, "prompt_embeddings")
    num_wavelet_tokens = _infer_prompt_shape(sd, "wavelet_prompt_embeddings")

    if num_layers is None:
        num_layers = AutoConfig.from_pretrained(ssl_model).num_hidden_layers

    wpt = WPTPromptsOnly(
        num_layers=num_layers,
        num_prompt_tokens=num_prompt_tokens,
        num_wavelet_tokens=num_wavelet_tokens,
        wavelet_block_cls=WaveletBlock,
    )
    mhfa_head = MHFAHeadImproved(
        feature_dim=1024,
        num_layers=num_layers,
        num_heads=8,
        compression_dim=128,
        embedding_dim=256,
        adapter_bottleneck=128,
        dropout=0.1,
    )
    arcface = ArcFaceLoss(
        in_features=256,
        out_features=num_languages,
        margin=0.3,
        scale=30.0,
    )
    model = LIDBranchOnly(wpt, mhfa_head, arcface, num_languages)
    lid_sd = {k: v for k, v in sd.items() if not k.startswith("wpt_w2vbert.model.")}
    model.load_state_dict(lid_sd, strict=True)
    model.to(device)
    model.eval()
    return model
