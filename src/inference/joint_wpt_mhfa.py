"""
Joint WPT + MHFA inference (single-backbone mode).

One frozen W2V-BERT (from the ASV checkpoint); separate ASV/LID prompt tokens and
MHFA heads. Audio is feature-projected once, then two prompt-conditioned passes
through the same encoder layers. Inference-only — no retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PromptBranch:
    prompt_embeddings: nn.Parameter
    wavelet_prompt_embeddings: nn.Parameter
    wavelet_block: nn.Module
    prompt_dropout: nn.Module
    num_prompt_tokens: int
    num_wavelet_tokens: int


def _audio_to_features(processor, audio_data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if audio_data.dim() == 1:
        audio_data = audio_data.unsqueeze(0)

    if audio_data.is_cuda:
        audio_cpu = audio_data.detach().cpu().numpy()
    elif isinstance(audio_data, torch.Tensor):
        audio_cpu = audio_data.numpy()
    else:
        audio_cpu = audio_data

    processed = processor(audio_cpu, sampling_rate=16000, return_tensors="pt")
    if "input_features" in processed:
        feat = processed["input_features"]
    elif hasattr(processed, "input_features"):
        feat = processed.input_features
    elif "input_values" in processed:
        feat = processed["input_values"]
    else:
        feat = list(processed.values())[0]

    feat = feat.to(device)
    if feat.dim() > 3:
        feat = feat.squeeze(0)
    elif feat.dim() < 3:
        feat = feat.unsqueeze(0)
    return feat


def _project_audio(
    ssl_model: nn.Module,
    processor,
    audio_data: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    feat = _audio_to_features(processor, audio_data, device)
    batch_size = feat.size(0)
    with torch.no_grad():
        hidden_state, _ = ssl_model.feature_projection(feat)
        hidden_state = ssl_model.encoder.dropout(hidden_state)
    return hidden_state, batch_size


def encode_layer_features(
    ssl_model: nn.Module,
    config,
    hidden_state: torch.Tensor,
    batch_size: int,
    branch: PromptBranch,
) -> List[torch.Tensor]:
    total_prompt_tokens = branch.num_wavelet_tokens + branch.num_prompt_tokens
    layer_features: List[torch.Tensor] = []

    for layer_idx in range(config.num_hidden_layers):
        prompt = branch.prompt_embeddings[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
        prompt = branch.prompt_dropout(prompt)

        wavelet_prompt = branch.wavelet_prompt_embeddings[layer_idx].unsqueeze(0).expand(
            batch_size, -1, -1
        )
        wavelet_prompt = branch.wavelet_block(wavelet_prompt)
        wavelet_prompt = branch.prompt_dropout(wavelet_prompt)

        if layer_idx == 0:
            hidden_state = torch.cat([wavelet_prompt, prompt, hidden_state], dim=1)
        else:
            audio_features = hidden_state[:, total_prompt_tokens:, :]
            hidden_state = torch.cat([wavelet_prompt, prompt, audio_features], dim=1)

        hidden_state = ssl_model.encoder.layers[layer_idx](hidden_state)[0]
        audio_only = hidden_state[:, total_prompt_tokens:, :].clone()
        layer_features.append(audio_only)

    return layer_features


def _branch_from_wpt(wpt_module) -> PromptBranch:
    return PromptBranch(
        prompt_embeddings=wpt_module.prompt_embeddings,
        wavelet_prompt_embeddings=wpt_module.wavelet_prompt_embeddings,
        wavelet_block=wpt_module.wavelet_block,
        prompt_dropout=wpt_module.prompt_dropout,
        num_prompt_tokens=wpt_module.num_prompt_tokens,
        num_wavelet_tokens=wpt_module.num_wavelet_tokens,
    )


class JointWPTMHFAInference(nn.Module):
    """Inference-only composition of ASV + LID WPT specialists."""

    def __init__(self, asv_model, lid_model, verify_backbone: bool = False):
        super().__init__()
        self.asv_model = asv_model
        self.lid_model = lid_model

        asv_wpt = asv_model.wpt_w2vbert
        lid_wpt = lid_model.wpt_w2vbert

        if verify_backbone and hasattr(lid_wpt, "model") and lid_wpt.model is not None:
            _assert_shared_backbone(asv_wpt.model, lid_wpt.model)

        self.ssl_model = asv_wpt.model
        self.processor = asv_wpt.processor
        self.config = asv_wpt.config
        self.asv_branch = _branch_from_wpt(asv_wpt)
        self.lid_branch = _branch_from_wpt(lid_wpt)
        self.asv_head = asv_model.mhfa_head
        self.lid_head = lid_model.mhfa_head
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def extract_both(
        self, audio_data: torch.Tensor, normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state, batch_size = _project_audio(
            self.ssl_model, self.processor, audio_data, self.device
        )
        lf_asv = encode_layer_features(
            self.ssl_model, self.config, hidden_state.clone(), batch_size, self.asv_branch
        )
        lf_lid = encode_layer_features(
            self.ssl_model, self.config, hidden_state.clone(), batch_size, self.lid_branch
        )
        emb_asv = self.asv_head(lf_asv)
        emb_lid = self.lid_head(lf_lid)
        if normalize:
            emb_asv = F.normalize(emb_asv, p=2, dim=1)
            emb_lid = F.normalize(emb_lid, p=2, dim=1)
        return emb_asv, emb_lid


def _assert_shared_backbone(asv_ssl: nn.Module, lid_ssl: nn.Module, atol: float = 0.0) -> None:
    asv_sd = asv_ssl.state_dict()
    lid_sd = lid_ssl.state_dict()
    if set(asv_sd.keys()) != set(lid_sd.keys()):
        raise ValueError("ASV and LID backbone keys differ; cannot share safely.")
    max_diff = 0.0
    for key in asv_sd:
        diff = (asv_sd[key].float() - lid_sd[key].float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > atol:
            raise ValueError(f"Backbone mismatch on {key}: max_abs_diff={diff:.6g}")
    print(f"Backbone parity OK (max_abs_diff={max_diff:.6g})")
