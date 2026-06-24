"""WPT + W2V-BERT-2.0 + MHFA language ID model (VoxLingua checkpoint)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2BertModel


# WaveletBlock + WPTW2VBERTMultiLayer are shared from src/wpt/peft_wpt.py so the
# --peft_mode (deep_prompt / shallow_prompt / prefix) and --use_wavelet options
# are available here too. The repo scripts put src/wpt on PYTHONPATH; the
# fallback below keeps a direct `import model` working from any CWD.
import os as _os
import sys as _sys
_WPT_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "wpt")
if _WPT_DIR not in _sys.path:
    _sys.path.insert(0, _WPT_DIR)
from peft_wpt import WaveletBlock, WPTW2VBERTMultiLayer  # noqa: E402,F401


class MHFAHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 8,
        compression_dim: int = 128,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.compression_dim = compression_dim
        self.embedding_dim = embedding_dim

        self.layer_weights_key = nn.Parameter(torch.zeros(num_layers))
        self.layer_weights_value = nn.Parameter(torch.zeros(num_layers))
        nn.init.uniform_(self.layer_weights_key.data, -0.1, 0.1)
        nn.init.uniform_(self.layer_weights_value.data, -0.1, 0.1)

        self.key_projection = nn.Linear(feature_dim, compression_dim)
        self.value_projection = nn.Linear(feature_dim, compression_dim)
        self.attention_projection = nn.Linear(compression_dim, num_heads)
        self.embedding_projection = nn.Sequential(
            nn.Linear(num_heads * compression_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_features):
        if isinstance(layer_features, list):
            layer_features = torch.stack(layer_features, dim=0)

        num_layers, batch_size, _, dim = layer_features.shape
        assert num_layers == self.num_layers
        assert dim == self.feature_dim

        w_k = F.softmax(self.layer_weights_key, dim=0).view(num_layers, 1, 1, 1)
        w_v = F.softmax(self.layer_weights_value, dim=0).view(num_layers, 1, 1, 1)
        k_feat = (layer_features * w_k).sum(dim=0)
        v_feat = (layer_features * w_v).sum(dim=0)

        k_proj = self.dropout(self.key_projection(k_feat))
        v_proj = self.dropout(self.value_projection(v_feat))
        attn = F.softmax(self.attention_projection(k_proj), dim=1)
        attn = attn.transpose(1, 2).unsqueeze(-1)
        v_proj = v_proj.unsqueeze(1)
        pooled = (attn * v_proj).sum(dim=2).view(batch_size, self.num_heads * self.compression_dim)
        return self.embedding_projection(pooled)


class LangIDModelWithMHFAHead(nn.Module):
    def __init__(
        self,
        model_dir: str,
        num_languages: int,
        embedding_dim: int = 256,
        num_prompt_tokens: int = 15,
        num_wavelet_tokens: int = 8,
        prompt_dropout: float = 0.0,
        num_heads: int = 8,
        compression_dim: int = 128,
        head_dropout: float = 0.0,
        peft_mode: str = "deep_prompt",
        use_wavelet: bool = True,
        num_prefix_tokens: int | None = None,
    ):
        super().__init__()
        self.wpt_w2vbert = WPTW2VBERTMultiLayer(
            model_dir=model_dir,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens=num_wavelet_tokens,
            prompt_dim=1024,
            dropout=prompt_dropout,
            peft_mode=peft_mode,
            use_wavelet=use_wavelet,
            num_prefix_tokens=num_prefix_tokens,
        )
        self.mhfa_head = MHFAHead(
            feature_dim=1024,
            num_layers=self.wpt_w2vbert.config.num_hidden_layers,
            num_heads=num_heads,
            compression_dim=compression_dim,
            embedding_dim=embedding_dim,
            dropout=head_dropout,
        )
        self.classifier = nn.Linear(embedding_dim, num_languages)

    def forward(self, audio_data: torch.Tensor):
        layer_features = self.wpt_w2vbert(audio_data)
        embeddings = self.mhfa_head(layer_features)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def train(self, mode=True):
        super().train(mode)
        self.wpt_w2vbert.model.eval()
        return self
