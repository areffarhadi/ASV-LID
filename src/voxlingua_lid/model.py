"""WPT + W2V-BERT-2.0 + MHFA language ID model (VoxLingua checkpoint)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2BertModel


class WaveletBlock(nn.Module):
    def __init__(self, wave="haar", J=1, input_dim=1024, output_dim=1024):
        super().__init__()
        from pytorch_wavelets import DWTForward

        self.dwt = DWTForward(J=J, wave=wave)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        bsz, _, dim = x.shape
        assert dim == self.input_dim
        x = x.unsqueeze(dim=1)
        ll, band = self.dwt(x)
        bands = band[0]
        ll = ll.unsqueeze(dim=2)
        return torch.cat((ll, bands), dim=2).view(bsz, -1, dim)


class WPTW2VBERTMultiLayer(nn.Module):
    def __init__(
        self,
        model_dir: str,
        num_prompt_tokens: int = 15,
        num_wavelet_tokens: int = 8,
        prompt_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.num_wavelet_tokens = num_wavelet_tokens
        self.prompt_dim = prompt_dim

        self.config = AutoConfig.from_pretrained(model_dir)
        self.processor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2BertModel.from_pretrained(model_dir)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        num_layers = self.config.num_hidden_layers
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers, num_prompt_tokens, prompt_dim)
        )
        self.wavelet_prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers, num_wavelet_tokens, prompt_dim)
        )
        self.wavelet_block = WaveletBlock(
            wave="haar", J=1, input_dim=prompt_dim, output_dim=prompt_dim
        )

        val = math.sqrt(6.0 / float(2 * prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        nn.init.uniform_(self.wavelet_prompt_embeddings.data, -val, val)
        self.prompt_dropout = nn.Dropout(p=dropout)

    def forward(self, audio_data: torch.Tensor):
        original_device = audio_data.device
        audio_cpu = audio_data.detach().cpu().numpy()
        processed = self.processor(audio_cpu, sampling_rate=16000, return_tensors="pt")

        if "input_features" in processed:
            feat = processed["input_features"].to(original_device)
        elif hasattr(processed, "input_features"):
            feat = processed.input_features.to(original_device)
        elif "input_values" in processed:
            feat = processed["input_values"].to(original_device)
        else:
            feat = list(processed.values())[0].to(original_device)

        if feat.dim() == 3:
            feat = feat.squeeze(0)

        batch_size = feat.size(0)
        with torch.no_grad():
            hidden_state, _ = self.model.feature_projection(feat)
            hidden_state = self.model.encoder.dropout(hidden_state)

        total_prompt_tokens = self.num_wavelet_tokens + self.num_prompt_tokens
        layer_features = []

        for layer_idx in range(self.config.num_hidden_layers):
            prompt = self.prompt_embeddings[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
            prompt = self.prompt_dropout(prompt)

            wavelet_prompt = self.wavelet_prompt_embeddings[layer_idx].unsqueeze(0).expand(
                batch_size, -1, -1
            )
            wavelet_prompt = self.wavelet_block(wavelet_prompt)
            wavelet_prompt = self.prompt_dropout(wavelet_prompt)

            if layer_idx == 0:
                hidden_state = torch.cat([wavelet_prompt, prompt, hidden_state], dim=1)
            else:
                audio_features = hidden_state[:, total_prompt_tokens:, :]
                hidden_state = torch.cat([wavelet_prompt, prompt, audio_features], dim=1)

            hidden_state = self.model.encoder.layers[layer_idx](hidden_state)[0]
            layer_features.append(hidden_state[:, total_prompt_tokens:, :].clone())

        return layer_features


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
    ):
        super().__init__()
        self.wpt_w2vbert = WPTW2VBERTMultiLayer(
            model_dir=model_dir,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens=num_wavelet_tokens,
            prompt_dim=1024,
            dropout=prompt_dropout,
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
