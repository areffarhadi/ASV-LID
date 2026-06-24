"""
Parameter-efficient adaptation backbone for W2V-BERT-2.0 — selectable PEFT modes.
=================================================================================

Single source of truth for the WPT front-end. The user chooses *how* the small
set of learnable vectors is injected into the frozen backbone, via two
orthogonal axes:

  --peft_mode :  how the learnable vectors enter the network
      'deep_prompt'    (DEFAULT)  prepend learnable prompt tokens to the INPUT
                                  sequence of EVERY layer; drop them after each
                                  layer.  == VPT-Deep.  *** This is exactly the
                                  mechanism used for the paper's results. ***
      'shallow_prompt'            prepend learnable prompt tokens only ONCE, at
                                  the first layer (classic Lester-2021 prompt
                                  tuning); they flow through all layers and are
                                  stripped at the end.
      'prefix'         (EXPERIMENTAL) learnable vectors are injected ONLY as
                                  extra Keys/Values inside self-attention — no
                                  query, never through the FFN.  == true
                                  prefix-tuning (Li & Liang 2021).  Performs
                                  surgery on the HF attention module; MUST be
                                  smoke-tested on your transformers version
                                  first (scripts/smoke_test_peft.py).

  --use_wavelet :  whether the learnable vectors are Haar-wavelet structured
      True  (DEFAULT)  pass them through a fixed Haar DWT (coarse+fine) before
                       injection  -> the "Wavelet" in WPT.
      False            use the raw learnable vectors (ablation).

Backward compatibility
-----------------------
With the defaults (peft_mode='deep_prompt', use_wavelet=True) this class creates
exactly the same sub-modules and parameter names as the original
WPTW2VBERTMultiLayer ('prompt_embeddings', 'wavelet_prompt_embeddings',
'wavelet_block', 'model'), so released checkpoints load unchanged with
strict=True. Extra parameters for the new modes are created ONLY when those
modes are selected, so they never appear in a default checkpoint.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2BertModel

PEFT_MODES = ("deep_prompt", "shallow_prompt", "prefix")


# ---------------------------------------------------------------------------
# Wavelet block (unchanged from the original implementation)
# ---------------------------------------------------------------------------
class WaveletBlock(nn.Module):
    """Haar DWT reparameterization of a (B, T, D) block of learnable vectors."""

    def __init__(self, wave="haar", J=1, input_dim=1024, output_dim=1024):
        super().__init__()
        from pytorch_wavelets import DWTForward

        self.dwt = DWTForward(J=J, wave=wave)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.input_dim
        x = x.unsqueeze(dim=1)
        LL, band = self.dwt(x)
        bands = band[0]
        LL = LL.unsqueeze(dim=2)
        features = torch.cat((LL, bands), dim=2).view(B, -1, D)
        return features


# ---------------------------------------------------------------------------
# EXPERIMENTAL: prefix injector for true prefix-tuning
# ---------------------------------------------------------------------------
class PrefixInjector(nn.Module):
    """
    EXPERIMENTAL — true prefix-tuning for a single Wav2Vec2Bert self-attention.

    Holds the learnable Key/Value prefixes for ONE layer and replaces that
    layer's attention forward so the prefixes are prepended to K and V only (no
    query is computed for them; they never pass through the FFN/conv -> output
    length == audio length). Prefix tokens get ZERO positional bias (treated as
    position-agnostic memory) — the standard simplification for prefix-tuning on
    relative/rotary-position backbones.

    *** Guarded by `_assert_supported()`; reimplements the scaled-dot-product of
    the HF attention. NOT exercised by released checkpoints. Verify with
    scripts/smoke_test_peft.py on your transformers version before use. ***
    """

    def __init__(self, attn: nn.Module, num_prefix: int, prompt_dim: int,
                 wavelet_block: nn.Module | None = None):
        super().__init__()
        object.__setattr__(self, "attn", attn)  # keep a reference WITHOUT registering as a child
        self.num_prefix = num_prefix
        self.prompt_dim = prompt_dim
        self.num_heads = getattr(attn, "num_heads", None)
        self.head_size = getattr(attn, "head_size", None)
        self.position_type = getattr(attn, "position_embeddings_type", None)
        self._assert_supported()
        self.wavelet_block = wavelet_block  # shared; may be None
        # learnable prefixes, stored flat (num_prefix, hidden) so the wavelet block applies cleanly
        self.prefix_key = nn.Parameter(torch.zeros(num_prefix, prompt_dim))
        self.prefix_value = nn.Parameter(torch.zeros(num_prefix, prompt_dim))
        val = math.sqrt(6.0 / float(2 * prompt_dim))
        nn.init.uniform_(self.prefix_key.data, -val, val)
        nn.init.uniform_(self.prefix_value.data, -val, val)

    def _assert_supported(self):
        a = self.attn
        missing = [n for n in ("linear_q", "linear_k", "linear_v", "linear_out") if not hasattr(a, n)]
        if missing or self.num_heads is None or self.head_size is None:
            raise NotImplementedError(
                "PrefixInjector: unexpected Wav2Vec2BertSelfAttention structure "
                f"(missing {missing or 'num_heads/head_size'}). Adapt the 'prefix' mode "
                "to your transformers version, or use --peft_mode deep_prompt (default). "
                "Run scripts/smoke_test_peft.py to diagnose."
            )
        if self.position_type not in (None, "rotary", "relative", "relative_key"):
            raise NotImplementedError(
                f"PrefixInjector: position_embeddings_type='{self.position_type}' not supported."
            )

    def _prefix_kv(self, B, device, dtype):
        pk, pv = self.prefix_key, self.prefix_value
        if self.wavelet_block is not None:
            pk = self.wavelet_block(pk.unsqueeze(0)).squeeze(0)
            pv = self.wavelet_block(pv.unsqueeze(0)).squeeze(0)
        H, Dh = self.num_heads, self.head_size
        # (P, hidden) -> (B, H, P, Dh)
        pk = pk.view(self.num_prefix, H, Dh).unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2)
        pv = pv.view(self.num_prefix, H, Dh).unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2)
        return pk.to(device=device, dtype=dtype), pv.to(device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, relative_position_embeddings=None,
                output_attentions=False, **kwargs):
        a = self.attn
        B, T, _ = hidden_states.shape
        H, Dh = self.num_heads, self.head_size

        def shape(x):  # (B, T, H*Dh) -> (B, H, T, Dh)
            return x.view(B, -1, H, Dh).transpose(1, 2)

        q = shape(a.linear_q(hidden_states))
        k = shape(a.linear_k(hidden_states))
        v = shape(a.linear_v(hidden_states))

        if self.position_type == "rotary" and relative_position_embeddings is not None:
            q, k = a._apply_rotary_embedding(q, k, relative_position_embeddings)

        if self.position_type == "relative" and relative_position_embeddings is not None:
            scores_real = a._apply_relative_embeddings(
                query=q, key=k, relative_position_embeddings=relative_position_embeddings)
        else:
            scores_real = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        pk, pv = self._prefix_kv(B, q.device, q.dtype)
        scores_prefix = torch.matmul(q, pk.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,T,P)

        if attention_mask is not None:
            scores_real = scores_real + attention_mask

        scores = torch.cat([scores_prefix, scores_real], dim=-1)               # (B,H,T,P+T)
        probs = torch.softmax(scores, dim=-1)
        if hasattr(a, "dropout"):
            probs = a.dropout(probs)

        ctx = torch.matmul(probs[..., :self.num_prefix], pv) + \
            torch.matmul(probs[..., self.num_prefix:], v)                      # (B,H,T,Dh)
        ctx = ctx.transpose(1, 2).reshape(B, T, H * Dh)
        out = a.linear_out(ctx)
        return (out, probs) if output_attentions else (out, None)


# ---------------------------------------------------------------------------
# Configurable WPT front-end
# ---------------------------------------------------------------------------
class WPTW2VBERTMultiLayer(nn.Module):
    """W2V-BERT-2.0 (frozen) + selectable PEFT injection; extracts ALL layers."""

    def __init__(self, model_dir, num_prompt_tokens=6, num_wavelet_tokens=4,
                 prompt_dim=1024, dropout=0.1,
                 peft_mode="deep_prompt", use_wavelet=True, num_prefix_tokens=None,
                 verbose=True):
        super().__init__()
        if peft_mode not in PEFT_MODES:
            raise ValueError(f"peft_mode must be one of {PEFT_MODES}, got '{peft_mode}'")
        self.peft_mode = peft_mode
        self.use_wavelet = bool(use_wavelet)
        self.num_prompt_tokens = num_prompt_tokens
        self.num_wavelet_tokens = num_wavelet_tokens if self.use_wavelet else 0
        self.prompt_dim = prompt_dim

        self.config = AutoConfig.from_pretrained(model_dir)
        self.processor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2BertModel.from_pretrained(model_dir)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        num_layers = self.config.num_hidden_layers
        val = math.sqrt(6.0 / float(2 * prompt_dim))

        if peft_mode in ("deep_prompt", "shallow_prompt"):
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(num_layers, num_prompt_tokens, prompt_dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.use_wavelet:
                self.wavelet_prompt_embeddings = nn.Parameter(
                    torch.zeros(num_layers, num_wavelet_tokens, prompt_dim))
                nn.init.uniform_(self.wavelet_prompt_embeddings.data, -val, val)
                self.wavelet_block = WaveletBlock(wave="haar", J=1,
                                                  input_dim=prompt_dim, output_dim=prompt_dim)
            self.prompt_dropout = nn.Dropout(p=dropout)

        elif peft_mode == "prefix":
            self.num_prefix_tokens = num_prefix_tokens or (num_prompt_tokens + self.num_wavelet_tokens)
            # The Haar DWT only preserves the token count when it is even; round up if needed.
            if self.use_wavelet and (self.num_prefix_tokens % 2 != 0):
                self.num_prefix_tokens += 1
                print(f"    [prefix+wavelet] rounded num_prefix_tokens up to "
                      f"{self.num_prefix_tokens} (Haar needs an even count).")
            shared_wavelet = (WaveletBlock(wave="haar", J=1, input_dim=prompt_dim,
                                           output_dim=prompt_dim) if self.use_wavelet else None)
            if shared_wavelet is not None:
                self.wavelet_block = shared_wavelet
            self.prompt_dropout = nn.Dropout(p=dropout)
            self.injectors = nn.ModuleList([
                PrefixInjector(self.model.encoder.layers[l].self_attn,
                               self.num_prefix_tokens, prompt_dim, shared_wavelet)
                for l in range(num_layers)
            ])
            for l in range(num_layers):  # route attention through the injector
                self.model.encoder.layers[l].self_attn.forward = self.injectors[l].forward

        if verbose:
            print(f"  WPT front-end:  peft_mode={peft_mode}  use_wavelet={self.use_wavelet}")
            if peft_mode in ("deep_prompt", "shallow_prompt"):
                print(f"    regular prompts/layer: {num_prompt_tokens} | "
                      f"wavelet prompts/layer: {self.num_wavelet_tokens} | layers: {num_layers}")
            else:
                print(f"    prefix tokens/layer: {self.num_prefix_tokens} "
                      f"(EXPERIMENTAL — verify with scripts/smoke_test_peft.py)")

    # -- helpers ---------------------------------------------------------
    def _features_from_audio(self, audio_data):
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
        if feat.dim() > 3:
            feat = feat.squeeze(0)
        elif feat.dim() < 3:
            feat = feat.unsqueeze(0)
        return feat

    def _prompt(self, layer_idx, batch_size):
        """Tokens to prepend at this layer, ordered [wavelet, regular] (matches the released code)."""
        prompt = self.prompt_embeddings[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
        prompt = self.prompt_dropout(prompt)
        if self.use_wavelet:
            w = self.wavelet_prompt_embeddings[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
            w = self.wavelet_block(w)
            w = self.prompt_dropout(w)
            return torch.cat([w, prompt], dim=1)
        return prompt

    # -- forward ---------------------------------------------------------
    def forward(self, audio_data):
        feat = self._features_from_audio(audio_data)
        batch_size = feat.size(0)
        with torch.no_grad():
            hidden_state, _ = self.model.feature_projection(feat)
            hidden_state = self.model.encoder.dropout(hidden_state)

        num_layers = self.config.num_hidden_layers
        layer_features = []

        if self.peft_mode == "deep_prompt":
            n_tok = self.num_prompt_tokens + self.num_wavelet_tokens
            for l in range(num_layers):
                prompts = self._prompt(l, batch_size)
                if l == 0:
                    hidden_state = torch.cat([prompts, hidden_state], dim=1)
                else:
                    audio = hidden_state[:, n_tok:, :]
                    hidden_state = torch.cat([prompts, audio], dim=1)
                hidden_state = self.model.encoder.layers[l](hidden_state)[0]
                layer_features.append(hidden_state[:, n_tok:, :].clone())

        elif self.peft_mode == "shallow_prompt":
            n_tok = self.num_prompt_tokens + self.num_wavelet_tokens
            hidden_state = torch.cat([self._prompt(0, batch_size), hidden_state], dim=1)
            for l in range(num_layers):
                hidden_state = self.model.encoder.layers[l](hidden_state)[0]
                layer_features.append(hidden_state[:, n_tok:, :].clone())

        elif self.peft_mode == "prefix":
            # prefixes are injected inside attention; the sequence stays audio-only
            for l in range(num_layers):
                hidden_state = self.model.encoder.layers[l](hidden_state)[0]
                layer_features.append(hidden_state.clone())

        return layer_features

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()  # backbone stays frozen / in eval
        return self
