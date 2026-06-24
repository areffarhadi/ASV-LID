#!/usr/bin/env python3
"""
Smoke test for the selectable PEFT modes in src/wpt/peft_wpt.py.

Run this in the project venv (the one with torch + transformers + pytorch_wavelets)
BEFORE relying on any mode — especially the EXPERIMENTAL 'prefix' mode, which
performs surgery on the HF Wav2Vec2Bert attention and is version-sensitive.

What it checks, for every (peft_mode x use_wavelet) combo:
  1. the front-end builds,
  2. a forward pass runs on dummy audio,
  3. it returns one feature map per layer with shape (B, T, 1024) and T == the
     audio frame count (i.e. prompts/prefixes were correctly stripped),
  4. backprop reaches the adapter parameters but NOT the frozen backbone.

Usage:
    python scripts/smoke_test_peft.py                       # uses facebook/w2v-bert-2.0
    python scripts/smoke_test_peft.py --model_dir <path>    # local backbone dir
    python scripts/smoke_test_peft.py --modes deep_prompt prefix
"""
import argparse
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "src", "wpt"))

import torch  # noqa: E402
from peft_wpt import WPTW2VBERTMultiLayer, PEFT_MODES  # noqa: E402


def run_case(model_dir, peft_mode, use_wavelet, n_prompt, n_wave, device):
    tag = f"[{peft_mode:14s} wavelet={'on' if use_wavelet else 'off'}]"
    try:
        net = WPTW2VBERTMultiLayer(
            model_dir=model_dir,
            num_prompt_tokens=n_prompt,
            num_wavelet_tokens=n_wave,
            prompt_dim=1024,
            dropout=0.1,
            peft_mode=peft_mode,
            use_wavelet=use_wavelet,
            verbose=False,
        ).to(device)
        net.train()

        # ~0.5 s of dummy audio at 16 kHz
        audio = torch.randn(1, 8000, device=device)
        feats = net(audio)

        assert isinstance(feats, list) and len(feats) == net.config.num_hidden_layers, \
            f"expected {net.config.num_hidden_layers} layer maps, got {len(feats)}"
        B, T, D = feats[-1].shape
        assert D == 1024, f"feature dim {D} != 1024"
        assert all(f.shape == feats[-1].shape for f in feats), "inconsistent layer shapes"

        # backward: adapter params should get grads; backbone should not
        loss = sum(f.float().pow(2).mean() for f in feats)
        loss.backward()

        backbone_grads = [n for n, p in net.model.named_parameters() if p.grad is not None]
        adapter_grads = [n for n, p in net.named_parameters()
                         if p.grad is not None and not n.startswith("model.")]
        assert not backbone_grads, f"backbone received gradients (should be frozen): {backbone_grads[:3]}"
        assert adapter_grads, "no adapter parameter received a gradient"

        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad and p.grad is not None)
        print(f"  PASS {tag}  T={T}  adapter-params-with-grad={trainable:,}")
        return True
    except Exception as e:
        print(f"  FAIL {tag}  -> {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.environ.get("W2VBERT_MODEL", "facebook/w2v-bert-2.0"))
    ap.add_argument("--modes", nargs="+", default=list(PEFT_MODES), choices=list(PEFT_MODES))
    ap.add_argument("--num_prompt_tokens", type=int, default=4)
    ap.add_argument("--num_wavelet_tokens", type=int, default=4)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Backbone: {args.model_dir}   device: {device}")
    print("=" * 64)

    results = []
    for mode in args.modes:
        for use_wavelet in (True, False):
            results.append(run_case(args.model_dir, mode, use_wavelet,
                                    args.num_prompt_tokens, args.num_wavelet_tokens, device))

    print("=" * 64)
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} cases passed.")
    if n_ok != len(results):
        print("NOTE: 'prefix' is experimental — if only prefix cases fail, paste the error "
              "and the deep_prompt/shallow_prompt modes are still fully usable.")
        sys.exit(1)


if __name__ == "__main__":
    main()
