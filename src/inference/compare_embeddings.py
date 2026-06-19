#!/usr/bin/env python3
"""
Compare joint inference embeddings against cached stage1_asv / stage2_lid .npy files.

Exit code 0 only if all checks pass.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch

from audio_utils import load_waveform
from joint_wpt_mhfa import JointWPTMHFAInference
from load_models import load_asv_model, load_lid_branch_only


def wav_to_npy_path(wav_path: str, emb_root: str) -> str:
    parts = wav_path.replace("\\", "/").split("/")
    try:
        idx = parts.index("TidyVoiceX_Dev")
        rel = "/".join(parts[idx + 1 :])
    except ValueError:
        rel = "/".join(parts[-3:])
    return os.path.join(emb_root, os.path.splitext(rel)[0] + ".npy")


def collect_wavs(wav_dir: str, num_samples: int) -> list[str]:
    paths = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No wav files under {wav_dir}")
    return paths[:num_samples]


def main() -> int:
    parser = argparse.ArgumentParser(description="Joint vs cached embedding parity test")
    parser.add_argument("--asv_ckpt", required=True)
    parser.add_argument("--lid_ckpt", required=True)
    parser.add_argument("--asv_code", required=True)
    parser.add_argument("--lid_code", required=True)
    parser.add_argument("--emb_asv_root", required=True)
    parser.add_argument("--emb_lid_root", required=True)
    parser.add_argument("--ssl_model", default="facebook/w2v-bert-2.0")
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    asv_model = load_asv_model(args.asv_code, args.asv_ckpt, args.ssl_model, device)
    lid_branch = load_lid_branch_only(
        args.lid_code,
        args.lid_ckpt,
        args.ssl_model,
        device,
        num_layers=asv_model.wpt_w2vbert.config.num_hidden_layers,
    )
    joint = JointWPTMHFAInference(asv_model, lid_branch, verify_backbone=False)

    wavs = collect_wavs(args.wav_dir, args.num_samples)
    print(f"Testing {len(wavs)} wav files")

    all_ok = True
    for path in wavs:
        wav_batch = load_waveform(path).to(device).unsqueeze(0)
        name = os.path.basename(path)

        npy_asv = wav_to_npy_path(path, args.emb_asv_root)
        npy_lid = wav_to_npy_path(path, args.emb_lid_root)
        if not os.path.isfile(npy_asv) or not os.path.isfile(npy_lid):
            print(f"  [SKIP] {name}: missing cached npy")
            continue

        cached_asv = torch.from_numpy(np.load(npy_asv).astype(np.float32)).to(device)
        cached_lid = torch.from_numpy(np.load(npy_lid).astype(np.float32)).to(device)
        if cached_asv.dim() == 1:
            cached_asv = cached_asv.unsqueeze(0)
        if cached_lid.dim() == 1:
            cached_lid = cached_lid.unsqueeze(0)

        with torch.no_grad():
            live_asv = asv_model.extract_embedding(wav_batch, normalize=True)
            joint_asv, joint_lid = joint.extract_both(wav_batch, normalize=True)

        ok_joint_asv = torch.allclose(joint_asv, cached_asv, rtol=args.rtol, atol=args.atol)
        ok_joint_lid = torch.allclose(joint_lid, cached_lid, rtol=args.rtol, atol=args.atol)
        ok_live_asv = torch.allclose(live_asv, cached_asv, rtol=args.rtol, atol=args.atol)
        ok = ok_joint_asv and ok_joint_lid and ok_live_asv
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {name}: joint_asv={ok_joint_asv} joint_lid={ok_joint_lid} live_asv={ok_live_asv}"
        )
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll parity checks PASSED.")
        return 0
    print("\nParity checks FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
