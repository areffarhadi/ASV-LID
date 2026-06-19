#!/usr/bin/env python3
"""
End-to-end inference: wav files -> language labels + pair similarity scores.

Uses unified joint WPT backbone + dual-path ASV/LID fusion heads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from audio_utils import cosine_score, load_lang_map, load_waveform
from joint_wpt_mhfa import JointWPTMHFAInference
from load_models import load_asv_model, load_lid_branch_only


@dataclass
class FileResult:
    path: str
    pure_lid_language: str
    pure_lid_confidence: float
    fused_lid_language: str
    fused_lid_confidence: float


@dataclass
class PairResult:
    file_a: str
    file_b: str
    asv_only_score: float
    lid_only_score: float
    asv_fused_score: float
    lid_fused_score: float
    fusion_recipe: str


class EndToEndPipeline:
    def __init__(
        self,
        asv_code: str,
        lid_code: str,
        fusion_asv_code: str,
        fusion_lid_code: str,
        asv_ckpt: str,
        lid_ckpt: str,
        fusion_asv_ckpt: str,
        fusion_lid_ckpt: str,
        manifest_path: str,
        ssl_model: str,
        device: torch.device,
    ):
        self.device = device
        self.lang_to_id, self.id_to_lang = load_lang_map(manifest_path)

        self.asv_model = load_asv_model(asv_code, asv_ckpt, ssl_model, device)
        self.lid_branch = load_lid_branch_only(
            lid_code,
            lid_ckpt,
            ssl_model,
            device,
            num_layers=self.asv_model.wpt_w2vbert.config.num_hidden_layers,
        )
        self.joint = JointWPTMHFAInference(self.asv_model, self.lid_branch, verify_backbone=False)

        if fusion_asv_code not in sys.path:
            sys.path.insert(0, fusion_asv_code)
        from train_fusion_dual_path_v12 import DualPathModel

        asv_fusion_ckpt = torch.load(fusion_asv_ckpt, map_location=device, weights_only=False)
        self.asv_fusion_params = asv_fusion_ckpt.get("results", {}).get(
            "sub+add-lid_p", (0.5, 0.5, 0.1)
        )
        self.asv_fusion = DualPathModel(
            embed_dim=256,
            hidden_dim=512,
            subspace_dim=64,
            num_heads=4,
            num_speakers=asv_fusion_ckpt.get("num_speakers", 3666),
            num_languages=asv_fusion_ckpt.get("num_languages", 40),
        ).to(device)
        self.asv_fusion.load_state_dict(asv_fusion_ckpt["model_state_dict"])
        self.asv_fusion.eval()

        if fusion_lid_code not in sys.path:
            sys.path.insert(0, fusion_lid_code)
        from train_lid_fusion_dual_path_manifest import DualPathLanguageExtractor

        lid_fusion_sd = torch.load(fusion_lid_ckpt, map_location=device, weights_only=False)
        num_classes = lid_fusion_sd["sub_classifier_weight"].shape[0]
        self.lid_fusion = DualPathLanguageExtractor(
            embed_dim=256,
            hidden_dim=512,
            subspace_dim=64,
            num_heads=4,
            num_classes=num_classes,
        ).to(device)
        self.lid_fusion.load_state_dict(lid_fusion_sd)
        self.lid_fusion.eval()

    @torch.no_grad()
    def predict_pure_lid(self, lid_emb: torch.Tensor) -> Tuple[str, float]:
        weight = F.normalize(self.lid_branch.arcface.weight, p=2, dim=1)
        emb = F.normalize(lid_emb, p=2, dim=1)
        logits = F.linear(emb, weight) * self.lid_branch.arcface.scale
        prob = torch.softmax(logits, dim=1)
        pred_id = int(torch.argmax(logits, dim=1).item())
        conf = float(prob[0, pred_id].item())
        return self.id_to_lang.get(pred_id, f"id_{pred_id}"), conf

    @torch.no_grad()
    def predict_fused_lid(self, asv_emb: torch.Tensor, lid_emb: torch.Tensor) -> Tuple[str, float]:
        sub_logits, add_logits, _, _ = self.lid_fusion(asv_emb, lid_emb)
        fused_logits = 0.5 * (sub_logits + add_logits)
        prob = torch.softmax(fused_logits, dim=1)
        pred_id = int(torch.argmax(fused_logits, dim=1).item())
        conf = float(prob[0, pred_id].item())
        return self.id_to_lang.get(pred_id, f"id_{pred_id}"), conf

    @torch.no_grad()
    def fused_lid_embedding(self, asv_emb: torch.Tensor, lid_emb: torch.Tensor) -> torch.Tensor:
        _, _, sub_emb, add_emb = self.lid_fusion(asv_emb, lid_emb)
        sub_n = F.normalize(sub_emb, p=2, dim=1)
        add_n = F.normalize(add_emb, p=2, dim=1)
        return F.normalize(0.5 * (sub_n + add_n), p=2, dim=1)

    @torch.no_grad()
    def process_file(self, wav_path: str) -> Tuple[FileResult, torch.Tensor, torch.Tensor]:
        wav = load_waveform(wav_path).to(self.device).unsqueeze(0)
        asv_emb, lid_emb = self.joint.extract_both(wav, normalize=True)
        pure_lang, pure_conf = self.predict_pure_lid(lid_emb)
        fused_lang, fused_conf = self.predict_fused_lid(asv_emb, lid_emb)
        return (
            FileResult(
                path=wav_path,
                pure_lid_language=pure_lang,
                pure_lid_confidence=pure_conf,
                fused_lid_language=fused_lang,
                fused_lid_confidence=fused_conf,
            ),
            asv_emb,
            lid_emb,
        )

    @torch.no_grad()
    def score_pair(
        self,
        asv_a: torch.Tensor,
        lid_a: torch.Tensor,
        asv_b: torch.Tensor,
        lid_b: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        asv_only = cosine_score(asv_a, asv_b)
        lid_only = cosine_score(lid_a, lid_b)

        sub_a = self.asv_fusion.extract_sub_embedding(asv_a, lid_a)
        add_a = self.asv_fusion.extract_add_embedding(asv_a, lid_a)
        sub_b = self.asv_fusion.extract_sub_embedding(asv_b, lid_b)
        add_b = self.asv_fusion.extract_add_embedding(asv_b, lid_b)

        sub_s = cosine_score(sub_a, sub_b)
        add_s = cosine_score(add_a, add_b)
        alpha, beta, delta = self.asv_fusion_params
        asv_fused = alpha * sub_s + beta * add_s - delta * lid_only

        lid_fused_a = self.fused_lid_embedding(asv_a, lid_a)
        lid_fused_b = self.fused_lid_embedding(asv_b, lid_b)
        lid_fused = cosine_score(lid_fused_a, lid_fused_b)

        return asv_only, lid_only, asv_fused, lid_fused

    def run(self, wav_paths: List[str]) -> dict:
        if len(wav_paths) < 2:
            raise ValueError("Need at least two wav files for pair scoring.")

        file_results = []
        embeddings = []
        for path in wav_paths:
            fr, asv_e, lid_e = self.process_file(path)
            file_results.append(fr)
            embeddings.append((asv_e, lid_e))

        pairs: List[PairResult] = []
        for i in range(len(wav_paths)):
            for j in range(i + 1, len(wav_paths)):
                asv_i, lid_i = embeddings[i]
                asv_j, lid_j = embeddings[j]
                a, lid_o, b, c = self.score_pair(asv_i, lid_i, asv_j, lid_j)
                alpha, beta, delta = self.asv_fusion_params
                pairs.append(
                    PairResult(
                        file_a=wav_paths[i],
                        file_b=wav_paths[j],
                        asv_only_score=a,
                        lid_only_score=lid_o,
                        asv_fused_score=b,
                        lid_fused_score=c,
                        fusion_recipe=f"{alpha:.2f}*sub + {beta:.2f}*add - {delta:.2f}*lid_cos",
                    )
                )

        return {
            "files": [fr.__dict__ for fr in file_results],
            "pairs": [pr.__dict__ for pr in pairs],
            "asv_fusion_recipe": pairs[0].fusion_recipe if pairs else "",
        }


def print_report(result: dict) -> None:
    print("\n" + "=" * 72)
    print("ASV-LID END-TO-END INFERENCE")
    print("=" * 72)

    print("\nPer-file language ID:")
    print(f"{'file':<40} {'LID-only':<12} {'conf':>6}  {'LID-fused':<12} {'conf':>6}")
    print("-" * 72)
    for item in result["files"]:
        name = os.path.basename(item["path"])
        print(
            f"{name:<40} {item['pure_lid_language']:<12} {item['pure_lid_confidence']:>6.3f}  "
            f"{item['fused_lid_language']:<12} {item['fused_lid_confidence']:>6.3f}"
        )

    print("\nPair similarity scores:")
    print(f"ASV fusion recipe: {result.get('asv_fusion_recipe', '')}")
    print(
        f"{'pair':<44} {'ASV-only':>9} {'LID-only':>9} {'ASV-fused':>9} {'LID-fused':>9}"
    )
    print("-" * 84)
    for item in result["pairs"]:
        label = f"{os.path.basename(item['file_a'])} <-> {os.path.basename(item['file_b'])}"
        print(
            f"{label:<44} {item['asv_only_score']:>9.4f} {item['lid_only_score']:>9.4f} "
            f"{item['asv_fused_score']:>9.4f} {item['lid_fused_score']:>9.4f}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end ASV-LID inference from wav files")
    parser.add_argument("--wav_dir", required=True, help="Folder with .wav files (>=2)")
    parser.add_argument("--output_json", default="", help="Optional JSON output path")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    repo_root = os.environ.get(
        "REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    )

    asv_ckpt = os.environ.get("CKPT_ASV", os.path.join(repo_root, "checkpoints/asv_wpt.pt"))
    lid_ckpt = os.environ.get("CKPT_LID", os.path.join(repo_root, "checkpoints/lid_wpt.pt"))
    fusion_asv_ckpt = os.environ.get("CKPT_FUSION_ASV", os.path.join(repo_root, "checkpoints/fusion_asv.pt"))
    fusion_lid_ckpt = os.environ.get("CKPT_FUSION_LID", os.path.join(repo_root, "checkpoints/fusion_lid.pth"))
    manifest = os.environ.get(
        "LID_MANIFEST", os.path.join(repo_root, "data/manifests/training_manifest.txt")
    )
    ssl_model = os.environ.get("W2VBERT_MODEL", "facebook/w2v-bert-2.0")
    asv_code = os.environ.get("ASV_CODE", os.path.join(repo_root, "src/wpt"))
    lid_code = os.environ.get("LID_CODE", os.path.join(repo_root, "src/wpt"))
    fusion_asv_code = os.environ.get("FUSION_ASV_CODE", os.path.join(repo_root, "src/fusion_asv"))
    fusion_lid_code = os.environ.get("FUSION_LID_CODE", os.path.join(repo_root, "src/fusion_lid"))

    wav_paths = sorted(
        os.path.join(args.wav_dir, f)
        for f in os.listdir(args.wav_dir)
        if f.lower().endswith(".wav")
    )
    if len(wav_paths) < 2:
        raise SystemExit(f"Need >=2 wav files in {args.wav_dir}, found {len(wav_paths)}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Processing {len(wav_paths)} wav files from {args.wav_dir}")

    pipe = EndToEndPipeline(
        asv_code=asv_code,
        lid_code=lid_code,
        fusion_asv_code=fusion_asv_code,
        fusion_lid_code=fusion_lid_code,
        asv_ckpt=asv_ckpt,
        lid_ckpt=lid_ckpt,
        fusion_asv_ckpt=fusion_asv_ckpt,
        fusion_lid_ckpt=fusion_lid_ckpt,
        manifest_path=manifest,
        ssl_model=ssl_model,
        device=device,
    )

    result = pipe.run(wav_paths)
    print_report(result)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
