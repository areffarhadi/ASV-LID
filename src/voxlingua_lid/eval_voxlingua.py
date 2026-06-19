#!/usr/bin/env python3
"""Evaluate VoxLingua LID checkpoint on dev_vox."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VoxLinguaLangIDDataset
from model import LangIDModelWithMHFAHead


def load_checkpoint(ckpt_dir: str, device: torch.device):
    args_path = os.path.join(ckpt_dir, "args.json")
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"Missing {args_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"model.pt not found in {ckpt_dir}")

    with open(args_path, "r") as f:
        train_args = json.load(f)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_map = checkpoint["label_map"]
    num_languages = len(label_map)

    model = LangIDModelWithMHFAHead(
        model_dir=train_args.get("xlsr", "facebook/w2v-bert-2.0"),
        num_languages=num_languages,
        embedding_dim=train_args["embedding_dim"],
        num_prompt_tokens=train_args["num_prompt_tokens"],
        num_wavelet_tokens=train_args["num_wavelet_tokens"],
        prompt_dropout=0.0,
        num_heads=train_args["num_heads"],
        compression_dim=train_args["compression_dim"],
        head_dropout=0.0,
    )

    state_dict = checkpoint["model_state_dict"]
    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    # Checkpoint may include frozen W2V-BERT weights from an older transformers layout.
    # Backbone is loaded from HuggingFace; keep only trainable WPT/MHFA/classifier weights.
    trainable = {
        k: v
        for k, v in cleaned.items()
        if not k.startswith("wpt_w2vbert.model.")
    }
    missing, unexpected = model.load_state_dict(trainable, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")
    allowed_missing = {
        k
        for k in missing
        if k.startswith("wpt_w2vbert.model.")
    }
    if set(missing) - allowed_missing:
        raise RuntimeError(f"Missing required keys when loading checkpoint: {set(missing) - allowed_missing}")
    model.to(device)
    model.eval()

    return model, train_args, label_map, checkpoint


@torch.no_grad()
def evaluate(model, dataset, device, batch_size: int, num_workers: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    correct = 0
    total = 0
    per_lang_correct: dict[str, int] = {}
    per_lang_total: dict[str, int] = {}

    for waveform, _filename, labels in tqdm(loader, desc="Evaluating"):
        waveform = waveform.to(device)
        labels = labels.to(device)
        logits, _ = model(waveform)
        predicted = torch.argmax(logits, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for pred_id, true_id in zip(predicted.tolist(), labels.tolist()):
            lang = dataset.id_to_lang[true_id]
            per_lang_total[lang] = per_lang_total.get(lang, 0) + 1
            if pred_id == true_id:
                per_lang_correct[lang] = per_lang_correct.get(lang, 0) + 1

    accuracy = 100.0 * correct / total if total else 0.0
    per_lang_acc = {
        lang: 100.0 * per_lang_correct.get(lang, 0) / count
        for lang, count in sorted(per_lang_total.items())
    }
    return accuracy, correct, total, per_lang_acc


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VoxLingua LID checkpoint")
    parser.add_argument(
        "--checkpoint_dir",
        default=os.environ.get(
            "CKPT_VOXLINGUA_LID_DIR",
            os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                "checkpoints/voxlingua_lid",
            ),
        ),
    )
    parser.add_argument(
        "--eval_audio_root",
        default=os.environ.get("VOXLINGUA_EVAL_ROOT", ""),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    if not os.path.isdir(args.eval_audio_root):
        raise SystemExit(
            f"Eval audio root not found: {args.eval_audio_root}\n"
            "Set VOXLINGUA_EVAL_ROOT to your VoxLingua dev_vox folder."
        )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Eval data:  {args.eval_audio_root}")

    model, train_args, label_map, checkpoint = load_checkpoint(args.checkpoint_dir, device)
    audio_len = train_args.get("audio_len", 96000)

    dataset = VoxLinguaLangIDDataset(
        root_dir=args.eval_audio_root,
        audio_length=audio_len,
        label_map=label_map,
    )
    print(f"Samples: {len(dataset)} | Languages in checkpoint: {len(label_map)}")

    accuracy, correct, total, per_lang_acc = evaluate(
        model, dataset, device, args.batch_size, args.num_workers
    )

    ckpt_val_acc = checkpoint.get("val_acc")
    print("\n" + "=" * 60)
    print("VOXLINGUA LID EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}% ({correct}/{total})")
    if ckpt_val_acc is not None:
        print(f"Checkpoint val acc (training log): {float(ckpt_val_acc):.4f}%")

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "eval_audio_root": args.eval_audio_root,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "checkpoint_val_acc": ckpt_val_acc,
        "per_language_accuracy": per_lang_acc,
    }

    out_json = args.output_json or os.environ.get(
        "OUT_EVAL_VOXLINGUA", ""
    )
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
