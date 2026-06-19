#!/usr/bin/env python3
"""
Evaluate LID dual-path fusion on TL26 evaluation files.

This script computes and saves:
1) Identification metrics (micro/macro accuracy) on tl26_lid.txt
   - Pure baseline from stage2 LID embeddings (leave-one-out centroid classifier)
   - Checkpoint-based transformed outputs (Sub/Add/Fused logits) if checkpoint is provided
2) Verification metrics (EER) on tl26_pairs.txt + tl26_enroll.tsv
   - Pure scores from stage1/stage2 embeddings
   - Checkpoint-transformed scores + fusion strategies if checkpoint is provided
3) Per-item score files
   - Per-utterance identification scores
   - Per-pair verification scores
4) Optional score dumps for validation splits from training manifest (val/val2)

Usage example:
  python evaluate_lid_fusion_dual_path_manifest_tl26.py \
      --emb_base /path/to/multilingual_embeddings_stages_tidyvoicex2_asv2 \
      --tl26_lid /path/to/tl26_lid.txt \
      --tl26_pairs /path/to/tl26_pairs.txt \
      --tl26_enroll /path/to/tl26_enroll.tsv \
      --checkpoint /path/to/best_acc.pth \
      --manifest_file /path/to/training_manifest.txt \
      --output_dir /path/to/output_eval
"""

import argparse
import json
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Reuse model definitions and manifest parsing from training script.
from train_lid_fusion_dual_path_manifest import DualPathLanguageExtractor, ManifestDataLoader


def parse_lid_file(path: str) -> List[Tuple[str, str]]:
    items = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                items.append((parts[0], parts[1]))
    return items


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r") as f:
        for _ in f:
            n += 1
    return n


def parse_enroll_file(path: str) -> Dict[str, List[str]]:
    enroll = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                enroll_id = parts[0]
                enroll[enroll_id] = [p for p in parts[1:] if p]
    return enroll


def wav_to_npy_name(wav_name: str) -> str:
    return wav_name[:-4] + ".npy" if wav_name.endswith(".wav") else wav_name + ".npy"


def build_embedding_index(stage_dir: str) -> Dict[str, str]:
    """
    Build basename->path index with dev preference over train.
    """
    index = {}
    for fold in ["dev", "train"]:
        root = os.path.join(stage_dir, fold)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(".npy"):
                    continue
                if fn not in index:
                    index[fn] = os.path.join(dirpath, fn)
    return index


def load_embedding(index: Dict[str, str], wav_name: str) -> np.ndarray:
    npy_name = wav_to_npy_name(wav_name)
    path = index.get(npy_name)
    if path is None:
        raise FileNotFoundError(f"Embedding not found for '{wav_name}' ({npy_name})")
    return np.load(path).astype(np.float32)


def build_cached_loader(index: Dict[str, str], maxsize: int = 50000):
    @lru_cache(maxsize=maxsize)
    def _load_from_npy_name(npy_name: str) -> np.ndarray:
        path = index.get(npy_name)
        if path is None:
            raise FileNotFoundError(f"Embedding not found for '{npy_name}'")
        return np.load(path).astype(np.float32)

    def _load(wav_name: str) -> np.ndarray:
        return _load_from_npy_name(wav_to_npy_name(wav_name))

    return _load


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def compute_micro_macro(y_true: List[str], y_pred: List[str]) -> Tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    micro = float(np.mean(y_true_arr == y_pred_arr))

    per_class_acc = []
    classes = sorted(set(y_true))
    for c in classes:
        mask = (y_true_arr == c)
        if mask.sum() > 0:
            per_class_acc.append(float(np.mean(y_pred_arr[mask] == y_true_arr[mask])))
    macro = float(np.mean(per_class_acc)) if per_class_acc else 0.0
    return micro, macro


def eer_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) == 0:
        return 1.0

    labels = labels.astype(np.int64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 1.0

    # Keep this identical to training-time TrialEvaluator._eer_from_scores
    # so standalone validation EER is directly comparable to metrics.json.
    thresholds = np.linspace(float(np.min(scores)) - 0.1, float(np.max(scores)) + 0.1, 1000)
    min_eer = float("inf")
    for thr in thresholds:
        preds = (scores > thr).astype(np.int64)
        far = float(np.sum((preds == 1) & (labels == 0)) / n_neg)
        frr = float(np.sum((preds == 0) & (labels == 1)) / n_pos)
        eer = 0.5 * (far + frr)
        if eer < min_eer:
            min_eer = eer
    return float(min_eer)


def search_best_linear_fusion(base: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, object]:
    """
    Mirrors training-time score fusion search for key strategies.
    Returns best params + scores for each strategy.
    """
    out: Dict[str, object] = {}

    scores_sub = base["sub"]
    scores_add = base["add"]
    scores_lid = base["lid"]
    scores_asv = base["asv"]

    out["sub"] = eer_from_scores(scores_sub, labels)
    out["add"] = eer_from_scores(scores_add, labels)
    out["lid"] = eer_from_scores(scores_lid, labels)
    out["asv"] = eer_from_scores(scores_asv, labels)

    betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    def best_single_minus(main_name: str) -> Tuple[float, float, np.ndarray]:
        main = base[main_name]
        best_eer = 1.0
        best_b = 0.1
        best_score = main - 0.1 * scores_asv
        for b in betas:
            s = main - b * scores_asv
            e = eer_from_scores(s, labels)
            if e < best_eer:
                best_eer = e
                best_b = b
                best_score = s
        return best_eer, best_b, best_score

    e, b, s = best_single_minus("sub")
    out["sub-asv"] = e
    out["sub-asv_b"] = b
    out["sub-asv_scores"] = s.tolist()

    e, b, s = best_single_minus("add")
    out["add-asv"] = e
    out["add-asv_b"] = b
    out["add-asv_scores"] = s.tolist()

    e, b, s = best_single_minus("lid")
    out["lid-asv"] = e
    out["lid-asv_b"] = b
    out["lid-asv_scores"] = s.tolist()

    # (sub+add)-asv
    best_eer = 1.0
    best_params = (0.5, 0.1)
    best_scores = None
    alphas = np.linspace(0.1, 0.9, 9)
    for a in alphas:
        for b in betas:
            s = a * scores_sub + (1.0 - a) * scores_add - b * scores_asv
            e = eer_from_scores(s, labels)
            if e < best_eer:
                best_eer = e
                best_params = (float(a), float(b))
                best_scores = s
    out["(sub+add)-asv"] = best_eer
    out["(sub+add)-asv_params"] = best_params
    out["(sub+add)-asv_scores"] = best_scores.tolist() if best_scores is not None else []

    # lid+sub+add
    best_eer = 1.0
    best_params3 = (0.3, 0.3, 0.4)
    best_scores = None
    for a in [0.2, 0.3, 0.4, 0.5]:
        for b in [0.2, 0.3, 0.4, 0.5]:
            c = 1.0 - a - b
            if c <= 0.05:
                continue
            s = a * scores_lid + b * scores_sub + c * scores_add
            e = eer_from_scores(s, labels)
            if e < best_eer:
                best_eer = e
                best_params3 = (float(a), float(b), float(c))
                best_scores = s
    out["lid+sub+add"] = best_eer
    out["lid+sub+add_params"] = best_params3
    out["lid+sub+add_scores"] = best_scores.tolist() if best_scores is not None else []

    # lid+sub+add-asv
    best_eer = 1.0
    best_params4 = (0.3, 0.3, 0.4, 0.1)
    best_scores = None
    for a in [0.2, 0.3, 0.4, 0.5]:
        for b in [0.2, 0.3, 0.4, 0.5]:
            c = 1.0 - a - b
            if c <= 0.05:
                continue
            for d in betas:
                s = a * scores_lid + b * scores_sub + c * scores_add - d * scores_asv
                e = eer_from_scores(s, labels)
                if e < best_eer:
                    best_eer = e
                    best_params4 = (float(a), float(b), float(c), float(d))
                    best_scores = s
    out["lid+sub+add-asv"] = best_eer
    out["lid+sub+add-asv_params"] = best_params4
    out["lid+sub+add-asv_scores"] = best_scores.tolist() if best_scores is not None else []

    return out


def find_training_style_embedding(base_dir: str, wav_path: str) -> Optional[str]:
    npy_rel = wav_path.replace(".wav", ".npy")
    for fold in ["train", "dev"]:
        cand = os.path.join(base_dir, fold, npy_rel)
        if os.path.exists(cand):
            return cand
    return None


def evaluate_manifest_val_scores(
    manifest_file: str,
    emb_stage1_base: str,
    emb_stage2_base: str,
    output_dir: str,
    model: Optional[DualPathLanguageExtractor],
    device: torch.device,
) -> Dict[str, float]:
    """
    Save per-utterance score dumps for manifest val/val2 splits.
    """
    loader = ManifestDataLoader(manifest_file)
    split_map = {"val": loader.val_data, "val2": loader.val2_data}
    metrics = {}

    for split_name, data_list in split_map.items():
        out_path = os.path.join(output_dir, f"{split_name}_utterance_scores.txt")
        y_true = []
        y_pred_sub = []
        y_pred_add = []
        y_pred_fused = []

        with open(out_path, "w") as fw:
            fw.write("wav_path\ttrue_lang\tpred_sub\tpred_add\tpred_fused\tscore_sub\tscore_add\tscore_fused\n")
            for item in tqdm(data_list, desc=f"Saving {split_name} scores", leave=False):
                wav_path = item["path"]
                true_lang = item["language"]
                p1 = find_training_style_embedding(emb_stage1_base, wav_path)
                p2 = find_training_style_embedding(emb_stage2_base, wav_path)
                if p1 is None or p2 is None:
                    continue

                emb1 = np.load(p1).astype(np.float32)
                emb2 = np.load(p2).astype(np.float32)

                if model is None:
                    # No checkpoint mode: skip model-based score dumping.
                    continue

                with torch.no_grad():
                    t1 = torch.from_numpy(emb1).unsqueeze(0).to(device)
                    t2 = torch.from_numpy(emb2).unsqueeze(0).to(device)
                    sub_logits, add_logits, _, _ = model(t1, t2)
                    fused_logits = 0.5 * (sub_logits + add_logits)

                    sub_prob = torch.softmax(sub_logits, dim=1)
                    add_prob = torch.softmax(add_logits, dim=1)
                    fused_prob = torch.softmax(fused_logits, dim=1)

                    id_to_lang = {v: k for k, v in loader.lang_to_id.items()}
                    pred_sub = id_to_lang[int(torch.argmax(sub_logits, dim=1).item())]
                    pred_add = id_to_lang[int(torch.argmax(add_logits, dim=1).item())]
                    pred_fused = id_to_lang[int(torch.argmax(fused_logits, dim=1).item())]

                    score_sub = float(torch.max(sub_prob).item())
                    score_add = float(torch.max(add_prob).item())
                    score_fused = float(torch.max(fused_prob).item())

                y_true.append(true_lang)
                y_pred_sub.append(pred_sub)
                y_pred_add.append(pred_add)
                y_pred_fused.append(pred_fused)

                fw.write(
                    f"{wav_path}\t{true_lang}\t{pred_sub}\t{pred_add}\t{pred_fused}\t"
                    f"{score_sub:.6f}\t{score_add:.6f}\t{score_fused:.6f}\n"
                )

        if model is not None and y_true:
            sub_micro, sub_macro = compute_micro_macro(y_true, y_pred_sub)
            add_micro, add_macro = compute_micro_macro(y_true, y_pred_add)
            fused_micro, fused_macro = compute_micro_macro(y_true, y_pred_fused)
            metrics[f"{split_name}_sub_micro"] = sub_micro
            metrics[f"{split_name}_sub_macro"] = sub_macro
            metrics[f"{split_name}_add_micro"] = add_micro
            metrics[f"{split_name}_add_macro"] = add_macro
            metrics[f"{split_name}_fused_micro"] = fused_micro
            metrics[f"{split_name}_fused_macro"] = fused_macro

    return metrics


def evaluate_validation_trial_verification(
    trial_file: str,
    enroll_manifest: str,
    emb_stage1_base: str,
    emb_stage2_base: str,
    output_dir: str,
    model: Optional[DualPathLanguageExtractor],
    device: torch.device,
    score_batch_size: int,
) -> Dict[str, object]:
    """Evaluate verification on validation trial pairs and save per-pair scores.

    Trial format: <label> <enroll_id> <test_wav_relpath>
    Enrollment manifest format: <enroll_id>\t<wav_relpath1>\t<wav_relpath2>...
    """

    def parse_trials(path: str) -> List[Tuple[int, str, str]]:
        out = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                out.append((int(parts[0]), parts[1], parts[2]))
        return out

    print("Evaluating validation verification trials...")
    trials = parse_trials(trial_file)
    enroll_map = parse_enroll_file(enroll_manifest)

    enroll_stage1: Dict[str, np.ndarray] = {}
    enroll_stage2: Dict[str, np.ndarray] = {}

    for eid, wavs in tqdm(enroll_map.items(), desc="Val enrollment averaging", leave=False):
        embs1 = []
        embs2 = []
        for w in wavs:
            p1 = find_training_style_embedding(emb_stage1_base, w)
            p2 = find_training_style_embedding(emb_stage2_base, w)
            if p1 is None or p2 is None:
                continue
            try:
                embs1.append(np.load(p1).astype(np.float32))
                embs2.append(np.load(p2).astype(np.float32))
            except Exception:
                continue
        if embs1 and embs2:
            enroll_stage1[eid] = np.mean(np.stack(embs1, axis=0), axis=0)
            enroll_stage2[eid] = np.mean(np.stack(embs2, axis=0), axis=0)

    enroll_sub: Dict[str, np.ndarray] = {}
    enroll_add: Dict[str, np.ndarray] = {}
    if model is not None:
        for eid in tqdm(sorted(enroll_stage1.keys()), desc="Val transformed enrollment", leave=False):
            t1 = torch.from_numpy(enroll_stage1[eid]).unsqueeze(0).to(device)
            t2 = torch.from_numpy(enroll_stage2[eid]).unsqueeze(0).to(device)
            with torch.no_grad():
                enroll_sub[eid] = model.extract_sub_embedding(t1, t2).squeeze(0).cpu().numpy()
                enroll_add[eid] = model.extract_add_embedding(t1, t2).squeeze(0).cpu().numpy()

    labels_arr: List[int] = []
    lid_scores: List[float] = []
    asv_scores: List[float] = []
    sub_scores: List[float] = []
    add_scores: List[float] = []
    rows: List[Tuple[int, str, str, float, float, float, float]] = []

    pending_labels: List[int] = []
    pending_eids: List[str] = []
    pending_test_wavs: List[str] = []
    pending_lid: List[float] = []
    pending_asv: List[float] = []
    pending_t1: List[np.ndarray] = []
    pending_t2: List[np.ndarray] = []

    skipped_missing_enroll = 0
    skipped_missing_test = 0

    def flush_pending() -> None:
        if not pending_labels:
            return

        if model is not None:
            b1 = torch.from_numpy(np.stack(pending_t1, axis=0)).to(device)
            b2 = torch.from_numpy(np.stack(pending_t2, axis=0)).to(device)
            with torch.no_grad():
                sub_b = model.extract_sub_embedding(b1, b2).cpu().numpy()
                add_b = model.extract_add_embedding(b1, b2).cpu().numpy()
        else:
            sub_b = None
            add_b = None

        for i in range(len(pending_labels)):
            lab = pending_labels[i]
            eid = pending_eids[i]
            tw = pending_test_wavs[i]
            lid_s = pending_lid[i]
            asv_s = pending_asv[i]
            sub_s = float("nan")
            add_s = float("nan")
            if sub_b is not None and add_b is not None:
                sub_s = cosine(sub_b[i], enroll_sub[eid])
                add_s = cosine(add_b[i], enroll_add[eid])

            labels_arr.append(lab)
            lid_scores.append(lid_s)
            asv_scores.append(asv_s)
            if model is not None:
                sub_scores.append(sub_s)
                add_scores.append(add_s)
            rows.append((lab, eid, tw, lid_s, asv_s, sub_s, add_s))

        pending_labels.clear()
        pending_eids.clear()
        pending_test_wavs.clear()
        pending_lid.clear()
        pending_asv.clear()
        pending_t1.clear()
        pending_t2.clear()

    for lab, eid, test_wav in tqdm(trials, desc="Val trial scoring", leave=False):
        if eid not in enroll_stage1 or eid not in enroll_stage2:
            skipped_missing_enroll += 1
            continue
        p1 = find_training_style_embedding(emb_stage1_base, test_wav)
        p2 = find_training_style_embedding(emb_stage2_base, test_wav)
        if p1 is None or p2 is None:
            skipped_missing_test += 1
            continue
        try:
            t1 = np.load(p1).astype(np.float32)
            t2 = np.load(p2).astype(np.float32)
        except Exception:
            skipped_missing_test += 1
            continue

        pending_labels.append(lab)
        pending_eids.append(eid)
        pending_test_wavs.append(test_wav)
        pending_lid.append(cosine(t2, enroll_stage2[eid]))
        pending_asv.append(cosine(t1, enroll_stage1[eid]))
        pending_t1.append(t1)
        pending_t2.append(t2)

        if len(pending_labels) >= score_batch_size:
            flush_pending()

    flush_pending()

    if not labels_arr:
        return {
            "n_trials": 0,
            "skipped_missing_enroll": int(skipped_missing_enroll),
            "skipped_missing_test": int(skipped_missing_test),
        }

    labels_np = np.array(labels_arr, dtype=np.int64)
    lid_np = np.array(lid_scores, dtype=np.float32)
    asv_np = np.array(asv_scores, dtype=np.float32)

    # Base strategies.
    best_lid_minus_asv = lid_np - 0.1 * asv_np
    best_beta = 0.1
    best_e = eer_from_scores(best_lid_minus_asv, labels_np)
    for b in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        s = lid_np - b * asv_np
        e = eer_from_scores(s, labels_np)
        if e < best_e:
            best_e = e
            best_beta = b
            best_lid_minus_asv = s

    strategy_scores: Dict[str, np.ndarray] = {
        "pure_lid": lid_np,
        "pure_asv": asv_np,
        "pure_lid-asv": best_lid_minus_asv,
    }
    metrics: Dict[str, object] = {
        "pure_lid_eer": eer_from_scores(lid_np, labels_np),
        "pure_asv_eer": eer_from_scores(asv_np, labels_np),
        "pure_lid-asv_eer": best_e,
        "pure_lid-asv_beta": float(best_beta),
    }

    if model is not None and len(sub_scores) == len(labels_arr):
        sub_np = np.array(sub_scores, dtype=np.float32)
        add_np = np.array(add_scores, dtype=np.float32)
        base = {"sub": sub_np, "add": add_np, "lid": lid_np, "asv": asv_np}
        search_out = search_best_linear_fusion(base, labels_np)

        b_sub = float(search_out.get("sub-asv_b", 0.1))
        b_add = float(search_out.get("add-asv_b", 0.1))
        b_lid = float(search_out.get("lid-asv_b", 0.1))
        a_sa, b_sa = search_out.get("(sub+add)-asv_params", (0.5, 0.1))
        a_lsa, b_lsa, c_lsa = search_out.get("lid+sub+add_params", (0.3, 0.3, 0.4))
        a_lsaa, b_lsaa, c_lsaa, d_lsaa = search_out.get("lid+sub+add-asv_params", (0.3, 0.3, 0.4, 0.1))

        sub_asv = sub_np - b_sub * asv_np
        add_asv = add_np - b_add * asv_np
        lid_asv = lid_np - b_lid * asv_np
        sub_add_asv = a_sa * sub_np + (1.0 - a_sa) * add_np - b_sa * asv_np
        lid_sub_add = a_lsa * lid_np + b_lsa * sub_np + c_lsa * add_np
        lid_sub_add_asv = a_lsaa * lid_np + b_lsaa * sub_np + c_lsaa * add_np - d_lsaa * asv_np

        strategy_scores.update(
            {
                "sub": sub_np,
                "add": add_np,
                "lid": lid_np,
                "asv": asv_np,
                "sub-asv": sub_asv,
                "add-asv": add_asv,
                "lid-asv": lid_asv,
                "(sub+add)-asv": sub_add_asv,
                "lid+sub+add": lid_sub_add,
                "lid+sub+add-asv": lid_sub_add_asv,
            }
        )

        metrics.update(
            {
                "sub": eer_from_scores(sub_np, labels_np),
                "add": eer_from_scores(add_np, labels_np),
                "lid": eer_from_scores(lid_np, labels_np),
                "asv": eer_from_scores(asv_np, labels_np),
                "sub-asv": eer_from_scores(sub_asv, labels_np),
                "sub-asv_b": b_sub,
                "add-asv": eer_from_scores(add_asv, labels_np),
                "add-asv_b": b_add,
                "lid-asv": eer_from_scores(lid_asv, labels_np),
                "lid-asv_b": b_lid,
                "(sub+add)-asv": eer_from_scores(sub_add_asv, labels_np),
                "(sub+add)-asv_params": (float(a_sa), float(b_sa)),
                "lid+sub+add": eer_from_scores(lid_sub_add, labels_np),
                "lid+sub+add_params": (float(a_lsa), float(b_lsa), float(c_lsa)),
                "lid+sub+add-asv": eer_from_scores(lid_sub_add_asv, labels_np),
                "lid+sub+add-asv_params": (float(a_lsaa), float(b_lsaa), float(c_lsaa), float(d_lsaa)),
            }
        )

    strategy_eers = {name: eer_from_scores(scores, labels_np) for name, scores in strategy_scores.items()}
    best_strategy = min(strategy_eers, key=strategy_eers.get)
    best_eer = float(strategy_eers[best_strategy])
    best_scores = np.asarray(strategy_scores[best_strategy], dtype=np.float32)

    all_out = os.path.join(output_dir, "val_trial_pair_scores.txt")
    with open(all_out, "w") as fw:
        fw.write(
            "label\tenroll_id\ttest_wav\tpure_lid\tpure_asv\tpure_lid_minus_asv\tsub\tadd\t"
            "best_sub_add_minus_asv\tbest_lid_sub_add\tbest_lid_sub_add_asv\n"
        )
        for i, (lab, eid, tw, lid_s, asv_s, sub_s, add_s) in enumerate(rows):
            lid_minus = float(best_lid_minus_asv[i])
            best_sub_add = float("nan")
            best_lsa = float("nan")
            best_lsaa = float("nan")
            if "(sub+add)-asv" in strategy_scores:
                best_sub_add = float(strategy_scores["(sub+add)-asv"][i])
            if "lid+sub+add" in strategy_scores:
                best_lsa = float(strategy_scores["lid+sub+add"][i])
            if "lid+sub+add-asv" in strategy_scores:
                best_lsaa = float(strategy_scores["lid+sub+add-asv"][i])
            fw.write(
                f"{lab}\t{eid}\t{tw}\t{lid_s:.6f}\t{asv_s:.6f}\t{lid_minus:.6f}\t"
                f"{sub_s:.6f}\t{add_s:.6f}\t{best_sub_add:.6f}\t{best_lsa:.6f}\t{best_lsaa:.6f}\n"
            )

    best_out = os.path.join(output_dir, f"val_trial_pair_scores_best_{best_strategy}.txt")
    with open(best_out, "w") as fw:
        fw.write("label\tenroll_id\ttest_wav\tstrategy\tscore\n")
        for i, (lab, eid, tw, _, _, _, _) in enumerate(rows):
            fw.write(f"{lab}\t{eid}\t{tw}\t{best_strategy}\t{float(best_scores[i]):.6f}\n")

    metrics.update(
        {
            "n_trials": int(len(labels_arr)),
            "skipped_missing_enroll": int(skipped_missing_enroll),
            "skipped_missing_test": int(skipped_missing_test),
            "strategy_eers": strategy_eers,
            "best_strategy": best_strategy,
            "best_strategy_eer": best_eer,
            "pair_score_file": all_out,
            "best_pair_score_file": best_out,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LID dual-path model on TL26")
    parser.add_argument("--emb_base", required=True, help="Embedding root containing stage1_asv and stage2_lid")
    parser.add_argument("--tl26_lid", required=True)
    parser.add_argument("--tl26_pairs", required=True)
    parser.add_argument("--tl26_enroll", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--checkpoint", default="", help="Checkpoint path (best_acc.pth or best_eer.pth)")
    parser.add_argument("--manifest_file", default="", help="Training manifest to rebuild language id mapping")

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--subspace_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--arcface_margin", type=float, default=0.3)
    parser.add_argument("--arcface_scale", type=float, default=30.0)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_manifest_val_scores", action="store_true")
    parser.add_argument("--emb_stage1_base", default="", help="Training-style stage1 base for manifest val/val2")
    parser.add_argument("--emb_stage2_base", default="", help="Training-style stage2 base for manifest val/val2")
    parser.add_argument("--val_trial_file", default="", help="Optional validation verification trial file")
    parser.add_argument("--val_enroll_manifest", default="", help="Optional enrollment manifest for validation trials")
    parser.add_argument("--max_pairs", type=int, default=0, help="Optional cap for TL26 pairs (0 = all)")
    parser.add_argument(
        "--score_batch_size",
        type=int,
        default=1024,
        help="Batch size for checkpoint pair scoring (Sub/Add embedding extraction)",
    )
    parser.add_argument(
        "--fusion_search_max_pairs",
        type=int,
        default=300000,
        help="Max pairs used for fusion weight search; 0 means use all pairs",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50000,
        help="Print explicit timing progress every N scored pairs (0 disables)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    stage1_dir = os.path.join(args.emb_base, "stage1_asv")
    stage2_dir = os.path.join(args.emb_base, "stage2_lid")

    print("Building embedding indices...")
    idx_stage1 = build_embedding_index(stage1_dir)
    idx_stage2 = build_embedding_index(stage2_dir)
    load_stage1 = build_cached_loader(idx_stage1)
    load_stage2 = build_cached_loader(idx_stage2)
    print(f"  stage1 index size: {len(idx_stage1)}")
    print(f"  stage2 index size: {len(idx_stage2)}")

    # -------------------------
    # Optional model loading
    # -------------------------
    model = None
    manifest_loader = None
    checkpoint_enabled = bool(args.checkpoint)

    if checkpoint_enabled:
        if not args.manifest_file:
            raise ValueError("--manifest_file is required when --checkpoint is provided")

        manifest_loader = ManifestDataLoader(args.manifest_file)
        model = DualPathLanguageExtractor(
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            subspace_dim=args.subspace_dim,
            num_heads=args.num_heads,
            num_classes=manifest_loader.num_classes,
            dropout=args.dropout,
            arcface_margin=args.arcface_margin,
            arcface_scale=args.arcface_scale,
        ).to(device)

        state = torch.load(args.checkpoint, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"Loaded checkpoint: {args.checkpoint}")

    # -------------------------
    # Identification evaluation
    # -------------------------
    print("Evaluating TL26 identification (tl26_lid)...")
    lid_items = parse_lid_file(args.tl26_lid)

    utt_out_path = os.path.join(args.output_dir, "tl26_identification_scores.txt")
    y_true = []

    # Pure leave-one-out centroid baseline on stage2 embeddings.
    emb2_by_utt = {}
    labels = []
    for wav, lang in tqdm(lid_items, desc="Loading TL26 LID embeddings", leave=False):
        try:
            emb2_by_utt[wav] = load_stage2(wav)
            labels.append(lang)
        except Exception:
            continue

    valid_items = [(w, l) for (w, l) in lid_items if w in emb2_by_utt]
    classes = sorted(set(l for _, l in valid_items))

    class_sum = {c: np.zeros(args.embed_dim, dtype=np.float32) for c in classes}
    class_cnt = {c: 0 for c in classes}
    for w, l in valid_items:
        class_sum[l] += emb2_by_utt[w]
        class_cnt[l] += 1

    pure_pred = []
    pure_true = []

    # Checkpoint predictions
    ckpt_sub_pred = []
    ckpt_add_pred = []
    ckpt_fused_pred = []
    ckpt_true = []

    with open(utt_out_path, "w") as fw:
        fw.write(
            "wav\ttrue_lang\tpure_pred\tpure_score\t"
            "ckpt_sub_pred\tckpt_sub_score\tckpt_add_pred\tckpt_add_score\t"
            "ckpt_fused_pred\tckpt_fused_score\n"
        )

        for wav, lang in tqdm(valid_items, desc="Scoring TL26 identification", leave=False):
            y_true.append(lang)
            emb2 = emb2_by_utt[wav]

            # Pure leave-one-out centroid score.
            best_lang = None
            best_score = -1e9
            for c in classes:
                cnt = class_cnt[c]
                if cnt <= 0:
                    continue
                if c == lang and cnt > 1:
                    centroid = (class_sum[c] - emb2) / (cnt - 1)
                else:
                    centroid = class_sum[c] / cnt
                s = cosine(emb2, centroid)
                if s > best_score:
                    best_score = s
                    best_lang = c

            pure_true.append(lang)
            pure_pred.append(best_lang if best_lang is not None else "")

            ck_sub_lang = "NA"
            ck_add_lang = "NA"
            ck_fused_lang = "NA"
            ck_sub_score = float("nan")
            ck_add_score = float("nan")
            ck_fused_score = float("nan")

            if model is not None and manifest_loader is not None:
                try:
                    emb1 = load_stage1(wav)
                    t1 = torch.from_numpy(emb1).unsqueeze(0).to(device)
                    t2 = torch.from_numpy(emb2).unsqueeze(0).to(device)
                    with torch.no_grad():
                        sub_logits, add_logits, _, _ = model(t1, t2)
                        fused_logits = 0.5 * (sub_logits + add_logits)

                        sub_prob = torch.softmax(sub_logits, dim=1)
                        add_prob = torch.softmax(add_logits, dim=1)
                        fused_prob = torch.softmax(fused_logits, dim=1)

                        id_to_lang = {v: k for k, v in manifest_loader.lang_to_id.items()}
                        sub_id = int(torch.argmax(sub_logits, dim=1).item())
                        add_id = int(torch.argmax(add_logits, dim=1).item())
                        fused_id = int(torch.argmax(fused_logits, dim=1).item())

                        ck_sub_lang = id_to_lang.get(sub_id, "UNK")
                        ck_add_lang = id_to_lang.get(add_id, "UNK")
                        ck_fused_lang = id_to_lang.get(fused_id, "UNK")

                        ck_sub_score = float(torch.max(sub_prob).item())
                        ck_add_score = float(torch.max(add_prob).item())
                        ck_fused_score = float(torch.max(fused_prob).item())

                    # Only score classes known by model mapping.
                    if lang in manifest_loader.lang_to_id:
                        ckpt_true.append(lang)
                        ckpt_sub_pred.append(ck_sub_lang)
                        ckpt_add_pred.append(ck_add_lang)
                        ckpt_fused_pred.append(ck_fused_lang)
                except Exception:
                    pass

            fw.write(
                f"{wav}\t{lang}\t{pure_pred[-1]}\t{best_score:.6f}\t"
                f"{ck_sub_lang}\t{ck_sub_score:.6f}\t"
                f"{ck_add_lang}\t{ck_add_score:.6f}\t"
                f"{ck_fused_lang}\t{ck_fused_score:.6f}\n"
            )

    pure_micro, pure_macro = compute_micro_macro(pure_true, pure_pred)

    # -------------------------
    # Pair/EER evaluation
    # -------------------------
    print("Evaluating TL26 pair verification (tl26_pairs + tl26_enroll)...")
    enroll_map = parse_enroll_file(args.tl26_enroll)

    enroll_stage1 = {}
    enroll_stage2 = {}

    for eid, wavs in tqdm(enroll_map.items(), desc="Averaging enrollment embeddings", leave=False):
        embs1 = []
        embs2 = []
        for w in wavs:
            try:
                embs1.append(load_stage1(w))
                embs2.append(load_stage2(w))
            except Exception:
                continue
        if embs1 and embs2:
            enroll_stage1[eid] = np.mean(np.stack(embs1, axis=0), axis=0)
            enroll_stage2[eid] = np.mean(np.stack(embs2, axis=0), axis=0)

    # Optional transformed enrollment embeddings cache.
    enroll_sub = {}
    enroll_add = {}
    if model is not None:
        for eid in tqdm(sorted(enroll_stage1.keys()), desc="Extracting transformed enrollment features", leave=False):
            t1 = torch.from_numpy(enroll_stage1[eid]).unsqueeze(0).to(device)
            t2 = torch.from_numpy(enroll_stage2[eid]).unsqueeze(0).to(device)
            with torch.no_grad():
                sub_e = model.extract_sub_embedding(t1, t2).squeeze(0).cpu().numpy()
                add_e = model.extract_add_embedding(t1, t2).squeeze(0).cpu().numpy()
            enroll_sub[eid] = sub_e
            enroll_add[eid] = add_e

    pair_out_path = os.path.join(args.output_dir, "tl26_pair_scores.txt")
    tmp_basic_path = os.path.join(args.output_dir, "tl26_pair_scores_basic.tmp")

    labels_arr = []
    pure_lid_scores = []
    pure_asv_scores = []
    sub_scores = []
    add_scores = []

    total_pairs = count_lines(args.tl26_pairs)
    processed = 0
    skipped_missing_enroll = 0
    skipped_missing_test = 0
    pair_start_time = time.time()

    print(
        f"Pair scoring config: total_lines={total_pairs}, score_batch_size={args.score_batch_size}, "
        f"log_every={args.log_every}"
    )

    # Pending queue for batched checkpoint inference.
    pending_labels = []
    pending_eids = []
    pending_wavs = []
    pending_origs = []
    pending_lid = []
    pending_asv = []
    pending_test1 = []
    pending_test2 = []

    def flush_pending_pairs(ft_handle) -> None:
        if not pending_labels:
            return

        if model is not None:
            batch1 = torch.from_numpy(np.stack(pending_test1, axis=0)).to(device)
            batch2 = torch.from_numpy(np.stack(pending_test2, axis=0)).to(device)
            with torch.no_grad():
                sub_batch = model.extract_sub_embedding(batch1, batch2).cpu().numpy()
                add_batch = model.extract_add_embedding(batch1, batch2).cpu().numpy()
        else:
            sub_batch = None
            add_batch = None

        for i in range(len(pending_labels)):
            label_i = pending_labels[i]
            eid_i = pending_eids[i]
            wav_i = pending_wavs[i]
            orig_i = pending_origs[i]
            lid_i = pending_lid[i]
            asv_i = pending_asv[i]

            sub_i = float("nan")
            add_i = float("nan")
            if sub_batch is not None and add_batch is not None:
                sub_i = cosine(sub_batch[i], enroll_sub[eid_i])
                add_i = cosine(add_batch[i], enroll_add[eid_i])

            labels_arr.append(label_i)
            pure_lid_scores.append(lid_i)
            pure_asv_scores.append(asv_i)
            if model is not None:
                sub_scores.append(sub_i)
                add_scores.append(add_i)

            orig_val = orig_i if orig_i is not None else float("nan")
            ft_handle.write(
                f"{label_i}\t{eid_i}\t{wav_i}\t{orig_val:.6f}\t{lid_i:.6f}\t{asv_i:.6f}\t{sub_i:.6f}\t{add_i:.6f}\n"
            )

        pending_labels.clear()
        pending_eids.clear()
        pending_wavs.clear()
        pending_origs.clear()
        pending_lid.clear()
        pending_asv.clear()
        pending_test1.clear()
        pending_test2.clear()

    with open(args.tl26_pairs, "r") as fr, open(tmp_basic_path, "w") as ft:
        ft.write("label\tenroll_id\ttest_wav\torig_score\tpure_lid\tpure_asv\tsub\tadd\n")

        for line in tqdm(fr, total=total_pairs, desc="Scoring pairs", leave=False):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            label = int(parts[0])
            eid = parts[1]
            test_wav = parts[2]
            orig_score = float(parts[3]) if len(parts) >= 4 else None

            if args.max_pairs > 0 and processed >= args.max_pairs:
                break

            if eid not in enroll_stage1 or eid not in enroll_stage2:
                skipped_missing_enroll += 1
                continue
            try:
                test1 = load_stage1(test_wav)
                test2 = load_stage2(test_wav)
            except Exception:
                skipped_missing_test += 1
                continue

            lid_s = cosine(test2, enroll_stage2[eid])
            asv_s = cosine(test1, enroll_stage1[eid])

            pending_labels.append(label)
            pending_eids.append(eid)
            pending_wavs.append(test_wav)
            pending_origs.append(orig_score)
            pending_lid.append(lid_s)
            pending_asv.append(asv_s)
            pending_test1.append(test1)
            pending_test2.append(test2)

            if len(pending_labels) >= args.score_batch_size:
                flush_pending_pairs(ft)
            processed += 1

            if args.log_every > 0 and processed % args.log_every == 0:
                elapsed = max(1e-6, time.time() - pair_start_time)
                speed = processed / elapsed
                rem = max(0, total_pairs - processed)
                eta_sec = rem / max(1e-6, speed)
                eta_min = eta_sec / 60.0
                print(
                    f"[PAIR-PROGRESS] scored={processed}/{total_pairs} "
                    f"({100.0 * processed / max(1, total_pairs):.2f}%), "
                    f"speed={speed:.1f} pairs/s, eta={eta_min:.1f} min"
                )

        # Flush the last partial batch.
        flush_pending_pairs(ft)

    total_elapsed = max(1e-6, time.time() - pair_start_time)
    print(
        f"[PAIR-DONE] scored={processed}, elapsed={total_elapsed/60.0:.2f} min, "
        f"avg_speed={processed/total_elapsed:.1f} pairs/s"
    )

    labels_np = np.array(labels_arr, dtype=np.int64)
    pure_lid_np = np.array(pure_lid_scores, dtype=np.float32)
    pure_asv_np = np.array(pure_asv_scores, dtype=np.float32)

    print("Computing EER metrics and fusion search...")

    # Pure baseline EERs.
    pure_results = {
        "pure_lid_eer": eer_from_scores(pure_lid_np, labels_np),
        "pure_asv_eer": eer_from_scores(pure_asv_np, labels_np),
    }

    # Search best pure lid-asv subtraction.
    best_e = 1.0
    best_b = 0.1
    best_scores = pure_lid_np - 0.1 * pure_asv_np
    for b in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        s = pure_lid_np - b * pure_asv_np
        e = eer_from_scores(s, labels_np)
        if e < best_e:
            best_e = e
            best_b = b
            best_scores = s
    pure_results["pure_lid-asv_eer"] = best_e
    pure_results["pure_lid-asv_beta"] = best_b

    transformed_results = {}
    transformed_results_summary = {}
    best_lid_sub_add = []
    best_lid_sub_add_asv = []
    best_sub_add_minus_asv = []

    # Strategy score containers for global best-strategy selection (v12 style).
    strategy_scores = {
        "pure_lid": pure_lid_np,
        "pure_asv": pure_asv_np,
        "pure_lid-asv": best_scores,
    }

    if model is not None and len(sub_scores) == len(labels_arr):
        base_full = {
            "sub": np.array(sub_scores, dtype=np.float32),
            "add": np.array(add_scores, dtype=np.float32),
            "lid": pure_lid_np,
            "asv": pure_asv_np,
        }

        # Search fusion params on a subset for speed, then evaluate on full arrays.
        n_pairs = len(labels_np)
        use_sample = args.fusion_search_max_pairs > 0 and n_pairs > args.fusion_search_max_pairs
        if use_sample:
            rng = np.random.default_rng(42)
            sel = rng.choice(n_pairs, size=args.fusion_search_max_pairs, replace=False)
            sel.sort()
            labels_search = labels_np[sel]
            base_search = {k: v[sel] for k, v in base_full.items()}
            print(
                f"Fusion search on subset: {len(labels_search)}/{n_pairs} pairs "
                f"(final EERs computed on full set)"
            )
        else:
            labels_search = labels_np
            base_search = base_full

        search_out = search_best_linear_fusion(base_search, labels_search)

        # Recover params and score full set with fixed fusion weights.
        sub_full = base_full["sub"]
        add_full = base_full["add"]
        lid_full = base_full["lid"]
        asv_full = base_full["asv"]

        b_sub = float(search_out.get("sub-asv_b", 0.1))
        b_add = float(search_out.get("add-asv_b", 0.1))
        b_lid = float(search_out.get("lid-asv_b", 0.1))
        a_sa, b_sa = search_out.get("(sub+add)-asv_params", (0.5, 0.1))
        a_lsa, b_lsa, c_lsa = search_out.get("lid+sub+add_params", (0.3, 0.3, 0.4))
        a_lsaa, b_lsaa, c_lsaa, d_lsaa = search_out.get("lid+sub+add-asv_params", (0.3, 0.3, 0.4, 0.1))

        sub_asv_full = sub_full - b_sub * asv_full
        add_asv_full = add_full - b_add * asv_full
        lid_asv_full = lid_full - b_lid * asv_full
        sub_add_asv_full = a_sa * sub_full + (1.0 - a_sa) * add_full - b_sa * asv_full
        lid_sub_add_full = a_lsa * lid_full + b_lsa * sub_full + c_lsa * add_full
        lid_sub_add_asv_full = a_lsaa * lid_full + b_lsaa * sub_full + c_lsaa * add_full - d_lsaa * asv_full

        transformed_results = {
            "sub": eer_from_scores(sub_full, labels_np),
            "add": eer_from_scores(add_full, labels_np),
            "lid": eer_from_scores(lid_full, labels_np),
            "asv": eer_from_scores(asv_full, labels_np),
            "sub-asv": eer_from_scores(sub_asv_full, labels_np),
            "sub-asv_b": b_sub,
            "add-asv": eer_from_scores(add_asv_full, labels_np),
            "add-asv_b": b_add,
            "lid-asv": eer_from_scores(lid_asv_full, labels_np),
            "lid-asv_b": b_lid,
            "(sub+add)-asv": eer_from_scores(sub_add_asv_full, labels_np),
            "(sub+add)-asv_params": (float(a_sa), float(b_sa)),
            "lid+sub+add": eer_from_scores(lid_sub_add_full, labels_np),
            "lid+sub+add_params": (float(a_lsa), float(b_lsa), float(c_lsa)),
            "lid+sub+add-asv": eer_from_scores(lid_sub_add_asv_full, labels_np),
            "lid+sub+add-asv_params": (float(a_lsaa), float(b_lsaa), float(c_lsaa), float(d_lsaa)),
        }

        transformed_results_summary = transformed_results
        best_sub_add_minus_asv = sub_add_asv_full.tolist()
        best_lid_sub_add = lid_sub_add_full.tolist()
        best_lid_sub_add_asv = lid_sub_add_asv_full.tolist()

        # Add checkpoint-based strategies to global strategy pool.
        strategy_scores.update(
            {
                "sub": sub_full,
                "add": add_full,
                "lid": lid_full,
                "asv": asv_full,
                "sub-asv": sub_asv_full,
                "add-asv": add_asv_full,
                "lid-asv": lid_asv_full,
                "(sub+add)-asv": sub_add_asv_full,
                "lid+sub+add": lid_sub_add_full,
                "lid+sub+add-asv": lid_sub_add_asv_full,
            }
        )

    # EER for every available strategy and global best strategy selection.
    strategy_eers = {name: eer_from_scores(scores, labels_np) for name, scores in strategy_scores.items()}
    best_strategy = min(strategy_eers, key=strategy_eers.get)
    best_strategy_eer = float(strategy_eers[best_strategy])
    best_strategy_scores = np.asarray(strategy_scores[best_strategy], dtype=np.float32)
    print(f"Best strategy: {best_strategy} (EER={best_strategy_eer:.6f})")

    # Save pair-level scores including best fused scores.
    print("Writing final pair score file with fused columns...")
    with open(pair_out_path, "w") as fw:
        fw.write(
            "label\tenroll_id\ttest_wav\torig_score\tpure_lid\tpure_asv\tpure_lid_minus_asv\t"
            "sub\tadd\tbest_sub_add_minus_asv\tbest_lid_sub_add\tbest_lid_sub_add_asv\n"
        )
        with open(tmp_basic_path, "r") as fr:
            header = fr.readline()
            for i, line in enumerate(fr):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 8:
                    continue
                label, eid, test_wav, orig, pure_lid, pure_asv, sub, add = parts
                lid_minus_asv = float(best_scores[i]) if i < len(best_scores) else float("nan")
                s1 = float(best_sub_add_minus_asv[i]) if i < len(best_sub_add_minus_asv) else float("nan")
                s2 = float(best_lid_sub_add[i]) if i < len(best_lid_sub_add) else float("nan")
                s3 = float(best_lid_sub_add_asv[i]) if i < len(best_lid_sub_add_asv) else float("nan")

                fw.write(
                    f"{label}\t{eid}\t{test_wav}\t{orig}\t"
                    f"{pure_lid}\t{pure_asv}\t{lid_minus_asv:.6f}\t"
                    f"{sub}\t{add}\t{s1:.6f}\t{s2:.6f}\t{s3:.6f}\n"
                )

    # Save best-strategy scores only (same spirit as eval_fusion_dual_path_v12_val).
    best_pair_out_path = os.path.join(args.output_dir, f"tl26_pair_scores_best_{best_strategy}.txt")
    print(f"Writing best-strategy score file: {best_pair_out_path}")
    with open(best_pair_out_path, "w") as fw_best, open(tmp_basic_path, "r") as fr:
        fw_best.write("label\tenroll_id\ttest_wav\torig_score\tstrategy\tscore\n")
        header = fr.readline()
        for i, line in enumerate(fr):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8:
                continue
            label, eid, test_wav, orig, pure_lid, pure_asv, sub, add = parts
            score = float(best_strategy_scores[i]) if i < len(best_strategy_scores) else float("nan")
            fw_best.write(f"{label}\t{eid}\t{test_wav}\t{orig}\t{best_strategy}\t{score:.6f}\n")

    if os.path.exists(tmp_basic_path):
        os.remove(tmp_basic_path)

    # -------------------------
    # Optional manifest val score dumping
    # -------------------------
    manifest_val_metrics = {}
    if args.save_manifest_val_scores:
        if not args.manifest_file or not args.emb_stage1_base or not args.emb_stage2_base:
            raise ValueError(
                "--save_manifest_val_scores requires --manifest_file, --emb_stage1_base, --emb_stage2_base"
            )
        manifest_val_metrics = evaluate_manifest_val_scores(
            args.manifest_file,
            args.emb_stage1_base,
            args.emb_stage2_base,
            args.output_dir,
            model,
            device,
        )

    validation_verification_metrics = {}
    if args.val_trial_file and args.val_enroll_manifest:
        if not args.emb_stage1_base or not args.emb_stage2_base:
            raise ValueError(
                "--val_trial_file/--val_enroll_manifest require --emb_stage1_base and --emb_stage2_base"
            )
        validation_verification_metrics = evaluate_validation_trial_verification(
            args.val_trial_file,
            args.val_enroll_manifest,
            args.emb_stage1_base,
            args.emb_stage2_base,
            args.output_dir,
            model,
            device,
            args.score_batch_size,
        )

    # -------------------------
    # Save summary
    # -------------------------
    summary = {
        "identification": {
            "n_items": len(valid_items),
            "pure_micro": pure_micro,
            "pure_macro": pure_macro,
            "pure_score_file": utt_out_path,
        },
        "verification": {
            "n_pairs_scored": int(len(labels_arr)),
            "pairs_total_lines": int(total_pairs),
            "pairs_skipped_missing_enroll": int(skipped_missing_enroll),
            "pairs_skipped_missing_test": int(skipped_missing_test),
            **pure_results,
            **transformed_results_summary,
            "strategy_eers": strategy_eers,
            "best_strategy": best_strategy,
            "best_strategy_eer": best_strategy_eer,
            "pair_score_file": pair_out_path,
            "best_pair_score_file": best_pair_out_path,
        },
        "manifest_val_metrics": manifest_val_metrics,
        "validation_verification": validation_verification_metrics,
    }

    if model is not None and ckpt_true:
        sub_micro, sub_macro = compute_micro_macro(ckpt_true, ckpt_sub_pred)
        add_micro, add_macro = compute_micro_macro(ckpt_true, ckpt_add_pred)
        fused_micro, fused_macro = compute_micro_macro(ckpt_true, ckpt_fused_pred)
        summary["identification"]["checkpoint_n_items"] = len(ckpt_true)
        summary["identification"]["ckpt_sub_micro"] = sub_micro
        summary["identification"]["ckpt_sub_macro"] = sub_macro
        summary["identification"]["ckpt_add_micro"] = add_micro
        summary["identification"]["ckpt_add_macro"] = add_macro
        summary["identification"]["ckpt_fused_micro"] = fused_micro
        summary["identification"]["ckpt_fused_macro"] = fused_macro

    summary_path = os.path.join(args.output_dir, "tl26_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Evaluation complete")
    print(f"Identification score file: {utt_out_path}")
    print(f"Pair score file:           {pair_out_path}")
    print(f"Best strategy score file:  {best_pair_out_path}")
    print(f"Summary JSON:              {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
