"""
Evaluate Dual-Path Fusion V12 on TidyVoiceX2_ASV trials
=========================================================

Two evaluation modes:
  1. Baseline EER — raw ASV/LID cosine similarity from pre-extracted embeddings
  2. Fusion EER  — load trained DualPathModel, produce Sub/Add embeddings,
                   then run full 4-way score fusion

Embeddings are loaded from flat .npy directories:
    emb_base/stage1_asv/dev/<filename>.npy   (ASV embeddings)
    emb_base/stage2_lid/dev/<filename>.npy   (LID embeddings)

Trial files: <enroll_id> <test_id> <target|nontarget>

Usage:
    python eval_fusion_dual_path_v12.py \
        --emb_base ./multilingual_embeddings_stages_tidyvoicex2_asv2 \
        --model_checkpoint ./path/to/best_model.pt \
        --trial_files trial_with_labels/task1_labels.txt trial_with_labels/task2_labels.txt \
        --gpu 0
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DualPathModel and dependencies from training code
from train_fusion_dual_path_v12 import DualPathModel, compute_eer


def compute_minDCF(target_scores, nontarget_scores, ptar=0.5, cfa=1, cfr=1):
    """Compute minimum Detection Cost Function
    
    Args:
        target_scores: scores for target speaker trials
        nontarget_scores: scores for non-target speaker trials
        ptar: prior probability of target speaker (default: 0.5)
        cfa: cost of false acceptance (default: 1)
        cfr: cost of false rejection (default: 1)
    
    Returns:
        minDCF value (in percentage)
    """
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return 100.0
    
    scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([np.ones(len(target_scores)), np.zeros(len(nontarget_scores))])
    
    # Sort by score descending
    desc_idx = np.argsort(scores)[::-1]
    labels_sorted = labels[desc_idx]
    
    # Compute cumulative FPR and FNR
    num_pos = labels.sum()
    num_neg = len(labels) - num_pos
    
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    
    fnr = 1 - tp / num_pos
    fpr = fp / num_neg
    
    # Compute DCF at each threshold
    dcf = ptar * cfr * fnr + (1 - ptar) * cfa * fpr
    
    # Return minimum DCF in percentage
    return np.min(dcf) * 100


def load_flat_embeddings(emb_dir):
    """Load all .npy files from a flat directory into a dict {filename: embedding}."""
    emb_dict = {}
    npy_files = sorted(f for f in os.listdir(emb_dir) if f.endswith('.npy'))
    print(f"  Loading {len(npy_files)} embeddings from {emb_dir}")
    for f in tqdm(npy_files, desc="  Loading", leave=False):
        key = os.path.splitext(f)[0]
        emb_dict[key] = np.load(os.path.join(emb_dir, f)).astype(np.float32)
    return emb_dict


def parse_trials(trial_file, available_keys):
    """Parse trial file and return enroll_ids, test_ids, labels (1=target, 0=nontarget)."""
    enroll_ids, test_ids, labels = [], [], []
    skipped = 0
    with open(trial_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            enroll_id, test_id, label_str = parts
            if enroll_id not in available_keys or test_id not in available_keys:
                skipped += 1
                continue
            enroll_ids.append(enroll_id)
            test_ids.append(test_id)
            labels.append(1 if label_str == 'target' else 0)
    labels = np.array(labels, dtype=np.int32)
    print(f"  Loaded {len(labels)} trials ({labels.sum()} target, {(1-labels).sum()} nontarget)")
    if skipped > 0:
        print(f"  Skipped {skipped} trials (embedding not found)")
    return enroll_ids, test_ids, labels


def compute_cosine_scores(emb_dict, enroll_ids, test_ids, batch_size=500000):
    """Compute cosine similarity scores for all trial pairs."""
    n = len(enroll_ids)
    scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        enroll = np.stack([emb_dict[enroll_ids[i]] for i in range(start, end)])
        test = np.stack([emb_dict[test_ids[i]] for i in range(start, end)])
        # L2 normalize
        enroll = enroll / np.clip(np.linalg.norm(enroll, axis=1, keepdims=True), 1e-8, None)
        test = test / np.clip(np.linalg.norm(test, axis=1, keepdims=True), 1e-8, None)
        scores[start:end] = np.sum(enroll * test, axis=1)
    return scores


def extract_fusion_embeddings(model, asv_embs_np, lid_embs_np, device, batch_size=512):
    """Run ASV+LID embeddings through the fusion model to get Sub and Add embeddings."""
    model.eval()
    sub_list, add_list = [], []
    n = len(asv_embs_np)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            asv = torch.tensor(asv_embs_np[i:end]).to(device)
            lid = torch.tensor(lid_embs_np[i:end]).to(device)
            sub_list.append(model.extract_sub_embedding(asv, lid).cpu().numpy())
            add_list.append(model.extract_add_embedding(asv, lid).cpu().numpy())
    return np.vstack(sub_list), np.vstack(add_list)


def save_scores_file(filepath, enroll_ids, test_ids, labels, scores):
    """Save trial scores: enroll_id test_id target/nontarget score"""
    with open(filepath, 'w') as f:
        for eid, tid, lab, sc in zip(enroll_ids, test_ids, labels, scores):
            label_str = 'target' if lab == 1 else 'nontarget'
            f.write(f"{eid} {tid} {label_str} {sc:.6f}\n")


def evaluate_trial(task_name, enroll_ids, test_ids, labels,
                   asv_dict, lid_dict, sub_dict=None, add_dict=None,
                   output_dir=None):
    """Evaluate a single trial file: baseline + fusion EER."""
    target_mask = labels == 1

    if output_dir:
        scores_dir = os.path.join(output_dir, 'scores')
        os.makedirs(scores_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {task_name}")
    print(f"{'='*70}")

    # Initialize results dictionary
    results = {}

    # --- Baseline: raw ASV cosine ---
    asv_scores = compute_cosine_scores(asv_dict, enroll_ids, test_ids)
    asv_eer, _ = compute_eer(asv_scores[target_mask], asv_scores[~target_mask])
    asv_minDCF = compute_minDCF(asv_scores[target_mask], asv_scores[~target_mask])
    print(f"\n  [Baseline] ASV-only:       {asv_eer:.4f}% EER | {asv_minDCF:.4f}% minDCF")
    results['asv_only'] = asv_eer
    results['asv_only_minDCF'] = asv_minDCF

    # --- Baseline: raw LID cosine ---
    lid_scores = compute_cosine_scores(lid_dict, enroll_ids, test_ids)
    lid_eer, _ = compute_eer(lid_scores[target_mask], lid_scores[~target_mask])
    lid_minDCF = compute_minDCF(lid_scores[target_mask], lid_scores[~target_mask])
    print(f"  [Baseline] LID-only:       {lid_eer:.4f}% EER | {lid_minDCF:.4f}% minDCF")
    results['lid_only'] = lid_eer
    results['lid_only_minDCF'] = lid_minDCF

    # --- ASV - LID baseline ---
    betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    best_eer_asv_lid, best_mindcf_asv_lid, best_b = 100.0, 100.0, 0.1
    for b in betas:
        eer, _ = compute_eer(
            (asv_scores - b * lid_scores)[target_mask],
            (asv_scores - b * lid_scores)[~target_mask],
        )
        mindcf = compute_minDCF(
            (asv_scores - b * lid_scores)[target_mask],
            (asv_scores - b * lid_scores)[~target_mask],
        )
        if eer < best_eer_asv_lid:
            best_eer_asv_lid, best_b = eer, b
        if mindcf < best_mindcf_asv_lid:
            best_mindcf_asv_lid = mindcf
    print(f"  [Baseline] ASV-LID:        {best_eer_asv_lid:.4f}% EER | {best_mindcf_asv_lid:.4f}% minDCF  (β={best_b:.2f})")

    results['asv-lid'] = best_eer_asv_lid
    results['asv-lid_minDCF'] = best_mindcf_asv_lid
    results['asv-lid_beta'] = best_b

    # Track all score arrays to find global best at the end
    all_score_sets = {
        'asv_only': asv_scores,
        'lid_only': lid_scores,
        'asv-lid': asv_scores - best_b * lid_scores,
    }

    # --- Fusion model embeddings ---
    if sub_dict is not None and add_dict is not None:
        sub_scores = compute_cosine_scores(sub_dict, enroll_ids, test_ids)
        add_scores = compute_cosine_scores(add_dict, enroll_ids, test_ids)

        # Individual
        sub_eer, _ = compute_eer(sub_scores[target_mask], sub_scores[~target_mask])
        sub_minDCF = compute_minDCF(sub_scores[target_mask], sub_scores[~target_mask])
        add_eer, _ = compute_eer(add_scores[target_mask], add_scores[~target_mask])
        add_minDCF = compute_minDCF(add_scores[target_mask], add_scores[~target_mask])
        print(f"\n  [Fusion]   Sub-only:       {sub_eer:.4f}% EER | {sub_minDCF:.4f}% minDCF")
        print(f"  [Fusion]   Add-only:       {add_eer:.4f}% EER | {add_minDCF:.4f}% minDCF")
        results['sub_only'] = sub_eer
        results['sub_only_minDCF'] = sub_minDCF
        results['add_only'] = add_eer
        results['add_only_minDCF'] = add_minDCF
        all_score_sets['sub_only'] = sub_scores
        all_score_sets['add_only'] = add_scores

        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # ASV + Sub
        best_eer, best_mindcf, best_a = 100.0, 100.0, 0.5
        for a in alphas:
            eer, _ = compute_eer(
                (a * asv_scores + (1 - a) * sub_scores)[target_mask],
                (a * asv_scores + (1 - a) * sub_scores)[~target_mask],
            )
            mindcf = compute_minDCF(
                (a * asv_scores + (1 - a) * sub_scores)[target_mask],
                (a * asv_scores + (1 - a) * sub_scores)[~target_mask],
            )
            if eer < best_eer:
                best_eer, best_a = eer, a
            if mindcf < best_mindcf:
                best_mindcf = mindcf
        results['asv+sub'] = best_eer
        results['asv+sub_minDCF'] = best_mindcf
        results['asv+sub_a'] = best_a
        all_score_sets['asv+sub'] = best_a * asv_scores + (1 - best_a) * sub_scores
        print(f"  [Fusion]   ASV+Sub:        {best_eer:.4f}% EER | {best_mindcf:.4f}% minDCF  (α={best_a:.1f})")

        # ASV + Add
        best_eer, best_mindcf, best_a = 100.0, 100.0, 0.5
        for a in alphas:
            eer, _ = compute_eer(
                (a * asv_scores + (1 - a) * add_scores)[target_mask],
                (a * asv_scores + (1 - a) * add_scores)[~target_mask],
            )
            mindcf = compute_minDCF(
                (a * asv_scores + (1 - a) * add_scores)[target_mask],
                (a * asv_scores + (1 - a) * add_scores)[~target_mask],
            )
            if eer < best_eer:
                best_eer, best_a = eer, a
            if mindcf < best_mindcf:
                best_mindcf = mindcf
        results['asv+add'] = best_eer
        results['asv+add_minDCF'] = best_mindcf
        results['asv+add_a'] = best_a
        all_score_sets['asv+add'] = best_a * asv_scores + (1 - best_a) * add_scores
        print(f"  [Fusion]   ASV+Add:        {best_eer:.4f}% EER | {best_mindcf:.4f}% minDCF  (α={best_a:.1f})")

        # Sub + Add
        best_eer, best_mindcf, best_a = 100.0, 100.0, 0.5
        for a in alphas:
            eer, _ = compute_eer(
                (a * sub_scores + (1 - a) * add_scores)[target_mask],
                (a * sub_scores + (1 - a) * add_scores)[~target_mask],
            )
            mindcf = compute_minDCF(
                (a * sub_scores + (1 - a) * add_scores)[target_mask],
                (a * sub_scores + (1 - a) * add_scores)[~target_mask],
            )
            if eer < best_eer:
                best_eer, best_a = eer, a
            if mindcf < best_mindcf:
                best_mindcf = mindcf
        results['sub+add'] = best_eer
        results['sub+add_minDCF'] = best_mindcf
        results['sub+add_a'] = best_a
        all_score_sets['sub+add'] = best_a * sub_scores + (1 - best_a) * add_scores
        print(f"  [Fusion]   Sub+Add:        {best_eer:.4f}% EER | {best_mindcf:.4f}% minDCF  (α={best_a:.1f})")

        # With LID subtraction
        # Sub - LID
        best_eer, best_mindcf, best_b = 100.0, 100.0, 0.1
        for b in betas:
            eer, _ = compute_eer(
                (sub_scores - b * lid_scores)[target_mask],
                (sub_scores - b * lid_scores)[~target_mask],
            )
            mindcf = compute_minDCF(
                (sub_scores - b * lid_scores)[target_mask],
                (sub_scores - b * lid_scores)[~target_mask],
            )
            if eer < best_eer:
                best_eer, best_b = eer, b
            if mindcf < best_mindcf:
                best_mindcf = mindcf
        results['sub-lid'] = best_eer
        results['sub-lid_minDCF'] = best_mindcf
        results['sub-lid_b'] = best_b
        all_score_sets['sub-lid'] = sub_scores - best_b * lid_scores
        print(f"  [Fusion]   Sub-LID:        {best_eer:.4f}% EER | {best_mindcf:.4f}% minDCF  (β={best_b:.2f})")

        # Add - LID
        best_eer, best_mindcf, best_b = 100.0, 100.0, 0.1
        for b in betas:
            eer, _ = compute_eer(
                (add_scores - b * lid_scores)[target_mask],
                (add_scores - b * lid_scores)[~target_mask],
            )
            mindcf = compute_minDCF(
                (add_scores - b * lid_scores)[target_mask],
                (add_scores - b * lid_scores)[~target_mask],
            )
            if eer < best_eer:
                best_eer, best_b = eer, b
            if mindcf < best_mindcf:
                best_mindcf = mindcf
        results['add-lid'] = best_eer
        results['add-lid_minDCF'] = best_mindcf
        results['add-lid_b'] = best_b
        all_score_sets['add-lid'] = add_scores - best_b * lid_scores
        print(f"  [Fusion]   Add-LID:        {best_eer:.4f}% EER | {best_mindcf:.4f}% minDCF  (β={best_b:.2f})")

        # Sub + Add - LID
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.5, 0.5, 0.1)
        for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for d in [0.05, 0.1, 0.15, 0.2, 0.25]:
                combined = a * sub_scores + (1 - a) * add_scores - d * lid_scores
                eer, _ = compute_eer(combined[target_mask], combined[~target_mask])
                mindcf = compute_minDCF(combined[target_mask], combined[~target_mask])
                if eer < best_eer:
                    best_eer, best_p = eer, (a, 1 - a, d)
                if mindcf < best_mindcf:
                    best_mindcf = mindcf
        results['sub+add-lid'] = best_eer
        results['sub+add-lid_minDCF'] = best_mindcf
        results['sub+add-lid_p'] = best_p
        all_score_sets['sub+add-lid'] = best_p[0] * sub_scores + best_p[1] * add_scores - best_p[2] * lid_scores
        print(f"  [Fusion]   Sub+Add-LID:    {best_eer:.4f}% EER | {best_mindcf:.4f}% minDCF  (α={best_p[0]:.1f}, β={best_p[1]:.1f}, δ={best_p[2]:.2f})")

        # ASV + Sub + Add
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.33, 0.33, 0.34)
        for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for b in [0.2, 0.3, 0.4, 0.5]:
                c = 1.0 - a - b
                if c > 0.05:
                    combined = a * asv_scores + b * sub_scores + c * add_scores
                    eer, _ = compute_eer(combined[target_mask], combined[~target_mask])
                    mindcf = compute_minDCF(combined[target_mask], combined[~target_mask])
                    if eer < best_eer:
                        best_eer, best_p = eer, (a, b, c)
                    if mindcf < best_mindcf:
                        best_mindcf = mindcf
        results['asv+sub+add'] = best_eer
        results['asv+sub+add_minDCF'] = best_mindcf
        results['asv+sub+add_p'] = best_p
        all_score_sets['asv+sub+add'] = best_p[0] * asv_scores + best_p[1] * sub_scores + best_p[2] * add_scores
        print(f"  [Fusion]   ASV+Sub+Add:    {best_eer:.4f}% EER | {best_mindcf:.4f}% minDCF  ({best_p[0]:.1f},{best_p[1]:.1f},{best_p[2]:.1f})")

        # Full: α*ASV + β*Sub + γ*Add - δ*LID
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.3, 0.4, 0.3, 0.1)
        for a in [0.1, 0.2, 0.3, 0.4]:
            for b in [0.2, 0.3, 0.4, 0.5, 0.6]:
                for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    for d in [0.05, 0.1, 0.15, 0.2]:
                        combined = a * asv_scores + b * sub_scores + c * add_scores - d * lid_scores
                        eer, _ = compute_eer(combined[target_mask], combined[~target_mask])
                        mindcf = compute_minDCF(combined[target_mask], combined[~target_mask])
                        if eer < best_eer:
                            best_eer, best_p = eer, (a, b, c, d)
                        if mindcf < best_mindcf:
                            best_mindcf = mindcf
        results['full'] = best_eer
        results['full_minDCF'] = best_mindcf
        results['full_params'] = best_p
        all_score_sets['full'] = best_p[0] * asv_scores + best_p[1] * sub_scores + best_p[2] * add_scores - best_p[3] * lid_scores
        print(f"  [Fusion]   Full:           {best_eer:.4f}% EER | {best_mindcf:.4f}% minDCF  (α={best_p[0]:.1f}*ASV + β={best_p[1]:.1f}*Sub + γ={best_p[2]:.1f}*Add - δ={best_p[3]:.2f}*LID)")

        # Summary
        strategies = ['asv_only', 'asv-lid', 'sub_only', 'add_only',
                      'asv+sub', 'asv+add', 'sub+add',
                      'sub-lid', 'add-lid', 'sub+add-lid',
                      'asv+sub+add', 'full']
        best_strategy = min(strategies, key=lambda x: results[x])
        print(f"\n  ★ Best: {best_strategy} = {results[best_strategy]:.4f}%")

    # Save only the best strategy's scores
    if output_dir:
        # Find overall best among all evaluated strategies
        scored_strategies = [k for k in all_score_sets]
        best_strategy = min(scored_strategies, key=lambda x: results[x])
        best_scores = all_score_sets[best_strategy]
        out_path = os.path.join(scores_dir, f'{task_name}_best_{best_strategy}.txt')
        save_scores_file(out_path, enroll_ids, test_ids, labels, best_scores)
        print(f"\n  Saved best scores ({best_strategy}, EER={results[best_strategy]:.4f}%) → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Dual-Path Fusion V12 on TidyVoiceX2_ASV trials'
    )
    parser.add_argument('--emb_base', type=str, required=True,
                        help='Base dir containing stage1_asv/ and stage2_lid/')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Path to trained best_model.pt (skip for baseline-only)')
    parser.add_argument('--trial_files', type=str, nargs='+', required=True,
                        help='One or more trial files (task1_labels.txt task2_labels.txt)')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--subspace_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Optional dir to save results JSON')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load flat embeddings ──────────────────────────────────────────────
    asv_dev_dir = os.path.join(args.emb_base, 'stage1_asv', 'dev')
    lid_dev_dir = os.path.join(args.emb_base, 'stage2_lid', 'dev')

    print("\nLoading ASV embeddings...")
    asv_dict = load_flat_embeddings(asv_dev_dir)
    print("Loading LID embeddings...")
    lid_dict = load_flat_embeddings(lid_dev_dir)

    available_keys = set(asv_dict.keys()) & set(lid_dict.keys())
    print(f"\nCommon embeddings: {len(available_keys)}")

    # ── Optional: load fusion model and extract Sub/Add embeddings ────────
    sub_dict = None
    add_dict = None

    if args.model_checkpoint and os.path.isfile(args.model_checkpoint):
        print(f"\nLoading fusion model: {args.model_checkpoint}")
        ckpt = torch.load(args.model_checkpoint, map_location=device, weights_only=False)

        num_speakers = ckpt.get('num_speakers', 3666)
        num_languages = ckpt.get('num_languages', 40)

        model = DualPathModel(
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            subspace_dim=args.subspace_dim,
            num_heads=args.num_heads,
            num_speakers=num_speakers,
            num_languages=num_languages,
            dropout=args.dropout,
        ).to(device)

        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print(f"  Loaded epoch {ckpt.get('epoch', '?')}, "
              f"train best_eer={ckpt.get('best_eer', '?'):.4f}% "
              f"({ckpt.get('best_strategy', '?')})")

        # Build ordered arrays for batch extraction
        ordered_keys = sorted(available_keys)
        key_to_idx = {k: i for i, k in enumerate(ordered_keys)}

        asv_arr = np.stack([asv_dict[k] for k in ordered_keys])
        lid_arr = np.stack([lid_dict[k] for k in ordered_keys])

        # L2 normalize (same as EvalDataset in training code)
        asv_arr = asv_arr / np.clip(np.linalg.norm(asv_arr, axis=1, keepdims=True), 1e-8, None)
        lid_arr = lid_arr / np.clip(np.linalg.norm(lid_arr, axis=1, keepdims=True), 1e-8, None)

        print("\nExtracting Sub/Add embeddings through fusion model...")
        sub_arr, add_arr = extract_fusion_embeddings(model, asv_arr, lid_arr, device)

        # Convert back to dicts
        sub_dict = {k: sub_arr[key_to_idx[k]] for k in ordered_keys}
        add_dict = {k: add_arr[key_to_idx[k]] for k in ordered_keys}
        print(f"  Sub embeddings: {sub_arr.shape}, Add embeddings: {add_arr.shape}")

        del asv_arr, lid_arr, sub_arr, add_arr
    else:
        if args.model_checkpoint:
            print(f"\nWARNING: Model checkpoint not found: {args.model_checkpoint}")
            print("  Running baseline-only evaluation.")
        else:
            print("\nNo model checkpoint provided — baseline-only evaluation.")

    # ── Evaluate each trial file ──────────────────────────────────────────
    output_dir = args.output_dir or os.path.dirname(args.trial_files[0])
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for trial_file in args.trial_files:
        task_name = os.path.splitext(os.path.basename(trial_file))[0]
        print(f"\nParsing trials: {trial_file}")
        enroll_ids, test_ids, labels = parse_trials(trial_file, available_keys)

        if len(labels) == 0:
            print(f"  WARNING: No valid trials found for {task_name}, skipping.")
            continue

        results = evaluate_trial(
            task_name, enroll_ids, test_ids, labels,
            asv_dict, lid_dict, sub_dict, add_dict,
            output_dir=output_dir,
        )
        all_results[task_name] = results

    # ── Save results ──────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, 'eval_results.json')

    # Convert numpy/tuple values
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    serializable = {
        task: {k: to_serializable(v) for k, v in r.items()}
        for task, r in all_results.items()
    }
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n\nResults saved to {results_path}")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for task, r in all_results.items():
        print(f"\n  {task}:")
        print(f"    Baseline ASV:      {r['asv_only']:.4f}% EER | {r['asv_only_minDCF']:.4f}% minDCF")
        print(f"    Baseline LID:      {r['lid_only']:.4f}% EER | {r['lid_only_minDCF']:.4f}% minDCF")
        print(f"    Baseline ASV-LID:  {r['asv-lid']:.4f}% EER | {r['asv-lid_minDCF']:.4f}% minDCF")
        if 'full' in r:
            eer_strategies = [k for k in r if isinstance(r[k], (int, float)) and '_' not in k and k not in ('asv-lid_beta', 'asv+sub_a', 'asv+add_a', 'sub+add_a', 'sub-lid_b', 'add-lid_b', 'sub+add-lid_p', 'asv+sub+add_p', 'full_params')]
            best_eer_val = min(r[k] for k in eer_strategies if k in r)
            best_minDCF_val = min(r.get(k + '_minDCF', 100.0) for k in eer_strategies if k + '_minDCF' in r)
            print(f"    Fusion best EER:   {best_eer_val:.4f}%")
            print(f"    Fusion best minDCF: {best_minDCF_val:.4f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
