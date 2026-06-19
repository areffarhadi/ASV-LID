#!/usr/bin/env python3
"""
Evaluate Both Trained Dual-Path Fusion V12 Models with minDCF Metrics
======================================================================

This script evaluates two pre-trained models on their respective datasets:

1. Model 1 (TidyVoice):
   - Loads: multilingual_embeddings_stages_tidyvoice/ckpt_fusion_v12_dual_path/best_model.pt
   - Data: Converted .npz embeddings (dev_asv.npz, dev_lid.npz)
   - Trials: test_4types_trials_dev2.txt
   - Output: Results with minDCF metrics

2. Model 2 (TidyVoiceX2_ASV):
   - Loads: multilingual_embeddings_stages_tidyvoice/ckpt_fusion_v12_dual_path/best_model.pt
   - Data: Flat .npy embeddings from multilingual_embeddings_stages_tidyvoicex2_asv2
   - Trials: task1_labels.txt, task2_labels.txt
   - Output: Results with minDCF metrics

Usage:
    python evaluate_both_models_mindcf.py [options]

Options:
    --gpu GPU_ID              GPU device ID (default: 0)
    --output_dir DIR          Output directory for results (default: ./eval_results_mindcf)
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_fusion_dual_path_v12 import DualPathModel, compute_eer


def compute_minDCF(target_scores, nontarget_scores, ptar=0.01, cmiss=1, cfa=1):
    """
    Compute normalized minimum Detection Cost Function (Wespeaker implementation).
    
    Formula:
    - C_det = min(cmiss * FNR * ptar + cfa * FPR * (1-ptar))
    - C_def = min(cmiss * ptar, cfa * (1-ptar))
    - minDCF = C_det / C_def
    
    Args:
        target_scores: Scores of target trials
        nontarget_scores: Scores of nontarget (impostor) trials
        ptar: Prior probability of target trials (default 0.01 for speaker verification)
        cmiss: Cost of false rejection (default 1)
        cfa: Cost of false acceptance (default 1)
    
    Returns:
        Normalized minDCF value (typically 0-1)
    """
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return 1.0
    
    # Combine scores and labels
    scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([np.ones(len(target_scores)), np.zeros(len(nontarget_scores))])
    
    # Sort by score in ascending order
    sorted_idx = np.argsort(scores)
    labels_sorted = labels[sorted_idx]
    
    # Compute weights for targets and imposters
    tgt_wghts = (labels_sorted == 1).astype('f8')
    imp_wghts = (labels_sorted == 0).astype('f8')
    
    # Compute FNR and FPR curves (Wespeaker's approach)
    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1.0 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    
    # Compute detection cost at each threshold
    c_det = cmiss * fnr * ptar + cfa * fpr * (1 - ptar)
    
    # Compute minimum detection cost
    min_c_det = np.min(c_det)
    
    # Compute default cost (cost of always rejecting or always accepting)
    c_def = min(cmiss * ptar, cfa * (1 - ptar))
    
    # Return normalized minDCF
    return min_c_det / c_def


def compute_cosine_scores(emb_dict, enroll_ids, test_ids, batch_size=500000):
    """Compute cosine similarity scores for all trial pairs"""
    n = len(enroll_ids)
    scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        enroll = np.stack([emb_dict[enroll_ids[i]] for i in range(start, end)])
        test = np.stack([emb_dict[test_ids[i]] for i in range(start, end)])
        enroll = enroll / np.clip(np.linalg.norm(enroll, axis=1, keepdims=True), 1e-8, None)
        test = test / np.clip(np.linalg.norm(test, axis=1, keepdims=True), 1e-8, None)
        scores[start:end] = np.sum(enroll * test, axis=1)
    return scores


def load_npz_embeddings(npz_file):
    """Load embeddings from NPZ file into dict"""
    data = np.load(npz_file, allow_pickle=True)
    
    # Handle two possible NPZ formats
    if 'embeddings' in data.files and 'filenames' in data.files:
        # Format: structured NPZ with separate embeddings and filenames arrays
        embeddings = data['embeddings'].astype(np.float32)
        filenames = data['filenames']
        emb_dict = {}
        for filename, embedding in zip(filenames, embeddings):
            # Handle both string and bytes filenames
            if isinstance(filename, bytes):
                filename = filename.decode('utf-8')
            emb_dict[filename] = embedding
        return emb_dict
    else:
        # Format: flat NPZ where each key is an embedding
        emb_dict = {}
        for key in data.files:
            val = data[key]
            if isinstance(val, np.ndarray) and val.dtype != object:
                emb_dict[key] = val.astype(np.float32)
        return emb_dict


def parse_trials_npz(trial_file, available_keys):
    """Parse trial file and return enroll_ids, test_ids, labels"""
    enroll_ids, test_ids, labels = [], [], []
    skipped = 0
    with open(trial_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label_str, enroll_path, test_path = parts
            
            # Extract filename (without extension) from path
            # Path format: id014463/en/en_36069944.wav -> en_36069944
            enroll_fn = os.path.splitext(os.path.basename(enroll_path))[0]
            test_fn = os.path.splitext(os.path.basename(test_path))[0]
            
            if enroll_fn not in available_keys or test_fn not in available_keys:
                skipped += 1
                continue
            enroll_ids.append(enroll_fn)
            test_ids.append(test_fn)
            labels.append(1 if label_str == '1' else 0)
    
    labels = np.array(labels, dtype=np.int32)
    print(f"  Loaded {len(labels)} trials ({labels.sum()} target, {(1-labels).sum()} nontarget)")
    if skipped > 0:
        print(f"  Skipped {skipped} trials (embedding not found)")
    return enroll_ids, test_ids, labels


def load_flat_embeddings(emb_dir):
    """Load all .npy files from a flat directory into a dict"""
    emb_dict = {}
    npy_files = sorted(f for f in os.listdir(emb_dir) if f.endswith('.npy'))
    print(f"  Loading {len(npy_files)} embeddings from {emb_dir}")
    for f in tqdm(npy_files, desc="  Loading", leave=False):
        key = os.path.splitext(f)[0]
        emb_dict[key] = np.load(os.path.join(emb_dir, f)).astype(np.float32)
    return emb_dict


def parse_trials_flat(trial_file, available_keys):
    """Parse trial file for flat embeddings"""
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


def extract_fusion_embeddings(model, asv_arr, lid_arr, device, batch_size=512):
    """Extract Sub and Add embeddings through fusion model"""
    model.eval()
    sub_list, add_list = [], []
    n = len(asv_arr)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            asv = torch.tensor(asv_arr[i:end]).to(device)
            lid = torch.tensor(lid_arr[i:end]).to(device)
            sub_list.append(model.extract_sub_embedding(asv, lid).cpu().numpy())
            add_list.append(model.extract_add_embedding(asv, lid).cpu().numpy())
    return np.vstack(sub_list), np.vstack(add_list)


def report_metrics(task_name, enroll_ids, test_ids, labels, asv_scores, lid_scores, 
                   sub_scores=None, add_scores=None):
    """Compute and report both EER and minDCF metrics"""
    target_mask = labels == 1
    results = {}
    
    print(f"\n{'='*70}")
    print(f"  {task_name}")
    print(f"{'='*70}")
    
    # Baseline metrics
    asv_eer, _ = compute_eer(asv_scores[target_mask], asv_scores[~target_mask])
    asv_mindcf = compute_minDCF(asv_scores[target_mask], asv_scores[~target_mask])
    lid_eer, _ = compute_eer(lid_scores[target_mask], lid_scores[~target_mask])
    lid_mindcf = compute_minDCF(lid_scores[target_mask], lid_scores[~target_mask])
    
    print(f"\n  [Baseline] ASV-only:        EER={asv_eer:.4f}%  minDCF={asv_mindcf:.4f}")
    print(f"  [Baseline] LID-only:        EER={lid_eer:.4f}%  minDCF={lid_mindcf:.4f}")
    results['asv_only_eer'] = asv_eer
    results['asv_only_minDCF'] = asv_mindcf
    results['lid_only_eer'] = lid_eer
    results['lid_only_minDCF'] = lid_mindcf
    
    # ASV - LID
    best_eer_asv_lid, best_mindcf_asv_lid, best_b = 100.0, 100.0, 0.1
    for b in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        eer, _ = compute_eer(
            (asv_scores - b * lid_scores)[target_mask],
            (asv_scores - b * lid_scores)[~target_mask],
        )
        mindcf = compute_minDCF(
            (asv_scores - b * lid_scores)[target_mask],
            (asv_scores - b * lid_scores)[~target_mask],
        )
        if mindcf < best_mindcf_asv_lid:
            best_eer_asv_lid, best_mindcf_asv_lid, best_b = eer, mindcf, b
    print(f"  [Baseline] ASV-LID:         EER={best_eer_asv_lid:.4f}%  minDCF={best_mindcf_asv_lid:.4f}  (β={best_b:.2f})")
    results['asv-lid_eer'] = best_eer_asv_lid
    results['asv-lid_minDCF'] = best_mindcf_asv_lid
    results['asv-lid_beta'] = best_b
    
    # Fusion metrics (if available)
    if sub_scores is not None and add_scores is not None:
        sub_eer, _ = compute_eer(sub_scores[target_mask], sub_scores[~target_mask])
        sub_mindcf = compute_minDCF(sub_scores[target_mask], sub_scores[~target_mask])
        add_eer, _ = compute_eer(add_scores[target_mask], add_scores[~target_mask])
        add_mindcf = compute_minDCF(add_scores[target_mask], add_scores[~target_mask])
        print(f"\n  [Fusion]   Sub-only:         EER={sub_eer:.4f}%  minDCF={sub_mindcf:.4f}")
        print(f"  [Fusion]   Add-only:         EER={add_eer:.4f}%  minDCF={add_mindcf:.4f}")
        results['sub_only_eer'] = sub_eer
        results['sub_only_minDCF'] = sub_mindcf
        results['add_only_eer'] = add_eer
        results['add_only_minDCF'] = add_mindcf
        
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
            if mindcf < best_mindcf:
                best_eer, best_mindcf, best_a = eer, mindcf, a
        results['asv+sub_eer'] = best_eer
        results['asv+sub_minDCF'] = best_mindcf
        results['asv+sub_a'] = best_a
        print(f"  [Fusion]   ASV+Sub:          EER={best_eer:.4f}%  minDCF={best_mindcf:.4f}  (α={best_a:.1f})")
        
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
            if mindcf < best_mindcf:
                best_eer, best_mindcf, best_a = eer, mindcf, a
        results['asv+add_eer'] = best_eer
        results['asv+add_minDCF'] = best_mindcf
        results['asv+add_a'] = best_a
        print(f"  [Fusion]   ASV+Add:          EER={best_eer:.4f}%  minDCF={best_mindcf:.4f}  (α={best_a:.1f})")
        
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
            if mindcf < best_mindcf:
                best_eer, best_mindcf, best_a = eer, mindcf, a
        results['sub+add_eer'] = best_eer
        results['sub+add_minDCF'] = best_mindcf
        results['sub+add_a'] = best_a
        print(f"  [Fusion]   Sub+Add:          EER={best_eer:.4f}%  minDCF={best_mindcf:.4f}  (α={best_a:.1f})")
        
        # Sub - LID
        best_eer, best_mindcf, best_b = 100.0, 100.0, 0.1
        for b in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            eer, _ = compute_eer(
                (sub_scores - b * lid_scores)[target_mask],
                (sub_scores - b * lid_scores)[~target_mask],
            )
            mindcf = compute_minDCF(
                (sub_scores - b * lid_scores)[target_mask],
                (sub_scores - b * lid_scores)[~target_mask],
            )
            if mindcf < best_mindcf:
                best_eer, best_mindcf, best_b = eer, mindcf, b
        results['sub-lid_eer'] = best_eer
        results['sub-lid_minDCF'] = best_mindcf
        results['sub-lid_b'] = best_b
        print(f"  [Fusion]   Sub-LID:          EER={best_eer:.4f}%  minDCF={best_mindcf:.4f}  (β={best_b:.2f})")
        
        # Add - LID
        best_eer, best_mindcf, best_b = 100.0, 100.0, 0.1
        for b in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            eer, _ = compute_eer(
                (add_scores - b * lid_scores)[target_mask],
                (add_scores - b * lid_scores)[~target_mask],
            )
            mindcf = compute_minDCF(
                (add_scores - b * lid_scores)[target_mask],
                (add_scores - b * lid_scores)[~target_mask],
            )
            if mindcf < best_mindcf:
                best_eer, best_mindcf, best_b = eer, mindcf, b
        results['add-lid_eer'] = best_eer
        results['add-lid_minDCF'] = best_mindcf
        results['add-lid_b'] = best_b
        print(f"  [Fusion]   Add-LID:          EER={best_eer:.4f}%  minDCF={best_mindcf:.4f}  (β={best_b:.2f})")
        
        # Sub + Add - LID
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.5, 0.5, 0.1)
        for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for d in [0.05, 0.1, 0.15, 0.2, 0.25]:
                combined = a * sub_scores + (1 - a) * add_scores - d * lid_scores
                eer, _ = compute_eer(combined[target_mask], combined[~target_mask])
                mindcf = compute_minDCF(combined[target_mask], combined[~target_mask])
                if mindcf < best_mindcf:
                    best_eer, best_mindcf, best_p = eer, mindcf, (a, 1 - a, d)
        results['sub+add-lid_eer'] = best_eer
        results['sub+add-lid_minDCF'] = best_mindcf
        results['sub+add-lid_p'] = best_p
        print(f"  [Fusion]   Sub+Add-LID:      EER={best_eer:.4f}%  minDCF={best_mindcf:.4f}  (α={best_p[0]:.1f}, β={best_p[1]:.1f}, δ={best_p[2]:.2f})")
        
        # ASV + Sub + Add
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.33, 0.33, 0.34)
        for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for b in [0.2, 0.3, 0.4, 0.5]:
                c = 1.0 - a - b
                if c > 0.05:
                    combined = a * asv_scores + b * sub_scores + c * add_scores
                    eer, _ = compute_eer(combined[target_mask], combined[~target_mask])
                    mindcf = compute_minDCF(combined[target_mask], combined[~target_mask])
                    if mindcf < best_mindcf:
                        best_eer, best_mindcf, best_p = eer, mindcf, (a, b, c)
        results['asv+sub+add_eer'] = best_eer
        results['asv+sub+add_minDCF'] = best_mindcf
        results['asv+sub+add_p'] = best_p
        print(f"  [Fusion]   ASV+Sub+Add:      EER={best_eer:.4f}%  minDCF={best_mindcf:.4f}  ({best_p[0]:.1f},{best_p[1]:.1f},{best_p[2]:.1f})")
        
        # Full: α*ASV + β*Sub + γ*Add - δ*LID
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.3, 0.4, 0.3, 0.1)
        for a in [0.1, 0.2, 0.3, 0.4]:
            for b in [0.2, 0.3, 0.4, 0.5, 0.6]:
                for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    for d in [0.05, 0.1, 0.15, 0.2]:
                        combined = a * asv_scores + b * sub_scores + c * add_scores - d * lid_scores
                        eer, _ = compute_eer(combined[target_mask], combined[~target_mask])
                        mindcf = compute_minDCF(combined[target_mask], combined[~target_mask])
                        if mindcf < best_mindcf:
                            best_eer, best_mindcf, best_p = eer, mindcf, (a, b, c, d)
        results['full_eer'] = best_eer
        results['full_minDCF'] = best_mindcf
        results['full_params'] = best_p
        print(f"  [Fusion]   Full:              EER={best_eer:.4f}%  minDCF={best_mindcf:.4f}  (α={best_p[0]:.1f}*ASV + β={best_p[1]:.1f}*Sub + γ={best_p[2]:.1f}*Add - δ={best_p[3]:.2f}*LID)")
        
        # Find best
        fusion_strategies = ['sub_only_minDCF', 'add_only_minDCF', 'asv+sub_minDCF', 'asv+add_minDCF', 
                           'sub+add_minDCF', 'sub-lid_minDCF', 'add-lid_minDCF', 'sub+add-lid_minDCF',
                           'asv+sub+add_minDCF', 'full_minDCF']
        best_fusion = min(fusion_strategies, key=lambda x: results[x])
        print(f"\n  ★ Best Fusion: {best_fusion} = {results[best_fusion]:.4f}%")
    
    return results


def evaluate_model_1_tidyvoice(device, output_dir):
    """Evaluate Model 1 (TidyVoice) with NPZ embeddings"""
    print(f"\n{'='*70}")
    print("EVALUATING MODEL 1: TidyVoice (NPZ Embeddings)")
    print(f"{'='*70}")
    
    model_path = os.environ.get("CKPT_FUSION_ASV", os.path.join(os.environ.get("REPO_ROOT", "."), "checkpoints/fusion_asv.pt"))
    eval_asv_file = os.environ.get("EVAL_ASV_NPZ", os.path.join(os.environ.get("EMB_TIDYVOICE", ""), "converted_npz/dev_asv.npz"))
    eval_lid_file = os.environ.get("EVAL_LID_NPZ", os.path.join(os.environ.get("EMB_TIDYVOICE", ""), "converted_npz/dev_lid.npz"))
    trial_file = os.environ.get("ASV_TRIAL_DEV2", "")
    
    # Validate files
    if not all([os.path.isfile(f) for f in [model_path, eval_asv_file, eval_lid_file, trial_file]]):
        print(f"ERROR: Missing required files for Model 1")
        return None
    
    print("\nLoading embeddings...")
    asv_embs = load_npz_embeddings(eval_asv_file)
    lid_embs = load_npz_embeddings(eval_lid_file)
    available_keys = set(asv_embs.keys()) & set(lid_embs.keys())
    print(f"Common embeddings: {len(available_keys)}")
    
    print(f"\nParsing trials: {trial_file}")
    enroll_ids, test_ids, labels = parse_trials_npz(trial_file, available_keys)
    
    if len(labels) == 0:
        print("ERROR: No valid trials found")
        return None
    
    # Compute baseline scores
    asv_scores = compute_cosine_scores(asv_embs, enroll_ids, test_ids)
    lid_scores = compute_cosine_scores(lid_embs, enroll_ids, test_ids)
    
    # Load model and extract fusion embeddings
    print(f"\nLoading model: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    num_speakers = ckpt.get('num_speakers', 3666)
    num_languages = ckpt.get('num_languages', 40)
    
    model = DualPathModel(
        embed_dim=256,
        hidden_dim=512,
        subspace_dim=64,
        num_heads=4,
        num_speakers=num_speakers,
        num_languages=num_languages,
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print(f"Loaded epoch {ckpt.get('epoch', '?')}, best_eer={ckpt.get('best_eer', '?'):.4f}%")
    
    # Build ordered arrays
    ordered_keys = sorted(available_keys)
    asv_arr = np.stack([asv_embs[k] for k in ordered_keys])
    lid_arr = np.stack([lid_embs[k] for k in ordered_keys])
    
    # L2 normalize
    asv_arr = asv_arr / np.clip(np.linalg.norm(asv_arr, axis=1, keepdims=True), 1e-8, None)
    lid_arr = lid_arr / np.clip(np.linalg.norm(lid_arr, axis=1, keepdims=True), 1e-8, None)
    
    print("\nExtracting Sub/Add embeddings...")
    sub_arr, add_arr = extract_fusion_embeddings(model, asv_arr, lid_arr, device)
    
    # Build dicts for score computation
    key_to_idx = {k: i for i, k in enumerate(ordered_keys)}
    sub_embs = {k: sub_arr[key_to_idx[k]] for k in ordered_keys}
    add_embs = {k: add_arr[key_to_idx[k]] for k in ordered_keys}
    
    sub_scores = compute_cosine_scores(sub_embs, enroll_ids, test_ids)
    add_scores = compute_cosine_scores(add_embs, enroll_ids, test_ids)
    
    # Report metrics
    results = report_metrics("TidyVoice (dev)", enroll_ids, test_ids, labels, 
                           asv_scores, lid_scores, sub_scores, add_scores)
    
    del asv_embs, lid_embs, asv_arr, lid_arr, sub_arr, add_arr, model
    torch.cuda.empty_cache()
    
    return results


def evaluate_model_2_tidyvoicex2(device, output_dir):
    """Evaluate Model 2 (TidyVoiceX2_ASV) with flat .npy embeddings"""
    print(f"\n{'='*70}")
    print("EVALUATING MODEL 2: TidyVoiceX2_ASV (Flat embeddings)")
    print(f"{'='*70}")
    
    model_path = os.environ.get("CKPT_FUSION_ASV", os.path.join(os.environ.get("REPO_ROOT", "."), "checkpoints/fusion_asv.pt"))
    emb_base = os.environ.get("EMB_TIDYVOICEX2", "")
    asv_dev_dir = os.path.join(emb_base, 'stage1_asv', 'dev')
    lid_dev_dir = os.path.join(emb_base, 'stage2_lid', 'dev')
    
    trial_files = [
        os.environ.get("ASV_TASK1", ""),
        os.environ.get("ASV_TASK2", ""),
    ]
    
    # Validate files
    if not os.path.isdir(asv_dev_dir) or not os.path.isdir(lid_dev_dir) or not os.path.isfile(model_path):
        print(f"ERROR: Missing required files/directories for Model 2")
        return {}
    
    print("\nLoading embeddings...")
    asv_dict = load_flat_embeddings(asv_dev_dir)
    lid_dict = load_flat_embeddings(lid_dev_dir)
    available_keys = set(asv_dict.keys()) & set(lid_dict.keys())
    print(f"Common embeddings: {len(available_keys)}")
    
    # Load model
    print(f"\nLoading model: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    num_speakers = ckpt.get('num_speakers', 3666)
    num_languages = ckpt.get('num_languages', 40)
    
    model = DualPathModel(
        embed_dim=256,
        hidden_dim=512,
        subspace_dim=64,
        num_heads=4,
        num_speakers=num_speakers,
        num_languages=num_languages,
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print(f"Loaded epoch {ckpt.get('epoch', '?')}, best_eer={ckpt.get('best_eer', '?'):.4f}%")
    
    # Build ordered arrays
    ordered_keys = sorted(available_keys)
    key_to_idx = {k: i for i, k in enumerate(ordered_keys)}
    
    asv_arr = np.stack([asv_dict[k] for k in ordered_keys])
    lid_arr = np.stack([lid_dict[k] for k in ordered_keys])
    
    # L2 normalize
    asv_arr = asv_arr / np.clip(np.linalg.norm(asv_arr, axis=1, keepdims=True), 1e-8, None)
    lid_arr = lid_arr / np.clip(np.linalg.norm(lid_arr, axis=1, keepdims=True), 1e-8, None)
    
    print("\nExtracting Sub/Add embeddings...")
    sub_arr, add_arr = extract_fusion_embeddings(model, asv_arr, lid_arr, device)
    
    # Evaluate on all trial files
    all_results = {}
    for trial_file in trial_files:
        if not os.path.isfile(trial_file):
            print(f"Skipping {trial_file} (not found)")
            continue
        
        task_name = os.path.splitext(os.path.basename(trial_file))[0]
        print(f"\nParsing trials: {trial_file}")
        enroll_ids, test_ids, labels = parse_trials_flat(trial_file, available_keys)
        
        if len(labels) == 0:
            print(f"No valid trials for {task_name}")
            continue
        
        # Compute scores
        asv_scores = compute_cosine_scores(asv_dict, enroll_ids, test_ids)
        lid_scores = compute_cosine_scores(lid_dict, enroll_ids, test_ids)
        
        sub_embs = {k: sub_arr[key_to_idx[k]] for k in ordered_keys}
        add_embs = {k: add_arr[key_to_idx[k]] for k in ordered_keys}
        sub_scores = compute_cosine_scores(sub_embs, enroll_ids, test_ids)
        add_scores = compute_cosine_scores(add_embs, enroll_ids, test_ids)
        
        # Report metrics
        results = report_metrics(task_name, enroll_ids, test_ids, labels, 
                               asv_scores, lid_scores, sub_scores, add_scores)
        all_results[task_name] = results
    
    del asv_dict, lid_dict, asv_arr, lid_arr, sub_arr, add_arr, model
    torch.cuda.empty_cache()
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Both Trained Models with minDCF')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--output_dir', type=str, default='./eval_results_mindcf',
                       help='Output directory for results')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    # Evaluate Model 1
    model1_results = evaluate_model_1_tidyvoice(device, args.output_dir)
    if model1_results is not None:
        all_results['tidyvoice_dev'] = model1_results
    
    # Evaluate Model 2
    model2_results = evaluate_model_2_tidyvoicex2(device, args.output_dir)
    if model2_results:
        all_results.update(model2_results)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results_mindcf.json')
    
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
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - All minDCF Results")
    print(f"{'='*70}")
    for task, r in all_results.items():
        print(f"\n{task}:")
        print(f"  Baseline ASV-only:    {r['asv_only_minDCF']:.4f}%")
        print(f"  Baseline LID-only:    {r['lid_only_minDCF']:.4f}%")
        print(f"  Baseline ASV-LID:     {r['asv-lid_minDCF']:.4f}%")
        if 'full_minDCF' in r:
            print(f"  Fusion best (Full):   {r['full_minDCF']:.4f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
