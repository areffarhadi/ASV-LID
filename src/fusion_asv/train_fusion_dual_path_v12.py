"""
Dual-Path Fusion V12 - Subtractive + Additive + Score Fusion
==============================================================

This version trains a SINGLE model with TWO independent paths:

  Path A (Subtractive - from V6):
    speaker_emb ──► Orthogonal Projection ──► clean_emb (language removed)
  
  Path B (Additive - from V11):
    speaker_emb ──► Q ─┐
    language_emb ──► K,V ┴► CrossAttention ──► enriched_emb (speaker info added)

At evaluation, we have 4 sets of scores:
  1. ASV scores    (original speaker embedding)
  2. LID scores    (original language embedding)
  3. Sub scores    (subtractive path - language removed)
  4. Add scores    (additive path - speaker enriched)

Score fusion searches over all combinations:
  - Individual: ASV, Sub, Add
  - Pairs: ASV+Sub, ASV+Add, Sub+Add
  - Triple: ASV+Sub+Add
  - With LID subtraction: all of the above minus LID
  - Full: α*ASV + β*Sub + γ*Add - δ*LID

Why this works:
  - Subtractive path removes language contamination
  - Additive path extracts complementary speaker info from LID
  - These capture DIFFERENT aspects of speaker identity
  - Score fusion combines their complementary strengths

Usage:
    python train_fusion_dual_path_v12.py \
        --train_asv_emb /path/to/train_embeddings_asv.npz \
        --train_lang_emb /path/to/train_embeddings_language.npz \
        --eval_asv_emb /path/to/embeddings.npz \
        --eval_lang_emb /path/to/embeddings_language.npz \
        --trial_file /path/to/trials.txt \
        --gpu 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
import random
from collections import defaultdict
from itertools import product
import warnings


# ============================================================================
# Path A: Subtractive (Orthogonal Projection from V6)
# ============================================================================

class OrthogonalProjection(nn.Module):
    """Remove language subspace from speaker embedding"""
    
    def __init__(self, embed_dim=256, subspace_dim=64, num_languages=40):
        super().__init__()
        
        self.basis = nn.Parameter(torch.randn(subspace_dim, embed_dim))
        
        self.lang_to_subspace = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.lang_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_languages)
        )
    
    def forward(self, speaker_emb, language_emb):
        basis_norm = F.normalize(self.basis, p=2, dim=1, eps=1e-8)
        proj_coeffs = torch.mm(speaker_emb, basis_norm.t())
        proj_basis = torch.mm(proj_coeffs, basis_norm)
        
        lang_direction = self.lang_to_subspace(language_emb)
        lang_direction = F.normalize(lang_direction, p=2, dim=1, eps=1e-8)
        proj_coeff_lang = (speaker_emb * lang_direction).sum(dim=1, keepdim=True)
        proj_lang = proj_coeff_lang * lang_direction
        
        alpha = torch.sigmoid(self.alpha)
        removed = alpha * proj_basis + (1 - alpha) * proj_lang
        clean = speaker_emb - removed
        
        return clean, removed
    
    def predict_language(self, removed):
        return self.lang_classifier(removed)


# ============================================================================
# Path B: Additive (Speaker Query Attention from V11)
# ============================================================================

class SpeakerQueryAttention(nn.Module):
    """Speaker queries Language for speaker-relevant info"""
    
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, speaker_emb, language_emb):
        B = speaker_emb.size(0)
        
        Q = self.W_q(speaker_emb).view(B, self.num_heads, self.head_dim)
        K = self.W_k(language_emb).view(B, self.num_heads, self.head_dim)
        V = self.W_v(language_emb).view(B, self.num_heads, self.head_dim)
        
        scale = math.sqrt(self.head_dim) * F.softplus(self.temperature)
        attn_scores = (Q * K).sum(dim=-1) / scale
        attn_weights = torch.sigmoid(attn_scores)
        
        attended = attn_weights.unsqueeze(-1) * V
        attended = attended.reshape(B, self.embed_dim)
        
        extracted = self.out_proj(self.dropout(attended))
        extracted = self.layer_norm(extracted)
        
        return extracted, attn_weights


class GatedAdditiveFusion(nn.Module):
    """Gated addition: output = speaker + gate * extracted"""
    
    def __init__(self, embed_dim=256):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        self.add_scale = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, speaker_emb, extracted_info):
        combined = torch.cat([speaker_emb, extracted_info], dim=-1)
        gate = self.gate_net(combined)
        scale = torch.sigmoid(self.add_scale)
        enriched = speaker_emb + scale * gate * extracted_info
        return enriched, gate


# ============================================================================
# Full Model: Dual-Path
# ============================================================================

class DualPathModel(nn.Module):
    """
    Dual-Path Model with Subtractive AND Additive paths.
    
    Each path has its own refinement and ArcFace classifier.
    Both paths are trained jointly.
    """
    
    def __init__(self, embed_dim=256, hidden_dim=512, subspace_dim=64,
                 num_heads=4, num_speakers=1000, num_languages=40,
                 dropout=0.1, arcface_margin=0.3, arcface_scale=30.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # ============================
        # Path A: Subtractive
        # ============================
        self.sub_projection = OrthogonalProjection(
            embed_dim=embed_dim,
            subspace_dim=subspace_dim,
            num_languages=num_languages
        )
        
        self.sub_refinement = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.sub_refine_scale = nn.Parameter(torch.tensor(0.1))
        
        self.sub_classifier = ArcFaceLoss(
            in_features=embed_dim,
            out_features=num_speakers,
            scale=arcface_scale,
            margin=arcface_margin
        )
        
        # ============================
        # Path B: Additive
        # ============================
        self.add_attention = SpeakerQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.add_fusion = GatedAdditiveFusion(embed_dim=embed_dim)
        
        self.add_refinement = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.add_refine_scale = nn.Parameter(torch.tensor(0.1))
        
        self.add_classifier = ArcFaceLoss(
            in_features=embed_dim,
            out_features=num_speakers,
            scale=arcface_scale,
            margin=arcface_margin
        )
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nDual-Path Model V12 initialized:")
        print(f"  - Embed dim: {embed_dim}")
        print(f"  - Path A (Subtractive): Orthogonal Projection (subspace={subspace_dim})")
        print(f"  - Path B (Additive): Cross-Attention ({num_heads} heads) + Gated Fusion")
        print(f"  - Num speakers: {num_speakers}")
        print(f"  - Num languages: {num_languages}")
        print(f"  - Total params: {total_params:,}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_sub(self, speaker_emb, language_emb, speaker_labels=None, language_labels=None):
        """Forward through subtractive path"""
        clean, removed = self.sub_projection(speaker_emb, language_emb)
        refined = clean + self.sub_refine_scale * self.sub_refinement(clean)
        sub_emb = F.normalize(refined, p=2, dim=1, eps=1e-8)
        
        if speaker_labels is not None:
            sub_loss, sub_logits = self.sub_classifier(sub_emb, speaker_labels)
        else:
            sub_loss, sub_logits = None, None
        
        lang_logits = self.sub_projection.predict_language(removed)
        if language_labels is not None:
            lang_loss = F.cross_entropy(lang_logits, language_labels)
        else:
            lang_loss = None
        
        return sub_emb, sub_loss, sub_logits, removed, lang_logits, lang_loss
    
    def forward_add(self, speaker_emb, language_emb, speaker_labels=None):
        """Forward through additive path"""
        extracted, attn_weights = self.add_attention(speaker_emb, language_emb)
        enriched, gate = self.add_fusion(speaker_emb, extracted)
        refined = enriched + self.add_refine_scale * self.add_refinement(enriched)
        add_emb = F.normalize(refined, p=2, dim=1, eps=1e-8)
        
        if speaker_labels is not None:
            add_loss, add_logits = self.add_classifier(add_emb, speaker_labels)
        else:
            add_loss, add_logits = None, None
        
        return add_emb, add_loss, add_logits, attn_weights, gate
    
    def forward(self, speaker_emb, language_emb, speaker_labels=None, language_labels=None):
        """Full forward through both paths"""
        sub_emb, sub_loss, sub_logits, removed, lang_logits, lang_loss = \
            self.forward_sub(speaker_emb, language_emb, speaker_labels, language_labels)
        
        add_emb, add_loss, add_logits, attn_weights, gate = \
            self.forward_add(speaker_emb, language_emb, speaker_labels)
        
        return {
            'sub_emb': sub_emb,
            'sub_loss': sub_loss,
            'sub_logits': sub_logits,
            'removed': removed,
            'lang_logits': lang_logits,
            'lang_loss': lang_loss,
            'add_emb': add_emb,
            'add_loss': add_loss,
            'add_logits': add_logits,
            'attn_weights': attn_weights,
            'gate': gate,
        }
    
    def extract_sub_embedding(self, speaker_emb, language_emb):
        """Extract subtractive embedding for inference"""
        clean, _ = self.sub_projection(speaker_emb, language_emb)
        refined = clean + self.sub_refine_scale * self.sub_refinement(clean)
        return F.normalize(refined, p=2, dim=1, eps=1e-8)
    
    def extract_add_embedding(self, speaker_emb, language_emb):
        """Extract additive embedding for inference"""
        extracted, _ = self.add_attention(speaker_emb, language_emb)
        enriched, _ = self.add_fusion(speaker_emb, extracted)
        refined = enriched + self.add_refine_scale * self.add_refinement(enriched)
        return F.normalize(refined, p=2, dim=1, eps=1e-8)
    
    def get_scales(self):
        return {
            'sub_alpha': torch.sigmoid(self.sub_projection.alpha).item(),
            'sub_refine': self.sub_refine_scale.item(),
            'add_scale': torch.sigmoid(self.add_fusion.add_scale).item(),
            'add_refine': self.add_refine_scale.item(),
        }


class ArcFaceLoss(nn.Module):
    """ArcFace Loss"""
    
    def __init__(self, in_features, out_features, scale=30.0, margin=0.3):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, input, label):
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        input_norm = F.normalize(input, p=2, dim=1)
        
        cosine = F.linear(input_norm, weight_norm)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return F.cross_entropy(output, label), output


# ============================================================================
# Loss Functions
# ============================================================================

class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, clean_emb, language_emb):
        sim = F.cosine_similarity(clean_emb, language_emb)
        return (sim ** 2).mean()


class LanguageInvarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, anchor, positive, is_cross_lang):
        mask = is_cross_lang > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=anchor.device)
        sim = F.cosine_similarity(anchor[mask], positive[mask])
        return (1 - sim).mean()


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()


class HardNegativeLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, negative):
        sim = F.cosine_similarity(anchor, negative)
        loss = F.relu(sim - self.margin + 0.5)
        return loss.mean()


class CrossPathAgreementLoss(nn.Module):
    """
    NEW: Both paths should agree on speaker identity.
    Same-speaker pairs from both paths should be similar.
    This encourages the two paths to capture complementary but consistent info.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, sub_emb, add_emb, speaker_labels):
        # Same-speaker embeddings from both paths should be similar
        sim = F.cosine_similarity(sub_emb, add_emb)
        return (1 - sim).mean()


# ============================================================================
# Dataset
# ============================================================================

class DualPathDataset(Dataset):
    """Dataset with positive AND negative samples for both paths"""
    
    def __init__(self, asv_emb_path, lang_emb_path):
        print(f"\nLoading embeddings...")
        
        asv_data = np.load(asv_emb_path, allow_pickle=True)
        self.asv_embeddings = asv_data['embeddings'].astype(np.float32)
        self.speaker_labels = asv_data['speaker_labels'].astype(np.int64)
        self.language_labels = asv_data['language_labels'].astype(np.int64)
        self.filenames = list(asv_data['filenames'])
        
        lang_data = np.load(lang_emb_path, allow_pickle=True)
        self.lang_embeddings = lang_data['embeddings'].astype(np.float32)
        
        # L2 normalize
        norms = np.linalg.norm(self.asv_embeddings, axis=1, keepdims=True)
        self.asv_embeddings = self.asv_embeddings / np.clip(norms, 1e-8, None)
        norms = np.linalg.norm(self.lang_embeddings, axis=1, keepdims=True)
        self.lang_embeddings = self.lang_embeddings / np.clip(norms, 1e-8, None)
        
        if np.isnan(self.asv_embeddings).any() or np.isnan(self.lang_embeddings).any():
            raise ValueError("NaN detected in embeddings!")
        
        self.num_samples = len(self.asv_embeddings)
        self.num_speakers = len(np.unique(self.speaker_labels))
        self.num_languages = len(np.unique(self.language_labels))
        
        print(f"  Samples: {self.num_samples}")
        print(f"  Speakers: {self.num_speakers}")
        print(f"  Languages: {self.num_languages}")
        
        self._build_indices()
    
    def _build_indices(self):
        print("  Building indices...")
        
        self.speaker_lang_to_indices = defaultdict(lambda: defaultdict(list))
        self.lang_to_indices = defaultdict(list)
        self.speaker_to_indices = defaultdict(list)
        self.speaker_languages = defaultdict(set)
        
        for idx in range(self.num_samples):
            spk = int(self.speaker_labels[idx])
            lang = int(self.language_labels[idx])
            self.speaker_lang_to_indices[spk][lang].append(idx)
            self.lang_to_indices[lang].append(idx)
            self.speaker_to_indices[spk].append(idx)
            self.speaker_languages[spk].add(lang)
        
        multi = sum(1 for langs in self.speaker_languages.values() if len(langs) > 1)
        print(f"  Multi-lingual speakers: {multi} / {self.num_speakers}")
    
    def get_positive(self, idx):
        spk = int(self.speaker_labels[idx])
        lang = int(self.language_labels[idx])
        
        other_langs = [l for l in self.speaker_languages[spk] if l != lang]
        if other_langs:
            pos_lang = random.choice(other_langs)
            return random.choice(self.speaker_lang_to_indices[spk][pos_lang]), True
        else:
            candidates = [i for i in self.speaker_to_indices[spk] if i != idx]
            return (random.choice(candidates), False) if candidates else (idx, False)
    
    def get_hard_negative(self, idx):
        spk = int(self.speaker_labels[idx])
        lang = int(self.language_labels[idx])
        
        candidates = [i for i in self.lang_to_indices[lang] if self.speaker_labels[i] != spk]
        if len(candidates) > 0:
            return random.choice(candidates)
        else:
            while True:
                neg_idx = random.randint(0, self.num_samples - 1)
                if self.speaker_labels[neg_idx] != spk:
                    return neg_idx
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        pos_idx, is_cross_lang = self.get_positive(idx)
        neg_idx = self.get_hard_negative(idx)
        return (
            self.asv_embeddings[idx],
            self.lang_embeddings[idx],
            self.speaker_labels[idx],
            self.language_labels[idx],
            pos_idx,
            neg_idx,
            1 if is_cross_lang else 0
        )


class EvalDataset(Dataset):
    def __init__(self, asv_emb_path, lang_emb_path):
        asv_data = np.load(asv_emb_path, allow_pickle=True)
        self.asv_embeddings = asv_data['embeddings'].astype(np.float32)
        self.filenames = list(asv_data['filenames'])
        
        lang_data = np.load(lang_emb_path, allow_pickle=True)
        self.lang_embeddings = lang_data['embeddings'].astype(np.float32)
        
        norms = np.linalg.norm(self.asv_embeddings, axis=1, keepdims=True)
        self.asv_embeddings = self.asv_embeddings / np.clip(norms, 1e-8, None)
        norms = np.linalg.norm(self.lang_embeddings, axis=1, keepdims=True)
        self.lang_embeddings = self.lang_embeddings / np.clip(norms, 1e-8, None)
        
        self.num_samples = len(self.asv_embeddings)
        print(f"\nEval dataset: {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples


# ============================================================================
# Score Fusion Evaluation (4-way)
# ============================================================================

def compute_eer(target_scores, nontarget_scores):
    """Fast EER computation without expensive interpolation"""
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return 100.0, 0.0
    
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
    
    tpr = tp / num_pos
    fpr = fp / num_neg
    fnr = 1 - tpr
    
    # Find crossing point
    diff = fpr - fnr
    idx = np.searchsorted(diff, 0)
    
    if idx == 0:
        eer = fpr[0]
    elif idx >= len(diff):
        eer = fpr[-1]
    else:
        # Linear interpolation between adjacent points
        alpha = diff[idx] / (diff[idx] - diff[idx - 1] + 1e-10)
        eer = alpha * fpr[idx - 1] + (1 - alpha) * fpr[idx]
    
    return eer * 100, 0


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


def compute_eer_and_minDCF(target_scores, nontarget_scores, ptar=0.5, cfa=1, cfr=1):
    """Compute both EER and minDCF in one pass"""
    eer, _ = compute_eer(target_scores, nontarget_scores)
    mindcf = compute_minDCF(target_scores, nontarget_scores, ptar, cfa, cfr)
    return eer, mindcf


class FourWayScoreFusionEvaluator:
    """
    Evaluates 4 embedding types and all their combinations:
    ASV (original), LID (original), Sub (subtractive path), Add (additive path)
    """
    
    def __init__(self, eval_dataset, trial_file):
        print(f"\n  Caching trial file...")
        
        self.fn_to_idx = {fn: i for i, fn in enumerate(eval_dataset.filenames)}
        spk_labels, lang_labels, enroll_idx, test_idx = [], [], [], []
        
        def _extract_language(file_path):
            parts = file_path.split('/')
            if len(parts) >= 2:
                return parts[1]
            return 'unknown'
        
        with open(trial_file, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        with open(trial_file, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="  Reading"):
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                
                enroll_path = parts[1]
                test_path = parts[2]
                enroll_fn = os.path.splitext(os.path.basename(enroll_path))[0]
                test_fn = os.path.splitext(os.path.basename(test_path))[0]
                
                if enroll_fn in self.fn_to_idx and test_fn in self.fn_to_idx:
                    spk_labels.append(int(parts[0]))
                    enroll_idx.append(self.fn_to_idx[enroll_fn])
                    test_idx.append(self.fn_to_idx[test_fn])
                    
                    enroll_lang = _extract_language(enroll_path)
                    test_lang = _extract_language(test_path)
                    lang_labels.append(1 if enroll_lang == test_lang else 0)
        
        self.labels = np.array(spk_labels)
        self.lang_labels = np.array(lang_labels)
        self.enroll_idx = np.array(enroll_idx)
        self.test_idx = np.array(test_idx)
        self.target_mask = self.labels == 1
        self.lang_target_mask = self.lang_labels == 1
        
        print(f"  Cached {len(self.labels):,} trials")
        print(f"  Speaker: {self.target_mask.sum():,} target, {(~self.target_mask).sum():,} non-target")
        print(f"  Language: {self.lang_target_mask.sum():,} same-lang, {(~self.lang_target_mask).sum():,} diff-lang")
    
    def compute_scores_batch(self, embeddings, batch_size=100000):
        num_trials = len(self.labels)
        scores = np.zeros(num_trials, dtype=np.float32)
        for start in range(0, num_trials, batch_size):
            end = min(start + batch_size, num_trials)
            enroll = embeddings[self.enroll_idx[start:end]]
            test = embeddings[self.test_idx[start:end]]
            scores[start:end] = np.sum(enroll * test, axis=1)
        return scores
    
    def _eer(self, scores):
        eer, _ = compute_eer(scores[self.target_mask], scores[~self.target_mask])
        return eer
    
    def _mindcf(self, scores):
        mindcf = compute_minDCF(scores[self.target_mask], scores[~self.target_mask])
        return mindcf
    
    def _metrics(self, scores):
        """Compute both EER and minDCF"""
        eer, _ = compute_eer(scores[self.target_mask], scores[~self.target_mask])
        mindcf = compute_minDCF(scores[self.target_mask], scores[~self.target_mask])
        return eer, mindcf
    
    def evaluate_all(self, model, eval_dataset, device):
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        model.eval()
        
        asv_embs = eval_dataset.asv_embeddings
        lid_embs = eval_dataset.lang_embeddings
        
        # Extract both path embeddings
        sub_list, add_list = [], []
        with torch.no_grad():
            for i in range(0, len(eval_dataset), 256):
                end = min(i + 256, len(eval_dataset))
                asv = torch.tensor(asv_embs[i:end]).to(device)
                lid = torch.tensor(lid_embs[i:end]).to(device)
                sub_list.append(model.extract_sub_embedding(asv, lid).cpu().numpy())
                add_list.append(model.extract_add_embedding(asv, lid).cpu().numpy())
        
        sub_embs = np.vstack(sub_list)
        add_embs = np.vstack(add_list)
        
        # Compute all 4 score sets
        asv_scores = self.compute_scores_batch(asv_embs)
        lid_scores = self.compute_scores_batch(lid_embs)
        sub_scores = self.compute_scores_batch(sub_embs)
        add_scores = self.compute_scores_batch(add_embs)
        
        results = {}
        
        # === Individual ===
        asv_eer, asv_minDCF = self._metrics(asv_scores)
        results['asv_only'] = asv_eer
        results['asv_only_minDCF'] = asv_minDCF
        
        eer, _ = compute_eer(lid_scores[self.lang_target_mask], lid_scores[~self.lang_target_mask])
        mindcf = compute_minDCF(lid_scores[self.lang_target_mask], lid_scores[~self.lang_target_mask])
        results['lid_only'] = eer
        results['lid_only_minDCF'] = mindcf
        
        sub_eer, sub_minDCF = self._metrics(sub_scores)
        results['sub_only'] = sub_eer
        results['sub_only_minDCF'] = sub_minDCF
        
        add_eer, add_minDCF = self._metrics(add_scores)
        results['add_only'] = add_eer
        results['add_only_minDCF'] = add_minDCF
        
        # === Pairs (grid search alpha) ===
        alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        # ASV + Sub
        best_eer, best_mindcf, best_a = 100.0, 100.0, 0.5
        for a in alphas:
            combined = a * asv_scores + (1 - a) * sub_scores
            eer, mindcf = self._metrics(combined)
            if eer < best_eer: best_eer, best_a = eer, a
            if mindcf < best_mindcf: best_mindcf = mindcf
        results['asv+sub'] = best_eer
        results['asv+sub_minDCF'] = best_mindcf
        results['asv+sub_a'] = best_a
        
        # ASV + Add
        best_eer, best_mindcf, best_a = 100.0, 100.0, 0.5
        for a in alphas:
            combined = a * asv_scores + (1 - a) * add_scores
            eer, mindcf = self._metrics(combined)
            if eer < best_eer: best_eer, best_a = eer, a
            if mindcf < best_mindcf: best_mindcf = mindcf
        results['asv+add'] = best_eer
        results['asv+add_minDCF'] = best_mindcf
        results['asv+add_a'] = best_a
        
        # Sub + Add
        best_eer, best_mindcf, best_a = 100.0, 100.0, 0.5
        for a in alphas:
            combined = a * sub_scores + (1 - a) * add_scores
            eer, mindcf = self._metrics(combined)
            if eer < best_eer: best_eer, best_a = eer, a
            if mindcf < best_mindcf: best_mindcf = mindcf
        results['sub+add'] = best_eer
        results['sub+add_minDCF'] = best_mindcf
        results['sub+add_a'] = best_a
        
        # === With LID subtraction ===
        betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        # Sub - LID
        best_eer, best_mindcf, best_b = 100.0, 100.0, 0.1
        for b in betas:
            combined = sub_scores - b * lid_scores
            eer, mindcf = self._metrics(combined)
            if eer < best_eer: best_eer, best_b = eer, b
            if mindcf < best_mindcf: best_mindcf = mindcf
        results['sub-lid'] = best_eer
        results['sub-lid_minDCF'] = best_mindcf
        results['sub-lid_b'] = best_b
        
        # Add - LID
        best_eer, best_mindcf, best_b = 100.0, 100.0, 0.1
        for b in betas:
            combined = add_scores - b * lid_scores
            eer, mindcf = self._metrics(combined)
            if eer < best_eer: best_eer, best_b = eer, b
            if mindcf < best_mindcf: best_mindcf = mindcf
        results['add-lid'] = best_eer
        results['add-lid_minDCF'] = best_mindcf
        results['add-lid_b'] = best_b
        
        # ASV - LID
        best_eer, best_mindcf, best_b = 100.0, 100.0, 0.1
        for b in betas:
            combined = asv_scores - b * lid_scores
            eer, mindcf = self._metrics(combined)
            if eer < best_eer: best_eer, best_b = eer, b
            if mindcf < best_mindcf: best_mindcf = mindcf
        results['asv-lid'] = best_eer
        results['asv-lid_minDCF'] = best_mindcf
        results['asv-lid_b'] = best_b
        
        # === Triple: ASV + Sub + Add ===
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.33, 0.33, 0.34)
        for a in [0.2, 0.3, 0.4]:
            for b in [0.3, 0.4, 0.5]:
                c = 1.0 - a - b
                if c > 0.05:
                    combined = a * asv_scores + b * sub_scores + c * add_scores
                    eer, mindcf = self._metrics(combined)
                    if eer < best_eer: best_eer, best_p = eer, (a, b, c)
                    if mindcf < best_mindcf: best_mindcf = mindcf
        results['asv+sub+add'] = best_eer
        results['asv+sub+add_minDCF'] = best_mindcf
        results['asv+sub+add_p'] = best_p
        
        # === Full: α*ASV + β*Sub + γ*Add - δ*LID ===
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.3, 0.4, 0.3, 0.1)
        for a in [0.2, 0.3, 0.4]:
            for b in [0.3, 0.4, 0.5, 0.6]:
                for c in [0.2, 0.3, 0.4, 0.5]:
                    for d in [0.05, 0.1, 0.15, 0.2]:
                        combined = a * asv_scores + b * sub_scores + c * add_scores - d * lid_scores
                        eer, mindcf = self._metrics(combined)
                        if eer < best_eer:
                            best_eer, best_p = eer, (a, b, c, d)
                        if mindcf < best_mindcf:
                            best_mindcf = mindcf
        results['full'] = best_eer
        results['full_minDCF'] = best_mindcf
        results['full_params'] = best_p
        
        # === Sub + Add - LID (without ASV) ===
        best_eer, best_mindcf, best_p = 100.0, 100.0, (0.5, 0.5, 0.1)
        for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for d in [0.05, 0.1, 0.15, 0.2]:
                combined = a * sub_scores + (1 - a) * add_scores - d * lid_scores
                eer, mindcf = self._metrics(combined)
                if eer < best_eer:
                    best_eer, best_p = eer, (a, 1 - a, d)
                if mindcf < best_mindcf:
                    best_mindcf = mindcf
        results['sub+add-lid'] = best_eer
        results['sub+add-lid_minDCF'] = best_mindcf
        results['sub+add-lid_p'] = best_p
        
        return results


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, train_dataset, train_loader, optimizer, device, epoch,
                ortho_weight=1.0, lang_inv_weight=1.0, lang_pred_weight=0.5,
                triplet_weight=0.5, hard_neg_weight=0.3, agreement_weight=0.5):
    model.train()
    
    ortho_fn = OrthogonalityLoss()
    inv_fn = LanguageInvarianceLoss()
    triplet_fn = TripletLoss(margin=0.3)
    hard_neg_fn = HardNegativeLoss(margin=0.3)
    agreement_fn = CrossPathAgreementLoss()
    
    metrics = defaultdict(float)
    correct_sub, correct_add, correct_lang, total = 0, 0, 0, 0
    cross_lang_pairs = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        (anchor_asv, anchor_lang, spk_labels, lang_labels,
         pos_idx, neg_idx, is_cross_lang) = batch
        
        anchor_asv = anchor_asv.to(device)
        anchor_lang = anchor_lang.to(device)
        spk_labels = spk_labels.to(device)
        lang_labels = lang_labels.to(device)
        is_cross_lang = is_cross_lang.to(device)
        
        pos_idx = pos_idx.numpy()
        neg_idx = neg_idx.numpy()
        pos_asv = torch.tensor(train_dataset.asv_embeddings[pos_idx]).to(device)
        pos_lang = torch.tensor(train_dataset.lang_embeddings[pos_idx]).to(device)
        neg_asv = torch.tensor(train_dataset.asv_embeddings[neg_idx]).to(device)
        neg_lang = torch.tensor(train_dataset.lang_embeddings[neg_idx]).to(device)
        
        optimizer.zero_grad()
        
        # Forward through both paths
        out = model(anchor_asv, anchor_lang, spk_labels, lang_labels)
        
        # Positive/negative embeddings from both paths
        pos_sub = model.extract_sub_embedding(pos_asv, pos_lang)
        neg_sub = model.extract_sub_embedding(neg_asv, neg_lang)
        pos_add = model.extract_add_embedding(pos_asv, pos_lang)
        neg_add = model.extract_add_embedding(neg_asv, neg_lang)
        
        loss = torch.tensor(0.0, device=device)
        
        # ============================================================
        # PATH A LOSSES (Subtractive)
        # ============================================================
        
        # ArcFace for subtractive path
        loss = loss + out['sub_loss']
        metrics['sub_arcface'] += out['sub_loss'].item()
        
        # Orthogonality: clean_emb ⊥ language
        ortho_loss = ortho_fn(out['sub_emb'], anchor_lang)
        loss = loss + ortho_weight * ortho_loss
        metrics['ortho'] += ortho_loss.item()
        
        # Language prediction on removed component
        loss = loss + lang_pred_weight * out['lang_loss']
        metrics['lang_pred'] += out['lang_loss'].item()
        
        # Language invariance for subtractive path
        sub_inv = inv_fn(out['sub_emb'], pos_sub, is_cross_lang)
        loss = loss + lang_inv_weight * sub_inv
        metrics['sub_inv'] += sub_inv.item()
        
        # Triplet for subtractive path
        sub_triplet = triplet_fn(out['sub_emb'], pos_sub, neg_sub)
        loss = loss + triplet_weight * sub_triplet
        metrics['sub_triplet'] += sub_triplet.item()
        
        # Hard negative for subtractive path
        sub_hn = hard_neg_fn(out['sub_emb'], neg_sub)
        loss = loss + hard_neg_weight * sub_hn
        metrics['sub_hn'] += sub_hn.item()
        
        # ============================================================
        # PATH B LOSSES (Additive)
        # ============================================================
        
        # ArcFace for additive path
        loss = loss + out['add_loss']
        metrics['add_arcface'] += out['add_loss'].item()
        
        # Language invariance for additive path
        add_inv = inv_fn(out['add_emb'], pos_add, is_cross_lang)
        loss = loss + lang_inv_weight * add_inv
        metrics['add_inv'] += add_inv.item()
        
        # Triplet for additive path
        add_triplet = triplet_fn(out['add_emb'], pos_add, neg_add)
        loss = loss + triplet_weight * add_triplet
        metrics['add_triplet'] += add_triplet.item()
        
        # Hard negative for additive path
        add_hn = hard_neg_fn(out['add_emb'], neg_add)
        loss = loss + hard_neg_weight * add_hn
        metrics['add_hn'] += add_hn.item()
        
        # ============================================================
        # CROSS-PATH LOSS (both paths should agree)
        # ============================================================
        agree_loss = agreement_fn(out['sub_emb'], out['add_emb'], spk_labels)
        loss = loss + agreement_weight * agree_loss
        metrics['agreement'] += agree_loss.item()
        
        # Skip if NaN
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        metrics['total'] += loss.item()
        
        correct_sub += (out['sub_logits'].argmax(dim=1) == spk_labels).sum().item()
        correct_add += (out['add_logits'].argmax(dim=1) == spk_labels).sum().item()
        correct_lang += (out['lang_logits'].argmax(dim=1) == lang_labels).sum().item()
        total += spk_labels.size(0)
        cross_lang_pairs += is_cross_lang.sum().item()
        
        scales = model.get_scales()
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'sub': f'{100.*correct_sub/total:.1f}%',
            'add': f'{100.*correct_add/total:.1f}%',
        })
    
    n = len(train_loader)
    return {k: v/n for k, v in metrics.items()} | {
        'sub_acc': 100.*correct_sub/total,
        'add_acc': 100.*correct_add/total,
        'lang_acc': 100.*correct_lang/total,
        'cross_lang_ratio': 100.*cross_lang_pairs/total
    }


def main():
    parser = argparse.ArgumentParser(description='Dual-Path Fusion V12')
    
    parser.add_argument('--train_asv_emb', type=str, required=True)
    parser.add_argument('--train_lang_emb', type=str, required=True)
    parser.add_argument('--eval_asv_emb', type=str, required=True)
    parser.add_argument('--eval_lang_emb', type=str, required=True)
    parser.add_argument('--trial_file', type=str, required=True)
    
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--subspace_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--arcface_margin', type=float, default=0.3)
    parser.add_argument('--arcface_scale', type=float, default=30.0)
    
    parser.add_argument('--ortho_weight', type=float, default=1.0)
    parser.add_argument('--lang_inv_weight', type=float, default=1.0)
    parser.add_argument('--lang_pred_weight', type=float, default=0.5)
    parser.add_argument('--triplet_weight', type=float, default=0.5)
    parser.add_argument('--hard_neg_weight', type=float, default=0.3)
    parser.add_argument('--agreement_weight', type=float, default=0.5,
                       help='Weight for cross-path agreement loss')
    
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./ckpt_fusion_v12')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n{'='*70}")
    print("Dual-Path Fusion V12 - Subtractive + Additive + 4-Way Score Fusion")
    print(f"{'='*70}")
    print("  Path A (Sub): Orthogonal Projection → remove language")
    print("  Path B (Add): Cross-Attention → extract speaker from LID")
    print("  Evaluation: ASV + Sub + Add - LID score fusion")
    
    train_dataset = DualPathDataset(args.train_asv_emb, args.train_lang_emb)
    eval_dataset = EvalDataset(args.eval_asv_emb, args.eval_lang_emb)
    evaluator = FourWayScoreFusionEvaluator(eval_dataset, args.trial_file)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    
    model = DualPathModel(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        subspace_dim=args.subspace_dim,
        num_heads=args.num_heads,
        num_speakers=train_dataset.num_speakers,
        num_languages=train_dataset.num_languages,
        dropout=args.dropout,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_eer = 100.0
    best_strategy = 'sub_only'
    
    print(f"\nLoss Weights:")
    print(f"  Ortho={args.ortho_weight}, LangInv={args.lang_inv_weight}, "
          f"LangPred={args.lang_pred_weight}")
    print(f"  Triplet={args.triplet_weight}, HardNeg={args.hard_neg_weight}, "
          f"Agreement={args.agreement_weight}")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"--- Epoch {epoch}/{args.epochs} ---")
        
        metrics = train_epoch(
            model, train_dataset, train_loader, optimizer, device, epoch,
            ortho_weight=args.ortho_weight,
            lang_inv_weight=args.lang_inv_weight,
            lang_pred_weight=args.lang_pred_weight,
            triplet_weight=args.triplet_weight,
            hard_neg_weight=args.hard_neg_weight,
            agreement_weight=args.agreement_weight
        )
        
        scheduler.step()
        
        scales = model.get_scales()
        print(f"\nPath A (Sub): ArcFace={metrics['sub_arcface']:.4f}, Ortho={metrics['ortho']:.4f}, "
              f"Triplet={metrics['sub_triplet']:.4f}, Acc={metrics['sub_acc']:.1f}%")
        print(f"Path B (Add): ArcFace={metrics['add_arcface']:.4f}, "
              f"Triplet={metrics['add_triplet']:.4f}, Acc={metrics['add_acc']:.1f}%")
        print(f"Cross-path: Agreement={metrics['agreement']:.4f}, "
              f"Lang Acc={metrics['lang_acc']:.1f}%")
        print(f"Scales: sub_α={scales['sub_alpha']:.3f}, add_scale={scales['add_scale']:.3f}")
        
        results = evaluator.evaluate_all(model, eval_dataset, device)
        
        print(f"\n  === Individual ===")
        print(f"  ASV only:       {results['asv_only']:.4f}% EER | {results['asv_only_minDCF']:.4f}% minDCF")
        print(f"  LID only:       {results['lid_only']:.4f}% EER | {results['lid_only_minDCF']:.4f}% minDCF")
        print(f"  Sub only:       {results['sub_only']:.4f}% EER | {results['sub_only_minDCF']:.4f}% minDCF")
        print(f"  Add only:       {results['add_only']:.4f}% EER | {results['add_only_minDCF']:.4f}% minDCF")
        
        print(f"\n  === Pairs ===")
        print(f"  ASV+Sub (α={results['asv+sub_a']:.1f}):  {results['asv+sub']:.4f}% EER | {results['asv+sub_minDCF']:.4f}% minDCF")
        print(f"  ASV+Add (α={results['asv+add_a']:.1f}):  {results['asv+add']:.4f}% EER | {results['asv+add_minDCF']:.4f}% minDCF")
        print(f"  Sub+Add (α={results['sub+add_a']:.1f}):  {results['sub+add']:.4f}% EER | {results['sub+add_minDCF']:.4f}% minDCF")
        
        print(f"\n  === With LID subtraction ===")
        print(f"  ASV-LID (β={results['asv-lid_b']:.2f}):  {results['asv-lid']:.4f}% EER | {results['asv-lid_minDCF']:.4f}% minDCF")
        print(f"  Sub-LID (β={results['sub-lid_b']:.2f}):  {results['sub-lid']:.4f}% EER | {results['sub-lid_minDCF']:.4f}% minDCF")
        print(f"  Add-LID (β={results['add-lid_b']:.2f}):  {results['add-lid']:.4f}% EER | {results['add-lid_minDCF']:.4f}% minDCF")
        p = results['sub+add-lid_p']
        print(f"  Sub+Add-LID ({p[0]:.1f},{p[1]:.1f},δ={p[2]:.2f}): {results['sub+add-lid']:.4f}% EER | {results['sub+add-lid_minDCF']:.4f}% minDCF")
        
        print(f"\n  === Triple + Full ===")
        p = results['asv+sub+add_p']
        print(f"  ASV+Sub+Add ({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}): {results['asv+sub+add']:.4f}% EER | {results['asv+sub+add_minDCF']:.4f}% minDCF")
        p = results['full_params']
        print(f"  Full: α={p[0]:.1f}*ASV + β={p[1]:.1f}*Sub + γ={p[2]:.1f}*Add - δ={p[3]:.2f}*LID: {results['full']:.4f}% EER | {results['full_minDCF']:.4f}% minDCF")
        
        # Find best
        strategies = ['asv_only', 'sub_only', 'add_only', 'asv+sub', 'asv+add', 'sub+add',
                      'asv-lid', 'sub-lid', 'add-lid', 'sub+add-lid', 'asv+sub+add', 'full']
        current_best = min(strategies, key=lambda x: results[x])
        current_eer = results[current_best]
        
        print(f"\n  ★ Best: {current_best} = {current_eer:.4f}%")
        
        if current_eer < best_eer:
            best_eer = current_eer
            best_strategy = current_best
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'results': results,
                'best_strategy': best_strategy,
                'best_eer': best_eer,
                'num_speakers': train_dataset.num_speakers,
                'num_languages': train_dataset.num_languages,
                'scales': scales,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model")
        
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(f"{epoch} "
                   f"asv_eer={results['asv_only']:.4f} asv_dcf={results['asv_only_minDCF']:.4f} "
                   f"sub_eer={results['sub_only']:.4f} sub_dcf={results['sub_only_minDCF']:.4f} "
                   f"add_eer={results['add_only']:.4f} add_dcf={results['add_only_minDCF']:.4f} "
                   f"sub+add_eer={results['sub+add']:.4f} sub+add_dcf={results['sub+add_minDCF']:.4f} "
                   f"sub-lid_eer={results['sub-lid']:.4f} sub-lid_dcf={results['sub-lid_minDCF']:.4f} "
                   f"add-lid_eer={results['add-lid']:.4f} add-lid_dcf={results['add-lid_minDCF']:.4f} "
                   f"full_eer={results['full']:.4f} full_dcf={results['full_minDCF']:.4f} "
                   f"best={current_eer:.4f}\n")
    
    print(f"\n{'='*70}")
    print(f"Done! Best: {best_strategy} = {best_eer:.4f}%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
