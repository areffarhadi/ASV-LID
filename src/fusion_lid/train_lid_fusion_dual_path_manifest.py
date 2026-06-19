#!/usr/bin/env python3
"""
Language Embedding Extractor - Dual-Path Fusion with Manifest-based Data Loading

Combines:
  1. Manifest-based data loading (on-the-fly embedding loading, mixed train/dev folders)
  2. Dual-path fusion architecture for LANGUAGE EMBEDDING EXTRACTION

Dual Paths (REVERSED from speaker extraction):
  Path A (Subtractive): Remove speaker contamination from language embedding
    - Input: language_emb (from stage2_lid)
    - Uses orthogonal projection on speaker subspace
    - Output: clean_lang_emb (speaker-independent)
    
  Path B (Additive): Extract complementary language info from speaker embedding
    - Language queries Speaker for language info
    - Cross-attention mechanism
    - Output: enriched_lang_emb (with additional speaker-side language cues)

Both paths contribute to language identification task.
"""

import argparse
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import math

# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# Manifest Data Loader
# ============================================================================
class ManifestDataLoader:
    """Loads manifest and organizes data by split"""
    
    def __init__(self, manifest_file):
        self.manifest_file = manifest_file
        self.train_data = []
        self.val_data = []
        self.val2_data = []
        self.languages = set()
        self.load_manifest()
    
    def load_manifest(self):
        """Parse manifest: flag \t path \t language"""
        with open(self.manifest_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                flag, path, language = int(parts[0]), parts[1], parts[2]
                self.languages.add(language)
                
                data_point = {'path': path, 'language': language}
                
                if flag == 1:
                    self.train_data.append(data_point)
                elif flag == 2:
                    self.val_data.append(data_point)
                elif flag == 3:
                    self.val2_data.append(data_point)
        
        logging.info(f"Manifest loaded: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.val2_data)} val2")
        logging.info(f"Languages: {sorted(self.languages)}")
        self.lang_to_id = {lang: idx for idx, lang in enumerate(sorted(self.languages))}
        self.num_classes = len(self.languages)
    
    def get_lang_id(self, language):
        return self.lang_to_id[language]

# ============================================================================
# Dataset
# ============================================================================
class EmbeddingDataset(Dataset):
    """Dataset that loads embeddings on-the-fly"""
    
    def __init__(self, data_list, emb_stage1_base, emb_stage2_base, manifest_loader, split='train'):
        self.data_list = data_list
        self.emb_stage1_base = emb_stage1_base
        self.emb_stage2_base = emb_stage2_base
        self.manifest_loader = manifest_loader
        self.split = split
    
    def find_embedding(self, base_path, wav_path):
        """
        Find embedding for given wav_path.
        Since data can be in either train or dev folder, we check both.
        wav_path format: speaker_id/language/filename.wav
        embedding path: {base}/{train,dev}/speaker_id/language/filename.npy
        """
        # Remove .wav extension and add .npy
        emb_name = wav_path.replace('.wav', '.npy')
        
        # Try train folder first, then dev folder
        for fold in ['train', 'dev']:
            emb_path = os.path.join(base_path, fold, emb_name)
            if os.path.exists(emb_path):
                return emb_path
        
        logging.warning(f"Embedding not found for: {wav_path}")
        return None
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_point = self.data_list[idx]
        wav_path = data_point['path']
        language = data_point['language']
        lang_id = self.manifest_loader.get_lang_id(language)
        
        # Load embeddings
        emb1_path = self.find_embedding(self.emb_stage1_base, wav_path)
        emb2_path = self.find_embedding(self.emb_stage2_base, wav_path)
        
        if emb1_path is None or emb2_path is None:
            missing = []
            if emb1_path is None:
                missing.append(f"stage1 speaker embedding for '{wav_path}' under '{self.emb_stage1_base}/{{train,dev}}'")
            if emb2_path is None:
                missing.append(f"stage2 language embedding for '{wav_path}' under '{self.emb_stage2_base}/{{train,dev}}'")
            raise FileNotFoundError("Missing required embedding(s): " + "; ".join(missing))

        emb1 = np.load(emb1_path).astype(np.float32)
        emb2 = np.load(emb2_path).astype(np.float32)
        
        return {
            'speaker_emb': torch.FloatTensor(emb1),    # stage1_asv (256-dim)
            'language_emb': torch.FloatTensor(emb2),   # stage2_lid (256-dim)
            'label': lang_id,
            'language': language,
            'path': wav_path
        }

# ============================================================================
# Path A: Subtractive (Remove Speaker from Language)
# ============================================================================
class OrthogonalProjection(nn.Module):
    """Remove speaker subspace from language embedding"""
    
    def __init__(self, embed_dim=256, subspace_dim=64, num_languages=40):
        super().__init__()
        
        self.basis = nn.Parameter(torch.randn(subspace_dim, embed_dim))
        
        self.speaker_to_subspace = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.speaker_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1000)  # Placeholder for speaker dim
        )
    
    def forward(self, language_emb, speaker_emb):
        """
        Remove speaker contamination from language embedding
        
        Args:
            language_emb: Language embedding (main input)
            speaker_emb: Speaker embedding (defines subspace to remove)
        
        Returns:
            clean: Language embedding with speaker info removed
            removed: The removed component
        """
        basis_norm = F.normalize(self.basis, p=2, dim=1, eps=1e-8)
        proj_coeffs = torch.mm(language_emb, basis_norm.t())
        proj_basis = torch.mm(proj_coeffs, basis_norm)
        
        speaker_direction = self.speaker_to_subspace(speaker_emb)
        speaker_direction = F.normalize(speaker_direction, p=2, dim=1, eps=1e-8)
        proj_coeff_speaker = (language_emb * speaker_direction).sum(dim=1, keepdim=True)
        proj_speaker = proj_coeff_speaker * speaker_direction
        
        alpha = torch.sigmoid(self.alpha)
        removed = alpha * proj_basis + (1 - alpha) * proj_speaker
        clean = language_emb - removed
        
        return clean, removed
    
    def predict_speaker(self, removed):
        """Predict speaker identity from removed component (for auxiliary loss)"""
        return self.speaker_classifier(removed)

# ============================================================================
# Path B: Additive (Extract Language from Speaker)
# ============================================================================
class LanguageQueryAttention(nn.Module):
    """Language queries Speaker for language-relevant information"""
    
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
    
    def forward(self, language_emb, speaker_emb):
        """
        Language attends to Speaker for language-relevant info
        
        Args:
            language_emb: Query (language embedding)
            speaker_emb: Key/Value (speaker embedding)
        
        Returns:
            extracted: Extracted language-relevant info from speaker
            attn_weights: Attention weights
        """
        B = language_emb.size(0)
        
        Q = self.W_q(language_emb).view(B, self.num_heads, self.head_dim)
        K = self.W_k(speaker_emb).view(B, self.num_heads, self.head_dim)
        V = self.W_v(speaker_emb).view(B, self.num_heads, self.head_dim)
        
        scale = math.sqrt(self.head_dim) * F.softplus(self.temperature)
        attn_scores = (Q * K).sum(dim=-1) / scale
        attn_weights = torch.sigmoid(attn_scores)
        
        attended = attn_weights.unsqueeze(-1) * V
        attended = attended.reshape(B, self.embed_dim)
        
        extracted = self.out_proj(self.dropout(attended))
        extracted = self.layer_norm(extracted)
        
        return extracted, attn_weights


class GatedAdditiveFusion(nn.Module):
    """Gated addition: output = language + gate * extracted"""
    
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
    
    def forward(self, language_emb, extracted_info):
        """
        Args:
            language_emb: Original language embedding
            extracted_info: Information extracted from speaker
        
        Returns:
            enriched: Enriched language embedding
            gate: Gate values
        """
        combined = torch.cat([language_emb, extracted_info], dim=-1)
        gate = self.gate_net(combined)
        scale = torch.sigmoid(self.add_scale)
        enriched = language_emb + scale * gate * extracted_info
        return enriched, gate

# ============================================================================
# Full Model: Dual-Path for Language Embedding Extraction
# ============================================================================
class DualPathLanguageExtractor(nn.Module):
    """
    Dual-Path Model for LANGUAGE EMBEDDING EXTRACTION
    
    Path A: Subtractive - Remove speaker contamination
    Path B: Additive - Extract complementary language info from speaker
    """
    
    def __init__(self, embed_dim=256, hidden_dim=512, subspace_dim=64,
                 num_heads=4, num_classes=35, num_speakers=1000,
                 dropout=0.1, arcface_margin=0.3, arcface_scale=30.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.arcface_margin = arcface_margin
        self.arcface_scale = arcface_scale
        
        # ============================
        # Path A: Subtractive
        # ============================
        self.sub_projection = OrthogonalProjection(
            embed_dim=embed_dim,
            subspace_dim=subspace_dim,
            num_languages=num_classes
        )
        
        # Refinement layers for Path A output
        self.sub_refine = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # ArcFace classifier for Path A
        self.sub_classifier_weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.sub_classifier_weight)
        
        # ============================
        # Path B: Additive
        # ============================
        self.lang_query_attention = LanguageQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.additive_fusion = GatedAdditiveFusion(embed_dim=embed_dim)
        
        # Refinement layers for Path B output
        self.add_refine = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # ArcFace classifier for Path B
        self.add_classifier_weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.add_classifier_weight)
        
        # ============================
        # Triplet Loss Components
        # ============================
        self.triplet_loss = nn.TripletMarginLoss(margin=0.2)
    
    def arcface_loss(self, embeds, labels, weight, scale=30.0, margin=0.3):
        """ArcFace loss"""
        # Normalize embeddings and weight
        embeds = F.normalize(embeds, p=2, dim=1)
        weight = F.normalize(weight, p=2, dim=1)
        
        # Compute cosine similarity
        logits = torch.mm(embeds, weight.t())  # (B, num_classes)
        
        # Add margin
        logits = logits.clamp(-1 + 1e-7, 1 - 1e-7)
        one_hot = F.one_hot(labels, self.num_classes).float()
        
        cos_m = torch.Tensor([1.0]).cuda() if embeds.is_cuda else torch.Tensor([1.0])
        sin_m = torch.sqrt((1.0 - cos_m * cos_m).clamp(0, 1))
        
        # Rotation matrix
        sin_theta = torch.asin(torch.clamp(torch.norm(embeds - one_hot * weight[labels], dim=1), max=1.0))
        cos_theta = torch.sqrt((1.0 - sin_theta * sin_theta).clamp(0, 1))
        
        cos_theta_m = cos_theta * 1.0 - sin_theta * sin_m
        
        logits = logits * (1 - one_hot) + cos_theta_m * scale * one_hot
        
        return F.cross_entropy(logits, labels)
    
    def forward(self, speaker_emb, language_emb):
        """
        Args:
            speaker_emb: Speaker embedding (256-dim)
            language_emb: Language embedding (256-dim)
        
        Returns:
            sub_logits: Path A (Subtractive) logits
            add_logits: Path B (Additive) logits
            sub_emb: Path A refined embedding
            add_emb: Path B refined embedding
        """
        # ============================
        # Path A: Subtractive
        # ============================
        clean_lang, removed_speaker = self.sub_projection(language_emb, speaker_emb)
        sub_emb = self.sub_refine(clean_lang)
        sub_emb_norm = F.normalize(sub_emb, p=2, dim=1)
        sub_logits = torch.mm(sub_emb_norm, self.sub_classifier_weight.t())
        
        # ============================
        # Path B: Additive
        # ============================
        extracted, attn_weights = self.lang_query_attention(language_emb, speaker_emb)
        enriched_lang, gate = self.additive_fusion(language_emb, extracted)
        add_emb = self.add_refine(enriched_lang)
        add_emb_norm = F.normalize(add_emb, p=2, dim=1)
        add_logits = torch.mm(add_emb_norm, self.add_classifier_weight.t())
        
        return sub_logits, add_logits, sub_emb, add_emb
    
    def extract_sub_embedding(self, speaker_emb, language_emb):
        """Extract Path A (Subtractive) embedding"""
        clean_lang, _ = self.sub_projection(language_emb, speaker_emb)
        sub_emb = self.sub_refine(clean_lang)
        return F.normalize(sub_emb, p=2, dim=1)
    
    def extract_add_embedding(self, speaker_emb, language_emb):
        """Extract Path B (Additive) embedding"""
        extracted, _ = self.lang_query_attention(language_emb, speaker_emb)
        enriched_lang, _ = self.additive_fusion(language_emb, extracted)
        add_emb = self.add_refine(enriched_lang)
        return F.normalize(add_emb, p=2, dim=1)

# ============================================================================
# Trial Evaluator (EER with Dual-Path Score Fusion)
# ============================================================================
class TrialEvaluator:
    """Evaluate EER on trial file with score fusion from both paths"""
    
    def __init__(self, trial_file, enrollment_manifest, emb_stage1, emb_stage2, device):
        self.trial_file = trial_file
        self.device = device
        self.emb_stage1 = emb_stage1
        self.emb_stage2 = emb_stage2
        
        # Store embeddings for enrollments (averaged from multiple samples)
        self.enrollments_stage1 = {}   # stage1_asv averaged per enrollment ID
        self.enrollments_stage2 = {}   # stage2_lid averaged per enrollment ID
        
        self.load_enrollment_manifest(enrollment_manifest)
        self.load_trial_file(trial_file)
    
    def get_embedding_path(self, base_dir, wav_path):
        """Convert wav_path to embedding path, check both train and dev folders"""
        emb_name = wav_path.replace('.wav', '.npy')
        
        for fold in ['train', 'dev']:
            emb_path = os.path.join(base_dir, fold, emb_name)
            if os.path.exists(emb_path):
                return emb_path
        return None
    
    def load_enrollment_manifest(self, enrollment_manifest):
        """Load and average enrollment embeddings from both stages"""
        with open(enrollment_manifest, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    enroll_id = parts[0]
                    enroll_wav_paths = parts[1:]
                    
                    embs_stage1 = []
                    embs_stage2 = []
                    
                    for wav_path in enroll_wav_paths:
                        # Get embeddings from both stages
                        emb1_path = self.get_embedding_path(self.emb_stage1, wav_path)
                        emb2_path = self.get_embedding_path(self.emb_stage2, wav_path)
                        
                        if emb1_path and os.path.exists(emb1_path):
                            try:
                                emb1 = np.load(emb1_path).astype(np.float32)
                                embs_stage1.append(emb1)
                            except:
                                pass
                        
                        if emb2_path and os.path.exists(emb2_path):
                            try:
                                emb2 = np.load(emb2_path).astype(np.float32)
                                embs_stage2.append(emb2)
                            except:
                                pass
                    
                    # Average embeddings
                    if embs_stage1:
                        self.enrollments_stage1[enroll_id] = np.mean(embs_stage1, axis=0)
                    if embs_stage2:
                        self.enrollments_stage2[enroll_id] = np.mean(embs_stage2, axis=0)
    
    def load_trial_file(self, trial_file):
        """Load trial pairs"""
        self.trials = []
        with open(trial_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = int(parts[0])
                    enroll_id = parts[1]
                    test_wav = parts[2]
                    self.trials.append((label, enroll_id, test_wav))
    
    def _eer_from_scores(self, scores, labels):
        """Compute EER from scores and binary labels"""
        if len(scores) == 0 or len(set(labels)) < 2:
            return 1.0
        
        thresholds = np.linspace(min(scores) - 0.1, max(scores) + 0.1, 1000)
        min_eer = float('inf')
        
        for thr in thresholds:
            preds = (scores > thr).astype(int)
            n_neg = np.sum(labels == 0)
            n_pos = np.sum(labels == 1)
            
            if n_neg > 0 and n_pos > 0:
                far = np.sum((preds == 1) & (labels == 0)) / n_neg
                frr = np.sum((preds == 0) & (labels == 1)) / n_pos
                eer = (far + frr) / 2.0
                
                if eer < min_eer:
                    min_eer = eer
        
        return min_eer
    
    def evaluate_eer(self, model, device):
        """
        Extract embeddings from both paths and try different score fusions.
        For LANGUAGE IDENTIFICATION: Remove/subtract speaker information.
        Returns best EER among different combinations.
        """
        model.eval()
        
        sub_scores = []   # Path A (Subtractive) - speaker removed
        add_scores = []   # Path B (Additive) - enriched
        lid_scores = []   # Raw language embedding scores
        asv_scores = []   # Raw speaker embedding scores (for removal)
        all_labels = []
        
        with torch.no_grad():
            for label, enroll_id, test_wav in tqdm(self.trials, desc='EER Evaluation', leave=False):
                try:
                    # Check enrollment exists
                    if enroll_id not in self.enrollments_stage1 or enroll_id not in self.enrollments_stage2:
                        continue
                    
                    # Load test embeddings
                    test_emb1_path = self.get_embedding_path(self.emb_stage1, test_wav)
                    test_emb2_path = self.get_embedding_path(self.emb_stage2, test_wav)
                    
                    if not test_emb1_path or not test_emb2_path:
                        continue
                    
                    test_emb1 = np.load(test_emb1_path).astype(np.float32)  # stage1_asv
                    test_emb2 = np.load(test_emb2_path).astype(np.float32)  # stage2_lid
                    
                    # Convert to tensors
                    test_spk_emb = torch.FloatTensor(test_emb1).unsqueeze(0).to(device)
                    test_lang_emb = torch.FloatTensor(test_emb2).unsqueeze(0).to(device)
                    
                    # Forward through model to get both path embeddings
                    sub_logits, add_logits, sub_emb, add_emb = model(test_spk_emb, test_lang_emb)
                    
                    # Get enrollment embeddings through both paths
                    enroll_spk_emb = torch.FloatTensor(self.enrollments_stage1[enroll_id]).unsqueeze(0).to(device)
                    enroll_lang_emb = torch.FloatTensor(self.enrollments_stage2[enroll_id]).unsqueeze(0).to(device)
                    
                    sub_enroll_logits, add_enroll_logits, sub_enroll_emb, add_enroll_emb = model(
                        enroll_spk_emb, enroll_lang_emb
                    )
                    
                    # Compute similarity scores
                    sub_score = F.cosine_similarity(sub_emb, sub_enroll_emb).item()
                    add_score = F.cosine_similarity(add_emb, add_enroll_emb).item()
                    lid_score = F.cosine_similarity(test_lang_emb, enroll_lang_emb).item()
                    asv_score = F.cosine_similarity(test_spk_emb, enroll_spk_emb).item()
                    
                    sub_scores.append(sub_score)
                    add_scores.append(add_score)
                    lid_scores.append(lid_score)
                    asv_scores.append(asv_score)
                    all_labels.append(label)
                    
                except Exception as e:
                    continue
        
        if len(all_labels) == 0:
            logging.warning("No valid trials for EER evaluation")
            return 1.0
        
        scores_sub = np.array(sub_scores)
        scores_add = np.array(add_scores)
        scores_lid = np.array(lid_scores)
        scores_asv = np.array(asv_scores)
        labels_arr = np.array(all_labels)
        
        logging.info(f"EER evaluation: {len(all_labels)} valid trials")
        
        # Try different fusion combinations for LANGUAGE IDENTIFICATION
        # Goal: Remove speaker contamination from language scores
        results = {}
        
        # Individual paths
        results['sub'] = self._eer_from_scores(scores_sub, labels_arr)
        results['add'] = self._eer_from_scores(scores_add, labels_arr)
        results['lid'] = self._eer_from_scores(scores_lid, labels_arr)
        results['asv'] = self._eer_from_scores(scores_asv, labels_arr)
        
        # Pairwise combinations with speaker subtraction
        betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        # Sub - ASV (remove speaker from subtractive path)
        best_eer = 1.0
        for b in betas:
            combined = scores_sub - b * scores_asv
            eer = self._eer_from_scores(combined, labels_arr)
            if eer < best_eer:
                best_eer = eer
        results['sub-asv'] = best_eer
        
        # Add - ASV (remove speaker from additive path)
        best_eer = 1.0
        for b in betas:
            combined = scores_add - b * scores_asv
            eer = self._eer_from_scores(combined, labels_arr)
            if eer < best_eer:
                best_eer = eer
        results['add-asv'] = best_eer
        
        # LID - ASV (remove speaker from raw language embedding)
        best_eer = 1.0
        for b in betas:
            combined = scores_lid - b * scores_asv
            eer = self._eer_from_scores(combined, labels_arr)
            if eer < best_eer:
                best_eer = eer
        results['lid-asv'] = best_eer
        
        # (Sub + Add) - ASV (fused paths minus speaker)
        alphas = np.linspace(0.1, 0.9, 9)
        best_eer = 1.0
        for a in alphas:
            for b in betas:
                combined = a * scores_sub + (1 - a) * scores_add - b * scores_asv
                eer = self._eer_from_scores(combined, labels_arr)
                if eer < best_eer:
                    best_eer = eer
        results['(sub+add)-asv'] = best_eer
        
        # LID + Sub + Add (all paths combined)
        best_eer = 1.0
        for a in [0.2, 0.3, 0.4, 0.5]:
            for b in [0.2, 0.3, 0.4, 0.5]:
                c = 1.0 - a - b
                if c > 0.05:
                    combined = a * scores_lid + b * scores_sub + c * scores_add
                    eer = self._eer_from_scores(combined, labels_arr)
                    if eer < best_eer:
                        best_eer = eer
        results['lid+sub+add'] = best_eer
        
        # LID + Sub + Add - ASV (all paths minus speaker contamination)
        best_eer = 1.0
        for a in [0.2, 0.3, 0.4, 0.5]:
            for b in [0.2, 0.3, 0.4, 0.5]:
                for d in betas:
                    c = 1.0 - a - b
                    if c > 0.05:
                        combined = a * scores_lid + b * scores_sub + c * scores_add - d * scores_asv
                        eer = self._eer_from_scores(combined, labels_arr)
                        if eer < best_eer:
                            best_eer = eer
        results['lid+sub+add-asv'] = best_eer
        
        # Return best EER and log all results
        best_eer = min(results.values())
        logging.info(f"EER Results (Language ID - ASV-focused): {results}")
        logging.info(f"Best EER: {best_eer:.4f}")
        
        return best_eer

# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_accuracy(model, data_loader, device, desc='Validation'):
    """Compute macro and micro accuracy"""
    model.eval()
    
    all_preds_sub = []
    all_preds_add = []
    all_labels = []
    per_lang_correct_sub = defaultdict(int)
    per_lang_correct_add = defaultdict(int)
    per_lang_total = defaultdict(int)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'{desc}', leave=False):
            speaker_emb = batch['speaker_emb'].to(device)
            language_emb = batch['language_emb'].to(device)
            labels = batch['label'].to(device)
            languages = batch['language']
            
            sub_logits, add_logits, _, _ = model(speaker_emb, language_emb)
            
            sub_preds = torch.argmax(sub_logits, dim=1)
            add_preds = torch.argmax(add_logits, dim=1)
            
            all_preds_sub.extend(sub_preds.cpu().numpy())
            all_preds_add.extend(add_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for pred_sub, pred_add, label, lang in zip(sub_preds, add_preds, labels, languages):
                per_lang_correct_sub[lang] += (pred_sub.item() == label.item())
                per_lang_correct_add[lang] += (pred_add.item() == label.item())
                per_lang_total[lang] += 1
    
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Path A accuracy
    sub_micro_acc = np.mean(np.array(all_preds_sub) == np.array(all_labels))
    sub_macro_acc = np.mean([
        per_lang_correct_sub[lang] / per_lang_total[lang]
        for lang in per_lang_total
    ])
    
    # Path B accuracy
    add_micro_acc = np.mean(np.array(all_preds_add) == np.array(all_labels))
    add_macro_acc = np.mean([
        per_lang_correct_add[lang] / per_lang_total[lang]
        for lang in per_lang_total
    ])
    
    # Return order matches training loop unpacking: macro first, then micro.
    return sub_macro_acc, sub_micro_acc, add_macro_acc, add_micro_acc

# ============================================================================
# Training Loop
# ============================================================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_file', required=True)
    parser.add_argument('--emb_stage1_base', required=True)
    parser.add_argument('--emb_stage2_base', required=True)
    parser.add_argument('--trial_file', required=True)
    parser.add_argument('--enrollment_manifest', required=True)
    parser.add_argument('--output_dir', required=True)
    
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
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--lr_decay_step', type=int, default=5)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--hard_neg_margin', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    log_file = os.path.join(args.output_dir, 'train.log')
    logger = setup_logging(log_file)
    
    logger.info("=" * 70)
    logger.info("Language Embedding Extractor - Dual-Path Fusion + Manifest Loading")
    logger.info("=" * 70)
    
    # Load manifest
    manifest_loader = ManifestDataLoader(args.manifest_file)
    
    # Create datasets
    train_dataset = EmbeddingDataset(
        manifest_loader.train_data, args.emb_stage1_base, args.emb_stage2_base, manifest_loader, 'train'
    )
    val_dataset = EmbeddingDataset(
        manifest_loader.val_data, args.emb_stage1_base, args.emb_stage2_base, manifest_loader, 'val'
    )
    val2_dataset = EmbeddingDataset(
        manifest_loader.val2_data, args.emb_stage1_base, args.emb_stage2_base, manifest_loader, 'val2'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val2_loader = DataLoader(val2_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = DualPathLanguageExtractor(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        subspace_dim=args.subspace_dim,
        num_heads=args.num_heads,
        num_classes=manifest_loader.num_classes,
        dropout=args.dropout,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    
    # Trial evaluator
    trial_evaluator = TrialEvaluator(
        args.trial_file, args.enrollment_manifest, args.emb_stage1_base, args.emb_stage2_base, device
    )
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    metrics_history = []
    best_acc = 0.0
    best_eer = 1.0
    best_acc_epoch = -1
    best_eer_epoch = -1
    best_acc_ckpt = os.path.join(args.output_dir, 'best_acc.pth')
    best_eer_ckpt = os.path.join(args.output_dir, 'best_eer.pth')
    
    for epoch in tqdm(range(args.epochs), desc='Training', unit='epoch'):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', leave=False):
            speaker_emb = batch['speaker_emb'].to(device)
            language_emb = batch['language_emb'].to(device)
            labels = batch['label'].to(device)
            
            sub_logits, add_logits, sub_emb, add_emb = model(speaker_emb, language_emb)
            
            # Combined loss
            loss = (F.cross_entropy(sub_logits, labels) + F.cross_entropy(add_logits, labels)) / 2.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation
        if epoch % args.eval_every == 0:
            sub_val_macro, sub_val_micro, add_val_macro, add_val_micro = evaluate_accuracy(
                model, val_loader, device, f'Epoch {epoch+1} [Val]'
            )
            sub_val2_macro, sub_val2_micro, add_val2_macro, add_val2_micro = evaluate_accuracy(
                model, val2_loader, device, f'Epoch {epoch+1} [Val2]'
            )
            
            # EER
            eer = trial_evaluator.evaluate_eer(model, device)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | EER: {eer:.4f}"
            )
            logger.info(
                f"  Val  | Sub: micro={sub_val_micro:.4f}, macro={sub_val_macro:.4f} | "
                f"Add: micro={add_val_micro:.4f}, macro={add_val_macro:.4f}"
            )
            logger.info(
                f"  Val2 | Sub: micro={sub_val2_micro:.4f}, macro={sub_val2_macro:.4f} | "
                f"Add: micro={add_val2_micro:.4f}, macro={add_val2_macro:.4f}"
            )
            
            metrics_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'val_micro': sub_val_micro,
                'val_macro': sub_val_macro,
                'val_add_micro': add_val_micro,
                'val_add_macro': add_val_macro,
                'val2_micro': sub_val2_micro,
                'val2_macro': sub_val2_macro,
                'val2_add_micro': add_val2_micro,
                'val2_add_macro': add_val2_macro,
                'eer': eer
            })
            
            # Checkpoint
            if sub_val_micro > best_acc:
                best_acc = sub_val_micro
                best_acc_epoch = epoch + 1
                torch.save(model.state_dict(), best_acc_ckpt)
                logger.info(
                    f"  [BEST-ACC] Updated at epoch {best_acc_epoch}: "
                    f"sub_val_micro={best_acc:.4f} -> {best_acc_ckpt}"
                )
            
            if eer < best_eer:
                best_eer = eer
                best_eer_epoch = epoch + 1
                torch.save(model.state_dict(), best_eer_ckpt)
                logger.info(
                    f"  [BEST-EER] Updated at epoch {best_eer_epoch}: "
                    f"eer={best_eer:.4f} -> {best_eer_ckpt}"
                )
        
        scheduler.step()
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)

    summary = {
        'best_acc': {
            'metric': 'sub_val_micro',
            'value': best_acc,
            'epoch': best_acc_epoch,
            'checkpoint': best_acc_ckpt
        },
        'best_eer': {
            'metric': 'eer',
            'value': best_eer,
            'epoch': best_eer_epoch,
            'checkpoint': best_eer_ckpt
        }
    }
    summary_path = os.path.join(args.output_dir, 'best_checkpoints.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(
        f"Best by accuracy: epoch={best_acc_epoch}, sub_val_micro={best_acc:.4f}, "
        f"checkpoint={best_acc_ckpt}"
    )
    logger.info(
        f"Best by EER:      epoch={best_eer_epoch}, eer={best_eer:.4f}, "
        f"checkpoint={best_eer_ckpt}"
    )
    logger.info(f"Checkpoint summary saved to: {summary_path}")

if __name__ == '__main__':
    train()
