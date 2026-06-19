"""
Evaluation script for Simple SV model trained with WPT + W2V-BERT-2.0 + MHFA
============================================================================

This script evaluates a trained model on the evaluation dataset.

Usage:
    python eval_simple_sv_wpt_w2vbert_mhfa.py \
        --checkpoint ./ckpt_simple_sv/best_model.pt \
        --eval_audio /path/to/eval/audio \
        --eval_label /path/to/eval/labels.csv \
        --trial_file /path/to/trials.txt \
        --xlsr facebook/w2v-bert-2.0 \
        --gpu 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import csv
import numpy as np
import math
from transformers import Wav2Vec2BertModel, AutoFeatureExtractor, AutoConfig
from dataset_asv import SpoofCelebASV
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses import ArcFaceLoss
import sys

# Import model classes from training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_train_simple_sv_wpt_w2vbert_mhfa import (
    WPTW2VBERTMultiLayer,
    compute_eer
)


class MHFAHead(nn.Module):
    """
    Basic Multi-Head Factorized Attention (MHFA) Head
    (Without adapter and deep embedding MLP)
    
    This is the original version used for training.
    """
    
    def __init__(self, feature_dim=1024, num_layers=24, num_heads=8, 
                 compression_dim=128, embedding_dim=256, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.compression_dim = compression_dim
        self.embedding_dim = embedding_dim
        
        # Layer-wise attention weights for Key and Value streams (factorized)
        self.layer_weights_key = nn.Parameter(torch.zeros(num_layers))
        self.layer_weights_value = nn.Parameter(torch.zeros(num_layers))
        
        # Initialize weights uniformly
        nn.init.uniform_(self.layer_weights_key.data, -0.1, 0.1)
        nn.init.uniform_(self.layer_weights_value.data, -0.1, 0.1)
        
        # Dimension compression: feature_dim → compression_dim
        self.key_projection = nn.Linear(feature_dim, compression_dim)
        self.value_projection = nn.Linear(feature_dim, compression_dim)
        
        # Attention projection for multi-head attention
        self.attention_projection = nn.Linear(compression_dim, num_heads)
        
        # Embedding projection: (num_heads * compression_dim) → embedding_dim
        self.embedding_projection = nn.Sequential(
            nn.Linear(num_heads * compression_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, layer_features):
        """
        Forward pass through MHFA head
        
        Args:
            layer_features: List of (B, T, D) tensors, one per layer
                           or Tensor of shape (L, B, T, D)
        
        Returns:
            embeddings: (B, embedding_dim) - utterance-level embeddings
        """
        # Convert list to tensor if needed: (L, B, T, D)
        if isinstance(layer_features, list):
            layer_features = torch.stack(layer_features, dim=0)
        
        L, B, T, D = layer_features.shape
        assert L == self.num_layers, f"Expected {self.num_layers} layers, got {L}"
        assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"
        
        # Step 1: Layer-wise weighted aggregation
        w_k = F.softmax(self.layer_weights_key, dim=0)  # (L,)
        w_v = F.softmax(self.layer_weights_value, dim=0)  # (L,)
        
        w_k = w_k.view(L, 1, 1, 1)
        w_v = w_v.view(L, 1, 1, 1)
        
        K_feat = (layer_features * w_k).sum(dim=0)  # (B, T, D)
        V_feat = (layer_features * w_v).sum(dim=0)  # (B, T, D)
        
        # Step 2: Dimension compression
        K = self.key_projection(K_feat)  # (B, T, compression_dim)
        V = self.value_projection(V_feat)  # (B, T, compression_dim)
        
        K = self.dropout(K)
        V = self.dropout(V)
        
        # Step 3: Multi-head attention pooling
        A = self.attention_projection(K)  # (B, T, num_heads)
        A = F.softmax(A, dim=1)  # (B, T, num_heads)
        
        # Apply attention to Value stream
        A = A.transpose(1, 2).unsqueeze(-1)  # (B, num_heads, T, 1)
        V = V.unsqueeze(1)  # (B, 1, T, compression_dim)
        
        # Weighted pooling: (B, num_heads, compression_dim)
        pooled = (A * V).sum(dim=2)  # (B, num_heads, compression_dim)
        
        # Flatten heads: (B, num_heads * compression_dim)
        pooled = pooled.view(B, self.num_heads * self.compression_dim)
        
        # Step 4: Embedding projection
        embeddings = self.embedding_projection(pooled)  # (B, embedding_dim)
        
        return embeddings


class MHFAHeadImproved(nn.Module):
    """Improved Multi-Head Factorized Attention (MHFA) Head with Adapter + Deep MLP"""
    
    def __init__(self, feature_dim=1024, num_layers=24, num_heads=8, 
                 compression_dim=128, embedding_dim=256, adapter_bottleneck=128,
                 dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.compression_dim = compression_dim
        self.embedding_dim = embedding_dim
        self.adapter_bottleneck = adapter_bottleneck
        
        # Layer-wise attention weights
        self.layer_weights_key = nn.Parameter(torch.zeros(num_layers))
        self.layer_weights_value = nn.Parameter(torch.zeros(num_layers))
        
        nn.init.uniform_(self.layer_weights_key.data, -0.1, 0.1)
        nn.init.uniform_(self.layer_weights_value.data, -0.1, 0.1)
        
        # Dimension compression
        self.key_projection = nn.Linear(feature_dim, compression_dim)
        self.value_projection = nn.Linear(feature_dim, compression_dim)
        
        # Attention projection
        self.attention_projection = nn.Linear(compression_dim, num_heads)
        
        # Embedding projection
        self.embedding_projection = nn.Sequential(
            nn.Linear(num_heads * compression_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Adapter module with residual connection
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, adapter_bottleneck),
            nn.BatchNorm1d(adapter_bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_bottleneck, embedding_dim),
        )
        self.adapter_norm = nn.LayerNorm(embedding_dim)
        
        # Deep embedding MLP
        self.embedding_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, layer_features):
        if isinstance(layer_features, list):
            layer_features = torch.stack(layer_features, dim=0)
        
        L, B, T, D = layer_features.shape
        
        # Layer-wise weighted aggregation
        w_k = F.softmax(self.layer_weights_key, dim=0).view(L, 1, 1, 1)
        w_v = F.softmax(self.layer_weights_value, dim=0).view(L, 1, 1, 1)
        
        K_feat = (layer_features * w_k).sum(dim=0)
        V_feat = (layer_features * w_v).sum(dim=0)
        
        # Dimension compression
        K = self.dropout(self.key_projection(K_feat))
        V = self.dropout(self.value_projection(V_feat))
        
        # Multi-head attention pooling
        A = F.softmax(self.attention_projection(K), dim=1)
        A = A.transpose(1, 2).unsqueeze(-1)
        V = V.unsqueeze(1)
        
        pooled = (A * V).sum(dim=2).view(B, self.num_heads * self.compression_dim)
        
        # Embedding projection
        embeddings = self.embedding_projection(pooled)
        
        # Adapter module with residual
        adapted = self.adapter(embeddings)
        adapted = self.adapter_norm(adapted + embeddings)
        
        # Deep embedding MLP
        embeddings = self.embedding_layer(adapted)
        
        return embeddings


class SimpleSVModelWPTW2VBERTMHFA(nn.Module):
    """
    Simple Speaker Verification with WPT + W2V-BERT-2.0 + MHFA Head
    """
    
    def __init__(self, model_dir, num_speakers, embedding_dim=256, 
                 num_prompt_tokens=6, num_wavelet_tokens=4, prompt_dropout=0.1,
                 num_heads=8, compression_dim=128, adapter_bottleneck=None, head_dropout=0.1,
                 use_arcface=True, arcface_margin=0.3, arcface_scale=30.0):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.use_arcface = use_arcface
        
        # Load W2V-BERT-2.0 with WPT
        self.wpt_w2vbert = WPTW2VBERTMultiLayer(
            model_dir=model_dir,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens=num_wavelet_tokens,
            prompt_dim=1024,
            dropout=prompt_dropout
        )
        
        # Get number of layers from config
        num_layers = self.wpt_w2vbert.config.num_hidden_layers
        
        # Choose MHFA Head version based on adapter_bottleneck
        if adapter_bottleneck is not None:
            # Improved MHFA Head with Adapter + Deep MLP
            self.mhfa_head = MHFAHeadImproved(
                feature_dim=1024,
                num_layers=num_layers,
                num_heads=num_heads,
                compression_dim=compression_dim,
                embedding_dim=embedding_dim,
                adapter_bottleneck=adapter_bottleneck,
                dropout=head_dropout
            )
        else:
            # Basic MHFA Head
            self.mhfa_head = MHFAHead(
                feature_dim=1024,
                num_layers=num_layers,
                num_heads=num_heads,
                compression_dim=compression_dim,
                embedding_dim=embedding_dim,
                dropout=head_dropout
            )
        
        # Classifier
        if use_arcface:
            self.arcface_loss = ArcFaceLoss(
                in_features=embedding_dim,
                out_features=num_speakers,
                scale=arcface_scale,
                margin=arcface_margin,
                easy_margin=False
            )
            self.classifier = None
        else:
            self.classifier = nn.Linear(embedding_dim, num_speakers)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
            self.arcface_loss = None
    
    def extract_embedding(self, audio_data, normalize=True):
        """Extract speaker embedding"""
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        
        # Get features from all layers
        layer_features = self.wpt_w2vbert(audio_data)
        
        # Pass through MHFA head
        embeddings = self.mhfa_head(layer_features)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, audio_data, labels=None):
        """Forward pass - returns embeddings and logits"""
        embeddings_unnorm = self.extract_embedding(audio_data, normalize=False)
        embeddings_norm = F.normalize(embeddings_unnorm, p=2, dim=1)
        
        # Classify
        if self.use_arcface and self.arcface_loss is not None:
            weight = F.normalize(self.arcface_loss.weight, p=2, dim=1)
            cosine = F.linear(embeddings_norm, weight)
            
            if labels is not None:
                loss, logits = self.arcface_loss(embeddings_norm, labels)
                return embeddings_norm, embeddings_unnorm, logits, loss
            else:
                logits = cosine * self.arcface_loss.scale
                return embeddings_norm, embeddings_unnorm, logits
        else:
            logits = self.classifier(embeddings_unnorm)
            return embeddings_norm, embeddings_unnorm, logits

torch.set_default_dtype(torch.float32)


def load_model_from_checkpoint(checkpoint_path, args, device):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        args: Arguments containing model configuration
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load args from checkpoint directory if not provided
    checkpoint_dir = os.path.dirname(checkpoint_path)
    args_file = os.path.join(checkpoint_dir, 'args.json')
    
    # Default values
    defaults = {
        'xlsr': 'facebook/w2v-bert-2.0',
        'num_prompt_tokens': 6,
        'num_wavelet_tokens': 4,
        'embedding_dim': 256,
        'num_heads': 8,
        'compression_dim': 128,
        'adapter_bottleneck': None,  # None for basic MHFA, 128 for improved MHFA
        'use_arcface': False,
        'arcface_margin': 0.3,
        'arcface_scale': 30.0
    }
    
    if os.path.exists(args_file):
        print(f"Loading model arguments from {args_file}...")
        try:
            with open(args_file, 'r') as f:
                saved_args = json.load(f)
            
            # Override with saved args if not provided in command line
            for key, default_value in defaults.items():
                current_value = getattr(args, key, None)
                if current_value is None:
                    saved_value = saved_args.get(key, default_value)
                    # Convert boolean from JSON (might be stored as bool or string)
                    if key == 'use_arcface' and isinstance(saved_value, str):
                        saved_value = saved_value.lower() == 'true'
                    setattr(args, key, saved_value)
        except Exception as e:
            print(f"  Warning: Could not load args.json: {e}")
            print(f"  Using defaults and command-line arguments")
            # Set defaults for missing values
            for key, default_value in defaults.items():
                if getattr(args, key, None) is None:
                    setattr(args, key, default_value)
    else:
        print(f"  args.json not found, using defaults and command-line arguments")
        # Set defaults for missing values
        for key, default_value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, default_value)
    
    # Ensure use_arcface is boolean
    if isinstance(args.use_arcface, str):
        args.use_arcface = args.use_arcface.lower() == 'true'
    
    # Get number of speakers from checkpoint (critical for loading ArcFace weights)
    if 'model_state_dict' in checkpoint:
        # Extract from arcface_loss.weight or classifier.weight
        if 'arcface_loss.weight' in checkpoint['model_state_dict']:
            num_speakers = checkpoint['model_state_dict']['arcface_loss.weight'].shape[0]
            print(f"  Detected {num_speakers} speakers from checkpoint (ArcFace)")
        elif 'classifier.weight' in checkpoint['model_state_dict']:
            num_speakers = checkpoint['model_state_dict']['classifier.weight'].shape[0]
            print(f"  Detected {num_speakers} speakers from checkpoint (Linear)")
        else:
            num_speakers = args.num_speakers if args.num_speakers else 1000
            print(f"  Warning: Could not detect number of speakers, using {num_speakers}")
    else:
        num_speakers = args.num_speakers if args.num_speakers else 1000
        print(f"  Using {num_speakers} speakers")
    
    # Detect model architecture from checkpoint state dict
    # Check if checkpoint has adapter layers (Improved MHFA) or not (Basic MHFA)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        has_adapter = any('mhfa_head.adapter.' in key for key in state_dict.keys())
        
        if has_adapter:
            # Checkpoint has adapter layers - should use Improved MHFA
            if args.adapter_bottleneck is None:
                print(f"  Warning: Checkpoint has adapter layers but args.adapter_bottleneck is None")
                print(f"  Setting adapter_bottleneck to 128 (default for Improved MHFA)")
                args.adapter_bottleneck = 128
        else:
            # Checkpoint does NOT have adapter layers - should use Basic MHFA
            if args.adapter_bottleneck is not None:
                print(f"  Warning: Checkpoint does NOT have adapter layers but args.adapter_bottleneck={args.adapter_bottleneck}")
                print(f"  Setting adapter_bottleneck to None (Basic MHFA) to match checkpoint")
                args.adapter_bottleneck = None
    else:
        # Can't detect from state dict, use args value
        pass
    
    # Initialize model
    print(f"\nInitializing model...")
    if args.adapter_bottleneck is not None:
        print(f"  Using improved MHFA head (adapter_bottleneck={args.adapter_bottleneck})")
    else:
        print(f"  Using basic MHFA head")
    
    model = SimpleSVModelWPTW2VBERTMHFA(
        model_dir=args.xlsr,
        num_speakers=num_speakers,
        embedding_dim=args.embedding_dim,
        num_prompt_tokens=args.num_prompt_tokens,
        num_wavelet_tokens=args.num_wavelet_tokens,
        prompt_dropout=args.prompt_dropout if hasattr(args, 'prompt_dropout') else 0.1,
        num_heads=args.num_heads,
        compression_dim=args.compression_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        head_dropout=args.head_dropout if hasattr(args, 'head_dropout') else 0.1,
        use_arcface=args.use_arcface,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale
    ).to(device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded model state from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'eer' in checkpoint:
            print(f"  ✓ Checkpoint EER: {checkpoint['eer']:.4f}%")
    else:
        # Try loading directly (in case checkpoint is just the state dict)
        model.load_state_dict(checkpoint)
        print(f"  ✓ Loaded model state")
    
    model.eval()
    
    return model


def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings for all samples in dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation dataset
        device: Device to use
    
    Returns:
        embeddings: List of embeddings (numpy arrays)
        filenames: List of filenames
        speaker_labels: List of speaker labels
    """
    print("\nExtracting embeddings...")
    
    all_embeddings = []
    all_filenames = []
    all_speaker_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            waveform, filename, speaker_labels = batch
            waveform = waveform.to(device)
            
            # Extract normalized embeddings
            embeddings = model.extract_embedding(waveform, normalize=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_filenames.extend(filename)
            all_speaker_labels.append(speaker_labels.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    speaker_labels = np.concatenate(all_speaker_labels)
    
    print(f"  Extracted {len(embeddings)} embeddings")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, all_filenames, speaker_labels


def evaluate_with_trial_file(embeddings, filenames, trial_file):
    """
    Evaluate using trial file
    
    Supports two formats:
    1. Space-separated: label enroll_file test_file (label is integer: 1=target, 0=non-target)
    2. CSV format: enroll_file,test_file,label (label is string: 'target', 'nontarget', or 'spoof')
       Note: 'spoof' trials are automatically filtered out
    
    Args:
        embeddings: Numpy array of embeddings
        filenames: List of filenames corresponding to embeddings
        trial_file: Path to trial file
    
    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
        target_scores: Target scores
        nontarget_scores: Non-target scores
    """
    print(f"\nEvaluating with trial file: {trial_file}")
    
    # Create filename to embedding mapping
    filename_to_emb = {}
    for fn, emb in zip(filenames, embeddings):
        # Store with basename (without extension) as key
        basename = os.path.splitext(os.path.basename(fn))[0]
        filename_to_emb[basename] = emb
    
    print(f"  Created embedding map for {len(filename_to_emb)} files")
    
    # Load trials and compute scores
    target_scores = []
    nontarget_scores = []
    all_trials = []  # Track all trial pairs with scores
    missing_files = set()
    spoof_count = 0
    
    # Detect format by reading first line
    with open(trial_file, 'r') as f:
        first_line = f.readline().strip()
        # Check if it's CSV format (contains comma)
        is_csv = ',' in first_line
        f.seek(0)  # Reset to beginning
        
        for line in tqdm(f, desc="Processing trials"):
            if is_csv:
                # CSV format: enroll_file,test_file,label
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                
                enroll_file, test_file, label = parts
                label = label.strip().lower()
                
                # Skip spoof trials (SASV protocol)
                if label == 'spoof':
                    spoof_count += 1
                    continue
                
                # Check if target or nontarget
                # Label can be: 'target', 'nontarget', or 'spoof' (already filtered)
                if label == 'target':
                    is_target = True
                elif label == 'nontarget':
                    is_target = False
                else:
                    # Unknown label, skip
                    continue
            else:
                # Space-separated format: label enroll_file test_file
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                
                label = int(parts[0])
                enroll_file = parts[1]
                test_file = parts[2]
                is_target = (label == 1)
            
            enroll_basename = os.path.splitext(os.path.basename(enroll_file))[0]
            test_basename = os.path.splitext(os.path.basename(test_file))[0]
            
            if enroll_basename not in filename_to_emb:
                missing_files.add(enroll_basename)
                continue
            if test_basename not in filename_to_emb:
                missing_files.add(test_basename)
                continue
            
            # Compute cosine similarity (dot product for normalized embeddings)
            score = np.dot(filename_to_emb[enroll_basename], filename_to_emb[test_basename])
            
            # Store trial with score
            trial_label = 'target' if is_target else 'nontarget'
            all_trials.append({
                'enroll_file': enroll_basename,
                'test_file': test_basename,
                'score': float(score),
                'label': trial_label
            })
            
            if is_target:
                target_scores.append(score)
            else:
                nontarget_scores.append(score)
    
    if spoof_count > 0:
        print(f"  Skipped {spoof_count} spoof trials (SASV protocol)")
    
    if missing_files:
        print(f"  Warning: {len(missing_files)} files from trial file not found in dataset")
        if len(missing_files) <= 10:
            print(f"    Missing files: {list(missing_files)[:10]}")
    
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        print("  Error: No valid trials found")
        print(f"    Target scores: {len(target_scores)}, Non-target scores: {len(nontarget_scores)}")
        return 100.0, 0.0, [], [], []
    
    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)
    
    print(f"  Target trials: {len(target_scores)}")
    print(f"  Non-target trials: {len(nontarget_scores)}")
    
    eer, threshold = compute_eer(target_scores, nontarget_scores)
    
    return eer, threshold, target_scores, nontarget_scores, all_trials


def evaluate_with_sampled_pairs(embeddings, speaker_labels, max_pairs=100000):
    """
    Evaluate by sampling pairs from embeddings
    
    Args:
        embeddings: Numpy array of embeddings
        speaker_labels: Numpy array of speaker labels
        max_pairs: Maximum number of pairs to sample
    
    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
        target_scores: Target scores
        nontarget_scores: Non-target scores
        all_trials: List of trial dicts with scores
    """
    print(f"\nEvaluating with sampled pairs (max {max_pairs})...")
    
    num_samples = embeddings.shape[0]
    total_pairs = num_samples * (num_samples - 1) // 2
    
    print(f"  Total possible pairs: {total_pairs}")
    
    all_trials = []
    
    if total_pairs > max_pairs:
        # Sample pairs randomly
        np.random.seed(42)  # For reproducibility
        idx_i, idx_j = np.triu_indices(num_samples, k=1)
        sample_indices = np.random.choice(len(idx_i), max_pairs, replace=False)
        idx_i_sampled = idx_i[sample_indices]
        idx_j_sampled = idx_j[sample_indices]
        
        # Compute scores for sampled pairs (vectorized)
        emb_i = embeddings[idx_i_sampled]  # (max_pairs, emb_dim)
        emb_j = embeddings[idx_j_sampled]  # (max_pairs, emb_dim)
        pair_scores = np.sum(emb_i * emb_j, axis=1)  # (max_pairs,)
        same_speaker = speaker_labels[idx_i_sampled] == speaker_labels[idx_j_sampled]
        
        # Track all trials
        for i, (sp_i, sp_j, score) in enumerate(zip(idx_i_sampled, idx_j_sampled, pair_scores)):
            trial_label = 'target' if same_speaker[i] else 'nontarget'
            all_trials.append({
                'enroll_file': f'sample_{sp_i}',
                'test_file': f'sample_{sp_j}',
                'score': float(score),
                'label': trial_label
            })
        
        print(f"  Sampled {max_pairs} pairs")
    else:
        # Compute all pairs (vectorized)
        sim_matrix = np.matmul(embeddings, embeddings.T)
        idx_i, idx_j = np.triu_indices(num_samples, k=1)
        pair_scores = sim_matrix[idx_i, idx_j]
        same_speaker = speaker_labels[idx_i] == speaker_labels[idx_j]
        
        # Track all trials
        for sp_i, sp_j, score, is_target in zip(idx_i, idx_j, pair_scores, same_speaker):
            trial_label = 'target' if is_target else 'nontarget'
            all_trials.append({
                'enroll_file': f'sample_{sp_i}',
                'test_file': f'sample_{sp_j}',
                'score': float(score),
                'label': trial_label
            })
        
        print(f"  Computed all {total_pairs} pairs")
    
    target_scores = pair_scores[same_speaker]
    nontarget_scores = pair_scores[~same_speaker]
    
    print(f"  Target pairs: {len(target_scores)}")
    print(f"  Non-target pairs: {len(nontarget_scores)}")
    
    eer, threshold = compute_eer(target_scores, nontarget_scores)
    
    return eer, threshold, target_scores, nontarget_scores, all_trials


def main():
    parser = argparse.ArgumentParser(description='Evaluate Simple SV model (WPT + W2V-BERT-2.0 + MHFA)')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (best_model.pt)')
    
    # Data paths
    parser.add_argument('--eval_audio', type=str, required=True,
                       help='Path to evaluation audio directory')
    parser.add_argument('--eval_label', type=str, required=True,
                       help='Path to evaluation labels CSV file')
    parser.add_argument('--trial_file', type=str, default=None,
                       help='Path to trial file for evaluation (optional). If not provided, will sample pairs.')
    
    # Audio parameters
    parser.add_argument('--audio_len', type=int, default=64600,
                       help='Audio length in samples (default: 64600 = 4.0375s @ 16kHz)')
    
    # Model paths
    parser.add_argument('--xlsr', type=str, default=None,
                       help='Path to W2V-BERT-2.0 model (default: loaded from checkpoint args)')
    
    # Model hyperparameters (will be loaded from checkpoint if not provided)
    parser.add_argument('--num_prompt_tokens', type=int, default=None,
                       help='Number of prompt tokens (default: loaded from checkpoint)')
    parser.add_argument('--num_wavelet_tokens', type=int, default=None,
                       help='Number of wavelet tokens (default: loaded from checkpoint)')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Embedding dimension (default: loaded from checkpoint)')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Number of attention heads (default: loaded from checkpoint)')
    parser.add_argument('--compression_dim', type=int, default=None,
                       help='Compression dimension (default: loaded from checkpoint)')
    parser.add_argument('--num_speakers', type=int, default=None,
                       help='Number of speakers (default: auto-detected from checkpoint)')
    
    # Loss parameters
    parser.add_argument('--use_arcface', type=str, default=None,
                       choices=['true', 'false', 'True', 'False'],
                       help='Use ArcFace loss: true/false (default: loaded from checkpoint)')
    parser.add_argument('--arcface_margin', type=float, default=None,
                       help='ArcFace margin (default: loaded from checkpoint)')
    parser.add_argument('--arcface_scale', type=float, default=None,
                       help='ArcFace scale (default: loaded from checkpoint)')
    
    # Evaluation parameters
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--max_pairs', type=int, default=100000,
                       help='Maximum number of pairs to sample if trial_file not provided')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save evaluation results (JSON format)')
    
    args = parser.parse_args()
    
    # Convert use_arcface string to boolean if provided
    if args.use_arcface is not None:
        args.use_arcface = args.use_arcface.lower() == 'true'
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args, device)
    
    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    eval_dataset = SpoofCelebASV(
        path_to_features=args.eval_audio,
        path_to_protocol=args.eval_label,
        audio_length=args.audio_len,
        bonafide_only=True,
        rawboost=False,
        musanrir=False
    )
    
    print(f"  Evaluation samples: {len(eval_dataset)}")
    print(f"  Number of speakers: {eval_dataset.num_speakers}")
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Extract embeddings
    embeddings, filenames, speaker_labels = extract_embeddings(model, eval_loader, device)
    
    # Evaluate
    if args.trial_file and os.path.exists(args.trial_file):
        eer, threshold, target_scores, nontarget_scores, all_trials = evaluate_with_trial_file(
            embeddings, filenames, args.trial_file
        )
    else:
        if args.trial_file:
            print(f"  Warning: Trial file {args.trial_file} not found. Using sampled pairs instead.")
        eer, threshold, target_scores, nontarget_scores, all_trials = evaluate_with_sampled_pairs(
            embeddings, speaker_labels, max_pairs=args.max_pairs
        )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    print(f"EER: {eer:.4f}%")
    print(f"Threshold: {threshold:.4f}")
    if len(target_scores) > 0:
        print(f"Target scores - Mean: {np.mean(target_scores):.4f}, Std: {np.std(target_scores):.4f}")
        print(f"  Min: {np.min(target_scores):.4f}, Max: {np.max(target_scores):.4f}")
    if len(nontarget_scores) > 0:
        print(f"Non-target scores - Mean: {np.mean(nontarget_scores):.4f}, Std: {np.std(nontarget_scores):.4f}")
        print(f"  Min: {np.min(nontarget_scores):.4f}, Max: {np.max(nontarget_scores):.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        'eer': float(eer),
        'threshold': float(threshold),
        'num_target_trials': len(target_scores),
        'num_nontarget_trials': len(nontarget_scores),
        'target_mean': float(np.mean(target_scores)) if len(target_scores) > 0 else None,
        'target_std': float(np.std(target_scores)) if len(target_scores) > 0 else None,
        'nontarget_mean': float(np.mean(nontarget_scores)) if len(nontarget_scores) > 0 else None,
        'nontarget_std': float(np.std(nontarget_scores)) if len(nontarget_scores) > 0 else None,
    }
    
    if args.output_file:
        # Save trial scores as CSV (primary output for score fusion)
        csv_path = args.output_file.replace('.json', '.csv') if args.output_file.endswith('.json') else args.output_file
        
        if all_trials:
            # Save individual trial pair scores for score fusion
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['enroll_file', 'test_file', 'score', 'label'])
                writer.writeheader()
                for trial in all_trials:
                    writer.writerow(trial)
            print(f"✓ Trial scores saved to: {csv_path}")
            print(f"  Total trials: {len(all_trials)}")
        
        # Also save summary metrics as JSON
        json_path = args.output_file if args.output_file.endswith('.json') else args.output_file + '.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✓ Summary metrics saved to: {json_path}")


if __name__ == '__main__':
    main()
