"""
Simple SV with WPT + W2V-BERT-2.0 + MHFA Head — Multilingual Data + Augmentation
===================================================================================

Based on main_train_simple_sv_wpt_w2vbert_mhfa.py, adapted for:
1. Multilingual ASV training data (speaker/language/wav folder structure)
2. MUSAN + RIR data augmentation (WeSpeaker-style)
3. SpecAugment (time/frequency masking on SSL features)

Architecture (unchanged):
    W2V-BERT-2.0 (frozen) + WPT (Wavelet Prompt Tuning)
        | Extract features from ALL layers
    All L layers: X in R^(L x T x D)
        | MHFA Head (layer-wise weighted aggregation)
    Key stream: K_feat = sum softmax(w^k_l) . Z_l
    Value stream: V_feat = sum softmax(w^v_l) . Z_l
        | Dimension compression + Attention pooling
    Embedding projection (1024->256)
        | Adapter (256->128->256) with residual
        | Deep MLP (256->256->256)
    Final Embedding (256D) -> AAM-Softmax Classifier

Data augmentation:
    1. MUSAN + RIR (raw waveform augmentation via LMDB)
    2. SpecAugment (feature-level time/frequency masking)
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
import math
import random
import torchaudio
from torchaudio import functional as AF
from functools import partial
from transformers import Wav2Vec2BertModel, AutoFeatureExtractor, AutoConfig
from dataset_multilingual_asv import MultilingualASVDataset, pad_dataset, normalize_audio, torchaudio_load
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm
from losses import ArcFaceLoss

# Import WeSpeaker-style augmentation
try:
    from audio_augmentation import AudioAugmentor
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("Warning: audio_augmentation module not found. MUSAN+RIR augmentation disabled.")

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# SpecAugment: feature-level time & frequency masking
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    """
    SpecAugment applied to SSL features of shape (B, T, D).

    Performs:
      - Time masking:  mask `num_time_masks` contiguous blocks of up to `time_mask_param` frames
      - Feature masking: mask `num_feat_masks` contiguous blocks of up to `feat_mask_param` dims

    Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method
               for Automatic Speech Recognition", Interspeech 2019.
    """

    def __init__(self, time_mask_param=20, freq_mask_param=40,
                 num_time_masks=2, num_freq_masks=2):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def forward(self, x):
        """
        Args:
            x: (B, T, D) feature tensor
        Returns:
            Masked feature tensor (B, T, D).  Uses element-wise multiply
            with a binary mask to avoid clone / in-place ops on graph tensors.
        """
        if not self.training:
            return x

        B, T, D = x.shape
        # Build a binary mask on the same device (detached from the graph)
        mask = torch.ones(B, T, D, device=x.device, dtype=x.dtype)

        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, T - 1))
            t0 = random.randint(0, T - t)
            mask[:, t0:t0 + t, :] = 0.0

        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, D - 1))
            f0 = random.randint(0, D - f)
            mask[:, :, f0:f0 + f] = 0.0

        return x * mask


# ---------------------------------------------------------------------------
# Dataset wrapper: adds MUSAN+RIR augmentation to MultilingualASVDataset
# ---------------------------------------------------------------------------

class MultilingualASVDatasetWithAug(Dataset):
    """
    Wraps MultilingualASVDataset and adds WeSpeaker-style MUSAN+RIR augmentation.

    Returns the same 3-tuple expected by the MHFA training loop:
        (waveform, filename, speaker_id)

    Uses raw audio files from RIRS_NOISES and MUSAN folders (not LMDB).
    """

    def __init__(self, base_dataset, musanrir=False,
                 rir_folder=None, noise_folder=None, aug_prob=0.6,
                 variable_length=False, max_eval_dur=60.0,
                 speed_perturbation=None, is_train=True):
        self.base_dataset = base_dataset
        self.musanrir = musanrir
        self.variable_length = variable_length
        self.max_eval_len = int(max_eval_dur * 16000)
        self.speed_perturbation = speed_perturbation if speed_perturbation is not None else []
        self.is_train = is_train

        # Store paths for augmentor
        self._rir_folder = rir_folder
        self._noise_folder = noise_folder
        self._aug_prob = aug_prob

        # Expose key attributes from the base dataset
        self.num_speakers = base_dataset.num_speakers
        self.num_languages = base_dataset.num_languages
        self.speaker_to_id = base_dataset.speaker_to_id
        self.id_to_speaker = base_dataset.id_to_speaker
        self.language_to_id = base_dataset.language_to_id
        self.id_to_language = base_dataset.id_to_language

        # WeSpeaker-style augmentor (using raw audio files)
        self.augmentor = None
        if self.musanrir and AUGMENTATION_AVAILABLE:
            self.augmentor = AudioAugmentor(
                rir_folder=rir_folder,
                noise_folder=noise_folder,
                aug_prob=aug_prob,
                target_sr=16000,
            )
            print(f"  MUSAN+RIR augmentation enabled (prob={aug_prob})")
            print(f"    RIR folder: {rir_folder}")
            print(f"    Noise folder: {noise_folder}")
        elif self.musanrir and not AUGMENTATION_AVAILABLE:
            print("  Warning: musanrir=True but audio_augmentation not available.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Variable-length mode uses tuple index: (idx, duration_seconds)
        dur_sec = None
        if isinstance(idx, tuple):
            idx, dur_sec = idx

        relative_path, speaker_id, _language_id, _speaker_folder, _language_folder = \
            self.base_dataset.files[idx]
        full_path = os.path.join(self.base_dataset.path_to_features, relative_path)
        waveform, sr = torchaudio_load(full_path)

        # Use mono for consistency
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Ensure 16k waveform domain for all downstream modules
        if sr != 16000:
            waveform = AF.resample(waveform.unsqueeze(0), sr, 16000).squeeze(0)

        # Optional speed perturbation (keeps same speaker label)
        if self.is_train and len(self.speed_perturbation) > 0:
            speed = random.choice([1.0] + list(self.speed_perturbation))
            if abs(speed - 1.0) > 1e-6:
                spk_sr = max(1, int(16000 * speed))
                waveform = AF.resample(waveform.unsqueeze(0), 16000, spk_sr).squeeze(0)
                waveform = AF.resample(waveform.unsqueeze(0), spk_sr, 16000).squeeze(0)

        # Apply MUSAN+RIR augmentation on the waveform
        if self.musanrir and self.augmentor is not None:
            waveform = self.augmentor(waveform)

        # Length policy
        if self.variable_length:
            if self.is_train:
                if dur_sec is None:
                    target_len = self.base_dataset.audio_length
                else:
                    target_len = max(4000, int(float(dur_sec) * 16000))

                # Random crop if long enough, else pad/repeat to target length
                if waveform.shape[0] > target_len:
                    start = random.randint(0, waveform.shape[0] - target_len)
                    waveform = waveform[start:start + target_len]
                else:
                    waveform = pad_dataset(waveform.unsqueeze(0), target_len)
            else:
                # Evaluation: keep long context, truncate only to max_eval_dur
                if waveform.shape[0] > self.max_eval_len:
                    waveform = waveform[:self.max_eval_len]
        else:
            # Legacy fixed-length mode
            waveform = pad_dataset(waveform.unsqueeze(0), self.base_dataset.audio_length)

        waveform = normalize_audio(waveform)
        filename = relative_path
        return waveform, filename, speaker_id


def worker_init_fn(worker_id, dataset=None):
    """Worker initialization (no longer needed for raw files, but kept for compatibility)."""
    if dataset is not None and hasattr(dataset, 'reinitialize_lmdb'):
        dataset.reinitialize_lmdb()


class WavBatchSampler:
    """Batch sampler that assigns one random duration to each training batch."""
    def __init__(self, dataset, dur_range=None, shuffle=False, batch_size=1, drop_last=False):
        self.dur_range = dur_range
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    def _renew(self):
        if self.dur_range is None:
            return [], None
        return [], random.uniform(float(self.dur_range[0]), float(self.dur_range[1]))

    def __iter__(self):
        batch, dur = self._renew()
        for idx in self.sampler:
            if self.dur_range is None:
                batch.append(idx)
            else:
                batch.append((idx, dur))
            if len(batch) == self.batch_size:
                yield batch
                batch, dur = self._renew()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        sampler_len = len(self.sampler)
        if self.drop_last:
            return sampler_len // self.batch_size
        return (sampler_len + self.batch_size - 1) // self.batch_size


class ECAPAStyleTemporalBlock(nn.Module):
    """Lightweight ECAPA-style residual temporal block."""
    def __init__(self, channels, kernel_size=3, dilation=2):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=kernel_size, padding=padding,
                dilation=dilation, groups=channels, bias=False
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        se_hidden = max(16, channels // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, se_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(se_hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.pre(x)
        x = self.dw(x)
        x = x * self.se(x)
        x = x + residual
        return self.out_act(x)


# ---------------------------------------------------------------------------
# Model components (identical to main_train_simple_sv_wpt_w2vbert_mhfa.py)
# ---------------------------------------------------------------------------

def compute_eer(target_scores, nontarget_scores):
    """Compute Equal Error Rate"""
    all_scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([np.ones(len(target_scores)),
                             np.zeros(len(nontarget_scores))])
    thresholds = np.sort(np.unique(all_scores))
    far = np.zeros(len(thresholds))
    frr = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        predictions = (all_scores >= threshold).astype(int)
        far[i] = np.sum((predictions == 1) & (labels == 0)) / np.sum(labels == 0)
        frr[i] = np.sum((predictions == 0) & (labels == 1)) / np.sum(labels == 1)
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    threshold = thresholds[min_index]
    return eer * 100, threshold


def load_trial_pairs(trial_file):
    """
    Load trial pairs from file
    Format: label wav_file1 wav_file2
    Example:
        1 id014463/en/en_36069944.wav id014463/de/de_35014131.wav
        0 id014122/be/be_38040215.wav id014077/en/en_41688276.wav
    
    Returns:
        list of tuples: (label, file1, file2)
    """
    trial_pairs = []
    try:
        with open(trial_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        label = int(parts[0])
                        file1 = parts[1]
                        file2 = parts[2]
                        trial_pairs.append((label, file1, file2))
                    except (ValueError, IndexError) as e:
                        if line_num <= 5:  # Only print errors for first few lines
                            print(f"Warning: Error parsing line {line_num}: {e}")
    except Exception as e:
        print(f"Error loading trial file {trial_file}: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    print(f"✓ Loaded {len(trial_pairs)} trial pairs from {trial_file}")
    return trial_pairs


def sample_trial_pairs_from_file(trial_file, max_trial_pairs=500000, rng_seed=42):
    """
    Reservoir-sample trial pairs from file with bounded memory.

    Keeps at most `max_trial_pairs` tuples in RAM regardless of file size.

    Returns:
        (sampled_pairs, total_valid_pairs)
    """
    if max_trial_pairs <= 0:
        return [], 0

    rng = np.random.default_rng(rng_seed)
    sampled_pairs = []
    total_valid = 0

    try:
        with open(trial_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    label = int(parts[0])
                    pair = (label, parts[1], parts[2])
                except (ValueError, IndexError):
                    if line_num <= 5:
                        print(f"Warning: Error parsing line {line_num}")
                    continue

                if len(sampled_pairs) < max_trial_pairs:
                    sampled_pairs.append(pair)
                else:
                    j = rng.integers(0, total_valid + 1)
                    if j < max_trial_pairs:
                        sampled_pairs[j] = pair

                total_valid += 1
    except Exception as e:
        print(f"Error sampling trial file {trial_file}: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

    print(f"✓ Reservoir-sampled {len(sampled_pairs)} trial pairs from {total_valid} total lines")
    return sampled_pairs, total_valid


def extract_all_embeddings(model, unique_files, eval_audio_dir, device, audio_length=64600,
                           batch_size=64):
    """
    Extract embeddings for ALL unique audio files in one batched pass.
    
    Args:
        model: SimpleSVModelWPTW2VBERTMHFA
        unique_files: List of unique relative file paths
        eval_audio_dir: Base directory for audio files
        device: GPU device
        audio_length: Audio length in samples
        batch_size: Batch size for inference
    
    Returns:
        dict mapping relative_path -> embedding tensor (embedding_dim,) on CPU
    """
    model.eval()
    embeddings_dict = {}
    
    # Process files in batches
    failed_files = 0
    for batch_start in tqdm(range(0, len(unique_files), batch_size),
                            desc="Extracting embeddings", 
                            total=(len(unique_files) + batch_size - 1) // batch_size):
        batch_files = unique_files[batch_start:batch_start + batch_size]
        waveforms = []
        valid_files = []
        
        for rel_path in batch_files:
            full_path = os.path.join(eval_audio_dir, rel_path)
            try:
                waveform, sr = torchaudio_load(full_path)
                waveform = pad_dataset(waveform, audio_length)
                waveform = normalize_audio(waveform)
                waveforms.append(waveform)
                valid_files.append(rel_path)
            except Exception as e:
                failed_files += 1
                if failed_files <= 5:
                    print(f"Warning: Failed to load {full_path}: {e}")
        
        if len(waveforms) == 0:
            continue
        
        # Stack into batch and run inference
        batch_tensor = torch.stack(waveforms, dim=0).to(device)  # (B, audio_length)
        
        with torch.no_grad():
            embs = model.extract_embedding(batch_tensor, normalize=True)
        
        # Store embeddings
        embs_cpu = embs.cpu()
        for i, rel_path in enumerate(valid_files):
            embeddings_dict[rel_path] = embs_cpu[i]
    
    if failed_files > 0:
        print(f"Warning: Failed to load {failed_files} files out of {len(unique_files)}")
    
    print(f"  Extracted {len(embeddings_dict)} embeddings")
    return embeddings_dict


def batch_cosine_scores(embeddings_dict, trial_pairs, score_batch_size=5000):
    """
    Compute cosine similarity scores for all trial pairs using batch processing.
    Computes target/nontarget scores incrementally to avoid large array allocations.
    
    Args:
        embeddings_dict: dict mapping relative_path -> embedding tensor (embedding_dim,)
        trial_pairs: List of (label, file1, file2) tuples
        score_batch_size: Number of pairs to score at once (default 5000)
    
    Returns:
        (target_scores, nontarget_scores) as numpy arrays
    """
    target_scores = []
    nontarget_scores = []
    
    skipped = 0

    num_pairs = len(trial_pairs)
    print(f"  Scoring {num_pairs} trial pairs in batches of {score_batch_size}...")

    if num_pairs == 0:
        return np.array([]), np.array([])

    # Process scores in small batches; do not build large intermediate lists
    for batch_start in tqdm(range(0, num_pairs, score_batch_size),
                            desc="Batch scoring",
                            total=(num_pairs + score_batch_size - 1) // score_batch_size):
        batch_end = min(batch_start + score_batch_size, num_pairs)

        emb1_list = []
        emb2_list = []
        batch_labels = []

        for label, file1, file2 in trial_pairs[batch_start:batch_end]:
            emb1 = embeddings_dict.get(file1, None)
            emb2 = embeddings_dict.get(file2, None)
            if emb1 is None or emb2 is None:
                skipped += 1
                continue
            emb1_list.append(emb1)
            emb2_list.append(emb2)
            batch_labels.append(label)

        if len(batch_labels) == 0:
            continue

        # Vectorized cosine similarity for valid pairs in this batch
        emb1_batch = torch.stack(emb1_list, dim=0)
        emb2_batch = torch.stack(emb2_list, dim=0)
        scores = F.cosine_similarity(emb1_batch, emb2_batch, dim=1).cpu().numpy()

        # Split into target/nontarget immediately (don't store all scores at once)
        for score, label in zip(scores, batch_labels):
            if label == 1:
                target_scores.append(score)
            else:
                nontarget_scores.append(score)

        del emb1_batch, emb2_batch, scores, emb1_list, emb2_list, batch_labels

    if skipped > 0:
        print(f"  Skipped {skipped} pairs (missing embeddings from dict with {len(embeddings_dict)} keys)")

    # Convert to numpy arrays
    target_scores = np.array(target_scores, dtype=np.float32)
    nontarget_scores = np.array(nontarget_scores, dtype=np.float32)

    print(f"  Target scores: {len(target_scores)}, Nontarget scores: {len(nontarget_scores)}")

    return target_scores, nontarget_scores


def evaluate_with_cached_embeddings(val_embeddings, val_filenames, trial_pairs, score_batch_size=5000):
    """
    Fast evaluation using CACHED embeddings from validation pass.
    Reuses embeddings already extracted during validation to avoid redundant I/O.
    
    Args:
        val_embeddings: Torch tensor (N, embedding_dim) from validation loader
        val_filenames: List of N filenames from validation loader 
                      (format: speaker_id/language/filename.wav - same as trial file)
        trial_pairs: List of (label, file1, file2) tuples
        score_batch_size: Batch size for scoring
    
    Returns:
        (eer, threshold) tuple
    """
    # Build embeddings dict from validation results
    embeddings_dict = {}
    for fn, emb in zip(val_filenames, val_embeddings):
        embeddings_dict[fn] = emb
    
    print(f"\n  Trial evaluation: {len(trial_pairs)} pairs, {len(val_filenames)} cached embeddings")
    
    # Debug: show sample formats
    sample_fns = val_filenames[:3] if val_filenames else []
    sample_trials = trial_pairs[:3] if trial_pairs else []
    if sample_fns:
        print(f"  Sample validation filenames: {sample_fns}")
    if sample_trials:
        print(f"  Sample trial pairs: {[(l, f1, f2) for l, f1, f2 in sample_trials]}")
    
    # Score all trial pairs using cached embeddings
    target_scores, nontarget_scores = batch_cosine_scores(
        embeddings_dict, trial_pairs,
        score_batch_size=score_batch_size
    )
    
    # Compute EER
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        print("  Warning: no valid trials in trial file")
        eer, threshold = 100.0, 0.0
    else:
        eer, threshold = compute_eer(target_scores, nontarget_scores)
    
    return eer, threshold



# WaveletBlock + WPTW2VBERTMultiLayer now live in peft_wpt.py so that the
# adaptation mechanism (--peft_mode) and wavelet toggle (--use_wavelet) are
# shared across entry points. The default (deep_prompt + wavelet) reproduces the
# original behaviour byte-for-byte, with identical parameter names, so released
# checkpoints load unchanged.
from peft_wpt import WaveletBlock, WPTW2VBERTMultiLayer  # noqa: E402


class MHFAHeadDynamicStats(nn.Module):
    """
    MHFA head with:
      1) Dynamic layer fusion (input-dependent weights per utterance)
      2) Attentive statistics pooling (weighted mean + weighted std)
    """
    def __init__(self, feature_dim=1024, num_layers=24, num_heads=8,
                 compression_dim=128, embedding_dim=256, adapter_bottleneck=128,
                 dynamic_fusion_hidden=256, head_type='dynstats', dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.compression_dim = compression_dim
        self.embedding_dim = embedding_dim
        self.adapter_bottleneck = adapter_bottleneck
        self.head_type = head_type

        # Dynamic layer fusion gates: per-utterance weights for key/value streams
        self.layer_gate = nn.Sequential(
            nn.Linear(feature_dim, dynamic_fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dynamic_fusion_hidden, 2 * num_layers)
        )

        self.key_projection = nn.Linear(feature_dim, compression_dim)
        self.value_projection = nn.Linear(feature_dim, compression_dim)
        self.attention_projection = nn.Linear(compression_dim, num_heads)
        self.temporal_block = None
        if self.head_type == 'dynstats_ecapa':
            self.temporal_block = ECAPAStyleTemporalBlock(
                channels=compression_dim, kernel_size=3, dilation=2
            )

        self.embedding_projection = nn.Sequential(
            nn.Linear(2 * num_heads * compression_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout))

        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, adapter_bottleneck),
            nn.BatchNorm1d(adapter_bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_bottleneck, embedding_dim))
        self.adapter_norm = nn.LayerNorm(embedding_dim)

        self.embedding_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

        print(f"  Dynamic-Stats MHFA Head initialised:")
        print(f"    Feature dim: {feature_dim}, Layers: {num_layers}")
        print(f"    Heads: {num_heads}, Compression: {compression_dim}")
        print(f"    Embedding: {embedding_dim}, Adapter: {adapter_bottleneck}")
        print(f"    Dynamic fusion hidden dim: {dynamic_fusion_hidden}")
        print(f"    Pooling: attentive mean + attentive std")
        print(f"    Head type: {head_type}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, layer_features):
        if isinstance(layer_features, list):
            layer_features = torch.stack(layer_features, dim=0)
        L, B, T, D = layer_features.shape

        # Build utterance summary (B, D) then predict dynamic per-layer weights
        utt_summary = layer_features.mean(dim=0).mean(dim=1)
        gate_logits = self.layer_gate(utt_summary)
        gate_k, gate_v = torch.split(gate_logits, self.num_layers, dim=1)
        w_k = F.softmax(gate_k, dim=1).transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        w_v = F.softmax(gate_v, dim=1).transpose(0, 1).unsqueeze(-1).unsqueeze(-1)

        K_feat = (layer_features * w_k).sum(dim=0)
        V_feat = (layer_features * w_v).sum(dim=0)

        K = self.dropout(self.key_projection(K_feat))
        V = self.dropout(self.value_projection(V_feat))
        if self.temporal_block is not None:
            # ECAPA-style temporal refinement before attentive stats pooling
            V = self.temporal_block(V.transpose(1, 2)).transpose(1, 2)

        # Attentive statistics pooling: weighted mean + weighted std
        # A: (B, T, H) -> (B, H, T, 1)
        A = F.softmax(self.attention_projection(K), dim=1)
        A = A.transpose(1, 2).unsqueeze(-1)
        V = V.unsqueeze(1)  # (B, 1, T, C)
        mean = (A * V).sum(dim=2)  # (B, H, C)
        second_moment = (A * (V ** 2)).sum(dim=2)
        var = torch.clamp(second_moment - mean ** 2, min=1e-6)
        std = torch.sqrt(var)
        pooled = torch.cat([mean, std], dim=-1).reshape(
            B, 2 * self.num_heads * self.compression_dim
        )

        embeddings = self.embedding_projection(pooled)
        adapted = self.adapter(embeddings)
        adapted = self.adapter_norm(adapted + embeddings)
        embeddings = self.embedding_layer(adapted)
        return embeddings


class SimpleSVModelWPTW2VBERTMHFA(nn.Module):
    """
    Simple SV with WPT + W2V-BERT-2.0 + MHFA Head + SpecAugment.
    Same architecture, with optional SpecAugment on per-layer features.
    """

    def __init__(self, model_dir, num_speakers, embedding_dim=256,
                 num_prompt_tokens=6, num_wavelet_tokens=4, prompt_dropout=0.1,
                 num_heads=8, compression_dim=128, adapter_bottleneck=128,
                 dynamic_fusion_hidden=256,
                 head_type='dynstats',
                 head_dropout=0.1, use_arcface=True, arcface_margin=0.3,
                 arcface_scale=30.0,
                 use_specaugment=False, time_mask_param=20, freq_mask_param=40,
                 num_time_masks=2, num_freq_masks=2,
                 peft_mode="deep_prompt", use_wavelet=True, num_prefix_tokens=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.use_arcface = use_arcface
        self.use_specaugment = use_specaugment

        # W2V-BERT-2.0 + WPT
        print(f"\nLoading W2V-BERT-2.0 with WPT from {model_dir}...")
        self.wpt_w2vbert = WPTW2VBERTMultiLayer(
            model_dir=model_dir,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens=num_wavelet_tokens,
            prompt_dim=1024,
            dropout=prompt_dropout,
            peft_mode=peft_mode,
            use_wavelet=use_wavelet,
            num_prefix_tokens=num_prefix_tokens)

        num_layers = self.wpt_w2vbert.config.num_hidden_layers

        # Dynamic MHFA Head with attentive statistics pooling
        print("\nInitialising Dynamic-Stats MHFA Head...")
        self.mhfa_head = MHFAHeadDynamicStats(
            feature_dim=1024,
            num_layers=num_layers,
            num_heads=num_heads,
            compression_dim=compression_dim,
            embedding_dim=embedding_dim,
            adapter_bottleneck=adapter_bottleneck,
            dynamic_fusion_hidden=dynamic_fusion_hidden,
            head_type=head_type,
            dropout=head_dropout)

        # SpecAugment (applied to each layer's features during training)
        if use_specaugment:
            self.specaugment = SpecAugment(
                time_mask_param=time_mask_param,
                freq_mask_param=freq_mask_param,
                num_time_masks=num_time_masks,
                num_freq_masks=num_freq_masks)
            print(f"\n  SpecAugment enabled:")
            print(f"    time_mask_param={time_mask_param}, num_time_masks={num_time_masks}")
            print(f"    freq_mask_param={freq_mask_param}, num_freq_masks={num_freq_masks}")
        else:
            self.specaugment = None

        # Classifier
        if use_arcface:
            print(f"\nInitialising ArcFace loss...")
            self.arcface_loss = ArcFaceLoss(
                in_features=embedding_dim,
                out_features=num_speakers,
                scale=arcface_scale,
                margin=arcface_margin,
                easy_margin=False)
            self.classifier = None
            print(f"  ArcFace: margin={arcface_margin}, scale={arcface_scale}")
        else:
            self.classifier = nn.Linear(embedding_dim, num_speakers)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
            self.arcface_loss = None
            print(f"  Using Linear classifier")

        print(f"\n  Model: WPT + W2V-BERT-2.0 + MHFA Head")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Num speakers: {num_speakers}")
        print(f"  ArcFace: {use_arcface}")
        print(f"  SpecAugment: {use_specaugment}")

    def extract_embedding(self, audio_data, normalize=True):
        """Extract speaker embedding (no SpecAugment during inference)."""
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        layer_features = self.wpt_w2vbert(audio_data)
        embeddings = self.mhfa_head(layer_features)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(self, audio_data, labels=None):
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)

        layer_features = self.wpt_w2vbert(audio_data)

        # Apply SpecAugment to each layer's features during training
        if self.use_specaugment and self.specaugment is not None and self.training:
            layer_features = [self.specaugment(lf) for lf in layer_features]

        embeddings_unnorm = self.mhfa_head(layer_features)
        embeddings_norm = F.normalize(embeddings_unnorm, p=2, dim=1)

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    use_variable_length = args.dur_range is not None
    if use_variable_length:
        print(f"Variable-length mode: dur_range={args.dur_range[0]:.2f}-{args.dur_range[1]:.2f}s")
        print(f"Validation max duration: {args.max_eval_dur:.1f}s")
    else:
        print(f"Fixed-length mode: {args.audio_len} samples ({args.audio_len/16000:.3f}s)")

    # ------------------------------------------------------------------
    # Load multilingual datasets
    # ------------------------------------------------------------------
    print("\nLoading multilingual training dataset...")
    base_train_dataset = MultilingualASVDataset(
        path_to_features=args.train_audio,
        audio_length=args.audio_len,
    )

    # Wrap with MUSAN+RIR augmentation
    train_dataset = MultilingualASVDatasetWithAug(
        base_dataset=base_train_dataset,
        musanrir=args.musanrir,
        rir_folder=args.rir_folder,
        noise_folder=args.noise_folder,
        aug_prob=args.aug_prob,
        variable_length=use_variable_length,
        max_eval_dur=args.max_eval_dur,
        speed_perturbation=args.speed_perturbation,
        is_train=True,
    )

    print("\nLoading multilingual validation dataset...")
    base_val_dataset = MultilingualASVDataset(
        path_to_features=args.eval_audio,
        audio_length=args.audio_len,
        speaker_to_id=base_train_dataset.speaker_to_id,
        id_to_speaker=base_train_dataset.id_to_speaker,
        language_to_id=base_train_dataset.language_to_id,
        id_to_language=base_train_dataset.id_to_language,
    )
    # No augmentation for validation
    val_dataset = MultilingualASVDatasetWithAug(
        base_dataset=base_val_dataset,
        musanrir=False,
        variable_length=use_variable_length,
        max_eval_dur=args.max_eval_dur,
        speed_perturbation=None,
        is_train=False,
    )

    # Setup DataLoader (no special worker_init_fn needed for raw files)
    use_persistent = args.num_workers > 0
    if use_variable_length:
        train_batch_sampler = WavBatchSampler(
            train_dataset,
            dur_range=args.dur_range,
            shuffle=True,
            batch_size=args.batch_size,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=use_persistent,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    num_speakers = train_dataset.num_speakers
    print(f"Number of speakers: {num_speakers}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nInitialising model...")
    model = SimpleSVModelWPTW2VBERTMHFA(
        model_dir=args.xlsr,
        num_speakers=num_speakers,
        embedding_dim=args.embedding_dim,
        num_prompt_tokens=args.num_prompt_tokens,
        num_wavelet_tokens=args.num_wavelet_tokens,
        prompt_dropout=args.prompt_dropout,
        num_heads=args.num_heads,
        compression_dim=args.compression_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        dynamic_fusion_hidden=args.dynamic_fusion_hidden,
        head_type=args.head_type,
        head_dropout=args.head_dropout,
        use_arcface=args.use_arcface,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        use_specaugment=args.use_specaugment,
        time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        num_time_masks=args.num_time_masks,
        num_freq_masks=args.num_freq_masks,
        peft_mode=args.peft_mode,
        use_wavelet=(args.use_wavelet == 'on'),
        num_prefix_tokens=args.num_prefix_tokens,
    ).to(device)
    if args.pretrain:
        if os.path.exists(args.pretrain):
            print(f"\nLoading checkpoint: {args.pretrain}")
            ckpt = torch.load(args.pretrain, map_location='cpu')
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded checkpoint with strict=False")
            print(f"  Missing keys: {len(load_result.missing_keys)}")
            print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")
        else:
            print(f"Warning: pretrain checkpoint not found: {args.pretrain}")

    # ------------------------------------------------------------------
    # Optimiser — only WPT + MHFA head + classifier
    # ------------------------------------------------------------------
    trainable_params = []
    trainable_params += [model.wpt_w2vbert.prompt_embeddings]
    trainable_params += [model.wpt_w2vbert.wavelet_prompt_embeddings]
    trainable_params += list(model.mhfa_head.parameters())
    if model.use_arcface and model.arcface_loss is not None:
        trainable_params += list(model.arcface_loss.parameters())
    else:
        trainable_params += list(model.classifier.parameters())
    # SpecAugment has no learnable parameters

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_eer = 100.0
    os.makedirs(args.out_fold, exist_ok=True)

    # Load only a bounded trial subset once (avoid loading full huge trial files in RAM)
    trial_pairs_cache = None
    if args.trial_file and os.path.exists(args.trial_file):
        print("\nSampling trial pairs with bounded memory (once)...")
        trial_pairs_cache, total_pairs = sample_trial_pairs_from_file(
            args.trial_file,
            max_trial_pairs=args.max_trial_pairs,
            rng_seed=args.trial_sample_seed
        )
        if trial_pairs_cache:
            pct = 100.0 * len(trial_pairs_cache) / max(1, total_pairs)
            print(f"  Using {len(trial_pairs_cache)} / {total_pairs} trial pairs ({pct:.2f}%)")
        else:
            trial_pairs_cache = None

    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        model.train()
        model.wpt_w2vbert.model.eval()  # keep backbone frozen
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch_idx, (waveform, filename, speaker_labels) in enumerate(pbar):
            waveform = waveform.to(device)
            speaker_labels = speaker_labels.to(device)

            optimizer.zero_grad()

            if args.use_arcface:
                embeddings_norm, embeddings_unnorm, logits, loss = \
                    model(waveform, labels=speaker_labels)
            else:
                embeddings_norm, embeddings_unnorm, logits = \
                    model(waveform, labels=speaker_labels)
                loss = F.cross_entropy(logits, speaker_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == speaker_labels).sum().item()
            train_total += speaker_labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })

        scheduler.step()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if (epoch + 1) % args.interval == 0:
            model.eval()
            val_embeddings = []
            val_filenames = []
            val_labels = []

            with torch.no_grad():
                for waveform, filename, speaker_labels in tqdm(val_loader,
                        desc=f"Epoch {epoch+1} [Val]"):
                    waveform = waveform.to(device)
                    embeddings = model.extract_embedding(waveform, normalize=True)
                    val_embeddings.append(embeddings.cpu())
                    val_filenames.extend(filename)
                    val_labels.append(speaker_labels)

            val_embeddings = torch.cat(val_embeddings, dim=0)
            val_labels = torch.cat(val_labels, dim=0)

            # EER via trial file (FAST using cached embeddings) or sampled pairs
            if trial_pairs_cache is not None:
                # Fast evaluation: REUSE pre-loaded trial pairs + cached embeddings
                eer, threshold = evaluate_with_cached_embeddings(
                    val_embeddings, val_filenames, trial_pairs_cache,
                    score_batch_size=args.score_batch_size
                )
            else:
                # Fallback: sampled pairs from validation set
                emb_np = val_embeddings.cpu().numpy()
                labels_np = val_labels.cpu().numpy()
                num_samples = emb_np.shape[0]
                max_pairs = 100000
                total_pairs = num_samples * (num_samples - 1) // 2

                if total_pairs > max_pairs:
                    np.random.seed(42)
                    idx_i, idx_j = np.triu_indices(num_samples, k=1)
                    sample_idx = np.random.choice(len(idx_i), max_pairs, replace=False)
                    emb_i = emb_np[idx_i[sample_idx]]
                    emb_j = emb_np[idx_j[sample_idx]]
                    pair_scores = np.sum(emb_i * emb_j, axis=1)
                    same_speaker = labels_np[idx_i[sample_idx]] == labels_np[idx_j[sample_idx]]
                else:
                    sim_matrix = np.matmul(emb_np, emb_np.T)
                    idx_i, idx_j = np.triu_indices(num_samples, k=1)
                    pair_scores = sim_matrix[idx_i, idx_j]
                    same_speaker = labels_np[idx_i] == labels_np[idx_j]

                target_scores = pair_scores[same_speaker]
                nontarget_scores = pair_scores[~same_speaker]
                eer, threshold = compute_eer(target_scores, nontarget_scores)

            print(f"\nEpoch {epoch+1} Validation:")
            print(f"  EER: {eer:.4f}%")
            print(f"  Threshold: {threshold:.4f}")

            if eer < best_eer:
                best_eer = eer
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eer': eer,
                    'peft_mode': args.peft_mode,
                    'use_wavelet': (args.use_wavelet == 'on'),
                    'num_prefix_tokens': args.num_prefix_tokens,
                }, os.path.join(args.out_fold, 'best_model.pt'))
                print(f"  -> Saved best model (EER: {eer:.4f}%)")

            with open(os.path.join(args.out_fold, 'val_eer.log'), 'a') as f:
                f.write(f"{epoch+1} {eer:.4f}\n")

            # Free validation memory to prevent accumulation across epochs
            del val_embeddings, val_filenames, val_labels
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nTraining complete! Best EER: {best_eer:.4f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WPT + W2V-BERT-2.0 + MHFA — Multilingual data + MUSAN/RIR + SpecAugment')

    # Data paths (multilingual folder structure)
    parser.add_argument('--train_audio', type=str, required=True,
                        help='Path to multilingual training data (speaker/language/wav)')
    parser.add_argument('--eval_audio', type=str, required=True,
                        help='Path to multilingual validation data (speaker/language/wav)')
    parser.add_argument('--trial_file', type=str, default=None,
                        help='Path to trial file for EER evaluation (optional)')

    # Audio parameters
    parser.add_argument('--audio_len', type=int, default=64600,
                        help='Audio length in samples (default: 64600 = 4.04s @ 16kHz)')
    parser.add_argument('--dur_range', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Variable-length training duration range in seconds, e.g. --dur_range 2 3')
    parser.add_argument('--max_eval_dur', type=float, default=60.0,
                        help='Maximum evaluation duration in seconds when variable-length mode is used')
    parser.add_argument('--speed_perturbation', type=float, nargs='*', default=[],
                        help='Optional speed perturbation factors, e.g. --speed_perturbation 0.9 1.1')

    # MUSAN+RIR augmentation (using raw audio files, not LMDB)
    parser.add_argument('--musanrir', action='store_true',
                        help='Enable MUSAN+RIR augmentation')
    parser.add_argument('--rir_folder', type=str, default=None,
                        help='Path to RIRS_NOISES raw audio folder')
    parser.add_argument('--noise_folder', type=str, default=None,
                        help='Path to MUSAN raw audio folder')
    parser.add_argument('--aug_prob', type=float, default=0.6,
                        help='Probability of applying MUSAN+RIR augmentation (default: 0.6)')

    # SpecAugment parameters
    parser.add_argument('--use_specaugment', action='store_true',
                        help='Enable SpecAugment (time/frequency masking on SSL features)')
    parser.add_argument('--time_mask_param', type=int, default=20,
                        help='Max time-mask length in frames (default: 20)')
    parser.add_argument('--freq_mask_param', type=int, default=40,
                        help='Max frequency-mask width in feature dims (default: 40)')
    parser.add_argument('--num_time_masks', type=int, default=2,
                        help='Number of time masks (default: 2)')
    parser.add_argument('--num_freq_masks', type=int, default=2,
                        help='Number of frequency masks (default: 2)')

    # Model paths
    parser.add_argument('--xlsr', type=str, default='facebook/w2v-bert-2.0',
                        help='Path to W2V-BERT-2.0 model')

    # Training hyperparameters
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--interval', type=int, default=1,
                        help='Validation interval (epochs)')
    parser.add_argument('--max_trial_pairs', type=int, default=500000,
                        help='Maximum number of trial pairs kept in RAM for validation scoring')
    parser.add_argument('--score_batch_size', type=int, default=3000,
                        help='Trial scoring batch size (reduce if host RAM is limited)')
    parser.add_argument('--trial_sample_seed', type=int, default=42,
                        help='Seed for trial reservoir sampling')

    # WPT parameters
    parser.add_argument('--num_prompt_tokens', type=int, default=15)
    parser.add_argument('--num_wavelet_tokens', type=int, default=8)
    parser.add_argument('--prompt_dropout', type=float, default=0.1)

    # MHFA head parameters
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads in MHFA')
    parser.add_argument('--compression_dim', type=int, default=128,
                        help='Compression dimension in MHFA')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Final embedding dimension')
    parser.add_argument('--adapter_bottleneck', type=int, default=128,
                        help='Adapter bottleneck dimension')
    parser.add_argument('--dynamic_fusion_hidden', type=int, default=256,
                        help='Hidden dimension for dynamic layer-fusion gate MLP')
    parser.add_argument('--head_type', type=str, default='dynstats',
                        choices=['dynstats', 'dynstats_ecapa'],
                        help='Head type for ablation')
    parser.add_argument('--head_dropout', type=float, default=0.1)

    # PEFT adaptation mechanism (see src/wpt/peft_wpt.py)
    parser.add_argument('--peft_mode', type=str, default='deep_prompt',
                        choices=['deep_prompt', 'shallow_prompt', 'prefix'],
                        help="How the learnable vectors enter the frozen backbone. "
                             "'deep_prompt' (default) = the paper's mechanism; "
                             "'shallow_prompt' = classic prompt tuning (input only); "
                             "'prefix' = EXPERIMENTAL true prefix-tuning (verify with "
                             "scripts/smoke_test_peft.py).")
    parser.add_argument('--use_wavelet', type=str, default='on', choices=['on', 'off'],
                        help="Haar-wavelet-structured prompts ('on', default) or raw ('off').")
    parser.add_argument('--num_prefix_tokens', type=int, default=None,
                        help="Prefix tokens/layer for --peft_mode prefix "
                             "(default: num_prompt_tokens + num_wavelet_tokens).")

    # Loss parameters
    parser.add_argument('--use_arcface', action='store_true',
                        help='Use ArcFace loss')
    parser.add_argument('--arcface_margin', type=float, default=0.3)
    parser.add_argument('--arcface_scale', type=float, default=30.0)
    parser.add_argument('--pretrain', type=str, default='',
                        help='Optional checkpoint to initialize model weights')

    # Output
    parser.add_argument('--out_fold', type=str, required=True,
                        help='Output directory')

    args = parser.parse_args()

    os.makedirs(args.out_fold, exist_ok=True)
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    train(args)
