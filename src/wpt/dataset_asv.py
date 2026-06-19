"""
Dataset loader for ASV (Automatic Speaker Verification) task
==============================================================

This module provides dataset loaders for training speaker verification models
where the classifier head predicts speaker identities.

Key difference from spoofceleb dataset:
- Returns speaker labels for classification
- No spoof/bonafide labels needed
- Focuses on speaker discrimination
"""

import torch
import os
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchaudio
from RawBoost import process_Rawboost_feature

# Import WeSpeaker-style augmentation
try:
    from audio_augmentation import AudioAugmentor
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("Warning: audio_augmentation module not found. Using basic augmentation.")


def torchaudio_load(file_path):
    """Load audio file using torchaudio"""
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


def pad_dataset(wav, audio_length=64600):
    """Pad or truncate audio to fixed length"""
    # Squeeze to remove channel dimension: (1, length) -> (length,)
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = audio_length
    
    if waveform_len >= cut:
        waveform = waveform[:cut]
    else:
        # need to pad
        num_repeats = int(cut / waveform_len) + 1
        waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    
    # Normalize audio to zero mean and unit variance
    # This ensures consistent amplitude ranges across different datasets (SpoofCeleb vs VoxCeleb2)
    waveform = normalize_audio(waveform)
    
    return waveform


def normalize_audio(waveform):
    """Normalize audio to zero mean and unit variance"""
    waveform = waveform - waveform.mean()
    waveform = waveform / (torch.sqrt(waveform.var() + 1e-7))
    return waveform


class AudioAugmentor:
    """Simple audio augmentation for ASV training"""
    def __init__(self):
        pass
    
    def apply_noise(self, waveform, noise_level=0.005):
        """Add Gaussian noise"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def apply_time_stretch(self, waveform, rate=1.0):
        """Simple time stretching (placeholder)"""
        # For simplicity, just return waveform
        # In production, use proper time stretching
        return waveform


class WavBatchSampler:
    """Variable-length batch sampler (inspired by w2v-BERT-2.0_SV).
    
    Each batch gets a random duration drawn uniformly from *dur_range*.
    Every sample index in that batch is paired with the same duration so
    the DataLoader can produce a single uniform-length tensor per batch
    without any padding or loop-tiling.
    
    Args:
        dataset: The dataset to sample from.
        dur_range: Tuple (min_dur, max_dur) in seconds, e.g. (2, 6).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle indices each epoch.
        drop_last: Drop the last incomplete batch.
    """

    def __init__(self, dataset, dur_range, batch_size=16, shuffle=True,
                 drop_last=True):
        self.dataset = dataset
        self.dur_range = dur_range
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def _renew(self):
        dur = random.uniform(self.dur_range[0], self.dur_range[1])
        return [], dur

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        batch, dur = self._renew()
        for idx in indices:
            batch.append((idx, dur))
            if len(batch) == self.batch_size:
                yield batch
                batch, dur = self._renew()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class SpoofCelebASV(Dataset):
    """
    SpoofCeleb dataset for ASV (Speaker Verification) training
    
    This dataset loader is designed for training speaker verification models
    where the goal is to classify audio by speaker identity.
    
    Supports two modes:
      1. **Fixed-length** (legacy): audio_length is set, all samples padded/truncated
         to the same length. __getitem__ receives a plain int index.
      2. **Variable-length**: variable_length=True. __getitem__ receives either
         (idx, dur) tuples (training, via WavBatchSampler) or plain int index
         (evaluation, clips capped at max_eval_dur).
    
    Returns:
        - waveform: Audio waveform tensor
        - filename: Audio filename (without extension)
        - speaker_id: Numeric speaker ID (for classification)
    """
    
    def __init__(self, path_to_features, path_to_protocol, 
                 rawboost=False, musanrir=False, audio_length=64600, 
                 rawboost_log=5, bonafide_only=True,
                 reverb_lmdb_path=None, noise_lmdb_path=None, aug_prob=0.6,
                 variable_length=False, max_eval_dur=60, sample_rate=16000):
        """
        Args:
            path_to_features: Path to audio files
            path_to_protocol: Path to protocol CSV file (file,speaker,attack format)
            rawboost: Whether to apply RawBoost augmentation
            musanrir: Whether to apply MUSAN+RIR augmentation (WeSpeaker-style)
            audio_length: Length of audio in samples (default 64600 = 4.0375s @ 16kHz)
                          Only used when variable_length=False.
            rawboost_log: RawBoost algorithm ID (1-5)
            bonafide_only: If True, only load bonafide samples (a00)
            reverb_lmdb_path: Path to RIRS LMDB database (for WeSpeaker-style augmentation)
            noise_lmdb_path: Path to MUSAN LMDB database (for WeSpeaker-style augmentation)
            aug_prob: Probability of applying reverb/noise augmentation (default: 0.6)
            variable_length: If True, use variable-length random cropping instead of
                             fixed pad/truncate.
            max_eval_dur: Maximum duration in seconds for evaluation (default: 60).
                          Only used when variable_length=True and plain int index is given.
            sample_rate: Audio sample rate in Hz (default: 16000).
        """
        super(SpoofCelebASV, self).__init__()
        
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.rawboost_log = rawboost_log
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.bonafide_only = bonafide_only
        self.variable_length = variable_length
        self.sample_rate = sample_rate
        self.max_eval_len = int(max_eval_dur * sample_rate)
        
        # Initialize WeSpeaker-style augmentation if available
        if self.musanrir and AUGMENTATION_AVAILABLE:
            self.wespeaker_augmentor = AudioAugmentor(
                reverb_lmdb_path=reverb_lmdb_path,
                noise_lmdb_path=noise_lmdb_path,
                aug_prob=aug_prob,
                target_sr=16000
            )
        else:
            self.wespeaker_augmentor = None
            if self.musanrir and not AUGMENTATION_AVAILABLE:
                print("Warning: musanrir=True but audio_augmentation not available. Using basic augmentation.")
        
        # Keep old AudioAugmentor for backward compatibility (basic augmentation)
        # Note: This is the simple AudioAugmentor class defined in this file, not the WeSpeaker one
        if not AUGMENTATION_AVAILABLE:
            # Use the simple AudioAugmentor class defined below
            self.AudioAugmentor = AudioAugmentor()
        else:
            # WeSpeaker augmentation is available, don't use simple one
            self.AudioAugmentor = None

        self.files = []
        self.speaker_to_id = {}  # Map speaker ID to numeric IDs
        self.id_to_speaker = {}  # Map numeric IDs to speaker ID
        speaker_counter = 0
        
        bonafide_count = 0
        spoof_count = 0
        
        with open(path_to_protocol, 'r') as f:
            next(f)  # Skip header
            
            for line_num, line in enumerate(f, 2):
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue  
                
                file_path = parts[0]  # e.g., "a00/id10318/YYsxcZ5saac-00002-006.flac"
                speaker = parts[1]    # e.g., "id10318"
                attack_type = parts[2]  # e.g., "a00" (bonafide) or attack type
                
                # Filter for bonafide only if specified
                is_bonafide = (attack_type == "a00")
                if self.bonafide_only and not is_bonafide:
                    spoof_count += 1
                    continue
                
                if is_bonafide:
                    bonafide_count += 1
                else:
                    spoof_count += 1
                
                # Map speaker to numeric ID
                if speaker not in self.speaker_to_id:
                    self.speaker_to_id[speaker] = speaker_counter
                    self.id_to_speaker[speaker_counter] = speaker
                    speaker_counter += 1
                
                speaker_id = self.speaker_to_id[speaker]
                
                self.files.append((file_path, speaker_id, speaker, attack_type))

        self.num_speakers = len(self.speaker_to_id)
        
        print(f"\n{'='*80}")
        print(f"Loaded SpoofCeleb ASV Dataset")
        print(f"{'='*80}")
        print(f"Protocol file: {path_to_protocol}")
        print(f"Audio path: {path_to_features}")
        print(f"Bonafide only: {bonafide_only}")
        if bonafide_only:
            print(f"  Loaded: {bonafide_count} bonafide samples")
            print(f"  Filtered: {spoof_count} spoof samples")
        else:
            print(f"  Bonafide: {bonafide_count}")
            print(f"  Spoof: {spoof_count}")
        print(f"Total samples: {len(self.files)}")
        print(f"Unique speakers: {self.num_speakers}")
        if self.variable_length:
            print(f"Mode: VARIABLE-LENGTH (random crop per batch)")
            print(f"  Max eval duration: {max_eval_dur}s ({self.max_eval_len} samples @ {sample_rate}Hz)")
        else:
            print(f"Mode: FIXED-LENGTH (legacy)")
            print(f"  Audio length: {audio_length} samples ({audio_length/16000:.3f}s @ 16kHz)")
        print(f"RawBoost: {rawboost}")
        print(f"MUSAN+RIR: {musanrir}")
        if musanrir and self.wespeaker_augmentor is not None:
            print(f"  Reverb LMDB: {reverb_lmdb_path}")
            print(f"  Noise LMDB: {noise_lmdb_path}")
            print(f"  Augmentation probability: {aug_prob}")
        print(f"{'='*80}\n")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx_data):
        # Support both (idx, dur) tuples from WavBatchSampler and plain int
        if isinstance(idx_data, (tuple, list)):
            idx, dur = idx_data
            target_len = int(dur * self.sample_rate)
        else:
            idx = idx_data
            dur = None
            target_len = None

        file_path, speaker_id, speaker, attack_type = self.files[idx]
        
        # Load audio file
        full_path = os.path.join(self.path_to_features, file_path)
        waveform, sr = torchaudio_load(full_path)

        # Apply RawBoost augmentation if enabled (before cropping)
        if self.rawboost:
            waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
            waveform = process_Rawboost_feature(waveform, sr=sr, algo=int(self.rawboost_log))
            waveform = torch.Tensor(np.expand_dims(waveform, axis=0))
        
        # Squeeze channel dim: (1, L) -> (L,)
        waveform = waveform.squeeze(0)

        if self.variable_length:
            if target_len is not None:
                # Training: random crop to the batch-level duration
                waveform = truncate_audio_random(waveform, target_len)
            else:
                # Eval / inference: use full audio, capped at max_eval_len
                if waveform.shape[0] > self.max_eval_len:
                    waveform = waveform[:self.max_eval_len]
            waveform = normalize_audio(waveform)
        else:
            # Legacy fixed-length mode
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = pad_dataset(waveform, self.audio_length)
        
        # Apply MUSAN+RIR augmentation if enabled
        if self.musanrir:
            if self.wespeaker_augmentor is not None:
                # Use WeSpeaker-style augmentation
                waveform = self.wespeaker_augmentor(waveform)
            else:
                # Fallback to basic augmentation
                wav_len = waveform.size(0)
                waveform = self._apply_augmentation(waveform, wav_len)
        
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Return: waveform, filename, speaker_id
        # Note: speaker_id is the label for classification
        return waveform, filename, speaker_id

    def _apply_augmentation(self, waveform, audio_length):
        """Apply MUSAN+RIR augmentation (placeholder)"""
        # For simplicity, apply basic augmentation
        augtype = random.randint(0, 4)
        
        if augtype == 0:
            # No augmentation
            return waveform
        elif augtype == 1:
            # Add noise
            return self.AudioAugmentor.apply_noise(waveform, noise_level=0.005)
        elif augtype == 2:
            # Add more noise
            return self.AudioAugmentor.apply_noise(waveform, noise_level=0.01)
        elif augtype == 3:
            # Time stretch (placeholder)
            return self.AudioAugmentor.apply_time_stretch(waveform, rate=0.95)
        else:
            # Time stretch (placeholder)
            return self.AudioAugmentor.apply_time_stretch(waveform, rate=1.05)
        
        return waveform
    
    def collate_fn(self, samples):
        return default_collate(samples)

    @staticmethod
    def variable_length_collate_fn(samples):
        """Collate function for variable-length evaluation.
        
        During eval each sample may have a different length. We pad
        all waveforms in the batch to the length of the longest one
        with zeros. This is fine because the MHFA attention pooling
        (softmax + weighted sum) naturally down-weights zero-padded
        regions since they carry no energy.
        """
        waveforms, filenames, speaker_ids = zip(*samples)
        lengths = [w.shape[0] for w in waveforms]
        max_len = max(lengths)
        padded = torch.zeros(len(waveforms), max_len)
        for i, w in enumerate(waveforms):
            padded[i, :w.shape[0]] = w
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
        return padded, list(filenames), speaker_ids


class VoxCelebASV(Dataset):
    """
    VoxCeleb dataset loader for ASV training
    
    This is a placeholder for VoxCeleb dataset support.
    Adapt this based on your VoxCeleb data structure.
    """
    
    def __init__(self, path_to_features, path_to_protocol, 
                 audio_length=64600, rawboost=False):
        """
        Args:
            path_to_features: Path to VoxCeleb audio files
            path_to_protocol: Path to protocol file
            audio_length: Audio length in samples
            rawboost: Whether to apply RawBoost
        """
        super(VoxCelebASV, self).__init__()
        
        self.path_to_features = path_to_features
        self.audio_length = audio_length
        self.rawboost = rawboost
        
        self.files = []
        self.speaker_to_id = {}
        self.id_to_speaker = {}
        speaker_counter = 0
        
        # Load protocol file
        # Format: speaker_id/session_id/utterance_id.wav
        with open(path_to_protocol, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Extract speaker ID from path
                parts = line.split('/')
                speaker = parts[0]
                
                if speaker not in self.speaker_to_id:
                    self.speaker_to_id[speaker] = speaker_counter
                    self.id_to_speaker[speaker_counter] = speaker
                    speaker_counter += 1
                
                speaker_id = self.speaker_to_id[speaker]
                self.files.append((line, speaker_id, speaker))
        
        self.num_speakers = len(self.speaker_to_id)
        
        print(f"\nLoaded VoxCeleb ASV Dataset:")
        print(f"  Total samples: {len(self.files)}")
        print(f"  Unique speakers: {self.num_speakers}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path, speaker_id, speaker = self.files[idx]
        
        full_path = os.path.join(self.path_to_features, file_path)
        waveform, sr = torchaudio_load(full_path)
        
        if self.rawboost:
            waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
            waveform = process_Rawboost_feature(waveform, sr=sr, algo=5)
            waveform = torch.Tensor(np.expand_dims(waveform, axis=0))
        
        waveform = pad_dataset(waveform, self.audio_length)
        
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        return waveform, filename, speaker_id
    
    def collate_fn(self, samples):
        return default_collate(samples)


# Test function
if __name__ == "__main__":
    print("Testing SpoofCelebASV dataset loader...")
    
    # Example paths (adjust based on your setup)
    audio_path = "/path/to/spoofceleb/flac/train"
    protocol_path = "/path/to/spoofceleb/metadata/train.csv"
    
    if os.path.exists(protocol_path):
        dataset = SpoofCelebASV(
            path_to_features=audio_path,
            path_to_protocol=protocol_path,
            bonafide_only=True,
            audio_length=64600
        )
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Number of speakers: {dataset.num_speakers}")
        
        # Test loading one sample
        waveform, filename, speaker_id = dataset[0]
        print(f"\nSample 0:")
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Filename: {filename}")
        print(f"  Speaker ID: {speaker_id}")
    else:
        print(f"Protocol file not found: {protocol_path}")
        print("Please update paths in the test section.")

