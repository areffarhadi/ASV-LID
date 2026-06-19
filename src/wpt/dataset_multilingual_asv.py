"""
Multilingual ASV Dataset Loader
================================

Dataset loader for multilingual speaker verification where:
- Structure: speaker_folder -> languageID_folder -> wav files
- Each speaker has data in multiple languages
- Goal: Train ASV head to be language-invariant

Returns:
    - waveform: Audio waveform tensor
    - filename: Audio filename
    - speaker_id: Numeric speaker ID (for ASV head)
    - language_id: Numeric language ID (for Language head)
"""

import torch
import os
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchaudio


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
    
    return waveform


def normalize_audio(waveform):
    """Normalize audio to zero mean and unit variance"""
    waveform = waveform - waveform.mean()
    waveform = waveform / (torch.sqrt(waveform.var() + 1e-7))
    return waveform


class MultilingualASVDataset(Dataset):
    """
    Multilingual ASV dataset for training language-invariant speaker verification
    
    Data structure:
        base_path/
            speaker_folder/ (e.g., id010001)
                languageID_folder/ (e.g., en, cy, de)
                    wav files
    
    Returns:
        - waveform: Audio waveform tensor
        - filename: Audio filename (without extension)
        - speaker_id: Numeric speaker ID (for ASV classification)
        - language_id: Numeric language ID (for Language classification)
        - speaker_folder: Original speaker folder name (e.g., id010001)
        - language_folder: Original language folder name (e.g., en)
    """
    
    def __init__(self, path_to_features, audio_length=64600, rawboost=False,
                 speaker_to_id=None, id_to_speaker=None,
                 language_to_id=None, id_to_language=None):
        """
        Args:
            path_to_features: Path to base directory containing speaker folders
            audio_length: Length of audio in samples (default 64600 = 4.0375s @ 16kHz)
            rawboost: Whether to apply RawBoost augmentation (not implemented yet)
            speaker_to_id: Optional pre-defined speaker mapping (for consistent mapping across datasets)
            id_to_speaker: Optional pre-defined reverse speaker mapping
            language_to_id: Optional pre-defined language mapping (for consistent mapping across datasets)
            id_to_language: Optional pre-defined reverse language mapping
        """
        super(MultilingualASVDataset, self).__init__()
        
        self.path_to_features = path_to_features
        self.audio_length = audio_length
        self.rawboost = rawboost
        
        self.files = []
        
        # Use provided mappings or create new ones
        if speaker_to_id is not None and language_to_id is not None:
            # Use provided mappings (for validation set to match training set)
            self.speaker_to_id = speaker_to_id.copy()
            self.id_to_speaker = id_to_speaker.copy()
            self.language_to_id = language_to_id.copy()
            self.id_to_language = id_to_language.copy()
            speaker_counter = max(self.speaker_to_id.values()) + 1 if self.speaker_to_id else 0
            language_counter = max(self.language_to_id.values()) + 1 if self.language_to_id else 0
        else:
            # Create new mappings (for training set)
            self.speaker_to_id = {}
            self.id_to_speaker = {}
            self.language_to_id = {}
            self.id_to_language = {}
            speaker_counter = 0
            language_counter = 0
        
        # Scan directory structure: speaker_folder -> languageID_folder -> wav files
        if not os.path.exists(path_to_features):
            raise ValueError(f"Path does not exist: {path_to_features}")
        
        print(f"\nScanning directory structure: {path_to_features}")
        
        # Iterate through speaker folders
        for speaker_folder in sorted(os.listdir(path_to_features)):
            speaker_path = os.path.join(path_to_features, speaker_folder)
            
            # Skip if not a directory
            if not os.path.isdir(speaker_path):
                continue
            
            # Map speaker to numeric ID
            if speaker_folder not in self.speaker_to_id:
                self.speaker_to_id[speaker_folder] = speaker_counter
                self.id_to_speaker[speaker_counter] = speaker_folder
                speaker_counter += 1
            
            speaker_id = self.speaker_to_id[speaker_folder]
            
            # Iterate through language folders
            if not os.path.isdir(speaker_path):
                continue
                
            for language_folder in sorted(os.listdir(speaker_path)):
                language_path = os.path.join(speaker_path, language_folder)
                
                # Skip if not a directory
                if not os.path.isdir(language_path):
                    continue
                
                # Map language to numeric ID
                if language_folder not in self.language_to_id:
                    # If using provided mapping, check if language exists
                    # If not in provided mapping, add it (for new languages in validation set)
                    self.language_to_id[language_folder] = language_counter
                    self.id_to_language[language_counter] = language_folder
                    language_counter += 1
                
                language_id = self.language_to_id[language_folder]
                
                # Find all wav files in this language folder
                for wav_file in sorted(os.listdir(language_path)):
                    if wav_file.endswith('.wav') or wav_file.endswith('.WAV'):
                        # Store relative path: speaker_folder/language_folder/wav_file
                        relative_path = os.path.join(speaker_folder, language_folder, wav_file)
                        self.files.append((relative_path, speaker_id, language_id, speaker_folder, language_folder))
        
        self.num_speakers = len(self.speaker_to_id)
        self.num_languages = len(self.language_to_id)
        
        print(f"\n{'='*80}")
        print(f"Loaded Multilingual ASV Dataset")
        print(f"{'='*80}")
        print(f"Base path: {path_to_features}")
        print(f"Total samples: {len(self.files)}")
        print(f"Unique speakers: {self.num_speakers}")
        print(f"Unique languages: {self.num_languages}")
        print(f"Audio length: {audio_length} samples ({audio_length/16000:.3f}s @ 16kHz)")
        print(f"\nLanguage mapping:")
        for lang_id, lang_name in sorted(self.id_to_language.items()):
            print(f"  {lang_id}: {lang_name}")
        print(f"{'='*80}\n")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        relative_path, speaker_id, language_id, speaker_folder, language_folder = self.files[idx]
        
        # Load audio file
        full_path = os.path.join(self.path_to_features, relative_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Audio file not found: {full_path}")
        
        waveform, sr = torchaudio_load(full_path)
        
        # Apply RawBoost augmentation if enabled (placeholder for now)
        if self.rawboost:
            # TODO: Implement RawBoost if needed
            pass
        
        # Pad/truncate to fixed length
        waveform = pad_dataset(waveform, self.audio_length)
        
        # Normalize audio
        waveform = normalize_audio(waveform)
        
        # Return full relative path (not just basename) for trial matching
        # relative_path format: speaker_id/language/filename.wav
        filename = relative_path
        
        # Return: waveform, filename, speaker_id, language_id, original folder names
        return waveform, filename, speaker_id, language_id, speaker_folder, language_folder
    
    def collate_fn(self, samples):
        return default_collate(samples)


# Test function
if __name__ == "__main__":
    print("Testing MultilingualASVDataset dataset loader...")
    
    train_path = os.environ.get("TIDYVOICEX_TRAIN", "")
    dev_path = os.environ.get("TIDYVOICEX_DEV", "")
    
    if os.path.exists(train_path):
        dataset = MultilingualASVDataset(
            path_to_features=train_path,
            audio_length=64600
        )
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Number of speakers: {dataset.num_speakers}")
        print(f"Number of languages: {dataset.num_languages}")
        
        # Test loading one sample
        waveform, filename, speaker_id, language_id = dataset[0]
        print(f"\nSample 0:")
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Filename: {filename}")
        print(f"  Speaker ID: {speaker_id}")
        print(f"  Language ID: {language_id}")
    else:
        print(f"Path not found: {train_path}")
        print("Please update paths in the test section.")

