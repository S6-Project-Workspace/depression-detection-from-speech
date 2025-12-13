"""
Dataset Classes for Depression Detection

This module implements PyTorch Dataset classes with:
- Data loading from preprocessed chunks
- Online data augmentation (Section 7)
- Stratified Group K-Fold cross-validation (Section 3)
"""

import os
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from collections import defaultdict
import logging

try:
    from audiomentations import (
        Compose, PitchShift, TimeStretch, AddGaussianNoise,
        Normalize, Gain
    )
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False

from config import (
    DepressionDetectionConfig, 
    AugmentationConfig,
    PreprocessingConfig
)
from preprocessing import AudioChunk, build_dataset_manifest, AudioPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaveformAugmenter:
    """Online waveform augmentation pipeline.
    
    Implements Section 7.1: Waveform Augmentations (Online)
    - Pitch Shifting: ±4 semitones
    - Time Stretching: 0.8x to 1.25x speed
    - Additive Noise: SNR 5-20 dB
    """
    
    def __init__(self, config: AugmentationConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        if HAS_AUDIOMENTATIONS:
            transforms = []
            
            if config.pitch_shift_enabled:
                transforms.append(
                    PitchShift(
                        min_semitones=config.pitch_shift_semitones[0],
                        max_semitones=config.pitch_shift_semitones[1],
                        p=config.pitch_shift_prob
                    )
                )
            
            if config.time_stretch_enabled:
                transforms.append(
                    TimeStretch(
                        min_rate=config.time_stretch_rate[0],
                        max_rate=config.time_stretch_rate[1],
                        p=config.time_stretch_prob
                    )
                )
            
            if config.additive_noise_enabled:
                transforms.append(
                    AddGaussianNoise(
                        min_amplitude=0.001,
                        max_amplitude=0.015,
                        p=config.additive_noise_prob
                    )
                )
            
            self.augmenter = Compose(transforms) if transforms else None
        else:
            self.augmenter = None
            logger.warning("audiomentations not installed. Using basic augmentation.")
    
    def _basic_pitch_shift(self, waveform: np.ndarray) -> np.ndarray:
        """Basic pitch shift without external libraries."""
        if random.random() > self.config.pitch_shift_prob:
            return waveform
        
        # Simple resampling-based pitch shift
        shift_factor = 2 ** (random.uniform(
            self.config.pitch_shift_semitones[0],
            self.config.pitch_shift_semitones[1]
        ) / 12)
        
        indices = np.arange(0, len(waveform), shift_factor)
        indices = indices[indices < len(waveform)].astype(int)
        shifted = waveform[indices]
        
        # Pad or trim to original length
        if len(shifted) < len(waveform):
            shifted = np.pad(shifted, (0, len(waveform) - len(shifted)))
        else:
            shifted = shifted[:len(waveform)]
        
        return shifted
    
    def _basic_time_stretch(self, waveform: np.ndarray) -> np.ndarray:
        """Basic time stretch without external libraries."""
        if random.random() > self.config.time_stretch_prob:
            return waveform
        
        rate = random.uniform(
            self.config.time_stretch_rate[0],
            self.config.time_stretch_rate[1]
        )
        
        # Simple interpolation-based time stretch
        original_length = len(waveform)
        new_length = int(original_length / rate)
        indices = np.linspace(0, original_length - 1, new_length).astype(int)
        stretched = waveform[indices]
        
        # Pad or trim to original length
        if len(stretched) < original_length:
            stretched = np.pad(stretched, (0, original_length - len(stretched)))
        else:
            stretched = stretched[:original_length]
        
        return stretched
    
    def _basic_add_noise(self, waveform: np.ndarray) -> np.ndarray:
        """Add Gaussian noise with random SNR."""
        if random.random() > self.config.additive_noise_prob:
            return waveform
        
        snr_db = random.uniform(
            self.config.snr_range_db[0],
            self.config.snr_range_db[1]
        )
        
        signal_power = np.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(waveform))
        
        return waveform + noise.astype(waveform.dtype)
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to waveform.
        
        Args:
            waveform: Input tensor of shape (1, samples) or (samples,)
            
        Returns:
            Augmented waveform tensor
        """
        # Convert to numpy
        if waveform.dim() == 2:
            waveform_np = waveform.squeeze(0).numpy()
        else:
            waveform_np = waveform.numpy()
        
        if self.augmenter is not None:
            waveform_np = self.augmenter(waveform_np, sample_rate=self.sample_rate)
        else:
            # Apply basic augmentations
            if self.config.pitch_shift_enabled:
                waveform_np = self._basic_pitch_shift(waveform_np)
            if self.config.time_stretch_enabled:
                waveform_np = self._basic_time_stretch(waveform_np)
            if self.config.additive_noise_enabled:
                waveform_np = self._basic_add_noise(waveform_np)
        
        # Convert back to tensor
        return torch.from_numpy(waveform_np).unsqueeze(0)


class SpecAugment:
    """SpecAugment for spectrogram augmentation.
    
    Implements Section 7.2: Spectrogram Augmentations (SpecAugment)
    - Frequency Masking: 2 bands, max width 30 bins
    - Time Masking: 2 blocks, max width 40 steps
    - Time Warping: sparse warping with W=5
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def frequency_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to spectrogram.
        
        Args:
            spectrogram: Shape (freq_bins, time_steps)
        """
        if not self.config.freq_mask_enabled:
            return spectrogram
        
        spec = spectrogram.clone()
        freq_bins = spec.shape[0]
        
        for _ in range(self.config.freq_mask_num):
            f = random.randint(0, min(self.config.freq_mask_param, freq_bins))
            f0 = random.randint(0, freq_bins - f)
            spec[f0:f0 + f, :] = 0
        
        return spec
    
    def time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking to spectrogram.
        
        Args:
            spectrogram: Shape (freq_bins, time_steps)
        """
        if not self.config.time_mask_enabled:
            return spectrogram
        
        spec = spectrogram.clone()
        time_steps = spec.shape[1]
        
        for _ in range(self.config.time_mask_num):
            t = random.randint(0, min(self.config.time_mask_param, time_steps))
            t0 = random.randint(0, time_steps - t)
            spec[:, t0:t0 + t] = 0
        
        return spec
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Shape (freq_bins, time_steps) or (batch, freq_bins, time_steps)
        """
        spectrogram = self.frequency_mask(spectrogram)
        spectrogram = self.time_mask(spectrogram)
        return spectrogram


class DepressionDataset(Dataset):
    """PyTorch Dataset for depression detection.
    
    Handles loading preprocessed audio chunks with augmentation.
    """
    
    def __init__(
        self,
        chunks: List[AudioChunk],
        config: DepressionDetectionConfig,
        is_training: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            chunks: List of preprocessed AudioChunk objects
            config: Configuration object
            is_training: Whether this is for training (enables augmentation)
            transform: Optional additional transform
        """
        self.chunks = chunks
        self.config = config
        self.is_training = is_training
        self.transform = transform
        
        # Initialize augmenters
        if is_training:
            self.waveform_augmenter = WaveformAugmenter(
                config.augmentation,
                config.preprocessing.sampling_rate
            )
            self.spec_augmenter = SpecAugment(config.augmentation)
        else:
            self.waveform_augmenter = None
            self.spec_augmenter = None
        
        # Build speaker to chunk index mapping
        self.speaker_to_chunks: Dict[str, List[int]] = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            self.speaker_to_chunks[chunk.speaker_id].append(idx)
        
        # Compute class weights for loss function
        labels = [chunk.label for chunk in chunks]
        self.class_counts = {0: labels.count(0), 1: labels.count(1)}
        total = len(labels)
        self.class_weights = torch.tensor([
            total / (2 * self.class_counts[0]),
            total / (2 * self.class_counts[1])
        ], dtype=torch.float32)
        
        logger.info(f"Dataset initialized with {len(chunks)} chunks")
        logger.info(f"Class distribution: {self.class_counts}")
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Returns:
            Dictionary with keys:
                - waveform: (1, chunk_samples)
                - attention_mask: (chunk_samples,)
                - label: scalar
                - speaker_id: string
        """
        chunk = self.chunks[idx]
        
        waveform = chunk.waveform.clone()
        attention_mask = chunk.attention_mask.clone()
        
        # Apply waveform augmentation during training
        if self.is_training and self.waveform_augmenter is not None:
            waveform = self.waveform_augmenter(waveform)
        
        # Apply custom transform
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        return {
            'waveform': waveform.float(),
            'attention_mask': attention_mask.float(),
            'label': torch.tensor(chunk.label, dtype=torch.long),
            'speaker_id': chunk.speaker_id,
            'file_path': chunk.file_path,
            'chunk_idx': chunk.chunk_idx
        }
    
    def get_speaker_ids(self) -> List[str]:
        """Get list of unique speaker IDs."""
        return list(self.speaker_to_chunks.keys())
    
    def get_labels(self) -> List[int]:
        """Get list of all labels."""
        return [chunk.label for chunk in self.chunks]
    
    def get_speaker_labels(self) -> Dict[str, int]:
        """Get speaker to label mapping."""
        speaker_labels = {}
        for chunk in self.chunks:
            speaker_labels[chunk.speaker_id] = chunk.label
        return speaker_labels


class MelSpectrogramDataset(Dataset):
    """Dataset that returns Mel spectrograms for ECAPA-TDNN.
    
    Computes Mel filterbank features from waveforms.
    """
    
    def __init__(
        self,
        chunks: List[AudioChunk],
        config: DepressionDetectionConfig,
        is_training: bool = True
    ):
        self.chunks = chunks
        self.config = config
        self.is_training = is_training
        
        # Mel spectrogram transform
        ecapa_config = config.ecapa_model
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.preprocessing.sampling_rate,
            n_mels=ecapa_config.n_mels,
            n_fft=int(ecapa_config.window_size_ms * config.preprocessing.sampling_rate / 1000),
            hop_length=int(ecapa_config.hop_size_ms * config.preprocessing.sampling_rate / 1000),
            f_min=0,
            f_max=config.preprocessing.sampling_rate // 2
        )
        
        # SpecAugment
        if is_training:
            self.spec_augmenter = SpecAugment(config.augmentation)
            self.waveform_augmenter = WaveformAugmenter(
                config.augmentation,
                config.preprocessing.sampling_rate
            )
        else:
            self.spec_augmenter = None
            self.waveform_augmenter = None
        
        # Class weights
        labels = [chunk.label for chunk in chunks]
        self.class_counts = {0: labels.count(0), 1: labels.count(1)}
        total = len(labels)
        self.class_weights = torch.tensor([
            total / (2 * self.class_counts[0]),
            total / (2 * self.class_counts[1])
        ], dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def _apply_cmvn(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply Cepstral Mean and Variance Normalization."""
        if not self.config.ecapa_model.apply_cmvn:
            return mel_spec
        
        mean = mel_spec.mean(dim=-1, keepdim=True)
        std = mel_spec.std(dim=-1, keepdim=True)
        return (mel_spec - mean) / (std + 1e-8)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        waveform = chunk.waveform.clone()
        
        # Apply waveform augmentation
        if self.is_training and self.waveform_augmenter is not None:
            waveform = self.waveform_augmenter(waveform)
        
        # Compute Mel spectrogram
        mel_spec = self.mel_transform(waveform)  # (1, n_mels, time)
        mel_spec = torch.log(mel_spec + 1e-9)  # Log mel
        mel_spec = mel_spec.squeeze(0)  # (n_mels, time)
        
        # Apply CMVN
        mel_spec = self._apply_cmvn(mel_spec)
        
        # Apply SpecAugment
        if self.is_training and self.spec_augmenter is not None:
            mel_spec = self.spec_augmenter(mel_spec)
        
        return {
            'mel_spectrogram': mel_spec.float(),
            'label': torch.tensor(chunk.label, dtype=torch.long),
            'speaker_id': chunk.speaker_id,
            'file_path': chunk.file_path
        }


class StratifiedGroupKFold:
    """Stratified Group K-Fold cross-validation.
    
    Implements Section 3.1: Stratified Group K-Fold Implementation
    - Prevents speaker overlap between folds (Group constraint)
    - Preserves class ratio across folds (Stratified constraint)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        max_class_deviation: float = 0.05
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.max_class_deviation = max_class_deviation
    
    def split(
        self,
        chunks: List[AudioChunk],
        groups: Optional[List[str]] = None
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate train/val indices for K folds.
        
        Args:
            chunks: List of AudioChunk objects
            groups: Optional list of speaker IDs (extracted from chunks if not provided)
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if groups is None:
            groups = [chunk.speaker_id for chunk in chunks]
        
        labels = [chunk.label for chunk in chunks]
        
        # Group chunks by speaker
        speaker_chunks: Dict[str, List[int]] = defaultdict(list)
        speaker_labels: Dict[str, int] = {}
        
        for idx, (chunk, group) in enumerate(zip(chunks, groups)):
            speaker_chunks[group].append(idx)
            speaker_labels[group] = chunk.label
        
        speakers = list(speaker_chunks.keys())
        speaker_label_list = [speaker_labels[s] for s in speakers]
        
        # Compute global class ratio
        global_ratio = sum(speaker_label_list) / len(speaker_label_list)
        
        # Shuffle speakers
        if self.shuffle:
            random.seed(self.random_state)
            combined = list(zip(speakers, speaker_label_list))
            random.shuffle(combined)
            speakers, speaker_label_list = zip(*combined)
            speakers = list(speakers)
            speaker_label_list = list(speaker_label_list)
        
        # Assign speakers to folds while maintaining class balance
        folds = [[] for _ in range(self.n_splits)]
        fold_labels = [[] for _ in range(self.n_splits)]
        
        # Sort by label to alternate assignment
        positive_speakers = [s for s, l in zip(speakers, speaker_label_list) if l == 1]
        negative_speakers = [s for s, l in zip(speakers, speaker_label_list) if l == 0]
        
        # Distribute positive speakers evenly
        for i, speaker in enumerate(positive_speakers):
            fold_idx = i % self.n_splits
            folds[fold_idx].append(speaker)
            fold_labels[fold_idx].append(1)
        
        # Distribute negative speakers evenly
        for i, speaker in enumerate(negative_speakers):
            fold_idx = i % self.n_splits
            folds[fold_idx].append(speaker)
            fold_labels[fold_idx].append(0)
        
        # Generate splits
        splits = []
        for val_fold_idx in range(self.n_splits):
            val_speakers = set(folds[val_fold_idx])
            train_speakers = set()
            for i in range(self.n_splits):
                if i != val_fold_idx:
                    train_speakers.update(folds[i])
            
            # Convert to chunk indices
            train_indices = []
            val_indices = []
            
            for speaker, chunk_indices in speaker_chunks.items():
                if speaker in train_speakers:
                    train_indices.extend(chunk_indices)
                elif speaker in val_speakers:
                    val_indices.extend(chunk_indices)
            
            # Verify class balance
            train_labels = [labels[i] for i in train_indices]
            val_labels = [labels[i] for i in val_indices]
            
            train_ratio = sum(train_labels) / len(train_labels) if train_labels else 0
            val_ratio = sum(val_labels) / len(val_labels) if val_labels else 0
            
            logger.info(f"Fold {val_fold_idx + 1}: "
                       f"Train={len(train_indices)} (ratio={train_ratio:.3f}), "
                       f"Val={len(val_indices)} (ratio={val_ratio:.3f})")
            
            splits.append((train_indices, val_indices))
        
        return splits


class BalancedSampler(Sampler):
    """Balanced sampler for handling class imbalance.
    
    Oversamples minority class to achieve balanced batches.
    """
    
    def __init__(
        self,
        dataset: DepressionDataset,
        replacement: bool = True
    ):
        self.dataset = dataset
        self.replacement = replacement
        
        labels = dataset.get_labels()
        self.indices = list(range(len(labels)))
        
        # Compute sample weights
        class_counts = dataset.class_counts
        total = sum(class_counts.values())
        weights = [total / (2 * class_counts[label]) for label in labels]
        self.weights = torch.DoubleTensor(weights)
    
    def __iter__(self):
        return iter(torch.multinomial(
            self.weights,
            len(self.weights),
            replacement=self.replacement
        ).tolist())
    
    def __len__(self):
        return len(self.dataset)


def create_dataloaders(
    config: DepressionDetectionConfig,
    fold_idx: int = 0,
    language: str = "combined"
) -> Tuple[DataLoader, DataLoader, List[AudioChunk]]:
    """Create train and validation dataloaders for a specific fold.
    
    Args:
        config: Configuration object
        fold_idx: Fold index for cross-validation
        language: Language to use ("tamil", "malayalam", or "combined")
        
    Returns:
        Tuple of (train_loader, val_loader, all_chunks)
    """
    # Build dataset manifest
    chunks, speaker_map = build_dataset_manifest(
        config.paths,
        config.preprocessing,
        language=language,
        is_training=True
    )
    
    # Create cross-validation splits
    cv = StratifiedGroupKFold(
        n_splits=config.training.n_folds,
        shuffle=True,
        random_state=config.training.cv_random_seed
    )
    
    splits = cv.split(chunks)
    train_indices, val_indices = splits[fold_idx]
    
    # Create train and val chunk lists
    train_chunks = [chunks[i] for i in train_indices]
    val_chunks = [chunks[i] for i in val_indices]
    
    # Create datasets
    train_dataset = DepressionDataset(train_chunks, config, is_training=True)
    val_dataset = DepressionDataset(val_chunks, config, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=BalancedSampler(train_dataset),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, chunks


def create_mel_dataloaders(
    config: DepressionDetectionConfig,
    fold_idx: int = 0,
    language: str = "combined"
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for ECAPA-TDNN (Mel spectrogram input)."""
    chunks, _ = build_dataset_manifest(
        config.paths,
        config.preprocessing,
        language=language,
        is_training=True
    )
    
    cv = StratifiedGroupKFold(
        n_splits=config.training.n_folds,
        shuffle=True,
        random_state=config.training.cv_random_seed
    )
    
    splits = cv.split(chunks)
    train_indices, val_indices = splits[fold_idx]
    
    train_chunks = [chunks[i] for i in train_indices]
    val_chunks = [chunks[i] for i in val_indices]
    
    train_dataset = MelSpectrogramDataset(train_chunks, config, is_training=True)
    val_dataset = MelSpectrogramDataset(val_chunks, config, is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    from config import get_config
    
    config = get_config("combined")
    
    print("Creating dataloaders...")
    train_loader, val_loader, chunks = create_dataloaders(config, fold_idx=0)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  Waveform shape: {batch['waveform'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels: {batch['label']}")
