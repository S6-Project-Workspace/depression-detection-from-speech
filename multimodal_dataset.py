"""
Multimodal Dataset for Depression Detection

This module implements PyTorch Dataset classes for multimodal (audio + text)
depression detection, handling audio-transcript pairing and speaker-independent
cross-validation. Extended to include linguistic features from traditional NLP tasks.

Reference: Multimodal NLP Upgrade - Requirements 6.1, 6.2, 6.3, 6.4
Reference: Traditional NLP Tasks - Task 6

Feature: multimodal-nlp-upgrade, traditional-nlp-tasks
"""

import os
import json
import logging
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from config import (
    DepressionDetectionConfig,
    TextModelConfig,
    FusionConfig
)
from preprocessing import AudioChunk, build_dataset_manifest
from text_preprocessing import TextPreprocessor, create_preprocessor, PreprocessedText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalSample:
    """A single multimodal sample containing audio, text, and linguistic data.
    
    Extended to include linguistic features from traditional NLP tasks.
    
    Reference: Multimodal NLP Upgrade - Requirement 6.4
    Reference: Traditional NLP Tasks - Task 6
    """
    # Audio data
    audio_path: str
    waveform: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    
    # Text data
    transcript_path: Optional[str] = None
    text: str = ""
    preprocessed_text: str = ""
    input_ids: Optional[torch.Tensor] = None
    text_attention_mask: Optional[torch.Tensor] = None
    
    # Linguistic features (Traditional NLP Tasks - Task 6)
    linguistic_features: Optional[torch.Tensor] = None
    linguistic_analysis: Optional[Any] = None  # LinguisticAnalysis object
    has_linguistic_features: bool = False
    
    # Metadata
    label: int = 0
    speaker_id: str = ""
    language: str = ""
    chunk_idx: int = 0
    
    def has_audio(self) -> bool:
        """Check if audio data is available."""
        return self.waveform is not None
    
    def has_text(self) -> bool:
        """Check if text data is available."""
        return bool(self.text.strip())
    
    def has_linguistic(self) -> bool:
        """Check if linguistic features are available."""
        return self.has_linguistic_features and self.linguistic_features is not None
    
    def is_complete(self) -> bool:
        """Check if all modalities are available."""
        return self.has_audio() and self.has_text() and self.has_linguistic()
    
    def is_bimodal_complete(self) -> bool:
        """Check if audio and text modalities are available."""
        return self.has_audio() and self.has_text()


@dataclass
class TranscriptData:
    """Transcript data loaded from JSON file."""
    text: str
    normalized_text: str
    morphemes: List[str]
    language: str
    speaker_id: str
    label: int
    confidence: float = 1.0
    audio_file: str = ""
    
    @classmethod
    def from_json(cls, json_path: str) -> Optional["TranscriptData"]:
        """Load transcript data from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return cls(
                text=data.get("transcript", ""),
                normalized_text=data.get("normalized_text", data.get("transcript", "")),
                morphemes=data.get("morphemes", []),
                language=data.get("language", ""),
                speaker_id=data.get("speaker_id", ""),
                label=data.get("label", 0),
                confidence=data.get("confidence", 1.0),
                audio_file=data.get("audio_file", "")
            )
        except Exception as e:
            logger.warning(f"Failed to load transcript from {json_path}: {e}")
            return None


class MultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal depression detection.
    
    Extended to include linguistic features from traditional NLP tasks.
    
    Handles:
    1. Audio-transcript pairing - Requirement 6.1
    2. Missing transcript handling via ASR - Requirement 6.2
    3. Aligned audio chunks and text segments - Requirement 6.4
    4. Linguistic feature extraction and caching - Task 6
    
    Reference: Multimodal NLP Upgrade - Requirements 6.1, 6.2, 6.4
    Reference: Traditional NLP Tasks - Task 6
    """
    
    def __init__(
        self,
        audio_chunks: List[AudioChunk],
        transcripts: Dict[str, TranscriptData],
        tokenizer: Any,
        config: DepressionDetectionConfig,
        text_preprocessor: Optional[TextPreprocessor] = None,
        linguistic_analyzer: Optional[Any] = None,
        is_training: bool = True,
        generate_missing_transcripts: bool = False,
        enable_linguistic_features: bool = True,
        linguistic_cache_dir: Optional[str] = None
    ):
        """Initialize multimodal dataset.
        
        Args:
            audio_chunks: List of preprocessed AudioChunk objects
            transcripts: Dictionary mapping audio filename to TranscriptData
            tokenizer: Text tokenizer (MuRIL tokenizer wrapper)
            config: Configuration object
            text_preprocessor: Optional text preprocessor for normalization
            linguistic_analyzer: Optional LinguisticAnalyzer for feature extraction
            is_training: Whether this is for training
            generate_missing_transcripts: Whether to generate missing transcripts via ASR
            enable_linguistic_features: Whether to extract linguistic features
            linguistic_cache_dir: Directory for caching linguistic features
        """
        self.audio_chunks = audio_chunks
        self.transcripts = transcripts
        self.tokenizer = tokenizer
        self.config = config
        self.text_preprocessor = text_preprocessor
        self.linguistic_analyzer = linguistic_analyzer
        self.is_training = is_training
        self.generate_missing_transcripts = generate_missing_transcripts
        self.enable_linguistic_features = enable_linguistic_features
        self.linguistic_cache_dir = linguistic_cache_dir
        
        # ASR pipeline for generating missing transcripts (lazy loaded)
        self._asr_pipeline = None
        
        # Linguistic feature cache
        self._linguistic_cache: Dict[str, torch.Tensor] = {}
        self._setup_linguistic_cache()
        
        # Build audio filename to chunk index mapping
        self.audio_to_chunks: Dict[str, List[int]] = defaultdict(list)
        for idx, chunk in enumerate(audio_chunks):
            audio_filename = os.path.basename(chunk.file_path)
            self.audio_to_chunks[audio_filename].append(idx)
        
        # Build speaker to chunk index mapping
        self.speaker_to_chunks: Dict[str, List[int]] = defaultdict(list)
        for idx, chunk in enumerate(audio_chunks):
            self.speaker_to_chunks[chunk.speaker_id].append(idx)
        
        # Compute class weights
        labels = [chunk.label for chunk in audio_chunks]
        self.class_counts = {0: labels.count(0), 1: labels.count(1)}
        total = len(labels)
        if self.class_counts[0] > 0 and self.class_counts[1] > 0:
            self.class_weights = torch.tensor([
                total / (2 * self.class_counts[0]),
                total / (2 * self.class_counts[1])
            ], dtype=torch.float32)
        else:
            self.class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        
        # Statistics
        self._count_transcript_coverage()
        self._count_linguistic_coverage()
        
        logger.info(f"MultimodalDataset initialized with {len(audio_chunks)} audio chunks")
        logger.info(f"Linguistic features enabled: {self.enable_linguistic_features}")
        if self.enable_linguistic_features:
            logger.info(f"Linguistic coverage: {self.linguistic_coverage:.1%}")
    
    def _setup_linguistic_cache(self):
        """Setup linguistic feature caching system."""
        if not self.enable_linguistic_features or not self.linguistic_cache_dir:
            return
        
        os.makedirs(self.linguistic_cache_dir, exist_ok=True)
        cache_file = os.path.join(self.linguistic_cache_dir, "linguistic_features.pkl")
        
        # Load existing cache if available
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self._linguistic_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._linguistic_cache)} cached linguistic features")
            except Exception as e:
                logger.warning(f"Failed to load linguistic cache: {e}")
                self._linguistic_cache = {}
        else:
            self._linguistic_cache = {}
    
    def _save_linguistic_cache(self):
        """Save linguistic feature cache to disk."""
        if not self.linguistic_cache_dir or not self._linguistic_cache:
            return
        
        cache_file = os.path.join(self.linguistic_cache_dir, "linguistic_features.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._linguistic_cache, f)
            logger.info(f"Saved {len(self._linguistic_cache)} linguistic features to cache")
        except Exception as e:
            logger.warning(f"Failed to save linguistic cache: {e}")
    
    def _count_linguistic_coverage(self):
        """Count how many samples have linguistic features available."""
        if not self.enable_linguistic_features:
            self.linguistic_coverage = 0.0
            return
        
        samples_with_linguistic = 0
        for chunk in self.audio_chunks:
            audio_filename = os.path.basename(chunk.file_path)
            if audio_filename in self.transcripts:
                transcript = self.transcripts[audio_filename]
                if transcript.text and transcript.text.strip():
                    # Check if cached or can be computed
                    cache_key = self._get_linguistic_cache_key(transcript.text, transcript.language)
                    if cache_key in self._linguistic_cache or self.linguistic_analyzer is not None:
                        samples_with_linguistic += 1
        
        self.linguistic_coverage = samples_with_linguistic / len(self.audio_chunks) if self.audio_chunks else 0.0
    
    def _get_linguistic_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for linguistic features."""
        import hashlib
        text_hash = hashlib.md5(f"{text}_{language}".encode()).hexdigest()
        return f"{language}_{text_hash}"
    
    def _extract_linguistic_features(self, text: str, language: str) -> Optional[torch.Tensor]:
        """Extract linguistic features from text with caching.
        
        Args:
            text: Raw text for analysis
            language: Language code
            
        Returns:
            Linguistic features tensor of shape (128,) or None if extraction fails
        """
        if not self.enable_linguistic_features or not text or not text.strip():
            return None
        
        # Check cache first
        cache_key = self._get_linguistic_cache_key(text, language)
        if cache_key in self._linguistic_cache:
            return self._linguistic_cache[cache_key]
        
        # Extract features if analyzer is available
        if self.linguistic_analyzer is None:
            return None
        
        try:
            # Analyze text using linguistic analyzer
            analysis = self.linguistic_analyzer.analyze_text(text, language)
            features = self.linguistic_analyzer.extract_linguistic_features(analysis)
            
            # Convert to tensor and cache
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            self._linguistic_cache[cache_key] = feature_tensor
            
            return feature_tensor
            
        except Exception as e:
            logger.warning(f"Failed to extract linguistic features for text: {e}")
            return None
        logger.info(f"Transcript coverage: {self.transcript_coverage:.1%}")
        logger.info(f"Class distribution: {self.class_counts}")
    
    def _count_transcript_coverage(self):
        """Count how many audio files have transcripts."""
        unique_audio_files = set(os.path.basename(c.file_path) for c in self.audio_chunks)
        files_with_transcripts = sum(1 for f in unique_audio_files if f in self.transcripts)
        self.transcript_coverage = files_with_transcripts / len(unique_audio_files) if unique_audio_files else 0
    
    def _get_asr_pipeline(self):
        """Lazy load ASR pipeline."""
        if self._asr_pipeline is None and self.generate_missing_transcripts:
            try:
                from asr_pipeline import create_asr_pipeline
                self._asr_pipeline = create_asr_pipeline(self.config.asr)
                logger.info("ASR pipeline loaded for generating missing transcripts")
            except Exception as e:
                logger.warning(f"Failed to load ASR pipeline: {e}")
        return self._asr_pipeline
    
    def _get_transcript_for_chunk(self, chunk: AudioChunk) -> Optional[TranscriptData]:
        """Get transcript for an audio chunk.
        
        Args:
            chunk: AudioChunk object
            
        Returns:
            TranscriptData or None if not available
        """
        audio_filename = os.path.basename(chunk.file_path)
        
        # Check if transcript exists
        if audio_filename in self.transcripts:
            return self.transcripts[audio_filename]
        
        # Try to generate transcript if enabled
        if self.generate_missing_transcripts:
            asr = self._get_asr_pipeline()
            if asr is not None:
                try:
                    # Determine language from path
                    language = "tamil" if "Tamil" in chunk.file_path else "malayalam"
                    result = asr.transcribe(chunk.file_path, language=language)
                    
                    if result.text and not result.error:
                        transcript = TranscriptData(
                            text=result.text,
                            normalized_text=result.text,
                            morphemes=[],
                            language=language,
                            speaker_id=chunk.speaker_id,
                            label=chunk.label,
                            confidence=result.confidence,
                            audio_file=audio_filename
                        )
                        # Cache the transcript
                        self.transcripts[audio_filename] = transcript
                        return transcript
                except Exception as e:
                    logger.warning(f"Failed to generate transcript for {audio_filename}: {e}")
        
        return None
    
    def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text using the text preprocessor.
        
        Args:
            text: Raw text
            language: Language code
            
        Returns:
            Preprocessed text (morpheme-segmented)
        """
        if self.text_preprocessor is None:
            return text
        
        try:
            result = self.text_preprocessor.preprocess(text, language=language)
            return result.morpheme_text if result.morpheme_text else result.normalized
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}")
            return text
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text for the model.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if self.tokenizer is None:
            # Return dummy tokens if no tokenizer
            return {
                "input_ids": torch.zeros(1, 32, dtype=torch.long),
                "attention_mask": torch.zeros(1, 32, dtype=torch.long)
            }
        
        try:
            return self.tokenizer.tokenize(text)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return {
                "input_ids": torch.zeros(1, 32, dtype=torch.long),
                "attention_mask": torch.zeros(1, 32, dtype=torch.long)
            }
    
    def __len__(self) -> int:
        return len(self.audio_chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single multimodal sample.
        
        Extended to include linguistic features from traditional NLP tasks.
        
        Returns:
            Dictionary with keys:
                - waveform: (1, chunk_samples)
                - audio_attention_mask: (chunk_samples,)
                - input_ids: (seq_len,)
                - text_attention_mask: (seq_len,)
                - linguistic_features: (128,) - linguistic feature vector
                - label: scalar
                - speaker_id: string
                - text: raw text string
                - language: language code
                - has_text: bool
                - has_audio: bool
                - has_linguistic: bool
        """
        chunk = self.audio_chunks[idx]
        
        # Get audio data
        waveform = chunk.waveform.clone()
        audio_attention_mask = chunk.attention_mask.clone()
        
        # Get transcript
        transcript = self._get_transcript_for_chunk(chunk)
        
        # Process text if available
        if transcript is not None and transcript.text:
            text = transcript.text
            language = transcript.language
            preprocessed_text = self._preprocess_text(text, language)
            tokens = self._tokenize_text(preprocessed_text)
            input_ids = tokens["input_ids"].squeeze(0)
            text_attention_mask = tokens["attention_mask"].squeeze(0)
            has_text = True
            
            # Extract linguistic features
            linguistic_features = self._extract_linguistic_features(text, language)
            has_linguistic = linguistic_features is not None
            
            if not has_linguistic:
                # Use zero features as fallback
                linguistic_features = torch.zeros(128, dtype=torch.float32)
        else:
            text = ""
            language = "tamil"  # Default language
            preprocessed_text = ""
            input_ids = torch.zeros(32, dtype=torch.long)
            text_attention_mask = torch.zeros(32, dtype=torch.long)
            linguistic_features = torch.zeros(128, dtype=torch.float32)
            has_text = False
            has_linguistic = False
        
        return {
            'waveform': waveform.float(),
            'audio_attention_mask': audio_attention_mask.float(),
            'input_ids': input_ids,
            'text_attention_mask': text_attention_mask,
            'linguistic_features': linguistic_features,
            'label': torch.tensor(chunk.label, dtype=torch.long),
            'speaker_id': chunk.speaker_id,
            'file_path': chunk.file_path,
            'chunk_idx': chunk.chunk_idx,
            'text': text,
            'language': language,
            'has_text': has_text,
            'has_audio': True,
            'has_linguistic': has_linguistic
        }
    
    def get_speaker_ids(self) -> List[str]:
        """Get list of unique speaker IDs."""
        return list(self.speaker_to_chunks.keys())
    
    def get_labels(self) -> List[int]:
        """Get list of all labels."""
        return [chunk.label for chunk in self.audio_chunks]
    
    def get_speaker_labels(self) -> Dict[str, int]:
        """Get speaker to label mapping."""
        speaker_labels = {}
        for chunk in self.audio_chunks:
            speaker_labels[chunk.speaker_id] = chunk.label
        return speaker_labels


class SpeakerIndependentSplitter:
    """Speaker-independent cross-validation splitter.
    
    Ensures no speaker overlap between train and validation sets.
    
    Reference: Multimodal NLP Upgrade - Requirement 6.3
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """Initialize splitter.
        
        Args:
            n_splits: Number of cross-validation folds
            shuffle: Whether to shuffle speakers before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        chunks: List[AudioChunk]
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate speaker-independent train/val splits.
        
        Args:
            chunks: List of AudioChunk objects
            
        Returns:
            List of (train_indices, val_indices) tuples
            
        Reference: Requirement 6.3 - Speaker-independent cross-validation
        """
        # Group chunks by speaker
        speaker_chunks: Dict[str, List[int]] = defaultdict(list)
        speaker_labels: Dict[str, int] = {}
        
        for idx, chunk in enumerate(chunks):
            speaker_chunks[chunk.speaker_id].append(idx)
            speaker_labels[chunk.speaker_id] = chunk.label
        
        speakers = list(speaker_chunks.keys())
        
        # Shuffle speakers if requested
        if self.shuffle:
            random.seed(self.random_state)
            random.shuffle(speakers)
        
        # Separate speakers by class for stratification
        positive_speakers = [s for s in speakers if speaker_labels[s] == 1]
        negative_speakers = [s for s in speakers if speaker_labels[s] == 0]
        
        # Distribute speakers to folds
        folds: List[List[str]] = [[] for _ in range(self.n_splits)]
        
        # Distribute positive speakers evenly
        for i, speaker in enumerate(positive_speakers):
            folds[i % self.n_splits].append(speaker)
        
        # Distribute negative speakers evenly
        for i, speaker in enumerate(negative_speakers):
            folds[i % self.n_splits].append(speaker)
        
        # Generate splits
        splits = []
        for val_fold_idx in range(self.n_splits):
            val_speakers = set(folds[val_fold_idx])
            train_speakers = set()
            for i in range(self.n_splits):
                if i != val_fold_idx:
                    train_speakers.update(folds[i])
            
            # Verify no overlap (Property 9)
            assert len(val_speakers & train_speakers) == 0, \
                "Speaker overlap detected between train and validation sets"
            
            # Convert to chunk indices
            train_indices = []
            val_indices = []
            
            for speaker, chunk_indices in speaker_chunks.items():
                if speaker in train_speakers:
                    train_indices.extend(chunk_indices)
                elif speaker in val_speakers:
                    val_indices.extend(chunk_indices)
            
            splits.append((train_indices, val_indices))
            
            # Log split statistics
            train_labels = [chunks[i].label for i in train_indices]
            val_labels = [chunks[i].label for i in val_indices]
            
            logger.info(
                f"Fold {val_fold_idx + 1}: "
                f"Train={len(train_indices)} samples ({len(train_speakers)} speakers), "
                f"Val={len(val_indices)} samples ({len(val_speakers)} speakers)"
            )
        
        return splits
    
    def get_speaker_sets(
        self,
        chunks: List[AudioChunk],
        fold_idx: int
    ) -> Tuple[Set[str], Set[str]]:
        """Get train and validation speaker sets for a specific fold.
        
        Args:
            chunks: List of AudioChunk objects
            fold_idx: Fold index
            
        Returns:
            Tuple of (train_speaker_ids, val_speaker_ids)
        """
        splits = self.split(chunks)
        train_indices, val_indices = splits[fold_idx]
        
        train_speakers = set(chunks[i].speaker_id for i in train_indices)
        val_speakers = set(chunks[i].speaker_id for i in val_indices)
        
        return train_speakers, val_speakers
    
    def verify_speaker_independence(
        self,
        chunks: List[AudioChunk]
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Verify that all splits maintain speaker independence.
        
        This method validates Property 9: Speaker-Independent Cross-Validation
        by checking that no speaker appears in both train and validation sets
        for any fold.
        
        Args:
            chunks: List of AudioChunk objects
            
        Returns:
            Tuple of (is_valid, fold_details) where:
                - is_valid: True if all folds have no speaker overlap
                - fold_details: List of dictionaries with fold statistics
                
        Reference: Requirement 6.3 - Speaker-independent cross-validation
        """
        splits = self.split(chunks)
        fold_details = []
        all_valid = True
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            train_speakers = set(chunks[i].speaker_id for i in train_indices)
            val_speakers = set(chunks[i].speaker_id for i in val_indices)
            
            # Check for overlap (Property 9)
            overlap = train_speakers & val_speakers
            is_valid = len(overlap) == 0
            
            if not is_valid:
                all_valid = False
                logger.error(
                    f"Fold {fold_idx + 1}: Speaker overlap detected! "
                    f"Overlapping speakers: {overlap}"
                )
            
            # Compute class distribution
            train_labels = [chunks[i].label for i in train_indices]
            val_labels = [chunks[i].label for i in val_indices]
            
            fold_details.append({
                'fold_idx': fold_idx,
                'train_samples': len(train_indices),
                'val_samples': len(val_indices),
                'train_speakers': len(train_speakers),
                'val_speakers': len(val_speakers),
                'train_positive_ratio': sum(train_labels) / len(train_labels) if train_labels else 0,
                'val_positive_ratio': sum(val_labels) / len(val_labels) if val_labels else 0,
                'overlap_speakers': list(overlap),
                'is_valid': is_valid
            })
        
        return all_valid, fold_details


def load_transcripts_from_directory(
    transcript_dir: str,
    language: Optional[str] = None
) -> Dict[str, TranscriptData]:
    """Load all transcripts from a directory.
    
    Args:
        transcript_dir: Directory containing transcript JSON files
        language: Optional language filter
        
    Returns:
        Dictionary mapping audio filename to TranscriptData
    """
    transcripts = {}
    
    if not os.path.exists(transcript_dir):
        logger.warning(f"Transcript directory not found: {transcript_dir}")
        return transcripts
    
    for root, dirs, files in os.walk(transcript_dir):
        for filename in files:
            if filename.endswith('_transcript.json'):
                json_path = os.path.join(root, filename)
                transcript = TranscriptData.from_json(json_path)
                
                if transcript is not None:
                    # Filter by language if specified
                    if language and transcript.language != language:
                        continue
                    
                    # Extract audio filename
                    audio_filename = filename.replace('_transcript.json', '.wav')
                    transcripts[audio_filename] = transcript
    
    logger.info(f"Loaded {len(transcripts)} transcripts from {transcript_dir}")
    return transcripts


def verify_speaker_independence_for_splits(
    train_indices: List[int],
    val_indices: List[int],
    chunks: List[Any]
) -> Tuple[bool, Set[str]]:
    """Verify speaker independence between train and validation splits.
    
    This is a standalone utility function that can be used to verify
    any train/val split maintains speaker independence.
    
    Args:
        train_indices: List of training sample indices
        val_indices: List of validation sample indices
        chunks: List of chunk objects with speaker_id attribute
        
    Returns:
        Tuple of (is_valid, overlapping_speakers) where:
            - is_valid: True if no speaker overlap exists
            - overlapping_speakers: Set of speaker IDs that appear in both sets
            
    Reference: Requirement 6.3 - Speaker-independent cross-validation
    """
    train_speakers = set(chunks[i].speaker_id for i in train_indices)
    val_speakers = set(chunks[i].speaker_id for i in val_indices)
    
    overlap = train_speakers & val_speakers
    is_valid = len(overlap) == 0
    
    return is_valid, overlap


def create_multimodal_dataloaders(
    config: DepressionDetectionConfig,
    tokenizer: Any,
    transcript_dir: str,
    fold_idx: int = 0,
    language: str = "combined",
    text_preprocessor: Optional[TextPreprocessor] = None,
    linguistic_analyzer: Optional[Any] = None,
    enable_linguistic_features: bool = True,
    linguistic_cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, List[AudioChunk]]:
    """Create multimodal train and validation dataloaders.
    
    Extended to support linguistic features from traditional NLP tasks.
    
    Args:
        config: Configuration object
        tokenizer: Text tokenizer
        transcript_dir: Directory containing transcript JSON files
        fold_idx: Fold index for cross-validation
        language: Language to use ("tamil", "malayalam", or "combined")
        text_preprocessor: Optional text preprocessor
        linguistic_analyzer: Optional LinguisticAnalyzer for feature extraction
        enable_linguistic_features: Whether to extract linguistic features
        linguistic_cache_dir: Directory for caching linguistic features
        
    Returns:
        Tuple of (train_loader, val_loader, all_chunks)
    """
    # Build audio dataset manifest
    chunks, speaker_map = build_dataset_manifest(
        config.paths,
        config.preprocessing,
        language=language,
        is_training=True
    )
    
    # Load transcripts
    transcripts = load_transcripts_from_directory(transcript_dir)
    
    # Create speaker-independent splits
    splitter = SpeakerIndependentSplitter(
        n_splits=config.training.n_folds,
        shuffle=True,
        random_state=config.training.cv_random_seed
    )
    
    splits = splitter.split(chunks)
    train_indices, val_indices = splits[fold_idx]
    
    # Create train and val chunk lists
    train_chunks = [chunks[i] for i in train_indices]
    val_chunks = [chunks[i] for i in val_indices]
    
    # Setup linguistic cache directory
    if linguistic_cache_dir is None and enable_linguistic_features:
        linguistic_cache_dir = os.path.join(config.paths.output_dir, "linguistic_cache")
    
    # Create datasets
    train_dataset = MultimodalDataset(
        audio_chunks=train_chunks,
        transcripts=transcripts,
        tokenizer=tokenizer,
        config=config,
        text_preprocessor=text_preprocessor,
        linguistic_analyzer=linguistic_analyzer,
        is_training=True,
        enable_linguistic_features=enable_linguistic_features,
        linguistic_cache_dir=linguistic_cache_dir
    )
    
    val_dataset = MultimodalDataset(
        audio_chunks=val_chunks,
        transcripts=transcripts,
        tokenizer=tokenizer,
        config=config,
        text_preprocessor=text_preprocessor,
        linguistic_analyzer=linguistic_analyzer,
        is_training=False,
        enable_linguistic_features=enable_linguistic_features,
        linguistic_cache_dir=linguistic_cache_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=multimodal_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=multimodal_collate_fn
    )
    
    # Save linguistic cache after dataset creation
    if enable_linguistic_features:
        train_dataset._save_linguistic_cache()
        val_dataset._save_linguistic_cache()
    
    return train_loader, val_loader, chunks


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for multimodal batches.
    
    Extended to handle linguistic features from traditional NLP tasks.
    Handles variable-length sequences and missing modalities.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with linguistic features
    """
    # Stack tensors
    waveforms = torch.stack([item['waveform'] for item in batch])
    audio_attention_masks = torch.stack([item['audio_attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Stack linguistic features
    linguistic_features = torch.stack([item['linguistic_features'] for item in batch])
    
    # Pad text sequences to same length
    input_ids_list = [item['input_ids'] for item in batch]
    text_attention_masks_list = [item['text_attention_mask'] for item in batch]
    
    max_text_len = max(ids.shape[0] for ids in input_ids_list)
    
    padded_input_ids = []
    padded_text_masks = []
    
    for ids, mask in zip(input_ids_list, text_attention_masks_list):
        pad_len = max_text_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        padded_input_ids.append(ids)
        padded_text_masks.append(mask)
    
    input_ids = torch.stack(padded_input_ids)
    text_attention_masks = torch.stack(padded_text_masks)
    
    # Collect metadata
    speaker_ids = [item['speaker_id'] for item in batch]
    file_paths = [item['file_path'] for item in batch]
    texts = [item['text'] for item in batch]
    languages = [item['language'] for item in batch]
    has_text = torch.tensor([item['has_text'] for item in batch], dtype=torch.bool)
    has_audio = torch.tensor([item['has_audio'] for item in batch], dtype=torch.bool)
    has_linguistic = torch.tensor([item['has_linguistic'] for item in batch], dtype=torch.bool)
    
    return {
        'waveform': waveforms,
        'audio_attention_mask': audio_attention_masks,
        'input_ids': input_ids,
        'text_attention_mask': text_attention_masks,
        'linguistic_features': linguistic_features,
        'label': labels,
        'speaker_id': speaker_ids,
        'file_path': file_paths,
        'text': texts,
        'language': languages,
        'has_text': has_text,
        'has_audio': has_audio,
        'has_linguistic': has_linguistic
    }


if __name__ == "__main__":
    from config import get_config
    
    config = get_config("combined")
    
    print("Testing MultimodalDataset...")
    
    # Test speaker-independent splitter
    print("\nTesting SpeakerIndependentSplitter...")
    
    # Create mock chunks for testing
    @dataclass
    class MockChunk:
        speaker_id: str
        label: int
        file_path: str = ""
        waveform: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 16000))
        attention_mask: torch.Tensor = field(default_factory=lambda: torch.ones(16000))
        chunk_idx: int = 0
    
    # Create test data with multiple speakers
    mock_chunks = []
    for i in range(10):
        speaker_id = f"speaker_{i % 5}"
        label = i % 2
        mock_chunks.append(MockChunk(
            speaker_id=speaker_id,
            label=label,
            file_path=f"/path/to/audio_{i}.wav"
        ))
    
    splitter = SpeakerIndependentSplitter(n_splits=3, shuffle=True, random_state=42)
    splits = splitter.split(mock_chunks)
    
    print(f"Created {len(splits)} folds")
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_speakers = set(mock_chunks[i].speaker_id for i in train_idx)
        val_speakers = set(mock_chunks[i].speaker_id for i in val_idx)
        overlap = train_speakers & val_speakers
        
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train speakers: {train_speakers}")
        print(f"  Val speakers: {val_speakers}")
        print(f"  Overlap: {overlap} (should be empty)")
        
        assert len(overlap) == 0, "Speaker overlap detected!"
    
    print("\n✓ All speaker-independent splits verified - no overlap!")
    
    # Test the verification function
    print("\nTesting verify_speaker_independence...")
    is_valid, fold_details = splitter.verify_speaker_independence(mock_chunks)
    print(f"All folds valid: {is_valid}")
    for detail in fold_details:
        print(f"  Fold {detail['fold_idx'] + 1}: "
              f"train={detail['train_samples']} samples ({detail['train_speakers']} speakers), "
              f"val={detail['val_samples']} samples ({detail['val_speakers']} speakers), "
              f"valid={detail['is_valid']}")
    
    # Test standalone verification function
    print("\nTesting verify_speaker_independence_for_splits...")
    train_idx, val_idx = splits[0]
    is_valid, overlap = verify_speaker_independence_for_splits(train_idx, val_idx, mock_chunks)
    print(f"Split 0 valid: {is_valid}, overlap: {overlap}")
    
    print("\n✓ All speaker independence tests passed!")
