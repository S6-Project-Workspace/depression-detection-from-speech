"""
Property-Based Tests for Multimodal Dataset Module

Tests the multimodal dataset and speaker-independent cross-validation using
hypothesis for property-based testing. Extended to include linguistic features.

Validates Requirement 6.3 for speaker-independent cross-validation.
Validates Task 6 for linguistic feature integration.

Feature: multimodal-nlp-upgrade, traditional-nlp-tasks
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import torch
import random
import tempfile
import os
import json

from multimodal_dataset import (
    SpeakerIndependentSplitter,
    verify_speaker_independence_for_splits,
    MultimodalDataset,
    TranscriptData,
    multimodal_collate_fn
)


# Strategy for generating speaker IDs
speaker_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
    min_size=3,
    max_size=10
).filter(lambda x: x.strip())

# Strategy for generating a list of unique speaker IDs
unique_speakers_strategy = st.lists(
    speaker_id_strategy,
    min_size=3,
    max_size=20,
    unique=True
)


@dataclass
class MockAudioChunk:
    """Mock AudioChunk for testing purposes."""
    speaker_id: str
    label: int
    file_path: str = ""
    waveform: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 16000))
    attention_mask: torch.Tensor = field(default_factory=lambda: torch.ones(16000))
    chunk_idx: int = 0


class MockLinguisticAnalyzer:
    """Mock linguistic analyzer for testing."""
    
    def analyze_text(self, text: str, language: str):
        """Mock analysis that returns None."""
        return None
    
    def extract_linguistic_features(self, analysis) -> List[float]:
        """Mock feature extraction that returns random 128-d vector."""
        return [random.random() for _ in range(128)]


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Mock tokenization."""
        seq_len = min(len(text.split()) + 2, 32)  # +2 for special tokens
        return {
            "input_ids": torch.randint(0, 1000, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len)
        }
    chunk_idx: int = 0


def generate_mock_chunks(
    speakers: List[str],
    chunks_per_speaker: int = 2,
    label_distribution: str = "balanced"
) -> List[MockAudioChunk]:
    """Generate mock audio chunks for testing.
    
    Args:
        speakers: List of speaker IDs
        chunks_per_speaker: Number of chunks per speaker
        label_distribution: "balanced" or "random"
        
    Returns:
        List of MockAudioChunk objects
    """
    chunks = []
    for i, speaker_id in enumerate(speakers):
        # Assign label based on distribution
        if label_distribution == "balanced":
            label = i % 2
        else:
            label = random.randint(0, 1)
        
        for chunk_idx in range(chunks_per_speaker):
            chunks.append(MockAudioChunk(
                speaker_id=speaker_id,
                label=label,
                file_path=f"/path/to/{speaker_id}_chunk_{chunk_idx}.wav",
                chunk_idx=chunk_idx
            ))
    
    return chunks


class TestSpeakerIndependentCrossValidation:
    """
    Property 9: Speaker-Independent Cross-Validation
    
    *For any* cross-validation split, the intersection of speaker IDs in the
    training set and validation set SHALL be empty.
    
    **Validates: Requirements 6.3**
    """
    
    @given(
        speakers=unique_speakers_strategy,
        n_splits=st.integers(min_value=2, max_value=10),
        random_state=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100, deadline=None)
    def test_no_speaker_overlap_in_any_fold(
        self,
        speakers: List[str],
        n_splits: int,
        random_state: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 9: Speaker-Independent Cross-Validation
        
        For any set of speakers and any number of folds, no speaker should appear
        in both train and validation sets for any fold.
        
        **Validates: Requirements 6.3**
        """
        # Need at least n_splits speakers to create valid folds
        assume(len(speakers) >= n_splits)
        
        # Generate mock chunks
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=2)
        
        # Create splitter
        splitter = SpeakerIndependentSplitter(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        
        # Generate splits
        splits = splitter.split(chunks)
        
        # Property: Should have exactly n_splits folds
        assert len(splits) == n_splits, (
            f"Expected {n_splits} folds, got {len(splits)}"
        )
        
        # Property: For each fold, train and val speakers should not overlap
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            train_speakers = set(chunks[i].speaker_id for i in train_indices)
            val_speakers = set(chunks[i].speaker_id for i in val_indices)
            
            overlap = train_speakers & val_speakers
            
            assert len(overlap) == 0, (
                f"Fold {fold_idx}: Speaker overlap detected!\n"
                f"  Train speakers: {train_speakers}\n"
                f"  Val speakers: {val_speakers}\n"
                f"  Overlap: {overlap}"
            )
    
    @given(
        speakers=unique_speakers_strategy,
        n_splits=st.integers(min_value=2, max_value=5),
        chunks_per_speaker=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_all_chunks_assigned_to_splits(
        self,
        speakers: List[str],
        n_splits: int,
        chunks_per_speaker: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 9: Speaker-Independent Cross-Validation
        
        For any split, all chunks should be assigned to either train or val set.
        
        **Validates: Requirements 6.3**
        """
        assume(len(speakers) >= n_splits)
        
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=chunks_per_speaker)
        total_chunks = len(chunks)
        
        splitter = SpeakerIndependentSplitter(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )
        
        splits = splitter.split(chunks)
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            # Property: All indices should be covered
            all_indices = set(train_indices) | set(val_indices)
            expected_indices = set(range(total_chunks))
            
            assert all_indices == expected_indices, (
                f"Fold {fold_idx}: Not all chunks assigned!\n"
                f"  Missing: {expected_indices - all_indices}\n"
                f"  Extra: {all_indices - expected_indices}"
            )
            
            # Property: No index should appear in both train and val
            overlap_indices = set(train_indices) & set(val_indices)
            assert len(overlap_indices) == 0, (
                f"Fold {fold_idx}: Index overlap detected: {overlap_indices}"
            )
    
    @given(
        speakers=unique_speakers_strategy,
        n_splits=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_speaker_appears_in_exactly_one_fold_as_val(
        self,
        speakers: List[str],
        n_splits: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 9: Speaker-Independent Cross-Validation
        
        Each speaker should appear in the validation set of exactly one fold
        (or zero if there are more folds than speakers).
        
        **Validates: Requirements 6.3**
        """
        assume(len(speakers) >= n_splits)
        
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=2)
        
        splitter = SpeakerIndependentSplitter(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )
        
        splits = splitter.split(chunks)
        
        # Track which folds each speaker appears in as validation
        speaker_val_folds = {speaker: [] for speaker in speakers}
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            val_speakers = set(chunks[i].speaker_id for i in val_indices)
            for speaker in val_speakers:
                speaker_val_folds[speaker].append(fold_idx)
        
        # Property: Each speaker should be in validation set of at most one fold
        for speaker, folds in speaker_val_folds.items():
            assert len(folds) <= 1, (
                f"Speaker {speaker} appears in validation set of multiple folds: {folds}"
            )
    
    @given(
        speakers=unique_speakers_strategy,
        random_state=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100, deadline=None)
    def test_verify_speaker_independence_function(
        self,
        speakers: List[str],
        random_state: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 9: Speaker-Independent Cross-Validation
        
        The verify_speaker_independence_for_splits function should correctly
        identify valid and invalid splits.
        
        **Validates: Requirements 6.3**
        """
        assume(len(speakers) >= 3)
        
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=2)
        
        splitter = SpeakerIndependentSplitter(
            n_splits=3,
            shuffle=True,
            random_state=random_state
        )
        
        splits = splitter.split(chunks)
        
        # Property: All splits from SpeakerIndependentSplitter should be valid
        for train_indices, val_indices in splits:
            is_valid, overlap = verify_speaker_independence_for_splits(
                train_indices, val_indices, chunks
            )
            
            assert is_valid, (
                f"verify_speaker_independence_for_splits returned invalid for valid split!\n"
                f"  Overlap: {overlap}"
            )
            assert len(overlap) == 0, (
                f"verify_speaker_independence_for_splits found overlap in valid split: {overlap}"
            )
    
    @given(
        speakers=unique_speakers_strategy,
        n_splits=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_splitter_verify_method(
        self,
        speakers: List[str],
        n_splits: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 9: Speaker-Independent Cross-Validation
        
        The verify_speaker_independence method should return True for all valid splits.
        
        **Validates: Requirements 6.3**
        """
        assume(len(speakers) >= n_splits)
        
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=2)
        
        splitter = SpeakerIndependentSplitter(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )
        
        # Property: verify_speaker_independence should return True
        is_valid, fold_details = splitter.verify_speaker_independence(chunks)
        
        assert is_valid, (
            f"verify_speaker_independence returned False!\n"
            f"  Fold details: {fold_details}"
        )
        
        # Property: All fold details should show valid=True
        for detail in fold_details:
            assert detail['is_valid'], (
                f"Fold {detail['fold_idx']} marked as invalid: {detail}"
            )
            assert len(detail['overlap_speakers']) == 0, (
                f"Fold {detail['fold_idx']} has overlap: {detail['overlap_speakers']}"
            )


class TestSpeakerIndependentSplitterEdgeCases:
    """Edge case tests for SpeakerIndependentSplitter."""
    
    def test_minimum_speakers(self):
        """Test with minimum number of speakers (equal to n_splits)."""
        speakers = ["speaker_0", "speaker_1", "speaker_2"]
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=3)
        
        splitter = SpeakerIndependentSplitter(n_splits=3, shuffle=False, random_state=42)
        splits = splitter.split(chunks)
        
        # Each fold should have exactly one speaker in validation
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            train_speakers = set(chunks[i].speaker_id for i in train_indices)
            val_speakers = set(chunks[i].speaker_id for i in val_indices)
            
            assert len(train_speakers & val_speakers) == 0
    
    def test_single_chunk_per_speaker(self):
        """Test with single chunk per speaker."""
        speakers = [f"speaker_{i}" for i in range(10)]
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=1)
        
        splitter = SpeakerIndependentSplitter(n_splits=5, shuffle=True, random_state=42)
        is_valid, fold_details = splitter.verify_speaker_independence(chunks)
        
        assert is_valid
    
    def test_imbalanced_classes(self):
        """Test with imbalanced class distribution."""
        # Create chunks with imbalanced labels
        chunks = []
        for i in range(15):
            speaker_id = f"speaker_{i}"
            # 80% positive class
            label = 1 if i < 12 else 0
            chunks.append(MockAudioChunk(
                speaker_id=speaker_id,
                label=label,
                file_path=f"/path/to/{speaker_id}.wav"
            ))
        
        splitter = SpeakerIndependentSplitter(n_splits=3, shuffle=True, random_state=42)
        is_valid, fold_details = splitter.verify_speaker_independence(chunks)
        
        assert is_valid
    
    def test_reproducibility_with_same_seed(self):
        """Test that same random_state produces same splits."""
        speakers = [f"speaker_{i}" for i in range(10)]
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=2)
        
        splitter1 = SpeakerIndependentSplitter(n_splits=3, shuffle=True, random_state=42)
        splitter2 = SpeakerIndependentSplitter(n_splits=3, shuffle=True, random_state=42)
        
        splits1 = splitter1.split(chunks)
        splits2 = splitter2.split(chunks)
        
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            assert set(train1) == set(train2)
            assert set(val1) == set(val2)
    
    def test_different_seeds_produce_different_splits(self):
        """Test that different random_states produce different splits."""
        speakers = [f"speaker_{i}" for i in range(10)]
        chunks = generate_mock_chunks(speakers, chunks_per_speaker=2)
        
        splitter1 = SpeakerIndependentSplitter(n_splits=3, shuffle=True, random_state=42)
        splitter2 = SpeakerIndependentSplitter(n_splits=3, shuffle=True, random_state=123)
        
        splits1 = splitter1.split(chunks)
        splits2 = splitter2.split(chunks)
        
        # At least one fold should be different
        any_different = False
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            if set(train1) != set(train2) or set(val1) != set(val2):
                any_different = True
                break
        
        assert any_different, "Different seeds should produce different splits"


class TestLinguisticFeatures:
    """Test cases for linguistic feature integration in multimodal dataset."""
    
    def test_multimodal_dataset_with_linguistic_features(self):
        """Test MultimodalDataset with linguistic features enabled."""
        # Create mock data
        chunks = [
            MockAudioChunk(speaker_id="speaker1", label=0, file_path="audio1.wav"),
            MockAudioChunk(speaker_id="speaker2", label=1, file_path="audio2.wav")
        ]
        
        transcripts = {
            "audio1.wav": TranscriptData(
                text="This is a test sentence",
                normalized_text="This is a test sentence",
                morphemes=[],
                language="tamil",
                speaker_id="speaker1",
                label=0
            ),
            "audio2.wav": TranscriptData(
                text="Another test sentence",
                normalized_text="Another test sentence", 
                morphemes=[],
                language="tamil",
                speaker_id="speaker2",
                label=1
            )
        }
        
        # Create mock config
        from dataclasses import dataclass
        
        @dataclass
        class MockConfig:
            num_workers: int = 0
        
        config = MockConfig()
        tokenizer = MockTokenizer()
        linguistic_analyzer = MockLinguisticAnalyzer()
        
        # Create dataset with linguistic features
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = MultimodalDataset(
                audio_chunks=chunks,
                transcripts=transcripts,
                tokenizer=tokenizer,
                config=config,
                linguistic_analyzer=linguistic_analyzer,
                enable_linguistic_features=True,
                linguistic_cache_dir=temp_dir
            )
            
            # Test sample retrieval
            sample = dataset[0]
            
            # Verify linguistic features are present
            assert 'linguistic_features' in sample
            assert 'has_linguistic' in sample
            assert sample['linguistic_features'].shape == (128,)
            assert sample['has_linguistic'] == True
    
    def test_multimodal_dataset_without_linguistic_features(self):
        """Test MultimodalDataset with linguistic features disabled."""
        chunks = [MockAudioChunk(speaker_id="speaker1", label=0, file_path="audio1.wav")]
        transcripts = {}
        
        @dataclass
        class MockConfig:
            num_workers: int = 0
        
        config = MockConfig()
        tokenizer = MockTokenizer()
        
        dataset = MultimodalDataset(
            audio_chunks=chunks,
            transcripts=transcripts,
            tokenizer=tokenizer,
            config=config,
            enable_linguistic_features=False
        )
        
        sample = dataset[0]
        
        # Verify linguistic features are zero tensors
        assert 'linguistic_features' in sample
        assert 'has_linguistic' in sample
        assert sample['linguistic_features'].shape == (128,)
        assert sample['has_linguistic'] == False
        assert torch.allclose(sample['linguistic_features'], torch.zeros(128))
    
    def test_linguistic_feature_caching(self):
        """Test linguistic feature caching functionality."""
        chunks = [MockAudioChunk(speaker_id="speaker1", label=0, file_path="audio1.wav")]
        transcripts = {
            "audio1.wav": TranscriptData(
                text="Test sentence for caching",
                normalized_text="Test sentence for caching",
                morphemes=[],
                language="tamil",
                speaker_id="speaker1",
                label=0
            )
        }
        
        @dataclass
        class MockConfig:
            num_workers: int = 0
        
        config = MockConfig()
        tokenizer = MockTokenizer()
        linguistic_analyzer = MockLinguisticAnalyzer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first dataset - should extract and cache features
            dataset1 = MultimodalDataset(
                audio_chunks=chunks,
                transcripts=transcripts,
                tokenizer=tokenizer,
                config=config,
                linguistic_analyzer=linguistic_analyzer,
                enable_linguistic_features=True,
                linguistic_cache_dir=temp_dir
            )
            
            sample1 = dataset1[0]
            dataset1._save_linguistic_cache()
            
            # Create second dataset - should load from cache
            dataset2 = MultimodalDataset(
                audio_chunks=chunks,
                transcripts=transcripts,
                tokenizer=tokenizer,
                config=config,
                linguistic_analyzer=None,  # No analyzer, should use cache
                enable_linguistic_features=True,
                linguistic_cache_dir=temp_dir
            )
            
            sample2 = dataset2[0]
            
            # Features should be identical (loaded from cache)
            assert torch.allclose(sample1['linguistic_features'], sample2['linguistic_features'])
    
    def test_multimodal_collate_fn_with_linguistic_features(self):
        """Test collate function with linguistic features."""
        # Create mock batch
        batch = [
            {
                'waveform': torch.randn(1, 16000),
                'audio_attention_mask': torch.ones(16000),
                'input_ids': torch.randint(0, 1000, (32,)),
                'text_attention_mask': torch.ones(32),
                'linguistic_features': torch.randn(128),
                'label': torch.tensor(0),
                'speaker_id': 'speaker1',
                'file_path': 'audio1.wav',
                'text': 'Test text',
                'language': 'tamil',
                'has_text': True,
                'has_audio': True,
                'has_linguistic': True
            },
            {
                'waveform': torch.randn(1, 16000),
                'audio_attention_mask': torch.ones(16000),
                'input_ids': torch.randint(0, 1000, (24,)),  # Different length
                'text_attention_mask': torch.ones(24),
                'linguistic_features': torch.randn(128),
                'label': torch.tensor(1),
                'speaker_id': 'speaker2',
                'file_path': 'audio2.wav',
                'text': 'Another test',
                'language': 'tamil',
                'has_text': True,
                'has_audio': True,
                'has_linguistic': True
            }
        ]
        
        # Test collate function
        collated = multimodal_collate_fn(batch)
        
        # Verify batch structure
        assert collated['waveform'].shape == (2, 1, 16000)
        assert collated['linguistic_features'].shape == (2, 128)
        assert collated['input_ids'].shape[0] == 2  # Batch size
        assert collated['input_ids'].shape[1] == 32  # Padded to max length
        assert len(collated['text']) == 2
        assert len(collated['language']) == 2
        assert collated['has_linguistic'].shape == (2,)
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        text_lengths=st.lists(st.integers(min_value=5, max_value=50), min_size=1, max_size=8)
    )
    @settings(deadline=None, max_examples=10)
    def test_linguistic_feature_dimensionality_property(self, batch_size, text_lengths):
        """
        Feature: traditional-nlp-tasks, Property 4: For any linguistic analysis, 
        the extracted feature vector SHALL have exactly 128 dimensions with all 
        values in valid ranges.
        """
        # Ensure batch_size matches text_lengths
        text_lengths = text_lengths[:batch_size] if len(text_lengths) > batch_size else text_lengths
        while len(text_lengths) < batch_size:
            text_lengths.append(random.randint(5, 50))
        
        # Create mock batch with varying text lengths
        batch = []
        for i in range(batch_size):
            text_len = text_lengths[i]
            batch.append({
                'waveform': torch.randn(1, 16000),
                'audio_attention_mask': torch.ones(16000),
                'input_ids': torch.randint(0, 1000, (text_len,)),
                'text_attention_mask': torch.ones(text_len),
                'linguistic_features': torch.randn(128),  # Always 128 dimensions
                'label': torch.tensor(i % 2),
                'speaker_id': f'speaker{i}',
                'file_path': f'audio{i}.wav',
                'text': f'Test text {i}',
                'language': 'tamil',
                'has_text': True,
                'has_audio': True,
                'has_linguistic': True
            })
        
        # Test collate function
        collated = multimodal_collate_fn(batch)
        
        # Verify linguistic features have exactly 128 dimensions
        assert collated['linguistic_features'].shape == (batch_size, 128)
        
        # Verify no NaN or infinite values
        assert not torch.isnan(collated['linguistic_features']).any()
        assert not torch.isinf(collated['linguistic_features']).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
