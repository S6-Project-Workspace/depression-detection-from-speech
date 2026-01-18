"""
Tests for Enhanced Multimodal Model with Linguistic Features

This module contains comprehensive tests for the enhanced multimodal model
that integrates traditional NLP features with audio and text embeddings.

Reference: Traditional NLP Tasks - Task 5
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
from hypothesis import given, strategies as st, settings
import numpy as np

from enhanced_multimodal_model import (
    EnhancedMultimodalModel,
    EnhancedFusionConfig,
    LinguisticFeatureExtractor,
    AttentionFusion,
    create_enhanced_multimodal_model
)


class MockAudioModel(nn.Module):
    """Mock audio model for testing."""
    
    def __init__(self, embedding_dim: int = 192):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(80, embedding_dim)
    
    def get_embeddings(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Simple pooling for testing
        return self.fc(x.mean(dim=-1))


class MockTextModel(nn.Module):
    """Mock text model for testing."""
    
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(512, embedding_dim)
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Simple embedding for testing
        return self.fc(torch.randn(input_ids.size(0), 512))


class MockLinguisticAnalyzer:
    """Mock linguistic analyzer for testing."""
    
    def analyze_text(self, text: str, language: str):
        """Mock analysis that returns None."""
        return None
    
    def extract_linguistic_features(self, analysis) -> np.ndarray:
        """Mock feature extraction that returns random 128-d vector."""
        return np.random.randn(128).astype(np.float32)


@pytest.fixture
def mock_audio_model():
    """Create mock audio model."""
    return MockAudioModel(embedding_dim=192)


@pytest.fixture
def mock_text_model():
    """Create mock text model."""
    return MockTextModel(embedding_dim=768)


@pytest.fixture
def mock_linguistic_analyzer():
    """Create mock linguistic analyzer."""
    return MockLinguisticAnalyzer()


@pytest.fixture
def enhanced_config():
    """Create enhanced fusion configuration."""
    return EnhancedFusionConfig(
        audio_embedding_dim=192,
        text_embedding_dim=768,
        linguistic_feature_dim=128,
        fusion_hidden_dim=512,
        fusion_dropout=0.3,
        num_classes=2,
        fusion_method="concatenation",
        allow_unimodal_fallback=True,
        allow_bimodal_fallback=True
    )


@pytest.fixture
def enhanced_model(mock_audio_model, mock_text_model, mock_linguistic_analyzer, enhanced_config):
    """Create enhanced multimodal model."""
    return EnhancedMultimodalModel(
        audio_model=mock_audio_model,
        text_model=mock_text_model,
        linguistic_analyzer=mock_linguistic_analyzer,
        config=enhanced_config
    )


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing."""
    batch_size = 4
    return {
        'audio_input': torch.randn(batch_size, 80, 500),
        'text_input_ids': torch.randint(0, 1000, (batch_size, 128)),
        'text_attention_mask': torch.ones(batch_size, 128),
        'text_raw': ["Sample text for analysis"] * batch_size,
        'language': "tamil"
    }


class TestLinguisticFeatureExtractor:
    """Test cases for LinguisticFeatureExtractor."""
    
    def test_initialization(self):
        """Test LinguisticFeatureExtractor initialization."""
        extractor = LinguisticFeatureExtractor(
            input_dim=128,
            output_dim=128,
            dropout=0.1
        )
        
        assert extractor.input_dim == 128
        assert extractor.output_dim == 128
        assert isinstance(extractor.layer_norm, nn.LayerNorm)
        assert isinstance(extractor.projection, nn.Sequential)
        assert extractor.feature_weights.shape == (128,)
    
    def test_forward_pass(self):
        """Test forward pass through linguistic feature extractor."""
        extractor = LinguisticFeatureExtractor(input_dim=128, output_dim=64)
        
        batch_size = 4
        input_features = torch.randn(batch_size, 128)
        
        output = extractor(input_features)
        
        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        extractor = LinguisticFeatureExtractor(input_dim=128)
        
        importance = extractor.get_feature_importance()
        
        assert importance.shape == (128,)
        assert torch.allclose(importance.sum(), torch.tensor(1.0), atol=1e-6)
        assert (importance >= 0).all()
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        input_dim=st.integers(min_value=32, max_value=256),
        output_dim=st.integers(min_value=32, max_value=256)
    )
    @settings(deadline=None, max_examples=10)
    def test_extractor_dimensions_property(self, batch_size, input_dim, output_dim):
        """Property test: Output dimensions should match configuration."""
        extractor = LinguisticFeatureExtractor(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        input_features = torch.randn(batch_size, input_dim)
        output = extractor(input_features)
        
        assert output.shape == (batch_size, output_dim)


class TestAttentionFusion:
    """Test cases for AttentionFusion."""
    
    def test_initialization(self):
        """Test AttentionFusion initialization."""
        fusion = AttentionFusion(
            audio_dim=192,
            text_dim=768,
            linguistic_dim=128,
            hidden_dim=256
        )
        
        assert fusion.audio_dim == 192
        assert fusion.text_dim == 768
        assert fusion.linguistic_dim == 128
        assert isinstance(fusion.attention, nn.MultiheadAttention)
    
    def test_forward_pass(self):
        """Test forward pass through attention fusion."""
        fusion = AttentionFusion(
            audio_dim=192,
            text_dim=768,
            linguistic_dim=128,
            hidden_dim=256
        )
        
        batch_size = 4
        audio_emb = torch.randn(batch_size, 192)
        text_emb = torch.randn(batch_size, 768)
        ling_emb = torch.randn(batch_size, 128)
        
        output = fusion(audio_emb, text_emb, ling_emb)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()


class TestEnhancedMultimodalModel:
    """Test cases for EnhancedMultimodalModel."""
    
    def test_initialization(self, enhanced_model, enhanced_config):
        """Test enhanced multimodal model initialization."""
        assert enhanced_model.config == enhanced_config
        assert enhanced_model.audio_embedding_dim == 192
        assert enhanced_model.text_embedding_dim == 768
        assert enhanced_model.linguistic_feature_dim == 128
        assert hasattr(enhanced_model, 'linguistic_extractor')
        assert hasattr(enhanced_model, 'fusion_mlp')
    
    def test_embedding_dimensions(self, enhanced_model):
        """Test embedding dimension getters."""
        dims = enhanced_model.get_embedding_dimensions()
        
        assert dims['audio'] == 192
        assert dims['text'] == 768
        assert dims['linguistic'] == 128
        assert dims['fused'] == 192 + 768 + 128
    
    def test_trimodal_forward_pass(self, enhanced_model, sample_inputs):
        """Test forward pass with all three modalities."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits = enhanced_model(
                sample_inputs['audio_input'],
                sample_inputs['text_input_ids'],
                sample_inputs['text_attention_mask'],
                text_raw=sample_inputs['text_raw'],
                language=sample_inputs['language']
            )
        
        batch_size = sample_inputs['audio_input'].size(0)
        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
    
    def test_trimodal_forward_with_embeddings(self, enhanced_model, sample_inputs):
        """Test forward pass returning embeddings."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits, embeddings = enhanced_model(
                sample_inputs['audio_input'],
                sample_inputs['text_input_ids'],
                sample_inputs['text_attention_mask'],
                text_raw=sample_inputs['text_raw'],
                language=sample_inputs['language'],
                return_embeddings=True
            )
        
        audio_emb, text_emb, ling_emb, fused_emb = embeddings
        batch_size = sample_inputs['audio_input'].size(0)
        
        assert logits.shape == (batch_size, 2)
        assert audio_emb.shape == (batch_size, 192)
        assert text_emb.shape == (batch_size, 768)
        assert ling_emb.shape == (batch_size, 128)
        assert fused_emb.shape == (batch_size, 192 + 768 + 128)
    
    def test_trimodal_forward_with_feature_importance(self, enhanced_model, sample_inputs):
        """Test forward pass returning feature importance."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits, embeddings, feature_importance = enhanced_model(
                sample_inputs['audio_input'],
                sample_inputs['text_input_ids'],
                sample_inputs['text_attention_mask'],
                text_raw=sample_inputs['text_raw'],
                language=sample_inputs['language'],
                return_embeddings=True,
                return_feature_importance=True
            )
        
        batch_size = sample_inputs['audio_input'].size(0)
        
        assert logits.shape == (batch_size, 2)
        assert feature_importance.shape == (128,)
        assert torch.allclose(feature_importance.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_bimodal_audio_text_fallback(self, enhanced_model, sample_inputs):
        """Test bimodal fallback with audio and text."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits = enhanced_model.forward_audio_text(
                sample_inputs['audio_input'],
                sample_inputs['text_input_ids'],
                sample_inputs['text_attention_mask']
            )
        
        batch_size = sample_inputs['audio_input'].size(0)
        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
    
    def test_bimodal_audio_linguistic_fallback(self, enhanced_model, sample_inputs):
        """Test bimodal fallback with audio and linguistic features."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits = enhanced_model.forward_audio_linguistic(
                sample_inputs['audio_input'],
                sample_inputs['text_raw'],
                language=sample_inputs['language']
            )
        
        batch_size = sample_inputs['audio_input'].size(0)
        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
    
    def test_bimodal_text_linguistic_fallback(self, enhanced_model, sample_inputs):
        """Test bimodal fallback with text and linguistic features."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits = enhanced_model.forward_text_linguistic(
                sample_inputs['text_input_ids'],
                sample_inputs['text_attention_mask'],
                sample_inputs['text_raw'],
                language=sample_inputs['language']
            )
        
        batch_size = sample_inputs['audio_input'].size(0)
        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
    
    def test_unimodal_audio_fallback(self, enhanced_model, sample_inputs):
        """Test unimodal fallback with audio only."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits = enhanced_model.forward_audio_only(
                sample_inputs['audio_input']
            )
        
        batch_size = sample_inputs['audio_input'].size(0)
        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
    
    def test_unimodal_text_fallback(self, enhanced_model, sample_inputs):
        """Test unimodal fallback with text only."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits = enhanced_model.forward_text_only(
                sample_inputs['text_input_ids'],
                sample_inputs['text_attention_mask']
            )
        
        batch_size = sample_inputs['audio_input'].size(0)
        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
    
    def test_unimodal_linguistic_fallback(self, enhanced_model, sample_inputs):
        """Test unimodal fallback with linguistic features only."""
        enhanced_model.eval()
        
        with torch.no_grad():
            logits = enhanced_model.forward_linguistic_only(
                sample_inputs['text_raw'],
                language=sample_inputs['language']
            )
        
        batch_size = len(sample_inputs['text_raw'])
        assert logits.shape == (batch_size, 2)
        assert not torch.isnan(logits).any()
    
    def test_attention_fusion_config(self, mock_audio_model, mock_text_model, mock_linguistic_analyzer):
        """Test model with attention fusion."""
        config = EnhancedFusionConfig(
            audio_embedding_dim=192,
            text_embedding_dim=768,
            linguistic_feature_dim=128,
            fusion_method="attention",
            fusion_hidden_dim=256
        )
        
        model = EnhancedMultimodalModel(
            audio_model=mock_audio_model,
            text_model=mock_text_model,
            linguistic_analyzer=mock_linguistic_analyzer,
            config=config
        )
        
        assert hasattr(model, 'attention_fusion')
        assert isinstance(model.attention_fusion, AttentionFusion)
    
    def test_fallback_disabled_errors(self, mock_audio_model, mock_text_model, mock_linguistic_analyzer, sample_inputs):
        """Test that disabled fallbacks raise appropriate errors."""
        config = EnhancedFusionConfig(
            audio_embedding_dim=192,
            text_embedding_dim=768,
            linguistic_feature_dim=128,
            allow_unimodal_fallback=False,
            allow_bimodal_fallback=False
        )
        
        model = EnhancedMultimodalModel(
            audio_model=mock_audio_model,
            text_model=mock_text_model,
            linguistic_analyzer=mock_linguistic_analyzer,
            config=config
        )
        
        # Test that bimodal fallback raises error
        with pytest.raises(ValueError, match="Bimodal fallback not enabled"):
            model.forward_audio_text(
                sample_inputs['audio_input'],
                sample_inputs['text_input_ids'],
                sample_inputs['text_attention_mask']
            )
        
        # Test that unimodal fallback raises error
        with pytest.raises(ValueError, match="Unimodal fallback not enabled"):
            model.forward_audio_only(sample_inputs['audio_input'])


class TestPropertyBasedTests:
    """Property-based tests for enhanced multimodal model."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        audio_seq_len=st.integers(min_value=100, max_value=1000),
        text_seq_len=st.integers(min_value=32, max_value=256)
    )
    @settings(deadline=None, max_examples=10)
    def test_integration_compatibility_property(
        self, 
        batch_size, 
        audio_seq_len, 
        text_seq_len
    ):
        """
        Feature: traditional-nlp-tasks, Property 6: For any enhanced multimodal model 
        forward pass, the linguistic features SHALL integrate seamlessly with existing 
        audio and text embeddings without dimension mismatches.
        """
        # Create mock models inside the test
        mock_audio_model = MockAudioModel(embedding_dim=192)
        mock_text_model = MockTextModel(embedding_dim=768)
        mock_linguistic_analyzer = MockLinguisticAnalyzer()
        
        config = EnhancedFusionConfig(
            audio_embedding_dim=192,
            text_embedding_dim=768,
            linguistic_feature_dim=128,
            fusion_hidden_dim=512
        )
        
        model = EnhancedMultimodalModel(
            audio_model=mock_audio_model,
            text_model=mock_text_model,
            linguistic_analyzer=mock_linguistic_analyzer,
            config=config
        )
        
        # Create inputs
        audio_input = torch.randn(batch_size, 80, audio_seq_len)
        text_input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        text_attention_mask = torch.ones(batch_size, text_seq_len)
        text_raw = ["Sample text"] * batch_size
        
        model.eval()
        with torch.no_grad():
            # Test that forward pass works without dimension errors
            logits, embeddings = model(
                audio_input,
                text_input_ids,
                text_attention_mask,
                text_raw=text_raw,
                language="tamil",
                return_embeddings=True
            )
            
            audio_emb, text_emb, ling_emb, fused_emb = embeddings
            
            # Verify no dimension mismatches
            assert logits.shape == (batch_size, 2)
            assert audio_emb.shape == (batch_size, 192)
            assert text_emb.shape == (batch_size, 768)
            assert ling_emb.shape == (batch_size, 128)
            assert fused_emb.shape == (batch_size, 192 + 768 + 128)
            
            # Verify no NaN values (indicates successful integration)
            assert not torch.isnan(logits).any()
            assert not torch.isnan(fused_emb).any()
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8)
    )
    @settings(deadline=None, max_examples=10)
    def test_linguistic_feature_dimensionality_property(
        self,
        batch_size
    ):
        """
        Feature: traditional-nlp-tasks, Property 4: For any linguistic analysis, 
        the extracted feature vector SHALL have exactly 128 dimensions with all 
        values in valid ranges.
        """
        mock_linguistic_analyzer = MockLinguisticAnalyzer()
        
        extractor = LinguisticFeatureExtractor(
            input_dim=128,
            output_dim=128
        )
        
        # Simulate linguistic features from analyzer
        linguistic_features = []
        for _ in range(batch_size):
            features = mock_linguistic_analyzer.extract_linguistic_features(None)
            linguistic_features.append(torch.tensor(features, dtype=torch.float32))
        
        linguistic_batch = torch.stack(linguistic_features)
        
        # Process through extractor
        processed_features = extractor(linguistic_batch)
        
        # Verify exactly 128 dimensions
        assert processed_features.shape == (batch_size, 128)
        
        # Verify values are in valid ranges (no NaN, no extreme values)
        assert not torch.isnan(processed_features).any()
        assert not torch.isinf(processed_features).any()
        assert processed_features.abs().max() < 1000  # Reasonable range


class TestFactoryFunctions:
    """Test factory functions for model creation."""
    
    def test_create_enhanced_multimodal_model(self):
        """Test factory function for creating enhanced model."""
        model = create_enhanced_multimodal_model()
        
        assert isinstance(model, EnhancedMultimodalModel)
        assert model.config is not None
        assert model.linguistic_analyzer is not None
    
    def test_create_enhanced_multimodal_model_with_params(
        self, 
        mock_audio_model, 
        mock_text_model, 
        mock_linguistic_analyzer,
        enhanced_config
    ):
        """Test factory function with custom parameters."""
        model = create_enhanced_multimodal_model(
            audio_model=mock_audio_model,
            text_model=mock_text_model,
            linguistic_analyzer=mock_linguistic_analyzer,
            config=enhanced_config
        )
        
        assert isinstance(model, EnhancedMultimodalModel)
        assert model.audio_model == mock_audio_model
        assert model.text_model == mock_text_model
        assert model.linguistic_analyzer == mock_linguistic_analyzer
        assert model.config == enhanced_config


if __name__ == "__main__":
    pytest.main([__file__])