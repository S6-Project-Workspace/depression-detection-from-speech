"""
Property-Based Tests for Multimodal Fusion Model

Tests the multimodal fusion model using hypothesis for property-based testing.
Validates Property 8: Fusion Embedding Concatenation (Requirements 5.1, 5.2).

Feature: multimodal-nlp-upgrade
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List, Dict, Any, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

# Skip all tests if torch is not available
if not HAS_TORCH:
    pytest.skip("torch not available", allow_module_level=True)

from config import FusionConfig
from multimodal_model import (
    MultimodalDepressionModel,
    FusionMLP,
    create_multimodal_model
)


# Strategy for generating embedding dimensions
audio_dim_strategy = st.sampled_from([192, 256, 512, 768])
text_dim_strategy = st.sampled_from([256, 512, 768, 1024])
batch_size_strategy = st.integers(min_value=1, max_value=8)


class MockAudioModel(nn.Module):
    """Mock audio model for testing multimodal fusion.
    
    Simulates ECAPA-TDNN or Wav2Vec2 embedding extraction.
    """
    
    def __init__(self, embedding_dim: int = 192):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._seed = 42
    
    def get_embeddings(
        self,
        audio_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return audio embeddings with correct shape."""
        batch_size = audio_input.shape[0]
        # Generate deterministic embeddings
        torch.manual_seed(self._seed)
        return torch.randn(batch_size, self.embedding_dim)
    
    def forward(
        self,
        audio_input: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Mock forward pass."""
        embeddings = self.get_embeddings(audio_input)
        logits = torch.zeros(embeddings.size(0), 2)
        if return_embeddings:
            return logits, embeddings
        return logits


class MockTextModel(nn.Module):
    """Mock text model for testing multimodal fusion.
    
    Simulates MuRIL embedding extraction.
    """
    
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._seed = 123
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return text embeddings with correct shape."""
        batch_size = input_ids.shape[0]
        # Generate deterministic embeddings
        torch.manual_seed(self._seed)
        return torch.randn(batch_size, self.embedding_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Mock forward pass."""
        embeddings = self.get_embeddings(input_ids, attention_mask)
        logits = torch.zeros(embeddings.size(0), 2)
        if return_embeddings:
            return logits, embeddings
        return logits


@pytest.fixture
def default_config():
    """Create default FusionConfig for testing."""
    return FusionConfig(
        audio_embedding_dim=192,
        text_embedding_dim=768,
        fusion_hidden_dim=512,
        fusion_dropout=0.3,
        num_classes=2,
        allow_unimodal_fallback=True
    )


@pytest.fixture
def mock_audio_model():
    """Create mock audio model."""
    return MockAudioModel(embedding_dim=192)


@pytest.fixture
def mock_text_model():
    """Create mock text model."""
    return MockTextModel(embedding_dim=768)


@pytest.fixture
def multimodal_model(mock_audio_model, mock_text_model, default_config):
    """Create multimodal model with mock components."""
    model = MultimodalDepressionModel(
        audio_model=mock_audio_model,
        text_model=mock_text_model,
        config=default_config
    )
    model.eval()
    return model


class TestFusionEmbeddingConcatenation:
    """
    Property 8: Fusion Embedding Concatenation
    
    *For any* audio embedding of dimension D_a and text embedding of dimension D_t,
    the fused representation SHALL have dimension D_a + D_t before the fusion MLP.
    
    **Validates: Requirements 5.1, 5.2**
    """
    
    @given(
        audio_dim=audio_dim_strategy,
        text_dim=text_dim_strategy,
        batch_size=batch_size_strategy
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_fused_embedding_dimension_equals_sum(
        self,
        audio_dim: int,
        text_dim: int,
        batch_size: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 8: Fusion Embedding Concatenation
        
        For any audio embedding of dimension D_a and text embedding of dimension D_t,
        the fused representation SHALL have dimension D_a + D_t.
        
        **Validates: Requirements 5.1, 5.2**
        """
        # Create config with specified dimensions
        config = FusionConfig(
            audio_embedding_dim=audio_dim,
            text_embedding_dim=text_dim,
            fusion_hidden_dim=256,
            fusion_dropout=0.3,
            num_classes=2,
            allow_unimodal_fallback=True
        )
        
        # Create mock models with matching dimensions
        audio_model = MockAudioModel(embedding_dim=audio_dim)
        text_model = MockTextModel(embedding_dim=text_dim)
        
        # Create multimodal model
        model = MultimodalDepressionModel(
            audio_model=audio_model,
            text_model=text_model,
            config=config
        )
        model.eval()
        
        # Create dummy inputs
        audio_input = torch.randn(batch_size, 80, 500)  # Mel spectrogram
        text_input_ids = torch.randint(0, 1000, (batch_size, 128))
        text_attention_mask = torch.ones(batch_size, 128)
        
        # Get fused embeddings
        with torch.no_grad():
            _, fused_embeddings = model(
                audio_input,
                text_input_ids,
                text_attention_mask,
                return_embeddings=True
            )
        
        # Property: Fused dimension should equal audio_dim + text_dim
        expected_dim = audio_dim + text_dim
        actual_dim = fused_embeddings.shape[1]
        
        assert actual_dim == expected_dim, (
            f"Fused embedding dimension {actual_dim} doesn't match "
            f"expected {expected_dim} (audio={audio_dim} + text={text_dim})"
        )
        
        # Also verify batch dimension
        assert fused_embeddings.shape[0] == batch_size, (
            f"Batch dimension {fused_embeddings.shape[0]} doesn't match "
            f"expected {batch_size}"
        )
    
    @given(
        audio_dim=audio_dim_strategy,
        text_dim=text_dim_strategy
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_get_fused_embedding_dim_returns_correct_value(
        self,
        audio_dim: int,
        text_dim: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 8: Fusion Embedding Concatenation
        
        get_fused_embedding_dim() SHALL return audio_dim + text_dim.
        
        **Validates: Requirements 5.1, 5.2**
        """
        config = FusionConfig(
            audio_embedding_dim=audio_dim,
            text_embedding_dim=text_dim,
            fusion_hidden_dim=256,
            fusion_dropout=0.3,
            num_classes=2
        )
        
        audio_model = MockAudioModel(embedding_dim=audio_dim)
        text_model = MockTextModel(embedding_dim=text_dim)
        
        model = MultimodalDepressionModel(
            audio_model=audio_model,
            text_model=text_model,
            config=config
        )
        
        expected_dim = audio_dim + text_dim
        actual_dim = model.get_fused_embedding_dim()
        
        assert actual_dim == expected_dim, (
            f"get_fused_embedding_dim() returned {actual_dim}, "
            f"expected {expected_dim}"
        )
    
    @given(batch_size=batch_size_strategy)
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_concatenation_preserves_batch_dimension(
        self,
        batch_size: int,
        multimodal_model
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 8: Fusion Embedding Concatenation
        
        Concatenation SHALL preserve the batch dimension.
        
        **Validates: Requirements 5.1, 5.2**
        """
        # Create dummy inputs
        audio_input = torch.randn(batch_size, 80, 500)
        text_input_ids = torch.randint(0, 1000, (batch_size, 128))
        text_attention_mask = torch.ones(batch_size, 128)
        
        with torch.no_grad():
            _, fused_embeddings = multimodal_model(
                audio_input,
                text_input_ids,
                text_attention_mask,
                return_embeddings=True
            )
        
        assert fused_embeddings.shape[0] == batch_size, (
            f"Batch dimension {fused_embeddings.shape[0]} doesn't match "
            f"expected {batch_size}"
        )


    @given(
        audio_dim=audio_dim_strategy,
        text_dim=text_dim_strategy
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_precomputed_embeddings_concatenation(
        self,
        audio_dim: int,
        text_dim: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 8: Fusion Embedding Concatenation
        
        forward_with_precomputed_embeddings SHALL concatenate embeddings correctly.
        
        **Validates: Requirements 5.1, 5.2**
        """
        config = FusionConfig(
            audio_embedding_dim=audio_dim,
            text_embedding_dim=text_dim,
            fusion_hidden_dim=256,
            fusion_dropout=0.3,
            num_classes=2
        )
        
        audio_model = MockAudioModel(embedding_dim=audio_dim)
        text_model = MockTextModel(embedding_dim=text_dim)
        
        model = MultimodalDepressionModel(
            audio_model=audio_model,
            text_model=text_model,
            config=config
        )
        model.eval()
        
        batch_size = 4
        
        # Create pre-computed embeddings
        audio_embeddings = torch.randn(batch_size, audio_dim)
        text_embeddings = torch.randn(batch_size, text_dim)
        
        with torch.no_grad():
            _, fused_embeddings = model.forward_with_precomputed_embeddings(
                audio_embeddings,
                text_embeddings,
                return_embeddings=True
            )
        
        expected_dim = audio_dim + text_dim
        assert fused_embeddings.shape == (batch_size, expected_dim), (
            f"Fused shape {fused_embeddings.shape} doesn't match "
            f"expected ({batch_size}, {expected_dim})"
        )


class TestFusionMLPWithDropout:
    """Tests for Fusion MLP with dropout (Requirement 5.4)."""
    
    def test_fusion_mlp_has_dropout_layers(self):
        """Fusion MLP SHALL have dropout layers to prevent modality dominance."""
        mlp = FusionMLP(
            input_dim=960,
            hidden_dim=512,
            num_classes=2,
            dropout=0.3
        )
        
        # Check that dropout layers exist
        dropout_layers = [m for m in mlp.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) >= 1, "FusionMLP should have at least one dropout layer"
    
    @given(dropout_rate=st.floats(min_value=0.1, max_value=0.5))
    @settings(max_examples=10, deadline=None)
    def test_fusion_mlp_respects_dropout_rate(self, dropout_rate: float):
        """Fusion MLP SHALL use the specified dropout rate."""
        mlp = FusionMLP(
            input_dim=960,
            hidden_dim=512,
            num_classes=2,
            dropout=dropout_rate
        )
        
        # Find dropout layers and verify rate
        for module in mlp.modules():
            if isinstance(module, nn.Dropout):
                assert abs(module.p - dropout_rate) < 1e-6, (
                    f"Dropout rate {module.p} doesn't match expected {dropout_rate}"
                )


class TestUnimodalFallback:
    """Tests for unimodal fallback functionality (Requirement 5.6)."""
    
    def test_audio_only_fallback_works(self, multimodal_model):
        """forward_audio_only SHALL work when text is unavailable."""
        batch_size = 4
        audio_input = torch.randn(batch_size, 80, 500)
        
        with torch.no_grad():
            logits, audio_emb = multimodal_model.forward_audio_only(
                audio_input, return_embeddings=True
            )
        
        assert logits.shape == (batch_size, 2)
        assert audio_emb.shape == (batch_size, 192)  # Default audio dim
    
    def test_text_only_fallback_works(self, multimodal_model):
        """forward_text_only SHALL work when audio is unavailable."""
        batch_size = 4
        text_input_ids = torch.randint(0, 1000, (batch_size, 128))
        text_attention_mask = torch.ones(batch_size, 128)
        
        with torch.no_grad():
            logits, text_emb = multimodal_model.forward_text_only(
                text_input_ids, text_attention_mask, return_embeddings=True
            )
        
        assert logits.shape == (batch_size, 2)
        assert text_emb.shape == (batch_size, 768)  # Default text dim
    
    def test_fallback_disabled_raises_error(self):
        """Fallback methods SHALL raise error when disabled."""
        config = FusionConfig(
            audio_embedding_dim=192,
            text_embedding_dim=768,
            allow_unimodal_fallback=False
        )
        
        audio_model = MockAudioModel(embedding_dim=192)
        text_model = MockTextModel(embedding_dim=768)
        
        model = MultimodalDepressionModel(
            audio_model=audio_model,
            text_model=text_model,
            config=config
        )
        
        audio_input = torch.randn(2, 80, 500)
        
        with pytest.raises(ValueError, match="fallback not enabled"):
            model.forward_audio_only(audio_input)


class TestMultimodalModelCreation:
    """Tests for multimodal model creation and configuration."""
    
    def test_create_multimodal_model_factory(self, mock_audio_model, mock_text_model, default_config):
        """create_multimodal_model factory SHALL create valid model."""
        model = create_multimodal_model(
            audio_model=mock_audio_model,
            text_model=mock_text_model,
            config=default_config
        )
        
        assert model is not None
        assert isinstance(model, MultimodalDepressionModel)
    
    def test_model_stores_correct_dimensions(self, multimodal_model, default_config):
        """Model SHALL store correct embedding dimensions."""
        assert multimodal_model.get_audio_embedding_dim() == default_config.audio_embedding_dim
        assert multimodal_model.get_text_embedding_dim() == default_config.text_embedding_dim
        assert multimodal_model.get_fused_embedding_dim() == (
            default_config.audio_embedding_dim + default_config.text_embedding_dim
        )
    
    def test_model_has_fusion_mlp(self, multimodal_model):
        """Model SHALL have fusion MLP component."""
        assert hasattr(multimodal_model, 'fusion_mlp')
        assert isinstance(multimodal_model.fusion_mlp, FusionMLP)
    
    def test_model_has_fallback_classifiers_when_enabled(self, multimodal_model):
        """Model SHALL have fallback classifiers when enabled."""
        assert multimodal_model.audio_only_classifier is not None
        assert multimodal_model.text_only_classifier is not None


class TestForwardPassShapes:
    """Tests for forward pass output shapes."""
    
    @given(batch_size=batch_size_strategy)
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_forward_returns_correct_logits_shape(
        self,
        batch_size: int,
        multimodal_model
    ):
        """Forward pass SHALL return logits of shape (batch, num_classes)."""
        audio_input = torch.randn(batch_size, 80, 500)
        text_input_ids = torch.randint(0, 1000, (batch_size, 128))
        text_attention_mask = torch.ones(batch_size, 128)
        
        with torch.no_grad():
            logits = multimodal_model(
                audio_input,
                text_input_ids,
                text_attention_mask
            )
        
        assert logits.shape == (batch_size, 2), (
            f"Logits shape {logits.shape} doesn't match expected ({batch_size}, 2)"
        )
    
    def test_forward_with_embeddings_returns_tuple(self, multimodal_model):
        """Forward with return_embeddings=True SHALL return tuple."""
        batch_size = 4
        audio_input = torch.randn(batch_size, 80, 500)
        text_input_ids = torch.randint(0, 1000, (batch_size, 128))
        text_attention_mask = torch.ones(batch_size, 128)
        
        with torch.no_grad():
            result = multimodal_model(
                audio_input,
                text_input_ids,
                text_attention_mask,
                return_embeddings=True
            )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        logits, embeddings = result
        assert logits.shape == (batch_size, 2)
        assert embeddings.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
