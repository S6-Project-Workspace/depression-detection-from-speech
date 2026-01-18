"""
Multimodal Fusion Model for Depression Detection

This module implements the multimodal fusion architecture that combines
audio embeddings (ECAPA-TDNN/Wav2Vec2) with text embeddings (MuRIL)
for improved depression detection.

Key components:
- Embedding concatenation for intermediate fusion
- Fusion MLP with dropout for classification
- Fallback methods for unimodal inference when one modality is missing

Reference: Multimodal NLP Upgrade - Requirement 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import logging

from config import FusionConfig, DepressionDetectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionMLP(nn.Module):
    """Fusion MLP for combining multimodal embeddings.
    
    Implements a two-layer MLP with dropout for classification
    from concatenated audio and text embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3
    ):
        """Initialize Fusion MLP.
        
        Args:
            input_dim: Dimension of concatenated embeddings
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fusion MLP.
        
        Args:
            x: Concatenated embeddings of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.mlp(x)


class MultimodalDepressionModel(nn.Module):
    """Multimodal Depression Detection Model.
    
    Combines audio and text embeddings through concatenation and
    passes them through a fusion MLP for classification.
    
    Implements Requirement 5:
    - Extract embeddings from audio model (192-d or 768-d) and text model (768-d)
    - Concatenate into a single fused representation
    - Pass through shared MLP for classification
    - Apply dropout to prevent modality dominance
    - Support fallback to single modality when one is missing
    
    Attributes:
        config: FusionConfig with model parameters
        audio_model: Pre-trained audio model (ECAPA-TDNN or Wav2Vec2)
        text_model: Pre-trained text model (MuRIL)
        fusion_mlp: MLP for classification from fused embeddings
        audio_only_classifier: Classifier for audio-only fallback
        text_only_classifier: Classifier for text-only fallback
    """
    
    def __init__(
        self,
        audio_model: Optional[nn.Module],
        text_model: Optional[nn.Module],
        config: FusionConfig
    ):
        """Initialize Multimodal Depression Model.
        
        Args:
            audio_model: Pre-trained audio model (can be None for text-only)
            text_model: Pre-trained text model (can be None for audio-only)
            config: FusionConfig containing model parameters
        """
        super().__init__()
        
        self.config = config
        self.audio_model = audio_model
        self.text_model = text_model
        
        # Store embedding dimensions
        self.audio_embedding_dim = config.audio_embedding_dim
        self.text_embedding_dim = config.text_embedding_dim
        
        # Compute fused embedding dimension
        self.fused_embedding_dim = self.audio_embedding_dim + self.text_embedding_dim
        
        # Main fusion MLP for multimodal classification
        self.fusion_mlp = FusionMLP(
            input_dim=self.fused_embedding_dim,
            hidden_dim=config.fusion_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.fusion_dropout
        )
        
        # Fallback classifiers for unimodal inference
        if config.allow_unimodal_fallback:
            # Audio-only classifier
            self.audio_only_classifier = FusionMLP(
                input_dim=self.audio_embedding_dim,
                hidden_dim=config.fusion_hidden_dim // 2,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
            
            # Text-only classifier
            self.text_only_classifier = FusionMLP(
                input_dim=self.text_embedding_dim,
                hidden_dim=config.fusion_hidden_dim // 2,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
        else:
            self.audio_only_classifier = None
            self.text_only_classifier = None
        
        logger.info(
            f"Created MultimodalDepressionModel with "
            f"audio_dim={self.audio_embedding_dim}, "
            f"text_dim={self.text_embedding_dim}, "
            f"fused_dim={self.fused_embedding_dim}"
        )
    
    def _extract_audio_embeddings(
        self,
        audio_input: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract embeddings from audio model.
        
        Args:
            audio_input: Audio input (waveform or mel spectrogram)
            audio_attention_mask: Optional attention mask for audio
            
        Returns:
            Audio embeddings of shape (batch, audio_embedding_dim)
        """
        if self.audio_model is None:
            raise ValueError("Audio model not provided")
        
        # Check if audio model has get_embeddings method
        if hasattr(self.audio_model, 'get_embeddings'):
            if audio_attention_mask is not None:
                return self.audio_model.get_embeddings(audio_input, audio_attention_mask)
            return self.audio_model.get_embeddings(audio_input)
        else:
            # Fallback: use forward with return_embeddings=True
            _, embeddings = self.audio_model(
                audio_input,
                return_embeddings=True
            )
            return embeddings
    
    def _extract_text_embeddings(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract embeddings from text model.
        
        Args:
            text_input_ids: Tokenized text input
            text_attention_mask: Attention mask for text
            
        Returns:
            Text embeddings of shape (batch, text_embedding_dim)
        """
        if self.text_model is None:
            raise ValueError("Text model not provided")
        
        # Use get_embeddings method from MuRIL model
        if hasattr(self.text_model, 'get_embeddings'):
            return self.text_model.get_embeddings(text_input_ids, text_attention_mask)
        else:
            # Fallback: use forward with return_embeddings=True
            _, embeddings = self.text_model(
                text_input_ids,
                text_attention_mask,
                return_embeddings=True
            )
            return embeddings
    
    def forward(
        self,
        audio_input: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through multimodal model.
        
        Args:
            audio_input: Audio input (waveform or mel spectrogram)
            text_input_ids: Tokenized text input
            text_attention_mask: Attention mask for text
            audio_attention_mask: Optional attention mask for audio
            return_embeddings: If True, also return fused embeddings
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
            fused_embeddings (optional): Concatenated embeddings
        """
        # Extract embeddings from both modalities
        audio_embeddings = self._extract_audio_embeddings(
            audio_input, audio_attention_mask
        )
        text_embeddings = self._extract_text_embeddings(
            text_input_ids, text_attention_mask
        )
        
        # Concatenate embeddings (Requirement 5.2)
        fused_embeddings = torch.cat([audio_embeddings, text_embeddings], dim=1)
        
        # Pass through fusion MLP (Requirement 5.3)
        logits = self.fusion_mlp(fused_embeddings)
        
        if return_embeddings:
            return logits, fused_embeddings
        return logits
    
    def forward_audio_only(
        self,
        audio_input: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using only audio modality (fallback).
        
        Used when text/ASR is unavailable (Requirement 5.6).
        
        Args:
            audio_input: Audio input (waveform or mel spectrogram)
            audio_attention_mask: Optional attention mask for audio
            return_embeddings: If True, also return audio embeddings
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
            audio_embeddings (optional): Audio embeddings
        """
        if self.audio_only_classifier is None:
            raise ValueError(
                "Audio-only fallback not enabled. "
                "Set allow_unimodal_fallback=True in FusionConfig."
            )
        
        # Extract audio embeddings
        audio_embeddings = self._extract_audio_embeddings(
            audio_input, audio_attention_mask
        )
        
        # Classify using audio-only classifier
        logits = self.audio_only_classifier(audio_embeddings)
        
        if return_embeddings:
            return logits, audio_embeddings
        return logits
    
    def forward_text_only(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using only text modality (fallback).
        
        Used when audio is unavailable (Requirement 5.6).
        
        Args:
            text_input_ids: Tokenized text input
            text_attention_mask: Attention mask for text
            return_embeddings: If True, also return text embeddings
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
            text_embeddings (optional): Text embeddings
        """
        if self.text_only_classifier is None:
            raise ValueError(
                "Text-only fallback not enabled. "
                "Set allow_unimodal_fallback=True in FusionConfig."
            )
        
        # Extract text embeddings
        text_embeddings = self._extract_text_embeddings(
            text_input_ids, text_attention_mask
        )
        
        # Classify using text-only classifier
        logits = self.text_only_classifier(text_embeddings)
        
        if return_embeddings:
            return logits, text_embeddings
        return logits
    
    def forward_with_precomputed_embeddings(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using pre-computed embeddings.
        
        Useful when embeddings are cached or computed separately.
        
        Args:
            audio_embeddings: Pre-computed audio embeddings
            text_embeddings: Pre-computed text embeddings
            return_embeddings: If True, also return fused embeddings
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
            fused_embeddings (optional): Concatenated embeddings
        """
        # Concatenate embeddings
        fused_embeddings = torch.cat([audio_embeddings, text_embeddings], dim=1)
        
        # Pass through fusion MLP
        logits = self.fusion_mlp(fused_embeddings)
        
        if return_embeddings:
            return logits, fused_embeddings
        return logits
    
    def get_fused_embedding_dim(self) -> int:
        """Return the dimension of fused embeddings.
        
        Returns:
            int: audio_embedding_dim + text_embedding_dim
        """
        return self.fused_embedding_dim
    
    def get_audio_embedding_dim(self) -> int:
        """Return the dimension of audio embeddings."""
        return self.audio_embedding_dim
    
    def get_text_embedding_dim(self) -> int:
        """Return the dimension of text embeddings."""
        return self.text_embedding_dim



class MultimodalDepressionModelWithProjection(MultimodalDepressionModel):
    """Multimodal model with projection layers for embedding alignment.
    
    Adds projection layers to map audio and text embeddings to a common
    dimension before concatenation. Useful when embedding dimensions
    are very different.
    """
    
    def __init__(
        self,
        audio_model: Optional[nn.Module],
        text_model: Optional[nn.Module],
        config: FusionConfig,
        projection_dim: int = 256
    ):
        """Initialize with projection layers.
        
        Args:
            audio_model: Pre-trained audio model
            text_model: Pre-trained text model
            config: FusionConfig containing model parameters
            projection_dim: Common dimension for projected embeddings
        """
        # Temporarily modify config for parent init
        original_audio_dim = config.audio_embedding_dim
        original_text_dim = config.text_embedding_dim
        
        # Set projected dimensions
        config.audio_embedding_dim = projection_dim
        config.text_embedding_dim = projection_dim
        
        super().__init__(audio_model, text_model, config)
        
        # Restore original dimensions
        self._original_audio_dim = original_audio_dim
        self._original_text_dim = original_text_dim
        
        # Add projection layers
        self.audio_projection = nn.Sequential(
            nn.Linear(original_audio_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(original_text_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )
        
        logger.info(
            f"Created MultimodalDepressionModelWithProjection: "
            f"audio {original_audio_dim} -> {projection_dim}, "
            f"text {original_text_dim} -> {projection_dim}"
        )
    
    def _extract_audio_embeddings(
        self,
        audio_input: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract and project audio embeddings."""
        # Get raw embeddings from parent
        if self.audio_model is None:
            raise ValueError("Audio model not provided")
        
        if hasattr(self.audio_model, 'get_embeddings'):
            if audio_attention_mask is not None:
                raw_embeddings = self.audio_model.get_embeddings(
                    audio_input, audio_attention_mask
                )
            else:
                raw_embeddings = self.audio_model.get_embeddings(audio_input)
        else:
            _, raw_embeddings = self.audio_model(
                audio_input, return_embeddings=True
            )
        
        # Project to common dimension
        return self.audio_projection(raw_embeddings)
    
    def _extract_text_embeddings(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract and project text embeddings."""
        if self.text_model is None:
            raise ValueError("Text model not provided")
        
        if hasattr(self.text_model, 'get_embeddings'):
            raw_embeddings = self.text_model.get_embeddings(
                text_input_ids, text_attention_mask
            )
        else:
            _, raw_embeddings = self.text_model(
                text_input_ids, text_attention_mask, return_embeddings=True
            )
        
        # Project to common dimension
        return self.text_projection(raw_embeddings)


def create_multimodal_model(
    audio_model: Optional[nn.Module] = None,
    text_model: Optional[nn.Module] = None,
    config: Optional[FusionConfig] = None,
    use_projection: bool = False,
    projection_dim: int = 256
) -> MultimodalDepressionModel:
    """Factory function to create multimodal model.
    
    Args:
        audio_model: Pre-trained audio model (ECAPA-TDNN or Wav2Vec2)
        text_model: Pre-trained text model (MuRIL)
        config: FusionConfig (uses default if None)
        use_projection: Whether to use projection layers
        projection_dim: Dimension for projected embeddings
        
    Returns:
        Initialized MultimodalDepressionModel
    """
    if config is None:
        config = FusionConfig()
    
    if use_projection:
        model = MultimodalDepressionModelWithProjection(
            audio_model=audio_model,
            text_model=text_model,
            config=config,
            projection_dim=projection_dim
        )
    else:
        model = MultimodalDepressionModel(
            audio_model=audio_model,
            text_model=text_model,
            config=config
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    logger.info(f"Created multimodal model with {total_params:,} parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def load_multimodal_checkpoint(
    checkpoint_path: str,
    audio_model: Optional[nn.Module] = None,
    text_model: Optional[nn.Module] = None,
    config: Optional[FusionConfig] = None
) -> MultimodalDepressionModel:
    """Load multimodal model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        audio_model: Pre-trained audio model
        text_model: Pre-trained text model
        config: FusionConfig (uses default if None)
        
    Returns:
        Loaded model
    """
    if config is None:
        config = FusionConfig()
    
    model = create_multimodal_model(
        audio_model=audio_model,
        text_model=text_model,
        config=config
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded multimodal model from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test model creation with mock models
    print("Testing MultimodalDepressionModel...")
    
    # Create mock audio and text models for testing
    class MockAudioModel(nn.Module):
        def __init__(self, embedding_dim: int = 192):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.fc = nn.Linear(80, embedding_dim)
        
        def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
            # Simple pooling for testing
            return self.fc(x.mean(dim=-1))
        
        def forward(
            self, x: torch.Tensor, return_embeddings: bool = False
        ) -> torch.Tensor:
            emb = self.get_embeddings(x)
            logits = torch.zeros(emb.size(0), 2)
            if return_embeddings:
                return logits, emb
            return logits
    
    class MockTextModel(nn.Module):
        def __init__(self, embedding_dim: int = 768):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.fc = nn.Linear(512, embedding_dim)
        
        def get_embeddings(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
        ) -> torch.Tensor:
            # Simple embedding for testing
            return self.fc(torch.randn(input_ids.size(0), 512))
        
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            return_embeddings: bool = False
        ) -> torch.Tensor:
            emb = self.get_embeddings(input_ids, attention_mask)
            logits = torch.zeros(emb.size(0), 2)
            if return_embeddings:
                return logits, emb
            return logits
    
    # Create config
    config = FusionConfig(
        audio_embedding_dim=192,
        text_embedding_dim=768,
        fusion_hidden_dim=512,
        fusion_dropout=0.3,
        num_classes=2,
        allow_unimodal_fallback=True
    )
    
    # Create mock models
    audio_model = MockAudioModel(embedding_dim=192)
    text_model = MockTextModel(embedding_dim=768)
    
    # Create multimodal model
    model = create_multimodal_model(
        audio_model=audio_model,
        text_model=text_model,
        config=config
    )
    
    # Test forward pass
    batch_size = 4
    audio_input = torch.randn(batch_size, 80, 500)  # Mel spectrogram
    text_input_ids = torch.randint(0, 1000, (batch_size, 128))
    text_attention_mask = torch.ones(batch_size, 128)
    
    print(f"\nInput shapes:")
    print(f"  Audio: {audio_input.shape}")
    print(f"  Text IDs: {text_input_ids.shape}")
    
    # Test multimodal forward
    model.eval()
    with torch.no_grad():
        logits, fused_emb = model(
            audio_input,
            text_input_ids,
            text_attention_mask,
            return_embeddings=True
        )
    
    print(f"\nMultimodal forward:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Fused embeddings shape: {fused_emb.shape}")
    print(f"  Expected fused dim: {config.audio_embedding_dim + config.text_embedding_dim}")
    
    # Test audio-only fallback
    with torch.no_grad():
        audio_logits, audio_emb = model.forward_audio_only(
            audio_input, return_embeddings=True
        )
    
    print(f"\nAudio-only forward:")
    print(f"  Logits shape: {audio_logits.shape}")
    print(f"  Audio embeddings shape: {audio_emb.shape}")
    
    # Test text-only fallback
    with torch.no_grad():
        text_logits, text_emb = model.forward_text_only(
            text_input_ids, text_attention_mask, return_embeddings=True
        )
    
    print(f"\nText-only forward:")
    print(f"  Logits shape: {text_logits.shape}")
    print(f"  Text embeddings shape: {text_emb.shape}")
    
    # Test with pre-computed embeddings
    with torch.no_grad():
        precomputed_logits = model.forward_with_precomputed_embeddings(
            audio_emb, text_emb
        )
    
    print(f"\nPre-computed embeddings forward:")
    print(f"  Logits shape: {precomputed_logits.shape}")
    
    # Verify embedding dimensions
    assert fused_emb.shape[1] == config.audio_embedding_dim + config.text_embedding_dim
    assert model.get_fused_embedding_dim() == config.audio_embedding_dim + config.text_embedding_dim
    
    print("\n✓ All tests passed!")
