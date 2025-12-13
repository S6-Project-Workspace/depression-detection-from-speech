"""
SSL Model Architecture (IndicWav2Vec / Wav2Vec2)

This module implements the first stream of the hybrid architecture
as specified in Section 4: Linguistic and Paralinguistic SSL

Key components:
- Fine-tuned Wav2Vec2/IndicWav2Vec backbone
- Attentive Statistics Pooling (ASP)
- Depression-specific classification head
- Layer-wise freezing strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    Wav2Vec2PreTrainedModel
)
from typing import Optional, Tuple, Dict, Union
import logging

from config import SSLModelConfig, DepressionDetectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling (ASP).
    
    Implements the pooling mechanism described in Section 4.2:
    - Computes attention-weighted mean and standard deviation
    - Captures dynamic range of speaker's voice (depression biomarker)
    
    The attention mechanism:
        α_t = softmax(v^T tanh(W H_t + b))
        μ = Σ α_t H_t
        σ = sqrt(Σ α_t (H_t - μ)^2)
    
    Final embedding: [μ, σ] with size 2 × D
    """
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        """
        Args:
            input_dim: Dimension of input features (D)
            attention_dim: Hidden dimension for attention computation
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Shape (batch, time, features)
            attention_mask: Shape (batch, time) with 1 for valid, 0 for padding
            
        Returns:
            Pooled representation of shape (batch, 2 * features)
        """
        # Compute attention weights: (batch, time, 1)
        attention_scores = self.attention(hidden_states)
        
        # Apply mask to attention scores
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            mask = attention_mask.unsqueeze(-1)  # (batch, time, 1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over time dimension
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, time, 1)
        
        # Weighted mean: μ = Σ α_t H_t
        weighted_mean = torch.sum(attention_weights * hidden_states, dim=1)  # (batch, features)
        
        # Weighted standard deviation: σ = sqrt(Σ α_t (H_t - μ)^2)
        diff = hidden_states - weighted_mean.unsqueeze(1)  # (batch, time, features)
        weighted_var = torch.sum(attention_weights * diff ** 2, dim=1)  # (batch, features)
        weighted_std = torch.sqrt(weighted_var + 1e-8)  # (batch, features)
        
        # Concatenate mean and std: [μ, σ]
        return torch.cat([weighted_mean, weighted_std], dim=1)  # (batch, 2 * features)


class DepressionClassificationHead(nn.Module):
    """Depression-specific classification head.
    
    Two-layer MLP as described in Section 4.2:
    - First layer: pooled_dim (2D) → projection_dim with GELU + Dropout
    - Second layer: projection_dim → num_classes
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes)
        )
    
    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_features: Shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.classifier(pooled_features)


class Wav2Vec2ForDepressionDetection(nn.Module):
    """Wav2Vec2/IndicWav2Vec model for depression detection.
    
    Implements the complete SSL stream architecture:
    1. Wav2Vec2 feature extractor (frozen initially)
    2. Wav2Vec2 Transformer encoder (fine-tuned)
    3. Attentive Statistics Pooling
    4. Classification head
    """
    
    def __init__(self, config: SSLModelConfig):
        super().__init__()
        
        self.config = config
        
        # Load pre-trained Wav2Vec2 model
        logger.info(f"Loading SSL backbone: {config.model_name}")
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.model_name)
        except Exception as e:
            logger.warning(f"Failed to load {config.model_name}: {e}")
            logger.info(f"Falling back to {config.fallback_model}")
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.fallback_model)
        
        # Get hidden size from model
        hidden_size = self.wav2vec2.config.hidden_size
        
        # Attentive Statistics Pooling
        self.pooling = AttentiveStatisticsPooling(
            input_dim=hidden_size,
            attention_dim=config.attention_hidden_dim
        )
        
        # Classification head
        # ASP outputs 2 * hidden_size (mean + std)
        self.classifier = DepressionClassificationHead(
            input_dim=hidden_size * 2,
            projection_dim=config.projection_dim,
            num_classes=config.num_classes,
            dropout=config.dropout_rate
        )
        
        # Freeze feature extractor initially
        if config.freeze_feature_extractor:
            self._freeze_feature_extractor()
        
        # Optional: weighted layer sum
        if config.use_weighted_layer_sum:
            num_layers = self.wav2vec2.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def _freeze_feature_extractor(self):
        """Freeze the CNN feature extractor (first 7 blocks)."""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        
        # Also freeze feature projection
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False
        
        logger.info("Frozen feature extractor layers")
    
    def _unfreeze_feature_extractor(self):
        """Unfreeze the CNN feature extractor for full fine-tuning."""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = True
        
        logger.info("Unfrozen feature extractor layers")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input_values: Raw waveform tensor of shape (batch, samples)
            attention_mask: Optional mask of shape (batch, samples)
            return_embeddings: If True, also return the pooled embeddings
            
        Returns:
            Logits of shape (batch, num_classes)
            Optionally also embeddings of shape (batch, hidden_size * 2)
        """
        # Wav2Vec2 forward pass
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.config.use_weighted_layer_sum
        )
        
        if self.config.use_weighted_layer_sum and hasattr(self, 'layer_weights'):
            # Weighted sum of all hidden states
            hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (layers, batch, time, hidden)
            weights = F.softmax(self.layer_weights, dim=0)
            weights = weights.view(-1, 1, 1, 1)
            hidden_states = (hidden_states * weights).sum(dim=0)  # (batch, time, hidden)
        else:
            hidden_states = outputs.last_hidden_state  # (batch, time, hidden)
        
        # Compute frame-level attention mask for transformer outputs
        if attention_mask is not None:
            # Wav2Vec2 downsamples the input, so we need to adjust the mask
            # The CNN downsamples by factor of ~320 (depending on config)
            # Use the extract_features output length
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            
            # Create mask for transformer outputs
            batch_size, seq_len, _ = hidden_states.shape
            transformer_mask = torch.zeros(batch_size, seq_len, device=hidden_states.device)
            for i, length in enumerate(output_lengths):
                transformer_mask[i, :length] = 1.0
        else:
            transformer_mask = None
        
        # Attentive Statistics Pooling
        pooled = self.pooling(hidden_states, transformer_mask)
        
        # Classification
        logits = self.classifier(pooled)
        
        if return_embeddings:
            return logits, pooled
        return logits
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after feature extraction."""
        # Wav2Vec2 CNN has specific downsampling
        # This is a simplified version - actual downsampling depends on config
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)
        
        # Get kernel sizes and strides from config
        conv_kernels = self.wav2vec2.config.conv_kernel
        conv_strides = self.wav2vec2.config.conv_stride
        
        for kernel, stride in zip(conv_kernels, conv_strides):
            input_lengths = _conv_out_length(input_lengths.float(), kernel, stride)
        
        return input_lengths.long()
    
    def get_embeddings(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract embeddings without classification.
        
        Useful for ensemble stacking where embeddings are concatenated.
        """
        _, embeddings = self.forward(input_values, attention_mask, return_embeddings=True)
        return embeddings


class XLSRForDepressionDetection(Wav2Vec2ForDepressionDetection):
    """XLS-R variant for cross-lingual robustness.
    
    Uses facebook/wav2vec2-xls-r-300m as backbone.
    """
    
    def __init__(self, config: Optional[SSLModelConfig] = None):
        if config is None:
            config = SSLModelConfig()
        
        # Override model name to XLS-R
        config.model_name = "facebook/wav2vec2-xls-r-300m"
        config.hidden_size = 1024  # XLS-R 300M has 1024 hidden size
        
        super().__init__(config)


class MultiStreamSSLModel(nn.Module):
    """Multi-stream SSL model combining IndicWav2Vec and XLS-R.
    
    For advanced ensembling at the embedding level.
    """
    
    def __init__(
        self,
        indic_config: Optional[SSLModelConfig] = None,
        xlsr_config: Optional[SSLModelConfig] = None,
        num_classes: int = 2
    ):
        super().__init__()
        
        if indic_config is None:
            indic_config = SSLModelConfig()
        if xlsr_config is None:
            xlsr_config = SSLModelConfig()
            xlsr_config.model_name = "facebook/wav2vec2-xls-r-300m"
            xlsr_config.hidden_size = 1024
        
        # Two parallel streams
        self.indic_stream = Wav2Vec2ForDepressionDetection(indic_config)
        self.xlsr_stream = Wav2Vec2ForDepressionDetection(xlsr_config)
        
        # Combined embedding dimension
        combined_dim = (indic_config.hidden_size + xlsr_config.hidden_size) * 2
        
        # Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get embeddings from both streams
        _, indic_emb = self.indic_stream(input_values, attention_mask, return_embeddings=True)
        _, xlsr_emb = self.xlsr_stream(input_values, attention_mask, return_embeddings=True)
        
        # Concatenate embeddings
        combined = torch.cat([indic_emb, xlsr_emb], dim=1)
        
        # Classify
        return self.fusion_classifier(combined)


def create_ssl_model(
    config: DepressionDetectionConfig,
    pretrained: bool = True
) -> Wav2Vec2ForDepressionDetection:
    """Factory function to create SSL model.
    
    Args:
        config: Full configuration object
        pretrained: Whether to load pretrained weights
        
    Returns:
        Initialized Wav2Vec2ForDepressionDetection model
    """
    model = Wav2Vec2ForDepressionDetection(config.ssl_model)
    
    logger.info(f"Created SSL model with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def load_ssl_checkpoint(
    checkpoint_path: str,
    config: DepressionDetectionConfig
) -> Wav2Vec2ForDepressionDetection:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        config: Configuration object
        
    Returns:
        Loaded model
    """
    model = create_ssl_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded SSL model from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    from config import get_config
    
    config = get_config("combined")
    
    # Use a smaller model for testing
    config.ssl_model.model_name = "facebook/wav2vec2-base"
    config.ssl_model.fallback_model = "facebook/wav2vec2-base"
    config.ssl_model.hidden_size = 768
    
    print("Creating SSL model...")
    model = create_ssl_model(config)
    
    # Test forward pass
    batch_size = 4
    seq_length = 80000  # 5 seconds at 16kHz
    
    dummy_input = torch.randn(batch_size, seq_length)
    dummy_mask = torch.ones(batch_size, seq_length)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        logits, embeddings = model(dummy_input, dummy_mask, return_embeddings=True)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Predicted classes: {logits.argmax(dim=1)}")
