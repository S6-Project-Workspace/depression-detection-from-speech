"""
Enhanced Multimodal Model with Linguistic Features

This module extends the existing multimodal architecture to include
linguistic features from traditional NLP tasks (POS tagging, NER, dependency parsing).

Reference: Traditional NLP Tasks - Task 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass
import logging

from config import FusionConfig, DepressionDetectionConfig
from multimodal_model import MultimodalDepressionModel, FusionMLP
from linguistic_analyzer import LinguisticAnalyzer, LinguisticAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedFusionConfig:
    """Configuration for enhanced multimodal fusion with linguistic features."""
    # Existing modality dimensions
    audio_embedding_dim: int = 192  # ECAPA-TDNN or 768 for Wav2Vec2
    text_embedding_dim: int = 768   # MuRIL
    linguistic_feature_dim: int = 128  # Linguistic features from NLP tasks
    
    # Fusion architecture
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.3
    num_classes: int = 2
    
    # Fusion strategy
    fusion_method: str = "concatenation"  # "concatenation", "attention", "gated"
    
    # Feature importance analysis
    enable_feature_importance: bool = True
    
    # Fallback behavior
    allow_unimodal_fallback: bool = True
    allow_bimodal_fallback: bool = True  # Audio+text without linguistic


class LinguisticFeatureExtractor(nn.Module):
    """Neural network layer for processing linguistic features.
    
    Applies normalization and projection to linguistic features
    before fusion with other modalities.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        """Initialize linguistic feature extractor.
        
        Args:
            input_dim: Input dimension of linguistic features
            output_dim: Output dimension after projection
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Feature importance weights (learnable)
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        
    def forward(self, linguistic_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through linguistic feature extractor.
        
        Args:
            linguistic_features: Tensor of shape (batch, input_dim)
            
        Returns:
            Processed features of shape (batch, output_dim)
        """
        # Apply feature importance weights
        weighted_features = linguistic_features * self.feature_weights
        
        # Normalize features
        normalized_features = self.layer_norm(weighted_features)
        
        # Project to output dimension
        projected_features = self.projection(normalized_features)
        
        return projected_features
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get learned feature importance weights.
        
        Returns:
            Feature importance weights of shape (input_dim,)
        """
        return torch.softmax(self.feature_weights, dim=0)


class AttentionFusion(nn.Module):
    """Attention-based fusion for multimodal embeddings."""
    
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        linguistic_dim: int,
        hidden_dim: int = 256
    ):
        """Initialize attention fusion.
        
        Args:
            audio_dim: Audio embedding dimension
            text_dim: Text embedding dimension
            linguistic_dim: Linguistic feature dimension
            hidden_dim: Hidden dimension for attention computation
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.linguistic_dim = linguistic_dim
        
        # Project all modalities to common dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.linguistic_proj = nn.Linear(linguistic_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        linguistic_emb: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through attention fusion.
        
        Args:
            audio_emb: Audio embeddings (batch, audio_dim)
            text_emb: Text embeddings (batch, text_dim)
            linguistic_emb: Linguistic embeddings (batch, linguistic_dim)
            modality_mask: Optional mask for missing modalities
            
        Returns:
            Fused embeddings (batch, hidden_dim)
        """
        batch_size = audio_emb.size(0)
        
        # Project to common dimension
        audio_proj = self.audio_proj(audio_emb)  # (batch, hidden_dim)
        text_proj = self.text_proj(text_emb)     # (batch, hidden_dim)
        ling_proj = self.linguistic_proj(linguistic_emb)  # (batch, hidden_dim)
        
        # Stack modalities as sequence
        modalities = torch.stack([audio_proj, text_proj, ling_proj], dim=1)  # (batch, 3, hidden_dim)
        
        # Apply attention
        attended, attention_weights = self.attention(
            modalities, modalities, modalities,
            key_padding_mask=modality_mask
        )
        
        # Pool attended representations
        fused = attended.mean(dim=1)  # (batch, hidden_dim)
        
        # Final projection
        output = self.output_proj(fused)
        
        return output


class EnhancedMultimodalModel(nn.Module):
    """Enhanced Multimodal Model with Linguistic Features.
    
    Extends the existing multimodal architecture to include linguistic features
    from traditional NLP tasks (POS tagging, NER, dependency parsing).
    
    Implements Task 5:
    - Integrate 128-dimensional linguistic features with audio and text embeddings
    - Implement three-way fusion (audio + text + linguistic)
    - Maintain backward compatibility with existing modes
    - Add feature importance analysis for linguistic contributions
    
    Attributes:
        config: EnhancedFusionConfig with model parameters
        audio_model: Pre-trained audio model (ECAPA-TDNN or Wav2Vec2)
        text_model: Pre-trained text model (MuRIL)
        linguistic_analyzer: LinguisticAnalyzer for extracting linguistic features
        linguistic_extractor: Neural network for processing linguistic features
        fusion_layer: Fusion mechanism for combining all modalities
    """
    
    def __init__(
        self,
        audio_model: Optional[nn.Module],
        text_model: Optional[nn.Module],
        linguistic_analyzer: LinguisticAnalyzer,
        config: EnhancedFusionConfig
    ):
        """Initialize Enhanced Multimodal Model.
        
        Args:
            audio_model: Pre-trained audio model (can be None for text+linguistic only)
            text_model: Pre-trained text model (can be None for audio+linguistic only)
            linguistic_analyzer: LinguisticAnalyzer for feature extraction
            config: EnhancedFusionConfig containing model parameters
        """
        super().__init__()
        
        self.config = config
        self.audio_model = audio_model
        self.text_model = text_model
        self.linguistic_analyzer = linguistic_analyzer
        
        # Store embedding dimensions
        self.audio_embedding_dim = config.audio_embedding_dim
        self.text_embedding_dim = config.text_embedding_dim
        self.linguistic_feature_dim = config.linguistic_feature_dim
        
        # Linguistic feature extractor
        self.linguistic_extractor = LinguisticFeatureExtractor(
            input_dim=config.linguistic_feature_dim,
            output_dim=config.linguistic_feature_dim,
            dropout=config.fusion_dropout
        )
        
        # Fusion mechanism
        if config.fusion_method == "concatenation":
            # Compute total fused dimension
            self.fused_embedding_dim = (
                self.audio_embedding_dim + 
                self.text_embedding_dim + 
                self.linguistic_feature_dim
            )
            
            # Main fusion MLP for trimodal classification
            self.fusion_mlp = FusionMLP(
                input_dim=self.fused_embedding_dim,
                hidden_dim=config.fusion_hidden_dim,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
            
        elif config.fusion_method == "attention":
            self.attention_fusion = AttentionFusion(
                audio_dim=self.audio_embedding_dim,
                text_dim=self.text_embedding_dim,
                linguistic_dim=self.linguistic_feature_dim,
                hidden_dim=config.fusion_hidden_dim
            )
            
            self.fusion_mlp = FusionMLP(
                input_dim=config.fusion_hidden_dim,
                hidden_dim=config.fusion_hidden_dim,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
            
        else:
            raise ValueError(f"Unsupported fusion method: {config.fusion_method}")
        
        # Fallback classifiers for partial modality combinations
        if config.allow_bimodal_fallback:
            # Audio + Text (existing multimodal)
            self.audio_text_classifier = FusionMLP(
                input_dim=self.audio_embedding_dim + self.text_embedding_dim,
                hidden_dim=config.fusion_hidden_dim // 2,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
            
            # Audio + Linguistic
            self.audio_linguistic_classifier = FusionMLP(
                input_dim=self.audio_embedding_dim + self.linguistic_feature_dim,
                hidden_dim=config.fusion_hidden_dim // 2,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
            
            # Text + Linguistic
            self.text_linguistic_classifier = FusionMLP(
                input_dim=self.text_embedding_dim + self.linguistic_feature_dim,
                hidden_dim=config.fusion_hidden_dim // 2,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
        
        if config.allow_unimodal_fallback:
            # Individual modality classifiers
            self.audio_only_classifier = FusionMLP(
                input_dim=self.audio_embedding_dim,
                hidden_dim=config.fusion_hidden_dim // 4,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
            
            self.text_only_classifier = FusionMLP(
                input_dim=self.text_embedding_dim,
                hidden_dim=config.fusion_hidden_dim // 4,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
            
            self.linguistic_only_classifier = FusionMLP(
                input_dim=self.linguistic_feature_dim,
                hidden_dim=config.fusion_hidden_dim // 4,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout
            )
        
        logger.info(
            f"Created EnhancedMultimodalModel with "
            f"audio_dim={self.audio_embedding_dim}, "
            f"text_dim={self.text_embedding_dim}, "
            f"linguistic_dim={self.linguistic_feature_dim}, "
            f"fusion_method={config.fusion_method}"
        )
    
    def _extract_audio_embeddings(
        self,
        audio_input: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract embeddings from audio model."""
        if self.audio_model is None:
            raise ValueError("Audio model not provided")
        
        if hasattr(self.audio_model, 'get_embeddings'):
            if audio_attention_mask is not None:
                return self.audio_model.get_embeddings(audio_input, audio_attention_mask)
            return self.audio_model.get_embeddings(audio_input)
        else:
            _, embeddings = self.audio_model(audio_input, return_embeddings=True)
            return embeddings
    
    def _extract_text_embeddings(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract embeddings from text model."""
        if self.text_model is None:
            raise ValueError("Text model not provided")
        
        if hasattr(self.text_model, 'get_embeddings'):
            return self.text_model.get_embeddings(text_input_ids, text_attention_mask)
        else:
            _, embeddings = self.text_model(
                text_input_ids, text_attention_mask, return_embeddings=True
            )
            return embeddings
    
    def _extract_linguistic_features(
        self,
        text_raw: List[str],
        language: str = "tamil"
    ) -> torch.Tensor:
        """Extract linguistic features from raw text.
        
        Args:
            text_raw: List of raw text strings
            language: Language for linguistic analysis
            
        Returns:
            Linguistic features tensor of shape (batch, linguistic_feature_dim)
        """
        batch_size = len(text_raw)
        device = next(self.parameters()).device
        
        # Extract linguistic features for each text
        linguistic_features = []
        
        for text in text_raw:
            if text and text.strip():
                try:
                    # Analyze text using linguistic analyzer
                    analysis = self.linguistic_analyzer.analyze_text(text, language)
                    features = self.linguistic_analyzer.extract_linguistic_features(analysis)
                    linguistic_features.append(torch.tensor(features, dtype=torch.float32))
                except Exception as e:
                    logger.warning(f"Failed to extract linguistic features: {e}")
                    # Use zero features as fallback
                    linguistic_features.append(torch.zeros(self.linguistic_feature_dim, dtype=torch.float32))
            else:
                # Empty text - use zero features
                linguistic_features.append(torch.zeros(self.linguistic_feature_dim, dtype=torch.float32))
        
        # Stack into batch tensor
        linguistic_batch = torch.stack(linguistic_features).to(device)
        
        # Process through linguistic extractor
        processed_features = self.linguistic_extractor(linguistic_batch)
        
        return processed_features
    
    def forward(
        self,
        audio_input: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_raw: Optional[List[str]] = None,
        language: str = "tamil",
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        return_feature_importance: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through enhanced multimodal model.
        
        Args:
            audio_input: Audio input (waveform or mel spectrogram)
            text_input_ids: Tokenized text input
            text_attention_mask: Attention mask for text
            text_raw: Raw text strings for linguistic analysis
            language: Language for linguistic analysis
            audio_attention_mask: Optional attention mask for audio
            return_embeddings: If True, also return individual embeddings
            return_feature_importance: If True, return linguistic feature importance
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
            embeddings (optional): Tuple of (audio_emb, text_emb, linguistic_emb, fused_emb)
            feature_importance (optional): Linguistic feature importance weights
        """
        # Extract embeddings from all modalities
        audio_embeddings = self._extract_audio_embeddings(audio_input, audio_attention_mask)
        text_embeddings = self._extract_text_embeddings(text_input_ids, text_attention_mask)
        
        # Extract linguistic features if raw text is provided
        if text_raw is not None:
            linguistic_embeddings = self._extract_linguistic_features(text_raw, language)
        else:
            # Use zero linguistic features as fallback
            batch_size = audio_embeddings.size(0)
            device = audio_embeddings.device
            linguistic_embeddings = torch.zeros(
                batch_size, self.linguistic_feature_dim, device=device
            )
        
        # Fusion
        if self.config.fusion_method == "concatenation":
            # Concatenate all embeddings
            fused_embeddings = torch.cat([
                audio_embeddings, text_embeddings, linguistic_embeddings
            ], dim=1)
            
            # Pass through fusion MLP
            logits = self.fusion_mlp(fused_embeddings)
            
        elif self.config.fusion_method == "attention":
            # Use attention-based fusion
            fused_embeddings = self.attention_fusion(
                audio_embeddings, text_embeddings, linguistic_embeddings
            )
            
            # Pass through fusion MLP
            logits = self.fusion_mlp(fused_embeddings)
        
        # Prepare return values
        result = [logits]
        
        if return_embeddings:
            result.append((audio_embeddings, text_embeddings, linguistic_embeddings, fused_embeddings))
        
        if return_feature_importance:
            feature_importance = self.linguistic_extractor.get_feature_importance()
            result.append(feature_importance)
        
        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)
    
    def forward_audio_text(
        self,
        audio_input: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using only audio and text modalities (bimodal fallback).
        
        Args:
            audio_input: Audio input
            text_input_ids: Tokenized text input
            text_attention_mask: Attention mask for text
            audio_attention_mask: Optional attention mask for audio
            return_embeddings: If True, also return embeddings
            
        Returns:
            logits: Classification logits
            embeddings (optional): Concatenated audio+text embeddings
        """
        if not self.config.allow_bimodal_fallback:
            raise ValueError("Bimodal fallback not enabled")
        
        # Extract embeddings
        audio_embeddings = self._extract_audio_embeddings(audio_input, audio_attention_mask)
        text_embeddings = self._extract_text_embeddings(text_input_ids, text_attention_mask)
        
        # Concatenate and classify
        fused_embeddings = torch.cat([audio_embeddings, text_embeddings], dim=1)
        logits = self.audio_text_classifier(fused_embeddings)
        
        if return_embeddings:
            return logits, fused_embeddings
        return logits
    
    def forward_audio_linguistic(
        self,
        audio_input: torch.Tensor,
        text_raw: List[str],
        language: str = "tamil",
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using audio and linguistic features (bimodal fallback).
        
        Args:
            audio_input: Audio input
            text_raw: Raw text for linguistic analysis
            language: Language for linguistic analysis
            audio_attention_mask: Optional attention mask for audio
            return_embeddings: If True, also return embeddings
            
        Returns:
            logits: Classification logits
            embeddings (optional): Concatenated audio+linguistic embeddings
        """
        if not self.config.allow_bimodal_fallback:
            raise ValueError("Bimodal fallback not enabled")
        
        # Extract embeddings
        audio_embeddings = self._extract_audio_embeddings(audio_input, audio_attention_mask)
        linguistic_embeddings = self._extract_linguistic_features(text_raw, language)
        
        # Concatenate and classify
        fused_embeddings = torch.cat([audio_embeddings, linguistic_embeddings], dim=1)
        logits = self.audio_linguistic_classifier(fused_embeddings)
        
        if return_embeddings:
            return logits, fused_embeddings
        return logits
    
    def forward_text_linguistic(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_raw: List[str],
        language: str = "tamil",
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using text and linguistic features (bimodal fallback).
        
        Args:
            text_input_ids: Tokenized text input
            text_attention_mask: Attention mask for text
            text_raw: Raw text for linguistic analysis
            language: Language for linguistic analysis
            return_embeddings: If True, also return embeddings
            
        Returns:
            logits: Classification logits
            embeddings (optional): Concatenated text+linguistic embeddings
        """
        if not self.config.allow_bimodal_fallback:
            raise ValueError("Bimodal fallback not enabled")
        
        # Extract embeddings
        text_embeddings = self._extract_text_embeddings(text_input_ids, text_attention_mask)
        linguistic_embeddings = self._extract_linguistic_features(text_raw, language)
        
        # Concatenate and classify
        fused_embeddings = torch.cat([text_embeddings, linguistic_embeddings], dim=1)
        logits = self.text_linguistic_classifier(fused_embeddings)
        
        if return_embeddings:
            return logits, fused_embeddings
        return logits
    
    def forward_audio_only(
        self,
        audio_input: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using only audio modality (unimodal fallback)."""
        if not self.config.allow_unimodal_fallback:
            raise ValueError("Unimodal fallback not enabled")
        
        audio_embeddings = self._extract_audio_embeddings(audio_input, audio_attention_mask)
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
        """Forward pass using only text modality (unimodal fallback)."""
        if not self.config.allow_unimodal_fallback:
            raise ValueError("Unimodal fallback not enabled")
        
        text_embeddings = self._extract_text_embeddings(text_input_ids, text_attention_mask)
        logits = self.text_only_classifier(text_embeddings)
        
        if return_embeddings:
            return logits, text_embeddings
        return logits
    
    def forward_linguistic_only(
        self,
        text_raw: List[str],
        language: str = "tamil",
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using only linguistic features (unimodal fallback)."""
        if not self.config.allow_unimodal_fallback:
            raise ValueError("Unimodal fallback not enabled")
        
        linguistic_embeddings = self._extract_linguistic_features(text_raw, language)
        logits = self.linguistic_only_classifier(linguistic_embeddings)
        
        if return_embeddings:
            return logits, linguistic_embeddings
        return logits
    
    def get_feature_importance_analysis(self) -> Dict[str, torch.Tensor]:
        """Get comprehensive feature importance analysis.
        
        Returns:
            Dictionary with feature importance information
        """
        analysis = {}
        
        # Linguistic feature importance
        if hasattr(self.linguistic_extractor, 'get_feature_importance'):
            analysis['linguistic_importance'] = self.linguistic_extractor.get_feature_importance()
        
        # Attention weights (if using attention fusion)
        if hasattr(self, 'attention_fusion'):
            # This would require storing attention weights during forward pass
            analysis['attention_weights'] = None  # Placeholder
        
        return analysis
    
    def get_embedding_dimensions(self) -> Dict[str, int]:
        """Get embedding dimensions for all modalities.
        
        Returns:
            Dictionary with embedding dimensions
        """
        return {
            'audio': self.audio_embedding_dim,
            'text': self.text_embedding_dim,
            'linguistic': self.linguistic_feature_dim,
            'fused': getattr(self, 'fused_embedding_dim', 
                           self.audio_embedding_dim + self.text_embedding_dim + self.linguistic_feature_dim)
        }


def create_enhanced_multimodal_model(
    audio_model: Optional[nn.Module] = None,
    text_model: Optional[nn.Module] = None,
    linguistic_analyzer: Optional[LinguisticAnalyzer] = None,
    config: Optional[EnhancedFusionConfig] = None
) -> EnhancedMultimodalModel:
    """Factory function to create enhanced multimodal model.
    
    Args:
        audio_model: Pre-trained audio model (ECAPA-TDNN or Wav2Vec2)
        text_model: Pre-trained text model (MuRIL)
        linguistic_analyzer: LinguisticAnalyzer for feature extraction
        config: EnhancedFusionConfig (uses default if None)
        
    Returns:
        Initialized EnhancedMultimodalModel
    """
    if config is None:
        config = EnhancedFusionConfig()
    
    if linguistic_analyzer is None:
        from linguistic_analyzer import create_linguistic_analyzer
        linguistic_analyzer = create_linguistic_analyzer()
    
    model = EnhancedMultimodalModel(
        audio_model=audio_model,
        text_model=text_model,
        linguistic_analyzer=linguistic_analyzer,
        config=config
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created enhanced multimodal model with {total_params:,} parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def load_enhanced_multimodal_checkpoint(
    checkpoint_path: str,
    audio_model: Optional[nn.Module] = None,
    text_model: Optional[nn.Module] = None,
    linguistic_analyzer: Optional[LinguisticAnalyzer] = None,
    config: Optional[EnhancedFusionConfig] = None
) -> EnhancedMultimodalModel:
    """Load enhanced multimodal model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        audio_model: Pre-trained audio model
        text_model: Pre-trained text model
        linguistic_analyzer: LinguisticAnalyzer for feature extraction
        config: EnhancedFusionConfig (uses default if None)
        
    Returns:
        Loaded model
    """
    model = create_enhanced_multimodal_model(
        audio_model=audio_model,
        text_model=text_model,
        linguistic_analyzer=linguistic_analyzer,
        config=config
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded enhanced multimodal model from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test enhanced model creation
    print("Testing EnhancedMultimodalModel...")
    
    # Import required modules
    from dataclasses import dataclass
    
    # Create mock models for testing
    class MockAudioModel(nn.Module):
        def __init__(self, embedding_dim: int = 192):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.fc = nn.Linear(80, embedding_dim)
        
        def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x.mean(dim=-1))
    
    class MockTextModel(nn.Module):
        def __init__(self, embedding_dim: int = 768):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.fc = nn.Linear(512, embedding_dim)
        
        def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.fc(torch.randn(input_ids.size(0), 512))
    
    class MockLinguisticAnalyzer:
        def analyze_text(self, text: str, language: str):
            return None
        
        def extract_linguistic_features(self, analysis):
            return torch.randn(128).numpy()
    
    # Create config
    config = EnhancedFusionConfig(
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
    
    # Create mock models
    audio_model = MockAudioModel(embedding_dim=192)
    text_model = MockTextModel(embedding_dim=768)
    linguistic_analyzer = MockLinguisticAnalyzer()
    
    # Create enhanced multimodal model
    model = EnhancedMultimodalModel(
        audio_model=audio_model,
        text_model=text_model,
        linguistic_analyzer=linguistic_analyzer,
        config=config
    )
    
    # Test forward pass
    batch_size = 4
    audio_input = torch.randn(batch_size, 80, 500)
    text_input_ids = torch.randint(0, 1000, (batch_size, 128))
    text_attention_mask = torch.ones(batch_size, 128)
    text_raw = ["Sample text for analysis"] * batch_size
    
    print(f"\nInput shapes:")
    print(f"  Audio: {audio_input.shape}")
    print(f"  Text IDs: {text_input_ids.shape}")
    print(f"  Text raw: {len(text_raw)} samples")
    
    # Test enhanced multimodal forward
    model.eval()
    with torch.no_grad():
        logits, embeddings, feature_importance = model(
            audio_input,
            text_input_ids,
            text_attention_mask,
            text_raw=text_raw,
            language="tamil",
            return_embeddings=True,
            return_feature_importance=True
        )
    
    audio_emb, text_emb, ling_emb, fused_emb = embeddings
    
    print(f"\nEnhanced multimodal forward:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Audio embeddings: {audio_emb.shape}")
    print(f"  Text embeddings: {text_emb.shape}")
    print(f"  Linguistic embeddings: {ling_emb.shape}")
    print(f"  Fused embeddings: {fused_emb.shape}")
    print(f"  Feature importance: {feature_importance.shape}")
    
    # Test bimodal fallbacks
    with torch.no_grad():
        audio_text_logits = model.forward_audio_text(
            audio_input, text_input_ids, text_attention_mask
        )
        audio_ling_logits = model.forward_audio_linguistic(
            audio_input, text_raw, language="tamil"
        )
        text_ling_logits = model.forward_text_linguistic(
            text_input_ids, text_attention_mask, text_raw, language="tamil"
        )
    
    print(f"\nBimodal fallbacks:")
    print(f"  Audio+Text: {audio_text_logits.shape}")
    print(f"  Audio+Linguistic: {audio_ling_logits.shape}")
    print(f"  Text+Linguistic: {text_ling_logits.shape}")
    
    # Test unimodal fallbacks
    with torch.no_grad():
        audio_only_logits = model.forward_audio_only(audio_input)
        text_only_logits = model.forward_text_only(text_input_ids, text_attention_mask)
        ling_only_logits = model.forward_linguistic_only(text_raw, language="tamil")
    
    print(f"\nUnimodal fallbacks:")
    print(f"  Audio only: {audio_only_logits.shape}")
    print(f"  Text only: {text_only_logits.shape}")
    print(f"  Linguistic only: {ling_only_logits.shape}")
    
    # Test embedding dimensions
    dims = model.get_embedding_dimensions()
    print(f"\nEmbedding dimensions: {dims}")
    
    # Verify expected dimensions
    expected_fused_dim = config.audio_embedding_dim + config.text_embedding_dim + config.linguistic_feature_dim
    assert fused_emb.shape[1] == expected_fused_dim, f"Expected {expected_fused_dim}, got {fused_emb.shape[1]}"
    
    print("\n✓ All tests passed!")