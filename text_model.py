"""
MuRIL Text Model for Depression Detection

This module implements the transformer-based text classification stream
using Google's MuRIL (Multilingual Representations for Indian Languages) model.

Key components:
- MuRIL encoder for Dravidian language text
- [CLS] token pooling for classification
- Classification head with dropout
- Embedding extraction for multimodal fusion

Reference: Multimodal NLP Upgrade - Requirement 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig
)
from typing import Optional, Tuple, Dict, Union, List
import logging

from config import TextModelConfig, DepressionDetectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MuRILForDepressionDetection(nn.Module):
    """MuRIL-based model for depression detection from text.
    
    Implements the text classification stream as described in Requirement 4:
    1. MuRIL encoder for multilingual Indian language text
    2. [CLS] token pooling for sentence-level representation
    3. Classification head with dropout for depression detection
    4. Embedding extraction for multimodal fusion
    
    Attributes:
        config: TextModelConfig with model parameters
        encoder: Pre-trained MuRIL transformer encoder
        dropout: Dropout layer for regularization
        classifier: Linear classification head
    """
    
    def __init__(self, config: TextModelConfig):
        """Initialize MuRIL model for depression detection.
        
        Args:
            config: TextModelConfig containing model parameters
        """
        super().__init__()
        
        self.config = config
        
        # Load pre-trained MuRIL model
        logger.info(f"Loading MuRIL backbone: {config.model_name}")
        try:
            self.encoder = AutoModel.from_pretrained(config.model_name)
        except Exception as e:
            logger.warning(f"Failed to load {config.model_name}: {e}")
            # Fallback to bert-base-multilingual-cased if MuRIL unavailable
            fallback_model = "bert-base-multilingual-cased"
            logger.info(f"Falling back to {fallback_model}")
            self.encoder = AutoModel.from_pretrained(fallback_model)
        
        # Get hidden size from encoder config
        self.hidden_size = self.encoder.config.hidden_size
        
        # Verify hidden size matches config expectation
        if self.hidden_size != config.hidden_size:
            logger.warning(
                f"Encoder hidden size ({self.hidden_size}) differs from "
                f"config ({config.hidden_size}). Using encoder's hidden size."
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, config.num_classes)
        
        # Freeze encoder if specified
        if config.freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen MuRIL encoder parameters")
    
    def _unfreeze_encoder(self):
        """Unfreeze all encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Unfrozen MuRIL encoder parameters")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through MuRIL model.
        
        Args:
            input_ids: Tokenized input tensor of shape (batch, seq_len)
            attention_mask: Attention mask tensor of shape (batch, seq_len)
            return_embeddings: If True, also return [CLS] embeddings
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
            embeddings (optional): [CLS] embeddings of shape (batch, hidden_size)
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token embedding (first token)
        # Shape: (batch, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(cls_embedding)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        if return_embeddings:
            return logits, cls_embedding
        return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract [CLS] embeddings without classification.
        
        Useful for multimodal fusion where text embeddings are concatenated
        with audio embeddings.
        
        Args:
            input_ids: Tokenized input tensor of shape (batch, seq_len)
            attention_mask: Attention mask tensor of shape (batch, seq_len)
            
        Returns:
            embeddings: [CLS] embeddings of shape (batch, hidden_size)
        """
        _, embeddings = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_embeddings=True
        )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension.
        
        Returns:
            int: The hidden size of the encoder (768 for MuRIL-base)
        """
        return self.hidden_size


class MuRILTokenizerWrapper:
    """Wrapper for MuRIL tokenizer with convenience methods.
    
    Provides easy tokenization for Tamil and Malayalam text with
    proper padding and truncation.
    """
    
    def __init__(self, config: TextModelConfig):
        """Initialize tokenizer.
        
        Args:
            config: TextModelConfig with tokenizer parameters
        """
        self.config = config
        
        logger.info(f"Loading tokenizer: {config.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {config.model_name}: {e}")
            fallback_model = "bert-base-multilingual-cased"
            logger.info(f"Falling back to {fallback_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text(s) for MuRIL model.
        
        Args:
            texts: Single text or list of texts to tokenize
            max_length: Maximum sequence length (uses config default if None)
            return_tensors: Return type ("pt" for PyTorch tensors)
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        if max_length is None:
            max_length = self.config.max_length
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def create_muril_model(
    config: Optional[TextModelConfig] = None
) -> MuRILForDepressionDetection:
    """Factory function to create MuRIL model.
    
    Args:
        config: TextModelConfig (uses default if None)
        
    Returns:
        Initialized MuRILForDepressionDetection model
    """
    if config is None:
        config = TextModelConfig()
    
    model = MuRILForDepressionDetection(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created MuRIL model with {total_params:,} parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_muril_tokenizer(
    config: Optional[TextModelConfig] = None
) -> MuRILTokenizerWrapper:
    """Factory function to create MuRIL tokenizer.
    
    Args:
        config: TextModelConfig (uses default if None)
        
    Returns:
        Initialized MuRILTokenizerWrapper
    """
    if config is None:
        config = TextModelConfig()
    
    return MuRILTokenizerWrapper(config)


def load_muril_checkpoint(
    checkpoint_path: str,
    config: Optional[TextModelConfig] = None
) -> MuRILForDepressionDetection:
    """Load MuRIL model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        config: TextModelConfig (uses default if None)
        
    Returns:
        Loaded model
    """
    if config is None:
        config = TextModelConfig()
    
    model = create_muril_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded MuRIL model from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    from config import get_config
    
    config = get_config("combined")
    
    print("Creating MuRIL model...")
    model = create_muril_model(config.text_model)
    
    print("\nCreating tokenizer...")
    tokenizer = create_muril_tokenizer(config.text_model)
    
    # Test with sample texts (Tamil and Malayalam)
    sample_texts = [
        "நான் மிகவும் சோர்வாக உணர்கிறேன்",  # Tamil: "I feel very tired"
        "എനിക്ക് വളരെ ക്ഷീണം തോന്നുന്നു",  # Malayalam: "I feel very tired"
        "I am feeling happy today",  # English (code-mixed support)
    ]
    
    print(f"\nTokenizing {len(sample_texts)} sample texts...")
    encoded = tokenizer.tokenize(sample_texts)
    
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    
    # Test forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(
            encoded['input_ids'],
            encoded['attention_mask'],
            return_embeddings=True
        )
    
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {model.get_embedding_dim()}")
    print(f"Predicted classes: {logits.argmax(dim=1)}")
    
    # Test get_embeddings method
    print("\nTesting get_embeddings method...")
    with torch.no_grad():
        emb = model.get_embeddings(encoded['input_ids'], encoded['attention_mask'])
    print(f"Embeddings from get_embeddings: {emb.shape}")
    
    # Verify embeddings match
    assert torch.allclose(embeddings, emb), "Embeddings should match"
    print("✓ Embeddings match between forward() and get_embeddings()")
