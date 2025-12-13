"""
ECAPA-TDNN Model Architecture

This module implements the second stream of the hybrid architecture
as specified in Section 5: Spectral and Prosodic Modeling (ECAPA-TDNN)

Key components:
- SE-Res2Net blocks with progressive dilations
- Squeeze-and-Excitation (SE) module
- Multi-Scale Feature Aggregation (MFA)
- Attentive Statistics Pooling (ASP)
- AAM Softmax loss support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union, List
import logging

from config import ECAPAModelConfig, DepressionDetectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) Module.
    
    Implements channel attention as described in Section 5.1:
    - Rescales channel weights adaptively
    - Emphasizes frequency bands correlating with depressive speech
    - Suppresses irrelevant spectral bands
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks"
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 8):
        """
        Args:
            channels: Number of input channels
            reduction_ratio: Bottleneck reduction ratio (r=8 in blueprint)
        """
        super().__init__()
        
        bottleneck = channels // reduction_ratio
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Channel-reweighted tensor of same shape
        """
        scale = self.se(x)  # (batch, channels, 1)
        return x * scale


class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale feature learning.
    
    Hierarchical residual-like connections within a single residual block.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8
    ):
        """
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor for temporal context
            scale: Number of feature groups (scale factor)
        """
        super().__init__()
        
        self.scale = scale
        self.width = channels // scale
        
        # Parallel convolutions at different scales
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.width,
                self.width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2
            )
            for _ in range(scale - 1)
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width)
            for _ in range(scale - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time)
        """
        # Split input into groups
        chunks = x.chunk(self.scale, dim=1)
        
        outputs = [chunks[0]]  # First chunk passes through directly
        
        for i in range(1, self.scale):
            if i == 1:
                y = self.convs[i - 1](chunks[i])
            else:
                y = self.convs[i - 1](chunks[i] + outputs[-1])
            y = F.relu(self.bns[i - 1](y))
            outputs.append(y)
        
        return torch.cat(outputs, dim=1)


class SERes2NetBlock(nn.Module):
    """SE-Res2Net block combining Res2Net and SE module.
    
    Implements the core building block described in Section 5.1:
    - Res2Net for multi-scale temporal features
    - SE module for channel attention
    - Progressive dilations (2, 3, 4) for expanding receptive field
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        se_reduction: int = 8
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.res2net = Res2NetBlock(
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            scale=scale
        )
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SEModule(out_channels, se_reduction)
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res2net(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        
        x = x + residual
        x = F.relu(x)
        
        return x


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling for ECAPA-TDNN.
    
    Same concept as in SSL model but adapted for spectrogram features.
    Produces fixed-size embedding from variable-length input.
    """
    
    def __init__(self, in_channels: int, attention_channels: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels * 3, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, in_channels, kernel_size=1),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Pooled tensor of shape (batch, channels * 2)
        """
        # Compute global statistics for attention context
        global_mean = x.mean(dim=2, keepdim=True).expand_as(x)
        global_std = x.std(dim=2, keepdim=True).expand_as(x)
        
        # Concatenate with input for attention computation
        attention_input = torch.cat([x, global_mean, global_std], dim=1)
        
        # Compute attention weights
        attention_weights = self.attention(attention_input)  # (batch, channels, time)
        
        # Weighted mean
        weighted_mean = torch.sum(attention_weights * x, dim=2)  # (batch, channels)
        
        # Weighted std
        weighted_std = torch.sqrt(
            torch.sum(attention_weights * (x ** 2), dim=2) - weighted_mean ** 2 + 1e-8
        )
        
        # Concatenate mean and std
        return torch.cat([weighted_mean, weighted_std], dim=1)


class AAMSoftmax(nn.Module):
    """Additive Angular Margin (AAM) Softmax Loss.
    
    Implements the loss function described in Section 5.2:
    - Margin m = 0.2
    - Scale s = 30
    - Enforces tighter intra-class compactness
    - Enhances decision boundary between "Depressed" and "Healthy" clusters
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0
    ):
        super().__init__()
        
        self.margin = margin
        self.scale = scale
        
        # Learnable class centers
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Input embeddings of shape (batch, in_features)
            labels: Ground truth labels of shape (batch,) - only needed for training
            
        Returns:
            Scaled logits for loss computation, or cosine similarities for inference
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight)  # (batch, num_classes)
        
        if labels is None:
            # Inference mode - return scaled cosine
            return self.scale * cosine
        
        # Training mode - apply angular margin
        sine = torch.sqrt(1.0 - cosine ** 2 + 1e-8)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)
        
        # Ensure numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        return self.scale * output


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation TDNN).
    
    Implements the complete acoustic stream architecture as described in Section 5:
    1. Initial convolution layer
    2. Three SE-Res2Net blocks with progressive dilations (2, 3, 4)
    3. Multi-Scale Feature Aggregation (MFA)
    4. Attentive Statistics Pooling (ASP)
    5. Bottleneck embedding layer
    6. Classification layer (with optional AAM Softmax)
    """
    
    def __init__(self, config: ECAPAModelConfig):
        super().__init__()
        
        self.config = config
        
        # Initial convolution
        self.conv1 = nn.Conv1d(
            config.n_mels,
            config.initial_channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(config.initial_channels)
        
        # SE-Res2Net blocks with progressive dilations
        self.se_res2net_blocks = nn.ModuleList()
        for i, dilation in enumerate(config.dilations):
            self.se_res2net_blocks.append(
                SERes2NetBlock(
                    in_channels=config.initial_channels,
                    out_channels=config.initial_channels,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    se_reduction=config.se_reduction_ratio
                )
            )
        
        # Multi-Scale Feature Aggregation (MFA)
        # Concatenate outputs from all blocks + initial conv
        mfa_channels = config.initial_channels * (len(config.dilations) + 1)
        
        self.mfa_conv = nn.Conv1d(mfa_channels, config.aggregation_channels, kernel_size=1)
        self.mfa_bn = nn.BatchNorm1d(config.aggregation_channels)
        
        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(
            config.aggregation_channels,
            config.attention_channels
        )
        
        # ASP outputs channels * 2 (mean + std)
        asp_output = config.aggregation_channels * 2
        
        # Bottleneck embedding
        self.embedding = nn.Sequential(
            nn.Linear(asp_output, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim)
        )
        
        # Classification layer
        self.classifier = AAMSoftmax(
            config.embedding_dim,
            config.num_classes,
            margin=config.aam_margin,
            scale=config.aam_scale
        )
        
        # Standard classifier for inference
        self.fc_classifier = nn.Linear(config.embedding_dim, config.num_classes)
    
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            mel_spectrogram: Input Mel spectrogram of shape (batch, n_mels, time)
            labels: Ground truth labels for AAM Softmax (training only)
            return_embeddings: If True, also return embeddings
            
        Returns:
            Logits of shape (batch, num_classes)
            Optionally also embeddings of shape (batch, embedding_dim)
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(mel_spectrogram)))
        
        # Store outputs for MFA
        outputs = [x]
        
        # SE-Res2Net blocks
        for block in self.se_res2net_blocks:
            x = block(x)
            outputs.append(x)
        
        # Multi-Scale Feature Aggregation
        x = torch.cat(outputs, dim=1)
        x = F.relu(self.mfa_bn(self.mfa_conv(x)))
        
        # Attentive Statistics Pooling
        x = self.asp(x)
        
        # Embedding
        embeddings = self.embedding(x)
        
        # Classification
        if self.training and labels is not None:
            # Use AAM Softmax during training
            logits = self.classifier(embeddings, labels)
        else:
            # Use standard classifier for inference
            logits = self.fc_classifier(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        return logits
    
    def get_embeddings(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without classification.
        
        Useful for ensemble stacking.
        """
        _, embeddings = self.forward(mel_spectrogram, return_embeddings=True)
        return embeddings


class ECAPATDNNWithCrossEntropy(ECAPATDNN):
    """ECAPA-TDNN variant using standard Cross-Entropy loss.
    
    For simpler training or when fine-tuning from AAM to CE.
    """
    
    def __init__(self, config: ECAPAModelConfig):
        super().__init__(config)
        
        # Override classifier to simple linear
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)
    
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with standard classification."""
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(mel_spectrogram)))
        
        outputs = [x]
        
        for block in self.se_res2net_blocks:
            x = block(x)
            outputs.append(x)
        
        x = torch.cat(outputs, dim=1)
        x = F.relu(self.mfa_bn(self.mfa_conv(x)))
        
        x = self.asp(x)
        
        embeddings = self.embedding(x)
        
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        return logits


class TinyECAPATDNN(nn.Module):
    """Lightweight ECAPA-TDNN for faster experimentation.
    
    Reduced model size while maintaining key architectural features.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 256,
        embedding_dim: int = 128,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        
        # Simplified SE-Res2Net blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, dilation=d, padding=d),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                SEModule(channels, reduction_ratio=4)
            )
            for d in [2, 3, 4]
        ])
        
        # Pooling
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes)
        )
    
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(mel_spectrogram)))
        
        for block in self.blocks:
            x = x + block(x)
        
        x = self.pool(x)
        logits = self.classifier(x)
        
        if return_embeddings:
            # Get embedding before final layer
            embedding = self.classifier[:-1](x)
            return logits, embedding
        return logits


def create_ecapa_model(
    config: DepressionDetectionConfig,
    use_aam_softmax: bool = True
) -> ECAPATDNN:
    """Factory function to create ECAPA-TDNN model.
    
    Args:
        config: Full configuration object
        use_aam_softmax: Whether to use AAM Softmax loss
        
    Returns:
        Initialized ECAPA-TDNN model
    """
    if use_aam_softmax:
        model = ECAPATDNN(config.ecapa_model)
    else:
        model = ECAPATDNNWithCrossEntropy(config.ecapa_model)
    
    logger.info(f"Created ECAPA-TDNN model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def load_ecapa_checkpoint(
    checkpoint_path: str,
    config: DepressionDetectionConfig
) -> ECAPATDNN:
    """Load ECAPA-TDNN model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        config: Configuration object
        
    Returns:
        Loaded model
    """
    model = create_ecapa_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded ECAPA-TDNN model from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    from config import get_config
    
    config = get_config("combined")
    
    print("Creating ECAPA-TDNN model...")
    model = create_ecapa_model(config)
    
    # Test forward pass
    batch_size = 4
    n_mels = config.ecapa_model.n_mels
    time_steps = 500  # ~5 seconds at 10ms hop
    
    dummy_input = torch.randn(batch_size, n_mels, time_steps)
    dummy_labels = torch.randint(0, 2, (batch_size,))
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Training mode
    model.train()
    logits, embeddings = model(dummy_input, dummy_labels, return_embeddings=True)
    print(f"Training - Logits shape: {logits.shape}")
    print(f"Training - Embeddings shape: {embeddings.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"Inference - Logits shape: {logits.shape}")
    print(f"Predicted classes: {logits.argmax(dim=1)}")
    
    # Test tiny model
    print("\n" + "=" * 50)
    print("Testing TinyECAPATDNN...")
    tiny_model = TinyECAPATDNN()
    with torch.no_grad():
        logits = tiny_model(dummy_input)
    print(f"Tiny model output shape: {logits.shape}")
    print(f"Tiny model parameters: {sum(p.numel() for p in tiny_model.parameters()):,}")
