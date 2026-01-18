"""
Configuration file for Depression Detection in Dravidian Speech
DravidianLangTech @ ACL 2026 Shared Task

This module contains all hyperparameters and configuration settings
for the Hybrid Multi-Stream Fusion architecture.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class Language(Enum):
    """Supported Dravidian languages."""
    TAMIL = "tamil"
    MALAYALAM = "malayalam"
    COMBINED = "combined"


@dataclass
class PathConfig:
    """Data path configurations."""
    # Base paths
    base_dir: str = "/Users/keerthivasan/NLP/NLP Dataset"
    output_dir: str = "/Users/keerthivasan/NLP/outputs"
    checkpoint_dir: str = "/Users/keerthivasan/NLP/checkpoints"
    
    # Language-specific paths
    tamil_depressed: str = field(default="")
    tamil_non_depressed: str = field(default="")
    malayalam_depressed: str = field(default="")
    malayalam_non_depressed: str = field(default="")
    
    def __post_init__(self):
        self.tamil_depressed = os.path.join(self.base_dir, "Tamil/Depressed/Train_set")
        self.tamil_non_depressed = os.path.join(self.base_dir, "Tamil/Non-depressed/Train_set")
        self.malayalam_depressed = os.path.join(self.base_dir, "Malayalam/Depressed/Train_set")
        self.malayalam_non_depressed = os.path.join(self.base_dir, "Malayalam/Non_depressed/Train_set")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@dataclass
class PreprocessingConfig:
    """Audio preprocessing hyperparameters.
    
    Reference: Section 2 - Data Engineering and Signal Preprocessing
    """
    # Audio format
    sampling_rate: int = 16000  # Standard for speech backbones (IndicWav2Vec, ECAPA)
    n_channels: int = 1  # Mono
    
    # Resampling (Sinc Interpolation with Hann window)
    lowpass_filter_width: int = 6
    rolloff: float = 0.99
    
    # Normalization
    peak_normalize_target: float = 0.95  # Prevents clipping
    
    # Voice Activity Detection
    vad_frame_size_ms: int = 30  # Granularity for silence detection
    vad_aggressiveness: int = 3  # High sensitivity (1-3)
    silence_threshold_db: float = -40.0  # Relative to signal peak
    silence_padding_ms: int = 200  # Preserves prosodic onset/offset
    
    # Chunking
    chunk_duration_sec: float = 5.0  # Input size for DNNs
    chunk_overlap_train_sec: float = 1.0  # Augments training data volume
    chunk_overlap_inference_sec: float = 2.5  # Dense coverage for evaluation
    
    # Minimum audio duration (in seconds)
    min_audio_duration: float = 0.5


@dataclass
class SSLModelConfig:
    """SSL Model (IndicWav2Vec) configuration.
    
    Reference: Section 4 - Model Architecture A: Linguistic and Paralinguistic SSL
    """
    # Model backbone - use wav2vec2-base for faster training on Mac
    model_name: str = "facebook/wav2vec2-base"  # Base model for faster iteration
    fallback_model: str = "facebook/wav2vec2-base"  # Same as primary for consistency
    
    # Model architecture (wav2vec2-base dimensions)
    hidden_size: int = 768  # Transformer width
    num_hidden_layers: int = 12
    
    # Attentive Statistics Pooling
    attention_hidden_dim: int = 128
    
    # Classification head
    projection_dim: int = 256
    dropout_rate: float = 0.3
    num_classes: int = 2
    
    # Layer-wise freezing
    freeze_feature_extractor: bool = True
    freeze_epochs: int = 5  # Freeze CNN for first N epochs
    
    # Fine-tuning
    use_weighted_layer_sum: bool = True
    layer_weights_trainable: bool = True


@dataclass
class ECAPAModelConfig:
    """ECAPA-TDNN Model configuration.
    
    Reference: Section 5 - Model Architecture B: Spectral and Prosodic Modeling
    """
    # Input features (Mel-filterbank)
    n_mels: int = 80
    window_size_ms: int = 25
    hop_size_ms: int = 10
    
    # CMVN (Cepstral Mean and Variance Normalization)
    apply_cmvn: bool = True
    
    # Core backbone
    initial_channels: int = 512
    kernel_size: int = 5
    
    # SE-Res2Net blocks with progressive dilations
    se_res2net_blocks: int = 3
    dilations: Tuple[int, ...] = (2, 3, 4)
    se_reduction_ratio: int = 8
    
    # Multi-scale feature aggregation
    aggregation_channels: int = 1536
    
    # Attentive Statistics Pooling
    attention_channels: int = 128
    
    # Embedding and classification
    embedding_dim: int = 192
    num_classes: int = 2
    
    # AAM Softmax parameters
    aam_margin: float = 0.2
    aam_scale: float = 30.0


@dataclass
class TrainingConfig:
    """Training hyperparameters.
    
    Reference: Section 6 - Training Pipeline and Optimization Dynamics
    """
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4  # Max learning rate
    weight_decay: float = 2e-5  # L2 regularization
    
    # Learning rate scheduler (Cyclic Triangular)
    scheduler: str = "cyclic"
    lr_min: float = 1e-6
    warmup_ratio: float = 0.1  # 10% warmup
    cycle_step_multiplier: int = 4  # step_size = 4 * iterations_per_epoch
    
    # Batch and epochs
    batch_size: int = 16
    gradient_accumulation_steps: int = 2  # Effective batch = 32
    epochs: int = 30
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Cross-validation
    n_folds: int = 5
    cv_random_seed: int = 42
    max_class_balance_deviation: float = 0.05  # 5% deviation threshold
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class LossConfig:
    """Loss function configuration.
    
    Reference: Section 6.1 - Hyperparameters and Loss Functions
    """
    # Weighted Cross-Entropy for SSL stream
    use_class_weights: bool = True
    
    # AAM Softmax for ECAPA stream
    use_aam_softmax: bool = True
    aam_margin: float = 0.2
    aam_scale: float = 30.0
    
    # Label smoothing
    label_smoothing: float = 0.1


@dataclass
class AugmentationConfig:
    """Data augmentation configuration.
    
    Reference: Section 7 - Data Augmentation Strategy
    """
    # Waveform augmentations (online)
    # Pitch shifting
    pitch_shift_enabled: bool = True
    pitch_shift_semitones: Tuple[int, int] = (-4, 4)
    pitch_shift_prob: float = 0.5
    
    # Time stretching
    time_stretch_enabled: bool = True
    time_stretch_rate: Tuple[float, float] = (0.8, 1.25)
    time_stretch_prob: float = 0.5
    
    # Additive noise (MUSAN)
    additive_noise_enabled: bool = True
    snr_range_db: Tuple[int, int] = (5, 20)
    additive_noise_prob: float = 0.5
    noise_types: List[str] = field(default_factory=lambda: ["white", "pink", "brown"])
    
    # Room Impulse Response convolution
    rir_enabled: bool = True
    rir_prob: float = 0.3
    
    # SpecAugment for ECAPA stream
    # Frequency masking
    freq_mask_enabled: bool = True
    freq_mask_param: int = 30  # Max width in frequency bins
    freq_mask_num: int = 2  # Number of masks
    
    # Time masking
    time_mask_enabled: bool = True
    time_mask_param: int = 40  # Max width in time steps
    time_mask_num: int = 2  # Number of masks
    
    # Time warping
    time_warp_enabled: bool = True
    time_warp_window: int = 5


@dataclass
class ThresholdConfig:
    """Threshold tuning configuration.
    
    Reference: Section 8 - Threshold Tuning for Macro-F1 Optimization
    """
    # Grid search parameters
    threshold_min: float = 0.01
    threshold_max: float = 0.99
    threshold_step: float = 0.01
    
    # Default threshold (before optimization)
    default_threshold: float = 0.5


@dataclass
class EnsembleConfig:
    """Ensemble and inference configuration.
    
    Reference: Section 9 - Ensembling and Inference Pipeline
    """
    # Weighted average fusion
    fusion_method: str = "weighted_average"  # or "stacking"
    fusion_weight_ssl: float = 0.5  # Alpha for SSL stream
    fusion_weight_ecapa: float = 0.5  # 1 - Alpha for ECAPA stream
    
    # Weight optimization
    weight_search_min: float = 0.0
    weight_search_max: float = 1.0
    weight_search_step: float = 0.05
    
    # Meta-ensemble stacking
    stacking_enabled: bool = False
    meta_learner: str = "logistic_regression"  # or "mlp"
    mlp_hidden_dim: int = 64


@dataclass
class ASRConfig:
    """ASR Pipeline configuration for IndicWhisper.
    
    Reference: Multimodal NLP Upgrade - Requirement 1
    """
    # Model backbone
    model_name: str = "ai4bharat/indicwhisper-large"
    
    # Audio chunking for long files
    chunk_length_s: int = 30
    
    # Batch processing
    batch_size: int = 8
    
    # Device configuration
    device: str = "cuda"
    
    # Supported languages
    supported_languages: List[str] = field(default_factory=lambda: ["tamil", "malayalam"])
    
    # Transcription settings
    return_timestamps: bool = True
    language_detection: bool = False  # Use explicit language when known
    
    # Output settings
    transcript_output_dir: str = "transcripts"


@dataclass
class TextModelConfig:
    """MuRIL Text Model configuration.
    
    Reference: Multimodal NLP Upgrade - Requirement 4
    """
    # Model backbone
    model_name: str = "google/muril-base-cased"
    
    # Tokenization
    max_length: int = 512
    
    # Model architecture
    hidden_size: int = 768
    
    # Classification head
    dropout_rate: float = 0.1
    num_classes: int = 2
    
    # Fine-tuning strategy
    freeze_encoder: bool = False
    freeze_epochs: int = 2  # Freeze encoder for first N epochs
    
    # Training settings
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1


@dataclass
class TextPreprocessingConfig:
    """Text preprocessing configuration for Dravidian languages.
    
    Reference: Multimodal NLP Upgrade - Requirement 2
    """
    # Language setting ("ta" for Tamil, "ml" for Malayalam)
    language: str = "ta"
    
    # Normalization
    apply_normalization: bool = True
    
    # Morphological segmentation
    apply_morphological_segmentation: bool = True
    morfessor_model_path: Optional[str] = None


@dataclass
class TFIDFConfig:
    """TF-IDF Classifier configuration.
    
    Reference: Multimodal NLP Upgrade - Requirement 3
    """
    # N-gram range
    ngram_range: Tuple[int, int] = (1, 3)
    
    # Vocabulary constraints
    min_df: int = 5
    max_features: int = 5000
    
    # Classifier type
    classifier: str = "svm"  # or "logistic_regression"
    
    # SVM parameters
    svm_kernel: str = "linear"
    svm_c: float = 1.0
    class_weight: str = "balanced"  # Handle class imbalance
    random_state: int = 42
    
    # Logistic Regression parameters
    lr_c: float = 1.0
    lr_max_iter: int = 1000


@dataclass
class FusionConfig:
    """Multimodal Fusion configuration.
    
    Reference: Multimodal NLP Upgrade - Requirement 5
    """
    # Input embedding dimensions
    audio_embedding_dim: int = 192  # ECAPA-TDNN default, or 768 for Wav2Vec2
    text_embedding_dim: int = 768   # MuRIL
    
    # Fusion MLP architecture
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.3
    
    # Output
    num_classes: int = 2
    
    # Fusion strategy
    fusion_method: str = "concatenation"  # or "attention", "gated"
    
    # Fallback behavior when modality is missing
    allow_unimodal_fallback: bool = True


@dataclass
class DepressionDetectionConfig:
    """Master configuration class combining all sub-configurations."""
    paths: PathConfig = field(default_factory=PathConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ssl_model: SSLModelConfig = field(default_factory=SSLModelConfig)
    ecapa_model: ECAPAModelConfig = field(default_factory=ECAPAModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # Multimodal NLP Upgrade configurations
    asr: ASRConfig = field(default_factory=ASRConfig)
    text_model: TextModelConfig = field(default_factory=TextModelConfig)
    text_preprocessing: TextPreprocessingConfig = field(default_factory=TextPreprocessingConfig)
    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    
    # General settings
    language: Language = Language.COMBINED
    seed: int = 42
    device: str = "cuda"  # or "cpu", "mps" for Apple Silicon
    num_workers: int = 4
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.preprocessing.sampling_rate == 16000, \
            "Sampling rate must be 16kHz for pre-trained speech models"
        assert 0 < self.ensemble.fusion_weight_ssl <= 1.0, \
            "Fusion weight must be between 0 and 1"
        # Sync device across ASR config
        self.asr.device = self.device


def get_config(language: str = "combined") -> DepressionDetectionConfig:
    """Factory function to create configuration.
    
    Args:
        language: One of "tamil", "malayalam", or "combined"
        
    Returns:
        DepressionDetectionConfig instance
    """
    config = DepressionDetectionConfig()
    config.language = Language(language.lower())
    return config


# Convenience function to print config summary
def print_config_summary(config: DepressionDetectionConfig):
    """Print a summary of the configuration."""
    print("=" * 60)
    print("Depression Detection Configuration Summary")
    print("=" * 60)
    print(f"Language: {config.language.value}")
    print(f"Device: {config.device}")
    print(f"Random Seed: {config.seed}")
    print()
    print("Preprocessing:")
    print(f"  Sampling Rate: {config.preprocessing.sampling_rate} Hz")
    print(f"  Chunk Duration: {config.preprocessing.chunk_duration_sec}s")
    print(f"  VAD Aggressiveness: {config.preprocessing.vad_aggressiveness}")
    print()
    print("SSL Model (IndicWav2Vec):")
    print(f"  Backbone: {config.ssl_model.model_name}")
    print(f"  Hidden Size: {config.ssl_model.hidden_size}")
    print(f"  Dropout: {config.ssl_model.dropout_rate}")
    print()
    print("ECAPA-TDNN Model:")
    print(f"  Mel Bins: {config.ecapa_model.n_mels}")
    print(f"  Embedding Dim: {config.ecapa_model.embedding_dim}")
    print(f"  AAM Margin: {config.ecapa_model.aam_margin}")
    print()
    print("Training:")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  K-Folds: {config.training.n_folds}")
    print()
    print("ASR Pipeline (Multimodal):")
    print(f"  Model: {config.asr.model_name}")
    print(f"  Chunk Length: {config.asr.chunk_length_s}s")
    print(f"  Supported Languages: {config.asr.supported_languages}")
    print()
    print("Text Model (MuRIL):")
    print(f"  Model: {config.text_model.model_name}")
    print(f"  Max Length: {config.text_model.max_length}")
    print(f"  Hidden Size: {config.text_model.hidden_size}")
    print()
    print("TF-IDF Baseline:")
    print(f"  N-gram Range: {config.tfidf.ngram_range}")
    print(f"  Max Features: {config.tfidf.max_features}")
    print(f"  Classifier: {config.tfidf.classifier}")
    print()
    print("Multimodal Fusion:")
    print(f"  Audio Embedding Dim: {config.fusion.audio_embedding_dim}")
    print(f"  Text Embedding Dim: {config.fusion.text_embedding_dim}")
    print(f"  Fusion Hidden Dim: {config.fusion.fusion_hidden_dim}")
    print(f"  Fusion Method: {config.fusion.fusion_method}")
    print("=" * 60)


if __name__ == "__main__":
    config = get_config("combined")
    print_config_summary(config)
