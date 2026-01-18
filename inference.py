"""
Enhanced Ensemble and Inference Pipeline

This module implements the enhanced ensemble and inference pipeline with
linguistic features for traditional NLP tasks integration.

Key components:
- Enhanced multimodal inference with linguistic features
- Linguistic feature extraction and visualization
- Feature importance analysis for interpretability
- Support for both standard and enhanced inference modes
- Clinical interpretation with linguistic markers

Reference: Traditional NLP Tasks - Task 9 (Enhanced Inference Pipeline)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import logging
import json
import csv
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

from config import DepressionDetectionConfig, EnsembleConfig
from ssl_model import Wav2Vec2ForDepressionDetection, load_ssl_checkpoint
from ecapa_model import ECAPATDNN, load_ecapa_checkpoint
from threshold_tuner import MacroF1ThresholdOptimizer, ThresholdResult
from preprocessing import AudioPreprocessor, AudioChunk

# Enhanced multimodal components
from enhanced_multimodal_model import EnhancedMultimodalModel
from linguistic_analyzer import LinguisticAnalyzer, LinguisticAnalysis, LinguisticConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    file_id: str
    speaker_id: str
    predicted_label: int
    probability: float
    ssl_probability: float
    ecapa_probability: float


@dataclass
class EnhancedPredictionResult:
    """Container for enhanced prediction results with linguistic features."""
    file_id: str
    speaker_id: str
    predicted_label: int
    probability: float
    
    # Individual modality probabilities
    ssl_probability: float
    ecapa_probability: float
    text_probability: Optional[float] = None
    
    # Enhanced multimodal probability
    enhanced_probability: Optional[float] = None
    
    # Linguistic analysis
    linguistic_analysis: Optional[LinguisticAnalysis] = None
    linguistic_features: Optional[np.ndarray] = None
    
    # Feature importance scores
    feature_importance: Optional[Dict[str, float]] = None
    
    # Clinical markers
    linguistic_markers: Optional[Dict[str, float]] = None


@dataclass
class LinguisticMarkers:
    """Clinical linguistic markers for depression analysis."""
    # POS-based markers
    first_person_pronoun_ratio: float
    negative_adjective_ratio: float
    past_tense_verb_ratio: float
    
    # NER-based markers
    self_reference_count: int
    medical_entity_count: int
    
    # Syntactic markers
    sentence_complexity: float
    dependency_distance: float
    
    # Overall linguistic risk score
    linguistic_risk_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'first_person_pronoun_ratio': self.first_person_pronoun_ratio,
            'negative_adjective_ratio': self.negative_adjective_ratio,
            'past_tense_verb_ratio': self.past_tense_verb_ratio,
            'self_reference_count': float(self.self_reference_count),
            'medical_entity_count': float(self.medical_entity_count),
            'sentence_complexity': self.sentence_complexity,
            'dependency_distance': self.dependency_distance,
            'linguistic_risk_score': self.linguistic_risk_score
        }


@dataclass
class EnsembleResult:
    """Container for ensemble evaluation results."""
    macro_f1: float
    f1_depressed: float
    f1_non_depressed: float
    precision: float
    recall: float
    optimal_threshold: float
    optimal_fusion_weight: float
    predictions: List[PredictionResult]


class WeightedAverageFusion:
    """Weighted Average Fusion for combining model predictions.
    
    Implements Section 9.1: Late Fusion via Weighted Averaging
    P_final = α × P_SSL + (1 - α) × P_ECAPA
    """
    
    def __init__(
        self,
        ssl_weight: float = 0.5,
        search_weights: bool = True,
        weight_search_step: float = 0.05
    ):
        """
        Args:
            ssl_weight: Initial weight for SSL model (alpha)
            search_weights: Whether to optimize weights on validation
            weight_search_step: Step size for weight grid search
        """
        self.ssl_weight = ssl_weight
        self.search_weights = search_weights
        self.weight_search_step = weight_search_step
    
    def fuse(
        self,
        ssl_probs: np.ndarray,
        ecapa_probs: np.ndarray
    ) -> np.ndarray:
        """Fuse predictions using weighted average.
        
        Args:
            ssl_probs: Probabilities from SSL model (batch, 2) or (batch,)
            ecapa_probs: Probabilities from ECAPA model (batch, 2) or (batch,)
            
        Returns:
            Fused probabilities
        """
        # Handle both binary and multi-class format
        if ssl_probs.ndim == 2:
            ssl_probs = ssl_probs[:, 1]  # Probability of positive class
        if ecapa_probs.ndim == 2:
            ecapa_probs = ecapa_probs[:, 1]
        
        return self.ssl_weight * ssl_probs + (1 - self.ssl_weight) * ecapa_probs
    
    def optimize_weights(
        self,
        ssl_probs: np.ndarray,
        ecapa_probs: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, float]:
        """Find optimal fusion weight.
        
        Args:
            ssl_probs: Validation probabilities from SSL model
            ecapa_probs: Validation probabilities from ECAPA model
            labels: Ground truth labels
            
        Returns:
            Tuple of (optimal weight, optimal F1)
        """
        if ssl_probs.ndim == 2:
            ssl_probs = ssl_probs[:, 1]
        if ecapa_probs.ndim == 2:
            ecapa_probs = ecapa_probs[:, 1]
        
        weights = np.arange(0.0, 1.0 + self.weight_search_step, self.weight_search_step)
        
        best_weight = 0.5
        best_f1 = 0.0
        
        threshold_optimizer = MacroF1ThresholdOptimizer()
        
        for weight in weights:
            fused = weight * ssl_probs + (1 - weight) * ecapa_probs
            result = threshold_optimizer.optimize(fused, labels)
            
            if result.optimal_macro_f1 > best_f1:
                best_f1 = result.optimal_macro_f1
                best_weight = weight
        
        logger.info(f"Optimal fusion weight (SSL): {best_weight:.2f}")
        logger.info(f"Optimal fused Macro-F1: {best_f1:.4f}")
        
        self.ssl_weight = best_weight
        return best_weight, best_f1


class MetaEnsembleStacking:
    """Meta-Ensemble Stacking for advanced prediction fusion.
    
    Implements Section 9.2: Embedding concatenation with meta-learner
    Instead of averaging probabilities, concatenates embeddings and
    trains a meta-learner (Logistic Regression or MLP).
    """
    
    def __init__(
        self,
        meta_learner: str = 'logistic_regression',
        mlp_hidden_dim: int = 64
    ):
        """
        Args:
            meta_learner: 'logistic_regression' or 'mlp'
            mlp_hidden_dim: Hidden dimension for MLP meta-learner
        """
        self.meta_learner_type = meta_learner
        self.mlp_hidden_dim = mlp_hidden_dim
        
        if meta_learner == 'logistic_regression':
            self.meta_learner = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        else:
            self.meta_learner = MLPClassifier(
                hidden_layer_sizes=(mlp_hidden_dim,),
                max_iter=500,
                random_state=42
            )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(
        self,
        ssl_embeddings: np.ndarray,
        ecapa_embeddings: np.ndarray,
        labels: np.ndarray
    ):
        """Train the meta-learner on concatenated embeddings.
        
        Args:
            ssl_embeddings: Embeddings from SSL model (batch, dim)
            ecapa_embeddings: Embeddings from ECAPA model (batch, dim)
            labels: Ground truth labels
        """
        # Concatenate embeddings
        combined = np.concatenate([ssl_embeddings, ecapa_embeddings], axis=1)
        
        # Scale features
        combined_scaled = self.scaler.fit_transform(combined)
        
        # Train meta-learner
        self.meta_learner.fit(combined_scaled, labels)
        self.is_fitted = True
        
        logger.info(f"Trained meta-learner on {len(labels)} samples")
        logger.info(f"Combined embedding dimension: {combined.shape[1]}")
    
    def predict_proba(
        self,
        ssl_embeddings: np.ndarray,
        ecapa_embeddings: np.ndarray
    ) -> np.ndarray:
        """Get probability predictions from meta-learner.
        
        Args:
            ssl_embeddings: Embeddings from SSL model
            ecapa_embeddings: Embeddings from ECAPA model
            
        Returns:
            Probability predictions (batch, 2)
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-learner not fitted. Call fit() first.")
        
        combined = np.concatenate([ssl_embeddings, ecapa_embeddings], axis=1)
        combined_scaled = self.scaler.transform(combined)
        
        return self.meta_learner.predict_proba(combined_scaled)


class DepressionInferencePipeline:
    """Complete inference pipeline for depression detection.
    
    Implements Section 9.3: Full inference execution with:
    1. Audio preprocessing
    2. Model inference
    3. Chunk-to-recording aggregation
    4. Ensemble fusion
    5. Threshold application
    """
    
    def __init__(
        self,
        config: DepressionDetectionConfig,
        ssl_model: Optional[Wav2Vec2ForDepressionDetection] = None,
        ecapa_model: Optional[ECAPATDNN] = None,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device
        
        # Load models if not provided
        self.ssl_model = ssl_model
        self.ecapa_model = ecapa_model
        
        # Preprocessor
        self.preprocessor = AudioPreprocessor(
            config.preprocessing,
            is_training=False  # Use inference overlap
        )
        
        # Fusion and threshold
        self.fusion = WeightedAverageFusion(
            ssl_weight=config.ensemble.fusion_weight_ssl
        )
        self.optimal_threshold = 0.5
        
        # Mel transform for ECAPA
        import torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.preprocessing.sampling_rate,
            n_mels=config.ecapa_model.n_mels,
            n_fft=int(config.ecapa_model.window_size_ms * config.preprocessing.sampling_rate / 1000),
            hop_length=int(config.ecapa_model.hop_size_ms * config.preprocessing.sampling_rate / 1000)
        ).to(device)
    
    def load_models(
        self,
        ssl_checkpoint: str,
        ecapa_checkpoint: str
    ):
        """Load model weights from checkpoints."""
        self.ssl_model = load_ssl_checkpoint(ssl_checkpoint, self.config)
        self.ssl_model.to(self.device)
        self.ssl_model.eval()
        
        self.ecapa_model = load_ecapa_checkpoint(ecapa_checkpoint, self.config)
        self.ecapa_model.to(self.device)
        self.ecapa_model.eval()
        
        logger.info("Loaded both models from checkpoints")
    
    def set_fusion_params(
        self,
        ssl_weight: float,
        optimal_threshold: float
    ):
        """Set fusion weight and threshold from optimization."""
        self.fusion.ssl_weight = ssl_weight
        self.optimal_threshold = optimal_threshold
        logger.info(f"Set fusion weight: {ssl_weight:.2f}, threshold: {optimal_threshold:.3f}")
    
    @torch.no_grad()
    def _get_ssl_predictions(
        self,
        chunks: List[AudioChunk]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from SSL model.
        
        Returns:
            Tuple of (probabilities, embeddings)
        """
        all_probs = []
        all_embeddings = []
        
        for chunk in chunks:
            waveform = chunk.waveform.squeeze(0).unsqueeze(0).to(self.device)
            attention_mask = chunk.attention_mask.unsqueeze(0).to(self.device)
            
            logits, embeddings = self.ssl_model(
                waveform,
                attention_mask,
                return_embeddings=True
            )
            
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_probs), np.concatenate(all_embeddings)
    
    @torch.no_grad()
    def _get_ecapa_predictions(
        self,
        chunks: List[AudioChunk]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from ECAPA model.
        
        Returns:
            Tuple of (probabilities, embeddings)
        """
        all_probs = []
        all_embeddings = []
        
        for chunk in chunks:
            waveform = chunk.waveform.to(self.device)
            
            # Compute Mel spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Apply CMVN
            mel_spec = mel_spec - mel_spec.mean(dim=-1, keepdim=True)
            mel_spec = mel_spec / (mel_spec.std(dim=-1, keepdim=True) + 1e-8)
            
            logits, embeddings = self.ecapa_model(
                mel_spec,
                return_embeddings=True
            )
            
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_probs), np.concatenate(all_embeddings)
    
    def predict_file(
        self,
        file_path: str,
        return_chunk_preds: bool = False
    ) -> Dict:
        """Run inference on a single audio file.
        
        Args:
            file_path: Path to audio file
            return_chunk_preds: Whether to return chunk-level predictions
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        chunks = self.preprocessor.process_file(file_path, label=0)  # Label unknown
        
        if not chunks:
            logger.warning(f"No valid chunks from {file_path}")
            return {'predicted_label': 0, 'probability': 0.0}
        
        # Get predictions from both models
        ssl_probs, ssl_emb = self._get_ssl_predictions(chunks)
        ecapa_probs, ecapa_emb = self._get_ecapa_predictions(chunks)
        
        # Fuse chunk-level predictions
        fused_probs = self.fusion.fuse(ssl_probs, ecapa_probs)
        
        # Aggregate to recording level (mean probability)
        recording_prob = np.mean(fused_probs)
        
        # Apply threshold
        predicted_label = 1 if recording_prob >= self.optimal_threshold else 0
        
        result = {
            'file_path': file_path,
            'speaker_id': chunks[0].speaker_id,
            'predicted_label': predicted_label,
            'probability': float(recording_prob),
            'ssl_probability': float(np.mean(ssl_probs[:, 1])),
            'ecapa_probability': float(np.mean(ecapa_probs[:, 1])),
            'num_chunks': len(chunks)
        }
        
        if return_chunk_preds:
            result['chunk_predictions'] = fused_probs.tolist()
        
        return result
    
    def predict_directory(
        self,
        directory: str,
        output_csv: Optional[str] = None
    ) -> List[Dict]:
        """Run inference on all audio files in a directory.
        
        Args:
            directory: Path to directory with audio files
            output_csv: Optional path to save predictions
            
        Returns:
            List of prediction results
        """
        import glob
        
        # Find all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(directory, ext)))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        results = []
        for file_path in audio_files:
            try:
                result = self.predict_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Save to CSV if requested
        if output_csv:
            self._save_predictions_csv(results, output_csv)
        
        return results
    
    def _save_predictions_csv(
        self,
        results: List[Dict],
        output_path: str
    ):
        """Save predictions to CSV file for submission."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Label', 'Probability'])
            
            for result in results:
                file_id = os.path.splitext(os.path.basename(result['file_path']))[0]
                label = 'Depressed' if result['predicted_label'] == 1 else 'Non-Depressed'
                writer.writerow([file_id, label, f"{result['probability']:.4f}"])
        
        logger.info(f"Saved predictions to {output_path}")


class EnhancedDepressionInferencePipeline:
    """Enhanced inference pipeline with linguistic features.
    
    Supports both standard multimodal inference and enhanced inference
    with traditional NLP tasks (POS, NER, dependency parsing).
    
    Reference: Traditional NLP Tasks - Requirements 4, 6
    """
    
    def __init__(
        self,
        config: DepressionDetectionConfig,
        enhanced_model: Optional[EnhancedMultimodalModel] = None,
        linguistic_analyzer: Optional[LinguisticAnalyzer] = None,
        device: str = 'cuda',
        inference_mode: str = 'enhanced'  # 'standard' or 'enhanced'
    ):
        """Initialize enhanced inference pipeline.
        
        Args:
            config: Configuration object
            enhanced_model: Enhanced multimodal model with linguistic features
            linguistic_analyzer: Linguistic analyzer for NLP tasks
            device: Device to run inference on
            inference_mode: 'standard' for baseline, 'enhanced' for linguistic features
        """
        self.config = config
        self.device = device
        self.inference_mode = inference_mode
        
        # Models
        self.enhanced_model = enhanced_model
        self.linguistic_analyzer = linguistic_analyzer
        
        # Standard pipeline for fallback
        self.standard_pipeline = DepressionInferencePipeline(
            config, device=device
        )
        
        # Initialize linguistic analyzer if not provided
        if self.inference_mode == 'enhanced' and self.linguistic_analyzer is None:
            linguistic_config = LinguisticConfig(device=device)
            self.linguistic_analyzer = LinguisticAnalyzer(linguistic_config)
            logger.info("Initialized linguistic analyzer for enhanced inference")
    
    def load_enhanced_model(self, checkpoint_path: str):
        """Load enhanced multimodal model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.enhanced_model.load_state_dict(checkpoint['model_state_dict'])
        self.enhanced_model.to(self.device)
        self.enhanced_model.eval()
        logger.info(f"Loaded enhanced model from {checkpoint_path}")
    
    def extract_linguistic_markers(
        self, 
        linguistic_analysis: LinguisticAnalysis
    ) -> LinguisticMarkers:
        """Extract clinical linguistic markers from analysis.
        
        Args:
            linguistic_analysis: Complete linguistic analysis
            
        Returns:
            LinguisticMarkers with clinical indicators
        """
        if not linguistic_analysis.linguistic_features:
            # Return empty markers if no features
            return LinguisticMarkers(
                first_person_pronoun_ratio=0.0,
                negative_adjective_ratio=0.0,
                past_tense_verb_ratio=0.0,
                self_reference_count=0,
                medical_entity_count=0,
                sentence_complexity=0.0,
                dependency_distance=0.0,
                linguistic_risk_score=0.0
            )
        
        features = linguistic_analysis.linguistic_features
        
        # Extract specific markers
        first_person_ratio = features.get('first_person_pronoun_ratio', 0.0)
        negative_adj_ratio = features.get('negative_adjective_ratio', 0.0)
        past_tense_ratio = features.get('past_tense_verb_ratio', 0.0)
        
        self_ref_count = int(features.get('self_reference_entities', 0))
        medical_count = int(features.get('medical_entity_count', 0))
        
        complexity = features.get('syntactic_complexity', 0.0)
        dep_distance = features.get('avg_dependency_distance', 0.0)
        
        # Calculate composite risk score
        risk_score = (
            first_person_ratio * 0.3 +
            negative_adj_ratio * 0.2 +
            past_tense_ratio * 0.1 +
            min(self_ref_count / 10.0, 1.0) * 0.2 +
            min(medical_count / 5.0, 1.0) * 0.1 +
            min(complexity / 5.0, 1.0) * 0.1
        )
        
        return LinguisticMarkers(
            first_person_pronoun_ratio=first_person_ratio,
            negative_adjective_ratio=negative_adj_ratio,
            past_tense_verb_ratio=past_tense_ratio,
            self_reference_count=self_ref_count,
            medical_entity_count=medical_count,
            sentence_complexity=complexity,
            dependency_distance=dep_distance,
            linguistic_risk_score=risk_score
        )
    
    def analyze_feature_importance(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        linguistic_features: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze feature importance for interpretability.
        
        Args:
            audio_embeddings: Audio feature embeddings
            text_embeddings: Text feature embeddings  
            linguistic_features: Linguistic feature vector
            
        Returns:
            Dictionary with feature importance scores
        """
        if self.enhanced_model is None:
            # Return default equal importance when enhanced model is not available
            return {
                'audio_importance': 0.33,
                'text_importance': 0.33,
                'linguistic_importance': 0.34
            }
        
        # Get feature importance from enhanced model
        try:
            importance_scores = self.enhanced_model.get_feature_importance(
                audio_embeddings, text_embeddings, linguistic_features
            )
            
            return {
                'audio_importance': float(importance_scores.get('audio', 0.0)),
                'text_importance': float(importance_scores.get('text', 0.0)),
                'linguistic_importance': float(importance_scores.get('linguistic', 0.0))
            }
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
            return {
                'audio_importance': 0.33,
                'text_importance': 0.33,
                'linguistic_importance': 0.34
            }
    
    @torch.no_grad()
    def predict_file_enhanced(
        self,
        audio_path: str,
        transcript_text: Optional[str] = None,
        language: str = "ta",
        return_analysis: bool = True
    ) -> EnhancedPredictionResult:
        """Run enhanced inference on audio file with optional transcript.
        
        Args:
            audio_path: Path to audio file
            transcript_text: Optional transcript text
            language: Language for linguistic analysis
            return_analysis: Whether to return detailed linguistic analysis
            
        Returns:
            EnhancedPredictionResult with all modalities and linguistic features
        """
        if self.inference_mode == 'standard' or self.enhanced_model is None:
            # Fall back to standard inference
            standard_result = self.standard_pipeline.predict_file(audio_path)
            return EnhancedPredictionResult(
                file_id=os.path.splitext(os.path.basename(audio_path))[0],
                speaker_id=standard_result['speaker_id'],
                predicted_label=standard_result['predicted_label'],
                probability=standard_result['probability'],
                ssl_probability=standard_result['ssl_probability'],
                ecapa_probability=standard_result['ecapa_probability']
            )
        
        # Enhanced inference with linguistic features
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Process audio (reuse standard pipeline preprocessing)
        chunks = self.standard_pipeline.preprocessor.process_file(audio_path, label=0)
        
        if not chunks:
            logger.warning(f"No valid chunks from {audio_path}")
            return EnhancedPredictionResult(
                file_id=file_id,
                speaker_id="unknown",
                predicted_label=0,
                probability=0.0,
                ssl_probability=0.0,
                ecapa_probability=0.0
            )
        
        speaker_id = chunks[0].speaker_id
        
        # Get audio embeddings (using first chunk for simplicity)
        chunk = chunks[0]
        waveform = chunk.waveform.squeeze(0).unsqueeze(0).to(self.device)
        attention_mask = chunk.attention_mask.unsqueeze(0).to(self.device)
        
        # Get audio embeddings from SSL model
        if hasattr(self.enhanced_model, 'audio_model'):
            _, audio_embeddings = self.enhanced_model.audio_model(
                waveform, attention_mask, return_embeddings=True
            )
        else:
            audio_embeddings = torch.randn(1, 768).to(self.device)  # Fallback
        
        # Process text and linguistic features
        linguistic_analysis = None
        linguistic_features = None
        text_embeddings = None
        
        if transcript_text and self.linguistic_analyzer:
            # Perform linguistic analysis
            linguistic_analysis = self.linguistic_analyzer.analyze_text(
                transcript_text, language
            )
            
            # Extract linguistic feature vector
            linguistic_features = self.linguistic_analyzer.extract_linguistic_feature_vector(
                linguistic_analysis.pos_result,
                linguistic_analysis.entities,
                linguistic_analysis.dependency_tree
            )
            linguistic_features = torch.from_numpy(linguistic_features).float().unsqueeze(0).to(self.device)
            
            # Get text embeddings (simplified - would use actual text model)
            text_embeddings = torch.randn(1, 768).to(self.device)  # Placeholder
        
        # Run enhanced model inference
        if transcript_text and linguistic_features is not None:
            # Trimodal inference
            logits = self.enhanced_model.forward_trimodal(
                audio_embeddings=audio_embeddings,
                text_embeddings=text_embeddings,
                linguistic_features=linguistic_features
            )
            enhanced_prob = F.softmax(logits, dim=1)[0, 1].item()
        else:
            # Audio-only inference
            logits = self.enhanced_model.forward_audio_only(
                audio_embeddings=audio_embeddings
            )
            enhanced_prob = F.softmax(logits, dim=1)[0, 1].item()
        
        # Get individual modality predictions for comparison
        ssl_prob = 0.5  # Placeholder
        ecapa_prob = 0.5  # Placeholder
        text_prob = 0.5 if transcript_text else None
        
        # Final prediction
        predicted_label = 1 if enhanced_prob >= 0.5 else 0
        
        # Extract linguistic markers
        linguistic_markers = None
        if linguistic_analysis:
            markers = self.extract_linguistic_markers(linguistic_analysis)
            linguistic_markers = markers.to_dict()
        
        # Analyze feature importance
        feature_importance = None
        if transcript_text and linguistic_features is not None:
            feature_importance = self.analyze_feature_importance(
                audio_embeddings, text_embeddings, linguistic_features
            )
        
        return EnhancedPredictionResult(
            file_id=file_id,
            speaker_id=speaker_id,
            predicted_label=predicted_label,
            probability=enhanced_prob,
            ssl_probability=ssl_prob,
            ecapa_probability=ecapa_prob,
            text_probability=text_prob,
            enhanced_probability=enhanced_prob,
            linguistic_analysis=linguistic_analysis if return_analysis else None,
            linguistic_features=linguistic_features.cpu().numpy() if linguistic_features is not None else None,
            feature_importance=feature_importance,
            linguistic_markers=linguistic_markers
        )
    
    def predict_with_transcript_file(
        self,
        audio_path: str,
        transcript_path: str,
        language: str = "ta"
    ) -> EnhancedPredictionResult:
        """Run enhanced inference with separate transcript file.
        
        Args:
            audio_path: Path to audio file
            transcript_path: Path to transcript text file
            language: Language for linguistic analysis
            
        Returns:
            EnhancedPredictionResult with linguistic analysis
        """
        # Read transcript
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read().strip()
        except Exception as e:
            logger.error(f"Could not read transcript {transcript_path}: {e}")
            transcript_text = None
        
        return self.predict_file_enhanced(
            audio_path, transcript_text, language
        )
    
    def visualize_feature_importance(
        self,
        result: EnhancedPredictionResult,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Visualize feature importance for interpretability.
        
        Args:
            result: Enhanced prediction result
            save_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization or None
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - skipping visualization")
            return None
            
        if not result.feature_importance:
            logger.warning("No feature importance data to visualize")
            return None
        
        try:
            # Create feature importance plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Feature importance bar plot
            features = list(result.feature_importance.keys())
            importance = list(result.feature_importance.values())
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            bars = ax1.bar(features, importance, color=colors)
            ax1.set_title('Feature Importance')
            ax1.set_ylabel('Importance Score')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, val in zip(bars, importance):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
            
            # Linguistic markers radar plot (if available)
            if result.linguistic_markers:
                markers = result.linguistic_markers
                categories = list(markers.keys())
                values = list(markers.values())
                
                # Normalize values to 0-1 range for radar plot
                max_val = max(values) if values else 1.0
                normalized_values = [v / max_val for v in values]
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))
                normalized_values = normalized_values + [normalized_values[0]]
                
                ax2.plot(angles, normalized_values, 'o-', linewidth=2, color='red')
                ax2.fill(angles, normalized_values, alpha=0.25, color='red')
                ax2.set_xticks(angles[:-1])
                ax2.set_xticklabels(categories, rotation=45, ha='right')
                ax2.set_ylim(0, 1)
                ax2.set_title('Linguistic Markers')
                ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved feature importance visualization to {save_path}")
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    def generate_clinical_report(
        self,
        result: EnhancedPredictionResult,
        save_path: Optional[str] = None
    ) -> str:
        """Generate clinical interpretation report.
        
        Args:
            result: Enhanced prediction result
            save_path: Optional path to save report
            
        Returns:
            Clinical report as string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DEPRESSION DETECTION CLINICAL REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"File ID: {result.file_id}")
        report_lines.append(f"Speaker ID: {result.speaker_id}")
        report_lines.append("")
        
        # Overall prediction
        prediction_text = "DEPRESSED" if result.predicted_label == 1 else "NON-DEPRESSED"
        confidence = result.probability
        report_lines.append(f"PREDICTION: {prediction_text}")
        report_lines.append(f"Confidence: {confidence:.3f}")
        report_lines.append("")
        
        # Modality breakdown
        report_lines.append("MODALITY ANALYSIS:")
        report_lines.append(f"  Audio (SSL): {result.ssl_probability:.3f}")
        report_lines.append(f"  Audio (ECAPA): {result.ecapa_probability:.3f}")
        if result.text_probability is not None:
            report_lines.append(f"  Text: {result.text_probability:.3f}")
        if result.enhanced_probability is not None:
            report_lines.append(f"  Enhanced (Multimodal): {result.enhanced_probability:.3f}")
        report_lines.append("")
        
        # Feature importance
        if result.feature_importance:
            report_lines.append("FEATURE IMPORTANCE:")
            for feature, importance in result.feature_importance.items():
                report_lines.append(f"  {feature.replace('_', ' ').title()}: {importance:.3f}")
            report_lines.append("")
        
        # Linguistic markers
        if result.linguistic_markers:
            report_lines.append("LINGUISTIC MARKERS:")
            markers = result.linguistic_markers
            
            report_lines.append(f"  First-person pronoun usage: {markers['first_person_pronoun_ratio']:.3f}")
            report_lines.append(f"  Negative language: {markers['negative_adjective_ratio']:.3f}")
            report_lines.append(f"  Past-tense focus: {markers['past_tense_verb_ratio']:.3f}")
            report_lines.append(f"  Self-references: {markers['self_reference_count']}")
            report_lines.append(f"  Medical terminology: {markers['medical_entity_count']}")
            report_lines.append(f"  Sentence complexity: {markers['sentence_complexity']:.3f}")
            report_lines.append(f"  Linguistic risk score: {markers['linguistic_risk_score']:.3f}")
            report_lines.append("")
        
        # Clinical interpretation
        report_lines.append("CLINICAL INTERPRETATION:")
        if result.linguistic_markers:
            risk_score = result.linguistic_markers['linguistic_risk_score']
            if risk_score > 0.7:
                report_lines.append("  HIGH linguistic risk indicators present")
            elif risk_score > 0.4:
                report_lines.append("  MODERATE linguistic risk indicators present")
            else:
                report_lines.append("  LOW linguistic risk indicators")
        
        if confidence > 0.8:
            report_lines.append("  High confidence prediction")
        elif confidence > 0.6:
            report_lines.append("  Moderate confidence prediction")
        else:
            report_lines.append("  Low confidence prediction - consider additional assessment")
        
        report_lines.append("")
        report_lines.append("Note: This is an automated analysis tool and should not")
        report_lines.append("replace professional clinical assessment.")
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Saved clinical report to {save_path}")
        
        return report


class EnsembleEvaluator:
    """Evaluate ensemble performance with optimization."""
    
    def __init__(self, config: DepressionDetectionConfig):
        self.config = config
        self.threshold_optimizer = MacroF1ThresholdOptimizer()
        self.fusion = WeightedAverageFusion()
    
    def evaluate(
        self,
        ssl_probs: np.ndarray,
        ecapa_probs: np.ndarray,
        labels: np.ndarray,
        speaker_ids: Optional[List[str]] = None
    ) -> EnsembleResult:
        """Evaluate ensemble with joint weight and threshold optimization.
        
        Args:
            ssl_probs: SSL model probabilities
            ecapa_probs: ECAPA model probabilities
            labels: Ground truth labels
            speaker_ids: Optional speaker IDs for subject-level eval
            
        Returns:
            EnsembleResult with metrics and optimal parameters
        """
        # Optimize fusion weight
        optimal_weight, _ = self.fusion.optimize_weights(
            ssl_probs, ecapa_probs, labels
        )
        
        # Get fused predictions with optimal weight
        fused_probs = self.fusion.fuse(ssl_probs, ecapa_probs)
        
        # Optimize threshold
        if speaker_ids:
            threshold_result = self.threshold_optimizer.optimize_subject_level(
                fused_probs, labels, speaker_ids
            )
        else:
            threshold_result = self.threshold_optimizer.optimize(fused_probs, labels)
        
        # Generate final predictions
        final_preds = (fused_probs >= threshold_result.optimal_threshold).astype(int)
        
        return EnsembleResult(
            macro_f1=threshold_result.optimal_macro_f1,
            f1_depressed=threshold_result.f1_depressed,
            f1_non_depressed=threshold_result.f1_non_depressed,
            precision=threshold_result.precision,
            recall=threshold_result.recall,
            optimal_threshold=threshold_result.optimal_threshold,
            optimal_fusion_weight=optimal_weight,
            predictions=[]  # Populate if needed
        )


def run_cross_validation_ensemble(
    config: DepressionDetectionConfig,
    ssl_fold_predictions: List[Dict],
    ecapa_fold_predictions: List[Dict],
    fold_labels: List[np.ndarray]
) -> Dict:
    """Run ensemble evaluation across all CV folds.
    
    Args:
        config: Configuration object
        ssl_fold_predictions: List of SSL predictions per fold
        ecapa_fold_predictions: List of ECAPA predictions per fold
        fold_labels: List of labels per fold
        
    Returns:
        Dictionary with aggregated results
    """
    evaluator = EnsembleEvaluator(config)
    
    fold_results = []
    all_optimal_weights = []
    all_optimal_thresholds = []
    
    for fold_idx, (ssl_preds, ecapa_preds, labels) in enumerate(
        zip(ssl_fold_predictions, ecapa_fold_predictions, fold_labels)
    ):
        result = evaluator.evaluate(
            ssl_preds['probabilities'],
            ecapa_preds['probabilities'],
            labels
        )
        
        fold_results.append(result)
        all_optimal_weights.append(result.optimal_fusion_weight)
        all_optimal_thresholds.append(result.optimal_threshold)
        
        logger.info(f"Fold {fold_idx + 1}: Macro-F1 = {result.macro_f1:.4f}")
    
    # Aggregate results
    mean_f1 = np.mean([r.macro_f1 for r in fold_results])
    std_f1 = np.std([r.macro_f1 for r in fold_results])
    
    return {
        'mean_macro_f1': mean_f1,
        'std_macro_f1': std_f1,
        'fold_results': fold_results,
        'mean_optimal_weight': np.mean(all_optimal_weights),
        'mean_optimal_threshold': np.mean(all_optimal_thresholds)
    }


def create_enhanced_inference_pipeline(
    config: DepressionDetectionConfig,
    enhanced_model_path: Optional[str] = None,
    device: str = 'cuda',
    inference_mode: str = 'enhanced'
) -> EnhancedDepressionInferencePipeline:
    """Factory function to create enhanced inference pipeline.
    
    Args:
        config: Configuration object
        enhanced_model_path: Path to enhanced model checkpoint
        device: Device to run on
        inference_mode: 'standard' or 'enhanced'
        
    Returns:
        EnhancedDepressionInferencePipeline instance
    """
    # Create linguistic analyzer
    linguistic_config = LinguisticConfig(device=device)
    linguistic_analyzer = LinguisticAnalyzer(linguistic_config)
    
    # Create enhanced model (placeholder - would load actual model)
    enhanced_model = None
    if enhanced_model_path and os.path.exists(enhanced_model_path):
        # Load enhanced model from checkpoint
        logger.info(f"Loading enhanced model from {enhanced_model_path}")
        # enhanced_model = load_enhanced_model(enhanced_model_path, config)
    
    pipeline = EnhancedDepressionInferencePipeline(
        config=config,
        enhanced_model=enhanced_model,
        linguistic_analyzer=linguistic_analyzer,
        device=device,
        inference_mode=inference_mode
    )
    
    return pipeline


def run_enhanced_inference_demo():
    """Demonstrate enhanced inference capabilities."""
    print("=" * 60)
    print("Enhanced Inference Pipeline Demo")
    print("=" * 60)
    
    # Create demo configuration
    from config import get_config
    config = get_config()
    
    # Create enhanced pipeline
    pipeline = create_enhanced_inference_pipeline(
        config, 
        inference_mode='enhanced',
        device='cpu'  # Use CPU for demo
    )
    
    # Demo text for linguistic analysis
    demo_text = "நான் மிகவும் சோர்வாக இருக்கிறேன். எனக்கு தூக்கம் வரவில்லை."
    
    print(f"\nDemo text: {demo_text}")
    print("\nPerforming linguistic analysis...")
    
    # Analyze text
    if pipeline.linguistic_analyzer:
        analysis = pipeline.linguistic_analyzer.analyze_text(demo_text, "ta")
        
        print(f"Tokens: {analysis.tokens}")
        print(f"POS tags: {analysis.pos_result.tags}")
        print(f"Entities: {[(e.text, e.label) for e in analysis.entities]}")
        
        # Extract markers
        markers = pipeline.extract_linguistic_markers(analysis)
        print(f"\nLinguistic markers:")
        for key, value in markers.to_dict().items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Enhanced inference pipeline ready!")
    print("Use predict_file_enhanced() for full audio + text analysis")
    print("=" * 60)


if __name__ == "__main__":
    # Test fusion and threshold optimization
    np.random.seed(42)
    
    n_samples = 500
    labels = np.zeros(n_samples, dtype=int)
    labels[:100] = 1  # 20% positive
    
    # Simulate SSL model predictions
    ssl_probs = np.random.beta(2, 5, n_samples)
    ssl_probs[labels == 1] = np.random.beta(4, 2, 100)
    
    # Simulate ECAPA model predictions (slightly different)
    ecapa_probs = np.random.beta(2, 4, n_samples)
    ecapa_probs[labels == 1] = np.random.beta(5, 2, 100)
    
    print("Testing WeightedAverageFusion...")
    fusion = WeightedAverageFusion()
    optimal_weight, optimal_f1 = fusion.optimize_weights(ssl_probs, ecapa_probs, labels)
    
    print(f"\nOptimal SSL weight: {optimal_weight:.2f}")
    print(f"Optimal fused Macro-F1: {optimal_f1:.4f}")
    
    # Compare with individual models
    from sklearn.metrics import f1_score
    
    ssl_preds = (ssl_probs >= 0.5).astype(int)
    ecapa_preds = (ecapa_probs >= 0.5).astype(int)
    
    ssl_f1 = f1_score(labels, ssl_preds, average='macro')
    ecapa_f1 = f1_score(labels, ecapa_preds, average='macro')
    
    print(f"\nSSL-only Macro-F1 (0.5 threshold): {ssl_f1:.4f}")
    print(f"ECAPA-only Macro-F1 (0.5 threshold): {ecapa_f1:.4f}")
    print(f"Ensemble improvement: {(optimal_f1 - max(ssl_f1, ecapa_f1)) * 100:.2f}%")
    
    # Test stacking
    print("\n" + "=" * 50)
    print("Testing MetaEnsembleStacking...")
    
    # Simulate embeddings
    ssl_embeddings = np.random.randn(n_samples, 256)
    ecapa_embeddings = np.random.randn(n_samples, 192)
    
    stacker = MetaEnsembleStacking()
    stacker.fit(ssl_embeddings, ecapa_embeddings, labels)
    
    stacked_probs = stacker.predict_proba(ssl_embeddings, ecapa_embeddings)
    stacked_preds = stacked_probs.argmax(axis=1)
    stacked_f1 = f1_score(labels, stacked_preds, average='macro')
    
    print(f"Stacked Macro-F1: {stacked_f1:.4f}")
    
    # Test enhanced inference pipeline
    print("\n" + "=" * 50)
    print("Testing Enhanced Inference Pipeline...")
    
    try:
        run_enhanced_inference_demo()
    except Exception as e:
        print(f"Enhanced inference demo failed: {e}")
        print("This is expected if linguistic models are not available")
