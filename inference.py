"""
Ensemble and Inference Pipeline

This module implements the ensemble and inference pipeline as described in
Section 9: Ensembling and Inference Pipeline

Key components:
- Weighted Average Fusion (Late Fusion)
- Meta-Ensemble Stacking
- Full inference pipeline with aggregation
- Submission file generation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import logging
import json
import csv

from config import DepressionDetectionConfig, EnsembleConfig
from ssl_model import Wav2Vec2ForDepressionDetection, load_ssl_checkpoint
from ecapa_model import ECAPATDNN, load_ecapa_checkpoint
from threshold_tuner import MacroF1ThresholdOptimizer, ThresholdResult
from preprocessing import AudioPreprocessor, AudioChunk

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
