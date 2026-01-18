"""
Multimodal Trainer for Depression Detection

This module implements the training pipeline for multimodal depression detection,
combining audio and text modalities.

Key components:
- Joint audio-text batch handling
- Weighted loss for class imbalance
- Comprehensive evaluation metrics (Macro-F1, per-class F1, precision, recall)
- Confusion matrix generation
- Support for unimodal fallback during training

Reference: Multimodal NLP Upgrade - Requirements 4.3, 5.4, 7.1, 7.2, 7.5
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from config import DepressionDetectionConfig, FusionConfig
from multimodal_model import MultimodalDepressionModel, create_multimodal_model
from trainer import (
    ModelTrainer,
    WeightedCrossEntropyLoss,
    EarlyStopping,
    TrainingMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalTrainingMetrics(TrainingMetrics):
    """Extended metrics for multimodal training."""
    confusion_matrix: Optional[List[List[int]]] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert confusion matrix to list if it's a numpy array
        if isinstance(d.get('confusion_matrix'), np.ndarray):
            d['confusion_matrix'] = d['confusion_matrix'].tolist()
        return d


@dataclass
class EvaluationResult:
    """Container for comprehensive evaluation results.
    
    Implements Requirements 7.1, 7.2, 7.5 for evaluation metrics.
    """
    # Primary metric (Requirement 7.1)
    macro_f1: float
    
    # Per-class metrics (Requirement 7.2)
    f1_depressed: float
    f1_non_depressed: float
    precision_depressed: float
    precision_non_depressed: float
    recall_depressed: float
    recall_non_depressed: float
    
    # Aggregate metrics
    macro_precision: float
    macro_recall: float
    
    # Confusion matrix (Requirement 7.5)
    confusion_matrix: np.ndarray
    
    # Loss
    loss: float
    
    # Raw predictions for further analysis
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'macro_f1': self.macro_f1,
            'f1_depressed': self.f1_depressed,
            'f1_non_depressed': self.f1_non_depressed,
            'precision_depressed': self.precision_depressed,
            'precision_non_depressed': self.precision_non_depressed,
            'recall_depressed': self.recall_depressed,
            'recall_non_depressed': self.recall_non_depressed,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'loss': self.loss
        }
    
    def print_summary(self):
        """Print a formatted summary of evaluation results."""
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Macro-F1 (Primary Metric): {self.macro_f1:.4f}")
        print(f"Macro Precision: {self.macro_precision:.4f}")
        print(f"Macro Recall: {self.macro_recall:.4f}")
        print()
        print("Per-Class Metrics:")
        print(f"  Depressed:")
        print(f"    F1: {self.f1_depressed:.4f}")
        print(f"    Precision: {self.precision_depressed:.4f}")
        print(f"    Recall: {self.recall_depressed:.4f}")
        print(f"  Non-Depressed:")
        print(f"    F1: {self.f1_non_depressed:.4f}")
        print(f"    Precision: {self.precision_non_depressed:.4f}")
        print(f"    Recall: {self.recall_non_depressed:.4f}")
        print()
        print("Confusion Matrix:")
        print(f"  [[TN={self.confusion_matrix[0,0]}, FP={self.confusion_matrix[0,1]}],")
        print(f"   [FN={self.confusion_matrix[1,0]}, TP={self.confusion_matrix[1,1]}]]")
        print("=" * 50)


def compute_evaluation_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    loss: float = 0.0
) -> EvaluationResult:
    """Compute comprehensive evaluation metrics.
    
    Implements Requirements 7.1, 7.2, 7.5 for evaluation.
    
    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        probabilities: Prediction probabilities (optional)
        loss: Validation loss
        
    Returns:
        EvaluationResult with all metrics
    """
    # Macro-F1 (Requirement 7.1 - primary metric)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Per-class F1 (Requirement 7.2)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    f1_non_depressed = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
    f1_depressed = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    
    # Per-class precision (Requirement 7.2)
    precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
    precision_non_depressed = precision_per_class[0] if len(precision_per_class) > 0 else 0.0
    precision_depressed = precision_per_class[1] if len(precision_per_class) > 1 else 0.0
    
    # Per-class recall (Requirement 7.2)
    recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
    recall_non_depressed = recall_per_class[0] if len(recall_per_class) > 0 else 0.0
    recall_depressed = recall_per_class[1] if len(recall_per_class) > 1 else 0.0
    
    # Macro precision and recall
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    
    # Confusion matrix (Requirement 7.5)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    
    return EvaluationResult(
        macro_f1=macro_f1,
        f1_depressed=f1_depressed,
        f1_non_depressed=f1_non_depressed,
        precision_depressed=precision_depressed,
        precision_non_depressed=precision_non_depressed,
        recall_depressed=recall_depressed,
        recall_non_depressed=recall_non_depressed,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        confusion_matrix=cm,
        loss=loss,
        predictions=predictions,
        labels=labels,
        probabilities=probabilities if probabilities is not None else np.array([])
    )


class MultimodalTrainer(ModelTrainer):
    """Trainer for multimodal depression detection models.
    
    Extends ModelTrainer to handle joint audio-text batches and
    support weighted loss for class imbalance.
    
    Implements:
    - Requirement 4.3: Weighted cross-entropy loss for class imbalance
    - Requirement 5.4: Dropout in fusion layer during training
    - Requirements 7.1, 7.2, 7.5: Comprehensive evaluation metrics
    """
    
    def __init__(
        self,
        model: MultimodalDepressionModel,
        config: DepressionDetectionConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        class_weights: Optional[torch.Tensor] = None
    ):
        """Initialize multimodal trainer.
        
        Args:
            model: MultimodalDepressionModel instance
            config: DepressionDetectionConfig
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            class_weights: Optional class weights for imbalanced data
        """
        self.class_weights = class_weights
        super().__init__(model, config, train_loader, val_loader, device)
        
        # Override loss function with class weights if provided
        if class_weights is not None:
            self.criterion = WeightedCrossEntropyLoss(
                class_weights=class_weights,
                label_smoothing=config.loss.label_smoothing
            )
            logger.info(f"Using weighted loss with weights: {class_weights}")
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare multimodal batch for model input.
        
        Handles both audio and text inputs from the batch.
        
        Args:
            batch: Dictionary containing audio and text data
            
        Returns:
            Dictionary with prepared inputs
        """
        prepared = {}
        
        # Audio input (waveform or mel spectrogram)
        if 'waveform' in batch:
            # For SSL models - waveform input
            waveform = batch['waveform']
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)  # (B, 1, T) -> (B, T)
            prepared['audio_input'] = waveform.to(self.device)
            
            if 'attention_mask' in batch:
                prepared['audio_attention_mask'] = batch['attention_mask'].to(self.device)
        
        elif 'mel_spectrogram' in batch:
            # For ECAPA models - mel spectrogram input
            prepared['audio_input'] = batch['mel_spectrogram'].to(self.device)
        
        # Text input
        if 'input_ids' in batch:
            prepared['text_input_ids'] = batch['input_ids'].to(self.device)
            prepared['text_attention_mask'] = batch['text_attention_mask'].to(self.device)
        
        return prepared
    
    def _forward(self, inputs: Dict) -> torch.Tensor:
        """Run forward pass through multimodal model.
        
        Handles different input combinations:
        - Full multimodal (audio + text)
        - Audio-only fallback
        - Text-only fallback
        
        Args:
            inputs: Prepared input dictionary
            
        Returns:
            Model logits
        """
        has_audio = 'audio_input' in inputs
        has_text = 'text_input_ids' in inputs
        
        if has_audio and has_text:
            # Full multimodal forward
            return self.model(
                audio_input=inputs['audio_input'],
                text_input_ids=inputs['text_input_ids'],
                text_attention_mask=inputs['text_attention_mask'],
                audio_attention_mask=inputs.get('audio_attention_mask')
            )
        elif has_audio:
            # Audio-only fallback
            return self.model.forward_audio_only(
                audio_input=inputs['audio_input'],
                audio_attention_mask=inputs.get('audio_attention_mask')
            )
        elif has_text:
            # Text-only fallback
            return self.model.forward_text_only(
                text_input_ids=inputs['text_input_ids'],
                text_attention_mask=inputs['text_attention_mask']
            )
        else:
            raise ValueError("Batch must contain either audio or text input")

    
    @torch.no_grad()
    def evaluate(self) -> EvaluationResult:
        """Run comprehensive evaluation on validation set.
        
        Implements Requirements 7.1, 7.2, 7.5 for evaluation metrics.
        
        Returns:
            EvaluationResult with all metrics including confusion matrix
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(
            self.val_loader,
            desc="Evaluating",
            leave=True,
            file=sys.stdout
        )
        
        for batch in pbar:
            inputs = self._prepare_batch(batch)
            labels = batch['label'].to(self.device)
            
            logits = self._forward(inputs)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        return compute_evaluation_metrics(
            predictions=np.array(all_preds),
            labels=np.array(all_labels),
            probabilities=np.array(all_probs),
            loss=avg_loss
        )
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute metrics.
        
        Overrides parent method to use comprehensive evaluation.
        
        Returns:
            Dictionary of validation metrics
        """
        eval_result = self.evaluate()
        
        return {
            'loss': eval_result.loss,
            'macro_f1': eval_result.macro_f1,
            'f1_depressed': eval_result.f1_depressed,
            'f1_non_depressed': eval_result.f1_non_depressed,
            'precision': eval_result.macro_precision,
            'recall': eval_result.macro_recall,
            'predictions': eval_result.predictions,
            'labels': eval_result.labels,
            'probabilities': eval_result.probabilities,
            'confusion_matrix': eval_result.confusion_matrix
        }
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Run full training loop with comprehensive metrics.
        
        Args:
            num_epochs: Override number of epochs from config
            
        Returns:
            Dictionary with final metrics and training history
        """
        if num_epochs is None:
            num_epochs = self.training_config.epochs
        
        print(f"\n{'='*60}")
        print(f"Starting Multimodal Training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()
            
            # Train
            train_loss, train_f1 = self.train_epoch()
            
            # Validate with comprehensive metrics
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Create metrics object
            metrics = MultimodalTrainingMetrics(
                epoch=self.current_epoch,
                train_loss=train_loss,
                val_loss=val_metrics['loss'],
                train_f1=train_f1,
                val_f1=val_metrics['macro_f1'],
                val_precision=val_metrics['precision'],
                val_recall=val_metrics['recall'],
                val_f1_depressed=val_metrics['f1_depressed'],
                val_f1_non_depressed=val_metrics['f1_non_depressed'],
                learning_rate=current_lr,
                best_val_f1=self.best_val_f1,
                confusion_matrix=val_metrics['confusion_matrix'].tolist()
            )
            
            self.training_history.append(metrics)
            
            # Log progress
            elapsed = time.time() - start_time
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs} Summary ({elapsed:.1f}s)")
            print(f"   Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            print(f"   F1(Depressed): {val_metrics['f1_depressed']:.4f} | F1(Non-Dep): {val_metrics['f1_non_depressed']:.4f}")
            print(f"   Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
            print(f"   LR: {current_lr:.2e}")
            
            # Print confusion matrix
            cm = val_metrics['confusion_matrix']
            print(f"   Confusion Matrix: [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")
            sys.stdout.flush()
            
            # Save best model
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.save_checkpoint('best_multimodal_model.pt')
                print(f"   ✅ New best model saved with Macro-F1: {self.best_val_f1:.4f}")
                sys.stdout.flush()
            
            # Early stopping check
            if self.early_stopping(val_metrics['macro_f1']):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                sys.stdout.flush()
                break
        
        # Save final checkpoint
        self.save_checkpoint('final_multimodal_model.pt')
        print(f"\n{'='*60}")
        print(f"Training complete! Best Macro-F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        # Load best model
        self.load_checkpoint('best_multimodal_model.pt')
        
        # Final evaluation
        final_eval = self.evaluate()
        final_eval.print_summary()
        
        return {
            'best_val_f1': self.best_val_f1,
            'final_epoch': self.current_epoch,
            'training_history': [m.to_dict() for m in self.training_history],
            'final_evaluation': final_eval.to_dict()
        }


class TextOnlyTrainer(ModelTrainer):
    """Trainer for text-only models (MuRIL or TF-IDF).
    
    Used for baseline comparison with multimodal model.
    """
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare text-only batch."""
        return {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['text_attention_mask'].to(self.device)
        }
    
    def _forward(self, inputs: Dict) -> torch.Tensor:
        """Forward pass through text model."""
        return self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )


def create_multimodal_trainer(
    audio_model: nn.Module,
    text_model: nn.Module,
    config: DepressionDetectionConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    class_weights: Optional[torch.Tensor] = None
) -> MultimodalTrainer:
    """Factory function to create multimodal trainer.
    
    Args:
        audio_model: Pre-trained audio model
        text_model: Pre-trained text model
        config: DepressionDetectionConfig
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        class_weights: Optional class weights for imbalanced data
        
    Returns:
        Initialized MultimodalTrainer
    """
    # Create multimodal model
    multimodal_model = create_multimodal_model(
        audio_model=audio_model,
        text_model=text_model,
        config=config.fusion
    )
    
    # Create trainer
    trainer = MultimodalTrainer(
        model=multimodal_model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights
    )
    
    return trainer


def compare_models(
    results: Dict[str, EvaluationResult],
    output_path: Optional[str] = None
) -> str:
    """Generate comparison table for multiple model variants.
    
    Implements Requirement 7.3 for model comparison.
    
    Args:
        results: Dictionary mapping model names to EvaluationResult
        output_path: Optional path to save comparison table
        
    Returns:
        Formatted comparison table as string
    """
    # Header
    table = "\n" + "=" * 80 + "\n"
    table += "Model Comparison Table\n"
    table += "=" * 80 + "\n\n"
    
    # Column headers
    headers = ["Model", "Macro-F1", "F1(Dep)", "F1(Non-Dep)", "Precision", "Recall"]
    header_line = " | ".join(f"{h:^12}" for h in headers)
    table += header_line + "\n"
    table += "-" * len(header_line) + "\n"
    
    # Data rows
    for model_name, result in results.items():
        row = [
            f"{model_name:^12}",
            f"{result.macro_f1:^12.4f}",
            f"{result.f1_depressed:^12.4f}",
            f"{result.f1_non_depressed:^12.4f}",
            f"{result.macro_precision:^12.4f}",
            f"{result.macro_recall:^12.4f}"
        ]
        table += " | ".join(row) + "\n"
    
    table += "=" * 80 + "\n"
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        logger.info(f"Saved comparison table to {output_path}")
    
    return table


if __name__ == "__main__":
    # Test multimodal trainer components
    import torch.nn as nn
    from config import get_config
    
    print("Testing MultimodalTrainer components...")
    
    # Test compute_evaluation_metrics
    print("\n1. Testing compute_evaluation_metrics...")
    preds = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    labels = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    
    result = compute_evaluation_metrics(preds, labels)
    print(f"   Macro-F1: {result.macro_f1:.4f}")
    print(f"   F1 Depressed: {result.f1_depressed:.4f}")
    print(f"   F1 Non-Depressed: {result.f1_non_depressed:.4f}")
    print(f"   Precision: {result.macro_precision:.4f}")
    print(f"   Recall: {result.macro_recall:.4f}")
    print(f"   Confusion Matrix:\n{result.confusion_matrix}")
    
    # Verify all required metrics are present
    assert result.macro_f1 is not None, "Missing macro_f1"
    assert result.f1_depressed is not None, "Missing f1_depressed"
    assert result.f1_non_depressed is not None, "Missing f1_non_depressed"
    assert result.precision_depressed is not None, "Missing precision_depressed"
    assert result.precision_non_depressed is not None, "Missing precision_non_depressed"
    assert result.recall_depressed is not None, "Missing recall_depressed"
    assert result.recall_non_depressed is not None, "Missing recall_non_depressed"
    assert result.confusion_matrix is not None, "Missing confusion_matrix"
    print("   ✓ All required metrics present")
    
    # Test EvaluationResult.to_dict()
    print("\n2. Testing EvaluationResult.to_dict()...")
    result_dict = result.to_dict()
    required_keys = [
        'macro_f1', 'f1_depressed', 'f1_non_depressed',
        'precision_depressed', 'precision_non_depressed',
        'recall_depressed', 'recall_non_depressed',
        'macro_precision', 'macro_recall', 'confusion_matrix', 'loss'
    ]
    for key in required_keys:
        assert key in result_dict, f"Missing key: {key}"
    print(f"   ✓ All {len(required_keys)} required keys present in dict")
    
    # Test compare_models
    print("\n3. Testing compare_models...")
    results = {
        'Audio-Only': result,
        'Text-Only': result,
        'Multimodal': result
    }
    table = compare_models(results)
    print(table)
    
    # Test EvaluationResult.print_summary()
    print("\n4. Testing EvaluationResult.print_summary()...")
    result.print_summary()
    
    print("\n✓ All component tests passed!")
