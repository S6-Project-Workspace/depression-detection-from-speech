"""
Enhanced Multimodal Trainer for Depression Detection with Linguistic Features

This module implements the training pipeline for enhanced multimodal depression detection,
combining audio, text, and linguistic features.

Key components:
- Joint audio-text-linguistic batch handling
- Progressive training strategy (baseline → enhanced)
- Feature importance analysis and ablation study capabilities
- Linguistic feature contribution metrics in training logs
- Support for trimodal, bimodal, and unimodal fallback modes

Reference: Traditional NLP Tasks - Task 7, Requirements REQ-6
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
from enhanced_multimodal_model import EnhancedMultimodalModel, create_enhanced_multimodal_model
from multimodal_trainer import (
    MultimodalTrainer,
    MultimodalTrainingMetrics,
    EvaluationResult,
    compute_evaluation_metrics,
    compare_models
)
from trainer import (
    ModelTrainer,
    WeightedCrossEntropyLoss,
    EarlyStopping,
    TrainingMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainingMetrics(MultimodalTrainingMetrics):
    """Extended metrics for enhanced multimodal training with linguistic features."""
    # Linguistic feature importance metrics
    linguistic_feature_importance: Optional[float] = None
    audio_feature_importance: Optional[float] = None
    text_feature_importance: Optional[float] = None
    
    # Ablation study metrics
    trimodal_f1: Optional[float] = None
    bimodal_audio_text_f1: Optional[float] = None
    bimodal_audio_linguistic_f1: Optional[float] = None
    bimodal_text_linguistic_f1: Optional[float] = None
    
    # Progressive training metrics
    training_phase: str = "enhanced"  # "baseline", "enhanced"
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert confusion matrix to list if it's a numpy array
        if isinstance(d.get('confusion_matrix'), np.ndarray):
            d['confusion_matrix'] = d['confusion_matrix'].tolist()
        return d


@dataclass
class EnhancedEvaluationResult(EvaluationResult):
    """Extended evaluation result with linguistic feature analysis."""
    # Feature importance scores
    linguistic_feature_importance: float = 0.0
    audio_feature_importance: float = 0.0
    text_feature_importance: float = 0.0
    
    # Ablation study results
    trimodal_performance: Dict[str, float] = field(default_factory=dict)
    bimodal_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    unimodal_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Linguistic contribution analysis
    linguistic_contribution_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'linguistic_feature_importance': self.linguistic_feature_importance,
            'audio_feature_importance': self.audio_feature_importance,
            'text_feature_importance': self.text_feature_importance,
            'trimodal_performance': self.trimodal_performance,
            'bimodal_performance': self.bimodal_performance,
            'unimodal_performance': self.unimodal_performance,
            'linguistic_contribution_score': self.linguistic_contribution_score
        })
        return base_dict
    
    def print_summary(self):
        """Print a formatted summary of enhanced evaluation results."""
        super().print_summary()
        print("\n" + "=" * 50)
        print("Enhanced Features Analysis")
        print("=" * 50)
        print(f"Linguistic Feature Importance: {self.linguistic_feature_importance:.4f}")
        print(f"Audio Feature Importance: {self.audio_feature_importance:.4f}")
        print(f"Text Feature Importance: {self.text_feature_importance:.4f}")
        print(f"Linguistic Contribution Score: {self.linguistic_contribution_score:.4f}")
        
        if self.trimodal_performance:
            print(f"\nTrimodal Performance:")
            for metric, value in self.trimodal_performance.items():
                print(f"  {metric}: {value:.4f}")
        
        if self.bimodal_performance:
            print(f"\nBimodal Performance:")
            for mode, metrics in self.bimodal_performance.items():
                print(f"  {mode}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
        
        print("=" * 50)


class EnhancedMultimodalTrainer(MultimodalTrainer):
    """Enhanced trainer for multimodal depression detection with linguistic features.
    
    Extends MultimodalTrainer to handle joint audio-text-linguistic batches and
    support progressive training and feature importance analysis.
    
    Implements:
    - Requirement REQ-6: Integration with existing multimodal system
    - Progressive training strategy (baseline → enhanced)
    - Feature importance analysis for linguistic contributions
    - Ablation study capabilities
    """
    
    def __init__(
        self,
        model: EnhancedMultimodalModel,
        config: DepressionDetectionConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        class_weights: Optional[torch.Tensor] = None,
        progressive_training: bool = True
    ):
        """Initialize enhanced multimodal trainer.
        
        Args:
            model: EnhancedMultimodalModel instance
            config: DepressionDetectionConfig
            train_loader: Training data loader with linguistic features
            val_loader: Validation data loader with linguistic features
            device: Device to train on
            class_weights: Optional class weights for imbalanced data
            progressive_training: Whether to use progressive training strategy
        """
        self.progressive_training = progressive_training
        self.current_training_phase = "enhanced"
        
        # Initialize parent class
        super().__init__(model, config, train_loader, val_loader, device, class_weights)
        
        logger.info(f"Enhanced multimodal trainer initialized with progressive training: {progressive_training}")
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare enhanced multimodal batch for model input.
        
        Handles audio, text, and linguistic inputs from the batch.
        
        Args:
            batch: Dictionary containing audio, text, and linguistic data
            
        Returns:
            Dictionary with prepared inputs
        """
        prepared = super()._prepare_batch(batch)
        
        # Add linguistic features if available
        if 'linguistic_features' in batch:
            prepared['linguistic_features'] = batch['linguistic_features'].to(self.device)
        
        # Add raw text for on-the-fly linguistic analysis if needed
        if 'text_raw' in batch:
            prepared['text_raw'] = batch['text_raw']
        
        return prepared
    
    def _forward(self, inputs: Dict) -> torch.Tensor:
        """Run forward pass through enhanced multimodal model.
        
        Handles different input combinations:
        - Full trimodal (audio + text + linguistic)
        - Bimodal fallbacks (audio+text, audio+linguistic, text+linguistic)
        - Unimodal fallbacks (audio, text, linguistic)
        
        Args:
            inputs: Prepared input dictionary
            
        Returns:
            Model logits
        """
        has_audio = 'audio_input' in inputs
        has_text = 'text_input_ids' in inputs
        has_linguistic = 'linguistic_features' in inputs or 'text_raw' in inputs
        
        # Determine the best forward mode based on available inputs
        if has_audio and has_text and has_linguistic:
            # Full trimodal forward
            return self.model(
                audio_input=inputs['audio_input'],
                text_input_ids=inputs['text_input_ids'],
                text_attention_mask=inputs['text_attention_mask'],
                linguistic_features=inputs.get('linguistic_features'),
                text_raw=inputs.get('text_raw'),
                audio_attention_mask=inputs.get('audio_attention_mask')
            )
        elif has_audio and has_text:
            # Bimodal audio-text fallback
            return self.model.forward_bimodal_audio_text(
                audio_input=inputs['audio_input'],
                text_input_ids=inputs['text_input_ids'],
                text_attention_mask=inputs['text_attention_mask'],
                audio_attention_mask=inputs.get('audio_attention_mask')
            )
        elif has_audio and has_linguistic:
            # Bimodal audio-linguistic fallback
            return self.model.forward_bimodal_audio_linguistic(
                audio_input=inputs['audio_input'],
                linguistic_features=inputs.get('linguistic_features'),
                text_raw=inputs.get('text_raw'),
                audio_attention_mask=inputs.get('audio_attention_mask')
            )
        elif has_text and has_linguistic:
            # Bimodal text-linguistic fallback
            return self.model.forward_bimodal_text_linguistic(
                text_input_ids=inputs['text_input_ids'],
                text_attention_mask=inputs['text_attention_mask'],
                linguistic_features=inputs.get('linguistic_features'),
                text_raw=inputs.get('text_raw')
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
        elif has_linguistic:
            # Linguistic-only fallback
            return self.model.forward_linguistic_only(
                linguistic_features=inputs.get('linguistic_features'),
                text_raw=inputs.get('text_raw')
            )
        else:
            raise ValueError("Batch must contain at least one modality (audio, text, or linguistic)")
    
    @torch.no_grad()
    def evaluate_with_ablation(self) -> EnhancedEvaluationResult:
        """Run comprehensive evaluation with ablation study.
        
        Tests different modality combinations to analyze feature importance.
        
        Returns:
            EnhancedEvaluationResult with ablation study results
        """
        self.model.eval()
        
        # Storage for different modality combinations
        results = {}
        
        # Test all possible combinations
        modality_combinations = [
            ('trimodal', ['audio', 'text', 'linguistic']),
            ('bimodal_audio_text', ['audio', 'text']),
            ('bimodal_audio_linguistic', ['audio', 'linguistic']),
            ('bimodal_text_linguistic', ['text', 'linguistic']),
            ('unimodal_audio', ['audio']),
            ('unimodal_text', ['text']),
            ('unimodal_linguistic', ['linguistic'])
        ]
        
        for combination_name, modalities in modality_combinations:
            all_preds = []
            all_labels = []
            all_probs = []
            total_loss = 0.0
            
            pbar = tqdm(
                self.val_loader,
                desc=f"Evaluating {combination_name}",
                leave=False,
                file=sys.stdout
            )
            
            for batch in pbar:
                # Prepare inputs based on available modalities
                inputs = {}
                labels = batch['label'].to(self.device)
                
                # Add inputs based on combination
                if 'audio' in modalities and 'waveform' in batch:
                    waveform = batch['waveform']
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(1)
                    inputs['audio_input'] = waveform.to(self.device)
                    if 'attention_mask' in batch:
                        inputs['audio_attention_mask'] = batch['attention_mask'].to(self.device)
                elif 'audio' in modalities and 'mel_spectrogram' in batch:
                    inputs['audio_input'] = batch['mel_spectrogram'].to(self.device)
                
                if 'text' in modalities and 'input_ids' in batch:
                    inputs['text_input_ids'] = batch['input_ids'].to(self.device)
                    inputs['text_attention_mask'] = batch['text_attention_mask'].to(self.device)
                
                if 'linguistic' in modalities:
                    if 'linguistic_features' in batch:
                        inputs['linguistic_features'] = batch['linguistic_features'].to(self.device)
                    if 'text_raw' in batch:
                        inputs['text_raw'] = batch['text_raw']
                
                # Skip if required modalities are not available
                if not inputs:
                    continue
                
                try:
                    logits = self._forward(inputs)
                    loss = self.criterion(logits, labels)
                    
                    total_loss += loss.item()
                    
                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                except Exception as e:
                    logger.warning(f"Skipping batch for {combination_name}: {e}")
                    continue
            
            if all_preds:
                avg_loss = total_loss / len(self.val_loader)
                eval_result = compute_evaluation_metrics(
                    predictions=np.array(all_preds),
                    labels=np.array(all_labels),
                    probabilities=np.array(all_probs),
                    loss=avg_loss
                )
                results[combination_name] = eval_result
            else:
                logger.warning(f"No valid predictions for {combination_name}")
        
        # Get feature importance from the model if available
        feature_importance = {}
        if hasattr(self.model, 'get_feature_importance'):
            try:
                feature_importance = self.model.get_feature_importance()
            except Exception as e:
                logger.warning(f"Could not get feature importance: {e}")
        
        # Create enhanced evaluation result
        if 'trimodal' in results:
            base_result = results['trimodal']
        elif 'bimodal_audio_text' in results:
            base_result = results['bimodal_audio_text']
        else:
            # Fallback to any available result
            base_result = list(results.values())[0] if results else EvaluationResult(
                macro_f1=0.0, f1_depressed=0.0, f1_non_depressed=0.0,
                precision_depressed=0.0, precision_non_depressed=0.0,
                recall_depressed=0.0, recall_non_depressed=0.0,
                macro_precision=0.0, macro_recall=0.0,
                confusion_matrix=np.zeros((2, 2)), loss=float('inf')
            )
        
        # Calculate linguistic contribution score
        linguistic_contribution = 0.0
        if 'trimodal' in results and 'bimodal_audio_text' in results:
            linguistic_contribution = results['trimodal'].macro_f1 - results['bimodal_audio_text'].macro_f1
        
        # Organize results by category
        trimodal_performance = {}
        bimodal_performance = {}
        unimodal_performance = {}
        
        for name, result in results.items():
            metrics = {
                'macro_f1': result.macro_f1,
                'precision': result.macro_precision,
                'recall': result.macro_recall,
                'loss': result.loss
            }
            
            if name == 'trimodal':
                trimodal_performance = metrics
            elif name.startswith('bimodal_'):
                mode = name.replace('bimodal_', '')
                bimodal_performance[mode] = metrics
            elif name.startswith('unimodal_'):
                mode = name.replace('unimodal_', '')
                unimodal_performance[mode] = metrics
        
        enhanced_result = EnhancedEvaluationResult(
            macro_f1=base_result.macro_f1,
            f1_depressed=base_result.f1_depressed,
            f1_non_depressed=base_result.f1_non_depressed,
            precision_depressed=base_result.precision_depressed,
            precision_non_depressed=base_result.precision_non_depressed,
            recall_depressed=base_result.recall_depressed,
            recall_non_depressed=base_result.recall_non_depressed,
            macro_precision=base_result.macro_precision,
            macro_recall=base_result.macro_recall,
            confusion_matrix=base_result.confusion_matrix,
            loss=base_result.loss,
            predictions=base_result.predictions,
            labels=base_result.labels,
            probabilities=base_result.probabilities,
            linguistic_feature_importance=feature_importance.get('linguistic', 0.0),
            audio_feature_importance=feature_importance.get('audio', 0.0),
            text_feature_importance=feature_importance.get('text', 0.0),
            trimodal_performance=trimodal_performance,
            bimodal_performance=bimodal_performance,
            unimodal_performance=unimodal_performance,
            linguistic_contribution_score=linguistic_contribution
        )
        
        return enhanced_result
    
    @torch.no_grad()
    def evaluate(self) -> EnhancedEvaluationResult:
        """Run comprehensive evaluation on validation set.
        
        Overrides parent method to return enhanced evaluation results.
        
        Returns:
            EnhancedEvaluationResult with feature importance analysis
        """
        return self.evaluate_with_ablation()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute enhanced metrics.
        
        Overrides parent method to include linguistic feature metrics.
        
        Returns:
            Dictionary of validation metrics including feature importance
        """
        eval_result = self.evaluate()
        
        metrics = {
            'loss': eval_result.loss,
            'macro_f1': eval_result.macro_f1,
            'f1_depressed': eval_result.f1_depressed,
            'f1_non_depressed': eval_result.f1_non_depressed,
            'precision': eval_result.macro_precision,
            'recall': eval_result.macro_recall,
            'predictions': eval_result.predictions,
            'labels': eval_result.labels,
            'probabilities': eval_result.probabilities,
            'confusion_matrix': eval_result.confusion_matrix,
            'linguistic_feature_importance': eval_result.linguistic_feature_importance,
            'audio_feature_importance': eval_result.audio_feature_importance,
            'text_feature_importance': eval_result.text_feature_importance,
            'linguistic_contribution_score': eval_result.linguistic_contribution_score
        }
        
        return metrics
    
    def train_progressive(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Run progressive training: baseline → enhanced.
        
        First trains without linguistic features, then fine-tunes with them.
        
        Args:
            num_epochs: Override number of epochs from config
            
        Returns:
            Dictionary with training results from both phases
        """
        if num_epochs is None:
            num_epochs = self.training_config.epochs
        
        baseline_epochs = num_epochs // 2
        enhanced_epochs = num_epochs - baseline_epochs
        
        print(f"\n{'='*60}")
        print(f"Starting Progressive Training")
        print(f"Phase 1 (Baseline): {baseline_epochs} epochs")
        print(f"Phase 2 (Enhanced): {enhanced_epochs} epochs")
        print(f"{'='*60}\n")
        
        # Phase 1: Baseline training (audio + text only)
        print(f"🚀 Phase 1: Baseline Training ({baseline_epochs} epochs)")
        self.current_training_phase = "baseline"
        
        # Temporarily disable linguistic features in the model
        if hasattr(self.model, 'set_linguistic_enabled'):
            self.model.set_linguistic_enabled(False)
        
        baseline_results = self.train(baseline_epochs)
        
        # Phase 2: Enhanced training (audio + text + linguistic)
        print(f"\n🚀 Phase 2: Enhanced Training ({enhanced_epochs} epochs)")
        self.current_training_phase = "enhanced"
        
        # Re-enable linguistic features
        if hasattr(self.model, 'set_linguistic_enabled'):
            self.model.set_linguistic_enabled(True)
        
        # Reset some training state for second phase
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        
        # Use lower learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        
        enhanced_results = self.train(enhanced_epochs)
        
        return {
            'baseline_results': baseline_results,
            'enhanced_results': enhanced_results,
            'progressive_training': True
        }
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Run training loop with enhanced metrics.
        
        Args:
            num_epochs: Override number of epochs from config
            
        Returns:
            Dictionary with final metrics and training history
        """
        if self.progressive_training and self.current_training_phase == "enhanced":
            return self.train_progressive(num_epochs)
        
        if num_epochs is None:
            num_epochs = self.training_config.epochs
        
        phase_name = self.current_training_phase.title()
        print(f"\n{'='*60}")
        print(f"Starting {phase_name} Training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()
            
            # Train
            train_loss, train_f1 = self.train_epoch()
            
            # Validate with enhanced metrics
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Create enhanced metrics object
            metrics = EnhancedTrainingMetrics(
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
                confusion_matrix=val_metrics['confusion_matrix'].tolist(),
                linguistic_feature_importance=val_metrics['linguistic_feature_importance'],
                audio_feature_importance=val_metrics['audio_feature_importance'],
                text_feature_importance=val_metrics['text_feature_importance'],
                training_phase=self.current_training_phase
            )
            
            self.training_history.append(metrics)
            
            # Log progress
            elapsed = time.time() - start_time
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs} Summary ({elapsed:.1f}s)")
            print(f"   Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            print(f"   F1(Depressed): {val_metrics['f1_depressed']:.4f} | F1(Non-Dep): {val_metrics['f1_non_depressed']:.4f}")
            print(f"   Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
            print(f"   LR: {current_lr:.2e} | Phase: {self.current_training_phase}")
            
            # Feature importance
            if val_metrics['linguistic_feature_importance'] > 0:
                print(f"   Feature Importance - Ling: {val_metrics['linguistic_feature_importance']:.3f}, "
                      f"Audio: {val_metrics['audio_feature_importance']:.3f}, "
                      f"Text: {val_metrics['text_feature_importance']:.3f}")
                print(f"   Linguistic Contribution: {val_metrics['linguistic_contribution_score']:.4f}")
            
            # Print confusion matrix
            cm = val_metrics['confusion_matrix']
            print(f"   Confusion Matrix: [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")
            sys.stdout.flush()
            
            # Save best model
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                checkpoint_name = f'best_enhanced_multimodal_model_{self.current_training_phase}.pt'
                self.save_checkpoint(checkpoint_name)
                print(f"   ✅ New best model saved with Macro-F1: {self.best_val_f1:.4f}")
                sys.stdout.flush()
            
            # Early stopping check
            if self.early_stopping(val_metrics['macro_f1']):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                sys.stdout.flush()
                break
        
        # Save final checkpoint
        final_checkpoint_name = f'final_enhanced_multimodal_model_{self.current_training_phase}.pt'
        self.save_checkpoint(final_checkpoint_name)
        print(f"\n{'='*60}")
        print(f"{phase_name} Training complete! Best Macro-F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        # Load best model
        best_checkpoint_name = f'best_enhanced_multimodal_model_{self.current_training_phase}.pt'
        self.load_checkpoint(best_checkpoint_name)
        
        # Final evaluation with ablation study
        final_eval = self.evaluate_with_ablation()
        final_eval.print_summary()
        
        return {
            'best_val_f1': self.best_val_f1,
            'final_epoch': self.current_epoch,
            'training_history': [m.to_dict() for m in self.training_history],
            'final_evaluation': final_eval.to_dict(),
            'training_phase': self.current_training_phase
        }


def create_enhanced_multimodal_trainer(
    audio_model: nn.Module,
    text_model: nn.Module,
    linguistic_analyzer,
    config: DepressionDetectionConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    class_weights: Optional[torch.Tensor] = None,
    progressive_training: bool = True
) -> EnhancedMultimodalTrainer:
    """Factory function to create enhanced multimodal trainer.
    
    Args:
        audio_model: Pre-trained audio model
        text_model: Pre-trained text model
        linguistic_analyzer: LinguisticAnalyzer instance
        config: DepressionDetectionConfig
        train_loader: Training data loader with linguistic features
        val_loader: Validation data loader with linguistic features
        device: Device to train on
        class_weights: Optional class weights for imbalanced data
        progressive_training: Whether to use progressive training strategy
        
    Returns:
        Initialized EnhancedMultimodalTrainer
    """
    # Create enhanced multimodal model
    enhanced_model = create_enhanced_multimodal_model(
        audio_model=audio_model,
        text_model=text_model,
        linguistic_analyzer=linguistic_analyzer,
        config=config.fusion
    )
    
    # Create trainer
    trainer = EnhancedMultimodalTrainer(
        model=enhanced_model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        progressive_training=progressive_training
    )
    
    return trainer


if __name__ == "__main__":
    # Test enhanced multimodal trainer components
    import torch.nn as nn
    from config import get_config
    
    print("Testing EnhancedMultimodalTrainer components...")
    
    # Test EnhancedEvaluationResult
    print("\n1. Testing EnhancedEvaluationResult...")
    preds = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    labels = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    
    # Create base result
    base_result = compute_evaluation_metrics(preds, labels)
    
    # Create enhanced result
    enhanced_result = EnhancedEvaluationResult(
        macro_f1=base_result.macro_f1,
        f1_depressed=base_result.f1_depressed,
        f1_non_depressed=base_result.f1_non_depressed,
        precision_depressed=base_result.precision_depressed,
        precision_non_depressed=base_result.precision_non_depressed,
        recall_depressed=base_result.recall_depressed,
        recall_non_depressed=base_result.recall_non_depressed,
        macro_precision=base_result.macro_precision,
        macro_recall=base_result.macro_recall,
        confusion_matrix=base_result.confusion_matrix,
        loss=base_result.loss,
        linguistic_feature_importance=0.3,
        audio_feature_importance=0.4,
        text_feature_importance=0.3,
        linguistic_contribution_score=0.05
    )
    
    print(f"   Linguistic Feature Importance: {enhanced_result.linguistic_feature_importance:.3f}")
    print(f"   Audio Feature Importance: {enhanced_result.audio_feature_importance:.3f}")
    print(f"   Text Feature Importance: {enhanced_result.text_feature_importance:.3f}")
    print(f"   Linguistic Contribution Score: {enhanced_result.linguistic_contribution_score:.3f}")
    
    # Test to_dict method
    print("\n2. Testing EnhancedEvaluationResult.to_dict()...")
    result_dict = enhanced_result.to_dict()
    enhanced_keys = [
        'linguistic_feature_importance', 'audio_feature_importance', 'text_feature_importance',
        'trimodal_performance', 'bimodal_performance', 'unimodal_performance',
        'linguistic_contribution_score'
    ]
    for key in enhanced_keys:
        assert key in result_dict, f"Missing enhanced key: {key}"
    print(f"   ✓ All {len(enhanced_keys)} enhanced keys present in dict")
    
    # Test EnhancedTrainingMetrics
    print("\n3. Testing EnhancedTrainingMetrics...")
    enhanced_metrics = EnhancedTrainingMetrics(
        epoch=1,
        train_loss=0.5,
        val_loss=0.4,
        train_f1=0.8,
        val_f1=0.75,
        val_precision=0.76,
        val_recall=0.74,
        val_f1_depressed=0.73,
        val_f1_non_depressed=0.77,
        learning_rate=1e-4,
        best_val_f1=0.75,
        linguistic_feature_importance=0.3,
        audio_feature_importance=0.4,
        text_feature_importance=0.3,
        training_phase="enhanced"
    )
    
    metrics_dict = enhanced_metrics.to_dict()
    assert 'linguistic_feature_importance' in metrics_dict
    assert 'training_phase' in metrics_dict
    assert metrics_dict['training_phase'] == "enhanced"
    print(f"   ✓ Enhanced training metrics created successfully")
    
    # Test print_summary method
    print("\n4. Testing EnhancedEvaluationResult.print_summary()...")
    enhanced_result.print_summary()
    
    print("\n✓ All enhanced trainer component tests passed!")