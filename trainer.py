"""
Training Pipeline for Depression Detection

This module implements the complete training pipeline as described in
Section 6: Training Pipeline and Optimization Dynamics

Key components:
- Weighted Cross-Entropy Loss with class balancing
- Cyclic learning rate scheduler
- Early stopping with patience
- Gradient accumulation for effective batch size
- Mixed precision training
- Model checkpointing
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
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from config import DepressionDetectionConfig, TrainingConfig, LossConfig
from ssl_model import Wav2Vec2ForDepressionDetection, create_ssl_model
from ecapa_model import ECAPATDNN, create_ecapa_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_f1: float
    val_f1: float
    val_precision: float
    val_recall: float
    val_f1_depressed: float
    val_f1_non_depressed: float
    learning_rate: float
    best_val_f1: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy Loss for class imbalance handling.
    
    Implements Section 6.1: Class weights computed inversely proportional
    to class frequencies: w_c = N_total / (N_c × num_classes)
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Shape (batch, num_classes)
            labels: Shape (batch,)
        """
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None
        
        return F.cross_entropy(
            logits,
            labels,
            weight=weights,
            label_smoothing=self.label_smoothing
        )


class FocalLoss(nn.Module):
    """Focal Loss for hard example mining.
    
    Alternative to weighted CE that down-weights easy examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            focal_loss = focal_loss * weights[labels]
        
        return focal_loss.mean()


class EarlyStopping:
    """Early stopping to prevent overfitting.
    
    Monitors validation Macro-F1 and stops if no improvement
    for 'patience' consecutive epochs.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelTrainer:
    """Base trainer class for depression detection models.
    
    Implements the training loop with:
    - Gradient accumulation
    - Mixed precision training
    - Learning rate scheduling
    - Checkpointing
    - Metric logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DepressionDetectionConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training config shortcuts
        self.training_config = config.training
        self.loss_config = config.loss
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Mixed precision scaler - only use for CUDA
        self.use_amp = config.training.use_mixed_precision and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='max'
        )
        
        # Tracking
        self.best_val_f1 = 0.0
        self.training_history: List[TrainingMetrics] = []
        self.current_epoch = 0
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.training_config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.training_config.learning_rate,
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """Create Cyclic LR scheduler as per Section 6.1."""
        steps_per_epoch = len(self.train_loader) // self.training_config.gradient_accumulation_steps
        
        if self.training_config.scheduler == 'cyclic':
            return CyclicLR(
                self.optimizer,
                base_lr=self.training_config.lr_min,
                max_lr=self.training_config.learning_rate,
                step_size_up=int(steps_per_epoch * self.training_config.warmup_ratio * self.training_config.epochs),
                step_size_down=int(steps_per_epoch * (1 - self.training_config.warmup_ratio) * self.training_config.epochs),
                mode='triangular2',
                cycle_momentum=False
            )
        else:
            # OneCycleLR as alternative
            return OneCycleLR(
                self.optimizer,
                max_lr=self.training_config.learning_rate,
                epochs=self.training_config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.training_config.warmup_ratio
            )
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function with class weights."""
        # Get class weights from training loader dataset
        if hasattr(self.train_loader.dataset, 'class_weights'):
            class_weights = self.train_loader.dataset.class_weights
        else:
            class_weights = None
        
        if self.loss_config.use_class_weights and class_weights is not None:
            logger.info(f"Using class weights: {class_weights}")
            return WeightedCrossEntropyLoss(
                class_weights=class_weights,
                label_smoothing=self.loss_config.label_smoothing
            )
        else:
            return nn.CrossEntropyLoss(
                label_smoothing=self.loss_config.label_smoothing
            )
    
    def train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch.
        
        Returns:
            Tuple of (average loss, train F1)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        # Progress bar for training
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=True,
            file=sys.stdout
        )
        
        for batch_idx, batch in pbar:
            # Move batch to device
            inputs = self._prepare_batch(batch)
            labels = batch['label'].to(self.device)
            
            # Forward pass with mixed precision (CUDA only)
            if self.use_amp:
                with autocast(device_type='cuda'):
                    logits = self._forward(inputs)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.training_config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                logits = self._forward(inputs)
                loss = self.criterion(logits, labels)
                loss = loss / self.training_config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.training_config.gradient_accumulation_steps
            
            # Collect predictions
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            avg_loss_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, train_f1
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute metrics.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Progress bar for validation
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
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
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        
        # Per-class F1
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        f1_non_depressed = f1_per_class[0]
        f1_depressed = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
        
        # Macro F1 (the target metric)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': avg_loss,
            'macro_f1': macro_f1,
            'f1_depressed': f1_depressed,
            'f1_non_depressed': f1_non_depressed,
            'precision': precision,
            'recall': recall,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs)
        }
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, float]:
        """Run full training loop.
        
        Args:
            num_epochs: Override number of epochs from config
            
        Returns:
            Dictionary with final metrics
        """
        if num_epochs is None:
            num_epochs = self.training_config.epochs
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()
            
            # Train
            train_loss, train_f1 = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Create metrics object
            metrics = TrainingMetrics(
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
                best_val_f1=self.best_val_f1
            )
            
            self.training_history.append(metrics)
            
            # Log progress with clear visual formatting
            elapsed = time.time() - start_time
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs} Summary ({elapsed:.1f}s)")
            print(f"   Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            print(f"   F1(Depressed): {val_metrics['f1_depressed']:.4f} | F1(Non-Dep): {val_metrics['f1_non_depressed']:.4f}")
            print(f"   LR: {current_lr:.2e}")
            sys.stdout.flush()
            
            # Save best model
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.save_checkpoint('best_model.pt')
                print(f"   ✅ New best model saved with Macro-F1: {self.best_val_f1:.4f}")
                sys.stdout.flush()
            
            # Early stopping check
            if self.early_stopping(val_metrics['macro_f1']):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                sys.stdout.flush()
                break
            
            # Handle layer unfreezing for SSL model
            self._handle_layer_unfreezing(epoch + 1)
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        print(f"\n{'='*60}")
        print(f"Training complete! Best Macro-F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        # Load best model for return
        self.load_checkpoint('best_model.pt')
        
        return {
            'best_val_f1': self.best_val_f1,
            'final_epoch': self.current_epoch,
            'training_history': [m.to_dict() for m in self.training_history]
        }
    
    def _handle_layer_unfreezing(self, epoch: int):
        """Unfreeze layers after specified number of epochs."""
        if hasattr(self.model, '_unfreeze_feature_extractor'):
            if hasattr(self.config.ssl_model, 'freeze_epochs'):
                if epoch == self.config.ssl_model.freeze_epochs:
                    self.model._unfreeze_feature_extractor()
                    logger.info(f"Unfroze feature extractor at epoch {epoch}")
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare batch for model input. Override in subclasses."""
        raise NotImplementedError
    
    def _forward(self, inputs: Dict) -> torch.Tensor:
        """Run forward pass. Override in subclasses."""
        raise NotImplementedError
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.paths.checkpoint_dir,
            filename
        )
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'training_history': [m.to_dict() for m in self.training_history],
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.paths.checkpoint_dir,
            filename
        )
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class SSLModelTrainer(ModelTrainer):
    """Trainer for SSL (Wav2Vec2/IndicWav2Vec) models."""
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare batch for SSL model."""
        waveform = batch['waveform'].squeeze(1)  # Remove channel dim: (B, 1, T) -> (B, T)
        attention_mask = batch['attention_mask']
        
        return {
            'input_values': waveform.to(self.device),
            'attention_mask': attention_mask.to(self.device)
        }
    
    def _forward(self, inputs: Dict) -> torch.Tensor:
        """Forward pass through SSL model."""
        return self.model(
            input_values=inputs['input_values'],
            attention_mask=inputs['attention_mask']
        )


class ECAPAModelTrainer(ModelTrainer):
    """Trainer for ECAPA-TDNN models."""
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare batch for ECAPA model."""
        return {
            'mel_spectrogram': batch['mel_spectrogram'].to(self.device)
        }
    
    def _forward(self, inputs: Dict) -> torch.Tensor:
        """Forward pass through ECAPA model."""
        # During training with AAM Softmax, we need labels
        if self.model.training:
            return self.model(inputs['mel_spectrogram'])
        else:
            return self.model(inputs['mel_spectrogram'])


class HybridModelTrainer:
    """Trainer for both SSL and ECAPA models in parallel.
    
    Trains both streams and saves checkpoints for ensemble.
    """
    
    def __init__(
        self,
        config: DepressionDetectionConfig,
        ssl_train_loader: DataLoader,
        ssl_val_loader: DataLoader,
        ecapa_train_loader: DataLoader,
        ecapa_val_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device
        
        # Create models
        self.ssl_model = create_ssl_model(config)
        self.ecapa_model = create_ecapa_model(config)
        
        # Create trainers
        self.ssl_trainer = SSLModelTrainer(
            self.ssl_model,
            config,
            ssl_train_loader,
            ssl_val_loader,
            device
        )
        
        self.ecapa_trainer = ECAPAModelTrainer(
            self.ecapa_model,
            config,
            ecapa_train_loader,
            ecapa_val_loader,
            device
        )
    
    def train(self) -> Dict[str, Dict]:
        """Train both models.
        
        Returns:
            Dictionary with results for both models
        """
        logger.info("=" * 60)
        logger.info("Training SSL Model (IndicWav2Vec)")
        logger.info("=" * 60)
        ssl_results = self.ssl_trainer.train()
        
        logger.info("=" * 60)
        logger.info("Training ECAPA-TDNN Model")
        logger.info("=" * 60)
        ecapa_results = self.ecapa_trainer.train()
        
        return {
            'ssl': ssl_results,
            'ecapa': ecapa_results
        }


def compute_subject_level_metrics(
    chunk_predictions: np.ndarray,
    chunk_labels: np.ndarray,
    chunk_speaker_ids: List[str],
    aggregation: str = 'mean'
) -> Dict[str, float]:
    """Compute subject-level metrics from chunk predictions.
    
    Implements Section 3.2: Evaluation at subject level, not chunk level.
    
    Args:
        chunk_predictions: Predicted probabilities for depressed class
        chunk_labels: Ground truth labels
        chunk_speaker_ids: Speaker IDs for each chunk
        aggregation: 'mean' or 'majority' for chunk aggregation
        
    Returns:
        Dictionary with subject-level metrics
    """
    # Group predictions by speaker
    speaker_preds = defaultdict(list)
    speaker_labels = {}
    
    for pred, label, speaker in zip(chunk_predictions, chunk_labels, chunk_speaker_ids):
        speaker_preds[speaker].append(pred)
        speaker_labels[speaker] = label  # All chunks from same speaker have same label
    
    # Aggregate predictions
    final_preds = []
    final_labels = []
    
    for speaker, preds in speaker_preds.items():
        if aggregation == 'mean':
            # Average probability
            prob = np.mean(preds)
            pred = 1 if prob >= 0.5 else 0
        else:
            # Majority voting
            pred = 1 if np.sum(preds) > len(preds) / 2 else 0
        
        final_preds.append(pred)
        final_labels.append(speaker_labels[speaker])
    
    # Compute metrics
    final_preds = np.array(final_preds)
    final_labels = np.array(final_labels)
    
    macro_f1 = f1_score(final_labels, final_preds, average='macro')
    f1_per_class = f1_score(final_labels, final_preds, average=None)
    
    return {
        'subject_macro_f1': macro_f1,
        'subject_f1_depressed': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        'subject_f1_non_depressed': f1_per_class[0],
        'num_subjects': len(speaker_preds)
    }


if __name__ == "__main__":
    # Test trainer setup
    from config import get_config
    from dataset import create_dataloaders, create_mel_dataloaders
    
    config = get_config("combined")
    
    # Use smaller model for testing
    config.ssl_model.model_name = "facebook/wav2vec2-base"
    config.ssl_model.fallback_model = "facebook/wav2vec2-base"
    config.training.epochs = 2
    config.training.batch_size = 4
    
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(config, fold_idx=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test batch preparation
    batch = next(iter(train_loader))
    print(f"\nBatch waveform shape: {batch['waveform'].shape}")
    print(f"Batch labels: {batch['label']}")
