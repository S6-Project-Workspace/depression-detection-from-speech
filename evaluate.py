"""
Evaluation Script for Depression Detection Models

This script evaluates trained models and computes F1 scores
for the DravidianLangTech @ ACL 2026 Shared Task.

Usage:
    python evaluate.py --checkpoint-dir outputs/experiment_name
    python evaluate.py --ssl-checkpoint path/to/ssl.pt --ecapa-checkpoint path/to/ecapa.pt
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, confusion_matrix, classification_report
)

from config import get_config, DepressionDetectionConfig
from preprocessing import build_dataset_manifest, AudioPreprocessor
from dataset import create_dataloaders, create_mel_dataloaders, StratifiedGroupKFold
from ssl_model import create_ssl_model
from ecapa_model import create_ecapa_model
from threshold_tuner import MacroF1ThresholdOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def evaluate_ssl_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> Dict:
    """Evaluate SSL model and return predictions."""
    model.eval()
    model.to(device)
    
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating SSL"):
            waveforms = batch['waveform'].to(device)
            # Handle different input shapes - squeeze extra dimensions
            if waveforms.dim() == 4:
                waveforms = waveforms.squeeze(1).squeeze(1)
            elif waveforms.dim() == 3:
                waveforms = waveforms.squeeze(1)
            labels = batch['label'].to(device)
            
            outputs = model(waveforms)
            # Model returns logits directly or dict with 'logits' key
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            probs = F.softmax(logits, dim=-1)[:, 1]  # Probability of depressed
            preds = (probs > 0.5).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return {
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels),
        'predictions': np.array(all_preds)
    }


def evaluate_ecapa_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> Dict:
    """Evaluate ECAPA model and return predictions."""
    model.eval()
    model.to(device)
    
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating ECAPA"):
            mel_specs = batch['mel_spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(mel_specs)
            # Model returns logits directly or dict with 'logits' key
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return {
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels),
        'predictions': np.array(all_preds)
    }


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, probs: Optional[np.ndarray] = None) -> Dict:
    """Compute comprehensive metrics."""
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro'),
        'weighted_f1': f1_score(labels, predictions, average='weighted'),
        'f1_depressed': f1_score(labels, predictions, pos_label=1),
        'f1_non_depressed': f1_score(labels, predictions, pos_label=0),
        'precision_macro': precision_score(labels, predictions, average='macro'),
        'recall_macro': recall_score(labels, predictions, average='macro'),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist()
    }
    
    # Optimize threshold if probabilities provided
    if probs is not None:
        optimizer = MacroF1ThresholdOptimizer()
        result = optimizer.optimize(probs, labels)
        best_threshold = result.optimal_threshold
        best_f1 = result.optimal_macro_f1
        optimized_preds = (probs >= best_threshold).astype(int)
        metrics['optimized_threshold'] = best_threshold
        metrics['optimized_macro_f1'] = best_f1
        metrics['optimized_f1_depressed'] = f1_score(labels, optimized_preds, pos_label=1)
        metrics['optimized_f1_non_depressed'] = f1_score(labels, optimized_preds, pos_label=0)
    
    return metrics


def evaluate_ensemble(
    ssl_probs: np.ndarray,
    ecapa_probs: np.ndarray,
    labels: np.ndarray,
    weights: Optional[List[float]] = None
) -> Dict:
    """Evaluate ensemble of SSL and ECAPA models."""
    if weights is None:
        # Grid search for optimal weight
        best_f1 = 0
        best_weight = 0.5
        for w in np.arange(0.1, 1.0, 0.05):
            fused_probs = w * ssl_probs + (1 - w) * ecapa_probs
            optimizer = MacroF1ThresholdOptimizer()
            result = optimizer.optimize(fused_probs, labels)
            if result.optimal_macro_f1 > best_f1:
                best_f1 = result.optimal_macro_f1
                best_weight = w
        weights = [best_weight, 1 - best_weight]
    
    fused_probs = weights[0] * ssl_probs + weights[1] * ecapa_probs
    
    # Optimize threshold
    optimizer = MacroF1ThresholdOptimizer()
    result = optimizer.optimize(fused_probs, labels)
    best_threshold = result.optimal_threshold
    predictions = (fused_probs >= best_threshold).astype(int)
    
    metrics = compute_metrics(labels, predictions, fused_probs)
    metrics['fusion_weights'] = {'ssl': weights[0], 'ecapa': weights[1]}
    
    return metrics


def cross_validate_evaluation(
    config: DepressionDetectionConfig,
    checkpoint_dir: str,
    device: str
) -> Dict:
    """Run cross-validation evaluation on trained models."""
    
    # Check what checkpoints exist
    ssl_folds = []
    ecapa_folds = []
    
    for fold_idx in range(config.training.n_folds):
        ssl_path = os.path.join(checkpoint_dir, f'ssl_fold_{fold_idx}', 'best_model.pt')
        ecapa_path = os.path.join(checkpoint_dir, f'ecapa_fold_{fold_idx}', 'best_model.pt')
        
        if os.path.exists(ssl_path):
            ssl_folds.append(fold_idx)
        if os.path.exists(ecapa_path):
            ecapa_folds.append(fold_idx)
    
    logger.info(f"Found SSL checkpoints for folds: {ssl_folds}")
    logger.info(f"Found ECAPA checkpoints for folds: {ecapa_folds}")
    
    # Build dataset manifest
    chunks, speaker_map = build_dataset_manifest(
        config.paths,
        config.preprocessing,
        language=config.language.value,
        is_training=True
    )
    
    # Create CV splitter
    cv = StratifiedGroupKFold(
        n_splits=config.training.n_folds,
        shuffle=True,
        random_state=config.training.cv_random_seed
    )
    splits = list(cv.split(chunks))
    
    results = {
        'ssl': {'fold_metrics': [], 'mean_f1': None, 'std_f1': None},
        'ecapa': {'fold_metrics': [], 'mean_f1': None, 'std_f1': None},
        'ensemble': {'fold_metrics': [], 'mean_f1': None, 'std_f1': None}
    }
    
    # Evaluate SSL folds
    for fold_idx in ssl_folds:
        logger.info(f"\nEvaluating SSL Fold {fold_idx}")
        
        # Load model
        model = create_ssl_model(config)
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, f'ssl_fold_{fold_idx}', 'best_model.pt'),
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create validation dataloader
        _, val_loader, _ = create_dataloaders(
            config,
            fold_idx=fold_idx,
            language=config.language.value
        )
        
        # Evaluate
        eval_results = evaluate_ssl_model(model, val_loader, device)
        metrics = compute_metrics(
            eval_results['labels'],
            eval_results['predictions'],
            eval_results['probabilities']
        )
        metrics['fold'] = fold_idx
        results['ssl']['fold_metrics'].append(metrics)
        
        logger.info(f"SSL Fold {fold_idx} Macro-F1: {metrics['macro_f1']:.4f} (optimized: {metrics.get('optimized_macro_f1', 'N/A')})")
    
    # Evaluate ECAPA folds
    for fold_idx in ecapa_folds:
        logger.info(f"\nEvaluating ECAPA Fold {fold_idx}")
        
        # Load model
        model = create_ecapa_model(config)
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, f'ecapa_fold_{fold_idx}', 'best_model.pt'),
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create validation dataloader
        train_loader, val_loader = create_mel_dataloaders(
            config,
            fold_idx=fold_idx,
            language=config.language.value
        )
        
        # Evaluate
        eval_results = evaluate_ecapa_model(model, val_loader, device)
        metrics = compute_metrics(
            eval_results['labels'],
            eval_results['predictions'],
            eval_results['probabilities']
        )
        metrics['fold'] = fold_idx
        results['ecapa']['fold_metrics'].append(metrics)
        
        logger.info(f"ECAPA Fold {fold_idx} Macro-F1: {metrics['macro_f1']:.4f} (optimized: {metrics.get('optimized_macro_f1', 'N/A')})")
    
    # Compute aggregated metrics
    if results['ssl']['fold_metrics']:
        ssl_f1s = [m['macro_f1'] for m in results['ssl']['fold_metrics']]
        results['ssl']['mean_f1'] = np.mean(ssl_f1s)
        results['ssl']['std_f1'] = np.std(ssl_f1s)
        
        ssl_opt_f1s = [m.get('optimized_macro_f1', m['macro_f1']) for m in results['ssl']['fold_metrics']]
        results['ssl']['mean_optimized_f1'] = np.mean(ssl_opt_f1s)
    
    if results['ecapa']['fold_metrics']:
        ecapa_f1s = [m['macro_f1'] for m in results['ecapa']['fold_metrics']]
        results['ecapa']['mean_f1'] = np.mean(ecapa_f1s)
        results['ecapa']['std_f1'] = np.std(ecapa_f1s)
        
        ecapa_opt_f1s = [m.get('optimized_macro_f1', m['macro_f1']) for m in results['ecapa']['fold_metrics']]
        results['ecapa']['mean_optimized_f1'] = np.mean(ecapa_opt_f1s)
    
    return results


def print_results(results: Dict):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    if results['ssl']['fold_metrics']:
        print("\n📊 SSL Model (IndicWav2Vec/Wav2Vec2)")
        print("-" * 50)
        for m in results['ssl']['fold_metrics']:
            opt_f1 = m.get('optimized_macro_f1')
            opt_str = f"{opt_f1:.4f}" if isinstance(opt_f1, float) else "N/A"
            print(f"  Fold {m['fold']}: Macro-F1 = {m['macro_f1']:.4f} | Optimized = {opt_str}")
        print(f"\n  Mean Macro-F1: {results['ssl']['mean_f1']:.4f} ± {results['ssl']['std_f1']:.4f}")
        if results['ssl'].get('mean_optimized_f1'):
            print(f"  Mean Optimized F1: {results['ssl']['mean_optimized_f1']:.4f}")
    
    if results['ecapa']['fold_metrics']:
        print("\n📊 ECAPA-TDNN Model")
        print("-" * 50)
        for m in results['ecapa']['fold_metrics']:
            opt_f1 = m.get('optimized_macro_f1')
            opt_str = f"{opt_f1:.4f}" if isinstance(opt_f1, float) else "N/A"
            print(f"  Fold {m['fold']}: Macro-F1 = {m['macro_f1']:.4f} | Optimized = {opt_str}")
        print(f"\n  Mean Macro-F1: {results['ecapa']['mean_f1']:.4f} ± {results['ecapa']['std_f1']:.4f}")
        if results['ecapa'].get('mean_optimized_f1'):
            print(f"  Mean Optimized F1: {results['ecapa']['mean_optimized_f1']:.4f}")
    
    if results['ensemble']['fold_metrics']:
        print("\n📊 Ensemble (SSL + ECAPA)")
        print("-" * 50)
        for m in results['ensemble']['fold_metrics']:
            print(f"  Fold {m['fold']}: Macro-F1 = {m['macro_f1']:.4f}")
        print(f"\n  Mean Macro-F1: {results['ensemble']['mean_f1']:.4f} ± {results['ensemble']['std_f1']:.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Depression Detection Models')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing model checkpoints')
    parser.add_argument('--language', type=str, default='combined',
                        choices=['tamil', 'malayalam', 'combined'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results JSON')
    
    args = parser.parse_args()
    
    # Get config
    config = get_config(args.language)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Find checkpoints directory
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.endswith('checkpoints'):
        checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
    
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    # Run evaluation
    results = cross_validate_evaluation(config, checkpoint_dir, device)
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(checkpoint_dir), 'evaluation_results.json')
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
