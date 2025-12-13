"""
Main Training Script for Depression Detection

This script orchestrates the complete training pipeline for the
DravidianLangTech @ ACL 2026 Shared Task on Depression Detection.

Usage:
    # Train both models with 5-fold CV
    python train.py --language combined --epochs 30
    
    # Train only SSL model
    python train.py --model ssl --language tamil --epochs 20
    
    # Train with custom config
    python train.py --config custom_config.json
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import torch

# Local imports
from config import (
    get_config,
    DepressionDetectionConfig,
    print_config_summary
)
from dataset import (
    create_dataloaders,
    create_mel_dataloaders,
    StratifiedGroupKFold
)
from preprocessing import build_dataset_manifest
from ssl_model import create_ssl_model, Wav2Vec2ForDepressionDetection
from ecapa_model import create_ecapa_model, ECAPATDNN
from trainer import (
    SSLModelTrainer,
    ECAPAModelTrainer,
    ModelTrainer,
    compute_subject_level_metrics
)
from threshold_tuner import MacroF1ThresholdOptimizer
from inference import (
    EnsembleEvaluator,
    WeightedAverageFusion,
    run_cross_validation_ensemble
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Depression Detection Models for Dravidian Languages'
    )
    
    # Data arguments
    parser.add_argument(
        '--language',
        type=str,
        default='combined',
        choices=['tamil', 'malayalam', 'combined'],
        help='Language to train on'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='both',
        choices=['ssl', 'ecapa', 'both'],
        help='Which model(s) to train'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for this experiment'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Config file (optional)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file (overrides other arguments)'
    )
    
    # Debug mode
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with smaller models'
    )
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine the best device to use."""
    if device_arg != 'auto':
        return device_arg
    
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def setup_experiment(args, config: DepressionDetectionConfig) -> str:
    """Setup experiment directory and logging."""
    # Generate experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"depression_{args.language}_{args.model}_{timestamp}"
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config.paths.output_dir, exp_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config paths
    config.paths.output_dir = output_dir
    config.paths.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    # Note: Full dataclass serialization would require custom handling
    with open(config_path, 'w') as f:
        json.dump({
            'language': args.language,
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'folds': args.folds,
            'seed': args.seed
        }, f, indent=2)
    
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Output directory: {output_dir}")
    
    return output_dir


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ssl_model_cv(
    config: DepressionDetectionConfig,
    device: str,
    output_dir: str
) -> Dict:
    """Train SSL model with cross-validation.
    
    Returns:
        Dictionary with fold results and aggregated metrics
    """
    logger.info("=" * 60)
    logger.info("Training SSL Model (IndicWav2Vec/Wav2Vec2)")
    logger.info("=" * 60)
    
    # Build dataset
    chunks, speaker_map = build_dataset_manifest(
        config.paths,
        config.preprocessing,
        language=config.language.value,
        is_training=True
    )
    
    # Cross-validation splits
    cv = StratifiedGroupKFold(
        n_splits=config.training.n_folds,
        shuffle=True,
        random_state=config.training.cv_random_seed
    )
    splits = cv.split(chunks)
    
    fold_results = []
    all_val_predictions = []
    all_val_labels = []
    all_speaker_ids = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        logger.info(f"\n{'='*40}")
        logger.info(f"SSL Fold {fold_idx + 1}/{config.training.n_folds}")
        logger.info(f"{'='*40}")
        
        # Create dataloaders for this fold
        train_loader, val_loader, _ = create_dataloaders(
            config,
            fold_idx=fold_idx,
            language=config.language.value
        )
        
        # Create model
        model = create_ssl_model(config)
        
        # Create trainer
        trainer = SSLModelTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Train
        fold_result = trainer.train()
        fold_results.append(fold_result)
        
        # Get validation predictions for ensemble
        val_metrics = trainer.validate()
        all_val_predictions.append(val_metrics['probabilities'])
        all_val_labels.append(val_metrics['labels'])
        
        # Get speaker IDs for subject-level evaluation
        val_chunks = [chunks[i] for i in val_indices]
        fold_speaker_ids = [c.speaker_id for c in val_chunks]
        all_speaker_ids.append(fold_speaker_ids)
        
        # Save fold checkpoint
        fold_checkpoint_dir = os.path.join(output_dir, 'checkpoints', f'ssl_fold_{fold_idx}')
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold_idx': fold_idx,
            'best_val_f1': fold_result['best_val_f1']
        }, os.path.join(fold_checkpoint_dir, 'best_model.pt'))
        
        logger.info(f"Fold {fold_idx + 1} Best Macro-F1: {fold_result['best_val_f1']:.4f}")
    
    # Aggregate results
    mean_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    std_f1 = np.std([r['best_val_f1'] for r in fold_results])
    
    logger.info(f"\nSSL Cross-Validation Results:")
    logger.info(f"  Mean Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    return {
        'fold_results': fold_results,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'all_predictions': all_val_predictions,
        'all_labels': all_val_labels,
        'all_speaker_ids': all_speaker_ids
    }


def train_ecapa_model_cv(
    config: DepressionDetectionConfig,
    device: str,
    output_dir: str
) -> Dict:
    """Train ECAPA-TDNN model with cross-validation.
    
    Returns:
        Dictionary with fold results and aggregated metrics
    """
    logger.info("=" * 60)
    logger.info("Training ECAPA-TDNN Model")
    logger.info("=" * 60)
    
    # Build dataset
    chunks, speaker_map = build_dataset_manifest(
        config.paths,
        config.preprocessing,
        language=config.language.value,
        is_training=True
    )
    
    # Cross-validation splits
    cv = StratifiedGroupKFold(
        n_splits=config.training.n_folds,
        shuffle=True,
        random_state=config.training.cv_random_seed
    )
    splits = cv.split(chunks)
    
    fold_results = []
    all_val_predictions = []
    all_val_labels = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        logger.info(f"\n{'='*40}")
        logger.info(f"ECAPA Fold {fold_idx + 1}/{config.training.n_folds}")
        logger.info(f"{'='*40}")
        
        # Create dataloaders for this fold
        train_loader, val_loader = create_mel_dataloaders(
            config,
            fold_idx=fold_idx,
            language=config.language.value
        )
        
        # Create model
        model = create_ecapa_model(config)
        
        # Create trainer
        trainer = ECAPAModelTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Train
        fold_result = trainer.train()
        fold_results.append(fold_result)
        
        # Get validation predictions for ensemble
        val_metrics = trainer.validate()
        all_val_predictions.append(val_metrics['probabilities'])
        all_val_labels.append(val_metrics['labels'])
        
        # Save fold checkpoint
        fold_checkpoint_dir = os.path.join(output_dir, 'checkpoints', f'ecapa_fold_{fold_idx}')
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold_idx': fold_idx,
            'best_val_f1': fold_result['best_val_f1']
        }, os.path.join(fold_checkpoint_dir, 'best_model.pt'))
        
        logger.info(f"Fold {fold_idx + 1} Best Macro-F1: {fold_result['best_val_f1']:.4f}")
    
    # Aggregate results
    mean_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    std_f1 = np.std([r['best_val_f1'] for r in fold_results])
    
    logger.info(f"\nECAPA Cross-Validation Results:")
    logger.info(f"  Mean Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    return {
        'fold_results': fold_results,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'all_predictions': all_val_predictions,
        'all_labels': all_val_labels
    }


def evaluate_ensemble(
    ssl_results: Dict,
    ecapa_results: Dict,
    config: DepressionDetectionConfig,
    output_dir: str
) -> Dict:
    """Evaluate ensemble of SSL and ECAPA models.
    
    Returns:
        Dictionary with ensemble metrics and optimal parameters
    """
    logger.info("=" * 60)
    logger.info("Evaluating Ensemble")
    logger.info("=" * 60)
    
    # Optimize fusion weights and threshold
    evaluator = EnsembleEvaluator(config)
    
    fold_ensemble_results = []
    
    for fold_idx in range(config.training.n_folds):
        ssl_probs = ssl_results['all_predictions'][fold_idx]
        ecapa_probs = ecapa_results['all_predictions'][fold_idx]
        labels = ssl_results['all_labels'][fold_idx]
        
        result = evaluator.evaluate(ssl_probs, ecapa_probs, labels)
        fold_ensemble_results.append(result)
        
        logger.info(
            f"Fold {fold_idx + 1}: Ensemble Macro-F1 = {result.macro_f1:.4f} "
            f"(weight={result.optimal_fusion_weight:.2f}, threshold={result.optimal_threshold:.3f})"
        )
    
    # Aggregate results
    mean_f1 = np.mean([r.macro_f1 for r in fold_ensemble_results])
    std_f1 = np.std([r.macro_f1 for r in fold_ensemble_results])
    mean_weight = np.mean([r.optimal_fusion_weight for r in fold_ensemble_results])
    mean_threshold = np.mean([r.optimal_threshold for r in fold_ensemble_results])
    
    logger.info(f"\nEnsemble Cross-Validation Results:")
    logger.info(f"  Mean Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"  Mean Optimal SSL Weight: {mean_weight:.3f}")
    logger.info(f"  Mean Optimal Threshold: {mean_threshold:.3f}")
    
    # Save ensemble config
    ensemble_config = {
        'mean_macro_f1': float(mean_f1),
        'std_macro_f1': float(std_f1),
        'optimal_ssl_weight': float(mean_weight),
        'optimal_threshold': float(mean_threshold),
        'fold_results': [
            {
                'macro_f1': float(r.macro_f1),
                'f1_depressed': float(r.f1_depressed),
                'f1_non_depressed': float(r.f1_non_depressed),
                'optimal_weight': float(r.optimal_fusion_weight),
                'optimal_threshold': float(r.optimal_threshold)
            }
            for r in fold_ensemble_results
        ]
    }
    
    with open(os.path.join(output_dir, 'ensemble_results.json'), 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    return ensemble_config


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Get configuration
    config = get_config(args.language)
    
    # Apply command line arguments
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.n_folds = args.folds
    config.seed = args.seed
    config.num_workers = args.num_workers
    
    # Get device
    device = get_device(args.device)
    config.device = device
    logger.info(f"Using device: {device}")
    
    # Debug mode adjustments
    if args.debug:
        logger.info("Debug mode enabled - using smaller models")
        config.ssl_model.model_name = "facebook/wav2vec2-base"
        config.ssl_model.fallback_model = "facebook/wav2vec2-base"
        config.ssl_model.hidden_size = 768
        config.training.epochs = 2
        config.training.n_folds = 2
    
    # Setup experiment
    output_dir = setup_experiment(args, config)
    
    # Set random seeds
    set_seed(config.seed)
    
    # Print config
    print_config_summary(config)
    
    # Train models
    ssl_results = None
    ecapa_results = None
    
    if args.model in ['ssl', 'both']:
        ssl_results = train_ssl_model_cv(config, device, output_dir)
    
    if args.model in ['ecapa', 'both']:
        ecapa_results = train_ecapa_model_cv(config, device, output_dir)
    
    # Ensemble evaluation
    if args.model == 'both' and ssl_results and ecapa_results:
        ensemble_results = evaluate_ensemble(
            ssl_results, ecapa_results, config, output_dir
        )
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"SSL Model Mean Macro-F1: {ssl_results['mean_f1']:.4f} ± {ssl_results['std_f1']:.4f}")
        logger.info(f"ECAPA Model Mean Macro-F1: {ecapa_results['mean_f1']:.4f} ± {ecapa_results['std_f1']:.4f}")
        logger.info(f"Ensemble Mean Macro-F1: {ensemble_results['mean_macro_f1']:.4f} ± {ensemble_results['std_macro_f1']:.4f}")
        logger.info(f"\nOptimal Ensemble Parameters:")
        logger.info(f"  SSL Weight: {ensemble_results['optimal_ssl_weight']:.3f}")
        logger.info(f"  Threshold: {ensemble_results['optimal_threshold']:.3f}")
    elif ssl_results:
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS (SSL Only)")
        logger.info("=" * 60)
        logger.info(f"SSL Model Mean Macro-F1: {ssl_results['mean_f1']:.4f} ± {ssl_results['std_f1']:.4f}")
    elif ecapa_results:
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS (ECAPA Only)")
        logger.info("=" * 60)
        logger.info(f"ECAPA Model Mean Macro-F1: {ecapa_results['mean_f1']:.4f} ± {ecapa_results['std_f1']:.4f}")
    
    logger.info(f"\nAll results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
