"""
Complete Training Script for NLP Multimodal Depression Detection

This script trains the full multimodal system:
1. Text Model (MuRIL) - trained on transcripts
2. TF-IDF Baseline - for comparison
3. Enhanced Multimodal (Audio + Text + Linguistic Features)

Usage:
    python train_nlp_multimodal.py --mode all --epochs 10 --batch-size 16
    python train_nlp_multimodal.py --mode text_only --epochs 5
    python train_nlp_multimodal.py --mode enhanced --epochs 15
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

# Import project modules
from config import get_config, DepressionDetectionConfig
from text_model import create_text_model
from text_preprocessing import TextPreprocessor
from tfidf_classifier import TFIDFClassifier
from linguistic_analyzer import create_linguistic_analyzer
from multimodal_dataset import MultimodalDataset, create_multimodal_dataloaders
from multimodal_trainer import MultimodalTrainer, create_multimodal_trainer
from enhanced_multimodal_trainer import EnhancedMultimodalTrainer, create_enhanced_multimodal_trainer
from ssl_model import create_ssl_model
from ecapa_model import create_ecapa_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train NLP Multimodal Depression Detection System'
    )
    
    # Training mode
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['text_only', 'tfidf', 'multimodal', 'enhanced', 'all'],
        help='Training mode: text_only, tfidf, multimodal, enhanced, or all'
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default='NLP Dataset',
        help='Path to audio dataset directory'
    )
    
    parser.add_argument(
        '--transcript-dir',
        type=str,
        default='transcripts',
        help='Path to transcript directory'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate for text model'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=3,
        help='Number of cross-validation folds'
    )
    
    # Model parameters
    parser.add_argument(
        '--audio-model',
        type=str,
        default='ssl',
        choices=['ssl', 'ecapa', 'both'],
        help='Audio model to use: ssl (Wav2Vec2), ecapa (ECAPA-TDNN), or both'
    )
    
    parser.add_argument(
        '--use-linguistic',
        action='store_true',
        help='Use linguistic features (POS, NER, DEP)'
    )
    
    parser.add_argument(
        '--progressive-training',
        action='store_true',
        help='Use progressive training (baseline → enhanced)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/nlp_multimodal_training',
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine the best available device."""
    if device_arg != 'auto':
        return device_arg
    
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_transcripts(transcript_dir: str, language: str) -> Dict[str, str]:
    """Load transcripts from JSON files.
    
    Args:
        transcript_dir: Directory containing transcript JSON files
        language: 'tamil' or 'malayalam'
        
    Returns:
        Dictionary mapping audio_id to transcript text
    """
    transcripts = {}
    
    # Try different directory structures
    lang_dir = Path(transcript_dir) / language
    
    # Structure 1: depressed_transcripts.json and non_depressed_transcripts.json
    depressed_file = lang_dir / 'depressed_transcripts.json'
    non_depressed_file = lang_dir / 'non_depressed_transcripts.json'
    
    if depressed_file.exists():
        with open(depressed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                audio_id = item['audio_id']
                text = item['text']
                transcripts[audio_id] = text
    
    if non_depressed_file.exists():
        with open(non_depressed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                audio_id = item['audio_id']
                text = item['text']
                transcripts[audio_id] = text
    
    # Structure 2: depressed/ and non_depressed/ subdirectories with individual JSON files
    if len(transcripts) == 0:
        for subdir in ['depressed', 'non_depressed']:
            subdir_path = lang_dir / subdir
            if subdir_path.exists():
                for json_file in subdir_path.glob('*.json'):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Handle both single item and list formats
                            if isinstance(data, list):
                                for item in data:
                                    audio_id = item.get('audio_id', json_file.stem)
                                    text = item.get('text', '')
                                    if text:
                                        transcripts[audio_id] = text
                            elif isinstance(data, dict):
                                audio_id = data.get('audio_id', json_file.stem)
                                text = data.get('text', '')
                                if text:
                                    transcripts[audio_id] = text
                    except Exception as e:
                        logger.warning(f"Failed to load {json_file}: {e}")
    
    logger.info(f"Loaded {len(transcripts)} transcripts for {language}")
    return transcripts


def train_tfidf_baseline(
    config: DepressionDetectionConfig,
    transcript_dir: str,
    output_dir: str,
    n_folds: int = 3
) -> Dict:
    """Train TF-IDF baseline classifier.
    
    Args:
        config: Configuration object
        transcript_dir: Directory containing transcripts
        output_dir: Output directory for results
        n_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with training results
    """
    logger.info("\n" + "="*60)
    logger.info("Training TF-IDF Baseline Classifier")
    logger.info("="*60)
    
    # Load transcripts for both languages
    tamil_transcripts = load_transcripts(transcript_dir, 'tamil')
    malayalam_transcripts = load_transcripts(transcript_dir, 'malayalam')
    
    # Combine transcripts
    all_texts = []
    all_labels = []
    all_audio_ids = []
    
    for audio_id, text in tamil_transcripts.items():
        all_texts.append(text)
        # Determine label from audio_id (D_ prefix = depressed)
        label = 1 if audio_id.startswith('D_') else 0
        all_labels.append(label)
        all_audio_ids.append(audio_id)
    
    for audio_id, text in malayalam_transcripts.items():
        all_texts.append(text)
        label = 1 if audio_id.startswith('D_') else 0
        all_labels.append(label)
        all_audio_ids.append(audio_id)
    
    logger.info(f"Total samples: {len(all_texts)}")
    logger.info(f"Depressed: {sum(all_labels)}, Non-Depressed: {len(all_labels) - sum(all_labels)}")
    
    # Create TF-IDF classifier
    tfidf_classifier = TFIDFClassifier(config=config.tfidf)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Split data
        train_texts = [all_texts[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_texts = [all_texts[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        # Train
        tfidf_classifier.fit(train_texts, train_labels)
        
        # Evaluate
        val_metrics = tfidf_classifier.evaluate(val_texts, val_labels)
        
        logger.info(f"Fold {fold + 1} Results:")
        logger.info(f"  Macro-F1: {val_metrics['macro_f1']:.4f}")
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  F1 Depressed: {val_metrics['f1_depressed']:.4f}")
        logger.info(f"  F1 Non-Depressed: {val_metrics['f1_non_depressed']:.4f}")
        
        fold_results.append(val_metrics)
        
        # Save model for this fold
        fold_output_dir = Path(output_dir) / 'tfidf' / f'fold_{fold}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        tfidf_classifier.save(str(fold_output_dir / 'tfidf_model.pkl'))
    
    # Compute average metrics
    avg_metrics = {
        'macro_f1': np.mean([r['macro_f1'] for r in fold_results]),
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'f1_depressed': np.mean([r['f1_depressed'] for r in fold_results]),
        'f1_non_depressed': np.mean([r['f1_non_depressed'] for r in fold_results])
    }
    
    logger.info("\n" + "="*60)
    logger.info("TF-IDF Baseline - Average Results")
    logger.info("="*60)
    logger.info(f"Macro-F1: {avg_metrics['macro_f1']:.4f} ± {np.std([r['macro_f1'] for r in fold_results]):.4f}")
    logger.info(f"Accuracy: {avg_metrics['accuracy']:.4f} ± {np.std([r['accuracy'] for r in fold_results]):.4f}")
    logger.info(f"F1 Depressed: {avg_metrics['f1_depressed']:.4f}")
    logger.info(f"F1 Non-Depressed: {avg_metrics['f1_non_depressed']:.4f}")
    
    # Save results
    results = {
        'fold_results': fold_results,
        'average_metrics': avg_metrics,
        'config': {
            'ngram_range': config.tfidf.ngram_range,
            'max_features': config.tfidf.max_features,
            'classifier': config.tfidf.classifier
        }
    }
    
    results_file = Path(output_dir) / 'tfidf' / 'results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def train_text_model(
    config: DepressionDetectionConfig,
    transcript_dir: str,
    output_dir: str,
    device: str,
    epochs: int = 10,
    batch_size: int = 16,
    n_folds: int = 3
) -> Dict:
    """Train MuRIL text model.
    
    Args:
        config: Configuration object
        transcript_dir: Directory containing transcripts
        output_dir: Output directory for checkpoints
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size
        n_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with training results
    """
    logger.info("\n" + "="*60)
    logger.info("Training MuRIL Text Model")
    logger.info("="*60)
    
    # Load transcripts
    tamil_transcripts = load_transcripts(transcript_dir, 'tamil')
    malayalam_transcripts = load_transcripts(transcript_dir, 'malayalam')
    
    # Prepare data
    all_texts = []
    all_labels = []
    all_audio_ids = []
    
    for audio_id, text in tamil_transcripts.items():
        all_texts.append(text)
        label = 1 if audio_id.startswith('D_') else 0
        all_labels.append(label)
        all_audio_ids.append(audio_id)
    
    for audio_id, text in malayalam_transcripts.items():
        all_texts.append(text)
        label = 1 if audio_id.startswith('D_') else 0
        all_labels.append(label)
        all_audio_ids.append(audio_id)
    
    logger.info(f"Total samples: {len(all_texts)}")
    
    # Create text preprocessor
    preprocessor = TextPreprocessor(language='ta')  # MuRIL handles both Tamil and Malayalam
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Create model
        text_model = create_text_model(config.text_model).to(device)
        
        # Prepare datasets
        train_texts = [all_texts[i] for i in train_idx]
        train_labels = torch.tensor([all_labels[i] for i in train_idx])
        val_texts = [all_texts[i] for i in val_idx]
        val_labels = torch.tensor([all_labels[i] for i in val_idx])
        
        # Create dataloaders
        from torch.utils.data import TensorDataset, DataLoader
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(config.text_model.model_name)
        
        # Tokenize
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=config.text_model.max_length,
            return_tensors='pt'
        )
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=config.text_model.max_length,
            return_tensors='pt'
        )
        
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            val_labels
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create trainer
        from trainer import ModelTrainer
        
        trainer = ModelTrainer(
            model=text_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Train
        config.training.epochs = epochs
        results = trainer.train()
        
        logger.info(f"Fold {fold + 1} Best Macro-F1: {results['best_val_f1']:.4f}")
        
        fold_results.append({
            'fold': fold,
            'best_val_f1': results['best_val_f1'],
            'final_epoch': results['final_epoch']
        })
        
        # Save model
        fold_output_dir = Path(output_dir) / 'text_model' / f'fold_{fold}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': text_model.state_dict(),
            'config': config.text_model,
            'results': results
        }, fold_output_dir / 'best_model.pt')
    
    # Compute average
    avg_f1 = np.mean([r['best_val_f1'] for r in fold_results])
    std_f1 = np.std([r['best_val_f1'] for r in fold_results])
    
    logger.info("\n" + "="*60)
    logger.info("Text Model - Average Results")
    logger.info("="*60)
    logger.info(f"Macro-F1: {avg_f1:.4f} ± {std_f1:.4f}")
    
    return {
        'fold_results': fold_results,
        'average_f1': avg_f1,
        'std_f1': std_f1
    }


def train_multimodal_system(
    config: DepressionDetectionConfig,
    data_dir: str,
    transcript_dir: str,
    output_dir: str,
    device: str,
    audio_model_type: str = 'ssl',
    use_linguistic: bool = False,
    progressive_training: bool = False,
    epochs: int = 15,
    batch_size: int = 16,
    n_folds: int = 3
) -> Dict:
    """Train full multimodal system (Audio + Text + Linguistic).
    
    Args:
        config: Configuration object
        data_dir: Audio dataset directory
        transcript_dir: Transcript directory
        output_dir: Output directory
        device: Device to train on
        audio_model_type: 'ssl', 'ecapa', or 'both'
        use_linguistic: Whether to use linguistic features
        progressive_training: Whether to use progressive training
        epochs: Number of epochs
        batch_size: Batch size
        n_folds: Number of folds
        
    Returns:
        Dictionary with training results
    """
    logger.info("\n" + "="*60)
    if use_linguistic:
        logger.info("Training Enhanced Multimodal System (Audio + Text + Linguistic)")
    else:
        logger.info("Training Multimodal System (Audio + Text)")
    logger.info("="*60)
    
    # Load transcripts
    tamil_transcripts = load_transcripts(transcript_dir, 'tamil')
    malayalam_transcripts = load_transcripts(transcript_dir, 'malayalam')
    all_transcripts = {**tamil_transcripts, **malayalam_transcripts}
    
    logger.info(f"Loaded {len(all_transcripts)} transcripts")
    
    # Create audio model
    if audio_model_type == 'ssl':
        audio_model = create_ssl_model(config).to(device)
        logger.info("Using SSL (Wav2Vec2) audio model")
    elif audio_model_type == 'ecapa':
        audio_model = create_ecapa_model(config).to(device)
        logger.info("Using ECAPA-TDNN audio model")
    else:
        raise ValueError(f"Unsupported audio model type: {audio_model_type}")
    
    # Create text model
    text_model = create_text_model(config.text_model).to(device)
    logger.info("Created MuRIL text model")
    
    # Create linguistic analyzer if needed
    linguistic_analyzer = None
    if use_linguistic:
        linguistic_analyzer = create_linguistic_analyzer()
        logger.info("Created linguistic analyzer")
    
    # Create dataloaders
    logger.info("Creating multimodal dataloaders...")
    train_loader, val_loader = create_multimodal_dataloaders(
        data_dir=data_dir,
        transcripts=all_transcripts,
        config=config,
        batch_size=batch_size,
        use_linguistic=use_linguistic
    )
    
    # Create trainer
    if use_linguistic:
        trainer = create_enhanced_multimodal_trainer(
            audio_model=audio_model,
            text_model=text_model,
            linguistic_analyzer=linguistic_analyzer,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            progressive_training=progressive_training
        )
    else:
        trainer = create_multimodal_trainer(
            audio_model=audio_model,
            text_model=text_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
    
    # Train
    config.training.epochs = epochs
    results = trainer.train()
    
    # Save model
    model_output_dir = Path(output_dir) / ('enhanced_multimodal' if use_linguistic else 'multimodal')
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config,
        'results': results
    }, model_output_dir / 'best_model.pt')
    
    # Save results
    with open(model_output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n" + "="*60)
    logger.info("Multimodal Training Complete")
    logger.info("="*60)
    logger.info(f"Best Macro-F1: {results['best_val_f1']:.4f}")
    
    return results


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    set_seed(args.seed)
    
    logger.info("\n" + "="*60)
    logger.info("NLP Multimodal Depression Detection Training")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"N-Folds: {args.n_folds}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = get_config('combined')
    config.device = device
    config.training.batch_size = args.batch_size
    config.text_model.learning_rate = args.learning_rate
    
    # Training results
    all_results = {}
    
    # Train based on mode
    if args.mode in ['tfidf', 'all']:
        logger.info("\n🚀 Starting TF-IDF Baseline Training...")
        tfidf_results = train_tfidf_baseline(
            config=config,
            transcript_dir=args.transcript_dir,
            output_dir=args.output_dir,
            n_folds=args.n_folds
        )
        all_results['tfidf'] = tfidf_results
    
    if args.mode in ['text_only', 'all']:
        logger.info("\n🚀 Starting Text Model Training...")
        text_results = train_text_model(
            config=config,
            transcript_dir=args.transcript_dir,
            output_dir=args.output_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_folds=args.n_folds
        )
        all_results['text_model'] = text_results
    
    if args.mode in ['multimodal', 'all']:
        logger.info("\n🚀 Starting Multimodal Training (Audio + Text)...")
        multimodal_results = train_multimodal_system(
            config=config,
            data_dir=args.data_dir,
            transcript_dir=args.transcript_dir,
            output_dir=args.output_dir,
            device=device,
            audio_model_type=args.audio_model,
            use_linguistic=False,
            progressive_training=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_folds=args.n_folds
        )
        all_results['multimodal'] = multimodal_results
    
    if args.mode in ['enhanced', 'all']:
        logger.info("\n🚀 Starting Enhanced Multimodal Training (Audio + Text + Linguistic)...")
        enhanced_results = train_multimodal_system(
            config=config,
            data_dir=args.data_dir,
            transcript_dir=args.transcript_dir,
            output_dir=args.output_dir,
            device=device,
            audio_model_type=args.audio_model,
            use_linguistic=True,
            progressive_training=args.progressive_training,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_folds=args.n_folds
        )
        all_results['enhanced_multimodal'] = enhanced_results
    
    # Save all results
    final_results_file = output_dir / 'training_results.json'
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("\n" + "="*60)
    logger.info("✅ All Training Complete!")
    logger.info("="*60)
    logger.info(f"Results saved to: {final_results_file}")
    
    # Print summary
    logger.info("\n📊 Training Summary:")
    for model_name, results in all_results.items():
        if 'average_f1' in results:
            logger.info(f"  {model_name}: Macro-F1 = {results['average_f1']:.4f}")
        elif 'best_val_f1' in results:
            logger.info(f"  {model_name}: Macro-F1 = {results['best_val_f1']:.4f}")
        elif 'average_metrics' in results:
            logger.info(f"  {model_name}: Macro-F1 = {results['average_metrics']['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
