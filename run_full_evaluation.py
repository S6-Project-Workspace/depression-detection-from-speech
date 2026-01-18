#!/usr/bin/env python3
"""
Full Model Evaluation Script

This script runs cross-validation for all model variants and generates
a comprehensive comparison table.

Task 14.1: Run full cross-validation with all model variants
- Audio-only (SSL + ECAPA + Ensemble)
- Text-only (TF-IDF)
- Text-only (MuRIL)
- Multimodal (Audio + Text)

Requirements: 7.3, 7.4
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Local imports
from config import get_config, print_config_summary
from train import (
    train_ssl_model_cv,
    train_ecapa_model_cv,
    train_multimodal_model_cv,
    train_text_only_model_cv,
    evaluate_ensemble,
    set_seed,
    get_device
)
from tfidf_classifier import TFIDFClassifier
from text_preprocessing import create_preprocessor
from multimodal_dataset import load_transcripts_from_directory
from dataset import build_dataset_manifest, StratifiedGroupKFold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_tfidf_baseline_cv(
    config,
    transcript_dir: str,
    output_dir: str
) -> Dict:
    """Train TF-IDF baseline with cross-validation.
    
    Args:
        config: DepressionDetectionConfig
        transcript_dir: Directory containing transcript JSON files
        output_dir: Output directory for results
        
    Returns:
        Dictionary with fold results and aggregated metrics
    """
    logger.info("=" * 60)
    logger.info("Training TF-IDF Baseline")
    logger.info("=" * 60)
    
    # Load transcripts
    transcripts = load_transcripts_from_directory(transcript_dir)
    logger.info(f"Loaded {len(transcripts)} transcripts")
    
    # Build dataset for speaker mapping
    chunks, speaker_map = build_dataset_manifest(
        config.paths,
        config.preprocessing,
        language=config.language.value,
        is_training=True
    )
    
    # Create text preprocessor
    text_preprocessor = create_preprocessor(config.text_preprocessing)
    
    # Prepare text data
    texts = []
    labels = []
    speaker_ids = []
    
    for chunk in chunks:
        audio_filename = os.path.basename(chunk.audio_path)
        if audio_filename in transcripts:
            transcript = transcripts[audio_filename]
            
            # Preprocess text
            processed_text = text_preprocessor.preprocess(transcript.text)
            
            texts.append(processed_text)
            labels.append(chunk.label)
            speaker_ids.append(chunk.speaker_id)
    
    logger.info(f"Prepared {len(texts)} text samples for TF-IDF training")
    
    # Cross-validation splits (speaker-independent)
    cv = StratifiedGroupKFold(
        n_splits=config.training.n_folds,
        shuffle=True,
        random_state=config.training.cv_random_seed
    )
    
    # Create dummy chunks for splitting
    dummy_chunks = [type('obj', (object,), {
        'label': labels[i],
        'speaker_id': speaker_ids[i]
    })() for i in range(len(texts))]
    
    splits = cv.split(dummy_chunks)
    
    fold_results = []
    all_val_predictions = []
    all_val_labels = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        logger.info(f"\n{'='*40}")
        logger.info(f"TF-IDF Fold {fold_idx + 1}/{config.training.n_folds}")
        logger.info(f"{'='*40}")
        
        # Split data
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        # Create and train TF-IDF classifier
        classifier = TFIDFClassifier(config.tfidf)
        classifier.fit(train_texts, train_labels)
        
        # Evaluate
        val_predictions = classifier.predict(val_texts)
        val_probabilities = classifier.predict_proba(val_texts)
        
        # Compute metrics
        from sklearn.metrics import classification_report, f1_score
        
        macro_f1 = f1_score(val_labels, val_predictions, average='macro')
        report = classification_report(val_labels, val_predictions, output_dict=True)
        
        fold_result = {
            'fold_idx': fold_idx,
            'macro_f1': macro_f1,
            'f1_depressed': report['1']['f1-score'],
            'f1_non_depressed': report['0']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall']
        }
        
        fold_results.append(fold_result)
        all_val_predictions.append(val_probabilities)
        all_val_labels.append(val_labels)
        
        logger.info(f"Fold {fold_idx + 1} Macro-F1: {macro_f1:.4f}")
        
        # Save fold model
        fold_checkpoint_dir = os.path.join(output_dir, 'checkpoints', f'tfidf_fold_{fold_idx}')
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        classifier.save(os.path.join(fold_checkpoint_dir, 'tfidf_model.pkl'))
    
    # Aggregate results
    mean_f1 = np.mean([r['macro_f1'] for r in fold_results])
    std_f1 = np.std([r['macro_f1'] for r in fold_results])
    
    logger.info(f"\nTF-IDF Cross-Validation Results:")
    logger.info(f"  Mean Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    return {
        'fold_results': fold_results,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'all_predictions': all_val_predictions,
        'all_labels': all_val_labels
    }


def generate_comparison_table(
    results: Dict[str, Dict],
    output_dir: str
) -> pd.DataFrame:
    """Generate comparison table for all model variants.
    
    Args:
        results: Dictionary mapping model names to their results
        output_dir: Directory to save the comparison table
        
    Returns:
        DataFrame with comparison results
    """
    logger.info("=" * 60)
    logger.info("Generating Model Comparison Table")
    logger.info("=" * 60)
    
    comparison_data = []
    
    for model_name, model_results in results.items():
        if model_results is None:
            continue
            
        # Extract metrics
        mean_f1 = model_results.get('mean_f1', 0.0)
        std_f1 = model_results.get('std_f1', 0.0)
        
        # Calculate additional metrics if available
        fold_results = model_results.get('fold_results', [])
        
        if fold_results:
            # Calculate per-class F1 if available
            f1_depressed_scores = []
            f1_non_depressed_scores = []
            
            for fold in fold_results:
                if 'f1_depressed' in fold:
                    f1_depressed_scores.append(fold['f1_depressed'])
                if 'f1_non_depressed' in fold:
                    f1_non_depressed_scores.append(fold['f1_non_depressed'])
            
            mean_f1_depressed = np.mean(f1_depressed_scores) if f1_depressed_scores else 0.0
            std_f1_depressed = np.std(f1_depressed_scores) if f1_depressed_scores else 0.0
            mean_f1_non_depressed = np.mean(f1_non_depressed_scores) if f1_non_depressed_scores else 0.0
            std_f1_non_depressed = np.std(f1_non_depressed_scores) if f1_non_depressed_scores else 0.0
        else:
            mean_f1_depressed = 0.0
            std_f1_depressed = 0.0
            mean_f1_non_depressed = 0.0
            std_f1_non_depressed = 0.0
        
        comparison_data.append({
            'Model': model_name,
            'Macro-F1 (Mean)': f"{mean_f1:.4f}",
            'Macro-F1 (Std)': f"{std_f1:.4f}",
            'Macro-F1 (Mean ± Std)': f"{mean_f1:.4f} ± {std_f1:.4f}",
            'F1-Depressed (Mean)': f"{mean_f1_depressed:.4f}",
            'F1-Depressed (Std)': f"{std_f1_depressed:.4f}",
            'F1-Non-Depressed (Mean)': f"{mean_f1_non_depressed:.4f}",
            'F1-Non-Depressed (Std)': f"{std_f1_non_depressed:.4f}",
            'Num Folds': len(fold_results)
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Sort by Macro-F1 (descending)
    df['_sort_key'] = df['Macro-F1 (Mean)'].astype(float)
    df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    
    # Save to JSON for programmatic access
    json_path = os.path.join(output_dir, 'model_comparison.json')
    comparison_dict = {
        'timestamp': datetime.now().isoformat(),
        'models': comparison_data,
        'best_model': comparison_data[0]['Model'] if comparison_data else None,
        'best_macro_f1': comparison_data[0]['Macro-F1 (Mean)'] if comparison_data else None
    }
    
    with open(json_path, 'w') as f:
        json.dump(comparison_dict, f, indent=2)
    
    # Print table
    logger.info("\nModel Comparison Results:")
    logger.info("=" * 100)
    print(df.to_string(index=False))
    logger.info("=" * 100)
    
    logger.info(f"\nComparison table saved to:")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  JSON: {json_path}")
    
    return df


def run_full_evaluation(
    language: str = "combined",
    device: str = "auto",
    output_dir: Optional[str] = None,
    transcript_dir: Optional[str] = None,
    debug: bool = False,
    models: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Run full evaluation of all model variants.
    
    Args:
        language: Language to evaluate on
        device: Device to use for training
        output_dir: Output directory for results
        transcript_dir: Directory containing transcripts
        debug: Enable debug mode with smaller models
        models: List of models to evaluate (default: all)
        
    Returns:
        Dictionary mapping model names to their results
    """
    # Get configuration
    config = get_config(language)
    
    # Get device
    device = get_device(device)
    config.device = device
    logger.info(f"Using device: {device}")
    
    # Debug mode adjustments
    if debug:
        logger.info("Debug mode enabled - using smaller models and fewer folds")
        config.ssl_model.model_name = "facebook/wav2vec2-base"
        config.ssl_model.fallback_model = "facebook/wav2vec2-base"
        config.ssl_model.hidden_size = 768
        config.training.epochs = 2
        config.training.n_folds = 2
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(config.paths.output_dir, f"full_evaluation_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    config.paths.output_dir = output_dir
    config.paths.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    
    # Set transcript directory
    if transcript_dir is None:
        transcript_dir = "transcripts"
    
    # Set random seed
    set_seed(config.seed)
    
    # Print config
    print_config_summary(config)
    
    # Determine which models to evaluate
    if models is None:
        models = ['ssl', 'ecapa', 'ensemble', 'tfidf', 'muril', 'multimodal']
    
    logger.info(f"Evaluating models: {models}")
    
    # Store results
    results = {}
    
    # Train SSL model
    if 'ssl' in models:
        try:
            logger.info("\n" + "🔄 Starting SSL Model Training...")
            results['SSL (Wav2Vec2)'] = train_ssl_model_cv(config, device, output_dir)
        except Exception as e:
            logger.error(f"SSL model training failed: {e}")
            results['SSL (Wav2Vec2)'] = None
    
    # Train ECAPA model
    if 'ecapa' in models:
        try:
            logger.info("\n" + "🔄 Starting ECAPA Model Training...")
            results['ECAPA-TDNN'] = train_ecapa_model_cv(config, device, output_dir)
        except Exception as e:
            logger.error(f"ECAPA model training failed: {e}")
            results['ECAPA-TDNN'] = None
    
    # Evaluate ensemble if both SSL and ECAPA are available
    if 'ensemble' in models and 'SSL (Wav2Vec2)' in results and 'ECAPA-TDNN' in results:
        if results['SSL (Wav2Vec2)'] and results['ECAPA-TDNN']:
            try:
                logger.info("\n" + "🔄 Evaluating Ensemble...")
                ensemble_results = evaluate_ensemble(
                    results['SSL (Wav2Vec2)'],
                    results['ECAPA-TDNN'],
                    config,
                    output_dir
                )
                results['Ensemble (SSL + ECAPA)'] = {
                    'mean_f1': ensemble_results['mean_macro_f1'],
                    'std_f1': ensemble_results['std_macro_f1'],
                    'fold_results': ensemble_results['fold_results']
                }
            except Exception as e:
                logger.error(f"Ensemble evaluation failed: {e}")
                results['Ensemble (SSL + ECAPA)'] = None
    
    # Train TF-IDF baseline
    if 'tfidf' in models:
        try:
            logger.info("\n" + "🔄 Starting TF-IDF Baseline Training...")
            results['TF-IDF Baseline'] = train_tfidf_baseline_cv(config, transcript_dir, output_dir)
        except Exception as e:
            logger.error(f"TF-IDF baseline training failed: {e}")
            results['TF-IDF Baseline'] = None
    
    # Train MuRIL text-only model
    if 'muril' in models:
        try:
            logger.info("\n" + "🔄 Starting MuRIL Text-Only Training...")
            results['MuRIL (Text-Only)'] = train_text_only_model_cv(config, device, output_dir, transcript_dir)
        except Exception as e:
            logger.error(f"MuRIL text-only training failed: {e}")
            results['MuRIL (Text-Only)'] = None
    
    # Train multimodal model
    if 'multimodal' in models:
        try:
            logger.info("\n" + "🔄 Starting Multimodal Training...")
            results['Multimodal (Audio + Text)'] = train_multimodal_model_cv(config, device, output_dir, transcript_dir)
        except Exception as e:
            logger.error(f"Multimodal training failed: {e}")
            results['Multimodal (Audio + Text)'] = None
    
    # Generate comparison table
    comparison_df = generate_comparison_table(results, output_dir)
    
    # Save all results
    all_results_path = os.path.join(output_dir, 'all_results.json')
    with open(all_results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            if model_results is not None:
                serializable_results[model_name] = {
                    'mean_f1': float(model_results['mean_f1']),
                    'std_f1': float(model_results['std_f1']),
                    'num_folds': len(model_results.get('fold_results', []))
                }
            else:
                serializable_results[model_name] = None
        
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'device': device,
            'debug_mode': debug,
            'results': serializable_results
        }, f, indent=2)
    
    logger.info(f"\n✅ Full evaluation completed!")
    logger.info(f"📁 All results saved to: {output_dir}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full evaluation of all model variants"
    )
    parser.add_argument(
        '--language',
        type=str,
        default='combined',
        choices=['tamil', 'malayalam', 'combined'],
        help='Language to evaluate on'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--transcript-dir',
        type=str,
        default='transcripts',
        help='Directory containing transcript JSON files'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with smaller models'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['ssl', 'ecapa', 'ensemble', 'tfidf', 'muril', 'multimodal'],
        default=None,
        help='Models to evaluate (default: all)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_full_evaluation(
        language=args.language,
        device=args.device,
        output_dir=args.output_dir,
        transcript_dir=args.transcript_dir,
        debug=args.debug,
        models=args.models
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())