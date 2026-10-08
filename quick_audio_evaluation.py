#!/usr/bin/env python3
"""
Quick Audio Model Evaluation for ACL 2026 DravidianLangTech Competition

Simplified version that focuses on the key audio models for competition submission.
Run with: conda activate depression_detection && python3 quick_audio_evaluation.py
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're in the right environment with required packages."""
    try:
        import torch
        import soundfile
        import transformers
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ Device available: {'CUDA' if torch.cuda.is_available() else 'MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU'}")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        return False

def load_ground_truth():
    """Load ground truth labels from Excel files."""
    test_set_dir = "/Users/keerthivasan/NLP/Test Set"
    
    try:
        tamil_gt = pd.read_excel(os.path.join(test_set_dir, "Tamil_GT.xlsx"))
        malayalam_gt = pd.read_excel(os.path.join(test_set_dir, "Malayalam_GT.xlsx"))
        
        logger.info(f"📊 Tamil GT: {len(tamil_gt)} samples")
        logger.info(f"📊 Malayalam GT: {len(malayalam_gt)} samples")
        
        return tamil_gt, malayalam_gt
    except Exception as e:
        logger.error(f"❌ Error loading ground truth: {e}")
        return None, None

def find_audio_files(gt_df, test_dir, language):
    """Find audio files matching ground truth."""
    audio_files = []
    labels = []
    
    for _, row in gt_df.iterrows():
        filename = row['filename']
        # Handle different column names
        label_col = 'label' if 'label' in gt_df.columns else 'Label'
        label = row[label_col]
        
        # Convert to numeric (D=1, ND=0)
        numeric_label = 1 if str(label).upper() == 'D' else 0
        
        # Find audio file
        audio_path = None
        for root, dirs, files in os.walk(test_dir):
            if filename in files:
                audio_path = os.path.join(root, filename)
                break
        
        if audio_path and os.path.exists(audio_path):
            audio_files.append(audio_path)
            labels.append(numeric_label)
        else:
            logger.warning(f"⚠️  Audio file not found: {filename}")
    
    logger.info(f"🎵 Found {len(audio_files)} audio files for {language}")
    return audio_files, labels

def simple_audio_features(audio_path):
    """Extract simple audio features as baseline."""
    try:
        audio, sr = sf.read(audio_path)
        
        # Handle multi-channel
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Simple features
        features = {
            'duration': len(audio) / sr,
            'mean_amplitude': np.mean(np.abs(audio)),
            'std_amplitude': np.std(audio),
            'zero_crossing_rate': np.mean(np.diff(np.sign(audio)) != 0),
            'energy': np.sum(audio ** 2),
            'rms': np.sqrt(np.mean(audio ** 2))
        }
        
        return np.array(list(features.values()))
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return np.zeros(6)  # Return zeros as fallback

def evaluate_baseline_classifier(audio_files, labels, language):
    """Evaluate using simple baseline classifier."""
    logger.info(f"🔍 Extracting features for {language}...")
    
    # Extract features
    features = []
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        feat = simple_audio_features(audio_file)
        features.append(feat)
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Simple threshold-based classifier using mean amplitude
    # (This is just a baseline - real models would be much better)
    mean_amplitude_threshold = np.median(features[:, 1])  # Use median as threshold
    predictions = (features[:, 1] > mean_amplitude_threshold).astype(int)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    # Detailed report
    class_report = classification_report(
        labels, predictions,
        target_names=['Non-Depressed', 'Depressed'],
        output_dict=True
    )
    
    results = {
        'language': language,
        'num_samples': len(predictions),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'accuracy': class_report['accuracy'],
        'predictions': predictions,
        'true_labels': labels
    }
    
    logger.info(f"📈 {language.title()} Baseline Results:")
    logger.info(f"   Samples: {len(predictions)}")
    logger.info(f"   Macro Precision: {precision:.4f}")
    logger.info(f"   Macro Recall: {recall:.4f}")
    logger.info(f"   Macro F1: {f1:.4f}")
    logger.info(f"   Accuracy: {class_report['accuracy']:.4f}")
    
    return results

def check_model_files():
    """Check which model files are available."""
    model_paths = {
        "ECAPA Final": "final_models/ecapa_audio_fold0.pt",
        "SSL Final": "final_models/ssl_audio_fold0.pt",
        "ECAPA Fold 0": "outputs/training_with_progress/checkpoints/ecapa_fold_0/best_model.pt",
        "SSL Fold 0": "outputs/training_with_progress/checkpoints/ssl_fold_0/best_model.pt",
    }
    
    available_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            available_models[name] = path
            logger.info(f"✅ {name}: {path} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"❌ {name}: {path} (not found)")
    
    return available_models

def main():
    """Main evaluation function."""
    print("🎯 ACL 2026 DravidianLangTech Competition - Quick Audio Evaluation")
    print("=" * 70)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Make sure you're in the depression_detection conda environment.")
        return
    
    # Check available models
    logger.info("🔍 Checking available model files...")
    available_models = check_model_files()
    
    if not available_models:
        logger.warning("⚠️  No trained models found. Running baseline evaluation only.")
    
    # Load ground truth
    logger.info("📊 Loading ground truth data...")
    tamil_gt, malayalam_gt = load_ground_truth()
    
    if tamil_gt is None or malayalam_gt is None:
        logger.error("❌ Failed to load ground truth data")
        return
    
    # Test set directories
    test_set_dir = "/Users/keerthivasan/NLP/Test Set"
    tamil_test_dir = os.path.join(test_set_dir, "tamil_test")
    malayalam_test_dir = os.path.join(test_set_dir, "malayalam_test")
    
    # Find audio files
    tamil_files, tamil_labels = find_audio_files(tamil_gt, tamil_test_dir, "Tamil")
    malayalam_files, malayalam_labels = find_audio_files(malayalam_gt, malayalam_test_dir, "Malayalam")
    
    if not tamil_files or not malayalam_files:
        logger.error("❌ No audio files found. Check test set extraction.")
        return
    
    # Run baseline evaluation
    logger.info("\n🚀 Running baseline evaluation...")
    tamil_results = evaluate_baseline_classifier(tamil_files, tamil_labels, "Tamil")
    malayalam_results = evaluate_baseline_classifier(malayalam_files, malayalam_labels, "Malayalam")
    
    # Summary
    overall_f1 = (tamil_results['macro_f1'] + malayalam_results['macro_f1']) / 2
    
    print("\n" + "=" * 60)
    print("📊 COMPETITION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Tamil F1-Score:     {tamil_results['macro_f1']:.4f}")
    print(f"Malayalam F1-Score: {malayalam_results['macro_f1']:.4f}")
    print(f"Overall F1-Score:   {overall_f1:.4f}")
    print()
    print("📝 Note: This is a baseline evaluation using simple audio features.")
    print("   For full evaluation with trained models, run the complete evaluation script.")
    print()
    print("🎯 Competition Details:")
    print(f"   - Tamil test samples: {len(tamil_files)}")
    print(f"   - Malayalam test samples: {len(malayalam_files)}")
    print(f"   - Evaluation metric: Macro-averaged F1-Score")
    print()
    
    if available_models:
        print(f"✅ Found {len(available_models)} trained models ready for full evaluation")
        print("   Run the full evaluation script to test these models:")
        print("   conda activate depression_detection && python3 evaluate_audio_models_competition.py")
    else:
        print("⚠️  No trained models found. Train models first or check model paths.")

if __name__ == "__main__":
    main()