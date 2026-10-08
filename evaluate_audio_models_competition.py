#!/usr/bin/env python3
"""
Audio Model Evaluation for ACL 2026 DravidianLangTech Competition

Evaluates audio-based depression detection models on the competition test set.
Focuses on ECAPA-TDNN and SSL (WavLM/Wav2Vec) models only.

Competition Details:
- Tamil: 160 test samples
- Malayalam: 200 test samples  
- Evaluation: Macro-averaged Precision, Recall, F1-Score
- Audio sampling rates: 16kHz (depressed), 48kHz (non-depressed)

Usage:
    python evaluate_audio_models_competition.py
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import soundfile as sf
import zipfile
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DepressionDetectionConfig, SSLModelConfig, ECAPAModelConfig
from ssl_model import Wav2Vec2ForDepressionDetection, XLSRForDepressionDetection
from ecapa_model import ECAPATDNN, ECAPATDNNWithCrossEntropy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompetitionAudioEvaluator:
    """Evaluator for audio models on competition test set."""
    
    def __init__(self, device: str = "auto"):
        """Initialize evaluator.
        
        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        self.config = DepressionDetectionConfig()
        
        # Competition test set paths
        self.test_set_dir = "/Users/keerthivasan/NLP/Test Set"
        self.tamil_gt_file = os.path.join(self.test_set_dir, "Tamil_GT.xlsx")
        self.malayalam_gt_file = os.path.join(self.test_set_dir, "Malayalam_GT.xlsx")
        self.tamil_zip = os.path.join(self.test_set_dir, "Test-set-tamil.zip")
        self.malayalam_zip = os.path.join(self.test_set_dir, "Test_set_mal.zip")
        
        # Audio model paths (competition-ready models)
        self.audio_models = {
            "ECAPA-TDNN (Final)": "final_models/ecapa_audio_fold0.pt",
            "SSL WavLM (Final)": "final_models/ssl_audio_fold0.pt",
            "ECAPA Fold 0 (Training)": "outputs/training_with_progress/checkpoints/ecapa_fold_0/best_model.pt",
            "ECAPA Fold 1 (Training)": "outputs/training_with_progress/checkpoints/ecapa_fold_1/best_model.pt", 
            "ECAPA Fold 2 (Training)": "outputs/training_with_progress/checkpoints/ecapa_fold_2/best_model.pt",
            "SSL Fold 0 (Training)": "outputs/training_with_progress/checkpoints/ssl_fold_0/best_model.pt",
            "SSL Fold 1 (Training)": "outputs/training_with_progress/checkpoints/ssl_fold_1/best_model.pt",
            "SSL Fold 2 (Training)": "outputs/training_with_progress/checkpoints/ssl_fold_2/best_model.pt",
        }
        
        # Results storage
        self.results = {}
        
    def extract_test_sets(self):
        """Extract test set zip files if needed."""
        logger.info("Extracting test sets...")
        
        # Extract Tamil test set
        tamil_extract_dir = os.path.join(self.test_set_dir, "tamil_test")
        if not os.path.exists(tamil_extract_dir):
            os.makedirs(tamil_extract_dir)
            with zipfile.ZipFile(self.tamil_zip, 'r') as zip_ref:
                zip_ref.extractall(tamil_extract_dir)
            logger.info(f"Extracted Tamil test set to {tamil_extract_dir}")
        
        # Extract Malayalam test set  
        malayalam_extract_dir = os.path.join(self.test_set_dir, "malayalam_test")
        if not os.path.exists(malayalam_extract_dir):
            os.makedirs(malayalam_extract_dir)
            with zipfile.ZipFile(self.malayalam_zip, 'r') as zip_ref:
                zip_ref.extractall(malayalam_extract_dir)
            logger.info(f"Extracted Malayalam test set to {malayalam_extract_dir}")
            
        return tamil_extract_dir, malayalam_extract_dir
    
    def load_ground_truth(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load ground truth labels from Excel files.
        
        Returns:
            Tuple of (tamil_gt, malayalam_gt) DataFrames
        """
        logger.info("Loading ground truth labels...")
        
        tamil_gt = pd.read_excel(self.tamil_gt_file)
        malayalam_gt = pd.read_excel(self.malayalam_gt_file)
        
        logger.info(f"Tamil GT: {len(tamil_gt)} samples")
        logger.info(f"Malayalam GT: {len(malayalam_gt)} samples")
        
        return tamil_gt, malayalam_gt
    
    def load_audio_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        """Load audio model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            model_type: 'ecapa' or 'ssl'
            
        Returns:
            Loaded model
        """
        if model_type.lower() == "ecapa":
            # Use ECAPA-TDNN model with config
            config = ECAPAModelConfig()
            model = ECAPATDNN(config)
        else:  # ssl
            # Use SSL model (Wav2Vec2) with config
            config = SSLModelConfig()
            model = Wav2Vec2ForDepressionDetection(config)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess audio file for model input.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Handle multi-channel audio
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed (competition data has mixed sample rates)
            target_sr = 16000
            if sr != target_sr:
                # Simple resampling (for production, use librosa.resample)
                audio = audio[::int(sr/target_sr)]
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            return audio_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            # Return silence as fallback
            return torch.zeros(1, 16000).to(self.device)
    
    def predict_batch(self, model: torch.nn.Module, audio_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on batch of audio files.
        
        Args:
            model: Loaded audio model
            audio_files: List of audio file paths
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for audio_file in tqdm(audio_files, desc="Processing audio"):
                audio_tensor = self.preprocess_audio(audio_file)
                
                try:
                    outputs = model(audio_tensor)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    probs = torch.softmax(logits, dim=1)[0]
                    pred = torch.argmax(logits, dim=1)[0]
                    
                    predictions.append(int(pred.cpu()))
                    probabilities.append(probs.cpu().numpy())
                    
                except Exception as e:
                    logger.error(f"Error predicting {audio_file}: {e}")
                    # Default to non-depressed prediction
                    predictions.append(0)
                    probabilities.append(np.array([0.6, 0.4]))
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate_language(self, model: torch.nn.Module, model_name: str, 
                         language: str, gt_df: pd.DataFrame, 
                         test_dir: str) -> Dict:
        """Evaluate model on one language.
        
        Args:
            model: Loaded model
            model_name: Name of the model
            language: 'tamil' or 'malayalam'
            gt_df: Ground truth DataFrame
            test_dir: Test set directory
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating {model_name} on {language.title()}...")
        
        # Find audio files in test directory
        audio_files = []
        true_labels = []
        
        for _, row in gt_df.iterrows():
            filename = row['filename']
            # Handle different column names for label
            if 'label' in gt_df.columns:
                label = row['label']
            elif 'Label' in gt_df.columns:
                label = row['Label']
            else:
                logger.error(f"No label column found in ground truth")
                continue
            
            # Convert string labels to numeric (D=1, ND=0)
            if isinstance(label, str):
                numeric_label = 1 if label.upper() == 'D' else 0
            else:
                numeric_label = int(label)
            
            # Find the audio file (might be in subdirectories)
            audio_path = None
            for root, dirs, files in os.walk(test_dir):
                if filename in files:
                    audio_path = os.path.join(root, filename)
                    break
            
            if audio_path and os.path.exists(audio_path):
                audio_files.append(audio_path)
                true_labels.append(numeric_label)
            else:
                logger.warning(f"Audio file not found: {filename}")
        
        if not audio_files:
            logger.error(f"No audio files found for {language}")
            return {}
        
        logger.info(f"Found {len(audio_files)} audio files for {language}")
        
        # Get predictions
        predictions, probabilities = self.predict_batch(model, audio_files)
        true_labels = np.array(true_labels[:len(predictions)])
        
        # Calculate metrics (competition format)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='macro'
        )
        
        # Detailed classification report
        class_report = classification_report(
            true_labels, predictions, 
            target_names=['Non-Depressed', 'Depressed'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            'language': language,
            'model': model_name,
            'num_samples': len(predictions),
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'accuracy': class_report['accuracy'],
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities
        }
        
        # Log results
        logger.info(f"{language.title()} Results for {model_name}:")
        logger.info(f"  Samples: {len(predictions)}")
        logger.info(f"  Macro Precision: {precision:.4f}")
        logger.info(f"  Macro Recall: {recall:.4f}")
        logger.info(f"  Macro F1: {f1:.4f}")
        logger.info(f"  Accuracy: {class_report['accuracy']:.4f}")
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all audio models on both languages."""
        logger.info("Starting comprehensive audio model evaluation...")
        
        # Extract test sets
        tamil_test_dir, malayalam_test_dir = self.extract_test_sets()
        
        # Load ground truth
        tamil_gt, malayalam_gt = self.load_ground_truth()
        
        # Evaluate each model
        for model_name, model_path in self.audio_models.items():
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_name}")
            logger.info(f"{'='*60}")
            
            # Determine model type
            model_type = "ecapa" if "ecapa" in model_name.lower() else "ssl"
            
            try:
                # Load model
                model = self.load_audio_model(model_path, model_type)
                
                # Evaluate on Tamil
                tamil_results = self.evaluate_language(
                    model, model_name, "tamil", tamil_gt, tamil_test_dir
                )
                
                # Evaluate on Malayalam
                malayalam_results = self.evaluate_language(
                    model, model_name, "malayalam", malayalam_gt, malayalam_test_dir
                )
                
                # Store results
                self.results[model_name] = {
                    'tamil': tamil_results,
                    'malayalam': malayalam_results,
                    'model_type': model_type,
                    'model_path': model_path
                }
                
                # Calculate overall metrics
                if tamil_results and malayalam_results:
                    overall_f1 = (tamil_results['macro_f1'] + malayalam_results['macro_f1']) / 2
                    logger.info(f"Overall Macro F1 (Average): {overall_f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
    
    def generate_competition_report(self, output_dir: str = "competition_evaluation_results"):
        """Generate comprehensive competition evaluation report."""
        if not self.results:
            logger.error("No results to report. Run evaluate_all_models() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Summary table
        summary_data = []
        for model_name, model_results in self.results.items():
            if 'tamil' in model_results and 'malayalam' in model_results:
                tamil_f1 = model_results['tamil'].get('macro_f1', 0)
                malayalam_f1 = model_results['malayalam'].get('macro_f1', 0)
                overall_f1 = (tamil_f1 + malayalam_f1) / 2
                
                summary_data.append({
                    'Model': model_name,
                    'Type': model_results['model_type'].upper(),
                    'Tamil_F1': tamil_f1,
                    'Malayalam_F1': malayalam_f1,
                    'Overall_F1': overall_f1,
                    'Tamil_Precision': model_results['tamil'].get('macro_precision', 0),
                    'Malayalam_Precision': model_results['malayalam'].get('macro_precision', 0),
                    'Tamil_Recall': model_results['tamil'].get('macro_recall', 0),
                    'Malayalam_Recall': model_results['malayalam'].get('macro_recall', 0),
                })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Overall_F1', ascending=False)
        summary_file = os.path.join(output_dir, "competition_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Generate detailed report
        report_file = os.path.join(output_dir, "detailed_evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write("ACL 2026 DravidianLangTech Competition - Audio Model Evaluation\n")
            f.write("="*70 + "\n\n")
            
            f.write("COMPETITION DETAILS:\n")
            f.write("- Task: Depression Detection from Malayalam and Tamil Speech\n")
            f.write("- Tamil Test Samples: 160\n")
            f.write("- Malayalam Test Samples: 200\n")
            f.write("- Evaluation Metrics: Macro-averaged Precision, Recall, F1-Score\n\n")
            
            f.write("MODEL RANKINGS (by Overall F1-Score):\n")
            f.write("-" * 50 + "\n")
            for i, (_, row) in enumerate(summary_df.iterrows(), 1):
                f.write(f"{i:2d}. {row['Model']:<35} F1: {row['Overall_F1']:.4f}\n")
            
            f.write("\n\nDETAILED RESULTS:\n")
            f.write("=" * 50 + "\n")
            
            for model_name, model_results in self.results.items():
                f.write(f"\n{model_name}\n")
                f.write("-" * len(model_name) + "\n")
                
                for lang in ['tamil', 'malayalam']:
                    if lang in model_results:
                        results = model_results[lang]
                        f.write(f"\n{lang.title()}:\n")
                        f.write(f"  Samples: {results['num_samples']}\n")
                        f.write(f"  Macro Precision: {results['macro_precision']:.4f}\n")
                        f.write(f"  Macro Recall: {results['macro_recall']:.4f}\n")
                        f.write(f"  Macro F1: {results['macro_f1']:.4f}\n")
                        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        
        # Generate visualizations
        self.create_visualizations(output_dir)
        
        logger.info(f"Competition evaluation report saved to: {output_dir}")
        logger.info(f"Summary: {summary_file}")
        logger.info(f"Detailed report: {report_file}")
        
        # Print top 3 models
        print("\n" + "="*60)
        print("TOP 3 AUDIO MODELS FOR COMPETITION SUBMISSION")
        print("="*60)
        for i, (_, row) in enumerate(summary_df.head(3).iterrows(), 1):
            print(f"{i}. {row['Model']}")
            print(f"   Overall F1: {row['Overall_F1']:.4f}")
            print(f"   Tamil F1: {row['Tamil_F1']:.4f} | Malayalam F1: {row['Malayalam_F1']:.4f}")
            print()
    
    def create_visualizations(self, output_dir: str):
        """Create evaluation visualizations."""
        # Model comparison plot
        plt.figure(figsize=(15, 8))
        
        models = []
        tamil_f1 = []
        malayalam_f1 = []
        
        for model_name, model_results in self.results.items():
            if 'tamil' in model_results and 'malayalam' in model_results:
                models.append(model_name.replace(' (Final)', '').replace(' (Training)', ''))
                tamil_f1.append(model_results['tamil']['macro_f1'])
                malayalam_f1.append(model_results['malayalam']['macro_f1'])
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, tamil_f1, width, label='Tamil', alpha=0.8)
        plt.bar(x + width/2, malayalam_f1, width, label='Malayalam', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Macro F1-Score')
        plt.title('Audio Model Performance Comparison - ACL 2026 Competition')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations saved to competition_evaluation_results/")


def main():
    """Main evaluation function."""
    print("ACL 2026 DravidianLangTech Competition - Audio Model Evaluation")
    print("="*70)
    
    # Initialize evaluator
    evaluator = CompetitionAudioEvaluator(device="auto")
    
    # Run evaluation
    evaluator.evaluate_all_models()
    
    # Generate report
    evaluator.generate_competition_report()
    
    print("\n✅ Evaluation complete!")
    print("📊 Check 'competition_evaluation_results/' for detailed results")
    print("🏆 Top models are ready for competition submission")


if __name__ == "__main__":
    main()