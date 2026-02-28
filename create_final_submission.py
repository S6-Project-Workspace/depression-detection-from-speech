"""
Final Competition Submission Generator
Uses best trained models (ECAPA fold 0) for predictions

DravidianLangTech@ACL 2026 - Depression Detection
Deadline: February 10, 2026
"""

import os
import csv
import zipfile
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import logging
import torchaudio

from config import DepressionDetectionConfig
from ecapa_model import ECAPATDNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCompetitionSubmissionGenerator:
    """Generate competition submission using trained ECAPA model."""
    
    def __init__(
        self,
        team_name: str,
        model_checkpoint: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            team_name: Your team name for submission
            model_checkpoint: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.team_name = team_name
        self.device = device
        self.config = DepressionDetectionConfig()
        
        # Load model
        logger.info(f"Loading model from {model_checkpoint}")
        self.model = self._load_model(model_checkpoint)
        self.model.eval()
        
        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.preprocessing.sampling_rate,
            n_mels=self.config.ecapa_model.n_mels,
            n_fft=int(self.config.ecapa_model.window_size_ms * 
                     self.config.preprocessing.sampling_rate / 1000),
            hop_length=int(self.config.ecapa_model.hop_size_ms * 
                          self.config.preprocessing.sampling_rate / 1000)
        ).to(device)
        
        # Optimal threshold from evaluation
        self.threshold = 0.23  # From ECAPA fold 0 evaluation
        
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Using threshold: {self.threshold}")
    
    def _load_model(self, checkpoint_path: str) -> ECAPATDNN:
        """Load ECAPA model from checkpoint."""
        model = ECAPATDNN(self.config.ecapa_model).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _load_and_preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        try:
            import soundfile as sf
            import librosa
            
            # Load audio using soundfile
            try:
                audio_data, sample_rate = sf.read(audio_path, dtype='float32')
            except:
                # Fallback to librosa
                audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            
            # Convert to torch tensor
            if len(audio_data.shape) == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_data.T)
            
            # Resample if needed
            if sample_rate != self.config.preprocessing.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 
                    self.config.preprocessing.sampling_rate
                )
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Normalize
            waveform = waveform - torch.mean(waveform)
            max_amp = torch.max(torch.abs(waveform))
            if max_amp > 0:
                waveform = waveform * (0.95 / max_amp)
            
            return waveform.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            return None
    
    def _extract_mel_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract Mel spectrogram features."""
        # Compute Mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Apply CMVN
        mel_spec = mel_spec - mel_spec.mean(dim=-1, keepdim=True)
        mel_spec = mel_spec / (mel_spec.std(dim=-1, keepdim=True) + 1e-8)
        
        return mel_spec
    
    @torch.no_grad()
    def predict_file(self, audio_path: str) -> Dict:
        """Predict depression for a single audio file."""
        # Load and preprocess
        waveform = self._load_and_preprocess_audio(audio_path)
        
        if waveform is None:
            # Return default prediction for failed files
            return {
                'id': Path(audio_path).stem,
                'label': 'Non-Depressed',
                'probability': 0.0
            }
        
        # Extract features
        mel_spec = self._extract_mel_features(waveform)
        
        # Get prediction
        logits = self.model(mel_spec)
        probs = F.softmax(logits, dim=1)
        prob_depressed = probs[0, 1].item()
        
        # Apply threshold
        predicted_label = 1 if prob_depressed >= self.threshold else 0
        label_text = 'Depressed' if predicted_label == 1 else 'Non-Depressed'
        
        return {
            'id': Path(audio_path).stem,
            'label': label_text,
            'probability': prob_depressed
        }
    
    def predict_language_dataset(
        self,
        data_dir: str,
        language: str
    ) -> List[Dict]:
        """Generate predictions for a language dataset."""
        logger.info(f"Processing {language} dataset from {data_dir}")
        
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(Path(data_dir).rglob(ext))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        predictions = []
        for i, audio_file in enumerate(audio_files):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing {i + 1}/{len(audio_files)}...")
            
            try:
                result = self.predict_file(str(audio_file))
                predictions.append(result)
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                # Add default prediction
                predictions.append({
                    'id': audio_file.stem,
                    'label': 'Non-Depressed',
                    'probability': 0.0
                })
        
        logger.info(f"Generated {len(predictions)} predictions for {language}")
        
        # Log prediction distribution
        depressed_count = sum(1 for p in predictions if p['label'] == 'Depressed')
        logger.info(f"  Depressed: {depressed_count}")
        logger.info(f"  Non-Depressed: {len(predictions) - depressed_count}")
        
        return predictions
    
    def save_predictions_csv(
        self,
        predictions: List[Dict],
        output_path: str
    ):
        """Save predictions to CSV file in competition format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['id', 'labels'])
            
            # Write predictions
            for pred in predictions:
                writer.writerow([pred['id'], pred['label']])
        
        logger.info(f"Saved predictions to {output_path}")
    
    def generate_submission(
        self,
        tamil_data_dir: str,
        malayalam_data_dir: str,
        output_dir: str = 'submissions'
    ):
        """Generate complete submission package."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate predictions for both languages
        logger.info("=" * 60)
        logger.info("Generating Tamil predictions...")
        logger.info("=" * 60)
        tamil_predictions = self.predict_language_dataset(
            tamil_data_dir, 'Tamil'
        )
        
        logger.info("=" * 60)
        logger.info("Generating Malayalam predictions...")
        logger.info("=" * 60)
        malayalam_predictions = self.predict_language_dataset(
            malayalam_data_dir, 'Malayalam'
        )
        
        # Save CSV files
        tamil_csv = os.path.join(output_dir, f'{self.team_name}_Tamil.csv')
        malayalam_csv = os.path.join(output_dir, f'{self.team_name}_Malayalam.csv')
        
        self.save_predictions_csv(tamil_predictions, tamil_csv)
        self.save_predictions_csv(malayalam_predictions, malayalam_csv)
        
        # Create zip file (competition format: Team_name_task.zip)
        zip_filename = f'{self.team_name}_depression.zip'
        zip_path = os.path.join(output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(tamil_csv, os.path.basename(tamil_csv))
            zipf.write(malayalam_csv, os.path.basename(malayalam_csv))
        
        logger.info("=" * 60)
        logger.info("SUBMISSION PACKAGE CREATED!")
        logger.info("=" * 60)
        logger.info(f"Location: {zip_path}")
        logger.info(f"Tamil predictions: {len(tamil_predictions)}")
        logger.info(f"Malayalam predictions: {len(malayalam_predictions)}")
        logger.info("=" * 60)
        logger.info("Files in submission:")
        logger.info(f"  - {os.path.basename(tamil_csv)}")
        logger.info(f"  - {os.path.basename(malayalam_csv)}")
        logger.info("=" * 60)
        
        return zip_path


def main():
    """Main function to generate submission."""
    
    # ===== CONFIGURATION =====
    TEAM_NAME = "ApexPatchers"  # Team name for competition submission
    
    # Best model checkpoint (ECAPA fold 0 with 0.994 F1)
    MODEL_CHECKPOINT = "outputs/training_with_progress/checkpoints/ecapa_fold_0/best_model.pt"
    
    # Data directories
    TAMIL_TEST_DIR = "NLP Dataset/Tamil"
    MALAYALAM_TEST_DIR = "NLP Dataset/Malayalam"
    
    OUTPUT_DIR = "submissions"
    # =========================
    
    # Verify model exists
    if not os.path.exists(MODEL_CHECKPOINT):
        logger.error(f"Model checkpoint not found: {MODEL_CHECKPOINT}")
        logger.error("Please ensure you have trained models in the checkpoints directory")
        return
    
    # Create generator
    logger.info("Initializing submission generator...")
    generator = SimpleCompetitionSubmissionGenerator(
        team_name=TEAM_NAME,
        model_checkpoint=MODEL_CHECKPOINT
    )
    
    # Generate submission
    submission_path = generator.generate_submission(
        tamil_data_dir=TAMIL_TEST_DIR,
        malayalam_data_dir=MALAYALAM_TEST_DIR,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "=" * 60)
    print("✅ SUBMISSION READY FOR COMPETITION!")
    print("=" * 60)
    print(f"📦 Submission file: {submission_path}")
    print(f"📅 Deadline: February 10, 2026 (TOMORROW!)")
    print("\n📋 Next steps:")
    print("1. Review the CSV files in the submissions/ directory")
    print("2. Submit via the Google Form provided by organizers")
    print("3. Keep the ZIP file for your records")
    print("=" * 60)


if __name__ == '__main__':
    main()
