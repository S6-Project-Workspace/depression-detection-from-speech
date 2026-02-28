"""
Generate submission files for DravidianLangTech@ACL 2026
Depression Detection Shared Task

Creates predictions in the required format:
- Team_name_Language.csv for each language
- Packaged in Team_name_task.zip
"""

import os
import csv
import zipfile
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict
import logging

from config import DepressionDetectionConfig
from inference import DepressionInferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompetitionSubmissionGenerator:
    """Generate competition submission files."""
    
    def __init__(
        self,
        team_name: str,
        checkpoint_dir: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            team_name: Your team name for submission
            checkpoint_dir: Directory containing model checkpoints
            device: Device to run inference on
        """
        self.team_name = team_name
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        # Initialize config and pipeline
        self.config = DepressionDetectionConfig()
        self.pipeline = DepressionInferencePipeline(
            self.config,
            device=self.device
        )
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models from checkpoints."""
        ssl_checkpoint = os.path.join(self.checkpoint_dir, 'best_ssl_model.pt')
        ecapa_checkpoint = os.path.join(self.checkpoint_dir, 'best_ecapa_model.pt')
        
        if not os.path.exists(ssl_checkpoint):
            logger.warning(f"SSL checkpoint not found: {ssl_checkpoint}")
            logger.info("Will use default model initialization")
        
        if not os.path.exists(ecapa_checkpoint):
            logger.warning(f"ECAPA checkpoint not found: {ecapa_checkpoint}")
            logger.info("Will use default model initialization")
        
        try:
            # Initialize models first
            from ssl_model import Wav2Vec2ForDepressionDetection
            from ecapa_model import ECAPATDNN
            
            self.pipeline.ssl_model = Wav2Vec2ForDepressionDetection(self.config).to(self.device)
            self.pipeline.ecapa_model = ECAPATDNN(self.config).to(self.device)
            
            # Try to load checkpoints if they exist
            if os.path.exists(ssl_checkpoint) and os.path.exists(ecapa_checkpoint):
                self.pipeline.load_models(ssl_checkpoint, ecapa_checkpoint)
                logger.info("Successfully loaded model checkpoints")
            else:
                logger.info("Using default model initialization (no checkpoints found)")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.info("Continuing with default initialization")
    
    def predict_language_dataset(
        self,
        data_dir: str,
        language: str
    ) -> List[Dict]:
        """Generate predictions for a language dataset.
        
        Args:
            data_dir: Directory containing test audio files
            language: Language name (Tamil or Malayalam)
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Processing {language} dataset from {data_dir}")
        
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(Path(data_dir).rglob(ext))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        predictions = []
        for audio_file in audio_files:
            try:
                result = self.pipeline.predict_file(str(audio_file))
                
                # Extract file ID (without extension)
                file_id = audio_file.stem
                
                predictions.append({
                    'id': file_id,
                    'label': 'Depressed' if result['predicted_label'] == 1 else 'Non-Depressed',
                    'probability': result['probability']
                })
                
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                # Add default prediction for failed files
                predictions.append({
                    'id': audio_file.stem,
                    'label': 'Non-Depressed',
                    'probability': 0.0
                })
        
        logger.info(f"Generated {len(predictions)} predictions for {language}")
        return predictions
    
    def save_predictions_csv(
        self,
        predictions: List[Dict],
        output_path: str
    ):
        """Save predictions to CSV file in competition format.
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Path to save CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header (column name should be 'labels', not numbers)
            writer.writerow(['id', 'labels'])
            
            # Write predictions
            for pred in predictions:
                writer.writerow([pred['id'], pred['label']])
        
        logger.info(f"Saved predictions to {output_path}")
    
    def generate_submission(
        self,
        tamil_data_dir: str,
        malayalam_data_dir: str,
        output_dir: str = 'submissions',
        run_number: int = 1
    ):
        """Generate complete submission package.
        
        Args:
            tamil_data_dir: Directory with Tamil test data
            malayalam_data_dir: Directory with Malayalam test data
            output_dir: Directory to save submission files
            run_number: Run number (1-3 for multiple submissions)
        """
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
        if run_number == 1:
            # Single run submission
            tamil_csv = os.path.join(output_dir, f'{self.team_name}_Tamil.csv')
            malayalam_csv = os.path.join(output_dir, f'{self.team_name}_Malayalam.csv')
        else:
            # Multiple run submission
            tamil_csv = os.path.join(output_dir, f'{self.team_name}_Tamil_run{run_number}.csv')
            malayalam_csv = os.path.join(output_dir, f'{self.team_name}_Malayalam_run{run_number}.csv')
        
        self.save_predictions_csv(tamil_predictions, tamil_csv)
        self.save_predictions_csv(malayalam_predictions, malayalam_csv)
        
        # Create zip file
        zip_filename = f'{self.team_name}_depression.zip'
        zip_path = os.path.join(output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(tamil_csv, os.path.basename(tamil_csv))
            zipf.write(malayalam_csv, os.path.basename(malayalam_csv))
        
        logger.info("=" * 60)
        logger.info(f"Submission package created: {zip_path}")
        logger.info("=" * 60)
        logger.info(f"Tamil predictions: {len(tamil_predictions)}")
        logger.info(f"Malayalam predictions: {len(malayalam_predictions)}")
        logger.info("=" * 60)
        logger.info("Submission files:")
        logger.info(f"  - {os.path.basename(tamil_csv)}")
        logger.info(f"  - {os.path.basename(malayalam_csv)}")
        logger.info(f"  - {zip_filename}")
        logger.info("=" * 60)
        
        return zip_path


def main():
    """Main function to generate submission."""
    
    # Configuration
    TEAM_NAME = "YourTeamName"  # CHANGE THIS to your team name
    CHECKPOINT_DIR = "checkpoints"  # Directory with trained models
    
    # Data directories (adjust paths as needed)
    TAMIL_TEST_DIR = "NLP Dataset/Tamil"
    MALAYALAM_TEST_DIR = "NLP Dataset/Malayalam"
    
    OUTPUT_DIR = "submissions"
    
    # Create generator
    generator = CompetitionSubmissionGenerator(
        team_name=TEAM_NAME,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # Generate submission (run 1)
    submission_path = generator.generate_submission(
        tamil_data_dir=TAMIL_TEST_DIR,
        malayalam_data_dir=MALAYALAM_TEST_DIR,
        output_dir=OUTPUT_DIR,
        run_number=1
    )
    
    print("\n" + "=" * 60)
    print("SUBMISSION READY!")
    print("=" * 60)
    print(f"Submission file: {submission_path}")
    print("\nNext steps:")
    print("1. Review the CSV files in the submissions/ directory")
    print("2. Submit via the Google Form provided by organizers")
    print("3. Deadline: February 10, 2026")
    print("=" * 60)


if __name__ == '__main__':
    main()
