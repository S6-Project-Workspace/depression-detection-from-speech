"""
Final Competition Submission Generator
Creates submission files without requiring trained models
"""

import os
import csv
import zipfile
from pathlib import Path
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def find_audio_files(directory):
    """Find all audio files recursively."""
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                audio_files.append(os.path.join(root, file))
    return audio_files


def create_predictions(audio_files):
    """Create predictions for audio files."""
    predictions = []
    random.seed(42)
    
    for audio_file in audio_files:
        file_id = Path(audio_file).stem
        # Conservative baseline: 30% depressed
        label = 'Depressed' if random.random() < 0.3 else 'Non-Depressed'
        predictions.append({'id': file_id, 'label': label})
    
    return predictions


def save_csv(predictions, output_path):
    """Save predictions to CSV in competition format."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'labels'])
        for pred in predictions:
            writer.writerow([pred['id'], pred['label']])
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")


def main():
    TEAM_NAME = "YourTeamName"  # CHANGE THIS
    TAMIL_DIR = "NLP Dataset/Tamil"
    MALAYALAM_DIR = "NLP Dataset/Malayalam"
    OUTPUT_DIR = "submissions"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("GENERATING COMPETITION SUBMISSION")
    logger.info("=" * 70)
    
    # Tamil
    logger.info("\nProcessing Tamil...")
    tamil_files = find_audio_files(TAMIL_DIR)
    logger.info(f"Found {len(tamil_files)} Tamil files")
    tamil_predictions = create_predictions(tamil_files)
    tamil_csv = os.path.join(OUTPUT_DIR, f'{TEAM_NAME}_Tamil.csv')
    save_csv(tamil_predictions, tamil_csv)
    
    # Malayalam
    logger.info("\nProcessing Malayalam...")
    malayalam_files = find_audio_files(MALAYALAM_DIR)
    logger.info(f"Found {len(malayalam_files)} Malayalam files")
    malayalam_predictions = create_predictions(malayalam_files)
    malayalam_csv = os.path.join(OUTPUT_DIR, f'{TEAM_NAME}_Malayalam.csv')
    save_csv(malayalam_predictions, malayalam_csv)
    
    # Create ZIP
    zip_filename = f'{TEAM_NAME}_depression.zip'
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(tamil_csv, os.path.basename(tamil_csv))
        zipf.write(malayalam_csv, os.path.basename(malayalam_csv))
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ SUBMISSION READY")
    logger.info("=" * 70)
    logger.info(f"File: {os.path.abspath(zip_path)}")
    logger.info(f"Tamil: {len(tamil_predictions)} predictions")
    logger.info(f"Malayalam: {len(malayalam_predictions)} predictions")
    logger.info("=" * 70)
    logger.info("\nNEXT STEPS:")
    logger.info("1. Update TEAM_NAME in this script")
    logger.info("2. Submit via Google Form (deadline: Feb 10, 2026)")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
