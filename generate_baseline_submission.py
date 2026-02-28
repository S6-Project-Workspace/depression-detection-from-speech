"""
Generate baseline submission for DravidianLangTech@ACL 2026
Depression Detection Shared Task

This creates a simple baseline submission when trained models are not available.
Uses random predictions or simple heuristics.
"""

import os
import csv
import zipfile
from pathlib import Path
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_baseline_predictions(data_dir: str, language: str) -> list:
    """Generate baseline predictions for a dataset.
    
    Args:
        data_dir: Directory containing audio files
        language: Language name
        
    Returns:
        List of predictions
    """
    logger.info(f"Generating baseline predictions for {language}")
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path(data_dir).rglob(ext))
    
    logger.info(f"Found {len(audio_files)} files")
    
    predictions = []
    random.seed(42)  # For reproducibility
    
    for audio_file in audio_files:
        file_id = audio_file.stem
        
        # Simple baseline: random prediction with slight bias toward non-depressed
        # (as depression is typically less common)
        label = 'Depressed' if random.random() < 0.3 else 'Non-Depressed'
        
        predictions.append({
            'id': file_id,
            'label': label
        })
    
    return predictions


def save_predictions_csv(predictions: list, output_path: str):
    """Save predictions to CSV in competition format."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'labels'])
        
        for pred in predictions:
            writer.writerow([pred['id'], pred['label']])
    
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")


def create_submission_package(
    team_name: str,
    tamil_dir: str,
    malayalam_dir: str,
    output_dir: str = 'submissions'
):
    """Create complete submission package.
    
    Args:
        team_name: Your team name
        tamil_dir: Tamil test data directory
        malayalam_dir: Malayalam test data directory
        output_dir: Output directory for submission files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions
    logger.info("=" * 60)
    logger.info("GENERATING BASELINE SUBMISSION")
    logger.info("=" * 60)
    
    tamil_predictions = generate_baseline_predictions(tamil_dir, 'Tamil')
    malayalam_predictions = generate_baseline_predictions(malayalam_dir, 'Malayalam')
    
    # Save CSV files
    tamil_csv = os.path.join(output_dir, f'{team_name}_Tamil.csv')
    malayalam_csv = os.path.join(output_dir, f'{team_name}_Malayalam.csv')
    
    save_predictions_csv(tamil_predictions, tamil_csv)
    save_predictions_csv(malayalam_predictions, malayalam_csv)
    
    # Create zip file
    zip_filename = f'{team_name}_depression.zip'
    zip_path = os.path.join(output_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(tamil_csv, os.path.basename(tamil_csv))
        zipf.write(malayalam_csv, os.path.basename(malayalam_csv))
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUBMISSION PACKAGE CREATED")
    logger.info("=" * 60)
    logger.info(f"Location: {zip_path}")
    logger.info(f"Tamil predictions: {len(tamil_predictions)}")
    logger.info(f"Malayalam predictions: {len(malayalam_predictions)}")
    logger.info("=" * 60)
    logger.info("Files in package:")
    logger.info(f"  - {os.path.basename(tamil_csv)}")
    logger.info(f"  - {os.path.basename(malayalam_csv)}")
    logger.info("=" * 60)
    
    return zip_path


def main():
    """Main function."""
    
    # CHANGE THIS to your team name
    TEAM_NAME = "YourTeamName"
    
    # Data directories
    TAMIL_DIR = "NLP Dataset/Tamil"
    MALAYALAM_DIR = "NLP Dataset/Malayalam"
    
    # Check if directories exist
    if not os.path.exists(TAMIL_DIR):
        logger.error(f"Tamil directory not found: {TAMIL_DIR}")
        return
    
    if not os.path.exists(MALAYALAM_DIR):
        logger.error(f"Malayalam directory not found: {MALAYALAM_DIR}")
        return
    
    # Generate submission
    submission_path = create_submission_package(
        team_name=TEAM_NAME,
        tamil_dir=TAMIL_DIR,
        malayalam_dir=MALAYALAM_DIR
    )
    
    print("\n" + "=" * 60)
    print("✓ SUBMISSION READY FOR UPLOAD")
    print("=" * 60)
    print(f"File: {submission_path}")
    print("\nIMPORTANT:")
    print("1. Update TEAM_NAME in this script before running")
    print("2. Submit via Google Form (link in competition page)")
    print("3. Deadline: February 10, 2026 (TOMORROW!)")
    print("=" * 60)


if __name__ == '__main__':
    main()
