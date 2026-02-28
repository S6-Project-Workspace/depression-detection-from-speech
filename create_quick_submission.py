"""
Quick submission generator for DravidianLangTech@ACL 2026
Creates submission files without requiring trained models.
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
    """Find all audio files in directory."""
    audio_files = []
    extensions = ['.wav', '.mp3', '.flac', '.m4a']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files


def create_predictions(audio_files, strategy='balanced'):
    """Create predictions for audio files.
    
    Args:
        audio_files: List of audio file paths
        strategy: 'balanced', 'conservative', or 'random'
    """
    predictions = []
    random.seed(42)
    
    for audio_file in audio_files:
        file_id = Path(audio_file).stem
        
        if strategy == 'balanced':
            # 50-50 split
            label = 'Depressed' if random.random() < 0.5 else 'Non-Depressed'
        elif strategy == 'conservative':
            # Bias toward non-depressed (more common in general population)
            label = 'Depressed' if random.random() < 0.3 else 'Non-Depressed'
        else:  # random
            label = random.choice(['Depressed', 'Non-Depressed'])
        
        predictions.append({
            'id': file_id,
            'label': label
        })
    
    return predictions


def save_csv(predictions, output_path):
    """Save predictions to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'labels'])
        
        for pred in predictions:
            writer.writerow([pred['id'], pred['label']])
    
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")


def create_submission(team_name, tamil_dir, malayalam_dir, output_dir='submissions', strategy='balanced'):
    """Create complete submission package."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("CREATING COMPETITION SUBMISSION")
    logger.info("=" * 70)
    logger.info(f"Team Name: {team_name}")
    logger.info(f"Strategy: {strategy}")
    logger.info("=" * 70)
    
    # Process Tamil
    logger.info("\n[1/4] Finding Tamil audio files...")
    tamil_files = find_audio_files(tamil_dir)
    logger.info(f"Found {len(tamil_files)} Tamil files")
    
    logger.info("[2/4] Generating Tamil predictions...")
    tamil_predictions = create_predictions(tamil_files, strategy)
    
    # Process Malayalam
    logger.info("\n[3/4] Finding Malayalam audio files...")
    malayalam_files = find_audio_files(malayalam_dir)
    logger.info(f"Found {len(malayalam_files)} Malayalam files")
    
    logger.info("[4/4] Generating Malayalam predictions...")
    malayalam_predictions = create_predictions(malayalam_files, strategy)
    
    # Save CSV files
    logger.info("\n" + "=" * 70)
    logger.info("SAVING FILES")
    logger.info("=" * 70)
    
    tamil_csv = os.path.join(output_dir, f'{team_name}_Tamil.csv')
    malayalam_csv = os.path.join(output_dir, f'{team_name}_Malayalam.csv')
    
    save_csv(tamil_predictions, tamil_csv)
    save_csv(malayalam_predictions, malayalam_csv)
    
    # Create ZIP file
    zip_filename = f'{team_name}_depression.zip'
    zip_path = os.path.join(output_dir, zip_filename)
    
    logger.info(f"\nCreating ZIP file: {zip_filename}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(tamil_csv, os.path.basename(tamil_csv))
        zipf.write(malayalam_csv, os.path.basename(malayalam_csv))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ SUBMISSION CREATED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"Location: {os.path.abspath(zip_path)}")
    logger.info(f"\nContents:")
    logger.info(f"  • {os.path.basename(tamil_csv)} ({len(tamil_predictions)} predictions)")
    logger.info(f"  • {os.path.basename(malayalam_csv)} ({len(malayalam_predictions)} predictions)")
    logger.info("=" * 70)
    
    # Statistics
    tamil_depressed = sum(1 for p in tamil_predictions if p['label'] == 'Depressed')
    malayalam_depressed = sum(1 for p in malayalam_predictions if p['label'] == 'Depressed')
    
    logger.info("\nPrediction Statistics:")
    logger.info(f"Tamil:")
    logger.info(f"  - Depressed: {tamil_depressed} ({tamil_depressed/len(tamil_predictions)*100:.1f}%)")
    logger.info(f"  - Non-Depressed: {len(tamil_predictions)-tamil_depressed} ({(len(tamil_predictions)-tamil_depressed)/len(tamil_predictions)*100:.1f}%)")
    logger.info(f"Malayalam:")
    logger.info(f"  - Depressed: {malayalam_depressed} ({malayalam_depressed/len(malayalam_predictions)*100:.1f}%)")
    logger.info(f"  - Non-Depressed: {len(malayalam_predictions)-malayalam_depressed} ({(len(malayalam_predictions)-malayalam_depressed)/len(malayalam_predictions)*100:.1f}%)")
    logger.info("=" * 70)
    
    return zip_path


def main():
    """Main function."""
    
    print("\n" + "=" * 70)
    print("DravidianLangTech@ACL 2026 - Quick Submission Generator")
    print("=" * 70)
    
    # Configuration
    TEAM_NAME = input("\nEnter your team name: ").strip()
    
    if not TEAM_NAME:
        TEAM_NAME = "MyTeam"
        logger.warning(f"No team name provided, using default: {TEAM_NAME}")
    
    # Data directories
    TAMIL_DIR = "NLP Dataset/Tamil"
    MALAYALAM_DIR = "NLP Dataset/Malayalam"
    
    # Verify directories exist
    if not os.path.exists(TAMIL_DIR):
        logger.error(f"Tamil directory not found: {TAMIL_DIR}")
        return
    
    if not os.path.exists(MALAYALAM_DIR):
        logger.error(f"Malayalam directory not found: {MALAYALAM_DIR}")
        return
    
    # Choose strategy
    print("\nPrediction Strategy:")
    print("  1. Balanced (50-50 split)")
    print("  2. Conservative (30% depressed, 70% non-depressed)")
    print("  3. Random")
    
    choice = input("\nSelect strategy (1-3) [default: 2]: ").strip()
    
    strategy_map = {
        '1': 'balanced',
        '2': 'conservative',
        '3': 'random',
        '': 'conservative'
    }
    
    strategy = strategy_map.get(choice, 'conservative')
    
    # Generate submission
    try:
        submission_path = create_submission(
            team_name=TEAM_NAME,
            tamil_dir=TAMIL_DIR,
            malayalam_dir=MALAYALAM_DIR,
            strategy=strategy
        )
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print(f"1. Verify the files in: submissions/")
        print(f"2. Upload {os.path.basename(submission_path)} via Google Form")
        print(f"3. Deadline: February 10, 2026 (TOMORROW!)")
        print("=" * 70)
        print("\nGoogle Form link: Check competition page for submission link")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error creating submission: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
