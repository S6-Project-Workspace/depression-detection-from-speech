#!/usr/bin/env python3
"""
Transcript Generation Script for Depression Detection Dataset

This script runs the ASR pipeline on all audio files in the NLP Dataset
and generates transcript JSON files with metadata.

Task 13.1: Generate Transcripts for Existing Dataset
- Tamil Depressed/Non-depressed
- Malayalam Depressed/Non-depressed

Requirements: 1.6, 6.5
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from config import get_config, ASRConfig
from asr_pipeline import ASRPipeline, TranscriptManager, create_asr_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset structure mapping
DATASET_STRUCTURE = {
    "tamil": {
        "depressed": {
            "path": "NLP Dataset/Tamil/Depressed/Train_set",
            "label": 1
        },
        "non_depressed": {
            "path": "NLP Dataset/Tamil/Non-depressed/Train_set",
            "label": 0
        }
    },
    "malayalam": {
        "depressed": {
            "path": "NLP Dataset/Malayalam/Depressed/Train_set",
            "label": 1
        },
        "non_depressed": {
            "path": "NLP Dataset/Malayalam/Non_depressed/Train_set",
            "label": 0
        }
    }
}


def get_audio_files(directory: str) -> List[str]:
    """Get all audio files from a directory.
    
    Args:
        directory: Path to directory containing audio files
        
    Returns:
        List of audio file paths
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return audio_files
    
    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in audio_extensions:
            audio_files.append(os.path.join(directory, filename))
    
    return audio_files


def generate_transcripts_for_subset(
    asr_pipeline: ASRPipeline,
    transcript_manager: TranscriptManager,
    audio_dir: str,
    language: str,
    label: int,
    skip_existing: bool = True,
    max_files: Optional[int] = None
) -> Dict[str, int]:
    """Generate transcripts for a subset of the dataset.
    
    Args:
        asr_pipeline: ASR pipeline instance
        transcript_manager: Transcript manager instance
        audio_dir: Directory containing audio files
        language: Language ("tamil" or "malayalam")
        label: Class label (0 or 1)
        skip_existing: Skip files with existing transcripts
        max_files: Maximum number of files to process (for testing)
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "errors": 0
    }
    
    audio_files = get_audio_files(audio_dir)
    stats["total"] = len(audio_files)
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    logger.info(f"Processing {len(audio_files)} files from {audio_dir}")
    
    for i, audio_path in enumerate(audio_files):
        filename = os.path.basename(audio_path)
        
        # Check if transcript already exists
        if skip_existing:
            existing = transcript_manager.load_transcript(audio_path, language, label)
            if existing and existing.text:
                logger.debug(f"Skipping (exists): {filename}")
                stats["skipped"] += 1
                continue
        
        logger.info(f"[{i+1}/{len(audio_files)}] Transcribing: {filename}")
        
        try:
            # Transcribe audio
            result = asr_pipeline.transcribe(audio_path, language)
            
            if result.error:
                logger.warning(f"Error transcribing {filename}: {result.error}")
                stats["errors"] += 1
            else:
                # Save transcript
                transcript_manager.save_transcript(result, language, label)
                stats["processed"] += 1
                logger.info(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
                
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            stats["errors"] += 1
    
    return stats


def generate_all_transcripts(
    output_dir: str = "transcripts",
    languages: Optional[List[str]] = None,
    skip_existing: bool = True,
    max_files_per_subset: Optional[int] = None,
    device: str = "cpu"
) -> Dict[str, Dict]:
    """Generate transcripts for the entire dataset.
    
    Args:
        output_dir: Base directory for transcript output
        languages: List of languages to process (default: all)
        skip_existing: Skip files with existing transcripts
        max_files_per_subset: Max files per subset (for testing)
        device: Device for ASR model ("cpu", "cuda", "mps")
        
    Returns:
        Dictionary with statistics for each subset
    """
    # Initialize ASR pipeline
    config = get_config("combined")
    config.asr.device = device
    
    logger.info(f"Initializing ASR pipeline on {device}...")
    asr_pipeline = create_asr_pipeline(config.asr)
    
    # Initialize transcript manager
    transcript_manager = TranscriptManager(output_dir)
    
    # Determine which languages to process
    if languages is None:
        languages = ["tamil", "malayalam"]
    
    all_stats = {}
    
    for language in languages:
        if language not in DATASET_STRUCTURE:
            logger.warning(f"Unknown language: {language}")
            continue
        
        all_stats[language] = {}
        
        for class_name, info in DATASET_STRUCTURE[language].items():
            audio_dir = info["path"]
            label = info["label"]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {language.upper()} - {class_name.upper()}")
            logger.info(f"Directory: {audio_dir}")
            logger.info(f"{'='*60}")
            
            stats = generate_transcripts_for_subset(
                asr_pipeline=asr_pipeline,
                transcript_manager=transcript_manager,
                audio_dir=audio_dir,
                language=language,
                label=label,
                skip_existing=skip_existing,
                max_files=max_files_per_subset
            )
            
            all_stats[language][class_name] = stats
            
            logger.info(f"Completed {language}/{class_name}:")
            logger.info(f"  Total: {stats['total']}")
            logger.info(f"  Processed: {stats['processed']}")
            logger.info(f"  Skipped: {stats['skipped']}")
            logger.info(f"  Errors: {stats['errors']}")
    
    return all_stats


def print_summary(stats: Dict[str, Dict]):
    """Print a summary of transcript generation."""
    print("\n" + "="*60)
    print("TRANSCRIPT GENERATION SUMMARY")
    print("="*60)
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    total_files = 0
    
    for language, lang_stats in stats.items():
        print(f"\n{language.upper()}:")
        for class_name, class_stats in lang_stats.items():
            print(f"  {class_name}:")
            print(f"    Total files: {class_stats['total']}")
            print(f"    Processed: {class_stats['processed']}")
            print(f"    Skipped: {class_stats['skipped']}")
            print(f"    Errors: {class_stats['errors']}")
            
            total_files += class_stats['total']
            total_processed += class_stats['processed']
            total_skipped += class_stats['skipped']
            total_errors += class_stats['errors']
    
    print(f"\nOVERALL:")
    print(f"  Total files: {total_files}")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Errors: {total_errors}")
    print("="*60)


def save_generation_report(stats: Dict[str, Dict], output_dir: str):
    """Save transcript generation report to JSON."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "summary": {
            "total_processed": sum(
                s["processed"] 
                for lang in stats.values() 
                for s in lang.values()
            ),
            "total_errors": sum(
                s["errors"] 
                for lang in stats.values() 
                for s in lang.values()
            )
        }
    }
    
    report_path = os.path.join(output_dir, "generation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate transcripts for depression detection dataset"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="transcripts",
        help="Output directory for transcripts (default: transcripts)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        choices=["tamil", "malayalam", "all"],
        default="all",
        help="Language to process (default: all)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device for ASR model (default: cpu)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing transcripts"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum files per subset (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually transcribing"
    )
    
    args = parser.parse_args()
    
    # Determine languages
    languages = None if args.language == "all" else [args.language]
    
    if args.dry_run:
        # Just show statistics
        print("\nDRY RUN - Showing dataset statistics:\n")
        for language, lang_info in DATASET_STRUCTURE.items():
            if languages and language not in languages:
                continue
            print(f"{language.upper()}:")
            for class_name, info in lang_info.items():
                audio_files = get_audio_files(info["path"])
                print(f"  {class_name}: {len(audio_files)} files")
        return
    
    # Generate transcripts
    stats = generate_all_transcripts(
        output_dir=args.output_dir,
        languages=languages,
        skip_existing=not args.no_skip,
        max_files_per_subset=args.max_files,
        device=args.device
    )
    
    # Print summary
    print_summary(stats)
    
    # Save report
    save_generation_report(stats, args.output_dir)


if __name__ == "__main__":
    main()
