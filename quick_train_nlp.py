#!/usr/bin/env python3
"""
Quick NLP Training Script - Start Training Immediately

This script starts training with sensible defaults.
Just run: python quick_train_nlp.py
"""

import os
import sys
import torch
from pathlib import Path

# Check if transcripts exist
transcript_dir = Path("transcripts")
if not transcript_dir.exists() or not (transcript_dir / "tamil").exists():
    print("❌ ERROR: Transcripts not found!")
    print("\nPlease generate transcripts first:")
    print("  python generate_transcripts.py")
    sys.exit(1)

print("✅ Transcripts found")

# Determine device
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple Silicon (MPS) acceleration")
elif torch.cuda.is_available():
    device = "cuda"
    print("✅ Using CUDA GPU acceleration")
else:
    device = "cpu"
    print("⚠️  Using CPU (training will be slower)")

print("\n" + "="*60)
print("Quick NLP Training - Starting Now!")
print("="*60)
print("\nConfiguration:")
print(f"  Device: {device}")
print(f"  Epochs: 10")
print(f"  Batch Size: 8 (reduced for stability)")
print(f"  Mode: TF-IDF Baseline (fastest)")
print("\n" + "="*60 + "\n")

# Import and run
from train_nlp_components import NLPTrainingPipeline
from config import get_config

# Get configuration
config = get_config("combined")
config.device = device
config.training.epochs = 10
config.training.batch_size = 8  # Reduced for stability

# Create pipeline
pipeline = NLPTrainingPipeline(
    config=config,
    transcript_dir="transcripts",
    output_dir="outputs/nlp_training",
    device=device
)

# Start with TF-IDF baseline (fastest)
print("🚀 Starting TF-IDF Baseline Training...")
print("   (This is the fastest model to train)\n")

try:
    pipeline.train_tfidf_baseline()
    pipeline.save_results()
    
    print("\n" + "="*60)
    print("✅ TF-IDF Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train MuRIL text model:")
    print("     python train_nlp_components.py --mode text")
    print("\n  2. Train multimodal fusion:")
    print("     python train_nlp_components.py --mode multimodal")
    print("\n  3. Train enhanced multimodal:")
    print("     python train_nlp_components.py --mode enhanced")
    print("\n  4. Or train everything:")
    print("     python train_nlp_components.py --mode all")
    print()
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
