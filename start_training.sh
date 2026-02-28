#!/bin/bash

# Quick Start Training Script for NLP Multimodal System
# This script trains all components of the depression detection system

echo "=========================================="
echo "NLP Multimodal Training - Quick Start"
echo "=========================================="
echo ""

# Check if transcripts exist
if [ ! -d "transcripts/tamil" ] || [ ! -d "transcripts/malayalam" ]; then
    echo "⚠️  Transcripts not found!"
    echo "Generating transcripts first..."
    python generate_transcripts.py
    echo ""
fi

# Check Python environment
echo "Checking Python environment..."
python --version
echo ""

# Start training
echo "🚀 Starting NLP Multimodal Training..."
echo ""
echo "This will train:"
echo "  1. TF-IDF Baseline (~5 min)"
echo "  2. MuRIL Text Model (~30 min)"
echo "  3. Multimodal (Audio + Text) (~2 hours)"
echo "  4. Enhanced Multimodal (Audio + Text + Linguistic) (~3 hours)"
echo ""
echo "Total estimated time: ~5-6 hours"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting training..."
    python train_nlp_multimodal.py \
        --mode all \
        --epochs 10 \
        --batch-size 16 \
        --n-folds 3 \
        --audio-model ssl \
        --use-linguistic \
        --progressive-training \
        --output-dir outputs/nlp_multimodal_training \
        --device auto \
        --seed 42
    
    echo ""
    echo "=========================================="
    echo "✅ Training Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: outputs/nlp_multimodal_training/"
    echo ""
    echo "Next steps:"
    echo "  1. View results: cat outputs/nlp_multimodal_training/training_results.json"
    echo "  2. Generate submission: python create_final_submission.py"
    echo "  3. Write paper with results"
else
    echo "Training cancelled."
fi
