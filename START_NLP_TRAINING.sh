#!/bin/bash

# NLP Component Training Starter Script
# This script starts the complete NLP training pipeline

echo "=========================================="
echo "NLP Component Training Pipeline"
echo "=========================================="
echo ""

# Check if transcripts exist
if [ ! -d "transcripts/tamil" ] || [ ! -d "transcripts/malayalam" ]; then
    echo "ERROR: Transcripts not found!"
    echo "Please run transcript generation first:"
    echo "  python generate_transcripts.py"
    exit 1
fi

echo "✓ Transcripts found"
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found!"
    exit 1
fi

echo "✓ Python environment ready"
echo ""

# Determine device
if python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="mps"
    echo "✓ Using Apple Silicon (MPS) acceleration"
elif python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
    echo "✓ Using CUDA GPU acceleration"
else
    DEVICE="cpu"
    echo "⚠ Using CPU (training will be slower)"
fi

echo ""
echo "=========================================="
echo "Starting Training Pipeline"
echo "=========================================="
echo ""
echo "Training modes:"
echo "  1. TF-IDF Baseline (fastest)"
echo "  2. MuRIL Text Model"
echo "  3. Multimodal Fusion (Audio + Text)"
echo "  4. Enhanced Multimodal (Audio + Text + Linguistic)"
echo "  5. ALL (complete pipeline)"
echo ""
read -p "Select mode (1-5) [default: 5]: " MODE_CHOICE

case $MODE_CHOICE in
    1)
        MODE="tfidf"
        echo "Training TF-IDF baseline..."
        ;;
    2)
        MODE="text"
        echo "Training MuRIL text model..."
        ;;
    3)
        MODE="multimodal"
        echo "Training multimodal fusion..."
        ;;
    4)
        MODE="enhanced"
        echo "Training enhanced multimodal..."
        ;;
    *)
        MODE="all"
        echo "Training complete pipeline..."
        ;;
esac

echo ""
read -p "Number of epochs [default: 10]: " EPOCHS
EPOCHS=${EPOCHS:-10}

read -p "Batch size [default: 16]: " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-16}

echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Device: $DEVICE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo ""
read -p "Start training? (y/n) [default: y]: " CONFIRM
CONFIRM=${CONFIRM:-y}

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "=========================================="
echo "Training Started"
echo "=========================================="
echo ""

# Run training
python train_nlp_components.py \
    --mode $MODE \
    --device $DEVICE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --transcript-dir transcripts \
    --output-dir outputs/nlp_training \
    2>&1 | tee outputs/nlp_training/logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Results saved to: outputs/nlp_training/results/"
echo "Models saved to: outputs/nlp_training/checkpoints/"
echo "Logs saved to: outputs/nlp_training/logs/"
echo ""
