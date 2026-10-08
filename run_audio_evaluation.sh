#!/bin/bash

# ACL 2026 DravidianLangTech Competition - Audio Model Evaluation
# This script activates the depression_detection environment and runs the evaluation

echo "🎯 ACL 2026 DravidianLangTech Competition - Audio Model Evaluation"
echo "=================================================================="

# Activate the depression_detection conda environment
echo "📦 Activating depression_detection environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate depression_detection

# Check if activation was successful
if [[ "$CONDA_DEFAULT_ENV" == "depression_detection" ]]; then
    echo "✅ Environment activated successfully: $CONDA_DEFAULT_ENV"
else
    echo "❌ Failed to activate depression_detection environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

# Run the evaluation
echo ""
echo "🚀 Starting audio model evaluation..."
echo "This will evaluate 8 audio models on Tamil (160 samples) and Malayalam (200 samples)"
echo ""

python3 evaluate_audio_models_competition.py

echo ""
echo "✅ Evaluation complete!"
echo "📊 Check 'competition_evaluation_results/' for detailed results"