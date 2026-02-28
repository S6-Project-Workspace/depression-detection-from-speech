#!/bin/bash

echo "🎤 Depression Detection - Model Testing Interface"
echo "=================================================="
echo ""
echo "Starting Streamlit app with all trained models..."
echo ""
echo "Available models:"
echo "  ✓ TF-IDF Baseline (2.2 MB)"
echo "  ✓ MuRIL Text Model (906 MB)"
echo "  ✓ ECAPA Audio (27 MB)"
echo "  ✓ SSL Audio (362 MB)"
echo "  ✓ Multimodal Baseline (3.7 GB)"
echo "  ✓ Enhanced Multimodal (3.7 GB)"
echo ""
echo "Opening browser..."
echo ""

streamlit run test_all_models.py
