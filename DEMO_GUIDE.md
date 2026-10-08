# 🎯 Depression Detection Demo Guide

## Quick Start

### Option 1: Run Script
```bash
./RUN_DEMO.sh
```

### Option 2: Direct Command
```bash
streamlit run streamlit_app.py
```

## How to Use

### 1. **Demo Buttons** (Easiest!)
Click any of the 4 demo buttons to auto-populate sample text:
- 😔 Depressed (Tamil)
- 😊 Non-Depressed (Tamil)
- 😔 Depressed (Malayalam)
- 😊 Non-Depressed (Malayalam)

### 2. **Analyze**
Click the big **"🚀 ANALYZE ALL MODELS"** button

### 3. **View Results**
See predictions from:
- **TF-IDF Baseline** - Traditional ML approach
- **MuRIL Transformer** - Deep learning text model
- **Enhanced Multimodal** - Text + Linguistic features

### 4. **Consensus**
Get the final consensus prediction from all models

## Features

✅ **Auto-populated demo data** - No typing needed!
✅ **All models tested** - Compare 3 different approaches
✅ **Instant results** - See predictions in seconds
✅ **Confidence scores** - Know how certain each model is
✅ **Linguistic analysis** - POS, NER, Dependencies included
✅ **Beautiful UI** - Clean, professional interface

## Models Loaded

1. **TF-IDF Baseline** (Traditional ML)
   - Fast, interpretable
   - Good baseline performance

2. **MuRIL Transformer** (Deep Learning)
   - State-of-the-art for Indian languages
   - High accuracy

3. **Enhanced Multimodal** (Advanced)
   - Combines text + linguistic features
   - Best overall performance
   - Includes POS tagging, NER, dependency parsing

## Showcase Tips

1. **Start with demo buttons** - Shows it works immediately
2. **Try different languages** - Tamil and Malayalam supported
3. **Compare predictions** - Show how models agree/disagree
4. **Highlight consensus** - Final prediction from all models
5. **Show linguistic features** - Expand to see NLP analysis

## Troubleshooting

If models don't load:
```bash
# Check if models exist
ls outputs/nlp_training_peak_performance/checkpoints/

# Should see:
# - tfidf_baseline.pkl
# - muril_text_model.pt
# - best_enhanced_multimodal_model_baseline.pt
```

## Performance

- **Model Loading**: ~10-15 seconds (first time)
- **Prediction Time**: ~1-2 seconds per model
- **Total Demo Time**: ~5 seconds from click to results

Perfect for live demonstrations! 🎬
