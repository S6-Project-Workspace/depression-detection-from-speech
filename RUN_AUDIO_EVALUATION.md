# ACL 2026 DravidianLangTech Competition - Audio Model Evaluation

## Quick Start

To evaluate your audio models for the competition, run these commands in your terminal:

```bash
# Activate the depression_detection environment
conda activate depression_detection

# Run the quick evaluation (baseline + model check)
python3 quick_audio_evaluation.py
```

## Full Evaluation

For complete evaluation of all trained models:

```bash
# Option 1: Use the shell script
./run_audio_evaluation.sh

# Option 2: Manual activation and run
conda activate depression_detection
python3 evaluate_audio_models_competition.py
```

## What This Does

### Quick Evaluation (`quick_audio_evaluation.py`)
- ✅ Checks your environment and available models
- 📊 Loads competition test data (Tamil: 160 samples, Malayalam: 200 samples)
- 🎵 Runs baseline audio feature evaluation
- 📈 Shows preliminary F1-scores for competition format

### Full Evaluation (`evaluate_audio_models_competition.py`)
- 🤖 Tests all 8 trained audio models:
  - ECAPA-TDNN models (Final + Training folds)
  - SSL models (WavLM/Wav2Vec2 - Final + Training folds)
- 📊 Generates competition-format results
- 📈 Creates detailed performance reports
- 🏆 Ranks models by macro-averaged F1-score
- 📁 Saves results to `competition_evaluation_results/`

## Competition Details

- **Task**: Depression Detection from Malayalam and Tamil Speech
- **Test Data**: 
  - Tamil: 160 samples
  - Malayalam: 200 samples
- **Evaluation**: Macro-averaged Precision, Recall, F1-Score
- **Audio Formats**: Mixed sample rates (16kHz depressed, 48kHz non-depressed)

## Expected Output

The evaluation will show:
1. Model availability check
2. Test data loading confirmation
3. Per-language results (Tamil, Malayalam)
4. Overall competition scores
5. Model rankings for submission

## Troubleshooting

If you get import errors:
```bash
# Make sure you're in the right environment
conda activate depression_detection
conda list | grep torch  # Should show PyTorch installation
```

If test data is missing:
- Check `/Users/keerthivasan/NLP/Test Set/` directory
- Ensure `Tamil_GT.xlsx` and `Malayalam_GT.xlsx` exist
- Verify audio files are extracted in `tamil_test/` and `malayalam_test/`

## Next Steps

After evaluation:
1. 📊 Review results in `competition_evaluation_results/`
2. 🏆 Select top-performing model for submission
3. 📝 Use results for competition paper/report
4. 🚀 Submit to ACL 2026 DravidianLangTech competition

---

**Ready to evaluate your models for the competition!** 🎯