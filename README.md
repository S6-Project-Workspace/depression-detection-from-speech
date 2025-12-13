# Depression Detection from Speech

🎯 **DravidianLangTech @ ACL 2026 Shared Task** - Speech-based Depression Detection for Tamil and Malayalam

A deep learning system for detecting depression markers in speech using a two-stream architecture combining **ECAPA-TDNN** and **Wav2Vec2 SSL** models.

## 🏆 Performance

| Model | Macro-F1 Score |
|-------|----------------|
| **ECAPA-TDNN** | **98.35%** |
| SSL (Wav2Vec2) | 77.42% |
| Ensemble | ~95% |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/S6-Project-Workspace/depression-detection-from-speech.git
cd depression-detection-from-speech
```

### 2. Install Dependencies
```bash
conda create -n depression_detection python=3.10
conda activate depression_detection
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🎤 Features

- **Real-time Voice Recording**: Record your voice directly in the browser
- **Audio File Upload**: Support for WAV, MP3, FLAC, M4A, OGG
- **Multiple Models**: Choose between ECAPA-TDNN, SSL, or Ensemble prediction
- **Visual Results**: Confidence scores and chunk-level analysis

## 🏗️ Architecture

### Two-Stream Fusion Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Audio Input (16kHz)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
    ┌───────────────────────┐  ┌───────────────────────┐
    │   ECAPA-TDNN Stream   │  │     SSL Stream        │
    │  (Spectral/Prosodic)  │  │   (Wav2Vec2-base)     │
    │                       │  │                       │
    │  • Mel Spectrogram    │  │  • Raw Waveform       │
    │  • SE-Res2Net Blocks  │  │  • 768-dim Features   │
    │  • Attentive Stats    │  │  • Transformer Enc    │
    │    Pooling            │  │                       │
    └───────────────────────┘  └───────────────────────┘
              │                          │
              │      192-dim             │      768-dim
              ▼                          ▼
    ┌───────────────────────────────────────────────────┐
    │              Late Fusion Layer                     │
    │         (Weighted Average: 0.7/0.3)               │
    └───────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │   Binary Classification   │
              │  Depressed / Non-Depressed│
              └───────────────────────────┘
```

### ECAPA-TDNN Architecture
- **Input**: 80-band Mel Spectrogram
- **Backbone**: SE-Res2Net blocks with multi-scale feature aggregation
- **Pooling**: Attentive Statistics Pooling
- **Output**: 192-dimensional speaker embedding

### SSL Architecture  
- **Backbone**: facebook/wav2vec2-base (12 transformer layers)
- **Features**: 768-dimensional contextualized representations
- **Pooling**: Mean pooling over time

## 📁 Project Structure

```
├── app.py                 # Streamlit web application
├── config.py              # Configuration dataclasses
├── preprocessing.py       # Audio preprocessing pipeline
├── dataset.py             # PyTorch dataset classes
├── ssl_model.py           # Wav2Vec2 SSL model
├── ecapa_model.py         # ECAPA-TDNN model
├── trainer.py             # Training loop with progress bars
├── threshold_tuner.py     # Macro-F1 threshold optimization
├── inference.py           # Inference utilities
├── train.py               # Main training script
├── evaluate.py            # Model evaluation script
├── requirements.txt       # Python dependencies
└── outputs/               # Trained model checkpoints
    └── training_with_progress/
        └── checkpoints/
            ├── ecapa_fold_0/best_model.pt
            └── ssl_fold_0/best_model.pt
```

## 🔧 Training (Optional)

If you have access to the dataset:

```bash
python train.py --language combined --epochs 10 --folds 3
```

### Training Configuration
- **Cross-Validation**: 3-fold Stratified Group K-Fold (speaker-independent)
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 1e-4 (SSL) / 1e-3 (ECAPA)
- **Scheduler**: Cosine annealing with warmup
- **Audio**: 5-second chunks at 16kHz

## 📊 Dataset

**DravidianLangTech Depression Dataset**
- Languages: Tamil, Malayalam
- Classes: Depressed, Non-Depressed
- Format: WAV audio files

> Note: Dataset not included due to size. Contact organizers for access.

## 🛠️ Technical Details

### Audio Preprocessing
- Resampling to 16kHz (Sinc interpolation with Hann window)
- Stereo to mono conversion
- DC offset removal and peak normalization
- 5-second chunking with 2.5s overlap (inference)

### Model Training
- Mixed precision training (where supported)
- Gradient clipping (max norm 1.0)
- Early stopping with patience
- Macro-F1 based threshold optimization

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{depression-detection-2026,
  title={Two-Stream Fusion Architecture for Depression Detection in Dravidian Languages},
  author={Your Name},
  booktitle={DravidianLangTech @ ACL 2026},
  year={2026}
}
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- DravidianLangTech organizers for the shared task
- Hugging Face for Wav2Vec2 pretrained models
- SpeechBrain for ECAPA-TDNN implementation reference
