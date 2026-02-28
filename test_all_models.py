"""
Complete Model Testing Interface - All 25 Trained Models

Streamlit app for testing ALL trained depression detection models with dropdown selection.

Model Groups:
  Final Models (6):
    - TF-IDF Baseline (2.2MB)    - MuRIL Text Model (906MB)
    - ECAPA Audio Fold 0 (27MB)  - SSL Audio Fold 0 (362MB)
    - Multimodal Baseline (3.7GB)- Enhanced Multimodal (3.7GB)
  Top-Level Checkpoints (2):
    - best_multimodal_model.pt (3.7GB)
    - final_multimodal_model.pt (3.7GB)
  Training With Progress - Audio Folds (8):
    - SSL folds 0-2, ECAPA folds 0-2, best/final model
  NLP Peak Performance (4):
    - TF-IDF, MuRIL, Best/Final Enhanced Multimodal
  Depression Combined Runs (5):
    - best/final models from two training runs + SSL fold

Usage:
    streamlit run test_all_models.py
"""

import os
import sys
import tempfile
import numpy as np
import torch
import streamlit as st
from pathlib import Path
import soundfile as sf
import pickle

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DepressionDetectionConfig
from text_preprocessing import TextPreprocessor
from text_model import create_muril_model, create_muril_tokenizer
from tfidf_classifier import TFIDFClassifier
from ssl_model import create_ssl_model
from ecapa_model import create_ecapa_model
from multimodal_model import create_multimodal_model
from enhanced_multimodal_model import create_enhanced_multimodal_model, EnhancedFusionConfig
from linguistic_analyzer import create_linguistic_analyzer

# Page configuration
st.set_page_config(
    page_title="Depression Detection - All Models",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid;
    }
    .depressed {
        background-color: #ffebee;
        border-color: #f44336 !important;
    }
    .non-depressed {
        background-color: #e8f5e9;
        border-color: #4caf50 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ALL TRAINED MODEL PATHS (25 models across the codebase)
# ============================================================

# --- Final Models (production-ready, copied to final_models/) ---
FINAL_MODELS = {
    "TF-IDF Baseline": "final_models/tfidf_baseline.pkl",
    "MuRIL Text Model": "final_models/muril_text_model.pt",
    "ECAPA Audio (Fold 0)": "final_models/ecapa_audio_fold0.pt",
    "SSL Audio (Fold 0)": "final_models/ssl_audio_fold0.pt",
    "Multimodal Baseline": "final_models/multimodal_baseline.pt",
    "Enhanced Multimodal (Audio+Text+Linguistic)": "final_models/enhanced_multimodal.pt",
}

# --- Checkpoints (top-level checkpoints/) ---
CHECKPOINT_MODELS = {
    "Best Multimodal (checkpoint)": "checkpoints/best_multimodal_model.pt",
    "Final Multimodal (checkpoint)": "checkpoints/final_multimodal_model.pt",
}

# --- Training With Progress: Audio Folds ---
TRAINING_PROGRESS_MODELS = {
    "SSL Audio Fold 0 (training)": "outputs/training_with_progress/checkpoints/ssl_fold_0/best_model.pt",
    "SSL Audio Fold 1 (training)": "outputs/training_with_progress/checkpoints/ssl_fold_1/best_model.pt",
    "SSL Audio Fold 2 (training)": "outputs/training_with_progress/checkpoints/ssl_fold_2/best_model.pt",
    "ECAPA Audio Fold 0 (training)": "outputs/training_with_progress/checkpoints/ecapa_fold_0/best_model.pt",
    "ECAPA Audio Fold 1 (training)": "outputs/training_with_progress/checkpoints/ecapa_fold_1/best_model.pt",
    "ECAPA Audio Fold 2 (training)": "outputs/training_with_progress/checkpoints/ecapa_fold_2/best_model.pt",
    "Best Model (training progress)": "outputs/training_with_progress/checkpoints/best_model.pt",
    "Final Model (training progress)": "outputs/training_with_progress/checkpoints/final_model.pt",
}

# --- NLP Peak Performance ---
NLP_PEAK_MODELS = {
    "TF-IDF Baseline (NLP peak)": "outputs/nlp_training_peak_performance/checkpoints/tfidf_baseline.pkl",
    "MuRIL Text (NLP peak)": "outputs/nlp_training_peak_performance/checkpoints/muril_text_model.pt",
    "Best Enhanced Multimodal (NLP peak)": "outputs/nlp_training_peak_performance/checkpoints/best_enhanced_multimodal_model_baseline.pt",
    "Final Enhanced Multimodal (NLP peak)": "outputs/nlp_training_peak_performance/checkpoints/final_enhanced_multimodal_model_baseline.pt",
}

# --- Depression Combined Runs ---
DEPRESSION_COMBINED_MODELS = {
    "Best Model (combined run 1)": "outputs/depression_combined_both_20251212_032146/checkpoints/best_model.pt",
    "Final Model (combined run 1)": "outputs/depression_combined_both_20251212_032146/checkpoints/final_model.pt",
    "Best Model (combined run 2)": "outputs/depression_combined_both_20251212_085532/checkpoints/best_model.pt",
    "Final Model (combined run 2)": "outputs/depression_combined_both_20251212_085532/checkpoints/final_model.pt",
    "SSL Fold 0 (combined run 2)": "outputs/depression_combined_both_20251212_085532/checkpoints/ssl_fold_0/best_model.pt",
}

# Category mapping for each model (used to pick the right loader)
MODEL_CATEGORY = {}
for name in FINAL_MODELS:
    if "TF-IDF" in name:
        MODEL_CATEGORY[name] = "tfidf"
    elif "MuRIL" in name:
        MODEL_CATEGORY[name] = "text"
    elif "ECAPA" in name:
        MODEL_CATEGORY[name] = "ecapa"
    elif "SSL" in name:
        MODEL_CATEGORY[name] = "ssl"
    elif "Enhanced" in name:
        MODEL_CATEGORY[name] = "enhanced"
    elif "Multimodal" in name:
        MODEL_CATEGORY[name] = "multimodal"

for name in CHECKPOINT_MODELS:
    MODEL_CATEGORY[name] = "multimodal"

for name in TRAINING_PROGRESS_MODELS:
    if "SSL" in name:
        MODEL_CATEGORY[name] = "ssl"
    elif "ECAPA" in name:
        MODEL_CATEGORY[name] = "ecapa"
    else:
        MODEL_CATEGORY[name] = "multimodal"

for name in NLP_PEAK_MODELS:
    if "TF-IDF" in name:
        MODEL_CATEGORY[name] = "tfidf"
    elif "MuRIL" in name:
        MODEL_CATEGORY[name] = "text"
    else:
        MODEL_CATEGORY[name] = "enhanced"

for name in DEPRESSION_COMBINED_MODELS:
    if "SSL" in name:
        MODEL_CATEGORY[name] = "ssl"
    else:
        MODEL_CATEGORY[name] = "multimodal"

# Combined dict of ALL model paths (used by the UI)
MODEL_PATHS = {}
MODEL_PATHS.update(FINAL_MODELS)
MODEL_PATHS.update(CHECKPOINT_MODELS)
MODEL_PATHS.update(TRAINING_PROGRESS_MODELS)
MODEL_PATHS.update(NLP_PEAK_MODELS)
MODEL_PATHS.update(DEPRESSION_COMBINED_MODELS)

# Group labels for sidebar display
MODEL_GROUPS = {
    "Final Models (Production)": list(FINAL_MODELS.keys()),
    "Top-Level Checkpoints": list(CHECKPOINT_MODELS.keys()),
    "Training Progress (Audio Folds)": list(TRAINING_PROGRESS_MODELS.keys()),
    "NLP Peak Performance": list(NLP_PEAK_MODELS.keys()),
    "Depression Combined Runs": list(DEPRESSION_COMBINED_MODELS.keys()),
}


@st.cache_resource
def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@st.cache_resource
def load_tfidf_model(model_path):
    """Load TF-IDF model."""
    config = DepressionDetectionConfig()
    model = TFIDFClassifier(config=config.tfidf)
    model.load(model_path)
    return model


@st.cache_resource
def load_text_model(model_path, device):
    """Load MuRIL text model."""
    config = DepressionDetectionConfig()
    model = create_muril_model(config.text_model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    tokenizer = create_muril_tokenizer(config.text_model)
    return model, tokenizer


@st.cache_resource
def load_audio_model(model_path, model_type, device):
    """Load audio model (ECAPA or SSL)."""
    config = DepressionDetectionConfig()
    
    if model_type == "ecapa":
        model = create_ecapa_model(config)
    else:  # ssl
        model = create_ssl_model(config)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_multimodal_model(model_path, device):
    """Load multimodal baseline model."""
    config = DepressionDetectionConfig()
    
    # Create component models
    audio_model = create_ssl_model(config)
    text_model = create_muril_model(config.text_model)
    
    # Create multimodal model
    model = create_multimodal_model(
        audio_model=audio_model,
        text_model=text_model,
        config=config.fusion
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    tokenizer = create_muril_tokenizer(config.text_model)
    return model, tokenizer


@st.cache_resource
def load_enhanced_model(model_path, device):
    """Load enhanced multimodal model with linguistic features."""
    config = DepressionDetectionConfig()
    
    # Create component models
    audio_model = create_ssl_model(config)
    text_model = create_muril_model(config.text_model)
    linguistic_analyzer = create_linguistic_analyzer()
    
    # Create enhanced config
    enhanced_config = EnhancedFusionConfig(
        audio_embedding_dim=config.fusion.audio_embedding_dim,
        text_embedding_dim=config.fusion.text_embedding_dim,
        linguistic_feature_dim=config.fusion.linguistic_feature_dim,
        fusion_hidden_dim=config.fusion.fusion_hidden_dim,
        fusion_dropout=config.fusion.fusion_dropout,
        num_classes=config.fusion.num_classes,
        fusion_method=config.fusion.fusion_method,
        allow_unimodal_fallback=True,
        allow_bimodal_fallback=True
    )
    
    # Create enhanced model
    model = create_enhanced_multimodal_model(
        audio_model=audio_model,
        text_model=text_model,
        linguistic_analyzer=linguistic_analyzer,
        config=enhanced_config
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    tokenizer = create_muril_tokenizer(config.text_model)
    return model, tokenizer, linguistic_analyzer


def predict_tfidf(model, text):
    """Predict using TF-IDF model."""
    preprocessor = TextPreprocessor()
    preprocessed = preprocessor.preprocess(text, 'ta')
    preprocessed_str = preprocessed.normalized if hasattr(preprocessed, 'normalized') else str(preprocessed)
    
    prediction = model.predict([preprocessed_str])[0]
    probabilities = model.predict_proba([preprocessed_str])[0]
    
    return {
        'prediction': int(prediction),
        'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
        'confidence': float(max(probabilities)),
        'prob_non_depressed': float(probabilities[0]),
        'prob_depressed': float(probabilities[1])
    }


def predict_text_model(model, tokenizer, text, device):
    """Predict using MuRIL text model."""
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(logits, dim=1)[0]
    
    return {
        'prediction': int(prediction.cpu()),
        'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
        'confidence': float(probabilities.max().cpu()),
        'prob_non_depressed': float(probabilities[0].cpu()),
        'prob_depressed': float(probabilities[1].cpu())
    }


def predict_audio_model(model, audio_array, device):
    """Predict using audio model."""
    audio_tensor = torch.from_numpy(audio_array).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(audio_tensor)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(logits, dim=1)[0]
    
    return {
        'prediction': int(prediction.cpu()),
        'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
        'confidence': float(probabilities.max().cpu()),
        'prob_non_depressed': float(probabilities[0].cpu()),
        'prob_depressed': float(probabilities[1].cpu())
    }


def predict_multimodal(model, tokenizer, audio_array, text, device):
    """Predict using multimodal model."""
    # Prepare audio
    audio_tensor = torch.from_numpy(audio_array).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.to(device)
    
    # Prepare text
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(
            audio_input=audio_tensor,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask
        )
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(logits, dim=1)[0]
    
    return {
        'prediction': int(prediction.cpu()),
        'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
        'confidence': float(probabilities.max().cpu()),
        'prob_non_depressed': float(probabilities[0].cpu()),
        'prob_depressed': float(probabilities[1].cpu())
    }


def predict_enhanced(model, tokenizer, audio_array, text, device):
    """Predict using enhanced multimodal model."""
    # Prepare audio
    audio_tensor = None
    if audio_array is not None:
        audio_tensor = torch.from_numpy(audio_array).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(device)
    
    # Prepare text
    text_input_ids = None
    text_attention_mask = None
    text_raw = None
    
    if text and text.strip():
        encoding = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_input_ids = encoding['input_ids'].to(device)
        text_attention_mask = encoding['attention_mask'].to(device)
        text_raw = [text]
    
    with torch.no_grad():
        logits = model(
            audio_input=audio_tensor,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_raw=text_raw
        )
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = torch.argmax(logits, dim=1)[0]
    
    return {
        'prediction': int(prediction.cpu()),
        'label': 'Depressed' if prediction == 1 else 'Non-Depressed',
        'confidence': float(probabilities.max().cpu()),
        'prob_non_depressed': float(probabilities[0].cpu()),
        'prob_depressed': float(probabilities[1].cpu())
    }


def display_result(result, model_name):
    """Display prediction result."""
    label = result['label']
    confidence = result['confidence']
    prob_dep = result['prob_depressed']
    prob_non = result['prob_non_depressed']
    
    # Color based on prediction
    if 'Depressed' in label:
        color = '#f44336'
        emoji = '⚠️'
        css_class = 'depressed'
    else:
        color = '#4caf50'
        emoji = '✅'
        css_class = 'non-depressed'
    
    st.markdown(f"""
    <div class="result-box {css_class}">
        <h3 style="color: {color}; margin: 0;">{emoji} {model_name}</h3>
        <h2 style="color: {color}; margin: 0.5rem 0;">{label}</h2>
        <p style="margin: 0.5rem 0;">Confidence: <strong>{confidence:.1%}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Non-Depressed", f"{prob_non:.1%}")
    with col2:
        st.metric("Depressed", f"{prob_dep:.1%}")
    
    st.progress(prob_dep)


def main():
    # Header
    st.markdown('<p class="main-header">🎤 Depression Detection - All Models Testing</p>', unsafe_allow_html=True)
    
    device = get_device()
    
    # Sidebar
    st.sidebar.header("⚙️ Model Selection")
    st.sidebar.info(f"🖥️ Device: **{device.upper()}**")

    # Scan which models actually exist on disk
    existing_models = {name: path for name, path in MODEL_PATHS.items() if os.path.exists(path)}
    missing_models = {name: path for name, path in MODEL_PATHS.items() if not os.path.exists(path)}

    st.sidebar.success(f"✅ {len(existing_models)}/{len(MODEL_PATHS)} models found on disk")

    # Let user pick a model group first, then individual models
    selected_group = st.sidebar.selectbox(
        "Model Group",
        ["All Models"] + list(MODEL_GROUPS.keys()),
    )

    if selected_group == "All Models":
        choosable = list(existing_models.keys())
    else:
        choosable = [m for m in MODEL_GROUPS[selected_group] if m in existing_models]

    selected_models = st.sidebar.multiselect(
        "Select Models to Test",
        choosable,
        default=[choosable[0]] if choosable else [],
    )

    if not selected_models:
        st.warning("⚠️ Please select at least one model from the sidebar")
        return

    # Show model info
    with st.sidebar.expander("📊 Selected Model Info"):
        for model_name in selected_models:
            model_path = MODEL_PATHS[model_name]
            cat = MODEL_CATEGORY.get(model_name, "unknown")
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024**2)
                st.write(f"✅ **{model_name}**")
                st.write(f"   Type: `{cat}` | Size: {size:.1f} MB")
                st.write(f"   Path: `{model_path}`")
            else:
                st.write(f"❌ **{model_name}** — not found")

    # Show missing models
    with st.sidebar.expander(f"⚠️ Missing Models ({len(missing_models)})"):
        for model_name, model_path in missing_models.items():
            st.write(f"❌ **{model_name}**")
            st.write(f"   `{model_path}`")
    
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        text_input = st.text_area(
            "Enter text (Tamil/Malayalam):",
            height=150,
            placeholder="Type or paste text here..."
        )
        
        st.header("🎤 Audio Input")
        audio_tab1, audio_tab2 = st.tabs(["🎙️ Record", "📁 Upload"])
        
        with audio_tab1:
            audio_recording = st.audio_input("Click to record")
        
        with audio_tab2:
            uploaded_file = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'ogg']
            )
    
    with col2:
        st.header("ℹ️ Instructions")
        st.markdown("""
        **How to use:**
        
        1. Pick a model group in the sidebar
        2. Select one or more models
        3. Provide text and/or audio
        4. Click **Analyze**
        
        **Model Categories:**
        - **tfidf** — Text only (TF-IDF)
        - **text** — Text only (MuRIL)
        - **ecapa** — Audio only (ECAPA-TDNN)
        - **ssl** — Audio only (WavLM/Wav2Vec)
        - **multimodal** — Requires text + audio
        - **enhanced** — Best with text + audio
        
        **25 models** across 5 groups.
        """)
    
    # Process audio
    audio_array = None
    if audio_recording is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_recording.read())
            audio_path = tmp_file.name
        st.audio(audio_recording)
        audio_array, sr = sf.read(audio_path)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
    elif uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name
        st.audio(uploaded_file)
        audio_array, sr = sf.read(audio_path)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
    
    st.markdown("---")
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not text_input and audio_array is None:
            st.warning("⚠️ Please provide text or audio input")
            return
        
        st.header("🎯 Prediction Results")
        
        for model_name in selected_models:
            model_path = MODEL_PATHS[model_name]
            category = MODEL_CATEGORY.get(model_name, "unknown")

            if not os.path.exists(model_path):
                st.error(f"❌ {model_name}: Model file not found at {model_path}")
                continue
            
            try:
                with st.spinner(f"Running {model_name}..."):
                    # --- TF-IDF models ---
                    if category == "tfidf":
                        if not text_input:
                            st.warning(f"⚠️ {model_name} requires text input")
                            continue
                        model = load_tfidf_model(model_path)
                        result = predict_tfidf(model, text_input)
                        display_result(result, model_name)

                    # --- Text (MuRIL) models ---
                    elif category == "text":
                        if not text_input:
                            st.warning(f"⚠️ {model_name} requires text input")
                            continue
                        model, tokenizer = load_text_model(model_path, device)
                        result = predict_text_model(model, tokenizer, text_input, device)
                        display_result(result, model_name)

                    # --- ECAPA audio models ---
                    elif category == "ecapa":
                        if audio_array is None:
                            st.warning(f"⚠️ {model_name} requires audio input")
                            continue
                        model = load_audio_model(model_path, "ecapa", device)
                        result = predict_audio_model(model, audio_array, device)
                        display_result(result, model_name)

                    # --- SSL audio models ---
                    elif category == "ssl":
                        if audio_array is None:
                            st.warning(f"⚠️ {model_name} requires audio input")
                            continue
                        model = load_audio_model(model_path, "ssl", device)
                        result = predict_audio_model(model, audio_array, device)
                        display_result(result, model_name)

                    # --- Multimodal (baseline) models ---
                    elif category == "multimodal":
                        if not text_input or audio_array is None:
                            st.warning(f"⚠️ {model_name} requires both text and audio")
                            continue
                        model, tokenizer = load_multimodal_model(model_path, device)
                        result = predict_multimodal(model, tokenizer, audio_array, text_input, device)
                        display_result(result, model_name)

                    # --- Enhanced multimodal models ---
                    elif category == "enhanced":
                        model, tokenizer, analyzer = load_enhanced_model(model_path, device)
                        result = predict_enhanced(model, tokenizer, audio_array, text_input, device)
                        display_result(result, model_name)
                        if text_input:
                            with st.expander("🔍 Linguistic Analysis"):
                                st.info("Model analyzed POS tags, dependencies, and NER")

                    else:
                        st.warning(f"⚠️ Unknown model category '{category}' for {model_name}")
            
            except Exception as e:
                st.error(f"❌ Error with {model_name}: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>🏆 DravidianLangTech @ ACL 2026 - All Models Ready for Testing</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
