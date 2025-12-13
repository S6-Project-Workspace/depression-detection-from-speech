"""
Real-Time Depression Detection Streamlit App

This app uses the trained ECAPA-TDNN and SSL models to detect
depression markers in speech audio.

Usage:
    streamlit run app.py
"""

import os
import sys
import tempfile
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import streamlit as st
from pathlib import Path
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from preprocessing import AudioPreprocessor
from ssl_model import create_ssl_model
from ecapa_model import create_ecapa_model

# Page configuration
st.set_page_config(
    page_title="Depression Detection in Speech",
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
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .depressed {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .non-depressed {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(checkpoint_dir: str, device: str):
    """Load trained models (cached)."""
    config = get_config('combined')
    config.device = device
    
    models = {}
    
    # Try to load ECAPA model (more reliable based on evaluation)
    ecapa_path = os.path.join(checkpoint_dir, 'ecapa_fold_0', 'best_model.pt')
    if os.path.exists(ecapa_path):
        try:
            model = create_ecapa_model(config)
            checkpoint = torch.load(ecapa_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            models['ecapa'] = model
            st.success("✅ ECAPA-TDNN model loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load ECAPA model: {e}")
    
    # Try to load SSL model
    ssl_path = os.path.join(checkpoint_dir, 'ssl_fold_0', 'best_model.pt')
    if os.path.exists(ssl_path):
        try:
            model = create_ssl_model(config)
            checkpoint = torch.load(ssl_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            models['ssl'] = model
            st.success("✅ SSL (Wav2Vec2) model loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load SSL model: {e}")
    
    return models, config


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def process_audio(audio_path: str, config) -> dict:
    """Process audio file and extract features."""
    import soundfile as sf
    import librosa
    
    # Load audio using soundfile or librosa
    try:
        audio_data, sample_rate = sf.read(audio_path, dtype='float32')
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert stereo to mono
    except Exception:
        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e:
            st.error(f"Failed to load audio: {e}")
            return None
    
    # Convert to tensor
    waveform = torch.from_numpy(audio_data).float()
    
    # Resample to target sample rate if needed
    target_sr = config.preprocessing.sampling_rate
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
    
    # Normalize: remove DC offset and peak normalize
    waveform = waveform - waveform.mean()
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
    
    # Chunk audio for processing
    chunk_samples = int(config.preprocessing.chunk_duration_sec * sample_rate)
    overlap_sec = config.preprocessing.chunk_overlap_inference_sec
    hop_samples = int((config.preprocessing.chunk_duration_sec - overlap_sec) * sample_rate)
    
    chunks = []
    total_samples = len(waveform)
    
    if total_samples <= chunk_samples:
        # Pad if shorter than one chunk
        if total_samples < chunk_samples:
            padding = torch.zeros(chunk_samples - total_samples)
            waveform_padded = torch.cat([waveform, padding])
        else:
            waveform_padded = waveform
        chunks.append(waveform_padded.numpy())
    else:
        # Create overlapping chunks
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                padding = torch.zeros(chunk_samples - len(chunk))
                chunk = torch.cat([chunk, padding])
            
            chunks.append(chunk.numpy())
            start += hop_samples
    
    return {
        'waveform': waveform.numpy(),
        'chunks': chunks,
        'sample_rate': sample_rate,
        'duration': len(waveform) / sample_rate
    }


def predict_ecapa(model, audio_data: dict, config, device: str) -> dict:
    """Make prediction using ECAPA-TDNN model."""
    import torchaudio
    
    all_probs = []
    
    # Process each chunk
    for chunk in audio_data['chunks']:
        # Convert to tensor
        waveform = torch.FloatTensor(chunk).unsqueeze(0)
        
        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.preprocessing.sampling_rate,
            n_fft=int(config.ecapa_model.window_size_ms * config.preprocessing.sampling_rate / 1000),
            hop_length=int(config.ecapa_model.hop_size_ms * config.preprocessing.sampling_rate / 1000),
            n_mels=config.ecapa_model.n_mels
        )
        
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = mel_spec.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(mel_spec)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs[0, 1].cpu().item())
    
    # Aggregate predictions
    mean_prob = np.mean(all_probs)
    
    return {
        'probability': mean_prob,
        'prediction': 'Depressed' if mean_prob > 0.5 else 'Non-Depressed',
        'confidence': max(mean_prob, 1 - mean_prob),
        'chunk_probs': all_probs
    }


def predict_ssl(model, audio_data: dict, config, device: str) -> dict:
    """Make prediction using SSL model."""
    all_probs = []
    
    # Process each chunk
    for chunk in audio_data['chunks']:
        # Convert to tensor
        waveform = torch.FloatTensor(chunk).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(waveform)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs[0, 1].cpu().item())
    
    # Aggregate predictions
    mean_prob = np.mean(all_probs)
    
    return {
        'probability': mean_prob,
        'prediction': 'Depressed' if mean_prob > 0.5 else 'Non-Depressed',
        'confidence': max(mean_prob, 1 - mean_prob),
        'chunk_probs': all_probs
    }


def display_results(results: dict, model_name: str):
    """Display prediction results with visualization."""
    prob = results['probability']
    prediction = results['prediction']
    confidence = results['confidence']
    
    # Color based on prediction
    if prediction == 'Depressed':
        color = '#f44336'
        bg_color = '#ffebee'
        emoji = '⚠️'
    else:
        color = '#4caf50'
        bg_color = '#e8f5e9'
        emoji = '✅'
    
    # Display result card
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 1.5rem; border-radius: 10px; 
                border: 2px solid {color}; margin: 1rem 0;">
        <h3 style="color: {color}; margin: 0;">{emoji} {model_name} Prediction</h3>
        <h2 style="color: {color}; margin: 0.5rem 0;">{prediction}</h2>
        <p style="margin: 0.5rem 0;">Confidence: <strong>{confidence:.1%}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability bar
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Non-Depressed Probability", f"{(1-prob):.1%}")
    with col2:
        st.metric("Depressed Probability", f"{prob:.1%}")
    
    # Progress bar visualization
    st.progress(prob)
    
    # Chunk-level analysis if multiple chunks
    if len(results['chunk_probs']) > 1:
        with st.expander("📊 Chunk-level Analysis"):
            import pandas as pd
            chunk_df = pd.DataFrame({
                'Chunk': range(1, len(results['chunk_probs']) + 1),
                'Depression Probability': results['chunk_probs']
            })
            st.bar_chart(chunk_df.set_index('Chunk'))


def main():
    # Header
    st.markdown('<p class="main-header">🎤 Depression Detection in Speech</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">DravidianLangTech @ ACL 2026 - Real-time Analysis Tool</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Checkpoint directory
    default_checkpoint = "outputs/training_with_progress/checkpoints"
    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint Directory",
        value=default_checkpoint,
        help="Path to the directory containing trained model checkpoints"
    )
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Model to Use",
        ["ECAPA-TDNN (Recommended)", "SSL (Wav2Vec2)", "Both (Ensemble)"],
        help="ECAPA-TDNN showed more consistent performance in evaluation"
    )
    
    # Threshold adjustment
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability threshold for classifying as Depressed"
    )
    
    # Device info
    device = get_device()
    st.sidebar.info(f"🖥️ Using device: **{device.upper()}**")
    
    # Load models
    if not os.path.exists(checkpoint_dir):
        st.error(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        st.info("Please run training first with: `python train.py --language combined --epochs 10 --folds 3`")
        return
    
    with st.spinner("Loading models..."):
        models, config = load_models(checkpoint_dir, device)
    
    if not models:
        st.error("❌ No models could be loaded. Please check the checkpoint directory.")
        return
    
    st.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎤 Audio Input")
        
        # Create tabs for different input methods
        input_tab1, input_tab2, input_tab3 = st.tabs(["🎙️ Record Voice", "📁 Upload File", "🎵 Dataset Sample"])
        
        with input_tab1:
            st.markdown("**Record your voice directly:**")
            audio_recording = st.audio_input(
                "Click to record",
                help="Click the microphone button to start recording. Click again to stop."
            )
            
            if audio_recording:
                st.success("✅ Recording captured! Click 'Analyze Audio' below.")
        
        with input_tab2:
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
                help="Upload a speech audio file for depression detection"
            )
        
        with input_tab3:
            st.markdown("**Try with sample audio from dataset:**")
            sample_dir = "NLP Dataset"
            
            if st.button("🎵 Use Random Sample from Dataset"):
                # Find a random audio file from the dataset
                import random
                audio_files = []
                for root, dirs, files in os.walk(sample_dir):
                    for f in files:
                        if f.endswith('.wav'):
                            audio_files.append(os.path.join(root, f))
                
                if audio_files:
                    sample_file = random.choice(audio_files)
                    st.session_state['sample_file'] = sample_file
                    st.success(f"Selected: {os.path.basename(sample_file)}")
                else:
                    st.warning("No audio files found in dataset directory")
    
    with col2:
        st.header("ℹ️ Model Info")
        st.markdown("""
        **Available Models:**
        - **ECAPA-TDNN**: Spectral/prosodic features
        - **SSL**: Wav2Vec2 linguistic features
        
        **Best Performance:**
        - ECAPA: ~98.4% Macro-F1
        - SSL: ~77.4% Macro-F1
        
        **Tips for Recording:**
        - Speak naturally for 5-10 seconds
        - Use a quiet environment
        - Speak about any topic
        """)
    
    st.markdown("---")
    
    # Process audio - handle all three input sources
    audio_path = None
    audio_source = None
    
    # Priority: Recording > Upload > Sample
    if audio_recording is not None:
        # Save recorded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_recording.read())
            audio_path = tmp_file.name
        st.subheader("🎙️ Your Recording:")
        st.audio(audio_recording)
        audio_source = "recording"
        
    elif uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name
        st.subheader("📁 Uploaded File:")
        st.audio(uploaded_file)
        audio_source = "upload"
        
    elif 'sample_file' in st.session_state:
        audio_path = st.session_state['sample_file']
        st.subheader("🎵 Dataset Sample:")
        st.audio(audio_path)
        audio_source = "sample"
        # Show label from path
        if 'Depressed' in audio_path or '/D_' in audio_path:
            st.info("📋 Ground Truth: **Depressed**")
        elif 'Non' in audio_path or '/ND' in audio_path:
            st.info("📋 Ground Truth: **Non-Depressed**")
    
    if audio_path and st.button("🔍 Analyze Audio", type="primary"):
        with st.spinner("Processing audio..."):
            # Process audio
            audio_data = process_audio(audio_path, config)
            
            if audio_data is None:
                st.error("❌ Failed to process audio file")
                return
            
            # Display audio info
            st.info(f"📊 Audio Duration: {audio_data['duration']:.2f}s | Chunks: {len(audio_data['chunks'])}")
            
            # Make predictions
            st.header("🎯 Prediction Results")
            
            results_cols = st.columns(2 if model_choice == "Both (Ensemble)" else 1)
            
            if model_choice in ["ECAPA-TDNN (Recommended)", "Both (Ensemble)"]:
                if 'ecapa' in models:
                    with results_cols[0] if model_choice == "Both (Ensemble)" else st.container():
                        ecapa_results = predict_ecapa(models['ecapa'], audio_data, config, device)
                        display_results(ecapa_results, "ECAPA-TDNN")
                else:
                    st.warning("ECAPA model not available")
            
            if model_choice in ["SSL (Wav2Vec2)", "Both (Ensemble)"]:
                if 'ssl' in models:
                    col_idx = 1 if model_choice == "Both (Ensemble)" else 0
                    with results_cols[col_idx] if model_choice == "Both (Ensemble)" else st.container():
                        ssl_results = predict_ssl(models['ssl'], audio_data, config, device)
                        display_results(ssl_results, "SSL (Wav2Vec2)")
                else:
                    st.warning("SSL model not available")
            
            # Ensemble result
            if model_choice == "Both (Ensemble)" and 'ecapa' in models and 'ssl' in models:
                st.markdown("---")
                st.header("🔗 Ensemble Result")
                
                # Weighted average (ECAPA gets higher weight based on performance)
                ensemble_prob = 0.7 * ecapa_results['probability'] + 0.3 * ssl_results['probability']
                ensemble_pred = 'Depressed' if ensemble_prob > threshold else 'Non-Depressed'
                ensemble_conf = max(ensemble_prob, 1 - ensemble_prob)
                
                ensemble_results = {
                    'probability': ensemble_prob,
                    'prediction': ensemble_pred,
                    'confidence': ensemble_conf,
                    'chunk_probs': [ensemble_prob]  # Single aggregated value
                }
                display_results(ensemble_results, "Ensemble (70% ECAPA + 30% SSL)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>🏆 DravidianLangTech @ ACL 2026 Shared Task</p>
        <p>Depression Detection in Tamil and Malayalam Speech</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
