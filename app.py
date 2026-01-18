"""
Real-Time Depression Detection Streamlit App

This app uses the trained ECAPA-TDNN, SSL, and Multimodal models to detect
depression markers in speech audio. Supports ASR transcription for multimodal
analysis.

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
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from preprocessing import AudioPreprocessor
from ssl_model import create_ssl_model
from ecapa_model import create_ecapa_model
from multimodal_model import create_multimodal_model
from text_model import create_muril_model, MuRILTokenizerWrapper as MuRILTokenizer
from text_preprocessing import create_preprocessor

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
    
    # Try to load multimodal model
    multimodal_path = os.path.join(checkpoint_dir, 'multimodal_fold_0', 'best_model.pt')
    if os.path.exists(multimodal_path):
        try:
            # Create component models
            audio_model = create_ssl_model(config)
            text_model = create_muril_model(config.text_model)
            
            # Create multimodal model
            multimodal_model = create_multimodal_model(
                audio_model=audio_model,
                text_model=text_model,
                config=config.fusion
            )
            
            checkpoint = torch.load(multimodal_path, map_location=device, weights_only=False)
            multimodal_model.load_state_dict(checkpoint['model_state_dict'])
            multimodal_model.to(device)
            multimodal_model.eval()
            models['multimodal'] = multimodal_model
            st.success("✅ Multimodal model loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load Multimodal model: {e}")
    
    return models, config


@st.cache_resource
def load_asr_pipeline():
    """Load ASR pipeline for transcription (cached)."""
    try:
        from asr_pipeline import create_asr_pipeline
        from config import get_config
        config = get_config('combined')
        asr = create_asr_pipeline(config.asr)
        return asr
    except Exception as e:
        st.warning(f"Could not load ASR pipeline: {e}")
        return None


@st.cache_resource
def load_text_components():
    """Load text tokenizer and preprocessor (cached)."""
    try:
        config = get_config('combined')
        tokenizer = MuRILTokenizer(config.text_model)
        preprocessor = create_preprocessor(config.text_preprocessing)
        return tokenizer, preprocessor
    except Exception as e:
        st.warning(f"Could not load text components: {e}")
        return None, None


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


def transcribe_audio(audio_path: str, language: str = "tamil") -> dict:
    """Transcribe audio using ASR pipeline.
    
    Args:
        audio_path: Path to audio file
        language: Language for transcription ("tamil" or "malayalam")
        
    Returns:
        Dictionary with transcript and metadata
    """
    asr = load_asr_pipeline()
    
    if asr is None:
        return {
            'text': '',
            'error': 'ASR pipeline not available',
            'confidence': 0.0
        }
    
    try:
        result = asr.transcribe(audio_path, language=language)
        return {
            'text': result.text,
            'error': result.error,
            'confidence': result.confidence,
            'language': language
        }
    except Exception as e:
        return {
            'text': '',
            'error': str(e),
            'confidence': 0.0
        }


def predict_multimodal(model, audio_data: dict, transcript: str, config, device: str) -> dict:
    """Make prediction using multimodal model (audio + text).
    
    Args:
        model: Multimodal model
        audio_data: Processed audio data
        transcript: Text transcript
        config: Configuration
        device: Device to use
        
    Returns:
        Dictionary with prediction results
    """
    tokenizer, preprocessor = load_text_components()
    
    if tokenizer is None:
        # Fall back to audio-only prediction
        return predict_multimodal_audio_only(model, audio_data, config, device)
    
    all_probs = []
    
    # Preprocess text
    if preprocessor is not None and transcript:
        try:
            preprocessed = preprocessor.preprocess(transcript)
            text_input = preprocessed.morpheme_text if preprocessed.morpheme_text else preprocessed.normalized
        except Exception:
            text_input = transcript
    else:
        text_input = transcript
    
    # Tokenize text
    try:
        tokens = tokenizer.tokenize(text_input)
        input_ids = tokens['input_ids'].to(device)
        text_attention_mask = tokens['attention_mask'].to(device)
    except Exception as e:
        st.warning(f"Tokenization failed: {e}")
        return predict_multimodal_audio_only(model, audio_data, config, device)
    
    # Process each audio chunk with the same text
    for chunk in audio_data['chunks']:
        waveform = torch.FloatTensor(chunk).unsqueeze(0).to(device)
        
        with torch.no_grad():
            try:
                logits = model(
                    audio_input=waveform,
                    text_input_ids=input_ids,
                    text_attention_mask=text_attention_mask
                )
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs[0, 1].cpu().item())
            except Exception as e:
                st.warning(f"Multimodal forward failed: {e}")
                # Fall back to audio-only
                logits = model.forward_audio_only(audio_input=waveform)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs[0, 1].cpu().item())
    
    mean_prob = np.mean(all_probs)
    
    return {
        'probability': mean_prob,
        'prediction': 'Depressed' if mean_prob > 0.5 else 'Non-Depressed',
        'confidence': max(mean_prob, 1 - mean_prob),
        'chunk_probs': all_probs,
        'used_text': bool(transcript)
    }


def predict_multimodal_audio_only(model, audio_data: dict, config, device: str) -> dict:
    """Make prediction using multimodal model with audio only (fallback).
    
    Args:
        model: Multimodal model
        audio_data: Processed audio data
        config: Configuration
        device: Device to use
        
    Returns:
        Dictionary with prediction results
    """
    all_probs = []
    
    for chunk in audio_data['chunks']:
        waveform = torch.FloatTensor(chunk).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model.forward_audio_only(audio_input=waveform)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs[0, 1].cpu().item())
    
    mean_prob = np.mean(all_probs)
    
    return {
        'probability': mean_prob,
        'prediction': 'Depressed' if mean_prob > 0.5 else 'Non-Depressed',
        'confidence': max(mean_prob, 1 - mean_prob),
        'chunk_probs': all_probs,
        'used_text': False
    }


def display_transcript(transcript_result: dict):
    """Display ASR transcript with styling."""
    if transcript_result.get('error'):
        st.warning(f"⚠️ Transcription issue: {transcript_result['error']}")
        return
    
    text = transcript_result.get('text', '')
    confidence = transcript_result.get('confidence', 0.0)
    language = transcript_result.get('language', 'unknown')
    
    if text:
        st.markdown(f"""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #1976d2; margin: 1rem 0;">
            <h4 style="color: #1976d2; margin: 0 0 0.5rem 0;">📝 ASR Transcript ({language.title()})</h4>
            <p style="font-size: 1.1rem; margin: 0; color: #333;">{text}</p>
            <p style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
                Confidence: {confidence:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📝 No transcript available for this audio.")


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
    
    # Model selection - updated to include multimodal
    model_choice = st.sidebar.selectbox(
        "Model to Use",
        [
            "ECAPA-TDNN (Recommended)",
            "SSL (Wav2Vec2)",
            "Multimodal (Audio + Text)",
            "Both (Ensemble)",
            "All Models Comparison"
        ],
        help="Select which model(s) to use for prediction"
    )
    
    # Language selection for ASR
    language_choice = st.sidebar.selectbox(
        "Audio Language",
        ["Tamil", "Malayalam"],
        help="Select the language of the audio for ASR transcription"
    )
    
    # Enable/disable ASR transcription
    enable_asr = st.sidebar.checkbox(
        "Enable ASR Transcription",
        value=True,
        help="Generate text transcript from audio using ASR"
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
        - **Multimodal**: Audio + Text fusion
        
        **Best Performance:**
        - ECAPA: ~98.4% Macro-F1
        - SSL: ~77.4% Macro-F1
        - Multimodal: Combines both modalities
        
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
            
            # ASR Transcription
            transcript_text = ""
            if enable_asr:
                with st.spinner("Transcribing audio..."):
                    transcript_result = transcribe_audio(
                        audio_path,
                        language=language_choice.lower()
                    )
                    display_transcript(transcript_result)
                    transcript_text = transcript_result.get('text', '')
            
            # Make predictions
            st.header("🎯 Prediction Results")
            
            # Determine number of columns based on model choice
            if model_choice == "All Models Comparison":
                num_cols = min(3, len(models))
                results_cols = st.columns(num_cols)
            elif model_choice == "Both (Ensemble)":
                results_cols = st.columns(2)
            else:
                results_cols = [st.container()]
            
            col_idx = 0
            
            # ECAPA-TDNN prediction
            if model_choice in ["ECAPA-TDNN (Recommended)", "Both (Ensemble)", "All Models Comparison"]:
                if 'ecapa' in models:
                    with results_cols[col_idx % len(results_cols)]:
                        ecapa_results = predict_ecapa(models['ecapa'], audio_data, config, device)
                        display_results(ecapa_results, "ECAPA-TDNN")
                    col_idx += 1
                else:
                    st.warning("ECAPA model not available")
            
            # SSL prediction
            if model_choice in ["SSL (Wav2Vec2)", "Both (Ensemble)", "All Models Comparison"]:
                if 'ssl' in models:
                    with results_cols[col_idx % len(results_cols)]:
                        ssl_results = predict_ssl(models['ssl'], audio_data, config, device)
                        display_results(ssl_results, "SSL (Wav2Vec2)")
                    col_idx += 1
                else:
                    st.warning("SSL model not available")
            
            # Multimodal prediction
            if model_choice in ["Multimodal (Audio + Text)", "All Models Comparison"]:
                if 'multimodal' in models:
                    with results_cols[col_idx % len(results_cols)] if col_idx < len(results_cols) else st.container():
                        multimodal_results = predict_multimodal(
                            models['multimodal'],
                            audio_data,
                            transcript_text,
                            config,
                            device
                        )
                        model_label = "Multimodal (Audio + Text)" if multimodal_results.get('used_text') else "Multimodal (Audio Only)"
                        display_results(multimodal_results, model_label)
                    col_idx += 1
                else:
                    st.warning("Multimodal model not available. Train with: `python train.py --model multimodal`")
            
            # Ensemble result for audio-only models
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
            
            # Model comparison summary for "All Models Comparison"
            if model_choice == "All Models Comparison":
                st.markdown("---")
                st.header("📊 Model Comparison Summary")
                
                comparison_data = []
                if 'ecapa' in models:
                    comparison_data.append({
                        'Model': 'ECAPA-TDNN',
                        'Prediction': ecapa_results['prediction'],
                        'Confidence': f"{ecapa_results['confidence']:.1%}",
                        'Depression Prob': f"{ecapa_results['probability']:.1%}"
                    })
                if 'ssl' in models:
                    comparison_data.append({
                        'Model': 'SSL (Wav2Vec2)',
                        'Prediction': ssl_results['prediction'],
                        'Confidence': f"{ssl_results['confidence']:.1%}",
                        'Depression Prob': f"{ssl_results['probability']:.1%}"
                    })
                if 'multimodal' in models:
                    comparison_data.append({
                        'Model': 'Multimodal',
                        'Prediction': multimodal_results['prediction'],
                        'Confidence': f"{multimodal_results['confidence']:.1%}",
                        'Depression Prob': f"{multimodal_results['probability']:.1%}"
                    })
                
                if comparison_data:
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    st.table(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>🏆 DravidianLangTech @ ACL 2026 Shared Task</p>
        <p>Depression Detection in Tamil and Malayalam Speech</p>
        <p>Supports Audio-Only, Text-Only, and Multimodal Analysis</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
