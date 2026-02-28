"""
Real-Time Depression Detection - Complete Model Testing Interface

Streamlit app for testing all trained models:
- TF-IDF Baseline
- MuRIL Text Model
- Audio Models (ECAPA, SSL)
- Enhanced Multimodal with Linguistic Features

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys
import tempfile
import numpy as np
import torch
import streamlit as st
from pathlib import Path
import soundfile as sf

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_interface.model_loader import get_model_loader
from web_interface.realtime_inference import get_inference

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
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
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
    .model-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load all trained models."""
    loader = get_model_loader()
    available = loader.get_available_models()
    return loader, available


def display_result(result: dict, model_name: str):
    """Display prediction result with styling."""
    if 'error' in result:
        st.error(f"❌ {model_name}: {result['error']}")
        return
    
    prediction = result.get('label', 'Unknown')
    confidence = result.get('confidence', 0.0)
    probs = result.get('probabilities', {})
    
    # Color based on prediction
    if 'Depressed' in prediction:
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
        <h3 style="color: {color}; margin: 0;">{emoji} {model_name}</h3>
        <h2 style="color: {color}; margin: 0.5rem 0;">{prediction}</h2>
        <p style="margin: 0.5rem 0;">Confidence: <strong>{confidence:.1%}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability bars
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Non-Depressed", f"{probs.get('non_depressed', 0):.1%}")
    with col2:
        st.metric("Depressed", f"{probs.get('depressed', 0):.1%}")
    
    # Progress bar
    st.progress(probs.get('depressed', 0))


def main():
    # Header
    st.markdown('<p class="main-header">🎤 Depression Detection - All Models</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Test all trained models in real-time</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        loader, available_models = load_all_models()
        inference = get_inference()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.info(f"🖥️ Device: **{loader.device.upper()}**")
    st.sidebar.success(f"✅ Loaded {len(available_models)} models")
    
    # Show available models
    st.sidebar.markdown("### Available Models:")
    for model in available_models:
        st.sidebar.markdown(f"- ✓ {model.upper()}")
    
    # Model selection
    test_mode = st.sidebar.selectbox(
        "Testing Mode",
        [
            "All Models Comparison",
            "Text Models Only",
            "Audio Models Only",
            "Enhanced Multimodal Only"
        ]
    )
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        text_input = st.text_area(
            "Enter text (Tamil/Malayalam):",
            height=150,
            placeholder="Type or paste text here for analysis..."
        )
        
        st.header("🎤 Audio Input")
        
        # Audio input tabs
        audio_tab1, audio_tab2 = st.tabs(["🎙️ Record", "📁 Upload"])
        
        with audio_tab1:
            audio_recording = st.audio_input("Click to record")
        
        with audio_tab2:
            uploaded_file = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'ogg']
            )
    
    with col2:
        st.header("ℹ️ Model Info")
        st.markdown("""
        **Available Models:**
        
        **Text Models:**
        - TF-IDF Baseline
        - MuRIL (Transformer)
        
        **Audio Models:**
        - ECAPA-TDNN
        - SSL (Wav2Vec2)
        
        **Multimodal:**
        - Enhanced (Audio + Text + Linguistic)
        
        **Features:**
        - Real-time prediction
        - Multiple model comparison
        - Confidence scores
        - Linguistic analysis
        """)
    
    st.markdown("---")
    
    # Process inputs
    audio_array = None
    audio_path = None
    
    # Handle audio input
    if audio_recording is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_recording.read())
            audio_path = tmp_file.name
        st.audio(audio_recording)
        
        # Load audio
        audio_array, sr = sf.read(audio_path)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
    elif uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name
        st.audio(uploaded_file)
        
        # Load audio
        audio_array, sr = sf.read(audio_path)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not text_input and audio_array is None:
            st.warning("⚠️ Please provide text or audio input")
            return
        
        st.header("🎯 Prediction Results")
        
        # Determine which models to run
        run_text = test_mode in ["All Models Comparison", "Text Models Only"]
        run_audio = test_mode in ["All Models Comparison", "Audio Models Only"]
        run_enhanced = test_mode in ["All Models Comparison", "Enhanced Multimodal Only"]
        
        # Text Models
        if run_text and text_input:
            st.subheader("📝 Text-Only Models")
            
            col1, col2 = st.columns(2)
            
            # TF-IDF
            if 'tfidf' in available_models:
                with col1:
                    with st.spinner("Running TF-IDF..."):
                        result = inference.predict_tfidf(text_input)
                        display_result(result, "TF-IDF Baseline")
            
            # MuRIL Text
            if 'text' in available_models:
                with col2:
                    with st.spinner("Running MuRIL..."):
                        result = inference.predict_text(text_input)
                        display_result(result, "MuRIL Text Model")
        
        # Audio Models
        if run_audio and audio_array is not None:
            st.subheader("🎤 Audio-Only Models")
            
            if 'audio' in available_models:
                with st.spinner("Running Audio Model..."):
                    result = inference.predict_audio(audio_array)
                    display_result(result, "Audio Model (SSL)")
        
        # Enhanced Multimodal
        if run_enhanced:
            st.subheader("🔗 Enhanced Multimodal Model")
            
            if 'enhanced' in available_models:
                with st.spinner("Running Enhanced Multimodal..."):
                    result = inference.predict_enhanced(audio_array, text_input)
                    
                    # Show modalities used
                    modalities = result.get('modalities_used', {})
                    modality_str = []
                    if modalities.get('audio'):
                        modality_str.append("Audio")
                    if modalities.get('text'):
                        modality_str.append("Text")
                    if modalities.get('linguistic'):
                        modality_str.append("Linguistic")
                    
                    model_label = f"Enhanced Multimodal ({' + '.join(modality_str)})"
                    display_result(result, model_label)
                    
                    # Show linguistic features if used
                    if modalities.get('linguistic'):
                        with st.expander("🔍 Linguistic Features"):
                            st.info("Model analyzed POS tags, dependency parsing, and NER")
        
        # All Models Comparison
        if test_mode == "All Models Comparison" and text_input and audio_array is not None:
            st.markdown("---")
            st.subheader("📊 Model Comparison Summary")
            
            with st.spinner("Running all models..."):
                all_results = inference.predict_all(audio_array, text_input)
            
            # Create comparison table
            import pandas as pd
            comparison_data = []
            
            for model_name, result in all_results.items():
                if 'error' not in result:
                    comparison_data.append({
                        'Model': model_name.upper(),
                        'Prediction': result.get('label', 'Unknown'),
                        'Confidence': f"{result.get('confidence', 0):.1%}",
                        'Depressed Prob': f"{result.get('probabilities', {}).get('depressed', 0):.1%}"
                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Consensus prediction
                depressed_count = sum(1 for d in comparison_data if 'Depressed' in d['Prediction'])
                total_count = len(comparison_data)
                
                if depressed_count > total_count / 2:
                    consensus = "Depressed"
                    consensus_color = "#f44336"
                else:
                    consensus = "Non-Depressed"
                    consensus_color = "#4caf50"
                
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; 
                            border-left: 4px solid #1976d2; margin: 1rem 0;">
                    <h3 style="color: #1976d2; margin: 0;">🎯 Consensus Prediction</h3>
                    <h2 style="color: {consensus_color}; margin: 0.5rem 0;">{consensus}</h2>
                    <p style="margin: 0;">Based on {depressed_count}/{total_count} models predicting Depressed</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>🏆 DravidianLangTech @ ACL 2026 - Depression Detection System</p>
        <p>All models trained and ready for real-time testing</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
