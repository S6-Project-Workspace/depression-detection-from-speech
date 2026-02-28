"""
FastAPI Main Application

Entry point for the NLP web interface backend.
Provides REST API endpoints and WebSocket support for real-time processing.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pathlib import Path
from typing import Optional

from .api import router as api_router, create_error_handlers
from .metrics_api import router as metrics_router
from .websocket_handler import (
    websocket_endpoint, 
    startup_websocket_manager, 
    shutdown_websocket_manager,
    get_websocket_manager
)
from .audio_processor import initialize_audio_processor, shutdown_audio_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="NLP Web Interface API",
    description="Modern web interface for multimodal NLP analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handlers
http_handler, general_handler = create_error_handlers()
app.add_exception_handler(HTTPException, http_handler)
app.add_exception_handler(Exception, general_handler)

# Include API routers
app.include_router(api_router)
app.include_router(metrics_router)

# Create static directory if it doesn't exist
static_dir = Path("web_interface/static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="web_interface/static"), name="static")

# Root route to serve index.html
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main index.html page."""
    try:
        with open("web_interface/static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index page not found")

# WebSocket endpoint
@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket, session_id: Optional[str] = None):
    """WebSocket endpoint for real-time audio processing."""
    await websocket_endpoint(websocket, session_id)

# Real-time inference endpoints
@app.post("/api/predict/text")
async def predict_text(text: str):
    """Predict depression from text using all text models."""
    from .realtime_inference import get_inference
    inference = get_inference()
    
    results = {}
    if text and text.strip():
        # TF-IDF
        tfidf_result = inference.predict_tfidf(text)
        if 'error' not in tfidf_result:
            results['tfidf'] = tfidf_result
        
        # MuRIL text model
        text_result = inference.predict_text(text)
        if 'error' not in text_result:
            results['text'] = text_result
    
    return results

@app.post("/api/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """Predict depression from audio."""
    import numpy as np
    import soundfile as sf
    from io import BytesIO
    
    from .realtime_inference import get_inference
    inference = get_inference()
    
    # Read audio file
    audio_bytes = await file.read()
    audio_array, sample_rate = sf.read(BytesIO(audio_bytes))
    
    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    
    # Predict
    result = inference.predict_audio(audio_array, sample_rate)
    return result

@app.post("/api/predict/multimodal")
async def predict_multimodal(
    text: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
):
    """Predict depression using multimodal enhanced model."""
    import numpy as np
    import soundfile as sf
    from io import BytesIO
    
    from .realtime_inference import get_inference
    inference = get_inference()
    
    audio_array = None
    sample_rate = 16000
    
    # Read audio if provided
    if file:
        audio_bytes = await file.read()
        audio_array, sample_rate = sf.read(BytesIO(audio_bytes))
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
    
    # Predict with enhanced model
    result = inference.predict_enhanced(audio_array, text, sample_rate)
    return result

@app.post("/api/predict/all")
async def predict_all(
    text: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
):
    """Predict using all available models."""
    import numpy as np
    import soundfile as sf
    from io import BytesIO
    
    from .realtime_inference import get_inference
    inference = get_inference()
    
    audio_array = None
    sample_rate = 16000
    
    # Read audio if provided
    if file:
        audio_bytes = await file.read()
        audio_array, sample_rate = sf.read(BytesIO(audio_bytes))
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
    
    # Predict with all models
    results = inference.predict_all(audio_array, text, sample_rate)
    return results

@app.get("/api/models/available")
async def get_available_models():
    """Get list of available models."""
    from .model_loader import get_model_loader
    loader = get_model_loader()
    return {
        "models": loader.get_available_models(),
        "device": loader.device
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await startup_websocket_manager()
    await initialize_audio_processor()
    
    # Initialize models
    from .model_loader import get_model_loader
    loader = get_model_loader()
    logger.info(f"Models loaded: {loader.get_available_models()}")
    
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on shutdown."""
    await shutdown_websocket_manager()
    await shutdown_audio_processor()
    logger.info("Application shutdown complete")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the metrics dashboard page."""
    try:
        with open("web_interface/static/dashboard.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard</h1><p>Dashboard loading...</p>")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    try:
        with open("web_interface/static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>NLP Web Interface</h1><p>Interface loading...</p>")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    websocket_manager = get_websocket_manager()
    return {
        "status": "healthy", 
        "service": "nlp-web-interface",
        "active_connections": websocket_manager.get_connection_count(),
        "websocket_manager": "running"
    }

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon or return 204 No Content."""
    from fastapi.responses import Response
    return Response(status_code=204)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)