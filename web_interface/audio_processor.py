"""
Audio Processor for Real-time ASR Integration

This module handles real-time audio processing, integrating with the existing
ASR pipeline and providing audio quality monitoring and feedback.

Requirements: 1.2, 1.4, 8.1
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import base64
import json

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from .models import AudioQuality, SessionData, SessionStatus

# Conditional imports for ASR pipeline
try:
    from asr_pipeline import ASRPipeline, TranscriptResult, create_asr_pipeline
    from config import ASRConfig
    HAS_ASR_PIPELINE = True
except ImportError:
    # Mock classes for testing when ASR pipeline is not available
    class ASRPipeline:
        def __init__(self, config=None):
            pass
        
        def transcribe(self, audio_path, language):
            from dataclasses import dataclass
            @dataclass
            class MockResult:
                text: str = "Mock transcription"
                confidence: float = 0.9
                error: str = None
            return MockResult()
    
    class TranscriptResult:
        def __init__(self):
            self.text = "Mock transcription"
            self.confidence = 0.9
            self.error = None
    
    def create_asr_pipeline(config=None):
        return ASRPipeline(config)
    
    class ASRConfig:
        def __init__(self):
            pass
    
    HAS_ASR_PIPELINE = False

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    data: np.ndarray
    sample_rate: int
    timestamp: float
    chunk_id: str
    session_id: str


@dataclass
class ProcessingResult:
    """Result of audio processing."""
    transcription: str
    confidence: float
    audio_quality: AudioQuality
    processing_time: float
    chunk_id: str
    session_id: str
    error: Optional[str] = None


class AudioQualityMonitor:
    """Monitors audio quality in real-time."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize audio quality monitor.
        
        Args:
            sample_rate: Expected sample rate for audio
        """
        self.sample_rate = sample_rate
        self.min_signal_threshold = 0.01  # Minimum signal level
        self.max_noise_threshold = 0.3    # Maximum acceptable noise level
        self.min_snr_db = 10.0            # Minimum SNR in dB
        
    def analyze_quality(self, audio_data: np.ndarray) -> AudioQuality:
        """Analyze audio quality metrics.
        
        Args:
            audio_data: Audio waveform as numpy array
            
        Returns:
            AudioQuality object with metrics and warnings
        """
        warnings = []
        
        # Handle empty or invalid audio
        if len(audio_data) == 0:
            return AudioQuality(
                signal_level=0.0,
                noise_level=0.0,
                snr=0.0,
                quality_score=0.0,
                warnings=["Empty audio data"]
            )
        
        # Calculate signal level (RMS)
        signal_level = float(np.sqrt(np.mean(audio_data ** 2)))
        
        # Handle NaN values
        if np.isnan(signal_level) or np.isinf(signal_level):
            signal_level = 0.0
        
        # Estimate noise level (using quieter segments)
        # Simple approach: use 10th percentile of absolute values
        sorted_abs = np.sort(np.abs(audio_data))
        if len(sorted_abs) > 0:
            noise_level = float(np.mean(sorted_abs[:max(1, len(sorted_abs)//10)]))
        else:
            noise_level = 0.0
        
        # Handle NaN values
        if np.isnan(noise_level) or np.isinf(noise_level):
            noise_level = 0.0
        
        # Calculate SNR
        if noise_level > 0:
            snr = 20 * np.log10(max(signal_level, 1e-10) / max(noise_level, 1e-10))
        else:
            snr = 60.0  # Very high SNR if no noise detected
        
        # Handle NaN values
        if np.isnan(snr) or np.isinf(snr):
            snr = 0.0
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio_data) > 0.95) / len(audio_data)
        if clipping_ratio > 0.01:  # More than 1% clipped
            warnings.append(f"Audio clipping detected ({clipping_ratio:.1%})")
        
        # Check signal level
        if signal_level < self.min_signal_threshold:
            warnings.append("Signal level too low - speak louder or move closer to microphone")
        elif signal_level > 0.8:
            warnings.append("Signal level too high - reduce input gain or move away from microphone")
        
        # Check noise level
        if noise_level > self.max_noise_threshold:
            warnings.append("High background noise detected - find a quieter environment")
        
        # Check SNR
        if snr < self.min_snr_db:
            warnings.append(f"Poor signal-to-noise ratio ({snr:.1f} dB) - reduce background noise")
        
        # Calculate overall quality score (0-1)
        quality_score = self._calculate_quality_score(signal_level, noise_level, snr, clipping_ratio)
        
        return AudioQuality(
            signal_level=min(max(signal_level, 0.0), 1.0),  # Clamp to [0, 1]
            noise_level=min(max(noise_level, 0.0), 1.0),   # Clamp to [0, 1]
            snr=float(snr),
            quality_score=min(max(quality_score, 0.0), 1.0),  # Clamp to [0, 1]
            warnings=warnings
        )
    
    def _calculate_quality_score(
        self, 
        signal_level: float, 
        noise_level: float, 
        snr: float, 
        clipping_ratio: float
    ) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0
        
        # Penalize low signal
        if signal_level < self.min_signal_threshold:
            score *= 0.3
        elif signal_level < 0.1:
            score *= 0.7
        
        # Penalize high noise
        if noise_level > self.max_noise_threshold:
            score *= 0.4
        elif noise_level > 0.1:
            score *= 0.8
        
        # Penalize low SNR
        if snr < self.min_snr_db:
            score *= 0.5
        elif snr < 20:
            score *= 0.8
        
        # Penalize clipping
        if clipping_ratio > 0.01:
            score *= 0.3
        
        return max(score, 0.0)


class RealTimeAudioProcessor:
    """Processes audio chunks in real-time with ASR integration."""
    
    def __init__(self, asr_config: Optional[ASRConfig] = None):
        """Initialize the real-time audio processor.
        
        Args:
            asr_config: Optional ASR configuration
        """
        self.asr_config = asr_config or ASRConfig()
        self.asr_pipeline: Optional[ASRPipeline] = None
        self.quality_monitor = AudioQualityMonitor()
        
        # Processing state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.result_callbacks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.processing_stats = {
            "total_chunks": 0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "avg_processing_time": 0.0
        }
        
        # Initialize ASR pipeline
        self._initialize_asr()
    
    def _initialize_asr(self):
        """Initialize the ASR pipeline."""
        try:
            if HAS_ASR_PIPELINE:
                self.asr_pipeline = create_asr_pipeline(self.asr_config)
                logger.info("ASR pipeline initialized successfully")
            else:
                logger.warning("ASR pipeline not available - using mock implementation")
                self.asr_pipeline = ASRPipeline(self.asr_config)
        except Exception as e:
            logger.error(f"Failed to initialize ASR pipeline: {e}")
            self.asr_pipeline = None
    
    async def start_session(
        self, 
        session_id: str, 
        language: str = "tamil",
        result_callback: Optional[Callable] = None
    ):
        """Start a new audio processing session.
        
        Args:
            session_id: Unique session identifier
            language: Target language for transcription
            result_callback: Optional callback for processing results
        """
        self.active_sessions[session_id] = {
            "language": language,
            "start_time": time.time(),
            "chunk_count": 0,
            "total_audio_duration": 0.0,
            "accumulated_audio": [],
            "last_transcription": "",
            "quality_history": []
        }
        
        if result_callback:
            self.result_callbacks[session_id] = result_callback
        
        logger.info(f"Started audio processing session: {session_id} (language: {language})")
    
    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Stop an audio processing session and return summary.
        
        Args:
            session_id: Session identifier to stop
            
        Returns:
            Session summary with statistics
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to stop non-existent session: {session_id}")
            return {}
        
        session_data = self.active_sessions[session_id]
        session_duration = time.time() - session_data["start_time"]
        
        summary = {
            "session_id": session_id,
            "duration": session_duration,
            "chunk_count": session_data["chunk_count"],
            "total_audio_duration": session_data["total_audio_duration"],
            "final_transcription": session_data["last_transcription"],
            "avg_quality": self._calculate_avg_quality(session_data["quality_history"])
        }
        
        # Clean up
        del self.active_sessions[session_id]
        if session_id in self.result_callbacks:
            del self.result_callbacks[session_id]
        
        logger.info(f"Stopped audio processing session: {session_id}")
        return summary
    
    def _calculate_avg_quality(self, quality_history: List[AudioQuality]) -> Dict[str, float]:
        """Calculate average quality metrics from history."""
        if not quality_history:
            return {"signal_level": 0.0, "noise_level": 0.0, "snr": 0.0, "quality_score": 0.0}
        
        return {
            "signal_level": np.mean([q.signal_level for q in quality_history]),
            "noise_level": np.mean([q.noise_level for q in quality_history]),
            "snr": np.mean([q.snr for q in quality_history]),
            "quality_score": np.mean([q.quality_score for q in quality_history])
        }
    
    async def process_audio_chunk(
        self, 
        session_id: str, 
        audio_data: bytes, 
        chunk_id: str,
        sample_rate: int = 16000
    ) -> ProcessingResult:
        """Process a single audio chunk.
        
        Args:
            session_id: Session identifier
            audio_data: Raw audio bytes
            chunk_id: Unique chunk identifier
            sample_rate: Audio sample rate
            
        Returns:
            ProcessingResult with transcription and quality metrics
        """
        start_time = time.time()
        
        if session_id not in self.active_sessions:
            return ProcessingResult(
                transcription="",
                confidence=0.0,
                audio_quality=AudioQuality(
                    signal_level=0.0, noise_level=0.0, snr=0.0, 
                    quality_score=0.0, warnings=["Session not found"]
                ),
                processing_time=0.0,
                chunk_id=chunk_id,
                session_id=session_id,
                error="Session not found"
            )
        
        try:
            # Convert audio bytes to numpy array
            audio_array = self._bytes_to_audio_array(audio_data, sample_rate)
            if audio_array is None:
                raise ValueError("Failed to decode audio data")
            
            # Analyze audio quality
            quality = self.quality_monitor.analyze_quality(audio_array)
            
            # Update session data
            session_data = self.active_sessions[session_id]
            session_data["chunk_count"] += 1
            session_data["total_audio_duration"] += len(audio_array) / sample_rate
            session_data["quality_history"].append(quality)
            session_data["accumulated_audio"].append(audio_array)
            
            # Perform transcription if ASR is available and quality is acceptable
            transcription = ""
            confidence = 0.0
            
            if self.asr_pipeline and quality.quality_score > 0.3:
                try:
                    # Combine recent audio chunks for better context
                    combined_audio = self._combine_recent_audio(session_data["accumulated_audio"])
                    
                    # Create temporary audio file for ASR
                    temp_audio_path = f"/tmp/audio_{session_id}_{chunk_id}.wav"
                    self._save_audio_array(combined_audio, temp_audio_path, sample_rate)
                    
                    # Transcribe
                    result = self.asr_pipeline.transcribe(
                        temp_audio_path, 
                        session_data["language"]
                    )
                    
                    if result.error is None:
                        transcription = result.text
                        confidence = result.confidence
                        session_data["last_transcription"] = transcription
                        self.processing_stats["successful_transcriptions"] += 1
                    else:
                        logger.warning(f"ASR failed for chunk {chunk_id}: {result.error}")
                        self.processing_stats["failed_transcriptions"] += 1
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                        
                except Exception as e:
                    logger.error(f"Transcription failed for chunk {chunk_id}: {e}")
                    self.processing_stats["failed_transcriptions"] += 1
            
            processing_time = time.time() - start_time
            
            # Update processing stats
            self.processing_stats["total_chunks"] += 1
            self.processing_stats["avg_processing_time"] = (
                (self.processing_stats["avg_processing_time"] * (self.processing_stats["total_chunks"] - 1) + processing_time) /
                self.processing_stats["total_chunks"]
            )
            
            result = ProcessingResult(
                transcription=transcription,
                confidence=confidence,
                audio_quality=quality,
                processing_time=processing_time,
                chunk_id=chunk_id,
                session_id=session_id
            )
            
            # Call result callback if available
            if session_id in self.result_callbacks:
                try:
                    await self.result_callbacks[session_id](result)
                except Exception as e:
                    logger.error(f"Result callback failed for session {session_id}: {e}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Audio processing failed for chunk {chunk_id}: {e}")
            
            return ProcessingResult(
                transcription="",
                confidence=0.0,
                audio_quality=AudioQuality(
                    signal_level=0.0, noise_level=0.0, snr=0.0,
                    quality_score=0.0, warnings=[f"Processing error: {str(e)}"]
                ),
                processing_time=processing_time,
                chunk_id=chunk_id,
                session_id=session_id,
                error=str(e)
            )
    
    def _bytes_to_audio_array(self, audio_data: bytes, sample_rate: int) -> Optional[np.ndarray]:
        """Convert audio bytes to numpy array."""
        try:
            # Try to decode as base64 first (common for WebSocket)
            try:
                decoded_data = base64.b64decode(audio_data)
            except:
                decoded_data = audio_data
            
            # Ensure the buffer size is a multiple of 2 (for 16-bit samples)
            if len(decoded_data) % 2 != 0:
                # Pad with a zero byte if odd length
                decoded_data = decoded_data + b'\x00'
            
            # Convert to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(decoded_data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1]
            audio_array = audio_array / 32768.0
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Failed to convert audio bytes to array: {e}")
            return None
    
    def _combine_recent_audio(self, audio_chunks: List[np.ndarray], max_chunks: int = 5) -> np.ndarray:
        """Combine recent audio chunks for better transcription context."""
        if not audio_chunks:
            return np.array([])
        
        # Use last N chunks
        recent_chunks = audio_chunks[-max_chunks:]
        return np.concatenate(recent_chunks)
    
    def _save_audio_array(self, audio_array: np.ndarray, file_path: str, sample_rate: int):
        """Save audio array to file."""
        try:
            if HAS_SOUNDFILE:
                sf.write(file_path, audio_array, sample_rate)
            else:
                # Fallback: use basic file writing
                # Convert to 16-bit PCM
                audio_int16 = (audio_array * 32767).astype(np.int16)
                with open(file_path, 'wb') as f:
                    f.write(audio_int16.tobytes())
        except Exception as e:
            logger.error(f"Failed to save audio file {file_path}: {e}")
            raise
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an active session."""
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id]
        current_time = time.time()
        
        return {
            "session_id": session_id,
            "language": session_data["language"],
            "duration": current_time - session_data["start_time"],
            "chunk_count": session_data["chunk_count"],
            "total_audio_duration": session_data["total_audio_duration"],
            "last_transcription": session_data["last_transcription"],
            "current_quality": session_data["quality_history"][-1] if session_data["quality_history"] else None
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        return {
            **self.processing_stats,
            "active_sessions": len(self.active_sessions),
            "asr_available": self.asr_pipeline is not None
        }
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is active."""
        return session_id in self.active_sessions


# Global audio processor instance
audio_processor = RealTimeAudioProcessor()


def get_audio_processor() -> RealTimeAudioProcessor:
    """Get the global audio processor instance."""
    return audio_processor


async def initialize_audio_processor(asr_config: Optional[ASRConfig] = None):
    """Initialize the global audio processor."""
    global audio_processor
    if asr_config:
        audio_processor = RealTimeAudioProcessor(asr_config)
    logger.info("Audio processor initialized")


async def shutdown_audio_processor():
    """Shutdown the audio processor and clean up resources."""
    global audio_processor
    
    # Stop all active sessions
    active_sessions = list(audio_processor.active_sessions.keys())
    for session_id in active_sessions:
        await audio_processor.stop_session(session_id)
    
    logger.info("Audio processor shut down")