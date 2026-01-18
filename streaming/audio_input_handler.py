"""
Audio Input Handler for Real-time Streaming

This module handles real-time audio capture, preprocessing, and quality monitoring
for the streaming depression detection system.

Key Features:
- Real-time audio capture at 16kHz
- Multiple audio device support
- Noise reduction and audio normalization
- Audio quality assessment
- Voice activity detection

Reference: Real-time Streaming Analysis - Task 1.1
"""

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Callable, Any
import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    webrtcvad = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

from .models import AudioQualityMetric, SessionStatus
from .audio_quality_monitor import AudioQualityMonitor

logger = logging.getLogger(__name__)


class AudioDevice:
    """Audio device information and configuration."""
    
    def __init__(
        self,
        device_id: int,
        name: str,
        max_input_channels: int,
        default_sample_rate: float,
        is_default: bool = False
    ):
        self.device_id = device_id
        self.name = name
        self.max_input_channels = max_input_channels
        self.default_sample_rate = default_sample_rate
        self.is_default = is_default
    
    def __str__(self) -> str:
        return f"AudioDevice(id={self.device_id}, name='{self.name}', channels={self.max_input_channels})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'device_id': self.device_id,
            'name': self.name,
            'max_input_channels': self.max_input_channels,
            'default_sample_rate': self.default_sample_rate,
            'is_default': self.is_default
        }


class AudioInputHandler:
    """
    Real-time audio input handler with quality monitoring and preprocessing.
    
    Handles:
    - Audio device enumeration and selection
    - Real-time audio capture at 16kHz
    - Noise reduction and normalization
    - Audio quality assessment
    - Voice activity detection
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 1000,
        channels: int = 1,
        enable_noise_reduction: bool = True,
        enable_auto_gain: bool = True,
        min_audio_quality: float = 0.6
    ):
        """
        Initialize audio input handler.
        
        Args:
            sample_rate: Audio sampling rate (default: 16kHz)
            chunk_duration_ms: Audio chunk duration in milliseconds
            channels: Number of audio channels (1 for mono)
            enable_noise_reduction: Enable noise reduction
            enable_auto_gain: Enable automatic gain control
            min_audio_quality: Minimum acceptable audio quality (0.0-1.0)
        """
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is required for audio input handling")
        
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.channels = channels
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_auto_gain = enable_auto_gain
        self.min_audio_quality = min_audio_quality
        
        # Calculate chunk size
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        # PyAudio setup
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.current_device: Optional[AudioDevice] = None
        
        # Audio processing
        self.quality_monitor = AudioQualityMonitor(sample_rate)
        
        # Voice activity detection
        if WEBRTCVAD_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        else:
            self.vad = None
            logger.warning("WebRTC VAD not available - using energy-based VAD")
        
        # Threading and state
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None
        self.audio_queue = deque(maxlen=100)  # Buffer for audio chunks
        self.callbacks: List[Callable[[np.ndarray, AudioQualityMetric], None]] = []
        
        # Statistics
        self.total_chunks_captured = 0
        self.total_dropouts = 0
        self.last_capture_time = 0.0
        
        # Auto gain control
        self.gain_factor = 1.0
        self.target_rms = 0.1
        self.gain_adaptation_rate = 0.01
        
        logger.info(f"AudioInputHandler initialized: {sample_rate}Hz, {chunk_duration_ms}ms chunks")
    
    def get_available_devices(self) -> List[AudioDevice]:
        """Get list of available audio input devices."""
        devices = []
        
        try:
            device_count = self.pyaudio_instance.get_device_count()
            default_device_id = self.pyaudio_instance.get_default_input_device_info()['index']
            
            for i in range(device_count):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    
                    # Only include input devices
                    if device_info['maxInputChannels'] > 0:
                        device = AudioDevice(
                            device_id=i,
                            name=device_info['name'],
                            max_input_channels=device_info['maxInputChannels'],
                            default_sample_rate=device_info['defaultSampleRate'],
                            is_default=(i == default_device_id)
                        )
                        devices.append(device)
                        
                except Exception as e:
                    logger.warning(f"Could not get info for device {i}: {e}")
                    
        except Exception as e:
            logger.error(f"Error enumerating audio devices: {e}")
        
        logger.info(f"Found {len(devices)} audio input devices")
        return devices
    
    def select_device(self, device_id: Optional[int] = None) -> bool:
        """
        Select audio input device.
        
        Args:
            device_id: Device ID to select (None for default)
            
        Returns:
            True if device selected successfully
        """
        try:
            if device_id is None:
                # Use default device
                device_info = self.pyaudio_instance.get_default_input_device_info()
                device_id = device_info['index']
            else:
                device_info = self.pyaudio_instance.get_device_info_by_index(device_id)
            
            # Validate device supports required format
            if device_info['maxInputChannels'] < self.channels:
                logger.error(f"Device {device_id} doesn't support {self.channels} channels")
                return False
            
            self.current_device = AudioDevice(
                device_id=device_id,
                name=device_info['name'],
                max_input_channels=device_info['maxInputChannels'],
                default_sample_rate=device_info['defaultSampleRate'],
                is_default=True
            )
            
            logger.info(f"Selected audio device: {self.current_device}")
            return True
            
        except Exception as e:
            logger.error(f"Error selecting audio device {device_id}: {e}")
            return False
    
    def add_callback(self, callback: Callable[[np.ndarray, AudioQualityMetric], None]):
        """Add callback for audio chunks."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[np.ndarray, AudioQualityMetric], None]):
        """Remove callback for audio chunks."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio data."""
        if status:
            logger.warning(f"Audio callback status: {status}")
            self.total_dropouts += 1
        
        # Convert to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Store in queue for processing
        self.audio_queue.append((audio_data.copy(), time.time()))
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_chunk(self, audio_data: np.ndarray, timestamp: float):
        """Process a single audio chunk."""
        try:
            # Apply automatic gain control
            if self.enable_auto_gain:
                audio_data = self._apply_auto_gain(audio_data)
            
            # Apply noise reduction
            if self.enable_noise_reduction:
                audio_data = self._apply_noise_reduction(audio_data)
            
            # Assess audio quality
            quality_metric = self.quality_monitor.assess_quality(audio_data, timestamp)
            
            # Voice activity detection
            has_speech = self._detect_voice_activity(audio_data)
            quality_metric.metadata = {'has_speech': has_speech}
            
            # Update statistics
            self.total_chunks_captured += 1
            self.last_capture_time = timestamp
            
            # Call registered callbacks
            for callback in self.callbacks:
                try:
                    callback(audio_data, quality_metric)
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def _apply_auto_gain(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply automatic gain control."""
        if len(audio_data) == 0:
            return audio_data
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        if rms > 0:
            # Adapt gain factor
            error = self.target_rms - rms
            self.gain_factor += error * self.gain_adaptation_rate
            
            # Limit gain factor
            self.gain_factor = np.clip(self.gain_factor, 0.1, 10.0)
            
            # Apply gain
            audio_data = audio_data * self.gain_factor
            
            # Prevent clipping
            audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data
    
    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction."""
        if not LIBROSA_AVAILABLE or len(audio_data) == 0:
            return audio_data
        
        try:
            # Simple spectral subtraction-based noise reduction
            # This is a basic implementation - more sophisticated methods could be used
            
            # Estimate noise floor from first 10% of signal
            noise_samples = max(1, len(audio_data) // 10)
            noise_level = np.std(audio_data[:noise_samples])
            
            # Apply soft thresholding
            threshold = noise_level * 2.0
            mask = np.abs(audio_data) > threshold
            
            # Preserve signal above threshold, attenuate below
            processed = np.where(mask, audio_data, audio_data * 0.1)
            
            return processed
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Detect voice activity in audio chunk."""
        if self.vad and len(audio_data) == self.chunk_size:
            try:
                # Convert to 16-bit PCM for WebRTC VAD
                audio_16bit = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_16bit.tobytes()
                
                # WebRTC VAD requires specific frame sizes
                if len(audio_bytes) == self.chunk_size * 2:  # 16-bit = 2 bytes per sample
                    return self.vad.is_speech(audio_bytes, self.sample_rate)
                    
            except Exception as e:
                logger.warning(f"WebRTC VAD failed: {e}")
        
        # Fallback: energy-based VAD
        if len(audio_data) > 0:
            energy = np.mean(audio_data ** 2)
            return energy > 0.001  # Simple energy threshold
        
        return False
    
    def _capture_loop(self):
        """Main audio capture loop."""
        logger.info("Audio capture loop started")
        
        while self.is_capturing:
            try:
                # Process queued audio chunks
                while self.audio_queue and self.is_capturing:
                    audio_data, timestamp = self.audio_queue.popleft()
                    self._process_audio_chunk(audio_data, timestamp)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
        
        logger.info("Audio capture loop stopped")
    
    def start_capture(self, device_id: Optional[int] = None) -> bool:
        """
        Start audio capture.
        
        Args:
            device_id: Audio device ID (None for default)
            
        Returns:
            True if capture started successfully
        """
        if self.is_capturing:
            logger.warning("Audio capture already running")
            return True
        
        try:
            # Select device
            if not self.select_device(device_id):
                return False
            
            # Open audio stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.current_device.device_id,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start capture thread
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Start audio stream
            self.stream.start_stream()
            
            logger.info(f"Audio capture started on device: {self.current_device.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.stop_capture()
            return False
    
    def stop_capture(self):
        """Stop audio capture."""
        if not self.is_capturing:
            return
        
        logger.info("Stopping audio capture...")
        
        # Stop capture
        self.is_capturing = False
        
        # Stop and close stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
        
        # Wait for capture thread
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Clear queue
        self.audio_queue.clear()
        
        logger.info("Audio capture stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get capture statistics."""
        return {
            'is_capturing': self.is_capturing,
            'current_device': self.current_device.to_dict() if self.current_device else None,
            'total_chunks_captured': self.total_chunks_captured,
            'total_dropouts': self.total_dropouts,
            'queue_size': len(self.audio_queue),
            'last_capture_time': self.last_capture_time,
            'gain_factor': self.gain_factor,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop_capture()
            if hasattr(self, 'pyaudio_instance') and self.pyaudio_instance:
                self.pyaudio_instance.terminate()
        except Exception:
            pass


# Async wrapper for audio input handler
class AsyncAudioInputHandler:
    """Async wrapper for AudioInputHandler."""
    
    def __init__(self, *args, **kwargs):
        self.handler = AudioInputHandler(*args, **kwargs)
        self._audio_queue = asyncio.Queue()
        
    async def start_capture(self, device_id: Optional[int] = None) -> bool:
        """Start audio capture asynchronously."""
        # Add callback to forward audio to async queue
        def audio_callback(audio_data: np.ndarray, quality_metric: AudioQualityMetric):
            try:
                self._audio_queue.put_nowait((audio_data, quality_metric))
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping chunk")
        
        self.handler.add_callback(audio_callback)
        return self.handler.start_capture(device_id)
    
    async def stop_capture(self):
        """Stop audio capture asynchronously."""
        self.handler.stop_capture()
    
    async def get_audio_chunk(self) -> tuple[np.ndarray, AudioQualityMetric]:
        """Get next audio chunk asynchronously."""
        return await self._audio_queue.get()
    
    def get_available_devices(self) -> List[AudioDevice]:
        """Get available audio devices."""
        return self.handler.get_available_devices()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get capture statistics."""
        return self.handler.get_statistics()