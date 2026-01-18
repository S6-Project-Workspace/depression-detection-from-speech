"""
Tests for Audio Input Handler

Tests the real-time audio capture, preprocessing, and quality monitoring
functionality of the streaming system.

Reference: Real-time Streaming Analysis - Task 1.1
"""

import asyncio
import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from streaming.audio_input_handler import (
    AudioInputHandler, 
    AudioDevice, 
    AsyncAudioInputHandler
)
from streaming.audio_quality_monitor import AudioQualityMonitor
from streaming.models import AudioQualityMetric


class TestAudioDevice:
    """Test AudioDevice class."""
    
    def test_audio_device_creation(self):
        """Test AudioDevice creation and properties."""
        device = AudioDevice(
            device_id=0,
            name="Test Microphone",
            max_input_channels=2,
            default_sample_rate=44100.0,
            is_default=True
        )
        
        assert device.device_id == 0
        assert device.name == "Test Microphone"
        assert device.max_input_channels == 2
        assert device.default_sample_rate == 44100.0
        assert device.is_default is True
    
    def test_audio_device_to_dict(self):
        """Test AudioDevice serialization."""
        device = AudioDevice(
            device_id=1,
            name="USB Microphone",
            max_input_channels=1,
            default_sample_rate=16000.0
        )
        
        device_dict = device.to_dict()
        
        assert device_dict['device_id'] == 1
        assert device_dict['name'] == "USB Microphone"
        assert device_dict['max_input_channels'] == 1
        assert device_dict['default_sample_rate'] == 16000.0
        assert device_dict['is_default'] is False
    
    def test_audio_device_str(self):
        """Test AudioDevice string representation."""
        device = AudioDevice(
            device_id=2,
            name="Built-in Microphone",
            max_input_channels=1,
            default_sample_rate=48000.0
        )
        
        device_str = str(device)
        assert "AudioDevice" in device_str
        assert "id=2" in device_str
        assert "Built-in Microphone" in device_str


class TestAudioQualityMonitor:
    """Test AudioQualityMonitor class."""
    
    def test_quality_monitor_initialization(self):
        """Test quality monitor initialization."""
        monitor = AudioQualityMonitor(
            sample_rate=16000,
            noise_floor_db=-50.0,
            clipping_threshold=0.9
        )
        
        assert monitor.sample_rate == 16000
        assert monitor.noise_floor_db == -50.0
        assert monitor.clipping_threshold == 0.9
        assert len(monitor.quality_history) == 0
    
    def test_assess_quality_empty_audio(self):
        """Test quality assessment with empty audio."""
        monitor = AudioQualityMonitor()
        
        empty_audio = np.array([])
        metric = monitor.assess_quality(empty_audio)
        
        assert isinstance(metric, AudioQualityMetric)
        assert metric.quality_score == 0.0
        assert metric.silence_ratio == 1.0
        assert not metric.clipping_detected
    
    def test_assess_quality_normal_audio(self):
        """Test quality assessment with normal audio."""
        monitor = AudioQualityMonitor()
        
        # Generate test audio (sine wave with some noise)
        duration = 1.0  # 1 second
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 440 Hz sine wave with amplitude 0.3
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        # Add small amount of noise
        audio += 0.01 * np.random.randn(len(audio))
        
        metric = monitor.assess_quality(audio)
        
        assert isinstance(metric, AudioQualityMetric)
        assert 0.0 <= metric.quality_score <= 1.0
        assert metric.signal_to_noise_ratio >= 0.0
        assert 0.0 <= metric.volume_level <= 1.0
        assert 0.0 <= metric.silence_ratio <= 1.0
    
    def test_clipping_detection(self):
        """Test clipping detection."""
        monitor = AudioQualityMonitor(clipping_threshold=0.9)
        
        # Create audio with clipping
        clipped_audio = np.array([0.95, 0.96, 0.97, 0.98, 0.99] * 100)
        
        metric = monitor.assess_quality(clipped_audio)
        
        assert metric.clipping_detected is True
        assert metric.quality_score < 0.5  # Should be penalized
    
    def test_silence_detection(self):
        """Test silence detection."""
        monitor = AudioQualityMonitor(silence_threshold=0.001)
        
        # Create mostly silent audio
        silent_audio = np.zeros(1000)
        silent_audio[100:200] = 0.1  # Small active region
        
        metric = monitor.assess_quality(silent_audio)
        
        assert metric.silence_ratio > 0.8  # Mostly silent
    
    def test_quality_history(self):
        """Test quality history tracking."""
        monitor = AudioQualityMonitor()
        
        # Assess multiple chunks
        for i in range(5):
            audio = np.random.randn(1000) * 0.1
            monitor.assess_quality(audio)
        
        assert len(monitor.quality_history) == 5
        
        # Test average quality
        avg_quality = monitor.get_average_quality(minutes=10)
        assert 0.0 <= avg_quality <= 1.0


@pytest.mark.skipif(
    not hasattr(pytest, 'pyaudio_available') or not pytest.pyaudio_available,
    reason="PyAudio not available"
)
class TestAudioInputHandler:
    """Test AudioInputHandler class (requires PyAudio)."""
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_handler_initialization(self, mock_pyaudio):
        """Test handler initialization."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        handler = AudioInputHandler(
            sample_rate=16000,
            chunk_duration_ms=1000,
            channels=1
        )
        
        assert handler.sample_rate == 16000
        assert handler.chunk_duration_ms == 1000
        assert handler.channels == 1
        assert handler.chunk_size == 16000  # 1 second at 16kHz
        assert not handler.is_capturing
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_device_enumeration(self, mock_pyaudio):
        """Test audio device enumeration."""
        # Mock PyAudio device info
        mock_pa = Mock()
        mock_pyaudio.PyAudio.return_value = mock_pa
        
        mock_pa.get_device_count.return_value = 2
        mock_pa.get_default_input_device_info.return_value = {'index': 0}
        
        mock_pa.get_device_info_by_index.side_effect = [
            {
                'name': 'Built-in Microphone',
                'maxInputChannels': 2,
                'defaultSampleRate': 44100.0
            },
            {
                'name': 'USB Microphone',
                'maxInputChannels': 1,
                'defaultSampleRate': 16000.0
            }
        ]
        
        handler = AudioInputHandler()
        devices = handler.get_available_devices()
        
        assert len(devices) == 2
        assert devices[0].name == 'Built-in Microphone'
        assert devices[0].is_default is True
        assert devices[1].name == 'USB Microphone'
        assert devices[1].is_default is False
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_device_selection(self, mock_pyaudio):
        """Test audio device selection."""
        mock_pa = Mock()
        mock_pyaudio.PyAudio.return_value = mock_pa
        
        mock_pa.get_device_info_by_index.return_value = {
            'name': 'Test Device',
            'maxInputChannels': 1,
            'defaultSampleRate': 16000.0
        }
        
        handler = AudioInputHandler()
        success = handler.select_device(device_id=0)
        
        assert success is True
        assert handler.current_device is not None
        assert handler.current_device.name == 'Test Device'
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_callback_management(self, mock_pyaudio):
        """Test audio callback management."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        handler = AudioInputHandler()
        
        # Test callback addition
        callback1 = Mock()
        callback2 = Mock()
        
        handler.add_callback(callback1)
        handler.add_callback(callback2)
        
        assert len(handler.callbacks) == 2
        assert callback1 in handler.callbacks
        assert callback2 in handler.callbacks
        
        # Test callback removal
        handler.remove_callback(callback1)
        
        assert len(handler.callbacks) == 1
        assert callback1 not in handler.callbacks
        assert callback2 in handler.callbacks
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_auto_gain_control(self, mock_pyaudio):
        """Test automatic gain control."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        handler = AudioInputHandler(enable_auto_gain=True)
        
        # Test with low amplitude audio
        low_audio = np.random.randn(1000) * 0.01  # Very quiet
        processed = handler._apply_auto_gain(low_audio)
        
        # Should be amplified
        assert np.mean(np.abs(processed)) > np.mean(np.abs(low_audio))
        
        # Test with high amplitude audio
        high_audio = np.random.randn(1000) * 0.8  # Loud
        processed = handler._apply_auto_gain(high_audio)
        
        # Should be attenuated or clipped
        assert np.max(np.abs(processed)) <= 1.0
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_noise_reduction(self, mock_pyaudio):
        """Test noise reduction."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        handler = AudioInputHandler(enable_noise_reduction=True)
        
        # Create noisy audio
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))  # 440 Hz tone
        noise = np.random.randn(16000) * 0.1
        noisy_audio = signal + noise
        
        processed = handler._apply_noise_reduction(noisy_audio)
        
        # Should be same length
        assert len(processed) == len(noisy_audio)
        
        # Should be within valid range
        assert np.max(np.abs(processed)) <= 1.0
    
    @patch('streaming.audio_input_handler.pyaudio')
    @patch('streaming.audio_input_handler.webrtcvad')
    def test_voice_activity_detection(self, mock_vad, mock_pyaudio):
        """Test voice activity detection."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        # Mock WebRTC VAD
        mock_vad_instance = Mock()
        mock_vad.Vad.return_value = mock_vad_instance
        mock_vad_instance.is_speech.return_value = True
        
        handler = AudioInputHandler()
        
        # Test with speech-like audio
        speech_audio = np.random.randn(16000) * 0.1
        has_speech = handler._detect_voice_activity(speech_audio)
        
        assert isinstance(has_speech, bool)
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_statistics(self, mock_pyaudio):
        """Test statistics collection."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        handler = AudioInputHandler()
        stats = handler.get_statistics()
        
        assert 'is_capturing' in stats
        assert 'total_chunks_captured' in stats
        assert 'total_dropouts' in stats
        assert 'sample_rate' in stats
        assert 'chunk_size' in stats
        
        assert stats['is_capturing'] is False
        assert stats['total_chunks_captured'] == 0
        assert stats['sample_rate'] == 16000


class TestAsyncAudioInputHandler:
    """Test AsyncAudioInputHandler class."""
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_async_handler_initialization(self, mock_pyaudio):
        """Test async handler initialization."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        async_handler = AsyncAudioInputHandler(sample_rate=16000)
        
        assert async_handler.handler is not None
        assert async_handler.handler.sample_rate == 16000
        assert hasattr(async_handler, '_audio_queue')
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_async_device_methods(self, mock_pyaudio):
        """Test async device methods."""
        mock_pa = Mock()
        mock_pyaudio.PyAudio.return_value = mock_pa
        
        mock_pa.get_device_count.return_value = 1
        mock_pa.get_default_input_device_info.return_value = {'index': 0}
        mock_pa.get_device_info_by_index.return_value = {
            'name': 'Test Device',
            'maxInputChannels': 1,
            'defaultSampleRate': 16000.0
        }
        
        async_handler = AsyncAudioInputHandler()
        
        # Test device enumeration
        devices = async_handler.get_available_devices()
        assert len(devices) == 1
        
        # Test statistics
        stats = async_handler.get_statistics()
        assert 'is_capturing' in stats


# Property-based tests using hypothesis
try:
    from hypothesis import given, strategies as st, settings
    
    class TestAudioInputHandlerProperties:
        """Property-based tests for audio input handler."""
        
        @given(
            sample_rate=st.integers(min_value=8000, max_value=48000),
            chunk_duration=st.integers(min_value=100, max_value=2000)
        )
        @settings(deadline=None, max_examples=10)
        @patch('streaming.audio_input_handler.pyaudio')
        def test_chunk_size_calculation(self, mock_pyaudio, sample_rate, chunk_duration):
            """Test chunk size calculation properties."""
            mock_pyaudio.PyAudio.return_value = Mock()
            
            handler = AudioInputHandler(
                sample_rate=sample_rate,
                chunk_duration_ms=chunk_duration
            )
            
            expected_chunk_size = int(sample_rate * chunk_duration / 1000)
            
            # Properties
            assert handler.chunk_size == expected_chunk_size
            assert handler.chunk_size > 0
            assert handler.sample_rate == sample_rate
        
        @given(
            audio_data=st.lists(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                min_size=100, max_size=16000
            )
        )
        @settings(deadline=None, max_examples=10)
        @patch('streaming.audio_input_handler.pyaudio')
        def test_auto_gain_properties(self, mock_pyaudio, audio_data):
            """Test auto gain control properties."""
            mock_pyaudio.PyAudio.return_value = Mock()
            
            handler = AudioInputHandler(enable_auto_gain=True)
            audio_array = np.array(audio_data, dtype=np.float32)
            
            processed = handler._apply_auto_gain(audio_array)
            
            # Properties
            assert len(processed) == len(audio_array)
            assert np.all(np.abs(processed) <= 1.0)  # No clipping
            assert np.all(np.isfinite(processed))  # No NaN or inf
        
        @given(
            audio_data=st.lists(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                min_size=100, max_size=16000
            )
        )
        @settings(deadline=None, max_examples=10)
        def test_quality_assessment_properties(self, audio_data):
            """Test quality assessment properties."""
            monitor = AudioQualityMonitor()
            audio_array = np.array(audio_data, dtype=np.float32)
            
            metric = monitor.assess_quality(audio_array)
            
            # Properties
            assert isinstance(metric, AudioQualityMetric)
            assert 0.0 <= metric.quality_score <= 1.0
            assert metric.signal_to_noise_ratio >= 0.0
            assert 0.0 <= metric.volume_level <= 1.0
            assert 0.0 <= metric.silence_ratio <= 1.0
            assert isinstance(metric.clipping_detected, bool)

except ImportError:
    # Hypothesis not available
    pass


# Integration tests
class TestAudioInputIntegration:
    """Integration tests for audio input components."""
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_handler_quality_monitor_integration(self, mock_pyaudio):
        """Test integration between handler and quality monitor."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        handler = AudioInputHandler()
        
        # Test that quality monitor is properly initialized
        assert handler.quality_monitor is not None
        assert isinstance(handler.quality_monitor, AudioQualityMonitor)
        
        # Test quality assessment integration
        test_audio = np.random.randn(1000) * 0.1
        
        # This should not raise an exception
        quality_metric = handler.quality_monitor.assess_quality(test_audio)
        assert isinstance(quality_metric, AudioQualityMetric)
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_callback_integration(self, mock_pyaudio):
        """Test callback integration with quality monitoring."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        handler = AudioInputHandler()
        
        # Add test callback
        received_chunks = []
        received_metrics = []
        
        def test_callback(audio_data, quality_metric):
            received_chunks.append(audio_data)
            received_metrics.append(quality_metric)
        
        handler.add_callback(test_callback)
        
        # Simulate audio processing
        test_audio = np.random.randn(1000) * 0.1
        handler._process_audio_chunk(test_audio, time.time())
        
        # Verify callback was called
        assert len(received_chunks) == 1
        assert len(received_metrics) == 1
        assert isinstance(received_metrics[0], AudioQualityMetric)


if __name__ == "__main__":
    # Check if PyAudio is available for testing
    try:
        import pyaudio
        pytest.pyaudio_available = True
    except ImportError:
        pytest.pyaudio_available = False
        print("Warning: PyAudio not available - some tests will be skipped")
    
    pytest.main([__file__])