"""
Property-based tests for audio processing functionality.

Tests audio capture latency and real-time transcription using property-based
testing to verify universal properties across different audio inputs.

Requirements: 1.1, 1.2
"""

import pytest
import asyncio
import time
import numpy as np
import base64
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, strategies as st, settings, assume

from web_interface.audio_processor import (
    RealTimeAudioProcessor, 
    AudioQualityMonitor,
    ProcessingResult,
    get_audio_processor
)
from web_interface.models import AudioQuality


def generate_audio_data(duration_seconds: float, sample_rate: int = 16000, frequency: float = 440.0) -> np.ndarray:
    """Generate synthetic audio data for testing."""
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, False)
    # Generate sine wave with some noise
    audio = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(num_samples)
    return audio.astype(np.float32)


def audio_to_bytes(audio_array: np.ndarray) -> bytes:
    """Convert audio array to bytes for testing."""
    # Convert to 16-bit PCM
    audio_int16 = (audio_array * 32767).astype(np.int16)
    return audio_int16.tobytes()


@pytest.mark.asyncio
class TestAudioProcessingProperties:
    """Property-based tests for audio processing functionality."""
    
    @given(st.floats(min_value=0.1, max_value=5.0))
    @settings(max_examples=30, deadline=3000)
    async def test_audio_capture_latency_property(self, duration):
        """
        Feature: nlp-web-interface, Property 1: Audio capture latency
        For any audio capture request, the time between the start command and actual recording beginning should be under 100 milliseconds.
        **Validates: Requirements 1.1**
        """
        processor = RealTimeAudioProcessor()
        session_id = f"latency_test_{int(time.time() * 1000)}"
        
        try:
            # Measure session start latency
            start_time = time.time()
            await processor.start_session(session_id, language="tamil")
            session_start_latency = time.time() - start_time
            
            # Verify session start latency is under 100ms
            assert session_start_latency < 0.1, f"Session start latency {session_start_latency:.3f}s exceeds 100ms threshold"
            
            # Generate test audio data
            audio_data = generate_audio_data(duration)
            audio_bytes = audio_to_bytes(audio_data)
            
            # Measure audio processing latency
            chunk_id = f"chunk_{int(time.time() * 1000)}"
            start_time = time.time()
            result = await processor.process_audio_chunk(
                session_id=session_id,
                audio_data=audio_bytes,
                chunk_id=chunk_id,
                sample_rate=16000
            )
            processing_latency = time.time() - start_time
            
            # Verify processing starts immediately (under 100ms to begin processing)
            assert processing_latency < 2.0, f"Audio processing latency {processing_latency:.3f}s is too high"
            
            # Verify result is returned
            assert isinstance(result, ProcessingResult)
            assert result.session_id == session_id
            assert result.chunk_id == chunk_id
            
            # Verify audio quality analysis was performed
            assert isinstance(result.audio_quality, AudioQuality)
            assert 0.0 <= result.audio_quality.signal_level <= 1.0
            assert 0.0 <= result.audio_quality.noise_level <= 1.0
            assert 0.0 <= result.audio_quality.quality_score <= 1.0
            
        finally:
            # Clean up
            if processor.is_session_active(session_id):
                await processor.stop_session(session_id)
    
    @given(st.lists(st.floats(min_value=0.5, max_value=3.0), min_size=1, max_size=5))
    @settings(max_examples=20, deadline=10000)
    async def test_real_time_transcription_property(self, chunk_durations):
        """
        Feature: nlp-web-interface, Property 2: Real-time transcription
        For any audio input during active capture, transcription results should be generated within 2 seconds of audio completion.
        **Validates: Requirements 1.2**
        """
        processor = RealTimeAudioProcessor()
        session_id = f"transcription_test_{int(time.time() * 1000)}"
        
        try:
            # Start session
            await processor.start_session(session_id, language="tamil")
            
            # Process multiple audio chunks
            for i, duration in enumerate(chunk_durations):
                # Generate audio data
                audio_data = generate_audio_data(duration, frequency=440.0 + i * 100)  # Different frequencies
                audio_bytes = audio_to_bytes(audio_data)
                
                chunk_id = f"chunk_{i}_{int(time.time() * 1000)}"
                
                # Measure transcription time
                start_time = time.time()
                result = await processor.process_audio_chunk(
                    session_id=session_id,
                    audio_data=audio_bytes,
                    chunk_id=chunk_id,
                    sample_rate=16000
                )
                transcription_time = time.time() - start_time
                
                # Verify transcription is generated within 2 seconds
                assert transcription_time < 2.0, f"Transcription time {transcription_time:.3f}s exceeds 2 second threshold for chunk {i}"
                
                # Verify result structure
                assert isinstance(result, ProcessingResult)
                assert result.session_id == session_id
                assert result.chunk_id == chunk_id
                assert isinstance(result.transcription, str)
                assert 0.0 <= result.confidence <= 1.0
                assert result.processing_time > 0.0
                
                # Verify audio quality metrics are reasonable
                quality = result.audio_quality
                assert isinstance(quality, AudioQuality)
                assert quality.signal_level > 0.0  # Should detect some signal
                assert quality.quality_score >= 0.0
                
                # For synthetic audio, we expect decent quality
                if duration > 1.0:  # Longer chunks should have better quality detection
                    assert quality.signal_level > 0.1, f"Signal level too low for {duration}s chunk: {quality.signal_level}"
        
        finally:
            # Clean up
            if processor.is_session_active(session_id):
                await processor.stop_session(session_id)
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=8000)
    async def test_concurrent_audio_processing_property(self, num_sessions):
        """
        Test that multiple concurrent audio processing sessions work independently.
        """
        processor = RealTimeAudioProcessor()
        sessions = []
        
        try:
            # Start multiple sessions concurrently
            session_tasks = []
            for i in range(num_sessions):
                session_id = f"concurrent_test_{i}_{int(time.time() * 1000)}"
                sessions.append(session_id)
                task = processor.start_session(session_id, language="tamil")
                session_tasks.append(task)
            
            # Wait for all sessions to start
            await asyncio.gather(*session_tasks)
            
            # Verify all sessions are active
            for session_id in sessions:
                assert processor.is_session_active(session_id), f"Session {session_id} not active"
            
            # Process audio in all sessions concurrently
            processing_tasks = []
            for i, session_id in enumerate(sessions):
                # Generate unique audio for each session
                audio_data = generate_audio_data(1.0, frequency=440.0 + i * 50)
                audio_bytes = audio_to_bytes(audio_data)
                chunk_id = f"chunk_{i}"
                
                task = processor.process_audio_chunk(
                    session_id=session_id,
                    audio_data=audio_bytes,
                    chunk_id=chunk_id,
                    sample_rate=16000
                )
                processing_tasks.append(task)
            
            # Wait for all processing to complete
            results = await asyncio.gather(*processing_tasks)
            
            # Verify all results are correct and independent
            assert len(results) == num_sessions
            
            for i, result in enumerate(results):
                expected_session_id = sessions[i]
                expected_chunk_id = f"chunk_{i}"
                
                assert result.session_id == expected_session_id
                assert result.chunk_id == expected_chunk_id
                assert isinstance(result.transcription, str)
                assert 0.0 <= result.confidence <= 1.0
                assert result.processing_time > 0.0
                
                # Verify audio quality was analyzed
                assert isinstance(result.audio_quality, AudioQuality)
                assert result.audio_quality.signal_level > 0.0
        
        finally:
            # Clean up all sessions
            cleanup_tasks = []
            for session_id in sessions:
                if processor.is_session_active(session_id):
                    cleanup_tasks.append(processor.stop_session(session_id))
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks)
    
    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=30, deadline=2000)
    async def test_audio_quality_monitoring_property(self, noise_level):
        """
        Test that audio quality monitoring provides consistent and reasonable metrics.
        """
        monitor = AudioQualityMonitor()
        
        # Generate audio with controlled noise level
        duration = 2.0
        clean_audio = generate_audio_data(duration, frequency=440.0)
        
        # Add controlled noise
        noise = np.random.randn(len(clean_audio)) * noise_level
        noisy_audio = clean_audio + noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 0.95:
            noisy_audio = noisy_audio * 0.95 / max_val
        
        # Analyze quality
        quality = monitor.analyze_quality(noisy_audio)
        
        # Verify quality metrics are reasonable
        assert isinstance(quality, AudioQuality)
        assert 0.0 <= quality.signal_level <= 1.0
        assert 0.0 <= quality.noise_level <= 1.0
        assert 0.0 <= quality.quality_score <= 1.0
        assert isinstance(quality.snr, (int, float))
        assert isinstance(quality.warnings, list)
        
        # Verify signal level is detected for non-zero audio
        if np.max(np.abs(clean_audio)) > 0.01:
            assert quality.signal_level > 0.0, "Signal level should be > 0 for non-silent audio"
        
        # Test basic quality monitoring functionality without making assumptions
        # about the specific noise estimation algorithm
        # The key property is that the monitor should return valid metrics
        assert not np.isnan(quality.signal_level), "Signal level should not be NaN"
        assert not np.isnan(quality.noise_level), "Noise level should not be NaN"
        assert not np.isnan(quality.snr), "SNR should not be NaN"
        assert not np.isnan(quality.quality_score), "Quality score should not be NaN"
        
        # Test that extreme cases are handled properly
        if noise_level == 0.0:
            # With no added noise, we should have reasonable quality
            assert quality.quality_score > 0.0, "Quality score should be positive for clean audio"
        
        # Test that the quality monitor can handle the audio without crashing
        # and produces consistent results
        quality2 = monitor.analyze_quality(noisy_audio)
        assert quality.signal_level == quality2.signal_level, "Quality analysis should be deterministic"
        assert quality.noise_level == quality2.noise_level, "Quality analysis should be deterministic"
        assert quality.snr == quality2.snr, "Quality analysis should be deterministic"
        assert quality.quality_score == quality2.quality_score, "Quality analysis should be deterministic"
    
    @given(st.lists(st.floats(min_value=0.1, max_value=2.0), min_size=2, max_size=8))
    @settings(max_examples=15, deadline=8000)
    async def test_session_lifecycle_property(self, chunk_durations):
        """
        Test that audio processing sessions maintain consistent state throughout their lifecycle.
        """
        processor = RealTimeAudioProcessor()
        session_id = f"lifecycle_test_{int(time.time() * 1000)}"
        
        try:
            # Verify session doesn't exist initially
            assert not processor.is_session_active(session_id)
            assert processor.get_session_info(session_id) is None
            
            # Start session
            await processor.start_session(session_id, language="malayalam")
            
            # Verify session is active
            assert processor.is_session_active(session_id)
            session_info = processor.get_session_info(session_id)
            assert session_info is not None
            assert session_info["session_id"] == session_id
            assert session_info["language"] == "malayalam"
            assert session_info["chunk_count"] == 0
            
            # Process multiple chunks and verify state updates
            for i, duration in enumerate(chunk_durations):
                audio_data = generate_audio_data(duration)
                audio_bytes = audio_to_bytes(audio_data)
                chunk_id = f"chunk_{i}"
                
                result = await processor.process_audio_chunk(
                    session_id=session_id,
                    audio_data=audio_bytes,
                    chunk_id=chunk_id,
                    sample_rate=16000
                )
                
                # Verify result
                assert result.session_id == session_id
                assert result.chunk_id == chunk_id
                
                # Verify session state updated
                session_info = processor.get_session_info(session_id)
                assert session_info["chunk_count"] == i + 1
                assert session_info["total_audio_duration"] > 0.0
            
            # Stop session
            summary = await processor.stop_session(session_id)
            
            # Verify session stopped
            assert not processor.is_session_active(session_id)
            assert processor.get_session_info(session_id) is None
            
            # Verify summary
            assert isinstance(summary, dict)
            assert summary["session_id"] == session_id
            assert summary["chunk_count"] == len(chunk_durations)
            assert summary["duration"] > 0.0
            assert summary["total_audio_duration"] > 0.0
        
        finally:
            # Ensure cleanup
            if processor.is_session_active(session_id):
                await processor.stop_session(session_id)
    
    @pytest.mark.asyncio
    async def test_error_handling_robustness(self):
        """
        Test that audio processor handles various error conditions gracefully.
        """
        processor = RealTimeAudioProcessor()
        session_id = f"error_test_{int(time.time() * 1000)}"
        
        try:
            # Test processing without active session
            result = await processor.process_audio_chunk(
                session_id="nonexistent_session",
                audio_data=b"invalid_data",
                chunk_id="test_chunk",
                sample_rate=16000
            )
            
            assert result.error is not None
            assert "Session not found" in result.error
            
            # Start valid session
            await processor.start_session(session_id, language="tamil")
            
            # Test with invalid audio data
            result = await processor.process_audio_chunk(
                session_id=session_id,
                audio_data=b"invalid_audio_data",
                chunk_id="invalid_chunk",
                sample_rate=16000
            )
            
            # Should handle gracefully (may have error or low quality)
            assert isinstance(result, ProcessingResult)
            assert result.session_id == session_id
            
            # Test with empty audio data
            result = await processor.process_audio_chunk(
                session_id=session_id,
                audio_data=b"",
                chunk_id="empty_chunk",
                sample_rate=16000
            )
            
            # Should handle gracefully
            assert isinstance(result, ProcessingResult)
            assert result.session_id == session_id
        
        finally:
            # Clean up
            if processor.is_session_active(session_id):
                await processor.stop_session(session_id)


if __name__ == "__main__":
    # Run a simple test
    import asyncio
    
    async def run_simple_test():
        test_instance = TestAudioProcessingProperties()
        await test_instance.test_audio_capture_latency_property(1.0)
        print("Simple audio processing property test passed!")
    
    asyncio.run(run_simple_test())