"""
Phase 1 Integration Tests

Tests the integration of core streaming infrastructure components:
- Audio Input Handler
- Buffer Manager  
- Stream Processor
- Session Manager

Reference: Real-time Streaming Analysis - Phase 1
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from streaming import (
    AudioInputHandler, BufferManager, StreamProcessor, 
    StreamingSessionManager, SessionConfig, AudioQualityMetric
)


class TestPhase1Integration:
    """Integration tests for Phase 1 components."""
    
    def test_component_imports(self):
        """Test that all Phase 1 components can be imported."""
        # This test verifies the module structure is correct
        assert AudioInputHandler is not None
        assert BufferManager is not None
        assert StreamProcessor is not None
        assert StreamingSessionManager is not None
    
    @patch('streaming.audio_input_handler.pyaudio')
    def test_audio_to_buffer_integration(self, mock_pyaudio):
        """Test integration between audio handler and buffer manager."""
        mock_pyaudio.PyAudio.return_value = Mock()
        
        # Create components
        buffer_manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=5,
            window_size_seconds=2
        )
        
        # Simulate audio data
        audio_data = np.random.randn(1000).astype(np.float32)  # 1 second
        quality_metric = AudioQualityMetric(
            timestamp=datetime.now(),
            signal_to_noise_ratio=20.0,
            volume_level=0.5,
            clipping_detected=False,
            silence_ratio=0.1,
            quality_score=0.8
        )
        
        # Write audio to buffer
        buffer_manager.write_audio_chunk(audio_data, quality_metric)
        
        # Verify buffer received data
        assert buffer_manager.total_samples_written == 1000
        assert len(buffer_manager.quality_metrics_buffer) == 1
        
        # Write more data to enable window extraction
        buffer_manager.write_audio_chunk(audio_data, quality_metric)
        
        # Extract window
        window = buffer_manager.get_latest_window()
        assert window is not None
        assert len(window.audio_data) == 2000  # 2 seconds
    
    @pytest.mark.asyncio
    async def test_buffer_to_processor_integration(self):
        """Test integration between buffer manager and stream processor."""
        # Create components
        buffer_manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=5,
            window_size_seconds=2
        )
        
        stream_processor = StreamProcessor(device='cpu')
        
        # Generate test data
        audio_data = np.random.randn(2000).astype(np.float32)  # 2 seconds
        
        # Process through stream processor
        result = await stream_processor.process_audio_chunk(
            audio_data=audio_data,
            chunk_id="integration_test"
        )
        
        # Verify processing result
        assert result is not None
        assert result.chunk_id == "integration_test"
        assert 0.0 <= result.risk_score <= 1.0
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_session_manager_integration(self):
        """Test session manager integration with all components."""
        # Create session manager
        session_manager = StreamingSessionManager(
            max_concurrent_sessions=2,
            device='cpu'
        )
        
        # Create session
        session_id = await session_manager.create_session(
            patient_id="test_patient_001",
            config=SessionConfig(
                patient_id="test_patient_001",
                sample_rate=1000,
                buffer_size_seconds=5
            )
        )
        
        assert session_id is not None
        assert len(session_manager.active_sessions) == 1
        
        # Get session statistics
        stats = session_manager.get_session_statistics(session_id)
        assert stats is not None
        assert stats['session_id'] == session_id
        assert stats['patient_id'] == "test_patient_001"
        
        # Cleanup
        await session_manager.remove_session(session_id)
        assert len(session_manager.active_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_full_pipeline_simulation(self):
        """Test full pipeline with simulated audio processing."""
        # Create session manager
        session_manager = StreamingSessionManager(device='cpu')
        
        # Create and start session
        session_id = await session_manager.create_session("test_patient")
        assert session_id is not None
        
        # Get session for inspection
        session = session_manager.get_session(session_id)
        assert session is not None
        assert session.status.value in ['initializing', 'active']
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Check session is still valid
        stats = session_manager.get_session_statistics(session_id)
        assert stats is not None
        
        # Cleanup
        await session_manager.remove_session(session_id)
    
    def test_component_configuration_compatibility(self):
        """Test that component configurations are compatible."""
        # Test compatible configurations
        sample_rate = 16000
        buffer_size = 30
        window_size = 10
        
        # Buffer manager
        buffer_manager = BufferManager(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size,
            window_size_seconds=window_size
        )
        
        # Stream processor
        stream_processor = StreamProcessor(
            device='cpu',
            target_latency_ms=2000.0
        )
        
        # Session config
        session_config = SessionConfig(
            patient_id="test",
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size
        )
        
        # Verify configurations are consistent
        assert buffer_manager.sample_rate == session_config.sample_rate
        assert buffer_manager.buffer_size_seconds == session_config.buffer_size_seconds
        assert stream_processor.target_latency_ms == 2000.0
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across components."""
        session_manager = StreamingSessionManager(device='cpu')
        
        # Test session creation with invalid config
        session_id = await session_manager.create_session(
            patient_id="",  # Invalid patient ID
            config=SessionConfig(patient_id="")
        )
        
        # Should still create session (validation is lenient)
        assert session_id is not None
        
        # Cleanup
        await session_manager.remove_session(session_id)
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring across components."""
        # Create components with performance tracking
        buffer_manager = BufferManager(sample_rate=1000)
        stream_processor = StreamProcessor(device='cpu')
        
        # Get initial statistics
        buffer_stats = buffer_manager.get_statistics()
        processor_stats = stream_processor.get_processing_statistics()
        
        # Verify statistics structure
        assert 'total_samples_written' in buffer_stats
        assert 'total_chunks_processed' in processor_stats
        
        # Initial values should be zero
        assert buffer_stats['total_samples_written'] == 0
        assert processor_stats['total_chunks_processed'] == 0


class TestPhase1ComponentInteraction:
    """Test specific component interactions."""
    
    def test_buffer_window_to_processor_format(self):
        """Test that buffer windows are compatible with processor input."""
        buffer_manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=3,
            window_size_seconds=2
        )
        
        # Add enough data for a window
        for i in range(3):
            audio_chunk = np.random.randn(1000).astype(np.float32)
            buffer_manager.write_audio_chunk(audio_chunk)
        
        # Get window
        window = buffer_manager.get_latest_window()
        assert window is not None
        
        # Verify window format is compatible with processor
        assert isinstance(window.audio_data, np.ndarray)
        assert window.audio_data.dtype == np.float32
        assert len(window.audio_data) > 0
        assert window.sample_rate == 1000
    
    @pytest.mark.asyncio
    async def test_session_lifecycle_integration(self):
        """Test complete session lifecycle."""
        session_manager = StreamingSessionManager(device='cpu')
        
        # Create session
        session_id = await session_manager.create_session("lifecycle_test")
        assert session_id is not None
        
        # Check initial state
        session = session_manager.get_session(session_id)
        assert session.status.value in ['initializing', 'active']
        
        # Simulate session operations
        await asyncio.sleep(0.05)
        
        # Stop session
        success = await session_manager.stop_session(session_id)
        assert success is True
        
        # Check final state
        session = session_manager.get_session(session_id)
        assert session.status.value in ['completed', 'active']  # May still be active
        
        # Remove session
        await session_manager.remove_session(session_id)
        assert session_id not in session_manager.active_sessions


if __name__ == "__main__":
    pytest.main([__file__])