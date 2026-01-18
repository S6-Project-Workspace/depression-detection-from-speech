"""
Tests for Buffer Manager and Circular Buffer

Tests the audio buffering, sliding window management, and circular buffer
functionality of the streaming system.

Reference: Real-time Streaming Analysis - Task 1.2
"""

import pytest
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from streaming.buffer_manager import BufferManager, BufferWindow, MultiSessionBufferManager
from streaming.circular_buffer import CircularAudioBuffer, HighPerformanceCircularBuffer
from streaming.models import AudioQualityMetric


class TestCircularAudioBuffer:
    """Test CircularAudioBuffer class."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = CircularAudioBuffer(
            buffer_size_seconds=5,
            sample_rate=16000
        )
        
        assert buffer.buffer_size_seconds == 5
        assert buffer.sample_rate == 16000
        assert buffer.buffer_size == 80000  # 5 * 16000
        assert buffer.write_pos == 0
        assert buffer.samples_written == 0
        assert len(buffer.audio_buffer) == 80000
    
    def test_simple_write_read(self):
        """Test basic write and read operations."""
        buffer = CircularAudioBuffer(
            buffer_size_seconds=2,
            sample_rate=1000  # Simple rate for testing
        )
        
        # Write some data
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        buffer.write(test_data)
        
        assert buffer.samples_written == 5
        assert buffer.get_available_samples() == 5
        
        # Read back the data
        read_data = buffer.read_window(5)
        
        assert len(read_data) == 5
        np.testing.assert_array_equal(read_data, test_data)
    
    def test_circular_wraparound(self):
        """Test circular buffer wraparound behavior."""
        buffer = CircularAudioBuffer(
            buffer_size_seconds=1,
            sample_rate=10  # 10 samples total
        )
        
        # Fill buffer completely
        data1 = np.arange(10, dtype=np.float32)
        buffer.write(data1)
        
        assert buffer.samples_written == 10
        assert buffer.write_pos == 0  # Wrapped around
        
        # Write more data (should overwrite beginning)
        data2 = np.array([100.0, 101.0, 102.0], dtype=np.float32)
        buffer.write(data2)
        
        assert buffer.samples_written == 13
        assert buffer.write_pos == 3
        assert buffer.overrun_count == 1  # Should detect overrun
        
        # Read recent data
        recent_data = buffer.read_window(5)
        expected = np.array([8.0, 9.0, 100.0, 101.0, 102.0])  # Last 5 samples
        np.testing.assert_array_equal(recent_data, expected)
    
    def test_timestamp_tracking(self):
        """Test timestamp tracking functionality."""
        buffer = CircularAudioBuffer(
            buffer_size_seconds=1,
            sample_rate=1000
        )
        
        # Write data with timestamp
        test_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        timestamp = datetime.now()
        buffer.write(test_data, timestamp)
        
        # Read with timestamps
        audio_data, timestamps = buffer.read_window_with_timestamps(3)
        
        assert len(audio_data) == 3
        assert len(timestamps) == 3
        np.testing.assert_array_equal(audio_data, test_data)
        
        # Check timestamp progression
        expected_timestamps = [
            timestamp.timestamp(),
            timestamp.timestamp() + 1/1000,
            timestamp.timestamp() + 2/1000
        ]
        
        np.testing.assert_array_almost_equal(timestamps, expected_timestamps, decimal=6)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        buffer = CircularAudioBuffer(
            buffer_size_seconds=1,
            sample_rate=1000
        )
        
        # Try to read from empty buffer
        data = buffer.read_window(100)
        assert len(data) == 0
        
        # Write small amount of data
        buffer.write(np.array([1.0, 2.0]))
        
        # Try to read more than available
        data = buffer.read_window(10)
        assert len(data) == 2  # Should return what's available
        
        # Check has_sufficient_data
        assert buffer.has_sufficient_data(2) is True
        assert buffer.has_sufficient_data(10) is False
    
    def test_time_range_calculation(self):
        """Test time range calculation."""
        buffer = CircularAudioBuffer(
            buffer_size_seconds=1,
            sample_rate=1000
        )
        
        # Empty buffer
        oldest, newest = buffer.get_time_range()
        assert oldest is None
        assert newest is None
        
        # Add some data
        timestamp = datetime.now()
        buffer.write(np.array([1.0, 2.0, 3.0]), timestamp)
        
        oldest, newest = buffer.get_time_range()
        assert oldest is not None
        assert newest is not None
        assert newest > oldest
    
    def test_buffer_statistics(self):
        """Test buffer statistics collection."""
        buffer = CircularAudioBuffer(
            buffer_size_seconds=2,
            sample_rate=1000
        )
        
        stats = buffer.get_statistics()
        
        assert 'buffer_size_samples' in stats
        assert 'samples_written' in stats
        assert 'utilization_percent' in stats
        assert 'overrun_count' in stats
        
        assert stats['buffer_size_samples'] == 2000
        assert stats['samples_written'] == 0
        assert stats['utilization_percent'] == 0.0


class TestBufferWindow:
    """Test BufferWindow class."""
    
    def test_window_creation(self):
        """Test buffer window creation."""
        audio_data = np.random.randn(16000).astype(np.float32)
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        # Create quality metrics
        quality_metrics = [
            AudioQualityMetric(
                timestamp=start_time,
                signal_to_noise_ratio=20.0,
                volume_level=0.5,
                clipping_detected=False,
                silence_ratio=0.1,
                quality_score=0.8
            )
        ]
        
        window = BufferWindow(
            window_id="test_window",
            audio_data=audio_data,
            start_time=start_time,
            end_time=end_time,
            quality_metrics=quality_metrics,
            sample_rate=16000
        )
        
        assert window.window_id == "test_window"
        assert len(window.audio_data) == 16000
        assert window.sample_rate == 16000
        assert window.duration_seconds == 1.0
        assert window.average_quality == 0.8
        assert window.has_clipping is False
    
    def test_quality_assessment(self):
        """Test window quality assessment."""
        audio_data = np.random.randn(1000).astype(np.float32)
        
        # High quality metrics
        high_quality_metrics = [
            AudioQualityMetric(
                timestamp=datetime.now(),
                signal_to_noise_ratio=25.0,
                volume_level=0.6,
                clipping_detected=False,
                silence_ratio=0.1,
                quality_score=0.9
            )
        ]
        
        window = BufferWindow(
            window_id="high_quality",
            audio_data=audio_data,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            quality_metrics=high_quality_metrics,
            sample_rate=1000
        )
        
        assert window.is_acceptable_quality(min_quality=0.7) is True
        assert window.is_acceptable_quality(min_quality=0.95) is False
        
        # Low quality with clipping
        low_quality_metrics = [
            AudioQualityMetric(
                timestamp=datetime.now(),
                signal_to_noise_ratio=5.0,
                volume_level=0.3,
                clipping_detected=True,
                silence_ratio=0.8,
                quality_score=0.3
            )
        ]
        
        window_low = BufferWindow(
            window_id="low_quality",
            audio_data=audio_data,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            quality_metrics=low_quality_metrics,
            sample_rate=1000
        )
        
        assert window_low.is_acceptable_quality() is False
        assert window_low.has_clipping is True
    
    def test_window_serialization(self):
        """Test window serialization to dictionary."""
        audio_data = np.random.randn(1000).astype(np.float32)
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        window = BufferWindow(
            window_id="serialize_test",
            audio_data=audio_data,
            start_time=start_time,
            end_time=end_time,
            quality_metrics=[],
            sample_rate=1000
        )
        
        window_dict = window.to_dict()
        
        assert window_dict['window_id'] == "serialize_test"
        assert window_dict['sample_count'] == 1000
        assert window_dict['sample_rate'] == 1000
        assert 'start_time' in window_dict
        assert 'end_time' in window_dict


class TestBufferManager:
    """Test BufferManager class."""
    
    def test_manager_initialization(self):
        """Test buffer manager initialization."""
        manager = BufferManager(
            sample_rate=16000,
            buffer_size_seconds=30,
            window_size_seconds=10,
            overlap_seconds=2
        )
        
        assert manager.sample_rate == 16000
        assert manager.buffer_size_seconds == 30
        assert manager.window_size_seconds == 10
        assert manager.overlap_seconds == 2
        assert manager.step_samples == 8 * 16000  # 10s - 2s = 8s
    
    def test_invalid_configuration(self):
        """Test invalid configuration handling."""
        # Overlap >= window size
        with pytest.raises(ValueError):
            BufferManager(
                window_size_seconds=5,
                overlap_seconds=5
            )
        
        # Window size > buffer size
        with pytest.raises(ValueError):
            BufferManager(
                buffer_size_seconds=10,
                window_size_seconds=15
            )
    
    def test_audio_chunk_writing(self):
        """Test writing audio chunks to buffer."""
        manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=5,
            window_size_seconds=2,
            overlap_seconds=1
        )
        
        # Write some audio chunks
        chunk1 = np.random.randn(1000).astype(np.float32)  # 1 second
        chunk2 = np.random.randn(1000).astype(np.float32)  # 1 second
        
        quality_metric = AudioQualityMetric(
            timestamp=datetime.now(),
            signal_to_noise_ratio=20.0,
            volume_level=0.5,
            clipping_detected=False,
            silence_ratio=0.1,
            quality_score=0.8
        )
        
        manager.write_audio_chunk(chunk1, quality_metric)
        manager.write_audio_chunk(chunk2, quality_metric)
        
        assert manager.total_samples_written == 2000
        assert len(manager.quality_metrics_buffer) == 2
    
    def test_window_extraction(self):
        """Test sliding window extraction."""
        manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=5,
            window_size_seconds=2,
            overlap_seconds=0
        )
        
        # Write enough data for a window
        for i in range(3):  # 3 seconds of data
            chunk = np.full(1000, i, dtype=np.float32)  # Fill with value i
            quality_metric = AudioQualityMetric(
                timestamp=datetime.now(),
                signal_to_noise_ratio=20.0,
                volume_level=0.5,
                clipping_detected=False,
                silence_ratio=0.1,
                quality_score=0.8
            )
            manager.write_audio_chunk(chunk, quality_metric)
            time.sleep(0.01)  # Small delay for timestamp differences
        
        # Extract window
        window = manager.get_latest_window()
        
        assert window is not None
        assert len(window.audio_data) == 2000  # 2 seconds at 1000 Hz
        assert window.sample_rate == 1000
        assert window.average_quality == 0.8
    
    def test_overlapping_windows(self):
        """Test overlapping window extraction."""
        manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=10,
            window_size_seconds=3,
            overlap_seconds=1
        )
        
        # Write enough data
        for i in range(6):  # 6 seconds of data
            chunk = np.random.randn(1000).astype(np.float32)
            quality_metric = AudioQualityMetric(
                timestamp=datetime.now(),
                signal_to_noise_ratio=20.0,
                volume_level=0.5,
                clipping_detected=False,
                silence_ratio=0.1,
                quality_score=0.8
            )
            manager.write_audio_chunk(chunk, quality_metric)
        
        # Extract multiple overlapping windows
        windows = manager.get_overlapping_windows(count=2)
        
        assert len(windows) <= 2  # May be less if insufficient data
        
        if len(windows) == 2:
            # Check overlap
            assert len(windows[0].audio_data) == 3000  # 3 seconds
            assert len(windows[1].audio_data) == 3000  # 3 seconds
            
            # Windows should have different IDs
            assert windows[0].window_id != windows[1].window_id
    
    def test_quality_filtering(self):
        """Test quality-based window filtering."""
        manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=5,
            window_size_seconds=2
        )
        
        # Write low quality data
        chunk = np.random.randn(2000).astype(np.float32)
        low_quality_metric = AudioQualityMetric(
            timestamp=datetime.now(),
            signal_to_noise_ratio=5.0,
            volume_level=0.1,
            clipping_detected=True,
            silence_ratio=0.9,
            quality_score=0.2  # Low quality
        )
        
        manager.write_audio_chunk(chunk, low_quality_metric)
        
        # Try to get window with high quality requirement
        window = manager.get_latest_window(min_quality=0.7)
        assert window is None  # Should be rejected due to low quality
        
        # Try with lower quality requirement
        window = manager.get_latest_window(min_quality=0.1)
        assert window is not None  # Should be accepted
    
    def test_buffer_health_monitoring(self):
        """Test buffer health monitoring."""
        manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=2
        )
        
        health = manager.get_buffer_health()
        
        assert 'status' in health
        assert 'utilization_percent' in health
        assert 'buffer_size_samples' in health
        assert health['status'] == 'healthy'  # Empty buffer is healthy
        
        # Fill buffer significantly
        for _ in range(15):  # 1.5 seconds of data
            chunk = np.random.randn(100).astype(np.float32)
            manager.write_audio_chunk(chunk)
        
        health = manager.get_buffer_health()
        assert health['utilization_percent'] > 0
    
    def test_window_caching(self):
        """Test window caching functionality."""
        manager = BufferManager(
            sample_rate=1000,
            buffer_size_seconds=5,
            window_size_seconds=2,
            max_windows_cached=3
        )
        
        # Generate windows
        for i in range(5):
            chunk = np.random.randn(1000).astype(np.float32)
            quality_metric = AudioQualityMetric(
                timestamp=datetime.now(),
                signal_to_noise_ratio=20.0,
                volume_level=0.5,
                clipping_detected=False,
                silence_ratio=0.1,
                quality_score=0.8
            )
            manager.write_audio_chunk(chunk, quality_metric)
            
            if i >= 1:  # Need at least 2 seconds for a window
                window = manager.get_latest_window()
                if window:
                    # Check if window is cached
                    cached_window = manager.get_cached_window(window.window_id)
                    assert cached_window is not None
                    assert cached_window.window_id == window.window_id
        
        # Cache should be limited to max_windows_cached
        assert len(manager.windows_cache) <= 3


class TestMultiSessionBufferManager:
    """Test MultiSessionBufferManager class."""
    
    def test_multi_session_initialization(self):
        """Test multi-session manager initialization."""
        manager = MultiSessionBufferManager(
            max_sessions=5,
            default_buffer_config={'sample_rate': 16000}
        )
        
        assert manager.max_sessions == 5
        assert manager.default_buffer_config['sample_rate'] == 16000
        assert len(manager.session_buffers) == 0
    
    def test_session_creation(self):
        """Test session buffer creation."""
        manager = MultiSessionBufferManager(max_sessions=3)
        
        # Create sessions
        success1 = manager.create_session_buffer("session1")
        success2 = manager.create_session_buffer("session2")
        
        assert success1 is True
        assert success2 is True
        assert len(manager.session_buffers) == 2
        
        # Get session buffer
        buffer1 = manager.get_session_buffer("session1")
        assert buffer1 is not None
        assert isinstance(buffer1, BufferManager)
    
    def test_session_limit(self):
        """Test session limit enforcement."""
        manager = MultiSessionBufferManager(max_sessions=2)
        
        # Create maximum sessions
        assert manager.create_session_buffer("session1") is True
        assert manager.create_session_buffer("session2") is True
        
        # Try to exceed limit
        assert manager.create_session_buffer("session3") is False
        assert len(manager.session_buffers) == 2
    
    def test_session_removal(self):
        """Test session buffer removal."""
        manager = MultiSessionBufferManager()
        
        # Create and remove session
        manager.create_session_buffer("test_session")
        assert "test_session" in manager.session_buffers
        
        success = manager.remove_session_buffer("test_session")
        assert success is True
        assert "test_session" not in manager.session_buffers
        
        # Try to remove non-existent session
        success = manager.remove_session_buffer("non_existent")
        assert success is False
    
    def test_session_statistics(self):
        """Test session statistics collection."""
        manager = MultiSessionBufferManager()
        
        # Create sessions
        manager.create_session_buffer("session1")
        manager.create_session_buffer("session2")
        
        stats = manager.get_session_statistics()
        
        assert stats['total_sessions'] == 2
        assert stats['max_sessions'] == 10  # Default
        assert 'sessions' in stats
        assert 'session1' in stats['sessions']
        assert 'session2' in stats['sessions']


class TestHighPerformanceCircularBuffer:
    """Test HighPerformanceCircularBuffer class."""
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        buffer = HighPerformanceCircularBuffer(
            buffer_size_seconds=1,
            sample_rate=1000
        )
        
        # Write some data
        test_data = np.random.randn(100).astype(np.float32)
        buffer.write(test_data)
        buffer.write(test_data)
        
        # Check performance metrics
        assert buffer.write_count == 2
        assert buffer.total_write_time > 0
        assert buffer.get_average_write_time() > 0
        
        perf_stats = buffer.get_performance_statistics()
        assert 'performance' in perf_stats
        assert 'write_count' in perf_stats['performance']
        assert 'average_write_time_ms' in perf_stats['performance']


# Property-based tests using hypothesis
try:
    from hypothesis import given, strategies as st, settings
    
    class TestBufferManagerProperties:
        """Property-based tests for buffer manager."""
        
        @given(
            buffer_size=st.integers(min_value=2, max_value=10),
            window_size=st.integers(min_value=1, max_value=5),
            overlap=st.integers(min_value=0, max_value=2)
        )
        @settings(deadline=None, max_examples=10)
        def test_buffer_configuration_properties(self, buffer_size, window_size, overlap):
            """Test buffer configuration properties."""
            # Ensure valid configuration
            if overlap >= window_size or window_size > buffer_size:
                return  # Skip invalid configurations
            
            manager = BufferManager(
                sample_rate=1000,
                buffer_size_seconds=buffer_size,
                window_size_seconds=window_size,
                overlap_seconds=overlap
            )
            
            # Properties
            assert manager.buffer_size_seconds == buffer_size
            assert manager.window_size_seconds == window_size
            assert manager.overlap_seconds == overlap
            assert manager.step_samples == (window_size - overlap) * 1000
            assert manager.step_samples > 0
        
        @given(
            audio_data=st.lists(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
                min_size=100, max_size=2000
            )
        )
        @settings(deadline=None, max_examples=10)
        def test_circular_buffer_properties(self, audio_data):
            """Test circular buffer properties."""
            buffer = CircularAudioBuffer(
                buffer_size_seconds=5,
                sample_rate=1000
            )
            
            audio_array = np.array(audio_data, dtype=np.float32)
            buffer.write(audio_array)
            
            # Properties
            assert buffer.samples_written == len(audio_array)
            assert buffer.get_available_samples() <= buffer.buffer_size
            assert 0.0 <= buffer.get_utilization() <= 1.0
            
            # Read back data
            if len(audio_array) <= buffer.buffer_size:
                read_data = buffer.read_window(len(audio_array))
                assert len(read_data) == len(audio_array)
                np.testing.assert_array_equal(read_data, audio_array)
        
        @given(
            chunk_count=st.integers(min_value=1, max_value=10),
            chunk_size=st.integers(min_value=100, max_value=1000)
        )
        @settings(deadline=None, max_examples=10)
        def test_buffer_manager_properties(self, chunk_count, chunk_size):
            """Test buffer manager properties."""
            manager = BufferManager(
                sample_rate=1000,
                buffer_size_seconds=20,  # Large buffer
                window_size_seconds=2
            )
            
            # Write multiple chunks
            total_samples = 0
            for i in range(chunk_count):
                chunk = np.random.randn(chunk_size).astype(np.float32)
                manager.write_audio_chunk(chunk)
                total_samples += chunk_size
            
            # Properties
            assert manager.total_samples_written == total_samples
            
            # Buffer health should be reasonable
            health = manager.get_buffer_health()
            assert health['utilization_percent'] >= 0.0
            assert health['status'] in ['healthy', 'warning', 'critical']

except ImportError:
    # Hypothesis not available
    pass


if __name__ == "__main__":
    pytest.main([__file__])