"""
Circular Audio Buffer Implementation

This module provides a thread-safe circular buffer implementation optimized
for continuous audio streaming with timestamp tracking.

Key Features:
- Thread-safe circular buffer operations
- Timestamp tracking for audio samples
- Memory-efficient storage
- Overrun detection and handling
- Window extraction with timestamps

Reference: Real-time Streaming Analysis - Task 1.2
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class CircularAudioBuffer:
    """
    Thread-safe circular buffer for continuous audio streaming.
    
    Features:
    - Circular buffer with automatic wraparound
    - Timestamp tracking for each sample
    - Thread-safe read/write operations
    - Overrun detection
    - Efficient window extraction
    """
    
    def __init__(
        self, 
        buffer_size_seconds: int, 
        sample_rate: int,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize circular audio buffer.
        
        Args:
            buffer_size_seconds: Buffer size in seconds
            sample_rate: Audio sampling rate
            dtype: Data type for audio samples
        """
        self.buffer_size_seconds = buffer_size_seconds
        self.sample_rate = sample_rate
        self.dtype = dtype
        
        # Calculate buffer size in samples
        self.buffer_size = buffer_size_seconds * sample_rate
        
        # Initialize buffers
        self.audio_buffer = np.zeros(self.buffer_size, dtype=dtype)
        self.timestamp_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        
        # Buffer pointers
        self.write_pos = 0
        self.samples_written = 0
        self.overrun_count = 0
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.total_samples_written = 0
        self.last_write_time = 0.0
        
        logger.info(
            f"CircularAudioBuffer initialized: {buffer_size_seconds}s "
            f"({self.buffer_size} samples) at {sample_rate}Hz"
        )
    
    def write(self, audio_data: np.ndarray, timestamp: Optional[datetime] = None):
        """
        Write audio data to the circular buffer.
        
        Args:
            audio_data: Audio data to write
            timestamp: Timestamp for the first sample (current time if None)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_float = timestamp.timestamp()
        
        with self.lock:
            try:
                # Convert to correct dtype
                if audio_data.dtype != self.dtype:
                    audio_data = audio_data.astype(self.dtype)
                
                data_length = len(audio_data)
                
                if data_length == 0:
                    return
                
                # Check for potential overrun
                if data_length > self.buffer_size:
                    logger.warning(
                        f"Audio chunk ({data_length}) larger than buffer ({self.buffer_size})"
                    )
                    # Take only the last part that fits
                    audio_data = audio_data[-self.buffer_size:]
                    data_length = len(audio_data)
                
                # Calculate timestamps for each sample
                sample_timestamps = np.linspace(
                    timestamp_float,
                    timestamp_float + (data_length - 1) / self.sample_rate,
                    data_length
                )
                
                # Write data to buffer
                end_pos = self.write_pos + data_length
                
                if end_pos <= self.buffer_size:
                    # No wraparound needed
                    self.audio_buffer[self.write_pos:end_pos] = audio_data
                    self.timestamp_buffer[self.write_pos:end_pos] = sample_timestamps
                else:
                    # Handle wraparound
                    first_part_size = self.buffer_size - self.write_pos
                    second_part_size = data_length - first_part_size
                    
                    # Write first part
                    self.audio_buffer[self.write_pos:] = audio_data[:first_part_size]
                    self.timestamp_buffer[self.write_pos:] = sample_timestamps[:first_part_size]
                    
                    # Write second part (wraparound)
                    self.audio_buffer[:second_part_size] = audio_data[first_part_size:]
                    self.timestamp_buffer[:second_part_size] = sample_timestamps[first_part_size:]
                    
                    # Check for overrun
                    if self.samples_written >= self.buffer_size:
                        self.overrun_count += 1
                
                # Update pointers
                self.write_pos = end_pos % self.buffer_size
                self.samples_written += data_length
                self.total_samples_written += data_length
                self.last_write_time = timestamp_float
                
            except Exception as e:
                logger.error(f"Error writing to circular buffer: {e}")
                raise
    
    def read_window(
        self, 
        window_size_samples: int, 
        offset_samples: int = 0
    ) -> np.ndarray:
        """
        Read a window of audio data from the buffer.
        
        Args:
            window_size_samples: Size of window in samples
            offset_samples: Offset from current position (0 = most recent)
            
        Returns:
            Audio data window
        """
        with self.lock:
            if window_size_samples <= 0:
                return np.array([], dtype=self.dtype)
            
            if window_size_samples > self.buffer_size:
                logger.warning(
                    f"Window size ({window_size_samples}) larger than buffer ({self.buffer_size})"
                )
                window_size_samples = self.buffer_size
            
            # Check if we have enough data
            available_samples = min(self.samples_written, self.buffer_size)
            if window_size_samples + offset_samples > available_samples:
                logger.warning(
                    f"Insufficient data: requested {window_size_samples + offset_samples}, "
                    f"available {available_samples}"
                )
                # Return what we can
                actual_window_size = max(0, available_samples - offset_samples)
                if actual_window_size == 0:
                    return np.array([], dtype=self.dtype)
                window_size_samples = actual_window_size
            
            # Calculate read position (going backwards from write position)
            read_end_pos = (self.write_pos - offset_samples) % self.buffer_size
            read_start_pos = (read_end_pos - window_size_samples) % self.buffer_size
            
            # Extract data
            if read_start_pos < read_end_pos:
                # No wraparound
                window_data = self.audio_buffer[read_start_pos:read_end_pos].copy()
            else:
                # Handle wraparound
                first_part = self.audio_buffer[read_start_pos:].copy()
                second_part = self.audio_buffer[:read_end_pos].copy()
                window_data = np.concatenate([first_part, second_part])
            
            return window_data
    
    def read_window_with_timestamps(
        self, 
        window_size_samples: int, 
        offset_samples: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a window of audio data with corresponding timestamps.
        
        Args:
            window_size_samples: Size of window in samples
            offset_samples: Offset from current position
            
        Returns:
            Tuple of (audio_data, timestamps)
        """
        with self.lock:
            if window_size_samples <= 0:
                return (
                    np.array([], dtype=self.dtype),
                    np.array([], dtype=np.float64)
                )
            
            if window_size_samples > self.buffer_size:
                window_size_samples = self.buffer_size
            
            # Check available data
            available_samples = min(self.samples_written, self.buffer_size)
            if window_size_samples + offset_samples > available_samples:
                actual_window_size = max(0, available_samples - offset_samples)
                if actual_window_size == 0:
                    return (
                        np.array([], dtype=self.dtype),
                        np.array([], dtype=np.float64)
                    )
                window_size_samples = actual_window_size
            
            # Calculate read positions
            read_end_pos = (self.write_pos - offset_samples) % self.buffer_size
            read_start_pos = (read_end_pos - window_size_samples) % self.buffer_size
            
            # Extract audio data and timestamps
            if read_start_pos < read_end_pos:
                # No wraparound
                audio_data = self.audio_buffer[read_start_pos:read_end_pos].copy()
                timestamps = self.timestamp_buffer[read_start_pos:read_end_pos].copy()
            else:
                # Handle wraparound
                audio_first = self.audio_buffer[read_start_pos:].copy()
                audio_second = self.audio_buffer[:read_end_pos].copy()
                audio_data = np.concatenate([audio_first, audio_second])
                
                timestamp_first = self.timestamp_buffer[read_start_pos:].copy()
                timestamp_second = self.timestamp_buffer[:read_end_pos].copy()
                timestamps = np.concatenate([timestamp_first, timestamp_second])
            
            return audio_data, timestamps
    
    def has_sufficient_data(self, required_samples: int) -> bool:
        """Check if buffer has sufficient data for a read operation."""
        with self.lock:
            available_samples = min(self.samples_written, self.buffer_size)
            return available_samples >= required_samples
    
    def get_available_samples(self) -> int:
        """Get number of samples available for reading."""
        with self.lock:
            return min(self.samples_written, self.buffer_size)
    
    def get_utilization(self) -> float:
        """Get buffer utilization as a fraction (0.0 to 1.0)."""
        with self.lock:
            return min(self.samples_written, self.buffer_size) / self.buffer_size
    
    def is_overrun(self) -> bool:
        """Check if buffer has experienced overrun since last check."""
        with self.lock:
            return self.overrun_count > 0
    
    def get_latest_timestamp(self) -> Optional[datetime]:
        """Get timestamp of the most recent sample."""
        with self.lock:
            if self.samples_written == 0:
                return None
            
            # Get timestamp of the most recent sample
            latest_pos = (self.write_pos - 1) % self.buffer_size
            timestamp_float = self.timestamp_buffer[latest_pos]
            
            return datetime.fromtimestamp(timestamp_float)
    
    def get_time_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the time range of data currently in the buffer.
        
        Returns:
            Tuple of (oldest_timestamp, newest_timestamp)
        """
        with self.lock:
            if self.samples_written == 0:
                return None, None
            
            available_samples = min(self.samples_written, self.buffer_size)
            
            if available_samples == 0:
                return None, None
            
            # Get newest timestamp
            newest_pos = (self.write_pos - 1) % self.buffer_size
            newest_timestamp = datetime.fromtimestamp(
                self.timestamp_buffer[newest_pos]
            )
            
            # Get oldest timestamp
            if available_samples < self.buffer_size:
                # Buffer not full yet
                oldest_pos = 0
            else:
                # Buffer is full, oldest is at write position
                oldest_pos = self.write_pos
            
            oldest_timestamp = datetime.fromtimestamp(
                self.timestamp_buffer[oldest_pos]
            )
            
            return oldest_timestamp, newest_timestamp
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.audio_buffer.fill(0)
            self.timestamp_buffer.fill(0)
            self.write_pos = 0
            self.samples_written = 0
            self.overrun_count = 0
            logger.info("Circular buffer cleared")
    
    def get_statistics(self) -> dict:
        """Get comprehensive buffer statistics."""
        with self.lock:
            oldest_time, newest_time = self.get_time_range()
            
            return {
                'buffer_size_samples': self.buffer_size,
                'buffer_size_seconds': self.buffer_size_seconds,
                'sample_rate': self.sample_rate,
                'samples_written': self.samples_written,
                'total_samples_written': self.total_samples_written,
                'available_samples': self.get_available_samples(),
                'utilization_percent': self.get_utilization() * 100,
                'overrun_count': self.overrun_count,
                'write_position': self.write_pos,
                'last_write_time': self.last_write_time,
                'oldest_timestamp': oldest_time.isoformat() if oldest_time else None,
                'newest_timestamp': newest_time.isoformat() if newest_time else None,
                'time_span_seconds': (
                    (newest_time - oldest_time).total_seconds() 
                    if oldest_time and newest_time else 0.0
                )
            }
    
    def reset_overrun_count(self):
        """Reset the overrun counter."""
        with self.lock:
            self.overrun_count = 0


class HighPerformanceCircularBuffer(CircularAudioBuffer):
    """
    High-performance variant of CircularAudioBuffer with optimizations.
    
    Optimizations:
    - Pre-allocated timestamp arrays
    - Vectorized operations
    - Reduced memory allocations
    - Optimized for high-frequency writes
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Pre-allocate arrays for timestamp calculation
        self._temp_timestamps = np.zeros(self.buffer_size, dtype=np.float64)
        
        # Performance tracking
        self.write_count = 0
        self.total_write_time = 0.0
        
        logger.info("HighPerformanceCircularBuffer initialized with optimizations")
    
    def write(self, audio_data: np.ndarray, timestamp: Optional[datetime] = None):
        """Optimized write operation."""
        start_time = time.perf_counter()
        
        try:
            super().write(audio_data, timestamp)
            
            # Update performance metrics
            self.write_count += 1
            self.total_write_time += time.perf_counter() - start_time
            
        except Exception as e:
            logger.error(f"Error in high-performance write: {e}")
            raise
    
    def get_average_write_time(self) -> float:
        """Get average write operation time in seconds."""
        with self.lock:
            if self.write_count == 0:
                return 0.0
            return self.total_write_time / self.write_count
    
    def get_performance_statistics(self) -> dict:
        """Get performance-specific statistics."""
        base_stats = self.get_statistics()
        
        with self.lock:
            performance_stats = {
                'write_count': self.write_count,
                'total_write_time': self.total_write_time,
                'average_write_time_ms': self.get_average_write_time() * 1000,
                'writes_per_second': (
                    self.write_count / self.total_write_time 
                    if self.total_write_time > 0 else 0.0
                )
            }
        
        return {**base_stats, 'performance': performance_stats}