"""
Buffer Manager for Real-time Streaming

This module manages audio buffering with sliding windows for the streaming
depression detection system. Implements circular buffers and sliding window
extraction for efficient memory usage.

Key Features:
- Circular buffer implementation for continuous audio
- Sliding window management (30s windows, 5s overlap)
- Thread-safe operations
- Memory-efficient storage
- Buffer health monitoring

Reference: Real-time Streaming Analysis - Task 1.2
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .models import AudioQualityMetric, SessionStatus
from .circular_buffer import CircularAudioBuffer

logger = logging.getLogger(__name__)


class BufferWindow:
    """Represents a sliding window of audio data."""
    
    def __init__(
        self,
        window_id: str,
        audio_data: np.ndarray,
        start_time: datetime,
        end_time: datetime,
        quality_metrics: List[AudioQualityMetric],
        sample_rate: int
    ):
        self.window_id = window_id
        self.audio_data = audio_data
        self.start_time = start_time
        self.end_time = end_time
        self.quality_metrics = quality_metrics
        self.sample_rate = sample_rate
        
        # Calculate derived properties
        self.duration_seconds = (end_time - start_time).total_seconds()
        self.sample_count = len(audio_data)
        
        # Quality summary
        if quality_metrics:
            self.average_quality = np.mean([m.quality_score for m in quality_metrics])
            self.min_quality = min(m.quality_score for m in quality_metrics)
            self.has_clipping = any(m.clipping_detected for m in quality_metrics)
        else:
            self.average_quality = 0.0
            self.min_quality = 0.0
            self.has_clipping = False
    
    def is_acceptable_quality(self, min_quality: float = 0.6) -> bool:
        """Check if window has acceptable audio quality."""
        return (
            self.average_quality >= min_quality and
            self.min_quality >= min_quality * 0.5 and
            not self.has_clipping
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'window_id': self.window_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'sample_count': self.sample_count,
            'sample_rate': self.sample_rate,
            'average_quality': self.average_quality,
            'min_quality': self.min_quality,
            'has_clipping': self.has_clipping,
            'quality_metric_count': len(self.quality_metrics)
        }


class BufferManager:
    """
    Manages audio buffering with sliding windows for streaming analysis.
    
    Features:
    - Circular buffer for continuous audio storage
    - Sliding window extraction with configurable overlap
    - Thread-safe operations
    - Buffer health monitoring
    - Memory usage optimization
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_size_seconds: int = 30,
        window_size_seconds: int = 30,
        overlap_seconds: int = 5,
        max_windows_cached: int = 10
    ):
        """
        Initialize buffer manager.
        
        Args:
            sample_rate: Audio sampling rate
            buffer_size_seconds: Total buffer size in seconds
            window_size_seconds: Size of each sliding window
            overlap_seconds: Overlap between consecutive windows
            max_windows_cached: Maximum number of windows to keep in memory
        """
        self.sample_rate = sample_rate
        self.buffer_size_seconds = buffer_size_seconds
        self.window_size_seconds = window_size_seconds
        self.overlap_seconds = overlap_seconds
        self.max_windows_cached = max_windows_cached
        
        # Validate parameters
        if overlap_seconds >= window_size_seconds:
            raise ValueError("Overlap must be less than window size")
        
        if window_size_seconds > buffer_size_seconds:
            raise ValueError("Window size must not exceed buffer size")
        
        # Calculate sizes in samples
        self.buffer_size_samples = buffer_size_seconds * sample_rate
        self.window_size_samples = window_size_seconds * sample_rate
        self.overlap_samples = overlap_seconds * sample_rate
        self.step_samples = self.window_size_samples - self.overlap_samples
        
        # Initialize circular buffer
        self.circular_buffer = CircularAudioBuffer(
            buffer_size_seconds=buffer_size_seconds,
            sample_rate=sample_rate
        )
        
        # Window management
        self.windows_cache: Dict[str, BufferWindow] = {}
        self.window_creation_times: Dict[str, datetime] = {}
        self.next_window_id = 0
        
        # Quality metrics tracking
        self.quality_metrics_buffer = deque(maxlen=1000)
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.total_samples_written = 0
        self.total_windows_created = 0
        self.buffer_overruns = 0
        self.last_window_time = None
        
        logger.info(
            f"BufferManager initialized: {buffer_size_seconds}s buffer, "
            f"{window_size_seconds}s windows, {overlap_seconds}s overlap"
        )
    
    def write_audio_chunk(
        self, 
        audio_data: np.ndarray, 
        quality_metric: Optional[AudioQualityMetric] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Write audio chunk to buffer.
        
        Args:
            audio_data: Audio data to write
            quality_metric: Associated quality metric
            timestamp: Timestamp of the audio chunk
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            try:
                # Write to circular buffer
                self.circular_buffer.write(audio_data, timestamp)
                
                # Store quality metric
                if quality_metric:
                    self.quality_metrics_buffer.append((timestamp, quality_metric))
                
                # Update statistics
                self.total_samples_written += len(audio_data)
                
                # Check for buffer overrun
                if self.circular_buffer.is_overrun():
                    self.buffer_overruns += 1
                    logger.warning("Buffer overrun detected")
                
            except Exception as e:
                logger.error(f"Error writing audio chunk: {e}")
                raise
    
    def get_latest_window(self, min_quality: float = 0.0) -> Optional[BufferWindow]:
        """
        Get the latest complete window from the buffer.
        
        Args:
            min_quality: Minimum acceptable quality threshold
            
        Returns:
            BufferWindow if available and meets quality requirements
        """
        with self.lock:
            try:
                # Check if we have enough data for a window
                if not self.circular_buffer.has_sufficient_data(self.window_size_samples):
                    return None
                
                # Calculate window timing
                current_time = datetime.now()
                window_start_time = current_time - timedelta(seconds=self.window_size_seconds)
                
                # Extract window data
                window_data, data_timestamps = self.circular_buffer.read_window_with_timestamps(
                    self.window_size_samples
                )
                
                if len(window_data) < self.window_size_samples:
                    logger.warning(f"Insufficient data for window: {len(window_data)} < {self.window_size_samples}")
                    return None
                
                # Get quality metrics for this window
                window_quality_metrics = self._get_quality_metrics_for_timerange(
                    window_start_time, current_time
                )
                
                # Create window
                window_id = f"window_{self.next_window_id}"
                self.next_window_id += 1
                
                window = BufferWindow(
                    window_id=window_id,
                    audio_data=window_data,
                    start_time=window_start_time,
                    end_time=current_time,
                    quality_metrics=window_quality_metrics,
                    sample_rate=self.sample_rate
                )
                
                # Check quality requirements
                if window.average_quality < min_quality:
                    logger.debug(f"Window quality too low: {window.average_quality} < {min_quality}")
                    return None
                
                # Cache window
                self._cache_window(window)
                
                # Update statistics
                self.total_windows_created += 1
                self.last_window_time = current_time
                
                return window
                
            except Exception as e:
                logger.error(f"Error getting latest window: {e}")
                return None
    
    def get_overlapping_windows(
        self, 
        count: int = 1, 
        min_quality: float = 0.0
    ) -> List[BufferWindow]:
        """
        Get multiple overlapping windows.
        
        Args:
            count: Number of windows to extract
            min_quality: Minimum acceptable quality threshold
            
        Returns:
            List of BufferWindow objects
        """
        windows = []
        
        with self.lock:
            try:
                # Check if we have enough data
                total_samples_needed = (
                    self.window_size_samples + 
                    (count - 1) * self.step_samples
                )
                
                if not self.circular_buffer.has_sufficient_data(total_samples_needed):
                    return windows
                
                current_time = datetime.now()
                
                # Extract overlapping windows
                for i in range(count):
                    # Calculate window position
                    window_offset_samples = i * self.step_samples
                    window_start_time = current_time - timedelta(
                        seconds=(self.window_size_seconds + window_offset_samples / self.sample_rate)
                    )
                    window_end_time = window_start_time + timedelta(seconds=self.window_size_seconds)
                    
                    # Extract window data
                    window_data, _ = self.circular_buffer.read_window_with_timestamps(
                        self.window_size_samples,
                        offset_samples=window_offset_samples
                    )
                    
                    if len(window_data) < self.window_size_samples:
                        break
                    
                    # Get quality metrics
                    window_quality_metrics = self._get_quality_metrics_for_timerange(
                        window_start_time, window_end_time
                    )
                    
                    # Create window
                    window_id = f"window_{self.next_window_id}_{i}"
                    
                    window = BufferWindow(
                        window_id=window_id,
                        audio_data=window_data,
                        start_time=window_start_time,
                        end_time=window_end_time,
                        quality_metrics=window_quality_metrics,
                        sample_rate=self.sample_rate
                    )
                    
                    # Check quality
                    if window.average_quality >= min_quality:
                        windows.append(window)
                        self._cache_window(window)
                
                # Update statistics
                self.total_windows_created += len(windows)
                if windows:
                    self.last_window_time = current_time
                
            except Exception as e:
                logger.error(f"Error getting overlapping windows: {e}")
        
        return windows
    
    def _get_quality_metrics_for_timerange(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[AudioQualityMetric]:
        """Get quality metrics within a time range."""
        metrics = []
        
        for timestamp, metric in self.quality_metrics_buffer:
            if start_time <= timestamp <= end_time:
                metrics.append(metric)
        
        return metrics
    
    def _cache_window(self, window: BufferWindow):
        """Cache a window and manage cache size."""
        self.windows_cache[window.window_id] = window
        self.window_creation_times[window.window_id] = datetime.now()
        
        # Clean up old windows if cache is full
        if len(self.windows_cache) > self.max_windows_cached:
            # Remove oldest windows
            sorted_windows = sorted(
                self.window_creation_times.items(),
                key=lambda x: x[1]
            )
            
            windows_to_remove = len(self.windows_cache) - self.max_windows_cached
            for window_id, _ in sorted_windows[:windows_to_remove]:
                self.windows_cache.pop(window_id, None)
                self.window_creation_times.pop(window_id, None)
    
    def get_cached_window(self, window_id: str) -> Optional[BufferWindow]:
        """Get a cached window by ID."""
        with self.lock:
            return self.windows_cache.get(window_id)
    
    def clear_cache(self):
        """Clear the window cache."""
        with self.lock:
            self.windows_cache.clear()
            self.window_creation_times.clear()
            logger.info("Window cache cleared")
    
    def get_buffer_utilization(self) -> float:
        """Get buffer utilization as a percentage."""
        with self.lock:
            return self.circular_buffer.get_utilization()
    
    def get_buffer_health(self) -> Dict[str, Any]:
        """Get comprehensive buffer health information."""
        with self.lock:
            utilization = self.get_buffer_utilization()
            
            # Calculate recent quality
            recent_metrics = list(self.quality_metrics_buffer)[-10:]
            if recent_metrics:
                recent_quality = np.mean([m[1].quality_score for m in recent_metrics])
            else:
                recent_quality = 0.0
            
            # Buffer status
            if utilization > 0.9:
                status = "critical"
            elif utilization > 0.7:
                status = "warning"
            else:
                status = "healthy"
            
            return {
                'status': status,
                'utilization_percent': utilization * 100,
                'buffer_size_samples': self.buffer_size_samples,
                'samples_available': self.circular_buffer.get_available_samples(),
                'recent_quality': recent_quality,
                'total_windows_created': self.total_windows_created,
                'cached_windows': len(self.windows_cache),
                'buffer_overruns': self.buffer_overruns,
                'last_window_time': self.last_window_time.isoformat() if self.last_window_time else None
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics."""
        with self.lock:
            health = self.get_buffer_health()
            
            return {
                **health,
                'configuration': {
                    'sample_rate': self.sample_rate,
                    'buffer_size_seconds': self.buffer_size_seconds,
                    'window_size_seconds': self.window_size_seconds,
                    'overlap_seconds': self.overlap_seconds,
                    'step_seconds': self.step_samples / self.sample_rate,
                    'max_windows_cached': self.max_windows_cached
                },
                'performance': {
                    'total_samples_written': self.total_samples_written,
                    'total_windows_created': self.total_windows_created,
                    'buffer_overruns': self.buffer_overruns,
                    'quality_metrics_count': len(self.quality_metrics_buffer)
                }
            }
    
    def reset_statistics(self):
        """Reset buffer statistics."""
        with self.lock:
            self.total_samples_written = 0
            self.total_windows_created = 0
            self.buffer_overruns = 0
            self.last_window_time = None
            self.quality_metrics_buffer.clear()
            logger.info("Buffer statistics reset")
    
    def cleanup(self):
        """Cleanup buffer resources."""
        with self.lock:
            self.clear_cache()
            self.circular_buffer.clear()
            self.quality_metrics_buffer.clear()
            logger.info("Buffer manager cleaned up")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass


class MultiSessionBufferManager:
    """
    Manages buffers for multiple concurrent sessions.
    
    Provides session isolation and resource management across
    multiple streaming sessions.
    """
    
    def __init__(
        self,
        max_sessions: int = 10,
        default_buffer_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-session buffer manager.
        
        Args:
            max_sessions: Maximum number of concurrent sessions
            default_buffer_config: Default configuration for new buffers
        """
        self.max_sessions = max_sessions
        self.default_buffer_config = default_buffer_config or {}
        
        # Session management
        self.session_buffers: Dict[str, BufferManager] = {}
        self.session_creation_times: Dict[str, datetime] = {}
        
        # Threading
        self.lock = threading.RLock()
        
        logger.info(f"MultiSessionBufferManager initialized for {max_sessions} sessions")
    
    def create_session_buffer(
        self, 
        session_id: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create buffer for a new session.
        
        Args:
            session_id: Unique session identifier
            config: Session-specific buffer configuration
            
        Returns:
            True if buffer created successfully
        """
        with self.lock:
            if session_id in self.session_buffers:
                logger.warning(f"Buffer for session {session_id} already exists")
                return True
            
            if len(self.session_buffers) >= self.max_sessions:
                logger.error(f"Maximum sessions ({self.max_sessions}) reached")
                return False
            
            try:
                # Merge configurations
                buffer_config = {**self.default_buffer_config}
                if config:
                    buffer_config.update(config)
                
                # Create buffer
                buffer = BufferManager(**buffer_config)
                
                self.session_buffers[session_id] = buffer
                self.session_creation_times[session_id] = datetime.now()
                
                logger.info(f"Created buffer for session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create buffer for session {session_id}: {e}")
                return False
    
    def get_session_buffer(self, session_id: str) -> Optional[BufferManager]:
        """Get buffer for a session."""
        with self.lock:
            return self.session_buffers.get(session_id)
    
    def remove_session_buffer(self, session_id: str) -> bool:
        """Remove buffer for a session."""
        with self.lock:
            if session_id not in self.session_buffers:
                return False
            
            try:
                # Cleanup buffer
                buffer = self.session_buffers[session_id]
                buffer.cleanup()
                
                # Remove from tracking
                del self.session_buffers[session_id]
                del self.session_creation_times[session_id]
                
                logger.info(f"Removed buffer for session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error removing buffer for session {session_id}: {e}")
                return False
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics for all sessions."""
        with self.lock:
            session_stats = {}
            
            for session_id, buffer in self.session_buffers.items():
                session_stats[session_id] = {
                    'creation_time': self.session_creation_times[session_id].isoformat(),
                    'buffer_stats': buffer.get_statistics()
                }
            
            return {
                'total_sessions': len(self.session_buffers),
                'max_sessions': self.max_sessions,
                'sessions': session_stats
            }
    
    def cleanup_all_sessions(self):
        """Cleanup all session buffers."""
        with self.lock:
            session_ids = list(self.session_buffers.keys())
            
            for session_id in session_ids:
                self.remove_session_buffer(session_id)
            
            logger.info("All session buffers cleaned up")