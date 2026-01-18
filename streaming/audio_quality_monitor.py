"""
Audio Quality Monitor for Real-time Streaming

This module provides comprehensive audio quality assessment for streaming audio,
including signal-to-noise ratio, clipping detection, and overall quality scoring.

Reference: Real-time Streaming Analysis - Task 1.1
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

from .models import AudioQualityMetric

logger = logging.getLogger(__name__)


class AudioQualityMonitor:
    """
    Comprehensive audio quality assessment for streaming audio.
    
    Monitors:
    - Signal-to-noise ratio (SNR)
    - Volume levels and dynamic range
    - Clipping detection
    - Silence ratio
    - Overall quality scoring
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_floor_db: float = -60.0,
        clipping_threshold: float = 0.95,
        silence_threshold: float = 0.001
    ):
        """
        Initialize audio quality monitor.
        
        Args:
            sample_rate: Audio sampling rate
            noise_floor_db: Noise floor threshold in dB
            clipping_threshold: Clipping detection threshold (0.0-1.0)
            silence_threshold: Silence detection threshold
        """
        self.sample_rate = sample_rate
        self.noise_floor_db = noise_floor_db
        self.clipping_threshold = clipping_threshold
        self.silence_threshold = silence_threshold
        
        # Quality assessment history
        self.quality_history = []
        self.max_history_length = 100
        
        # Noise estimation
        self.noise_estimate = 0.001
        self.noise_adaptation_rate = 0.01
        
        logger.info(f"AudioQualityMonitor initialized for {sample_rate}Hz audio")
    
    def assess_quality(
        self, 
        audio_data: np.ndarray, 
        timestamp: Optional[float] = None
    ) -> AudioQualityMetric:
        """
        Assess audio quality for a single chunk.
        
        Args:
            audio_data: Audio data array
            timestamp: Timestamp of the audio chunk
            
        Returns:
            AudioQualityMetric with quality assessment
        """
        if timestamp is None:
            timestamp = time.time()
        
        dt_timestamp = datetime.fromtimestamp(timestamp)
        
        # Handle empty or invalid audio
        if len(audio_data) == 0:
            return AudioQualityMetric(
                timestamp=dt_timestamp,
                signal_to_noise_ratio=0.0,
                volume_level=0.0,
                clipping_detected=False,
                silence_ratio=1.0,
                quality_score=0.0
            )
        
        # Calculate basic metrics
        snr = self._calculate_snr(audio_data)
        volume_level = self._calculate_volume_level(audio_data)
        clipping_detected = self._detect_clipping(audio_data)
        silence_ratio = self._calculate_silence_ratio(audio_data)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            snr, volume_level, clipping_detected, silence_ratio
        )
        
        # Create quality metric
        metric = AudioQualityMetric(
            timestamp=dt_timestamp,
            signal_to_noise_ratio=snr,
            volume_level=volume_level,
            clipping_detected=clipping_detected,
            silence_ratio=silence_ratio,
            quality_score=quality_score
        )
        
        # Update history
        self._update_history(metric)
        
        return metric
    
    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        try:
            # Calculate RMS of the signal
            signal_rms = np.sqrt(np.mean(audio_data ** 2))
            
            if signal_rms == 0:
                return 0.0
            
            # Update noise estimate (use minimum RMS as noise estimate)
            current_rms = signal_rms
            if current_rms < self.noise_estimate or self.noise_estimate == 0.001:
                self.noise_estimate = (
                    (1 - self.noise_adaptation_rate) * self.noise_estimate +
                    self.noise_adaptation_rate * current_rms
                )
            
            # Calculate SNR in dB
            if self.noise_estimate > 0:
                snr_db = 20 * np.log10(signal_rms / self.noise_estimate)
                return max(0.0, snr_db)  # Ensure non-negative
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_level(self, audio_data: np.ndarray) -> float:
        """Calculate volume level (RMS amplitude)."""
        try:
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Convert to dB scale
            if rms > 0:
                db_level = 20 * np.log10(rms)
                # Normalize to 0-1 range (assuming -60dB to 0dB range)
                normalized = (db_level - self.noise_floor_db) / (-self.noise_floor_db)
                return np.clip(normalized, 0.0, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Volume level calculation failed: {e}")
            return 0.0
    
    def _detect_clipping(self, audio_data: np.ndarray) -> bool:
        """Detect audio clipping."""
        try:
            # Check for samples near the clipping threshold
            clipped_samples = np.sum(np.abs(audio_data) >= self.clipping_threshold)
            clipping_ratio = clipped_samples / len(audio_data)
            
            # Consider clipping if more than 1% of samples are clipped
            return clipping_ratio > 0.01
            
        except Exception as e:
            logger.warning(f"Clipping detection failed: {e}")
            return False
    
    def _calculate_silence_ratio(self, audio_data: np.ndarray) -> float:
        """Calculate ratio of silent samples."""
        try:
            silent_samples = np.sum(np.abs(audio_data) < self.silence_threshold)
            return silent_samples / len(audio_data)
            
        except Exception as e:
            logger.warning(f"Silence ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_quality_score(
        self,
        snr: float,
        volume_level: float,
        clipping_detected: bool,
        silence_ratio: float
    ) -> float:
        """
        Calculate overall quality score (0.0 to 1.0).
        
        Quality factors:
        - SNR: Higher is better (up to 40dB)
        - Volume: Optimal range 0.3-0.8
        - Clipping: Heavily penalized
        - Silence: Moderate penalty for high silence
        """
        try:
            # SNR component (0-1, saturates at 40dB)
            snr_score = min(1.0, snr / 40.0)
            
            # Volume component (optimal range 0.3-0.8)
            if 0.3 <= volume_level <= 0.8:
                volume_score = 1.0
            elif volume_level < 0.3:
                volume_score = volume_level / 0.3
            else:  # volume_level > 0.8
                volume_score = max(0.0, (1.0 - volume_level) / 0.2)
            
            # Clipping penalty
            clipping_penalty = 0.5 if clipping_detected else 0.0
            
            # Silence penalty (high silence is bad for speech)
            silence_penalty = min(0.3, silence_ratio * 0.5)
            
            # Combine scores
            quality_score = (
                0.4 * snr_score +
                0.3 * volume_score +
                0.2 * (1.0 - silence_penalty) +
                0.1  # Base score
            )
            
            # Apply clipping penalty
            quality_score = max(0.0, quality_score - clipping_penalty)
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _update_history(self, metric: AudioQualityMetric):
        """Update quality history."""
        self.quality_history.append(metric)
        
        # Limit history size
        if len(self.quality_history) > self.max_history_length:
            self.quality_history = self.quality_history[-self.max_history_length:]
    
    def get_average_quality(self, minutes: int = 5) -> float:
        """Get average quality over the last N minutes."""
        if not self.quality_history:
            return 0.0
        
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            m for m in self.quality_history
            if m.timestamp.timestamp() >= cutoff_time
        ]
        
        if not recent_metrics:
            return 0.0
        
        return np.mean([m.quality_score for m in recent_metrics])
    
    def get_quality_trend(self, minutes: int = 5) -> float:
        """Get quality trend (slope) over the last N minutes."""
        if len(self.quality_history) < 2:
            return 0.0
        
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            m for m in self.quality_history
            if m.timestamp.timestamp() >= cutoff_time
        ]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Calculate linear trend
        x = np.arange(len(recent_metrics))
        y = np.array([m.quality_score for m in recent_metrics])
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except Exception:
            return 0.0
    
    def is_quality_acceptable(self, min_quality: float = 0.6) -> bool:
        """Check if current quality is acceptable."""
        if not self.quality_history:
            return False
        
        recent_quality = self.get_average_quality(minutes=1)
        return recent_quality >= min_quality
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quality statistics."""
        if not self.quality_history:
            return {
                'average_quality': 0.0,
                'quality_trend': 0.0,
                'average_snr': 0.0,
                'average_volume': 0.0,
                'clipping_incidents': 0,
                'total_assessments': 0
            }
        
        recent_metrics = self.quality_history[-50:]  # Last 50 assessments
        
        return {
            'average_quality': np.mean([m.quality_score for m in recent_metrics]),
            'quality_trend': self.get_quality_trend(),
            'average_snr': np.mean([m.signal_to_noise_ratio for m in recent_metrics]),
            'average_volume': np.mean([m.volume_level for m in recent_metrics]),
            'clipping_incidents': sum(1 for m in recent_metrics if m.clipping_detected),
            'average_silence_ratio': np.mean([m.silence_ratio for m in recent_metrics]),
            'total_assessments': len(self.quality_history),
            'noise_estimate': self.noise_estimate
        }
    
    def reset_statistics(self):
        """Reset quality statistics."""
        self.quality_history.clear()
        self.noise_estimate = 0.001
        logger.info("Audio quality statistics reset")


class AdvancedAudioQualityMonitor(AudioQualityMonitor):
    """
    Advanced audio quality monitor with spectral analysis.
    
    Requires librosa for advanced features.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available - advanced features disabled")
    
    def assess_quality(
        self, 
        audio_data: np.ndarray, 
        timestamp: Optional[float] = None
    ) -> AudioQualityMetric:
        """Enhanced quality assessment with spectral analysis."""
        # Get basic quality assessment
        metric = super().assess_quality(audio_data, timestamp)
        
        if LIBROSA_AVAILABLE and len(audio_data) > 0:
            try:
                # Add spectral features
                spectral_features = self._analyze_spectral_quality(audio_data)
                
                # Adjust quality score based on spectral features
                spectral_quality = self._calculate_spectral_quality_score(spectral_features)
                
                # Combine with basic quality score
                metric.quality_score = (
                    0.7 * metric.quality_score +
                    0.3 * spectral_quality
                )
                
                # Store spectral features in metadata
                if not hasattr(metric, 'metadata'):
                    metric.metadata = {}
                metric.metadata.update(spectral_features)
                
            except Exception as e:
                logger.warning(f"Spectral analysis failed: {e}")
        
        return metric
    
    def _analyze_spectral_quality(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze spectral quality features."""
        try:
            # Compute spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate))
            }
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            return {}
    
    def _calculate_spectral_quality_score(self, spectral_features: Dict[str, float]) -> float:
        """Calculate quality score from spectral features."""
        if not spectral_features:
            return 0.5  # Neutral score if no features
        
        try:
            # Normalize spectral features to quality indicators
            centroid = spectral_features.get('spectral_centroid_mean', 0)
            bandwidth = spectral_features.get('spectral_bandwidth_mean', 0)
            rolloff = spectral_features.get('spectral_rolloff_mean', 0)
            zcr = spectral_features.get('zero_crossing_rate_mean', 0)
            
            # Quality heuristics for speech
            # Good speech typically has:
            # - Centroid around 1000-3000 Hz
            # - Moderate bandwidth
            # - Reasonable rolloff
            # - Moderate zero crossing rate
            
            centroid_score = 1.0
            if centroid > 0:
                # Optimal range 1000-3000 Hz
                if 1000 <= centroid <= 3000:
                    centroid_score = 1.0
                elif centroid < 1000:
                    centroid_score = centroid / 1000
                else:
                    centroid_score = max(0.0, 1.0 - (centroid - 3000) / 5000)
            
            # Bandwidth score (moderate bandwidth is good)
            bandwidth_score = 1.0
            if bandwidth > 0:
                bandwidth_score = min(1.0, 2000 / bandwidth) if bandwidth > 2000 else 1.0
            
            # Zero crossing rate (moderate is good for speech)
            zcr_score = 1.0
            if zcr > 0:
                if 0.05 <= zcr <= 0.15:
                    zcr_score = 1.0
                elif zcr < 0.05:
                    zcr_score = zcr / 0.05
                else:
                    zcr_score = max(0.0, 1.0 - (zcr - 0.15) / 0.1)
            
            # Combine scores
            spectral_quality = (
                0.4 * centroid_score +
                0.3 * bandwidth_score +
                0.3 * zcr_score
            )
            
            return np.clip(spectral_quality, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Spectral quality calculation failed: {e}")
            return 0.5