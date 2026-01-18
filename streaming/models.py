"""
Data Models for Real-time Streaming Analysis

This module defines the core data structures used throughout the streaming system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class SessionStatus(Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    INITIALIZING = "initializing"


class AlertType(Enum):
    """Alert type enumeration."""
    HIGH_RISK = "high_risk"
    TREND_WARNING = "trend_warning"
    SYSTEM_ERROR = "system_error"
    AUDIO_QUALITY = "audio_quality"
    MODEL_FAILURE = "model_failure"


class AlertSeverity(Enum):
    """Alert severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProcessingMode(Enum):
    """Processing mode enumeration."""
    TRIMODAL = "trimodal"      # Audio + Text + Linguistic
    BIMODAL = "bimodal"        # Audio + Text only
    AUDIO_ONLY = "audio_only"  # Audio only
    BASIC = "basic"            # Simple threshold-based
    FAST = "fast"              # Optimized for speed
    FULL = "full"              # Full quality processing


@dataclass
class AudioQualityMetric:
    """Audio quality assessment metrics."""
    timestamp: datetime
    signal_to_noise_ratio: float
    volume_level: float
    clipping_detected: bool
    silence_ratio: float
    quality_score: float  # 0.0 to 1.0
    
    def is_acceptable(self, min_quality: float = 0.6) -> bool:
        """Check if audio quality is acceptable."""
        return (
            self.quality_score >= min_quality and
            not self.clipping_detected and
            self.silence_ratio < 0.8
        )


@dataclass
class RiskDataPoint:
    """Individual risk assessment data point."""
    timestamp: datetime
    risk_score: float
    confidence: float
    
    # Modality contributions
    audio_contribution: float
    text_contribution: Optional[float] = None
    linguistic_contribution: Optional[float] = None
    
    # Linguistic markers
    linguistic_markers: Optional[Dict[str, float]] = None
    
    # Processing metadata
    processing_time_ms: float = 0.0
    chunk_id: str = ""
    processing_mode: ProcessingMode = ProcessingMode.TRIMODAL
    
    # Audio quality
    audio_quality: Optional[AudioQualityMetric] = None
    
    def get_risk_level(self) -> str:
        """Get categorical risk level."""
        if self.risk_score >= 0.8:
            return "HIGH"
        elif self.risk_score >= 0.5:
            return "MODERATE"
        else:
            return "LOW"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'audio_contribution': self.audio_contribution,
            'text_contribution': self.text_contribution,
            'linguistic_contribution': self.linguistic_contribution,
            'linguistic_markers': self.linguistic_markers,
            'processing_time_ms': self.processing_time_ms,
            'chunk_id': self.chunk_id,
            'processing_mode': self.processing_mode.value,
            'risk_level': self.get_risk_level()
        }


@dataclass
class AlertEvent:
    """Alert event data structure."""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    
    # Alert content
    message: str
    risk_score: float
    trigger_condition: str
    
    # Response tracking
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    # Context
    session_id: str = ""
    patient_id: str = ""
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def acknowledge(self, user_id: str):
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'risk_score': self.risk_score,
            'trigger_condition': self.trigger_condition,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'session_id': self.session_id,
            'patient_id': self.patient_id,
            'metadata': self.metadata
        }


@dataclass
class StreamingSession:
    """Streaming session data structure."""
    session_id: str
    patient_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    audio_source: str = "default"
    language: str = "ta"
    status: SessionStatus = SessionStatus.INITIALIZING
    
    # Real-time data
    current_risk_score: float = 0.0
    risk_history: List[RiskDataPoint] = field(default_factory=list)
    alert_history: List[AlertEvent] = field(default_factory=list)
    
    # Configuration
    buffer_size_seconds: int = 30
    overlap_seconds: int = 5
    risk_threshold: float = 0.8
    processing_mode: ProcessingMode = ProcessingMode.TRIMODAL
    
    # Performance metrics
    processing_latency: List[float] = field(default_factory=list)
    audio_quality_metrics: List[AudioQualityMetric] = field(default_factory=list)
    
    # Session metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_risk_datapoint(self, datapoint: RiskDataPoint):
        """Add a new risk data point."""
        self.risk_history.append(datapoint)
        self.current_risk_score = datapoint.risk_score
        
        # Keep only recent history (last 1000 points)
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
    
    def add_alert(self, alert: AlertEvent):
        """Add a new alert."""
        alert.session_id = self.session_id
        alert.patient_id = self.patient_id
        self.alert_history.append(alert)
    
    def get_recent_risk_scores(self, minutes: int = 5) -> List[float]:
        """Get risk scores from the last N minutes."""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        recent_scores = []
        
        for datapoint in reversed(self.risk_history):
            if datapoint.timestamp.timestamp() >= cutoff_time:
                recent_scores.append(datapoint.risk_score)
            else:
                break
        
        return list(reversed(recent_scores))
    
    def get_average_risk_score(self, minutes: int = 5) -> float:
        """Get average risk score over the last N minutes."""
        recent_scores = self.get_recent_risk_scores(minutes)
        return np.mean(recent_scores) if recent_scores else 0.0
    
    def get_risk_trend(self, minutes: int = 5) -> float:
        """Get risk trend (slope) over the last N minutes."""
        recent_scores = self.get_recent_risk_scores(minutes)
        
        if len(recent_scores) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        
        return 0.0
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def get_average_processing_latency(self) -> float:
        """Get average processing latency."""
        return np.mean(self.processing_latency) if self.processing_latency else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'patient_id': self.patient_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'audio_source': self.audio_source,
            'language': self.language,
            'status': self.status.value,
            'current_risk_score': self.current_risk_score,
            'buffer_size_seconds': self.buffer_size_seconds,
            'overlap_seconds': self.overlap_seconds,
            'risk_threshold': self.risk_threshold,
            'processing_mode': self.processing_mode.value,
            'session_duration': self.get_session_duration(),
            'average_latency': self.get_average_processing_latency(),
            'total_alerts': len(self.alert_history),
            'metadata': self.metadata
        }


@dataclass
class SessionConfig:
    """Configuration for streaming sessions."""
    patient_id: str
    audio_source: str = "default"
    language: str = "ta"
    buffer_size_seconds: int = 30
    overlap_seconds: int = 5
    risk_threshold: float = 0.8
    processing_mode: ProcessingMode = ProcessingMode.TRIMODAL
    
    # Audio settings
    sample_rate: int = 16000
    chunk_duration_ms: int = 1000
    
    # Quality settings
    min_audio_quality: float = 0.6
    enable_noise_reduction: bool = True
    enable_auto_gain: bool = True
    
    # Performance settings
    max_processing_latency_ms: float = 2000.0
    enable_adaptive_processing: bool = True
    
    # Privacy settings
    enable_anonymization: bool = True
    data_retention_hours: int = 24
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'patient_id': self.patient_id,
            'audio_source': self.audio_source,
            'language': self.language,
            'buffer_size_seconds': self.buffer_size_seconds,
            'overlap_seconds': self.overlap_seconds,
            'risk_threshold': self.risk_threshold,
            'processing_mode': self.processing_mode.value,
            'sample_rate': self.sample_rate,
            'chunk_duration_ms': self.chunk_duration_ms,
            'min_audio_quality': self.min_audio_quality,
            'enable_noise_reduction': self.enable_noise_reduction,
            'enable_auto_gain': self.enable_auto_gain,
            'max_processing_latency_ms': self.max_processing_latency_ms,
            'enable_adaptive_processing': self.enable_adaptive_processing,
            'enable_anonymization': self.enable_anonymization,
            'data_retention_hours': self.data_retention_hours
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    
    # Latency metrics
    audio_capture_latency_ms: float
    processing_latency_ms: float
    end_to_end_latency_ms: float
    
    # Throughput metrics
    chunks_processed_per_second: float
    sessions_active: int
    
    # Quality metrics
    average_audio_quality: float
    average_model_confidence: float
    
    # Resource metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_memory_usage_percent: float
    
    # Error metrics
    processing_errors: int
    audio_dropouts: int
    model_failures: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'audio_capture_latency_ms': self.audio_capture_latency_ms,
            'processing_latency_ms': self.processing_latency_ms,
            'end_to_end_latency_ms': self.end_to_end_latency_ms,
            'chunks_processed_per_second': self.chunks_processed_per_second,
            'sessions_active': self.sessions_active,
            'average_audio_quality': self.average_audio_quality,
            'average_model_confidence': self.average_model_confidence,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'gpu_memory_usage_percent': self.gpu_memory_usage_percent,
            'processing_errors': self.processing_errors,
            'audio_dropouts': self.audio_dropouts,
            'model_failures': self.model_failures
        }