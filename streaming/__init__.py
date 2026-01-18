"""
Real-time Streaming Analysis Module

This module implements real-time streaming analysis capabilities for the
multimodal depression detection system.

Components:
- Audio Input Handler: Real-time audio capture and preprocessing
- Buffer Manager: Sliding window audio buffering
- Stream Processor: Integration with enhanced inference pipeline
- Risk Monitor: Real-time risk assessment and tracking
- Alert System: Real-time alert generation and management
- Session Manager: Multi-session coordination and management

Reference: Real-time Streaming Analysis Specification
"""

__version__ = "1.0.0"
__author__ = "Depression Detection Team"

# Core streaming components
from .audio_input_handler import AudioInputHandler, AudioDevice
from .buffer_manager import BufferManager, CircularAudioBuffer
from .stream_processor import StreamProcessor, StreamingInferenceAdapter
# from .risk_monitor import RiskMonitor, RiskAnalyzer
# from .alert_system import AlertSystem, AlertManager
from .session_manager import StreamingSessionManager, SessionManager

# Data models
from .models import (
    StreamingSession,
    SessionConfig,
    RiskDataPoint,
    AlertEvent,
    AudioQualityMetric,
    SessionStatus,
    AlertType,
    AlertSeverity,
    ProcessingMode
)

# Utilities
# from .performance_optimizer import PerformanceOptimizer
# from .resource_manager import ResourceManager

__all__ = [
    # Core components
    'AudioInputHandler',
    'AudioDevice',
    'BufferManager', 
    'CircularAudioBuffer',
    'StreamProcessor',
    'StreamingInferenceAdapter',
    # 'RiskMonitor',
    # 'RiskAnalyzer',
    # 'AlertSystem',
    # 'AlertManager',
    'StreamingSessionManager',
    'SessionManager',
    'StreamingSession',
    
    # Data models
    'RiskDataPoint',
    'AlertEvent',
    'AudioQualityMetric',
    'SessionStatus',
    'AlertType',
    'AlertSeverity',
    'ProcessingMode',
    'SessionConfig',
    
    # Utilities
    # 'PerformanceOptimizer',
    # 'ResourceManager'
]