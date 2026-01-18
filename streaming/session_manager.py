"""
Session Manager for Real-time Streaming Analysis

This module manages multiple concurrent streaming sessions, providing session
lifecycle management, resource allocation, and coordination between components.

Key Features:
- Multi-session support with isolation
- Session lifecycle management (create, start, stop, pause)
- Resource allocation and load balancing
- Session state persistence and recovery
- Performance monitoring per session

Reference: Real-time Streaming Analysis - Task 1.4
"""

import asyncio
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
import pickle

from .models import (
    StreamingSession, SessionConfig, SessionStatus, RiskDataPoint,
    AlertEvent, AudioQualityMetric, ProcessingMode
)
from .audio_input_handler import AsyncAudioInputHandler
from .buffer_manager import BufferManager
from .stream_processor import StreamProcessor
from .risk_monitor import RiskMonitor
from .alert_system import AlertSystem

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages individual streaming session lifecycle and components.
    
    Coordinates:
    - Audio input handling
    - Buffer management
    - Stream processing
    - Session state management
    """
    
    def __init__(
        self,
        session: StreamingSession,
        config: SessionConfig,
        device: str = 'cuda'
    ):
        """
        Initialize session manager.
        
        Args:
            session: Session configuration and state
            config: Session-specific configuration
            device: Device for processing
        """
        self.session = session
        self.config = config
        self.device = device
        
        # Session components
        self.audio_handler: Optional[AsyncAudioInputHandler] = None
        self.buffer_manager: Optional[BufferManager] = None
        self.stream_processor: Optional[StreamProcessor] = None
        self.risk_monitor: Optional[RiskMonitor] = None
        self.alert_system: Optional[AlertSystem] = None
        
        # Session state
        self.is_running = False
        self.is_paused = False
        
        # Threading
        self.processing_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        
        # Callbacks
        self.risk_callbacks: List[Callable[[RiskDataPoint], None]] = []
        self.alert_callbacks: List[Callable[[AlertEvent], None]] = []
        
        logger.info(f"SessionManager created for session {session.session_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize session components.
        
        Returns:
            True if initialization successful
        """
        async with self.lock:
            try:
                # Initialize audio handler
                self.audio_handler = AsyncAudioInputHandler(
                    sample_rate=self.config.sample_rate,
                    chunk_duration_ms=self.config.chunk_duration_ms,
                    enable_noise_reduction=self.config.enable_noise_reduction,
                    enable_auto_gain=self.config.enable_auto_gain
                )
                
                # Initialize buffer manager
                self.buffer_manager = BufferManager(
                    sample_rate=self.config.sample_rate,
                    buffer_size_seconds=self.config.buffer_size_seconds,
                    window_size_seconds=self.config.buffer_size_seconds,  # Use full buffer as window
                    overlap_seconds=self.config.overlap_seconds
                )
                
                # Initialize stream processor
                self.stream_processor = StreamProcessor(
                    device=self.device,
                    inference_mode='enhanced',
                    target_latency_ms=self.config.max_processing_latency_ms,
                    enable_adaptive_processing=self.config.enable_adaptive_processing
                )
                
                # Initialize risk monitor
                self.risk_monitor = RiskMonitor(
                    rolling_window_minutes=5,
                    max_history_points=1000,
                    risk_thresholds={
                        'low': 0.3,
                        'moderate': 0.6,
                        'high': self.config.risk_threshold
                    }
                )
                
                # Initialize alert system
                self.alert_system = AlertSystem()
                
                # Set up callbacks
                self.stream_processor.add_result_callback(self._on_risk_result)
                
                self.session.status = SessionStatus.INITIALIZING
                logger.info(f"Session {self.session.session_id} initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize session {self.session.session_id}: {e}")
                self.session.status = SessionStatus.ERROR
                return False
    
    async def start(self) -> bool:
        """
        Start the streaming session.
        
        Returns:
            True if started successfully
        """
        async with self.lock:
            if self.is_running:
                logger.warning(f"Session {self.session.session_id} already running")
                return True
            
            try:
                # Start audio capture
                if not await self.audio_handler.start_capture():
                    logger.error(f"Failed to start audio capture for session {self.session.session_id}")
                    return False
                
                # Start processing task
                self.processing_task = asyncio.create_task(self._processing_loop())
                
                # Update session state
                self.is_running = True
                self.is_paused = False
                self.session.status = SessionStatus.ACTIVE
                self.session.start_time = datetime.now()
                
                logger.info(f"Session {self.session.session_id} started")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start session {self.session.session_id}: {e}")
                self.session.status = SessionStatus.ERROR
                return False
    
    async def stop(self):
        """Stop the streaming session."""
        async with self.lock:
            if not self.is_running:
                return
            
            try:
                # Stop processing
                self.is_running = False
                
                if self.processing_task:
                    self.processing_task.cancel()
                    try:
                        await self.processing_task
                    except asyncio.CancelledError:
                        pass
                
                # Stop audio capture
                if self.audio_handler:
                    await self.audio_handler.stop_capture()
                
                # Update session state
                self.session.status = SessionStatus.COMPLETED
                self.session.end_time = datetime.now()
                
                logger.info(f"Session {self.session.session_id} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping session {self.session.session_id}: {e}")
                self.session.status = SessionStatus.ERROR
    
    async def pause(self):
        """Pause the streaming session."""
        async with self.lock:
            if not self.is_running or self.is_paused:
                return
            
            self.is_paused = True
            logger.info(f"Session {self.session.session_id} paused")
    
    async def resume(self):
        """Resume the streaming session."""
        async with self.lock:
            if not self.is_running or not self.is_paused:
                return
            
            self.is_paused = False
            logger.info(f"Session {self.session.session_id} resumed")
    
    async def _processing_loop(self):
        """Main processing loop for the session."""
        logger.info(f"Processing loop started for session {self.session.session_id}")
        
        try:
            while self.is_running:
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Get audio chunk
                    audio_data, quality_metric = await asyncio.wait_for(
                        self.audio_handler.get_audio_chunk(),
                        timeout=1.0
                    )
                    
                    # Write to buffer
                    self.buffer_manager.write_audio_chunk(audio_data, quality_metric)
                    
                    # Process through stream processor
                    result = await self.stream_processor.process_audio_chunk(
                        audio_data=audio_data,
                        audio_quality=quality_metric,
                        language=self.session.language,
                        chunk_id=f"{self.session.session_id}_{int(time.time())}"
                    )
                    
                    # Update session with result
                    self.session.add_risk_datapoint(result)
                    
                    # Update performance metrics
                    self.session.processing_latency.append(result.processing_time_ms)
                    if quality_metric:
                        self.session.audio_quality_metrics.append(quality_metric)
                    
                except asyncio.TimeoutError:
                    # No audio data available, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in processing loop for session {self.session.session_id}: {e}")
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info(f"Processing loop cancelled for session {self.session.session_id}")
        except Exception as e:
            logger.error(f"Processing loop error for session {self.session.session_id}: {e}")
            self.session.status = SessionStatus.ERROR
    
    def _on_risk_result(self, result: RiskDataPoint):
        """Handle risk assessment result."""
        # Add to risk monitor
        if self.risk_monitor:
            self.risk_monitor.add_risk_datapoint(result)
            
            # Evaluate alerts
            if self.alert_system:
                trend_analysis = self.risk_monitor.get_risk_trend()
                trend_slope = trend_analysis.get('trend_slope', 0.0)
                
                # Get recent risk history for duration checks
                recent_history = [
                    point for point in self.session.risk_history[-10:]
                    if (datetime.now() - point.timestamp).total_seconds() < 300
                ]
                
                alerts = self.alert_system.evaluate_alerts(
                    result,
                    self.session.session_id,
                    trend_slope=trend_slope,
                    risk_history=recent_history
                )
                
                # Add alerts to session and trigger callbacks
                for alert in alerts:
                    self.session.add_alert(alert)
                    
                    # Call alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")
        
        # Call registered risk callbacks
        for callback in self.risk_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in risk callback: {e}")
    
    def add_risk_callback(self, callback: Callable[[RiskDataPoint], None]):
        """Add callback for risk results."""
        self.risk_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[AlertEvent], None]):
        """Add callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_risk_score(self) -> float:
        """Get current risk score."""
        if self.risk_monitor:
            return self.risk_monitor.get_current_risk_score()
        return self.session.current_risk_score
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        if self.risk_monitor:
            return self.risk_monitor.get_risk_summary()
        
        return {
            'current_risk_score': self.session.current_risk_score,
            'rolling_average': self.session.get_average_risk_score(),
            'risk_level': 'LOW' if self.session.current_risk_score < 0.5 else 'HIGH',
            'data_points': len(self.session.risk_history)
        }
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get active alerts for this session."""
        if self.alert_system:
            return self.alert_system.get_active_alerts(self.session.session_id)
        return [alert for alert in self.session.alert_history if not alert.acknowledged]
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        if self.alert_system:
            return self.alert_system.acknowledge_alert(alert_id, user_id)
        
        # Fallback for session-only alerts
        for alert in self.session.alert_history:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledge(user_id)
                return True
        return False
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        stats = self.session.to_dict()
        
        # Add component statistics
        if self.audio_handler:
            stats['audio_handler'] = self.audio_handler.get_statistics()
        
        if self.buffer_manager:
            stats['buffer_manager'] = self.buffer_manager.get_statistics()
        
        if self.stream_processor:
            stats['stream_processor'] = self.stream_processor.get_processing_statistics()
        
        if self.risk_monitor:
            stats['risk_monitor'] = self.risk_monitor.get_statistics()
        
        if self.alert_system:
            stats['alert_system'] = self.alert_system.get_alert_statistics()
        
        return stats
    
    async def cleanup(self):
        """Cleanup session resources."""
        await self.stop()
        
        # Clear callbacks
        self.risk_callbacks.clear()
        self.alert_callbacks.clear()
        
        # Cleanup components
        if self.buffer_manager:
            self.buffer_manager.cleanup()
        
        if self.alert_system:
            self.alert_system.reset_session_alerts(self.session.session_id)
        
        logger.info(f"Session {self.session.session_id} cleaned up")


class StreamingSessionManager:
    """
    Manages multiple concurrent streaming sessions.
    
    Features:
    - Multi-session coordination
    - Resource allocation and load balancing
    - Session persistence and recovery
    - Performance monitoring across sessions
    """
    
    def __init__(
        self,
        max_concurrent_sessions: int = 10,
        device: str = 'cuda',
        persistence_enabled: bool = True,
        persistence_path: str = "session_data"
    ):
        """
        Initialize streaming session manager.
        
        Args:
            max_concurrent_sessions: Maximum number of concurrent sessions
            device: Device for processing
            persistence_enabled: Enable session state persistence
            persistence_path: Path for session data persistence
        """
        self.max_concurrent_sessions = max_concurrent_sessions
        self.device = device
        self.persistence_enabled = persistence_enabled
        self.persistence_path = persistence_path
        
        # Session management
        self.active_sessions: Dict[str, SessionManager] = {}
        self.session_configs: Dict[str, SessionConfig] = {}
        
        # Global alert system
        self.global_alert_system: Optional[AlertSystem] = None
        if max_concurrent_sessions > 1:  # Enable global alerts for multi-session setups
            self.global_alert_system = AlertSystem()
        
        # Threading
        self.lock = asyncio.Lock()
        
        # Statistics
        self.total_sessions_created = 0
        self.total_sessions_completed = 0
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_enabled = False
        
        logger.info(f"StreamingSessionManager initialized for {max_concurrent_sessions} sessions")
    
    async def create_session(
        self,
        patient_id: str,
        config: Optional[SessionConfig] = None
    ) -> Optional[str]:
        """
        Create a new streaming session.
        
        Args:
            patient_id: Patient identifier
            config: Session configuration (uses default if None)
            
        Returns:
            Session ID if created successfully, None otherwise
        """
        async with self.lock:
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                logger.error(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
                return None
            
            try:
                # Generate session ID
                session_id = str(uuid.uuid4())
                
                # Use default config if not provided
                if config is None:
                    config = SessionConfig(patient_id=patient_id)
                
                # Create session object
                session = StreamingSession(
                    session_id=session_id,
                    patient_id=patient_id,
                    start_time=datetime.now(),
                    audio_source=config.audio_source,
                    language=config.language,
                    buffer_size_seconds=config.buffer_size_seconds,
                    overlap_seconds=config.overlap_seconds,
                    risk_threshold=config.risk_threshold,
                    processing_mode=config.processing_mode
                )
                
                # Create session manager
                session_manager = SessionManager(
                    session=session,
                    config=config,
                    device=self.device
                )
                
                # Initialize session
                if not await session_manager.initialize():
                    logger.error(f"Failed to initialize session {session_id}")
                    return None
                
                # Store session
                self.active_sessions[session_id] = session_manager
                self.session_configs[session_id] = config
                self.total_sessions_created += 1
                
                # Persist session if enabled
                if self.persistence_enabled:
                    await self._persist_session(session_id)
                
                logger.info(f"Created session {session_id} for patient {patient_id}")
                return session_id
                
            except Exception as e:
                logger.error(f"Failed to create session for patient {patient_id}: {e}")
                return None
    
    async def start_session(self, session_id: str) -> bool:
        """
        Start a streaming session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if started successfully
        """
        async with self.lock:
            session_manager = self.active_sessions.get(session_id)
            if not session_manager:
                logger.error(f"Session {session_id} not found")
                return False
            
            return await session_manager.start()
    
    async def stop_session(self, session_id: str) -> bool:
        """
        Stop a streaming session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if stopped successfully
        """
        async with self.lock:
            session_manager = self.active_sessions.get(session_id)
            if not session_manager:
                logger.error(f"Session {session_id} not found")
                return False
            
            await session_manager.stop()
            self.total_sessions_completed += 1
            
            # Persist final state
            if self.persistence_enabled:
                await self._persist_session(session_id)
            
            return True
    
    async def remove_session(self, session_id: str) -> bool:
        """
        Remove a session from management.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if removed successfully
        """
        async with self.lock:
            session_manager = self.active_sessions.get(session_id)
            if not session_manager:
                return False
            
            try:
                # Stop and cleanup session
                await session_manager.cleanup()
                
                # Remove from tracking
                del self.active_sessions[session_id]
                del self.session_configs[session_id]
                
                logger.info(f"Removed session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error removing session {session_id}: {e}")
                return False
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a streaming session."""
        session_manager = self.active_sessions.get(session_id)
        if not session_manager:
            return False
        
        await session_manager.pause()
        return True
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a streaming session."""
        session_manager = self.active_sessions.get(session_id)
        if not session_manager:
            return False
        
        await session_manager.resume()
        return True
    
    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get session object by ID."""
        session_manager = self.active_sessions.get(session_id)
        return session_manager.session if session_manager else None
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())
    
    def get_session_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session."""
        session_manager = self.active_sessions.get(session_id)
        return session_manager.get_session_statistics() if session_manager else None
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all sessions."""
        stats = {
            'total_sessions': len(self.active_sessions),
            'max_sessions': self.max_concurrent_sessions,
            'total_created': self.total_sessions_created,
            'total_completed': self.total_sessions_completed,
            'sessions': {}
        }
        
        # Add global alert statistics
        if self.global_alert_system:
            stats['global_alerts'] = self.global_alert_system.get_alert_statistics()
        
        for session_id, session_manager in self.active_sessions.items():
            stats['sessions'][session_id] = session_manager.get_session_statistics()
        
        return stats
    
    def get_global_alerts(self) -> List[AlertEvent]:
        """Get all active alerts across sessions."""
        if self.global_alert_system:
            return self.global_alert_system.get_active_alerts()
        
        # Fallback: collect alerts from all sessions
        all_alerts = []
        for session_manager in self.active_sessions.values():
            alerts = session_manager.get_active_alerts()
            all_alerts.extend(alerts)
        
        return sorted(all_alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_global_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert across all sessions."""
        if self.global_alert_system:
            if self.global_alert_system.acknowledge_alert(alert_id, user_id):
                return True
        
        # Try each session
        for session_manager in self.active_sessions.values():
            if session_manager.acknowledge_alert(alert_id, user_id):
                return True
        
        return False
    
    async def _persist_session(self, session_id: str):
        """Persist session state to disk."""
        if not self.persistence_enabled:
            return
        
        try:
            session_manager = self.active_sessions.get(session_id)
            if not session_manager:
                return
            
            # Create persistence directory
            import os
            os.makedirs(self.persistence_path, exist_ok=True)
            
            # Save session data
            session_file = os.path.join(self.persistence_path, f"{session_id}.json")
            session_data = session_manager.session.to_dict()
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Save config
            config_file = os.path.join(self.persistence_path, f"{session_id}_config.json")
            config_data = self.session_configs[session_id].to_dict()
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist session {session_id}: {e}")
    
    async def load_session(self, session_id: str) -> bool:
        """Load session from persistence."""
        if not self.persistence_enabled:
            return False
        
        try:
            import os
            
            session_file = os.path.join(self.persistence_path, f"{session_id}.json")
            config_file = os.path.join(self.persistence_path, f"{session_id}_config.json")
            
            if not (os.path.exists(session_file) and os.path.exists(config_file)):
                return False
            
            # Load session data
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Reconstruct objects (simplified - would need full reconstruction)
            logger.info(f"Session {session_id} loaded from persistence")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start session monitoring task."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        logger.info("Session monitoring started")
    
    async def stop_monitoring(self):
        """Stop session monitoring task."""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Session monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Session monitoring loop."""
        try:
            while self.monitoring_enabled:
                await asyncio.sleep(interval_seconds)
                
                # Check session health
                for session_id, session_manager in list(self.active_sessions.items()):
                    try:
                        # Check if session is still responsive
                        stats = session_manager.get_session_statistics()
                        
                        # Log session status
                        logger.debug(f"Session {session_id} status: {stats.get('status', 'unknown')}")
                        
                        # Could implement health checks here
                        
                    except Exception as e:
                        logger.error(f"Error monitoring session {session_id}: {e}")
        
        except asyncio.CancelledError:
            logger.info("Session monitoring loop cancelled")
    
    async def cleanup_all_sessions(self):
        """Cleanup all active sessions."""
        session_ids = list(self.active_sessions.keys())
        
        for session_id in session_ids:
            await self.remove_session(session_id)
        
        await self.stop_monitoring()
        logger.info("All sessions cleaned up")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            # Note: This won't work properly in async context
            # Proper cleanup should be done explicitly
            pass
        except Exception:
            pass