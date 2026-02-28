"""
Session Manager for NLP Web Interface

This module manages user sessions for the NLP web interface, providing session
lifecycle management, data persistence, and resource cleanup.

Key Features:
- Unique session ID generation and state initialization
- Session data persistence and history management
- Resource cleanup with configurable timeouts
- Session export functionality

Reference: NLP Web Interface - Task 10.1, 10.2
"""

import asyncio
import json
import logging
import os
import pickle
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import threading
from concurrent.futures import ThreadPoolExecutor

from .models import SessionData, SessionStatus, AnalysisResult

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages NLP web interface sessions with lifecycle management and persistence.
    
    Features:
    - Session creation with unique IDs
    - State persistence and recovery
    - Automatic resource cleanup
    - Session history and export
    """
    
    def __init__(
        self,
        session_timeout_minutes: int = 60,
        max_sessions: int = 1000,
        persistence_path: str = "session_data",
        cleanup_interval_minutes: int = 10,
        enable_persistence: bool = True
    ):
        """
        Initialize session manager.
        
        Args:
            session_timeout_minutes: Session timeout in minutes
            max_sessions: Maximum number of concurrent sessions
            persistence_path: Directory for session data persistence
            cleanup_interval_minutes: Cleanup task interval in minutes
            enable_persistence: Enable session data persistence
        """
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions = max_sessions
        self.persistence_path = Path(persistence_path)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        self.enable_persistence = enable_persistence
        
        # Session storage
        self.active_sessions: Dict[str, SessionData] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
        # Global lock for session management
        self.global_lock = asyncio.Lock()
        
        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_running = False
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="session_io")
        
        # Statistics
        self.total_sessions_created = 0
        self.total_sessions_expired = 0
        self.total_sessions_completed = 0
        
        # Create persistence directory
        if self.enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"SessionManager initialized with {session_timeout_minutes}min timeout")
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session with unique ID.
        
        Args:
            user_id: Optional user identifier
            metadata: Optional session metadata
            
        Returns:
            Unique session ID
            
        Raises:
            RuntimeError: If maximum sessions exceeded
        """
        async with self.global_lock:
            # Check session limit
            if len(self.active_sessions) >= self.max_sessions:
                # Try to cleanup expired sessions first
                await self._cleanup_expired_sessions()
                
                if len(self.active_sessions) >= self.max_sessions:
                    raise RuntimeError(f"Maximum sessions ({self.max_sessions}) exceeded")
            
            # Generate unique session ID
            session_id = self._generate_session_id()
            
            # Create session data
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=SessionStatus.ACTIVE,
                metadata=metadata or {}
            )
            
            # Store session
            self.active_sessions[session_id] = session_data
            self.session_locks[session_id] = asyncio.Lock()
            self.total_sessions_created += 1
            
            # Persist session if enabled
            if self.enable_persistence:
                await self._persist_session(session_id)
            
            # Start cleanup task if not running
            if not self.cleanup_running:
                await self._start_cleanup_task()
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if found and active, None otherwise
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        # Check if session is expired
        if self._is_session_expired(session):
            await self.expire_session(session_id)
            return None
        
        # Update last access time
        async with self.session_locks[session_id]:
            session.updated_at = datetime.now()
            
            # Persist updated session
            if self.enable_persistence:
                await self._persist_session(session_id)
        
        return session
    
    async def update_session(
        self,
        session_id: str,
        **updates
    ) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            **updates: Fields to update
            
        Returns:
            True if updated successfully, False if session not found
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        async with self.session_locks[session_id]:
            # Update fields
            for field, value in updates.items():
                if hasattr(session, field):
                    setattr(session, field, value)
            
            # Update timestamp
            session.updated_at = datetime.now()
            
            # Persist updated session
            if self.enable_persistence:
                await self._persist_session(session_id)
        
        logger.debug(f"Updated session {session_id}")
        return True
    
    async def add_analysis_result(
        self,
        session_id: str,
        analysis_result: AnalysisResult
    ) -> bool:
        """
        Add analysis result to session.
        
        Args:
            session_id: Session identifier
            analysis_result: Analysis result to add
            
        Returns:
            True if added successfully, False if session not found
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        async with self.session_locks[session_id]:
            session.analysis_results.append(analysis_result)
            session.updated_at = datetime.now()
            
            # Persist updated session
            if self.enable_persistence:
                await self._persist_session(session_id)
        
        logger.debug(f"Added analysis result to session {session_id}")
        return True
    
    async def complete_session(self, session_id: str) -> bool:
        """
        Mark session as completed.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if completed successfully, False if session not found
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        async with self.session_locks[session_id]:
            session.status = SessionStatus.COMPLETED
            session.updated_at = datetime.now()
            
            # Persist final state
            if self.enable_persistence:
                await self._persist_session(session_id)
        
        self.total_sessions_completed += 1
        logger.info(f"Completed session {session_id}")
        return True
    
    async def expire_session(self, session_id: str) -> bool:
        """
        Expire a session and clean up resources.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if expired successfully, False if session not found
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        async with self.session_locks[session_id]:
            session.status = SessionStatus.TIMEOUT
            session.updated_at = datetime.now()
            
            # Persist final state
            if self.enable_persistence:
                await self._persist_session(session_id)
        
        # Remove from active sessions
        async with self.global_lock:
            del self.active_sessions[session_id]
            del self.session_locks[session_id]
        
        self.total_sessions_expired += 1
        logger.info(f"Expired session {session_id}")
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted successfully, False if session not found
        """
        # Remove from active sessions
        async with self.global_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.session_locks:
                del self.session_locks[session_id]
        
        # Remove persisted data
        if self.enable_persistence:
            await self._delete_persisted_session(session_id)
        
        logger.info(f"Deleted session {session_id}")
        return True
    
    async def get_session_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        include_completed: bool = True,
        include_expired: bool = False
    ) -> List[SessionData]:
        """
        Get session history.
        
        Args:
            user_id: Filter by user ID (optional)
            limit: Maximum number of sessions to return
            include_completed: Include completed sessions
            include_expired: Include expired sessions
            
        Returns:
            List of session data sorted by creation time (newest first)
        """
        sessions = []
        
        # Get active sessions
        for session in self.active_sessions.values():
            if user_id and session.user_id != user_id:
                continue
            sessions.append(session)
        
        # Get persisted sessions if enabled
        if self.enable_persistence:
            persisted_sessions = await self._load_persisted_sessions(
                user_id=user_id,
                include_completed=include_completed,
                include_expired=include_expired
            )
            
            # Avoid duplicates (active sessions are already included)
            active_ids = {s.session_id for s in sessions}
            for session in persisted_sessions:
                if session.session_id not in active_ids:
                    sessions.append(session)
        
        # Sort by creation time (newest first) and limit
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]
    
    async def export_session(
        self,
        session_id: str,
        format: str = "json"
    ) -> Optional[Union[str, bytes]]:
        """
        Export session data in specified format.
        
        Args:
            session_id: Session identifier
            format: Export format ("json", "csv", "pickle")
            
        Returns:
            Exported data as string or bytes, None if session not found
        """
        session = await self.get_session(session_id)
        if not session:
            # Try to load from persistence
            if self.enable_persistence:
                session = await self._load_session_from_disk(session_id)
            
            if not session:
                return None
        
        if format.lower() == "json":
            return await self._export_session_json(session)
        elif format.lower() == "csv":
            return await self._export_session_csv(session)
        elif format.lower() == "pickle":
            return await self._export_session_pickle(session)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get session manager statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        active_count = len(self.active_sessions)
        
        # Calculate status distribution
        status_counts = {}
        for session in self.active_sessions.values():
            status = session.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate average session duration for completed sessions
        completed_sessions = [
            s for s in self.active_sessions.values()
            if s.status == SessionStatus.COMPLETED
        ]
        
        avg_duration = 0.0
        if completed_sessions:
            durations = [
                (s.updated_at - s.created_at).total_seconds()
                for s in completed_sessions
            ]
            avg_duration = sum(durations) / len(durations)
        
        return {
            "active_sessions": active_count,
            "max_sessions": self.max_sessions,
            "total_created": self.total_sessions_created,
            "total_completed": self.total_sessions_completed,
            "total_expired": self.total_sessions_expired,
            "status_distribution": status_counts,
            "average_duration_seconds": avg_duration,
            "cleanup_running": self.cleanup_running,
            "persistence_enabled": self.enable_persistence
        }
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = int(time.time() * 1000)  # milliseconds
        random_part = str(uuid.uuid4()).replace('-', '')[:8]
        return f"sess_{timestamp}_{random_part}"
    
    def _is_session_expired(self, session: SessionData) -> bool:
        """Check if session is expired."""
        if session.status in [SessionStatus.COMPLETED, SessionStatus.ERROR, SessionStatus.TIMEOUT]:
            return False  # Don't auto-expire these states
        
        time_since_update = datetime.now() - session.updated_at
        return time_since_update > self.session_timeout
    
    async def _start_cleanup_task(self):
        """Start the cleanup task."""
        if self.cleanup_running:
            return
        
        self.cleanup_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Started session cleanup task")
    
    async def _cleanup_loop(self):
        """Main cleanup loop."""
        try:
            while self.cleanup_running:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                await self._cleanup_expired_sessions()
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in session cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        expired_sessions = []
        
        # Find expired sessions
        for session_id, session in list(self.active_sessions.items()):
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        # Expire sessions
        for session_id in expired_sessions:
            await self.expire_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _persist_session(self, session_id: str):
        """Persist session to disk."""
        if not self.enable_persistence:
            return
        
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        try:
            session_file = self.persistence_path / f"{session_id}.json"
            session_data = session.dict()
            
            # Use thread pool for I/O
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._write_session_file,
                session_file,
                session_data
            )
            
        except Exception as e:
            logger.error(f"Failed to persist session {session_id}: {e}")
    
    def _write_session_file(self, file_path: Path, data: Dict[str, Any]):
        """Write session data to file (runs in thread pool)."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def _load_session_from_disk(self, session_id: str) -> Optional[SessionData]:
        """Load session from disk."""
        if not self.enable_persistence:
            return None
        
        try:
            session_file = self.persistence_path / f"{session_id}.json"
            if not session_file.exists():
                return None
            
            # Use thread pool for I/O
            loop = asyncio.get_event_loop()
            session_data = await loop.run_in_executor(
                self.executor,
                self._read_session_file,
                session_file
            )
            
            return SessionData(**session_data)
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def _read_session_file(self, file_path: Path) -> Dict[str, Any]:
        """Read session data from file (runs in thread pool)."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    async def _load_persisted_sessions(
        self,
        user_id: Optional[str] = None,
        include_completed: bool = True,
        include_expired: bool = False
    ) -> List[SessionData]:
        """Load persisted sessions from disk."""
        if not self.enable_persistence:
            return []
        
        try:
            sessions = []
            
            # Use thread pool for I/O
            loop = asyncio.get_event_loop()
            session_files = await loop.run_in_executor(
                self.executor,
                self._list_session_files
            )
            
            for session_file in session_files:
                try:
                    session_data = await loop.run_in_executor(
                        self.executor,
                        self._read_session_file,
                        session_file
                    )
                    
                    session = SessionData(**session_data)
                    
                    # Apply filters
                    if user_id and session.user_id != user_id:
                        continue
                    
                    if not include_completed and session.status == SessionStatus.COMPLETED:
                        continue
                    
                    if not include_expired and session.status == SessionStatus.TIMEOUT:
                        continue
                    
                    sessions.append(session)
                    
                except Exception as e:
                    logger.error(f"Failed to load session from {session_file}: {e}")
                    continue
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to load persisted sessions: {e}")
            return []
    
    def _list_session_files(self) -> List[Path]:
        """List session files (runs in thread pool)."""
        return list(self.persistence_path.glob("sess_*.json"))
    
    async def _delete_persisted_session(self, session_id: str):
        """Delete persisted session data."""
        if not self.enable_persistence:
            return
        
        try:
            session_file = self.persistence_path / f"{session_id}.json"
            if session_file.exists():
                # Use thread pool for I/O
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    session_file.unlink
                )
                
        except Exception as e:
            logger.error(f"Failed to delete persisted session {session_id}: {e}")
    
    async def _export_session_json(self, session: SessionData) -> str:
        """Export session as JSON."""
        return json.dumps(session.dict(), indent=2, default=str)
    
    async def _export_session_csv(self, session: SessionData) -> str:
        """Export session as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "session_id", "user_id", "created_at", "updated_at", "status",
            "transcription", "analysis_count"
        ])
        
        # Write session data
        writer.writerow([
            session.session_id,
            session.user_id or "",
            session.created_at.isoformat(),
            session.updated_at.isoformat(),
            session.status.value,
            session.transcription or "",
            len(session.analysis_results)
        ])
        
        # Write analysis results
        if session.analysis_results:
            writer.writerow([])  # Empty row
            writer.writerow(["Analysis Results"])
            writer.writerow([
                "timestamp", "text", "language", "sentiment", "processing_time"
            ])
            
            for result in session.analysis_results:
                writer.writerow([
                    result.timestamp.isoformat(),
                    result.text[:100] + "..." if len(result.text) > 100 else result.text,
                    result.language,
                    result.sentiment.overall_sentiment,
                    result.processing_time
                ])
        
        return output.getvalue()
    
    async def _export_session_pickle(self, session: SessionData) -> bytes:
        """Export session as pickle."""
        return pickle.dumps(session.dict())
    
    async def shutdown(self):
        """Shutdown session manager and cleanup resources."""
        logger.info("Shutting down session manager")
        
        # Stop cleanup task
        if self.cleanup_task:
            self.cleanup_running = False
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Persist all active sessions
        if self.enable_persistence:
            for session_id in list(self.active_sessions.keys()):
                await self._persist_session(session_id)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Session manager shutdown complete")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def initialize_session_manager(**kwargs) -> SessionManager:
    """Initialize global session manager with custom settings."""
    global _session_manager
    _session_manager = SessionManager(**kwargs)
    return _session_manager