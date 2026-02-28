"""
WebSocket handler for real-time communication.

This module manages WebSocket connections for real-time audio processing,
session management, and concurrent client support.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, Set, Optional, Any
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field

from .models import WebSocketMessage, SessionData, SessionStatus, AudioQuality
from .audio_processor import get_audio_processor, ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    session_id: str
    user_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()


class WebSocketManager:
    """Manages WebSocket connections and real-time communication."""
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self.sessions: Dict[str, SessionData] = {}
        self.connection_timeout = timedelta(minutes=30)  # 30 minute timeout
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_inactive_connections())
    
    async def stop_cleanup_task(self):
        """Stop the background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None, 
                     user_id: Optional[str] = None) -> str:
        """
        Accept a WebSocket connection and register it.
        
        Args:
            websocket: The WebSocket connection
            session_id: Optional existing session ID
            user_id: Optional user identifier
            
        Returns:
            The session ID for this connection
        """
        await websocket.accept()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        # Create connection info
        connection_info = ConnectionInfo(
            websocket=websocket,
            session_id=session_id,
            user_id=user_id
        )
        
        # Register connection
        self.active_connections[session_id] = connection_info
        
        # Create or update session data
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData(
                session_id=session_id,
                user_id=user_id,
                status=SessionStatus.ACTIVE
            )
        else:
            # Update existing session
            self.sessions[session_id].updated_at = datetime.now()
            self.sessions[session_id].status = SessionStatus.ACTIVE
        
        logger.info(f"WebSocket connected: session_id={session_id}, user_id={user_id}")
        
        # Send connection confirmation
        await self.send_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Start cleanup task if not running
        await self.start_cleanup_task()
        
        return session_id
    
    async def disconnect(self, session_id: str):
        """
        Disconnect a WebSocket connection and clean up resources.
        
        Args:
            session_id: The session ID to disconnect
        """
        if session_id in self.active_connections:
            connection_info = self.active_connections[session_id]
            connection_info.is_active = False
            
            try:
                await connection_info.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket for session {session_id}: {e}")
            
            # Remove from active connections
            del self.active_connections[session_id]
            
            # Update session status
            if session_id in self.sessions:
                self.sessions[session_id].status = SessionStatus.COMPLETED
                self.sessions[session_id].updated_at = datetime.now()
            
            logger.info(f"WebSocket disconnected: session_id={session_id}")
    
    async def send_message(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            session_id: The session ID to send to
            data: The message data
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if session_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent session: {session_id}")
            return False
        
        connection_info = self.active_connections[session_id]
        
        if not connection_info.is_active:
            logger.warning(f"Attempted to send message to inactive session: {session_id}")
            return False
        
        try:
            # Create WebSocket message
            message = WebSocketMessage(
                type=data.get("type", "message"),
                session_id=session_id,
                data=data
            )
            
            # Send message
            await connection_info.websocket.send_text(message.model_dump_json())
            connection_info.update_activity()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to session {session_id}: {e}")
            # Mark connection as inactive and disconnect
            connection_info.is_active = False
            await self.disconnect(session_id)
            return False
    
    async def broadcast_message(self, data: Dict[str, Any], 
                              exclude_sessions: Optional[Set[str]] = None) -> int:
        """
        Broadcast a message to all active connections.
        
        Args:
            data: The message data
            exclude_sessions: Optional set of session IDs to exclude
            
        Returns:
            Number of connections the message was sent to
        """
        if exclude_sessions is None:
            exclude_sessions = set()
        
        sent_count = 0
        
        for session_id in list(self.active_connections.keys()):
            if session_id not in exclude_sessions:
                if await self.send_message(session_id, data):
                    sent_count += 1
        
        return sent_count
    
    async def handle_message(self, session_id: str, message: str):
        """
        Handle an incoming WebSocket message.
        
        Args:
            session_id: The session ID that sent the message
            message: The raw message string
        """
        if session_id not in self.active_connections:
            logger.warning(f"Received message from non-existent session: {session_id}")
            return
        
        connection_info = self.active_connections[session_id]
        connection_info.update_activity()
        
        try:
            # Parse message
            data = json.loads(message)
            message_type = data.get("type", "unknown")
            
            logger.debug(f"Received message from {session_id}: type={message_type}")
            
            # Handle different message types
            if message_type == "ping":
                await self.send_message(session_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "audio_chunk":
                await self._handle_audio_chunk(session_id, data)
            
            elif message_type == "start_recording":
                await self._handle_start_recording(session_id, data)
            
            elif message_type == "stop_recording":
                await self._handle_stop_recording(session_id, data)
            
            elif message_type == "get_session_info":
                await self._handle_get_session_info(session_id)
            
            else:
                logger.warning(f"Unknown message type from {session_id}: {message_type}")
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "unknown_message_type",
                    "message": f"Unknown message type: {message_type}"
                })
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from session {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "error": "invalid_json",
                "message": "Invalid JSON format"
            })
        
        except Exception as e:
            logger.error(f"Error handling message from session {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "error": "processing_error",
                "message": "Error processing message"
            })
    
    async def _handle_audio_chunk(self, session_id: str, data: Dict[str, Any]):
        """Handle incoming audio chunk data."""
        try:
            audio_processor = get_audio_processor()
            
            # Extract audio data and metadata
            audio_data = data.get("audio_data", "")
            chunk_id = data.get("chunk_id", f"chunk_{int(time.time() * 1000)}")
            sample_rate = data.get("sample_rate", 16000)
            
            if not audio_data:
                await self.send_message(session_id, {
                    "type": "error",
                    "error": "missing_audio_data",
                    "message": "No audio data provided"
                })
                return
            
            # Convert audio data from base64 if needed
            if isinstance(audio_data, str):
                import base64
                try:
                    audio_bytes = base64.b64decode(audio_data)
                except Exception as e:
                    await self.send_message(session_id, {
                        "type": "error",
                        "error": "invalid_audio_data",
                        "message": f"Failed to decode audio data: {e}"
                    })
                    return
            else:
                audio_bytes = audio_data
            
            # Process the audio chunk
            result = await audio_processor.process_audio_chunk(
                session_id=session_id,
                audio_data=audio_bytes,
                chunk_id=chunk_id,
                sample_rate=sample_rate
            )
            
            # Send processing result back to client
            response_data = {
                "type": "audio_processing_result",
                "chunk_id": chunk_id,
                "transcription": result.transcription,
                "confidence": result.confidence,
                "audio_quality": result.audio_quality.model_dump(mode='json'),
                "processing_time": result.processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            if result.error:
                response_data["error"] = result.error
            
            await self.send_message(session_id, response_data)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for session {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "error": "audio_processing_failed",
                "message": f"Audio processing failed: {str(e)}"
            })
    
    async def _handle_start_recording(self, session_id: str, data: Dict[str, Any]):
        """Handle start recording request."""
        try:
            audio_processor = get_audio_processor()
            language = data.get("language", "tamil")
            
            # Start audio processing session
            await audio_processor.start_session(
                session_id=session_id,
                language=language,
                result_callback=self._create_result_callback(session_id)
            )
            
            # Update session status
            if session_id in self.sessions:
                self.sessions[session_id].updated_at = datetime.now()
            
            await self.send_message(session_id, {
                "type": "recording_started",
                "session_id": session_id,
                "language": language,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error starting recording for session {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "error": "start_recording_failed",
                "message": f"Failed to start recording: {str(e)}"
            })
    
    async def _handle_stop_recording(self, session_id: str, data: Dict[str, Any]):
        """Handle stop recording request."""
        try:
            audio_processor = get_audio_processor()
            
            # Stop audio processing session and get summary
            session_summary = await audio_processor.stop_session(session_id)
            
            # Update session status
            if session_id in self.sessions:
                self.sessions[session_id].updated_at = datetime.now()
                self.sessions[session_id].status = SessionStatus.COMPLETED
                # Store final transcription if available
                if session_summary.get("final_transcription"):
                    self.sessions[session_id].transcription = session_summary["final_transcription"]
            
            await self.send_message(session_id, {
                "type": "recording_stopped",
                "session_id": session_id,
                "session_summary": session_summary,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error stopping recording for session {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "error": "stop_recording_failed",
                "message": f"Failed to stop recording: {str(e)}"
            })
    
    async def _handle_get_session_info(self, session_id: str):
        """Handle session info request."""
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
            await self.send_message(session_id, {
                "type": "session_info",
                "session_data": session_data.model_dump(mode='json')
            })
        else:
            await self.send_message(session_id, {
                "type": "error",
                "error": "session_not_found",
                "message": "Session not found"
            })
    
    def _create_result_callback(self, session_id: str):
        """Create a result callback for audio processing."""
        async def callback(result: ProcessingResult):
            """Callback to handle processing results."""
            try:
                await self.send_message(session_id, {
                    "type": "real_time_result",
                    "transcription": result.transcription,
                    "confidence": result.confidence,
                    "audio_quality": result.audio_quality.model_dump(mode='json'),
                    "processing_time": result.processing_time,
                    "chunk_id": result.chunk_id,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error in result callback for session {session_id}: {e}")
        
        return callback
    
    async def _cleanup_inactive_connections(self):
        """Background task to clean up inactive connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                inactive_sessions = []
                
                for session_id, connection_info in self.active_connections.items():
                    if (current_time - connection_info.last_activity) > self.connection_timeout:
                        inactive_sessions.append(session_id)
                
                # Disconnect inactive sessions
                for session_id in inactive_sessions:
                    logger.info(f"Cleaning up inactive session: {session_id}")
                    await self.disconnect(session_id)
                
                # Clean up old session data (keep for 24 hours)
                session_cleanup_time = current_time - timedelta(hours=24)
                old_sessions = [
                    session_id for session_id, session_data in self.sessions.items()
                    if session_data.updated_at < session_cleanup_time
                    and session_data.status != SessionStatus.ACTIVE
                ]
                
                for session_id in old_sessions:
                    logger.info(f"Cleaning up old session data: {session_id}")
                    del self.sessions[session_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_session_info(self, session_id: str) -> Optional[SessionData]:
        """Get session information."""
        return self.sessions.get(session_id)
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is active."""
        return session_id in self.active_connections and \
               self.active_connections[session_id].is_active


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def websocket_endpoint(websocket: WebSocket, session_id: Optional[str] = None):
    """
    WebSocket endpoint handler.
    
    Args:
        websocket: The WebSocket connection
        session_id: Optional session ID from query parameters
    """
    session_id = await websocket_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Wait for message
            message = await websocket.receive_text()
            await websocket_manager.handle_message(session_id, message)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        await websocket_manager.disconnect(session_id)


# Utility functions for integration with FastAPI

async def startup_websocket_manager():
    """Startup function to initialize WebSocket manager."""
    await websocket_manager.start_cleanup_task()
    logger.info("WebSocket manager started")


async def shutdown_websocket_manager():
    """Shutdown function to clean up WebSocket manager."""
    await websocket_manager.stop_cleanup_task()
    
    # Disconnect all active connections
    active_sessions = list(websocket_manager.active_connections.keys())
    for session_id in active_sessions:
        await websocket_manager.disconnect(session_id)
    
    logger.info("WebSocket manager shut down")


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    return websocket_manager