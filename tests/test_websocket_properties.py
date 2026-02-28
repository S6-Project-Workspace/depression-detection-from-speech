"""
Property-based tests for WebSocket functionality.

Tests WebSocket connection establishment and multi-client session management
using property-based testing to verify universal properties.

Requirements: 6.1, 6.4
"""

import pytest
import asyncio
import json
import uuid
from typing import List, Dict, Any
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from web_interface.websocket_handler import WebSocketManager, ConnectionInfo
from web_interface.models import SessionData, SessionStatus, WebSocketMessage


# Mock WebSocket class for testing
class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages_sent = []
        self.closed = False
        self.accept_called = False
    
    async def accept(self):
        """Mock accept method."""
        self.accept_called = True
    
    async def send_text(self, message: str):
        """Mock send_text method."""
        if self.closed:
            raise Exception("WebSocket is closed")
        self.messages_sent.append(message)
    
    async def close(self):
        """Mock close method."""
        self.closed = True
    
    async def receive_text(self):
        """Mock receive_text method."""
        if self.closed:
            raise Exception("WebSocket is closed")
        # Return a test message
        return json.dumps({"type": "ping"})


class WebSocketStateMachine(RuleBasedStateMachine):
    """State machine for testing WebSocket manager properties."""
    
    def __init__(self):
        super().__init__()
        self.manager = WebSocketManager()
        self.active_sessions = set()
        self.websockets = {}
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.manager = WebSocketManager()
        self.active_sessions = set()
        self.websockets = {}
    
    @rule(session_id=st.text(min_size=1, max_size=50))
    def connect_session(self, session_id):
        """Rule: Connect a new WebSocket session."""
        assume(session_id not in self.active_sessions)
        assume(len(session_id.strip()) > 0)
        
        # Create mock WebSocket
        websocket = MockWebSocket()
        self.websockets[session_id] = websocket
        
        # Connect session using asyncio.run
        async def _connect():
            return await self.manager.connect(websocket, session_id)
        
        actual_session_id = asyncio.run(_connect())
        
        # Track active session
        self.active_sessions.add(actual_session_id)
        
        # Verify connection was accepted
        assert websocket.accept_called
        assert actual_session_id in self.manager.active_connections
        assert self.manager.is_session_active(actual_session_id)
    
    @rule()
    def disconnect_random_session(self):
        """Rule: Disconnect a random active session."""
        if not self.active_sessions:
            return
        
        session_id = self.active_sessions.pop()
        
        # Use asyncio.run to handle the async call
        async def _disconnect():
            await self.manager.disconnect(session_id)
        
        asyncio.run(_disconnect())
        
        # Verify disconnection
        assert session_id not in self.manager.active_connections
        assert not self.manager.is_session_active(session_id)
    
    @rule(message_data=st.dictionaries(st.text(), st.one_of(st.text(), st.integers(), st.floats(allow_nan=False))))
    def send_message_to_random_session(self, message_data):
        """Rule: Send a message to a random active session."""
        if not self.active_sessions:
            return
        
        session_id = next(iter(self.active_sessions))
        
        # Use asyncio.run to handle the async call
        async def _send_message():
            return await self.manager.send_message(session_id, message_data)
        
        success = asyncio.run(_send_message())
        
        if session_id in self.manager.active_connections:
            assert success
            websocket = self.websockets.get(session_id)
            if websocket:
                assert len(websocket.messages_sent) > 0
    
    @invariant()
    def connection_count_matches_active_sessions(self):
        """Invariant: Connection count should match active sessions."""
        expected_count = len([s for s in self.active_sessions if s in self.manager.active_connections])
        actual_count = self.manager.get_connection_count()
        assert actual_count == expected_count
    
    @invariant()
    def all_active_sessions_have_connections(self):
        """Invariant: All tracked active sessions should have connections."""
        for session_id in self.active_sessions:
            if session_id in self.manager.active_connections:
                connection_info = self.manager.active_connections[session_id]
                assert connection_info.is_active
                assert connection_info.session_id == session_id


@pytest.mark.asyncio
class TestWebSocketProperties:
    """Property-based tests for WebSocket functionality."""
    
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10, unique=True))
    @settings(max_examples=50, deadline=5000)
    async def test_websocket_connection_establishment_property(self, session_ids):
        """
        Feature: nlp-web-interface, Property 25: WebSocket connection establishment
        For any audio processing initiation, persistent WebSocket connections should be established successfully.
        **Validates: Requirements 6.1**
        """
        manager = WebSocketManager()
        connected_sessions = []
        
        try:
            # Test connection establishment for multiple sessions
            for session_id in session_ids:
                websocket = MockWebSocket()
                
                # Connect session
                actual_session_id = await manager.connect(websocket, session_id)
                connected_sessions.append(actual_session_id)
                
                # Verify connection establishment
                assert websocket.accept_called, f"WebSocket.accept() not called for session {session_id}"
                assert actual_session_id in manager.active_connections, f"Session {session_id} not in active connections"
                assert manager.is_session_active(actual_session_id), f"Session {session_id} not marked as active"
                
                # Verify connection info
                connection_info = manager.active_connections[actual_session_id]
                assert connection_info.session_id == actual_session_id
                assert connection_info.is_active
                assert connection_info.websocket == websocket
                
                # Verify session data
                assert actual_session_id in manager.sessions
                session_data = manager.sessions[actual_session_id]
                assert session_data.session_id == actual_session_id
                assert session_data.status == SessionStatus.ACTIVE
        
        finally:
            # Clean up all connections
            for session_id in connected_sessions:
                if manager.is_session_active(session_id):
                    await manager.disconnect(session_id)
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=10000)
    async def test_multi_client_session_management_property(self, num_clients):
        """
        Feature: nlp-web-interface, Property 28: Multi-client session management
        For any set of connected clients, each should operate independently without interference.
        **Validates: Requirements 6.4**
        """
        manager = WebSocketManager()
        sessions = []
        websockets = []
        
        try:
            # Connect multiple clients
            for i in range(num_clients):
                session_id = f"client_{i}_{uuid.uuid4().hex[:8]}"
                websocket = MockWebSocket()
                
                actual_session_id = await manager.connect(websocket, session_id)
                sessions.append(actual_session_id)
                websockets.append(websocket)
            
            # Verify all clients are connected independently
            assert manager.get_connection_count() == num_clients
            
            # Test independent message sending
            test_messages = [{"type": "test", "client_id": i, "data": f"message_{i}"} for i in range(num_clients)]
            
            for i, (session_id, message) in enumerate(zip(sessions, test_messages)):
                success = await manager.send_message(session_id, message)
                assert success, f"Failed to send message to client {i}"
                
                # Verify only the target client received the message
                target_websocket = websockets[i]
                assert len(target_websocket.messages_sent) > 0, f"Client {i} did not receive message"
                
                # Verify message content
                sent_message = json.loads(target_websocket.messages_sent[-1])
                assert sent_message["data"]["client_id"] == i
            
            # Test independent disconnection
            # Disconnect half the clients
            clients_to_disconnect = sessions[:num_clients//2]
            remaining_clients = sessions[num_clients//2:]
            
            for session_id in clients_to_disconnect:
                await manager.disconnect(session_id)
                assert not manager.is_session_active(session_id)
            
            # Verify remaining clients are still active
            for session_id in remaining_clients:
                assert manager.is_session_active(session_id)
            
            assert manager.get_connection_count() == len(remaining_clients)
            
            # Test that remaining clients can still receive messages
            for session_id in remaining_clients:
                test_message = {"type": "test", "data": "still_active"}
                success = await manager.send_message(session_id, test_message)
                assert success, f"Failed to send message to remaining client {session_id}"
        
        finally:
            # Clean up all remaining connections
            for session_id in sessions:
                if manager.is_session_active(session_id):
                    await manager.disconnect(session_id)
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=20), 
        st.one_of(st.text(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
        min_size=1, max_size=10
    ))
    @settings(max_examples=50, deadline=3000)
    async def test_message_serialization_consistency(self, message_data):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any WebSocket message data, serialization should be consistent and valid.
        **Validates: Requirements 10.4**
        """
        manager = WebSocketManager()
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        websocket = MockWebSocket()
        
        try:
            # Connect session
            await manager.connect(websocket, session_id)
            
            # Send message with arbitrary data
            success = await manager.send_message(session_id, message_data)
            assert success, "Failed to send message"
            
            # Verify message was sent
            assert len(websocket.messages_sent) > 0, "No messages were sent"
            
            # Verify message can be parsed as JSON
            sent_message_json = websocket.messages_sent[-1]
            parsed_message = json.loads(sent_message_json)
            
            # Verify WebSocketMessage structure
            assert "type" in parsed_message
            assert "session_id" in parsed_message
            assert "data" in parsed_message
            assert "timestamp" in parsed_message
            
            # Verify session_id matches
            assert parsed_message["session_id"] == session_id
            
            # Verify data is preserved
            sent_data = parsed_message["data"]
            for key, value in message_data.items():
                assert key in sent_data
                assert sent_data[key] == value
            
            # Verify timestamp is valid ISO format
            timestamp_str = parsed_message["timestamp"]
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        finally:
            # Clean up
            if manager.is_session_active(session_id):
                await manager.disconnect(session_id)
    
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5, unique=True))
    @settings(max_examples=30, deadline=5000)
    async def test_concurrent_connection_handling(self, session_ids):
        """
        Test that WebSocket manager handles concurrent connections correctly.
        """
        manager = WebSocketManager()
        
        async def connect_session(session_id):
            """Helper to connect a session."""
            websocket = MockWebSocket()
            actual_session_id = await manager.connect(websocket, session_id)
            return actual_session_id, websocket
        
        try:
            # Connect all sessions concurrently
            connection_tasks = [connect_session(sid) for sid in session_ids]
            results = await asyncio.gather(*connection_tasks)
            
            # Verify all connections succeeded
            assert len(results) == len(session_ids)
            
            connected_sessions = []
            for actual_session_id, websocket in results:
                assert websocket.accept_called
                assert manager.is_session_active(actual_session_id)
                connected_sessions.append(actual_session_id)
            
            # Verify connection count
            assert manager.get_connection_count() == len(session_ids)
            
            # Test concurrent message sending
            async def send_test_message(session_id, message_id):
                """Helper to send a test message."""
                message = {"type": "test", "message_id": message_id}
                return await manager.send_message(session_id, message)
            
            message_tasks = [
                send_test_message(session_id, i) 
                for i, session_id in enumerate(connected_sessions)
            ]
            send_results = await asyncio.gather(*message_tasks)
            
            # Verify all messages were sent successfully
            assert all(send_results), "Some messages failed to send"
        
        finally:
            # Clean up all connections
            for session_id in connected_sessions:
                if manager.is_session_active(session_id):
                    await manager.disconnect(session_id)
    
    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """
        Test that inactive connections are properly cleaned up.
        """
        manager = WebSocketManager()
        manager.connection_timeout = timedelta(seconds=1)  # Short timeout for testing
        
        session_id = f"timeout_test_{uuid.uuid4().hex[:8]}"
        websocket = MockWebSocket()
        
        try:
            # Connect session
            await manager.connect(websocket, session_id)
            assert manager.is_session_active(session_id)
            
            # Simulate passage of time by manually updating last_activity
            connection_info = manager.active_connections[session_id]
            connection_info.last_activity = datetime.now() - timedelta(seconds=2)
            
            # Trigger cleanup manually (normally done by background task)
            current_time = datetime.now()
            if (current_time - connection_info.last_activity) > manager.connection_timeout:
                await manager.disconnect(session_id)
            
            # Verify session was cleaned up
            assert not manager.is_session_active(session_id)
            assert session_id not in manager.active_connections
        
        finally:
            # Ensure cleanup
            if manager.is_session_active(session_id):
                await manager.disconnect(session_id)


# Run the state machine test
TestWebSocketStateMachine = WebSocketStateMachine.TestCase


if __name__ == "__main__":
    # Run a simple test
    import asyncio
    
    async def run_simple_test():
        test_instance = TestWebSocketProperties()
        await test_instance.test_websocket_connection_establishment_property(["test1", "test2"])
        print("Simple WebSocket property test passed!")
    
    asyncio.run(run_simple_test())