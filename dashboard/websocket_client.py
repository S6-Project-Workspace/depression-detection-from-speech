"""
WebSocket Client for Real-time Dashboard Updates

Provides real-time data streaming from the backend streaming system
to the Streamlit dashboard interface.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

logger = logging.getLogger(__name__)


class DashboardWebSocketClient:
    """
    WebSocket client for real-time dashboard data streaming.
    
    Handles connection management, data streaming, and error recovery
    for the clinical dashboard interface.
    """
    
    def __init__(self, 
                 server_url: str = "ws://localhost:8765",
                 reconnect_interval: float = 5.0,
                 max_reconnect_attempts: int = 10):
        """
        Initialize WebSocket client.
        
        Args:
            server_url: WebSocket server URL
            reconnect_interval: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.server_url = server_url
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Connection state
        self.websocket = None
        self.connected = False
        self.reconnect_count = 0
        
        # Data handlers
        self.data_handlers: Dict[str, List[Callable]] = {
            'risk_data': [],
            'alert_data': [],
            'session_data': [],
            'system_metrics': []
        }
        
        # Threading
        self.event_loop = None
        self.client_thread = None
        self.running = False
        
        # Data cache for dashboard
        self.latest_data: Dict[str, Any] = {
            'risk_data': {},
            'alert_data': {},
            'session_data': {},
            'system_metrics': {}
        }
        
        self._lock = threading.Lock()
    
    def add_data_handler(self, data_type: str, handler: Callable[[Dict], None]):
        """
        Add a data handler for specific data types.
        
        Args:
            data_type: Type of data ('risk_data', 'alert_data', etc.)
            handler: Callback function to handle the data
        """
        if data_type in self.data_handlers:
            self.data_handlers[data_type].append(handler)
        else:
            logger.warning(f"Unknown data type: {data_type}")
    
    def start(self):
        """Start the WebSocket client in a separate thread."""
        if self.running:
            logger.warning("WebSocket client already running")
            return
        
        self.running = True
        self.client_thread = threading.Thread(target=self._run_client, daemon=True)
        self.client_thread.start()
        logger.info("WebSocket client started")
    
    def stop(self):
        """Stop the WebSocket client."""
        self.running = False
        
        if self.event_loop and not self.event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._disconnect(), self.event_loop)
        
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.join(timeout=5.0)
        
        logger.info("WebSocket client stopped")
    
    def _run_client(self):
        """Run the WebSocket client event loop."""
        try:
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_until_complete(self._client_loop())
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
        finally:
            if self.event_loop and not self.event_loop.is_closed():
                self.event_loop.close()
    
    async def _client_loop(self):
        """Main client loop with reconnection logic."""
        while self.running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.error(f"Connection error: {e}")
                
                if self.reconnect_count < self.max_reconnect_attempts:
                    self.reconnect_count += 1
                    logger.info(f"Reconnecting in {self.reconnect_interval}s "
                              f"(attempt {self.reconnect_count}/{self.max_reconnect_attempts})")
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    logger.error("Max reconnection attempts reached")
                    break
    
    async def _connect_and_listen(self):
        """Connect to WebSocket server and listen for messages."""
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.websocket = websocket
                self.connected = True
                self.reconnect_count = 0
                logger.info(f"Connected to WebSocket server: {self.server_url}")
                
                # Send subscription message
                await self._subscribe_to_data()
                
                # Listen for messages
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        await self._handle_message(message)
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                        
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except InvalidURI:
            logger.error(f"Invalid WebSocket URI: {self.server_url}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.connected = False
            self.websocket = None
    
    async def _subscribe_to_data(self):
        """Subscribe to real-time data streams."""
        subscription_message = {
            'type': 'subscribe',
            'data_types': ['risk_data', 'alert_data', 'session_data', 'system_metrics'],
            'client_id': f"dashboard_{int(time.time())}"
        }
        
        await self.websocket.send(json.dumps(subscription_message))
        logger.info("Subscribed to data streams")
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'data_update':
                await self._handle_data_update(data)
            elif message_type == 'error':
                logger.error(f"Server error: {data.get('message')}")
            elif message_type == 'ping':
                await self._send_pong()
            else:
                logger.debug(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_data_update(self, data: Dict):
        """Handle data update messages."""
        data_type = data.get('data_type')
        payload = data.get('payload', {})
        
        if data_type in self.data_handlers:
            # Update cache
            with self._lock:
                self.latest_data[data_type].update(payload)
            
            # Call handlers
            for handler in self.data_handlers[data_type]:
                try:
                    handler(payload)
                except Exception as e:
                    logger.error(f"Error in data handler: {e}")
    
    async def _send_pong(self):
        """Send pong response to ping."""
        pong_message = {'type': 'pong', 'timestamp': datetime.now().isoformat()}
        await self.websocket.send(json.dumps(pong_message))
    
    async def _disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.connected = False
    
    def get_latest_data(self, data_type: str) -> Dict:
        """
        Get the latest cached data for a specific type.
        
        Args:
            data_type: Type of data to retrieve
            
        Returns:
            Latest data dictionary
        """
        with self._lock:
            return self.latest_data.get(data_type, {}).copy()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status.
        
        Returns:
            Connection status information
        """
        return {
            'connected': self.connected,
            'server_url': self.server_url,
            'reconnect_count': self.reconnect_count,
            'running': self.running
        }


class MockWebSocketClient(DashboardWebSocketClient):
    """
    Mock WebSocket client for testing and development.
    
    Generates simulated real-time data for dashboard testing
    without requiring a live WebSocket server.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize mock client.
        
        Args:
            update_interval: Seconds between mock data updates
        """
        super().__init__()
        self.update_interval = update_interval
        self.mock_session_ids = ["session_001", "session_002", "session_003"]
        self.mock_data_thread = None
    
    def start(self):
        """Start the mock data generator."""
        if self.running:
            return
        
        self.running = True
        self.connected = True
        self.mock_data_thread = threading.Thread(target=self._generate_mock_data, daemon=True)
        self.mock_data_thread.start()
        logger.info("Mock WebSocket client started")
    
    def stop(self):
        """Stop the mock data generator."""
        self.running = False
        if self.mock_data_thread and self.mock_data_thread.is_alive():
            self.mock_data_thread.join(timeout=2.0)
        self.connected = False
        logger.info("Mock WebSocket client stopped")
    
    def _generate_mock_data(self):
        """Generate mock real-time data."""
        import random
        
        while self.running:
            try:
                # Generate mock risk data
                for session_id in self.mock_session_ids:
                    risk_data = {
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat(),
                        'risk_score': random.uniform(0.2, 0.9),
                        'confidence': random.uniform(0.7, 0.95),
                        'audio_contribution': random.uniform(0.3, 0.8),
                        'text_contribution': random.uniform(0.2, 0.7),
                        'linguistic_contribution': random.uniform(0.1, 0.6),
                        'processing_time_ms': random.uniform(800, 1500)
                    }
                    
                    # Update cache and call handlers
                    with self._lock:
                        if session_id not in self.latest_data['risk_data']:
                            self.latest_data['risk_data'][session_id] = []
                        self.latest_data['risk_data'][session_id].append(risk_data)
                        
                        # Keep only recent data
                        if len(self.latest_data['risk_data'][session_id]) > 100:
                            self.latest_data['risk_data'][session_id] = \
                                self.latest_data['risk_data'][session_id][-100:]
                    
                    for handler in self.data_handlers['risk_data']:
                        try:
                            handler({session_id: risk_data})
                        except Exception as e:
                            logger.error(f"Mock handler error: {e}")
                
                # Generate mock alerts occasionally
                if random.random() < 0.1:  # 10% chance
                    session_id = random.choice(self.mock_session_ids)
                    alert_data = {
                        'alert_id': f"alert_{int(time.time())}",
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat(),
                        'alert_type': random.choice(['high_risk', 'trend_warning']),
                        'severity': random.choice(['medium', 'high']),
                        'message': f"Risk threshold exceeded for {session_id}",
                        'risk_score': random.uniform(0.8, 0.95)
                    }
                    
                    with self._lock:
                        if 'alerts' not in self.latest_data['alert_data']:
                            self.latest_data['alert_data']['alerts'] = []
                        self.latest_data['alert_data']['alerts'].append(alert_data)
                    
                    for handler in self.data_handlers['alert_data']:
                        try:
                            handler({'new_alert': alert_data})
                        except Exception as e:
                            logger.error(f"Mock alert handler error: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Mock data generation error: {e}")
                time.sleep(1.0)