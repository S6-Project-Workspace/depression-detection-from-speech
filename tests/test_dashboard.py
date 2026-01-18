"""
Tests for Clinical Dashboard Components

Tests the dashboard components including authentication, WebSocket client,
and visualization components.
"""

import pytest
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import dashboard components
from dashboard.websocket_client import DashboardWebSocketClient, MockWebSocketClient
from dashboard.auth.authentication import (
    AuthenticationManager, User, UserRole, Permission
)
from dashboard.components.risk_charts import RiskVisualization
from dashboard.components.session_monitor import SessionMonitor
from dashboard.components.alert_display import AlertDisplay
from dashboard.components.patient_management import PatientManager


class TestWebSocketClient:
    """Test WebSocket client functionality."""
    
    def test_mock_websocket_client_initialization(self):
        """Test mock WebSocket client initialization."""
        client = MockWebSocketClient(update_interval=0.1)
        
        assert client.update_interval == 0.1
        assert not client.running
        assert not client.connected
        assert len(client.mock_session_ids) == 3
    
    def test_mock_websocket_client_lifecycle(self):
        """Test mock WebSocket client start/stop lifecycle."""
        client = MockWebSocketClient(update_interval=0.1)
        
        # Start client
        client.start()
        assert client.running
        assert client.connected
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop client
        client.stop()
        assert not client.running
        assert not client.connected
    
    def test_websocket_data_handlers(self):
        """Test WebSocket data handler functionality."""
        client = MockWebSocketClient(update_interval=0.1)
        
        # Add data handlers
        risk_data_received = []
        alert_data_received = []
        
        def risk_handler(data):
            risk_data_received.append(data)
        
        def alert_handler(data):
            alert_data_received.append(data)
        
        client.add_data_handler('risk_data', risk_handler)
        client.add_data_handler('alert_data', alert_handler)
        
        # Start client and let it generate data
        client.start()
        time.sleep(0.5)
        client.stop()
        
        # Verify data was received
        assert len(risk_data_received) > 0
        # Alert data is generated randomly, so it might be empty
        
        # Verify data structure
        if risk_data_received:
            risk_data = risk_data_received[0]
            assert isinstance(risk_data, dict)
            # Should contain session data
            for session_id, data in risk_data.items():
                assert 'timestamp' in data
                assert 'risk_score' in data
                assert 'confidence' in data


class TestAuthentication:
    """Test authentication system."""
    
    def test_authentication_manager_initialization(self):
        """Test authentication manager initialization."""
        auth_manager = AuthenticationManager()
        
        # Should have default users
        assert len(auth_manager.users) >= 3
        assert 'admin' in auth_manager.users
        assert 'dr_smith' in auth_manager.users
        assert 'observer' in auth_manager.users
    
    def test_user_authentication(self):
        """Test user authentication."""
        auth_manager = AuthenticationManager()
        
        # Valid authentication
        user = auth_manager.authenticate_user('admin', 'admin123')
        assert user is not None
        assert user.username == 'admin'
        assert user.role == UserRole.ADMIN
        
        # Invalid authentication
        user = auth_manager.authenticate_user('admin', 'wrong_password')
        assert user is None
        
        user = auth_manager.authenticate_user('nonexistent', 'password')
        assert user is None
    
    def test_session_management(self):
        """Test session creation and validation."""
        auth_manager = AuthenticationManager()
        
        # Authenticate user
        user = auth_manager.authenticate_user('admin', 'admin123')
        assert user is not None
        
        # Create session
        session_token = auth_manager.create_session(user)
        assert session_token is not None
        assert len(session_token) > 20  # Should be a proper token
        
        # Validate session
        session = auth_manager.validate_session(session_token)
        assert session is not None
        assert session['username'] == 'admin'
        assert session['role'] == UserRole.ADMIN
        
        # Invalidate session
        auth_manager.invalidate_session(session_token)
        session = auth_manager.validate_session(session_token)
        assert session is None
    
    def test_permission_checking(self):
        """Test permission checking."""
        auth_manager = AuthenticationManager()
        
        # Admin user
        admin_user = auth_manager.authenticate_user('admin', 'admin123')
        admin_token = auth_manager.create_session(admin_user)
        
        # Admin should have all permissions
        assert auth_manager.check_permission(admin_token, Permission.VIEW_SESSIONS)
        assert auth_manager.check_permission(admin_token, Permission.SYSTEM_ADMIN)
        assert auth_manager.check_permission(admin_token, Permission.CLINICAL_NOTES)
        
        # Observer user
        observer_user = auth_manager.authenticate_user('observer', 'observe123')
        observer_token = auth_manager.create_session(observer_user)
        
        # Observer should have limited permissions
        assert auth_manager.check_permission(observer_token, Permission.VIEW_SESSIONS)
        assert not auth_manager.check_permission(observer_token, Permission.SYSTEM_ADMIN)
        assert not auth_manager.check_permission(observer_token, Permission.CLINICAL_NOTES)


class TestRiskVisualization:
    """Test risk visualization component."""
    
    def test_risk_visualization_initialization(self):
        """Test risk visualization initialization."""
        risk_viz = RiskVisualization()
        
        assert isinstance(risk_viz.risk_data_cache, dict)
        assert isinstance(risk_viz.linguistic_data_cache, dict)
        assert len(risk_viz.risk_data_cache) == 0
        assert 'low_risk' in risk_viz.colors
        assert 'high_risk' in risk_viz.colors
    
    def test_risk_data_update(self):
        """Test risk data update functionality."""
        risk_viz = RiskVisualization()
        
        # Mock risk data
        risk_data = {
            'session_001': {
                'timestamp': datetime.now().isoformat(),
                'risk_score': 0.75,
                'confidence': 0.9,
                'audio_contribution': 0.6,
                'text_contribution': 0.5,
                'linguistic_contribution': 0.4,
                'linguistic_markers': {
                    'negative_sentiment': 0.7,
                    'speech_rate': 0.3,
                    'pause_frequency': 0.6
                },
                'processing_time_ms': 1200.0
            }
        }
        
        # Update data
        risk_viz.update_data(risk_data)
        
        # Verify data was stored
        assert 'session_001' in risk_viz.risk_data_cache
        assert len(risk_viz.risk_data_cache['session_001']) == 1
        
        stored_data = risk_viz.risk_data_cache['session_001'][0]
        assert stored_data['risk_score'] == 0.75
        assert stored_data['confidence'] == 0.9
        
        # Verify linguistic data was stored
        assert 'session_001' in risk_viz.linguistic_data_cache
        assert len(risk_viz.linguistic_data_cache['session_001']) == 1
    
    def test_session_summary(self):
        """Test session summary generation."""
        risk_viz = RiskVisualization()
        
        # Add multiple data points
        for i in range(5):
            risk_data = {
                'session_001': {
                    'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                    'risk_score': 0.5 + (i * 0.1),
                    'confidence': 0.8 + (i * 0.02),
                    'processing_time_ms': 1000.0 + (i * 100)
                }
            }
            risk_viz.update_data(risk_data)
        
        # Get summary
        summary = risk_viz.get_session_summary('session_001')
        
        assert summary['total_points'] == 5
        assert summary['current_risk'] == 0.9  # Last added
        assert 0.5 <= summary['average_risk'] <= 0.9
        assert summary['max_risk'] == 0.9
        assert summary['min_risk'] == 0.5
        assert summary['risk_trend'] in ['increasing', 'decreasing', 'stable']


class TestSessionMonitor:
    """Test session monitoring component."""
    
    def test_session_monitor_initialization(self):
        """Test session monitor initialization."""
        session_monitor = SessionMonitor()
        
        # Should have mock sessions
        assert len(session_monitor.session_data_cache) >= 3
        assert 'session_001' in session_monitor.session_data_cache
        assert 'session_002' in session_monitor.session_data_cache
        assert 'session_003' in session_monitor.session_data_cache
        
        # Should have session history
        assert len(session_monitor.session_history_cache) >= 3
    
    def test_session_data_update(self):
        """Test session data update."""
        session_monitor = SessionMonitor()
        
        # Mock session update
        session_data = {
            'session_004': {
                'session_id': 'session_004',
                'patient_id': 'patient_004',
                'status': 'active',
                'current_risk_score': 0.65,
                'average_latency': 1100.0
            }
        }
        
        session_monitor.update_data(session_data)
        
        # Verify data was added
        assert 'session_004' in session_monitor.session_data_cache
        assert session_monitor.session_data_cache['session_004']['status'] == 'active'
        
        # Verify history was updated
        assert 'session_004' in session_monitor.session_history_cache
        assert len(session_monitor.session_history_cache['session_004']) > 0
    
    def test_active_sessions(self):
        """Test active sessions retrieval."""
        session_monitor = SessionMonitor()
        
        active_sessions = session_monitor.get_active_sessions()
        
        # Should have at least some active sessions from mock data
        assert isinstance(active_sessions, dict)
        
        # All returned sessions should be active
        for session_id, session_info in active_sessions.items():
            assert session_info.get('status') == 'active'
    
    def test_session_statistics(self):
        """Test session statistics calculation."""
        session_monitor = SessionMonitor()
        
        stats = session_monitor.get_session_statistics()
        
        assert 'total_sessions' in stats
        assert 'active_sessions' in stats
        assert 'high_risk_sessions' in stats
        assert 'total_alerts' in stats
        assert 'average_latency' in stats
        
        assert stats['total_sessions'] >= 3  # From mock data
        assert stats['average_latency'] > 0


class TestAlertDisplay:
    """Test alert display component."""
    
    def test_alert_display_initialization(self):
        """Test alert display initialization."""
        alert_display = AlertDisplay()
        
        # Should have mock alerts
        assert 'alerts' in alert_display.alert_data_cache
        assert len(alert_display.alert_data_cache['alerts']) >= 3
        
        # Should have acknowledged alerts list
        assert isinstance(alert_display.acknowledged_alerts, list)
    
    def test_alert_data_update(self):
        """Test alert data update."""
        alert_display = AlertDisplay()
        
        # Mock new alert
        new_alert_data = {
            'new_alert': {
                'alert_id': 'alert_test_001',
                'session_id': 'session_001',
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'high_risk',
                'severity': 'high',
                'message': 'Test alert message',
                'risk_score': 0.85
            }
        }
        
        initial_count = len(alert_display.alert_data_cache['alerts'])
        alert_display.update_data(new_alert_data)
        
        # Verify alert was added
        assert len(alert_display.alert_data_cache['alerts']) == initial_count + 1
        
        # Find the new alert
        new_alert = None
        for alert in alert_display.alert_data_cache['alerts']:
            if alert['alert_id'] == 'alert_test_001':
                new_alert = alert
                break
        
        assert new_alert is not None
        assert new_alert['message'] == 'Test alert message'
        assert new_alert['severity'] == 'high'
    
    def test_alert_statistics(self):
        """Test alert statistics calculation."""
        alert_display = AlertDisplay()
        
        stats = alert_display.get_alert_statistics()
        
        assert 'total_alerts' in stats
        assert 'acknowledged_alerts' in stats
        assert 'unacknowledged_alerts' in stats
        assert 'acknowledgment_rate' in stats
        assert 'severity_counts' in stats
        
        assert stats['total_alerts'] >= 3  # From mock data
        assert 0 <= stats['acknowledgment_rate'] <= 1


class TestPatientManager:
    """Test patient management component."""
    
    def test_patient_manager_initialization(self):
        """Test patient manager initialization."""
        patient_manager = PatientManager()
        
        # Should have mock patients
        assert len(patient_manager.patient_data_cache) >= 3
        assert 'patient_001' in patient_manager.patient_data_cache
        assert 'patient_002' in patient_manager.patient_data_cache
        assert 'patient_003' in patient_manager.patient_data_cache
        
        # Should have clinical notes
        assert len(patient_manager.clinical_notes_cache) >= 3
    
    def test_patient_data_structure(self):
        """Test patient data structure."""
        patient_manager = PatientManager()
        
        patient = patient_manager.patient_data_cache['patient_001']
        
        # Verify required fields
        assert 'patient_id' in patient
        assert 'display_name' in patient
        assert 'language' in patient
        assert 'total_sessions' in patient
        assert 'average_risk_score' in patient
        assert 'primary_clinician' in patient
        assert 'status' in patient
    
    def test_clinical_notes_structure(self):
        """Test clinical notes structure."""
        patient_manager = PatientManager()
        
        notes = patient_manager.clinical_notes_cache['patient_001']
        
        assert len(notes) >= 2  # From mock data
        
        note = notes[0]
        assert 'note_id' in note
        assert 'timestamp' in note
        assert 'clinician' in note
        assert 'note_type' in note
        assert 'content' in note
        assert 'risk_level' in note


def test_dashboard_integration():
    """Test integration between dashboard components."""
    # Initialize all components
    websocket_client = MockWebSocketClient(update_interval=0.1)
    risk_viz = RiskVisualization()
    session_monitor = SessionMonitor()
    alert_display = AlertDisplay()
    
    # Set up data handlers
    websocket_client.add_data_handler('risk_data', risk_viz.update_data)
    websocket_client.add_data_handler('alert_data', alert_display.update_data)
    websocket_client.add_data_handler('session_data', session_monitor.update_data)
    
    # Start WebSocket client
    websocket_client.start()
    
    # Let it run briefly to generate data
    time.sleep(0.3)
    
    # Stop client
    websocket_client.stop()
    
    # Verify data was propagated to components
    assert len(risk_viz.risk_data_cache) > 0
    # Alert data might be empty due to random generation
    
    # Verify session monitor has data
    stats = session_monitor.get_session_statistics()
    assert stats['total_sessions'] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])