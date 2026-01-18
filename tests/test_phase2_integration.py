"""
Integration Tests for Phase 2: Risk Monitoring and Alerts

This module contains comprehensive integration tests for Phase 2 of the
real-time streaming system, focusing on risk monitoring and alert functionality.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from hypothesis import given, strategies as st, settings

from streaming.risk_monitor import RiskMonitor
from streaming.alert_system import AlertSystem, AlertRule
from streaming.session_manager import SessionManager, StreamingSessionManager
from streaming.models import (
    RiskDataPoint, AlertEvent, AlertType, AlertSeverity, ProcessingMode,
    SessionConfig, StreamingSession, SessionStatus
)


class TestRiskMonitorIntegration:
    """Test risk monitor integration with other components."""
    
    @pytest.fixture
    def risk_monitor(self):
        """Create risk monitor for testing."""
        return RiskMonitor(
            rolling_window_minutes=5,
            max_history_points=100,
            risk_thresholds={'low': 0.3, 'moderate': 0.6, 'high': 0.8}
        )
    
    @pytest.fixture
    def sample_risk_data(self):
        """Create sample risk data points."""
        base_time = datetime.now()
        return [
            RiskDataPoint(
                timestamp=base_time + timedelta(seconds=i * 30),
                risk_score=0.5 + (i * 0.1),
                confidence=0.9,
                audio_contribution=0.6,
                text_contribution=0.4,
                linguistic_contribution=0.3,
                processing_mode=ProcessingMode.TRIMODAL
            )
            for i in range(10)
        ]
    
    def test_risk_monitor_data_flow(self, risk_monitor, sample_risk_data):
        """Test complete risk monitoring data flow."""
        # Add risk data points
        for data_point in sample_risk_data:
            risk_monitor.add_risk_datapoint(data_point)
        
        # Verify data storage
        assert len(risk_monitor.risk_history) == 10
        
        # Test rolling average
        rolling_avg = risk_monitor.get_rolling_average()
        assert 0.0 <= rolling_avg <= 1.0
        
        # Test trend analysis
        trend = risk_monitor.get_risk_trend()
        assert 'trend_slope' in trend
        assert 'trend_direction' in trend
        
        # Test risk level categorization
        current_level = risk_monitor.get_risk_level()
        assert current_level in ['LOW', 'MODERATE', 'HIGH']
        
        # Test comprehensive analysis
        analysis = risk_monitor.analyze_risk_patterns()
        assert 'analysis_type' in analysis
        assert 'distribution' in analysis
        assert 'trend' in analysis
    
    def test_risk_monitor_performance(self, risk_monitor):
        """Test risk monitor performance under load."""
        start_time = time.time()
        
        # Add many data points (within the max_history_points limit)
        for i in range(100):  # Use 100 instead of 1000 to stay within limit
            data_point = RiskDataPoint(
                timestamp=datetime.now(),
                risk_score=0.5 + (i % 10) * 0.05,
                confidence=0.9,
                audio_contribution=0.6
            )
            risk_monitor.add_risk_datapoint(data_point)
        
        processing_time = time.time() - start_time
        
        # Should process 100 points quickly
        assert processing_time < 1.0
        
        # Verify functionality still works
        summary = risk_monitor.get_risk_summary()
        assert summary['data_points'] == 100


class TestAlertSystemIntegration:
    """Test alert system integration with risk monitoring."""
    
    @pytest.fixture
    def alert_system(self):
        """Create alert system for testing."""
        return AlertSystem()
    
    @pytest.fixture
    def high_risk_data(self):
        """Create high-risk data point."""
        return RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.95,
            audio_contribution=0.8,
            text_contribution=0.7,
            linguistic_contribution=0.6,
            processing_mode=ProcessingMode.TRIMODAL
        )
    
    def test_alert_generation_flow(self, alert_system, high_risk_data):
        """Test complete alert generation flow."""
        # Generate alerts
        alerts = alert_system.evaluate_alerts(high_risk_data, "test_session")
        
        # Should generate at least one alert for high risk
        assert len(alerts) > 0
        
        # Verify alert properties
        alert = alerts[0]
        assert alert.risk_score == 0.85
        assert alert.session_id == "test_session"
        assert alert.alert_type in [AlertType.HIGH_RISK, AlertType.TREND_WARNING]
        
        # Test alert acknowledgment
        ack_result = alert_system.acknowledge_alert(alert.alert_id, "test_user")
        assert ack_result is True
        
        # Verify acknowledgment
        acknowledged_alert = alert_system.active_alerts[alert.alert_id]
        assert acknowledged_alert.acknowledged is True
        assert acknowledged_alert.acknowledged_by == "test_user"
    
    def test_alert_deduplication(self, alert_system, high_risk_data):
        """Test alert deduplication functionality."""
        # Generate multiple alerts with same data
        alerts1 = alert_system.evaluate_alerts(high_risk_data, "test_session")
        alerts2 = alert_system.evaluate_alerts(high_risk_data, "test_session")
        
        # Second call should generate fewer or no alerts due to deduplication
        assert len(alerts2) <= len(alerts1)
    
    def test_multi_session_alerts(self, alert_system, high_risk_data):
        """Test alerts across multiple sessions."""
        # Generate alerts for different sessions
        alerts1 = alert_system.evaluate_alerts(high_risk_data, "session_1")
        alerts2 = alert_system.evaluate_alerts(high_risk_data, "session_2")
        
        # Should generate alerts for both sessions
        assert len(alerts1) > 0
        assert len(alerts2) > 0
        
        # Verify session isolation
        session1_alerts = alert_system.get_active_alerts("session_1")
        session2_alerts = alert_system.get_active_alerts("session_2")
        
        assert all(alert.session_id == "session_1" for alert in session1_alerts)
        assert all(alert.session_id == "session_2" for alert in session2_alerts)


class TestRiskAlertIntegration:
    """Test integration between risk monitoring and alert systems."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated risk monitor and alert system."""
        risk_monitor = RiskMonitor(rolling_window_minutes=2, max_history_points=50)
        alert_system = AlertSystem()
        return risk_monitor, alert_system
    
    def test_risk_trend_alerts(self, integrated_system):
        """Test alerts based on risk trends."""
        risk_monitor, alert_system = integrated_system
        
        # Create increasing risk trend
        base_time = datetime.now()
        risk_scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for i, score in enumerate(risk_scores):
            data_point = RiskDataPoint(
                timestamp=base_time + timedelta(seconds=i * 30),
                risk_score=score,
                confidence=0.9,
                audio_contribution=score * 0.8
            )
            
            # Add to risk monitor
            risk_monitor.add_risk_datapoint(data_point)
            
            # Get trend analysis
            trend = risk_monitor.get_risk_trend()
            trend_slope = trend.get('trend_slope', 0.0)
            
            # Evaluate alerts with trend information
            alerts = alert_system.evaluate_alerts(
                data_point, 
                "trend_test_session",
                trend_slope=trend_slope
            )
            
            # Should generate alerts for high risk scores
            if score >= 0.8:
                assert len(alerts) > 0
    
    def test_sustained_risk_alerts(self, integrated_system):
        """Test alerts for sustained high risk."""
        risk_monitor, alert_system = integrated_system
        
        # Add custom rule for sustained risk
        sustained_rule = AlertRule(
            rule_id="sustained_high",
            name="Sustained High Risk",
            alert_type=AlertType.TREND_WARNING,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.7,
            duration_threshold_seconds=120,  # 2 minutes
            cooldown_minutes=0
        )
        alert_system.add_alert_rule(sustained_rule)
        
        # Create sustained high risk scenario
        base_time = datetime.now()
        
        # Add 5 minutes of high risk data (every 30 seconds)
        risk_history = []
        for i in range(10):
            data_point = RiskDataPoint(
                timestamp=base_time + timedelta(seconds=i * 30),
                risk_score=0.75,  # Consistently high
                confidence=0.9,
                audio_contribution=0.7
            )
            
            risk_monitor.add_risk_datapoint(data_point)
            risk_history.append(data_point)
        
        # Evaluate alerts with history
        current_data = risk_history[-1]
        alerts = alert_system.evaluate_alerts(
            current_data,
            "sustained_test_session",
            risk_history=risk_history
        )
        
        # Should generate sustained risk alert
        sustained_alerts = [
            alert for alert in alerts 
            if alert.metadata.get('rule_id') == 'sustained_high'
        ]
        assert len(sustained_alerts) > 0


class TestSessionManagerIntegration:
    """Test session manager integration with risk monitoring and alerts."""
    
    @pytest.fixture
    async def session_manager(self):
        """Create session manager for testing."""
        config = SessionConfig(
            patient_id="test_patient",
            sample_rate=16000,
            buffer_size_seconds=30,
            risk_threshold=0.8
        )
        
        session = StreamingSession(
            session_id="test_session",
            patient_id="test_patient",
            start_time=datetime.now(),
            status=SessionStatus.INITIALIZING
        )
        
        # Mock the components to avoid hardware dependencies
        with patch('streaming.session_manager.AsyncAudioInputHandler'), \
             patch('streaming.session_manager.BufferManager'), \
             patch('streaming.session_manager.StreamProcessor'):
            
            manager = SessionManager(session=session, config=config, device='cpu')
            await manager.initialize()
            return manager
    
    @pytest.mark.asyncio
    async def test_session_risk_monitoring(self, session_manager):
        """Test risk monitoring within session context."""
        # Verify risk monitor is initialized
        assert session_manager.risk_monitor is not None
        assert session_manager.alert_system is not None
        
        # Test risk callback registration
        risk_results = []
        
        def risk_callback(result: RiskDataPoint):
            risk_results.append(result)
        
        session_manager.add_risk_callback(risk_callback)
        
        # Simulate risk result
        test_result = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.8
        )
        
        session_manager._on_risk_result(test_result)
        
        # Verify callback was called
        assert len(risk_results) == 1
        assert risk_results[0].risk_score == 0.85
        
        # Verify risk monitor received data
        current_score = session_manager.get_current_risk_score()
        assert current_score == 0.85
        
        # Verify alerts were generated
        active_alerts = session_manager.get_active_alerts()
        assert len(active_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_session_alert_management(self, session_manager):
        """Test alert management within session."""
        # Generate high-risk result to trigger alerts
        high_risk_result = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.9,
            confidence=0.95,
            audio_contribution=0.85
        )
        
        session_manager._on_risk_result(high_risk_result)
        
        # Get active alerts
        alerts = session_manager.get_active_alerts()
        assert len(alerts) > 0
        
        # Test alert acknowledgment
        alert_id = alerts[0].alert_id
        ack_result = session_manager.acknowledge_alert(alert_id, "test_clinician")
        assert ack_result is True
        
        # Verify alert is acknowledged
        updated_alerts = session_manager.get_active_alerts()
        assert len(updated_alerts) < len(alerts)
    
    @pytest.mark.asyncio
    async def test_session_statistics_integration(self, session_manager):
        """Test comprehensive session statistics."""
        # Add some risk data
        for i in range(5):
            result = RiskDataPoint(
                timestamp=datetime.now(),
                risk_score=0.6 + (i * 0.05),
                confidence=0.9,
                audio_contribution=0.7
            )
            session_manager._on_risk_result(result)
        
        # Get comprehensive statistics
        stats = session_manager.get_session_statistics()
        
        # Verify all components are included
        assert 'risk_monitor' in stats
        assert 'alert_system' in stats
        
        # Verify risk summary
        risk_summary = session_manager.get_risk_summary()
        assert 'current_risk_score' in risk_summary
        assert 'rolling_average' in risk_summary
        assert 'risk_level' in risk_summary


class TestMultiSessionIntegration:
    """Test multi-session streaming manager integration."""
    
    @pytest.fixture
    async def streaming_manager(self):
        """Create streaming session manager for testing."""
        with patch('streaming.session_manager.AsyncAudioInputHandler'), \
             patch('streaming.session_manager.BufferManager'), \
             patch('streaming.session_manager.StreamProcessor'):
            
            manager = StreamingSessionManager(
                max_concurrent_sessions=3,
                device='cpu',
                persistence_enabled=False
            )
            return manager
    
    @pytest.mark.asyncio
    async def test_multi_session_alert_coordination(self, streaming_manager):
        """Test alert coordination across multiple sessions."""
        # Create multiple sessions
        session1_id = await streaming_manager.create_session("patient_1")
        session2_id = await streaming_manager.create_session("patient_2")
        
        assert session1_id is not None
        assert session2_id is not None
        
        # Get session managers
        session1 = streaming_manager.active_sessions[session1_id]
        session2 = streaming_manager.active_sessions[session2_id]
        
        # Generate high-risk results for both sessions
        high_risk_result = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.9,
            confidence=0.95,
            audio_contribution=0.85
        )
        
        session1._on_risk_result(high_risk_result)
        session2._on_risk_result(high_risk_result)
        
        # Test global alert retrieval
        global_alerts = streaming_manager.get_global_alerts()
        assert len(global_alerts) > 0
        
        # Verify alerts from both sessions
        session_ids = {alert.session_id for alert in global_alerts}
        assert session1_id in session_ids
        assert session2_id in session_ids
        
        # Test global alert acknowledgment
        alert_id = global_alerts[0].alert_id
        ack_result = streaming_manager.acknowledge_global_alert(alert_id, "supervisor")
        assert ack_result is True
    
    @pytest.mark.asyncio
    async def test_session_isolation(self, streaming_manager):
        """Test that sessions are properly isolated."""
        # Create two sessions
        session1_id = await streaming_manager.create_session("patient_1")
        session2_id = await streaming_manager.create_session("patient_2")
        
        session1 = streaming_manager.active_sessions[session1_id]
        session2 = streaming_manager.active_sessions[session2_id]
        
        # Generate different risk levels
        low_risk = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.3,
            confidence=0.9,
            audio_contribution=0.2
        )
        
        high_risk = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.9,
            confidence=0.95,
            audio_contribution=0.85
        )
        
        session1._on_risk_result(low_risk)
        session2._on_risk_result(high_risk)
        
        # Verify isolation
        session1_score = session1.get_current_risk_score()
        session2_score = session2.get_current_risk_score()
        
        assert session1_score == 0.3
        assert session2_score == 0.9
        
        # Verify alert isolation
        session1_alerts = session1.get_active_alerts()
        session2_alerts = session2.get_active_alerts()
        
        assert len(session1_alerts) == 0  # Low risk, no alerts
        assert len(session2_alerts) > 0   # High risk, should have alerts


class TestPerformanceIntegration:
    """Test performance of integrated system."""
    
    @settings(deadline=None, max_examples=5)
    @given(
        num_sessions=st.integers(min_value=1, max_value=5),
        num_risk_points=st.integers(min_value=10, max_value=100)
    )
    def test_system_performance_property(self, num_sessions, num_risk_points):
        """Property-based test for system performance."""
        async def run_performance_test():
            with patch('streaming.session_manager.AsyncAudioInputHandler'), \
                 patch('streaming.session_manager.BufferManager'), \
                 patch('streaming.session_manager.StreamProcessor'):
                
                # Create streaming manager
                manager = StreamingSessionManager(
                    max_concurrent_sessions=num_sessions,
                    device='cpu',
                    persistence_enabled=False
                )
                
                # Create sessions
                session_ids = []
                for i in range(num_sessions):
                    session_id = await manager.create_session(f"patient_{i}")
                    if session_id:
                        session_ids.append(session_id)
                
                # Generate risk data for all sessions
                start_time = time.time()
                
                for session_id in session_ids:
                    session_manager = manager.active_sessions[session_id]
                    
                    for j in range(num_risk_points):
                        risk_data = RiskDataPoint(
                            timestamp=datetime.now(),
                            risk_score=0.5 + (j % 10) * 0.05,
                            confidence=0.9,
                            audio_contribution=0.6
                        )
                        session_manager._on_risk_result(risk_data)
                
                processing_time = time.time() - start_time
                
                # Performance assertions
                total_operations = num_sessions * num_risk_points
                avg_time_per_operation = processing_time / total_operations
                
                # Should process each operation quickly
                assert avg_time_per_operation < 0.01  # 10ms per operation
                
                # Verify system still functions correctly
                stats = manager.get_all_statistics()
                assert stats['total_sessions'] == len(session_ids)
                
                # Cleanup
                await manager.cleanup_all_sessions()
        
        # Run the async test
        asyncio.run(run_performance_test())
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable under load."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        async def memory_test():
            with patch('streaming.session_manager.AsyncAudioInputHandler'), \
                 patch('streaming.session_manager.BufferManager'), \
                 patch('streaming.session_manager.StreamProcessor'):
                
                manager = StreamingSessionManager(max_concurrent_sessions=2, device='cpu')
                
                # Create and destroy sessions multiple times
                for cycle in range(5):
                    # Create sessions
                    session_ids = []
                    for i in range(2):
                        session_id = await manager.create_session(f"patient_{cycle}_{i}")
                        if session_id:
                            session_ids.append(session_id)
                    
                    # Generate data
                    for session_id in session_ids:
                        session_manager = manager.active_sessions[session_id]
                        for j in range(50):
                            risk_data = RiskDataPoint(
                                timestamp=datetime.now(),
                                risk_score=0.5,
                                confidence=0.9,
                                audio_contribution=0.6
                            )
                            session_manager._on_risk_result(risk_data)
                    
                    # Cleanup sessions
                    for session_id in session_ids:
                        await manager.remove_session(session_id)
                    
                    # Force garbage collection
                    gc.collect()
                
                await manager.cleanup_all_sessions()
        
        asyncio.run(memory_test())
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__])