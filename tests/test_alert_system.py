"""
Tests for Alert System

This module contains comprehensive tests for the alert system including
alert generation, deduplication, notifications, and analytics.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings

from streaming.alert_system import (
    AlertSystem, AlertRule, AlertDeduplicator, NotificationHandler, AlertAnalyzer
)
from streaming.models import (
    AlertEvent, AlertType, AlertSeverity, RiskDataPoint, ProcessingMode
)


class TestAlertRule:
    """Test AlertRule functionality."""
    
    def test_alert_rule_creation(self):
        """Test basic alert rule creation."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.alert_type == AlertType.HIGH_RISK
        assert rule.severity == AlertSeverity.HIGH
        assert rule.risk_threshold == 0.8
        assert rule.enabled is True
    
    def test_alert_rule_matches_condition_risk_threshold(self):
        """Test risk threshold matching."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8
        )
        
        # Create risk data points
        high_risk = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.7
        )
        
        low_risk = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.5,
            confidence=0.9,
            audio_contribution=0.4
        )
        
        assert rule.matches_condition(high_risk) is True
        assert rule.matches_condition(low_risk) is False
    
    def test_alert_rule_matches_condition_trend_threshold(self):
        """Test trend threshold matching."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.TREND_WARNING,
            severity=AlertSeverity.MEDIUM,
            trend_threshold=0.1
        )
        
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.6,
            confidence=0.9,
            audio_contribution=0.5
        )
        
        assert rule.matches_condition(risk_data, trend_slope=0.15) is True
        assert rule.matches_condition(risk_data, trend_slope=0.05) is False
        assert rule.matches_condition(risk_data, trend_slope=-0.15) is True  # Absolute value
    
    def test_alert_rule_disabled(self):
        """Test disabled rule behavior."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8,
            enabled=False
        )
        
        high_risk = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.9,
            confidence=0.9,
            audio_contribution=0.8
        )
        
        assert rule.matches_condition(high_risk) is False


class TestAlertDeduplicator:
    """Test AlertDeduplicator functionality."""
    
    def test_deduplicator_creation(self):
        """Test deduplicator initialization."""
        deduplicator = AlertDeduplicator()
        
        assert len(deduplicator.recent_alerts) == 0
        assert len(deduplicator.last_alert_time) == 0
        assert len(deduplicator.alert_counts) == 0
    
    def test_should_generate_alert_first_time(self):
        """Test first alert generation."""
        deduplicator = AlertDeduplicator()
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            cooldown_minutes=5
        )
        
        assert deduplicator.should_generate_alert(rule, "session_1") is True
    
    def test_should_generate_alert_cooldown(self):
        """Test cooldown period enforcement."""
        deduplicator = AlertDeduplicator()
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            cooldown_minutes=5
        )
        
        current_time = datetime.now()
        
        # First alert should be allowed
        assert deduplicator.should_generate_alert(rule, "session_1", current_time) is True
        
        # Second alert within cooldown should be blocked
        assert deduplicator.should_generate_alert(
            rule, "session_1", current_time + timedelta(minutes=2)
        ) is False
        
        # Alert after cooldown should be allowed
        assert deduplicator.should_generate_alert(
            rule, "session_1", current_time + timedelta(minutes=6)
        ) is True
    
    def test_should_generate_alert_rate_limiting(self):
        """Test rate limiting enforcement."""
        deduplicator = AlertDeduplicator()
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            cooldown_minutes=0,  # No cooldown
            max_alerts_per_hour=2
        )
        
        # Use a fixed time in the middle of an hour to avoid hour boundary issues
        current_time = datetime(2026, 1, 18, 14, 30, 0)  # 2:30 PM
        
        # First two alerts should be allowed (same hour)
        assert deduplicator.should_generate_alert(rule, "session_1", current_time) is True
        assert deduplicator.should_generate_alert(
            rule, "session_1", current_time + timedelta(minutes=10)
        ) is True
        
        # Third alert in same hour should be blocked
        assert deduplicator.should_generate_alert(
            rule, "session_1", current_time + timedelta(minutes=20)
        ) is False
        
        # Alert in next hour should be allowed
        assert deduplicator.should_generate_alert(
            rule, "session_1", current_time + timedelta(hours=1, minutes=10)
        ) is True
    
    def test_reset_session(self):
        """Test session reset functionality."""
        deduplicator = AlertDeduplicator()
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH
        )
        
        # Generate some alerts
        deduplicator.should_generate_alert(rule, "session_1")
        deduplicator.should_generate_alert(rule, "session_2")
        
        # Reset session 1
        deduplicator.reset_session("session_1")
        
        # Session 1 should be reset, session 2 should remain
        assert len([k for k in deduplicator.last_alert_time.keys() if k.startswith("session_1:")]) == 0
        assert len([k for k in deduplicator.last_alert_time.keys() if k.startswith("session_2:")]) == 1


class TestNotificationHandler:
    """Test NotificationHandler functionality."""
    
    def test_notification_handler_creation(self):
        """Test notification handler initialization."""
        handler = NotificationHandler()
        
        assert len(handler.notification_callbacks) == 0
        assert len(handler.notification_history) == 0
        assert len(handler.failed_notifications) == 0
        assert 'visual' in handler.channels
        assert 'audio' in handler.channels
        assert 'log' in handler.channels
    
    def test_register_notification_callback(self):
        """Test callback registration."""
        handler = NotificationHandler()
        callback = Mock()
        
        handler.register_notification_callback("custom", callback)
        
        assert "custom" in handler.notification_callbacks
        assert callback in handler.notification_callbacks["custom"]
    
    def test_send_notification_builtin_channels(self):
        """Test sending notifications through built-in channels."""
        handler = NotificationHandler()
        
        alert = AlertEvent(
            alert_id="test_alert",
            timestamp=datetime.now(),
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            risk_score=0.85,
            trigger_condition="risk_score >= 0.8"
        )
        
        results = handler.send_notification(alert, ["visual", "log"])
        
        assert results["visual"] is True
        assert results["log"] is True
        assert len(handler.notification_history) == 1
    
    def test_send_notification_custom_callback(self):
        """Test sending notifications through custom callbacks."""
        handler = NotificationHandler()
        callback = Mock()
        handler.register_notification_callback("custom", callback)
        
        alert = AlertEvent(
            alert_id="test_alert",
            timestamp=datetime.now(),
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            risk_score=0.85,
            trigger_condition="risk_score >= 0.8"
        )
        
        results = handler.send_notification(alert, ["custom"])
        
        assert results["custom"] is True
        callback.assert_called_once_with(alert)
    
    def test_send_notification_failure_handling(self):
        """Test notification failure handling."""
        handler = NotificationHandler()
        failing_callback = Mock(side_effect=Exception("Callback failed"))
        handler.register_notification_callback("failing", failing_callback)
        
        alert = AlertEvent(
            alert_id="test_alert",
            timestamp=datetime.now(),
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            risk_score=0.85,
            trigger_condition="risk_score >= 0.8"
        )
        
        results = handler.send_notification(alert, ["failing"], retry_count=1)
        
        assert results["failing"] is False
        assert len(handler.failed_notifications) == 1
    
    def test_get_notification_statistics(self):
        """Test notification statistics calculation."""
        handler = NotificationHandler()
        
        alert = AlertEvent(
            alert_id="test_alert",
            timestamp=datetime.now(),
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            risk_score=0.85,
            trigger_condition="risk_score >= 0.8"
        )
        
        # Send successful notification
        handler.send_notification(alert, ["visual"])
        
        stats = handler.get_notification_statistics()
        
        assert stats["total_notifications"] == 1
        assert stats["success_rate"] == 1.0
        assert "visual" in stats["channel_performance"]
        assert stats["channel_performance"]["visual"]["success_rate"] == 1.0


class TestAlertAnalyzer:
    """Test AlertAnalyzer functionality."""
    
    def test_analyzer_creation(self):
        """Test analyzer initialization."""
        analyzer = AlertAnalyzer()
        
        assert len(analyzer.alert_history) == 0
        assert len(analyzer.acknowledgment_times) == 0
        assert len(analyzer.false_positive_reports) == 0
    
    def test_record_alert(self):
        """Test alert recording."""
        analyzer = AlertAnalyzer()
        
        alert = AlertEvent(
            alert_id="test_alert",
            timestamp=datetime.now(),
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            risk_score=0.85,
            trigger_condition="risk_score >= 0.8"
        )
        
        analyzer.record_alert(alert)
        
        assert len(analyzer.alert_history) == 1
        assert analyzer.alert_history[0] == alert
    
    def test_record_acknowledgment(self):
        """Test acknowledgment recording."""
        analyzer = AlertAnalyzer()
        
        analyzer.record_acknowledgment("test_alert", 30.5)
        
        assert analyzer.acknowledgment_times["test_alert"] == 30.5
    
    def test_report_false_positive(self):
        """Test false positive reporting."""
        analyzer = AlertAnalyzer()
        
        analyzer.report_false_positive("test_alert")
        analyzer.report_false_positive("test_alert")  # Report twice
        
        assert analyzer.false_positive_reports["test_alert"] == 2
    
    def test_analyze_alert_patterns(self):
        """Test alert pattern analysis."""
        analyzer = AlertAnalyzer()
        
        # Create test alerts
        base_time = datetime.now()
        alerts = []
        
        for i in range(5):
            alert = AlertEvent(
                alert_id=f"alert_{i}",
                timestamp=base_time + timedelta(minutes=i * 10),
                alert_type=AlertType.HIGH_RISK,
                severity=AlertSeverity.HIGH if i % 2 == 0 else AlertSeverity.MEDIUM,
                message=f"Test alert {i}",
                risk_score=0.8 + (i * 0.02),
                trigger_condition="risk_score >= 0.8"
            )
            alerts.append(alert)
            analyzer.record_alert(alert)
        
        # Record some acknowledgments
        analyzer.record_acknowledgment("alert_0", 15.0)
        analyzer.record_acknowledgment("alert_2", 25.0)
        
        # Report false positive
        analyzer.report_false_positive("alert_1")
        
        analysis = analyzer.analyze_alert_patterns(hours=1)
        
        assert analysis["total_alerts"] == 5
        assert analysis["alert_frequency"] == 5.0  # 5 alerts per hour
        assert analysis["severity_distribution"]["high"] == 3
        assert analysis["severity_distribution"]["medium"] == 2
        assert analysis["acknowledgment_rate"] == 0.4  # 2/5
        assert analysis["false_positive_rate"] == 0.2  # 1/5
        assert analysis["average_response_time"] == 20.0  # (15+25)/2
    
    def test_get_alert_effectiveness_metrics(self):
        """Test alert effectiveness metrics."""
        analyzer = AlertAnalyzer()
        
        # Add test data
        alert = AlertEvent(
            alert_id="test_alert",
            timestamp=datetime.now(),
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            risk_score=0.85,
            trigger_condition="risk_score >= 0.8"
        )
        
        analyzer.record_alert(alert)
        analyzer.record_acknowledgment("test_alert", 30.0)
        analyzer.report_false_positive("other_alert")
        
        metrics = analyzer.get_alert_effectiveness_metrics()
        
        assert metrics["total_alerts_recorded"] == 1
        assert metrics["overall_acknowledgment_rate"] == 1.0
        assert metrics["overall_false_positive_rate"] == 1.0  # 1 FP report / 1 alert
        assert metrics["average_response_time"] == 30.0


class TestAlertSystem:
    """Test AlertSystem functionality."""
    
    def test_alert_system_creation(self):
        """Test alert system initialization."""
        system = AlertSystem()
        
        assert len(system.alert_rules) > 0  # Default rules should be loaded
        assert system.deduplicator is not None
        assert system.notification_handler is not None
        assert system.analyzer is not None
        assert system.total_alerts_generated == 0
        assert system.total_alerts_acknowledged == 0
    
    def test_alert_system_custom_rules(self):
        """Test alert system with custom rules."""
        custom_rule = AlertRule(
            rule_id="custom_rule",
            name="Custom Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.CRITICAL,
            risk_threshold=0.95
        )
        
        system = AlertSystem(default_rules=[custom_rule])
        
        assert len(system.alert_rules) == 1
        assert "custom_rule" in system.alert_rules
        assert system.alert_rules["custom_rule"].risk_threshold == 0.95
    
    def test_add_alert_rule(self):
        """Test adding alert rules."""
        system = AlertSystem(default_rules=[])
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8
        )
        
        system.add_alert_rule(rule)
        
        assert "test_rule" in system.alert_rules
        assert system.alert_rules["test_rule"] == rule
    
    def test_remove_alert_rule(self):
        """Test removing alert rules."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8
        )
        
        system = AlertSystem(default_rules=[rule])
        assert "test_rule" in system.alert_rules
        
        system.remove_alert_rule("test_rule")
        assert "test_rule" not in system.alert_rules
    
    def test_enable_disable_alert_rule(self):
        """Test enabling/disabling alert rules."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8
        )
        
        system = AlertSystem(default_rules=[rule])
        
        # Rule should be enabled by default
        assert system.alert_rules["test_rule"].enabled is True
        
        # Disable rule
        system.enable_alert_rule("test_rule", False)
        assert system.alert_rules["test_rule"].enabled is False
        
        # Re-enable rule
        system.enable_alert_rule("test_rule", True)
        assert system.alert_rules["test_rule"].enabled is True
    
    def test_evaluate_alerts_threshold(self):
        """Test alert evaluation with threshold rules."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8,
            cooldown_minutes=0  # No cooldown for testing
        )
        
        system = AlertSystem(default_rules=[rule])
        
        # High risk data should trigger alert
        high_risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.7
        )
        
        alerts = system.evaluate_alerts(high_risk_data, "session_1")
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.HIGH_RISK
        assert alerts[0].risk_score == 0.85
        assert alerts[0].session_id == "session_1"
        
        # Low risk data should not trigger alert
        low_risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.5,
            confidence=0.9,
            audio_contribution=0.4
        )
        
        alerts = system.evaluate_alerts(low_risk_data, "session_1")
        assert len(alerts) == 0
    
    def test_evaluate_alerts_trend(self):
        """Test alert evaluation with trend rules."""
        rule = AlertRule(
            rule_id="trend_rule",
            name="Trend Rule",
            alert_type=AlertType.TREND_WARNING,
            severity=AlertSeverity.MEDIUM,
            trend_threshold=0.1,
            cooldown_minutes=0
        )
        
        system = AlertSystem(default_rules=[rule])
        
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.6,
            confidence=0.9,
            audio_contribution=0.5
        )
        
        # High trend slope should trigger alert
        alerts = system.evaluate_alerts(risk_data, "session_1", trend_slope=0.15)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.TREND_WARNING
        
        # Low trend slope should not trigger alert
        alerts = system.evaluate_alerts(risk_data, "session_1", trend_slope=0.05)
        assert len(alerts) == 0
    
    def test_evaluate_alerts_duration_threshold(self):
        """Test alert evaluation with duration thresholds."""
        rule = AlertRule(
            rule_id="duration_rule",
            name="Duration Rule",
            alert_type=AlertType.TREND_WARNING,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.7,
            duration_threshold_seconds=60,  # 1 minute
            cooldown_minutes=0
        )
        
        system = AlertSystem(default_rules=[rule])
        
        base_time = datetime.now()
        
        # Create risk history with sustained high risk
        risk_history = []
        for i in range(5):
            risk_point = RiskDataPoint(
                timestamp=base_time + timedelta(seconds=i * 20),
                risk_score=0.75,
                confidence=0.9,
                audio_contribution=0.6
            )
            risk_history.append(risk_point)
        
        current_risk = RiskDataPoint(
            timestamp=base_time + timedelta(seconds=100),
            risk_score=0.75,
            confidence=0.9,
            audio_contribution=0.6
        )
        
        # Should trigger alert due to sustained high risk
        alerts = system.evaluate_alerts(
            current_risk, "session_1", risk_history=risk_history
        )
        assert len(alerts) == 1
        
        # Create history with brief high risk (should not trigger)
        short_history = [
            RiskDataPoint(
                timestamp=base_time + timedelta(seconds=80),
                risk_score=0.75,
                confidence=0.9,
                audio_contribution=0.6
            )
        ]
        
        alerts = system.evaluate_alerts(
            current_risk, "session_2", risk_history=short_history
        )
        assert len(alerts) == 0
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8,
            cooldown_minutes=0
        )
        
        system = AlertSystem(default_rules=[rule])
        
        # Generate alert
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.7
        )
        
        alerts = system.evaluate_alerts(risk_data, "session_1")
        assert len(alerts) == 1
        
        alert_id = alerts[0].alert_id
        
        # Acknowledge alert
        result = system.acknowledge_alert(alert_id, "user_1")
        assert result is True
        
        # Check alert is acknowledged
        alert = system.active_alerts[alert_id]
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "user_1"
        assert system.total_alerts_acknowledged == 1
    
    def test_report_false_positive(self):
        """Test false positive reporting."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8,
            cooldown_minutes=0
        )
        
        system = AlertSystem(default_rules=[rule])
        
        # Generate alert
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.7
        )
        
        alerts = system.evaluate_alerts(risk_data, "session_1")
        alert_id = alerts[0].alert_id
        
        # Report false positive
        result = system.report_false_positive(alert_id)
        assert result is True
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8,
            cooldown_minutes=0
        )
        
        system = AlertSystem(default_rules=[rule])
        
        # Generate alerts for different sessions
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.7
        )
        
        alerts1 = system.evaluate_alerts(risk_data, "session_1")
        alerts2 = system.evaluate_alerts(risk_data, "session_2")
        
        # Get all active alerts
        active_alerts = system.get_active_alerts()
        assert len(active_alerts) == 2
        
        # Get active alerts for specific session
        session1_alerts = system.get_active_alerts("session_1")
        assert len(session1_alerts) == 1
        assert session1_alerts[0].session_id == "session_1"
        
        # Acknowledge one alert
        system.acknowledge_alert(alerts1[0].alert_id, "user_1")
        
        # Should now have only one active alert
        active_alerts = system.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].session_id == "session_2"
    
    def test_get_alert_statistics(self):
        """Test alert statistics."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8,
            cooldown_minutes=0
        )
        
        system = AlertSystem(default_rules=[rule])
        
        # Generate and acknowledge some alerts
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.7
        )
        
        alerts = system.evaluate_alerts(risk_data, "session_1")
        system.acknowledge_alert(alerts[0].alert_id, "user_1")
        
        stats = system.get_alert_statistics()
        
        assert stats["total_alerts_generated"] == 1
        assert stats["total_alerts_acknowledged"] == 1
        assert stats["active_alerts"] == 0  # Alert was acknowledged
        assert stats["alert_rules_configured"] >= 1
        assert stats["alert_rules_enabled"] >= 1
    
    def test_cleanup_old_alerts(self):
        """Test cleanup of old alerts."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8,
            cooldown_minutes=0
        )
        
        system = AlertSystem(default_rules=[rule])
        
        # Generate and acknowledge alert
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.7
        )
        
        alerts = system.evaluate_alerts(risk_data, "session_1")
        alert_id = alerts[0].alert_id
        system.acknowledge_alert(alert_id, "user_1")
        
        # Manually set acknowledgment time to old time
        system.active_alerts[alert_id].acknowledged_at = datetime.now() - timedelta(hours=25)
        
        # Cleanup should remove old acknowledged alert
        system.cleanup_old_alerts(hours=24)
        
        assert alert_id not in system.active_alerts
    
    def test_reset_session_alerts(self):
        """Test resetting session alerts."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.HIGH,
            risk_threshold=0.8,
            cooldown_minutes=0
        )
        
        system = AlertSystem(default_rules=[rule])
        
        # Generate alerts for multiple sessions
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.85,
            confidence=0.9,
            audio_contribution=0.7
        )
        
        alerts1 = system.evaluate_alerts(risk_data, "session_1")
        alerts2 = system.evaluate_alerts(risk_data, "session_2")
        
        assert len(system.get_active_alerts()) == 2
        
        # Reset session 1
        system.reset_session_alerts("session_1")
        
        # Should only have session 2 alerts remaining
        active_alerts = system.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].session_id == "session_2"


class TestAlertSystemIntegration:
    """Integration tests for alert system components."""
    
    @settings(deadline=None, max_examples=10)
    @given(
        risk_score=st.floats(min_value=0.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        trend_slope=st.floats(min_value=-0.5, max_value=0.5)
    )
    def test_alert_system_property_based(self, risk_score, confidence, trend_slope):
        """Property-based test for alert system behavior."""
        system = AlertSystem()
        
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=risk_score,
            confidence=confidence,
            audio_contribution=risk_score * 0.8
        )
        
        alerts = system.evaluate_alerts(risk_data, "test_session", trend_slope=trend_slope)
        
        # Properties that should always hold
        assert isinstance(alerts, list)
        assert all(isinstance(alert, AlertEvent) for alert in alerts)
        assert all(alert.risk_score == risk_score for alert in alerts)
        assert all(alert.session_id == "test_session" for alert in alerts)
        
        # If risk is very high, should generate at least one alert
        if risk_score >= 0.9:
            assert len(alerts) > 0
    
    def test_end_to_end_alert_flow(self):
        """Test complete alert flow from generation to acknowledgment."""
        # Create custom notification callback
        notifications_received = []
        
        def custom_notification(alert):
            notifications_received.append(alert)
        
        # Set up alert system with no default rules
        system = AlertSystem(default_rules=[])
        system.register_notification_callback("custom", custom_notification)
        
        # Add custom rule with custom notification
        rule = AlertRule(
            rule_id="e2e_rule",
            name="End-to-End Rule",
            alert_type=AlertType.HIGH_RISK,
            severity=AlertSeverity.CRITICAL,
            risk_threshold=0.85,
            cooldown_minutes=0,
            notification_channels=["visual", "custom", "log"]
        )
        
        system.add_alert_rule(rule)
        
        # Generate high-risk data
        risk_data = RiskDataPoint(
            timestamp=datetime.now(),
            risk_score=0.9,
            confidence=0.95,
            audio_contribution=0.8,
            text_contribution=0.7,
            linguistic_contribution=0.6,
            processing_mode=ProcessingMode.TRIMODAL
        )
        
        # Evaluate alerts
        alerts = system.evaluate_alerts(risk_data, "e2e_session")
        
        # Verify alert generation
        assert len(alerts) >= 1
        
        # Find the alert from our custom rule
        generated_alert = None
        for alert in alerts:
            if alert.metadata.get('rule_id') == 'e2e_rule':
                generated_alert = alert
                break
        
        assert generated_alert is not None, "Custom rule alert not found"
        assert generated_alert.risk_score == 0.9
        assert generated_alert.severity == AlertSeverity.CRITICAL
        
        # Verify notification was sent
        assert len(notifications_received) >= 1
        
        # Find notification for our alert
        custom_notification_found = False
        for notification in notifications_received:
            if notification.alert_id == generated_alert.alert_id:
                custom_notification_found = True
                break
        
        assert custom_notification_found, "Custom notification not found"
        
        # Acknowledge alert
        ack_result = system.acknowledge_alert(generated_alert.alert_id, "test_user")
        assert ack_result is True
        
        # Verify acknowledgment
        acknowledged_alert = system.active_alerts[generated_alert.alert_id]
        assert acknowledged_alert.acknowledged is True
        assert acknowledged_alert.acknowledged_by == "test_user"
        
        # Verify statistics
        stats = system.get_alert_statistics()
        assert stats["total_alerts_generated"] >= 1
        assert stats["total_alerts_acknowledged"] >= 1
    
    def test_concurrent_alert_processing(self):
        """Test alert system under concurrent load."""
        import threading
        import time
        
        system = AlertSystem()
        results = []
        
        def generate_alerts(session_id, num_alerts):
            session_results = []
            for i in range(num_alerts):
                risk_data = RiskDataPoint(
                    timestamp=datetime.now(),
                    risk_score=0.85 + (i * 0.01),
                    confidence=0.9,
                    audio_contribution=0.7
                )
                
                alerts = system.evaluate_alerts(risk_data, session_id)
                session_results.extend(alerts)
                time.sleep(0.01)  # Small delay to simulate real processing
            
            results.append((session_id, session_results))
        
        # Create multiple threads generating alerts
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=generate_alerts,
                args=(f"session_{i}", 5)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 3
        total_alerts = sum(len(session_results) for _, session_results in results)
        assert total_alerts > 0
        
        # Verify system state is consistent
        stats = system.get_alert_statistics()
        assert stats["total_alerts_generated"] == total_alerts


if __name__ == "__main__":
    pytest.main([__file__])