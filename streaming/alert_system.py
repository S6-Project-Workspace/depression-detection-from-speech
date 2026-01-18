"""
Alert System for Real-time Streaming Analysis

This module provides comprehensive alerting capabilities for the streaming
depression detection system, including threshold-based alerts, trend analysis,
notification management, and alert analytics.

Key Features:
- Threshold-based alert generation (configurable thresholds)
- Trend-based alert detection (rapid changes, sustained patterns)
- Alert prioritization and deduplication
- Multi-channel notification system
- Alert acknowledgment and escalation
- Alert analytics and performance tracking

Reference: Real-time Streaming Analysis - Task 2.2
"""

import logging
import time
import uuid
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
import asyncio
import threading
from dataclasses import dataclass, field

from .models import (
    AlertEvent, AlertType, AlertSeverity, RiskDataPoint, 
    ProcessingMode, SessionStatus
)

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """
    Configuration for alert generation rules.
    
    Defines conditions and thresholds for generating alerts
    based on risk scores, trends, and other metrics.
    """
    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    
    # Threshold conditions
    risk_threshold: Optional[float] = None
    trend_threshold: Optional[float] = None
    duration_threshold_seconds: Optional[int] = None
    
    # Trend conditions
    trend_window_minutes: int = 5
    min_trend_points: int = 3
    
    # Deduplication settings
    cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10
    
    # Notification settings
    notification_channels: List[str] = field(default_factory=lambda: ["visual"])
    auto_escalate_minutes: Optional[int] = None
    
    # Metadata
    description: str = ""
    enabled: bool = True
    
    def matches_condition(self, risk_data: RiskDataPoint, trend_slope: float = 0.0) -> bool:
        """Check if risk data matches alert conditions."""
        if not self.enabled:
            return False
        
        # Risk threshold check
        if self.risk_threshold is not None:
            if risk_data.risk_score < self.risk_threshold:
                return False
        
        # Trend threshold check
        if self.trend_threshold is not None:
            if abs(trend_slope) < self.trend_threshold:
                return False
        
        return True


class AlertDeduplicator:
    """
    Manages alert deduplication to prevent alert spam.
    
    Tracks recent alerts and applies cooldown periods and rate limiting
    to ensure meaningful alerts without overwhelming users.
    """
    
    def __init__(self):
        """Initialize alert deduplicator."""
        self.recent_alerts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.lock = threading.RLock()
    
    def should_generate_alert(
        self, 
        rule: AlertRule, 
        session_id: str,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if an alert should be generated based on deduplication rules.
        
        Args:
            rule: Alert rule configuration
            session_id: Session identifier
            current_time: Current timestamp (uses now() if None)
            
        Returns:
            True if alert should be generated, False if deduplicated
        """
        if current_time is None:
            current_time = datetime.now()
        
        with self.lock:
            rule_key = f"{session_id}:{rule.rule_id}"
            
            # Check cooldown period
            if rule_key in self.last_alert_time:
                time_since_last = current_time - self.last_alert_time[rule_key]
                if time_since_last.total_seconds() < (rule.cooldown_minutes * 60):
                    return False
            
            # Check rate limiting (alerts per hour) - check BEFORE incrementing
            hour_key = current_time.strftime("%Y-%m-%d-%H")
            current_hour_count = self.alert_counts[rule_key][hour_key]
            
            if current_hour_count >= rule.max_alerts_per_hour:
                return False
            
            # Update tracking - increment count after checking
            self.alert_counts[rule_key][hour_key] += 1
            self.last_alert_time[rule_key] = current_time
            
            # Add to recent alerts
            self.recent_alerts[rule_key].append(current_time)
            
            # Clean old hour counts (keep last 24 hours)
            cutoff_time = current_time - timedelta(hours=24)
            cutoff_hour = cutoff_time.strftime("%Y-%m-%d-%H")
            
            hours_to_remove = []
            for hour in self.alert_counts[rule_key]:
                if hour < cutoff_hour:
                    hours_to_remove.append(hour)
            
            for hour in hours_to_remove:
                del self.alert_counts[rule_key][hour]
            
            return True
    
    def get_recent_alerts(self, session_id: str, minutes: int = 60) -> List[datetime]:
        """Get recent alert timestamps for a session."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent = []
            
            for rule_key in self.recent_alerts:
                if rule_key.startswith(f"{session_id}:"):
                    for alert_time in self.recent_alerts[rule_key]:
                        if alert_time >= cutoff_time:
                            recent.append(alert_time)
            
            return sorted(recent)
    
    def reset_session(self, session_id: str):
        """Reset deduplication state for a session."""
        with self.lock:
            keys_to_remove = []
            
            # Check all tracking dictionaries for session keys
            for rule_key in list(self.recent_alerts.keys()):
                if rule_key.startswith(f"{session_id}:"):
                    keys_to_remove.append(rule_key)
            
            for rule_key in list(self.last_alert_time.keys()):
                if rule_key.startswith(f"{session_id}:"):
                    keys_to_remove.append(rule_key)
            
            for rule_key in list(self.alert_counts.keys()):
                if rule_key.startswith(f"{session_id}:"):
                    keys_to_remove.append(rule_key)
            
            # Remove duplicates
            keys_to_remove = list(set(keys_to_remove))
            
            for key in keys_to_remove:
                if key in self.recent_alerts:
                    del self.recent_alerts[key]
                if key in self.last_alert_time:
                    del self.last_alert_time[key]
                if key in self.alert_counts:
                    del self.alert_counts[key]


class NotificationHandler:
    """
    Handles multi-channel alert notifications.
    
    Supports various notification channels including visual alerts,
    audio notifications, and external integrations.
    """
    
    def __init__(self):
        """Initialize notification handler."""
        self.notification_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.notification_history: deque = deque(maxlen=1000)
        self.failed_notifications: deque = deque(maxlen=100)
        self.lock = threading.RLock()
        
        # Built-in notification channels
        self.channels = {
            'visual': self._send_visual_notification,
            'audio': self._send_audio_notification,
            'log': self._send_log_notification
        }
    
    def register_notification_callback(self, channel: str, callback: Callable):
        """Register a callback for a notification channel."""
        with self.lock:
            self.notification_callbacks[channel].append(callback)
            logger.info(f"Registered notification callback for channel: {channel}")
    
    def send_notification(
        self, 
        alert: AlertEvent, 
        channels: List[str],
        retry_count: int = 3
    ) -> Dict[str, bool]:
        """
        Send notification through specified channels.
        
        Args:
            alert: Alert event to notify about
            channels: List of notification channels
            retry_count: Number of retry attempts for failed notifications
            
        Returns:
            Dictionary mapping channel names to success status
        """
        results = {}
        
        for channel in channels:
            success = False
            
            for attempt in range(retry_count + 1):
                try:
                    # Try built-in channel first
                    if channel in self.channels:
                        success = self.channels[channel](alert)
                    
                    # Try registered callbacks
                    elif channel in self.notification_callbacks:
                        for callback in self.notification_callbacks[channel]:
                            callback(alert)
                        success = True
                    
                    else:
                        logger.warning(f"Unknown notification channel: {channel}")
                        success = False
                    
                    if success:
                        break
                        
                except Exception as e:
                    logger.error(f"Notification failed (attempt {attempt + 1}): {e}")
                    if attempt == retry_count:
                        self._record_failed_notification(alert, channel, str(e))
            
            results[channel] = success
        
        # Record notification attempt
        with self.lock:
            self.notification_history.append({
                'timestamp': datetime.now(),
                'alert_id': alert.alert_id,
                'channels': channels,
                'results': results
            })
        
        return results
    
    def _send_visual_notification(self, alert: AlertEvent) -> bool:
        """Send visual notification (placeholder implementation)."""
        try:
            # This would integrate with the dashboard UI
            logger.info(f"VISUAL ALERT: {alert.message} (Risk: {alert.risk_score:.3f})")
            return True
        except Exception as e:
            logger.error(f"Visual notification failed: {e}")
            return False
    
    def _send_audio_notification(self, alert: AlertEvent) -> bool:
        """Send audio notification (placeholder implementation)."""
        try:
            # This would play an audio alert
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                logger.info(f"AUDIO ALERT: High priority alert - {alert.message}")
            return True
        except Exception as e:
            logger.error(f"Audio notification failed: {e}")
            return False
    
    def _send_log_notification(self, alert: AlertEvent) -> bool:
        """Send log notification."""
        try:
            log_level = {
                AlertSeverity.LOW: logging.INFO,
                AlertSeverity.MEDIUM: logging.WARNING,
                AlertSeverity.HIGH: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.INFO)
            
            logger.log(log_level, f"ALERT: {alert.message} (Risk: {alert.risk_score:.3f})")
            return True
        except Exception as e:
            logger.error(f"Log notification failed: {e}")
            return False
    
    def _record_failed_notification(self, alert: AlertEvent, channel: str, error: str):
        """Record failed notification for analysis."""
        with self.lock:
            self.failed_notifications.append({
                'timestamp': datetime.now(),
                'alert_id': alert.alert_id,
                'channel': channel,
                'error': error,
                'alert_severity': alert.severity.value
            })
    
    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification performance statistics."""
        with self.lock:
            if not self.notification_history:
                return {
                    'total_notifications': 0,
                    'success_rate': 0.0,
                    'channel_performance': {},
                    'failed_notifications': len(self.failed_notifications)
                }
            
            total_notifications = len(self.notification_history)
            successful_notifications = 0
            channel_stats = defaultdict(lambda: {'sent': 0, 'successful': 0})
            
            for notification in self.notification_history:
                for channel, success in notification['results'].items():
                    channel_stats[channel]['sent'] += 1
                    if success:
                        channel_stats[channel]['successful'] += 1
                        successful_notifications += 1
            
            # Calculate channel performance
            channel_performance = {}
            for channel, stats in channel_stats.items():
                channel_performance[channel] = {
                    'success_rate': stats['successful'] / stats['sent'] if stats['sent'] > 0 else 0.0,
                    'total_sent': stats['sent'],
                    'total_successful': stats['successful']
                }
            
            return {
                'total_notifications': total_notifications,
                'success_rate': successful_notifications / total_notifications if total_notifications > 0 else 0.0,
                'channel_performance': channel_performance,
                'failed_notifications': len(self.failed_notifications)
            }


class AlertAnalyzer:
    """
    Analyzes alert patterns and effectiveness.
    
    Provides insights into alert frequency, accuracy, and clinical relevance
    to help optimize alert rules and reduce false positives.
    """
    
    def __init__(self):
        """Initialize alert analyzer."""
        self.alert_history: deque = deque(maxlen=10000)
        self.acknowledgment_times: Dict[str, float] = {}
        self.false_positive_reports: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
    
    def record_alert(self, alert: AlertEvent):
        """Record an alert for analysis."""
        with self.lock:
            self.alert_history.append(alert)
    
    def record_acknowledgment(self, alert_id: str, response_time_seconds: float):
        """Record alert acknowledgment time."""
        with self.lock:
            self.acknowledgment_times[alert_id] = response_time_seconds
    
    def report_false_positive(self, alert_id: str):
        """Report an alert as false positive."""
        with self.lock:
            self.false_positive_reports[alert_id] += 1
    
    def analyze_alert_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze alert patterns over specified time period.
        
        Args:
            hours: Time window for analysis
            
        Returns:
            Dictionary with alert pattern analysis
        """
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]
            
            if not recent_alerts:
                return {
                    'analysis_period_hours': hours,
                    'total_alerts': 0,
                    'alert_frequency': 0.0,
                    'severity_distribution': {},
                    'type_distribution': {},
                    'average_response_time': 0.0
                }
            
            # Basic statistics
            total_alerts = len(recent_alerts)
            alert_frequency = total_alerts / hours
            
            # Severity distribution
            severity_counts = defaultdict(int)
            for alert in recent_alerts:
                severity_counts[alert.severity.value] += 1
            
            # Type distribution
            type_counts = defaultdict(int)
            for alert in recent_alerts:
                type_counts[alert.alert_type.value] += 1
            
            # Response time analysis
            response_times = []
            acknowledged_alerts = 0
            
            for alert in recent_alerts:
                if alert.alert_id in self.acknowledgment_times:
                    response_times.append(self.acknowledgment_times[alert.alert_id])
                    acknowledged_alerts += 1
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            acknowledgment_rate = acknowledged_alerts / total_alerts if total_alerts > 0 else 0.0
            
            # False positive analysis
            false_positives = sum(
                1 for alert in recent_alerts
                if alert.alert_id in self.false_positive_reports
            )
            false_positive_rate = false_positives / total_alerts if total_alerts > 0 else 0.0
            
            return {
                'analysis_period_hours': hours,
                'total_alerts': total_alerts,
                'alert_frequency': alert_frequency,
                'severity_distribution': dict(severity_counts),
                'type_distribution': dict(type_counts),
                'average_response_time': avg_response_time,
                'acknowledgment_rate': acknowledgment_rate,
                'false_positive_rate': false_positive_rate,
                'false_positives': false_positives
            }
    
    def get_alert_effectiveness_metrics(self) -> Dict[str, Any]:
        """Get comprehensive alert effectiveness metrics."""
        with self.lock:
            if not self.alert_history:
                return {
                    'total_alerts_recorded': 0,
                    'overall_acknowledgment_rate': 0.0,
                    'overall_false_positive_rate': 0.0,
                    'average_response_time': 0.0
                }
            
            total_alerts = len(self.alert_history)
            acknowledged_count = len(self.acknowledgment_times)
            false_positive_count = len(self.false_positive_reports)
            
            acknowledgment_rate = acknowledged_count / total_alerts if total_alerts > 0 else 0.0
            false_positive_rate = false_positive_count / total_alerts if total_alerts > 0 else 0.0
            
            avg_response_time = (
                sum(self.acknowledgment_times.values()) / len(self.acknowledgment_times)
                if self.acknowledgment_times else 0.0
            )
            
            return {
                'total_alerts_recorded': total_alerts,
                'overall_acknowledgment_rate': acknowledgment_rate,
                'overall_false_positive_rate': false_positive_rate,
                'average_response_time': avg_response_time,
                'acknowledged_alerts': acknowledged_count,
                'false_positive_reports': false_positive_count
            }


class AlertSystem:
    """
    Comprehensive alert system for real-time streaming analysis.
    
    Features:
    - Configurable alert rules with threshold and trend detection
    - Alert deduplication and rate limiting
    - Multi-channel notification system
    - Alert acknowledgment and escalation
    - Alert analytics and performance tracking
    """
    
    def __init__(
        self,
        default_rules: Optional[List[AlertRule]] = None,
        enable_analytics: bool = True
    ):
        """
        Initialize alert system.
        
        Args:
            default_rules: Default alert rules to configure
            enable_analytics: Whether to enable alert analytics
        """
        # Core components
        self.deduplicator = AlertDeduplicator()
        self.notification_handler = NotificationHandler()
        self.analyzer = AlertAnalyzer() if enable_analytics else None
        
        # Alert rules and state
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.total_alerts_generated = 0
        self.total_alerts_acknowledged = 0
        
        # Set up default rules
        if default_rules:
            for rule in default_rules:
                self.add_alert_rule(rule)
        else:
            self._setup_default_rules()
        
        logger.info(f"AlertSystem initialized with {len(self.alert_rules)} rules")
    
    def _setup_default_rules(self):
        """Set up default alert rules."""
        default_rules = [
            # High risk threshold alert
            AlertRule(
                rule_id="high_risk_threshold",
                name="High Risk Threshold",
                alert_type=AlertType.HIGH_RISK,
                severity=AlertSeverity.HIGH,
                risk_threshold=0.8,
                cooldown_minutes=2,
                max_alerts_per_hour=6,
                notification_channels=["visual", "audio", "log"],
                description="Alert when risk score exceeds 0.8"
            ),
            
            # Critical risk threshold alert
            AlertRule(
                rule_id="critical_risk_threshold",
                name="Critical Risk Threshold",
                alert_type=AlertType.HIGH_RISK,
                severity=AlertSeverity.CRITICAL,
                risk_threshold=0.9,
                cooldown_minutes=1,
                max_alerts_per_hour=10,
                notification_channels=["visual", "audio", "log"],
                auto_escalate_minutes=5,
                description="Alert when risk score exceeds 0.9"
            ),
            
            # Rapid risk increase trend
            AlertRule(
                rule_id="rapid_risk_increase",
                name="Rapid Risk Increase",
                alert_type=AlertType.TREND_WARNING,
                severity=AlertSeverity.MEDIUM,
                trend_threshold=0.1,  # 0.1 increase per minute
                trend_window_minutes=3,
                cooldown_minutes=5,
                max_alerts_per_hour=4,
                notification_channels=["visual", "log"],
                description="Alert when risk increases rapidly"
            ),
            
            # Sustained high risk
            AlertRule(
                rule_id="sustained_high_risk",
                name="Sustained High Risk",
                alert_type=AlertType.TREND_WARNING,
                severity=AlertSeverity.HIGH,
                risk_threshold=0.7,
                duration_threshold_seconds=300,  # 5 minutes
                cooldown_minutes=10,
                max_alerts_per_hour=2,
                notification_channels=["visual", "audio", "log"],
                description="Alert when risk remains high for extended period"
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule."""
        with self.lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        with self.lock:
            if rule_id in self.alert_rules:
                rule = self.alert_rules.pop(rule_id)
                logger.info(f"Removed alert rule: {rule.name} ({rule_id})")
            else:
                logger.warning(f"Alert rule not found: {rule_id}")
    
    def enable_alert_rule(self, rule_id: str, enabled: bool = True):
        """Enable or disable an alert rule."""
        with self.lock:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = enabled
                status = "enabled" if enabled else "disabled"
                logger.info(f"Alert rule {rule_id} {status}")
            else:
                logger.warning(f"Alert rule not found: {rule_id}")
    
    def evaluate_alerts(
        self, 
        risk_data: RiskDataPoint, 
        session_id: str,
        trend_slope: float = 0.0,
        risk_history: Optional[List[RiskDataPoint]] = None
    ) -> List[AlertEvent]:
        """
        Evaluate risk data against alert rules and generate alerts.
        
        Args:
            risk_data: Current risk assessment data
            session_id: Session identifier
            trend_slope: Current risk trend slope
            risk_history: Recent risk history for duration checks
            
        Returns:
            List of generated alert events
        """
        generated_alerts = []
        
        with self.lock:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # Check if rule conditions are met
                if not rule.matches_condition(risk_data, trend_slope):
                    continue
                
                # Check duration threshold if specified
                if rule.duration_threshold_seconds and risk_history:
                    if not self._check_duration_threshold(
                        rule, risk_data, risk_history
                    ):
                        continue
                
                # Check deduplication
                if not self.deduplicator.should_generate_alert(
                    rule, session_id, risk_data.timestamp
                ):
                    continue
                
                # Generate alert
                alert = self._create_alert(rule, risk_data, session_id)
                generated_alerts.append(alert)
                
                # Store alert
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                self.total_alerts_generated += 1
                
                # Send notifications
                notification_results = self.notification_handler.send_notification(
                    alert, rule.notification_channels
                )
                
                # Record for analytics
                if self.analyzer:
                    self.analyzer.record_alert(alert)
                
                logger.info(
                    f"Generated alert: {alert.message} "
                    f"(Risk: {alert.risk_score:.3f}, Severity: {alert.severity.value})"
                )
        
        return generated_alerts
    
    def _check_duration_threshold(
        self, 
        rule: AlertRule, 
        current_data: RiskDataPoint,
        risk_history: List[RiskDataPoint]
    ) -> bool:
        """Check if risk has been above threshold for required duration."""
        if not rule.duration_threshold_seconds or not rule.risk_threshold:
            return True
        
        # Check how long risk has been above threshold
        threshold_start_time = None
        
        for data_point in reversed(risk_history):
            if data_point.risk_score >= rule.risk_threshold:
                threshold_start_time = data_point.timestamp
            else:
                break
        
        if threshold_start_time is None:
            return False
        
        duration_seconds = (current_data.timestamp - threshold_start_time).total_seconds()
        return duration_seconds >= rule.duration_threshold_seconds
    
    def _create_alert(
        self, 
        rule: AlertRule, 
        risk_data: RiskDataPoint, 
        session_id: str
    ) -> AlertEvent:
        """Create an alert event from rule and risk data."""
        alert_id = str(uuid.uuid4())
        
        # Generate alert message
        message = self._generate_alert_message(rule, risk_data)
        
        # Determine trigger condition
        trigger_condition = self._get_trigger_condition(rule, risk_data)
        
        alert = AlertEvent(
            alert_id=alert_id,
            timestamp=risk_data.timestamp,
            alert_type=rule.alert_type,
            severity=rule.severity,
            message=message,
            risk_score=risk_data.risk_score,
            trigger_condition=trigger_condition,
            session_id=session_id,
            metadata={
                'rule_id': rule.rule_id,
                'rule_name': rule.name,
                'confidence': risk_data.confidence,
                'processing_mode': risk_data.processing_mode.value,
                'audio_contribution': risk_data.audio_contribution,
                'text_contribution': risk_data.text_contribution,
                'linguistic_contribution': risk_data.linguistic_contribution
            }
        )
        
        return alert
    
    def _generate_alert_message(self, rule: AlertRule, risk_data: RiskDataPoint) -> str:
        """Generate human-readable alert message."""
        if rule.alert_type == AlertType.HIGH_RISK:
            return (
                f"High risk detected: {risk_data.risk_score:.3f} "
                f"(threshold: {rule.risk_threshold:.3f})"
            )
        elif rule.alert_type == AlertType.TREND_WARNING:
            return (
                f"Risk trend warning: {rule.name} "
                f"(current risk: {risk_data.risk_score:.3f})"
            )
        else:
            return f"Alert: {rule.name} (risk: {risk_data.risk_score:.3f})"
    
    def _get_trigger_condition(self, rule: AlertRule, risk_data: RiskDataPoint) -> str:
        """Get description of what triggered the alert."""
        conditions = []
        
        if rule.risk_threshold is not None:
            conditions.append(f"risk_score >= {rule.risk_threshold}")
        
        if rule.trend_threshold is not None:
            conditions.append(f"trend_slope >= {rule.trend_threshold}")
        
        if rule.duration_threshold_seconds is not None:
            conditions.append(f"duration >= {rule.duration_threshold_seconds}s")
        
        return " AND ".join(conditions) if conditions else "rule_matched"
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            user_id: User acknowledging the alert
            
        Returns:
            True if alert was acknowledged, False if not found
        """
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                
                if not alert.acknowledged:
                    alert.acknowledge(user_id)
                    self.total_alerts_acknowledged += 1
                    
                    # Record response time for analytics
                    if self.analyzer:
                        response_time = (datetime.now() - alert.timestamp).total_seconds()
                        self.analyzer.record_acknowledgment(alert_id, response_time)
                    
                    logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
                    return True
                else:
                    logger.warning(f"Alert already acknowledged: {alert_id}")
                    return False
            else:
                logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False
    
    def report_false_positive(self, alert_id: str) -> bool:
        """
        Report an alert as false positive.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if report was recorded, False if alert not found
        """
        with self.lock:
            if alert_id in self.active_alerts or any(
                alert.alert_id == alert_id for alert in self.alert_history
            ):
                if self.analyzer:
                    self.analyzer.report_false_positive(alert_id)
                
                logger.info(f"False positive reported: {alert_id}")
                return True
            else:
                logger.warning(f"Alert not found for false positive report: {alert_id}")
                return False
    
    def get_active_alerts(self, session_id: Optional[str] = None) -> List[AlertEvent]:
        """Get currently active (unacknowledged) alerts."""
        with self.lock:
            active = []
            
            for alert in self.active_alerts.values():
                if not alert.acknowledged:
                    if session_id is None or alert.session_id == session_id:
                        active.append(alert)
            
            return sorted(active, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(
        self, 
        session_id: Optional[str] = None,
        hours: int = 24
    ) -> List[AlertEvent]:
        """Get alert history for specified time period."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            history = []
            for alert in self.alert_history:
                if alert.timestamp >= cutoff_time:
                    if session_id is None or alert.session_id == session_id:
                        history.append(alert)
            
            return sorted(history, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert system statistics."""
        with self.lock:
            active_count = len([
                alert for alert in self.active_alerts.values()
                if not alert.acknowledged
            ])
            
            # Get notification statistics
            notification_stats = self.notification_handler.get_notification_statistics()
            
            # Get analytics if available
            analytics = {}
            if self.analyzer:
                analytics = self.analyzer.get_alert_effectiveness_metrics()
            
            return {
                'total_alerts_generated': self.total_alerts_generated,
                'total_alerts_acknowledged': self.total_alerts_acknowledged,
                'active_alerts': active_count,
                'alert_rules_configured': len(self.alert_rules),
                'alert_rules_enabled': len([
                    rule for rule in self.alert_rules.values() if rule.enabled
                ]),
                'notification_statistics': notification_stats,
                'analytics': analytics
            }
    
    def cleanup_old_alerts(self, hours: int = 24):
        """Clean up old acknowledged alerts."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            alerts_to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if alert.acknowledged and alert.acknowledged_at and alert.acknowledged_at < cutoff_time:
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
            
            if alerts_to_remove:
                logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")
    
    def reset_session_alerts(self, session_id: str):
        """Reset all alerts for a specific session."""
        with self.lock:
            # Remove active alerts for session
            alerts_to_remove = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.session_id == session_id
            ]
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
            
            # Reset deduplication state
            self.deduplicator.reset_session(session_id)
            
            logger.info(f"Reset alerts for session: {session_id}")
    
    def register_notification_callback(self, channel: str, callback: Callable):
        """Register a custom notification callback."""
        self.notification_handler.register_notification_callback(channel, callback)
    
    def get_alert_rules(self) -> Dict[str, AlertRule]:
        """Get all configured alert rules."""
        with self.lock:
            return self.alert_rules.copy()