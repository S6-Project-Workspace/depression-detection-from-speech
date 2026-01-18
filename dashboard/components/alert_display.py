"""
Alert Display Components

Provides alert visualization, management, and acknowledgment interface
for the clinical dashboard.

Reference: Real-time Streaming Analysis - Task 3.2
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from ..auth import streamlit_auth, Permission

logger = logging.getLogger(__name__)


class AlertDisplay:
    """
    Alert display and management component.
    
    Provides real-time alert visualization, acknowledgment interface,
    and alert analytics for clinical staff.
    """
    
    def __init__(self):
        """Initialize alert display component."""
        self.alert_data_cache: Dict[str, List[Dict]] = {'alerts': []}
        self.acknowledged_alerts: List[str] = []
        
        # Initialize mock alerts for development
        self._initialize_mock_alerts()
    
    def _initialize_mock_alerts(self):
        """Initialize mock alert data for development."""
        mock_alerts = [
            {
                'alert_id': 'alert_001',
                'session_id': 'session_001',
                'patient_id': 'patient_001',
                'timestamp': datetime.now() - timedelta(minutes=15),
                'alert_type': 'high_risk',
                'severity': 'high',
                'message': 'Risk score exceeded threshold (0.85)',
                'risk_score': 0.85,
                'trigger_condition': 'risk_threshold',
                'acknowledged': False,
                'acknowledged_by': None,
                'acknowledged_at': None
            },
            {
                'alert_id': 'alert_002',
                'session_id': 'session_003',
                'patient_id': 'patient_003',
                'timestamp': datetime.now() - timedelta(minutes=8),
                'alert_type': 'trend_warning',
                'severity': 'medium',
                'message': 'Rapid risk increase detected (0.45 → 0.78)',
                'risk_score': 0.78,
                'trigger_condition': 'trend_analysis',
                'acknowledged': True,
                'acknowledged_by': 'dr_kumar',
                'acknowledged_at': datetime.now() - timedelta(minutes=5)
            },
            {
                'alert_id': 'alert_003',
                'session_id': 'session_001',
                'patient_id': 'patient_001',
                'timestamp': datetime.now() - timedelta(minutes=3),
                'alert_type': 'audio_quality',
                'severity': 'low',
                'message': 'Audio quality degraded (SNR: 12dB)',
                'risk_score': 0.65,
                'trigger_condition': 'audio_quality',
                'acknowledged': False,
                'acknowledged_by': None,
                'acknowledged_at': None
            }
        ]
        
        self.alert_data_cache['alerts'] = mock_alerts
    
    def update_data(self, data: Dict):
        """
        Update alert data cache with new data.
        
        Args:
            data: Alert data from WebSocket client
        """
        try:
            if 'new_alert' in data:
                # Add new alert
                new_alert = data['new_alert']
                self.alert_data_cache['alerts'].append(new_alert)
                
                # Keep only recent alerts (last 100)
                if len(self.alert_data_cache['alerts']) > 100:
                    self.alert_data_cache['alerts'] = self.alert_data_cache['alerts'][-100:]
            
            if 'alert_updates' in data:
                # Update existing alerts
                for update in data['alert_updates']:
                    alert_id = update.get('alert_id')
                    
                    for alert in self.alert_data_cache['alerts']:
                        if alert['alert_id'] == alert_id:
                            alert.update(update)
                            break
        
        except Exception as e:
            logger.error(f"Error updating alert data: {e}")
    
    def render_alert_summary(self):
        """Render alert summary for overview page."""
        try:
            alerts = self.alert_data_cache['alerts']
            
            if not alerts:
                st.info("No alerts to display")
                return
            
            # Filter active (unacknowledged) alerts
            active_alerts = [a for a in alerts if not a.get('acknowledged', False)]
            
            if not active_alerts:
                st.success("✅ No active alerts")
                return
            
            # Sort by severity and timestamp
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            active_alerts.sort(
                key=lambda x: (
                    severity_order.get(x.get('severity', 'low'), 3),
                    x.get('timestamp', datetime.min)
                ),
                reverse=True
            )
            
            # Display alerts
            for alert in active_alerts[:5]:  # Show top 5 alerts
                self._render_alert_card(alert, compact=True)
            
            # Show more alerts button
            if len(active_alerts) > 5:
                st.info(f"... and {len(active_alerts) - 5} more alerts")
            
            # Quick acknowledge all button
            if streamlit_auth.check_permission(Permission.ACKNOWLEDGE_ALERTS):
                if st.button("🔕 Acknowledge All", key="ack_all_summary"):
                    self._acknowledge_all_alerts()
        
        except Exception as e:
            logger.error(f"Error rendering alert summary: {e}")
            st.error("Error rendering alert summary")
    
    def render_alert_management(self):
        """Render full alert management interface."""
        try:
            st.subheader("🚨 Alert Management")
            
            alerts = self.alert_data_cache['alerts']
            
            if not alerts:
                st.info("No alerts to display")
                return
            
            # Alert filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_acknowledged = st.checkbox("Show Acknowledged", value=False)
            
            with col2:
                severity_filter = st.selectbox(
                    "Filter by Severity",
                    ["All", "Critical", "High", "Medium", "Low"]
                )
            
            with col3:
                session_filter = st.selectbox(
                    "Filter by Session",
                    ["All"] + list(set(a.get('session_id', '') for a in alerts))
                )
            
            # Apply filters
            filtered_alerts = self._filter_alerts(
                alerts, show_acknowledged, severity_filter, session_filter
            )
            
            if not filtered_alerts:
                st.info("No alerts match the current filters")
                return
            
            # Alert statistics
            self._render_alert_statistics(filtered_alerts)
            
            st.markdown("---")
            
            # Alert list
            for alert in filtered_alerts:
                self._render_alert_card(alert, compact=False)
        
        except Exception as e:
            logger.error(f"Error rendering alert management: {e}")
            st.error("Error rendering alert management")
    
    def _render_alert_card(self, alert: Dict, compact: bool = False):
        """
        Render individual alert card.
        
        Args:
            alert: Alert data dictionary
            compact: Whether to render in compact mode
        """
        # Determine alert styling
        severity = alert.get('severity', 'low')
        acknowledged = alert.get('acknowledged', False)
        
        if acknowledged:
            border_color = "#4CAF50"  # Green
            bg_color = "#E8F5E8"
        elif severity == 'critical':
            border_color = "#D32F2F"  # Dark Red
            bg_color = "#FFEBEE"
        elif severity == 'high':
            border_color = "#F44336"  # Red
            bg_color = "#FFEBEE"
        elif severity == 'medium':
            border_color = "#FF9800"  # Orange
            bg_color = "#FFF3E0"
        else:
            border_color = "#2196F3"  # Blue
            bg_color = "#E3F2FD"
        
        # Create alert container
        with st.container():
            st.markdown(f"""
            <div style="
                border-left: 4px solid {border_color};
                background-color: {bg_color};
                padding: 1rem;
                margin-bottom: 0.5rem;
                border-radius: 0.25rem;
            ">
            """, unsafe_allow_html=True)
            
            if compact:
                # Compact layout for summary
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{alert.get('message', 'Unknown alert')}**")
                    st.caption(f"Session: {alert.get('session_id', 'Unknown')} | "
                             f"Risk: {alert.get('risk_score', 0):.3f} | "
                             f"{self._format_timestamp(alert.get('timestamp'))}")
                
                with col2:
                    severity_emoji = self._get_severity_emoji(severity)
                    st.markdown(f"**{severity_emoji} {severity.upper()}**")
                    
                    if not acknowledged and streamlit_auth.check_permission(Permission.ACKNOWLEDGE_ALERTS):
                        if st.button("✓", key=f"ack_{alert['alert_id']}", help="Acknowledge"):
                            self._acknowledge_alert(alert['alert_id'])
            
            else:
                # Full layout for management page
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### {alert.get('message', 'Unknown alert')}")
                    st.markdown(f"**Alert ID**: {alert.get('alert_id', 'Unknown')}")
                    st.markdown(f"**Session**: {alert.get('session_id', 'Unknown')}")
                    st.markdown(f"**Patient**: {alert.get('patient_id', 'Unknown')}")
                    st.markdown(f"**Trigger**: {alert.get('trigger_condition', 'Unknown')}")
                
                with col2:
                    severity_emoji = self._get_severity_emoji(severity)
                    st.markdown(f"**Severity**: {severity_emoji} {severity.upper()}")
                    st.markdown(f"**Risk Score**: {alert.get('risk_score', 0):.3f}")
                    st.markdown(f"**Time**: {self._format_timestamp(alert.get('timestamp'))}")
                
                with col3:
                    if acknowledged:
                        st.success("✅ Acknowledged")
                        st.caption(f"By: {alert.get('acknowledged_by', 'Unknown')}")
                        st.caption(f"At: {self._format_timestamp(alert.get('acknowledged_at'))}")
                    else:
                        if streamlit_auth.check_permission(Permission.ACKNOWLEDGE_ALERTS):
                            if st.button("Acknowledge", key=f"ack_full_{alert['alert_id']}"):
                                self._acknowledge_alert(alert['alert_id'])
                        else:
                            st.warning("⏳ Pending")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def _render_alert_statistics(self, alerts: List[Dict]):
        """
        Render alert statistics.
        
        Args:
            alerts: List of alert dictionaries
        """
        col1, col2, col3, col4 = st.columns(4)
        
        total_alerts = len(alerts)
        acknowledged_alerts = len([a for a in alerts if a.get('acknowledged', False)])
        high_severity_alerts = len([a for a in alerts if a.get('severity') in ['critical', 'high']])
        
        with col1:
            st.metric("Total Alerts", total_alerts)
        
        with col2:
            st.metric("Acknowledged", acknowledged_alerts)
        
        with col3:
            st.metric("High Priority", high_severity_alerts)
        
        with col4:
            ack_rate = (acknowledged_alerts / total_alerts * 100) if total_alerts > 0 else 0
            st.metric("Ack Rate", f"{ack_rate:.1f}%")
    
    def _filter_alerts(self, alerts: List[Dict], show_acknowledged: bool, 
                      severity_filter: str, session_filter: str) -> List[Dict]:
        """
        Filter alerts based on criteria.
        
        Args:
            alerts: List of alert dictionaries
            show_acknowledged: Whether to include acknowledged alerts
            severity_filter: Severity filter
            session_filter: Session filter
            
        Returns:
            Filtered list of alerts
        """
        filtered = alerts.copy()
        
        # Filter by acknowledgment status
        if not show_acknowledged:
            filtered = [a for a in filtered if not a.get('acknowledged', False)]
        
        # Filter by severity
        if severity_filter != "All":
            filtered = [a for a in filtered if a.get('severity', '').lower() == severity_filter.lower()]
        
        # Filter by session
        if session_filter != "All":
            filtered = [a for a in filtered if a.get('session_id') == session_filter]
        
        # Sort by timestamp (newest first)
        filtered.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        return filtered
    
    def _acknowledge_alert(self, alert_id: str):
        """
        Acknowledge a specific alert.
        
        Args:
            alert_id: Alert ID to acknowledge
        """
        user = streamlit_auth.get_current_user()
        if not user:
            st.error("User not authenticated")
            return
        
        # Find and update alert
        for alert in self.alert_data_cache['alerts']:
            if alert['alert_id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_by'] = user.username
                alert['acknowledged_at'] = datetime.now()
                
                self.acknowledged_alerts.append(alert_id)
                st.success(f"Alert {alert_id} acknowledged")
                st.rerun()
                break
        else:
            st.error(f"Alert {alert_id} not found")
    
    def _acknowledge_all_alerts(self):
        """Acknowledge all unacknowledged alerts."""
        user = streamlit_auth.get_current_user()
        if not user:
            st.error("User not authenticated")
            return
        
        acknowledged_count = 0
        
        for alert in self.alert_data_cache['alerts']:
            if not alert.get('acknowledged', False):
                alert['acknowledged'] = True
                alert['acknowledged_by'] = user.username
                alert['acknowledged_at'] = datetime.now()
                
                self.acknowledged_alerts.append(alert['alert_id'])
                acknowledged_count += 1
        
        if acknowledged_count > 0:
            st.success(f"Acknowledged {acknowledged_count} alerts")
            st.rerun()
        else:
            st.info("No alerts to acknowledge")
    
    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for severity level."""
        emoji_map = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🔵'
        }
        return emoji_map.get(severity.lower(), '⚪')
    
    def _format_timestamp(self, timestamp) -> str:
        """Format timestamp for display."""
        if not timestamp:
            return "Unknown"
        
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        now = datetime.now()
        diff = now - timestamp
        
        if diff.total_seconds() < 60:
            return "Just now"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            return timestamp.strftime("%m/%d %H:%M")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Alert statistics dictionary
        """
        alerts = self.alert_data_cache['alerts']
        
        if not alerts:
            return {}
        
        total_alerts = len(alerts)
        acknowledged_alerts = len([a for a in alerts if a.get('acknowledged', False)])
        
        severity_counts = {}
        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recent alerts (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = [
            a for a in alerts 
            if a.get('timestamp', datetime.min) > recent_cutoff
        ]
        
        return {
            'total_alerts': total_alerts,
            'acknowledged_alerts': acknowledged_alerts,
            'unacknowledged_alerts': total_alerts - acknowledged_alerts,
            'acknowledgment_rate': acknowledged_alerts / total_alerts if total_alerts > 0 else 0,
            'severity_counts': severity_counts,
            'recent_alerts_24h': len(recent_alerts),
            'high_priority_alerts': len([a for a in alerts if a.get('severity') in ['critical', 'high']])
        }