"""
Main Clinical Dashboard Application

Streamlit-based web interface for real-time monitoring of depression detection
sessions, risk assessment, and clinical workflow management.

Key Features:
- Real-time risk score visualization
- Multi-patient session monitoring
- Alert management and acknowledgment
- Clinical workflow integration
- Role-based access control

Reference: Real-time Streaming Analysis - Task 3.1
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Any

# Import dashboard components
from .auth import streamlit_auth, Permission
from .websocket_client import MockWebSocketClient
from .components.risk_charts import RiskVisualization
from .components.session_monitor import SessionMonitor
from .components.alert_display import AlertDisplay
from .components.patient_management import PatientManager

logger = logging.getLogger(__name__)


class ClinicalDashboard:
    """
    Main clinical dashboard application.
    
    Coordinates all dashboard components and manages the overall
    user interface and data flow.
    """
    
    def __init__(self):
        """Initialize the clinical dashboard."""
        self.websocket_client = None
        self.risk_viz = RiskVisualization()
        self.session_monitor = SessionMonitor()
        self.alert_display = AlertDisplay()
        self.patient_manager = PatientManager()
        
        # Dashboard state
        self.selected_session = None
        self.auto_refresh = True
        self.refresh_interval = 2  # seconds
        
        # Initialize WebSocket client
        self._initialize_websocket()
    
    def _initialize_websocket(self):
        """Initialize WebSocket client for real-time data."""
        try:
            # Use mock client for development
            self.websocket_client = MockWebSocketClient(update_interval=1.0)
            
            # Add data handlers
            self.websocket_client.add_data_handler('risk_data', self._handle_risk_data)
            self.websocket_client.add_data_handler('alert_data', self._handle_alert_data)
            self.websocket_client.add_data_handler('session_data', self._handle_session_data)
            
            # Start client
            self.websocket_client.start()
            logger.info("WebSocket client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket client: {e}")
            st.error("Failed to connect to real-time data stream")
    
    def _handle_risk_data(self, data: Dict):
        """Handle incoming risk data."""
        # Update risk visualization
        self.risk_viz.update_data(data)
    
    def _handle_alert_data(self, data: Dict):
        """Handle incoming alert data."""
        # Update alert display
        self.alert_display.update_data(data)
    
    def _handle_session_data(self, data: Dict):
        """Handle incoming session data."""
        # Update session monitor
        self.session_monitor.update_data(data)
    
    def run(self):
        """Run the main dashboard application."""
        # Configure Streamlit page
        st.set_page_config(
            page_title="Clinical Dashboard",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._apply_custom_css()
        
        # Authentication check
        if not streamlit_auth.login_form():
            return
        
        # Main dashboard interface
        self._render_dashboard()
    
    def _apply_custom_css(self):
        """Apply custom CSS styling."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin-bottom: 1rem;
        }
        
        .alert-high {
            background-color: #ffebee;
            border-left-color: #f44336;
        }
        
        .alert-medium {
            background-color: #fff3e0;
            border-left-color: #ff9800;
        }
        
        .status-active {
            color: #4caf50;
            font-weight: bold;
        }
        
        .status-inactive {
            color: #9e9e9e;
        }
        
        .risk-high {
            color: #f44336;
            font-weight: bold;
        }
        
        .risk-moderate {
            color: #ff9800;
            font-weight: bold;
        }
        
        .risk-low {
            color: #4caf50;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_dashboard(self):
        """Render the main dashboard interface."""
        # Header
        st.markdown('<h1 class="main-header">🏥 Clinical Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # User info in sidebar
        streamlit_auth.display_user_info()
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content area
        self._render_main_content()
        
        # Auto-refresh mechanism
        if self.auto_refresh:
            time.sleep(self.refresh_interval)
            st.rerun()
    
    def _render_sidebar(self):
        """Render sidebar navigation and controls."""
        with st.sidebar:
            st.markdown("## Navigation")
            
            # Page selection
            page = st.selectbox(
                "Select View",
                ["Overview", "Session Details", "Patient Management", "System Status"],
                key="page_selector"
            )
            
            st.session_state['current_page'] = page
            
            st.markdown("---")
            
            # Refresh controls
            st.markdown("## Refresh Settings")
            
            self.auto_refresh = st.checkbox("Auto Refresh", value=True)
            
            if self.auto_refresh:
                self.refresh_interval = st.slider(
                    "Refresh Interval (seconds)", 
                    min_value=1, 
                    max_value=10, 
                    value=2
                )
            
            if st.button("Manual Refresh"):
                st.rerun()
            
            st.markdown("---")
            
            # Connection status
            self._display_connection_status()
    
    def _display_connection_status(self):
        """Display WebSocket connection status."""
        st.markdown("## Connection Status")
        
        if self.websocket_client:
            status = self.websocket_client.get_connection_status()
            
            if status['connected']:
                st.success("🟢 Connected")
            else:
                st.error("🔴 Disconnected")
            
            st.caption(f"Server: {status['server_url']}")
            
            if status['reconnect_count'] > 0:
                st.warning(f"Reconnections: {status['reconnect_count']}")
        else:
            st.error("🔴 Client Not Initialized")
    
    def _render_main_content(self):
        """Render main content based on selected page."""
        page = st.session_state.get('current_page', 'Overview')
        
        if page == "Overview":
            self._render_overview_page()
        elif page == "Session Details":
            self._render_session_details_page()
        elif page == "Patient Management":
            self._render_patient_management_page()
        elif page == "System Status":
            self._render_system_status_page()
    
    def _render_overview_page(self):
        """Render overview page with key metrics and alerts."""
        # Check permissions
        if not streamlit_auth.check_permission(Permission.VIEW_SESSIONS):
            st.error("Access denied. Insufficient permissions.")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active Sessions",
                value="3",
                delta="1"
            )
        
        with col2:
            st.metric(
                label="High Risk Patients",
                value="1",
                delta="-1"
            )
        
        with col3:
            st.metric(
                label="Active Alerts",
                value="2",
                delta="0"
            )
        
        with col4:
            st.metric(
                label="System Health",
                value="98%",
                delta="2%"
            )
        
        st.markdown("---")
        
        # Main content columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk visualization
            st.subheader("📊 Real-time Risk Monitoring")
            self.risk_viz.render_overview_chart()
        
        with col2:
            # Active alerts
            st.subheader("🚨 Active Alerts")
            self.alert_display.render_alert_summary()
        
        # Session overview
        st.markdown("---")
        st.subheader("👥 Session Overview")
        self.session_monitor.render_session_table()
    
    def _render_session_details_page(self):
        """Render detailed session monitoring page."""
        # Check permissions
        if not streamlit_auth.check_permission(Permission.VIEW_SESSIONS):
            st.error("Access denied. Insufficient permissions.")
            return
        
        st.subheader("📋 Session Details")
        
        # Session selector
        sessions = self.session_monitor.get_active_sessions()
        if sessions:
            selected_session = st.selectbox(
                "Select Session",
                options=list(sessions.keys()),
                format_func=lambda x: f"Session {x} - Patient {sessions[x].get('patient_id', 'Unknown')}"
            )
            
            if selected_session:
                self.selected_session = selected_session
                
                # Session details tabs
                tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Linguistic Markers", "Session History"])
                
                with tab1:
                    self.risk_viz.render_detailed_chart(selected_session)
                
                with tab2:
                    self.risk_viz.render_linguistic_markers(selected_session)
                
                with tab3:
                    self.session_monitor.render_session_history(selected_session)
        else:
            st.info("No active sessions found.")
    
    def _render_patient_management_page(self):
        """Render patient management page."""
        # Check permissions
        if not streamlit_auth.check_permission(Permission.VIEW_PATIENT_DATA):
            st.error("Access denied. Insufficient permissions.")
            return
        
        st.subheader("👤 Patient Management")
        self.patient_manager.render_patient_overview()
    
    def _render_system_status_page(self):
        """Render system status and diagnostics page."""
        # Check permissions
        if not streamlit_auth.check_permission(Permission.SYSTEM_ADMIN):
            st.error("Access denied. Administrator permissions required.")
            return
        
        st.subheader("⚙️ System Status")
        
        # System metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance Metrics")
            
            # Mock performance data
            perf_data = {
                'CPU Usage': '45%',
                'Memory Usage': '62%',
                'GPU Usage': '78%',
                'Disk Usage': '34%'
            }
            
            for metric, value in perf_data.items():
                st.metric(metric, value)
        
        with col2:
            st.markdown("### System Health")
            
            health_data = {
                'WebSocket Server': '🟢 Online',
                'Database': '🟢 Connected',
                'Model Server': '🟢 Running',
                'Alert System': '🟢 Active'
            }
            
            for component, status in health_data.items():
                st.markdown(f"**{component}**: {status}")
        
        # Recent logs
        st.markdown("---")
        st.subheader("📋 Recent System Logs")
        
        # Mock log data
        log_data = [
            {"timestamp": "2024-01-18 14:30:15", "level": "INFO", "message": "Session session_001 started"},
            {"timestamp": "2024-01-18 14:29:45", "level": "WARNING", "message": "High risk detected for patient_002"},
            {"timestamp": "2024-01-18 14:29:12", "level": "INFO", "message": "Alert acknowledged by dr_smith"},
            {"timestamp": "2024-01-18 14:28:33", "level": "ERROR", "message": "Temporary model loading failure, using fallback"},
            {"timestamp": "2024-01-18 14:27:58", "level": "INFO", "message": "WebSocket client connected"}
        ]
        
        df = pd.DataFrame(log_data)
        st.dataframe(df, use_container_width=True)
    
    def cleanup(self):
        """Cleanup resources when dashboard is closed."""
        if self.websocket_client:
            self.websocket_client.stop()


def main():
    """Main entry point for the dashboard application."""
    dashboard = ClinicalDashboard()
    
    try:
        dashboard.run()
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        st.error(f"Dashboard error: {e}")
    finally:
        dashboard.cleanup()


if __name__ == "__main__":
    main()