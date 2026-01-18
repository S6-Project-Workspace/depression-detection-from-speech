"""
Session Monitoring Components

Provides session status monitoring, performance tracking, and session
management interface for the clinical dashboard.

Reference: Real-time Streaming Analysis - Task 3.2
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SessionMonitor:
    """
    Session monitoring and management component.
    
    Provides real-time session status, performance metrics,
    and session lifecycle management interface.
    """
    
    def __init__(self):
        """Initialize session monitor component."""
        self.session_data_cache: Dict[str, Dict] = {}
        self.session_history_cache: Dict[str, List[Dict]] = {}
        
        # Mock session data for development
        self._initialize_mock_sessions()
    
    def _initialize_mock_sessions(self):
        """Initialize mock session data for development."""
        mock_sessions = {
            'session_001': {
                'session_id': 'session_001',
                'patient_id': 'patient_001',
                'start_time': datetime.now() - timedelta(minutes=45),
                'status': 'active',
                'current_risk_score': 0.65,
                'language': 'ta',
                'audio_source': 'microphone_1',
                'processing_mode': 'trimodal',
                'total_alerts': 2,
                'acknowledged_alerts': 1,
                'average_latency': 1250.0,
                'audio_quality': 0.85,
                'clinician': 'dr_smith'
            },
            'session_002': {
                'session_id': 'session_002',
                'patient_id': 'patient_002',
                'start_time': datetime.now() - timedelta(minutes=23),
                'status': 'active',
                'current_risk_score': 0.32,
                'language': 'ta',
                'audio_source': 'microphone_2',
                'processing_mode': 'bimodal',
                'total_alerts': 0,
                'acknowledged_alerts': 0,
                'average_latency': 980.0,
                'audio_quality': 0.92,
                'clinician': 'dr_patel'
            },
            'session_003': {
                'session_id': 'session_003',
                'patient_id': 'patient_003',
                'start_time': datetime.now() - timedelta(minutes=12),
                'status': 'paused',
                'current_risk_score': 0.78,
                'language': 'ml',
                'audio_source': 'microphone_3',
                'processing_mode': 'audio_only',
                'total_alerts': 3,
                'acknowledged_alerts': 2,
                'average_latency': 1850.0,
                'audio_quality': 0.67,
                'clinician': 'dr_kumar'
            }
        }
        
        self.session_data_cache.update(mock_sessions)
        
        # Initialize session history
        for session_id in mock_sessions:
            self.session_history_cache[session_id] = self._generate_mock_history(session_id)
    
    def _generate_mock_history(self, session_id: str) -> List[Dict]:
        """Generate mock session history data."""
        import random
        
        history = []
        base_time = datetime.now() - timedelta(minutes=60)
        
        for i in range(20):
            timestamp = base_time + timedelta(minutes=i * 3)
            
            history.append({
                'timestamp': timestamp,
                'event_type': random.choice(['risk_update', 'alert_generated', 'status_change', 'quality_update']),
                'description': f"Mock event {i+1} for {session_id}",
                'risk_score': random.uniform(0.2, 0.9),
                'processing_time': random.uniform(800, 2000),
                'audio_quality': random.uniform(0.6, 0.95)
            })
        
        return history
    
    def update_data(self, data: Dict):
        """
        Update session data cache with new data.
        
        Args:
            data: Session data from WebSocket client
        """
        try:
            for session_id, session_info in data.items():
                if session_id in self.session_data_cache:
                    self.session_data_cache[session_id].update(session_info)
                else:
                    self.session_data_cache[session_id] = session_info
                
                # Add to history
                if session_id not in self.session_history_cache:
                    self.session_history_cache[session_id] = []
                
                history_entry = {
                    'timestamp': datetime.now(),
                    'event_type': 'data_update',
                    'description': 'Session data updated',
                    **session_info
                }
                
                self.session_history_cache[session_id].append(history_entry)
                
                # Keep recent history only
                if len(self.session_history_cache[session_id]) > 100:
                    self.session_history_cache[session_id] = self.session_history_cache[session_id][-100:]
        
        except Exception as e:
            logger.error(f"Error updating session data: {e}")
    
    def render_session_table(self):
        """Render session overview table."""
        try:
            if not self.session_data_cache:
                st.info("No active sessions found.")
                return
            
            # Prepare data for table
            table_data = []
            
            for session_id, session_info in self.session_data_cache.items():
                duration = self._calculate_duration(session_info.get('start_time'))
                
                table_data.append({
                    'Session ID': session_id,
                    'Patient ID': session_info.get('patient_id', 'Unknown'),
                    'Status': session_info.get('status', 'Unknown'),
                    'Duration': duration,
                    'Current Risk': f"{session_info.get('current_risk_score', 0):.3f}",
                    'Risk Level': self._get_risk_level(session_info.get('current_risk_score', 0)),
                    'Alerts': f"{session_info.get('acknowledged_alerts', 0)}/{session_info.get('total_alerts', 0)}",
                    'Latency (ms)': f"{session_info.get('average_latency', 0):.0f}",
                    'Audio Quality': f"{session_info.get('audio_quality', 0):.2f}",
                    'Clinician': session_info.get('clinician', 'Unknown'),
                    'Language': session_info.get('language', 'Unknown').upper()
                })
            
            df = pd.DataFrame(table_data)
            
            # Style the dataframe
            def style_risk_level(val):
                if val == 'HIGH':
                    return 'background-color: #ffebee; color: #c62828'
                elif val == 'MODERATE':
                    return 'background-color: #fff3e0; color: #ef6c00'
                else:
                    return 'background-color: #e8f5e8; color: #2e7d32'
            
            def style_status(val):
                if val == 'active':
                    return 'background-color: #e8f5e8; color: #2e7d32'
                elif val == 'paused':
                    return 'background-color: #fff3e0; color: #ef6c00'
                else:
                    return 'background-color: #ffebee; color: #c62828'
            
            styled_df = df.style.applymap(
                style_risk_level, subset=['Risk Level']
            ).applymap(
                style_status, subset=['Status']
            )
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Session actions
            self._render_session_actions()
            
        except Exception as e:
            logger.error(f"Error rendering session table: {e}")
            st.error("Error rendering session table")
    
    def _render_session_actions(self):
        """Render session action buttons."""
        st.markdown("### Session Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Sessions"):
                st.rerun()
        
        with col2:
            if st.button("➕ New Session"):
                self._show_new_session_dialog()
        
        with col3:
            if st.button("⏸️ Pause All"):
                self._pause_all_sessions()
        
        with col4:
            if st.button("📊 Export Data"):
                self._export_session_data()
    
    def _show_new_session_dialog(self):
        """Show new session creation dialog."""
        with st.expander("Create New Session", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                patient_id = st.text_input("Patient ID")
                language = st.selectbox("Language", ["ta", "ml", "en"])
                audio_source = st.selectbox("Audio Source", ["microphone_1", "microphone_2", "microphone_3"])
            
            with col2:
                processing_mode = st.selectbox("Processing Mode", ["trimodal", "bimodal", "audio_only"])
                risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.8)
                clinician = st.text_input("Clinician ID")
            
            if st.button("Create Session"):
                if patient_id and clinician:
                    # Mock session creation
                    new_session_id = f"session_{len(self.session_data_cache) + 1:03d}"
                    
                    new_session = {
                        'session_id': new_session_id,
                        'patient_id': patient_id,
                        'start_time': datetime.now(),
                        'status': 'initializing',
                        'current_risk_score': 0.0,
                        'language': language,
                        'audio_source': audio_source,
                        'processing_mode': processing_mode,
                        'risk_threshold': risk_threshold,
                        'total_alerts': 0,
                        'acknowledged_alerts': 0,
                        'average_latency': 0.0,
                        'audio_quality': 0.0,
                        'clinician': clinician
                    }
                    
                    self.session_data_cache[new_session_id] = new_session
                    st.success(f"Session {new_session_id} created successfully!")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")
    
    def _pause_all_sessions(self):
        """Pause all active sessions."""
        paused_count = 0
        
        for session_id, session_info in self.session_data_cache.items():
            if session_info.get('status') == 'active':
                session_info['status'] = 'paused'
                paused_count += 1
        
        if paused_count > 0:
            st.success(f"Paused {paused_count} sessions")
            st.rerun()
        else:
            st.info("No active sessions to pause")
    
    def _export_session_data(self):
        """Export session data to CSV."""
        try:
            df = pd.DataFrame.from_dict(self.session_data_cache, orient='index')
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download Session Data CSV",
                data=csv,
                file_name=f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error exporting data: {e}")
    
    def render_session_history(self, session_id: str):
        """
        Render session history for specific session.
        
        Args:
            session_id: Session ID to display history for
        """
        try:
            if session_id not in self.session_history_cache:
                st.info(f"No history available for session {session_id}")
                return
            
            history = self.session_history_cache[session_id]
            
            if not history:
                st.info("No history events found")
                return
            
            # Create timeline visualization
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Event timeline chart
            fig = go.Figure()
            
            # Color map for event types
            color_map = {
                'risk_update': '#2196F3',
                'alert_generated': '#F44336',
                'status_change': '#FF9800',
                'quality_update': '#4CAF50',
                'data_update': '#9C27B0'
            }
            
            for event_type in df['event_type'].unique():
                event_data = df[df['event_type'] == event_type]
                
                fig.add_trace(go.Scatter(
                    x=event_data['timestamp'],
                    y=[event_type] * len(event_data),
                    mode='markers',
                    name=event_type.replace('_', ' ').title(),
                    marker=dict(
                        color=color_map.get(event_type, '#607D8B'),
                        size=10
                    ),
                    text=event_data['description'],
                    hovertemplate='<b>%{text}</b><br>Time: %{x}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f"Session Timeline - {session_id}",
                xaxis_title="Time",
                yaxis_title="Event Type",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Event details table
            st.subheader("Event Details")
            
            # Format data for display
            display_df = df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            display_df = display_df.sort_values('timestamp', ascending=False)
            
            st.dataframe(
                display_df[['timestamp', 'event_type', 'description']],
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            logger.error(f"Error rendering session history: {e}")
            st.error("Error rendering session history")
    
    def get_active_sessions(self) -> Dict[str, Dict]:
        """
        Get all active sessions.
        
        Returns:
            Dictionary of active sessions
        """
        return {
            session_id: session_info
            for session_id, session_info in self.session_data_cache.items()
            if session_info.get('status') == 'active'
        }
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get overall session statistics.
        
        Returns:
            Session statistics dictionary
        """
        if not self.session_data_cache:
            return {}
        
        sessions = list(self.session_data_cache.values())
        
        active_sessions = [s for s in sessions if s.get('status') == 'active']
        high_risk_sessions = [s for s in sessions if s.get('current_risk_score', 0) >= 0.8]
        
        total_alerts = sum(s.get('total_alerts', 0) for s in sessions)
        acknowledged_alerts = sum(s.get('acknowledged_alerts', 0) for s in sessions)
        
        avg_latency = sum(s.get('average_latency', 0) for s in sessions) / len(sessions)
        avg_quality = sum(s.get('audio_quality', 0) for s in sessions) / len(sessions)
        
        return {
            'total_sessions': len(sessions),
            'active_sessions': len(active_sessions),
            'high_risk_sessions': len(high_risk_sessions),
            'total_alerts': total_alerts,
            'acknowledged_alerts': acknowledged_alerts,
            'unacknowledged_alerts': total_alerts - acknowledged_alerts,
            'average_latency': avg_latency,
            'average_audio_quality': avg_quality,
            'alert_acknowledgment_rate': acknowledged_alerts / total_alerts if total_alerts > 0 else 0
        }
    
    def _calculate_duration(self, start_time) -> str:
        """Calculate session duration string."""
        if not start_time:
            return "Unknown"
        
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        
        duration = datetime.now() - start_time
        
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level string."""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.5:
            return "MODERATE"
        else:
            return "LOW"