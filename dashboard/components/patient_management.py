"""
Patient Management Components

Provides patient overview, session history, and clinical workflow
integration for the dashboard.

Reference: Real-time Streaming Analysis - Task 3.3
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


class PatientManager:
    """
    Patient management and clinical workflow component.
    
    Provides patient overview, session history, clinical notes,
    and care coordination features.
    """
    
    def __init__(self):
        """Initialize patient manager component."""
        self.patient_data_cache: Dict[str, Dict] = {}
        self.clinical_notes_cache: Dict[str, List[Dict]] = {}
        
        # Initialize mock patient data
        self._initialize_mock_patients()
    
    def _initialize_mock_patients(self):
        """Initialize mock patient data for development."""
        mock_patients = {
            'patient_001': {
                'patient_id': 'patient_001',
                'display_name': 'Patient A',  # Anonymized
                'age_group': '25-35',
                'language': 'Tamil',
                'enrollment_date': datetime.now() - timedelta(days=30),
                'total_sessions': 8,
                'active_sessions': 1,
                'last_session': datetime.now() - timedelta(minutes=45),
                'average_risk_score': 0.65,
                'highest_risk_score': 0.89,
                'total_alerts': 12,
                'acknowledged_alerts': 10,
                'primary_clinician': 'dr_smith',
                'status': 'active_monitoring',
                'risk_trend': 'increasing',
                'notes_count': 5
            },
            'patient_002': {
                'patient_id': 'patient_002',
                'display_name': 'Patient B',
                'age_group': '35-45',
                'language': 'Tamil',
                'enrollment_date': datetime.now() - timedelta(days=15),
                'total_sessions': 4,
                'active_sessions': 1,
                'last_session': datetime.now() - timedelta(minutes=23),
                'average_risk_score': 0.32,
                'highest_risk_score': 0.48,
                'total_alerts': 2,
                'acknowledged_alerts': 2,
                'primary_clinician': 'dr_patel',
                'status': 'routine_monitoring',
                'risk_trend': 'stable',
                'notes_count': 2
            },
            'patient_003': {
                'patient_id': 'patient_003',
                'display_name': 'Patient C',
                'age_group': '45-55',
                'language': 'Malayalam',
                'enrollment_date': datetime.now() - timedelta(days=7),
                'total_sessions': 2,
                'active_sessions': 0,
                'last_session': datetime.now() - timedelta(minutes=12),
                'average_risk_score': 0.78,
                'highest_risk_score': 0.92,
                'total_alerts': 6,
                'acknowledged_alerts': 4,
                'primary_clinician': 'dr_kumar',
                'status': 'high_risk_monitoring',
                'risk_trend': 'concerning',
                'notes_count': 8
            }
        }
        
        self.patient_data_cache.update(mock_patients)
        
        # Initialize clinical notes
        for patient_id in mock_patients:
            self.clinical_notes_cache[patient_id] = self._generate_mock_notes(patient_id)
    
    def _generate_mock_notes(self, patient_id: str) -> List[Dict]:
        """Generate mock clinical notes."""
        notes = [
            {
                'note_id': f"note_{patient_id}_001",
                'timestamp': datetime.now() - timedelta(days=2),
                'clinician': 'dr_smith',
                'note_type': 'assessment',
                'content': 'Initial assessment completed. Patient shows signs of mild depression. Recommended continued monitoring.',
                'risk_level': 'moderate',
                'follow_up_required': True,
                'follow_up_date': datetime.now() + timedelta(days=7)
            },
            {
                'note_id': f"note_{patient_id}_002",
                'timestamp': datetime.now() - timedelta(hours=6),
                'clinician': 'dr_patel',
                'note_type': 'session_review',
                'content': 'Reviewed recent session data. Risk scores showing improvement. Continue current monitoring protocol.',
                'risk_level': 'low',
                'follow_up_required': False,
                'follow_up_date': None
            }
        ]
        
        return notes
    
    def render_patient_overview(self):
        """Render patient overview interface."""
        try:
            # Check permissions
            if not streamlit_auth.check_permission(Permission.VIEW_PATIENT_DATA):
                st.error("Access denied. Insufficient permissions.")
                return
            
            st.subheader("👥 Patient Overview")
            
            # Patient summary metrics
            self._render_patient_metrics()
            
            st.markdown("---")
            
            # Patient selection and details
            col1, col2 = st.columns([1, 2])
            
            with col1:
                self._render_patient_list()
            
            with col2:
                selected_patient = st.session_state.get('selected_patient')
                if selected_patient:
                    self._render_patient_details(selected_patient)
                else:
                    st.info("Select a patient to view details")
        
        except Exception as e:
            logger.error(f"Error rendering patient overview: {e}")
            st.error("Error rendering patient overview")
    
    def _render_patient_metrics(self):
        """Render overall patient metrics."""
        patients = list(self.patient_data_cache.values())
        
        if not patients:
            st.info("No patient data available")
            return
        
        # Calculate metrics
        total_patients = len(patients)
        active_patients = len([p for p in patients if p.get('active_sessions', 0) > 0])
        high_risk_patients = len([p for p in patients if p.get('average_risk_score', 0) >= 0.7])
        
        total_sessions = sum(p.get('total_sessions', 0) for p in patients)
        total_alerts = sum(p.get('total_alerts', 0) for p in patients)
        acknowledged_alerts = sum(p.get('acknowledged_alerts', 0) for p in patients)
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Patients", total_patients)
        
        with col2:
            st.metric("Active Sessions", active_patients)
        
        with col3:
            st.metric("High Risk", high_risk_patients, 
                     delta=f"{high_risk_patients/total_patients*100:.1f}%" if total_patients > 0 else "0%")
        
        with col4:
            st.metric("Total Sessions", total_sessions)
        
        with col5:
            ack_rate = (acknowledged_alerts / total_alerts * 100) if total_alerts > 0 else 0
            st.metric("Alert Ack Rate", f"{ack_rate:.1f}%")
    
    def _render_patient_list(self):
        """Render patient selection list."""
        st.markdown("### Patient List")
        
        patients = list(self.patient_data_cache.values())
        
        if not patients:
            st.info("No patients found")
            return
        
        # Sort patients by risk level and last session
        patients.sort(key=lambda p: (
            -p.get('average_risk_score', 0),
            -(p.get('last_session', datetime.min).timestamp() if p.get('last_session') else 0)
        ))
        
        # Display patient cards
        for patient in patients:
            self._render_patient_card(patient)
    
    def _render_patient_card(self, patient: Dict):
        """
        Render individual patient card.
        
        Args:
            patient: Patient data dictionary
        """
        patient_id = patient['patient_id']
        display_name = patient.get('display_name', patient_id)
        risk_score = patient.get('average_risk_score', 0)
        status = patient.get('status', 'unknown')
        
        # Determine card styling based on risk level
        if risk_score >= 0.7:
            border_color = "#F44336"  # Red
            bg_color = "#FFEBEE"
        elif risk_score >= 0.5:
            border_color = "#FF9800"  # Orange
            bg_color = "#FFF3E0"
        else:
            border_color = "#4CAF50"  # Green
            bg_color = "#E8F5E8"
        
        # Create clickable patient card
        with st.container():
            if st.button(
                f"{display_name}",
                key=f"patient_select_{patient_id}",
                help=f"Risk: {risk_score:.3f} | Status: {status}"
            ):
                st.session_state['selected_patient'] = patient_id
                st.rerun()
            
            # Patient summary info
            st.markdown(f"""
            <div style="
                border-left: 4px solid {border_color};
                background-color: {bg_color};
                padding: 0.5rem;
                margin-bottom: 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.8rem;
            ">
                <strong>Risk:</strong> {risk_score:.3f}<br>
                <strong>Sessions:</strong> {patient.get('total_sessions', 0)}<br>
                <strong>Alerts:</strong> {patient.get('total_alerts', 0)}<br>
                <strong>Status:</strong> {status.replace('_', ' ').title()}
            </div>
            """, unsafe_allow_html=True)
    
    def _render_patient_details(self, patient_id: str):
        """
        Render detailed patient information.
        
        Args:
            patient_id: Patient ID to display
        """
        if patient_id not in self.patient_data_cache:
            st.error(f"Patient {patient_id} not found")
            return
        
        patient = self.patient_data_cache[patient_id]
        
        st.markdown(f"### {patient.get('display_name', patient_id)} Details")
        
        # Patient details tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview", "Session History", "Clinical Notes", "Risk Analysis"
        ])
        
        with tab1:
            self._render_patient_overview_tab(patient)
        
        with tab2:
            self._render_session_history_tab(patient_id)
        
        with tab3:
            self._render_clinical_notes_tab(patient_id)
        
        with tab4:
            self._render_risk_analysis_tab(patient_id)
    
    def _render_patient_overview_tab(self, patient: Dict):
        """Render patient overview tab."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            st.markdown(f"**Patient ID**: {patient['patient_id']}")
            st.markdown(f"**Age Group**: {patient.get('age_group', 'Unknown')}")
            st.markdown(f"**Language**: {patient.get('language', 'Unknown')}")
            st.markdown(f"**Enrollment**: {self._format_date(patient.get('enrollment_date'))}")
            st.markdown(f"**Primary Clinician**: {patient.get('primary_clinician', 'Unknown')}")
        
        with col2:
            st.markdown("#### Current Status")
            st.markdown(f"**Status**: {patient.get('status', 'Unknown').replace('_', ' ').title()}")
            st.markdown(f"**Risk Trend**: {patient.get('risk_trend', 'Unknown').title()}")
            st.markdown(f"**Last Session**: {self._format_timestamp(patient.get('last_session'))}")
            st.markdown(f"**Active Sessions**: {patient.get('active_sessions', 0)}")
        
        # Risk metrics
        st.markdown("#### Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Risk",
                f"{patient.get('average_risk_score', 0):.3f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Highest Risk",
                f"{patient.get('highest_risk_score', 0):.3f}",
                delta=None
            )
        
        with col3:
            alert_rate = (patient.get('acknowledged_alerts', 0) / 
                         patient.get('total_alerts', 1) * 100)
            st.metric(
                "Alert Response",
                f"{alert_rate:.1f}%",
                delta=None
            )
    
    def _render_session_history_tab(self, patient_id: str):
        """Render session history tab."""
        st.markdown("#### Session History")
        
        # Mock session history data
        sessions = [
            {
                'session_id': f'session_{i:03d}',
                'start_time': datetime.now() - timedelta(days=i*2, hours=i),
                'duration': f"{30 + i*5}m",
                'risk_score': 0.3 + (i * 0.1),
                'alerts': i % 3,
                'status': 'completed' if i > 0 else 'active'
            }
            for i in range(5)
        ]
        
        # Create session table
        df = pd.DataFrame(sessions)
        df['start_time'] = df['start_time'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Session trend chart
        if len(sessions) > 1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(sessions))),
                y=[s['risk_score'] for s in sessions],
                mode='lines+markers',
                name='Risk Score Trend',
                line=dict(color='#F44336')
            ))
            
            fig.update_layout(
                title="Risk Score Trend Over Sessions",
                xaxis_title="Session Number",
                yaxis_title="Risk Score",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_clinical_notes_tab(self, patient_id: str):
        """Render clinical notes tab."""
        # Check permissions for clinical notes
        if not streamlit_auth.check_permission(Permission.CLINICAL_NOTES):
            st.error("Access denied. Clinical notes permission required.")
            return
        
        st.markdown("#### Clinical Notes")
        
        notes = self.clinical_notes_cache.get(patient_id, [])
        
        # Add new note form
        with st.expander("Add New Note"):
            note_type = st.selectbox("Note Type", ["assessment", "session_review", "follow_up", "other"])
            content = st.text_area("Note Content")
            risk_level = st.selectbox("Risk Level", ["low", "moderate", "high"])
            follow_up = st.checkbox("Follow-up Required")
            
            if st.button("Add Note"):
                if content:
                    new_note = {
                        'note_id': f"note_{patient_id}_{len(notes)+1:03d}",
                        'timestamp': datetime.now(),
                        'clinician': streamlit_auth.get_current_user().username,
                        'note_type': note_type,
                        'content': content,
                        'risk_level': risk_level,
                        'follow_up_required': follow_up,
                        'follow_up_date': datetime.now() + timedelta(days=7) if follow_up else None
                    }
                    
                    notes.append(new_note)
                    self.clinical_notes_cache[patient_id] = notes
                    st.success("Note added successfully")
                    st.rerun()
                else:
                    st.error("Please enter note content")
        
        # Display existing notes
        if notes:
            for note in sorted(notes, key=lambda x: x['timestamp'], reverse=True):
                self._render_clinical_note(note)
        else:
            st.info("No clinical notes found")
    
    def _render_clinical_note(self, note: Dict):
        """Render individual clinical note."""
        risk_colors = {
            'low': '#4CAF50',
            'moderate': '#FF9800',
            'high': '#F44336'
        }
        
        risk_color = risk_colors.get(note.get('risk_level', 'low'), '#4CAF50')
        
        with st.container():
            st.markdown(f"""
            <div style="
                border-left: 4px solid {risk_color};
                background-color: #f8f9fa;
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 0.25rem;
            ">
                <strong>{note.get('note_type', 'Note').replace('_', ' ').title()}</strong> - 
                {self._format_timestamp(note.get('timestamp'))}<br>
                <em>By: {note.get('clinician', 'Unknown')}</em><br><br>
                {note.get('content', 'No content')}
            </div>
            """, unsafe_allow_html=True)
            
            if note.get('follow_up_required'):
                st.warning(f"⏰ Follow-up required by: {self._format_date(note.get('follow_up_date'))}")
    
    def _render_risk_analysis_tab(self, patient_id: str):
        """Render risk analysis tab."""
        st.markdown("#### Risk Analysis")
        
        patient = self.patient_data_cache.get(patient_id, {})
        
        # Risk summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_risk = patient.get('average_risk_score', 0)
            risk_level = self._get_risk_level(current_risk)
            st.metric("Current Risk Level", risk_level, delta=f"{current_risk:.3f}")
        
        with col2:
            trend = patient.get('risk_trend', 'stable')
            trend_emoji = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}.get(trend, '➡️')
            st.metric("Risk Trend", f"{trend_emoji} {trend.title()}")
        
        with col3:
            total_alerts = patient.get('total_alerts', 0)
            st.metric("Total Alerts", total_alerts)
        
        # Mock risk timeline
        st.markdown("#### Risk Timeline (Last 30 Days)")
        
        # Generate mock timeline data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        risk_scores = [0.3 + 0.4 * abs(np.sin(i/5)) + np.random.normal(0, 0.05) for i in range(30)]
        
        import numpy as np
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=risk_scores,
            mode='lines+markers',
            name='Daily Risk Score',
            line=dict(color='#F44336')
        ))
        
        # Add risk thresholds
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="High Risk")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")
        
        fig.update_layout(
            title="30-Day Risk Score Timeline",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            yaxis=dict(range=[0, 1]),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level string."""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.5:
            return "MODERATE"
        else:
            return "LOW"
    
    def _format_date(self, date) -> str:
        """Format date for display."""
        if not date:
            return "Unknown"
        
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        
        return date.strftime("%Y-%m-%d")
    
    def _format_timestamp(self, timestamp) -> str:
        """Format timestamp for display."""
        if not timestamp:
            return "Unknown"
        
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return timestamp.strftime("%Y-%m-%d %H:%M")