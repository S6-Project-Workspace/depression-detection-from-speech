"""
Risk Visualization Components

Provides real-time risk score visualization, trend analysis, and linguistic
marker displays for the clinical dashboard.

Reference: Real-time Streaming Analysis - Task 3.2
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RiskVisualization:
    """
    Real-time risk score visualization component.
    
    Provides interactive charts for risk monitoring, trend analysis,
    and linguistic marker interpretation.
    """
    
    def __init__(self):
        """Initialize risk visualization component."""
        self.risk_data_cache: Dict[str, List[Dict]] = {}
        self.linguistic_data_cache: Dict[str, List[Dict]] = {}
        
        # Chart configuration
        self.chart_config = {
            'displayModeBar': False,
            'staticPlot': False,
            'responsive': True
        }
        
        # Color scheme
        self.colors = {
            'low_risk': '#4CAF50',      # Green
            'moderate_risk': '#FF9800',  # Orange
            'high_risk': '#F44336',     # Red
            'confidence': '#2196F3',    # Blue
            'trend': '#9C27B0'          # Purple
        }
    
    def update_data(self, data: Dict):
        """
        Update risk data cache with new data.
        
        Args:
            data: Risk data from WebSocket client
        """
        try:
            for session_id, risk_point in data.items():
                if session_id not in self.risk_data_cache:
                    self.risk_data_cache[session_id] = []
                
                # Add new data point
                self.risk_data_cache[session_id].append(risk_point)
                
                # Keep only recent data (last 100 points)
                if len(self.risk_data_cache[session_id]) > 100:
                    self.risk_data_cache[session_id] = self.risk_data_cache[session_id][-100:]
                
                # Update linguistic data if available
                if 'linguistic_markers' in risk_point and risk_point['linguistic_markers']:
                    if session_id not in self.linguistic_data_cache:
                        self.linguistic_data_cache[session_id] = []
                    
                    linguistic_point = {
                        'timestamp': risk_point['timestamp'],
                        'markers': risk_point['linguistic_markers']
                    }
                    self.linguistic_data_cache[session_id].append(linguistic_point)
                    
                    # Keep recent linguistic data
                    if len(self.linguistic_data_cache[session_id]) > 50:
                        self.linguistic_data_cache[session_id] = self.linguistic_data_cache[session_id][-50:]
        
        except Exception as e:
            logger.error(f"Error updating risk data: {e}")
    
    def render_overview_chart(self):
        """Render overview chart showing all active sessions."""
        try:
            if not self.risk_data_cache:
                st.info("No risk data available. Waiting for real-time updates...")
                return
            
            # Create subplot for multiple sessions
            fig = make_subplots(
                rows=len(self.risk_data_cache),
                cols=1,
                subplot_titles=[f"Session {sid}" for sid in self.risk_data_cache.keys()],
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            for i, (session_id, data_points) in enumerate(self.risk_data_cache.items(), 1):
                if not data_points:
                    continue
                
                # Prepare data
                df = pd.DataFrame(data_points)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Risk score line
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['risk_score'],
                        mode='lines+markers',
                        name=f'Risk Score - {session_id}',
                        line=dict(color=self._get_risk_color(df['risk_score'].iloc[-1])),
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )
                
                # Risk threshold lines
                fig.add_hline(
                    y=0.8, line_dash="dash", line_color="red",
                    annotation_text="High Risk", row=i, col=1
                )
                fig.add_hline(
                    y=0.5, line_dash="dash", line_color="orange",
                    annotation_text="Moderate Risk", row=i, col=1
                )
            
            # Update layout
            fig.update_layout(
                title="Real-time Risk Monitoring - All Sessions",
                height=200 * len(self.risk_data_cache),
                showlegend=True,
                template="plotly_white"
            )
            
            fig.update_yaxes(title_text="Risk Score", range=[0, 1])
            fig.update_xaxes(title_text="Time")
            
            st.plotly_chart(fig, use_container_width=True, config=self.chart_config)
            
        except Exception as e:
            logger.error(f"Error rendering overview chart: {e}")
            st.error("Error rendering risk overview chart")
    
    def render_detailed_chart(self, session_id: str):
        """
        Render detailed chart for specific session.
        
        Args:
            session_id: Session ID to display
        """
        try:
            if session_id not in self.risk_data_cache or not self.risk_data_cache[session_id]:
                st.info(f"No data available for session {session_id}")
                return
            
            data_points = self.risk_data_cache[session_id]
            df = pd.DataFrame(data_points)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create detailed chart with multiple metrics
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    "Risk Score with Confidence",
                    "Modality Contributions",
                    "Processing Performance"
                ],
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Risk score with confidence bands
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['risk_score'],
                    mode='lines+markers',
                    name='Risk Score',
                    line=dict(color=self.colors['high_risk'], width=3)
                ),
                row=1, col=1
            )
            
            # Confidence intervals
            if 'confidence' in df.columns:
                upper_bound = np.minimum(df['risk_score'] + (1 - df['confidence']) * 0.1, 1.0)
                lower_bound = np.maximum(df['risk_score'] - (1 - df['confidence']) * 0.1, 0.0)
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Confidence Band',
                        fillcolor='rgba(33, 150, 243, 0.2)'
                    ),
                    row=1, col=1
                )
            
            # Modality contributions
            if 'audio_contribution' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['audio_contribution'],
                        mode='lines',
                        name='Audio',
                        line=dict(color='#FF5722')
                    ),
                    row=2, col=1
                )
            
            if 'text_contribution' in df.columns and df['text_contribution'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['text_contribution'],
                        mode='lines',
                        name='Text',
                        line=dict(color='#4CAF50')
                    ),
                    row=2, col=1
                )
            
            if 'linguistic_contribution' in df.columns and df['linguistic_contribution'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['linguistic_contribution'],
                        mode='lines',
                        name='Linguistic',
                        line=dict(color='#9C27B0')
                    ),
                    row=2, col=1
                )
            
            # Processing performance
            if 'processing_time_ms' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['processing_time_ms'],
                        mode='lines+markers',
                        name='Processing Time',
                        line=dict(color='#607D8B')
                    ),
                    row=3, col=1
                )
                
                # Add latency threshold line
                fig.add_hline(
                    y=2000, line_dash="dash", line_color="red",
                    annotation_text="Latency Target (2s)", row=3, col=1
                )
            
            # Add risk threshold lines to first subplot
            fig.add_hline(
                y=0.8, line_dash="dash", line_color="red",
                annotation_text="High Risk", row=1, col=1
            )
            fig.add_hline(
                y=0.5, line_dash="dash", line_color="orange",
                annotation_text="Moderate Risk", row=1, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"Detailed Analysis - Session {session_id}",
                height=800,
                showlegend=True,
                template="plotly_white"
            )
            
            fig.update_yaxes(title_text="Risk Score", range=[0, 1], row=1, col=1)
            fig.update_yaxes(title_text="Contribution", range=[0, 1], row=2, col=1)
            fig.update_yaxes(title_text="Time (ms)", row=3, col=1)
            fig.update_xaxes(title_text="Time", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True, config=self.chart_config)
            
            # Display current metrics
            self._display_current_metrics(df.iloc[-1] if not df.empty else None)
            
        except Exception as e:
            logger.error(f"Error rendering detailed chart: {e}")
            st.error("Error rendering detailed risk chart")
    
    def render_linguistic_markers(self, session_id: str):
        """
        Render linguistic markers visualization.
        
        Args:
            session_id: Session ID to display
        """
        try:
            if session_id not in self.linguistic_data_cache or not self.linguistic_data_cache[session_id]:
                st.info(f"No linguistic data available for session {session_id}")
                return
            
            linguistic_points = self.linguistic_data_cache[session_id]
            
            # Get latest linguistic markers
            latest_markers = linguistic_points[-1]['markers'] if linguistic_points else {}
            
            if not latest_markers:
                st.info("No linguistic markers available")
                return
            
            # Create radar chart for linguistic markers
            categories = list(latest_markers.keys())
            values = list(latest_markers.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Linguistic Markers',
                line_color='rgb(156, 39, 176)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Current Linguistic Markers",
                showlegend=True,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True, config=self.chart_config)
            
            # Display marker details
            st.subheader("Linguistic Marker Details")
            
            col1, col2 = st.columns(2)
            
            for i, (marker, value) in enumerate(latest_markers.items()):
                col = col1 if i % 2 == 0 else col2
                
                with col:
                    # Color code based on value
                    if value >= 0.7:
                        color = "🔴"
                    elif value >= 0.4:
                        color = "🟡"
                    else:
                        color = "🟢"
                    
                    st.metric(
                        label=f"{color} {marker.replace('_', ' ').title()}",
                        value=f"{value:.3f}",
                        delta=None
                    )
            
        except Exception as e:
            logger.error(f"Error rendering linguistic markers: {e}")
            st.error("Error rendering linguistic markers")
    
    def _display_current_metrics(self, current_data: Optional[pd.Series]):
        """
        Display current session metrics.
        
        Args:
            current_data: Latest data point
        """
        if current_data is None:
            return
        
        st.subheader("Current Session Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_level = self._get_risk_level(current_data['risk_score'])
            st.metric(
                label="Current Risk",
                value=f"{current_data['risk_score']:.3f}",
                delta=risk_level
            )
        
        with col2:
            if 'confidence' in current_data:
                st.metric(
                    label="Confidence",
                    value=f"{current_data['confidence']:.3f}",
                    delta=None
                )
        
        with col3:
            if 'processing_time_ms' in current_data:
                st.metric(
                    label="Processing Time",
                    value=f"{current_data['processing_time_ms']:.0f}ms",
                    delta=None
                )
        
        with col4:
            if 'processing_mode' in current_data:
                st.metric(
                    label="Processing Mode",
                    value=current_data['processing_mode'],
                    delta=None
                )
    
    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score."""
        if risk_score >= 0.8:
            return self.colors['high_risk']
        elif risk_score >= 0.5:
            return self.colors['moderate_risk']
        else:
            return self.colors['low_risk']
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level string."""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.5:
            return "MODERATE"
        else:
            return "LOW"
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Summary statistics dictionary
        """
        if session_id not in self.risk_data_cache or not self.risk_data_cache[session_id]:
            return {}
        
        data_points = self.risk_data_cache[session_id]
        df = pd.DataFrame(data_points)
        
        return {
            'total_points': len(data_points),
            'current_risk': df['risk_score'].iloc[-1] if not df.empty else 0,
            'average_risk': df['risk_score'].mean(),
            'max_risk': df['risk_score'].max(),
            'min_risk': df['risk_score'].min(),
            'risk_trend': self._calculate_trend(df['risk_score'].values),
            'average_confidence': df['confidence'].mean() if 'confidence' in df else 0,
            'average_processing_time': df['processing_time_ms'].mean() if 'processing_time_ms' in df else 0
        }
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"