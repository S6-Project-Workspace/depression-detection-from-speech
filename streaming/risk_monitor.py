"""
Risk Monitor for Real-time Streaming Analysis

This module provides real-time risk assessment, trend analysis, and monitoring
for the streaming depression detection system.

Key Features:
- Rolling risk score averages and trend analysis
- Risk level categorization (LOW/MODERATE/HIGH)
- Statistical analysis and confidence intervals
- Risk pattern detection and anomaly identification
- Historical risk data management

Reference: Real-time Streaming Analysis - Task 2.1
"""

import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
import threading

from .models import RiskDataPoint, ProcessingMode

logger = logging.getLogger(__name__)


class RiskTrendAnalyzer:
    """
    Analyzes risk trends and patterns over time.
    
    Provides statistical analysis of risk progression including:
    - Linear trend calculation
    - Change point detection
    - Volatility analysis
    - Pattern recognition
    """
    
    def __init__(self, min_data_points: int = 5):
        """
        Initialize risk trend analyzer.
        
        Args:
            min_data_points: Minimum data points needed for analysis
        """
        self.min_data_points = min_data_points
        
    def calculate_trend(self, risk_scores: List[float], timestamps: List[datetime]) -> Dict[str, float]:
        """
        Calculate risk trend over time.
        
        Args:
            risk_scores: List of risk scores
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with trend analysis results
        """
        if len(risk_scores) < self.min_data_points:
            return {
                'trend_slope': 0.0,
                'trend_r_squared': 0.0,
                'trend_p_value': 1.0,
                'trend_direction': 'insufficient_data'
            }
        
        try:
            # Convert timestamps to seconds since first timestamp
            time_seconds = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_seconds, risk_scores)
            
            # Determine trend direction
            if p_value < 0.05:  # Statistically significant
                if slope > 0.01:  # Increasing
                    direction = 'increasing'
                elif slope < -0.01:  # Decreasing
                    direction = 'decreasing'
                else:
                    direction = 'stable'
            else:
                direction = 'no_trend'
            
            return {
                'trend_slope': float(slope),
                'trend_r_squared': float(r_value ** 2),
                'trend_p_value': float(p_value),
                'trend_direction': direction,
                'trend_intercept': float(intercept),
                'trend_std_error': float(std_err)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return {
                'trend_slope': 0.0,
                'trend_r_squared': 0.0,
                'trend_p_value': 1.0,
                'trend_direction': 'error'
            }
    
    def detect_change_points(self, risk_scores: List[float], sensitivity: float = 2.0) -> List[int]:
        """
        Detect significant change points in risk scores.
        
        Args:
            risk_scores: List of risk scores
            sensitivity: Sensitivity threshold (standard deviations)
            
        Returns:
            List of indices where change points occur
        """
        if len(risk_scores) < self.min_data_points:
            return []
        
        try:
            # Calculate moving statistics
            window_size = max(3, len(risk_scores) // 5)
            change_points = []
            
            for i in range(window_size, len(risk_scores) - window_size):
                # Compare before and after windows
                before_window = risk_scores[i-window_size:i]
                after_window = risk_scores[i:i+window_size]
                
                # Statistical test for difference
                t_stat, p_value = stats.ttest_ind(before_window, after_window)
                
                # Check for significant change
                if p_value < 0.05 and abs(t_stat) > sensitivity:
                    change_points.append(i)
            
            return change_points
            
        except Exception as e:
            logger.error(f"Error detecting change points: {e}")
            return []
    
    def calculate_volatility(self, risk_scores: List[float]) -> Dict[str, float]:
        """
        Calculate risk score volatility metrics.
        
        Args:
            risk_scores: List of risk scores
            
        Returns:
            Dictionary with volatility metrics
        """
        if len(risk_scores) < 2:
            return {
                'volatility': 0.0,
                'coefficient_of_variation': 0.0,
                'max_change': 0.0,
                'volatility_category': 'insufficient_data'
            }
        
        try:
            # Calculate basic volatility metrics
            std_dev = np.std(risk_scores)
            mean_score = np.mean(risk_scores)
            
            # Coefficient of variation
            cv = std_dev / mean_score if mean_score > 0 else 0.0
            
            # Maximum change between consecutive points
            changes = [abs(risk_scores[i] - risk_scores[i-1]) for i in range(1, len(risk_scores))]
            max_change = max(changes) if changes else 0.0
            
            # Categorize volatility
            if std_dev < 0.1:
                category = 'low'
            elif std_dev < 0.2:
                category = 'moderate'
            else:
                category = 'high'
            
            return {
                'volatility': float(std_dev),
                'coefficient_of_variation': float(cv),
                'max_change': float(max_change),
                'volatility_category': category
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return {
                'volatility': 0.0,
                'coefficient_of_variation': 0.0,
                'max_change': 0.0,
                'volatility_category': 'error'
            }


class RiskAnalyzer:
    """
    Comprehensive risk analysis with statistical methods.
    
    Provides advanced risk assessment including:
    - Confidence interval calculation
    - Risk level classification
    - Anomaly detection
    - Statistical significance testing
    """
    
    def __init__(self):
        """Initialize risk analyzer."""
        self.trend_analyzer = RiskTrendAnalyzer()
    
    def analyze_risk_distribution(self, risk_scores: List[float]) -> Dict[str, Any]:
        """
        Analyze the distribution of risk scores.
        
        Args:
            risk_scores: List of risk scores
            
        Returns:
            Dictionary with distribution analysis
        """
        if not risk_scores:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'percentiles': {},
                'distribution_type': 'no_data'
            }
        
        try:
            scores_array = np.array(risk_scores)
            
            # Basic statistics
            mean_score = float(np.mean(scores_array))
            median_score = float(np.median(scores_array))
            std_score = float(np.std(scores_array))
            min_score = float(np.min(scores_array))
            max_score = float(np.max(scores_array))
            
            # Percentiles
            percentiles = {
                '25th': float(np.percentile(scores_array, 25)),
                '75th': float(np.percentile(scores_array, 75)),
                '90th': float(np.percentile(scores_array, 90)),
                '95th': float(np.percentile(scores_array, 95))
            }
            
            # Distribution type (simplified)
            if std_score < 0.1:
                dist_type = 'concentrated'
            elif max_score - min_score > 0.7:
                dist_type = 'wide'
            else:
                dist_type = 'normal'
            
            return {
                'mean': mean_score,
                'median': median_score,
                'std': std_score,
                'min': min_score,
                'max': max_score,
                'percentiles': percentiles,
                'distribution_type': dist_type,
                'sample_size': len(risk_scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk distribution: {e}")
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'percentiles': {},
                'distribution_type': 'error'
            }
    
    def calculate_confidence_intervals(
        self, 
        risk_scores: List[float], 
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence intervals for risk scores.
        
        Args:
            risk_scores: List of risk scores
            confidence_level: Confidence level (0.0 to 1.0)
            
        Returns:
            Dictionary with confidence interval bounds
        """
        if len(risk_scores) < 2:
            return {
                'lower_bound': 0.0,
                'upper_bound': 1.0,
                'confidence_level': confidence_level,
                'margin_of_error': 0.5
            }
        
        try:
            scores_array = np.array(risk_scores)
            mean_score = np.mean(scores_array)
            std_error = stats.sem(scores_array)  # Standard error of mean
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            degrees_freedom = len(risk_scores) - 1
            t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
            
            margin_of_error = t_critical * std_error
            lower_bound = mean_score - margin_of_error
            upper_bound = mean_score + margin_of_error
            
            # Clamp to valid range [0, 1]
            lower_bound = max(0.0, float(lower_bound))
            upper_bound = min(1.0, float(upper_bound))
            
            return {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence_level': confidence_level,
                'margin_of_error': float(margin_of_error)
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {
                'lower_bound': 0.0,
                'upper_bound': 1.0,
                'confidence_level': confidence_level,
                'margin_of_error': 0.5
            }
    
    def detect_anomalies(
        self, 
        risk_scores: List[float], 
        method: str = 'iqr',
        sensitivity: float = 1.5
    ) -> List[int]:
        """
        Detect anomalous risk scores.
        
        Args:
            risk_scores: List of risk scores
            method: Anomaly detection method ('iqr', 'zscore', 'isolation')
            sensitivity: Sensitivity parameter
            
        Returns:
            List of indices of anomalous scores
        """
        if len(risk_scores) < 5:
            return []
        
        try:
            scores_array = np.array(risk_scores)
            anomaly_indices = []
            
            if method == 'iqr':
                # Interquartile Range method
                q1 = np.percentile(scores_array, 25)
                q3 = np.percentile(scores_array, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - sensitivity * iqr
                upper_bound = q3 + sensitivity * iqr
                
                for i, score in enumerate(scores_array):
                    if score < lower_bound or score > upper_bound:
                        anomaly_indices.append(i)
            
            elif method == 'zscore':
                # Z-score method
                mean_score = np.mean(scores_array)
                std_score = np.std(scores_array)
                
                if std_score > 0:
                    z_scores = np.abs((scores_array - mean_score) / std_score)
                    anomaly_indices = [i for i, z in enumerate(z_scores) if z > sensitivity]
            
            return anomaly_indices
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []


class RiskMonitor:
    """
    Real-time risk monitoring with rolling averages and trend analysis.
    
    Features:
    - Rolling risk score averages (configurable windows)
    - Risk trend analysis and pattern detection
    - Risk level categorization and thresholds
    - Statistical analysis and confidence intervals
    - Historical risk data management
    """
    
    def __init__(
        self,
        rolling_window_minutes: int = 5,
        max_history_points: int = 1000,
        risk_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize risk monitor.
        
        Args:
            rolling_window_minutes: Window size for rolling averages
            max_history_points: Maximum number of historical points to keep
            risk_thresholds: Custom risk level thresholds
        """
        self.rolling_window_minutes = rolling_window_minutes
        self.max_history_points = max_history_points
        
        # Default risk thresholds
        self.risk_thresholds = risk_thresholds or {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8
        }
        
        # Risk data storage
        self.risk_history: deque = deque(maxlen=max_history_points)
        
        # Analysis components
        self.risk_analyzer = RiskAnalyzer()
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.total_risk_points = 0
        self.last_analysis_time = None
        
        logger.info(f"RiskMonitor initialized with {rolling_window_minutes}min window")
    
    def add_risk_datapoint(self, risk_datapoint: RiskDataPoint):
        """
        Add a new risk data point to the monitor.
        
        Args:
            risk_datapoint: Risk assessment result
        """
        with self.lock:
            self.risk_history.append(risk_datapoint)
            self.total_risk_points += 1
            self.last_analysis_time = datetime.now()
    
    def get_current_risk_score(self) -> float:
        """Get the most recent risk score."""
        with self.lock:
            if not self.risk_history:
                return 0.0
            return self.risk_history[-1].risk_score
    
    def get_rolling_average(self, minutes: Optional[int] = None) -> float:
        """
        Get rolling average risk score.
        
        Args:
            minutes: Window size in minutes (uses default if None)
            
        Returns:
            Rolling average risk score
        """
        if minutes is None:
            minutes = self.rolling_window_minutes
        
        with self.lock:
            if not self.risk_history:
                return 0.0
            
            # Get recent data points within window
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_points = [
                point for point in self.risk_history
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_points:
                return 0.0
            
            return np.mean([point.risk_score for point in recent_points])
    
    def get_risk_trend(self, minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Get risk trend analysis.
        
        Args:
            minutes: Window size for trend analysis
            
        Returns:
            Dictionary with trend analysis results
        """
        if minutes is None:
            minutes = self.rolling_window_minutes
        
        with self.lock:
            if not self.risk_history:
                return {
                    'trend_slope': 0.0,
                    'trend_direction': 'no_data',
                    'data_points': 0
                }
            
            # Get recent data points
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_points = [
                point for point in self.risk_history
                if point.timestamp >= cutoff_time
            ]
            
            if len(recent_points) < 3:
                return {
                    'trend_slope': 0.0,
                    'trend_direction': 'insufficient_data',
                    'data_points': len(recent_points)
                }
            
            # Extract data for trend analysis
            risk_scores = [point.risk_score for point in recent_points]
            timestamps = [point.timestamp for point in recent_points]
            
            # Calculate trend
            trend_analysis = self.risk_analyzer.trend_analyzer.calculate_trend(
                risk_scores, timestamps
            )
            trend_analysis['data_points'] = len(recent_points)
            
            return trend_analysis
    
    def get_risk_level(self, risk_score: Optional[float] = None) -> str:
        """
        Get categorical risk level.
        
        Args:
            risk_score: Risk score to categorize (uses current if None)
            
        Returns:
            Risk level category ('LOW', 'MODERATE', 'HIGH')
        """
        if risk_score is None:
            risk_score = self.get_current_risk_score()
        
        if risk_score >= self.risk_thresholds['high']:
            return 'HIGH'
        elif risk_score >= self.risk_thresholds['moderate']:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def analyze_risk_patterns(self, minutes: int = 30) -> Dict[str, Any]:
        """
        Comprehensive risk pattern analysis.
        
        Args:
            minutes: Analysis window in minutes
            
        Returns:
            Dictionary with comprehensive risk analysis
        """
        with self.lock:
            if not self.risk_history:
                return {
                    'analysis_type': 'no_data',
                    'data_points': 0,
                    'analysis_time': datetime.now().isoformat()
                }
            
            # Get data for analysis
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            analysis_points = [
                point for point in self.risk_history
                if point.timestamp >= cutoff_time
            ]
            
            if len(analysis_points) < 5:
                return {
                    'analysis_type': 'insufficient_data',
                    'data_points': len(analysis_points),
                    'analysis_time': datetime.now().isoformat()
                }
            
            # Extract risk scores and timestamps
            risk_scores = [point.risk_score for point in analysis_points]
            timestamps = [point.timestamp for point in analysis_points]
            
            # Comprehensive analysis
            analysis = {
                'analysis_type': 'comprehensive',
                'data_points': len(analysis_points),
                'analysis_time': datetime.now().isoformat(),
                'window_minutes': minutes
            }
            
            # Distribution analysis
            analysis['distribution'] = self.risk_analyzer.analyze_risk_distribution(risk_scores)
            
            # Trend analysis
            analysis['trend'] = self.risk_analyzer.trend_analyzer.calculate_trend(
                risk_scores, timestamps
            )
            
            # Volatility analysis
            analysis['volatility'] = self.risk_analyzer.trend_analyzer.calculate_volatility(risk_scores)
            
            # Confidence intervals
            analysis['confidence_intervals'] = self.risk_analyzer.calculate_confidence_intervals(risk_scores)
            
            # Change point detection
            analysis['change_points'] = self.risk_analyzer.trend_analyzer.detect_change_points(risk_scores)
            
            # Anomaly detection
            analysis['anomalies'] = self.risk_analyzer.detect_anomalies(risk_scores)
            
            # Risk level distribution
            risk_levels = [self.get_risk_level(score) for score in risk_scores]
            level_counts = {
                'LOW': risk_levels.count('LOW'),
                'MODERATE': risk_levels.count('MODERATE'),
                'HIGH': risk_levels.count('HIGH')
            }
            analysis['risk_level_distribution'] = level_counts
            
            return analysis
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        with self.lock:
            current_score = self.get_current_risk_score()
            rolling_avg = self.get_rolling_average()
            risk_level = self.get_risk_level(current_score)
            trend = self.get_risk_trend()
            
            return {
                'current_risk_score': current_score,
                'rolling_average': rolling_avg,
                'risk_level': risk_level,
                'trend_direction': trend.get('trend_direction', 'unknown'),
                'trend_slope': trend.get('trend_slope', 0.0),
                'data_points': len(self.risk_history),
                'last_update': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                'thresholds': self.risk_thresholds
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        with self.lock:
            if not self.risk_history:
                return {
                    'total_risk_points': self.total_risk_points,
                    'current_history_size': 0,
                    'max_history_points': self.max_history_points,
                    'rolling_window_minutes': self.rolling_window_minutes
                }
            
            # Calculate time span
            oldest_point = self.risk_history[0]
            newest_point = self.risk_history[-1]
            time_span = (newest_point.timestamp - oldest_point.timestamp).total_seconds() / 60
            
            return {
                'total_risk_points': self.total_risk_points,
                'current_history_size': len(self.risk_history),
                'max_history_points': self.max_history_points,
                'rolling_window_minutes': self.rolling_window_minutes,
                'time_span_minutes': time_span,
                'oldest_point': oldest_point.timestamp.isoformat(),
                'newest_point': newest_point.timestamp.isoformat(),
                'current_risk_score': self.get_current_risk_score(),
                'rolling_average': self.get_rolling_average()
            }
    
    def reset_history(self):
        """Reset risk history."""
        with self.lock:
            self.risk_history.clear()
            self.total_risk_points = 0
            self.last_analysis_time = None
            logger.info("Risk monitor history reset")
    
    def set_risk_thresholds(self, thresholds: Dict[str, float]):
        """Update risk level thresholds."""
        with self.lock:
            self.risk_thresholds.update(thresholds)
            logger.info(f"Risk thresholds updated: {self.risk_thresholds}")
    
    def export_risk_data(self, minutes: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Export risk data for analysis.
        
        Args:
            minutes: Time window to export (all data if None)
            
        Returns:
            List of risk data points as dictionaries
        """
        with self.lock:
            if minutes is None:
                export_points = list(self.risk_history)
            else:
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                export_points = [
                    point for point in self.risk_history
                    if point.timestamp >= cutoff_time
                ]
            
            return [point.to_dict() for point in export_points]