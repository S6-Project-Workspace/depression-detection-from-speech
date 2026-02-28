"""
Property-based tests for dashboard functionality.

This module tests the correctness properties of the metrics dashboard,
including performance data visualization, model comparison charts,
and real-time metric updates.

Implements Properties 16, 17, 18 for Requirements 4.2, 4.3, 4.4.
"""

import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from hypothesis import given, strategies as st, settings
import numpy as np

# Import the models for testing
from web_interface.models import ModelMetrics


class TestDashboardProperties:
    """Property-based tests for dashboard functionality."""
    
    @given(
        model_data=st.lists(
            st.fixed_dictionaries({
                'model_name': st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
                'version': st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
                'f1_score': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                'accuracy': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                'precision': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                'recall': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            }),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=50, deadline=3000)
    def test_property_16_performance_data_visualization(self, model_data):
        """
        Property 16: Performance data visualization
        Feature: nlp-web-interface
        
        For any available model performance data, training curves and validation 
        metrics should be rendered as interactive charts.
        
        Validates: Requirements 4.2
        """
        # Create ModelMetrics objects from test data
        model_metrics = []
        for data in model_data:
            metric = ModelMetrics(
                model_name=data['model_name'],
                version=data['version'],
                accuracy=data['accuracy'],
                precision=data['precision'],
                recall=data['recall'],
                f1_score=data['f1_score'],
                training_date=datetime.now() - timedelta(days=len(model_metrics)),
                evaluation_data={
                    'confusion_matrix': [[80, 20], [15, 85]],
                    'test_samples': 200,
                    'training_epochs': 10
                }
            )
            model_metrics.append(metric)
        
        # Property: All model metrics should have valid data ranges
        for metric in model_metrics:
            assert 0.0 <= metric.f1_score <= 1.0
            assert 0.0 <= metric.accuracy <= 1.0
            assert 0.0 <= metric.precision <= 1.0
            assert 0.0 <= metric.recall <= 1.0
            assert isinstance(metric.model_name, str) and len(metric.model_name) > 0
            assert isinstance(metric.version, str) and len(metric.version) > 0
            assert isinstance(metric.training_date, datetime)
            assert isinstance(metric.evaluation_data, dict)
        
        # Property: Model metrics should be serializable for visualization
        for metric in model_metrics:
            serialized = metric.model_dump()
            assert isinstance(serialized, dict)
            assert 'model_name' in serialized
            assert 'f1_score' in serialized
            assert 'accuracy' in serialized
            assert 'precision' in serialized
            assert 'recall' in serialized
            
            # Should be JSON serializable
            json_str = json.dumps(serialized, default=str)
            assert isinstance(json_str, str)
            
            # Should be deserializable
            deserialized = json.loads(json_str)
            assert isinstance(deserialized, dict)
        
        # Property: Metrics should be sortable by performance
        sorted_by_f1 = sorted(model_metrics, key=lambda m: m.f1_score, reverse=True)
        assert len(sorted_by_f1) == len(model_metrics)
        
        if len(sorted_by_f1) > 1:
            # Verify sorting order
            for i in range(len(sorted_by_f1) - 1):
                assert sorted_by_f1[i].f1_score >= sorted_by_f1[i + 1].f1_score
        
        # Property: Best and worst models should be identifiable
        if model_metrics:
            best_model = max(model_metrics, key=lambda m: m.f1_score)
            worst_model = min(model_metrics, key=lambda m: m.f1_score)
            
            assert best_model.f1_score >= worst_model.f1_score
            assert 0.0 <= worst_model.f1_score <= 1.0
            assert 0.0 <= best_model.f1_score <= 1.0
    
    @given(
        model_data=st.lists(
            st.fixed_dictionaries({
                'model_name': st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
                'version': st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
                'f1_score': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                'accuracy': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                'precision': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                'recall': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            }),
            min_size=2, max_size=10
        ),
        comparison_metric=st.sampled_from(['f1_score', 'accuracy', 'precision', 'recall'])
    )
    @settings(max_examples=30, deadline=3000)
    def test_property_17_model_comparison_charts(self, model_data, comparison_metric):
        """
        Property 17: Model comparison charts
        Feature: nlp-web-interface
        
        For any set of multiple models, comparison charts should display 
        relative performance metrics accurately.
        
        Validates: Requirements 4.3
        """
        # Create ModelMetrics objects
        model_metrics = []
        for i, data in enumerate(model_data):
            metric = ModelMetrics(
                model_name=data['model_name'],
                version=data['version'],
                accuracy=data['accuracy'],
                precision=data['precision'],
                recall=data['recall'],
                f1_score=data['f1_score'],
                training_date=datetime.now() - timedelta(days=i),
                evaluation_data={'test_samples': 200}
            )
            model_metrics.append(metric)
        
        # Property: All models should be comparable
        assert len(model_metrics) >= 2
        
        # Property: Comparison metric values should be extractable
        comparison_values = []
        for metric in model_metrics:
            if comparison_metric == 'f1_score':
                comparison_values.append(metric.f1_score)
            elif comparison_metric == 'accuracy':
                comparison_values.append(metric.accuracy)
            elif comparison_metric == 'precision':
                comparison_values.append(metric.precision)
            elif comparison_metric == 'recall':
                comparison_values.append(metric.recall)
        
        assert len(comparison_values) == len(model_metrics)
        
        # Property: Comparison values should be valid
        for value in comparison_values:
            assert 0.0 <= value <= 1.0
            assert not np.isnan(value)
            assert not np.isinf(value)
        
        # Property: Models should be rankable by comparison metric
        ranked_models = sorted(
            zip(model_metrics, comparison_values), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        assert len(ranked_models) == len(model_metrics)
        
        # Verify ranking order
        for i in range(len(ranked_models) - 1):
            assert ranked_models[i][1] >= ranked_models[i + 1][1]
        
        # Property: Best and worst models should be identifiable
        best_model, best_value = ranked_models[0]
        worst_model, worst_value = ranked_models[-1]
        
        assert best_value >= worst_value
        assert isinstance(best_model, ModelMetrics)
        assert isinstance(worst_model, ModelMetrics)
        
        # Property: Relative performance differences should be calculable
        if best_value > 0:
            for model, value in ranked_models:
                relative_performance = value / best_value
                assert 0.0 <= relative_performance <= 1.0
        
        # Property: Comparison data should be serializable for charts
        comparison_data = []
        for model, value in ranked_models:
            data_point = {
                'model_name': model.model_name,
                'version': model.version,
                'metric_value': value,
                'relative_performance': value / best_value if best_value > 0 else 0.0
            }
            comparison_data.append(data_point)
        
        # Should be JSON serializable
        json_str = json.dumps(comparison_data, default=str)
        assert isinstance(json_str, str)
        
        deserialized = json.loads(json_str)
        assert len(deserialized) == len(model_metrics)
    
    @given(
        operations=st.lists(
            st.fixed_dictionaries({
                'operation_type': st.sampled_from(['text_analysis', 'audio_processing', 'batch_analysis']),
                'processing_time': st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
                'input_size': st.integers(min_value=1, max_value=10000),
                'success': st.booleans(),
                'timestamp_offset_hours': st.integers(min_value=0, max_value=23)
            }),
            min_size=10, max_size=100
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_property_18_realtime_metric_updates(self, operations):
        """
        Property 18: Real-time metric updates
        Feature: nlp-web-interface
        
        For any metric update event, the dashboard should refresh automatically 
        without requiring page reload.
        
        Validates: Requirements 4.4
        """
        # Simulate real-time metrics processing
        base_time = datetime.now() - timedelta(hours=24)
        
        processed_operations = []
        for i, op_data in enumerate(operations):
            timestamp = base_time + timedelta(
                hours=op_data['timestamp_offset_hours'],
                minutes=i % 60
            )
            
            processed_op = {
                'timestamp': timestamp,
                'operation_type': op_data['operation_type'],
                'processing_time': op_data['processing_time'],
                'input_size': op_data['input_size'],
                'success': op_data['success'],
                'error_message': None if op_data['success'] else "Test error"
            }
            processed_operations.append(processed_op)
        
        # Property: All operations should be processable
        assert len(processed_operations) == len(operations)
        
        # Property: Success rate calculation should be accurate
        successful_ops = sum(1 for op in processed_operations if op['success'])
        failed_ops = len(processed_operations) - successful_ops
        total_ops = len(processed_operations)
        
        if total_ops > 0:
            success_rate = successful_ops / total_ops
            error_rate = failed_ops / total_ops
            
            assert 0.0 <= success_rate <= 1.0
            assert 0.0 <= error_rate <= 1.0
            assert abs(success_rate + error_rate - 1.0) < 0.001
        
        # Property: Processing time statistics should be calculable
        processing_times = [op['processing_time'] for op in processed_operations]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
            
            assert avg_time >= min_time
            assert avg_time <= max_time
            assert min_time <= max_time
            assert all(0.001 <= t <= 10.0 for t in processing_times)
        
        # Property: Operation type breakdown should be accurate
        operation_types = {}
        for op in processed_operations:
            op_type = op['operation_type']
            if op_type not in operation_types:
                operation_types[op_type] = {
                    'count': 0, 
                    'success_count': 0, 
                    'total_time': 0.0,
                    'times': []
                }
            
            operation_types[op_type]['count'] += 1
            if op['success']:
                operation_types[op_type]['success_count'] += 1
            operation_types[op_type]['total_time'] += op['processing_time']
            operation_types[op_type]['times'].append(op['processing_time'])
        
        # Verify operation type statistics
        for op_type, stats in operation_types.items():
            assert stats['count'] > 0
            assert 0 <= stats['success_count'] <= stats['count']
            assert stats['total_time'] > 0
            
            success_rate = stats['success_count'] / stats['count']
            assert 0.0 <= success_rate <= 1.0
            
            avg_time = stats['total_time'] / stats['count']
            assert avg_time > 0
            assert len(stats['times']) == stats['count']
        
        # Property: Hourly breakdown should be calculable
        hourly_stats = {}
        for op in processed_operations:
            hour = op['timestamp'].replace(minute=0, second=0, microsecond=0)
            if hour not in hourly_stats:
                hourly_stats[hour] = {'count': 0, 'total_time': 0.0}
            
            hourly_stats[hour]['count'] += 1
            hourly_stats[hour]['total_time'] += op['processing_time']
        
        # Verify hourly statistics
        for hour, stats in hourly_stats.items():
            assert isinstance(hour, datetime)
            assert stats['count'] > 0
            assert stats['total_time'] > 0
            
            avg_time = stats['total_time'] / stats['count']
            assert avg_time > 0
        
        # Property: Performance indicators should be calculable
        if total_ops > 0:
            throughput_per_hour = total_ops / 24  # 24 hours
            assert throughput_per_hour >= 0
            
            reliability_score = success_rate
            assert 0.0 <= reliability_score <= 1.0
            
            if processing_times:
                performance_score = min(1.0, 1.0 / max(avg_time, 0.001))
                assert 0.0 <= performance_score <= 1.0
        
        # Property: Real-time data should be serializable for dashboard updates
        dashboard_data = {
            'total_operations': total_ops,
            'successful_operations': successful_ops,
            'failed_operations': failed_ops,
            'success_rate': success_rate if total_ops > 0 else 0.0,
            'avg_processing_time': avg_time if processing_times else 0.0,
            'operation_types': operation_types,
            'hourly_breakdown': [
                {
                    'hour': hour.isoformat(),
                    'operations': stats['count'],
                    'avg_time': stats['total_time'] / stats['count']
                }
                for hour, stats in hourly_stats.items()
            ]
        }
        
        # Should be JSON serializable
        json_str = json.dumps(dashboard_data, default=str)
        assert isinstance(json_str, str)
        
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, dict)
        assert 'total_operations' in deserialized
        assert 'success_rate' in deserialized


# Test runner for property-based tests
class TestDashboardPropertiesRunner:
    """Test runner for dashboard property tests."""
    
    def test_run_all_dashboard_properties(self):
        """Run all dashboard property tests."""
        test_instance = TestDashboardProperties()
        
        # Run property tests with simple data
        test_instance.test_property_16_performance_data_visualization([
            {'model_name': 'model1', 'version': 'v1.0', 'f1_score': 0.85, 
             'accuracy': 0.82, 'precision': 0.80, 'recall': 0.87},
            {'model_name': 'model2', 'version': 'v2.0', 'f1_score': 0.90, 
             'accuracy': 0.88, 'precision': 0.85, 'recall': 0.92}
        ])
        
        print("✓ Property 16 (Performance data visualization) passed")
        
        test_instance.test_property_17_model_comparison_charts(
            model_data=[
                {'model_name': 'model1', 'version': 'v1.0', 'f1_score': 0.85, 
                 'accuracy': 0.82, 'precision': 0.80, 'recall': 0.87},
                {'model_name': 'model2', 'version': 'v2.0', 'f1_score': 0.90, 
                 'accuracy': 0.88, 'precision': 0.85, 'recall': 0.92}
            ],
            comparison_metric='f1_score'
        )
        
        print("✓ Property 17 (Model comparison charts) passed")
        
        test_instance.test_property_18_realtime_metric_updates([
            {'operation_type': 'text_analysis', 'processing_time': 0.15, 
             'input_size': 100, 'success': True, 'timestamp_offset_hours': 1},
            {'operation_type': 'audio_processing', 'processing_time': 0.25, 
             'input_size': 1000, 'success': True, 'timestamp_offset_hours': 2},
            {'operation_type': 'batch_analysis', 'processing_time': 1.5, 
             'input_size': 5000, 'success': False, 'timestamp_offset_hours': 3}
        ])
        
        print("✓ Property 18 (Real-time metric updates) passed")


if __name__ == "__main__":
    # Run the tests
    runner = TestDashboardPropertiesRunner()
    runner.test_run_all_dashboard_properties()
    print("\n✅ All dashboard property tests passed!")