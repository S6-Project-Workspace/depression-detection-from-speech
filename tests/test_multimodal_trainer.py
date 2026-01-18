"""
Property-Based Tests for Multimodal Trainer

Tests the multimodal trainer evaluation metrics using hypothesis for property-based testing.
Validates Property 10: Evaluation Metrics Completeness (Requirements 7.1, 7.2, 7.5).

Feature: multimodal-nlp-upgrade
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Check if required modules are available
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

# Skip all tests if torch is not available
if not HAS_TORCH:
    pytest.skip("torch not available", allow_module_level=True)

from multimodal_trainer import (
    compute_evaluation_metrics,
    EvaluationResult,
    compare_models
)


# Strategy for generating binary predictions
binary_array_strategy = st.lists(
    st.integers(min_value=0, max_value=1),
    min_size=10,
    max_size=100
)

# Strategy for generating sample sizes
sample_size_strategy = st.integers(min_value=10, max_value=200)


class TestEvaluationMetricsCompleteness:
    """
    Property 10: Evaluation Metrics Completeness
    
    *For any* model evaluation, the output SHALL contain Macro-F1, per-class F1,
    precision, recall, and confusion matrix, all with valid numerical values.
    
    **Validates: Requirements 7.1, 7.2, 7.5**
    """
    
    @given(
        predictions=binary_array_strategy,
        labels=binary_array_strategy
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_evaluation_contains_all_required_metrics(
        self,
        predictions: List[int],
        labels: List[int]
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 10: Evaluation Metrics Completeness
        
        For any evaluation, output SHALL contain all required metrics.
        
        **Validates: Requirements 7.1, 7.2, 7.5**
        """
        # Ensure same length
        min_len = min(len(predictions), len(labels))
        assume(min_len >= 10)
        
        predictions = np.array(predictions[:min_len])
        labels = np.array(labels[:min_len])
        
        # Compute metrics
        result = compute_evaluation_metrics(predictions, labels)
        
        # Property: All required metrics must be present
        # Requirement 7.1 - Macro-F1
        assert hasattr(result, 'macro_f1'), "Missing macro_f1"
        assert result.macro_f1 is not None, "macro_f1 is None"
        
        # Requirement 7.2 - Per-class F1
        assert hasattr(result, 'f1_depressed'), "Missing f1_depressed"
        assert hasattr(result, 'f1_non_depressed'), "Missing f1_non_depressed"
        assert result.f1_depressed is not None, "f1_depressed is None"
        assert result.f1_non_depressed is not None, "f1_non_depressed is None"
        
        # Requirement 7.2 - Per-class Precision
        assert hasattr(result, 'precision_depressed'), "Missing precision_depressed"
        assert hasattr(result, 'precision_non_depressed'), "Missing precision_non_depressed"
        assert result.precision_depressed is not None, "precision_depressed is None"
        assert result.precision_non_depressed is not None, "precision_non_depressed is None"
        
        # Requirement 7.2 - Per-class Recall
        assert hasattr(result, 'recall_depressed'), "Missing recall_depressed"
        assert hasattr(result, 'recall_non_depressed'), "Missing recall_non_depressed"
        assert result.recall_depressed is not None, "recall_depressed is None"
        assert result.recall_non_depressed is not None, "recall_non_depressed is None"
        
        # Requirement 7.5 - Confusion Matrix
        assert hasattr(result, 'confusion_matrix'), "Missing confusion_matrix"
        assert result.confusion_matrix is not None, "confusion_matrix is None"
    
    @given(
        predictions=binary_array_strategy,
        labels=binary_array_strategy
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_metrics_have_valid_numerical_values(
        self,
        predictions: List[int],
        labels: List[int]
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 10: Evaluation Metrics Completeness
        
        All metrics SHALL have valid numerical values (not NaN, in valid range).
        
        **Validates: Requirements 7.1, 7.2, 7.5**
        """
        min_len = min(len(predictions), len(labels))
        assume(min_len >= 10)
        
        predictions = np.array(predictions[:min_len])
        labels = np.array(labels[:min_len])
        
        result = compute_evaluation_metrics(predictions, labels)
        
        # Property: All F1 scores should be in [0, 1]
        assert 0.0 <= result.macro_f1 <= 1.0, f"macro_f1 {result.macro_f1} out of range"
        assert 0.0 <= result.f1_depressed <= 1.0, f"f1_depressed {result.f1_depressed} out of range"
        assert 0.0 <= result.f1_non_depressed <= 1.0, f"f1_non_depressed {result.f1_non_depressed} out of range"
        
        # Property: All precision scores should be in [0, 1]
        assert 0.0 <= result.precision_depressed <= 1.0, f"precision_depressed out of range"
        assert 0.0 <= result.precision_non_depressed <= 1.0, f"precision_non_depressed out of range"
        assert 0.0 <= result.macro_precision <= 1.0, f"macro_precision out of range"
        
        # Property: All recall scores should be in [0, 1]
        assert 0.0 <= result.recall_depressed <= 1.0, f"recall_depressed out of range"
        assert 0.0 <= result.recall_non_depressed <= 1.0, f"recall_non_depressed out of range"
        assert 0.0 <= result.macro_recall <= 1.0, f"macro_recall out of range"
        
        # Property: No NaN values
        assert not np.isnan(result.macro_f1), "macro_f1 is NaN"
        assert not np.isnan(result.f1_depressed), "f1_depressed is NaN"
        assert not np.isnan(result.f1_non_depressed), "f1_non_depressed is NaN"
        assert not np.isnan(result.macro_precision), "macro_precision is NaN"
        assert not np.isnan(result.macro_recall), "macro_recall is NaN"
    
    @given(
        predictions=binary_array_strategy,
        labels=binary_array_strategy
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_confusion_matrix_has_correct_shape(
        self,
        predictions: List[int],
        labels: List[int]
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 10: Evaluation Metrics Completeness
        
        Confusion matrix SHALL have shape (2, 2) for binary classification.
        
        **Validates: Requirements 7.5**
        """
        min_len = min(len(predictions), len(labels))
        assume(min_len >= 10)
        
        predictions = np.array(predictions[:min_len])
        labels = np.array(labels[:min_len])
        
        result = compute_evaluation_metrics(predictions, labels)
        
        # Property: Confusion matrix should be 2x2
        assert result.confusion_matrix.shape == (2, 2), (
            f"Confusion matrix shape {result.confusion_matrix.shape} != (2, 2)"
        )
    
    @given(
        predictions=binary_array_strategy,
        labels=binary_array_strategy
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_confusion_matrix_sums_to_sample_count(
        self,
        predictions: List[int],
        labels: List[int]
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 10: Evaluation Metrics Completeness
        
        Confusion matrix entries SHALL sum to total sample count.
        
        **Validates: Requirements 7.5**
        """
        min_len = min(len(predictions), len(labels))
        assume(min_len >= 10)
        
        predictions = np.array(predictions[:min_len])
        labels = np.array(labels[:min_len])
        
        result = compute_evaluation_metrics(predictions, labels)
        
        # Property: Sum of confusion matrix should equal sample count
        cm_sum = result.confusion_matrix.sum()
        assert cm_sum == min_len, (
            f"Confusion matrix sum {cm_sum} != sample count {min_len}"
        )
    
    @given(
        predictions=binary_array_strategy,
        labels=binary_array_strategy
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_confusion_matrix_entries_non_negative(
        self,
        predictions: List[int],
        labels: List[int]
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 10: Evaluation Metrics Completeness
        
        All confusion matrix entries SHALL be non-negative integers.
        
        **Validates: Requirements 7.5**
        """
        min_len = min(len(predictions), len(labels))
        assume(min_len >= 10)
        
        predictions = np.array(predictions[:min_len])
        labels = np.array(labels[:min_len])
        
        result = compute_evaluation_metrics(predictions, labels)
        
        # Property: All entries should be non-negative
        assert np.all(result.confusion_matrix >= 0), (
            f"Confusion matrix has negative entries: {result.confusion_matrix}"
        )



class TestEvaluationResultSerialization:
    """Tests for EvaluationResult serialization."""
    
    @given(
        predictions=binary_array_strategy,
        labels=binary_array_strategy
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_to_dict_contains_all_keys(
        self,
        predictions: List[int],
        labels: List[int]
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 10: Evaluation Metrics Completeness
        
        to_dict() SHALL contain all required metric keys.
        
        **Validates: Requirements 7.1, 7.2, 7.5**
        """
        min_len = min(len(predictions), len(labels))
        assume(min_len >= 10)
        
        predictions = np.array(predictions[:min_len])
        labels = np.array(labels[:min_len])
        
        result = compute_evaluation_metrics(predictions, labels)
        result_dict = result.to_dict()
        
        # Required keys
        required_keys = [
            'macro_f1',  # Requirement 7.1
            'f1_depressed', 'f1_non_depressed',  # Requirement 7.2
            'precision_depressed', 'precision_non_depressed',  # Requirement 7.2
            'recall_depressed', 'recall_non_depressed',  # Requirement 7.2
            'macro_precision', 'macro_recall',
            'confusion_matrix',  # Requirement 7.5
            'loss'
        ]
        
        for key in required_keys:
            assert key in result_dict, f"Missing key in to_dict(): {key}"
    
    def test_to_dict_values_are_serializable(self):
        """to_dict() values SHALL be JSON-serializable."""
        import json
        
        predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        
        result = compute_evaluation_metrics(predictions, labels)
        result_dict = result.to_dict()
        
        # Should not raise
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0


class TestCompareModels:
    """Tests for model comparison functionality."""
    
    def test_compare_models_generates_table(self):
        """compare_models SHALL generate a formatted comparison table."""
        predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        
        result = compute_evaluation_metrics(predictions, labels)
        
        results = {
            'Audio-Only': result,
            'Text-Only': result,
            'Multimodal': result
        }
        
        table = compare_models(results)
        
        # Table should contain model names
        assert 'Audio-Only' in table
        assert 'Text-Only' in table
        assert 'Multimodal' in table
        
        # Table should contain metric headers
        assert 'Macro-F1' in table
        assert 'F1(Dep)' in table
        assert 'Precision' in table
        assert 'Recall' in table
    
    @given(num_models=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, deadline=None)
    def test_compare_models_handles_variable_model_count(self, num_models: int):
        """compare_models SHALL handle variable number of models."""
        predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        
        result = compute_evaluation_metrics(predictions, labels)
        
        results = {f'Model_{i}': result for i in range(num_models)}
        
        table = compare_models(results)
        
        # All model names should appear in table
        for model_name in results.keys():
            assert model_name in table


class TestEdgeCases:
    """Tests for edge cases in evaluation metrics."""
    
    def test_all_same_predictions(self):
        """Metrics SHALL handle all-same predictions gracefully."""
        predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        
        result = compute_evaluation_metrics(predictions, labels)
        
        # Should not raise and should have valid values
        assert 0.0 <= result.macro_f1 <= 1.0
        assert result.confusion_matrix.shape == (2, 2)
    
    def test_all_correct_predictions(self):
        """Metrics SHALL handle perfect predictions."""
        labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        predictions = labels.copy()
        
        result = compute_evaluation_metrics(predictions, labels)
        
        # Perfect predictions should give F1 = 1.0
        assert result.macro_f1 == 1.0
        assert result.f1_depressed == 1.0
        assert result.f1_non_depressed == 1.0
    
    def test_all_wrong_predictions(self):
        """Metrics SHALL handle all-wrong predictions."""
        labels = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        predictions = 1 - labels  # Flip all predictions
        
        result = compute_evaluation_metrics(predictions, labels)
        
        # All wrong should give F1 = 0.0
        assert result.macro_f1 == 0.0
        assert result.f1_depressed == 0.0
        assert result.f1_non_depressed == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
