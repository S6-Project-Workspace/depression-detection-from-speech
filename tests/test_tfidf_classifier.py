"""
Property-Based Tests for TF-IDF Classifier Module

Tests the TF-IDF classifier using hypothesis for property-based testing.
Validates Requirements 3.1, 3.2, 3.3, and 3.5 for feature extraction and classification.

Feature: multimodal-nlp-upgrade
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import List, Tuple

from tfidf_classifier import (
    TFIDFClassifier,
    TFIDFConfig,
    create_tfidf_classifier
)


# Strategy for generating text samples (simple ASCII for reliability)
simple_text = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N'),  # Letters and numbers
        whitelist_characters=' ',
        blacklist_categories=('Cs',)  # Exclude surrogates
    ),
    min_size=1,
    max_size=100
)

# Strategy for generating a corpus of texts
text_corpus = st.lists(
    simple_text,
    min_size=10,  # Need enough samples for min_df
    max_size=50
)

# Strategy for generating binary labels
binary_label = st.integers(min_value=0, max_value=1)

# Strategy for generating labeled corpus
labeled_corpus = st.tuples(
    text_corpus,
    st.lists(binary_label, min_size=10, max_size=50)
).filter(lambda x: len(x[0]) == len(x[1]))


# Strategy for generating TF-IDF config parameters
tfidf_config_params = st.fixed_dictionaries({
    'ngram_min': st.integers(min_value=1, max_value=2),
    'ngram_max': st.integers(min_value=1, max_value=3),
    'min_df': st.integers(min_value=1, max_value=5),
    'max_features': st.integers(min_value=100, max_value=5000)
}).filter(lambda x: x['ngram_min'] <= x['ngram_max'])


class TestTFIDFFeatureDimensionBounds:
    """
    Property 5: TF-IDF Feature Dimension Bounds
    
    *For any* corpus of texts, the TF-IDF vectorizer SHALL produce a feature matrix
    with at most max_features columns (5000) and features appearing in at least
    min_df documents (5).
    
    **Validates: Requirements 3.1, 3.2**
    """
    
    @given(
        texts=st.lists(
            st.text(
                alphabet=st.sampled_from(list('abcdefghijklmnopqrstuvwxyz ')),
                min_size=5,
                max_size=50
            ),
            min_size=20,
            max_size=100
        ),
        max_features=st.integers(min_value=10, max_value=1000),
        min_df=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_feature_count_respects_max_features(
        self,
        texts: List[str],
        max_features: int,
        min_df: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 5: TF-IDF Feature Dimension Bounds
        
        For any corpus, the number of features SHALL NOT exceed max_features.
        **Validates: Requirements 3.1, 3.2**
        """
        # Filter out empty texts and ensure we have enough samples
        texts = [t for t in texts if t.strip()]
        assume(len(texts) >= 10)
        
        # Generate labels (balanced)
        labels = [i % 2 for i in range(len(texts))]
        
        # Create classifier with specified parameters
        classifier = create_tfidf_classifier(
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            classifier="svm"
        )
        
        # Fit the classifier
        try:
            classifier.fit(texts, labels)
        except ValueError:
            # May fail if vocabulary is empty after filtering
            assume(False)
        
        # Property: Feature count should not exceed max_features
        feature_count = classifier.get_feature_count()
        assert feature_count <= max_features, (
            f"Feature count ({feature_count}) exceeds max_features ({max_features})"
        )
    
    @given(
        ngram_min=st.integers(min_value=1, max_value=2),
        ngram_max=st.integers(min_value=2, max_value=3)
    )
    @settings(max_examples=100, deadline=None)
    def test_ngram_range_produces_valid_features(
        self,
        ngram_min: int,
        ngram_max: int
    ):
        """
        Feature: multimodal-nlp-upgrade, Property 5: TF-IDF Feature Dimension Bounds
        
        For any valid ngram_range, the vectorizer SHALL produce features.
        **Validates: Requirements 3.1**
        """
        assume(ngram_min <= ngram_max)
        
        # Create a corpus with repeated words to ensure features pass min_df
        texts = [
            "hello world test",
            "hello world example",
            "hello test sample",
            "world test data",
            "hello world test data",
            "example test sample",
            "world example data",
            "hello sample data",
            "test example world",
            "sample data hello"
        ]
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        
        classifier = create_tfidf_classifier(
            ngram_range=(ngram_min, ngram_max),
            min_df=2,
            max_features=5000,
            classifier="svm"
        )
        
        classifier.fit(texts, labels)
        
        # Property: Should have at least some features
        feature_count = classifier.get_feature_count()
        assert feature_count > 0, (
            f"No features extracted with ngram_range=({ngram_min}, {ngram_max})"
        )
        
        # Property: Feature names should be valid strings
        feature_names = classifier.get_feature_names()
        assert all(isinstance(name, str) for name in feature_names), (
            "Feature names should all be strings"
        )
    
    @given(max_features=st.integers(min_value=50, max_value=500))
    @settings(max_examples=100, deadline=None)
    def test_default_config_respects_bounds(self, max_features: int):
        """
        Feature: multimodal-nlp-upgrade, Property 5: TF-IDF Feature Dimension Bounds
        
        For any max_features setting, the vectorizer SHALL respect the bound.
        **Validates: Requirements 3.2**
        """
        # Create corpus with many unique words
        base_words = ["word", "test", "sample", "data", "example", "hello", "world"]
        texts = []
        for i in range(30):
            # Create texts with variations
            text = " ".join([f"{w}{i % 5}" for w in base_words[:3 + i % 4]])
            texts.append(text)
        
        labels = [i % 2 for i in range(len(texts))]
        
        classifier = create_tfidf_classifier(
            ngram_range=(1, 3),
            min_df=1,
            max_features=max_features,
            classifier="svm"
        )
        
        classifier.fit(texts, labels)
        
        # Property: Feature count should not exceed max_features
        feature_count = classifier.get_feature_count()
        assert feature_count <= max_features, (
            f"Feature count ({feature_count}) exceeds max_features ({max_features})"
        )


class TestClassifierPredictionValidity:
    """
    Property 6: Classifier Prediction Validity
    
    *For any* input to the TF-IDF classifier, the predicted label SHALL be in
    the valid label set {0, 1} and the probability SHALL be in the range [0, 1].
    
    **Validates: Requirements 3.3, 3.5**
    """
    
    @given(
        test_texts=st.lists(
            st.text(
                alphabet=st.sampled_from(list('abcdefghijklmnopqrstuvwxyz ')),
                min_size=1,
                max_size=50
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_predictions_are_valid_labels(self, test_texts: List[str]):
        """
        Feature: multimodal-nlp-upgrade, Property 6: Classifier Prediction Validity
        
        For any input, predicted labels SHALL be in {0, 1}.
        **Validates: Requirements 3.3**
        """
        # Filter empty texts
        test_texts = [t for t in test_texts if t.strip()]
        assume(len(test_texts) > 0)
        
        # Create and fit classifier with training data
        train_texts = [
            "positive happy good",
            "negative sad bad",
            "positive great excellent",
            "negative terrible awful",
            "positive wonderful amazing",
            "negative horrible poor",
            "positive fantastic super",
            "negative miserable depressed",
            "positive joyful cheerful",
            "negative gloomy dark"
        ]
        train_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        
        classifier = create_tfidf_classifier(
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000,
            classifier="svm"
        )
        classifier.fit(train_texts, train_labels)
        
        # Get predictions
        predictions = classifier.predict(test_texts)
        
        # Property: All predictions should be valid labels (0 or 1)
        assert all(pred in [0, 1] for pred in predictions), (
            f"Invalid predictions found: {predictions}"
        )
        
        # Property: Number of predictions should match number of inputs
        assert len(predictions) == len(test_texts), (
            f"Prediction count ({len(predictions)}) doesn't match "
            f"input count ({len(test_texts)})"
        )
    
    @given(
        test_texts=st.lists(
            st.text(
                alphabet=st.sampled_from(list('abcdefghijklmnopqrstuvwxyz ')),
                min_size=1,
                max_size=50
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_probabilities_are_valid(self, test_texts: List[str]):
        """
        Feature: multimodal-nlp-upgrade, Property 6: Classifier Prediction Validity
        
        For any input, probabilities SHALL be in range [0, 1] and sum to 1.
        **Validates: Requirements 3.5**
        """
        # Filter empty texts
        test_texts = [t for t in test_texts if t.strip()]
        assume(len(test_texts) > 0)
        
        # Create and fit classifier
        train_texts = [
            "positive happy good",
            "negative sad bad",
            "positive great excellent",
            "negative terrible awful",
            "positive wonderful amazing",
            "negative horrible poor",
            "positive fantastic super",
            "negative miserable depressed",
            "positive joyful cheerful",
            "negative gloomy dark"
        ]
        train_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        
        classifier = create_tfidf_classifier(
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000,
            classifier="svm"
        )
        classifier.fit(train_texts, train_labels)
        
        # Get probabilities
        probabilities = classifier.predict_proba(test_texts)
        
        # Property: Probabilities should be 2D array with shape (n_samples, 2)
        assert probabilities.shape == (len(test_texts), 2), (
            f"Probability shape {probabilities.shape} doesn't match "
            f"expected ({len(test_texts)}, 2)"
        )
        
        # Property: All probabilities should be in [0, 1]
        assert np.all(probabilities >= 0), (
            f"Found negative probabilities: {probabilities[probabilities < 0]}"
        )
        assert np.all(probabilities <= 1), (
            f"Found probabilities > 1: {probabilities[probabilities > 1]}"
        )
        
        # Property: Probabilities should sum to approximately 1 for each sample
        row_sums = np.sum(probabilities, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"Probability rows don't sum to 1: {row_sums}"
        )
    
    @given(classifier_type=st.sampled_from(["svm", "logistic_regression"]))
    @settings(max_examples=100, deadline=None)
    def test_both_classifier_types_produce_valid_output(self, classifier_type: str):
        """
        Feature: multimodal-nlp-upgrade, Property 6: Classifier Prediction Validity
        
        For both SVM and Logistic Regression, outputs SHALL be valid.
        **Validates: Requirements 3.3, 3.5**
        """
        # Training data
        train_texts = [
            "positive happy good",
            "negative sad bad",
            "positive great excellent",
            "negative terrible awful",
            "positive wonderful amazing",
            "negative horrible poor",
            "positive fantastic super",
            "negative miserable depressed",
            "positive joyful cheerful",
            "negative gloomy dark"
        ]
        train_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        
        # Test data
        test_texts = ["happy good day", "sad terrible night", "neutral text"]
        
        classifier = create_tfidf_classifier(
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000,
            classifier=classifier_type
        )
        classifier.fit(train_texts, train_labels)
        
        # Test predictions
        predictions = classifier.predict(test_texts)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probabilities
        probabilities = classifier.predict_proba(test_texts)
        assert probabilities.shape == (len(test_texts), 2)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-5)
    
    @given(
        n_samples=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_empty_input_handling(self, n_samples: int):
        """
        Feature: multimodal-nlp-upgrade, Property 6: Classifier Prediction Validity
        
        For empty input list, classifier SHALL return empty arrays.
        **Validates: Requirements 3.3, 3.5**
        """
        # Create and fit classifier with diverse training data
        train_texts = [
            "positive happy good excellent",
            "negative sad bad terrible",
            "positive great wonderful amazing",
            "negative awful horrible poor",
            "positive fantastic super joyful",
            "negative miserable depressed gloomy",
            "positive cheerful bright sunny",
            "negative dark dreary hopeless",
            "positive optimistic hopeful bright",
            "negative pessimistic worried anxious"
        ]
        train_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        
        classifier = create_tfidf_classifier(
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000,
            classifier="svm"
        )
        classifier.fit(train_texts, train_labels)
        
        # Test with empty input
        empty_predictions = classifier.predict([])
        empty_probabilities = classifier.predict_proba([])
        
        # Property: Empty input should return empty arrays
        assert len(empty_predictions) == 0
        assert empty_probabilities.shape == (0, 2)


class TestClassifierEvaluation:
    """Tests for classifier evaluation metrics."""
    
    def test_evaluate_returns_all_metrics(self):
        """
        Evaluate method should return all required metrics.
        """
        # Training data
        train_texts = [
            "positive happy good",
            "negative sad bad",
            "positive great excellent",
            "negative terrible awful",
            "positive wonderful amazing",
            "negative horrible poor",
            "positive fantastic super",
            "negative miserable depressed",
            "positive joyful cheerful",
            "negative gloomy dark"
        ]
        train_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        
        classifier = create_tfidf_classifier(
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000,
            classifier="svm"
        )
        classifier.fit(train_texts, train_labels)
        
        # Evaluate
        metrics = classifier.evaluate(train_texts, train_labels)
        
        # Check all required metrics are present
        required_metrics = [
            "macro_f1",
            "f1_non_depressed",
            "f1_depressed",
            "precision_macro",
            "recall_macro",
            "accuracy",
            "confusion_matrix",
            "n_samples",
            "n_correct"
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Check metric values are valid
        assert 0 <= metrics["macro_f1"] <= 1
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["n_samples"] == len(train_labels)


class TestClassifierPersistence:
    """Tests for model save/load functionality."""
    
    def test_save_and_load_preserves_model(self, tmp_path):
        """
        Save and load should preserve model state.
        """
        # Training data
        train_texts = [
            "positive happy good",
            "negative sad bad",
            "positive great excellent",
            "negative terrible awful",
            "positive wonderful amazing",
            "negative horrible poor"
        ]
        train_labels = [0, 1, 0, 1, 0, 1]
        
        # Create and fit classifier
        classifier = create_tfidf_classifier(
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000,
            classifier="svm"
        )
        classifier.fit(train_texts, train_labels)
        
        # Get predictions before save
        test_texts = ["happy good", "sad bad"]
        predictions_before = classifier.predict(test_texts)
        
        # Save model
        model_path = tmp_path / "tfidf_model.pkl"
        classifier.save(str(model_path))
        
        # Load into new classifier
        new_classifier = TFIDFClassifier()
        new_classifier.load(str(model_path))
        
        # Get predictions after load
        predictions_after = new_classifier.predict(test_texts)
        
        # Predictions should be identical
        assert np.array_equal(predictions_before, predictions_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
