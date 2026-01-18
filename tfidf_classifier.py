"""
TF-IDF Classifier Module for Depression Detection

This module provides a traditional NLP baseline using TF-IDF vectorization
with SVM classification for depression detection from text transcripts.

Reference: Multimodal NLP Upgrade - Requirement 3

Feature: multimodal-nlp-upgrade
"""

import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TFIDFConfig:
    """Configuration for TF-IDF Classifier.
    
    Reference: Multimodal NLP Upgrade - Requirement 3
    """
    # N-gram range for feature extraction
    # (1, 3) means unigrams, bigrams, and trigrams
    ngram_range: Tuple[int, int] = (1, 3)
    
    # Minimum document frequency for vocabulary
    # Features appearing in fewer than min_df documents are ignored
    min_df: int = 5
    
    # Maximum number of features to keep
    max_features: int = 5000
    
    # Classifier type: "svm" or "logistic_regression"
    classifier: str = "svm"
    
    # SVM parameters
    svm_kernel: str = "linear"
    svm_c: float = 1.0
    
    # Logistic regression parameters
    lr_c: float = 1.0
    lr_max_iter: int = 1000
    
    # Class weight handling for imbalanced data
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced"
    
    # Random state for reproducibility
    random_state: int = 42


class TFIDFClassifier:
    """TF-IDF based classifier for depression detection.
    
    Uses N-gram TF-IDF vectorization with SVM or Logistic Regression
    for binary classification (Depressed/Non-Depressed).
    
    Handles:
    1. N-gram feature extraction (unigrams, bigrams, trigrams) - Requirement 3.1
    2. Vocabulary constraints (min_df, max_features) - Requirement 3.2
    3. Classification predictions - Requirement 3.3
    4. Macro-F1 evaluation - Requirement 3.4
    5. Probability scores and predicted labels - Requirement 3.5
    
    Reference: Multimodal NLP Upgrade - Requirement 3
    """
    
    def __init__(self, config: Optional[TFIDFConfig] = None):
        """Initialize the TF-IDF classifier.
        
        Args:
            config: TFIDFConfig instance. If None, uses defaults.
        """
        self.config = config or TFIDFConfig()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._classifier: Optional[Union[SVC, LogisticRegression]] = None
        self._calibrated_classifier: Optional[CalibratedClassifierCV] = None
        self._is_fitted: bool = False
        self._classes: Optional[np.ndarray] = None
        
        self._init_vectorizer()
        self._init_classifier()
    
    def _init_vectorizer(self) -> None:
        """Initialize the TF-IDF vectorizer with configured parameters."""
        self._vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_features=self.config.max_features,
            sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
            strip_accents=None,  # Don't strip accents (important for Dravidian)
            lowercase=False,  # Don't lowercase (Dravidian scripts don't have case)
            token_pattern=r'(?u)\b\w+\b',  # Unicode-aware word tokenization
        )
        logger.info(
            f"Initialized TF-IDF vectorizer: "
            f"ngram_range={self.config.ngram_range}, "
            f"min_df={self.config.min_df}, "
            f"max_features={self.config.max_features}"
        )
    
    def _init_classifier(self) -> None:
        """Initialize the classifier (SVM or Logistic Regression)."""
        if self.config.classifier.lower() == "svm":
            # SVM with linear kernel for text classification
            self._classifier = SVC(
                kernel=self.config.svm_kernel,
                C=self.config.svm_c,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                probability=False  # We'll use calibration for probabilities
            )
            logger.info(
                f"Initialized SVM classifier: "
                f"kernel={self.config.svm_kernel}, C={self.config.svm_c}"
            )
        else:
            # Logistic Regression as alternative
            self._classifier = LogisticRegression(
                C=self.config.lr_c,
                max_iter=self.config.lr_max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                solver='lbfgs'
            )
            logger.info(
                f"Initialized Logistic Regression classifier: "
                f"C={self.config.lr_c}"
            )
    
    def fit(self, texts: List[str], labels: List[int]) -> "TFIDFClassifier":
        """Fit the TF-IDF vectorizer and classifier on training data.
        
        Args:
            texts: List of preprocessed text strings (morpheme-segmented)
            labels: List of integer labels (0 = Non-Depressed, 1 = Depressed)
            
        Returns:
            self for method chaining
            
        Reference: Requirement 3.1, 3.2
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")
        
        if len(texts) != len(labels):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of labels ({len(labels)})"
            )
        
        # Convert labels to numpy array
        labels_array = np.array(labels)
        self._classes = np.unique(labels_array)
        
        # Step 1: Fit and transform TF-IDF vectorizer
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts...")
        X = self._vectorizer.fit_transform(texts)
        
        actual_features = X.shape[1]
        logger.info(
            f"TF-IDF vectorization complete: "
            f"{X.shape[0]} samples, {actual_features} features"
        )
        
        # Step 2: Fit the classifier
        logger.info("Fitting classifier...")
        self._classifier.fit(X, labels_array)
        
        # Step 3: Create calibrated classifier for probability estimates
        # This is needed for SVM which doesn't natively support predict_proba
        if isinstance(self._classifier, SVC):
            logger.info("Calibrating classifier for probability estimates...")
            # Use cross-validation based calibration (cv=3 for small datasets)
            n_samples = X.shape[0]
            cv_folds = min(3, n_samples // 2) if n_samples >= 4 else 2
            self._calibrated_classifier = CalibratedClassifierCV(
                estimator=SVC(
                    kernel=self.config.svm_kernel,
                    C=self.config.svm_c,
                    class_weight=self.config.class_weight,
                    random_state=self.config.random_state,
                    probability=False
                ),
                cv=cv_folds,
                method='sigmoid'
            )
            self._calibrated_classifier.fit(X, labels_array)
        
        self._is_fitted = True
        logger.info("TF-IDF classifier training complete")
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict class labels for input texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Array of predicted labels (0 or 1)
            
        Reference: Requirement 3.3
        """
        self._check_is_fitted()
        
        if not texts:
            return np.array([], dtype=int)
        
        # Transform texts to TF-IDF features
        X = self._vectorizer.transform(texts)
        
        # Predict using classifier
        predictions = self._classifier.predict(X)
        
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probabilities for input texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
            
        Reference: Requirement 3.5
        """
        self._check_is_fitted()
        
        if not texts:
            return np.array([]).reshape(0, 2)
        
        # Transform texts to TF-IDF features
        X = self._vectorizer.transform(texts)
        
        # Get probabilities
        if isinstance(self._classifier, SVC):
            # Use calibrated classifier for SVM
            if self._calibrated_classifier is not None:
                probabilities = self._calibrated_classifier.predict_proba(X)
            else:
                # Fallback: use decision function and convert to pseudo-probabilities
                decision = self._classifier.decision_function(X)
                # Apply sigmoid to convert to probabilities
                prob_positive = 1 / (1 + np.exp(-decision))
                probabilities = np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Logistic Regression has native predict_proba
            probabilities = self._classifier.predict_proba(X)
        
        return probabilities
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, Any]:
        """Evaluate the classifier on test data.
        
        Computes Macro-F1 (primary metric), per-class F1, precision, recall,
        accuracy, and confusion matrix.
        
        Args:
            texts: List of preprocessed text strings
            labels: List of true labels
            
        Returns:
            Dictionary containing evaluation metrics
            
        Reference: Requirement 3.4
        """
        self._check_is_fitted()
        
        if not texts or not labels:
            raise ValueError("Cannot evaluate on empty data")
        
        # Get predictions
        predictions = self.predict(texts)
        labels_array = np.array(labels)
        
        # Compute metrics
        metrics = {
            # Primary metric: Macro-F1
            "macro_f1": f1_score(labels_array, predictions, average='macro'),
            
            # Per-class F1 scores
            "f1_non_depressed": f1_score(
                labels_array, predictions, pos_label=0, average='binary'
            ) if 0 in labels_array else 0.0,
            "f1_depressed": f1_score(
                labels_array, predictions, pos_label=1, average='binary'
            ) if 1 in labels_array else 0.0,
            
            # Precision and Recall (macro-averaged)
            "precision_macro": precision_score(
                labels_array, predictions, average='macro', zero_division=0
            ),
            "recall_macro": recall_score(
                labels_array, predictions, average='macro', zero_division=0
            ),
            
            # Per-class precision and recall
            "precision_non_depressed": precision_score(
                labels_array, predictions, pos_label=0, average='binary', zero_division=0
            ) if 0 in labels_array else 0.0,
            "precision_depressed": precision_score(
                labels_array, predictions, pos_label=1, average='binary', zero_division=0
            ) if 1 in labels_array else 0.0,
            "recall_non_depressed": recall_score(
                labels_array, predictions, pos_label=0, average='binary', zero_division=0
            ) if 0 in labels_array else 0.0,
            "recall_depressed": recall_score(
                labels_array, predictions, pos_label=1, average='binary', zero_division=0
            ) if 1 in labels_array else 0.0,
            
            # Accuracy
            "accuracy": accuracy_score(labels_array, predictions),
            
            # Confusion matrix
            "confusion_matrix": confusion_matrix(labels_array, predictions).tolist(),
            
            # Sample counts
            "n_samples": len(labels),
            "n_correct": int(np.sum(predictions == labels_array)),
        }
        
        return metrics
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names (vocabulary) from the fitted vectorizer.
        
        Returns:
            List of feature names (n-grams)
        """
        self._check_is_fitted()
        return self._vectorizer.get_feature_names_out().tolist()
    
    def get_feature_count(self) -> int:
        """Get the number of features in the fitted vectorizer.
        
        Returns:
            Number of features
        """
        self._check_is_fitted()
        return len(self._vectorizer.get_feature_names_out())
    
    def save(self, path: str) -> None:
        """Save the fitted classifier to disk.
        
        Saves both the vectorizer and classifier as a pickle file.
        
        Args:
            path: Path to save the model
        """
        self._check_is_fitted()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        model_data = {
            "config": self.config,
            "vectorizer": self._vectorizer,
            "classifier": self._classifier,
            "calibrated_classifier": self._calibrated_classifier,
            "classes": self._classes,
            "is_fitted": self._is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved TF-IDF classifier to: {path}")
    
    def load(self, path: str) -> "TFIDFClassifier":
        """Load a fitted classifier from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            self for method chaining
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data["config"]
        self._vectorizer = model_data["vectorizer"]
        self._classifier = model_data["classifier"]
        self._calibrated_classifier = model_data.get("calibrated_classifier")
        self._classes = model_data["classes"]
        self._is_fitted = model_data["is_fitted"]
        
        logger.info(f"Loaded TF-IDF classifier from: {path}")
        
        return self
    
    def _check_is_fitted(self) -> None:
        """Check if the classifier has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "Classifier has not been fitted. Call fit() first."
            )
    
    @property
    def is_fitted(self) -> bool:
        """Check if the classifier is fitted."""
        return self._is_fitted
    
    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Get the class labels."""
        return self._classes
    
    def get_classification_report(
        self,
        texts: List[str],
        labels: List[int]
    ) -> str:
        """Generate a detailed classification report.
        
        Args:
            texts: List of preprocessed text strings
            labels: List of true labels
            
        Returns:
            Formatted classification report string
        """
        self._check_is_fitted()
        
        predictions = self.predict(texts)
        
        target_names = ["Non-Depressed", "Depressed"]
        report = classification_report(
            labels,
            predictions,
            target_names=target_names,
            digits=4
        )
        
        return report


def create_tfidf_classifier(
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 5,
    max_features: int = 5000,
    classifier: str = "svm",
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced"
) -> TFIDFClassifier:
    """Factory function to create a TFIDFClassifier.
    
    Args:
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction
        min_df: Minimum document frequency for features
        max_features: Maximum number of features to keep
        classifier: Type of classifier ("svm" or "logistic_regression")
        class_weight: Class weight strategy for imbalanced data
        
    Returns:
        Configured TFIDFClassifier instance
    """
    config = TFIDFConfig(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        classifier=classifier,
        class_weight=class_weight
    )
    return TFIDFClassifier(config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    texts = [
        "நான் மிகவும் சோகமாக உணர்கிறேன்",  # Tamil: I feel very sad
        "எனக்கு நம்பிக்கை இல்லை",  # Tamil: I have no hope
        "நான் மகிழ்ச்சியாக இருக்கிறேன்",  # Tamil: I am happy
        "வாழ்க்கை அழகானது",  # Tamil: Life is beautiful
        "எல்லாம் கஷ்டமாக இருக்கிறது",  # Tamil: Everything is difficult
        "நான் நன்றாக இருக்கிறேன்",  # Tamil: I am fine
    ] * 10  # Repeat to meet min_df requirement
    
    labels = [1, 1, 0, 0, 1, 0] * 10  # 1 = Depressed, 0 = Non-Depressed
    
    # Create and train classifier
    classifier = create_tfidf_classifier(
        ngram_range=(1, 3),
        min_df=2,  # Lower for demo
        max_features=1000,
        classifier="svm"
    )
    
    classifier.fit(texts, labels)
    
    # Evaluate
    metrics = classifier.evaluate(texts, labels)
    
    print("\nEvaluation Metrics:")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    
    # Test prediction
    test_texts = ["நான் சோகமாக இருக்கிறேன்"]  # I am sad
    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)
    
    print(f"\nTest Prediction:")
    print(f"  Text: {test_texts[0]}")
    print(f"  Predicted Label: {predictions[0]}")
    print(f"  Probabilities: {probabilities[0]}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classifier.get_classification_report(texts, labels))
