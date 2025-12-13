"""
Threshold Tuning for Macro-F1 Optimization

This module implements the threshold optimization algorithm
as described in Section 8: Threshold Tuning for Macro-F1 Optimization

The algorithm finds the optimal decision threshold that maximizes
the Macro-F1 score on the validation set, rather than using the
default 0.5 threshold which often yields suboptimal results on
imbalanced datasets.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from sklearn.metrics import f1_score, precision_score, recall_score
from dataclasses import dataclass
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""
    optimal_threshold: float
    optimal_macro_f1: float
    f1_depressed: float
    f1_non_depressed: float
    precision: float
    recall: float
    all_thresholds: List[float]
    all_f1_scores: List[float]


class MacroF1ThresholdOptimizer:
    """Threshold optimization for Macro-F1 score.
    
    Implements Section 8: Grid search over thresholds to find
    the one that maximizes Macro-F1 on the validation set.
    
    The optimal threshold is often significantly lower than 0.5
    for imbalanced datasets where the positive class (Depressed)
    is rare.
    """
    
    def __init__(
        self,
        threshold_min: float = 0.01,
        threshold_max: float = 0.99,
        threshold_step: float = 0.01
    ):
        """
        Args:
            threshold_min: Minimum threshold to search
            threshold_max: Maximum threshold to search
            threshold_step: Step size for grid search
        """
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step
    
    def optimize(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        positive_class: int = 1
    ) -> ThresholdResult:
        """Find optimal threshold using grid search.
        
        Args:
            probabilities: Predicted probabilities for positive class (batch,)
            labels: Ground truth labels (batch,)
            positive_class: Index of positive class (default: 1 for depressed)
            
        Returns:
            ThresholdResult with optimal threshold and metrics
        """
        # Ensure probabilities are for positive class
        if probabilities.ndim == 2:
            probabilities = probabilities[:, positive_class]
        
        # Generate threshold candidates
        thresholds = np.arange(
            self.threshold_min,
            self.threshold_max + self.threshold_step,
            self.threshold_step
        )
        
        best_threshold = 0.5
        best_f1 = 0.0
        best_metrics = {}
        all_f1_scores = []
        
        for threshold in thresholds:
            # Generate predictions at this threshold
            predictions = (probabilities >= threshold).astype(int)
            
            # Compute Macro-F1
            f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
            
            # Handle edge cases
            if len(f1_per_class) < 2:
                f1_non_dep = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
                f1_dep = 0.0
            else:
                f1_non_dep = f1_per_class[0]
                f1_dep = f1_per_class[1]
            
            macro_f1 = (f1_dep + f1_non_dep) / 2
            all_f1_scores.append(macro_f1)
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_threshold = threshold
                best_metrics = {
                    'f1_depressed': f1_dep,
                    'f1_non_depressed': f1_non_dep,
                    'precision': precision_score(labels, predictions, average='macro', zero_division=0),
                    'recall': recall_score(labels, predictions, average='macro', zero_division=0)
                }
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} (Macro-F1: {best_f1:.4f})")
        logger.info(f"  F1 (Depressed): {best_metrics['f1_depressed']:.4f}")
        logger.info(f"  F1 (Non-Depressed): {best_metrics['f1_non_depressed']:.4f}")
        
        return ThresholdResult(
            optimal_threshold=best_threshold,
            optimal_macro_f1=best_f1,
            f1_depressed=best_metrics['f1_depressed'],
            f1_non_depressed=best_metrics['f1_non_depressed'],
            precision=best_metrics['precision'],
            recall=best_metrics['recall'],
            all_thresholds=thresholds.tolist(),
            all_f1_scores=all_f1_scores
        )
    
    def optimize_subject_level(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        speaker_ids: List[str],
        aggregation: str = 'mean'
    ) -> ThresholdResult:
        """Find optimal threshold at subject level.
        
        Aggregates chunk predictions by speaker before threshold search.
        
        Args:
            probabilities: Chunk-level probabilities
            labels: Chunk-level labels
            speaker_ids: Speaker ID for each chunk
            aggregation: 'mean' or 'max' for chunk aggregation
            
        Returns:
            ThresholdResult optimized at subject level
        """
        # Aggregate by speaker
        speaker_probs = defaultdict(list)
        speaker_labels = {}
        
        for prob, label, speaker in zip(probabilities, labels, speaker_ids):
            speaker_probs[speaker].append(prob)
            speaker_labels[speaker] = label
        
        # Compute aggregated probabilities
        agg_probs = []
        agg_labels = []
        
        for speaker, probs in speaker_probs.items():
            if aggregation == 'mean':
                agg_prob = np.mean(probs)
            elif aggregation == 'max':
                agg_prob = np.max(probs)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            agg_probs.append(agg_prob)
            agg_labels.append(speaker_labels[speaker])
        
        agg_probs = np.array(agg_probs)
        agg_labels = np.array(agg_labels)
        
        logger.info(f"Subject-level optimization with {len(agg_probs)} subjects")
        
        return self.optimize(agg_probs, agg_labels)


class AdaptiveThresholdOptimizer:
    """Adaptive threshold optimization using binary search.
    
    More efficient than grid search for fine-grained optimization.
    """
    
    def __init__(
        self,
        tolerance: float = 0.001,
        max_iterations: int = 50
    ):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def _compute_f1(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float
    ) -> float:
        """Compute Macro-F1 at given threshold."""
        preds = (probs >= threshold).astype(int)
        return f1_score(labels, preds, average='macro', zero_division=0)
    
    def optimize(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, float]:
        """Find optimal threshold using golden section search.
        
        More efficient than grid search for smooth F1 curves.
        """
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        a, b = 0.01, 0.99
        
        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio
        
        fc = self._compute_f1(probabilities, labels, c)
        fd = self._compute_f1(probabilities, labels, d)
        
        for _ in range(self.max_iterations):
            if abs(b - a) < self.tolerance:
                break
            
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / golden_ratio
                fc = self._compute_f1(probabilities, labels, c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / golden_ratio
                fd = self._compute_f1(probabilities, labels, d)
        
        optimal_threshold = (a + b) / 2
        optimal_f1 = self._compute_f1(probabilities, labels, optimal_threshold)
        
        return optimal_threshold, optimal_f1


class CrossValidatedThresholdOptimizer:
    """Cross-validated threshold optimization to prevent overfitting.
    
    Uses inner cross-validation on validation set to find robust threshold.
    """
    
    def __init__(
        self,
        n_splits: int = 3,
        threshold_min: float = 0.01,
        threshold_max: float = 0.99,
        threshold_step: float = 0.02
    ):
        self.n_splits = n_splits
        self.base_optimizer = MacroF1ThresholdOptimizer(
            threshold_min, threshold_max, threshold_step
        )
    
    def optimize(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        speaker_ids: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """Find optimal threshold using cross-validation.
        
        Args:
            probabilities: Predicted probabilities
            labels: Ground truth labels
            speaker_ids: Optional speaker IDs for grouped CV
            
        Returns:
            Tuple of (optimal threshold, mean F1 across folds)
        """
        n_samples = len(probabilities)
        indices = np.arange(n_samples)
        
        # Simple k-fold (ignoring speaker grouping for simplicity)
        fold_size = n_samples // self.n_splits
        
        optimal_thresholds = []
        
        for fold in range(self.n_splits):
            # Create train/val split
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_splits - 1 else n_samples
            
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            # Find optimal threshold on this fold's training set
            result = self.base_optimizer.optimize(
                probabilities[train_idx],
                labels[train_idx]
            )
            optimal_thresholds.append(result.optimal_threshold)
        
        # Average thresholds across folds
        final_threshold = np.mean(optimal_thresholds)
        
        # Evaluate on full validation set
        final_preds = (probabilities >= final_threshold).astype(int)
        final_f1 = f1_score(labels, final_preds, average='macro', zero_division=0)
        
        logger.info(f"Cross-validated threshold: {final_threshold:.3f}")
        logger.info(f"Threshold variance: {np.std(optimal_thresholds):.4f}")
        
        return final_threshold, final_f1


def optimize_threshold_for_ensemble(
    ssl_probs: np.ndarray,
    ecapa_probs: np.ndarray,
    labels: np.ndarray,
    fusion_weight: float = 0.5,
    threshold_optimizer: Optional[MacroF1ThresholdOptimizer] = None
) -> Tuple[float, float, float]:
    """Optimize threshold for ensemble predictions.
    
    First fuses predictions, then optimizes threshold.
    
    Args:
        ssl_probs: Probabilities from SSL model
        ecapa_probs: Probabilities from ECAPA model
        labels: Ground truth labels
        fusion_weight: Weight for SSL model (1-weight for ECAPA)
        threshold_optimizer: Optimizer instance
        
    Returns:
        Tuple of (optimal threshold, optimal F1, fused probabilities mean)
    """
    if threshold_optimizer is None:
        threshold_optimizer = MacroF1ThresholdOptimizer()
    
    # Fuse predictions
    fused_probs = fusion_weight * ssl_probs + (1 - fusion_weight) * ecapa_probs
    
    # Optimize threshold
    result = threshold_optimizer.optimize(fused_probs, labels)
    
    return result.optimal_threshold, result.optimal_macro_f1, np.mean(fused_probs)


def plot_threshold_curve(result: ThresholdResult, save_path: Optional[str] = None):
    """Plot F1 score vs threshold curve.
    
    Useful for visualizing the threshold landscape.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(result.all_thresholds, result.all_f1_scores, 'b-', linewidth=2)
        ax.axvline(result.optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal: {result.optimal_threshold:.3f}')
        ax.axhline(result.optimal_macro_f1, color='g', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Decision Threshold', fontsize=12)
        ax.set_ylabel('Macro-F1 Score', fontsize=12)
        ax.set_title('Threshold Optimization for Macro-F1', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved threshold curve to {save_path}")
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    # Test threshold optimization
    np.random.seed(42)
    
    # Simulate imbalanced predictions
    n_samples = 1000
    labels = np.zeros(n_samples, dtype=int)
    labels[:200] = 1  # 20% positive (depressed)
    
    # Generate probabilities (model slightly better than random)
    probs = np.random.beta(2, 5, n_samples)  # Skewed towards 0
    probs[labels == 1] = np.random.beta(5, 3, 200)  # Higher for positive class
    
    print("Testing MacroF1ThresholdOptimizer...")
    optimizer = MacroF1ThresholdOptimizer()
    result = optimizer.optimize(probs, labels)
    
    print(f"\nOptimal threshold: {result.optimal_threshold:.3f}")
    print(f"Optimal Macro-F1: {result.optimal_macro_f1:.4f}")
    
    # Compare with default threshold
    default_preds = (probs >= 0.5).astype(int)
    default_f1 = f1_score(labels, default_preds, average='macro')
    print(f"\nDefault (0.5) Macro-F1: {default_f1:.4f}")
    print(f"Improvement: {(result.optimal_macro_f1 - default_f1) * 100:.2f}%")
    
    # Test subject-level optimization
    print("\n" + "=" * 50)
    print("Testing subject-level optimization...")
    speaker_ids = [f"speaker_{i // 10}" for i in range(n_samples)]
    
    result_subject = optimizer.optimize_subject_level(probs, labels, speaker_ids)
    print(f"Subject-level optimal threshold: {result_subject.optimal_threshold:.3f}")
    print(f"Subject-level Macro-F1: {result_subject.optimal_macro_f1:.4f}")
