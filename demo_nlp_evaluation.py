#!/usr/bin/env python3
"""
Demonstration of NLP Evaluation Module

This script demonstrates the comprehensive NLP evaluation capabilities
for traditional NLP tasks (POS tagging, NER, dependency parsing).

Usage:
    python demo_nlp_evaluation.py
"""

import numpy as np
from typing import List, Dict, Any

from nlp_evaluation import (
    POSEvaluationMetrics,
    NEREvaluationMetrics,
    DependencyEvaluationMetrics,
    ComprehensiveEvaluationResult,
    POSEvaluator,
    NEREvaluator,
    DependencyEvaluator,
    ComprehensiveNLPEvaluator,
    compare_nlp_systems,
    statistical_significance_test
)

from pos_tagger import POSTagger, POSConfig
from ner_system import NERSystem, NERConfig, Entity
from dependency_parser import DependencyParser, DependencyConfig, DependencyTree
from linguistic_analyzer import LinguisticAnalyzer, LinguisticConfig


def create_sample_test_data() -> Dict[str, Any]:
    """Create sample test data for demonstration."""
    
    # Sample POS test data
    pos_test_data = [
        (['நான்', 'வீட்டிற்கு', 'செல்கிறேன்'], ['PRON', 'NOUN', 'VERB']),
        (['அவர்', 'புத்தகம்', 'படிக்கிறார்'], ['PRON', 'NOUN', 'VERB']),
        (['இது', 'நல்ல', 'நாள்'], ['PRON', 'ADJ', 'NOUN'])
    ]
    
    # Sample NER test data
    ner_test_data = [
        (['John', 'lives', 'in', 'Chennai'], [
            Entity('John', 'PERSON', 0, 1, 0.9, '[PERSON_1]'),
            Entity('Chennai', 'LOCATION', 3, 4, 0.8, '[LOCATION_1]')
        ]),
        (['Dr.', 'Smith', 'prescribed', 'medication'], [
            Entity('Dr. Smith', 'PERSON', 0, 2, 0.95, '[PERSON_2]'),
            Entity('medication', 'MEDICAL', 3, 4, 0.7, '[MEDICAL_1]')
        ])
    ]
    
    # Sample dependency test data
    dependency_test_data = [
        (['John', 'walks'], DependencyTree(
            tokens=['John', 'walks'],
            heads=[1, -1],  # John depends on walks, walks is root
            relations=['nsubj', 'root'],
            confidence_scores=[0.9, 0.95]
        )),
        (['The', 'cat', 'sleeps'], DependencyTree(
            tokens=['The', 'cat', 'sleeps'],
            heads=[1, 2, -1],  # The->cat, cat->sleeps, sleeps=root
            relations=['det', 'nsubj', 'root'],
            confidence_scores=[0.8, 0.9, 0.95]
        ))
    ]
    
    return {
        'pos': pos_test_data,
        'ner': ner_test_data,
        'dependency': dependency_test_data
    }


def demo_individual_evaluators():
    """Demonstrate individual task evaluators."""
    print("=" * 80)
    print("DEMONSTRATION: Individual NLP Task Evaluators")
    print("=" * 80)
    
    print("\nNote: This demonstration shows the evaluation data structures")
    print("and metrics. In practice, these would be computed from actual")
    print("model predictions on test data.")
    
    # 1. POS Tagger Evaluation Metrics
    print("\n1. POS Tagger Evaluation Metrics")
    print("-" * 40)
    
    pos_metrics = POSEvaluationMetrics(
        accuracy=0.85,
        macro_f1=0.82,
        macro_precision=0.83,
        macro_recall=0.81,
        per_tag_f1={'NOUN': 0.9, 'VERB': 0.8, 'ADJ': 0.75, 'PRON': 0.88},
        per_tag_precision={'NOUN': 0.92, 'VERB': 0.78, 'ADJ': 0.73, 'PRON': 0.90},
        per_tag_recall={'NOUN': 0.88, 'VERB': 0.82, 'ADJ': 0.77, 'PRON': 0.86},
        confusion_matrix=np.array([[15, 2, 1, 0], [1, 18, 2, 1], [2, 1, 12, 0], [0, 1, 0, 14]]),
        tag_labels=['NOUN', 'VERB', 'ADJ', 'PRON'],
        oov_accuracy=0.75,
        tag_distribution={'NOUN': 18, 'VERB': 22, 'ADJ': 15, 'PRON': 15}
    )
    
    print(f"   Overall Accuracy: {pos_metrics.accuracy:.4f}")
    print(f"   Macro-F1: {pos_metrics.macro_f1:.4f}")
    print(f"   Macro Precision: {pos_metrics.macro_precision:.4f}")
    print(f"   Macro Recall: {pos_metrics.macro_recall:.4f}")
    print(f"   OOV Accuracy: {pos_metrics.oov_accuracy:.4f}")
    print(f"   Per-tag F1 scores:")
    for tag, f1 in pos_metrics.per_tag_f1.items():
        print(f"     {tag}: {f1:.4f}")
    
    # 2. NER System Evaluation Metrics
    print("\n2. NER System Evaluation Metrics")
    print("-" * 40)
    
    ner_metrics = NEREvaluationMetrics(
        entity_f1=0.78,
        entity_precision=0.80,
        entity_recall=0.76,
        token_f1=0.85,
        token_precision=0.87,
        token_recall=0.83,
        per_type_f1={'PERSON': 0.9, 'LOCATION': 0.7, 'MEDICAL': 0.65, 'ORGANIZATION': 0.72},
        per_type_precision={'PERSON': 0.92, 'LOCATION': 0.68, 'MEDICAL': 0.62, 'ORGANIZATION': 0.75},
        per_type_recall={'PERSON': 0.88, 'LOCATION': 0.72, 'MEDICAL': 0.68, 'ORGANIZATION': 0.69},
        confusion_matrix=np.array([[12, 1, 0, 1], [2, 15, 1, 0], [1, 0, 10, 2], [0, 1, 1, 11]]),
        entity_labels=['PERSON', 'LOCATION', 'MEDICAL', 'ORGANIZATION'],
        entity_density=2.5,
        boundary_accuracy=0.82,
        type_distribution={'PERSON': 14, 'LOCATION': 18, 'MEDICAL': 13, 'ORGANIZATION': 13}
    )
    
    print(f"   Entity-level F1: {ner_metrics.entity_f1:.4f}")
    print(f"   Entity Precision: {ner_metrics.entity_precision:.4f}")
    print(f"   Entity Recall: {ner_metrics.entity_recall:.4f}")
    print(f"   Token-level F1: {ner_metrics.token_f1:.4f}")
    print(f"   Boundary Accuracy: {ner_metrics.boundary_accuracy:.4f}")
    print(f"   Entity Density: {ner_metrics.entity_density:.2f} entities/sentence")
    print(f"   Per-type F1 scores:")
    for entity_type, f1 in ner_metrics.per_type_f1.items():
        print(f"     {entity_type}: {f1:.4f}")
    
    # 3. Dependency Parser Evaluation Metrics
    print("\n3. Dependency Parser Evaluation Metrics")
    print("-" * 40)
    
    dep_metrics = DependencyEvaluationMetrics(
        las=0.72,
        uas=0.78,
        per_relation_f1={'nsubj': 0.8, 'obj': 0.7, 'root': 0.9, 'det': 0.85, 'amod': 0.68},
        per_relation_precision={'nsubj': 0.82, 'obj': 0.68, 'root': 0.92, 'det': 0.87, 'amod': 0.65},
        per_relation_recall={'nsubj': 0.78, 'obj': 0.72, 'root': 0.88, 'det': 0.83, 'amod': 0.71},
        complete_trees=0.65,
        root_accuracy=0.88,
        confusion_matrix=np.array([[10, 2, 0, 1, 0], [1, 12, 1, 0, 1], [0, 0, 15, 0, 0], [1, 0, 0, 13, 1], [0, 1, 0, 2, 9]]),
        relation_labels=['nsubj', 'obj', 'root', 'det', 'amod'],
        avg_tree_depth=3.2,
        avg_dependency_distance=2.1,
        relation_distribution={'nsubj': 13, 'obj': 15, 'root': 15, 'det': 15, 'amod': 12}
    )
    
    print(f"   LAS (Labeled Attachment Score): {dep_metrics.las:.4f}")
    print(f"   UAS (Unlabeled Attachment Score): {dep_metrics.uas:.4f}")
    print(f"   Complete Trees: {dep_metrics.complete_trees:.4f}")
    print(f"   Root Accuracy: {dep_metrics.root_accuracy:.4f}")
    print(f"   Average Tree Depth: {dep_metrics.avg_tree_depth:.2f}")
    print(f"   Average Dependency Distance: {dep_metrics.avg_dependency_distance:.2f}")
    print(f"   Per-relation F1 scores:")
    for relation, f1 in dep_metrics.per_relation_f1.items():
        print(f"     {relation}: {f1:.4f}")


def demo_comprehensive_evaluator():
    """Demonstrate comprehensive NLP evaluator."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Comprehensive NLP Evaluator")
    print("=" * 80)
    
    print("\nNote: This shows a comprehensive evaluation result combining")
    print("all traditional NLP tasks with timing and significance testing.")
    
    # Create comprehensive evaluation result
    pos_metrics = POSEvaluationMetrics(
        accuracy=0.85, macro_f1=0.82, macro_precision=0.83, macro_recall=0.81,
        per_tag_f1={'NOUN': 0.9, 'VERB': 0.8, 'ADJ': 0.75},
        per_tag_precision={'NOUN': 0.92, 'VERB': 0.78, 'ADJ': 0.73},
        per_tag_recall={'NOUN': 0.88, 'VERB': 0.82, 'ADJ': 0.77},
        confusion_matrix=np.array([[10, 2, 1], [1, 15, 2], [2, 1, 12]]),
        tag_labels=['NOUN', 'VERB', 'ADJ'],
        oov_accuracy=0.75,
        tag_distribution={'NOUN': 13, 'VERB': 18, 'ADJ': 15}
    )
    
    ner_metrics = NEREvaluationMetrics(
        entity_f1=0.78, entity_precision=0.80, entity_recall=0.76,
        token_f1=0.85, token_precision=0.87, token_recall=0.83,
        per_type_f1={'PERSON': 0.9, 'LOCATION': 0.7, 'MEDICAL': 0.65},
        per_type_precision={'PERSON': 0.92, 'LOCATION': 0.68, 'MEDICAL': 0.62},
        per_type_recall={'PERSON': 0.88, 'LOCATION': 0.72, 'MEDICAL': 0.68},
        confusion_matrix=np.array([[8, 1, 0], [2, 12, 1], [1, 0, 9]]),
        entity_labels=['PERSON', 'LOCATION', 'MEDICAL'],
        entity_density=2.5, boundary_accuracy=0.82,
        type_distribution={'PERSON': 10, 'LOCATION': 15, 'MEDICAL': 10}
    )
    
    dep_metrics = DependencyEvaluationMetrics(
        las=0.72, uas=0.78,
        per_relation_f1={'nsubj': 0.8, 'obj': 0.7, 'root': 0.9},
        per_relation_precision={'nsubj': 0.82, 'obj': 0.68, 'root': 0.92},
        per_relation_recall={'nsubj': 0.78, 'obj': 0.72, 'root': 0.88},
        complete_trees=0.65, root_accuracy=0.88,
        confusion_matrix=np.array([[6, 2, 0], [1, 9, 1], [0, 1, 8]]),
        relation_labels=['nsubj', 'obj', 'root'],
        avg_tree_depth=3.2, avg_dependency_distance=2.1,
        relation_distribution={'nsubj': 8, 'obj': 11, 'root': 9}
    )
    
    result = ComprehensiveEvaluationResult(
        pos_metrics=pos_metrics,
        ner_metrics=ner_metrics,
        dependency_metrics=dep_metrics,
        feature_extraction_time=0.15,
        total_evaluation_time=2.5,
        significance_tests={
            'pos_vs_baseline': {'p_value': 0.03, 'significant': True},
            'ner_vs_baseline': {'p_value': 0.02, 'significant': True},
            'dependency_vs_baseline': {'p_value': 0.08, 'significant': False}
        }
    )
    
    print("\nComprehensive evaluation result:")
    result.print_summary()
    
    return result


def demo_system_comparison():
    """Demonstrate system comparison functionality."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: NLP System Comparison")
    print("=" * 80)
    
    # Create mock results for different systems
    
    # System A (Baseline)
    pos_metrics_a = POSEvaluationMetrics(
        accuracy=0.82, macro_f1=0.79, macro_precision=0.80, macro_recall=0.78,
        per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
        confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.72,
        tag_distribution={}
    )
    
    ner_metrics_a = NEREvaluationMetrics(
        entity_f1=0.75, entity_precision=0.77, entity_recall=0.73,
        token_f1=0.82, token_precision=0.84, token_recall=0.80,
        per_type_f1={}, per_type_precision={}, per_type_recall={},
        confusion_matrix=np.array([[1]]), entity_labels=[], entity_density=2.3,
        boundary_accuracy=0.79, type_distribution={}
    )
    
    dep_metrics_a = DependencyEvaluationMetrics(
        las=0.69, uas=0.75, per_relation_f1={}, per_relation_precision={},
        per_relation_recall={}, complete_trees=0.62, root_accuracy=0.85,
        confusion_matrix=np.array([[1]]), relation_labels=[], avg_tree_depth=3.0,
        avg_dependency_distance=2.0, relation_distribution={}
    )
    
    result_a = ComprehensiveEvaluationResult(
        pos_metrics=pos_metrics_a, ner_metrics=ner_metrics_a,
        dependency_metrics=dep_metrics_a, feature_extraction_time=0.20,
        total_evaluation_time=3.0, significance_tests={}
    )
    
    # System B (Enhanced)
    pos_metrics_b = POSEvaluationMetrics(
        accuracy=0.85, macro_f1=0.82, macro_precision=0.83, macro_recall=0.81,
        per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
        confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.75,
        tag_distribution={}
    )
    
    ner_metrics_b = NEREvaluationMetrics(
        entity_f1=0.78, entity_precision=0.80, entity_recall=0.76,
        token_f1=0.85, token_precision=0.87, token_recall=0.83,
        per_type_f1={}, per_type_precision={}, per_type_recall={},
        confusion_matrix=np.array([[1]]), entity_labels=[], entity_density=2.5,
        boundary_accuracy=0.82, type_distribution={}
    )
    
    dep_metrics_b = DependencyEvaluationMetrics(
        las=0.72, uas=0.78, per_relation_f1={}, per_relation_precision={},
        per_relation_recall={}, complete_trees=0.65, root_accuracy=0.88,
        confusion_matrix=np.array([[1]]), relation_labels=[], avg_tree_depth=3.2,
        avg_dependency_distance=2.1, relation_distribution={}
    )
    
    result_b = ComprehensiveEvaluationResult(
        pos_metrics=pos_metrics_b, ner_metrics=ner_metrics_b,
        dependency_metrics=dep_metrics_b, feature_extraction_time=0.15,
        total_evaluation_time=2.5, significance_tests={}
    )
    
    # System C (Advanced)
    pos_metrics_c = POSEvaluationMetrics(
        accuracy=0.87, macro_f1=0.84, macro_precision=0.85, macro_recall=0.83,
        per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
        confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.77,
        tag_distribution={}
    )
    
    ner_metrics_c = NEREvaluationMetrics(
        entity_f1=0.81, entity_precision=0.83, entity_recall=0.79,
        token_f1=0.88, token_precision=0.90, token_recall=0.86,
        per_type_f1={}, per_type_precision={}, per_type_recall={},
        confusion_matrix=np.array([[1]]), entity_labels=[], entity_density=2.7,
        boundary_accuracy=0.85, type_distribution={}
    )
    
    dep_metrics_c = DependencyEvaluationMetrics(
        las=0.75, uas=0.81, per_relation_f1={}, per_relation_precision={},
        per_relation_recall={}, complete_trees=0.68, root_accuracy=0.91,
        confusion_matrix=np.array([[1]]), relation_labels=[], avg_tree_depth=3.4,
        avg_dependency_distance=2.3, relation_distribution={}
    )
    
    result_c = ComprehensiveEvaluationResult(
        pos_metrics=pos_metrics_c, ner_metrics=ner_metrics_c,
        dependency_metrics=dep_metrics_c, feature_extraction_time=0.12,
        total_evaluation_time=2.2, significance_tests={}
    )
    
    # Compare systems
    results = {
        'Baseline': result_a,
        'Enhanced': result_b,
        'Advanced': result_c
    }
    
    print("\nSystem Comparison Table:")
    comparison_table = compare_nlp_systems(results)
    print(comparison_table)
    
    # Statistical significance testing
    print("\nStatistical Significance Tests:")
    print("-" * 40)
    
    sig_tests_ab = statistical_significance_test(result_a, result_b)
    sig_tests_bc = statistical_significance_test(result_b, result_c)
    
    print("\nBaseline vs Enhanced:")
    for task, metrics in sig_tests_ab.items():
        print(f"  {task.upper()}:")
        for metric, test in metrics.items():
            sig_marker = "✓" if test['significant'] else "✗"
            print(f"    {metric}: {test['difference']:+.4f} (p={test['p_value']:.3f}) {sig_marker}")
    
    print("\nEnhanced vs Advanced:")
    for task, metrics in sig_tests_bc.items():
        print(f"  {task.upper()}:")
        for metric, test in metrics.items():
            sig_marker = "✓" if test['significant'] else "✗"
            print(f"    {metric}: {test['difference']:+.4f} (p={test['p_value']:.3f}) {sig_marker}")


def main():
    """Main demonstration function."""
    print("🔍 NLP Evaluation Module Demonstration")
    print("=" * 80)
    print("This demonstration shows the comprehensive evaluation capabilities")
    print("for traditional NLP tasks in the depression detection system.")
    print("=" * 80)
    
    # Run demonstrations
    demo_individual_evaluators()
    result = demo_comprehensive_evaluator()
    demo_system_comparison()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Individual task evaluators (POS, NER, Dependency)")
    print("✅ Comprehensive evaluation with all metrics")
    print("✅ System comparison and statistical significance testing")
    print("✅ Per-task metrics (F1, precision, recall, accuracy)")
    print("✅ Confusion matrices and error analysis")
    print("✅ Performance timing and feature extraction metrics")
    print("\nThe NLP evaluation module is ready for comprehensive")
    print("assessment of traditional NLP tasks in the enhanced")
    print("multimodal depression detection system!")
    print("=" * 80)


if __name__ == "__main__":
    main()