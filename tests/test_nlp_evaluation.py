"""
Test Suite for NLP Evaluation Module

This module contains comprehensive tests for the NLP evaluation system,
including unit tests, integration tests, and property-based tests.

Reference: Traditional NLP Tasks - Task 8 Testing
"""

import pytest
import numpy as np
import torch
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
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

from pos_tagger import POSTagger, POSResult, POSConfig
from ner_system import NERSystem, Entity, NERConfig
from dependency_parser import DependencyParser, DependencyTree, DependencyConfig
from linguistic_analyzer import LinguisticAnalyzer, LinguisticConfig


class TestPOSEvaluationMetrics:
    """Test POS evaluation metrics data structure."""
    
    def test_pos_metrics_creation(self):
        """Test POSEvaluationMetrics creation and basic properties."""
        metrics = POSEvaluationMetrics(
            accuracy=0.85,
            macro_f1=0.82,
            macro_precision=0.83,
            macro_recall=0.81,
            per_tag_f1={'NOUN': 0.9, 'VERB': 0.8},
            per_tag_precision={'NOUN': 0.92, 'VERB': 0.78},
            per_tag_recall={'NOUN': 0.88, 'VERB': 0.82},
            confusion_matrix=np.array([[10, 2], [1, 15]]),
            tag_labels=['NOUN', 'VERB'],
            oov_accuracy=0.75,
            tag_distribution={'NOUN': 12, 'VERB': 16}
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.macro_f1 == 0.82
        assert len(metrics.per_tag_f1) == 2
        assert metrics.confusion_matrix.shape == (2, 2)
        assert len(metrics.tag_labels) == 2
    
    def test_pos_metrics_to_dict(self):
        """Test POSEvaluationMetrics serialization."""
        metrics = POSEvaluationMetrics(
            accuracy=0.85,
            macro_f1=0.82,
            macro_precision=0.83,
            macro_recall=0.81,
            per_tag_f1={'NOUN': 0.9},
            per_tag_precision={'NOUN': 0.92},
            per_tag_recall={'NOUN': 0.88},
            confusion_matrix=np.array([[10, 2], [1, 15]]),
            tag_labels=['NOUN', 'VERB'],
            oov_accuracy=0.75,
            tag_distribution={'NOUN': 12}
        )
        
        result_dict = metrics.to_dict()
        
        # Check all required keys are present
        required_keys = [
            'accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
            'per_tag_f1', 'per_tag_precision', 'per_tag_recall',
            'confusion_matrix', 'tag_labels', 'oov_accuracy', 'tag_distribution'
        ]
        
        for key in required_keys:
            assert key in result_dict
        
        # Check confusion matrix is converted to list
        assert isinstance(result_dict['confusion_matrix'], list)
        assert len(result_dict['confusion_matrix']) == 2
    
    @given(
        accuracy=st.floats(min_value=0.0, max_value=1.0),
        f1=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(deadline=None, max_examples=10)
    def test_pos_metrics_property_valid_ranges(self, accuracy, f1):
        """Property: All metric values should be in valid ranges [0, 1]."""
        metrics = POSEvaluationMetrics(
            accuracy=accuracy,
            macro_f1=f1,
            macro_precision=f1,
            macro_recall=f1,
            per_tag_f1={'NOUN': f1},
            per_tag_precision={'NOUN': f1},
            per_tag_recall={'NOUN': f1},
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            tag_labels=['NOUN', 'VERB'],
            oov_accuracy=accuracy,
            tag_distribution={'NOUN': 1}
        )
        
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert 0.0 <= metrics.oov_accuracy <= 1.0


class TestNEREvaluationMetrics:
    """Test NER evaluation metrics data structure."""
    
    def test_ner_metrics_creation(self):
        """Test NEREvaluationMetrics creation and basic properties."""
        metrics = NEREvaluationMetrics(
            entity_f1=0.78,
            entity_precision=0.80,
            entity_recall=0.76,
            token_f1=0.85,
            token_precision=0.87,
            token_recall=0.83,
            per_type_f1={'PERSON': 0.9, 'LOCATION': 0.7},
            per_type_precision={'PERSON': 0.92, 'LOCATION': 0.68},
            per_type_recall={'PERSON': 0.88, 'LOCATION': 0.72},
            confusion_matrix=np.array([[8, 1], [2, 12]]),
            entity_labels=['PERSON', 'LOCATION'],
            entity_density=2.5,
            boundary_accuracy=0.82,
            type_distribution={'PERSON': 10, 'LOCATION': 13}
        )
        
        assert metrics.entity_f1 == 0.78
        assert metrics.token_f1 == 0.85
        assert len(metrics.per_type_f1) == 2
        assert metrics.entity_density == 2.5
    
    def test_ner_metrics_to_dict(self):
        """Test NEREvaluationMetrics serialization."""
        metrics = NEREvaluationMetrics(
            entity_f1=0.78,
            entity_precision=0.80,
            entity_recall=0.76,
            token_f1=0.85,
            token_precision=0.87,
            token_recall=0.83,
            per_type_f1={'PERSON': 0.9},
            per_type_precision={'PERSON': 0.92},
            per_type_recall={'PERSON': 0.88},
            confusion_matrix=np.array([[8, 1]]),
            entity_labels=['PERSON'],
            entity_density=2.5,
            boundary_accuracy=0.82,
            type_distribution={'PERSON': 10}
        )
        
        result_dict = metrics.to_dict()
        
        required_keys = [
            'entity_f1', 'entity_precision', 'entity_recall',
            'token_f1', 'token_precision', 'token_recall',
            'per_type_f1', 'per_type_precision', 'per_type_recall',
            'confusion_matrix', 'entity_labels', 'entity_density',
            'boundary_accuracy', 'type_distribution'
        ]
        
        for key in required_keys:
            assert key in result_dict
    
    @given(
        entity_f1=st.floats(min_value=0.0, max_value=1.0),
        boundary_acc=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(deadline=None, max_examples=10)
    def test_ner_metrics_property_valid_ranges(self, entity_f1, boundary_acc):
        """Property: All NER metric values should be in valid ranges."""
        metrics = NEREvaluationMetrics(
            entity_f1=entity_f1,
            entity_precision=entity_f1,
            entity_recall=entity_f1,
            token_f1=entity_f1,
            token_precision=entity_f1,
            token_recall=entity_f1,
            per_type_f1={'PERSON': entity_f1},
            per_type_precision={'PERSON': entity_f1},
            per_type_recall={'PERSON': entity_f1},
            confusion_matrix=np.array([[1]]),
            entity_labels=['PERSON'],
            entity_density=1.0,
            boundary_accuracy=boundary_acc,
            type_distribution={'PERSON': 1}
        )
        
        assert 0.0 <= metrics.entity_f1 <= 1.0
        assert 0.0 <= metrics.boundary_accuracy <= 1.0


class TestDependencyEvaluationMetrics:
    """Test dependency parsing evaluation metrics."""
    
    def test_dependency_metrics_creation(self):
        """Test DependencyEvaluationMetrics creation."""
        metrics = DependencyEvaluationMetrics(
            las=0.72,
            uas=0.78,
            per_relation_f1={'nsubj': 0.8, 'obj': 0.7},
            per_relation_precision={'nsubj': 0.82, 'obj': 0.68},
            per_relation_recall={'nsubj': 0.78, 'obj': 0.72},
            complete_trees=0.65,
            root_accuracy=0.88,
            confusion_matrix=np.array([[6, 2], [1, 9]]),
            relation_labels=['nsubj', 'obj'],
            avg_tree_depth=3.2,
            avg_dependency_distance=2.1,
            relation_distribution={'nsubj': 8, 'obj': 10}
        )
        
        assert metrics.las == 0.72
        assert metrics.uas == 0.78
        assert metrics.complete_trees == 0.65
        assert metrics.avg_tree_depth == 3.2
    
    @given(
        las=st.floats(min_value=0.0, max_value=1.0),
        uas=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(deadline=None, max_examples=10)
    def test_dependency_metrics_property_las_uas_relationship(self, las, uas):
        """Property: LAS should be less than or equal to UAS."""
        # Ensure LAS <= UAS
        if las > uas:
            las, uas = uas, las
        
        metrics = DependencyEvaluationMetrics(
            las=las,
            uas=uas,
            per_relation_f1={'nsubj': 0.8},
            per_relation_precision={'nsubj': 0.8},
            per_relation_recall={'nsubj': 0.8},
            complete_trees=0.5,
            root_accuracy=0.8,
            confusion_matrix=np.array([[1]]),
            relation_labels=['nsubj'],
            avg_tree_depth=2.0,
            avg_dependency_distance=1.5,
            relation_distribution={'nsubj': 1}
        )
        
        assert metrics.las <= metrics.uas


class TestComprehensiveEvaluationResult:
    """Test comprehensive evaluation result structure."""
    
    def test_comprehensive_result_creation(self):
        """Test ComprehensiveEvaluationResult creation."""
        pos_metrics = POSEvaluationMetrics(
            accuracy=0.85, macro_f1=0.82, macro_precision=0.83, macro_recall=0.81,
            per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
            confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.75,
            tag_distribution={}
        )
        
        ner_metrics = NEREvaluationMetrics(
            entity_f1=0.78, entity_precision=0.80, entity_recall=0.76,
            token_f1=0.85, token_precision=0.87, token_recall=0.83,
            per_type_f1={}, per_type_precision={}, per_type_recall={},
            confusion_matrix=np.array([[1]]), entity_labels=[], entity_density=2.5,
            boundary_accuracy=0.82, type_distribution={}
        )
        
        dep_metrics = DependencyEvaluationMetrics(
            las=0.72, uas=0.78, per_relation_f1={}, per_relation_precision={},
            per_relation_recall={}, complete_trees=0.65, root_accuracy=0.88,
            confusion_matrix=np.array([[1]]), relation_labels=[], avg_tree_depth=3.2,
            avg_dependency_distance=2.1, relation_distribution={}
        )
        
        result = ComprehensiveEvaluationResult(
            pos_metrics=pos_metrics,
            ner_metrics=ner_metrics,
            dependency_metrics=dep_metrics,
            feature_extraction_time=0.15,
            total_evaluation_time=2.5,
            significance_tests={}
        )
        
        assert result.pos_metrics.accuracy == 0.85
        assert result.ner_metrics.entity_f1 == 0.78
        assert result.dependency_metrics.las == 0.72
        assert result.feature_extraction_time == 0.15
    
    def test_comprehensive_result_to_dict(self):
        """Test ComprehensiveEvaluationResult serialization."""
        pos_metrics = POSEvaluationMetrics(
            accuracy=0.85, macro_f1=0.82, macro_precision=0.83, macro_recall=0.81,
            per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
            confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.75,
            tag_distribution={}
        )
        
        ner_metrics = NEREvaluationMetrics(
            entity_f1=0.78, entity_precision=0.80, entity_recall=0.76,
            token_f1=0.85, token_precision=0.87, token_recall=0.83,
            per_type_f1={}, per_type_precision={}, per_type_recall={},
            confusion_matrix=np.array([[1]]), entity_labels=[], entity_density=2.5,
            boundary_accuracy=0.82, type_distribution={}
        )
        
        dep_metrics = DependencyEvaluationMetrics(
            las=0.72, uas=0.78, per_relation_f1={}, per_relation_precision={},
            per_relation_recall={}, complete_trees=0.65, root_accuracy=0.88,
            confusion_matrix=np.array([[1]]), relation_labels=[], avg_tree_depth=3.2,
            avg_dependency_distance=2.1, relation_distribution={}
        )
        
        result = ComprehensiveEvaluationResult(
            pos_metrics=pos_metrics,
            ner_metrics=ner_metrics,
            dependency_metrics=dep_metrics,
            feature_extraction_time=0.15,
            total_evaluation_time=2.5,
            significance_tests={}
        )
        
        result_dict = result.to_dict()
        
        required_keys = [
            'pos_metrics', 'ner_metrics', 'dependency_metrics',
            'feature_extraction_time', 'total_evaluation_time', 'significance_tests'
        ]
        
        for key in required_keys:
            assert key in result_dict
    
    def test_comprehensive_result_print_summary(self, capsys):
        """Test ComprehensiveEvaluationResult print_summary method."""
        pos_metrics = POSEvaluationMetrics(
            accuracy=0.85, macro_f1=0.82, macro_precision=0.83, macro_recall=0.81,
            per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
            confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.75,
            tag_distribution={}
        )
        
        ner_metrics = NEREvaluationMetrics(
            entity_f1=0.78, entity_precision=0.80, entity_recall=0.76,
            token_f1=0.85, token_precision=0.87, token_recall=0.83,
            per_type_f1={}, per_type_precision={}, per_type_recall={},
            confusion_matrix=np.array([[1]]), entity_labels=[], entity_density=2.5,
            boundary_accuracy=0.82, type_distribution={}
        )
        
        dep_metrics = DependencyEvaluationMetrics(
            las=0.72, uas=0.78, per_relation_f1={}, per_relation_precision={},
            per_relation_recall={}, complete_trees=0.65, root_accuracy=0.88,
            confusion_matrix=np.array([[1]]), relation_labels=[], avg_tree_depth=3.2,
            avg_dependency_distance=2.1, relation_distribution={}
        )
        
        result = ComprehensiveEvaluationResult(
            pos_metrics=pos_metrics,
            ner_metrics=ner_metrics,
            dependency_metrics=dep_metrics,
            feature_extraction_time=0.15,
            total_evaluation_time=2.5,
            significance_tests={}
        )
        
        result.print_summary()
        
        captured = capsys.readouterr()
        assert "Comprehensive NLP Evaluation Results" in captured.out
        assert "Part-of-Speech Tagging" in captured.out
        assert "Named Entity Recognition" in captured.out
        assert "Dependency Parsing" in captured.out


class TestPOSEvaluator:
    """Test POS tagger evaluator."""
    
    def test_pos_evaluator_creation(self):
        """Test POSEvaluator creation."""
        mock_pos_tagger = Mock(spec=POSTagger)
        evaluator = POSEvaluator(mock_pos_tagger)
        
        assert evaluator.pos_tagger == mock_pos_tagger
    
    def test_pos_evaluator_evaluate(self):
        """Test POSEvaluator.evaluate method."""
        # Create mock POS tagger
        mock_pos_tagger = Mock(spec=POSTagger)
        
        # Mock POS result
        mock_result = Mock(spec=POSResult)
        mock_result.tags = ['NOUN', 'VERB', 'ADJ']
        mock_pos_tagger.tag_sentence.return_value = mock_result
        
        evaluator = POSEvaluator(mock_pos_tagger)
        
        # Test data
        test_data = [
            (['word1', 'word2', 'word3'], ['NOUN', 'VERB', 'ADJ']),
            (['word4', 'word5'], ['NOUN', 'VERB'])
        ]
        
        # Mock second call
        mock_result2 = Mock(spec=POSResult)
        mock_result2.tags = ['NOUN', 'VERB']
        mock_pos_tagger.tag_sentence.side_effect = [mock_result, mock_result2]
        
        result = evaluator.evaluate(test_data, language="ta")
        
        assert isinstance(result, POSEvaluationMetrics)
        assert result.accuracy >= 0.0
        assert result.macro_f1 >= 0.0
        assert mock_pos_tagger.tag_sentence.call_count == 2
    
    @given(
        num_sentences=st.integers(min_value=1, max_value=5),
        sentence_length=st.integers(min_value=1, max_value=10)
    )
    @settings(deadline=None, max_examples=10)
    def test_pos_evaluator_property_consistent_evaluation(self, num_sentences, sentence_length):
        """Property: Evaluation should handle variable sentence lengths consistently."""
        mock_pos_tagger = Mock(spec=POSTagger)
        evaluator = POSEvaluator(mock_pos_tagger)
        
        # Generate test data
        test_data = []
        mock_results = []
        
        for i in range(num_sentences):
            tokens = [f'word{j}' for j in range(sentence_length)]
            tags = ['NOUN'] * sentence_length
            test_data.append((tokens, tags))
            
            # Mock result
            mock_result = Mock(spec=POSResult)
            mock_result.tags = tags
            mock_results.append(mock_result)
        
        mock_pos_tagger.tag_sentence.side_effect = mock_results
        
        result = evaluator.evaluate(test_data, language="ta")
        
        # Should complete without errors
        assert isinstance(result, POSEvaluationMetrics)
        assert mock_pos_tagger.tag_sentence.call_count == num_sentences


class TestNEREvaluator:
    """Test NER system evaluator."""
    
    def test_ner_evaluator_creation(self):
        """Test NEREvaluator creation."""
        mock_ner_system = Mock(spec=NERSystem)
        evaluator = NEREvaluator(mock_ner_system)
        
        assert evaluator.ner_system == mock_ner_system
    
    def test_ner_evaluator_evaluate(self):
        """Test NEREvaluator.evaluate method."""
        mock_ner_system = Mock(spec=NERSystem)
        
        # Mock entities
        mock_entity = Entity(
            text="John",
            label="PERSON",
            start_idx=0,
            end_idx=1,
            confidence=0.9,
            anonymized_form="[PERSON_1]"
        )
        
        mock_ner_system.extract_entities.return_value = [mock_entity]
        
        evaluator = NEREvaluator(mock_ner_system)
        
        # Test data
        gold_entity = Entity(
            text="John",
            label="PERSON", 
            start_idx=0,
            end_idx=1,
            confidence=1.0,
            anonymized_form="[PERSON_1]"
        )
        
        test_data = [
            (['John', 'walks'], [gold_entity])
        ]
        
        result = evaluator.evaluate(test_data, language="ta")
        
        assert isinstance(result, NEREvaluationMetrics)
        assert result.entity_f1 >= 0.0
        assert result.token_f1 >= 0.0
    
    def test_ner_evaluator_entities_to_bio(self):
        """Test _entities_to_bio conversion method."""
        mock_ner_system = Mock(spec=NERSystem)
        evaluator = NEREvaluator(mock_ner_system)
        
        tokens = ['John', 'Smith', 'walks', 'to', 'New', 'York']
        entities = [
            Entity("John Smith", "PERSON", 0, 2, 0.9, "[PERSON_1]"),
            Entity("New York", "LOCATION", 4, 6, 0.8, "[LOCATION_1]")
        ]
        
        bio_tags = evaluator._entities_to_bio(tokens, entities)
        
        expected = ['B-PERSON', 'I-PERSON', 'O', 'O', 'B-LOCATION', 'I-LOCATION']
        assert bio_tags == expected
    
    @given(
        num_entities=st.integers(min_value=0, max_value=5),
        sentence_length=st.integers(min_value=1, max_value=10)
    )
    @settings(deadline=None, max_examples=10)
    def test_ner_evaluator_property_entity_boundaries(self, num_entities, sentence_length):
        """Property: Entity boundaries should be within sentence bounds."""
        mock_ner_system = Mock(spec=NERSystem)
        evaluator = NEREvaluator(mock_ner_system)
        
        tokens = [f'word{i}' for i in range(sentence_length)]
        
        # Generate valid entities
        entities = []
        for i in range(min(num_entities, sentence_length)):
            entity = Entity(
                text=f'word{i}',
                label="PERSON",
                start_idx=i,
                end_idx=i + 1,
                confidence=0.9,
                anonymized_form=f"[PERSON_{i}]"
            )
            entities.append(entity)
        
        bio_tags = evaluator._entities_to_bio(tokens, entities)
        
        # All BIO tags should be valid
        assert len(bio_tags) == len(tokens)
        for tag in bio_tags:
            assert tag.startswith(('B-', 'I-', 'O'))


class TestDependencyEvaluator:
    """Test dependency parser evaluator."""
    
    def test_dependency_evaluator_creation(self):
        """Test DependencyEvaluator creation."""
        mock_parser = Mock(spec=DependencyParser)
        evaluator = DependencyEvaluator(mock_parser)
        
        assert evaluator.dependency_parser == mock_parser
    
    def test_dependency_evaluator_evaluate(self):
        """Test DependencyEvaluator.evaluate method."""
        mock_parser = Mock(spec=DependencyParser)
        
        # Mock dependency tree
        mock_tree = DependencyTree(
            tokens=['John', 'walks'],
            heads=[-1, 0],  # John is root, walks depends on John
            relations=['root', 'nsubj'],
            confidence_scores=[0.9, 0.8]
        )
        
        # Add get_depth method to mock
        mock_tree.get_depth = Mock(return_value=2)
        
        mock_parser.parse_sentence.return_value = mock_tree
        
        evaluator = DependencyEvaluator(mock_parser)
        
        # Test data - same structure as prediction
        gold_tree = DependencyTree(
            tokens=['John', 'walks'],
            heads=[-1, 0],
            relations=['root', 'nsubj'],
            confidence_scores=[1.0, 1.0]
        )
        gold_tree.get_depth = Mock(return_value=2)
        
        test_data = [
            (['John', 'walks'], gold_tree)
        ]
        
        result = evaluator.evaluate(test_data, language="ta")
        
        assert isinstance(result, DependencyEvaluationMetrics)
        assert result.las >= 0.0
        assert result.uas >= 0.0
        assert result.las <= result.uas  # LAS should be <= UAS
    
    @given(
        sentence_length=st.integers(min_value=2, max_value=8)
    )
    @settings(deadline=None, max_examples=10)
    def test_dependency_evaluator_property_tree_structure(self, sentence_length):
        """Property: Dependency trees should maintain structural validity."""
        mock_parser = Mock(spec=DependencyParser)
        evaluator = DependencyEvaluator(mock_parser)
        
        tokens = [f'word{i}' for i in range(sentence_length)]
        
        # Create valid dependency tree (each token depends on previous, except root)
        heads = [-1] + list(range(sentence_length - 1))
        relations = ['root'] + ['dep'] * (sentence_length - 1)
        
        mock_tree = DependencyTree(
            tokens=tokens,
            heads=heads,
            relations=relations,
            confidence_scores=[0.9] * sentence_length
        )
        mock_tree.get_depth = Mock(return_value=sentence_length)
        
        gold_tree = DependencyTree(
            tokens=tokens,
            heads=heads,
            relations=relations,
            confidence_scores=[1.0] * sentence_length
        )
        gold_tree.get_depth = Mock(return_value=sentence_length)
        
        mock_parser.parse_sentence.return_value = mock_tree
        
        test_data = [(tokens, gold_tree)]
        
        result = evaluator.evaluate(test_data, language="ta")
        
        # Should complete without errors and have valid metrics
        assert isinstance(result, DependencyEvaluationMetrics)
        assert 0.0 <= result.las <= 1.0
        assert 0.0 <= result.uas <= 1.0


class TestComprehensiveNLPEvaluator:
    """Test comprehensive NLP evaluator."""
    
    def test_comprehensive_evaluator_creation(self):
        """Test ComprehensiveNLPEvaluator creation."""
        mock_analyzer = Mock(spec=LinguisticAnalyzer)
        mock_analyzer.pos_tagger = Mock(spec=POSTagger)
        mock_analyzer.ner_system = Mock(spec=NERSystem)
        mock_analyzer.dependency_parser = Mock(spec=DependencyParser)
        
        evaluator = ComprehensiveNLPEvaluator(mock_analyzer)
        
        assert evaluator.linguistic_analyzer == mock_analyzer
        assert isinstance(evaluator.pos_evaluator, POSEvaluator)
        assert isinstance(evaluator.ner_evaluator, NEREvaluator)
        assert isinstance(evaluator.dependency_evaluator, DependencyEvaluator)
    
    @patch('nlp_evaluation.POSEvaluator')
    @patch('nlp_evaluation.NEREvaluator')
    @patch('nlp_evaluation.DependencyEvaluator')
    def test_comprehensive_evaluator_evaluate_comprehensive(
        self, mock_dep_eval_class, mock_ner_eval_class, mock_pos_eval_class
    ):
        """Test ComprehensiveNLPEvaluator.evaluate_comprehensive method."""
        # Setup mocks
        mock_analyzer = Mock(spec=LinguisticAnalyzer)
        mock_analyzer.pos_tagger = Mock(spec=POSTagger)
        mock_analyzer.ner_system = Mock(spec=NERSystem)
        mock_analyzer.dependency_parser = Mock(spec=DependencyParser)
        
        # Mock evaluator instances
        mock_pos_eval = Mock()
        mock_ner_eval = Mock()
        mock_dep_eval = Mock()
        
        mock_pos_eval_class.return_value = mock_pos_eval
        mock_ner_eval_class.return_value = mock_ner_eval
        mock_dep_eval_class.return_value = mock_dep_eval
        
        # Mock evaluation results
        mock_pos_metrics = Mock(spec=POSEvaluationMetrics)
        mock_ner_metrics = Mock(spec=NEREvaluationMetrics)
        mock_dep_metrics = Mock(spec=DependencyEvaluationMetrics)
        
        mock_pos_eval.evaluate.return_value = mock_pos_metrics
        mock_ner_eval.evaluate.return_value = mock_ner_metrics
        mock_dep_eval.evaluate.return_value = mock_dep_metrics
        
        # Mock linguistic analysis
        mock_analysis = Mock()
        mock_analyzer.analyze_text.return_value = mock_analysis
        
        evaluator = ComprehensiveNLPEvaluator(mock_analyzer)
        
        test_data = {
            'pos': [(['word1'], ['NOUN'])],
            'ner': [(['word1'], [])],
            'dependency': [(['word1'], Mock())]
        }
        
        result = evaluator.evaluate_comprehensive(test_data, language="ta")
        
        assert isinstance(result, ComprehensiveEvaluationResult)
        assert result.pos_metrics == mock_pos_metrics
        assert result.ner_metrics == mock_ner_metrics
        assert result.dependency_metrics == mock_dep_metrics
        assert result.feature_extraction_time >= 0.0
        assert result.total_evaluation_time >= 0.0


class TestComparisonFunctions:
    """Test comparison and statistical functions."""
    
    def test_compare_nlp_systems(self):
        """Test compare_nlp_systems function."""
        # Create mock results
        pos_metrics = POSEvaluationMetrics(
            accuracy=0.85, macro_f1=0.82, macro_precision=0.83, macro_recall=0.81,
            per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
            confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.75,
            tag_distribution={}
        )
        
        ner_metrics = NEREvaluationMetrics(
            entity_f1=0.78, entity_precision=0.80, entity_recall=0.76,
            token_f1=0.85, token_precision=0.87, token_recall=0.83,
            per_type_f1={}, per_type_precision={}, per_type_recall={},
            confusion_matrix=np.array([[1]]), entity_labels=[], entity_density=2.5,
            boundary_accuracy=0.82, type_distribution={}
        )
        
        dep_metrics = DependencyEvaluationMetrics(
            las=0.72, uas=0.78, per_relation_f1={}, per_relation_precision={},
            per_relation_recall={}, complete_trees=0.65, root_accuracy=0.88,
            confusion_matrix=np.array([[1]]), relation_labels=[], avg_tree_depth=3.2,
            avg_dependency_distance=2.1, relation_distribution={}
        )
        
        result1 = ComprehensiveEvaluationResult(
            pos_metrics=pos_metrics,
            ner_metrics=ner_metrics,
            dependency_metrics=dep_metrics,
            feature_extraction_time=0.15,
            total_evaluation_time=2.5,
            significance_tests={}
        )
        
        result2 = ComprehensiveEvaluationResult(
            pos_metrics=pos_metrics,
            ner_metrics=ner_metrics,
            dependency_metrics=dep_metrics,
            feature_extraction_time=0.20,
            total_evaluation_time=3.0,
            significance_tests={}
        )
        
        results = {
            'System A': result1,
            'System B': result2
        }
        
        table = compare_nlp_systems(results)
        
        assert isinstance(table, str)
        assert 'System A' in table
        assert 'System B' in table
        assert 'POS F1' in table
        assert 'NER F1' in table
        assert 'LAS' in table
    
    def test_statistical_significance_test(self):
        """Test statistical_significance_test function."""
        # Create two different results
        pos_metrics1 = POSEvaluationMetrics(
            accuracy=0.85, macro_f1=0.82, macro_precision=0.83, macro_recall=0.81,
            per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
            confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.75,
            tag_distribution={}
        )
        
        pos_metrics2 = POSEvaluationMetrics(
            accuracy=0.87, macro_f1=0.84, macro_precision=0.85, macro_recall=0.83,
            per_tag_f1={}, per_tag_precision={}, per_tag_recall={},
            confusion_matrix=np.array([[1]]), tag_labels=[], oov_accuracy=0.77,
            tag_distribution={}
        )
        
        ner_metrics = NEREvaluationMetrics(
            entity_f1=0.78, entity_precision=0.80, entity_recall=0.76,
            token_f1=0.85, token_precision=0.87, token_recall=0.83,
            per_type_f1={}, per_type_precision={}, per_type_recall={},
            confusion_matrix=np.array([[1]]), entity_labels=[], entity_density=2.5,
            boundary_accuracy=0.82, type_distribution={}
        )
        
        dep_metrics = DependencyEvaluationMetrics(
            las=0.72, uas=0.78, per_relation_f1={}, per_relation_precision={},
            per_relation_recall={}, complete_trees=0.65, root_accuracy=0.88,
            confusion_matrix=np.array([[1]]), relation_labels=[], avg_tree_depth=3.2,
            avg_dependency_distance=2.1, relation_distribution={}
        )
        
        result1 = ComprehensiveEvaluationResult(
            pos_metrics=pos_metrics1,
            ner_metrics=ner_metrics,
            dependency_metrics=dep_metrics,
            feature_extraction_time=0.15,
            total_evaluation_time=2.5,
            significance_tests={}
        )
        
        result2 = ComprehensiveEvaluationResult(
            pos_metrics=pos_metrics2,
            ner_metrics=ner_metrics,
            dependency_metrics=dep_metrics,
            feature_extraction_time=0.20,
            total_evaluation_time=3.0,
            significance_tests={}
        )
        
        sig_tests = statistical_significance_test(result1, result2)
        
        assert isinstance(sig_tests, dict)
        assert 'pos' in sig_tests
        assert 'ner' in sig_tests
        assert 'dependency' in sig_tests
        
        # Check POS tests
        assert 'accuracy' in sig_tests['pos']
        assert 'macro_f1' in sig_tests['pos']
        
        # Each test should have difference, p_value, and significant
        for task_tests in sig_tests.values():
            for metric_test in task_tests.values():
                assert 'difference' in metric_test
                assert 'p_value' in metric_test
                assert 'significant' in metric_test


class TestIntegration:
    """Integration tests for the evaluation system."""
    
    @given(
        num_test_samples=st.integers(min_value=1, max_value=3)
    )
    @settings(deadline=None, max_examples=10)
    def test_evaluation_pipeline_property_completeness(self, num_test_samples):
        """Property: Evaluation pipeline should handle any number of test samples."""
        # This is a simplified integration test
        # In practice, would use real components
        
        mock_analyzer = Mock(spec=LinguisticAnalyzer)
        mock_analyzer.pos_tagger = Mock(spec=POSTagger)
        mock_analyzer.ner_system = Mock(spec=NERSystem)
        mock_analyzer.dependency_parser = Mock(spec=DependencyParser)
        
        evaluator = ComprehensiveNLPEvaluator(mock_analyzer)
        
        # Generate test data
        test_data = {
            'pos': [(['word'], ['NOUN'])] * num_test_samples,
            'ner': [(['word'], [])] * num_test_samples,
            'dependency': [(['word'], Mock())] * num_test_samples
        }
        
        # Mock the individual evaluators to return valid results
        with patch.object(evaluator.pos_evaluator, 'evaluate') as mock_pos_eval, \
             patch.object(evaluator.ner_evaluator, 'evaluate') as mock_ner_eval, \
             patch.object(evaluator.dependency_evaluator, 'evaluate') as mock_dep_eval:
            
            # Setup mock returns
            mock_pos_eval.return_value = Mock(spec=POSEvaluationMetrics)
            mock_ner_eval.return_value = Mock(spec=NEREvaluationMetrics)
            mock_dep_eval.return_value = Mock(spec=DependencyEvaluationMetrics)
            
            mock_analyzer.analyze_text.return_value = Mock()
            
            result = evaluator.evaluate_comprehensive(test_data)
            
            # Should complete successfully regardless of sample count
            assert isinstance(result, ComprehensiveEvaluationResult)
            assert result.total_evaluation_time >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])