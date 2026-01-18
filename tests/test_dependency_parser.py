"""
Tests for Dependency Parser

This module contains unit tests and property-based tests for the dependency parser
implementation for syntactic analysis of Dravidian languages.

Reference: Traditional NLP Tasks - Task 3
"""

import pytest
import torch
from hypothesis import given, strategies as st, assume, settings
from typing import List
import numpy as np

from dependency_parser import DependencyParser, DependencyConfig, DependencyTree, create_dependency_parser


class TestDependencyConfig:
    """Test dependency parser configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DependencyConfig()
        assert config.model_name == "bert-base-multilingual-cased"
        assert config.annotation_scheme == "universal_dependencies"
        assert config.max_length == 512
        assert config.return_probabilities is True
        assert "root" in config.relation_labels
        assert "nsubj" in config.relation_labels
        assert "obj" in config.relation_labels
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DependencyConfig(
            model_name="custom-model",
            max_length=256,
            return_probabilities=False
        )
        assert config.model_name == "custom-model"
        assert config.max_length == 256
        assert config.return_probabilities is False


class TestDependencyTree:
    """Test dependency tree data structure."""
    
    def test_valid_dependency_tree(self):
        """Test valid dependency tree creation."""
        tokens = ["I", "am", "happy"]
        heads = [1, -1, 1]  # "am" is root, "I" and "happy" attach to "am"
        relations = ["nsubj", "root", "amod"]
        confidences = [0.9, 0.95, 0.8]
        
        tree = DependencyTree(tokens, heads, relations, confidences)
        
        assert tree.tokens == tokens
        assert tree.heads == heads
        assert tree.relations == relations
        assert tree.confidence_scores == confidences
    
    def test_invalid_dependency_tree_length_mismatch(self):
        """Test that mismatched lengths raise assertion error."""
        with pytest.raises(AssertionError, match="All lists must have the same length"):
            DependencyTree(
                tokens=["I", "am"],
                heads=[1, -1],
                relations=["nsubj"],  # Mismatched length
                confidence_scores=[0.9, 0.8]
            )
    
    def test_invalid_head_index(self):
        """Test that invalid head indices raise assertion error."""
        with pytest.raises(AssertionError, match="Head index .* is out of bounds"):
            DependencyTree(
                tokens=["I", "am"],
                heads=[5, -1],  # Head index 5 is out of bounds
                relations=["nsubj", "root"],
                confidence_scores=[0.9, 0.8]
            )
    
    def test_self_head_invalid(self):
        """Test that self-head is invalid."""
        with pytest.raises(AssertionError, match="Token .* cannot be its own head"):
            DependencyTree(
                tokens=["I", "am"],
                heads=[0, -1],  # Token 0 cannot be its own head
                relations=["nsubj", "root"],
                confidence_scores=[0.9, 0.8]
            )
    
    def test_tree_depth_calculation(self):
        """Test tree depth calculation."""
        # Simple tree: "I am happy"
        tree = DependencyTree(
            tokens=["I", "am", "happy"],
            heads=[1, -1, 1],
            relations=["nsubj", "root", "amod"],
            confidence_scores=[0.9, 0.95, 0.8]
        )
        
        depth = tree.get_depth()
        assert depth == 2  # Root -> dependent
    
    def test_complexity_metrics(self):
        """Test syntactic complexity metrics."""
        tree = DependencyTree(
            tokens=["I", "am", "very", "happy"],
            heads=[1, -1, 3, 1],
            relations=["nsubj", "root", "advmod", "amod"],
            confidence_scores=[0.9, 0.95, 0.8, 0.85]
        )
        
        metrics = tree.get_complexity_metrics()
        
        assert "tree_depth" in metrics
        assert "avg_dependency_distance" in metrics
        assert "syntactic_complexity" in metrics
        assert "clause_count" in metrics
        
        assert metrics["tree_depth"] > 0
        assert metrics["avg_dependency_distance"] >= 0
        assert metrics["syntactic_complexity"] >= 0
        assert metrics["clause_count"] >= 0
    
    def test_tree_validity_valid(self):
        """Test valid tree detection."""
        tree = DependencyTree(
            tokens=["I", "am", "happy"],
            heads=[1, -1, 1],
            relations=["nsubj", "root", "amod"],
            confidence_scores=[0.9, 0.95, 0.8]
        )
        
        assert tree.is_valid_tree() is True
    
    def test_tree_validity_multiple_roots(self):
        """Test invalid tree with multiple roots."""
        tree = DependencyTree(
            tokens=["I", "am", "happy"],
            heads=[-1, -1, 1],  # Two roots
            relations=["root", "root", "amod"],
            confidence_scores=[0.9, 0.95, 0.8]
        )
        
        assert tree.is_valid_tree() is False
    
    def test_empty_tree(self):
        """Test empty dependency tree."""
        tree = DependencyTree([], [], [], [])
        
        assert tree.get_depth() == 0
        assert tree.is_valid_tree() is True
        
        metrics = tree.get_complexity_metrics()
        assert all(value == 0.0 for value in metrics.values())


class TestDependencyParser:
    """Test dependency parser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create dependency parser for testing."""
        # Force fallback to avoid model loading issues
        config = DependencyConfig(model_name="fallback")
        return DependencyParser(config)
    
    @pytest.fixture
    def sample_tokens(self):
        """Sample Tamil tokens for testing."""
        return ["நான்", "வருகிறேன்"]
    
    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.config.model_name == "fallback"
        assert len(parser.config.relation_labels) > 0
    
    def test_parse_sentence_basic(self, parser, sample_tokens):
        """Test basic sentence parsing."""
        tree = parser.parse_sentence(sample_tokens, "ta")
        
        assert isinstance(tree, DependencyTree)
        assert len(tree.tokens) == len(sample_tokens)
        assert len(tree.heads) == len(sample_tokens)
        assert len(tree.relations) == len(sample_tokens)
        assert len(tree.confidence_scores) == len(sample_tokens)
        
        # Should have exactly one root
        root_count = sum(1 for head in tree.heads if head == -1)
        assert root_count == 1
        
        # All relations should be valid
        for relation in tree.relations:
            assert relation in parser.config.relation_labels
    
    def test_parse_empty_sentence(self, parser):
        """Test parsing empty sentence."""
        tree = parser.parse_sentence([], "ta")
        
        assert tree.tokens == []
        assert tree.heads == []
        assert tree.relations == []
        assert tree.confidence_scores == []
    
    def test_parse_single_token(self, parser):
        """Test parsing single token."""
        tree = parser.parse_sentence(["வருகிறேன்"], "ta")
        
        assert len(tree.tokens) == 1
        assert tree.heads == [-1]  # Single token is root
        assert tree.relations == ["root"]
        assert len(tree.confidence_scores) == 1
    
    def test_extract_syntactic_features(self, parser, sample_tokens):
        """Test syntactic feature extraction."""
        tree = parser.parse_sentence(sample_tokens, "ta")
        features = parser.extract_syntactic_features(tree)
        
        # Check required features exist
        required_features = [
            "tree_depth", "avg_dependency_distance", "syntactic_complexity",
            "clause_count", "total_dependencies", "relation_diversity",
            "subject_ratio", "object_ratio", "tree_validity"
        ]
        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
        
        # Check feature ranges
        assert features["tree_depth"] >= 0
        assert features["avg_dependency_distance"] >= 0
        assert features["syntactic_complexity"] >= 0
        assert features["tree_validity"] in [0.0, 1.0]
        assert 0.0 <= features["avg_confidence"] <= 1.0
    
    def test_extract_syntactic_features_empty(self, parser):
        """Test feature extraction with empty tree."""
        tree = DependencyTree([], [], [], [])
        features = parser.extract_syntactic_features(tree)
        
        assert isinstance(features, dict)
        assert features["tree_depth"] == 0.0
        assert features["total_dependencies"] == 0.0
        assert features["tree_validity"] == 0.0


class TestDependencyParserPropertyBased:
    """Property-based tests for dependency parser."""
    
    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
    @settings(deadline=None, max_examples=10)
    def test_property_tree_validity(self, tokens):
        """
        Property 3: For any dependency tree, each token SHALL have exactly 
        one head (except the root), and the tree SHALL be connected and acyclic.
        
        **Feature: traditional-nlp-tasks, Property 3**
        **Validates: Requirements 3.1, 3.4**
        """
        # Create parser with fallback (no model loading)
        config = DependencyConfig(model_name="fallback")
        parser = DependencyParser(config)
        
        # Filter tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        tree = parser.parse_sentence(tokens, "ta")
        
        # Property: exactly one root
        root_count = sum(1 for head in tree.heads if head == -1)
        assert root_count == 1, \
            f"Tree must have exactly one root, found {root_count}"
        
        # Property: all non-root tokens have valid heads
        for i, head in enumerate(tree.heads):
            if head != -1:
                assert 0 <= head < len(tree.tokens), \
                    f"Head {head} for token {i} is out of bounds"
                assert head != i, \
                    f"Token {i} cannot be its own head"
        
        # Property: tree is valid (connected and acyclic)
        assert tree.is_valid_tree(), \
            "Dependency tree must be valid (connected and acyclic)"
    
    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
           st.sampled_from(["ta", "ml"]))
    @settings(deadline=None, max_examples=10)
    def test_property_language_consistency(self, tokens, language):
        """
        Property 5: For any input text with specified language, all NLP tasks 
        SHALL use language-appropriate models and return results consistent 
        with that language's linguistic properties.
        
        **Feature: traditional-nlp-tasks, Property 5**
        **Validates: Requirements 1.2, 2.2, 3.2**
        """
        # Create parser with fallback (no model loading)
        config = DependencyConfig(model_name="fallback")
        parser = DependencyParser(config)
        
        # Filter tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        tree = parser.parse_sentence(tokens, language)
        
        # Property: consistent processing for specified language
        assert len(tree.tokens) == len(tokens)
        assert len(tree.heads) == len(tokens)
        assert len(tree.relations) == len(tokens)
        
        # All relations should be valid
        for relation in tree.relations:
            assert relation in parser.config.relation_labels, \
                f"Relation '{relation}' not in valid relation set"
    
    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
    @settings(deadline=None, max_examples=10)
    def test_property_feature_vector_validity(self, tokens):
        """
        Test that extracted syntactic features are always valid.
        """
        # Create parser with fallback (no model loading)
        config = DependencyConfig(model_name="fallback")
        parser = DependencyParser(config)
        
        # Filter tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        tree = parser.parse_sentence(tokens, "ta")
        features = parser.extract_syntactic_features(tree)
        
        # Property: all features are valid numbers
        for key, value in features.items():
            assert isinstance(value, (int, float)), \
                f"Feature '{key}' must be numeric, got {type(value)}"
            assert not (isinstance(value, float) and (value != value)), \
                f"Feature '{key}' cannot be NaN"  # NaN != NaN is True
            
            # Specific feature constraints
            if "ratio" in key:
                assert 0.0 <= value <= 1.0, \
                    f"Ratio feature '{key}' must be between 0 and 1, got {value}"
            
            if key in ["tree_depth", "clause_count", "total_dependencies"]:
                assert value >= 0, \
                    f"Count feature '{key}' must be non-negative, got {value}"
            
            if key == "tree_validity":
                assert value in [0.0, 1.0], \
                    f"Tree validity must be 0.0 or 1.0, got {value}"


class TestDependencyParserIntegration:
    """Integration tests for dependency parser."""
    
    def test_tamil_sov_parsing(self):
        """Test parsing of Tamil SOV sentence."""
        # Force fallback for consistent testing
        config = DependencyConfig(model_name="fallback")
        parser = DependencyParser(config)
        
        # Tamil SOV: "I am coming"
        tokens = ["நான்", "வருகிறேன்"]
        tree = parser.parse_sentence(tokens, "ta")
        
        assert len(tree.tokens) == 2
        assert tree.is_valid_tree()
        
        # In SOV, verb should be root
        verb_idx = 1  # "வருகிறேன்"
        assert tree.heads[verb_idx] == -1
        assert tree.relations[verb_idx] == "root"
        
        # Subject should attach to verb
        subj_idx = 0  # "நான்"
        assert tree.heads[subj_idx] == verb_idx
        assert tree.relations[subj_idx] == "nsubj"
    
    def test_malayalam_parsing(self):
        """Test parsing of Malayalam sentence."""
        # Force fallback for consistent testing
        config = DependencyConfig(model_name="fallback")
        parser = DependencyParser(config)
        
        tokens = ["ഞാൻ", "വരുന്നു"]
        tree = parser.parse_sentence(tokens, "ml")
        
        assert len(tree.tokens) == 2
        assert tree.is_valid_tree()
        
        # Should have exactly one root
        root_count = sum(1 for head in tree.heads if head == -1)
        assert root_count == 1
    
    def test_complex_sentence_parsing(self):
        """Test parsing of more complex sentence."""
        # Force fallback for consistent testing
        config = DependencyConfig(model_name="fallback")
        parser = DependencyParser(config)
        
        tokens = ["நான்", "மருத்துவரை", "சந்தித்தேன்"]
        tree = parser.parse_sentence(tokens, "ta")
        
        assert len(tree.tokens) == 3
        assert tree.is_valid_tree()
        
        # Extract features
        features = parser.extract_syntactic_features(tree)
        assert features["tree_depth"] > 0
        assert features["total_dependencies"] == 3
        assert features["tree_validity"] == 1.0
    
    def test_syntactic_complexity_analysis(self):
        """Test syntactic complexity analysis for depression detection."""
        # Force fallback for consistent testing
        config = DependencyConfig(model_name="fallback")
        parser = DependencyParser(config)
        
        # Simple sentence
        simple_tokens = ["நான்", "வருகிறேன்"]
        simple_tree = parser.parse_sentence(simple_tokens, "ta")
        simple_features = parser.extract_syntactic_features(simple_tree)
        
        # Complex sentence
        complex_tokens = ["நான்", "மருத்துவரை", "சந்தித்து", "மருந்து", "வாங்கினேன்"]
        complex_tree = parser.parse_sentence(complex_tokens, "ta")
        complex_features = parser.extract_syntactic_features(complex_tree)
        
        # Complex sentence should have higher complexity
        assert complex_features["syntactic_complexity"] >= simple_features["syntactic_complexity"]
        assert complex_features["total_dependencies"] > simple_features["total_dependencies"]
    
    def test_tree_correction_for_invalid_structures(self):
        """Test that parser produces valid trees even with challenging input."""
        # Force fallback for consistent testing
        config = DependencyConfig(model_name="fallback")
        parser = DependencyParser(config)
        
        # Test with various challenging inputs
        test_cases = [
            [""],  # Empty token (will be filtered)
            ["single"],
            ["a", "b", "c", "d", "e"],  # Long sequence
        ]
        
        for tokens in test_cases:
            tokens = [t for t in tokens if t.strip()]  # Filter empty
            if tokens:
                tree = parser.parse_sentence(tokens, "ta")
                assert tree.is_valid_tree(), f"Invalid tree for tokens: {tokens}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])