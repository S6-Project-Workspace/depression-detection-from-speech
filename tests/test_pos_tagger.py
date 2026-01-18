"""
Tests for Part-of-Speech Tagger

This module contains unit tests and property-based tests for the POS tagger
implementation for Dravidian languages.

Reference: Traditional NLP Tasks - Task 1
"""

import pytest
import torch
from hypothesis import given, strategies as st, assume, settings
from typing import List, Tuple
import numpy as np

from pos_tagger import POSTagger, POSConfig, POSResult, create_pos_tagger


class TestPOSConfig:
    """Test POS configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = POSConfig()
        assert config.model_name == "bert-base-multilingual-cased"
        assert config.tagset == "universal"
        assert config.max_length == 512
        assert config.batch_size == 16
        assert "ta" in config.languages
        assert "ml" in config.languages
        assert len(config.pos_tags) > 0
        assert "NOUN" in config.pos_tags
        assert "VERB" in config.pos_tags
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = POSConfig(
            model_name="custom-model",
            max_length=256,
            languages=["ta"]
        )
        assert config.model_name == "custom-model"
        assert config.max_length == 256
        assert config.languages == ["ta"]


class TestPOSResult:
    """Test POS result data structure."""
    
    def test_valid_pos_result(self):
        """Test valid POS result creation."""
        tokens = ["நான்", "வருகிறேன்"]
        tags = ["PRON", "VERB"]
        scores = [0.9, 0.8]
        
        result = POSResult(
            tokens=tokens,
            tags=tags,
            confidence_scores=scores,
            pos_distribution={"PRON": 1, "VERB": 1},
            language="ta"
        )
        
        assert result.tokens == tokens
        assert result.tags == tags
        assert result.confidence_scores == scores
        assert result.language == "ta"
    
    def test_invalid_pos_result_length_mismatch(self):
        """Test that mismatched lengths raise assertion error."""
        with pytest.raises(AssertionError):
            POSResult(
                tokens=["நான்", "வருகிறேன்"],
                tags=["PRON"],  # Mismatched length
                confidence_scores=[0.9, 0.8],
                pos_distribution={"PRON": 1},
                language="ta"
            )


class TestPOSTagger:
    """Test POS tagger functionality."""
    
    @pytest.fixture
    def tagger(self):
        """Create POS tagger for testing."""
        return create_pos_tagger("ta", "cpu")
    
    @pytest.fixture
    def sample_tokens(self):
        """Sample Tamil tokens for testing."""
        return ["நான்", "மகிழ்ச்சியாக", "இருக்கிறேன்"]
    
    def test_tagger_initialization(self, tagger):
        """Test tagger initialization."""
        assert tagger.config.languages == ["ta"]
        assert tagger.device.type == "cpu"
        assert len(tagger.config.pos_tags) > 0
    
    def test_tag_sentence_basic(self, tagger, sample_tokens):
        """Test basic sentence tagging."""
        tagged = tagger.tag_sentence(sample_tokens, "ta")
        
        assert len(tagged) == len(sample_tokens)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in tagged)
        
        tokens, tags = zip(*tagged)
        assert list(tokens) == sample_tokens
        assert all(tag in tagger.config.pos_tags for tag in tags)
    
    def test_tag_empty_sentence(self, tagger):
        """Test tagging empty sentence."""
        tagged = tagger.tag_sentence([], "ta")
        assert tagged == []
    
    def test_tag_batch(self, tagger, sample_tokens):
        """Test batch tagging."""
        sentences = [sample_tokens, ["அவன்", "வருகிறான்"]]
        tagged_batch = tagger.tag_batch(sentences, "ta")
        
        assert len(tagged_batch) == 2
        assert len(tagged_batch[0]) == len(sample_tokens)
        assert len(tagged_batch[1]) == 2
    
    def test_extract_pos_features(self, tagger, sample_tokens):
        """Test POS feature extraction."""
        tagged = tagger.tag_sentence(sample_tokens, "ta")
        features = tagger.extract_pos_features(tagged)
        
        # Check required features exist
        required_features = [
            "pos_diversity", "function_word_ratio", "content_word_ratio",
            "pronoun_ratio", "first_person_pronoun_ratio"
        ]
        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert 0.0 <= features[feature] <= 1.0 or feature == "pos_diversity"
    
    def test_extract_pos_features_empty(self, tagger):
        """Test feature extraction with empty input."""
        features = tagger.extract_pos_features([])
        
        assert isinstance(features, dict)
        assert features["pos_diversity"] == 0.0
        assert features["function_word_ratio"] == 0.0
    
    def test_create_pos_result(self, tagger):
        """Test POS result creation."""
        tokens = ["நான்", "வருகிறேன்"]
        tags = ["PRON", "VERB"]
        scores = [0.9, 0.8]
        
        result = tagger.create_pos_result(tokens, tags, scores, "ta")
        
        assert isinstance(result, POSResult)
        assert result.tokens == tokens
        assert result.tags == tags
        assert result.confidence_scores == scores
        assert result.language == "ta"
        assert result.pos_distribution == {"PRON": 1, "VERB": 1}


class TestPOSTaggerPropertyBased:
    """Property-based tests for POS tagger."""
    
    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
    @settings(deadline=None, max_examples=10)  # Reduce examples and remove deadline
    def test_property_tag_count_consistency(self, tokens):
        """
        Property 1: For any tokenized sentence, the number of POS tags 
        returned SHALL equal the number of input tokens.
        
        **Feature: traditional-nlp-tasks, Property 1**
        **Validates: Requirements 1.1, 1.4**
        """
        # Create tagger with fallback (no model loading)
        config = POSConfig(model_name="fallback")
        tagger = POSTagger(config)
        
        # Filter out empty tokens and very long tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        tagged = tagger.tag_sentence(tokens, "ta")
        
        # Property: tag count equals token count
        assert len(tagged) == len(tokens), \
            f"Tag count ({len(tagged)}) must equal token count ({len(tokens)})"
        
        # Verify structure
        for item in tagged:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # token
            assert isinstance(item[1], str)  # tag
    
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
        # Create tagger with fallback (no model loading)
        config = POSConfig(model_name="fallback", languages=[language])
        tagger = POSTagger(config)
        
        # Filter tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        tagged = tagger.tag_sentence(tokens, language)
        
        # Property: consistent processing for specified language
        assert len(tagged) == len(tokens)
        
        # All tags should be valid POS tags
        _, tags = zip(*tagged)
        for tag in tags:
            assert tag in tagger.config.pos_tags, \
                f"Tag '{tag}' not in valid POS tagset"
    
    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
    @settings(deadline=None, max_examples=10)
    def test_property_feature_vector_validity(self, tokens):
        """
        Test that extracted features are always valid.
        """
        # Create tagger with fallback (no model loading)
        config = POSConfig(model_name="fallback")
        tagger = POSTagger(config)
        
        # Filter tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        tagged = tagger.tag_sentence(tokens, "ta")
        features = tagger.extract_pos_features(tagged)
        
        # Property: all features are valid numbers
        for key, value in features.items():
            assert isinstance(value, (int, float)), \
                f"Feature '{key}' must be numeric, got {type(value)}"
            assert not np.isnan(value), \
                f"Feature '{key}' cannot be NaN"
            assert not np.isinf(value), \
                f"Feature '{key}' cannot be infinite"
            
            # Most ratios should be between 0 and 1
            if "ratio" in key:
                assert 0.0 <= value <= 1.0, \
                    f"Ratio feature '{key}' must be between 0 and 1, got {value}"


class TestPOSTaggerIntegration:
    """Integration tests for POS tagger."""
    
    def test_tamil_sentence_tagging(self):
        """Test tagging of actual Tamil sentence."""
        tagger = create_pos_tagger("ta")
        
        # Tamil sentence: "I am happy"
        tokens = ["நான்", "மகிழ்ச்சியாக", "இருக்கிறேன்"]
        tagged = tagger.tag_sentence(tokens, "ta")
        
        assert len(tagged) == 3
        tokens_out, tags_out = zip(*tagged)
        assert list(tokens_out) == tokens
        
        # Extract features
        features = tagger.extract_pos_features(tagged)
        assert "first_person_pronoun_ratio" in features
        assert features["first_person_pronoun_ratio"] > 0  # Should detect "நான்"
    
    def test_malayalam_sentence_tagging(self):
        """Test tagging of actual Malayalam sentence."""
        tagger = create_pos_tagger("ml")
        
        # Malayalam sentence: "I am coming"
        tokens = ["ഞാൻ", "വരുന്നു"]
        tagged = tagger.tag_sentence(tokens, "ml")
        
        assert len(tagged) == 2
        tokens_out, tags_out = zip(*tagged)
        assert list(tokens_out) == tokens
    
    def test_punctuation_handling(self):
        """Test handling of punctuation."""
        # Force fallback tagger
        config = POSConfig(model_name="fallback")
        tagger = POSTagger(config)
        
        tokens = ["நான்", "வருகிறேன்", "."]
        tagged = tagger.tag_sentence(tokens, "ta")
        
        assert len(tagged) == 3
        # Last token should be tagged as punctuation
        assert tagged[2][1] == "PUNCT"
    
    def test_batch_processing_consistency(self):
        """Test that batch processing gives same results as individual processing."""
        tagger = create_pos_tagger("ta")
        
        sentences = [
            ["நான்", "வருகிறேன்"],
            ["அவன்", "போகிறான்"]
        ]
        
        # Process individually
        individual_results = [tagger.tag_sentence(sent, "ta") for sent in sentences]
        
        # Process in batch
        batch_results = tagger.tag_batch(sentences, "ta")
        
        assert individual_results == batch_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])