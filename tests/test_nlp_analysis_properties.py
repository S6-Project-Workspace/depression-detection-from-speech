"""
Property-Based Tests for NLP Analysis Components

This module contains property-based tests for the NLP analysis functionality
in the web interface, validating universal properties across all inputs.

Properties tested:
- Property 6: POS tagging integration
- Property 7: Named entity extraction  
- Property 8: Dependency parsing
- Property 9: Sentiment analysis computation

Reference: NLP Web Interface - Requirements 2.1, 2.2, 2.3, 2.4
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any
import logging

# Import components to test
try:
    from web_interface.linguistic_analyzer_wrapper import (
        WebLinguisticAnalyzer, WebLinguisticConfig, create_web_linguistic_analyzer
    )
    from web_interface.sentiment_analyzer import SentimentAnalyzer, create_sentiment_analyzer
    HAS_WEB_COMPONENTS = True
except ImportError as e:
    logging.warning(f"Web components not available for testing: {e}")
    HAS_WEB_COMPONENTS = False

# Test data generators
@st.composite
def valid_text_input(draw):
    """Generate valid text input for analysis."""
    # Generate text with 1-50 words
    words = draw(st.lists(
        st.text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='நான் அவர் மகிழ்ச்சி வருத்தம் ഞാൻ അവർ സന്തോഷം ദുഃഖം'
            ),
            min_size=1,
            max_size=15
        ).filter(lambda x: x.strip() and not any(ord(c) < 32 and c not in '\t\n\r' for c in x)),
        min_size=1,
        max_size=50
    ))
    
    # Join words with spaces
    text = ' '.join(words)
    assume(len(text.strip()) > 0)
    assume(len(text) <= 1000)  # Reasonable length limit
    
    return text

@st.composite
def valid_language_code(draw):
    """Generate valid language codes."""
    return draw(st.sampled_from(['ta', 'ml', 'en']))

@st.composite
def valid_tokens(draw):
    """Generate valid token lists."""
    tokens = draw(st.lists(
        st.text(
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='நான் அவர் மகிழ்ச்சி வருத்தம் ഞാൻ അവർ സന്തോഷം ദുഃഖം'
            ),
            min_size=1,
            max_size=10
        ).filter(lambda x: x.strip()),
        min_size=1,
        max_size=20
    ))
    
    return tokens


class TestNLPAnalysisProperties:
    """Property-based tests for NLP analysis components."""
    
    @pytest.fixture(scope="class")
    def web_analyzer(self):
        """Create web linguistic analyzer for testing."""
        if not HAS_WEB_COMPONENTS:
            pytest.skip("Web components not available")
        
        return create_web_linguistic_analyzer(
            device="cpu",
            max_concurrent_tasks=2,
            enable_fallback=True
        )
    
    @pytest.fixture(scope="class")
    def sentiment_analyzer(self):
        """Create sentiment analyzer for testing."""
        if not HAS_WEB_COMPONENTS:
            pytest.skip("Web components not available")
        
        return create_sentiment_analyzer(device="cpu")
    
    @given(text=valid_text_input(), language=valid_language_code())
    @settings(max_examples=100, deadline=30000)  # 30 second timeout
    def test_property_6_pos_tagging_integration(self, web_analyzer, text, language):
        """
        Property 6: POS tagging integration
        For any text input, POS tags should be generated using the existing POS tagger 
        with consistent format and accuracy.
        
        **Validates: Requirements 2.1**
        """
        # Run analysis
        result = asyncio.run(web_analyzer.analyze_text(text, language))
        
        # Property: Analysis should complete without critical errors
        assert result is not None, "Analysis should return a result"
        
        # Property: Tokens should be generated
        assert isinstance(result.tokens, list), "Tokens should be a list"
        assert len(result.tokens) > 0, "Should generate at least one token for non-empty text"
        
        # Property: POS result should be consistent with tokens
        if result.pos_result is not None:
            # If POS analysis succeeded, it should have consistent structure
            assert hasattr(result.pos_result, 'tokens'), "POS result should have tokens"
            assert hasattr(result.pos_result, 'tags'), "POS result should have tags"
            assert hasattr(result.pos_result, 'confidence_scores'), "POS result should have confidence scores"
            
            # Property: Token count consistency
            assert len(result.pos_result.tokens) == len(result.pos_result.tags), \
                "POS tokens and tags should have same length"
            assert len(result.pos_result.tokens) == len(result.pos_result.confidence_scores), \
                "POS tokens and confidence scores should have same length"
            
            # Property: Confidence scores should be valid
            for conf in result.pos_result.confidence_scores:
                assert 0.0 <= conf <= 1.0, f"Confidence score {conf} should be between 0 and 1"
            
            # Property: Tags should be valid POS tags
            valid_pos_tags = {
                "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", 
                "VERB", "X", "O"
            }
            for tag in result.pos_result.tags:
                assert tag in valid_pos_tags, f"POS tag '{tag}' should be valid"
        
        # Property: Component status should indicate POS tagger status
        assert "pos" in result.component_status or len(result.error_messages) > 0, \
            "Should have POS component status or error messages"
    
    @given(text=valid_text_input(), language=valid_language_code())
    @settings(max_examples=100, deadline=30000)
    def test_property_7_named_entity_extraction(self, web_analyzer, text, language):
        """
        Property 7: Named entity extraction
        For any text containing identifiable entities, the NER system should extract them 
        with appropriate labels and confidence scores.
        
        **Validates: Requirements 2.2**
        """
        # Run analysis
        result = asyncio.run(web_analyzer.analyze_text(text, language))
        
        # Property: Analysis should complete
        assert result is not None, "Analysis should return a result"
        
        # Property: Entities should be a list
        assert isinstance(result.entities, list), "Entities should be a list"
        
        # Property: Each entity should have required attributes
        for entity in result.entities:
            assert hasattr(entity, 'text'), "Entity should have text attribute"
            assert hasattr(entity, 'label'), "Entity should have label attribute"
            assert hasattr(entity, 'confidence'), "Entity should have confidence attribute"
            
            # Property: Entity text should not be empty
            assert len(getattr(entity, 'text', '')) > 0, "Entity text should not be empty"
            
            # Property: Entity label should be valid
            valid_labels = {"PERSON", "LOCATION", "ORGANIZATION", "MEDICAL", "MISC", "O"}
            entity_label = getattr(entity, 'label', '')
            assert entity_label in valid_labels, f"Entity label '{entity_label}' should be valid"
            
            # Property: Confidence should be valid
            confidence = getattr(entity, 'confidence', 0.0)
            assert 0.0 <= confidence <= 1.0, f"Entity confidence {confidence} should be between 0 and 1"
            
            # Property: Entity positions should be valid if available
            if hasattr(entity, 'start_idx') and hasattr(entity, 'end_idx'):
                start_idx = getattr(entity, 'start_idx', 0)
                end_idx = getattr(entity, 'end_idx', 0)
                assert start_idx >= 0, "Entity start index should be non-negative"
                assert end_idx > start_idx, "Entity end index should be greater than start index"
        
        # Property: Component status should indicate NER system status
        assert "ner" in result.component_status or len(result.error_messages) > 0, \
            "Should have NER component status or error messages"
    
    @given(text=valid_text_input(), language=valid_language_code())
    @settings(max_examples=100, deadline=30000)
    def test_property_8_dependency_parsing(self, web_analyzer, text, language):
        """
        Property 8: Dependency parsing
        For any grammatically valid sentence, dependency parse trees should be generated 
        with correct head-dependent relationships.
        
        **Validates: Requirements 2.3**
        """
        # Run analysis
        result = asyncio.run(web_analyzer.analyze_text(text, language))
        
        # Property: Analysis should complete
        assert result is not None, "Analysis should return a result"
        
        # Property: Dependency tree structure should be valid if present
        if result.dependency_tree is not None:
            dep_tree = result.dependency_tree
            
            # Property: Dependency tree should have required attributes
            assert hasattr(dep_tree, 'tokens'), "Dependency tree should have tokens"
            assert hasattr(dep_tree, 'heads'), "Dependency tree should have heads"
            assert hasattr(dep_tree, 'relations'), "Dependency tree should have relations"
            assert hasattr(dep_tree, 'confidence_scores'), "Dependency tree should have confidence scores"
            
            # Property: All lists should have same length
            tokens = getattr(dep_tree, 'tokens', [])
            heads = getattr(dep_tree, 'heads', [])
            relations = getattr(dep_tree, 'relations', [])
            confidences = getattr(dep_tree, 'confidence_scores', [])
            
            if tokens:  # Only check if we have tokens
                assert len(tokens) == len(heads), "Tokens and heads should have same length"
                assert len(tokens) == len(relations), "Tokens and relations should have same length"
                assert len(tokens) == len(confidences), "Tokens and confidences should have same length"
                
                # Property: Head indices should be valid
                for i, head in enumerate(heads):
                    if head != -1:  # Not root
                        assert 0 <= head < len(tokens), f"Head index {head} should be valid"
                        assert head != i, f"Token {i} should not be its own head"
                
                # Property: Exactly one root should exist
                root_count = sum(1 for head in heads if head == -1)
                assert root_count == 1, f"Should have exactly one root, found {root_count}"
                
                # Property: Relations should be valid
                valid_relations = {
                    "root", "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp",
                    "obl", "vocative", "expl", "dislocated", "advcl", "advmod",
                    "discourse", "aux", "auxpass", "cop", "mark", "nmod", "appos",
                    "nummod", "acl", "amod", "det", "clf", "case", "conj", "cc",
                    "fixed", "flat", "compound", "list", "parataxis", "orphan",
                    "goeswith", "reparandum", "punct", "dep"
                }
                
                for relation in relations:
                    assert relation in valid_relations, f"Relation '{relation}' should be valid"
                
                # Property: Confidence scores should be valid
                for conf in confidences:
                    assert 0.0 <= conf <= 1.0, f"Confidence {conf} should be between 0 and 1"
        
        # Property: Component status should indicate dependency parser status
        assert "dep" in result.component_status or len(result.error_messages) > 0, \
            "Should have dependency parser component status or error messages"
    
    @given(text=valid_text_input(), language=valid_language_code())
    @settings(max_examples=100, deadline=30000)
    def test_property_9_sentiment_analysis_computation(self, web_analyzer, text, language):
        """
        Property 9: Sentiment analysis computation
        For any text input, sentiment scores should be computed and returned 
        within the expected range [-1, 1] or [0, 1].
        
        **Validates: Requirements 2.4**
        """
        # Run analysis
        result = asyncio.run(web_analyzer.analyze_text(text, language))
        
        # Property: Analysis should complete
        assert result is not None, "Analysis should return a result"
        
        # Property: Sentiment result should be valid if present
        if result.sentiment_result is not None:
            sentiment = result.sentiment_result
            
            # Property: Sentiment should have required attributes
            assert hasattr(sentiment, 'overall_sentiment'), "Sentiment should have overall_sentiment"
            assert hasattr(sentiment, 'confidence'), "Sentiment should have confidence"
            assert hasattr(sentiment, 'scores'), "Sentiment should have scores"
            
            # Property: Overall sentiment should be valid
            overall_sentiment = getattr(sentiment, 'overall_sentiment', '')
            valid_sentiments = {"positive", "negative", "neutral"}
            assert overall_sentiment in valid_sentiments, \
                f"Overall sentiment '{overall_sentiment}' should be valid"
            
            # Property: Confidence should be in valid range
            confidence = getattr(sentiment, 'confidence', 0.0)
            assert 0.0 <= confidence <= 1.0, f"Sentiment confidence {confidence} should be between 0 and 1"
            
            # Property: Scores should be valid
            scores = getattr(sentiment, 'scores', {})
            assert isinstance(scores, dict), "Sentiment scores should be a dictionary"
            
            required_score_keys = {"positive", "negative", "neutral"}
            for key in required_score_keys:
                assert key in scores, f"Sentiment scores should contain '{key}'"
                score = scores[key]
                assert 0.0 <= score <= 1.0, f"Sentiment score for '{key}' ({score}) should be between 0 and 1"
            
            # Property: Scores should approximately sum to 1.0 (allowing for floating point errors)
            total_score = sum(scores.values())
            assert 0.8 <= total_score <= 1.2, f"Sentiment scores should approximately sum to 1.0, got {total_score}"
        
        # Property: Component status should indicate sentiment analyzer status
        assert "sentiment" in result.component_status or len(result.error_messages) > 0, \
            "Should have sentiment component status or error messages"
    
    @given(tokens=valid_tokens(), language=valid_language_code())
    @settings(max_examples=50, deadline=20000)
    def test_sentiment_analyzer_direct_properties(self, sentiment_analyzer, tokens, language):
        """
        Additional property test for sentiment analyzer directly.
        Tests sentiment computation properties with token input.
        """
        if not HAS_WEB_COMPONENTS:
            pytest.skip("Web components not available")
        
        # Create text from tokens
        text = ' '.join(tokens)
        
        # Run sentiment analysis
        result = sentiment_analyzer.analyze_sentiment(text, tokens, language)
        
        # Property: Result should be valid
        assert result is not None, "Sentiment analysis should return a result"
        assert hasattr(result, 'overall_sentiment'), "Should have overall_sentiment"
        assert hasattr(result, 'confidence'), "Should have confidence"
        assert hasattr(result, 'scores'), "Should have scores"
        
        # Property: Sentiment values should be valid
        assert result.overall_sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        
        # Property: Scores should be valid probabilities
        for sentiment_type, score in result.scores.items():
            assert 0.0 <= score <= 1.0, f"Score for {sentiment_type} should be between 0 and 1"
        
        # Property: Required sentiment types should be present
        required_types = {"positive", "negative", "neutral"}
        assert required_types.issubset(result.scores.keys()), "Should have all required sentiment types"
    
    @given(texts=st.lists(valid_text_input(), min_size=1, max_size=5), language=valid_language_code())
    @settings(max_examples=20, deadline=60000)  # Longer timeout for batch processing
    def test_batch_analysis_properties(self, web_analyzer, texts, language):
        """
        Property test for batch analysis functionality.
        Tests that batch processing maintains individual analysis properties.
        """
        # Run batch analysis
        results = asyncio.run(web_analyzer.analyze_batch(texts, language))
        
        # Property: Should return same number of results as inputs
        assert len(results) == len(texts), "Batch analysis should return same number of results as inputs"
        
        # Property: Each result should be valid
        for i, result in enumerate(results):
            assert result is not None, f"Result {i} should not be None"
            assert hasattr(result, 'text'), f"Result {i} should have text attribute"
            assert hasattr(result, 'tokens'), f"Result {i} should have tokens attribute"
            assert hasattr(result, 'component_status'), f"Result {i} should have component_status"
            
            # Property: Text should match input
            assert result.text == texts[i], f"Result {i} text should match input text"
            
            # Property: Should have processing time
            assert hasattr(result, 'processing_time'), f"Result {i} should have processing_time"
            assert result.processing_time >= 0.0, f"Result {i} processing time should be non-negative"


# Utility functions for running tests
def run_property_tests():
    """Run all property tests."""
    if not HAS_WEB_COMPONENTS:
        print("Web components not available, skipping property tests")
        return
    
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_property_tests()