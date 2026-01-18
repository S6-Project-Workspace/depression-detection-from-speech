"""
Tests for Unified Linguistic Analyzer

This module contains unit tests and property-based tests for the unified
linguistic analyzer that combines POS tagging, NER, and dependency parsing.

Reference: Traditional NLP Tasks - Task 4
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from typing import List

from linguistic_analyzer import (
    LinguisticAnalyzer, LinguisticConfig, LinguisticAnalysis, 
    create_linguistic_analyzer
)
from pos_tagger import POSConfig, POSResult
from ner_system import NERConfig, Entity
from dependency_parser import DependencyConfig, DependencyTree


class TestLinguisticConfig:
    """Test linguistic analyzer configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LinguisticConfig()
        assert isinstance(config.pos_config, POSConfig)
        assert isinstance(config.ner_config, NERConfig)
        assert isinstance(config.dependency_config, DependencyConfig)
        assert config.feature_extraction is True
        assert config.device == "cpu"
    
    def test_device_sync(self):
        """Test that device is synced across all configs."""
        config = LinguisticConfig(device="cuda")
        assert config.pos_config.device == "cuda"
        assert config.ner_config.device == "cuda"
        assert config.dependency_config.device == "cuda"
    
    def test_custom_config(self):
        """Test custom configuration."""
        pos_config = POSConfig(model_name="custom-pos")
        ner_config = NERConfig(model_name="custom-ner")
        
        config = LinguisticConfig(
            pos_config=pos_config,
            ner_config=ner_config,
            feature_extraction=False
        )
        
        assert config.pos_config.model_name == "custom-pos"
        assert config.ner_config.model_name == "custom-ner"
        assert config.feature_extraction is False


class TestLinguisticAnalysis:
    """Test linguistic analysis data structure."""
    
    def test_valid_linguistic_analysis(self):
        """Test valid linguistic analysis creation."""
        tokens = ["நான்", "வருகிறேன்"]
        
        pos_result = POSResult(
            tokens=tokens,
            tags=["PRON", "VERB"],
            confidence_scores=[0.9, 0.8],
            pos_distribution={"PRON": 1, "VERB": 1},
            language="ta"
        )
        
        entities = [Entity("நான்", "PERSON", 0, 4, 0.9)]
        
        dependency_tree = DependencyTree(
            tokens=tokens,
            heads=[1, -1],
            relations=["nsubj", "root"],
            confidence_scores=[0.8, 0.9]
        )
        
        analysis = LinguisticAnalysis(
            text="நான் வருகிறேன்",
            tokens=tokens,
            pos_result=pos_result,
            entities=entities,
            dependency_tree=dependency_tree,
            linguistic_features={"complexity": 0.5},
            language="ta"
        )
        
        assert analysis.text == "நான் வருகிறேன்"
        assert analysis.tokens == tokens
        assert analysis.language == "ta"
    
    def test_invalid_analysis_token_mismatch(self):
        """Test that token count mismatch raises assertion error."""
        tokens = ["நான்", "வருகிறேன்"]
        
        pos_result = POSResult(
            tokens=["நான்"],  # Mismatched length
            tags=["PRON"],
            confidence_scores=[0.9],
            pos_distribution={"PRON": 1},
            language="ta"
        )
        
        dependency_tree = DependencyTree(
            tokens=tokens,
            heads=[1, -1],
            relations=["nsubj", "root"],
            confidence_scores=[0.8, 0.9]
        )
        
        with pytest.raises(AssertionError, match="Token count mismatch"):
            LinguisticAnalysis(
                text="நான் வருகிறேன்",
                tokens=tokens,
                pos_result=pos_result,
                entities=[],
                dependency_tree=dependency_tree,
                linguistic_features={},
                language="ta"
            )


class TestLinguisticAnalyzer:
    """Test linguistic analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create linguistic analyzer for testing."""
        return create_linguistic_analyzer("ta", "cpu", feature_extraction=True)
    
    @pytest.fixture
    def sample_text(self):
        """Sample Tamil text for testing."""
        return "நான் மருத்துவரை சந்தித்தேன்"
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config.feature_extraction is True
        assert analyzer.pos_tagger is not None
        assert analyzer.ner_system is not None
        assert analyzer.dependency_parser is not None
    
    def test_analyze_text_basic(self, analyzer, sample_text):
        """Test basic text analysis."""
        analysis = analyzer.analyze_text(sample_text, "ta")
        
        assert isinstance(analysis, LinguisticAnalysis)
        assert analysis.text == sample_text
        assert len(analysis.tokens) > 0
        assert len(analysis.pos_result.tags) == len(analysis.tokens)
        assert len(analysis.dependency_tree.tokens) == len(analysis.tokens)
        assert analysis.language == "ta"
        
        # Check that linguistic features were extracted
        assert isinstance(analysis.linguistic_features, dict)
    
    def test_analyze_empty_text(self, analyzer):
        """Test analysis of empty text."""
        analysis = analyzer.analyze_text("", "ta")
        
        assert analysis.text == ""
        assert analysis.tokens == []
        assert analysis.pos_result.tokens == []
        assert analysis.entities == []
        assert analysis.dependency_tree.tokens == []
    
    def test_analyze_whitespace_text(self, analyzer):
        """Test analysis of whitespace-only text."""
        analysis = analyzer.analyze_text("   ", "ta")
        
        assert analysis.tokens == []
        assert analysis.pos_result.tokens == []
    
    def test_extract_linguistic_features(self, analyzer, sample_text):
        """Test linguistic feature extraction."""
        analysis = analyzer.analyze_text(sample_text, "ta")
        
        feature_vector = analyzer.extract_linguistic_feature_vector(
            analysis.pos_result, analysis.entities, analysis.dependency_tree
        )
        
        assert isinstance(feature_vector, np.ndarray)
        assert feature_vector.shape == (128,)
        assert np.all(np.isfinite(feature_vector))
        
        # Check that features are in reasonable ranges
        assert np.all(feature_vector >= -10.0)
        assert np.all(feature_vector <= 100.0)
    
    def test_batch_analyze(self, analyzer):
        """Test batch text analysis."""
        texts = ["நான் வருகிறேன்", "அவன் போகிறான்", "நாம் செல்கிறோம்"]
        
        analyses = analyzer.batch_analyze(texts, "ta")
        
        assert len(analyses) == len(texts)
        for i, analysis in enumerate(analyses):
            assert isinstance(analysis, LinguisticAnalysis)
            assert analysis.text == texts[i]
            assert analysis.language == "ta"
    
    def test_feature_vector_consistency(self, analyzer):
        """Test that feature vectors are consistent for same input."""
        text = "நான் வருகிறேன்"
        
        analysis1 = analyzer.analyze_text(text, "ta")
        analysis2 = analyzer.analyze_text(text, "ta")
        
        features1 = analyzer.extract_linguistic_feature_vector(
            analysis1.pos_result, analysis1.entities, analysis1.dependency_tree
        )
        features2 = analyzer.extract_linguistic_feature_vector(
            analysis2.pos_result, analysis2.entities, analysis2.dependency_tree
        )
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_different_languages(self, analyzer):
        """Test analysis with different languages."""
        tamil_text = "நான் வருகிறேன்"
        malayalam_text = "ഞാൻ വരുന്നു"
        
        tamil_analysis = analyzer.analyze_text(tamil_text, "ta")
        malayalam_analysis = analyzer.analyze_text(malayalam_text, "ml")
        
        assert tamil_analysis.language == "ta"
        assert malayalam_analysis.language == "ml"
        
        # Both should produce valid analyses
        assert len(tamil_analysis.tokens) > 0
        assert len(malayalam_analysis.tokens) > 0


class TestLinguisticAnalyzerPropertyBased:
    """Property-based tests for linguistic analyzer."""
    
    @given(st.text(min_size=1, max_size=100))
    @settings(deadline=None, max_examples=10)
    def test_property_feature_vector_dimensionality(self, text):
        """
        Property 4: For any linguistic analysis, the extracted feature vector 
        SHALL have exactly 128 dimensions with all values in valid ranges.
        
        **Feature: traditional-nlp-tasks, Property 4**
        **Validates: Requirements 4.4, 4.5**
        """
        # Filter text
        text = text.strip()
        assume(len(text) > 0 and len(text) < 200)
        assume(not text.isspace())
        
        analyzer = create_linguistic_analyzer("ta", "cpu", feature_extraction=True)
        analysis = analyzer.analyze_text(text, "ta")
        
        if analysis.tokens:  # Only test if we have tokens
            feature_vector = analyzer.extract_linguistic_feature_vector(
                analysis.pos_result, analysis.entities, analysis.dependency_tree
            )
            
            # Property: exactly 128 dimensions
            assert feature_vector.shape == (128,), \
                f"Feature vector must have 128 dimensions, got {feature_vector.shape}"
            
            # Property: all values are finite
            assert np.all(np.isfinite(feature_vector)), \
                "All feature values must be finite"
            
            # Property: values in reasonable ranges
            assert np.all(feature_vector >= -10.0), \
                "Feature values must be >= -10.0"
            assert np.all(feature_vector <= 100.0), \
                "Feature values must be <= 100.0"
    
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
    @settings(deadline=None, max_examples=10)
    def test_property_batch_consistency(self, texts):
        """
        Test that batch analysis produces same results as individual analysis.
        """
        # Filter texts
        texts = [text.strip() for text in texts if text.strip() and len(text.strip()) < 50]
        assume(len(texts) > 0)
        
        analyzer = create_linguistic_analyzer("ta", "cpu", feature_extraction=False)
        
        # Individual analysis
        individual_results = [analyzer.analyze_text(text, "ta") for text in texts]
        
        # Batch analysis
        batch_results = analyzer.batch_analyze(texts, "ta")
        
        # Property: same number of results
        assert len(individual_results) == len(batch_results)
        
        # Property: corresponding results should be equivalent
        for individual, batch in zip(individual_results, batch_results):
            assert individual.text == batch.text
            assert individual.tokens == batch.tokens
            assert individual.language == batch.language
    
    @given(st.text(min_size=1, max_size=50), st.sampled_from(["ta", "ml"]))
    @settings(deadline=None, max_examples=10)
    def test_property_language_consistency(self, text, language):
        """
        Property 5: For any input text with specified language, all NLP tasks 
        SHALL use language-appropriate models and return results consistent 
        with that language's linguistic properties.
        
        **Feature: traditional-nlp-tasks, Property 5**
        **Validates: Requirements 1.2, 2.2, 3.2**
        """
        # Filter text
        text = text.strip()
        assume(len(text) > 0 and len(text) < 100)
        assume(not text.isspace())
        
        analyzer = create_linguistic_analyzer(language, "cpu", feature_extraction=False)
        analysis = analyzer.analyze_text(text, language)
        
        # Property: language consistency
        assert analysis.language == language
        
        # Property: all components processed the same tokens
        if analysis.tokens:
            assert len(analysis.pos_result.tokens) == len(analysis.tokens)
            assert len(analysis.dependency_tree.tokens) == len(analysis.tokens)
            
            # Property: POS tags are valid
            for tag in analysis.pos_result.tags:
                assert tag in analyzer.pos_tagger.config.pos_tags
            
            # Property: dependency tree is valid
            assert analysis.dependency_tree.is_valid_tree()


class TestLinguisticAnalyzerIntegration:
    """Integration tests for linguistic analyzer."""
    
    def test_tamil_comprehensive_analysis(self):
        """Test comprehensive analysis of Tamil text."""
        analyzer = create_linguistic_analyzer("ta", "cpu", feature_extraction=True)
        
        text = "நான் மருத்துவரை சந்தித்து மருந்து வாங்கினேன்"
        analysis = analyzer.analyze_text(text, "ta")
        
        # Should have tokens
        assert len(analysis.tokens) > 0
        
        # Should have POS tags
        assert len(analysis.pos_result.tags) == len(analysis.tokens)
        assert "PRON" in analysis.pos_result.tags  # "நான்"
        
        # Should have entities
        person_entities = [e for e in analysis.entities if e.label == "PERSON"]
        medical_entities = [e for e in analysis.entities if e.label == "MEDICAL"]
        assert len(person_entities) > 0 or len(medical_entities) > 0
        
        # Should have valid dependency tree
        assert analysis.dependency_tree.is_valid_tree()
        assert len(analysis.dependency_tree.relations) == len(analysis.tokens)
        
        # Should have linguistic features
        feature_vector = analyzer.extract_linguistic_feature_vector(
            analysis.pos_result, analysis.entities, analysis.dependency_tree
        )
        assert feature_vector.shape == (128,)
        assert np.any(feature_vector > 0)  # Should have some non-zero features
    
    def test_malayalam_analysis(self):
        """Test analysis of Malayalam text."""
        analyzer = create_linguistic_analyzer("ml", "cpu", feature_extraction=True)
        
        text = "ഞാൻ ഡോക്ടറെ കണ്ടു"
        analysis = analyzer.analyze_text(text, "ml")
        
        assert analysis.language == "ml"
        assert len(analysis.tokens) > 0
        assert len(analysis.pos_result.tags) == len(analysis.tokens)
        assert analysis.dependency_tree.is_valid_tree()
    
    def test_feature_extraction_components(self):
        """Test that all feature components contribute to final vector."""
        analyzer = create_linguistic_analyzer("ta", "cpu", feature_extraction=True)
        
        # Text with diverse linguistic features
        text = "நான் மருத்துவரை சந்தித்து மருந்து வாங்கினேன் சென்னையில்"
        analysis = analyzer.analyze_text(text, "ta")
        
        # Extract individual component features
        pos_features = analyzer.pos_tagger.extract_pos_features(
            list(zip(analysis.pos_result.tokens, analysis.pos_result.tags))
        )
        ner_features = analyzer.ner_system.extract_ner_features(analysis.entities)
        syntactic_features = analyzer.dependency_parser.extract_syntactic_features(
            analysis.dependency_tree
        )
        
        # All components should contribute features
        assert len(pos_features) > 0
        assert len(ner_features) > 0
        assert len(syntactic_features) > 0
        
        # Combined feature vector should incorporate all
        feature_vector = analyzer.extract_linguistic_feature_vector(
            analysis.pos_result, analysis.entities, analysis.dependency_tree
        )
        
        # Should have reasonable feature values
        assert np.sum(feature_vector > 0) > 10  # At least 10 non-zero features
    
    def test_error_handling_robustness(self):
        """Test that analyzer handles errors gracefully."""
        analyzer = create_linguistic_analyzer("ta", "cpu", feature_extraction=True)
        
        # Test with various challenging inputs
        challenging_texts = [
            "a",  # Single character
            "123 456",  # Numbers only
            "!@#$%",  # Special characters only
            "a" * 1000,  # Very long text
        ]
        
        for text in challenging_texts:
            try:
                analysis = analyzer.analyze_text(text, "ta")
                
                # Should always produce valid analysis
                assert isinstance(analysis, LinguisticAnalysis)
                assert analysis.text == text
                assert analysis.language == "ta"
                
                # If tokens exist, should have consistent structure
                if analysis.tokens:
                    assert len(analysis.pos_result.tags) == len(analysis.tokens)
                    assert len(analysis.dependency_tree.tokens) == len(analysis.tokens)
                    assert analysis.dependency_tree.is_valid_tree()
                
            except Exception as e:
                pytest.fail(f"Analyzer failed on text '{text}': {e}")
    
    def test_feature_vector_stability(self):
        """Test that feature vectors are stable across multiple runs."""
        analyzer = create_linguistic_analyzer("ta", "cpu", feature_extraction=True)
        
        text = "நான் வருகிறேன்"
        
        # Run analysis multiple times
        vectors = []
        for _ in range(3):
            analysis = analyzer.analyze_text(text, "ta")
            vector = analyzer.extract_linguistic_feature_vector(
                analysis.pos_result, analysis.entities, analysis.dependency_tree
            )
            vectors.append(vector)
        
        # All vectors should be identical
        for i in range(1, len(vectors)):
            np.testing.assert_array_equal(vectors[0], vectors[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])