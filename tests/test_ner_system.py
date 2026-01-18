"""
Tests for Named Entity Recognition System

This module contains unit tests and property-based tests for the NER system
implementation for clinical context and Dravidian languages.

Reference: Traditional NLP Tasks - Task 2
"""

import pytest
import torch
from hypothesis import given, strategies as st, assume, settings
from typing import List
import re

from ner_system import NERSystem, NERConfig, Entity, create_ner_system


class TestNERConfig:
    """Test NER configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NERConfig()
        assert config.model_name == "bert-base-multilingual-cased"
        assert "PERSON" in config.entity_types
        assert "MEDICAL" in config.entity_types
        assert "LOCATION" in config.entity_types
        assert config.max_length == 512
        assert config.anonymize_entities is True
        assert config.confidence_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = NERConfig(
            model_name="custom-model",
            entity_types=["PERSON", "LOCATION"],
            anonymize_entities=False
        )
        assert config.model_name == "custom-model"
        assert config.entity_types == ["PERSON", "LOCATION"]
        assert config.anonymize_entities is False


class TestEntity:
    """Test Entity data structure."""
    
    def test_entity_creation(self):
        """Test entity creation with basic fields."""
        entity = Entity(
            text="John",
            label="PERSON",
            start_idx=0,
            end_idx=4,
            confidence=0.9
        )
        
        assert entity.text == "John"
        assert entity.label == "PERSON"
        assert entity.start_idx == 0
        assert entity.end_idx == 4
        assert entity.confidence == 0.9
        assert entity.anonymized_form.startswith("PERSON_")
    
    def test_entity_anonymization(self):
        """Test entity anonymization."""
        entity = Entity(
            text="Dr. Smith",
            label="PERSON",
            start_idx=0,
            end_idx=9,
            confidence=0.8
        )
        
        # Should generate consistent anonymized form
        assert entity.anonymized_form.startswith("PERSON_")
        assert len(entity.anonymized_form) == 13  # "PERSON_" + 6 chars
        
        # Same text should generate same anonymized form
        entity2 = Entity(
            text="Dr. Smith",
            label="PERSON",
            start_idx=10,
            end_idx=19,
            confidence=0.9
        )
        assert entity.anonymized_form == entity2.anonymized_form
    
    def test_different_entity_types_anonymization(self):
        """Test anonymization for different entity types."""
        person = Entity("John", "PERSON", 0, 4, 0.9)
        location = Entity("Hospital", "LOCATION", 0, 8, 0.8)
        medical = Entity("depression", "MEDICAL", 0, 10, 0.7)
        
        assert person.anonymized_form.startswith("PERSON_")
        assert location.anonymized_form.startswith("LOCATION_")
        assert medical.anonymized_form == "MEDICAL_TERM"


class TestNERSystem:
    """Test NER system functionality."""
    
    @pytest.fixture
    def ner_system(self):
        """Create NER system for testing."""
        return create_ner_system("ta", "cpu", anonymize=True)
    
    @pytest.fixture
    def sample_tokens(self):
        """Sample tokens for testing."""
        return ["நான்", "மருத்துவரை", "சந்தித்தேன்"]
    
    def test_ner_system_initialization(self, ner_system):
        """Test NER system initialization."""
        assert ner_system.config.anonymize_entities is True
        assert ner_system.device.type == "cpu"
        assert len(ner_system.config.entity_types) > 0
    
    def test_extract_entities_basic(self, ner_system, sample_tokens):
        """Test basic entity extraction."""
        entities = ner_system.extract_entities(sample_tokens, "ta")
        
        assert isinstance(entities, list)
        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.label in ner_system.config.entity_types
            assert 0.0 <= entity.confidence <= 1.0
            assert entity.start_idx >= 0
            assert entity.end_idx > entity.start_idx
    
    def test_extract_entities_empty(self, ner_system):
        """Test entity extraction with empty input."""
        entities = ner_system.extract_entities([], "ta")
        assert entities == []
    
    def test_anonymize_text(self, ner_system):
        """Test text anonymization."""
        text = "நான் மருத்துவரை சந்தித்தேன்"
        entities = [
            Entity("நான்", "PERSON", 0, 4, 0.9),
            Entity("மருத்துவரை", "MEDICAL", 5, 15, 0.8)
        ]
        
        anonymized = ner_system.anonymize_text(text, entities)
        
        # Original entities should be replaced
        assert "நான்" not in anonymized
        assert "மருத்துவரை" not in anonymized
        assert "PERSON_" in anonymized
        assert "MEDICAL_TERM" in anonymized
    
    def test_anonymize_text_disabled(self):
        """Test anonymization when disabled."""
        config = NERConfig(anonymize_entities=False)
        ner_system = NERSystem(config)
        
        text = "John went to hospital"
        entities = [Entity("John", "PERSON", 0, 4, 0.9)]
        
        anonymized = ner_system.anonymize_text(text, entities)
        assert anonymized == text  # Should be unchanged
    
    def test_extract_ner_features(self, ner_system):
        """Test NER feature extraction."""
        entities = [
            Entity("நான்", "PERSON", 0, 4, 0.9),
            Entity("மருத்துவரை", "MEDICAL", 5, 15, 0.8),
            Entity("சென்னை", "LOCATION", 16, 22, 0.7)
        ]
        
        features = ner_system.extract_ner_features(entities)
        
        # Check required features exist
        required_features = [
            "entity_density", "entity_diversity", "avg_entity_confidence",
            "person_entity_count", "medical_entity_count", "person_entity_ratio"
        ]
        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
        
        # Check specific values
        assert features["entity_density"] == 3
        assert features["entity_diversity"] == 3
        assert features["person_entity_count"] == 1
        assert features["medical_entity_count"] == 1
        assert features["person_entity_ratio"] == 1/3
    
    def test_extract_ner_features_empty(self, ner_system):
        """Test feature extraction with empty entities."""
        features = ner_system.extract_ner_features([])
        
        assert isinstance(features, dict)
        assert features["entity_density"] == 0.0
        assert features["entity_diversity"] == 0.0
        assert features["person_entity_count"] == 0.0


class TestNERSystemPropertyBased:
    """Property-based tests for NER system."""
    
    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
    @settings(deadline=None, max_examples=10)
    def test_property_entity_boundary_validity(self, tokens):
        """
        Property 2: For any extracted entity, the start and end indices 
        SHALL be valid token boundaries within the input text.
        
        **Feature: traditional-nlp-tasks, Property 2**
        **Validates: Requirements 2.1, 2.4**
        """
        # Create NER system with fallback (no model loading)
        config = NERConfig(model_name="fallback")
        ner_system = NERSystem(config)
        
        # Filter tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        text = " ".join(tokens)
        entities = ner_system.extract_entities(tokens, "ta")
        
        # Property: all entity boundaries are valid
        for entity in entities:
            assert 0 <= entity.start_idx < entity.end_idx <= len(text), \
                f"Entity '{entity.text}' has invalid boundaries [{entity.start_idx}:{entity.end_idx}] for text length {len(text)}"
            
            # Entity text should match the text at those indices
            extracted_text = text[entity.start_idx:entity.end_idx]
            assert entity.text == extracted_text, \
                f"Entity text '{entity.text}' doesn't match extracted text '{extracted_text}'"
    
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
        # Create NER system with fallback (no model loading)
        config = NERConfig(model_name="fallback")
        ner_system = NERSystem(config)
        
        # Filter tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        entities = ner_system.extract_entities(tokens, language)
        
        # Property: consistent processing for specified language
        for entity in entities:
            assert entity.label in ner_system.config.entity_types, \
                f"Entity label '{entity.label}' not in valid entity types"
            assert 0.0 <= entity.confidence <= 1.0, \
                f"Entity confidence {entity.confidence} not in valid range [0, 1]"
    
    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
    @settings(deadline=None, max_examples=10)
    def test_property_feature_vector_validity(self, tokens):
        """
        Test that extracted NER features are always valid.
        """
        # Create NER system with fallback (no model loading)
        config = NERConfig(model_name="fallback")
        ner_system = NERSystem(config)
        
        # Filter tokens
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) < 50]
        assume(len(tokens) > 0)
        
        entities = ner_system.extract_entities(tokens, "ta")
        features = ner_system.extract_ner_features(entities)
        
        # Property: all features are valid numbers
        for key, value in features.items():
            assert isinstance(value, (int, float)), \
                f"Feature '{key}' must be numeric, got {type(value)}"
            assert not (isinstance(value, float) and (value != value)), \
                f"Feature '{key}' cannot be NaN"  # NaN != NaN is True
            
            # Count features should be non-negative
            if "count" in key:
                assert value >= 0, \
                    f"Count feature '{key}' must be non-negative, got {value}"
            
            # Ratio features should be between 0 and 1
            if "ratio" in key:
                assert 0.0 <= value <= 1.0, \
                    f"Ratio feature '{key}' must be between 0 and 1, got {value}"


class TestNERSystemIntegration:
    """Integration tests for NER system."""
    
    def test_tamil_medical_entities(self):
        """Test extraction of Tamil medical entities."""
        # Force fallback for consistent testing
        config = NERConfig(model_name="fallback")
        ner = NERSystem(config)
        
        tokens = ["நான்", "மருந்து", "எடுத்துக்கொள்கிறேன்"]
        entities = ner.extract_entities(tokens, "ta")
        
        # Should find person and medical entities
        entity_labels = [e.label for e in entities]
        assert "PERSON" in entity_labels  # "நான்"
        assert "MEDICAL" in entity_labels  # "மருந்து"
    
    def test_malayalam_entities(self):
        """Test extraction of Malayalam entities."""
        # Force fallback for consistent testing
        config = NERConfig(model_name="fallback")
        ner = NERSystem(config)
        
        tokens = ["ഞാൻ", "ഡോക്ടറെ", "കണ്ടു"]
        entities = ner.extract_entities(tokens, "ml")
        
        # Should find person entity
        entity_labels = [e.label for e in entities]
        assert "PERSON" in entity_labels  # "ഞാൻ"
    
    def test_anonymization_consistency(self):
        """Test that anonymization is consistent."""
        config = NERConfig(model_name="fallback", anonymize_entities=True)
        ner = NERSystem(config)
        
        # Same entity should get same anonymized form
        entity1 = Entity("John", "PERSON", 0, 4, 0.9)
        entity2 = Entity("John", "PERSON", 10, 14, 0.8)
        
        assert entity1.anonymized_form == entity2.anonymized_form
    
    def test_medical_context_detection(self):
        """Test detection of medical context."""
        config = NERConfig(model_name="fallback")
        ner = NERSystem(config)
        
        tokens = ["depression", "therapy", "medication"]
        entities = ner.extract_entities(tokens, "en")
        
        # Should detect medical entities
        medical_entities = [e for e in entities if e.label == "MEDICAL"]
        assert len(medical_entities) > 0
        
        # Extract features
        features = ner.extract_ner_features(entities)
        assert features["medical_entity_count"] > 0
        assert features["medical_context_ratio"] > 0
    
    def test_privacy_preservation(self):
        """Test privacy-preserving anonymization."""
        config = NERConfig(model_name="fallback", anonymize_entities=True)
        ner = NERSystem(config)
        
        text = "நான் Dr. Smith ஐ சந்தித்தேன்"
        tokens = text.split()
        entities = ner.extract_entities(tokens, "ta")
        
        anonymized = ner.anonymize_text(text, entities)
        
        # Original person names should not appear in anonymized text
        assert "Smith" not in anonymized
        # But anonymized forms should be present
        assert "PERSON_" in anonymized or "MEDICAL_" in anonymized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])