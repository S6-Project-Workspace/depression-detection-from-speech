"""
Property-based tests for web interface data models.

Tests data model serialization, validation, and consistency properties.
"""

import pytest
from hypothesis import given, strategies as st, assume
from hypothesis.strategies import composite
from datetime import datetime, timedelta
import json
from typing import Dict, Any

from web_interface.models import (
    POSTag, NamedEntity, DependencyRelation, SentimentAnalysis,
    LinguisticFeatures, AnalysisResult, SessionData, AudioQuality,
    ModelMetrics, TextAnalysisRequest, FileUploadResponse, ErrorResponse,
    WebSocketMessage, EntityLabel, SessionStatus
)


# Custom strategies for generating test data

@composite
def pos_tag_strategy(draw):
    """Generate valid POSTag instances."""
    token = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    tag = draw(st.sampled_from(['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PRT', 'PUNCT', 'X']))
    position = draw(st.integers(min_value=0, max_value=1000))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return POSTag(token=token, tag=tag, position=position, confidence=confidence)


@composite
def named_entity_strategy(draw):
    """Generate valid NamedEntity instances."""
    text = draw(st.text(min_size=1, max_size=100))
    label = draw(st.sampled_from(EntityLabel))
    start = draw(st.integers(min_value=0, max_value=500))
    end = draw(st.integers(min_value=start + 1, max_value=start + len(text) + 100))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return NamedEntity(text=text, label=label, start=start, end=end, confidence=confidence)


@composite
def dependency_relation_strategy(draw):
    """Generate valid DependencyRelation instances."""
    head = draw(st.integers(min_value=-1, max_value=100))
    dependent = draw(st.integers(min_value=0, max_value=100))
    relation = draw(st.sampled_from(['nsubj', 'obj', 'root', 'det', 'amod', 'nmod', 'advmod', 'aux', 'cop']))
    head_text = draw(st.text(min_size=1, max_size=50))
    dependent_text = draw(st.text(min_size=1, max_size=50))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return DependencyRelation(
        head=head, dependent=dependent, relation=relation,
        head_text=head_text, dependent_text=dependent_text, confidence=confidence
    )


@composite
def sentiment_analysis_strategy(draw):
    """Generate valid SentimentAnalysis instances."""
    overall_sentiment = draw(st.sampled_from(['positive', 'negative', 'neutral']))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Generate scores that sum to approximately 1.0
    pos_score = draw(st.floats(min_value=0.0, max_value=1.0))
    neg_score = draw(st.floats(min_value=0.0, max_value=1.0 - pos_score))
    neu_score = 1.0 - pos_score - neg_score
    
    scores = {
        'positive': pos_score,
        'negative': neg_score,
        'neutral': max(0.0, neu_score)  # Ensure non-negative
    }
    
    return SentimentAnalysis(overall_sentiment=overall_sentiment, confidence=confidence, scores=scores)


@composite
def linguistic_features_strategy(draw):
    """Generate valid LinguisticFeatures instances."""
    sentence_count = draw(st.integers(min_value=0, max_value=100))
    word_count = draw(st.integers(min_value=0, max_value=1000))
    avg_sentence_length = draw(st.floats(min_value=0.0, max_value=100.0))
    lexical_diversity = draw(st.floats(min_value=0.0, max_value=1.0))
    function_word_ratio = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Generate POS distribution that sums to 1.0
    pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']
    pos_values = draw(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=len(pos_tags), max_size=len(pos_tags)))
    total = sum(pos_values)
    if total > 0:
        pos_distribution = {tag: val / total for tag, val in zip(pos_tags, pos_values)}
    else:
        pos_distribution = {tag: 0.0 for tag in pos_tags}
    
    return LinguisticFeatures(
        sentence_count=sentence_count,
        word_count=word_count,
        avg_sentence_length=avg_sentence_length,
        lexical_diversity=lexical_diversity,
        function_word_ratio=function_word_ratio,
        pos_distribution=pos_distribution
    )


@composite
def analysis_result_strategy(draw):
    """Generate valid AnalysisResult instances."""
    text = draw(st.text(min_size=1, max_size=1000))
    tokens = draw(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=50))
    
    # Generate POS tags matching token count
    pos_tags = [draw(pos_tag_strategy()) for _ in range(len(tokens))]
    for i, pos_tag in enumerate(pos_tags):
        pos_tag.token = tokens[i]
        pos_tag.position = i
    
    named_entities = draw(st.lists(named_entity_strategy(), max_size=10))
    dependencies = draw(st.lists(dependency_relation_strategy(), max_size=20))
    sentiment = draw(sentiment_analysis_strategy())
    linguistic_features = draw(linguistic_features_strategy())
    language = draw(st.sampled_from(['en', 'ta', 'ml', 'hi']))
    processing_time = draw(st.floats(min_value=0.0, max_value=10.0))
    
    return AnalysisResult(
        text=text,
        tokens=tokens,
        pos_tags=pos_tags,
        named_entities=named_entities,
        dependencies=dependencies,
        sentiment=sentiment,
        linguistic_features=linguistic_features,
        language=language,
        processing_time=processing_time
    )


class TestDataModelSerialization:
    """Property tests for data model serialization and validation."""
    
    @given(pos_tag_strategy())
    def test_pos_tag_serialization_roundtrip(self, pos_tag):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any POSTag instance, serialization and deserialization should preserve data.
        **Validates: Requirements 10.4**
        """
        # Serialize to dict
        data = pos_tag.dict()
        
        # Deserialize back to object
        reconstructed = POSTag(**data)
        
        # Verify all fields match
        assert reconstructed.token == pos_tag.token
        assert reconstructed.tag == pos_tag.tag
        assert reconstructed.position == pos_tag.position
        assert abs(reconstructed.confidence - pos_tag.confidence) < 1e-10
    
    @given(named_entity_strategy())
    def test_named_entity_serialization_roundtrip(self, entity):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any NamedEntity instance, JSON serialization should be consistent.
        **Validates: Requirements 10.4**
        """
        # Serialize to JSON
        json_data = entity.json()
        
        # Deserialize from JSON
        reconstructed = NamedEntity.parse_raw(json_data)
        
        # Verify all fields match
        assert reconstructed.text == entity.text
        assert reconstructed.label == entity.label
        assert reconstructed.start == entity.start
        assert reconstructed.end == entity.end
        assert abs(reconstructed.confidence - entity.confidence) < 1e-10
    
    @given(sentiment_analysis_strategy())
    def test_sentiment_analysis_validation(self, sentiment):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any SentimentAnalysis instance, validation rules should be enforced.
        **Validates: Requirements 10.4**
        """
        # Verify confidence is in valid range
        assert 0.0 <= sentiment.confidence <= 1.0
        
        # Verify all required score keys exist
        required_keys = {'positive', 'negative', 'neutral'}
        assert required_keys.issubset(sentiment.scores.keys())
        
        # Verify all scores are in valid range
        for score in sentiment.scores.values():
            assert 0.0 <= score <= 1.0
    
    @given(analysis_result_strategy())
    def test_analysis_result_consistency(self, result):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any AnalysisResult instance, internal consistency should be maintained.
        **Validates: Requirements 10.4**
        """
        # Verify POS tags match token count
        assert len(result.pos_tags) == len(result.tokens)
        
        # Verify POS tag positions are consistent
        for i, pos_tag in enumerate(result.pos_tags):
            assert pos_tag.position == i
            assert pos_tag.token == result.tokens[i]
        
        # Verify processing time is non-negative
        assert result.processing_time >= 0.0
        
        # Verify timestamp is reasonable (within last year to next year)
        now = datetime.now()
        assert (now - timedelta(days=365)) <= result.timestamp <= (now + timedelta(days=365))
    
    @given(st.text(min_size=1, max_size=10000))
    def test_text_analysis_request_validation(self, text):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any valid text input, TextAnalysisRequest should validate correctly.
        **Validates: Requirements 10.4**
        """
        request = TextAnalysisRequest(text=text)
        
        # Verify text is preserved
        assert request.text == text
        
        # Verify defaults are set
        assert request.include_sentiment is True
        assert request.include_features is True
        
        # Verify serialization works
        data = request.dict()
        reconstructed = TextAnalysisRequest(**data)
        assert reconstructed.text == text
    
    @given(st.dictionaries(st.text(), st.one_of(st.text(), st.integers(), st.floats(allow_nan=False, allow_infinity=False))))
    def test_websocket_message_serialization(self, data_dict):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any WebSocket message data, serialization should be consistent.
        **Validates: Requirements 10.4**
        """
        message = WebSocketMessage(
            type="test_message",
            session_id="test_session",
            data=data_dict
        )
        
        # Serialize to JSON
        json_data = message.json()
        
        # Verify JSON is valid
        parsed_json = json.loads(json_data)
        assert parsed_json['type'] == "test_message"
        assert parsed_json['session_id'] == "test_session"
        assert parsed_json['data'] == data_dict
    
    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_audio_quality_validation(self, quality_score):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any audio quality metrics, validation should enforce constraints.
        **Validates: Requirements 10.4**
        """
        audio_quality = AudioQuality(
            signal_level=0.8,
            noise_level=0.2,
            snr=15.0,
            quality_score=quality_score,
            warnings=[]
        )
        
        # Verify all values are in valid ranges
        assert 0.0 <= audio_quality.signal_level <= 1.0
        assert 0.0 <= audio_quality.noise_level <= 1.0
        assert 0.0 <= audio_quality.quality_score <= 1.0
        
        # Verify serialization preserves precision
        data = audio_quality.dict()
        reconstructed = AudioQuality(**data)
        assert abs(reconstructed.quality_score - quality_score) < 1e-10
    
    @given(st.lists(analysis_result_strategy(), max_size=5))
    def test_session_data_with_results(self, results):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any session with analysis results, data integrity should be maintained.
        **Validates: Requirements 10.4**
        """
        session = SessionData(
            session_id="test_session",
            analysis_results=results
        )
        
        # Verify results are preserved
        assert len(session.analysis_results) == len(results)
        
        # Verify session status is valid
        assert session.status in [status.value for status in SessionStatus]
        
        # Verify timestamps are reasonable
        assert session.created_at <= session.updated_at
        
        # Test serialization with complex nested data
        json_data = session.json()
        parsed = json.loads(json_data)
        assert parsed['session_id'] == "test_session"
        assert len(parsed['analysis_results']) == len(results)


class TestModelValidation:
    """Tests for model validation rules."""
    
    def test_named_entity_end_after_start_validation(self):
        """Test that NamedEntity validates end > start."""
        with pytest.raises(ValueError, match="end must be greater than start"):
            NamedEntity(
                text="test",
                label=EntityLabel.PERSON,
                start=10,
                end=5,  # Invalid: end <= start
                confidence=0.9
            )
    
    def test_sentiment_scores_validation(self):
        """Test that SentimentAnalysis validates score requirements."""
        with pytest.raises(ValueError, match="scores must contain keys"):
            SentimentAnalysis(
                overall_sentiment="positive",
                confidence=0.8,
                scores={"positive": 0.7}  # Missing required keys
            )
        
        with pytest.raises(ValueError, match="all scores must be between 0.0 and 1.0"):
            SentimentAnalysis(
                overall_sentiment="positive",
                confidence=0.8,
                scores={"positive": 1.5, "negative": 0.2, "neutral": 0.3}  # Invalid score > 1.0
            )
    
    def test_pos_tags_length_validation(self):
        """Test that AnalysisResult validates POS tags match tokens."""
        with pytest.raises(ValueError, match="pos_tags length must match tokens length"):
            AnalysisResult(
                text="test text",
                tokens=["test", "text"],
                pos_tags=[POSTag(token="test", tag="NOUN", position=0, confidence=0.9)],  # Length mismatch
                named_entities=[],
                dependencies=[],
                sentiment=SentimentAnalysis(
                    overall_sentiment="neutral",
                    confidence=0.5,
                    scores={"positive": 0.3, "negative": 0.2, "neutral": 0.5}
                ),
                linguistic_features=LinguisticFeatures(
                    sentence_count=1,
                    word_count=2,
                    avg_sentence_length=2.0,
                    lexical_diversity=1.0,
                    function_word_ratio=0.0,
                    pos_distribution={"NOUN": 1.0}
                ),
                language="en",
                processing_time=0.1
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])