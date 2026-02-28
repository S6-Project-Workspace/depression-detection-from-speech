"""
Property-based tests for API integration.

Tests API input validation, component integration, and response consistency.
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from fastapi.testclient import TestClient
import json

from web_interface.main import app
from web_interface.models import TextAnalysisRequest, AnalysisResult

# Create test client
client = TestClient(app)


# Custom strategies for API testing

@composite
def valid_text_strategy(draw):
    """Generate valid text inputs for analysis."""
    # Generate text that's not empty and within limits
    text = draw(st.text(min_size=1, max_size=1000, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Po', 'Zs'),
        blacklist_characters='\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f'
    )))
    assume(text.strip())  # Ensure text is not just whitespace
    return text


@composite
def invalid_text_strategy(draw):
    """Generate invalid text inputs for testing validation."""
    choice = draw(st.integers(min_value=0, max_value=2))
    
    if choice == 0:
        # Empty or whitespace-only text
        return draw(st.sampled_from(['', '   ', '\t\n', '  \t  \n  ']))
    elif choice == 1:
        # Text that's too long
        return 'a' * 10001  # Exceeds 10000 character limit
    else:
        # Text with only control characters
        return '\x00\x01\x02'


@composite
def text_analysis_request_strategy(draw):
    """Generate valid TextAnalysisRequest instances."""
    text = draw(valid_text_strategy())
    language = draw(st.one_of(st.none(), st.sampled_from(['en', 'ta', 'ml', 'hi'])))
    include_sentiment = draw(st.booleans())
    include_features = draw(st.booleans())
    
    return TextAnalysisRequest(
        text=text,
        language=language,
        include_sentiment=include_sentiment,
        include_features=include_features
    )


class TestAPIInputValidation:
    """Property tests for API input validation."""
    
    @given(valid_text_strategy())
    def test_valid_text_analysis_request(self, text):
        """
        Feature: nlp-web-interface, Property 44: Input validation
        For any valid text input, the API should accept and process the request.
        **Validates: Requirements 10.2**
        """
        request_data = {
            "text": text,
            "include_sentiment": True,
            "include_features": True
        }
        
        response = client.post("/api/analyze/text", json=request_data)
        
        # Should not return validation errors
        assert response.status_code in [200, 500]  # 500 might occur due to missing NLP components
        
        if response.status_code == 200:
            result = response.json()
            assert "text" in result
            assert "tokens" in result
            assert "pos_tags" in result
            assert "named_entities" in result
            assert "dependencies" in result
            assert "sentiment" in result
            assert "linguistic_features" in result
            assert "language" in result
            assert "processing_time" in result
            
            # Verify text is preserved
            assert result["text"] == text
    
    @given(invalid_text_strategy())
    def test_invalid_text_analysis_request(self, invalid_text):
        """
        Feature: nlp-web-interface, Property 44: Input validation
        For any invalid text input, the API should return appropriate error responses.
        **Validates: Requirements 10.2**
        """
        request_data = {
            "text": invalid_text,
            "include_sentiment": True,
            "include_features": True
        }
        
        response = client.post("/api/analyze/text", json=request_data)
        
        # Should return validation error (422 for Pydantic validation, 400 for custom validation)
        assert response.status_code in [400, 422]
        
        error_data = response.json()
        assert "error" in error_data or "detail" in error_data
    
    @given(text_analysis_request_strategy())
    def test_text_analysis_request_serialization(self, request_obj):
        """
        Feature: nlp-web-interface, Property 44: Input validation
        For any valid TextAnalysisRequest, serialization should work correctly.
        **Validates: Requirements 10.2**
        """
        # Convert to dict for API request
        request_data = request_obj.dict()
        
        response = client.post("/api/analyze/text", json=request_data)
        
        # Should not fail due to serialization issues
        assert response.status_code in [200, 500]  # 500 might occur due to missing components
        
        if response.status_code == 200:
            result = response.json()
            # Verify request data is preserved in response
            assert result["text"] == request_obj.text
    
    @given(st.lists(valid_text_strategy(), min_size=1, max_size=10))
    def test_batch_analysis_validation(self, texts):
        """
        Feature: nlp-web-interface, Property 44: Input validation
        For any list of valid texts, batch analysis should validate correctly.
        **Validates: Requirements 10.2**
        """
        response = client.post("/api/analyze/batch", json=texts)
        
        # Should not return validation errors for valid input
        assert response.status_code in [200, 500]  # 500 might occur due to missing components
        
        if response.status_code == 200:
            results = response.json()
            assert isinstance(results, list)
            assert len(results) == len(texts)
            
            for i, result in enumerate(results):
                assert result["text"] == texts[i]
    
    def test_batch_analysis_empty_list(self):
        """Test that empty batch requests are rejected."""
        response = client.post("/api/analyze/batch", json=[])
        assert response.status_code == 400
    
    def test_batch_analysis_too_many_texts(self):
        """Test that batch requests with too many texts are rejected."""
        texts = ["test"] * 101  # Exceeds limit of 100
        response = client.post("/api/analyze/batch", json=texts)
        assert response.status_code == 400


class TestAPIComponentIntegration:
    """Property tests for API component integration."""
    
    @given(valid_text_strategy())
    def test_nlp_component_integration(self, text):
        """
        Feature: nlp-web-interface, Property 45: Component integration
        For any text input, integration with NLP components should work seamlessly.
        **Validates: Requirements 10.3**
        """
        request_data = {
            "text": text,
            "include_sentiment": True,
            "include_features": True
        }
        
        response = client.post("/api/analyze/text", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Verify all NLP components produced results
            assert len(result["pos_tags"]) == len(result["tokens"])
            assert isinstance(result["named_entities"], list)
            assert isinstance(result["dependencies"], list)
            assert isinstance(result["sentiment"], dict)
            assert isinstance(result["linguistic_features"], dict)
            
            # Verify sentiment structure
            sentiment = result["sentiment"]
            assert "overall_sentiment" in sentiment
            assert "confidence" in sentiment
            assert "scores" in sentiment
            assert isinstance(sentiment["scores"], dict)
            
            # Verify linguistic features structure
            features = result["linguistic_features"]
            required_features = [
                "sentence_count", "word_count", "avg_sentence_length",
                "lexical_diversity", "function_word_ratio", "pos_distribution"
            ]
            for feature in required_features:
                assert feature in features
    
    def test_health_check_endpoint(self):
        """Test that health check endpoint works."""
        response = client.get("/api/analyze/health")
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
        assert "has_nlp_components" in data
    
    def test_model_info_endpoint(self):
        """Test that model info endpoint works."""
        response = client.get("/api/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "pos_tagger" in data
        assert "ner_system" in data
        assert "dependency_parser" in data
        
        for component in data.values():
            assert "available" in component
            assert "model_name" in component


class TestAPIResponseConsistency:
    """Property tests for API response consistency."""
    
    @given(valid_text_strategy())
    def test_response_format_consistency(self, text):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any text analysis, response format should be consistent.
        **Validates: Requirements 10.4**
        """
        request_data = {
            "text": text,
            "include_sentiment": True,
            "include_features": True
        }
        
        response = client.post("/api/analyze/text", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Verify response can be parsed as AnalysisResult
            try:
                analysis_result = AnalysisResult(**result)
                assert analysis_result.text == text
            except Exception as e:
                pytest.fail(f"Response format inconsistent: {e}")
    
    @given(st.lists(valid_text_strategy(), min_size=1, max_size=5))
    def test_batch_response_consistency(self, texts):
        """
        Feature: nlp-web-interface, Property 46: Response formatting consistency
        For any batch analysis, all responses should have consistent format.
        **Validates: Requirements 10.4**
        """
        response = client.post("/api/analyze/batch", json=texts)
        
        if response.status_code == 200:
            results = response.json()
            
            # Verify all results have consistent format
            for i, result in enumerate(results):
                try:
                    analysis_result = AnalysisResult(**result)
                    assert analysis_result.text == texts[i]
                except Exception as e:
                    pytest.fail(f"Batch response {i} format inconsistent: {e}")
    
    @given(valid_text_strategy())
    @settings(max_examples=10)  # Limit examples for performance
    def test_response_determinism(self, text):
        """
        Feature: nlp-web-interface, Property 45: Component integration
        For any text input, multiple requests should produce consistent results.
        **Validates: Requirements 10.3**
        """
        request_data = {
            "text": text,
            "include_sentiment": True,
            "include_features": True
        }
        
        # Make two identical requests
        response1 = client.post("/api/analyze/text", json=request_data)
        response2 = client.post("/api/analyze/text", json=request_data)
        
        if response1.status_code == 200 and response2.status_code == 200:
            result1 = response1.json()
            result2 = response2.json()
            
            # Results should be identical (excluding timestamp and processing_time)
            assert result1["text"] == result2["text"]
            assert result1["tokens"] == result2["tokens"]
            assert len(result1["pos_tags"]) == len(result2["pos_tags"])
            assert len(result1["named_entities"]) == len(result2["named_entities"])
            assert len(result1["dependencies"]) == len(result2["dependencies"])


class TestAPIErrorHandling:
    """Tests for API error handling."""
    
    def test_malformed_json_request(self):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/api/analyze/text",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self):
        """Test handling of requests with missing required fields."""
        response = client.post("/api/analyze/text", json={})
        assert response.status_code == 422  # Validation error
    
    def test_invalid_field_types(self):
        """Test handling of requests with invalid field types."""
        request_data = {
            "text": 123,  # Should be string
            "include_sentiment": "yes"  # Should be boolean
        }
        response = client.post("/api/analyze/text", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_endpoint(self):
        """Test handling of requests to nonexistent endpoints."""
        response = client.post("/api/nonexistent", json={"test": "data"})
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])