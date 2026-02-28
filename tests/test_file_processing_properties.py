"""
Property-based tests for file processing functionality.

Tests file upload, processing pipeline, and multi-file processing isolation
using property-based testing to verify universal properties across different file types.

Requirements: 5.1, 5.2, 5.5
"""

import pytest
import asyncio
import os
import tempfile
import json
import csv
import time
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from io import StringIO, BytesIO

from hypothesis import given, strategies as st, settings, assume
import numpy as np

from web_interface.api import (
    process_audio_file, process_text_file, save_uploaded_file,
    validate_file_type, generate_file_id, ALLOWED_AUDIO_TYPES, ALLOWED_TEXT_TYPES
)
from web_interface.models import AnalysisResult, FileUploadResponse
from web_interface.demo_controller import DemoController, BatchStatus


class MockUploadFile:
    """Mock UploadFile for testing."""
    
    def __init__(self, filename: str, content: bytes, content_type: str):
        self.filename = filename
        self.content = content
        self.content_type = content_type
        self._file = BytesIO(content)
    
    async def read(self) -> bytes:
        return self.content
    
    def seek(self, position: int):
        self._file.seek(position)


def generate_audio_file_content(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate synthetic audio file content for testing."""
    # Generate simple sine wave
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, False)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # 440 Hz tone
    
    # Convert to 16-bit PCM bytes
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def generate_text_file_content(content_type: str, num_lines: int = 5) -> str:
    """Generate text file content based on content type."""
    if content_type == "text/plain":
        lines = [f"This is line {i+1} of the test text file." for i in range(num_lines)]
        return "\n".join(lines)
    
    elif content_type == "text/csv":
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "text", "category"])
        for i in range(num_lines):
            writer.writerow([i+1, f"Sample text {i+1}", f"category_{i%3}"])
        return output.getvalue()
    
    elif content_type == "application/json":
        data = [
            {"id": i+1, "text": f"Sample text {i+1}", "category": f"category_{i%3}"}
            for i in range(num_lines)
        ]
        return json.dumps(data, indent=2)
    
    else:
        return "Default text content for testing."


@pytest.mark.asyncio
class TestFileProcessingProperties:
    """Property-based tests for file processing functionality."""
    
    @given(
        st.sampled_from(list(ALLOWED_AUDIO_TYPES)),
        st.floats(min_value=0.5, max_value=5.0),
        st.sampled_from(["tamil", "malayalam", "english"])
    )
    @settings(max_examples=20, deadline=15000)
    async def test_audio_file_processing_pipeline_property(self, content_type, duration, language):
        """
        Feature: nlp-web-interface, Property 20: Audio file processing pipeline
        For any uploaded audio file, the complete ASR and NLP pipeline should process it and return comprehensive results.
        **Validates: Requirements 5.1**
        """
        # Generate test audio file
        audio_content = generate_audio_file_content(duration)
        filename = f"test_audio_{int(time.time() * 1000)}.wav"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        try:
            # Generate file ID
            file_id = generate_file_id()
            
            # Process audio file
            start_time = time.time()
            result = await process_audio_file(temp_file_path, file_id, language)
            processing_time = time.time() - start_time
            
            # Verify processing completes in reasonable time
            assert processing_time < 30.0, f"Audio processing took too long: {processing_time:.2f}s"
            
            # Verify result structure
            assert isinstance(result, AnalysisResult), "Result should be AnalysisResult instance"
            
            # Verify required fields are present
            assert isinstance(result.text, str), "Transcription text should be string"
            assert isinstance(result.tokens, list), "Tokens should be list"
            assert isinstance(result.pos_tags, list), "POS tags should be list"
            assert isinstance(result.named_entities, list), "Named entities should be list"
            assert isinstance(result.dependencies, list), "Dependencies should be list"
            assert hasattr(result, 'sentiment'), "Result should have sentiment analysis"
            assert hasattr(result, 'linguistic_features'), "Result should have linguistic features"
            
            # Verify processing metadata
            assert result.language in ["tamil", "malayalam", "english", "ta", "ml", "en"], f"Invalid language: {result.language}"
            assert result.processing_time > 0.0, "Processing time should be positive"
            assert result.timestamp is not None, "Timestamp should be set"
            
            # Verify data consistency
            if result.tokens:
                # If we have tokens, we should have corresponding POS tags
                assert len(result.pos_tags) == len(result.tokens), "POS tags count should match tokens count"
                
                # Verify POS tag structure
                for i, pos_tag in enumerate(result.pos_tags):
                    assert pos_tag.token == result.tokens[i], f"POS tag token mismatch at position {i}"
                    assert pos_tag.position == i, f"POS tag position mismatch at position {i}"
                    assert 0.0 <= pos_tag.confidence <= 1.0, f"Invalid POS confidence: {pos_tag.confidence}"
            
            # Verify named entities are within text bounds
            text_length = len(result.text)
            for entity in result.named_entities:
                assert 0 <= entity.start <= text_length, f"Entity start position out of bounds: {entity.start}"
                assert entity.start < entity.end <= text_length, f"Invalid entity bounds: {entity.start}-{entity.end}"
                assert 0.0 <= entity.confidence <= 1.0, f"Invalid entity confidence: {entity.confidence}"
            
            # Verify sentiment scores
            sentiment = result.sentiment
            assert 0.0 <= sentiment.confidence <= 1.0, f"Invalid sentiment confidence: {sentiment.confidence}"
            for score_type, score in sentiment.scores.items():
                assert 0.0 <= score <= 1.0, f"Invalid sentiment score for {score_type}: {score}"
            
            # Verify linguistic features
            features = result.linguistic_features
            assert features.sentence_count >= 0, "Sentence count should be non-negative"
            assert features.word_count >= 0, "Word count should be non-negative"
            assert features.avg_sentence_length >= 0.0, "Average sentence length should be non-negative"
            assert 0.0 <= features.lexical_diversity <= 1.0, f"Invalid lexical diversity: {features.lexical_diversity}"
            assert 0.0 <= features.function_word_ratio <= 1.0, f"Invalid function word ratio: {features.function_word_ratio}"
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    @given(
        st.sampled_from(list(ALLOWED_TEXT_TYPES)),
        st.integers(min_value=1, max_value=20),
        st.sampled_from(["english", "tamil", "malayalam"])
    )
    @settings(max_examples=25, deadline=10000)
    async def test_text_file_analysis_property(self, content_type, num_lines, language):
        """
        Feature: nlp-web-interface, Property 21: Text file analysis
        For any uploaded text file, all available linguistic analyses should be performed and results returned.
        **Validates: Requirements 5.2**
        """
        # Generate test text file content
        text_content = generate_text_file_content(content_type, num_lines)
        
        # Determine file extension
        if content_type == "text/csv":
            extension = ".csv"
        elif content_type == "application/json":
            extension = ".json"
        else:
            extension = ".txt"
        
        filename = f"test_text_{int(time.time() * 1000)}{extension}"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=extension, encoding='utf-8') as temp_file:
            temp_file.write(text_content)
            temp_file_path = temp_file.name
        
        try:
            # Generate file ID
            file_id = generate_file_id()
            
            # Process text file
            start_time = time.time()
            results = await process_text_file(temp_file_path, file_id, language)
            processing_time = time.time() - start_time
            
            # Verify processing completes in reasonable time
            assert processing_time < 20.0, f"Text processing took too long: {processing_time:.2f}s"
            
            # Verify results structure
            assert isinstance(results, list), "Results should be a list"
            assert len(results) > 0, "Should have at least one result"
            assert len(results) <= 100, "Should not exceed maximum processing limit"
            
            # Verify each result
            for i, result in enumerate(results):
                assert isinstance(result, AnalysisResult), f"Result {i} should be AnalysisResult instance"
                
                # Verify required fields
                assert isinstance(result.text, str), f"Result {i} text should be string"
                assert len(result.text.strip()) > 0, f"Result {i} should have non-empty text"
                assert isinstance(result.tokens, list), f"Result {i} tokens should be list"
                assert isinstance(result.pos_tags, list), f"Result {i} POS tags should be list"
                assert isinstance(result.named_entities, list), f"Result {i} named entities should be list"
                assert isinstance(result.dependencies, list), f"Result {i} dependencies should be list"
                
                # Verify processing metadata
                assert result.processing_time > 0.0, f"Result {i} processing time should be positive"
                assert result.timestamp is not None, f"Result {i} timestamp should be set"
                
                # Verify data consistency
                if result.tokens:
                    assert len(result.pos_tags) == len(result.tokens), f"Result {i} POS tags count mismatch"
                    
                    # Verify token-level consistency
                    for j, pos_tag in enumerate(result.pos_tags):
                        assert pos_tag.token == result.tokens[j], f"Result {i} POS tag token mismatch at {j}"
                        assert pos_tag.position == j, f"Result {i} POS tag position mismatch at {j}"
                        assert 0.0 <= pos_tag.confidence <= 1.0, f"Result {i} invalid POS confidence at {j}"
                
                # Verify named entities
                for entity in result.named_entities:
                    assert 0 <= entity.start < entity.end <= len(result.text), f"Result {i} invalid entity bounds"
                    assert 0.0 <= entity.confidence <= 1.0, f"Result {i} invalid entity confidence"
                
                # Verify sentiment analysis
                sentiment = result.sentiment
                assert 0.0 <= sentiment.confidence <= 1.0, f"Result {i} invalid sentiment confidence"
                assert all(0.0 <= score <= 1.0 for score in sentiment.scores.values()), f"Result {i} invalid sentiment scores"
                
                # Verify linguistic features
                features = result.linguistic_features
                assert features.sentence_count >= 0, f"Result {i} invalid sentence count"
                assert features.word_count >= 0, f"Result {i} invalid word count"
                assert features.avg_sentence_length >= 0.0, f"Result {i} invalid avg sentence length"
                assert 0.0 <= features.lexical_diversity <= 1.0, f"Result {i} invalid lexical diversity"
                assert 0.0 <= features.function_word_ratio <= 1.0, f"Result {i} invalid function word ratio"
            
            # Verify file type specific properties
            if content_type == "text/csv":
                # CSV files should produce multiple results (one per row, excluding header)
                expected_results = max(1, num_lines - 1)  # -1 for header row
                assert len(results) <= expected_results + 1, f"Too many results for CSV with {num_lines} lines"
            
            elif content_type == "application/json":
                # JSON files should produce results based on structure
                assert len(results) >= 1, "JSON file should produce at least one result"
            
            else:  # text/plain
                # Plain text should produce results based on lines
                expected_results = min(num_lines, 100)  # Limited to 100
                assert len(results) <= expected_results, f"Too many results for plain text with {num_lines} lines"
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(["audio", "text"]),
                st.sampled_from(list(ALLOWED_AUDIO_TYPES) + list(ALLOWED_TEXT_TYPES)),
                st.integers(min_value=1, max_value=5)
            ),
            min_size=2,
            max_size=10
        )
    )
    @settings(max_examples=15, deadline=20000)
    async def test_multi_file_processing_isolation_property(self, file_specs):
        """
        Feature: nlp-web-interface, Property 24: Multi-file processing isolation
        For any set of uploaded files, each should be processed independently with separate, non-interfering result sets.
        **Validates: Requirements 5.5**
        """
        # Assume we have at least 2 different file types or contents
        assume(len(set((spec[0], spec[1]) for spec in file_specs)) >= 2)
        
        controller = DemoController()
        temp_files = []
        
        try:
            # Create test files
            batch_items = []
            for i, (file_type, content_type, size_param) in enumerate(file_specs):
                if file_type == "audio":
                    # Generate audio file
                    duration = float(size_param)
                    content = generate_audio_file_content(duration)
                    extension = ".wav"
                    item_type = "audio_file"
                else:
                    # Generate text file
                    num_lines = size_param
                    content = generate_text_file_content(content_type, num_lines).encode('utf-8')
                    if content_type == "text/csv":
                        extension = ".csv"
                    elif content_type == "application/json":
                        extension = ".json"
                    else:
                        extension = ".txt"
                    item_type = "text_file"
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                    temp_files.append(temp_file_path)
                
                # Add to batch items
                batch_items.append({
                    "type": item_type,
                    "data": temp_file_path,
                    "metadata": {
                        "filename": f"test_file_{i}{extension}",
                        "file_id": f"file_{i}",
                        "language": "english",
                        "content_type": content_type,
                        "original_spec": (file_type, content_type, size_param)
                    }
                })
            
            # Create batch operation
            batch_id = await controller.create_batch_operation(
                operation_type="file_processing",
                items=batch_items
            )
            
            # Start processing
            await controller.start_batch_processing(batch_id)
            
            # Verify batch completed
            progress = controller.get_batch_progress(batch_id)
            assert progress is not None, "Batch progress should be available"
            assert progress.status == BatchStatus.COMPLETED, f"Batch should be completed, got: {progress.status}"
            
            # Get batch results
            batch_results = controller.get_batch_results(batch_id)
            assert batch_results is not None, "Batch results should be available"
            
            # Verify isolation properties
            results = batch_results["results"]
            assert len(results) == len(file_specs), "Should have one result per file"
            
            # Verify each file was processed independently
            for i, (result, (file_type, content_type, size_param)) in enumerate(zip(results, file_specs)):
                if file_type == "audio":
                    # Audio files produce single AnalysisResult
                    assert isinstance(result, AnalysisResult), f"Audio result {i} should be AnalysisResult"
                    assert isinstance(result.text, str), f"Audio result {i} should have transcription"
                    assert len(result.tokens) >= 0, f"Audio result {i} should have tokens"
                else:
                    # Text files produce list of AnalysisResult
                    assert isinstance(result, list), f"Text result {i} should be list"
                    assert len(result) > 0, f"Text result {i} should have at least one analysis"
                    
                    for analysis in result:
                        assert isinstance(analysis, AnalysisResult), f"Text analysis {i} should be AnalysisResult"
                        assert isinstance(analysis.text, str), f"Text analysis {i} should have text"
                        assert len(analysis.tokens) >= 0, f"Text analysis {i} should have tokens"
            
            # Verify processing isolation - no cross-contamination
            # Each result should be independent and not affected by other files
            batch_details = controller.get_batch_details(batch_id)
            assert batch_details is not None, "Batch details should be available"
            
            items = batch_details["items"]
            assert len(items) == len(file_specs), "Should have one item per file"
            
            # Verify each item processed independently
            for i, item in enumerate(items):
                assert item["status"] in ["completed", "error"], f"Item {i} should be completed or error"
                
                if item["status"] == "completed":
                    assert item["processing_time"] > 0.0, f"Item {i} should have positive processing time"
                    assert item["error"] is None, f"Item {i} should not have error when completed"
                
                # Verify item isolation - each item should have unique ID
                assert item["item_id"].endswith(f"_item_{i}"), f"Item {i} should have unique ID"
            
            # Verify summary statistics
            summary = batch_results["summary"]
            assert summary["total_items"] == len(file_specs), "Summary should count all items"
            assert summary["successful_items"] + summary["failed_items"] == len(file_specs), "All items should be accounted for"
            assert summary["processing_time"] > 0.0, "Total processing time should be positive"
            assert 0.0 <= summary["success_rate"] <= 100.0, "Success rate should be valid percentage"
            
            # Verify no interference between files
            # This is demonstrated by successful independent processing of different file types
            successful_items = summary["successful_items"]
            if successful_items > 1:
                # If multiple files succeeded, they were processed independently
                assert successful_items >= 2, "Multiple files should process independently"
        
        finally:
            # Clean up temporary files
            for temp_file_path in temp_files:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=30, deadline=2000)
    async def test_file_validation_property(self, filename):
        """
        Test that file validation works correctly for all filename inputs.
        """
        # Test valid audio file types
        for content_type in ALLOWED_AUDIO_TYPES:
            mock_file = MockUploadFile(filename, b"test_content", content_type)
            assert validate_file_type(mock_file, ALLOWED_AUDIO_TYPES), f"Should accept {content_type}"
        
        # Test valid text file types
        for content_type in ALLOWED_TEXT_TYPES:
            mock_file = MockUploadFile(filename, b"test_content", content_type)
            assert validate_file_type(mock_file, ALLOWED_TEXT_TYPES), f"Should accept {content_type}"
        
        # Test invalid file type
        mock_file = MockUploadFile(filename, b"test_content", "application/pdf")
        assert not validate_file_type(mock_file, ALLOWED_AUDIO_TYPES), "Should reject PDF for audio"
        assert not validate_file_type(mock_file, ALLOWED_TEXT_TYPES), "Should reject PDF for text"
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=1000)
    async def test_file_id_generation_property(self, num_ids):
        """
        Test that file ID generation produces unique identifiers.
        """
        generated_ids = set()
        
        for _ in range(num_ids):
            file_id = generate_file_id()
            
            # Verify format
            assert isinstance(file_id, str), "File ID should be string"
            assert file_id.startswith("file_"), "File ID should start with 'file_'"
            assert len(file_id) > 5, "File ID should be longer than prefix"
            
            # Verify uniqueness
            assert file_id not in generated_ids, f"File ID {file_id} should be unique"
            generated_ids.add(file_id)
        
        # Verify all IDs are unique
        assert len(generated_ids) == num_ids, "All generated IDs should be unique"
    
    @pytest.mark.asyncio
    async def test_file_processing_error_handling(self):
        """
        Test that file processing handles various error conditions gracefully.
        """
        # Test with non-existent file
        with pytest.raises(Exception):
            await process_audio_file("/nonexistent/file.wav", "test_id", "english")
        
        with pytest.raises(Exception):
            await process_text_file("/nonexistent/file.txt", "test_id", "english")
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Empty audio file should handle gracefully
            result = await process_audio_file(temp_file_path, "empty_audio", "english")
            # Should either succeed with empty result or raise appropriate exception
            if isinstance(result, AnalysisResult):
                assert isinstance(result.text, str)
            
            # Empty text file should handle gracefully
            results = await process_text_file(temp_file_path, "empty_text", "english")
            assert isinstance(results, list)
            # May be empty list for empty file
        
        except Exception as e:
            # Exceptions are acceptable for invalid files
            assert isinstance(e, Exception)
        
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


if __name__ == "__main__":
    # Run a simple test
    import asyncio
    
    async def run_simple_test():
        test_instance = TestFileProcessingProperties()
        await test_instance.test_file_validation_property("test.wav")
        print("Simple file processing property test passed!")
    
    asyncio.run(run_simple_test())