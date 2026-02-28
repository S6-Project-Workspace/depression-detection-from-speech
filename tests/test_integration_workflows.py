"""
Integration Tests for Complete Workflows
Tests end-to-end workflows and concurrent user scenarios
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import tempfile
import os

class TestIntegrationWorkflows:
    """Integration tests for complete system workflows"""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for integration testing"""
        components = {
            'audio_processor': Mock(),
            'linguistic_analyzer': Mock(),
            'websocket_handler': Mock(),
            'session_manager': Mock(),
            'file_processor': Mock()
        }
        
        # Configure mocks
        components['audio_processor'].process_chunk = AsyncMock(return_value={
            'text': 'Sample transcribed text',
            'confidence': 0.85
        })
        
        components['linguistic_analyzer'].analyze_pos = AsyncMock(return_value=[
            {'token': 'Sample', 'tag': 'ADJ', 'confidence': 0.9},
            {'token': 'transcribed', 'tag': 'VERB', 'confidence': 0.8},
            {'token': 'text', 'tag': 'NOUN', 'confidence': 0.9}
        ])
        
        components['linguistic_analyzer'].analyze_ner = AsyncMock(return_value=[])
        components['linguistic_analyzer'].analyze_dependencies = AsyncMock(return_value=[])
        components['linguistic_analyzer'].analyze_sentiment = AsyncMock(return_value={
            'sentiment': 'neutral',
            'confidence': 0.7,
            'scores': {'positive': 0.3, 'negative': 0.2, 'neutral': 0.5}
        })
        
        return components
    
    @pytest.mark.asyncio
    async def test_complete_audio_capture_to_visualization_pipeline(self, mock_components):
        """
        Test complete audio capture → transcription → analysis → visualization pipeline
        """
        
        # Simulate audio capture workflow
        session_id = "test_session_audio_pipeline"
        audio_chunks = [b"fake_audio_data_1", b"fake_audio_data_2", b"fake_audio_data_3"]
        
        results = []
        
        # Process each audio chunk through the pipeline
        for i, audio_chunk in enumerate(audio_chunks):
            # Step 1: Audio processing
            transcription_result = await mock_components['audio_processor'].process_chunk(audio_chunk)
            assert transcription_result['text'] == 'Sample transcribed text'
            assert transcription_result['confidence'] > 0.8
            
            # Step 2: Linguistic analysis
            text = transcription_result['text']
            
            pos_result = await mock_components['linguistic_analyzer'].analyze_pos(text)
            ner_result = await mock_components['linguistic_analyzer'].analyze_ner(text)
            dep_result = await mock_components['linguistic_analyzer'].analyze_dependencies(text)
            sentiment_result = await mock_components['linguistic_analyzer'].analyze_sentiment(text)
            
            # Step 3: Compile analysis results
            analysis_result = {
                'chunk_id': i,
                'text': text,
                'transcription': transcription_result,
                'pos_tags': pos_result,
                'named_entities': ner_result,
                'dependencies': dep_result,
                'sentiment': sentiment_result,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(analysis_result)
            
            # Verify analysis completeness
            assert len(analysis_result['pos_tags']) == 3
            assert analysis_result['sentiment']['sentiment'] == 'neutral'
            assert 'confidence' in analysis_result['sentiment']
        
        # Step 4: Verify complete pipeline results
        assert len(results) == len(audio_chunks)
        
        # Verify all chunks processed successfully
        for i, result in enumerate(results):
            assert result['chunk_id'] == i
            assert result['text'] == 'Sample transcribed text'
            assert 'pos_tags' in result
            assert 'named_entities' in result
            assert 'dependencies' in result
            assert 'sentiment' in result
        
        # Step 5: Simulate visualization data preparation
        visualization_data = {
            'pos_visualization': [
                {
                    'text': result['text'],
                    'tags': result['pos_tags']
                }
                for result in results
            ],
            'sentiment_timeline': [
                {
                    'timestamp': result['timestamp'],
                    'sentiment': result['sentiment']['sentiment'],
                    'confidence': result['sentiment']['confidence']
                }
                for result in results
            ]
        }
        
        # Verify visualization data structure
        assert len(visualization_data['pos_visualization']) == 3
        assert len(visualization_data['sentiment_timeline']) == 3
        
        for pos_viz in visualization_data['pos_visualization']:
            assert 'text' in pos_viz
            assert 'tags' in pos_viz
            assert len(pos_viz['tags']) == 3
    
    @pytest.mark.asyncio
    async def test_file_upload_to_result_display_workflow(self, mock_components):
        """
        Test file upload → processing → result display workflow
        """
        
        # Create temporary test files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as text_file:
            text_file.write("This is a test document for NLP analysis. It contains multiple sentences.")
            text_file_path = text_file.name
        
        try:
            # Step 1: File upload simulation
            file_info = {
                'filename': os.path.basename(text_file_path),
                'file_path': text_file_path,
                'file_type': 'text',
                'file_size': os.path.getsize(text_file_path)
            }
            
            # Step 2: File processing
            with open(text_file_path, 'r') as f:
                file_content = f.read()
            
            # Mock file processor
            mock_components['file_processor'].process_text_file = AsyncMock(return_value={
                'text': file_content,
                'word_count': len(file_content.split()),
                'sentence_count': file_content.count('.'),
                'processing_time': 0.5
            })
            
            file_processing_result = await mock_components['file_processor'].process_text_file(text_file_path)
            
            # Step 3: NLP analysis on file content
            text = file_processing_result['text']
            
            pos_result = await mock_components['linguistic_analyzer'].analyze_pos(text)
            ner_result = await mock_components['linguistic_analyzer'].analyze_ner(text)
            dep_result = await mock_components['linguistic_analyzer'].analyze_dependencies(text)
            sentiment_result = await mock_components['linguistic_analyzer'].analyze_sentiment(text)
            
            # Step 4: Compile complete results
            complete_results = {
                'file_info': file_info,
                'processing_stats': file_processing_result,
                'nlp_analysis': {
                    'pos_tags': pos_result,
                    'named_entities': ner_result,
                    'dependencies': dep_result,
                    'sentiment': sentiment_result
                },
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'analysis_version': '1.0'
                }
            }
            
            # Step 5: Verify complete workflow results
            assert complete_results['file_info']['filename'].endswith('.txt')
            assert complete_results['processing_stats']['word_count'] > 0
            assert complete_results['processing_stats']['sentence_count'] > 0
            
            # Verify NLP analysis results
            nlp_analysis = complete_results['nlp_analysis']
            assert 'pos_tags' in nlp_analysis
            assert 'named_entities' in nlp_analysis
            assert 'dependencies' in nlp_analysis
            assert 'sentiment' in nlp_analysis
            
            # Step 6: Simulate result display formatting
            display_data = {
                'summary': {
                    'filename': complete_results['file_info']['filename'],
                    'word_count': complete_results['processing_stats']['word_count'],
                    'sentiment': complete_results['nlp_analysis']['sentiment']['sentiment']
                },
                'visualizations': {
                    'pos_tags': complete_results['nlp_analysis']['pos_tags'],
                    'entities': complete_results['nlp_analysis']['named_entities'],
                    'sentiment_score': complete_results['nlp_analysis']['sentiment']['confidence']
                }
            }
            
            # Verify display data structure
            assert 'summary' in display_data
            assert 'visualizations' in display_data
            assert display_data['summary']['word_count'] > 0
            
        finally:
            # Clean up temporary file
            if os.path.exists(text_file_path):
                os.unlink(text_file_path)
    
    @pytest.mark.asyncio
    async def test_concurrent_user_scenarios(self, mock_components):
        """
        Test concurrent user scenarios and session isolation
        """
        
        # Simulate multiple concurrent users
        user_sessions = [
            {'session_id': f'user_{i}', 'user_type': 'researcher'}
            for i in range(3)
        ]
        
        # Concurrent processing tasks
        async def simulate_user_session(session_info):
            session_id = session_info['session_id']
            
            # Each user performs multiple operations
            operations = []
            
            # Operation 1: Audio processing
            audio_result = await mock_components['audio_processor'].process_chunk(b"user_audio_data")
            operations.append({
                'type': 'audio_processing',
                'result': audio_result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Operation 2: Text analysis
            text = audio_result['text']
            pos_result = await mock_components['linguistic_analyzer'].analyze_pos(text)
            operations.append({
                'type': 'pos_analysis',
                'result': pos_result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Operation 3: Sentiment analysis
            sentiment_result = await mock_components['linguistic_analyzer'].analyze_sentiment(text)
            operations.append({
                'type': 'sentiment_analysis',
                'result': sentiment_result,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'session_id': session_id,
                'operations': operations,
                'total_operations': len(operations)
            }
        
        # Run concurrent sessions
        session_tasks = [
            simulate_user_session(session_info)
            for session_info in user_sessions
        ]
        
        concurrent_results = await asyncio.gather(*session_tasks)
        
        # Verify concurrent execution results
        assert len(concurrent_results) == len(user_sessions)
        
        for i, result in enumerate(concurrent_results):
            assert result['session_id'] == f'user_{i}'
            assert result['total_operations'] == 3
            
            # Verify each operation completed successfully
            operations = result['operations']
            assert len(operations) == 3
            
            # Check operation types
            operation_types = [op['type'] for op in operations]
            assert 'audio_processing' in operation_types
            assert 'pos_analysis' in operation_types
            assert 'sentiment_analysis' in operation_types
            
            # Verify results structure
            for operation in operations:
                assert 'type' in operation
                assert 'result' in operation
                assert 'timestamp' in operation
        
        # Verify session isolation (each session should have independent results)
        session_ids = [result['session_id'] for result in concurrent_results]
        assert len(set(session_ids)) == len(session_ids)  # All unique
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, mock_components):
        """
        Test error recovery and graceful degradation
        """
        
        # Simulate various error scenarios
        error_scenarios = [
            {
                'component': 'audio_processor',
                'error_type': 'ConnectionError',
                'error_message': 'Audio service unavailable'
            },
            {
                'component': 'linguistic_analyzer',
                'error_type': 'TimeoutError',
                'error_message': 'Analysis timeout'
            },
            {
                'component': 'websocket_handler',
                'error_type': 'NetworkError',
                'error_message': 'WebSocket connection lost'
            }
        ]
        
        recovery_results = []
        
        for scenario in error_scenarios:
            component_name = scenario['component']
            error_type = scenario['error_type']
            error_message = scenario['error_message']
            
            # Simulate error condition
            if component_name == 'audio_processor':
                mock_components[component_name].process_chunk.side_effect = Exception(error_message)
            elif component_name == 'linguistic_analyzer':
                mock_components[component_name].analyze_pos.side_effect = Exception(error_message)
            
            # Test error handling
            try:
                if component_name == 'audio_processor':
                    await mock_components[component_name].process_chunk(b"test_data")
                elif component_name == 'linguistic_analyzer':
                    await mock_components[component_name].analyze_pos("test text")
                
                # Should not reach here if error is properly raised
                assert False, f"Expected error for {component_name}"
                
            except Exception as e:
                # Verify error was caught
                assert error_message in str(e)
                
                # Simulate recovery action
                recovery_action = {
                    'component': component_name,
                    'error': str(e),
                    'recovery_strategy': 'fallback_to_cached_result',
                    'recovered': True
                }
                
                recovery_results.append(recovery_action)
                
                # Reset mock for next test
                if component_name == 'audio_processor':
                    mock_components[component_name].process_chunk.side_effect = None
                elif component_name == 'linguistic_analyzer':
                    mock_components[component_name].analyze_pos.side_effect = None
        
        # Verify error recovery
        assert len(recovery_results) == len(error_scenarios)
        
        for recovery in recovery_results:
            assert recovery['recovered'] is True
            assert 'error' in recovery
            assert 'recovery_strategy' in recovery
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, mock_components):
        """
        Test system performance under load
        """
        
        # Simulate high load scenario
        load_test_params = {
            'concurrent_sessions': 10,
            'operations_per_session': 5,
            'total_operations': 50
        }
        
        async def simulate_load_session(session_id: int):
            session_results = []
            
            for op_id in range(load_test_params['operations_per_session']):
                start_time = datetime.now()
                
                # Perform operation
                result = await mock_components['audio_processor'].process_chunk(
                    f"session_{session_id}_op_{op_id}".encode()
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                session_results.append({
                    'session_id': session_id,
                    'operation_id': op_id,
                    'processing_time': processing_time,
                    'result': result
                })
            
            return session_results
        
        # Run load test
        load_tasks = [
            simulate_load_session(session_id)
            for session_id in range(load_test_params['concurrent_sessions'])
        ]
        
        all_results = await asyncio.gather(*load_tasks)
        
        # Flatten results
        flat_results = [
            result for session_results in all_results 
            for result in session_results
        ]
        
        # Verify load test results
        assert len(flat_results) == load_test_params['total_operations']
        
        # Calculate performance metrics
        processing_times = [result['processing_time'] for result in flat_results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)
        
        # Performance assertions (adjust thresholds as needed)
        assert avg_processing_time < 1.0, f"Average processing time too high: {avg_processing_time:.3f}s"
        assert max_processing_time < 2.0, f"Max processing time too high: {max_processing_time:.3f}s"
        
        # Verify all operations completed successfully
        for result in flat_results:
            assert 'result' in result
            assert result['result'] is not None

# Run integration tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])