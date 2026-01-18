"""
Tests for Stream Processor

Tests the real-time stream processing, inference integration, and adaptive
processing functionality of the streaming system.

Reference: Real-time Streaming Analysis - Task 1.3
"""

import asyncio
import pytest
import numpy as np
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from streaming.stream_processor import (
    StreamProcessor,
    StreamingInferenceAdapter,
    AdaptiveProcessor,
    CircuitBreaker
)
from streaming.models import (
    RiskDataPoint, ProcessingMode, AudioQualityMetric
)


class TestStreamingInferenceAdapter:
    """Test StreamingInferenceAdapter class."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = StreamingInferenceAdapter(
            device='cpu',
            inference_mode='enhanced'
        )
        
        assert adapter.device == 'cpu'
        assert adapter.inference_mode == 'enhanced'
        assert adapter.total_inferences == 0
        assert adapter.inference_errors == 0
    
    @pytest.mark.asyncio
    async def test_mock_audio_processing(self):
        """Test audio processing with mock inference."""
        adapter = StreamingInferenceAdapter(device='cpu')
        
        # Create test audio data
        audio_data = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        
        # Process audio chunk
        result = await adapter.process_audio_chunk_async(
            audio_data=audio_data,
            transcript_text="Test transcript",
            language="ta",
            chunk_id="test_chunk_1"
        )
        
        # Verify result
        assert isinstance(result, RiskDataPoint)
        assert result.chunk_id == "test_chunk_1"
        assert 0.0 <= result.risk_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time_ms > 0
        assert result.processing_mode in [
            ProcessingMode.TRIMODAL, ProcessingMode.BIMODAL, 
            ProcessingMode.AUDIO_ONLY, ProcessingMode.BASIC
        ]
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in audio processing."""
        adapter = StreamingInferenceAdapter(device='cpu')
        
        # Mock the sync processing to raise an error
        with patch.object(adapter, '_process_audio_chunk_sync', side_effect=Exception("Test error")):
            result = await adapter.process_audio_chunk_async(
                audio_data=np.array([1.0, 2.0, 3.0]),
                chunk_id="error_chunk"
            )
            
            # Should return error result
            assert result.risk_score == 0.0
            assert result.confidence == 0.0
            assert result.chunk_id == "error_chunk"
            assert adapter.inference_errors == 1
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        adapter = StreamingInferenceAdapter(device='cpu')
        
        # Simulate some processing
        adapter.total_inferences = 10
        adapter.total_inference_time = 5000.0  # 5 seconds total
        adapter.inference_errors = 2
        
        metrics = adapter.get_performance_metrics()
        
        assert metrics['total_inferences'] == 10
        assert metrics['total_inference_time_ms'] == 5000.0
        assert metrics['average_inference_time_ms'] == 500.0
        assert metrics['inference_errors'] == 2
        assert metrics['error_rate'] == 0.2
    
    def test_mock_result_generation(self):
        """Test mock result generation."""
        adapter = StreamingInferenceAdapter(device='cpu')
        
        timestamp = datetime.now()
        result = adapter._create_mock_result(timestamp, "mock_chunk")
        
        assert isinstance(result, RiskDataPoint)
        assert result.timestamp == timestamp
        assert result.chunk_id == "mock_chunk"
        assert 0.0 <= result.risk_score <= 1.0
        assert result.linguistic_markers is not None
        assert 'linguistic_risk_score' in result.linguistic_markers


class TestAdaptiveProcessor:
    """Test AdaptiveProcessor class."""
    
    def test_adaptive_processor_initialization(self):
        """Test adaptive processor initialization."""
        processor = AdaptiveProcessor(
            target_latency_ms=1500.0,
            performance_window_size=5
        )
        
        assert processor.target_latency_ms == 1500.0
        assert processor.performance_window_size == 5
        assert processor.current_mode == ProcessingMode.TRIMODAL
        assert len(processor.recent_latencies) == 0
    
    def test_performance_tracking(self):
        """Test performance tracking and mode adaptation."""
        processor = AdaptiveProcessor(target_latency_ms=1000.0)
        
        # Add some good performance measurements
        for latency in [800, 850, 900]:
            mode = processor.update_performance(latency)
            
        assert processor.current_mode == ProcessingMode.TRIMODAL
        assert len(processor.recent_latencies) == 3
    
    def test_mode_downgrade(self):
        """Test processing mode downgrade under poor performance."""
        processor = AdaptiveProcessor(target_latency_ms=1000.0)
        
        # Add poor performance measurements (> 1.5 * target)
        for latency in [1600, 1700, 1800]:
            mode = processor.update_performance(latency)
        
        # Should downgrade from TRIMODAL
        assert processor.current_mode != ProcessingMode.TRIMODAL
    
    def test_mode_upgrade(self):
        """Test processing mode upgrade under good performance."""
        processor = AdaptiveProcessor(target_latency_ms=1000.0)
        
        # Start with downgraded mode
        processor.current_mode = ProcessingMode.AUDIO_ONLY
        
        # Add good performance measurements (< 0.8 * target)
        for latency in [600, 650, 700]:
            mode = processor.update_performance(latency)
        
        # Should upgrade towards better quality
        assert processor.current_mode in [ProcessingMode.BIMODAL, ProcessingMode.TRIMODAL]
    
    def test_mode_transitions(self):
        """Test all mode transitions."""
        processor = AdaptiveProcessor()
        
        # Test downgrade sequence
        assert processor._downgrade_mode(ProcessingMode.TRIMODAL) == ProcessingMode.BIMODAL
        assert processor._downgrade_mode(ProcessingMode.BIMODAL) == ProcessingMode.AUDIO_ONLY
        assert processor._downgrade_mode(ProcessingMode.AUDIO_ONLY) == ProcessingMode.FAST
        assert processor._downgrade_mode(ProcessingMode.FAST) == ProcessingMode.BASIC
        
        # Test upgrade sequence
        assert processor._upgrade_mode(ProcessingMode.BASIC) == ProcessingMode.FAST
        assert processor._upgrade_mode(ProcessingMode.FAST) == ProcessingMode.AUDIO_ONLY
        assert processor._upgrade_mode(ProcessingMode.AUDIO_ONLY) == ProcessingMode.BIMODAL
        assert processor._upgrade_mode(ProcessingMode.BIMODAL) == ProcessingMode.TRIMODAL
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        processor = AdaptiveProcessor(target_latency_ms=1000.0)
        
        # Add some measurements
        processor.update_performance(800)
        processor.update_performance(900)
        processor.update_performance(1200)
        
        summary = processor.get_performance_summary()
        
        assert 'current_mode' in summary
        assert 'average_latency_ms' in summary
        assert 'target_latency_ms' in summary
        assert 'performance_ratio' in summary
        assert summary['target_latency_ms'] == 1000.0


class TestStreamProcessor:
    """Test StreamProcessor class."""
    
    def test_processor_initialization(self):
        """Test stream processor initialization."""
        processor = StreamProcessor(
            device='cpu',
            inference_mode='enhanced',
            target_latency_ms=1500.0,
            enable_adaptive_processing=True
        )
        
        assert processor.device == 'cpu'
        assert processor.inference_mode == 'enhanced'
        assert processor.target_latency_ms == 1500.0
        assert processor.enable_adaptive_processing is True
        assert processor.inference_adapter is not None
        assert processor.adaptive_processor is not None
    
    def test_processor_without_adaptive(self):
        """Test processor without adaptive processing."""
        processor = StreamProcessor(
            enable_adaptive_processing=False
        )
        
        assert processor.adaptive_processor is None
    
    @pytest.mark.asyncio
    async def test_audio_chunk_processing(self):
        """Test audio chunk processing."""
        processor = StreamProcessor(device='cpu')
        
        # Create test data
        audio_data = np.random.randn(16000).astype(np.float32)
        quality_metric = AudioQualityMetric(
            timestamp=datetime.now(),
            signal_to_noise_ratio=20.0,
            volume_level=0.5,
            clipping_detected=False,
            silence_ratio=0.1,
            quality_score=0.8
        )
        
        # Process chunk
        result = await processor.process_audio_chunk(
            audio_data=audio_data,
            audio_quality=quality_metric,
            transcript_text="Test transcript",
            language="ta",
            chunk_id="test_chunk"
        )
        
        # Verify result
        assert isinstance(result, RiskDataPoint)
        assert result.chunk_id == "test_chunk"
        assert processor.total_chunks_processed == 1
        assert processor.processing_errors == 0
    
    @pytest.mark.asyncio
    async def test_callback_system(self):
        """Test result callback system."""
        processor = StreamProcessor(device='cpu')
        
        # Add callback
        received_results = []
        
        def test_callback(result: RiskDataPoint):
            received_results.append(result)
        
        processor.add_result_callback(test_callback)
        
        # Process chunk
        audio_data = np.random.randn(1000).astype(np.float32)
        result = await processor.process_audio_chunk(audio_data)
        
        # Verify callback was called
        assert len(received_results) == 1
        assert received_results[0].chunk_id == result.chunk_id
        
        # Remove callback
        processor.remove_result_callback(test_callback)
        
        # Process another chunk
        await processor.process_audio_chunk(audio_data)
        
        # Should still be only 1 result (callback removed)
        assert len(received_results) == 1
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self):
        """Test error handling during processing."""
        processor = StreamProcessor(device='cpu')
        
        # Mock the inference adapter to raise an error
        with patch.object(
            processor.inference_adapter, 
            'process_audio_chunk_async', 
            side_effect=Exception("Processing error")
        ):
            result = await processor.process_audio_chunk(
                audio_data=np.array([1.0, 2.0, 3.0]),
                chunk_id="error_test"
            )
            
            # Should return error result
            assert result.risk_score == 0.0
            assert result.confidence == 0.0
            assert processor.processing_errors == 1
    
    def test_statistics_collection(self):
        """Test statistics collection."""
        processor = StreamProcessor(device='cpu')
        
        # Simulate some processing
        processor.total_chunks_processed = 5
        processor.processing_errors = 1
        
        stats = processor.get_processing_statistics()
        
        assert stats['total_chunks_processed'] == 5
        assert stats['processing_errors'] == 1
        assert stats['error_rate'] == 0.2
        assert 'inference_metrics' in stats
        
        if processor.adaptive_processor:
            assert 'adaptive_processing' in stats
    
    def test_statistics_reset(self):
        """Test statistics reset."""
        processor = StreamProcessor(device='cpu')
        
        # Set some statistics
        processor.total_chunks_processed = 10
        processor.processing_errors = 2
        processor.inference_adapter.total_inferences = 8
        
        # Reset
        processor.reset_statistics()
        
        # Verify reset
        assert processor.total_chunks_processed == 0
        assert processor.processing_errors == 0
        assert processor.inference_adapter.total_inferences == 0


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_seconds=30,
            success_threshold=2
        )
        
        assert breaker.failure_threshold == 3
        assert breaker.timeout_seconds == 30
        assert breaker.success_threshold == 2
        assert breaker.state == 'closed'
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """Test successful operation through circuit breaker."""
        breaker = CircuitBreaker()
        
        async def successful_func():
            return "success"
        
        result = await breaker.call(successful_func)
        
        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == 'closed'
    
    @pytest.mark.asyncio
    async def test_circuit_opening(self):
        """Test circuit breaker opening after failures."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        async def failing_func():
            raise Exception("Test failure")
        
        # Cause failures to open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        
        assert breaker.state == 'open'
        assert breaker.failure_count == 3
        
        # Next call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_recovery(self):
        """Test circuit breaker recovery."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            timeout_seconds=1,  # Short timeout for testing
            success_threshold=2
        )
        
        async def failing_func():
            raise Exception("Test failure")
        
        async def successful_func():
            return "success"
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        
        assert breaker.state == 'open'
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        
        # Should move to half-open and allow successful calls
        result = await breaker.call(successful_func)
        assert result == "success"
        assert breaker.state == 'half_open'
        
        # Another success should close the circuit
        result = await breaker.call(successful_func)
        assert result == "success"
        assert breaker.state == 'closed'
    
    def test_circuit_breaker_state(self):
        """Test circuit breaker state reporting."""
        breaker = CircuitBreaker()
        
        state = breaker.get_state()
        
        assert 'state' in state
        assert 'failure_count' in state
        assert 'success_count' in state
        assert 'time_until_retry' in state
        
        assert state['state'] == 'closed'
        assert state['failure_count'] == 0


# Property-based tests using hypothesis
try:
    from hypothesis import given, strategies as st, settings
    
    class TestStreamProcessorProperties:
        """Property-based tests for stream processor."""
        
        @given(
            target_latency=st.floats(min_value=100.0, max_value=5000.0),
            window_size=st.integers(min_value=3, max_value=20)
        )
        @settings(deadline=None, max_examples=10)
        def test_adaptive_processor_properties(self, target_latency, window_size):
            """Test adaptive processor properties."""
            processor = AdaptiveProcessor(
                target_latency_ms=target_latency,
                performance_window_size=window_size
            )
            
            # Properties
            assert processor.target_latency_ms == target_latency
            assert processor.performance_window_size == window_size
            assert processor.current_mode == ProcessingMode.TRIMODAL
            
            # Add some measurements
            for latency in np.random.uniform(target_latency * 0.5, target_latency * 2.0, 5):
                mode = processor.update_performance(latency)
                assert isinstance(mode, ProcessingMode)
            
            # Performance summary should be valid
            summary = processor.get_performance_summary()
            assert summary['target_latency_ms'] == target_latency
            assert summary['performance_ratio'] >= 0.0
        
        @given(
            failure_threshold=st.integers(min_value=1, max_value=10),
            timeout_seconds=st.integers(min_value=1, max_value=60)
        )
        @settings(deadline=None, max_examples=10)
        def test_circuit_breaker_properties(self, failure_threshold, timeout_seconds):
            """Test circuit breaker properties."""
            breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                timeout_seconds=timeout_seconds
            )
            
            # Properties
            assert breaker.failure_threshold == failure_threshold
            assert breaker.timeout_seconds == timeout_seconds
            assert breaker.state == 'closed'
            
            # Simulate failures
            for _ in range(failure_threshold):
                breaker._on_failure()
            
            # Should be open after threshold failures
            assert breaker.state == 'open'
            assert breaker.failure_count == failure_threshold
        
        @pytest.mark.asyncio
        @given(
            audio_length=st.integers(min_value=1000, max_value=48000)
        )
        @settings(deadline=None, max_examples=5)
        async def test_audio_processing_properties(self, audio_length):
            """Test audio processing properties."""
            processor = StreamProcessor(device='cpu')
            
            # Generate random audio
            audio_data = np.random.randn(audio_length).astype(np.float32)
            
            # Process audio
            result = await processor.process_audio_chunk(audio_data)
            
            # Properties
            assert isinstance(result, RiskDataPoint)
            assert 0.0 <= result.risk_score <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time_ms >= 0.0
            assert isinstance(result.processing_mode, ProcessingMode)

except ImportError:
    # Hypothesis not available
    pass


# Integration tests
class TestStreamProcessorIntegration:
    """Integration tests for stream processor components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test end-to-end audio processing pipeline."""
        processor = StreamProcessor(
            device='cpu',
            enable_adaptive_processing=True
        )
        
        # Process multiple chunks to test adaptive behavior
        results = []
        
        for i in range(5):
            audio_data = np.random.randn(16000).astype(np.float32)
            
            result = await processor.process_audio_chunk(
                audio_data=audio_data,
                chunk_id=f"integration_chunk_{i}"
            )
            
            results.append(result)
        
        # Verify all chunks processed
        assert len(results) == 5
        assert processor.total_chunks_processed == 5
        
        # All results should be valid
        for result in results:
            assert isinstance(result, RiskDataPoint)
            assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_processing_integration(self):
        """Test integration between stream processor and adaptive processing."""
        processor = StreamProcessor(
            device='cpu',
            target_latency_ms=1000.0,
            enable_adaptive_processing=True
        )
        
        # Simulate varying processing times
        processing_times = [800, 1200, 1500, 900, 1100]
        
        for i, expected_time in enumerate(processing_times):
            # Mock processing time
            with patch.object(
                processor.inference_adapter,
                'process_audio_chunk_async',
                return_value=RiskDataPoint(
                    timestamp=datetime.now(),
                    risk_score=0.5,
                    confidence=0.8,
                    audio_contribution=0.7,
                    chunk_id=f"adaptive_chunk_{i}",
                    processing_mode=ProcessingMode.TRIMODAL,
                    processing_time_ms=expected_time
                )
            ):
                result = await processor.process_audio_chunk(
                    audio_data=np.random.randn(1000).astype(np.float32)
                )
                
                assert result.processing_time_ms == expected_time
        
        # Check adaptive processor state
        if processor.adaptive_processor:
            summary = processor.adaptive_processor.get_performance_summary()
            assert len(summary['recent_latencies']) > 0


if __name__ == "__main__":
    pytest.main([__file__])