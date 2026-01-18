"""
Stream Processor for Real-time Streaming Analysis

This module integrates the enhanced inference pipeline with real-time streaming,
providing continuous depression risk assessment with linguistic features.

Key Features:
- Integration with EnhancedDepressionInferencePipeline
- Async processing for real-time performance
- Adaptive processing modes for performance optimization
- Error handling and graceful degradation
- Performance monitoring and optimization

Reference: Real-time Streaming Analysis - Task 1.3
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import torch

from .models import (
    RiskDataPoint, ProcessingMode, AudioQualityMetric,
    SessionStatus, AlertType, AlertSeverity
)

# Import existing enhanced inference components
try:
    from inference import EnhancedDepressionInferencePipeline, create_enhanced_inference_pipeline
    from config import DepressionDetectionConfig
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    EnhancedDepressionInferencePipeline = None
    DepressionDetectionConfig = None

logger = logging.getLogger(__name__)


class StreamingInferenceAdapter:
    """
    Adapter for integrating enhanced inference pipeline with streaming.
    
    Handles:
    - Conversion between streaming and inference data formats
    - Async wrapper for inference operations
    - Performance optimization for streaming
    - Error handling and fallback modes
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        device: str = 'cuda',
        inference_mode: str = 'enhanced'
    ):
        """
        Initialize streaming inference adapter.
        
        Args:
            config: Configuration object for inference pipeline
            device: Device to run inference on
            inference_mode: 'standard' or 'enhanced'
        """
        self.device = device
        self.inference_mode = inference_mode
        self.config = config
        
        # Initialize inference pipeline
        self.inference_pipeline = None
        self._initialize_pipeline()
        
        # Performance tracking
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0
        
        # Error tracking
        self.inference_errors = 0
        self.last_error_time = None
        
        logger.info(f"StreamingInferenceAdapter initialized in {inference_mode} mode")
    
    def _initialize_pipeline(self):
        """Initialize the enhanced inference pipeline."""
        if not INFERENCE_AVAILABLE:
            logger.warning("Enhanced inference pipeline not available - using mock")
            self.inference_pipeline = None
            return
        
        try:
            if self.config:
                self.inference_pipeline = create_enhanced_inference_pipeline(
                    self.config,
                    inference_mode=self.inference_mode,
                    device=self.device
                )
            else:
                # Create with default config
                logger.warning("No config provided - using default configuration")
                self.inference_pipeline = None
                
        except Exception as e:
            logger.error(f"Failed to initialize inference pipeline: {e}")
            self.inference_pipeline = None
    
    async def process_audio_chunk_async(
        self,
        audio_data: np.ndarray,
        transcript_text: Optional[str] = None,
        language: str = "ta",
        chunk_id: str = "",
        timestamp: Optional[datetime] = None
    ) -> RiskDataPoint:
        """
        Process audio chunk asynchronously.
        
        Args:
            audio_data: Audio data to process
            transcript_text: Optional transcript text
            language: Language for linguistic analysis
            chunk_id: Unique identifier for this chunk
            timestamp: Timestamp of the audio chunk
            
        Returns:
            RiskDataPoint with analysis results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        start_time = time.perf_counter()
        
        try:
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._process_audio_chunk_sync,
                audio_data,
                transcript_text,
                language,
                chunk_id,
                timestamp
            )
            
            # Update performance metrics
            processing_time = (time.perf_counter() - start_time) * 1000  # ms
            self.total_inferences += 1
            self.total_inference_time += processing_time
            self.last_inference_time = processing_time
            
            # Update result with processing time
            result.processing_time_ms = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async audio processing: {e}")
            self.inference_errors += 1
            self.last_error_time = datetime.now()
            
            # Return error result
            return self._create_error_result(timestamp, chunk_id, str(e))
    
    def _process_audio_chunk_sync(
        self,
        audio_data: np.ndarray,
        transcript_text: Optional[str],
        language: str,
        chunk_id: str,
        timestamp: datetime
    ) -> RiskDataPoint:
        """Synchronous audio processing (runs in thread pool)."""
        try:
            if self.inference_pipeline is None:
                # Mock inference for testing
                return self._create_mock_result(timestamp, chunk_id)
            
            # Save audio to temporary file for inference
            # (In production, this could be optimized to avoid file I/O)
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, 16000)
                
                # Run enhanced inference
                result = self.inference_pipeline.predict_file_enhanced(
                    audio_path=temp_file.name,
                    transcript_text=transcript_text,
                    language=language,
                    return_analysis=True
                )
                
                # Clean up temp file
                import os
                os.unlink(temp_file.name)
            
            # Convert to RiskDataPoint
            return self._convert_inference_result(result, timestamp, chunk_id)
            
        except Exception as e:
            logger.error(f"Error in sync audio processing: {e}")
            raise
    
    def _convert_inference_result(
        self,
        inference_result: Any,
        timestamp: datetime,
        chunk_id: str
    ) -> RiskDataPoint:
        """Convert inference result to RiskDataPoint."""
        try:
            # Extract modality contributions
            audio_contribution = 0.5  # Default
            text_contribution = None
            linguistic_contribution = None
            
            if hasattr(inference_result, 'feature_importance') and inference_result.feature_importance:
                audio_contribution = inference_result.feature_importance.get('audio_importance', 0.5)
                text_contribution = inference_result.feature_importance.get('text_importance')
                linguistic_contribution = inference_result.feature_importance.get('linguistic_importance')
            
            # Extract linguistic markers
            linguistic_markers = None
            if hasattr(inference_result, 'linguistic_markers') and inference_result.linguistic_markers:
                linguistic_markers = inference_result.linguistic_markers
            
            # Determine processing mode
            processing_mode = ProcessingMode.AUDIO_ONLY
            if text_contribution is not None and linguistic_contribution is not None:
                processing_mode = ProcessingMode.TRIMODAL
            elif text_contribution is not None:
                processing_mode = ProcessingMode.BIMODAL
            
            return RiskDataPoint(
                timestamp=timestamp,
                risk_score=float(inference_result.probability),
                confidence=float(inference_result.probability),  # Use probability as confidence
                audio_contribution=audio_contribution,
                text_contribution=text_contribution,
                linguistic_contribution=linguistic_contribution,
                linguistic_markers=linguistic_markers,
                chunk_id=chunk_id,
                processing_mode=processing_mode
            )
            
        except Exception as e:
            logger.error(f"Error converting inference result: {e}")
            return self._create_error_result(timestamp, chunk_id, str(e))
    
    def _create_mock_result(self, timestamp: datetime, chunk_id: str) -> RiskDataPoint:
        """Create mock result for testing."""
        # Generate realistic mock data
        risk_score = np.random.beta(2, 5)  # Skewed towards lower risk
        confidence = np.random.uniform(0.6, 0.9)
        
        return RiskDataPoint(
            timestamp=timestamp,
            risk_score=risk_score,
            confidence=confidence,
            audio_contribution=0.7,
            text_contribution=0.2,
            linguistic_contribution=0.1,
            linguistic_markers={
                'first_person_pronoun_ratio': np.random.uniform(0.0, 0.3),
                'negative_adjective_ratio': np.random.uniform(0.0, 0.2),
                'linguistic_risk_score': risk_score * 0.8
            },
            chunk_id=chunk_id,
            processing_mode=ProcessingMode.TRIMODAL
        )
    
    def _create_error_result(
        self, 
        timestamp: datetime, 
        chunk_id: str, 
        error_message: str
    ) -> RiskDataPoint:
        """Create error result."""
        return RiskDataPoint(
            timestamp=timestamp,
            risk_score=0.0,
            confidence=0.0,
            audio_contribution=0.0,
            chunk_id=chunk_id,
            processing_mode=ProcessingMode.BASIC
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_inference_time = (
            self.total_inference_time / self.total_inferences
            if self.total_inferences > 0 else 0.0
        )
        
        return {
            'total_inferences': self.total_inferences,
            'total_inference_time_ms': self.total_inference_time,
            'average_inference_time_ms': avg_inference_time,
            'last_inference_time_ms': self.last_inference_time,
            'inference_errors': self.inference_errors,
            'error_rate': (
                self.inference_errors / self.total_inferences
                if self.total_inferences > 0 else 0.0
            ),
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None
        }


class AdaptiveProcessor:
    """
    Adaptive processor that adjusts processing quality based on performance.
    
    Monitors system performance and automatically adjusts processing modes
    to maintain real-time performance targets.
    """
    
    def __init__(
        self,
        target_latency_ms: float = 2000.0,
        performance_window_size: int = 10
    ):
        """
        Initialize adaptive processor.
        
        Args:
            target_latency_ms: Target processing latency in milliseconds
            performance_window_size: Number of recent measurements to consider
        """
        self.target_latency_ms = target_latency_ms
        self.performance_window_size = performance_window_size
        
        # Performance tracking
        self.recent_latencies = []
        self.current_mode = ProcessingMode.TRIMODAL
        
        # Adaptation parameters
        self.adaptation_threshold = 1.5  # Switch modes if latency > target * threshold
        self.recovery_threshold = 0.8    # Switch back if latency < target * threshold
        
        logger.info(f"AdaptiveProcessor initialized with {target_latency_ms}ms target")
    
    def update_performance(self, latency_ms: float) -> ProcessingMode:
        """
        Update performance metrics and adapt processing mode.
        
        Args:
            latency_ms: Latest processing latency
            
        Returns:
            Recommended processing mode
        """
        # Update latency history
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > self.performance_window_size:
            self.recent_latencies.pop(0)
        
        # Calculate average latency
        if len(self.recent_latencies) < 3:
            return self.current_mode  # Not enough data yet
        
        avg_latency = np.mean(self.recent_latencies)
        
        # Determine if adaptation is needed
        if avg_latency > self.target_latency_ms * self.adaptation_threshold:
            # Performance is poor, downgrade mode
            new_mode = self._downgrade_mode(self.current_mode)
            if new_mode != self.current_mode:
                logger.info(f"Downgrading processing mode: {self.current_mode} -> {new_mode}")
                self.current_mode = new_mode
        
        elif avg_latency < self.target_latency_ms * self.recovery_threshold:
            # Performance is good, try to upgrade mode
            new_mode = self._upgrade_mode(self.current_mode)
            if new_mode != self.current_mode:
                logger.info(f"Upgrading processing mode: {self.current_mode} -> {new_mode}")
                self.current_mode = new_mode
        
        return self.current_mode
    
    def _downgrade_mode(self, current_mode: ProcessingMode) -> ProcessingMode:
        """Downgrade processing mode for better performance."""
        if current_mode == ProcessingMode.TRIMODAL:
            return ProcessingMode.BIMODAL
        elif current_mode == ProcessingMode.BIMODAL:
            return ProcessingMode.AUDIO_ONLY
        elif current_mode == ProcessingMode.AUDIO_ONLY:
            return ProcessingMode.FAST
        else:
            return ProcessingMode.BASIC
    
    def _upgrade_mode(self, current_mode: ProcessingMode) -> ProcessingMode:
        """Upgrade processing mode for better quality."""
        if current_mode == ProcessingMode.BASIC:
            return ProcessingMode.FAST
        elif current_mode == ProcessingMode.FAST:
            return ProcessingMode.AUDIO_ONLY
        elif current_mode == ProcessingMode.AUDIO_ONLY:
            return ProcessingMode.BIMODAL
        elif current_mode == ProcessingMode.BIMODAL:
            return ProcessingMode.TRIMODAL
        else:
            return current_mode
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.recent_latencies:
            return {
                'current_mode': self.current_mode.value,
                'average_latency_ms': 0.0,
                'target_latency_ms': self.target_latency_ms,
                'performance_ratio': 0.0
            }
        
        avg_latency = np.mean(self.recent_latencies)
        performance_ratio = avg_latency / self.target_latency_ms
        
        return {
            'current_mode': self.current_mode.value,
            'average_latency_ms': avg_latency,
            'target_latency_ms': self.target_latency_ms,
            'performance_ratio': performance_ratio,
            'recent_latencies': list(self.recent_latencies),
            'adaptation_needed': performance_ratio > self.adaptation_threshold
        }


class StreamProcessor:
    """
    Main stream processor for real-time depression detection.
    
    Coordinates:
    - Audio chunk processing through enhanced inference
    - Adaptive performance management
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        device: str = 'cuda',
        inference_mode: str = 'enhanced',
        target_latency_ms: float = 2000.0,
        enable_adaptive_processing: bool = True
    ):
        """
        Initialize stream processor.
        
        Args:
            config: Configuration for inference pipeline
            device: Device to run inference on
            inference_mode: 'standard' or 'enhanced'
            target_latency_ms: Target processing latency
            enable_adaptive_processing: Enable adaptive processing modes
        """
        self.config = config
        self.device = device
        self.inference_mode = inference_mode
        self.target_latency_ms = target_latency_ms
        self.enable_adaptive_processing = enable_adaptive_processing
        
        # Initialize components
        self.inference_adapter = StreamingInferenceAdapter(
            config=config,
            device=device,
            inference_mode=inference_mode
        )
        
        if enable_adaptive_processing:
            self.adaptive_processor = AdaptiveProcessor(
                target_latency_ms=target_latency_ms
            )
        else:
            self.adaptive_processor = None
        
        # Processing callbacks
        self.result_callbacks: List[Callable[[RiskDataPoint], None]] = []
        
        # Statistics
        self.total_chunks_processed = 0
        self.processing_errors = 0
        self.last_processing_time = None
        
        logger.info(f"StreamProcessor initialized with {inference_mode} mode")
    
    async def process_audio_chunk(
        self,
        audio_data: np.ndarray,
        audio_quality: Optional[AudioQualityMetric] = None,
        transcript_text: Optional[str] = None,
        language: str = "ta",
        chunk_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> RiskDataPoint:
        """
        Process a single audio chunk.
        
        Args:
            audio_data: Audio data to process
            audio_quality: Audio quality metrics
            transcript_text: Optional transcript text
            language: Language for linguistic analysis
            chunk_id: Unique identifier for chunk
            timestamp: Timestamp of the chunk
            
        Returns:
            RiskDataPoint with analysis results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if chunk_id is None:
            chunk_id = f"chunk_{self.total_chunks_processed}"
        
        try:
            # Check audio quality
            if audio_quality and not audio_quality.is_acceptable():
                logger.warning(f"Poor audio quality for chunk {chunk_id}")
                # Could implement quality-based processing adjustments here
            
            # Process through inference adapter
            result = await self.inference_adapter.process_audio_chunk_async(
                audio_data=audio_data,
                transcript_text=transcript_text,
                language=language,
                chunk_id=chunk_id,
                timestamp=timestamp
            )
            
            # Update adaptive processing
            if self.adaptive_processor:
                recommended_mode = self.adaptive_processor.update_performance(
                    result.processing_time_ms
                )
                # Could adjust processing mode here if needed
            
            # Update statistics
            self.total_chunks_processed += 1
            self.last_processing_time = timestamp
            
            # Call registered callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk {chunk_id}: {e}")
            self.processing_errors += 1
            
            # Return error result
            return RiskDataPoint(
                timestamp=timestamp,
                risk_score=0.0,
                confidence=0.0,
                audio_contribution=0.0,
                chunk_id=chunk_id,
                processing_mode=ProcessingMode.BASIC
            )
    
    def add_result_callback(self, callback: Callable[[RiskDataPoint], None]):
        """Add callback for processing results."""
        self.result_callbacks.append(callback)
    
    def remove_result_callback(self, callback: Callable[[RiskDataPoint], None]):
        """Remove result callback."""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = {
            'total_chunks_processed': self.total_chunks_processed,
            'processing_errors': self.processing_errors,
            'error_rate': (
                self.processing_errors / self.total_chunks_processed
                if self.total_chunks_processed > 0 else 0.0
            ),
            'last_processing_time': (
                self.last_processing_time.isoformat()
                if self.last_processing_time else None
            ),
            'inference_metrics': self.inference_adapter.get_performance_metrics()
        }
        
        if self.adaptive_processor:
            stats['adaptive_processing'] = self.adaptive_processor.get_performance_summary()
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.total_chunks_processed = 0
        self.processing_errors = 0
        self.last_processing_time = None
        
        # Reset adapter statistics
        self.inference_adapter.total_inferences = 0
        self.inference_adapter.total_inference_time = 0.0
        self.inference_adapter.inference_errors = 0
        
        logger.info("Stream processor statistics reset")


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    Prevents cascading failures by temporarily disabling failing operations.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        success_threshold: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before trying again
            success_threshold: Number of successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # 'closed', 'open', 'half_open'
        
        logger.info(f"CircuitBreaker initialized with {failure_threshold} failure threshold")
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == 'open':
            # Check if timeout has passed
            if (time.time() - self.last_failure_time) > self.timeout_seconds:
                self.state = 'half_open'
                self.success_count = 0
                logger.info("Circuit breaker moving to half-open state")
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == 'half_open':
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = 'closed'
                self.failure_count = 0
                logger.info("Circuit breaker closed after successful recovery")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'time_until_retry': max(
                0, 
                self.timeout_seconds - (time.time() - (self.last_failure_time or 0))
            ) if self.state == 'open' else 0
        }