"""
Demo Controller for Batch Operations

This module provides orchestration for interactive demonstrations,
including progress tracking, result aggregation, and sequential processing.

Requirements: 5.3, 5.4, 5.5
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .models import (
    AnalysisResult, SessionData, SessionStatus, FileUploadResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Batch processing status values."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class BatchProgress:
    """Progress tracking for batch operations."""
    batch_id: str
    total_items: int
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    current_item: Optional[str] = None
    status: BatchStatus = BatchStatus.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100.0
    
    @property
    def processing_time(self) -> Optional[float]:
        """Calculate total processing time in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


@dataclass
class BatchItem:
    """Individual item in a batch operation."""
    item_id: str
    item_type: str  # "file", "text", "audio"
    data: Any  # File path, text content, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class BatchOperation:
    """Complete batch operation definition."""
    batch_id: str
    operation_type: str  # "file_processing", "text_analysis", "audio_transcription"
    items: List[BatchItem]
    progress: BatchProgress
    results: List[Any] = field(default_factory=list)
    aggregated_results: Optional[Dict[str, Any]] = None
    progress_callback: Optional[Callable] = None


class DemoController:
    """Controller for orchestrating interactive demonstrations."""
    
    def __init__(self):
        """Initialize demo controller."""
        self.active_batches: Dict[str, BatchOperation] = {}
        self.completed_batches: Dict[str, BatchOperation] = {}
        self.max_concurrent_batches = 5
        self.max_batch_size = 100
        self.batch_timeout = 3600  # 1 hour timeout
        
        # Statistics
        self.stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_items_processed": 0,
            "avg_processing_time": 0.0
        }
    
    def generate_batch_id(self) -> str:
        """Generate unique batch identifier."""
        return f"batch_{uuid.uuid4().hex[:12]}"
    
    async def create_batch_operation(
        self,
        operation_type: str,
        items: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Create a new batch operation.
        
        Args:
            operation_type: Type of operation to perform
            items: List of items to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Batch ID for tracking
        """
        # Validate input
        if len(items) > self.max_batch_size:
            raise ValueError(f"Batch size too large (max {self.max_batch_size})")
        
        if len(self.active_batches) >= self.max_concurrent_batches:
            raise ValueError("Too many concurrent batches")
        
        # Generate batch ID
        batch_id = self.generate_batch_id()
        
        # Create batch items
        batch_items = []
        for i, item_data in enumerate(items):
            item_id = f"{batch_id}_item_{i}"
            batch_items.append(BatchItem(
                item_id=item_id,
                item_type=item_data.get("type", "unknown"),
                data=item_data.get("data"),
                metadata=item_data.get("metadata", {})
            ))
        
        # Create progress tracker
        progress = BatchProgress(
            batch_id=batch_id,
            total_items=len(batch_items)
        )
        
        # Create batch operation
        batch_operation = BatchOperation(
            batch_id=batch_id,
            operation_type=operation_type,
            items=batch_items,
            progress=progress,
            progress_callback=progress_callback
        )
        
        # Store batch
        self.active_batches[batch_id] = batch_operation
        self.stats["total_batches"] += 1
        
        logger.info(f"Created batch operation {batch_id} with {len(batch_items)} items")
        return batch_id
    
    async def start_batch_processing(self, batch_id: str):
        """
        Start processing a batch operation.
        
        Args:
            batch_id: Batch identifier
        """
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch_operation = self.active_batches[batch_id]
        
        # Update status
        batch_operation.progress.status = BatchStatus.PROCESSING
        batch_operation.progress.started_at = datetime.now()
        
        # Notify progress callback
        if batch_operation.progress_callback:
            try:
                await batch_operation.progress_callback(batch_operation.progress)
            except Exception as e:
                logger.warning(f"Progress callback failed for batch {batch_id}: {e}")
        
        logger.info(f"Started processing batch {batch_id}")
        
        # Process items sequentially
        try:
            await self._process_batch_items(batch_operation)
            
            # Mark as completed
            batch_operation.progress.status = BatchStatus.COMPLETED
            batch_operation.progress.completed_at = datetime.now()
            
            # Aggregate results
            batch_operation.aggregated_results = await self._aggregate_results(batch_operation)
            
            # Update statistics
            self.stats["successful_batches"] += 1
            self.stats["total_items_processed"] += len(batch_operation.items)
            
            # Calculate average processing time
            if batch_operation.progress.processing_time:
                current_avg = self.stats["avg_processing_time"]
                total_batches = self.stats["successful_batches"]
                self.stats["avg_processing_time"] = (
                    (current_avg * (total_batches - 1) + batch_operation.progress.processing_time) / total_batches
                )
            
            logger.info(f"Completed batch processing {batch_id}")
            
        except Exception as e:
            # Mark as failed
            batch_operation.progress.status = BatchStatus.ERROR
            batch_operation.progress.error_message = str(e)
            batch_operation.progress.completed_at = datetime.now()
            
            self.stats["failed_batches"] += 1
            
            logger.error(f"Batch processing failed for {batch_id}: {e}")
        
        finally:
            # Final progress callback
            if batch_operation.progress_callback:
                try:
                    await batch_operation.progress_callback(batch_operation.progress)
                except Exception as e:
                    logger.warning(f"Final progress callback failed for batch {batch_id}: {e}")
            
            # Move to completed batches
            self.completed_batches[batch_id] = batch_operation
            del self.active_batches[batch_id]
    
    async def _process_batch_items(self, batch_operation: BatchOperation):
        """Process all items in a batch sequentially."""
        from .api import get_nlp_service, process_audio_file, process_text_file
        
        for item in batch_operation.items:
            try:
                # Update current item
                batch_operation.progress.current_item = item.item_id
                
                # Process based on item type
                start_time = time.time()
                
                if item.item_type == "audio_file":
                    # Process audio file
                    file_path = item.data
                    language = item.metadata.get("language", "tamil")
                    result = await process_audio_file(file_path, item.item_id, language)
                    item.result = result
                    
                elif item.item_type == "text_file":
                    # Process text file
                    file_path = item.data
                    language = item.metadata.get("language", "english")
                    results = await process_text_file(file_path, item.item_id, language)
                    item.result = results
                    
                elif item.item_type == "text":
                    # Process text directly
                    text = item.data
                    language = item.metadata.get("language", "english")
                    service = await get_nlp_service()
                    result = await service.analyze_text(
                        text=text,
                        language=language,
                        include_sentiment=True,
                        include_features=True
                    )
                    item.result = result
                    
                else:
                    raise ValueError(f"Unsupported item type: {item.item_type}")
                
                # Update item status
                item.status = "completed"
                item.processing_time = time.time() - start_time
                batch_operation.progress.successful_items += 1
                batch_operation.results.append(item.result)
                
            except Exception as e:
                # Handle item processing error
                item.status = "error"
                item.error = str(e)
                item.processing_time = time.time() - start_time
                batch_operation.progress.failed_items += 1
                
                logger.error(f"Failed to process item {item.item_id}: {e}")
            
            finally:
                # Update progress
                batch_operation.progress.processed_items += 1
                
                # Progress callback
                if batch_operation.progress_callback:
                    try:
                        await batch_operation.progress_callback(batch_operation.progress)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
    
    async def _aggregate_results(self, batch_operation: BatchOperation) -> Dict[str, Any]:
        """Aggregate results from batch processing."""
        aggregated = {
            "batch_id": batch_operation.batch_id,
            "operation_type": batch_operation.operation_type,
            "summary": {
                "total_items": len(batch_operation.items),
                "successful_items": batch_operation.progress.successful_items,
                "failed_items": batch_operation.progress.failed_items,
                "processing_time": batch_operation.progress.processing_time,
                "success_rate": (
                    batch_operation.progress.successful_items / len(batch_operation.items) * 100
                    if batch_operation.items else 0
                )
            },
            "results": batch_operation.results,
            "errors": [
                {"item_id": item.item_id, "error": item.error}
                for item in batch_operation.items
                if item.error
            ]
        }
        
        # Add operation-specific aggregations
        if batch_operation.operation_type == "text_analysis":
            aggregated["text_analysis_summary"] = self._aggregate_text_analysis(batch_operation.results)
        elif batch_operation.operation_type == "audio_transcription":
            aggregated["transcription_summary"] = self._aggregate_transcription_results(batch_operation.results)
        
        return aggregated
    
    def _aggregate_text_analysis(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Aggregate text analysis results."""
        if not results:
            return {}
        
        # Calculate averages
        total_processing_time = sum(r.processing_time for r in results)
        total_words = sum(len(r.tokens) for r in results)
        
        # Aggregate sentiment scores
        sentiment_scores = {"positive": [], "negative": [], "neutral": []}
        for result in results:
            for sentiment_type, score in result.sentiment.scores.items():
                if sentiment_type in sentiment_scores:
                    sentiment_scores[sentiment_type].append(score)
        
        avg_sentiment = {
            sentiment_type: sum(scores) / len(scores) if scores else 0.0
            for sentiment_type, scores in sentiment_scores.items()
        }
        
        # Aggregate POS distribution
        pos_distribution = {}
        for result in results:
            for pos_tag, count in result.linguistic_features.pos_distribution.items():
                pos_distribution[pos_tag] = pos_distribution.get(pos_tag, 0) + count
        
        # Normalize POS distribution
        total_pos_count = sum(pos_distribution.values())
        if total_pos_count > 0:
            pos_distribution = {
                pos: count / total_pos_count
                for pos, count in pos_distribution.items()
            }
        
        return {
            "total_texts": len(results),
            "total_processing_time": total_processing_time,
            "avg_processing_time": total_processing_time / len(results),
            "total_words": total_words,
            "avg_words_per_text": total_words / len(results),
            "avg_sentiment_scores": avg_sentiment,
            "pos_distribution": pos_distribution,
            "languages": list(set(r.language for r in results))
        }
    
    def _aggregate_transcription_results(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Aggregate transcription results."""
        if not results:
            return {}
        
        total_processing_time = sum(r.processing_time for r in results)
        total_words = sum(len(r.tokens) for r in results)
        
        return {
            "total_transcriptions": len(results),
            "total_processing_time": total_processing_time,
            "avg_processing_time": total_processing_time / len(results),
            "total_words": total_words,
            "avg_words_per_transcription": total_words / len(results),
            "languages": list(set(r.language for r in results))
        }
    
    def get_batch_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get progress information for a batch."""
        if batch_id in self.active_batches:
            return self.active_batches[batch_id].progress
        elif batch_id in self.completed_batches:
            return self.completed_batches[batch_id].progress
        return None
    
    def get_batch_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a completed batch."""
        if batch_id in self.completed_batches:
            batch_operation = self.completed_batches[batch_id]
            return batch_operation.aggregated_results
        return None
    
    def get_batch_details(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a batch."""
        batch_operation = None
        if batch_id in self.active_batches:
            batch_operation = self.active_batches[batch_id]
        elif batch_id in self.completed_batches:
            batch_operation = self.completed_batches[batch_id]
        
        if not batch_operation:
            return None
        
        return {
            "batch_id": batch_id,
            "operation_type": batch_operation.operation_type,
            "progress": batch_operation.progress,
            "items": [
                {
                    "item_id": item.item_id,
                    "item_type": item.item_type,
                    "status": item.status,
                    "processing_time": item.processing_time,
                    "error": item.error
                }
                for item in batch_operation.items
            ],
            "aggregated_results": batch_operation.aggregated_results
        }
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel an active batch operation."""
        if batch_id not in self.active_batches:
            return False
        
        batch_operation = self.active_batches[batch_id]
        batch_operation.progress.status = BatchStatus.CANCELLED
        batch_operation.progress.completed_at = datetime.now()
        
        # Move to completed batches
        self.completed_batches[batch_id] = batch_operation
        del self.active_batches[batch_id]
        
        logger.info(f"Cancelled batch operation {batch_id}")
        return True
    
    def cleanup_old_batches(self, max_age_hours: int = 24):
        """Clean up old completed batches."""
        current_time = datetime.now()
        to_remove = []
        
        for batch_id, batch_operation in self.completed_batches.items():
            if batch_operation.progress.completed_at:
                age_hours = (current_time - batch_operation.progress.completed_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    to_remove.append(batch_id)
        
        for batch_id in to_remove:
            del self.completed_batches[batch_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old batch operations")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            **self.stats,
            "active_batches": len(self.active_batches),
            "completed_batches": len(self.completed_batches)
        }


# Global demo controller instance
demo_controller = DemoController()


def get_demo_controller() -> DemoController:
    """Get the global demo controller instance."""
    return demo_controller