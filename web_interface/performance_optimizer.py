"""
Performance Optimizer for NLP Web Interface
Implements caching, efficient data structures, and lazy loading
Requirements: 2.5
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache, wraps
from collections import defaultdict, deque
import hashlib
import logging

logger = logging.getLogger(__name__)

class ResultCache:
    """LRU cache for analysis results with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order = deque()
        
    def _generate_key(self, text: str, analysis_type: str) -> str:
        """Generate cache key from text and analysis type"""
        content = f"{analysis_type}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, analysis_type: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        key = self._generate_key(text, analysis_type)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                self._remove(key)
                return None
            
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            logger.debug(f"Cache hit for {analysis_type}: {text[:50]}...")
            return result
        
        return None
    
    def put(self, text: str, analysis_type: str, result: Any):
        """Store result in cache"""
        key = self._generate_key(text, analysis_type)
        
        # Remove if already exists
        if key in self.cache:
            self.access_order.remove(key)
        
        # Evict oldest if at capacity
        elif len(self.cache) >= self.max_size:
            oldest_key = self.access_order.popleft()
            del self.cache[oldest_key]
        
        # Add new entry
        self.cache[key] = (result, time.time())
        self.access_order.append(key)
        
        logger.debug(f"Cached {analysis_type}: {text[:50]}...")
    
    def _remove(self, key: str):
        """Remove key from cache"""
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        self.access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }

class WebSocketMessageOptimizer:
    """Optimizes WebSocket message handling with batching and compression"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.message_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        
    async def queue_message(self, session_id: str, message: Dict[str, Any], websocket):
        """Queue message for batched sending"""
        self.message_queues[session_id].append(message)
        
        # Send immediately if batch is full
        if len(self.message_queues[session_id]) >= self.batch_size:
            await self._flush_batch(session_id, websocket)
        else:
            # Set timer for batch timeout
            if session_id not in self.batch_timers:
                self.batch_timers[session_id] = asyncio.create_task(
                    self._batch_timeout(session_id, websocket)
                )
    
    async def _batch_timeout(self, session_id: str, websocket):
        """Handle batch timeout"""
        await asyncio.sleep(self.batch_timeout)
        await self._flush_batch(session_id, websocket)
    
    async def _flush_batch(self, session_id: str, websocket):
        """Send batched messages"""
        if session_id not in self.message_queues:
            return
            
        messages = self.message_queues[session_id]
        if not messages:
            return
        
        # Cancel timer if exists
        if session_id in self.batch_timers:
            self.batch_timers[session_id].cancel()
            del self.batch_timers[session_id]
        
        # Send batch
        batch_message = {
            'type': 'batch',
            'messages': messages,
            'count': len(messages),
            'timestamp': time.time()
        }
        
        try:
            await websocket.send_text(json.dumps(batch_message))
            logger.debug(f"Sent batch of {len(messages)} messages for session {session_id}")
        except Exception as e:
            logger.error(f"Error sending batch: {e}")
        
        # Clear queue
        self.message_queues[session_id] = []
    
    async def flush_all(self, websocket):
        """Flush all pending batches"""
        for session_id in list(self.message_queues.keys()):
            await self._flush_batch(session_id, websocket)

class LazyDataLoader:
    """Implements lazy loading for large datasets"""
    
    def __init__(self, page_size: int = 50):
        self.page_size = page_size
        self.data_cache: Dict[str, List[Any]] = {}
        
    def paginate_results(self, data: List[Any], page: int = 0) -> Dict[str, Any]:
        """Paginate large result sets"""
        start_idx = page * self.page_size
        end_idx = start_idx + self.page_size
        
        page_data = data[start_idx:end_idx]
        
        return {
            'data': page_data,
            'page': page,
            'page_size': self.page_size,
            'total_items': len(data),
            'total_pages': (len(data) + self.page_size - 1) // self.page_size,
            'has_next': end_idx < len(data),
            'has_prev': page > 0
        }
    
    def load_visualization_data(self, analysis_results: List[Dict], visualization_type: str) -> Dict[str, Any]:
        """Load visualization data with lazy loading"""
        
        # Filter data based on visualization type
        if visualization_type == 'pos_tags':
            filtered_data = [
                {
                    'text': result.get('text', ''),
                    'pos_tags': result.get('pos_tags', [])
                }
                for result in analysis_results
                if result.get('pos_tags')
            ]
        elif visualization_type == 'named_entities':
            filtered_data = [
                {
                    'text': result.get('text', ''),
                    'entities': result.get('named_entities', [])
                }
                for result in analysis_results
                if result.get('named_entities')
            ]
        elif visualization_type == 'dependencies':
            filtered_data = [
                {
                    'text': result.get('text', ''),
                    'dependencies': result.get('dependencies', [])
                }
                for result in analysis_results
                if result.get('dependencies')
            ]
        else:
            filtered_data = analysis_results
        
        # Return paginated results
        return self.paginate_results(filtered_data)

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        self.result_cache = ResultCache()
        self.websocket_optimizer = WebSocketMessageOptimizer()
        self.lazy_loader = LazyDataLoader()
        self.performance_metrics = defaultdict(list)
        
    def cached_analysis(self, analysis_func):
        """Decorator for caching analysis results"""
        @wraps(analysis_func)
        async def wrapper(text: str, *args, **kwargs):
            analysis_type = analysis_func.__name__
            
            # Check cache first
            cached_result = self.result_cache.get(text, analysis_type)
            if cached_result is not None:
                return cached_result
            
            # Perform analysis and cache result
            start_time = time.time()
            result = await analysis_func(text, *args, **kwargs)
            end_time = time.time()
            
            # Cache the result
            self.result_cache.put(text, analysis_type, result)
            
            # Track performance
            self.performance_metrics[analysis_type].append(end_time - start_time)
            
            return result
        
        return wrapper
    
    async def optimized_websocket_send(self, session_id: str, message: Dict[str, Any], websocket):
        """Send WebSocket message with optimization"""
        await self.websocket_optimizer.queue_message(session_id, message, websocket)
    
    def get_paginated_data(self, data: List[Any], page: int = 0, page_size: int = None) -> Dict[str, Any]:
        """Get paginated data with lazy loading"""
        if page_size:
            self.lazy_loader.page_size = page_size
        
        return self.lazy_loader.paginate_results(data, page)
    
    def get_visualization_data(self, analysis_results: List[Dict], viz_type: str, page: int = 0) -> Dict[str, Any]:
        """Get optimized visualization data"""
        viz_data = self.lazy_loader.load_visualization_data(analysis_results, viz_type)
        return self.lazy_loader.paginate_results(viz_data['data'], page)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'cache': self.result_cache.stats(),
            'analysis_times': {}
        }
        
        # Calculate average analysis times
        for analysis_type, times in self.performance_metrics.items():
            if times:
                stats['analysis_times'][analysis_type] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_calls': len(times)
                }
        
        return stats
    
    def clear_cache(self):
        """Clear all caches"""
        self.result_cache.clear()
        self.performance_metrics.clear()
    
    async def shutdown(self):
        """Cleanup resources"""
        await self.websocket_optimizer.flush_all(None)
        self.clear_cache()

# Global optimizer instance
performance_optimizer = PerformanceOptimizer()