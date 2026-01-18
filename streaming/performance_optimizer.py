"""
Performance Optimizer for Real-time Streaming Analysis

Provides performance optimization, resource management, and adaptive
processing capabilities for the streaming system.

Reference: Real-time Streaming Analysis - Task 4.1
"""

import logging
import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np

from .models import ProcessingMode, SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceTarget:
    """Performance target configuration."""
    max_latency_ms: float = 2000.0
    min_throughput_chunks_per_sec: float = 1.0
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_percent: float = 70.0
    max_gpu_memory_usage_percent: float = 85.0


@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration."""
    enable_adaptive_processing: bool = True
    enable_model_caching: bool = True
    enable_batch_processing: bool = True
    enable_gpu_optimization: bool = True
    enable_memory_management: bool = True


class PerformanceMonitor:
    """
    Real-time performance monitoring and metrics collection.
    
    Tracks system performance, processing latency, and resource usage
    to enable adaptive optimization strategies.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Number of metrics to keep in history
        """
        self.history_size = history_size
        
        # Performance metrics history
        self.latency_history = deque(maxlen=history_size)
        self.throughput_history = deque(maxlen=history_size)
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.gpu_memory_history = deque(maxlen=history_size)
        
        # Current metrics
        self.current_metrics = SystemMetrics(
            timestamp=datetime.now(),
            audio_capture_latency_ms=0.0,
            processing_latency_ms=0.0,
            end_to_end_latency_ms=0.0,
            chunks_processed_per_second=0.0,
            sessions_active=0,
            average_audio_quality=0.0,
            average_model_confidence=0.0,
            cpu_usage_percent=0.0,
            memory_usage_percent=0.0,
            gpu_memory_usage_percent=0.0,
            processing_errors=0,
            audio_dropouts=0,
            model_failures=0
        )
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        self._lock = threading.Lock()
    
    def start_monitoring(self, update_interval: float = 1.0):
        """
        Start performance monitoring.
        
        Args:
            update_interval: Seconds between metric updates
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(update_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, update_interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # GPU memory (if available)
            gpu_memory_percent = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated()
                    gpu_memory_total = torch.cuda.max_memory_allocated()
                    if gpu_memory_total > 0:
                        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            except ImportError:
                pass
            
            # Update current metrics
            with self._lock:
                self.current_metrics.timestamp = datetime.now()
                self.current_metrics.cpu_usage_percent = cpu_percent
                self.current_metrics.memory_usage_percent = memory.percent
                self.current_metrics.gpu_memory_usage_percent = gpu_memory_percent
                
                # Add to history
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory.percent)
                self.gpu_memory_history.append(gpu_memory_percent)
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def record_latency(self, latency_ms: float, latency_type: str = "processing"):
        """
        Record processing latency.
        
        Args:
            latency_ms: Latency in milliseconds
            latency_type: Type of latency (processing, audio_capture, end_to_end)
        """
        with self._lock:
            if latency_type == "processing":
                self.current_metrics.processing_latency_ms = latency_ms
                self.latency_history.append(latency_ms)
            elif latency_type == "audio_capture":
                self.current_metrics.audio_capture_latency_ms = latency_ms
            elif latency_type == "end_to_end":
                self.current_metrics.end_to_end_latency_ms = latency_ms
    
    def record_throughput(self, chunks_per_second: float):
        """
        Record processing throughput.
        
        Args:
            chunks_per_second: Chunks processed per second
        """
        with self._lock:
            self.current_metrics.chunks_processed_per_second = chunks_per_second
            self.throughput_history.append(chunks_per_second)
    
    def record_error(self, error_type: str):
        """
        Record processing error.
        
        Args:
            error_type: Type of error (processing, audio_dropout, model_failure)
        """
        with self._lock:
            if error_type == "processing":
                self.current_metrics.processing_errors += 1
            elif error_type == "audio_dropout":
                self.current_metrics.audio_dropouts += 1
            elif error_type == "model_failure":
                self.current_metrics.model_failures += 1
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        with self._lock:
            return SystemMetrics(
                timestamp=self.current_metrics.timestamp,
                audio_capture_latency_ms=self.current_metrics.audio_capture_latency_ms,
                processing_latency_ms=self.current_metrics.processing_latency_ms,
                end_to_end_latency_ms=self.current_metrics.end_to_end_latency_ms,
                chunks_processed_per_second=self.current_metrics.chunks_processed_per_second,
                sessions_active=self.current_metrics.sessions_active,
                average_audio_quality=self.current_metrics.average_audio_quality,
                average_model_confidence=self.current_metrics.average_model_confidence,
                cpu_usage_percent=self.current_metrics.cpu_usage_percent,
                memory_usage_percent=self.current_metrics.memory_usage_percent,
                gpu_memory_usage_percent=self.current_metrics.gpu_memory_usage_percent,
                processing_errors=self.current_metrics.processing_errors,
                audio_dropouts=self.current_metrics.audio_dropouts,
                model_failures=self.current_metrics.model_failures
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            latency_array = np.array(list(self.latency_history)) if self.latency_history else np.array([0])
            throughput_array = np.array(list(self.throughput_history)) if self.throughput_history else np.array([0])
            cpu_array = np.array(list(self.cpu_history)) if self.cpu_history else np.array([0])
            memory_array = np.array(list(self.memory_history)) if self.memory_history else np.array([0])
            
            return {
                'latency_stats': {
                    'mean': float(np.mean(latency_array)),
                    'median': float(np.median(latency_array)),
                    'p95': float(np.percentile(latency_array, 95)),
                    'p99': float(np.percentile(latency_array, 99)),
                    'max': float(np.max(latency_array))
                },
                'throughput_stats': {
                    'mean': float(np.mean(throughput_array)),
                    'median': float(np.median(throughput_array)),
                    'min': float(np.min(throughput_array)),
                    'max': float(np.max(throughput_array))
                },
                'resource_stats': {
                    'cpu_mean': float(np.mean(cpu_array)),
                    'cpu_max': float(np.max(cpu_array)),
                    'memory_mean': float(np.mean(memory_array)),
                    'memory_max': float(np.max(memory_array))
                },
                'error_counts': {
                    'processing_errors': self.current_metrics.processing_errors,
                    'audio_dropouts': self.current_metrics.audio_dropouts,
                    'model_failures': self.current_metrics.model_failures
                }
            }


class AdaptiveProcessor:
    """
    Adaptive processing optimization based on performance metrics.
    
    Automatically adjusts processing parameters based on system performance
    to maintain target latency and throughput while optimizing resource usage.
    """
    
    def __init__(self, 
                 performance_targets: PerformanceTarget,
                 optimization_strategy: OptimizationStrategy):
        """
        Initialize adaptive processor.
        
        Args:
            performance_targets: Target performance metrics
            optimization_strategy: Optimization strategy configuration
        """
        self.targets = performance_targets
        self.strategy = optimization_strategy
        
        # Current processing mode
        self.current_mode = ProcessingMode.TRIMODAL
        self.mode_history = deque(maxlen=100)
        
        # Optimization callbacks
        self.mode_change_callbacks: List[Callable[[ProcessingMode], None]] = []
        
        # Performance tracking
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(seconds=10)
        
        self._lock = threading.Lock()
    
    def add_mode_change_callback(self, callback: Callable[[ProcessingMode], None]):
        """Add callback for processing mode changes."""
        self.mode_change_callbacks.append(callback)
    
    def optimize_processing(self, performance_monitor: PerformanceMonitor):
        """
        Optimize processing based on current performance metrics.
        
        Args:
            performance_monitor: Performance monitor instance
        """
        if not self.strategy.enable_adaptive_processing:
            return
        
        # Check if optimization interval has passed
        if datetime.now() - self.last_optimization < self.optimization_interval:
            return
        
        try:
            metrics = performance_monitor.get_current_metrics()
            summary = performance_monitor.get_performance_summary()
            
            # Determine if optimization is needed
            needs_optimization = self._needs_optimization(metrics, summary)
            
            if needs_optimization:
                new_mode = self._select_optimal_mode(metrics, summary)
                
                if new_mode != self.current_mode:
                    self._change_processing_mode(new_mode)
            
            self.last_optimization = datetime.now()
        
        except Exception as e:
            logger.error(f"Error in adaptive optimization: {e}")
    
    def _needs_optimization(self, metrics: SystemMetrics, summary: Dict[str, Any]) -> bool:
        """Check if optimization is needed based on performance metrics."""
        # Check latency targets
        if summary['latency_stats']['p95'] > self.targets.max_latency_ms:
            logger.info(f"Latency optimization needed: {summary['latency_stats']['p95']:.1f}ms > {self.targets.max_latency_ms}ms")
            return True
        
        # Check throughput targets
        if summary['throughput_stats']['mean'] < self.targets.min_throughput_chunks_per_sec:
            logger.info(f"Throughput optimization needed: {summary['throughput_stats']['mean']:.2f} < {self.targets.min_throughput_chunks_per_sec}")
            return True
        
        # Check resource usage
        if metrics.cpu_usage_percent > self.targets.max_cpu_usage_percent:
            logger.info(f"CPU optimization needed: {metrics.cpu_usage_percent:.1f}% > {self.targets.max_cpu_usage_percent}%")
            return True
        
        if metrics.memory_usage_percent > self.targets.max_memory_usage_percent:
            logger.info(f"Memory optimization needed: {metrics.memory_usage_percent:.1f}% > {self.targets.max_memory_usage_percent}%")
            return True
        
        return False
    
    def _select_optimal_mode(self, metrics: SystemMetrics, summary: Dict[str, Any]) -> ProcessingMode:
        """Select optimal processing mode based on performance metrics."""
        current_latency = summary['latency_stats']['p95']
        current_cpu = metrics.cpu_usage_percent
        current_memory = metrics.memory_usage_percent
        
        # If performance is severely degraded, use fastest mode
        if (current_latency > self.targets.max_latency_ms * 1.5 or
            current_cpu > self.targets.max_cpu_usage_percent * 1.2):
            return ProcessingMode.FAST
        
        # If performance is moderately degraded, reduce processing complexity
        if (current_latency > self.targets.max_latency_ms or
            current_cpu > self.targets.max_cpu_usage_percent):
            
            if self.current_mode == ProcessingMode.TRIMODAL:
                return ProcessingMode.BIMODAL
            elif self.current_mode == ProcessingMode.BIMODAL:
                return ProcessingMode.AUDIO_ONLY
            elif self.current_mode == ProcessingMode.AUDIO_ONLY:
                return ProcessingMode.FAST
        
        # If performance is good, try to upgrade processing quality
        if (current_latency < self.targets.max_latency_ms * 0.7 and
            current_cpu < self.targets.max_cpu_usage_percent * 0.7):
            
            if self.current_mode == ProcessingMode.FAST:
                return ProcessingMode.AUDIO_ONLY
            elif self.current_mode == ProcessingMode.AUDIO_ONLY:
                return ProcessingMode.BIMODAL
            elif self.current_mode == ProcessingMode.BIMODAL:
                return ProcessingMode.TRIMODAL
        
        # Keep current mode if no clear optimization path
        return self.current_mode
    
    def _change_processing_mode(self, new_mode: ProcessingMode):
        """Change processing mode and notify callbacks."""
        with self._lock:
            old_mode = self.current_mode
            self.current_mode = new_mode
            self.mode_history.append((datetime.now(), old_mode, new_mode))
        
        logger.info(f"Processing mode changed: {old_mode.value} → {new_mode.value}")
        
        # Notify callbacks
        for callback in self.mode_change_callbacks:
            try:
                callback(new_mode)
            except Exception as e:
                logger.error(f"Error in mode change callback: {e}")
    
    def get_current_mode(self) -> ProcessingMode:
        """Get current processing mode."""
        with self._lock:
            return self.current_mode
    
    def get_mode_history(self) -> List[tuple]:
        """Get processing mode change history."""
        with self._lock:
            return list(self.mode_history)


class ResourceManager:
    """
    System resource management and optimization.
    
    Manages memory usage, GPU resources, and system resources
    to ensure stable operation under varying loads.
    """
    
    def __init__(self):
        """Initialize resource manager."""
        self.memory_threshold_mb = 1024  # 1GB threshold for cleanup
        self.gpu_memory_threshold_percent = 80.0
        
        # Resource cleanup callbacks
        self.cleanup_callbacks: List[Callable[[], None]] = []
        
        # Monitoring
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)
    
    def add_cleanup_callback(self, callback: Callable[[], None]):
        """Add callback for resource cleanup."""
        self.cleanup_callbacks.append(callback)
    
    def manage_resources(self, performance_monitor: PerformanceMonitor):
        """
        Manage system resources based on current usage.
        
        Args:
            performance_monitor: Performance monitor instance
        """
        if datetime.now() - self.last_cleanup < self.cleanup_interval:
            return
        
        try:
            metrics = performance_monitor.get_current_metrics()
            
            # Check memory usage
            if metrics.memory_usage_percent > 80.0:
                self._cleanup_memory()
            
            # Check GPU memory usage
            if metrics.gpu_memory_usage_percent > self.gpu_memory_threshold_percent:
                self._cleanup_gpu_memory()
            
            self.last_cleanup = datetime.now()
        
        except Exception as e:
            logger.error(f"Error in resource management: {e}")
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        logger.info("Performing memory cleanup")
        
        # Python garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Call cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
    
    def _cleanup_gpu_memory(self):
        """Perform GPU memory cleanup."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Performing GPU memory cleanup")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        try:
            memory = psutil.virtual_memory()
            
            status = {
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            }
            
            # GPU status
            try:
                import torch
                if torch.cuda.is_available():
                    status['gpu_available'] = True
                    status['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                    status['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
                else:
                    status['gpu_available'] = False
            except ImportError:
                status['gpu_available'] = False
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting resource status: {e}")
            return {}


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    Coordinates performance monitoring, adaptive processing,
    and resource management for optimal system performance.
    """
    
    def __init__(self,
                 performance_targets: Optional[PerformanceTarget] = None,
                 optimization_strategy: Optional[OptimizationStrategy] = None):
        """
        Initialize performance optimizer.
        
        Args:
            performance_targets: Performance target configuration
            optimization_strategy: Optimization strategy configuration
        """
        self.targets = performance_targets or PerformanceTarget()
        self.strategy = optimization_strategy or OptimizationStrategy()
        
        # Components
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_processor = AdaptiveProcessor(self.targets, self.strategy)
        self.resource_manager = ResourceManager()
        
        # Optimization thread
        self.optimization_thread = None
        self.optimization_active = False
    
    def start_optimization(self, optimization_interval: float = 5.0):
        """
        Start performance optimization.
        
        Args:
            optimization_interval: Seconds between optimization cycles
        """
        if self.optimization_active:
            return
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Start optimization loop
        self.optimization_active = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            args=(optimization_interval,),
            daemon=True
        )
        self.optimization_thread.start()
        
        logger.info("Performance optimization started")
    
    def stop_optimization(self):
        """Stop performance optimization."""
        self.optimization_active = False
        
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=2.0)
        
        self.performance_monitor.stop_monitoring()
        logger.info("Performance optimization stopped")
    
    def _optimization_loop(self, optimization_interval: float):
        """Main optimization loop."""
        while self.optimization_active:
            try:
                # Run adaptive processing optimization
                self.adaptive_processor.optimize_processing(self.performance_monitor)
                
                # Run resource management
                self.resource_manager.manage_resources(self.performance_monitor)
                
                time.sleep(optimization_interval)
            
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(1.0)
    
    def record_processing_latency(self, latency_ms: float):
        """Record processing latency for optimization."""
        self.performance_monitor.record_latency(latency_ms, "processing")
    
    def record_throughput(self, chunks_per_second: float):
        """Record processing throughput for optimization."""
        self.performance_monitor.record_throughput(chunks_per_second)
    
    def record_error(self, error_type: str):
        """Record processing error for monitoring."""
        self.performance_monitor.record_error(error_type)
    
    def add_mode_change_callback(self, callback: Callable[[ProcessingMode], None]):
        """Add callback for processing mode changes."""
        self.adaptive_processor.add_mode_change_callback(callback)
    
    def add_cleanup_callback(self, callback: Callable[[], None]):
        """Add callback for resource cleanup."""
        self.resource_manager.add_cleanup_callback(callback)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'current_metrics': self.performance_monitor.get_current_metrics().to_dict(),
            'performance_summary': self.performance_monitor.get_performance_summary(),
            'current_processing_mode': self.adaptive_processor.get_current_mode().value,
            'mode_history': [(t.isoformat(), old.value, new.value) 
                           for t, old, new in self.adaptive_processor.get_mode_history()],
            'resource_status': self.resource_manager.get_resource_status(),
            'targets': {
                'max_latency_ms': self.targets.max_latency_ms,
                'min_throughput': self.targets.min_throughput_chunks_per_sec,
                'max_cpu_percent': self.targets.max_cpu_usage_percent,
                'max_memory_percent': self.targets.max_memory_usage_percent
            }
        }