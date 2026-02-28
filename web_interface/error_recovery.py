"""
Comprehensive Error Recovery System
Implements exponential backoff, graceful degradation, and automatic retry mechanisms
Requirements: 6.3
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from enum import Enum
import random

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"

class ErrorRecoveryManager:
    """Manages error recovery strategies and fallback mechanisms"""
    
    def __init__(self):
        self.retry_configs = {
            'websocket_connection': {
                'max_retries': 5,
                'base_delay': 1.0,
                'max_delay': 30.0,
                'strategy': RetryStrategy.EXPONENTIAL_BACKOFF,
                'jitter': True
            },
            'api_request': {
                'max_retries': 3,
                'base_delay': 0.5,
                'max_delay': 10.0,
                'strategy': RetryStrategy.EXPONENTIAL_BACKOFF,
                'jitter': True
            },
            'nlp_analysis': {
                'max_retries': 2,
                'base_delay': 1.0,
                'max_delay': 5.0,
                'strategy': RetryStrategy.LINEAR_BACKOFF,
                'jitter': False
            },
            'file_upload': {
                'max_retries': 3,
                'base_delay': 2.0,
                'max_delay': 15.0,
                'strategy': RetryStrategy.EXPONENTIAL_BACKOFF,
                'jitter': True
            }
        }
        
        self.fallback_handlers = {}
        self.error_history = []
        self.circuit_breakers = {}
        
    def register_fallback(self, operation_type: str, fallback_func: Callable):
        """Register fallback function for operation type"""
        self.fallback_handlers[operation_type] = fallback_func
        logger.info(f"Registered fallback for {operation_type}")
    
    def with_retry(self, operation_type: str = 'default'):
        """Decorator for automatic retry with exponential backoff"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                config = self.retry_configs.get(operation_type, self.retry_configs['api_request'])
                
                last_exception = None
                
                for attempt in range(config['max_retries'] + 1):
                    try:
                        # Check circuit breaker
                        if self._is_circuit_open(operation_type):
                            raise CircuitBreakerOpenError(f"Circuit breaker open for {operation_type}")
                        
                        result = await func(*args, **kwargs)
                        
                        # Reset circuit breaker on success
                        self._reset_circuit_breaker(operation_type)
                        
                        if attempt > 0:
                            logger.info(f"Operation {func.__name__} succeeded after {attempt} retries")
                        
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        
                        # Record failure
                        self._record_failure(operation_type, e)
                        
                        # Don't retry on final attempt
                        if attempt == config['max_retries']:
                            break
                        
                        # Calculate delay
                        delay = self._calculate_delay(config, attempt)
                        
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                        
                        await asyncio.sleep(delay)
                
                # All retries exhausted, try fallback
                if operation_type in self.fallback_handlers:
                    try:
                        logger.info(f"Attempting fallback for {operation_type}")
                        return await self.fallback_handlers[operation_type](*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback failed for {operation_type}: {fallback_error}")
                
                # No fallback or fallback failed
                logger.error(f"Operation {func.__name__} failed after {config['max_retries']} retries")
                raise last_exception
            
            return wrapper
        return decorator
    
    def _calculate_delay(self, config: Dict[str, Any], attempt: int) -> float:
        """Calculate delay based on retry strategy"""
        base_delay = config['base_delay']
        max_delay = config['max_delay']
        strategy = config['strategy']
        jitter = config.get('jitter', False)
        
        if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (2 ** attempt)
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * (attempt + 1)
        elif strategy == RetryStrategy.FIXED_DELAY:
            delay = base_delay
        else:  # IMMEDIATE
            delay = 0
        
        # Apply max delay limit
        delay = min(delay, max_delay)
        
        # Add jitter to prevent thundering herd
        if jitter:
            delay += random.uniform(0, delay * 0.1)
        
        return delay
    
    def _record_failure(self, operation_type: str, error: Exception):
        """Record failure for circuit breaker logic"""
        failure_record = {
            'operation_type': operation_type,
            'error': str(error),
            'timestamp': time.time(),
            'severity': self._classify_error_severity(error)
        }
        
        self.error_history.append(failure_record)
        
        # Keep only recent failures (last hour)
        cutoff_time = time.time() - 3600
        self.error_history = [
            record for record in self.error_history 
            if record['timestamp'] > cutoff_time
        ]
        
        # Update circuit breaker
        self._update_circuit_breaker(operation_type, error)
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in ['memory', 'disk', 'system']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_message for keyword in ['timeout', 'connection', 'network']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_message for keyword in ['permission', 'auth', 'forbidden']):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _update_circuit_breaker(self, operation_type: str, error: Exception):
        """Update circuit breaker state"""
        if operation_type not in self.circuit_breakers:
            self.circuit_breakers[operation_type] = {
                'failure_count': 0,
                'last_failure_time': 0,
                'state': 'closed',  # closed, open, half_open
                'failure_threshold': 5,
                'recovery_timeout': 60
            }
        
        breaker = self.circuit_breakers[operation_type]
        breaker['failure_count'] += 1
        breaker['last_failure_time'] = time.time()
        
        # Open circuit if threshold exceeded
        if breaker['failure_count'] >= breaker['failure_threshold']:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {operation_type}")
    
    def _is_circuit_open(self, operation_type: str) -> bool:
        """Check if circuit breaker is open"""
        if operation_type not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[operation_type]
        
        if breaker['state'] == 'closed':
            return False
        
        if breaker['state'] == 'open':
            # Check if recovery timeout has passed
            if time.time() - breaker['last_failure_time'] > breaker['recovery_timeout']:
                breaker['state'] = 'half_open'
                logger.info(f"Circuit breaker half-opened for {operation_type}")
                return False
            return True
        
        # half_open state - allow one attempt
        return False
    
    def _reset_circuit_breaker(self, operation_type: str):
        """Reset circuit breaker on successful operation"""
        if operation_type in self.circuit_breakers:
            breaker = self.circuit_breakers[operation_type]
            breaker['failure_count'] = 0
            breaker['state'] = 'closed'
            logger.info(f"Circuit breaker reset for {operation_type}")

class GracefulDegradationManager:
    """Manages graceful degradation strategies"""
    
    def __init__(self):
        self.degradation_levels = {
            'full_functionality': 0,
            'reduced_features': 1,
            'basic_functionality': 2,
            'emergency_mode': 3
        }
        
        self.current_level = 0
        self.feature_availability = {
            'real_time_analysis': True,
            'file_upload': True,
            'visualization': True,
            'metrics_dashboard': True,
            'session_history': True
        }
    
    def degrade_service(self, severity: ErrorSeverity):
        """Degrade service based on error severity"""
        if severity == ErrorSeverity.CRITICAL:
            self._set_degradation_level(3)  # Emergency mode
        elif severity == ErrorSeverity.HIGH:
            self._set_degradation_level(2)  # Basic functionality
        elif severity == ErrorSeverity.MEDIUM:
            self._set_degradation_level(1)  # Reduced features
    
    def _set_degradation_level(self, level: int):
        """Set degradation level and update feature availability"""
        if level <= self.current_level:
            return  # Don't degrade further
        
        self.current_level = level
        
        if level >= 1:  # Reduced features
            self.feature_availability['metrics_dashboard'] = False
            self.feature_availability['session_history'] = False
        
        if level >= 2:  # Basic functionality
            self.feature_availability['file_upload'] = False
            self.feature_availability['visualization'] = False
        
        if level >= 3:  # Emergency mode
            self.feature_availability['real_time_analysis'] = False
        
        logger.warning(f"Service degraded to level {level}")
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if feature is available at current degradation level"""
        return self.feature_availability.get(feature, False)
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        return {
            'level': self.current_level,
            'features': self.feature_availability.copy()
        }
    
    def recover_service(self):
        """Attempt to recover service to full functionality"""
        self.current_level = 0
        self.feature_availability = {
            'real_time_analysis': True,
            'file_upload': True,
            'visualization': True,
            'metrics_dashboard': True,
            'session_history': True
        }
        logger.info("Service recovered to full functionality")

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

# Global instances
error_recovery_manager = ErrorRecoveryManager()
degradation_manager = GracefulDegradationManager()

# Convenience decorators
def with_retry(operation_type: str = 'default'):
    """Convenience decorator for retry functionality"""
    return error_recovery_manager.with_retry(operation_type)

def register_fallback(operation_type: str):
    """Convenience decorator for registering fallback functions"""
    def decorator(func):
        error_recovery_manager.register_fallback(operation_type, func)
        return func
    return decorator