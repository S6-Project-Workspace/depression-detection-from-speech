"""
Tests for Phase 4: Production Readiness

Tests performance optimization, security, encryption, and production
readiness components of the streaming system.
"""

import pytest
import time
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import Phase 4 components
from streaming.performance_optimizer import (
    PerformanceMonitor, AdaptiveProcessor, ResourceManager, 
    PerformanceOptimizer, PerformanceTarget, OptimizationStrategy
)
from streaming.models import ProcessingMode, SystemMetrics
from security.encryption_manager import (
    KeyManager, DataEncryption, SecureStorage, EncryptionManager
)


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(history_size=100)
        
        assert monitor.history_size == 100
        assert len(monitor.latency_history) == 0
        assert len(monitor.throughput_history) == 0
        assert monitor.current_metrics.timestamp is not None
    
    def test_latency_recording(self):
        """Test latency recording functionality."""
        monitor = PerformanceMonitor()
        
        # Record processing latency
        monitor.record_latency(1500.0, "processing")
        assert monitor.current_metrics.processing_latency_ms == 1500.0
        assert len(monitor.latency_history) == 1
        assert monitor.latency_history[0] == 1500.0
        
        # Record audio capture latency
        monitor.record_latency(50.0, "audio_capture")
        assert monitor.current_metrics.audio_capture_latency_ms == 50.0
        
        # Record end-to-end latency
        monitor.record_latency(2000.0, "end_to_end")
        assert monitor.current_metrics.end_to_end_latency_ms == 2000.0
    
    def test_throughput_recording(self):
        """Test throughput recording functionality."""
        monitor = PerformanceMonitor()
        
        monitor.record_throughput(2.5)
        assert monitor.current_metrics.chunks_processed_per_second == 2.5
        assert len(monitor.throughput_history) == 1
        assert monitor.throughput_history[0] == 2.5
    
    def test_error_recording(self):
        """Test error recording functionality."""
        monitor = PerformanceMonitor()
        
        # Record different types of errors
        monitor.record_error("processing")
        assert monitor.current_metrics.processing_errors == 1
        
        monitor.record_error("audio_dropout")
        assert monitor.current_metrics.audio_dropouts == 1
        
        monitor.record_error("model_failure")
        assert monitor.current_metrics.model_failures == 1
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = PerformanceMonitor()
        
        # Add some test data
        latencies = [1000, 1200, 1500, 1800, 2000]
        throughputs = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        for latency, throughput in zip(latencies, throughputs):
            monitor.record_latency(latency, "processing")
            monitor.record_throughput(throughput)
        
        summary = monitor.get_performance_summary()
        
        assert 'latency_stats' in summary
        assert 'throughput_stats' in summary
        assert 'resource_stats' in summary
        assert 'error_counts' in summary
        
        # Check latency statistics
        assert summary['latency_stats']['mean'] == 1500.0
        assert summary['latency_stats']['max'] == 2000.0
        assert summary['latency_stats']['p95'] >= 1800.0
        
        # Check throughput statistics
        assert summary['throughput_stats']['mean'] == 2.0
        assert summary['throughput_stats']['min'] == 1.0
        assert summary['throughput_stats']['max'] == 3.0
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring(update_interval=0.1)
        assert monitor.monitoring_active
        assert monitor.monitoring_thread is not None
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active


class TestAdaptiveProcessor:
    """Test adaptive processing optimization."""
    
    def test_adaptive_processor_initialization(self):
        """Test adaptive processor initialization."""
        targets = PerformanceTarget(max_latency_ms=2000.0)
        strategy = OptimizationStrategy(enable_adaptive_processing=True)
        
        processor = AdaptiveProcessor(targets, strategy)
        
        assert processor.targets.max_latency_ms == 2000.0
        assert processor.strategy.enable_adaptive_processing
        assert processor.current_mode == ProcessingMode.TRIMODAL
    
    def test_mode_change_callbacks(self):
        """Test processing mode change callbacks."""
        targets = PerformanceTarget()
        strategy = OptimizationStrategy()
        processor = AdaptiveProcessor(targets, strategy)
        
        # Add callback
        mode_changes = []
        def mode_callback(new_mode):
            mode_changes.append(new_mode)
        
        processor.add_mode_change_callback(mode_callback)
        
        # Trigger mode change
        processor._change_processing_mode(ProcessingMode.BIMODAL)
        
        assert len(mode_changes) == 1
        assert mode_changes[0] == ProcessingMode.BIMODAL
        assert processor.get_current_mode() == ProcessingMode.BIMODAL
    
    def test_optimization_logic(self):
        """Test optimization decision logic."""
        targets = PerformanceTarget(max_latency_ms=2000.0, max_cpu_usage_percent=80.0)
        strategy = OptimizationStrategy(enable_adaptive_processing=True)
        processor = AdaptiveProcessor(targets, strategy)
        
        # Create mock performance monitor
        monitor = PerformanceMonitor()
        
        # Simulate high latency scenario
        for _ in range(10):
            monitor.record_latency(2500.0, "processing")  # Above threshold
        
        # Mock system metrics
        with patch.object(monitor, 'get_current_metrics') as mock_metrics:
            mock_metrics.return_value = SystemMetrics(
                timestamp=datetime.now(),
                audio_capture_latency_ms=100.0,
                processing_latency_ms=2500.0,
                end_to_end_latency_ms=2600.0,
                chunks_processed_per_second=1.0,
                sessions_active=1,
                average_audio_quality=0.8,
                average_model_confidence=0.9,
                cpu_usage_percent=85.0,  # Above threshold
                memory_usage_percent=60.0,
                gpu_memory_usage_percent=70.0,
                processing_errors=0,
                audio_dropouts=0,
                model_failures=0
            )
            
            # Force optimization interval to pass
            processor.last_optimization = datetime.now() - timedelta(seconds=15)
            
            # Run optimization
            processor.optimize_processing(monitor)
            
            # Should have downgraded from TRIMODAL
            assert processor.get_current_mode() in [
                ProcessingMode.BIMODAL, 
                ProcessingMode.AUDIO_ONLY, 
                ProcessingMode.FAST
            ]


class TestResourceManager:
    """Test resource management functionality."""
    
    def test_resource_manager_initialization(self):
        """Test resource manager initialization."""
        manager = ResourceManager()
        
        assert manager.memory_threshold_mb == 1024
        assert manager.gpu_memory_threshold_percent == 80.0
        assert len(manager.cleanup_callbacks) == 0
    
    def test_cleanup_callbacks(self):
        """Test resource cleanup callbacks."""
        manager = ResourceManager()
        
        # Add cleanup callback
        cleanup_called = []
        def cleanup_callback():
            cleanup_called.append(True)
        
        manager.add_cleanup_callback(cleanup_callback)
        
        # Trigger cleanup
        manager._cleanup_memory()
        
        assert len(cleanup_called) == 1
    
    def test_resource_status(self):
        """Test resource status reporting."""
        manager = ResourceManager()
        
        status = manager.get_resource_status()
        
        assert 'memory_total_gb' in status
        assert 'memory_used_gb' in status
        assert 'memory_percent' in status
        assert 'memory_available_gb' in status
        assert 'gpu_available' in status
        
        # Memory values should be reasonable
        assert status['memory_total_gb'] > 0
        assert 0 <= status['memory_percent'] <= 100


class TestPerformanceOptimizer:
    """Test main performance optimizer coordination."""
    
    def test_performance_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizer()
        
        assert optimizer.targets is not None
        assert optimizer.strategy is not None
        assert optimizer.performance_monitor is not None
        assert optimizer.adaptive_processor is not None
        assert optimizer.resource_manager is not None
    
    def test_optimization_lifecycle(self):
        """Test optimization start/stop lifecycle."""
        optimizer = PerformanceOptimizer()
        
        # Start optimization
        optimizer.start_optimization(optimization_interval=0.1)
        assert optimizer.optimization_active
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop optimization
        optimizer.stop_optimization()
        assert not optimizer.optimization_active
    
    def test_performance_recording(self):
        """Test performance metric recording."""
        optimizer = PerformanceOptimizer()
        
        # Record metrics
        optimizer.record_processing_latency(1500.0)
        optimizer.record_throughput(2.0)
        optimizer.record_error("processing")
        
        # Verify metrics were recorded
        metrics = optimizer.performance_monitor.get_current_metrics()
        assert metrics.processing_latency_ms == 1500.0
        assert metrics.chunks_processed_per_second == 2.0
        assert metrics.processing_errors == 1
    
    def test_performance_report(self):
        """Test comprehensive performance report."""
        optimizer = PerformanceOptimizer()
        
        # Add some test data
        optimizer.record_processing_latency(1200.0)
        optimizer.record_throughput(1.5)
        
        report = optimizer.get_performance_report()
        
        assert 'current_metrics' in report
        assert 'performance_summary' in report
        assert 'current_processing_mode' in report
        assert 'mode_history' in report
        assert 'resource_status' in report
        assert 'targets' in report
        
        # Verify report structure
        assert report['current_processing_mode'] == 'trimodal'
        assert report['targets']['max_latency_ms'] > 0


class TestKeyManager:
    """Test encryption key management."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_key_manager_initialization(self):
        """Test key manager initialization."""
        self.setUp()
        try:
            key_manager = KeyManager(key_storage_path=self.test_dir)
            
            assert key_manager.master_key is not None
            assert len(key_manager.master_key) == 44  # Fernet key length
            
            # Master key file should exist
            master_key_file = os.path.join(self.test_dir, "master.key")
            assert os.path.exists(master_key_file)
            
            # File should have secure permissions
            file_mode = oct(os.stat(master_key_file).st_mode)[-3:]
            assert file_mode == '600'
        finally:
            self.tearDown()
    
    def test_key_derivation(self):
        """Test key derivation functionality."""
        self.setUp()
        try:
            key_manager = KeyManager(key_storage_path=self.test_dir)
            
            # Derive keys for different purposes
            key1 = key_manager.derive_key("test_purpose_1")
            key2 = key_manager.derive_key("test_purpose_2")
            key3 = key_manager.derive_key("test_purpose_1")  # Same purpose
            
            assert len(key1) == 32  # 256-bit key
            assert len(key2) == 32
            assert key1 != key2  # Different purposes should give different keys
            assert key1 == key3  # Same purpose should give same key
        finally:
            self.tearDown()
    
    def test_data_key_management(self):
        """Test data key management."""
        self.setUp()
        try:
            key_manager = KeyManager(key_storage_path=self.test_dir)
            
            # Get data keys
            key1 = key_manager.get_data_key("session_data")
            key2 = key_manager.get_data_key("patient_data")
            key3 = key_manager.get_data_key("session_data")  # Same ID
            
            assert len(key1) == 32
            assert len(key2) == 32
            assert key1 != key2  # Different IDs should give different keys
            assert key1 == key3  # Same ID should give same key
        finally:
            self.tearDown()
    
    def test_key_rotation(self):
        """Test key rotation functionality."""
        self.setUp()
        try:
            key_manager = KeyManager(key_storage_path=self.test_dir)
            
            # Get initial key
            original_key = key_manager.get_data_key("test_key")
            
            # Rotate key
            new_key = key_manager.rotate_key("test_key")
            
            assert len(new_key) == 32
            assert new_key != original_key  # Should be different after rotation
            
            # Current key should be the new key
            current_key = key_manager.get_data_key("test_key")
            assert current_key == new_key
        finally:
            self.tearDown()


class TestDataEncryption:
    """Test data encryption functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.key_manager = KeyManager(key_storage_path=self.test_dir)
        self.encryption = DataEncryption(self.key_manager)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_string_encryption_decryption(self):
        """Test string encryption and decryption."""
        self.setUp()
        try:
            test_data = "This is sensitive clinical data that needs encryption."
            
            # Encrypt data
            encrypted_package = self.encryption.encrypt_data(test_data, "test_key")
            
            assert 'encrypted_data' in encrypted_package
            assert 'iv' in encrypted_package
            assert 'key_id' in encrypted_package
            assert 'hmac' in encrypted_package
            assert 'timestamp' in encrypted_package
            assert 'algorithm' in encrypted_package
            
            # Decrypt data
            decrypted_bytes = self.encryption.decrypt_data(encrypted_package)
            decrypted_string = decrypted_bytes.decode('utf-8')
            
            assert decrypted_string == test_data
        finally:
            self.tearDown()
    
    def test_bytes_encryption_decryption(self):
        """Test bytes encryption and decryption."""
        self.setUp()
        try:
            test_data = b"Binary data for encryption testing"
            
            # Encrypt data
            encrypted_package = self.encryption.encrypt_data(test_data, "test_key")
            
            # Decrypt data
            decrypted_bytes = self.encryption.decrypt_data(encrypted_package)
            
            assert decrypted_bytes == test_data
        finally:
            self.tearDown()
    
    def test_integrity_verification(self):
        """Test data integrity verification."""
        self.setUp()
        try:
            test_data = "Data for integrity testing"
            
            # Encrypt data
            encrypted_package = self.encryption.encrypt_data(test_data, "test_key")
            
            # Tamper with encrypted data
            tampered_package = encrypted_package.copy()
            tampered_package['hmac'] = 'invalid_hmac'
            
            # Decryption should fail with tampered data
            with pytest.raises(ValueError, match="Data integrity check failed"):
                self.encryption.decrypt_data(tampered_package)
        finally:
            self.tearDown()


class TestSecureStorage:
    """Test secure storage functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.storage = SecureStorage(storage_path=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_data_storage_retrieval(self):
        """Test data storage and retrieval."""
        self.setUp()
        try:
            test_data = {
                "patient_id": "patient_001",
                "session_data": {
                    "risk_scores": [0.3, 0.5, 0.7],
                    "timestamps": ["2024-01-01T10:00:00", "2024-01-01T10:01:00", "2024-01-01T10:02:00"]
                }
            }
            
            # Store data
            success = self.storage.store_data("test_session", test_data)
            assert success
            
            # Retrieve data
            retrieved_data = self.storage.retrieve_data("test_session")
            assert retrieved_data == test_data
        finally:
            self.tearDown()
    
    def test_nonexistent_data_retrieval(self):
        """Test retrieval of nonexistent data."""
        self.setUp()
        try:
            # Try to retrieve nonexistent data
            retrieved_data = self.storage.retrieve_data("nonexistent")
            assert retrieved_data is None
        finally:
            self.tearDown()
    
    def test_data_deletion(self):
        """Test secure data deletion."""
        self.setUp()
        try:
            test_data = {"test": "data"}
            
            # Store data
            self.storage.store_data("test_data", test_data)
            
            # Verify data exists
            retrieved_data = self.storage.retrieve_data("test_data")
            assert retrieved_data == test_data
            
            # Delete data
            success = self.storage.delete_data("test_data")
            assert success
            
            # Verify data is gone
            retrieved_data = self.storage.retrieve_data("test_data")
            assert retrieved_data is None
        finally:
            self.tearDown()
    
    def test_list_stored_data(self):
        """Test listing stored data."""
        self.setUp()
        try:
            # Store multiple data items
            self.storage.store_data("item1", {"data": 1})
            self.storage.store_data("item2", {"data": 2})
            self.storage.store_data("item3", {"data": 3})
            
            # List stored data
            data_list = self.storage.list_stored_data()
            
            assert len(data_list) == 3
            assert "item1" in data_list
            assert "item2" in data_list
            assert "item3" in data_list
        finally:
            self.tearDown()


class TestEncryptionManager:
    """Test main encryption manager."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.encryption_manager = EncryptionManager(storage_path=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_session_data_encryption(self):
        """Test session data encryption and decryption."""
        self.setUp()
        try:
            session_data = {
                "session_id": "session_001",
                "patient_id": "patient_001",
                "risk_scores": [0.3, 0.5, 0.7, 0.9],
                "timestamps": ["2024-01-01T10:00:00", "2024-01-01T10:01:00"]
            }
            
            # Encrypt session data
            success = self.encryption_manager.encrypt_session_data("session_001", session_data)
            assert success
            
            # Decrypt session data
            decrypted_data = self.encryption_manager.decrypt_session_data("session_001")
            assert decrypted_data == session_data
        finally:
            self.tearDown()
    
    def test_patient_data_encryption(self):
        """Test patient data encryption and decryption."""
        self.setUp()
        try:
            patient_data = {
                "patient_id": "patient_001",
                "demographics": {
                    "age_group": "25-35",
                    "language": "Tamil"
                },
                "clinical_history": {
                    "previous_sessions": 5,
                    "average_risk": 0.45
                }
            }
            
            # Encrypt patient data
            success = self.encryption_manager.encrypt_patient_data("patient_001", patient_data)
            assert success
            
            # Decrypt patient data
            decrypted_data = self.encryption_manager.decrypt_patient_data("patient_001")
            assert decrypted_data == patient_data
        finally:
            self.tearDown()
    
    def test_clinical_notes_encryption(self):
        """Test clinical notes encryption and decryption."""
        self.setUp()
        try:
            note_data = {
                "note_id": "note_001",
                "patient_id": "patient_001",
                "clinician": "dr_smith",
                "content": "Patient shows improvement in mood and speech patterns.",
                "timestamp": "2024-01-01T15:30:00",
                "risk_assessment": "moderate"
            }
            
            # Encrypt clinical notes
            success = self.encryption_manager.encrypt_clinical_notes("note_001", note_data)
            assert success
            
            # Decrypt clinical notes
            decrypted_data = self.encryption_manager.decrypt_clinical_notes("note_001")
            assert decrypted_data == note_data
        finally:
            self.tearDown()
    
    def test_secure_deletion(self):
        """Test secure deletion of encrypted data."""
        self.setUp()
        try:
            session_data = {"test": "data"}
            
            # Store and verify data
            self.encryption_manager.encrypt_session_data("test_session", session_data)
            retrieved_data = self.encryption_manager.decrypt_session_data("test_session")
            assert retrieved_data == session_data
            
            # Securely delete data
            success = self.encryption_manager.secure_delete_session("test_session")
            assert success
            
            # Verify data is gone
            retrieved_data = self.encryption_manager.decrypt_session_data("test_session")
            assert retrieved_data is None
        finally:
            self.tearDown()
    
    def test_encryption_status(self):
        """Test encryption status reporting."""
        self.setUp()
        try:
            # Add some test data
            self.encryption_manager.encrypt_session_data("session_001", {"test": "data1"})
            self.encryption_manager.encrypt_patient_data("patient_001", {"test": "data2"})
            self.encryption_manager.encrypt_clinical_notes("note_001", {"test": "data3"})
            
            status = self.encryption_manager.get_encryption_status()
            
            assert status['encryption_enabled']
            assert status['algorithm'] == 'AES-256-CBC'
            assert status['stored_data_count'] >= 3
            assert status['stored_sessions'] >= 1
            assert status['stored_patients'] >= 1
            assert status['stored_notes'] >= 1
        finally:
            self.tearDown()


def test_phase4_integration():
    """Test integration between Phase 4 components."""
    # Test performance optimization with encryption
    optimizer = PerformanceOptimizer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        encryption_manager = EncryptionManager(storage_path=temp_dir)
        
        # Record some performance metrics
        optimizer.record_processing_latency(1200.0)
        optimizer.record_throughput(2.0)
        
        # Encrypt some test data
        test_data = {"performance_test": "data"}
        success = encryption_manager.encrypt_session_data("perf_test", test_data)
        assert success
        
        # Get performance report
        report = optimizer.get_performance_report()
        assert 'current_metrics' in report
        
        # Get encryption status
        status = encryption_manager.get_encryption_status()
        assert status['encryption_enabled']
        
        # Verify data can be decrypted
        decrypted_data = encryption_manager.decrypt_session_data("perf_test")
        assert decrypted_data == test_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])