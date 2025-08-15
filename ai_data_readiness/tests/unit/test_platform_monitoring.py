"""Unit tests for platform monitoring system."""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from ai_data_readiness.core.platform_monitor import (
    PlatformMonitor, SystemMetrics, PlatformMetrics, PerformanceMetrics,
    get_platform_monitor
)
from ai_data_readiness.core.resource_optimizer import (
    ResourceOptimizer, ResourceUsage, OptimizationRecommendation,
    get_resource_optimizer
)
from ai_data_readiness.models.base_models import DatasetStatus


class TestSystemMetrics:
    """Test SystemMetrics data class."""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation and serialization."""
        metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=75.5,
            memory_percent=60.2,
            memory_used_gb=8.5,
            memory_available_gb=7.5,
            disk_usage_percent=45.0,
            disk_free_gb=100.0,
            network_bytes_sent=1024000,
            network_bytes_recv=2048000,
            active_connections=25
        )
        
        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 60.2
        assert metrics.active_connections == 25
        
        # Test serialization
        data = metrics.to_dict()
        assert 'timestamp' in data
        assert data['cpu_percent'] == 75.5
        assert data['memory_percent'] == 60.2


class TestPlatformMetrics:
    """Test PlatformMetrics data class."""
    
    def test_platform_metrics_creation(self):
        """Test PlatformMetrics creation and serialization."""
        metrics = PlatformMetrics(
            timestamp=datetime.utcnow(),
            active_datasets=10,
            processing_datasets=3,
            failed_datasets=1,
            total_data_processed_gb=150.5,
            avg_processing_time_seconds=45.2,
            quality_assessments_completed=8,
            bias_analyses_completed=5,
            api_requests_per_minute=120.0,
            error_rate_percent=2.5
        )
        
        assert metrics.active_datasets == 10
        assert metrics.processing_datasets == 3
        assert metrics.error_rate_percent == 2.5
        
        # Test serialization
        data = metrics.to_dict()
        assert data['active_datasets'] == 10
        assert data['error_rate_percent'] == 2.5


class TestPerformanceMetrics:
    """Test PerformanceMetrics data class."""
    
    def test_performance_metrics_lifecycle(self):
        """Test PerformanceMetrics lifecycle."""
        start_time = datetime.utcnow()
        metrics = PerformanceMetrics(
            operation_name="data_ingestion",
            start_time=start_time,
            metadata={"dataset_size": "100MB"}
        )
        
        assert metrics.operation_name == "data_ingestion"
        assert metrics.success is True
        assert metrics.end_time is None
        assert metrics.duration_seconds is None
        
        # Complete the operation
        time.sleep(0.1)  # Small delay to ensure duration > 0
        metrics.complete(success=True)
        
        assert metrics.end_time is not None
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds > 0
        assert metrics.success is True
        
        # Test serialization
        data = metrics.to_dict()
        assert data['operation_name'] == "data_ingestion"
        assert data['success'] is True
        assert data['duration_seconds'] is not None


class TestPlatformMonitor:
    """Test PlatformMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a PlatformMonitor instance for testing."""
        return PlatformMonitor(metrics_retention_hours=1)
    
    @patch('ai_data_readiness.core.platform_monitor.psutil')
    def test_collect_system_metrics(self, mock_psutil, monitor):
        """Test system metrics collection."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 75.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=60.0,
            used=8 * 1024**3,  # 8GB
            available=8 * 1024**3  # 8GB
        )
        mock_psutil.disk_usage.return_value = Mock(
            total=1000 * 1024**3,  # 1TB
            used=500 * 1024**3,    # 500GB
            free=500 * 1024**3     # 500GB
        )
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1024000,
            bytes_recv=2048000
        )
        mock_psutil.net_connections.return_value = [Mock()] * 25
        
        metrics = monitor.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 75.0
        assert metrics.memory_percent == 60.0
        assert metrics.active_connections == 25
        assert len(monitor.system_metrics) == 1
    
    def test_collect_platform_metrics(self, monitor):
        """Test platform metrics collection."""
        # Set up some counters
        monitor.operation_counters['active_datasets'] = 10
        monitor.operation_counters['processing_datasets'] = 3
        monitor.error_counters['dataset_processing_failed'] = 1
        monitor.api_request_counter = 120
        
        metrics = monitor.collect_platform_metrics()
        
        assert isinstance(metrics, PlatformMetrics)
        assert metrics.active_datasets == 10
        assert metrics.processing_datasets == 3
        assert metrics.failed_datasets == 1
        assert len(monitor.platform_metrics) == 1
    
    def test_operation_tracking(self, monitor):
        """Test operation tracking functionality."""
        # Start an operation
        operation_id = monitor.start_operation("test_operation", {"test": "data"})
        
        assert operation_id in monitor.active_operations
        assert monitor.operation_counters["test_operation"] == 1
        
        # Complete the operation
        monitor.complete_operation(operation_id, success=True)
        
        assert operation_id not in monitor.active_operations
        assert len(monitor.performance_metrics) == 1
        
        # Test failed operation
        failed_id = monitor.start_operation("failed_operation")
        monitor.complete_operation(failed_id, success=False, error_message="Test error")
        
        assert monitor.error_counters["failed_operation"] == 1
    
    def test_dataset_count_tracking(self, monitor):
        """Test dataset count tracking."""
        monitor.update_dataset_count(DatasetStatus.READY, 5)
        monitor.update_dataset_count(DatasetStatus.PROCESSING, 2)
        monitor.update_dataset_count(DatasetStatus.ERROR, 1)
        
        assert monitor.operation_counters['active_datasets'] == 5
        assert monitor.operation_counters['processing_datasets'] == 2
        assert monitor.operation_counters['failed_datasets'] == 1
    
    def test_health_status_healthy(self, monitor):
        """Test health status when system is healthy."""
        # Add healthy metrics
        healthy_system = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            active_connections=10
        )
        monitor.system_metrics.append(healthy_system)
        
        healthy_platform = PlatformMetrics(
            timestamp=datetime.utcnow(),
            active_datasets=10,
            processing_datasets=2,
            failed_datasets=0,
            total_data_processed_gb=100.0,
            avg_processing_time_seconds=30.0,
            quality_assessments_completed=10,
            bias_analyses_completed=5,
            api_requests_per_minute=50.0,
            error_rate_percent=1.0
        )
        monitor.platform_metrics.append(healthy_platform)
        
        health = monitor.get_health_status()
        assert health['status'] == 'healthy'
    
    def test_health_status_warning(self, monitor):
        """Test health status when system has warnings."""
        # Add warning-level metrics
        warning_system = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=85.0,  # Above warning threshold
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            active_connections=10
        )
        monitor.system_metrics.append(warning_system)
        
        warning_platform = PlatformMetrics(
            timestamp=datetime.utcnow(),
            active_datasets=10,
            processing_datasets=2,
            failed_datasets=0,
            total_data_processed_gb=100.0,
            avg_processing_time_seconds=30.0,
            quality_assessments_completed=10,
            bias_analyses_completed=5,
            api_requests_per_minute=50.0,
            error_rate_percent=1.0
        )
        monitor.platform_metrics.append(warning_platform)
        
        health = monitor.get_health_status()
        assert health['status'] == 'warning'
        assert len(health['issues']) > 0
    
    def test_monitoring_thread_lifecycle(self, monitor):
        """Test monitoring thread start/stop."""
        assert not monitor.monitoring_active
        
        # Start monitoring
        monitor.start_monitoring(interval_seconds=1)
        assert monitor.monitoring_active
        assert monitor.monitoring_thread is not None
        
        # Wait a bit for thread to start
        time.sleep(0.1)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active
    
    def test_alert_callbacks(self, monitor):
        """Test alert callback functionality."""
        callback_called = []
        
        def test_callback(severity, alert_data):
            callback_called.append((severity, alert_data))
        
        monitor.add_alert_callback(test_callback)
        
        # Add critical metrics to trigger alert
        critical_system = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=95.0,  # Critical level
            memory_percent=95.0,  # Critical level
            memory_used_gb=8.0,
            memory_available_gb=1.0,
            disk_usage_percent=95.0,  # Critical level
            disk_free_gb=10.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            active_connections=10
        )
        monitor.system_metrics.append(critical_system)
        
        critical_platform = PlatformMetrics(
            timestamp=datetime.utcnow(),
            active_datasets=10,
            processing_datasets=2,
            failed_datasets=0,
            total_data_processed_gb=100.0,
            avg_processing_time_seconds=30.0,
            quality_assessments_completed=10,
            bias_analyses_completed=5,
            api_requests_per_minute=50.0,
            error_rate_percent=15.0  # Critical error rate
        )
        monitor.platform_metrics.append(critical_platform)
        
        # Trigger alert check
        monitor._check_alerts()
        
        assert len(callback_called) > 0
        assert callback_called[0][0] == 'critical'
    
    def test_metrics_export(self, monitor, tmp_path):
        """Test metrics export functionality."""
        # Add some test metrics
        monitor.system_metrics.append(SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            active_connections=10
        ))
        
        # Export metrics
        export_file = tmp_path / "test_metrics.json"
        monitor.export_metrics(str(export_file), hours=1)
        
        assert export_file.exists()
        
        # Verify export content
        import json
        with open(export_file) as f:
            data = json.load(f)
        
        assert 'export_timestamp' in data
        assert 'system_metrics' in data
        assert 'health_status' in data


class TestResourceOptimizer:
    """Test ResourceOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a ResourceOptimizer instance for testing."""
        return ResourceOptimizer()
    
    @patch('ai_data_readiness.core.resource_optimizer.psutil')
    def test_track_resource_usage(self, mock_psutil, optimizer):
        """Test resource usage tracking."""
        # Mock psutil functions
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=8 * 1024 * 1024 * 1024)  # 8GB
        mock_psutil.Process.return_value = mock_process
        mock_psutil.cpu_percent.return_value = 75.0
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.disk_io_counters.return_value = Mock()
        mock_psutil.net_io_counters.return_value = Mock()
        mock_psutil.pids.return_value = list(range(100))
        
        with patch('threading.active_count', return_value=10):
            usage = optimizer.track_resource_usage()
        
        assert isinstance(usage, ResourceUsage)
        assert usage.cpu_cores_used == 6.0  # 75% of 8 cores
        assert usage.memory_mb_used == 8192.0  # 8GB in MB
        assert usage.active_threads == 10
        assert len(optimizer.resource_history) == 1
    
    def test_optimization_recommendations(self, optimizer):
        """Test optimization recommendation generation."""
        # Add some resource usage history
        for i in range(10):
            usage = ResourceUsage(
                timestamp=datetime.utcnow() - timedelta(minutes=i),
                cpu_cores_used=4.0 + i * 0.1,  # Increasing trend
                memory_mb_used=8000.0 + i * 50,  # Increasing trend
                disk_io_mb_per_sec=10.0,
                network_io_mb_per_sec=5.0,
                active_threads=20,
                active_processes=100
            )
            optimizer.resource_history.append(usage)
        
        optimizer.generate_recommendations()
        
        recommendations = optimizer.get_optimization_recommendations()
        assert len(recommendations) > 0
        
        # Check for trend-based recommendations
        cpu_recs = [r for r in recommendations if r['category'] == 'cpu']
        memory_recs = [r for r in recommendations if r['category'] == 'memory']
        
        assert len(cpu_recs) > 0 or len(memory_recs) > 0
    
    def test_resource_efficiency_metrics(self, optimizer):
        """Test resource efficiency calculation."""
        # Add resource usage data
        for i in range(10):
            usage = ResourceUsage(
                timestamp=datetime.utcnow() - timedelta(minutes=i),
                cpu_cores_used=4.0,
                memory_mb_used=8000.0,
                disk_io_mb_per_sec=10.0,
                network_io_mb_per_sec=5.0,
                active_threads=20,
                active_processes=100
            )
            optimizer.resource_history.append(usage)
        
        with patch('ai_data_readiness.core.resource_optimizer.psutil.cpu_count', return_value=8):
            with patch('ai_data_readiness.core.resource_optimizer.psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB
                
                efficiency = optimizer.get_resource_efficiency_metrics()
        
        assert 'cpu_efficiency' in efficiency
        assert 'memory_efficiency' in efficiency
        assert 'resource_utilization_score' in efficiency
        assert 0 <= efficiency['cpu_efficiency'] <= 1
        assert 0 <= efficiency['memory_efficiency'] <= 1
    
    def test_optimal_batch_size_calculation(self, optimizer):
        """Test optimal batch size calculation."""
        # Test with different memory scenarios
        batch_size_8gb = optimizer.get_optimal_batch_size(100.0, 8000.0)
        batch_size_32gb = optimizer.get_optimal_batch_size(100.0, 32000.0)
        
        assert batch_size_8gb >= 100  # Minimum batch size
        assert batch_size_32gb >= batch_size_8gb  # More memory = larger batches
        assert batch_size_32gb <= 10000  # Maximum batch size
    
    def test_optimization_lifecycle(self, optimizer):
        """Test optimization thread lifecycle."""
        assert not optimizer.optimization_active
        
        # Start optimization
        optimizer.start_optimization(interval_seconds=1)
        assert optimizer.optimization_active
        assert optimizer.optimization_thread is not None
        
        # Wait a bit for thread to start
        time.sleep(0.1)
        
        # Stop optimization
        optimizer.stop_optimization()
        assert not optimizer.optimization_active


class TestGlobalInstances:
    """Test global instance functions."""
    
    def test_get_platform_monitor_singleton(self):
        """Test that get_platform_monitor returns singleton."""
        monitor1 = get_platform_monitor()
        monitor2 = get_platform_monitor()
        
        assert monitor1 is monitor2
        assert isinstance(monitor1, PlatformMonitor)
    
    def test_get_resource_optimizer_singleton(self):
        """Test that get_resource_optimizer returns singleton."""
        optimizer1 = get_resource_optimizer()
        optimizer2 = get_resource_optimizer()
        
        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, ResourceOptimizer)


class TestIntegration:
    """Integration tests for monitoring components."""
    
    def test_monitor_optimizer_integration(self):
        """Test integration between monitor and optimizer."""
        monitor = get_platform_monitor()
        optimizer = get_resource_optimizer()
        
        # Start an operation in monitor
        operation_id = monitor.start_operation("test_integration")
        
        # Track resource usage in optimizer
        with patch('ai_data_readiness.core.resource_optimizer.psutil') as mock_psutil:
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(rss=8 * 1024 * 1024 * 1024)
            mock_psutil.Process.return_value = mock_process
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.cpu_count.return_value = 8
            mock_psutil.disk_io_counters.return_value = Mock()
            mock_psutil.net_io_counters.return_value = Mock()
            mock_psutil.pids.return_value = list(range(50))
            
            with patch('threading.active_count', return_value=5):
                optimizer.track_resource_usage()
        
        # Complete operation
        monitor.complete_operation(operation_id, success=True)
        
        # Verify both systems have data
        assert len(monitor.performance_metrics) > 0
        assert len(optimizer.resource_history) > 0
        
        # Test health status includes both systems
        health = monitor.get_health_status()
        assert 'status' in health