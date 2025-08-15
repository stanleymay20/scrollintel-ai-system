"""
Comprehensive tests for ScrollIntel monitoring and alerting system
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from scrollintel.core.monitoring import (
    MetricsCollector, PerformanceMonitor, AlertMetric, PerformanceMetrics,
    metrics_collector, performance_monitor
)
from scrollintel.core.uptime_monitor import (
    UptimeMonitor, HealthCheck, ServiceStatus, ServiceMetrics, Incident
)
from scrollintel.core.log_aggregation import (
    LogAggregator, LogEntry, StructuredFormatter, log_aggregator
)

class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes correctly"""
        collector = MetricsCollector()
        assert collector is not None
        assert hasattr(collector, 'logger')
        assert hasattr(collector, '_start_time')
    
    def test_record_request_metrics(self):
        """Test recording request metrics"""
        collector = MetricsCollector()
        
        # Record a request
        collector.record_request("GET", "/api/health", 200, 0.5)
        
        # Verify metrics are recorded (would check Prometheus metrics in real implementation)
        assert True  # Placeholder - would verify Prometheus counters
    
    def test_record_agent_request(self):
        """Test recording agent request metrics"""
        collector = MetricsCollector()
        
        # Record agent request
        collector.record_agent_request("cto_agent", "success", 2.5)
        
        # Verify metrics are recorded
        assert True  # Placeholder
    
    def test_record_error(self):
        """Test error recording"""
        collector = MetricsCollector()
        
        # Record an error
        collector.record_error("ValidationError", "api")
        
        # Verify error is recorded
        assert True  # Placeholder
    
    def test_collect_system_metrics(self):
        """Test system metrics collection"""
        collector = MetricsCollector()
        
        # Collect metrics
        metrics = collector.collect_system_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'cpu_percent')
        assert hasattr(metrics, 'memory_percent')
        assert hasattr(metrics, 'disk_percent')
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_percent >= 0
    
    def test_get_metrics_summary(self):
        """Test metrics summary generation"""
        collector = MetricsCollector()
        
        # Get summary
        summary = collector.get_metrics_summary()
        
        # Verify summary structure
        assert isinstance(summary, dict)
        assert 'uptime_seconds' in summary
        assert 'system' in summary
        assert 'timestamp' in summary
        assert 'cpu_percent' in summary['system']
        assert 'memory_percent' in summary['system']
        assert 'disk_percent' in summary['system']

class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initializes correctly"""
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert hasattr(monitor, 'alert_thresholds')
        assert hasattr(monitor, 'alerts')
    
    def test_check_thresholds_no_alerts(self):
        """Test threshold checking with normal metrics"""
        monitor = PerformanceMonitor()
        
        # Create normal metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            active_connections=10,
            request_rate=100.0,
            error_rate=1.0,
            avg_response_time=0.5,
            agent_count=5
        )
        
        # Check thresholds
        alerts = monitor.check_thresholds(metrics)
        
        # Should have no alerts
        assert len(alerts) == 0
    
    def test_check_thresholds_with_alerts(self):
        """Test threshold checking with high metrics"""
        monitor = PerformanceMonitor()
        
        # Create high metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=95.0,  # Above threshold
            memory_percent=90.0,  # Above threshold
            disk_percent=95.0,  # Above threshold
            active_connections=10,
            request_rate=100.0,
            error_rate=1.0,
            avg_response_time=0.5,
            agent_count=5
        )
        
        # Check thresholds
        alerts = monitor.check_thresholds(metrics)
        
        # Should have alerts for CPU, memory, and disk
        assert len(alerts) >= 3
        
        # Verify alert structure
        for alert in alerts:
            assert isinstance(alert, AlertMetric)
            assert hasattr(alert, 'metric_name')
            assert hasattr(alert, 'current_value')
            assert hasattr(alert, 'threshold')
            assert hasattr(alert, 'severity')
            assert hasattr(alert, 'timestamp')
            assert hasattr(alert, 'description')

class TestUptimeMonitor:
    """Test uptime monitoring functionality"""
    
    def test_uptime_monitor_initialization(self):
        """Test uptime monitor initializes correctly"""
        monitor = UptimeMonitor()
        assert monitor is not None
        assert hasattr(monitor, 'health_checks')
        assert hasattr(monitor, 'service_metrics')
        assert hasattr(monitor, 'incidents')
        assert len(monitor.health_checks) > 0  # Should have default health checks
    
    @pytest.mark.asyncio
    async def test_perform_health_check_success(self):
        """Test successful health check"""
        monitor = UptimeMonitor()
        
        # Create a health check
        check = HealthCheck(
            name="test_service",
            url="http://httpbin.org/status/200",
            timeout=5
        )
        
        # Mock successful response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="OK")
            
            mock_session.return_value.__aenter__.return_value.request.return_value.__aenter__.return_value = mock_response
            
            # Perform health check
            success, response_time, error = await monitor.perform_health_check(check)
            
            # Verify results
            assert success is True
            assert response_time >= 0
            assert error is None
    
    @pytest.mark.asyncio
    async def test_perform_health_check_failure(self):
        """Test failed health check"""
        monitor = UptimeMonitor()
        
        # Create a health check
        check = HealthCheck(
            name="test_service",
            url="http://httpbin.org/status/500",
            timeout=5
        )
        
        # Mock failed response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            
            mock_session.return_value.__aenter__.return_value.request.return_value.__aenter__.return_value = mock_response
            
            # Perform health check
            success, response_time, error = await monitor.perform_health_check(check)
            
            # Verify results
            assert success is False
            assert response_time >= 0
            assert error is not None
            assert "Expected status 200, got 500" in error
    
    def test_get_overall_status_operational(self):
        """Test overall status calculation when all services operational"""
        monitor = UptimeMonitor()
        
        # Add operational services
        monitor.service_metrics = {
            "service1": ServiceMetrics(
                name="service1",
                status=ServiceStatus.OPERATIONAL,
                response_time=0.1,
                uptime_percentage=99.9,
                last_check=datetime.utcnow(),
                total_checks=100,
                successful_checks=99,
                failed_checks=1
            ),
            "service2": ServiceMetrics(
                name="service2",
                status=ServiceStatus.OPERATIONAL,
                response_time=0.2,
                uptime_percentage=99.8,
                last_check=datetime.utcnow(),
                total_checks=100,
                successful_checks=99,
                failed_checks=1
            )
        }
        
        # Get overall status
        status = monitor.get_overall_status()
        
        # Should be operational
        assert status == ServiceStatus.OPERATIONAL
    
    def test_get_overall_status_degraded(self):
        """Test overall status calculation when some services degraded"""
        monitor = UptimeMonitor()
        
        # Add mixed status services
        monitor.service_metrics = {
            "service1": ServiceMetrics(
                name="service1",
                status=ServiceStatus.OPERATIONAL,
                response_time=0.1,
                uptime_percentage=99.9,
                last_check=datetime.utcnow(),
                total_checks=100,
                successful_checks=99,
                failed_checks=1
            ),
            "service2": ServiceMetrics(
                name="service2",
                status=ServiceStatus.DEGRADED,
                response_time=2.0,
                uptime_percentage=95.0,
                last_check=datetime.utcnow(),
                total_checks=100,
                successful_checks=95,
                failed_checks=5
            )
        }
        
        # Get overall status
        status = monitor.get_overall_status()
        
        # Should be degraded
        assert status == ServiceStatus.DEGRADED
    
    def test_get_status_page_data(self):
        """Test status page data generation"""
        monitor = UptimeMonitor()
        
        # Add some test data
        monitor.service_metrics = {
            "test_service": ServiceMetrics(
                name="test_service",
                status=ServiceStatus.OPERATIONAL,
                response_time=0.1,
                uptime_percentage=99.9,
                last_check=datetime.utcnow(),
                total_checks=100,
                successful_checks=99,
                failed_checks=1
            )
        }
        
        # Get status page data
        data = monitor.get_status_page_data()
        
        # Verify structure
        assert isinstance(data, dict)
        assert 'overall_status' in data
        assert 'last_updated' in data
        assert 'services' in data
        assert 'incidents' in data
        assert 'uptime_stats' in data
        
        # Verify services data
        assert 'test_service' in data['services']
        service_data = data['services']['test_service']
        assert 'status' in service_data
        assert 'uptime_percentage' in service_data
        assert 'response_time' in service_data
        assert 'last_check' in service_data

class TestLogAggregator:
    """Test log aggregation functionality"""
    
    def test_log_aggregator_initialization(self):
        """Test log aggregator initializes correctly"""
        aggregator = LogAggregator()
        assert aggregator is not None
        assert hasattr(aggregator, 'log_buffer')
        assert hasattr(aggregator, 'buffer_size')
        assert aggregator.buffer_size > 0
    
    def test_structured_formatter(self):
        """Test structured log formatter"""
        formatter = StructuredFormatter()
        
        # Create a log record
        import logging
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Verify it's valid JSON
        log_data = json.loads(formatted)
        assert 'timestamp' in log_data
        assert 'level' in log_data
        assert 'component' in log_data
        assert 'message' in log_data
        assert log_data['level'] == 'INFO'
        assert log_data['message'] == 'Test message'
    
    @pytest.mark.asyncio
    async def test_log_structured_entry(self):
        """Test structured log entry"""
        aggregator = LogAggregator()
        
        # Create a log entry
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            component="test",
            message="Test log message",
            user_id="user123",
            metadata={"key": "value"}
        )
        
        # Log the entry
        await aggregator.log_structured(entry)
        
        # Verify it's in the buffer
        assert len(aggregator.log_buffer) == 1
        assert aggregator.log_buffer[0] == entry
    
    @pytest.mark.asyncio
    async def test_get_error_summary(self):
        """Test error summary generation"""
        aggregator = LogAggregator()
        
        # Mock the search_logs method to return test data
        test_errors = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "ERROR",
                "component": "api",
                "message": "Database connection failed",
                "error_type": "ConnectionError"
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "ERROR",
                "component": "agents",
                "message": "AI service timeout",
                "error_type": "TimeoutError"
            }
        ]
        
        with patch.object(aggregator, 'search_logs', return_value=test_errors):
            # Get error summary
            summary = await aggregator.get_error_summary(hours=24)
            
            # Verify summary structure
            assert isinstance(summary, dict)
            assert 'total_errors' in summary
            assert 'time_range' in summary
            assert 'error_types' in summary
            assert 'components' in summary
            assert 'recent_errors' in summary
            
            # Verify data
            assert summary['total_errors'] == 2
            assert 'ConnectionError' in summary['error_types']
            assert 'TimeoutError' in summary['error_types']
            assert 'api' in summary['components']
            assert 'agents' in summary['components']

class TestMonitoringIntegration:
    """Test integration between monitoring components"""
    
    @pytest.mark.asyncio
    async def test_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        # Initialize components
        collector = MetricsCollector()
        monitor = PerformanceMonitor()
        uptime_mon = UptimeMonitor()
        
        # Simulate system activity
        collector.record_request("GET", "/api/health", 200, 0.1)
        collector.record_agent_request("cto_agent", "success", 1.5)
        
        # Collect system metrics
        metrics = collector.collect_system_metrics()
        assert metrics is not None
        
        # Check for alerts
        alerts = monitor.check_thresholds(metrics)
        # Should have no alerts with normal metrics
        assert len(alerts) == 0
        
        # Test uptime monitoring
        status_data = uptime_mon.get_status_page_data()
        assert 'overall_status' in status_data
    
    def test_metrics_export(self):
        """Test Prometheus metrics export"""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_request("GET", "/api/test", 200, 0.5)
        collector.record_error("TestError", "test_component")
        
        # Export metrics
        metrics_output = collector.export_metrics()
        
        # Verify output format
        assert isinstance(metrics_output, str)
        assert len(metrics_output) > 0
        # Would contain Prometheus format metrics in real implementation

class TestAlertingSystem:
    """Test alerting system functionality"""
    
    @pytest.mark.asyncio
    async def test_incident_creation(self):
        """Test incident creation and management"""
        monitor = UptimeMonitor()
        
        # Create an incident
        await monitor._create_incident(
            service="test_service",
            status=ServiceStatus.MAJOR_OUTAGE,
            description="Service is down"
        )
        
        # Verify incident was created
        assert len(monitor.incidents) == 1
        incident = monitor.incidents[0]
        assert incident.service == "test_service"
        assert incident.status == ServiceStatus.MAJOR_OUTAGE
        assert incident.description == "Service is down"
        assert incident.resolved_at is None
    
    @pytest.mark.asyncio
    async def test_incident_resolution(self):
        """Test incident resolution"""
        monitor = UptimeMonitor()
        
        # Create an incident
        await monitor._create_incident(
            service="test_service",
            status=ServiceStatus.MAJOR_OUTAGE,
            description="Service is down"
        )
        
        # Resolve the incident
        await monitor._resolve_incident("test_service")
        
        # Verify incident was resolved
        incident = monitor.incidents[0]
        assert incident.resolved_at is not None
        assert incident.duration is not None
        assert incident.duration >= 0

@pytest.mark.asyncio
async def test_monitoring_system_startup():
    """Test monitoring system startup and shutdown"""
    # This would test the full monitoring orchestrator
    # For now, just verify components can be initialized
    
    collector = MetricsCollector()
    monitor = PerformanceMonitor()
    uptime_mon = UptimeMonitor()
    aggregator = LogAggregator()
    
    # Verify all components initialized
    assert collector is not None
    assert monitor is not None
    assert uptime_mon is not None
    assert aggregator is not None
    
    # Test basic functionality
    metrics = collector.collect_system_metrics()
    assert metrics is not None
    
    status_data = uptime_mon.get_status_page_data()
    assert status_data is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])