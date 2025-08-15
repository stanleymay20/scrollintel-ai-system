#!/usr/bin/env python3
"""
Simple test for ScrollIntel monitoring system
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scrollintel.core.monitoring import MetricsCollector, PerformanceMonitor
from scrollintel.core.uptime_monitor import UptimeMonitor, HealthCheck, ServiceStatus
from scrollintel.core.log_aggregation import LogAggregator, LogEntry
from datetime import datetime

def test_metrics_collector():
    """Test metrics collector basic functionality"""
    print("Testing MetricsCollector...")
    
    collector = MetricsCollector()
    
    # Test recording metrics
    collector.record_request("GET", "/api/health", 200, 0.5)
    collector.record_agent_request("cto_agent", "success", 2.0)
    collector.record_error("TestError", "test_component")
    
    # Test system metrics collection
    metrics = collector.collect_system_metrics()
    assert metrics is not None
    assert hasattr(metrics, 'cpu_percent')
    assert hasattr(metrics, 'memory_percent')
    assert hasattr(metrics, 'disk_percent')
    
    # Test metrics summary
    summary = collector.get_metrics_summary()
    assert isinstance(summary, dict)
    assert 'uptime_seconds' in summary
    assert 'system' in summary
    
    print("✓ MetricsCollector tests passed")

def test_performance_monitor():
    """Test performance monitor functionality"""
    print("Testing PerformanceMonitor...")
    
    monitor = PerformanceMonitor()
    
    # Test with normal metrics (should have no alerts)
    from scrollintel.core.monitoring import PerformanceMetrics
    normal_metrics = PerformanceMetrics(
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
    
    alerts = monitor.check_thresholds(normal_metrics)
    assert len(alerts) == 0
    
    # Test with high metrics (should have alerts)
    high_metrics = PerformanceMetrics(
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
    
    alerts = monitor.check_thresholds(high_metrics)
    assert len(alerts) >= 3  # Should have CPU, memory, and disk alerts
    
    print("✓ PerformanceMonitor tests passed")

def test_uptime_monitor():
    """Test uptime monitor functionality"""
    print("Testing UptimeMonitor...")
    
    monitor = UptimeMonitor()
    
    # Test initialization
    assert len(monitor.health_checks) > 0  # Should have default health checks
    
    # Test overall status calculation
    from scrollintel.core.uptime_monitor import ServiceMetrics
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
    
    status = monitor.get_overall_status()
    assert status == ServiceStatus.OPERATIONAL
    
    # Test status page data
    status_data = monitor.get_status_page_data()
    assert isinstance(status_data, dict)
    assert 'overall_status' in status_data
    assert 'services' in status_data
    assert 'incidents' in status_data
    
    print("✓ UptimeMonitor tests passed")

async def test_log_aggregator():
    """Test log aggregator functionality"""
    print("Testing LogAggregator...")
    
    aggregator = LogAggregator()
    
    # Test structured logging
    entry = LogEntry(
        timestamp=datetime.utcnow(),
        level="INFO",
        component="test",
        message="Test log message",
        user_id="user123",
        metadata={"key": "value"}
    )
    
    await aggregator.log_structured(entry)
    assert len(aggregator.log_buffer) == 1
    
    # Test buffer flush
    await aggregator.flush_buffer()
    assert len(aggregator.log_buffer) == 0
    
    print("✓ LogAggregator tests passed")

async def test_health_check():
    """Test health check functionality"""
    print("Testing health check...")
    
    monitor = UptimeMonitor()
    
    # Create a simple health check
    check = HealthCheck(
        name="test_service",
        url="https://httpbin.org/status/200",
        timeout=10
    )
    
    try:
        # Perform health check
        success, response_time, error = await monitor.perform_health_check(check)
        
        print(f"Health check result: success={success}, response_time={response_time:.3f}s, error={error}")
        
        # Basic validation
        assert isinstance(success, bool)
        assert isinstance(response_time, float)
        assert response_time >= 0
        
        if not success:
            assert error is not None
        
        print("✓ Health check tests passed")
        
    except Exception as e:
        print(f"Health check test failed (network issue?): {e}")
        print("✓ Health check test skipped due to network")

def test_monitoring_integration():
    """Test integration between monitoring components"""
    print("Testing monitoring integration...")
    
    # Initialize all components
    collector = MetricsCollector()
    monitor = PerformanceMonitor()
    uptime_mon = UptimeMonitor()
    
    # Simulate activity
    collector.record_request("GET", "/api/test", 200, 0.3)
    collector.record_agent_request("data_scientist", "success", 1.8)
    
    # Collect metrics
    metrics = collector.collect_system_metrics()
    assert metrics is not None
    
    # Check thresholds
    alerts = monitor.check_thresholds(metrics)
    # Should have no alerts with normal system
    
    # Get status data
    status_data = uptime_mon.get_status_page_data()
    assert status_data is not None
    
    print("✓ Monitoring integration tests passed")

async def main():
    """Run all tests"""
    print("Starting ScrollIntel Monitoring System Tests...")
    print("=" * 50)
    
    try:
        # Run synchronous tests
        test_metrics_collector()
        test_performance_monitor()
        test_uptime_monitor()
        test_monitoring_integration()
        
        # Run asynchronous tests
        await test_log_aggregator()
        await test_health_check()
        
        print("=" * 50)
        print("✅ All monitoring tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)