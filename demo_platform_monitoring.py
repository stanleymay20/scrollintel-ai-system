"""Demo script for AI Data Readiness Platform monitoring and metrics system."""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from ai_data_readiness.core.platform_monitor import get_platform_monitor
from ai_data_readiness.core.resource_optimizer import get_resource_optimizer
from ai_data_readiness.models.base_models import DatasetStatus


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_metrics(title: str, metrics: Dict[str, Any]):
    """Print formatted metrics."""
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        elif isinstance(value, datetime):
            print(f"  {key}: {value.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"  {key}: {value}")


def simulate_workload():
    """Simulate platform workload for demonstration."""
    monitor = get_platform_monitor()
    
    print("Simulating platform workload...")
    
    # Simulate dataset processing operations
    operations = []
    for i in range(5):
        op_id = monitor.start_operation(f"dataset_processing_{i}", {
            "dataset_size_mb": 100 + i * 50,
            "operation_type": "quality_assessment"
        })
        operations.append(op_id)
        
        # Update dataset counts
        monitor.update_dataset_count(DatasetStatus.PROCESSING, 1)
        
        # Simulate some processing time
        time.sleep(0.1)
    
    # Complete some operations successfully
    for i, op_id in enumerate(operations[:3]):
        monitor.complete_operation(op_id, success=True)
        monitor.update_dataset_count(DatasetStatus.PROCESSING, -1)
        monitor.update_dataset_count(DatasetStatus.READY, 1)
        monitor.record_data_processed((100 + i * 50) / 1024)  # Convert to GB
    
    # Simulate some failures
    for op_id in operations[3:]:
        monitor.complete_operation(op_id, success=False, error_message="Simulated processing error")
        monitor.update_dataset_count(DatasetStatus.PROCESSING, -1)
        monitor.update_dataset_count(DatasetStatus.ERROR, 1)
    
    # Record API requests
    for _ in range(50):
        monitor.record_api_request()
    
    print("Workload simulation completed.")


def demo_system_monitoring():
    """Demonstrate system monitoring capabilities."""
    print_section("SYSTEM MONITORING DEMO")
    
    monitor = get_platform_monitor()
    
    print("Collecting system metrics...")
    try:
        system_metrics = monitor.collect_system_metrics()
        print_metrics("Current System Metrics", system_metrics.to_dict())
    except Exception as e:
        print(f"Note: System metrics collection requires psutil: {e}")
        print("Using mock data for demonstration...")
        
        # Create mock system metrics for demo
        from ai_data_readiness.core.platform_monitor import SystemMetrics
        system_metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=65.5,
            memory_percent=72.3,
            memory_used_gb=11.6,
            memory_available_gb=4.4,
            disk_usage_percent=58.2,
            disk_free_gb=250.8,
            network_bytes_sent=1024000,
            network_bytes_recv=2048000,
            active_connections=42
        )
        monitor.system_metrics.append(system_metrics)
        print_metrics("Mock System Metrics", system_metrics.to_dict())


def demo_platform_monitoring():
    """Demonstrate platform-specific monitoring."""
    print_section("PLATFORM MONITORING DEMO")
    
    monitor = get_platform_monitor()
    
    # Simulate workload first
    simulate_workload()
    
    print("\nCollecting platform metrics...")
    platform_metrics = monitor.collect_platform_metrics()
    print_metrics("Current Platform Metrics", platform_metrics.to_dict())


def demo_performance_tracking():
    """Demonstrate performance tracking capabilities."""
    print_section("PERFORMANCE TRACKING DEMO")
    
    monitor = get_platform_monitor()
    
    print("Tracking various operations...")
    
    # Track different types of operations
    operations = [
        ("data_ingestion", {"source": "database", "size_mb": 500}),
        ("quality_assessment", {"dataset_id": "test-123", "dimensions": 6}),
        ("bias_analysis", {"protected_attributes": ["age", "gender"]}),
        ("feature_engineering", {"features_generated": 25}),
    ]
    
    for op_name, metadata in operations:
        op_id = monitor.start_operation(op_name, metadata)
        
        # Simulate processing time
        processing_time = 0.1 + (hash(op_name) % 100) / 1000  # Variable processing time
        time.sleep(processing_time)
        
        # Complete operation (90% success rate)
        success = hash(op_name) % 10 != 0
        error_msg = None if success else f"Simulated error in {op_name}"
        monitor.complete_operation(op_id, success=success, error_message=error_msg)
    
    print(f"Tracked {len(operations)} operations")
    print(f"Performance metrics collected: {len(monitor.performance_metrics)}")
    
    # Show recent performance metrics
    if monitor.performance_metrics:
        recent_metric = monitor.performance_metrics[-1]
        print_metrics("Latest Performance Metric", recent_metric.to_dict())


def demo_resource_optimization():
    """Demonstrate resource optimization capabilities."""
    print_section("RESOURCE OPTIMIZATION DEMO")
    
    optimizer = get_resource_optimizer()
    
    print("Tracking resource usage...")
    try:
        resource_usage = optimizer.track_resource_usage()
        print_metrics("Current Resource Usage", resource_usage.to_dict())
    except Exception as e:
        print(f"Note: Resource tracking requires psutil: {e}")
        print("Using mock data for demonstration...")
        
        # Create mock resource usage for demo
        from ai_data_readiness.core.resource_optimizer import ResourceUsage
        resource_usage = ResourceUsage(
            timestamp=datetime.utcnow(),
            cpu_cores_used=5.2,
            memory_mb_used=12288.0,
            disk_io_mb_per_sec=15.5,
            network_io_mb_per_sec=8.2,
            active_threads=28,
            active_processes=156
        )
        optimizer.resource_history.append(resource_usage)
        print_metrics("Mock Resource Usage", resource_usage.to_dict())
    
    # Generate optimization recommendations
    print("\nGenerating optimization recommendations...")
    optimizer.generate_recommendations()
    
    recommendations = optimizer.get_optimization_recommendations()
    if recommendations:
        print(f"\nFound {len(recommendations)} optimization recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']} ({rec['category'].upper()})")
            print(f"   Priority: {rec['priority']}")
            print(f"   Description: {rec['description']}")
            print(f"   Action: {rec['action']}")
            print(f"   Estimated Impact: {rec['estimated_improvement']}")
    else:
        print("No optimization recommendations at this time.")
    
    # Show resource efficiency metrics
    try:
        efficiency = optimizer.get_resource_efficiency_metrics()
        if efficiency:
            print_metrics("Resource Efficiency Metrics", efficiency)
    except Exception:
        print("Resource efficiency calculation requires more data points.")


def demo_health_monitoring():
    """Demonstrate health monitoring and alerting."""
    print_section("HEALTH MONITORING DEMO")
    
    monitor = get_platform_monitor()
    
    print("Checking platform health status...")
    health_status = monitor.get_health_status()
    
    print(f"\nOverall Health Status: {health_status['status'].upper()}")
    
    if health_status['status'] == 'healthy':
        print("✅ All systems operating normally")
        if 'message' in health_status:
            print(f"   {health_status['message']}")
    else:
        print("⚠️  Issues detected:")
        for issue in health_status.get('issues', []):
            print(f"   • {issue}")
    
    # Show detailed metrics if available
    if 'system_metrics' in health_status:
        print_metrics("System Health Details", health_status['system_metrics'])
    
    if 'platform_metrics' in health_status:
        print_metrics("Platform Health Details", health_status['platform_metrics'])


def demo_alert_system():
    """Demonstrate alert system capabilities."""
    print_section("ALERT SYSTEM DEMO")
    
    monitor = get_platform_monitor()
    
    # Set up alert callback
    alerts_received = []
    
    def alert_callback(severity: str, alert_data: Dict[str, Any]):
        alerts_received.append((severity, alert_data))
        print(f"🚨 ALERT: {severity.upper()}")
        print(f"   Timestamp: {alert_data['timestamp']}")
        print(f"   Issues: {', '.join(alert_data.get('issues', []))}")
    
    monitor.add_alert_callback(alert_callback)
    
    print("Alert system configured. Checking for alerts...")
    
    # Trigger alert check
    monitor._check_alerts()
    
    if alerts_received:
        print(f"\n{len(alerts_received)} alerts triggered during demo")
    else:
        print("No alerts triggered - system is healthy")
    
    print("\nAlert thresholds configured:")
    for metric, threshold in monitor.alert_thresholds.items():
        print(f"  {metric}: {threshold}")


def demo_metrics_history():
    """Demonstrate metrics history and reporting."""
    print_section("METRICS HISTORY & REPORTING DEMO")
    
    monitor = get_platform_monitor()
    
    print("Retrieving metrics history...")
    history = monitor.get_metrics_history(hours=1)
    
    print(f"System metrics collected: {len(history['system_metrics'])}")
    print(f"Platform metrics collected: {len(history['platform_metrics'])}")
    print(f"Performance metrics collected: {len(history['performance_metrics'])}")
    
    if history['system_metrics']:
        latest_system = history['system_metrics'][-1]
        print_metrics("Latest System Metrics", latest_system)
    
    if history['platform_metrics']:
        latest_platform = history['platform_metrics'][-1]
        print_metrics("Latest Platform Metrics", latest_platform)


def demo_capacity_planning():
    """Demonstrate capacity planning capabilities."""
    print_section("CAPACITY PLANNING DEMO")
    
    optimizer = get_resource_optimizer()
    
    print("Analyzing capacity requirements...")
    
    # Calculate optimal batch sizes for different scenarios
    scenarios = [
        ("Small dataset", 50.0, 4000.0),    # 50MB data, 4GB available memory
        ("Medium dataset", 500.0, 8000.0),  # 500MB data, 8GB available memory
        ("Large dataset", 2000.0, 16000.0), # 2GB data, 16GB available memory
    ]
    
    print("\nOptimal batch size recommendations:")
    for scenario_name, data_size, available_memory in scenarios:
        batch_size = optimizer.get_optimal_batch_size(data_size, available_memory)
        print(f"  {scenario_name}: {batch_size:,} rows")
        print(f"    (Data: {data_size}MB, Memory: {available_memory/1024:.1f}GB)")
    
    # Show resource utilization trends
    if len(optimizer.resource_history) >= 2:
        print("\nResource utilization trends:")
        recent_usage = list(optimizer.resource_history)[-5:]  # Last 5 measurements
        
        if len(recent_usage) >= 2:
            cpu_trend = optimizer._analyze_trend([u.cpu_cores_used for u in recent_usage])
            memory_trend = optimizer._analyze_trend([u.memory_mb_used for u in recent_usage])
            
            print(f"  CPU trend: {'↗️ Increasing' if cpu_trend > 0.1 else '↘️ Decreasing' if cpu_trend < -0.1 else '➡️ Stable'}")
            print(f"  Memory trend: {'↗️ Increasing' if memory_trend > 10 else '↘️ Decreasing' if memory_trend < -10 else '➡️ Stable'}")


def demo_monitoring_lifecycle():
    """Demonstrate monitoring system lifecycle."""
    print_section("MONITORING LIFECYCLE DEMO")
    
    monitor = get_platform_monitor()
    optimizer = get_resource_optimizer()
    
    print("Starting continuous monitoring...")
    
    # Start monitoring with short intervals for demo
    monitor.start_monitoring(interval_seconds=2)
    optimizer.start_optimization(interval_seconds=5)
    
    print("Monitoring active. Collecting data for 10 seconds...")
    
    # Let it run for a bit
    time.sleep(10)
    
    print("Stopping monitoring...")
    monitor.stop_monitoring()
    optimizer.stop_optimization()
    
    print("Monitoring stopped.")
    
    # Show collected data
    print(f"\nData collected during monitoring:")
    print(f"  System metrics: {len(monitor.system_metrics)}")
    print(f"  Platform metrics: {len(monitor.platform_metrics)}")
    print(f"  Resource usage: {len(optimizer.resource_history)}")


def demo_metrics_export():
    """Demonstrate metrics export functionality."""
    print_section("METRICS EXPORT DEMO")
    
    monitor = get_platform_monitor()
    
    print("Exporting metrics to file...")
    
    import tempfile
    import os
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        export_file = f.name
    
    try:
        monitor.export_metrics(export_file, hours=1)
        
        # Read and display export info
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        print(f"✅ Metrics exported to: {export_file}")
        print(f"Export timestamp: {export_data.get('export_timestamp', 'N/A')}")
        print(f"System metrics: {len(export_data.get('system_metrics', []))}")
        print(f"Platform metrics: {len(export_data.get('platform_metrics', []))}")
        print(f"Performance metrics: {len(export_data.get('performance_metrics', []))}")
        
        # Show file size
        file_size = os.path.getsize(export_file)
        print(f"Export file size: {file_size:,} bytes")
        
    finally:
        # Cleanup
        if os.path.exists(export_file):
            os.unlink(export_file)
            print(f"Cleaned up export file: {export_file}")


async def main():
    """Run the complete monitoring demo."""
    print("🚀 AI Data Readiness Platform - Monitoring & Metrics Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive monitoring capabilities")
    print("of the AI Data Readiness Platform, including:")
    print("• System resource monitoring")
    print("• Platform-specific metrics")
    print("• Performance tracking")
    print("• Resource optimization")
    print("• Health monitoring & alerting")
    print("• Capacity planning")
    print("• Metrics export")
    
    try:
        # Run all demo sections
        demo_system_monitoring()
        demo_platform_monitoring()
        demo_performance_tracking()
        demo_resource_optimization()
        demo_health_monitoring()
        demo_alert_system()
        demo_metrics_history()
        demo_capacity_planning()
        demo_monitoring_lifecycle()
        demo_metrics_export()
        
        print_section("DEMO COMPLETED SUCCESSFULLY")
        print("✅ All monitoring components demonstrated successfully!")
        print("\nKey Features Demonstrated:")
        print("• Real-time system and platform metrics collection")
        print("• Performance tracking with operation-level granularity")
        print("• Intelligent resource optimization recommendations")
        print("• Proactive health monitoring and alerting")
        print("• Capacity planning and trend analysis")
        print("• Comprehensive metrics export capabilities")
        print("\nThe monitoring system is now ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure cleanup
        try:
            monitor = get_platform_monitor()
            optimizer = get_resource_optimizer()
            monitor.stop_monitoring()
            optimizer.stop_optimization()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())