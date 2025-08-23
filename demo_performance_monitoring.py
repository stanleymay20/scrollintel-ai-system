#!/usr/bin/env python3
"""
Demo script for performance monitoring and optimization system.
"""
import asyncio
import time
from datetime import datetime
from scrollintel.engines.performance_monitoring_engine import PerformanceMonitoringEngine

async def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("🚀 Performance Monitoring Demo")
    print("=" * 50)
    
    # Initialize monitoring engine
    engine = PerformanceMonitoringEngine()
    
    # Test data
    pipeline_id = "demo-pipeline-001"
    execution_id = f"exec-{int(time.time())}"
    
    try:
        # 1. Start monitoring
        print(f"\n📊 Starting monitoring for pipeline: {pipeline_id}")
        start_result = await engine.start_monitoring(pipeline_id, execution_id)
        print(f"✅ Monitoring started: {start_result}")
        
        metrics_id = start_result.get("metrics_id", 1)
        
        # 2. Simulate some monitoring cycles
        print(f"\n📈 Collecting metrics for 3 cycles...")
        for i in range(3):
            print(f"   Cycle {i+1}/3...")
            metrics_result = await engine.collect_metrics(metrics_id)
            print(f"   CPU: {metrics_result.get('cpu_usage', 'N/A')}%, "
                  f"Memory: {metrics_result.get('memory_usage', 'N/A')}%")
            await asyncio.sleep(1)  # Short delay between collections
        
        # 3. Stop monitoring
        print(f"\n🛑 Stopping monitoring...")
        stop_result = await engine.stop_monitoring(metrics_id)
        print(f"✅ Monitoring stopped: {stop_result}")
        
        # 4. Get dashboard data
        print(f"\n📋 Generating dashboard data...")
        dashboard_data = await engine.get_performance_dashboard_data(
            pipeline_id=pipeline_id,
            time_range_hours=1
        )
        
        print(f"📊 Dashboard Summary:")
        summary = dashboard_data.get("summary", {})
        print(f"   Total Executions: {summary.get('total_executions', 0)}")
        print(f"   Avg Duration: {summary.get('avg_duration_seconds', 0):.2f}s")
        print(f"   Avg CPU Usage: {summary.get('avg_cpu_usage', 0):.1f}%")
        print(f"   Error Rate: {summary.get('error_rate', 0)*100:.2f}%")
        
        # 5. Test auto-tuning
        print(f"\n⚙️ Testing auto-tuning...")
        tuning_result = await engine.apply_auto_tuning(pipeline_id)
        print(f"🔧 Auto-tuning result: {tuning_result.get('status', 'unknown')}")
        
        if tuning_result.get("actions"):
            print("   Recommended actions:")
            for action in tuning_result["actions"]:
                print(f"   - {action.get('action', 'unknown')}: {action.get('reason', 'no reason')}")
        
        print(f"\n✨ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_optimization_recommendations():
    """Demonstrate optimization recommendation logic."""
    print(f"\n🎯 Optimization Recommendations Demo")
    print("=" * 50)
    
    # Test scenarios for different optimization types
    scenarios = [
        {
            "name": "High CPU Usage",
            "cpu_usage": 95.0,
            "memory_mb": 2048.0,
            "duration": 300.0,
            "error_rate": 0.001
        },
        {
            "name": "High Memory Usage", 
            "cpu_usage": 60.0,
            "memory_mb": 12288.0,  # 12GB
            "duration": 300.0,
            "error_rate": 0.001
        },
        {
            "name": "Long Execution Time",
            "cpu_usage": 60.0,
            "memory_mb": 2048.0,
            "duration": 7200.0,  # 2 hours
            "error_rate": 0.001
        },
        {
            "name": "High Error Rate",
            "cpu_usage": 60.0,
            "memory_mb": 2048.0,
            "duration": 300.0,
            "error_rate": 0.08  # 8%
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   CPU: {scenario['cpu_usage']}%")
        print(f"   Memory: {scenario['memory_mb']} MB")
        print(f"   Duration: {scenario['duration']} seconds")
        print(f"   Error Rate: {scenario['error_rate']*100:.1f}%")
        
        # Simulate recommendation logic
        recommendations = []
        
        if scenario['cpu_usage'] > 80:
            priority = 'high' if scenario['cpu_usage'] > 90 else 'medium'
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': priority,
                'improvement': '25%'
            })
        
        if scenario['memory_mb'] > 8192:  # > 8GB
            recommendations.append({
                'type': 'memory_optimization', 
                'priority': 'medium',
                'improvement': '30%'
            })
        
        if scenario['duration'] > 3600:  # > 1 hour
            recommendations.append({
                'type': 'performance_optimization',
                'priority': 'high', 
                'improvement': '40%'
            })
        
        if scenario['error_rate'] > 0.01:  # > 1%
            recommendations.append({
                'type': 'reliability_optimization',
                'priority': 'critical',
                'improvement': '50%'
            })
        
        if recommendations:
            print(f"   🔧 Recommendations:")
            for rec in recommendations:
                print(f"     - {rec['type']} ({rec['priority']} priority): {rec['improvement']} improvement")
        else:
            print(f"   ✅ No optimizations needed")

def demo_sla_monitoring():
    """Demonstrate SLA monitoring and alerting."""
    print(f"\n🚨 SLA Monitoring Demo")
    print("=" * 50)
    
    # SLA thresholds
    thresholds = {
        'cpu_usage': 85.0,
        'memory_usage': 90.0,
        'error_rate': 0.05,
        'latency_ms': 10000
    }
    
    print(f"📏 SLA Thresholds:")
    for metric, threshold in thresholds.items():
        unit = '%' if 'usage' in metric or 'rate' in metric else 'ms'
        if 'rate' in metric:
            threshold_display = f"{threshold*100:.1f}%"
        else:
            threshold_display = f"{threshold}{unit}"
        print(f"   {metric}: {threshold_display}")
    
    # Test cases
    test_cases = [
        {"cpu": 95.0, "memory": 85.0, "error_rate": 0.02, "latency": 5000},
        {"cpu": 75.0, "memory": 95.0, "error_rate": 0.08, "latency": 15000},
        {"cpu": 70.0, "memory": 80.0, "error_rate": 0.001, "latency": 3000}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}:")
        print(f"   CPU: {case['cpu']}%")
        print(f"   Memory: {case['memory']}%") 
        print(f"   Error Rate: {case['error_rate']*100:.1f}%")
        print(f"   Latency: {case['latency']}ms")
        
        violations = []
        
        if case['cpu'] > thresholds['cpu_usage']:
            severity = 'critical' if case['cpu'] > 95 else 'warning'
            violations.append(f"CPU usage violation ({severity})")
        
        if case['memory'] > thresholds['memory_usage']:
            severity = 'critical' if case['memory'] > 95 else 'warning'
            violations.append(f"Memory usage violation ({severity})")
        
        if case['error_rate'] > thresholds['error_rate']:
            severity = 'critical' if case['error_rate'] > 0.1 else 'warning'
            violations.append(f"Error rate violation ({severity})")
        
        if case['latency'] > thresholds['latency_ms']:
            violations.append(f"Latency violation (warning)")
        
        if violations:
            print(f"   🚨 SLA Violations:")
            for violation in violations:
                print(f"     - {violation}")
        else:
            print(f"   ✅ All SLAs met")

if __name__ == "__main__":
    print("🎯 ScrollIntel Performance Monitoring System Demo")
    print("=" * 60)
    
    # Run demos
    try:
        # Async demo
        asyncio.run(demo_performance_monitoring())
        
        # Sync demos
        demo_optimization_recommendations()
        demo_sla_monitoring()
        
        print(f"\n🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()