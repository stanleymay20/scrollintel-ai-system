#!/usr/bin/env python3
"""
Demo script for Advanced Recovery and Self-Healing System.
Demonstrates autonomous system repair, intelligent dependency management,
self-optimizing performance tuning, and predictive maintenance.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from scrollintel.core.advanced_recovery_system import (
    advanced_recovery_system,
    PerformanceMetrics,
    RecoveryAction
)


async def demo_health_monitoring():
    """Demonstrate health monitoring capabilities."""
    print("🏥 HEALTH MONITORING DEMO")
    print("=" * 50)
    
    # Check health of all system components
    nodes = ['database', 'ai_services', 'visualization_engine', 'file_system', 'web_server']
    
    for node in nodes:
        health_score = await advanced_recovery_system.perform_health_check(node)
        status = advanced_recovery_system._get_health_status(health_score)
        
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'critical': '🔴',
            'failing': '💀'
        }.get(status, '❓')
        
        print(f"{status_emoji} {node}: {health_score:.2f} ({status})")
    
    print()


async def demo_autonomous_repair():
    """Demonstrate autonomous system repair."""
    print("🔧 AUTONOMOUS REPAIR DEMO")
    print("=" * 50)
    
    # Simulate a failing service
    print("Simulating ai_services failure...")
    ai_node = advanced_recovery_system.dependency_graph['ai_services']
    original_health = ai_node.health_score
    ai_node.health_score = 0.3  # Simulate degraded state
    ai_node.failure_count = 2
    
    try:
        print(f"AI Services health before repair: {ai_node.health_score:.2f}")
        
        # Trigger autonomous repair
        print("Triggering autonomous repair...")
        success = await advanced_recovery_system.autonomous_system_repair('ai_services')
        
        if success:
            print("✅ Autonomous repair successful!")
            print(f"AI Services health after repair: {ai_node.health_score:.2f}")
        else:
            print("❌ Autonomous repair failed")
        
        print(f"Recovery attempts: {ai_node.recovery_attempts}")
        
    finally:
        # Restore original state
        ai_node.health_score = original_health
        ai_node.failure_count = 0
        ai_node.recovery_attempts = 0
    
    print()


async def demo_dependency_management():
    """Demonstrate intelligent dependency management."""
    print("🕸️ DEPENDENCY MANAGEMENT DEMO")
    print("=" * 50)
    
    print("Running intelligent dependency management...")
    status = await advanced_recovery_system.intelligent_dependency_management()
    
    print("Dependency Status:")
    for node_name, node_status in status.items():
        health_score = node_status['health_score']
        node_status_str = node_status['status']
        failure_count = node_status['failure_count']
        
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'critical': '🔴',
            'failing': '💀'
        }.get(node_status_str, '❓')
        
        print(f"  {status_emoji} {node_name}: {health_score:.2f} ({node_status_str})")
        if failure_count > 0:
            print(f"    Failures: {failure_count}")
        if node_status.get('recovery_attempted'):
            recovery_success = node_status.get('recovery_successful', False)
            print(f"    Recovery: {'✅ Success' if recovery_success else '❌ Failed'}")
    
    print()


async def demo_performance_optimization():
    """Demonstrate self-optimizing performance tuning."""
    print("⚡ PERFORMANCE OPTIMIZATION DEMO")
    print("=" * 50)
    
    # Add some performance history to enable optimization
    print("Adding performance history...")
    for i in range(15):
        metrics = PerformanceMetrics(
            cpu_usage=60 + i * 2,  # Increasing CPU usage
            memory_usage=50 + i * 1.5,  # Increasing memory usage
            disk_io=10.0 + i * 0.5,
            network_io=5.0 + i * 0.3,
            response_time=1.0 + i * 0.1,  # Increasing response time
            throughput=100.0 - i * 2,  # Decreasing throughput
            error_rate=0.01 + i * 0.002
        )
        advanced_recovery_system.performance_history.append(metrics)
    
    print("Running self-optimizing performance tuning...")
    result = await advanced_recovery_system.self_optimizing_performance_tuning()
    
    print("Current Performance Metrics:")
    current_metrics = result['current_metrics']
    print(f"  CPU Usage: {current_metrics['cpu_usage']:.1f}%")
    print(f"  Memory Usage: {current_metrics['memory_usage']:.1f}%")
    print(f"  Response Time: {current_metrics['response_time']:.2f}s")
    print(f"  Throughput: {current_metrics['throughput']:.1f} req/s")
    print(f"  Error Rate: {current_metrics['error_rate']:.3f}")
    
    print("\nOptimizations Applied:")
    optimizations = result['optimizations_applied']
    if optimizations:
        for opt_type, success in optimizations.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"  {opt_type}: {status}")
    else:
        print("  No optimizations needed")
    
    print("\nPerformance Trends:")
    trends = result['performance_trend']
    for metric, trend in trends.items():
        trend_emoji = {
            'increasing': '📈',
            'decreasing': '📉',
            'stable': '➡️'
        }.get(trend, '❓')
        print(f"  {metric}: {trend_emoji} {trend}")
    
    print()


async def demo_predictive_maintenance():
    """Demonstrate predictive maintenance."""
    print("🔮 PREDICTIVE MAINTENANCE DEMO")
    print("=" * 50)
    
    # Add performance history with concerning trends
    print("Adding performance history with concerning trends...")
    for i in range(25):
        metrics = PerformanceMetrics(
            cpu_usage=70 + i * 1.5,  # Rapidly increasing CPU
            memory_usage=60 + i * 1.2,  # Increasing memory
            disk_io=10.0,
            network_io=5.0,
            response_time=1.0 + i * 0.08,  # Increasing response time
            throughput=100.0,
            error_rate=0.01 + i * 0.001
        )
        advanced_recovery_system.performance_history.append(metrics)
    
    print("Running predictive maintenance...")
    result = await advanced_recovery_system.predictive_maintenance()
    
    print("Predicted Issues:")
    issues = result['predicted_issues']
    if issues:
        for issue in issues:
            severity_emoji = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🔴',
                'critical': '💀'
            }.get(issue['severity'], '❓')
            
            print(f"  {severity_emoji} {issue['type']} ({issue['severity']})")
            print(f"    Estimated time: {issue['estimated_time']}")
            print(f"    Confidence: {issue['confidence']:.1%}")
    else:
        print("  No issues predicted")
    
    print("\nScheduled Maintenance Tasks:")
    tasks = result['scheduled_tasks']
    if tasks:
        for task in tasks:
            priority_emoji = "🔴" if task['priority'] >= 8 else "🟡" if task['priority'] >= 6 else "🟢"
            print(f"  {priority_emoji} {task['task_id']} ({task['type']})")
            print(f"    Priority: {task['priority']}/10")
            print(f"    Duration: {task['estimated_duration']}")
            if task['scheduled_time']:
                print(f"    Scheduled: {task['scheduled_time']}")
    else:
        print("  No tasks scheduled")
    
    print("\nPreventive Actions Taken:")
    actions = result['preventive_actions']
    if actions:
        for action in actions:
            print(f"  ✅ {action}")
    else:
        print("  No preventive actions needed")
    
    print()


async def demo_recovery_patterns():
    """Demonstrate recovery patterns and strategies."""
    print("🎯 RECOVERY PATTERNS DEMO")
    print("=" * 50)
    
    print("Available Recovery Patterns:")
    patterns = advanced_recovery_system.recovery_patterns
    
    for pattern_name, actions in patterns.items():
        print(f"\n📋 {pattern_name}:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action.value}")
    
    print("\nOptimization Rules:")
    rules = list(advanced_recovery_system.optimization_rules.keys())
    for rule in rules:
        print(f"  🔧 {rule}")
    
    print()


async def demo_system_status():
    """Demonstrate comprehensive system status."""
    print("📊 SYSTEM STATUS DEMO")
    print("=" * 50)
    
    status = advanced_recovery_system.get_system_status()
    
    print(f"Timestamp: {status['timestamp']}")
    print(f"Auto Recovery: {'✅ Enabled' if status['auto_recovery_enabled'] else '❌ Disabled'}")
    print(f"Monitoring: {'✅ Active' if status['monitoring_active'] else '❌ Inactive'}")
    
    print("\nDependency Health:")
    for node_name, health_info in status['dependency_health'].items():
        health_score = health_info['health_score']
        failure_count = health_info['failure_count']
        recovery_attempts = health_info['recovery_attempts']
        
        health_emoji = "✅" if health_score > 0.8 else "⚠️" if health_score > 0.5 else "🔴"
        print(f"  {health_emoji} {node_name}: {health_score:.2f}")
        if failure_count > 0:
            print(f"    Failures: {failure_count}")
        if recovery_attempts > 0:
            print(f"    Recovery attempts: {recovery_attempts}")
    
    print("\nMaintenance Tasks:")
    tasks_info = status['maintenance_tasks']
    print(f"  Total: {tasks_info['total']}")
    print(f"  Completed: {tasks_info['completed']}")
    print(f"  Pending: {tasks_info['pending']}")
    
    print("\nPerformance Thresholds:")
    thresholds = status['performance_thresholds']
    for metric, threshold in thresholds.items():
        print(f"  {metric}: {threshold}")
    
    print()


async def demo_failure_simulation():
    """Demonstrate failure simulation and recovery."""
    print("💥 FAILURE SIMULATION DEMO")
    print("=" * 50)
    
    # Simulate database failure
    print("Simulating database failure...")
    db_node = advanced_recovery_system.dependency_graph['database']
    original_health = db_node.health_score
    original_failures = db_node.failure_count
    
    try:
        # Set database to critical state
        db_node.health_score = 0.1
        db_node.failure_count = 5
        
        print(f"Database health: {db_node.health_score:.2f} (critical)")
        
        # Run dependency management to detect and handle failure
        print("Running dependency management...")
        status = await advanced_recovery_system.intelligent_dependency_management()
        
        # Check if recovery was attempted
        db_status = status.get('database', {})
        if db_status.get('recovery_attempted'):
            recovery_success = db_status.get('recovery_successful', False)
            print(f"Recovery attempted: {'✅ Success' if recovery_success else '❌ Failed'}")
        
        # Show impact on dependent services
        print("\nImpact on dependent services:")
        dependent_services = ['ai_services', 'visualization_engine']
        for service in dependent_services:
            if service in status:
                service_health = status[service]['health_score']
                print(f"  {service}: {service_health:.2f}")
        
    finally:
        # Restore original state
        db_node.health_score = original_health
        db_node.failure_count = original_failures
        db_node.recovery_attempts = 0
    
    print()


async def demo_performance_trends():
    """Demonstrate performance trend analysis."""
    print("📈 PERFORMANCE TRENDS DEMO")
    print("=" * 50)
    
    # Clear existing history
    advanced_recovery_system.performance_history.clear()
    
    # Add performance history with different trends
    print("Adding performance history with various trends...")
    
    # Phase 1: Stable performance
    for i in range(10):
        metrics = PerformanceMetrics(
            cpu_usage=50 + (i % 3 - 1) * 2,  # Stable around 50%
            memory_usage=60 + (i % 2) * 2,   # Stable around 60%
            disk_io=10.0,
            network_io=5.0,
            response_time=1.0 + (i % 2) * 0.1,  # Stable around 1.0s
            throughput=100.0,
            error_rate=0.01
        )
        advanced_recovery_system.performance_history.append(metrics)
    
    # Phase 2: Degrading performance
    for i in range(10):
        metrics = PerformanceMetrics(
            cpu_usage=50 + i * 3,  # Increasing CPU
            memory_usage=60 + i * 2,  # Increasing memory
            disk_io=10.0 + i * 0.5,
            network_io=5.0,
            response_time=1.0 + i * 0.15,  # Increasing response time
            throughput=100.0 - i * 3,  # Decreasing throughput
            error_rate=0.01 + i * 0.005  # Increasing errors
        )
        advanced_recovery_system.performance_history.append(metrics)
    
    # Analyze trends
    trends = advanced_recovery_system._analyze_performance_trends()
    
    print("Performance Trend Analysis:")
    for trend_type, should_optimize in trends.items():
        status = "🔴 Needs optimization" if should_optimize else "✅ Stable"
        print(f"  {trend_type}: {status}")
    
    # Get trend summary
    summary = advanced_recovery_system._get_performance_trend_summary()
    
    print("\nTrend Summary:")
    for metric, trend in summary.items():
        trend_emoji = {
            'increasing': '📈',
            'decreasing': '📉',
            'stable': '➡️'
        }.get(trend, '❓')
        print(f"  {metric}: {trend_emoji} {trend}")
    
    print()


async def main():
    """Run all demos."""
    print("🚀 ADVANCED RECOVERY AND SELF-HEALING SYSTEM DEMO")
    print("=" * 60)
    print("Demonstrating autonomous system repair, intelligent dependency")
    print("management, self-optimizing performance tuning, and predictive maintenance.")
    print("=" * 60)
    print()
    
    # Run all demo functions
    demos = [
        demo_health_monitoring,
        demo_autonomous_repair,
        demo_dependency_management,
        demo_performance_optimization,
        demo_predictive_maintenance,
        demo_recovery_patterns,
        demo_system_status,
        demo_failure_simulation,
        demo_performance_trends
    ]
    
    for demo in demos:
        try:
            await demo()
            await asyncio.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print()
    
    print("🎉 DEMO COMPLETE!")
    print("The Advanced Recovery System provides:")
    print("✅ Autonomous system repair with minimal user impact")
    print("✅ Intelligent dependency management with automatic failover")
    print("✅ Self-optimizing performance tuning based on usage patterns")
    print("✅ Predictive maintenance with proactive issue resolution")
    print("✅ Comprehensive monitoring and health management")
    print("✅ Cascade failure prevention")
    print("✅ Real-time performance optimization")
    print("✅ Intelligent recovery pattern matching")


if __name__ == "__main__":
    asyncio.run(main())