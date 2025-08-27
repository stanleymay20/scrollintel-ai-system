"""
Demo: Real-time Analytics Dashboard System
Demonstrates real-time data processing, alerting, and dashboard updates
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

from scrollintel.core.realtime_data_processor import RealtimeDataProcessor, StreamType, ProcessingRule
from scrollintel.core.intelligent_alerting_system import IntelligentAlertingSystem, ThresholdRule, AnomalyRule, AlertSeverity
from scrollintel.core.realtime_websocket_manager import RealtimeWebSocketManager, MessageType
from scrollintel.core.notification_system import NotificationSystem, NotificationRule, NotificationChannel, NotificationPriority
from scrollintel.core.data_quality_monitor import DataQualityMonitor, DataQualityRule, DataQualityIssueType, QualityCheckSeverity

class RealtimeDashboardDemo:
    """Comprehensive demo of real-time dashboard system"""
    
    def __init__(self):
        self.processor = None
        self.alerting = None
        self.websocket_manager = None
        self.notification_system = None
        self.quality_monitor = None
        
        # Demo data generators
        self.metrics_generators = {
            'cpu_usage': lambda: random.uniform(20, 95),
            'memory_usage': lambda: random.uniform(30, 85),
            'response_time': lambda: random.lognormal(3, 0.5),
            'error_rate': lambda: random.uniform(0, 10),
            'throughput': lambda: random.uniform(100, 1000),
            'user_count': lambda: random.randint(50, 500),
            'revenue': lambda: random.uniform(1000, 10000),
            'conversion_rate': lambda: random.uniform(2, 8)
        }
        
        self.running = False
    
    async def initialize(self):
        """Initialize all real-time systems"""
        print("🚀 Initializing Real-time Dashboard System...")
        
        # Initialize components
        self.processor = RealtimeDataProcessor()
        self.alerting = IntelligentAlertingSystem()
        self.websocket_manager = RealtimeWebSocketManager()
        self.notification_system = NotificationSystem()
        self.quality_monitor = DataQualityMonitor()
        
        await self.processor.initialize()
        await self.alerting.initialize()
        await self.websocket_manager.initialize()
        await self.notification_system.initialize()
        await self.quality_monitor.initialize()
        
        print("✅ All systems initialized successfully!")
        
        # Configure systems
        await self._configure_systems()
    
    async def _configure_systems(self):
        """Configure all systems with demo rules and settings"""
        print("⚙️  Configuring systems...")
        
        # Configure data processing rules
        self.processor.add_processing_rule(ProcessingRule(
            name="high_cpu_alert",
            condition="data.get('metric_name') == 'cpu_usage' and data.get('value', 0) > 80",
            action="alert",
            parameters={
                'alert_type': 'critical',
                'title': 'High CPU Usage Alert',
                'description': 'CPU usage exceeded 80%'
            }
        ))
        
        self.processor.add_processing_rule(ProcessingRule(
            name="revenue_milestone",
            condition="data.get('metric_name') == 'revenue' and data.get('value', 0) > 8000",
            action="notify",
            parameters={
                'notification_type': 'milestone',
                'title': 'Revenue Milestone Reached',
                'description': 'Daily revenue exceeded $8,000'
            }
        ))
        
        # Configure alerting rules
        self.alerting.add_threshold_rule(ThresholdRule(
            name="cpu_threshold",
            metric_name="cpu_usage",
            operator=">",
            threshold_value=75.0,
            severity=AlertSeverity.HIGH,
            cooldown_minutes=2,
            description="CPU usage is too high"
        ))
        
        self.alerting.add_threshold_rule(ThresholdRule(
            name="memory_threshold",
            metric_name="memory_usage",
            operator=">",
            threshold_value=80.0,
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=1,
            description="Memory usage is critical"
        ))
        
        self.alerting.add_threshold_rule(ThresholdRule(
            name="error_rate_threshold",
            metric_name="error_rate",
            operator=">",
            threshold_value=5.0,
            severity=AlertSeverity.MEDIUM,
            cooldown_minutes=3,
            description="Error rate is elevated"
        ))
        
        # Configure anomaly detection
        self.alerting.add_anomaly_rule(AnomalyRule(
            name="response_time_anomaly",
            metric_name="response_time",
            sensitivity=0.15,
            min_samples=50,
            description="Detect anomalous response times"
        ))
        
        self.alerting.add_anomaly_rule(AnomalyRule(
            name="throughput_anomaly",
            metric_name="throughput",
            sensitivity=0.2,
            min_samples=30,
            description="Detect throughput anomalies"
        ))
        
        # Configure notifications
        self.notification_system.configure_channel(NotificationChannel.EMAIL, {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'from_email': 'alerts@scrollintel.com'
        })
        
        self.notification_system.add_rule(NotificationRule(
            name="critical_alerts",
            conditions={"severity": "critical"},
            channels=[NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET],
            priority=NotificationPriority.URGENT,
            template_name="alert",
            recipients=["admin@scrollintel.com"],
            rate_limit=5
        ))
        
        self.notification_system.add_rule(NotificationRule(
            name="milestone_notifications",
            conditions={"type": "milestone"},
            channels=[NotificationChannel.WEBSOCKET],
            priority=NotificationPriority.NORMAL,
            template_name="insight",
            recipients=["team@scrollintel.com"]
        ))
        
        # Configure data quality monitoring
        self.quality_monitor.add_quality_rule("metrics", DataQualityRule(
            name="metric_completeness",
            description="Check metric data completeness",
            rule_type=DataQualityIssueType.MISSING_VALUES,
            severity=QualityCheckSeverity.HIGH,
            threshold=0.05,
            metadata={"columns": ["metric_name", "value", "timestamp"]}
        ))
        
        self.quality_monitor.add_quality_rule("metrics", DataQualityRule(
            name="metric_freshness",
            description="Check metric data freshness",
            rule_type=DataQualityIssueType.FRESHNESS,
            severity=QualityCheckSeverity.CRITICAL,
            threshold=5,  # 5 minutes
            metadata={"timestamp_column": "timestamp"}
        ))
        
        print("✅ Systems configured successfully!")
    
    async def start_systems(self):
        """Start all real-time systems"""
        print("🎯 Starting real-time systems...")
        
        # Start all systems concurrently
        tasks = [
            asyncio.create_task(self.processor.start_processing()),
            asyncio.create_task(self.alerting.start_monitoring()),
            asyncio.create_task(self.notification_system.start_processing()),
            asyncio.create_task(self.quality_monitor.start_monitoring())
        ]
        
        # Start WebSocket server on a different port for demo
        websocket_task = asyncio.create_task(
            self.websocket_manager.start_server(host="localhost", port=8767)
        )
        
        print("✅ All systems started!")
        
        return tasks + [websocket_task]
    
    async def generate_demo_data(self):
        """Generate realistic demo data"""
        print("📊 Starting demo data generation...")
        
        self.running = True
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                current_time = datetime.now()
                
                # Generate metrics with some patterns and anomalies
                for metric_name, generator in self.metrics_generators.items():
                    base_value = generator()
                    
                    # Add some patterns and anomalies
                    if metric_name == 'cpu_usage':
                        # Simulate CPU spikes every 30 iterations
                        if iteration % 30 == 0:
                            base_value = random.uniform(85, 95)
                        # Add daily pattern
                        hour_factor = 1 + 0.3 * np.sin(2 * np.pi * current_time.hour / 24)
                        base_value *= hour_factor
                    
                    elif metric_name == 'response_time':
                        # Simulate occasional latency spikes
                        if random.random() < 0.05:  # 5% chance
                            base_value *= random.uniform(3, 8)
                    
                    elif metric_name == 'error_rate':
                        # Simulate error bursts
                        if iteration % 50 == 0:
                            base_value = random.uniform(8, 15)
                    
                    elif metric_name == 'user_count':
                        # Simulate user activity patterns
                        base_value = int(base_value * (1 + 0.5 * np.sin(2 * np.pi * current_time.hour / 24)))
                    
                    # Publish metric
                    await self.processor.publish_message(
                        StreamType.METRICS,
                        {
                            'metric_name': metric_name,
                            'value': base_value,
                            'timestamp': current_time.isoformat(),
                            'source': 'demo_generator',
                            'iteration': iteration
                        },
                        'demo_system'
                    )
                    
                    # Also broadcast via WebSocket
                    await self.websocket_manager.broadcast_metric_update(
                        metric_name,
                        base_value,
                        {
                            'timestamp': current_time.isoformat(),
                            'iteration': iteration,
                            'trend': 'up' if base_value > 50 else 'down'
                        }
                    )
                
                # Generate some events
                if iteration % 10 == 0:
                    event_types = ['user_login', 'purchase', 'error', 'deployment', 'backup']
                    event_type = random.choice(event_types)
                    
                    await self.processor.publish_message(
                        StreamType.EVENTS,
                        {
                            'event_type': event_type,
                            'user_id': f'user_{random.randint(1, 1000)}',
                            'timestamp': current_time.isoformat(),
                            'metadata': {
                                'source_ip': f'192.168.1.{random.randint(1, 255)}',
                                'user_agent': 'ScrollIntel Dashboard'
                            }
                        },
                        'demo_system'
                    )
                
                # Generate insights periodically
                if iteration % 25 == 0:
                    insights = [
                        "User engagement increased by 15% in the last hour",
                        "Response time optimization reduced latency by 23%",
                        "Revenue trending 8% above daily target",
                        "New user acquisition rate exceeding expectations",
                        "System performance optimal across all metrics"
                    ]
                    
                    insight = random.choice(insights)
                    
                    await self.processor.publish_message(
                        StreamType.INSIGHTS,
                        {
                            'title': 'Automated Insight',
                            'description': insight,
                            'confidence': random.uniform(0.7, 0.95),
                            'category': 'performance',
                            'timestamp': current_time.isoformat()
                        },
                        'ai_insights'
                    )
                    
                    # Broadcast insight
                    await self.websocket_manager.broadcast_insight({
                        'title': 'Automated Insight',
                        'description': insight,
                        'timestamp': current_time.isoformat()
                    })
                
                # Update dashboard periodically
                if iteration % 15 == 0:
                    dashboard_data = {
                        'last_updated': current_time.isoformat(),
                        'total_metrics': len(self.metrics_generators),
                        'active_alerts': len(self.alerting.get_active_alerts()),
                        'system_health': 'optimal',
                        'uptime': f"{iteration * 2} seconds"
                    }
                    
                    await self.websocket_manager.broadcast_dashboard_update(
                        'main_dashboard',
                        dashboard_data
                    )
                
                # Print status every 20 iterations
                if iteration % 20 == 0:
                    active_alerts = self.alerting.get_active_alerts()
                    connection_stats = self.websocket_manager.get_connection_stats()
                    
                    print(f"📈 Iteration {iteration}:")
                    print(f"   Active Alerts: {len(active_alerts)}")
                    print(f"   WebSocket Connections: {connection_stats['total_connections']}")
                    print(f"   Dashboard Subscriptions: {connection_stats['dashboard_subscriptions']}")
                    
                    if active_alerts:
                        print("   Recent Alerts:")
                        for alert in active_alerts[-3:]:  # Show last 3 alerts
                            print(f"     - {alert.title} ({alert.severity.value})")
                
                # Wait before next iteration
                await asyncio.sleep(2)  # Generate data every 2 seconds
                
            except Exception as e:
                print(f"❌ Error in data generation: {e}")
                await asyncio.sleep(1)
    
    async def run_demo(self, duration_minutes: int = 5):
        """Run the complete demo"""
        print(f"🎬 Starting Real-time Dashboard Demo (Duration: {duration_minutes} minutes)")
        print("=" * 60)
        
        try:
            # Initialize systems
            await self.initialize()
            
            # Start systems
            system_tasks = await self.start_systems()
            
            # Wait for systems to start
            await asyncio.sleep(3)
            
            print("\n🌟 Demo Features:")
            print("   • Real-time metric processing and streaming")
            print("   • Intelligent threshold and anomaly alerting")
            print("   • WebSocket dashboard updates")
            print("   • Multi-channel notifications")
            print("   • Data quality monitoring")
            print("   • Performance analytics")
            print("\n📡 WebSocket server running on ws://localhost:8767")
            print("   Connect your dashboard to see real-time updates!")
            print("\n" + "=" * 60)
            
            # Start data generation
            data_task = asyncio.create_task(self.generate_demo_data())
            
            # Run for specified duration
            await asyncio.sleep(duration_minutes * 60)
            
            print(f"\n⏰ Demo completed after {duration_minutes} minutes")
            
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
        finally:
            # Stop data generation
            self.running = False
            
            # Stop all systems
            await self._stop_systems()
            
            # Cancel tasks
            for task in system_tasks + [data_task]:
                if not task.done():
                    task.cancel()
            
            print("🏁 Demo cleanup completed")
    
    async def _stop_systems(self):
        """Stop all systems gracefully"""
        print("🛑 Stopping systems...")
        
        try:
            if self.processor:
                await self.processor.stop_processing()
            if self.alerting:
                await self.alerting.stop_monitoring()
            if self.websocket_manager:
                await self.websocket_manager.stop_server()
            if self.notification_system:
                await self.notification_system.stop_processing()
            if self.quality_monitor:
                await self.quality_monitor.stop_monitoring()
        except Exception as e:
            print(f"⚠️  Error stopping systems: {e}")
    
    async def show_final_stats(self):
        """Show final demo statistics"""
        print("\n📊 Final Demo Statistics:")
        print("=" * 40)
        
        try:
            # Alert statistics
            active_alerts = self.alerting.get_active_alerts()
            print(f"Active Alerts: {len(active_alerts)}")
            
            if active_alerts:
                severity_counts = {}
                for alert in active_alerts:
                    severity = alert.severity.value
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                for severity, count in severity_counts.items():
                    print(f"  {severity.title()}: {count}")
            
            # Connection statistics
            if self.websocket_manager:
                stats = self.websocket_manager.get_connection_stats()
                print(f"\nWebSocket Statistics:")
                print(f"  Total Connections: {stats['total_connections']}")
                print(f"  Dashboard Subscriptions: {stats['dashboard_subscriptions']}")
                print(f"  Metric Subscriptions: {stats['metric_subscriptions']}")
            
            # Quality issues
            if self.quality_monitor:
                issues = await self.quality_monitor.get_active_issues()
                print(f"\nData Quality Issues: {len(issues)}")
                
                if issues:
                    for issue in issues[-3:]:  # Show last 3 issues
                        print(f"  - {issue.description}")
            
        except Exception as e:
            print(f"Error getting final stats: {e}")

async def run_interactive_demo():
    """Run interactive demo with user choices"""
    print("🎯 ScrollIntel Real-time Analytics Dashboard Demo")
    print("=" * 50)
    
    demo = RealtimeDashboardDemo()
    
    try:
        # Get demo duration from user
        duration_input = input("Enter demo duration in minutes (default: 5): ").strip()
        duration = int(duration_input) if duration_input.isdigit() else 5
        
        print(f"\n🚀 Starting {duration}-minute demo...")
        
        # Run the demo
        await demo.run_demo(duration)
        
        # Show final statistics
        await demo.show_final_stats()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

async def run_quick_demo():
    """Run a quick 2-minute demo"""
    print("⚡ Quick Real-time Dashboard Demo (2 minutes)")
    
    demo = RealtimeDashboardDemo()
    await demo.run_demo(duration_minutes=2)
    await demo.show_final_stats()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(run_quick_demo())
    else:
        asyncio.run(run_interactive_demo())