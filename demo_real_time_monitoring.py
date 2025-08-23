"""
ScrollIntel Real-Time Monitoring System Demo
Demonstrates enterprise-grade monitoring, analytics, and business impact tracking
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any
import json

from scrollintel.core.real_time_monitoring import (
    real_time_monitor,
    business_impact_tracker,
    executive_reporting,
    automated_alerting,
    get_real_time_dashboard
)
from scrollintel.core.monitoring import metrics_collector
from scrollintel.core.analytics import event_tracker
from scrollintel.core.alerting import alert_manager
from scrollintel.core.logging_config import get_logger

logger = get_logger(__name__)

class MonitoringDemo:
    """Comprehensive monitoring system demonstration"""
    
    def __init__(self):
        self.demo_agents = [
            {"id": "cto-agent-001", "type": "cto_agent"},
            {"id": "data-scientist-001", "type": "data_scientist"},
            {"id": "ml-engineer-001", "type": "ml_engineer"},
            {"id": "bi-agent-001", "type": "bi_agent"},
            {"id": "qa-agent-001", "type": "qa_agent"},
            {"id": "forecast-agent-001", "type": "forecast_agent"}
        ]
        self.simulation_running = False
        
    async def run_complete_demo(self):
        """Run comprehensive monitoring system demo"""
        print("🚀 ScrollIntel Real-Time Monitoring System Demo")
        print("=" * 60)
        
        try:
            # 1. Initialize monitoring system
            await self.initialize_monitoring_system()
            
            # 2. Register demo agents
            await self.register_demo_agents()
            
            # 3. Start real-time simulation
            await self.start_real_time_simulation()
            
            # 4. Demonstrate monitoring capabilities
            await self.demonstrate_monitoring_features()
            
            # 5. Generate executive reports
            await self.generate_executive_reports()
            
            # 6. Test alerting system
            await self.test_alerting_system()
            
            # 7. Show business impact metrics
            await self.show_business_impact_metrics()
            
            print("\n✅ Demo completed successfully!")
            print("Real-time monitoring system is fully operational.")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")
            
    async def initialize_monitoring_system(self):
        """Initialize the monitoring system components"""
        print("\n📊 Initializing Real-Time Monitoring System...")
        
        # Initialize metrics collector
        system_metrics = metrics_collector.collect_system_metrics()
        if system_metrics:
            print(f"   ✅ System metrics collector initialized")
            print(f"      CPU: {system_metrics.cpu_percent:.1f}%")
            print(f"      Memory: {system_metrics.memory_percent:.1f}%")
            print(f"      Disk: {system_metrics.disk_percent:.1f}%")
        
        # Initialize event tracker
        event_tracker.track_event(
            user_id="demo-user",
            session_id="demo-session-001",
            event_type="system_event",
            event_name="monitoring_system_initialized",
            properties={"demo_mode": True}
        )
        print("   ✅ Event tracking system initialized")
        
        # Initialize alert manager
        active_alerts = alert_manager.get_active_alerts()
        print(f"   ✅ Alert management system initialized ({len(active_alerts)} active alerts)")
        
        print("   🎯 All monitoring components ready!")
        
    async def register_demo_agents(self):
        """Register demo agents for monitoring"""
        print("\n🤖 Registering Demo Agents...")
        
        for agent in self.demo_agents:
            await real_time_monitor.register_agent(agent["id"], agent["type"])
            print(f"   ✅ Registered {agent['type']}: {agent['id']}")
            
        print(f"   🎯 {len(self.demo_agents)} agents registered for monitoring")
        
    async def start_real_time_simulation(self):
        """Start real-time metrics simulation"""
        print("\n⚡ Starting Real-Time Metrics Simulation...")
        
        self.simulation_running = True
        
        # Start background simulation task
        simulation_task = asyncio.create_task(self._simulate_agent_activity())
        
        # Let simulation run for a bit
        await asyncio.sleep(5)
        
        print("   🎯 Real-time simulation active!")
        
        return simulation_task
        
    async def _simulate_agent_activity(self):
        """Simulate realistic agent activity and metrics"""
        while self.simulation_running:
            try:
                for agent in self.demo_agents:
                    # Generate realistic metrics
                    metrics = self._generate_realistic_metrics(agent["type"])
                    
                    # Update agent metrics
                    await real_time_monitor.update_agent_metrics(agent["id"], metrics)
                    
                    # Track user interactions
                    event_tracker.track_agent_interaction(
                        user_id=f"user-{random.randint(1, 100)}",
                        session_id=f"session-{random.randint(1, 50)}",
                        agent_type=agent["type"],
                        operation="process_request",
                        success=metrics["success_rate"] > 90,
                        duration=metrics["avg_response_time"],
                        properties={"business_value": metrics["business_value_generated"]}
                    )
                
                # Update system metrics
                metrics_collector.collect_system_metrics()
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                await asyncio.sleep(5)
                
    def _generate_realistic_metrics(self, agent_type: str) -> Dict[str, Any]:
        """Generate realistic metrics based on agent type"""
        base_metrics = {
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(30, 85),
            "success_rate": random.uniform(88, 99.5),
            "avg_response_time": random.uniform(0.5, 4.0),
            "request_count": random.randint(50, 200),
            "error_count": random.randint(0, 10),
            "throughput_per_minute": random.uniform(20, 100),
            "business_value_generated": random.uniform(1000, 10000),
            "cost_savings": random.uniform(500, 5000)
        }
        
        # Adjust metrics based on agent type
        if agent_type == "cto_agent":
            base_metrics["business_value_generated"] *= 2.5
            base_metrics["avg_response_time"] *= 0.8
        elif agent_type == "data_scientist":
            base_metrics["cpu_usage"] *= 1.2
            base_metrics["avg_response_time"] *= 1.5
        elif agent_type == "ml_engineer":
            base_metrics["memory_usage"] *= 1.3
            base_metrics["cpu_usage"] *= 1.4
        elif agent_type == "bi_agent":
            base_metrics["throughput_per_minute"] *= 1.5
            base_metrics["success_rate"] = min(99.5, base_metrics["success_rate"] * 1.02)
        
        return base_metrics
        
    async def demonstrate_monitoring_features(self):
        """Demonstrate key monitoring features"""
        print("\n📈 Demonstrating Monitoring Features...")
        
        # 1. Show real-time agent metrics
        print("\n   📊 Real-Time Agent Performance:")
        agent_metrics = real_time_monitor.get_all_agent_metrics()
        
        for agent in agent_metrics[:3]:  # Show first 3 agents
            print(f"      🤖 {agent.agent_type} ({agent.agent_id}):")
            print(f"         Status: {agent.status}")
            print(f"         Success Rate: {agent.success_rate:.1f}%")
            print(f"         Response Time: {agent.avg_response_time:.2f}s")
            print(f"         Business Value: ${agent.business_value_generated:,.2f}")
            print(f"         Cost Savings: ${agent.cost_savings:,.2f}")
        
        # 2. Show system health
        print("\n   🏥 System Health Status:")
        dashboard = await get_real_time_dashboard()
        system_health = dashboard["system_health"]
        
        print(f"      Overall Health: {system_health['overall_health_score']:.1f}%")
        print(f"      Performance: {system_health['performance_score']:.1f}%")
        print(f"      Availability: {system_health['availability_score']:.1f}%")
        print(f"      Security: {system_health['security_score']:.1f}%")
        
        # 3. Show active alerts
        print("\n   🚨 Alert Status:")
        active_alerts = alert_manager.get_active_alerts()
        if active_alerts:
            for alert in active_alerts[:3]:
                print(f"      ⚠️  {alert.name}: {alert.description}")
        else:
            print("      ✅ No active alerts - system running smoothly")
            
    async def generate_executive_reports(self):
        """Generate and display executive reports"""
        print("\n📋 Generating Executive Reports...")
        
        # Generate comprehensive dashboard
        dashboard = await executive_reporting.generate_executive_dashboard()
        
        print("\n   💼 Executive Summary:")
        summary = dashboard["executive_summary"]
        print(f"      Total ROI: {summary['total_roi']:.1f}%")
        print(f"      Monthly Cost Savings: ${summary['monthly_cost_savings']:,.2f}")
        print(f"      Revenue Impact: ${summary['revenue_impact']:,.2f}")
        print(f"      Productivity Gain: {summary['productivity_gain']:.1f}%")
        print(f"      System Health: {summary['system_health']:.1f}%")
        print(f"      User Satisfaction: {summary['user_satisfaction']:.1f}%")
        
        print("\n   🎯 Key Performance Indicators:")
        kpis = dashboard["key_performance_indicators"]
        print(f"      Requests/Minute: {kpis['requests_per_minute']}")
        print(f"      Avg Resolution Time: {kpis['avg_resolution_time']}s")
        print(f"      Customer Satisfaction: {kpis['customer_satisfaction']}/5.0")
        print(f"      Automation Rate: {kpis['automation_rate']:.1f}%")
        print(f"      Accuracy Rate: {kpis['accuracy_rate']:.1f}%")
        
        print("\n   🏆 Competitive Positioning:")
        competitive = dashboard["competitive_positioning"]
        print(f"      Advantage Score: {competitive['advantage_score']:.1f}%")
        print(f"      Unique Capabilities: {len(competitive['unique_capabilities'])}")
        
        differentiation = competitive["market_differentiation"]
        print(f"      Performance: {differentiation['performance_advantage']}")
        print(f"      Cost: {differentiation['cost_advantage']}")
        print(f"      Features: {differentiation['feature_advantage']}")
        
    async def test_alerting_system(self):
        """Test the automated alerting system"""
        print("\n🚨 Testing Automated Alerting System...")
        
        # Simulate performance degradation
        print("\n   ⚠️  Simulating Performance Issues...")
        
        # Update agent with poor performance metrics
        test_agent_id = self.demo_agents[0]["id"]
        degraded_metrics = {
            "cpu_usage": 95.0,  # High CPU
            "memory_usage": 92.0,  # High memory
            "success_rate": 75.0,  # Low success rate
            "avg_response_time": 8.5,  # High response time
            "request_count": 50,
            "error_count": 25,  # High error count
            "business_value_generated": 500.0  # Low business value
        }
        
        await real_time_monitor.update_agent_metrics(test_agent_id, degraded_metrics)
        
        # Check for alerts
        await asyncio.sleep(1)
        
        # Simulate alert evaluation
        alert_metrics = {
            "agent_avg_response_time": 8.5,
            "agent_success_rate": 75.0,
            "cpu_percent": 95.0,
            "memory_percent": 92.0
        }
        
        alert_manager.evaluate_metrics(alert_metrics)
        
        # Show alert status
        active_alerts = alert_manager.get_active_alerts()
        print(f"   📊 Generated {len(active_alerts)} alerts from performance issues")
        
        for alert in active_alerts[:2]:
            print(f"      🚨 {alert.severity.value.upper()}: {alert.name}")
            print(f"         {alert.description}")
            print(f"         Current: {alert.current_value}, Threshold: {alert.threshold}")
        
        # Restore normal performance
        print("\n   ✅ Restoring Normal Performance...")
        normal_metrics = {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "success_rate": 96.5,
            "avg_response_time": 1.2,
            "request_count": 150,
            "error_count": 2,
            "business_value_generated": 5000.0
        }
        
        await real_time_monitor.update_agent_metrics(test_agent_id, normal_metrics)
        
    async def show_business_impact_metrics(self):
        """Show detailed business impact metrics"""
        print("\n💰 Business Impact Analysis...")
        
        # Calculate business metrics
        business_metrics = await business_impact_tracker.calculate_roi_metrics()
        
        print("\n   📈 ROI & Cost Savings:")
        print(f"      Total ROI: {business_metrics.total_roi:.1f}%")
        print(f"      24h Cost Savings: ${business_metrics.cost_savings_24h:,.2f}")
        print(f"      7d Cost Savings: ${business_metrics.cost_savings_7d:,.2f}")
        print(f"      30d Cost Savings: ${business_metrics.cost_savings_30d:,.2f}")
        print(f"      Revenue Impact: ${business_metrics.revenue_impact:,.2f}")
        
        print("\n   🚀 Performance Improvements:")
        print(f"      Productivity Gain: {business_metrics.productivity_gain:.1f}%")
        print(f"      Decision Accuracy: +{business_metrics.decision_accuracy_improvement:.1f}%")
        print(f"      Time-to-Insight Reduction: {business_metrics.time_to_insight_reduction:.1f}%")
        
        print("\n   😊 User & Market Impact:")
        print(f"      User Satisfaction: {business_metrics.user_satisfaction_score:.1f}%")
        print(f"      Competitive Advantage: {business_metrics.competitive_advantage_score:.1f}%")
        
        # Calculate payback period
        monthly_savings = business_metrics.cost_savings_30d
        monthly_costs = 7000  # Mock monthly operational costs
        
        if monthly_savings > monthly_costs:
            payback_months = monthly_costs / (monthly_savings - monthly_costs)
            print(f"\n   💡 Investment Analysis:")
            print(f"      Monthly Operational Cost: ${monthly_costs:,.2f}")
            print(f"      Monthly Net Savings: ${monthly_savings - monthly_costs:,.2f}")
            print(f"      Payback Period: {payback_months:.1f} months")
        
    async def cleanup_demo(self):
        """Clean up demo resources"""
        print("\n🧹 Cleaning up demo resources...")
        
        self.simulation_running = False
        
        # Clear demo data
        real_time_monitor.agent_metrics.clear()
        real_time_monitor.agent_history.clear()
        
        print("   ✅ Demo cleanup completed")

async def run_monitoring_benchmarks():
    """Run performance benchmarks for the monitoring system"""
    print("\n🏃‍♂️ Running Monitoring System Benchmarks...")
    
    # Benchmark 1: Agent registration performance
    print("\n   📊 Benchmark 1: Agent Registration Performance")
    start_time = time.time()
    
    for i in range(100):
        await real_time_monitor.register_agent(f"benchmark-agent-{i}", f"type-{i % 5}")
    
    registration_time = time.time() - start_time
    print(f"      Registered 100 agents in {registration_time:.3f}s")
    print(f"      Rate: {100/registration_time:.1f} registrations/second")
    
    # Benchmark 2: Metrics update performance
    print("\n   📊 Benchmark 2: Metrics Update Performance")
    start_time = time.time()
    
    for i in range(100):
        metrics = {
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(30, 70),
            "success_rate": random.uniform(90, 99),
            "avg_response_time": random.uniform(0.5, 3.0),
            "request_count": random.randint(50, 200),
            "business_value_generated": random.uniform(1000, 5000)
        }
        await real_time_monitor.update_agent_metrics(f"benchmark-agent-{i}", metrics)
    
    update_time = time.time() - start_time
    print(f"      Updated 100 agent metrics in {update_time:.3f}s")
    print(f"      Rate: {100/update_time:.1f} updates/second")
    
    # Benchmark 3: Dashboard generation performance
    print("\n   📊 Benchmark 3: Dashboard Generation Performance")
    start_time = time.time()
    
    for i in range(10):
        dashboard = await get_real_time_dashboard()
    
    dashboard_time = time.time() - start_time
    print(f"      Generated 10 dashboards in {dashboard_time:.3f}s")
    print(f"      Rate: {10/dashboard_time:.1f} dashboards/second")
    
    # Benchmark 4: Business impact calculation performance
    print("\n   📊 Benchmark 4: Business Impact Calculation Performance")
    start_time = time.time()
    
    for i in range(5):
        business_metrics = await business_impact_tracker.calculate_roi_metrics()
    
    business_time = time.time() - start_time
    print(f"      Calculated 5 business impact reports in {business_time:.3f}s")
    print(f"      Rate: {5/business_time:.1f} calculations/second")
    
    print("\n   🎯 Benchmark Results Summary:")
    print(f"      ✅ Agent Registration: {100/registration_time:.1f} ops/sec")
    print(f"      ✅ Metrics Updates: {100/update_time:.1f} ops/sec")
    print(f"      ✅ Dashboard Generation: {10/dashboard_time:.1f} ops/sec")
    print(f"      ✅ Business Calculations: {5/business_time:.1f} ops/sec")

async def main():
    """Main demo execution"""
    demo = MonitoringDemo()
    
    try:
        # Run complete demo
        await demo.run_complete_demo()
        
        # Run performance benchmarks
        await run_monitoring_benchmarks()
        
        # Keep simulation running for a bit longer
        print("\n⏱️  Monitoring system running... (Press Ctrl+C to stop)")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        logger.error(f"Demo error: {e}")
    finally:
        await demo.cleanup_demo()

if __name__ == "__main__":
    asyncio.run(main())