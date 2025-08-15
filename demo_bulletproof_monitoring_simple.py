"""
Simplified demo for bulletproof monitoring and analytics system.
Demonstrates real-time monitoring, failure pattern analysis, user satisfaction tracking,
and system health visualization with predictive alerts.
"""

import asyncio
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import time

from scrollintel.core.bulletproof_monitoring_analytics import (
    bulletproof_analytics,
    UserExperienceMetric,
    MetricType,
    AlertSeverity
)

class BulletproofMonitoringDemo:
    """Demo class for bulletproof monitoring and analytics system."""
    
    def __init__(self):
        self.demo_users = [f"user_{i}" for i in range(1, 21)]  # 20 demo users
        self.demo_components = ["api_gateway", "database", "ui_frontend", "auth_service", "analytics_engine"]
        self.demo_actions = ["login", "view_dashboard", "create_report", "export_data", "update_profile"]
        
    async def run_comprehensive_demo(self):
        """Run comprehensive demo of the monitoring system."""
        print("🚀 Starting Bulletproof Monitoring & Analytics Demo")
        print("=" * 60)
        
        # Run demo scenarios
        await self._demo_normal_operations()
        await self._demo_performance_degradation()
        await self._demo_failure_patterns()
        await self._demo_user_satisfaction_tracking()
        await self._demo_anomaly_detection()
        
        # Show analytics and reports
        await self._show_real_time_dashboard()
        await self._show_failure_pattern_analysis()
        await self._show_user_satisfaction_analysis()
        await self._show_health_report()
        
        print("\n✅ Demo completed successfully!")
        print("The bulletproof monitoring system is now actively protecting user experience.")
    
    async def _demo_normal_operations(self):
        """Demo normal system operations."""
        print("\n🔄 Simulating normal operations...")
        
        for i in range(50):
            user = random.choice(self.demo_users)
            action = random.choice(self.demo_actions)
            component = random.choice(self.demo_components)
            
            # Normal performance (100-500ms)
            duration = random.uniform(100, 500)
            
            # Record component performance
            metric = UserExperienceMetric(
                timestamp=datetime.now(),
                user_id=user,
                metric_type=MetricType.PERFORMANCE,
                value=duration,
                context={"action": action, "normal": True},
                component=component
            )
            await bulletproof_analytics.record_metric(metric)
            
            # Small delay to simulate real operations
            await asyncio.sleep(0.01)
        
        print(f"✅ Recorded 50 normal operations across {len(self.demo_components)} components")
    
    async def _demo_performance_degradation(self):
        """Demo performance degradation scenario."""
        print("\n⚠️  Simulating performance degradation...")
        
        # Simulate database performance issues
        degraded_component = "database"
        
        for i in range(20):
            user = random.choice(self.demo_users)
            
            # Degraded performance (2-5 seconds)
            duration = random.uniform(2000, 5000)
            
            metric = UserExperienceMetric(
                timestamp=datetime.now(),
                user_id=user,
                metric_type=MetricType.PERFORMANCE,
                value=duration,
                context={
                    "degradation_scenario": True,
                    "issue": "database_slow_query"
                },
                component=degraded_component
            )
            await bulletproof_analytics.record_metric(metric)
            
            await asyncio.sleep(0.02)
        
        print(f"✅ Simulated performance degradation in {degraded_component}")
        print("   - Response times: 2-5 seconds (above 1s threshold)")
        print("   - Should trigger predictive alerts")
    
    async def _demo_failure_patterns(self):
        """Demo failure pattern detection."""
        print("\n🚨 Simulating failure patterns...")
        
        # Simulate authentication service failures
        failing_component = "auth_service"
        
        for i in range(25):  # Above pattern detection threshold
            user = random.choice(self.demo_users)
            
            # High failure rate
            failure_metric = UserExperienceMetric(
                timestamp=datetime.now(),
                user_id=user,
                metric_type=MetricType.FAILURE_RATE,
                value=0.8,  # 80% failure rate
                context={
                    "error_type": "authentication_timeout",
                    "failure_scenario": True
                },
                component=failing_component
            )
            await bulletproof_analytics.record_metric(failure_metric)
            
            await asyncio.sleep(0.01)
        
        print(f"✅ Simulated failure pattern in {failing_component}")
        print("   - Failure rate: 80% (above 5% threshold)")
        print("   - Should detect failure pattern and escalate severity")
    
    async def _demo_user_satisfaction_tracking(self):
        """Demo user satisfaction tracking."""
        print("\n😊 Simulating user satisfaction feedback...")
        
        # Simulate satisfaction feedback with patterns
        current_hour = datetime.now().hour
        
        for i in range(30):
            user = random.choice(self.demo_users)
            
            # Lower satisfaction during "peak hours" (simulate system stress)
            if 9 <= current_hour <= 17:  # Business hours
                satisfaction = random.uniform(2.0, 3.5)  # Lower satisfaction
                context = {"period": "peak_hours", "stress_level": "high"}
            else:
                satisfaction = random.uniform(3.5, 5.0)  # Higher satisfaction
                context = {"period": "off_hours", "stress_level": "low"}
            
            satisfaction_metric = UserExperienceMetric(
                timestamp=datetime.now(),
                user_id=user,
                metric_type=MetricType.USER_SATISFACTION,
                value=satisfaction,
                context=context,
                component="user_feedback"
            )
            await bulletproof_analytics.record_metric(satisfaction_metric)
            
            await asyncio.sleep(0.01)
        
        print("✅ Recorded 30 user satisfaction ratings")
        print("   - Pattern: Lower satisfaction during peak hours")
        print("   - Should detect satisfaction patterns and trends")
    
    async def _demo_anomaly_detection(self):
        """Demo anomaly detection."""
        print("\n🔍 Simulating anomalous behavior...")
        
        # Add normal baseline metrics
        for i in range(50):
            user = random.choice(self.demo_users)
            component = random.choice(self.demo_components)
            
            # Normal performance
            normal_value = random.uniform(200, 400)
            
            metric = UserExperienceMetric(
                timestamp=datetime.now(),
                user_id=user,
                metric_type=MetricType.PERFORMANCE,
                value=normal_value,
                context={"baseline": True},
                component=component
            )
            await bulletproof_analytics.record_metric(metric)
        
        # Add anomalous metrics
        for i in range(10):
            user = random.choice(self.demo_users)
            component = random.choice(self.demo_components)
            
            # Anomalous performance (very high)
            anomalous_value = random.uniform(8000, 15000)  # 8-15 seconds
            
            metric = UserExperienceMetric(
                timestamp=datetime.now(),
                user_id=user,
                metric_type=MetricType.PERFORMANCE,
                value=anomalous_value,
                context={"anomaly": True, "type": "extreme_latency"},
                component=component
            )
            await bulletproof_analytics.record_metric(metric)
            
            await asyncio.sleep(0.01)
        
        print("✅ Simulated anomalous behavior")
        print("   - Baseline: 200-400ms response times")
        print("   - Anomalies: 8-15 second response times")
        print("   - Should trigger ML-based anomaly detection")
    
    async def _show_real_time_dashboard(self):
        """Show real-time dashboard data."""
        print("\n📈 Real-Time Dashboard Data")
        print("-" * 40)
        
        dashboard_data = await bulletproof_analytics.get_real_time_dashboard_data()
        
        if "error" in dashboard_data:
            print(f"❌ Error: {dashboard_data['error']}")
            return
        
        metrics = dashboard_data["metrics"]
        
        print(f"🎯 System Health Score: {metrics['system_health_score']:.1f}/100")
        print(f"⚡ Avg Response Time: {metrics['avg_response_time']:.1f}ms")
        print(f"😊 User Satisfaction: {metrics['user_satisfaction']:.1f}/5.0")
        print(f"🚨 Critical Issues: {metrics['active_critical_issues']}")
        print(f"👥 Active Users: {metrics['total_users_active']}")
        print(f"📊 Requests/Min: {metrics['requests_per_minute']}")
        
        # Show active alerts
        alerts = dashboard_data["alerts"]
        if alerts:
            print(f"\n🚨 Active Alerts ({len(alerts)}):")
            for alert in alerts[-5:]:  # Show last 5 alerts
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "low": "🟢"
                }.get(alert["severity"], "⚪")
                
                print(f"   {severity_emoji} {alert['predicted_issue']} "
                      f"(Probability: {alert['probability']:.1%})")
        
        # Show component health
        component_health = dashboard_data["component_health"]
        if component_health:
            print(f"\n🏗️  Component Health:")
            for component, health in component_health.items():
                health_score = health["health_score"]
                status_emoji = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
                print(f"   {status_emoji} {component}: {health_score:.1f}/100 "
                      f"(Avg Response: {health['avg_response_time']:.1f}ms)")
    
    async def _show_failure_pattern_analysis(self):
        """Show failure pattern analysis."""
        print("\n🔍 Failure Pattern Analysis")
        print("-" * 40)
        
        analysis = await bulletproof_analytics.get_failure_pattern_analysis()
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        if "message" in analysis:
            print(f"ℹ️  {analysis['message']}")
            return
        
        patterns = analysis["patterns"]
        insights = analysis["insights"]
        
        print(f"📊 Total Patterns Detected: {analysis['total_patterns']}")
        
        if patterns:
            print("\n🚨 Top Failure Patterns:")
            for i, pattern in enumerate(patterns[:3], 1):
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡", 
                    "low": "🟢"
                }.get(pattern["severity"], "⚪")
                
                print(f"   {i}. {severity_emoji} {pattern['pattern_type']}")
                print(f"      Frequency: {pattern['frequency']} occurrences")
                print(f"      Impact Score: {pattern['impact_score']:.1f}")
                print(f"      Components: {', '.join(pattern['components_affected'])}")
                print(f"      Active: {'Yes' if pattern['is_active'] else 'No'}")
        
        if insights:
            print("\n💡 Key Insights:")
            for insight in insights:
                insight_emoji = {
                    "most_frequent_pattern": "📈",
                    "critical_patterns": "🚨",
                    "recent_patterns": "⏰"
                }.get(insight["type"], "💡")
                
                print(f"   {insight_emoji} {insight['description']}")
    
    async def _show_user_satisfaction_analysis(self):
        """Show user satisfaction analysis."""
        print("\n😊 User Satisfaction Analysis")
        print("-" * 40)
        
        analysis = await bulletproof_analytics.analyze_user_satisfaction_patterns()
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        if "message" in analysis:
            print(f"ℹ️  {analysis['message']}")
            return
        
        stats = analysis["overall_stats"]
        patterns = analysis["patterns"]
        
        print(f"📊 Overall Statistics:")
        print(f"   Mean Satisfaction: {stats['mean_satisfaction']:.2f}/5.0")
        print(f"   Median Satisfaction: {stats['median_satisfaction']:.2f}/5.0")
        print(f"   Standard Deviation: {stats['std_satisfaction']:.2f}")
        print(f"   Total Responses: {stats['total_responses']}")
        
        # Show satisfaction distribution
        if "satisfaction_distribution" in analysis:
            print(f"\n📈 Satisfaction Distribution:")
            distribution = analysis["satisfaction_distribution"]
            for score in sorted(distribution.keys()):
                count = distribution[score]
                bar = "█" * min(20, int(count / max(distribution.values()) * 20))
                print(f"   {score:.1f}: {bar} ({count})")
        
        if patterns:
            print(f"\n🔍 Detected Patterns:")
            for pattern in patterns:
                pattern_emoji = {
                    "low_satisfaction_hours": "⏰",
                    "declining_satisfaction": "📉",
                    "improving_satisfaction": "📈"
                }.get(pattern["type"], "🔍")
                
                print(f"   {pattern_emoji} {pattern['description']}")
                print(f"      Recommendation: {pattern['recommendation']}")
    
    async def _show_health_report(self):
        """Show comprehensive health report."""
        print("\n🏥 System Health Report")
        print("-" * 40)
        
        report = await bulletproof_analytics.generate_health_report()
        
        if "error" in report:
            print(f"❌ Error: {report['error']}")
            return
        
        print(f"📊 Overall Health Score: {report['system_health_score']:.1f}/100")
        print(f"📅 Report Time: {report['report_timestamp']}")
        print(f"📈 Data Points Analyzed: {report['data_points_analyzed']:,}")
        
        # Performance summary
        perf = report["performance_summary"]
        print(f"\n⚡ Performance Summary:")
        print(f"   Avg Response Time: {perf['avg_response_time']:.1f}ms")
        print(f"   95th Percentile: {perf['p95_response_time']:.1f}ms")
        print(f"   99th Percentile: {perf['p99_response_time']:.1f}ms")
        print(f"   Avg Error Rate: {perf['avg_error_rate']:.1%}")
        
        # Satisfaction summary
        satisfaction = report["satisfaction_summary"]
        print(f"\n😊 Satisfaction Summary:")
        print(f"   Avg Satisfaction: {satisfaction['avg_satisfaction']:.2f}/5.0")
        print(f"   Total Feedback: {satisfaction['total_feedback']}")
        print(f"   Trend: {satisfaction['satisfaction_trend'].title()}")
        
        # Alert summary
        alerts = report["alert_summary"]
        print(f"\n🚨 Alert Summary:")
        print(f"   Total Active: {alerts['total_active_alerts']}")
        print(f"   Critical: {alerts['critical_alerts']}")
        print(f"   High Priority: {alerts['high_alerts']}")
        print(f"   Medium Priority: {alerts['medium_alerts']}")
        
        # Recommendations
        recommendations = report["recommendations"]
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

async def main():
    """Main demo function."""
    demo = BulletproofMonitoringDemo()
    
    try:
        await demo.run_comprehensive_demo()
        
        print("\n🔄 Monitoring system demonstrates comprehensive capabilities:")
        print("   • Real-time user experience monitoring")
        print("   • Failure pattern analysis and learning")
        print("   • User satisfaction tracking with feedback integration")
        print("   • System health visualization with predictive alerts")
        print("   • ML-based anomaly detection")
        print("   • Comprehensive health reporting")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Bulletproof User Experience - Monitoring & Analytics Demo")
    print("=" * 70)
    print("This demo showcases comprehensive monitoring and analytics capabilities:")
    print("• Real-time user experience monitoring")
    print("• Failure pattern analysis and learning")
    print("• User satisfaction tracking with feedback integration")
    print("• System health visualization with predictive alerts")
    print("• ML-based anomaly detection")
    print("• Comprehensive health reporting")
    print("=" * 70)
    
    asyncio.run(main())