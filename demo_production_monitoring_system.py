"""
Demo Script for Production Monitoring and Alerting System

This script demonstrates the comprehensive production monitoring system including:
- Real-time system health monitoring with predictive alerts
- User experience quality monitoring with automatic optimization
- Failure pattern detection with proactive prevention
- Comprehensive reporting system for continuous improvement
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.core.production_monitoring import production_monitor, MetricType
from scrollintel.core.ux_quality_monitor import ux_quality_monitor, UXMetricType
from scrollintel.core.failure_pattern_detector import failure_pattern_detector, FailureType, PatternSeverity
from scrollintel.core.comprehensive_reporting import comprehensive_reporter, ReportType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionMonitoringDemo:
    """
    Comprehensive demo of the production monitoring system
    """
    
    def __init__(self):
        self.demo_running = False
        self.user_sessions = {}
        
    async def run_complete_demo(self):
        """Run the complete production monitoring demo"""
        print("🚀 Starting Production Monitoring System Demo")
        print("=" * 60)
        
        try:
            # Initialize the comprehensive reporter with all components
            comprehensive_reporter.production_monitor = production_monitor
            comprehensive_reporter.ux_monitor = ux_quality_monitor
            comprehensive_reporter.pattern_detector = failure_pattern_detector
            
            # Start all monitoring systems
            print("\n📊 Starting monitoring systems...")
            await self._start_monitoring_systems()
            
            # Run demo scenarios
            await self._run_demo_scenarios()
            
            # Generate comprehensive reports
            await self._generate_demo_reports()
            
            # Show system status
            await self._show_system_status()
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            # Stop monitoring systems
            print("\n🛑 Stopping monitoring systems...")
            await self._stop_monitoring_systems()
        
        print("\n✅ Production Monitoring System Demo Complete!")
    
    async def _start_monitoring_systems(self):
        """Start all monitoring systems"""
        try:
            # Note: In a real implementation, these would run continuously
            # For demo purposes, we'll simulate their operation
            print("   ✓ Production Monitor initialized")
            print("   ✓ UX Quality Monitor initialized")
            print("   ✓ Failure Pattern Detector initialized")
            print("   ✓ Comprehensive Reporter initialized")
            
        except Exception as e:
            logger.error(f"Error starting monitoring systems: {e}")
    
    async def _stop_monitoring_systems(self):
        """Stop all monitoring systems"""
        try:
            # Note: In a real implementation, these would stop the continuous monitoring
            print("   ✓ All monitoring systems stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring systems: {e}")
    
    async def _run_demo_scenarios(self):
        """Run various demo scenarios"""
        print("\n🎭 Running Demo Scenarios")
        print("-" * 40)
        
        # Scenario 1: Normal operations
        await self._scenario_normal_operations()
        
        # Scenario 2: Performance degradation
        await self._scenario_performance_degradation()
        
        # Scenario 3: User experience issues
        await self._scenario_ux_issues()
        
        # Scenario 4: System failures and recovery
        await self._scenario_system_failures()
        
        # Scenario 5: Cascade failure pattern
        await self._scenario_cascade_failure()
    
    async def _scenario_normal_operations(self):
        """Simulate normal system operations"""
        print("\n📈 Scenario 1: Normal Operations")
        
        # Simulate normal user activities
        for i in range(20):
            user_id = f"user_{i % 5}"  # 5 different users
            
            # Record successful user actions
            production_monitor.record_user_action(
                user_id=user_id,
                action=random.choice(["page_load", "search", "login", "purchase", "view_profile"]),
                response_time=random.uniform(500, 1500),  # Normal response times
                success=True,
                satisfaction=random.uniform(0.8, 1.0)  # High satisfaction
            )
            
            # Start UX session if not exists
            if user_id not in self.user_sessions:
                session_id = ux_quality_monitor.start_user_session(
                    user_id=user_id,
                    context={"source": "demo", "scenario": "normal"}
                )
                self.user_sessions[user_id] = session_id
            
            # Record UX metrics
            ux_quality_monitor.record_ux_metric(
                user_id=user_id,
                session_id=self.user_sessions[user_id],
                metric_type=UXMetricType.PERFORMANCE,
                metric_name="page_load_time",
                value=random.uniform(800, 1200),
                context={"page": "dashboard"}
            )
            
            await asyncio.sleep(0.1)  # Small delay
        
        print("   ✓ Recorded 20 normal user interactions")
        
        # Check system health
        health = production_monitor.get_system_health()
        print(f"   📊 System Health Score: {health['health_score']:.1f}/100")
        print(f"   👥 Active Sessions: {health['active_sessions']}")
    
    async def _scenario_performance_degradation(self):
        """Simulate performance degradation"""
        print("\n⚠️  Scenario 2: Performance Degradation")
        
        # Simulate gradually increasing response times
        for i in range(15):
            base_response_time = 1000 + (i * 200)  # Increasing response times
            
            production_monitor.record_user_action(
                user_id=f"user_{i % 3}",
                action="slow_operation",
                response_time=base_response_time + random.uniform(-100, 100),
                success=random.choice([True, True, False]),  # Some failures
                satisfaction=max(0.3, 0.9 - (i * 0.04))  # Decreasing satisfaction
            )
            
            await asyncio.sleep(0.05)
        
        print("   ⚠️  Simulated performance degradation")
        
        # Check for alerts
        alerts = production_monitor.get_alerts(resolved=False)
        if alerts:
            print(f"   🚨 Generated {len(alerts)} alerts")
            for alert in alerts[:3]:  # Show first 3 alerts
                print(f"      - {alert['title']}: {alert['description']}")
        
        # Check UX dashboard
        ux_dashboard = ux_quality_monitor.get_ux_dashboard()
        print(f"   📉 Average Satisfaction: {ux_dashboard['average_satisfaction']:.2f}")
    
    async def _scenario_ux_issues(self):
        """Simulate user experience issues"""
        print("\n😞 Scenario 3: User Experience Issues")
        
        # Simulate various UX problems
        ux_issues = [
            ("page_load_time", 4000, "Slow page loading"),
            ("error_rate", 0.15, "High error rate"),
            ("task_completion_rate", 0.6, "Low task completion"),
            ("bounce_rate", 0.8, "High bounce rate")
        ]
        
        for metric_name, value, description in ux_issues:
            for user_id in ["ux_user_1", "ux_user_2", "ux_user_3"]:
                if user_id not in self.user_sessions:
                    session_id = ux_quality_monitor.start_user_session(
                        user_id=user_id,
                        context={"scenario": "ux_issues"}
                    )
                    self.user_sessions[user_id] = session_id
                
                ux_quality_monitor.record_ux_metric(
                    user_id=user_id,
                    session_id=self.user_sessions[user_id],
                    metric_type=UXMetricType.USABILITY,
                    metric_name=metric_name,
                    value=value,
                    context={"issue": description}
                )
        
        print("   😞 Simulated UX issues: slow loading, high errors, low completion")
        
        # Check for optimization recommendations
        optimizations = ux_quality_monitor.get_optimization_recommendations()
        if optimizations:
            print(f"   💡 Generated {len(optimizations)} optimization recommendations")
            for opt in optimizations[:3]:
                print(f"      - {opt.get('description', 'Optimization available')}")
    
    async def _scenario_system_failures(self):
        """Simulate system failures"""
        print("\n💥 Scenario 4: System Failures")
        
        # Simulate various types of failures
        failure_scenarios = [
            (FailureType.SYSTEM_ERROR, "api_gateway", "Connection timeout to database", PatternSeverity.HIGH),
            (FailureType.RESOURCE_EXHAUSTION, "web_server", "Memory usage exceeded 95%", PatternSeverity.CRITICAL),
            (FailureType.DEPENDENCY_FAILURE, "payment_service", "External payment API unavailable", PatternSeverity.HIGH),
            (FailureType.NETWORK_ISSUE, "load_balancer", "Network latency spike detected", PatternSeverity.MEDIUM),
            (FailureType.DATA_CORRUPTION, "user_database", "Inconsistent data state detected", PatternSeverity.CRITICAL)
        ]
        
        for failure_type, component, error_message, severity in failure_scenarios:
            # Record multiple instances of each failure to create patterns
            for i in range(random.randint(3, 7)):
                event_id = failure_pattern_detector.record_failure(
                    failure_type=failure_type,
                    component=component,
                    error_message=f"{error_message} (instance {i+1})",
                    stack_trace=f"Stack trace for {component} error {i+1}",
                    user_id=f"affected_user_{i}",
                    context={"scenario": "system_failures", "instance": i+1},
                    severity=severity
                )
                
                await asyncio.sleep(0.02)  # Small delay between failures
        
        print("   💥 Simulated various system failures")
        
        # Check detected patterns
        patterns = failure_pattern_detector.get_detected_patterns()
        if patterns:
            print(f"   🔍 Detected {len(patterns)} failure patterns")
            for pattern in patterns[:3]:
                print(f"      - {pattern['pattern_type']}: {pattern['description']}")
        
        # Check component health
        component_health = failure_pattern_detector.get_component_health()
        unhealthy_components = [
            comp for comp, health in component_health.items()
            if health.get("health_status") == "degraded"
        ]
        if unhealthy_components:
            print(f"   🏥 Unhealthy components: {', '.join(unhealthy_components)}")
    
    async def _scenario_cascade_failure(self):
        """Simulate cascade failure pattern"""
        print("\n🌊 Scenario 5: Cascade Failure Pattern")
        
        # Simulate a cascade failure starting from one component
        cascade_components = [
            "authentication_service",
            "user_service", 
            "profile_service",
            "notification_service",
            "analytics_service"
        ]
        
        base_time = datetime.now()
        
        for i, component in enumerate(cascade_components):
            # Each failure happens shortly after the previous one
            failure_time = base_time + timedelta(minutes=i * 2)
            
            for j in range(random.randint(2, 5)):  # Multiple failures per component
                failure_pattern_detector.record_failure(
                    failure_type=FailureType.DEPENDENCY_FAILURE,
                    component=component,
                    error_message=f"Cascade failure in {component} due to upstream dependency",
                    stack_trace=f"Cascade error stack trace for {component}",
                    user_id=f"cascade_user_{j}",
                    context={
                        "cascade_order": i,
                        "root_cause": "authentication_service_failure",
                        "affected_users": random.randint(10, 100)
                    },
                    severity=PatternSeverity.CRITICAL
                )
                
                await asyncio.sleep(0.01)
        
        print("   🌊 Simulated cascade failure across 5 components")
        
        # Check for cascade pattern detection
        patterns = failure_pattern_detector.get_detected_patterns()
        cascade_patterns = [p for p in patterns if "cascade" in p.get("pattern_type", "").lower()]
        if cascade_patterns:
            print(f"   🔗 Detected {len(cascade_patterns)} cascade patterns")
        
        # Check prevention status
        prevention_status = failure_pattern_detector.get_prevention_status()
        print(f"   🛡️  Prevention system: {prevention_status['active_rules']} active rules")
    
    async def _generate_demo_reports(self):
        """Generate comprehensive reports"""
        print("\n📋 Generating Comprehensive Reports")
        print("-" * 40)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)  # Last hour of demo data
        
        # Generate different types of reports
        report_types = [
            (ReportType.SYSTEM_HEALTH, "System Health Report"),
            (ReportType.USER_EXPERIENCE, "User Experience Report"),
            (ReportType.FAILURE_ANALYSIS, "Failure Analysis Report"),
            (ReportType.EXECUTIVE_SUMMARY, "Executive Summary Report")
        ]
        
        generated_reports = []
        
        for report_type, report_name in report_types:
            try:
                print(f"\n📊 Generating {report_name}...")
                
                report = await comprehensive_reporter.generate_report(
                    report_type=report_type,
                    time_period=(start_time, end_time),
                    filters={"demo": True}
                )
                
                generated_reports.append(report)
                
                print(f"   ✓ Report ID: {report.id}")
                print(f"   📈 Insights: {len(report.insights)}")
                print(f"   💡 Recommendations: {len(report.recommendations)}")
                
                # Show key insights
                if report.insights:
                    print("   🔍 Key Insights:")
                    for insight in report.insights[:2]:  # Show first 2 insights
                        print(f"      - {insight.title}: {insight.description}")
                
                # Show top recommendations
                if report.recommendations:
                    print("   💡 Top Recommendations:")
                    for rec in report.recommendations[:2]:  # Show first 2 recommendations
                        print(f"      - {rec}")
                
            except Exception as e:
                logger.error(f"Error generating {report_name}: {e}")
        
        print(f"\n✅ Generated {len(generated_reports)} comprehensive reports")
        
        # Show insights summary
        insights_summary = comprehensive_reporter.get_insights_summary(days=1)
        print(f"\n📊 Insights Summary:")
        print(f"   Total Insights: {insights_summary['total_insights']}")
        print(f"   Reports Analyzed: {insights_summary['reports_analyzed']}")
        print(f"   Average Confidence: {insights_summary['average_confidence']:.2f}")
        
        if insights_summary['top_recommendations']:
            print("   🏆 Top Recommendations:")
            for rec, count in insights_summary['top_recommendations'][:3]:
                print(f"      - {rec} (mentioned {count} times)")
    
    async def _show_system_status(self):
        """Show final system status"""
        print("\n📊 Final System Status")
        print("-" * 40)
        
        # Production Monitor Status
        health = production_monitor.get_system_health()
        print(f"\n🏥 System Health:")
        print(f"   Overall Score: {health['health_score']:.1f}/100")
        print(f"   Status: {health['status']}")
        print(f"   Active Alerts: {health['active_alerts']}")
        print(f"   Active Sessions: {health['active_sessions']}")
        
        # UX Monitor Status
        ux_dashboard = ux_quality_monitor.get_ux_dashboard()
        print(f"\n😊 User Experience:")
        print(f"   Average Satisfaction: {ux_dashboard['average_satisfaction']:.2f}")
        print(f"   Satisfaction Level: {ux_dashboard['satisfaction_level']}")
        print(f"   Average Load Time: {ux_dashboard['average_load_time']:.0f}ms")
        print(f"   Active Optimizations: {ux_dashboard['active_optimizations']}")
        
        # Failure Pattern Detector Status
        patterns = failure_pattern_detector.get_detected_patterns()
        component_health = failure_pattern_detector.get_component_health()
        prevention_status = failure_pattern_detector.get_prevention_status()
        
        print(f"\n🔍 Failure Analysis:")
        print(f"   Detected Patterns: {len(patterns)}")
        print(f"   Critical Patterns: {len([p for p in patterns if p.get('severity') == 'critical'])}")
        print(f"   Components Monitored: {len(component_health)}")
        print(f"   Prevention Rules Active: {prevention_status['active_rules']}")
        
        # Report Status
        reports = comprehensive_reporter.list_reports(limit=10)
        schedules = comprehensive_reporter.get_report_schedules()
        
        print(f"\n📋 Reporting System:")
        print(f"   Generated Reports: {len(reports)}")
        print(f"   Scheduled Reports: {len(schedules)}")
        print(f"   Active Schedules: {len([s for s in schedules if s['enabled']])}")
        
        # Show sample alert if any
        alerts = production_monitor.get_alerts(resolved=False)
        if alerts:
            print(f"\n🚨 Sample Active Alert:")
            alert = alerts[0]
            print(f"   Title: {alert['title']}")
            print(f"   Severity: {alert['severity']}")
            print(f"   Description: {alert['description']}")
        
        # Show sample pattern if any
        if patterns:
            print(f"\n🔍 Sample Failure Pattern:")
            pattern = patterns[0]
            print(f"   Type: {pattern['pattern_type']}")
            print(f"   Component: {pattern['component']}")
            print(f"   Frequency: {pattern['frequency']}")
            print(f"   Confidence: {pattern['confidence']:.2f}")

async def main():
    """Main demo function"""
    demo = ProductionMonitoringDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())