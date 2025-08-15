"""Demo script for usage tracking and compliance reporting system."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import uuid

from ai_data_readiness.engines.usage_tracker import UsageTracker
from ai_data_readiness.engines.compliance_reporter import ComplianceReporter
from ai_data_readiness.engines.audit_logger import AuditLogger
from ai_data_readiness.models.governance_models import (
    AuditEventType, DataClassification, User
)


class UsageTrackingComplianceDemo:
    """Demo class for usage tracking and compliance reporting."""
    
    def __init__(self):
        self.usage_tracker = UsageTracker()
        self.compliance_reporter = ComplianceReporter()
        self.audit_logger = AuditLogger()
        
        print("🚀 AI Data Readiness Platform - Usage Tracking & Compliance Demo")
        print("=" * 70)
    
    def simulate_user_activity(self) -> None:
        """Simulate user activity for demo purposes."""
        print("\n📊 Simulating User Activity...")
        
        # Simulate various user activities
        users = ["alice_analyst", "bob_scientist", "charlie_admin", "diana_steward"]
        resources = [
            {"id": "customer_data", "type": "dataset"},
            {"id": "financial_reports", "type": "dataset"},
            {"id": "ml_model_v1", "type": "model"},
            {"id": "compliance_dashboard", "type": "dashboard"}
        ]
        actions = ["read", "write", "execute", "export", "delete"]
        
        # Generate sample audit events
        for i in range(100):
            user = users[i % len(users)]
            resource = resources[i % len(resources)]
            action = actions[i % len(actions)]
            
            # Simulate some failed attempts
            success = i % 15 != 0  # ~7% failure rate
            
            try:
                self.audit_logger.log_data_access(
                    user_id=user,
                    resource_id=resource["id"],
                    resource_type=resource["type"],
                    action=action,
                    details={
                        "session_id": f"session_{i % 10}",
                        "query_complexity": "medium" if i % 3 == 0 else "low",
                        "data_volume_mb": (i % 100) + 10
                    },
                    ip_address=f"192.168.1.{(i % 50) + 100}",
                    user_agent="DataAnalytics/1.0"
                )
                
                # Track usage
                self.usage_tracker.track_data_access(
                    user_id=user,
                    resource_id=resource["id"],
                    resource_type=resource["type"],
                    action=action,
                    metadata={"simulated": True}
                )
                
            except Exception as e:
                print(f"⚠️  Error simulating activity: {e}")
        
        print("✅ Simulated 100 user activities")
    
    def demonstrate_usage_analytics(self) -> None:
        """Demonstrate usage analytics capabilities."""
        print("\n📈 Usage Analytics Demo")
        print("-" * 30)
        
        try:
            # Get comprehensive usage analytics
            analytics = self.usage_tracker.get_usage_analytics(
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow(),
                aggregation_level="daily"
            )
            
            print("📊 Usage Summary:")
            summary = analytics['summary']
            print(f"  • Total Events: {summary['total_events']}")
            print(f"  • Unique Users: {summary['unique_users']}")
            print(f"  • Unique Resources: {summary['unique_resources']}")
            print(f"  • Success Rate: {summary['success_rate']:.2%}")
            
            print("\n🔍 Event Types:")
            for event_type, count in summary['event_types'].items():
                print(f"  • {event_type}: {count}")
            
            print("\n⚡ Top Actions:")
            for action, count in list(summary['actions'].items())[:5]:
                print(f"  • {action}: {count}")
            
            # Show access patterns
            patterns = analytics['access_patterns']
            print("\n🕐 Access Patterns:")
            print("  Hourly Distribution (top 5 hours):")
            hourly = patterns['hourly_distribution']
            top_hours = sorted(hourly.items(), key=lambda x: x[1], reverse=True)[:5]
            for hour, count in top_hours:
                print(f"    • {hour:02d}:00 - {count} accesses")
            
        except Exception as e:
            print(f"❌ Error in usage analytics: {e}")
    
    def demonstrate_user_activity_report(self) -> None:
        """Demonstrate user activity reporting."""
        print("\n👤 User Activity Report Demo")
        print("-" * 35)
        
        try:
            # Get detailed user activity report
            report = self.usage_tracker.get_user_activity_report(
                user_id="alice_analyst",
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow(),
                include_details=False
            )
            
            print("📋 User Activity Summary:")
            activity = report['activity_summary']
            print(f"  • Total Events: {activity['total_events']}")
            print(f"  • Success Rate: {activity['success_rate']:.2%}")
            
            print("\n📁 Resource Usage:")
            resource_usage = report['resource_usage']
            print(f"  • Resources Accessed: {resource_usage['total_resources_accessed']}")
            
            if resource_usage['most_accessed_resources']:
                print("  • Most Accessed Resources:")
                for resource, count in resource_usage['most_accessed_resources'][:3]:
                    print(f"    - {resource}: {count} times")
            
            print("\n⚠️  Risk Indicators:")
            risk = report['risk_indicators']
            print(f"  • Failed Attempts: {risk['failed_attempts']}")
            print(f"  • Unusual Hour Access: {risk['unusual_hour_access']}")
            print(f"  • Risk Score: {risk['risk_score']}/100")
            
        except Exception as e:
            print(f"❌ Error in user activity report: {e}")
    
    def demonstrate_resource_usage_report(self) -> None:
        """Demonstrate resource usage reporting."""
        print("\n📊 Resource Usage Report Demo")
        print("-" * 37)
        
        try:
            # Get detailed resource usage report
            report = self.usage_tracker.get_resource_usage_report(
                resource_id="customer_data",
                resource_type="dataset",
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow(),
                include_user_breakdown=True
            )
            
            print("📈 Resource Usage Summary:")
            usage = report['usage_summary']
            print(f"  • Total Accesses: {usage['total_events']}")
            print(f"  • Unique Users: {usage['unique_users']}")
            print(f"  • Success Rate: {usage['success_rate']:.2%}")
            
            print("\n👥 User Breakdown:")
            user_breakdown = report['user_breakdown']
            for user_id, stats in list(user_breakdown.items())[:5]:
                print(f"  • {user_id}: {stats['access_count']} accesses")
            
            print("\n🔒 Security Metrics:")
            security = report['security_metrics']
            print(f"  • Failed Access Attempts: {security['failed_access_attempts']}")
            print(f"  • Security Incidents: {security['security_incidents']}")
            
        except Exception as e:
            print(f"❌ Error in resource usage report: {e}")
    
    def demonstrate_usage_trends(self) -> None:
        """Demonstrate usage trend analysis."""
        print("\n📊 Usage Trends Demo")
        print("-" * 25)
        
        try:
            # Generate usage trends
            trends = self.usage_tracker.generate_usage_trends(
                metric_type="access_count",
                time_period="7d",
                granularity="daily"
            )
            
            print("📈 Trend Analysis:")
            summary = trends['summary']
            print(f"  • Metric Type: {trends['metric_type']}")
            print(f"  • Time Period: {trends['time_period']}")
            print(f"  • Data Points: {summary['total_data_points']}")
            print(f"  • Trend Direction: {summary['trend_direction']}")
            print(f"  • Average Value: {summary['average_value']:.1f}")
            
            if summary['peak_value']:
                peak = summary['peak_value']
                print(f"  • Peak: {peak['value']} on {peak['timestamp']}")
            
            print("\n📅 Daily Trend Data (last 5 days):")
            for trend_point in trends['trends'][-5:]:
                print(f"  • {trend_point['timestamp']}: {trend_point['value']} accesses")
            
        except Exception as e:
            print(f"❌ Error in usage trends: {e}")
    
    def demonstrate_compliance_frameworks(self) -> None:
        """Demonstrate compliance framework support."""
        print("\n🛡️  Compliance Frameworks Demo")
        print("-" * 35)
        
        frameworks = self.compliance_reporter.compliance_frameworks
        
        print("📋 Supported Compliance Frameworks:")
        for code, info in frameworks.items():
            print(f"\n  🔹 {code} - {info['name']}")
            print(f"    Requirements: {len(info['requirements'])}")
            for req in info['requirements'][:3]:  # Show first 3 requirements
                print(f"      • {req.replace('_', ' ').title()}")
            if len(info['requirements']) > 3:
                print(f"      • ... and {len(info['requirements']) - 3} more")
    
    def demonstrate_compliance_report_generation(self) -> None:
        """Demonstrate compliance report generation."""
        print("\n📊 Compliance Report Generation Demo")
        print("-" * 42)
        
        try:
            # Generate GDPR compliance report
            print("🔍 Generating GDPR Compliance Report...")
            
            report = self.compliance_reporter.generate_compliance_report(
                framework="GDPR",
                scope=["customer_data", "financial_reports"],
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow(),
                generated_by="demo_user"
            )
            
            print("✅ GDPR Compliance Report Generated:")
            print(f"  • Report ID: {report.id}")
            print(f"  • Compliance Score: {report.compliance_score:.1f}/100")
            print(f"  • Violations Found: {len(report.violations)}")
            print(f"  • Recommendations: {len(report.recommendations)}")
            
            if report.violations:
                print("\n⚠️  Top Violations:")
                for violation in report.violations[:3]:
                    print(f"    • {violation.get('type', 'Unknown')}: {violation.get('description', 'No description')}")
            
            if report.recommendations:
                print("\n💡 Top Recommendations:")
                for rec in report.recommendations[:3]:
                    print(f"    • {rec.get('type', 'Unknown')}: {rec.get('description', 'No description')}")
            
        except Exception as e:
            print(f"❌ Error generating compliance report: {e}")
    
    def demonstrate_data_classification_validation(self) -> None:
        """Demonstrate data classification validation."""
        print("\n🏷️  Data Classification Validation Demo")
        print("-" * 44)
        
        try:
            # Validate data classification compliance
            validation = self.compliance_reporter.validate_data_classification_compliance(
                dataset_id="customer_data"
            )
            
            print("🔍 Data Classification Validation Results:")
            print(f"  • Dataset ID: {validation['dataset_id']}")
            print(f"  • Classification: {validation['classification']}")
            print(f"  • Compliance Status: {validation['compliance_status']}")
            
            if validation['issues']:
                print("\n⚠️  Issues Found:")
                for issue in validation['issues']:
                    print(f"    • {issue['type']} ({issue['severity']}): {issue['description']}")
            
            if validation['recommendations']:
                print("\n💡 Recommendations:")
                for rec in validation['recommendations']:
                    print(f"    • {rec['type']}: {rec['description']}")
            
        except Exception as e:
            print(f"❌ Error in data classification validation: {e}")
    
    def demonstrate_access_compliance_audit(self) -> None:
        """Demonstrate access compliance auditing."""
        print("\n🔐 Access Compliance Audit Demo")
        print("-" * 36)
        
        try:
            # Perform access compliance audit
            audit_result = self.compliance_reporter.audit_access_compliance(
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow()
            )
            
            print("🔍 Access Compliance Audit Results:")
            print(f"  • Total Access Events: {audit_result['total_access_events']}")
            print(f"  • Compliance Violations: {len(audit_result['compliance_violations'])}")
            
            # Show access patterns
            patterns = audit_result['access_patterns']
            print(f"\n📊 Access Patterns:")
            print(f"  • Unusual Hour Accesses: {patterns['unusual_access_count']}")
            print(f"  • Top Users: {len(patterns['top_users'])}")
            print(f"  • Top Resources: {len(patterns['top_resources'])}")
            
            # Show risk indicators
            risk = audit_result['risk_indicators']
            print(f"\n⚠️  Risk Indicators:")
            print(f"  • Failure Rate: {risk['failure_rate']:.2%}")
            print(f"  • Unusual Hour Rate: {risk['unusual_hour_rate']:.2%}")
            print(f"  • Risk Score: {risk['risk_score']}/100")
            
            if audit_result['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in audit_result['recommendations'][:3]:
                    print(f"    • {rec['type']} ({rec['priority']}): {rec['description']}")
            
        except Exception as e:
            print(f"❌ Error in access compliance audit: {e}")
    
    def demonstrate_compliance_dashboard(self) -> None:
        """Demonstrate compliance dashboard."""
        print("\n📊 Compliance Dashboard Demo")
        print("-" * 33)
        
        try:
            # Get compliance dashboard
            dashboard = self.compliance_reporter.get_compliance_dashboard(
                frameworks=["GDPR", "CCPA", "SOX"],
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow()
            )
            
            print("🎛️  Compliance Dashboard Overview:")
            
            # Overall metrics
            overall = dashboard['overall_metrics']
            print(f"\n📈 Overall Metrics:")
            print(f"  • Average Score: {overall['average_score']:.1f}/100")
            print(f"  • Total Violations: {overall['total_violations']}")
            print(f"  • Compliant Frameworks: {overall['compliant_frameworks']}/{overall['total_frameworks']}")
            
            # Framework breakdown
            print(f"\n🛡️  Framework Status:")
            for framework, metrics in dashboard['frameworks'].items():
                status_icon = "✅" if metrics['status'] == 'compliant' else "❌"
                print(f"  {status_icon} {framework}: {metrics['latest_score']:.1f}/100 ({metrics['violations_count']} violations)")
            
            # Alerts
            if dashboard['alerts']:
                print(f"\n🚨 Active Alerts:")
                for alert in dashboard['alerts'][:3]:
                    severity_icon = "🔴" if alert['severity'] == 'critical' else "🟡" if alert['severity'] == 'high' else "🟢"
                    print(f"  {severity_icon} {alert['type']}: {alert['message']}")
            
        except Exception as e:
            print(f"❌ Error in compliance dashboard: {e}")
    
    def demonstrate_audit_trail_report(self) -> None:
        """Demonstrate audit trail reporting."""
        print("\n📋 Audit Trail Report Demo")
        print("-" * 32)
        
        try:
            # Generate audit trail report
            report = self.compliance_reporter.generate_audit_trail_report(
                resource_id="customer_data",
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow(),
                event_types=["data_access", "data_modification"]
            )
            
            print("📊 Audit Trail Report Summary:")
            summary = report['summary']
            print(f"  • Total Events: {summary['total_events']}")
            print(f"  • Unique Users: {summary['unique_users']}")
            print(f"  • Unique Resources: {summary['unique_resources']}")
            print(f"  • Success Rate: {summary['success_rate']:.2%}")
            
            print(f"\n📈 Event Type Breakdown:")
            for event_type, count in summary['event_type_breakdown'].items():
                print(f"  • {event_type}: {count}")
            
            print(f"\n📅 Timeline Summary:")
            timeline = report['timeline']
            if timeline:
                print(f"  • Data Points: {len(timeline)}")
                print(f"  • Date Range: {timeline[0]['date']} to {timeline[-1]['date']}")
                total_events = sum(point['event_count'] for point in timeline)
                avg_daily = total_events / len(timeline) if timeline else 0
                print(f"  • Average Daily Events: {avg_daily:.1f}")
            
            print(f"\n📝 Compliance Notes:")
            for note in report['compliance_notes']:
                print(f"  • {note}")
            
        except Exception as e:
            print(f"❌ Error in audit trail report: {e}")
    
    def demonstrate_system_overview(self) -> None:
        """Demonstrate system usage overview."""
        print("\n🌐 System Usage Overview Demo")
        print("-" * 34)
        
        try:
            # Get system-wide usage overview
            overview = self.usage_tracker.get_system_usage_overview(
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow()
            )
            
            print("🎯 System Metrics:")
            metrics = overview['system_metrics']
            print(f"  • Total Users: {metrics['total_users']}")
            print(f"  • Total Resources: {metrics['total_resources']}")
            print(f"  • Total Events: {metrics['total_events']}")
            
            print(f"\n📊 Activity Summary:")
            activity = overview['activity_summary']
            print(f"  • Unique Users Active: {activity['unique_users']}")
            print(f"  • Unique Resources Accessed: {activity['unique_resources']}")
            print(f"  • Overall Success Rate: {activity['success_rate']:.2%}")
            
            print(f"\n🏆 Top Users:")
            for user_info in overview['top_users'][:5]:
                print(f"  • {user_info['user_id']}: {user_info['event_count']} events")
            
            print(f"\n📁 Top Resources:")
            for resource_info in overview['top_resources'][:5]:
                print(f"  • {resource_info['resource']}: {resource_info['access_count']} accesses")
            
            print(f"\n🔒 Security Overview:")
            security = overview['security_overview']
            print(f"  • Security Events: {security['total_security_events']}")
            print(f"  • Incident Rate: {security['security_incident_rate']:.2%}")
            
        except Exception as e:
            print(f"❌ Error in system overview: {e}")
    
    def run_demo(self) -> None:
        """Run the complete demo."""
        try:
            # Simulate data first
            self.simulate_user_activity()
            
            # Usage tracking demonstrations
            self.demonstrate_usage_analytics()
            self.demonstrate_user_activity_report()
            self.demonstrate_resource_usage_report()
            self.demonstrate_usage_trends()
            self.demonstrate_system_overview()
            
            # Compliance demonstrations
            self.demonstrate_compliance_frameworks()
            self.demonstrate_compliance_report_generation()
            self.demonstrate_data_classification_validation()
            self.demonstrate_access_compliance_audit()
            self.demonstrate_compliance_dashboard()
            self.demonstrate_audit_trail_report()
            
            print("\n" + "=" * 70)
            print("✅ Usage Tracking & Compliance Demo Completed Successfully!")
            print("\n🎯 Key Capabilities Demonstrated:")
            print("  • Comprehensive usage analytics and reporting")
            print("  • Multi-framework compliance assessment (GDPR, CCPA, SOX, HIPAA)")
            print("  • Real-time audit trail generation and analysis")
            print("  • Risk-based access pattern analysis")
            print("  • Data classification compliance validation")
            print("  • Executive compliance dashboards")
            print("  • Automated violation detection and recommendations")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the demo."""
    demo = UsageTrackingComplianceDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()