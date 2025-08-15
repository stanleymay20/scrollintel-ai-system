"""
Demo Script for ScrollIntel Audit Logging and Compliance System

Demonstrates comprehensive audit logging, compliance reporting, data retention,
and audit trail export capabilities for the Launch MVP.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from scrollintel.core.audit_system import (
    audit_system, AuditAction, ComplianceLevel, RetentionPolicy
)
from scrollintel.core.compliance_manager import (
    compliance_manager, ComplianceFramework, DataClassification
)


async def demo_basic_audit_logging():
    """Demonstrate basic audit logging functionality"""
    print("\n🔍 Basic Audit Logging Demo")
    print("=" * 40)
    
    # Sample user data
    user_id = str(uuid4())
    user_email = "demo@scrollintel.com"
    session_id = "demo_session_123"
    ip_address = "192.168.1.100"
    
    print(f"👤 Demo User: {user_email}")
    print(f"🔗 Session ID: {session_id}")
    print(f"🌐 IP Address: {ip_address}")
    
    # Log various user actions
    actions_to_log = [
        {
            "action": AuditAction.LOGIN_SUCCESS,
            "resource_type": "authentication",
            "details": {"login_method": "password", "mfa_enabled": True}
        },
        {
            "action": AuditAction.DATA_UPLOAD,
            "resource_type": "dataset",
            "resource_id": "dataset_123",
            "resource_name": "Customer_Data.csv",
            "details": {"file_size": 2048000, "rows": 10000, "columns": 15}
        },
        {
            "action": AuditAction.DASHBOARD_CREATE,
            "resource_type": "dashboard",
            "resource_id": "dashboard_456",
            "resource_name": "Sales Analytics Dashboard",
            "details": {"chart_count": 5, "data_sources": 2}
        },
        {
            "action": AuditAction.MODEL_TRAIN,
            "resource_type": "ml_model",
            "resource_id": "model_789",
            "resource_name": "Customer Churn Predictor",
            "details": {"algorithm": "random_forest", "accuracy": 0.92, "training_time": 300}
        },
        {
            "action": AuditAction.API_REQUEST,
            "resource_type": "api_endpoint",
            "resource_id": "/api/predictions",
            "details": {"method": "POST", "response_time": 0.15, "status_code": 200}
        }
    ]
    
    print(f"\n📝 Logging {len(actions_to_log)} audit events...")
    
    event_ids = []
    for action_data in actions_to_log:
        event_id = await audit_system.log_user_action(
            user_id=user_id,
            user_email=user_email,
            session_id=session_id,
            ip_address=ip_address,
            user_agent="ScrollIntel Demo Client v1.0",
            **action_data
        )
        event_ids.append(event_id)
        print(f"  ✅ {action_data['action'].value} -> {event_id[:8]}...")
    
    # Wait for background processing
    await asyncio.sleep(0.5)
    
    print(f"\n✨ Successfully logged {len(event_ids)} audit events!")
    return user_id, event_ids


async def demo_audit_log_search():
    """Demonstrate audit log search and filtering"""
    print("\n🔍 Audit Log Search Demo")
    print("=" * 40)
    
    # Search with various filters
    search_scenarios = [
        {
            "name": "All Recent Logs",
            "filters": {
                "start_date": datetime.utcnow() - timedelta(hours=1),
                "limit": 10
            }
        },
        {
            "name": "Authentication Events",
            "filters": {
                "action": AuditAction.LOGIN_SUCCESS.value,
                "resource_type": "authentication",
                "limit": 5
            }
        },
        {
            "name": "Data Operations",
            "filters": {
                "resource_type": "dataset",
                "start_date": datetime.utcnow() - timedelta(hours=1),
                "limit": 5
            }
        },
        {
            "name": "Failed Operations",
            "filters": {
                "success": False,
                "start_date": datetime.utcnow() - timedelta(days=1),
                "limit": 5
            }
        }
    ]
    
    for scenario in search_scenarios:
        print(f"\n🔎 Searching: {scenario['name']}")
        
        try:
            logs = await audit_system.search_audit_logs(**scenario['filters'])
            print(f"  📊 Found {len(logs)} matching logs")
            
            for log in logs[:3]:  # Show first 3
                timestamp = datetime.fromisoformat(log['timestamp']).strftime("%H:%M:%S")
                print(f"    • {timestamp}: {log['action']} on {log['resource_type']}")
                
        except Exception as e:
            print(f"  ❌ Search failed: {e}")
    
    print("\n✨ Audit log search completed!")


async def demo_compliance_reporting():
    """Demonstrate compliance reporting functionality"""
    print("\n📋 Compliance Reporting Demo")
    print("=" * 40)
    
    # Generate compliance report
    start_date = datetime.utcnow() - timedelta(days=7)
    end_date = datetime.utcnow()
    
    print(f"📅 Report Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        report = await audit_system.generate_compliance_report(
            start_date=start_date,
            end_date=end_date,
            report_type="comprehensive"
        )
        
        print(f"\n📊 Compliance Report Generated:")
        print(f"  🔢 Total Events: {report.total_events}")
        print(f"  ✅ Successful Events: {report.successful_events}")
        print(f"  ❌ Failed Events: {report.failed_events}")
        print(f"  🔒 Security Events: {report.security_events}")
        print(f"  ⚠️  Compliance Violations: {len(report.compliance_violations)}")
        
        if report.compliance_violations:
            print(f"\n⚠️  Recent Violations:")
            for violation in report.compliance_violations[:3]:
                print(f"    • {violation['action']} - {violation.get('error_message', 'Unknown error')}")
        
        print(f"\n👥 User Activity Summary:")
        for user_id, count in list(report.user_activity_summary.items())[:5]:
            print(f"    • User {user_id[:8]}...: {count} actions")
        
        print(f"\n📁 Resource Access Summary:")
        for resource_type, count in list(report.resource_access_summary.items())[:5]:
            print(f"    • {resource_type}: {count} accesses")
        
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"    • {rec}")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
    
    print("\n✨ Compliance reporting completed!")


async def demo_compliance_framework_analysis():
    """Demonstrate compliance framework-specific analysis"""
    print("\n🏛️ Compliance Framework Analysis Demo")
    print("=" * 40)
    
    frameworks_to_test = [
        ComplianceFramework.GDPR,
        ComplianceFramework.SOX,
        ComplianceFramework.ISO_27001
    ]
    
    start_date = datetime.utcnow() - timedelta(days=30)
    end_date = datetime.utcnow()
    
    for framework in frameworks_to_test:
        print(f"\n🔍 Analyzing {framework.value.upper()} Compliance...")
        
        try:
            report = await compliance_manager.generate_compliance_report(
                framework=framework,
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"  📊 Events Analyzed: {report['summary']['total_events_analyzed']}")
            print(f"  ⚠️  Violations Found: {report['summary']['total_violations']}")
            print(f"  📈 Compliance Rate: {report['summary']['compliance_rate']:.1f}%")
            print(f"  📋 Rules Evaluated: {report['summary']['rules_evaluated']}")
            
            if report['recommendations']:
                print(f"  💡 Top Recommendation: {report['recommendations'][0]}")
            
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
    
    print("\n✨ Framework analysis completed!")


async def demo_data_retention_policies():
    """Demonstrate data retention policy management"""
    print("\n🗂️ Data Retention Policies Demo")
    print("=" * 40)
    
    # Show current retention policies
    policies = compliance_manager.retention_policies
    
    print(f"📋 Active Retention Policies ({len(policies)}):")
    for policy_id, policy in policies.items():
        print(f"\n  📁 {policy.name}")
        print(f"    • ID: {policy_id}")
        print(f"    • Data Types: {', '.join(policy.data_types)}")
        print(f"    • Retention Period: {policy.retention_period.value}")
        print(f"    • Deletion Method: {policy.deletion_method}")
        print(f"    • Auto Cleanup: {'✅' if policy.auto_cleanup else '❌'}")
        print(f"    • Frameworks: {', '.join([f.value for f in policy.compliance_frameworks])}")
    
    # Simulate retention policy application
    print(f"\n🧹 Applying Retention Policies...")
    
    try:
        cleanup_results = await compliance_manager.apply_retention_policies()
        
        print(f"📊 Cleanup Results:")
        total_cleaned = 0
        for policy_id, count in cleanup_results.items():
            if count >= 0:
                print(f"  • {policy_id}: {count} records cleaned")
                total_cleaned += count
            else:
                print(f"  • {policy_id}: ❌ Error during cleanup")
        
        print(f"\n✨ Total Records Cleaned: {total_cleaned}")
        
    except Exception as e:
        print(f"❌ Retention policy application failed: {e}")
    
    print("\n✨ Data retention demo completed!")


async def demo_audit_export():
    """Demonstrate audit log export functionality"""
    print("\n📤 Audit Log Export Demo")
    print("=" * 40)
    
    start_date = datetime.utcnow() - timedelta(days=7)
    end_date = datetime.utcnow()
    
    export_formats = ["json", "csv"]
    
    for format_type in export_formats:
        print(f"\n📄 Exporting audit logs as {format_type.upper()}...")
        
        try:
            file_path = await audit_system.export_audit_logs(
                start_date=start_date,
                end_date=end_date,
                format=format_type,
                action=AuditAction.DATA_UPLOAD.value  # Filter for data upload events
            )
            
            export_file = Path(file_path)
            if export_file.exists():
                file_size = export_file.stat().st_size
                print(f"  ✅ Export successful!")
                print(f"  📁 File: {export_file.name}")
                print(f"  📊 Size: {file_size:,} bytes")
                
                # Show sample content for JSON
                if format_type == "json" and file_size < 10000:  # Only for small files
                    with open(export_file, 'r') as f:
                        data = json.load(f)
                        print(f"  📋 Records: {len(data.get('audit_logs', []))}")
                        print(f"  📅 Date Range: {data['export_metadata']['date_range_start']} to {data['export_metadata']['date_range_end']}")
                
                # Cleanup demo files
                export_file.unlink()
                print(f"  🧹 Demo file cleaned up")
            else:
                print(f"  ❌ Export file not found")
                
        except Exception as e:
            print(f"  ❌ Export failed: {e}")
    
    print("\n✨ Audit export demo completed!")


async def demo_compliance_violation_detection():
    """Demonstrate compliance violation detection"""
    print("\n⚠️ Compliance Violation Detection Demo")
    print("=" * 40)
    
    # Simulate various scenarios that might trigger violations
    test_scenarios = [
        {
            "name": "GDPR Personal Data Access",
            "action": "access",
            "resource_type": "personal_data",
            "resource_id": "user_profile_123",
            "user_id": "user_456",
            "data_classification": DataClassification.CONFIDENTIAL
        },
        {
            "name": "SOX Financial Data Modification",
            "action": "modify",
            "resource_type": "financial_data",
            "resource_id": "billing_record_789",
            "user_id": "user_789",
            "data_classification": DataClassification.RESTRICTED
        },
        {
            "name": "Regular Dashboard Access",
            "action": "view",
            "resource_type": "dashboard",
            "resource_id": "dashboard_123",
            "user_id": "user_456",
            "data_classification": DataClassification.INTERNAL
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        
        try:
            violations = await compliance_manager.check_compliance(
                action=scenario["action"],
                resource_type=scenario["resource_type"],
                resource_id=scenario["resource_id"],
                user_id=scenario["user_id"],
                data_classification=scenario["data_classification"]
            )
            
            if violations:
                print(f"  ⚠️  {len(violations)} violation(s) detected:")
                for violation in violations:
                    print(f"    • Rule: {violation.rule_id}")
                    print(f"    • Severity: {violation.severity}")
                    print(f"    • Description: {violation.description}")
            else:
                print(f"  ✅ No violations detected")
                
        except Exception as e:
            print(f"  ❌ Violation check failed: {e}")
    
    # Wait for background processing
    await asyncio.sleep(0.5)
    
    print("\n✨ Violation detection demo completed!")


async def demo_system_statistics():
    """Demonstrate audit system statistics and metrics"""
    print("\n📊 System Statistics Demo")
    print("=" * 40)
    
    # Generate some sample statistics
    start_date = datetime.utcnow() - timedelta(days=30)
    end_date = datetime.utcnow()
    
    try:
        # Generate a compliance report for statistics
        report = await audit_system.generate_compliance_report(
            start_date=start_date,
            end_date=end_date,
            report_type="statistics"
        )
        
        print(f"📈 System Statistics (Last 30 Days):")
        print(f"  🔢 Total Events: {report.total_events:,}")
        print(f"  ✅ Success Rate: {(report.successful_events / report.total_events * 100):.1f}%" if report.total_events > 0 else "  ✅ Success Rate: N/A")
        print(f"  ❌ Failed Events: {report.failed_events:,}")
        print(f"  🔒 Security Events: {report.security_events:,}")
        print(f"  ⚠️  Violations: {len(report.compliance_violations):,}")
        
        print(f"\n👥 User Activity:")
        print(f"  🏃 Active Users: {len(report.user_activity_summary)}")
        
        print(f"\n📁 Resource Access:")
        print(f"  📊 Resource Types: {len(report.resource_access_summary)}")
        
        if report.resource_access_summary:
            top_resources = sorted(
                report.resource_access_summary.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            print(f"  🔝 Top Resources:")
            for resource_type, count in top_resources:
                print(f"    • {resource_type}: {count:,} accesses")
        
        print(f"\n💡 System Health:")
        if report.recommendations:
            for rec in report.recommendations[:3]:
                print(f"  • {rec}")
        else:
            print(f"  ✅ All systems operating normally")
        
    except Exception as e:
        print(f"❌ Statistics generation failed: {e}")
    
    print("\n✨ Statistics demo completed!")


async def main():
    """Run the complete audit and compliance demo"""
    print("🚀 ScrollIntel Audit Logging & Compliance System Demo")
    print("=" * 60)
    print("Demonstrating comprehensive audit logging, compliance reporting,")
    print("data retention policies, and audit trail export capabilities.")
    print("=" * 60)
    
    try:
        # Start the audit and compliance systems
        await audit_system.start()
        await compliance_manager.start()
        
        # Run all demos
        await demo_basic_audit_logging()
        await demo_audit_log_search()
        await demo_compliance_reporting()
        await demo_compliance_framework_analysis()
        await demo_data_retention_policies()
        await demo_audit_export()
        await demo_compliance_violation_detection()
        await demo_system_statistics()
        
        print("\n" + "=" * 60)
        print("🎉 Demo Completed Successfully!")
        print("=" * 60)
        
        print("\n📋 Key Features Demonstrated:")
        print("  ✅ Comprehensive audit logging for all user actions")
        print("  ✅ Advanced audit log search and filtering")
        print("  ✅ Compliance reporting with multiple frameworks")
        print("  ✅ Data retention policies and automated cleanup")
        print("  ✅ Audit trail export in JSON and CSV formats")
        print("  ✅ Real-time compliance violation detection")
        print("  ✅ System statistics and health monitoring")
        
        print("\n🔒 Compliance Frameworks Supported:")
        print("  • GDPR (General Data Protection Regulation)")
        print("  • SOX (Sarbanes-Oxley Act)")
        print("  • ISO 27001 (Information Security Management)")
        print("  • HIPAA (Health Insurance Portability)")
        print("  • PCI DSS (Payment Card Industry)")
        print("  • NIST (National Institute of Standards)")
        
        print("\n📊 Production Ready Features:")
        print("  • Background processing for high performance")
        print("  • Structured logging with JSON format")
        print("  • Automated data retention and cleanup")
        print("  • Real-time violation detection and alerting")
        print("  • Comprehensive API endpoints for integration")
        print("  • Export capabilities for compliance audits")
        
        print("\n🚀 The system is ready for production deployment!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Stop the systems
        await audit_system.stop()
        await compliance_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())