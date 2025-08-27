"""
Demo script for audit and compliance system.
"""
import os
import sys
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.models.audit_models import Base, AuditAction, ComplianceRuleCreate
from scrollintel.core.audit_logger import AuditLogger
from scrollintel.core.compliance_manager import ComplianceManager
from scrollintel.core.access_control import AccessControlManager
from scrollintel.core.change_approval import ChangeApprovalManager


def setup_demo_database():
    """Set up demo database."""
    engine = create_engine("sqlite:///demo_audit_compliance.db")
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def demo_audit_logging(session):
    """Demonstrate audit logging functionality."""
    print("\n" + "="*60)
    print("🔍 AUDIT LOGGING DEMO")
    print("="*60)
    
    audit_logger = AuditLogger(session)
    
    # Demo 1: Basic audit logging
    print("\n1. Basic Audit Logging")
    print("-" * 30)
    
    audit_id = audit_logger.log_action(
        user_id="user123",
        user_email="john.doe@company.com",
        action=AuditAction.CREATE,
        resource_type="prompt",
        resource_id="prompt_001",
        resource_name="Customer Service Prompt",
        new_values={
            "content": "How can I help you today?",
            "category": "customer_service",
            "tags": ["greeting", "support"]
        },
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        metadata={"department": "customer_support"}
    )
    
    print(f"✅ Created audit log: {audit_id}")
    
    # Demo 2: Change tracking
    print("\n2. Change Tracking")
    print("-" * 20)
    
    old_values = {
        "content": "How can I help you today?",
        "tags": ["greeting", "support"]
    }
    
    new_values = {
        "content": "Hello! How may I assist you today?",
        "tags": ["greeting", "support", "polite"]
    }
    
    audit_logger.log_action(
        user_id="user456",
        user_email="jane.smith@company.com",
        action=AuditAction.UPDATE,
        resource_type="prompt",
        resource_id="prompt_001",
        resource_name="Customer Service Prompt",
        old_values=old_values,
        new_values=new_values
    )
    
    print("✅ Logged prompt update with change tracking")
    
    # Demo 3: High-risk action
    print("\n3. High-Risk Action Logging")
    print("-" * 30)
    
    audit_logger.log_action(
        user_id="admin789",
        user_email="admin@company.com",
        action=AuditAction.DELETE,
        resource_type="production_prompt",
        resource_id="prompt_001",
        resource_name="Customer Service Prompt",
        old_values=new_values,
        compliance_tags=["production", "critical"]
    )
    
    print("✅ Logged high-risk deletion")
    
    # Demo 4: Get audit trail
    print("\n4. Audit Trail Retrieval")
    print("-" * 30)
    
    logs = audit_logger.get_audit_trail(resource_id="prompt_001", limit=10)
    
    print(f"Found {len(logs)} audit entries for prompt_001:")
    for log in logs:
        print(f"  • {log.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {log.action} by {log.user_email} (Risk: {log.risk_level})")
    
    # Demo 5: Search functionality
    print("\n5. Audit Search")
    print("-" * 20)
    
    search_results = audit_logger.search_audit_logs("Customer Service")
    print(f"Found {len(search_results)} entries matching 'Customer Service'")
    
    # Demo 6: Statistics
    print("\n6. Audit Statistics")
    print("-" * 25)
    
    stats = audit_logger.get_audit_statistics()
    print(f"Total actions: {stats['total_actions']}")
    print("Action breakdown:")
    for action, count in stats['action_counts'].items():
        print(f"  • {action}: {count}")
    
    print("Risk level distribution:")
    for risk, count in stats['risk_counts'].items():
        print(f"  • {risk}: {count}")


def demo_compliance_management(session):
    """Demonstrate compliance management functionality."""
    print("\n" + "="*60)
    print("⚖️  COMPLIANCE MANAGEMENT DEMO")
    print("="*60)
    
    compliance_manager = ComplianceManager(session)
    
    # Demo 1: Create custom compliance rule
    print("\n1. Creating Custom Compliance Rule")
    print("-" * 40)
    
    custom_rule = ComplianceRuleCreate(
        name="PII Detection Rule",
        description="Detect personally identifiable information in prompts",
        rule_type="content",
        conditions={
            "patterns": [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"  # Credit card
            ],
            "keywords": ["social security", "credit card", "driver license"]
        },
        actions={
            "block": True,
            "alert": True,
            "require_approval": True
        },
        severity="critical"
    )
    
    rule = compliance_manager.create_rule(custom_rule, "compliance_officer")
    print(f"✅ Created compliance rule: {rule.name}")
    
    # Demo 2: Test compliance with clean content
    print("\n2. Testing Compliant Content")
    print("-" * 35)
    
    clean_resource = {
        "content": "Welcome to our service! How can we help you today?",
        "category": "greeting",
        "tags": ["customer_service"]
    }
    
    result = compliance_manager.check_compliance(
        resource_type="prompt",
        resource_data=clean_resource,
        action="create",
        user_context={"user_id": "user123", "department": "support"}
    )
    
    print(f"Compliance check result: {'✅ COMPLIANT' if result['compliant'] else '❌ NON-COMPLIANT'}")
    print(f"Risk level: {result['risk_level']}")
    
    # Demo 3: Test compliance with sensitive content
    print("\n3. Testing Non-Compliant Content")
    print("-" * 40)
    
    sensitive_resource = {
        "content": "Please provide your social security number 123-45-6789 for verification",
        "category": "verification",
        "tags": ["identity"]
    }
    
    result = compliance_manager.check_compliance(
        resource_type="prompt",
        resource_data=sensitive_resource,
        action="create",
        user_context={"user_id": "user456"}
    )
    
    print(f"Compliance check result: {'✅ COMPLIANT' if result['compliant'] else '❌ NON-COMPLIANT'}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Violations found: {len(result['violations'])}")
    
    for violation in result['violations']:
        print(f"  • {violation['violation_type']}: {violation['description']}")
    
    print(f"Required actions: {', '.join(result['actions_required'])}")
    
    # Record the violation
    if result['violations']:
        violation_id = compliance_manager.record_violation(
            result['violations'][0], "prompt", "prompt_sensitive"
        )
        print(f"✅ Recorded violation: {violation_id}")
    
    # Demo 4: Test production approval requirement
    print("\n4. Testing Production Approval Requirement")
    print("-" * 50)
    
    production_resource = {
        "content": "Updated customer greeting message",
        "tags": ["production", "customer_service"],
        "environment": "production"
    }
    
    result = compliance_manager.check_compliance(
        resource_type="prompt",
        resource_data=production_resource,
        action="update",
        user_context={"user_id": "user789"}
    )
    
    print(f"Compliance check result: {'✅ COMPLIANT' if result['compliant'] else '❌ REQUIRES APPROVAL'}")
    
    if not result['compliant']:
        approval_violations = [v for v in result['violations'] if v['violation_type'] == 'approval_required']
        if approval_violations:
            print("  • Production changes require approval")
    
    # Demo 5: Get compliance violations
    print("\n5. Compliance Violations Report")
    print("-" * 40)
    
    violations = compliance_manager.get_violations(limit=10)
    print(f"Found {len(violations)} compliance violations:")
    
    for violation in violations:
        print(f"  • {violation.violation_type} - {violation.severity} - {violation.status}")
    
    # Demo 6: Generate compliance report
    print("\n6. Compliance Report Generation")
    print("-" * 40)
    
    start_date = datetime.utcnow() - timedelta(days=7)
    end_date = datetime.utcnow()
    
    report = compliance_manager.generate_compliance_report(start_date, end_date)
    
    print(f"Compliance Report (Last 7 days):")
    print(f"  • Total actions: {report.total_actions}")
    print(f"  • Violations: {report.violations_count}")
    print(f"  • Compliance score: {report.compliance_score:.1f}%")
    print(f"  • Risk summary: {report.risk_summary}")
    
    if report.recommendations:
        print("  • Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")


def demo_access_control(session):
    """Demonstrate access control functionality."""
    print("\n" + "="*60)
    print("🔐 ACCESS CONTROL DEMO")
    print("="*60)
    
    access_manager = AccessControlManager(session)
    
    # Demo 1: Grant basic access
    print("\n1. Granting Basic Access")
    print("-" * 30)
    
    access_id = access_manager.grant_access(
        resource_type="prompt",
        resource_id="prompt_001",
        granted_by="admin",
        user_id="user123",
        permissions=["read", "write"]
    )
    
    print(f"✅ Granted access: {access_id}")
    
    # Demo 2: Check permissions
    print("\n2. Checking Permissions")
    print("-" * 30)
    
    # Test read permission
    result = access_manager.check_permission(
        user_id="user123",
        resource_type="prompt",
        resource_id="prompt_001",
        permission="read"
    )
    
    print(f"Read permission: {'✅ ALLOWED' if result['allowed'] else '❌ DENIED'}")
    
    # Test delete permission (not granted)
    result = access_manager.check_permission(
        user_id="user123",
        resource_type="prompt",
        resource_id="prompt_001",
        permission="delete"
    )
    
    print(f"Delete permission: {'✅ ALLOWED' if result['allowed'] else '❌ DENIED'}")
    
    # Demo 3: Role-based access
    print("\n3. Role-Based Access Control")
    print("-" * 40)
    
    # Grant access to editor role
    access_manager.grant_access(
        resource_type="template",
        resource_id="template_001",
        granted_by="admin",
        role="editor",
        permissions=["read", "write", "approve"]
    )
    
    # Check permission with role
    result = access_manager.check_permission(
        user_id="user456",
        resource_type="template",
        resource_id="template_001",
        permission="approve",
        user_roles=["editor", "reviewer"]
    )
    
    print(f"Approve permission (editor role): {'✅ ALLOWED' if result['allowed'] else '❌ DENIED'}")
    
    # Demo 4: Time-based access
    print("\n4. Time-Based Access Control")
    print("-" * 40)
    
    # Grant access with business hours restriction
    conditions = {
        "time_restrictions": {
            "allowed_hours": list(range(9, 17)),  # 9 AM to 5 PM
            "allowed_days": [0, 1, 2, 3, 4]  # Monday to Friday
        }
    }
    
    access_manager.grant_access(
        resource_type="sensitive_prompt",
        resource_id="prompt_sensitive",
        granted_by="admin",
        user_id="user789",
        permissions=["read"],
        conditions=conditions
    )
    
    print("✅ Granted time-restricted access (business hours only)")
    
    # Demo 5: Temporary access
    print("\n5. Temporary Access Control")
    print("-" * 40)
    
    # Grant access that expires in 1 hour
    expires_at = datetime.utcnow() + timedelta(hours=1)
    
    temp_access_id = access_manager.grant_access(
        resource_type="experiment",
        resource_id="exp_001",
        granted_by="admin",
        user_id="contractor123",
        permissions=["read"],
        expires_at=expires_at
    )
    
    print(f"✅ Granted temporary access (expires in 1 hour): {temp_access_id}")
    
    # Demo 6: Get user permissions
    print("\n6. User Permissions Summary")
    print("-" * 40)
    
    permissions = access_manager.get_user_permissions("user123")
    print(f"User123 has {len(permissions)} access grants:")
    
    for perm in permissions:
        print(f"  • {perm['resource_type']}:{perm['resource_id']} - {perm['permissions']}")
    
    # Demo 7: Access statistics
    print("\n7. Access Control Statistics")
    print("-" * 40)
    
    stats = access_manager.get_access_statistics()
    print(f"Total access controls: {stats['total_access_controls']}")
    print(f"Active access controls: {stats['active_access_controls']}")
    print(f"Expired access controls: {stats['expired_access_controls']}")
    
    print("Permission distribution:")
    for perm, count in stats['permission_distribution'].items():
        print(f"  • {perm}: {count}")


def demo_change_approval(session):
    """Demonstrate change approval workflow."""
    print("\n" + "="*60)
    print("📋 CHANGE APPROVAL WORKFLOW DEMO")
    print("="*60)
    
    approval_manager = ChangeApprovalManager(session)
    
    # Demo 1: Request approval for production change
    print("\n1. Requesting Production Change Approval")
    print("-" * 50)
    
    approval_id = approval_manager.request_approval(
        resource_type="production_prompt",
        resource_id="prompt_prod_001",
        change_description="Update customer greeting to be more friendly",
        proposed_changes={
            "content": {
                "old": "Hello. How can I help?",
                "new": "Hi there! How can I help you today? 😊"
            },
            "tone": {
                "old": "formal",
                "new": "friendly"
            }
        },
        requested_by="user123",
        priority="normal"
    )
    
    if approval_id:
        print(f"✅ Approval request created: {approval_id}")
    else:
        print("ℹ️  No approval required for this change")
    
    # Demo 2: Request urgent approval
    print("\n2. Requesting Urgent Approval")
    print("-" * 35)
    
    urgent_approval_id = approval_manager.request_approval(
        resource_type="production_prompt",
        resource_id="prompt_urgent_001",
        change_description="Fix critical bug in payment prompt",
        proposed_changes={
            "content": {
                "old": "Payment failed. Try again.",
                "new": "Payment processing failed. Please check your payment method and try again."
            }
        },
        requested_by="user456",
        priority="urgent",
        deadline=datetime.utcnow() + timedelta(hours=2)
    )
    
    if urgent_approval_id:
        print(f"✅ Urgent approval request created: {urgent_approval_id}")
    
    # Demo 3: Get pending approvals
    print("\n3. Pending Approvals Dashboard")
    print("-" * 40)
    
    pending = approval_manager.get_pending_approvals()
    print(f"Found {len(pending)} pending approvals:")
    
    for approval in pending:
        print(f"  • {approval.resource_type}:{approval.resource_id}")
        print(f"    Priority: {approval.priority}")
        print(f"    Requested by: {approval.requested_by}")
        print(f"    Description: {approval.change_description}")
        if approval.deadline:
            print(f"    Deadline: {approval.deadline.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Demo 4: Approve a change (mock approval permissions)
    if pending:
        print("4. Approving a Change Request")
        print("-" * 40)
        
        # Mock the permission check to allow approval
        with patch.object(approval_manager, '_can_approve', return_value=True):
            result = approval_manager.approve_change(
                approval_id=pending[0].id,
                approver_id="approver123",
                approval_notes="Change looks good. Approved for deployment."
            )
        
        if result["success"]:
            print(f"✅ Change approved: {result['approval_id']}")
        else:
            print(f"❌ Approval failed: {result['error']}")
    
    # Demo 5: Check approval status
    print("\n5. Checking Approval Status")
    print("-" * 35)
    
    status = approval_manager.check_approval_status("production_prompt", "prompt_prod_001")
    
    if status["has_pending"]:
        print("⏳ Resource has pending approval")
        print(f"   Approval ID: {status['approval_id']}")
    elif status.get("recently_approved"):
        print("✅ Resource was recently approved")
        print(f"   Approved at: {status['approved_at']}")
    else:
        print("ℹ️  No pending or recent approvals")
    
    # Demo 6: Approval statistics
    print("\n6. Approval Workflow Statistics")
    print("-" * 40)
    
    stats = approval_manager.get_approval_statistics()
    print(f"Total approval requests: {stats['total_requests']}")
    print(f"Approval rate: {stats['approval_rate']:.1f}%")
    print(f"Average approval time: {stats['average_approval_time_hours']:.1f} hours")
    
    print("Status distribution:")
    for status, count in stats['status_distribution'].items():
        print(f"  • {status}: {count}")


def demo_integration_scenarios(session):
    """Demonstrate integration scenarios."""
    print("\n" + "="*60)
    print("🔗 INTEGRATION SCENARIOS DEMO")
    print("="*60)
    
    audit_logger = AuditLogger(session)
    compliance_manager = ComplianceManager(session)
    access_manager = AccessControlManager(session)
    approval_manager = ChangeApprovalManager(session)
    
    # Scenario 1: Complete workflow for sensitive prompt creation
    print("\n1. Sensitive Prompt Creation Workflow")
    print("-" * 50)
    
    user_id = "data_scientist_001"
    resource_data = {
        "content": "Please enter your email address user@company.com for verification",
        "category": "data_collection",
        "tags": ["sensitive", "pii"]
    }
    
    # Step 1: Check compliance
    print("Step 1: Checking compliance...")
    compliance_result = compliance_manager.check_compliance(
        resource_type="prompt",
        resource_data=resource_data,
        action="create",
        user_context={"user_id": user_id}
    )
    
    print(f"Compliance: {'✅ PASS' if compliance_result['compliant'] else '❌ FAIL'}")
    
    if not compliance_result['compliant']:
        print("Violations detected:")
        for violation in compliance_result['violations']:
            print(f"  • {violation['description']}")
        
        # Step 2: Request approval if required
        if "require_approval" in compliance_result['actions_required']:
            print("\nStep 2: Requesting approval...")
            approval_id = approval_manager.request_approval(
                resource_type="prompt",
                resource_id="prompt_sensitive_001",
                change_description="Create prompt with PII collection",
                proposed_changes={"content": resource_data},
                requested_by=user_id,
                priority="high"
            )
            print(f"Approval requested: {approval_id}")
    
    # Step 3: Log the action
    print("\nStep 3: Logging audit trail...")
    audit_logger.log_action(
        user_id=user_id,
        user_email="scientist@company.com",
        action=AuditAction.CREATE,
        resource_type="prompt",
        resource_id="prompt_sensitive_001",
        new_values=resource_data,
        compliance_tags=["pii_detected", "approval_required"],
        metadata={
            "compliance_violations": compliance_result['violations'],
            "workflow_step": "creation_attempt",
            "compliance_risk_level": compliance_result['risk_level']
        }
    )
    print("✅ Audit trail logged")
    
    # Scenario 2: Access control with audit
    print("\n2. Access Control with Audit Trail")
    print("-" * 45)
    
    # Grant access
    print("Granting access to sensitive resource...")
    access_id = access_manager.grant_access(
        resource_type="sensitive_prompt",
        resource_id="prompt_sensitive_001",
        granted_by="security_admin",
        user_id="analyst_001",
        permissions=["read"],
        conditions={
            "time_restrictions": {
                "allowed_hours": list(range(9, 17))
            }
        }
    )
    
    # Check permission with audit
    print("Checking permission (with audit)...")
    result = access_manager.validate_permission_request(
        user_id="analyst_001",
        resource_type="sensitive_prompt",
        resource_id="prompt_sensitive_001",
        permission="read",
        context={
            "user_email": "analyst@company.com",
            "ip_address": "192.168.1.50"
        }
    )
    
    print(f"Access: {'✅ GRANTED' if result['allowed'] else '❌ DENIED'}")
    
    # Scenario 3: Compliance violation handling
    print("\n3. Compliance Violation Handling")
    print("-" * 45)
    
    # Simulate bulk operation that triggers violation
    print("Simulating bulk operations...")
    for i in range(12):  # Exceed bulk threshold
        audit_logger.log_action(
            user_id="bulk_user_001",
            user_email="bulk@company.com",
            action=AuditAction.CREATE,
            resource_type="prompt",
            resource_id=f"bulk_prompt_{i}",
            metadata={"batch_operation": True}
        )
    
    # Check compliance for next operation
    compliance_result = compliance_manager.check_compliance(
        resource_type="prompt",
        resource_data={"content": "Another prompt"},
        action="create",
        user_context={"user_id": "bulk_user_001"}
    )
    
    if not compliance_result['compliant']:
        bulk_violations = [v for v in compliance_result['violations'] if v['violation_type'] == 'bulk_operation']
        if bulk_violations:
            print("⚠️  Bulk operation violation detected")
            violation_id = compliance_manager.record_violation(
                bulk_violations[0], "prompt", "bulk_operation_001"
            )
            print(f"Violation recorded: {violation_id}")


def main():
    """Run the complete audit and compliance demo."""
    print("🚀 AUDIT AND COMPLIANCE SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive audit and compliance")
    print("system for prompt management, including:")
    print("• Audit logging and trail management")
    print("• Compliance rule enforcement")
    print("• Access control and permissions")
    print("• Change approval workflows")
    print("• Integration scenarios")
    
    # Set up demo database
    session = setup_demo_database()
    
    try:
        # Run all demos
        demo_audit_logging(session)
        demo_compliance_management(session)
        demo_access_control(session)
        demo_change_approval(session)
        demo_integration_scenarios(session)
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("• ✅ Comprehensive audit logging with change tracking")
        print("• ✅ Automated compliance checking and violation detection")
        print("• ✅ Flexible access control with time/condition-based restrictions")
        print("• ✅ Change approval workflow for sensitive operations")
        print("• ✅ Integration between all audit and compliance components")
        print("• ✅ Reporting and analytics capabilities")
        
        print(f"\nDemo database created: demo_audit_compliance.db")
        print("You can examine the database to see all the audit trails,")
        print("compliance rules, access controls, and approval records.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        session.close()


if __name__ == "__main__":
    main()