"""
Comprehensive tests for audit and compliance system.
"""
import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch

from scrollintel.models.audit_models import (
    Base, AuditAction, AuditLogCreate, ComplianceRuleCreate,
    AccessControlCreate, ChangeApprovalCreate
)
from scrollintel.core.audit_logger import AuditLogger
from scrollintel.core.compliance_manager import ComplianceManager
from scrollintel.core.access_control import AccessControlManager
from scrollintel.core.change_approval import ChangeApprovalManager


# Test database setup
@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


class TestAuditLogger:
    """Test audit logging functionality."""
    
    def test_log_action_basic(self, db_session):
        """Test basic audit logging."""
        audit_logger = AuditLogger(db_session)
        
        audit_id = audit_logger.log_action(
            user_id="user123",
            user_email="test@example.com",
            action=AuditAction.CREATE,
            resource_type="prompt",
            resource_id="prompt123",
            resource_name="Test Prompt"
        )
        
        assert audit_id is not None
        
        # Verify log was created
        logs = audit_logger.get_audit_trail(limit=1)
        assert len(logs) == 1
        assert logs[0].user_id == "user123"
        assert logs[0].action == "create"
        assert logs[0].resource_type == "prompt"
    
    def test_log_action_with_changes(self, db_session):
        """Test audit logging with change tracking."""
        audit_logger = AuditLogger(db_session)
        
        old_values = {"content": "Old prompt", "tags": ["old"]}
        new_values = {"content": "New prompt", "tags": ["new", "updated"]}
        
        audit_id = audit_logger.log_action(
            user_id="user123",
            user_email="test@example.com",
            action=AuditAction.UPDATE,
            resource_type="prompt",
            resource_id="prompt123",
            old_values=old_values,
            new_values=new_values
        )
        
        logs = audit_logger.get_audit_trail(limit=1)
        assert len(logs) == 1
        
        # Check changes were calculated
        log = logs[0]
        assert log.changes is not None
        assert "content" in log.changes
        assert log.changes["content"]["action"] == "modified"
    
    def test_risk_assessment(self, db_session):
        """Test risk level assessment."""
        audit_logger = AuditLogger(db_session)
        
        # High risk action
        audit_logger.log_action(
            user_id="user123",
            user_email="test@example.com",
            action=AuditAction.DELETE,
            resource_type="production_prompt",
            resource_id="prompt123"
        )
        
        logs = audit_logger.get_audit_trail(limit=1)
        assert logs[0].risk_level == "high"
    
    def test_get_resource_history(self, db_session):
        """Test getting resource history."""
        audit_logger = AuditLogger(db_session)
        
        # Create multiple log entries for same resource
        for action in [AuditAction.CREATE, AuditAction.UPDATE, AuditAction.DELETE]:
            audit_logger.log_action(
                user_id="user123",
                user_email="test@example.com",
                action=action,
                resource_type="prompt",
                resource_id="prompt123"
            )
        
        history = audit_logger.get_resource_history("prompt", "prompt123")
        assert len(history) == 3
        
        # Should be ordered by timestamp descending
        actions = [log.action for log in history]
        assert actions == ["delete", "update", "create"]
    
    def test_search_audit_logs(self, db_session):
        """Test searching audit logs."""
        audit_logger = AuditLogger(db_session)
        
        audit_logger.log_action(
            user_id="user123",
            user_email="test@example.com",
            action=AuditAction.CREATE,
            resource_type="prompt",
            resource_id="prompt123",
            resource_name="Sensitive Prompt"
        )
        
        # Search by resource name
        results = audit_logger.search_audit_logs("Sensitive")
        assert len(results) == 1
        assert results[0].resource_name == "Sensitive Prompt"
    
    def test_audit_statistics(self, db_session):
        """Test audit statistics generation."""
        audit_logger = AuditLogger(db_session)
        
        # Create various audit entries
        actions = [AuditAction.CREATE, AuditAction.UPDATE, AuditAction.DELETE]
        for i, action in enumerate(actions):
            audit_logger.log_action(
                user_id=f"user{i}",
                user_email=f"user{i}@example.com",
                action=action,
                resource_type="prompt",
                resource_id=f"prompt{i}"
            )
        
        stats = audit_logger.get_audit_statistics()
        
        assert stats["total_actions"] == 3
        assert stats["action_counts"]["create"] == 1
        assert stats["action_counts"]["update"] == 1
        assert stats["action_counts"]["delete"] == 1


class TestComplianceManager:
    """Test compliance management functionality."""
    
    def test_create_rule(self, db_session):
        """Test creating compliance rules."""
        compliance_manager = ComplianceManager(db_session)
        
        rule_data = ComplianceRuleCreate(
            name="Test Rule",
            description="Test compliance rule",
            rule_type="content",
            conditions={"patterns": [r"\b\d{3}-\d{2}-\d{4}\b"]},
            actions={"block": True, "alert": True},
            severity="high"
        )
        
        rule = compliance_manager.create_rule(rule_data, "admin")
        
        assert rule.name == "Test Rule"
        assert rule.rule_type == "content"
        assert rule.severity == "high"
    
    def test_content_compliance_check(self, db_session):
        """Test content-based compliance checking."""
        compliance_manager = ComplianceManager(db_session)
        
        # Create rule for SSN detection
        rule_data = ComplianceRuleCreate(
            name="SSN Detection",
            rule_type="content",
            conditions={"patterns": [r"\b\d{3}-\d{2}-\d{4}\b"]},
            actions={"block": True},
            severity="high"
        )
        compliance_manager.create_rule(rule_data, "admin")
        
        # Test with content containing SSN
        resource_data = {
            "content": "Please use SSN 123-45-6789 for verification"
        }
        
        result = compliance_manager.check_compliance(
            resource_type="prompt",
            resource_data=resource_data,
            action="create",
            user_context={"user_id": "user123"}
        )
        
        assert not result["compliant"]
        assert len(result["violations"]) == 1
        assert result["violations"][0]["violation_type"] == "sensitive_pattern"
        assert "block" in result["actions_required"]
    
    def test_keyword_compliance_check(self, db_session):
        """Test keyword-based compliance checking."""
        compliance_manager = ComplianceManager(db_session)
        
        # Test with existing default rules (should include password detection)
        resource_data = {
            "content": "The password is secret123"
        }
        
        result = compliance_manager.check_compliance(
            resource_type="prompt",
            resource_data=resource_data,
            action="create",
            user_context={"user_id": "user123"}
        )
        
        # Should detect sensitive keyword
        assert not result["compliant"]
        violations = [v for v in result["violations"] if v["violation_type"] == "sensitive_keyword"]
        assert len(violations) > 0
    
    def test_approval_compliance_check(self, db_session):
        """Test approval-based compliance checking."""
        compliance_manager = ComplianceManager(db_session)
        
        # Test with production tagged resource
        resource_data = {
            "tags": ["production"],
            "content": "Production prompt"
        }
        
        result = compliance_manager.check_compliance(
            resource_type="prompt",
            resource_data=resource_data,
            action="update",
            user_context={"user_id": "user123"}
        )
        
        # Should require approval for production updates
        assert not result["compliant"]
        approval_violations = [v for v in result["violations"] if v["violation_type"] == "approval_required"]
        assert len(approval_violations) > 0
    
    def test_bulk_operation_detection(self, db_session):
        """Test bulk operation detection."""
        audit_logger = AuditLogger(db_session)
        compliance_manager = ComplianceManager(db_session)
        
        # Create multiple recent actions
        user_id = "user123"
        for i in range(15):  # Exceed bulk threshold
            audit_logger.log_action(
                user_id=user_id,
                user_email="test@example.com",
                action=AuditAction.CREATE,
                resource_type="prompt",
                resource_id=f"prompt{i}"
            )
        
        # Check compliance for another action
        result = compliance_manager.check_compliance(
            resource_type="prompt",
            resource_data={"content": "test"},
            action="create",
            user_context={"user_id": user_id}
        )
        
        # Should detect bulk operation
        bulk_violations = [v for v in result["violations"] if v["violation_type"] == "bulk_operation"]
        assert len(bulk_violations) > 0
    
    def test_generate_compliance_report(self, db_session):
        """Test compliance report generation."""
        audit_logger = AuditLogger(db_session)
        compliance_manager = ComplianceManager(db_session)
        
        # Create some audit entries and violations
        audit_logger.log_action(
            user_id="user123",
            user_email="test@example.com",
            action=AuditAction.CREATE,
            resource_type="prompt",
            resource_id="prompt123"
        )
        
        # Record a violation
        violation_data = {
            "rule_id": str(uuid.uuid4()),
            "violation_type": "test_violation",
            "description": "Test violation",
            "severity": "medium"
        }
        compliance_manager.record_violation(violation_data, "prompt", "prompt123")
        
        # Generate report
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)
        
        report = compliance_manager.generate_compliance_report(start_date, end_date)
        
        assert report.total_actions == 1
        assert report.violations_count == 1
        assert report.compliance_score < 100  # Should be reduced due to violation


class TestAccessControlManager:
    """Test access control functionality."""
    
    def test_grant_access(self, db_session):
        """Test granting access to resources."""
        access_manager = AccessControlManager(db_session)
        
        access_id = access_manager.grant_access(
            resource_type="prompt",
            resource_id="prompt123",
            granted_by="admin",
            user_id="user123",
            permissions=["read", "write"]
        )
        
        assert access_id is not None
    
    def test_check_permission_direct(self, db_session):
        """Test checking direct permissions."""
        access_manager = AccessControlManager(db_session)
        
        # Grant access
        access_manager.grant_access(
            resource_type="prompt",
            resource_id="prompt123",
            granted_by="admin",
            user_id="user123",
            permissions=["read", "write"]
        )
        
        # Check read permission
        result = access_manager.check_permission(
            user_id="user123",
            resource_type="prompt",
            resource_id="prompt123",
            permission="read"
        )
        
        assert result["allowed"] is True
        
        # Check write permission
        result = access_manager.check_permission(
            user_id="user123",
            resource_type="prompt",
            resource_id="prompt123",
            permission="write"
        )
        
        assert result["allowed"] is True
        
        # Check delete permission (not granted)
        result = access_manager.check_permission(
            user_id="user123",
            resource_type="prompt",
            resource_id="prompt123",
            permission="delete"
        )
        
        assert result["allowed"] is False
    
    def test_check_permission_hierarchical(self, db_session):
        """Test hierarchical permission checking."""
        access_manager = AccessControlManager(db_session)
        
        # Grant admin permission
        access_manager.grant_access(
            resource_type="prompt",
            resource_id="prompt123",
            granted_by="admin",
            user_id="user123",
            permissions=["admin"]
        )
        
        # Admin should have all permissions
        for permission in ["read", "write", "delete", "approve"]:
            result = access_manager.check_permission(
                user_id="user123",
                resource_type="prompt",
                resource_id="prompt123",
                permission=permission
            )
            assert result["allowed"] is True
    
    def test_role_based_access(self, db_session):
        """Test role-based access control."""
        access_manager = AccessControlManager(db_session)
        
        # Grant access to role
        access_manager.grant_access(
            resource_type="prompt",
            resource_id="prompt123",
            granted_by="admin",
            role="editor",
            permissions=["read", "write"]
        )
        
        # Check permission with role
        result = access_manager.check_permission(
            user_id="user123",
            resource_type="prompt",
            resource_id="prompt123",
            permission="read",
            user_roles=["editor"]
        )
        
        assert result["allowed"] is True
    
    def test_time_based_conditions(self, db_session):
        """Test time-based access conditions."""
        access_manager = AccessControlManager(db_session)
        
        # Grant access with time restrictions (business hours only)
        conditions = {
            "time_restrictions": {
                "allowed_hours": list(range(9, 17))  # 9 AM to 5 PM
            }
        }
        
        access_manager.grant_access(
            resource_type="prompt",
            resource_id="prompt123",
            granted_by="admin",
            user_id="user123",
            permissions=["read"],
            conditions=conditions
        )
        
        # Mock current time to be during business hours
        with patch('scrollintel.core.access_control.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 10, 0)  # 10 AM
            
            result = access_manager.check_permission(
                user_id="user123",
                resource_type="prompt",
                resource_id="prompt123",
                permission="read"
            )
            
            assert result["allowed"] is True
    
    def test_expired_access(self, db_session):
        """Test expired access controls."""
        access_manager = AccessControlManager(db_session)
        
        # Grant access that expires in the past
        expires_at = datetime.utcnow() - timedelta(hours=1)
        
        access_manager.grant_access(
            resource_type="prompt",
            resource_id="prompt123",
            granted_by="admin",
            user_id="user123",
            permissions=["read"],
            expires_at=expires_at
        )
        
        # Should not have access due to expiration
        result = access_manager.check_permission(
            user_id="user123",
            resource_type="prompt",
            resource_id="prompt123",
            permission="read"
        )
        
        assert result["allowed"] is False
    
    def test_revoke_access(self, db_session):
        """Test revoking access."""
        access_manager = AccessControlManager(db_session)
        
        # Grant access
        access_id = access_manager.grant_access(
            resource_type="prompt",
            resource_id="prompt123",
            granted_by="admin",
            user_id="user123",
            permissions=["read"]
        )
        
        # Verify access works
        result = access_manager.check_permission(
            user_id="user123",
            resource_type="prompt",
            resource_id="prompt123",
            permission="read"
        )
        assert result["allowed"] is True
        
        # Revoke access
        success = access_manager.revoke_access(access_id, "admin")
        assert success is True
        
        # Verify access is revoked
        result = access_manager.check_permission(
            user_id="user123",
            resource_type="prompt",
            resource_id="prompt123",
            permission="read"
        )
        assert result["allowed"] is False


class TestChangeApprovalManager:
    """Test change approval workflow functionality."""
    
    def test_request_approval(self, db_session):
        """Test requesting change approval."""
        approval_manager = ChangeApprovalManager(db_session)
        
        approval_id = approval_manager.request_approval(
            resource_type="production_prompt",
            resource_id="prompt123",
            change_description="Update prompt content",
            proposed_changes={"content": {"old": "old", "new": "new"}},
            requested_by="user123"
        )
        
        assert approval_id is not None
    
    def test_approve_change(self, db_session):
        """Test approving a change request."""
        approval_manager = ChangeApprovalManager(db_session)
        
        # Request approval
        approval_id = approval_manager.request_approval(
            resource_type="production_prompt",
            resource_id="prompt123",
            change_description="Update prompt content",
            proposed_changes={"content": {"old": "old", "new": "new"}},
            requested_by="user123"
        )
        
        # Mock approval permission check
        with patch.object(approval_manager, '_can_approve', return_value=True):
            result = approval_manager.approve_change(
                approval_id=approval_id,
                approver_id="approver123",
                approval_notes="Looks good"
            )
        
        assert result["success"] is True
        assert result["status"] == "approved"
    
    def test_reject_change(self, db_session):
        """Test rejecting a change request."""
        approval_manager = ChangeApprovalManager(db_session)
        
        # Request approval
        approval_id = approval_manager.request_approval(
            resource_type="production_prompt",
            resource_id="prompt123",
            change_description="Update prompt content",
            proposed_changes={"content": {"old": "old", "new": "new"}},
            requested_by="user123"
        )
        
        # Mock approval permission check
        with patch.object(approval_manager, '_can_approve', return_value=True):
            result = approval_manager.reject_change(
                approval_id=approval_id,
                approver_id="approver123",
                rejection_reason="Needs more work"
            )
        
        assert result["success"] is True
        assert result["status"] == "rejected"
    
    def test_get_pending_approvals(self, db_session):
        """Test getting pending approvals."""
        approval_manager = ChangeApprovalManager(db_session)
        
        # Create multiple approval requests
        for i in range(3):
            approval_manager.request_approval(
                resource_type="production_prompt",
                resource_id=f"prompt{i}",
                change_description=f"Update prompt {i}",
                proposed_changes={"content": {"old": "old", "new": "new"}},
                requested_by="user123",
                priority="high" if i == 0 else "normal"
            )
        
        pending = approval_manager.get_pending_approvals()
        
        assert len(pending) == 3
        # Should be ordered by priority then time
        assert pending[0].priority == "high"
    
    def test_check_approval_status(self, db_session):
        """Test checking approval status for resource."""
        approval_manager = ChangeApprovalManager(db_session)
        
        # Request approval
        approval_manager.request_approval(
            resource_type="production_prompt",
            resource_id="prompt123",
            change_description="Update prompt content",
            proposed_changes={"content": {"old": "old", "new": "new"}},
            requested_by="user123"
        )
        
        status = approval_manager.check_approval_status("production_prompt", "prompt123")
        
        assert status["has_pending"] is True
        assert "approval_id" in status
    
    def test_minor_change_auto_approval(self, db_session):
        """Test auto-approval of minor changes."""
        approval_manager = ChangeApprovalManager(db_session)
        
        # Minor change (only description)
        approval_id = approval_manager.request_approval(
            resource_type="sensitive_template",  # Has auto_approve_minor = True
            resource_id="template123",
            change_description="Update description",
            proposed_changes={"description": {"old": "old desc", "new": "new desc"}},
            requested_by="user123"
        )
        
        # Should not require approval for minor changes
        assert approval_id is None


class TestIntegration:
    """Test integration between audit and compliance components."""
    
    def test_audit_compliance_integration(self, db_session):
        """Test integration between audit logging and compliance checking."""
        audit_logger = AuditLogger(db_session)
        compliance_manager = ComplianceManager(db_session)
        
        # Log an action that should trigger compliance check
        resource_data = {
            "content": "Password is secret123",
            "tags": ["production"]
        }
        
        # Check compliance first
        compliance_result = compliance_manager.check_compliance(
            resource_type="prompt",
            resource_data=resource_data,
            action="create",
            user_context={"user_id": "user123"}
        )
        
        # Log the action with compliance results
        audit_logger.log_action(
            user_id="user123",
            user_email="test@example.com",
            action=AuditAction.CREATE,
            resource_type="prompt",
            resource_id="prompt123",
            new_values=resource_data,
            compliance_tags=["sensitive_content", "production"],
            risk_level=compliance_result["risk_level"],
            metadata={"compliance_violations": compliance_result["violations"]}
        )
        
        # Verify audit log includes compliance information
        logs = audit_logger.get_audit_trail(limit=1)
        assert len(logs) == 1
        assert "sensitive_content" in logs[0].compliance_tags
        assert logs[0].risk_level in ["medium", "high"]
    
    def test_access_control_audit_integration(self, db_session):
        """Test integration between access control and audit logging."""
        access_manager = AccessControlManager(db_session)
        
        # Grant access (should be audited)
        access_id = access_manager.grant_access(
            resource_type="prompt",
            resource_id="prompt123",
            granted_by="admin",
            user_id="user123",
            permissions=["read", "write"]
        )
        
        # Check permission (should be audited)
        result = access_manager.validate_permission_request(
            user_id="user123",
            resource_type="prompt",
            resource_id="prompt123",
            permission="read",
            context={"user_email": "test@example.com"}
        )
        
        assert result["allowed"] is True
        
        # Verify audit trail
        audit_logger = AuditLogger(db_session)
        logs = audit_logger.get_audit_trail()
        
        # Should have audit log for permission check
        permission_logs = [log for log in logs if log.action == "permission_check"]
        assert len(permission_logs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])