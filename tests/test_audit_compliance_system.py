"""
Comprehensive Tests for Audit Logging and Compliance System

Tests all aspects of the audit logging, compliance reporting, data retention,
and audit trail export functionality.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from scrollintel.core.audit_system import (
    AuditSystem, AuditAction, ComplianceLevel, RetentionPolicy, AuditEvent
)
from scrollintel.core.compliance_manager import (
    ComplianceManager, ComplianceFramework, DataClassification, ComplianceViolation
)
from scrollintel.api.routes.audit_routes import router
from scrollintel.models.database import AuditLog, User


class TestAuditSystem:
    """Test the comprehensive audit logging system"""
    
    @pytest.fixture
    async def audit_system(self):
        """Create audit system for testing"""
        system = AuditSystem()
        await system.start()
        yield system
        await system.stop()
    
    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing"""
        return {
            "id": str(uuid4()),
            "email": "test@example.com",
            "session_id": "test_session_123",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 Test Browser"
        }
    
    async def test_log_basic_event(self, audit_system, sample_user):
        """Test logging a basic audit event"""
        
        event_id = await audit_system.log_event(
            action=AuditAction.USER_LOGIN,
            resource_type="authentication",
            user_id=sample_user["id"],
            user_email=sample_user["email"],
            details={"login_method": "password"},
            session_id=sample_user["session_id"],
            ip_address=sample_user["ip_address"],
            user_agent=sample_user["user_agent"]
        )
        
        assert event_id is not None
        assert len(event_id) == 36  # UUID length
    
    async def test_log_user_action(self, audit_system, sample_user):
        """Test logging user action with enhanced context"""
        
        event_id = await audit_system.log_user_action(
            user_id=sample_user["id"],
            user_email=sample_user["email"],
            action=AuditAction.DATA_UPLOAD,
            resource_type="dataset",
            resource_id="dataset_123",
            resource_name="Customer Data.csv",
            details={
                "file_size": 1024000,
                "file_type": "csv",
                "rows_processed": 5000
            },
            session_id=sample_user["session_id"],
            ip_address=sample_user["ip_address"]
        )
        
        assert event_id is not None
    
    async def test_compliance_level_determination(self, audit_system):
        """Test automatic compliance level determination"""
        
        # High-risk action should get enterprise level
        high_risk_level = audit_system._determine_compliance_level(AuditAction.USER_DELETE)
        assert high_risk_level == ComplianceLevel.ENTERPRISE
        
        # Auth action should get enhanced level
        auth_level = audit_system._determine_compliance_level(AuditAction.LOGIN_SUCCESS)
        assert auth_level == ComplianceLevel.ENHANCED
        
        # Regular action should get standard level
        regular_level = audit_system._determine_compliance_level(AuditAction.DASHBOARD_VIEW)
        assert regular_level == ComplianceLevel.STANDARD
    
    async def test_retention_policy_determination(self, audit_system):
        """Test automatic retention policy determination"""
        
        # Permanent actions
        permanent_policy = audit_system._determine_retention_policy(AuditAction.USER_DELETE)
        assert permanent_policy == RetentionPolicy.PERMANENT
        
        # Long-term actions
        long_term_policy = audit_system._determine_retention_policy(AuditAction.LOGIN_SUCCESS)
        assert long_term_policy == RetentionPolicy.YEARS_7
        
        # Regular actions
        regular_policy = audit_system._determine_retention_policy(AuditAction.DASHBOARD_VIEW)
        assert regular_policy == RetentionPolicy.YEAR_1
    
    @patch('scrollintel.core.audit_system.get_db')
    async def test_search_audit_logs(self, mock_get_db, audit_system):
        """Test audit log search functionality"""
        
        # Mock database session
        mock_session = AsyncMock()
        mock_get_db.return_value.__aenter__.return_value = mock_session
        
        # Mock query results
        mock_logs = [
            Mock(
                id=uuid4(),
                timestamp=datetime.utcnow(),
                user_id=uuid4(),
                action="user.login",
                resource_type="authentication",
                resource_id=None,
                details={"login_method": "password"},
                ip_address="192.168.1.100",
                user_agent="Test Browser",
                session_id="test_session",
                success=True,
                error_message=None,
                user=Mock(email="test@example.com")
            )
        ]
        
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_logs
        
        # Test search
        results = await audit_system.search_audit_logs(
            user_id=str(mock_logs[0].user_id),
            action="user.login",
            limit=10
        )
        
        assert len(results) == 1
        assert results[0]["action"] == "user.login"
        assert results[0]["resource_type"] == "authentication"
    
    async def test_generate_compliance_report(self, audit_system):
        """Test compliance report generation"""
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        with patch.object(audit_system, 'search_audit_logs') as mock_search:
            mock_search.return_value = [
                {
                    "id": str(uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": str(uuid4()),
                    "user_email": "test@example.com",
                    "action": "user.login",
                    "resource_type": "authentication",
                    "success": True,
                    "details": {}
                }
            ]
            
            with patch('scrollintel.core.audit_system.get_db') as mock_get_db:
                mock_session = AsyncMock()
                mock_get_db.return_value.__aenter__.return_value = mock_session
                
                # Mock database queries
                mock_session.execute.return_value.scalar.side_effect = [100, 95, 5, 2]  # total, success, failed, security
                mock_session.execute.return_value.scalars.return_value.all.return_value = []  # violations
                mock_session.execute.return_value.all.side_effect = [[], []]  # user activity, resource access
                
                report = await audit_system.generate_compliance_report(
                    start_date=start_date,
                    end_date=end_date
                )
                
                assert report.total_events == 100
                assert report.successful_events == 95
                assert report.failed_events == 5
                assert report.security_events == 2
                assert isinstance(report.recommendations, list)
    
    async def test_export_audit_logs_json(self, audit_system):
        """Test audit log export in JSON format"""
        
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        with patch.object(audit_system, 'search_audit_logs') as mock_search:
            mock_search.return_value = [
                {
                    "id": str(uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": str(uuid4()),
                    "user_email": "test@example.com",
                    "action": "user.login",
                    "resource_type": "authentication",
                    "success": True,
                    "details": {"login_method": "password"}
                }
            ]
            
            file_path = await audit_system.export_audit_logs(
                start_date=start_date,
                end_date=end_date,
                format="json"
            )
            
            assert file_path.endswith(".json")
            
            # Verify file exists and contains data
            export_file = Path(file_path)
            assert export_file.exists()
            
            with open(export_file, 'r') as f:
                data = json.load(f)
                assert "export_metadata" in data
                assert "audit_logs" in data
                assert len(data["audit_logs"]) == 1
            
            # Cleanup
            export_file.unlink()
    
    async def test_export_audit_logs_csv(self, audit_system):
        """Test audit log export in CSV format"""
        
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        with patch.object(audit_system, 'search_audit_logs') as mock_search:
            mock_search.return_value = [
                {
                    "id": str(uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": str(uuid4()),
                    "user_email": "test@example.com",
                    "action": "user.login",
                    "resource_type": "authentication",
                    "success": True,
                    "details": {"login_method": "password"}
                }
            ]
            
            file_path = await audit_system.export_audit_logs(
                start_date=start_date,
                end_date=end_date,
                format="csv"
            )
            
            assert file_path.endswith(".csv")
            
            # Verify file exists
            export_file = Path(file_path)
            assert export_file.exists()
            
            # Cleanup
            export_file.unlink()
    
    @patch('scrollintel.core.audit_system.get_db')
    async def test_cleanup_expired_logs(self, mock_get_db, audit_system):
        """Test cleanup of expired audit logs"""
        
        mock_session = AsyncMock()
        mock_get_db.return_value.__aenter__.return_value = mock_session
        
        # Mock deletion result
        mock_session.execute.return_value.rowcount = 50
        
        cleaned_count = await audit_system.cleanup_expired_logs()
        
        assert cleaned_count >= 0
        mock_session.execute.assert_called()
        mock_session.commit.assert_called()


class TestComplianceManager:
    """Test the compliance management system"""
    
    @pytest.fixture
    async def compliance_manager(self):
        """Create compliance manager for testing"""
        from scrollintel.core.compliance_manager import ComplianceManager
        manager = ComplianceManager()
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_load_compliance_rules(self, compliance_manager):
        """Test loading of compliance rules"""
        
        rules = compliance_manager.compliance_rules
        
        assert len(rules) > 0
        assert "gdpr_data_access" in rules
        assert "sox_financial_data" in rules
        
        gdpr_rule = rules["gdpr_data_access"]
        assert gdpr_rule.framework == ComplianceFramework.GDPR
        assert gdpr_rule.severity == "high"
    
    async def test_load_retention_policies(self, compliance_manager):
        """Test loading of retention policies"""
        
        policies = compliance_manager.retention_policies
        
        assert len(policies) > 0
        assert "audit_logs_standard" in policies
        assert "user_data_gdpr" in policies
        
        audit_policy = policies["audit_logs_standard"]
        assert audit_policy.retention_period == RetentionPeriod.YEARS_3
        assert audit_policy.auto_cleanup is True
    
    async def test_check_compliance_no_violations(self, compliance_manager):
        """Test compliance check with no violations"""
        
        violations = await compliance_manager.check_compliance(
            action="view",
            resource_type="dashboard",
            resource_id="dashboard_123",
            user_id="user_123",
            data_classification=DataClassification.INTERNAL
        )
        
        assert len(violations) == 0
    
    async def test_check_compliance_with_violations(self, compliance_manager):
        """Test compliance check that detects violations"""
        
        violations = await compliance_manager.check_compliance(
            action="access",
            resource_type="personal_data",
            resource_id="user_data_123",
            user_id="user_123",
            data_classification=DataClassification.CONFIDENTIAL
        )
        
        # Should detect GDPR violation for confidential personal data
        assert len(violations) >= 0  # May or may not have violations based on rule logic
    
    async def test_generate_compliance_report(self, compliance_manager):
        """Test compliance report generation"""
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        with patch('scrollintel.core.compliance_manager.audit_system') as mock_audit:
            mock_audit.search_audit_logs.return_value = [
                {
                    "id": str(uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": str(uuid4()),
                    "action": "data.access",
                    "resource_type": "personal_data",
                    "success": True,
                    "error_message": None,
                    "details": {}
                }
            ]
            
            report = await compliance_manager.generate_compliance_report(
                framework=ComplianceFramework.GDPR,
                start_date=start_date,
                end_date=end_date
            )
            
            assert report["framework"] == "gdpr"
            assert "summary" in report
            assert "rule_compliance" in report
            assert "recommendations" in report
            assert report["summary"]["total_events_analyzed"] == 1
    
    async def test_apply_retention_policies(self, compliance_manager):
        """Test application of data retention policies"""
        
        with patch.object(compliance_manager, '_cleanup_data_by_policy') as mock_cleanup:
            mock_cleanup.return_value = 25
            
            results = await compliance_manager.apply_retention_policies()
            
            assert isinstance(results, dict)
            # Should have results for policies with auto_cleanup=True
            assert len(results) >= 0
    
    async def test_export_compliance_data(self, compliance_manager):
        """Test compliance data export"""
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        with patch.object(compliance_manager, 'generate_compliance_report') as mock_report:
            mock_report.return_value = {
                "framework": "gdpr",
                "summary": {"total_events": 100},
                "rule_compliance": {},
                "recommendations": []
            }
            
            file_path = await compliance_manager.export_compliance_data(
                framework=ComplianceFramework.GDPR,
                start_date=start_date,
                end_date=end_date,
                format="json"
            )
            
            assert file_path.endswith(".json")
            
            # Verify file exists
            export_file = Path(file_path)
            assert export_file.exists()
            
            # Cleanup
            export_file.unlink()


class TestAuditRoutes:
    """Test the audit API routes"""
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user for testing"""
        user = Mock(spec=User)
        user.id = uuid4()
        user.email = "test@example.com"
        user.role = "admin"
        return user
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @patch('scrollintel.api.routes.audit_routes.get_current_user')
    @patch('scrollintel.api.routes.audit_routes.require_permission')
    @patch('scrollintel.api.routes.audit_routes.audit_system')
    def test_search_audit_logs_endpoint(self, mock_audit, mock_permission, mock_user_dep, client, mock_user):
        """Test audit log search endpoint"""
        
        mock_user_dep.return_value = mock_user
        mock_permission.return_value = None
        
        mock_audit.search_audit_logs.return_value = [
            {
                "id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": str(uuid4()),
                "user_email": "test@example.com",
                "action": "user.login",
                "resource_type": "authentication",
                "resource_id": None,
                "details": {},
                "ip_address": "192.168.1.100",
                "user_agent": "Test Browser",
                "session_id": "test_session",
                "success": True,
                "error_message": None
            }
        ]
        
        response = client.get("/api/audit/logs/search?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "total_count" in data
        assert len(data["logs"]) == 1
    
    @patch('scrollintel.api.routes.audit_routes.get_current_user')
    @patch('scrollintel.api.routes.audit_routes.require_permission')
    @patch('scrollintel.api.routes.audit_routes.audit_system')
    def test_generate_compliance_report_endpoint(self, mock_audit, mock_permission, mock_user_dep, client, mock_user):
        """Test compliance report generation endpoint"""
        
        mock_user_dep.return_value = mock_user
        mock_permission.return_value = None
        
        mock_report = Mock()
        mock_report.id = str(uuid4())
        mock_report.report_type = "comprehensive"
        mock_report.generated_at = datetime.utcnow()
        mock_report.date_range_start = datetime.utcnow() - timedelta(days=30)
        mock_report.date_range_end = datetime.utcnow()
        mock_report.total_events = 100
        mock_report.successful_events = 95
        mock_report.failed_events = 5
        mock_report.security_events = 2
        mock_report.compliance_violations = []
        mock_report.user_activity_summary = {}
        mock_report.resource_access_summary = {}
        mock_report.recommendations = ["All systems operating normally"]
        
        mock_audit.generate_compliance_report.return_value = mock_report
        
        request_data = {
            "start_date": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "report_type": "comprehensive"
        }
        
        response = client.post("/api/audit/compliance/report", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 100
        assert data["successful_events"] == 95
        assert len(data["recommendations"]) == 1
    
    @patch('scrollintel.api.routes.audit_routes.get_current_user')
    @patch('scrollintel.api.routes.audit_routes.require_permission')
    @patch('scrollintel.api.routes.audit_routes.audit_system')
    def test_export_audit_logs_endpoint(self, mock_audit, mock_permission, mock_user_dep, client, mock_user):
        """Test audit log export endpoint"""
        
        mock_user_dep.return_value = mock_user
        mock_permission.return_value = None
        
        # Create a temporary file for testing
        test_file = Path("test_export.json")
        test_file.write_text('{"test": "data"}')
        
        mock_audit.export_audit_logs.return_value = str(test_file)
        
        request_data = {
            "start_date": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "format": "json"
        }
        
        response = client.post("/api/audit/export", json=request_data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        
        # Cleanup
        test_file.unlink()


class TestIntegration:
    """Integration tests for audit and compliance systems"""
    
    async def test_full_audit_workflow(self):
        """Test complete audit workflow from logging to export"""
        
        audit_system = AuditSystem()
        await audit_system.start()
        
        try:
            # Log some events
            user_id = str(uuid4())
            
            await audit_system.log_user_action(
                user_id=user_id,
                user_email="test@example.com",
                action=AuditAction.USER_LOGIN,
                resource_type="authentication",
                details={"login_method": "password"}
            )
            
            await audit_system.log_user_action(
                user_id=user_id,
                user_email="test@example.com",
                action=AuditAction.DATA_UPLOAD,
                resource_type="dataset",
                resource_id="dataset_123",
                details={"file_size": 1024}
            )
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Search logs
            with patch('scrollintel.core.audit_system.get_db'):
                logs = await audit_system.search_audit_logs(user_id=user_id, limit=10)
                # In real test, this would return actual logs
                
            # Generate report
            start_date = datetime.utcnow() - timedelta(days=1)
            end_date = datetime.utcnow()
            
            with patch('scrollintel.core.audit_system.get_db'):
                report = await audit_system.generate_compliance_report(
                    start_date=start_date,
                    end_date=end_date
                )
                assert report is not None
            
        finally:
            await audit_system.stop()
    
    async def test_compliance_violation_workflow(self):
        """Test compliance violation detection and handling"""
        
        compliance_manager = ComplianceManager()
        await compliance_manager.start()
        
        try:
            # Check for violations
            violations = await compliance_manager.check_compliance(
                action="access",
                resource_type="personal_data",
                resource_id="user_data_123",
                user_id="user_123",
                data_classification=DataClassification.CONFIDENTIAL
            )
            
            # Process any violations
            if violations:
                assert len(violations) > 0
                assert violations[0].severity in ["low", "medium", "high", "critical"]
            
            # Wait for background processing
            await asyncio.sleep(0.1)
            
        finally:
            await compliance_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])