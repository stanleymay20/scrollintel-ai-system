"""
Tests for Compliance and Audit Framework
Tests immutable audit logging, compliance reporting, evidence generation,
violation detection, and data privacy controls
"""

import pytest
import tempfile
import os
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path

from security.compliance.immutable_audit_logger import ImmutableAuditLogger, AuditEvent
from security.compliance.compliance_reporting import ComplianceReportingEngine, ComplianceFramework, ControlStatus
from security.compliance.evidence_generator import EvidenceGenerator
from security.compliance.violation_detector import ComplianceViolationDetector, ViolationSeverity, ViolationStatus
from security.compliance.data_privacy_controls import DataPrivacyControls, RequestType, DataCategory


class TestImmutableAuditLogger:
    """Test immutable audit logging with blockchain verification"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_audit.db")
        self.logger = ImmutableAuditLogger(self.db_path)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_audit_event(self):
        """Test logging audit events"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().timestamp(),
            event_type="login",
            user_id="test_user",
            resource="web_app",
            action="authenticate",
            outcome="success",
            details={"method": "password"},
            source_ip="192.168.1.1",
            user_agent="Test-Agent",
            session_id=str(uuid.uuid4())
        )
        
        event_id = self.logger.log_event(event)
        assert event_id == event.event_id
    
    def test_blockchain_integrity(self):
        """Test blockchain integrity verification"""
        # Log multiple events
        for i in range(5):
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().timestamp(),
                event_type="test_event",
                user_id=f"user_{i}",
                resource="test_resource",
                action="test_action",
                outcome="success",
                details={"test": True},
                source_ip="127.0.0.1",
                user_agent="Test-Agent",
                session_id=str(uuid.uuid4())
            )
            self.logger.log_event(event)
            time.sleep(0.01)  # Small delay for different timestamps
        
        # Force block creation
        self.logger.force_block_creation()
        
        # Verify integrity
        verification = self.logger.verify_integrity()
        assert verification['valid'] is True
        assert verification['total_events'] >= 5
        assert len(verification['verification_errors']) == 0
    
    def test_audit_trail_retrieval(self):
        """Test audit trail retrieval with filters"""
        # Log test events
        test_events = []
        for i in range(3):
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().timestamp(),
                event_type="data_access",
                user_id=f"user_{i}",
                resource="database",
                action="select",
                outcome="success",
                details={"table": "users"},
                source_ip="192.168.1.1",
                user_agent="Test-Agent",
                session_id=str(uuid.uuid4())
            )
            test_events.append(event)
            self.logger.log_event(event)
            time.sleep(0.01)
        
        # Retrieve trail
        trail = self.logger.get_audit_trail(
            event_type="data_access",
            limit=10
        )
        
        assert len(trail) >= 3
        for event in trail:
            assert event['event_type'] == 'data_access'
    
    def test_blockchain_stats(self):
        """Test blockchain statistics"""
        stats = self.logger.get_blockchain_stats()
        
        assert 'total_blocks' in stats
        assert 'total_events' in stats
        assert 'pending_events' in stats
        assert isinstance(stats['total_blocks'], int)
        assert isinstance(stats['total_events'], int)


class TestComplianceReporting:
    """Test compliance reporting engine"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_compliance.db")
        self.engine = ComplianceReportingEngine(self.db_path)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_compliance_report(self):
        """Test compliance report generation"""
        report = self.engine.generate_compliance_report(
            framework=ComplianceFramework.SOC2_TYPE_II,
            assessor="Test User"
        )
        
        assert report.framework == ComplianceFramework.SOC2_TYPE_II
        assert report.assessor == "Test User"
        assert report.total_controls > 0
        assert report.compliance_score >= 0
        assert len(report.controls) > 0
        assert isinstance(report.recommendations, list)
    
    def test_update_control_status(self):
        """Test updating control status"""
        control_id = "SOC2-CC1.1"
        evidence = ["test_evidence.pdf"]
        notes = "Test update"
        
        self.engine.update_control_status(
            control_id=control_id,
            status=ControlStatus.COMPLIANT,
            evidence=evidence,
            notes=notes
        )
        
        # Verify update by generating report
        report = self.engine.generate_compliance_report(ComplianceFramework.SOC2_TYPE_II)
        updated_control = next((c for c in report.controls if c.control_id == control_id), None)
        
        assert updated_control is not None
        assert updated_control.status == ControlStatus.COMPLIANT
        assert updated_control.evidence == evidence
    
    def test_compliance_dashboard(self):
        """Test compliance dashboard data"""
        dashboard = self.engine.get_compliance_dashboard()
        
        assert 'frameworks' in dashboard
        assert 'overall_metrics' in dashboard
        assert 'total_controls' in dashboard['overall_metrics']
        assert 'compliant_controls' in dashboard['overall_metrics']
        assert 'average_compliance_score' in dashboard['overall_metrics']


class TestEvidenceGenerator:
    """Test evidence generation system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_compliance.db")
        self.evidence_storage = os.path.join(self.temp_dir, "evidence")
        self.generator = EvidenceGenerator(self.db_path, self.evidence_storage)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_evidence_package(self):
        """Test evidence package generation"""
        package = self.generator.generate_evidence_package(
            framework="SOC2_TYPE_II",
            control_ids=["SOC2-CC1.1", "SOC2-CC2.1"],
            period_days=30
        )
        
        assert package.framework == "SOC2_TYPE_II"
        assert len(package.evidence_items) > 0
        assert os.path.exists(package.package_path)
        assert package.package_hash is not None
        assert len(package.package_hash) == 64  # SHA-256 hash length
    
    def test_evidence_collection(self):
        """Test evidence collection for specific controls"""
        # This would test the evidence collection methods
        # For now, we'll test that the methods exist and can be called
        evidence_items = self.generator._collect_evidence_for_control(
            "SOC2-CC1.1",
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        )
        
        assert isinstance(evidence_items, list)
        # Evidence items might be empty in test environment, but should be a list


class TestViolationDetector:
    """Test violation detection and remediation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_compliance.db")
        self.detector = ComplianceViolationDetector(self.db_path)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if hasattr(self.detector, 'stop_monitoring'):
            self.detector.stop_monitoring()
    
    def test_violation_detection_rules(self):
        """Test violation detection rules initialization"""
        assert hasattr(self.detector, 'detection_rules')
        assert len(self.detector.detection_rules) > 0
        assert 'failed_login_threshold' in self.detector.detection_rules
        assert 'privileged_access_without_mfa' in self.detector.detection_rules
    
    def test_remediation_workflows(self):
        """Test remediation workflow initialization"""
        workflow = self.detector._get_remediation_workflow(
            'failed_login_threshold',
            ViolationSeverity.HIGH
        )
        
        # Workflow might not exist in test environment, but method should work
        assert workflow is None or hasattr(workflow, 'automated_steps')
    
    def test_violations_dashboard(self):
        """Test violations dashboard"""
        dashboard = self.detector.get_violations_dashboard()
        
        assert 'metrics' in dashboard
        assert 'by_severity' in dashboard
        assert 'recent_violations' in dashboard
        assert 'total_violations' in dashboard['metrics']


class TestDataPrivacyControls:
    """Test data privacy controls and request handling"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_privacy.db")
        self.controls = DataPrivacyControls(self.db_path)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if hasattr(self.controls, 'stop_processing'):
            self.controls.stop_processing()
    
    def test_submit_data_subject_request(self):
        """Test submitting data subject requests"""
        request_id = self.controls.submit_data_subject_request(
            request_type=RequestType.ACCESS,
            data_subject_email="test@example.com",
            data_subject_name="Test User",
            requested_categories=[DataCategory.PERSONAL_IDENTIFIABLE],
            reason="Test request"
        )
        
        assert request_id is not None
        assert isinstance(request_id, str)
    
    def test_record_consent(self):
        """Test recording consent"""
        consent_id = self.controls.record_consent(
            data_subject_email="test@example.com",
            purpose="Service provision",
            legal_basis="Contract",
            consent_given=True,
            consent_method="web_form",
            data_categories=[DataCategory.PERSONAL_IDENTIFIABLE]
        )
        
        assert consent_id is not None
        assert isinstance(consent_id, str)
    
    def test_withdraw_consent(self):
        """Test withdrawing consent"""
        # First record consent
        consent_id = self.controls.record_consent(
            data_subject_email="test@example.com",
            purpose="Marketing",
            legal_basis="Consent",
            consent_given=True,
            consent_method="email_opt_in",
            data_categories=[DataCategory.PERSONAL_IDENTIFIABLE]
        )
        
        # Then withdraw it
        success = self.controls.withdraw_consent(consent_id, "user_request")
        assert success is True
        
        # Try to withdraw again (should fail)
        success = self.controls.withdraw_consent(consent_id, "user_request")
        assert success is False
    
    def test_privacy_dashboard(self):
        """Test privacy dashboard"""
        dashboard = self.controls.get_privacy_dashboard()
        
        assert 'metrics' in dashboard
        assert 'by_type' in dashboard
        assert 'recent_requests' in dashboard
        assert 'total_requests' in dashboard['metrics']


class TestIntegration:
    """Test integration between compliance components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize all components with same base directory
        self.audit_logger = ImmutableAuditLogger(
            os.path.join(self.temp_dir, "audit.db")
        )
        self.compliance_engine = ComplianceReportingEngine(
            os.path.join(self.temp_dir, "compliance.db")
        )
        self.privacy_controls = DataPrivacyControls(
            os.path.join(self.temp_dir, "privacy.db")
        )
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_audit_privacy_integration(self):
        """Test integration between audit logging and privacy controls"""
        # Submit a privacy request
        request_id = self.privacy_controls.submit_data_subject_request(
            request_type=RequestType.ACCESS,
            data_subject_email="integration@example.com",
            data_subject_name="Integration Test"
        )
        
        # Log an audit event for the privacy request
        audit_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().timestamp(),
            event_type="privacy_request_submitted",
            user_id="integration@example.com",
            resource="privacy_system",
            action="submit_request",
            outcome="success",
            details={"request_id": request_id, "request_type": "access"},
            source_ip="192.168.1.1",
            user_agent="Test-Agent",
            session_id=str(uuid.uuid4())
        )
        
        event_id = self.audit_logger.log_event(audit_event)
        
        # Verify both systems have the related data
        assert request_id is not None
        assert event_id is not None
        
        # Retrieve audit trail to verify the event was logged
        trail = self.audit_logger.get_audit_trail(
            event_type="privacy_request_submitted",
            limit=10
        )
        
        privacy_events = [e for e in trail if e['event_type'] == 'privacy_request_submitted']
        assert len(privacy_events) > 0
    
    def test_compliance_audit_integration(self):
        """Test integration between compliance reporting and audit logging"""
        # Update a compliance control
        self.compliance_engine.update_control_status(
            control_id="SOC2-CC1.1",
            status=ControlStatus.COMPLIANT,
            evidence=["test_evidence.pdf"],
            notes="Integration test update"
        )
        
        # Log an audit event for the control update
        audit_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().timestamp(),
            event_type="compliance_control_updated",
            user_id="compliance_admin",
            resource="compliance_system",
            action="update_control_status",
            outcome="success",
            details={
                "control_id": "SOC2-CC1.1",
                "new_status": "compliant",
                "evidence_count": 1
            },
            source_ip="192.168.1.1",
            user_agent="Test-Agent",
            session_id=str(uuid.uuid4())
        )
        
        event_id = self.audit_logger.log_event(audit_event)
        
        # Verify the integration
        assert event_id is not None
        
        # Generate compliance report to verify the update
        report = self.compliance_engine.generate_compliance_report(
            ComplianceFramework.SOC2_TYPE_II
        )
        
        updated_control = next(
            (c for c in report.controls if c.control_id == "SOC2-CC1.1"),
            None
        )
        
        assert updated_control is not None
        assert updated_control.status == ControlStatus.COMPLIANT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])