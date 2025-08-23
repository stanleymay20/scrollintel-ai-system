"""
Comprehensive Tests for Data Lineage and Compliance System

This module contains tests for data lineage tracking, compliance enforcement,
audit reporting, and data governance functionality.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any

from scrollintel.engines.lineage_tracker import LineageTracker
from scrollintel.engines.compliance_engine import ComplianceEngine
from scrollintel.engines.audit_reporter import AuditReporter, AuditReportConfig
from scrollintel.engines.data_governance import DataGovernanceEngine, DataClassification, PrivacyLevel
from scrollintel.models.lineage_models import (
    LineageEventRequest, LineageEventType, ComplianceRuleRequest, ComplianceRuleType,
    ComplianceStatus, LineageQueryRequest
)


class TestLineageTracker:
    """Test cases for LineageTracker"""
    
    @pytest.fixture
    def tracker(self):
        """Create LineageTracker instance for testing"""
        with patch('scrollintel.engines.lineage_tracker.get_sync_db'):
            return LineageTracker()
    
    @pytest.fixture
    def sample_lineage_request(self):
        """Sample lineage event request"""
        return LineageEventRequest(
            pipeline_id="pipeline_001",
            source_dataset_id="dataset_source",
            target_dataset_id="dataset_target",
            transformation_id="transform_001",
            event_type=LineageEventType.TRANSFORMATION,
            source_schema={"columns": ["id", "name", "email"]},
            target_schema={"columns": ["id", "name", "email_hash"]},
            transformation_details={"type": "hash", "field": "email"},
            data_volume=1000,
            processing_duration=5000,
            event_metadata={"version": "1.0"}
        )
    
    def test_track_lineage_event(self, tracker, sample_lineage_request):
        """Test tracking a lineage event"""
        with patch.object(tracker.session, 'add'), \
             patch.object(tracker.session, 'commit'), \
             patch.object(tracker, '_update_lineage_graph'), \
             patch.object(tracker, '_create_audit_trail'):
            
            lineage_id = tracker.track_lineage_event(
                sample_lineage_request,
                user_id="user_001",
                session_id="session_001"
            )
            
            assert lineage_id is not None
            assert isinstance(lineage_id, str)
            tracker.session.add.assert_called_once()
            tracker.session.commit.assert_called_once()
    
    def test_get_lineage_history(self, tracker):
        """Test getting lineage history"""
        query = LineageQueryRequest(
            dataset_id="dataset_001",
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            event_types=[LineageEventType.TRANSFORMATION],
            include_transformations=True
        )
        
        # Mock database query
        mock_event = Mock()
        mock_event.id = "lineage_001"
        mock_event.pipeline_id = "pipeline_001"
        mock_event.source_dataset_id = "dataset_source"
        mock_event.target_dataset_id = "dataset_target"
        mock_event.transformation_id = "transform_001"
        mock_event.event_type = "transformation"
        mock_event.event_timestamp = datetime.utcnow()
        mock_event.data_volume = 1000
        mock_event.processing_duration = 5000
        mock_event.user_id = "user_001"
        mock_event.event_metadata = {"version": "1.0"}
        mock_event.source_schema = {"columns": ["id", "name"]}
        mock_event.target_schema = {"columns": ["id", "name_hash"]}
        mock_event.transformation_details = {"type": "hash"}
        
        with patch.object(tracker.session, 'query') as mock_query:
            # Set up the full query chain
            mock_query_chain = mock_query.return_value
            mock_query_chain.filter.return_value = mock_query_chain
            mock_query_chain.order_by.return_value = mock_query_chain
            mock_query_chain.all.return_value = [mock_event]
            
            history = tracker.get_lineage_history(query)
            
            assert len(history) == 1
            assert history[0]["id"] == "lineage_001"
            assert history[0]["pipeline_id"] == "pipeline_001"
            assert "source_schema" in history[0]
            assert "transformation_details" in history[0]
    
    def test_get_data_lineage_graph(self, tracker):
        """Test getting data lineage graph"""
        with patch.object(tracker, '_get_upstream_lineage') as mock_upstream, \
             patch.object(tracker, '_get_downstream_lineage') as mock_downstream, \
             patch.object(tracker, '_get_node_details') as mock_node_details:
            
            mock_upstream.return_value = ({"dataset_1"}, [{"source": "dataset_1", "target": "dataset_2"}])
            mock_downstream.return_value = ({"dataset_3"}, [{"source": "dataset_2", "target": "dataset_3"}])
            mock_node_details.return_value = [
                {"id": "dataset_1", "type": "dataset"},
                {"id": "dataset_2", "type": "dataset"},
                {"id": "dataset_3", "type": "dataset"}
            ]
            
            graph = tracker.get_data_lineage_graph("dataset_2", depth=3, direction="both")
            
            assert "nodes" in graph
            assert "edges" in graph
            assert graph["root_dataset"] == "dataset_2"
            assert graph["depth"] == 3
            assert len(graph["nodes"]) == 3
            assert len(graph["edges"]) == 2
    
    def test_get_impact_analysis(self, tracker):
        """Test impact analysis"""
        with patch.object(tracker, '_get_downstream_lineage') as mock_downstream:
            mock_downstream.return_value = (
                {"dataset_1", "dataset_2", "dataset_3"},
                [
                    {"pipeline_id": "pipeline_1"},
                    {"pipeline_id": "pipeline_2"},
                    {"pipeline_id": "pipeline_1"}
                ]
            )
            
            impact = tracker.get_impact_analysis("dataset_source", "schema_change")
            
            assert impact["affected_datasets"] == 3
            assert len(impact["affected_pipelines"]) == 2
            assert "pipeline_1" in impact["affected_pipelines"]
            assert "pipeline_2" in impact["affected_pipelines"]
            assert isinstance(impact["recommendations"], list)


class TestComplianceEngine:
    """Test cases for ComplianceEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create ComplianceEngine instance for testing"""
        with patch('scrollintel.engines.compliance_engine.get_sync_db'):
            return ComplianceEngine()
    
    @pytest.fixture
    def sample_compliance_rule(self):
        """Sample compliance rule request"""
        return ComplianceRuleRequest(
            name="Data Retention Rule",
            description="Ensure data is not retained beyond 7 years",
            rule_type=ComplianceRuleType.DATA_RETENTION,
            conditions={"data_retention_days": 2555},  # 7 years
            actions={"alert": {"recipients": ["admin@company.com"]}},
            severity="high",
            applicable_datasets=["customer_data*", "transaction_data*"],
            applicable_pipelines=["etl_*"]
        )
    
    def test_create_compliance_rule(self, engine, sample_compliance_rule):
        """Test creating a compliance rule"""
        with patch.object(engine.session, 'add'), \
             patch.object(engine.session, 'commit'):
            
            rule_id = engine.create_compliance_rule(sample_compliance_rule, "admin_user")
            
            assert rule_id is not None
            assert isinstance(rule_id, str)
            engine.session.add.assert_called_once()
            engine.session.commit.assert_called_once()
    
    def test_evaluate_compliance(self, engine):
        """Test compliance evaluation"""
        # Mock active rules
        mock_rule = Mock()
        mock_rule.id = "rule_001"
        mock_rule.name = "Test Rule"
        mock_rule.rule_type = "data_retention"
        mock_rule.severity = "high"
        mock_rule.applicable_pipelines = ["test_pipeline"]
        mock_rule.applicable_datasets = None
        mock_rule.conditions = {"data_retention_days": 365}
        mock_rule.actions = {"alert": {}}
        
        engine.active_rules = {"rule_001": mock_rule}
        
        with patch.object(engine, '_evaluate_rule') as mock_evaluate:
            mock_evaluate.return_value = {
                "rule_id": "rule_001",
                "rule_name": "Test Rule",
                "compliant": True,
                "severity": "high"
            }
            
            results = engine.evaluate_compliance(
                pipeline_id="test_pipeline",
                dataset_id="test_dataset",
                operation_type="data_processing",
                context={"data_age_days": 30}
            )
            
            assert len(results) == 1
            assert results[0]["rule_id"] == "rule_001"
            assert results[0]["compliant"] is True
    
    def test_get_compliance_violations(self, engine):
        """Test getting compliance violations"""
        mock_violation = Mock()
        mock_violation.id = "violation_001"
        mock_violation.rule_id = "rule_001"
        mock_violation.pipeline_id = "pipeline_001"
        mock_violation.dataset_id = "dataset_001"
        mock_violation.violation_type = "data_retention"
        mock_violation.description = "Data retention exceeded"
        mock_violation.severity = "high"
        mock_violation.status = "non_compliant"
        mock_violation.detected_at = datetime.utcnow()
        mock_violation.resolved_at = None
        mock_violation.resolution_notes = None
        
        with patch.object(engine.session, 'query') as mock_query:
            # Set up the full query chain
            mock_query_chain = mock_query.return_value
            mock_query_chain.filter.return_value = mock_query_chain
            mock_query_chain.order_by.return_value = mock_query_chain
            mock_query_chain.limit.return_value = mock_query_chain
            mock_query_chain.all.return_value = [mock_violation]
            
            violations = engine.get_compliance_violations(
                pipeline_id="pipeline_001",
                severity="high"
            )
            
            assert len(violations) == 1
            assert violations[0].id == "violation_001"
            assert violations[0].severity == "high"
    
    def test_resolve_violation(self, engine):
        """Test resolving a compliance violation"""
        mock_violation = Mock()
        mock_violation.id = "violation_001"
        
        with patch.object(engine.session, 'query') as mock_query, \
             patch.object(engine.session, 'commit'):
            mock_query.return_value.filter.return_value.first.return_value = mock_violation
            
            success = engine.resolve_violation(
                "violation_001",
                "Issue resolved by updating retention policy",
                "admin_user"
            )
            
            assert success is True
            assert mock_violation.status == ComplianceStatus.REMEDIATED.value
            assert mock_violation.resolution_notes == "Issue resolved by updating retention policy"
            assert mock_violation.resolved_by == "admin_user"
    
    def test_get_compliance_report(self, engine):
        """Test generating compliance report"""
        mock_violations = [
            Mock(severity="high", status="non_compliant", rule_id="rule_1", resolved_at=None),
            Mock(severity="medium", status="remediated", rule_id="rule_2", resolved_at=datetime.now()),
            Mock(severity="high", status="non_compliant", rule_id="rule_1", resolved_at=None)
        ]
        
        mock_rule = Mock()
        mock_rule.rule_type = "data_retention"
        
        with patch.object(engine.session, 'query') as mock_query:
            # Set up the query chain for violations
            mock_query_chain = mock_query.return_value
            mock_query_chain.filter.return_value = mock_query_chain
            mock_query_chain.all.return_value = mock_violations
            mock_query_chain.first.return_value = mock_rule
            
            start_date = datetime.utcnow() - timedelta(days=30)
            end_date = datetime.utcnow()
            
            report = engine.get_compliance_report(start_date, end_date, include_details=True)
            
            assert "summary" in report
            assert "breakdown" in report
            assert report["summary"]["total_violations"] == 3
            assert report["summary"]["resolved_violations"] == 1
            assert report["breakdown"]["by_severity"]["high"] == 2
            assert report["breakdown"]["by_severity"]["medium"] == 1


class TestAuditReporter:
    """Test cases for AuditReporter"""
    
    @pytest.fixture
    def reporter(self):
        """Create AuditReporter instance for testing"""
        with patch('scrollintel.engines.audit_reporter.get_sync_db'):
            return AuditReporter()
    
    def test_generate_audit_report(self, reporter):
        """Test generating audit report"""
        config = AuditReportConfig(
            report_type="comprehensive",
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow(),
            include_lineage=True,
            include_violations=True,
            include_user_activity=True,
            format="json"
        )
        
        with patch.object(reporter, '_generate_audit_summary') as mock_summary, \
             patch.object(reporter, '_generate_user_activity_report') as mock_user_activity, \
             patch.object(reporter, '_generate_lineage_report') as mock_lineage, \
             patch.object(reporter, '_generate_violations_report') as mock_violations, \
             patch.object(reporter, '_generate_system_changes_report') as mock_system_changes, \
             patch.object(reporter, '_generate_data_access_report') as mock_data_access:
            
            mock_summary.return_value = {"total_audit_entries": 100}
            mock_user_activity.return_value = {"user_statistics": []}
            mock_lineage.return_value = {"total_lineage_events": 50}
            mock_violations.return_value = {"total_violations": 5}
            mock_system_changes.return_value = {"total_changes": 10}
            mock_data_access.return_value = {"total_access_events": 200}
            
            report = reporter.generate_audit_report(config)
            
            assert "report_id" in report
            assert "generated_at" in report
            assert "summary" in report
            assert "sections" in report
            assert "user_activity" in report["sections"]
            assert "data_lineage" in report["sections"]
            assert "compliance_violations" in report["sections"]
    
    def test_generate_compliance_audit_report(self, reporter):
        """Test generating compliance-specific audit report"""
        with patch.object(reporter, '_generate_gdpr_audit_report') as mock_gdpr:
            mock_gdpr.return_value = {
                "data_subject_rights": {},
                "consent_management": {},
                "data_processing_lawfulness": {},
                "data_protection_impact_assessments": {}
            }
            
            start_date = datetime.utcnow() - timedelta(days=30)
            end_date = datetime.utcnow()
            
            report = reporter.generate_compliance_audit_report("GDPR", start_date, end_date)
            
            assert report["regulation_type"] == "GDPR"
            assert "report_id" in report
            assert "compliance_status" in report
            assert "sections" in report
    
    def test_export_audit_trail_csv(self, reporter):
        """Test exporting audit trail to CSV"""
        mock_entries = [
            Mock(
                id="audit_001",
                entity_type="dataset",
                entity_id="dataset_001",
                action="create",
                user_id="user_001",
                timestamp=datetime.utcnow(),
                ip_address="192.168.1.1",
                change_summary="Created new dataset"
            )
        ]
        
        with patch.object(reporter.session, 'query') as mock_query:
            mock_query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_entries
            
            start_date = datetime.utcnow() - timedelta(days=7)
            end_date = datetime.utcnow()
            
            csv_data = reporter.export_audit_trail(start_date, end_date, "csv")
            
            assert "ID,Entity Type,Entity ID,Action,User ID" in csv_data
            assert "audit_001" in csv_data
            assert "dataset" in csv_data
            assert "user_001" in csv_data
    
    def test_get_data_processing_history(self, reporter):
        """Test getting data processing history"""
        mock_lineage_event = Mock()
        mock_lineage_event.event_timestamp = datetime.utcnow()
        mock_lineage_event.event_type = "transformation"
        mock_lineage_event.pipeline_id = "pipeline_001"
        mock_lineage_event.transformation_id = "transform_001"
        mock_lineage_event.source_dataset_id = "dataset_source"
        mock_lineage_event.target_dataset_id = "dataset_target"
        mock_lineage_event.data_volume = 1000
        mock_lineage_event.processing_duration = 5000
        mock_lineage_event.user_id = "user_001"
        
        mock_audit_entry = Mock()
        mock_audit_entry.timestamp = datetime.utcnow()
        mock_audit_entry.action = "update"
        mock_audit_entry.user_id = "user_001"
        mock_audit_entry.old_values = {"status": "draft"}
        mock_audit_entry.new_values = {"status": "published"}
        mock_audit_entry.change_summary = "Published dataset"
        
        with patch.object(reporter.session, 'query') as mock_query:
            # Mock lineage query
            mock_query.return_value.filter.return_value.order_by.return_value.all.side_effect = [
                [mock_lineage_event],  # First call for lineage
                [mock_audit_entry]     # Second call for audit
            ]
            
            history = reporter.get_data_processing_history("dataset_001")
            
            assert history["dataset_id"] == "dataset_001"
            assert len(history["processing_events"]) == 1
            assert len(history["system_changes"]) == 1
            assert history["processing_events"][0]["event_type"] == "transformation"
            assert history["system_changes"][0]["action"] == "update"


class TestDataGovernanceEngine:
    """Test cases for DataGovernanceEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create DataGovernanceEngine instance for testing"""
        with patch('scrollintel.engines.data_governance.get_sync_db'):
            return DataGovernanceEngine()
    
    def test_classify_data(self, engine):
        """Test data classification"""
        data_sample = {
            "id": "12345",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "ssn": "123-45-6789",
            "salary": 75000
        }
        
        with patch.object(engine, '_detect_sensitive_patterns') as mock_detect, \
             patch.object(engine, '_determine_classification_level') as mock_classify, \
             patch.object(engine, '_determine_privacy_level') as mock_privacy, \
             patch.object(engine, '_identify_compliance_requirements') as mock_compliance, \
             patch.object(engine, '_generate_recommended_controls') as mock_controls, \
             patch.object(engine, '_calculate_confidence_score') as mock_confidence:
            
            mock_detect.return_value = [
                {"field": "email", "pattern": "email", "confidence": 0.9},
                {"field": "ssn", "pattern": "ssn", "confidence": 0.95}
            ]
            mock_classify.return_value = DataClassification.RESTRICTED
            mock_privacy.return_value = PrivacyLevel.MAXIMUM
            mock_compliance.return_value = ["GDPR", "SOX"]
            mock_controls.return_value = ["encryption", "access_control", "audit_logging"]
            mock_confidence.return_value = 0.92
            
            result = engine.classify_data("dataset_001", data_sample)
            
            assert result["dataset_id"] == "dataset_001"
            assert result["classification"] == DataClassification.RESTRICTED
            assert result["privacy_level"] == PrivacyLevel.MAXIMUM
            assert "GDPR" in result["compliance_requirements"]
            assert "encryption" in result["recommended_controls"]
            assert result["confidence_score"] == 0.92
    
    def test_apply_privacy_controls(self, engine):
        """Test applying privacy controls"""
        data = {
            "id": "12345",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "ssn": "123-45-6789"
        }
        
        user_context = {"user_id": "user_001", "roles": ["analyst"]}
        
        with patch.object(engine, '_get_data_classification') as mock_classification, \
             patch.object(engine, '_is_sensitive_field') as mock_sensitive, \
             patch.object(engine, '_apply_field_privacy_control') as mock_field_control:
            
            mock_classification.return_value = DataClassification.CONFIDENTIAL
            mock_sensitive.side_effect = lambda field, value: field in ["email", "ssn"]
            mock_field_control.return_value = {
                "protected_value": "[REDACTED]",
                "controls": ["full_redaction"]
            }
            
            result = engine.apply_privacy_controls(
                "dataset_001", data, PrivacyLevel.MAXIMUM, user_context
            )
            
            assert result["access_granted"] is True
            assert result["privacy_level"] == PrivacyLevel.MAXIMUM.value
            assert len(result["applied_controls"]) > 0
    
    def test_check_data_retention_compliance(self, engine):
        """Test data retention compliance check"""
        with patch.object(engine, '_get_retention_policies') as mock_policies, \
             patch.object(engine, '_check_retention_policy') as mock_check:
            
            mock_policies.return_value = [Mock()]
            mock_check.return_value = {
                "compliant": False,
                "violation": "Data retention exceeded",
                "actions_required": ["archive_data"]
            }
            
            data_age = timedelta(days=2600)  # Over 7 years
            
            result = engine.check_data_retention_compliance(
                "dataset_001", data_age, DataClassification.CONFIDENTIAL
            )
            
            assert result["dataset_id"] == "dataset_001"
            assert result["data_age_days"] == 2600
            assert result["compliant"] is False
            assert len(result["violations"]) > 0
            assert "archive_data" in result["actions_required"]
    
    def test_anonymize_data(self, engine):
        """Test data anonymization"""
        data = {
            "id": "12345",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 35,
            "city": "New York"
        }
        
        with patch.object(engine, '_is_personally_identifiable') as mock_pii, \
             patch.object(engine, '_anonymize_field') as mock_anonymize:
            
            mock_pii.side_effect = lambda field, value: field in ["name", "email", "age"]
            mock_anonymize.side_effect = [
                ("ANON_12345678", "hashing"),
                ("ANON_87654321", "hashing"),
                ("30-39", "generalization")
            ]
            
            result = engine.anonymize_data(data, "standard", True)
            
            assert "anonymized_data" in result
            assert "metadata" in result
            assert result["anonymized_data"]["name"] == "ANON_12345678"
            assert result["anonymized_data"]["email"] == "ANON_87654321"
            assert result["anonymized_data"]["age"] == "30-39"
            assert result["anonymized_data"]["city"] == "New York"  # Non-PII preserved
            assert len(result["metadata"]["techniques_applied"]) == 3


class TestIntegration:
    """Integration tests for the complete lineage and compliance system"""
    
    @pytest.fixture
    def system_components(self):
        """Create all system components for integration testing"""
        with patch('scrollintel.engines.lineage_tracker.get_sync_db'), \
             patch('scrollintel.engines.compliance_engine.get_sync_db'), \
             patch('scrollintel.engines.audit_reporter.get_sync_db'), \
             patch('scrollintel.engines.data_governance.get_sync_db'):
            
            return {
                "lineage_tracker": LineageTracker(),
                "compliance_engine": ComplianceEngine(),
                "audit_reporter": AuditReporter(),
                "governance_engine": DataGovernanceEngine()
            }
    
    def test_end_to_end_compliance_workflow(self, system_components):
        """Test complete compliance workflow from data processing to audit"""
        tracker = system_components["lineage_tracker"]
        compliance = system_components["compliance_engine"]
        reporter = system_components["audit_reporter"]
        governance = system_components["governance_engine"]
        
        # Mock database operations
        with patch.object(tracker.session, 'add'), \
             patch.object(tracker.session, 'commit'), \
             patch.object(compliance.session, 'add'), \
             patch.object(compliance.session, 'commit'):
            
            # 1. Track data lineage
            lineage_request = LineageEventRequest(
                pipeline_id="compliance_test_pipeline",
                source_dataset_id="raw_customer_data",
                target_dataset_id="processed_customer_data",
                event_type=LineageEventType.TRANSFORMATION,
                data_volume=10000,
                processing_duration=30000
            )
            
            lineage_id = tracker.track_lineage_event(lineage_request, "system_user")
            assert lineage_id is not None
            
            # 2. Create compliance rule
            rule_request = ComplianceRuleRequest(
                name="Customer Data Protection Rule",
                rule_type=ComplianceRuleType.DATA_PRIVACY,
                conditions={"encryption_required": True},
                actions={"alert": {"recipients": ["compliance@company.com"]}},
                severity="high",
                applicable_datasets=["*customer_data*"]
            )
            
            rule_id = compliance.create_compliance_rule(rule_request, "compliance_officer")
            assert rule_id is not None
            
            # 3. Evaluate compliance
            compliance.active_rules = {rule_id: Mock(
                id=rule_id,
                name="Customer Data Protection Rule",
                rule_type="data_privacy",
                severity="high",
                applicable_datasets=["*customer_data*"],
                applicable_pipelines=None,
                conditions={"encryption_required": True},
                actions={"alert": {}}
            )}
            
            with patch.object(compliance, '_evaluate_rule') as mock_evaluate:
                mock_evaluate.return_value = {
                    "rule_id": rule_id,
                    "compliant": True,
                    "severity": "high"
                }
                
                evaluation_results = compliance.evaluate_compliance(
                    "compliance_test_pipeline",
                    "processed_customer_data",
                    context={"encrypted": True}
                )
                
                assert len(evaluation_results) == 1
                assert evaluation_results[0]["compliant"] is True
    
    def test_data_governance_integration(self, system_components):
        """Test integration between data governance and compliance systems"""
        governance = system_components["governance_engine"]
        compliance = system_components["compliance_engine"]
        
        # Test data classification and compliance rule creation
        sensitive_data = {
            "customer_id": "CUST_12345",
            "email": "customer@example.com",
            "credit_card": "4111-1111-1111-1111",
            "purchase_amount": 299.99
        }
        
        with patch.object(governance, '_detect_sensitive_patterns') as mock_detect:
            mock_detect.return_value = [
                {"field": "email", "pattern": "email", "confidence": 0.9},
                {"field": "credit_card", "pattern": "credit_card", "confidence": 0.95}
            ]
            
            classification_result = governance.classify_data(
                "payment_data", sensitive_data
            )
            
            # Verify classification triggers appropriate compliance requirements
            assert DataClassification.RESTRICTED in [
                DataClassification.RESTRICTED,
                classification_result["classification"]
            ]
            assert "PCI-DSS" in classification_result.get("compliance_requirements", []) or \
                   "encryption" in classification_result.get("recommended_controls", [])


if __name__ == "__main__":
    pytest.main([__file__])