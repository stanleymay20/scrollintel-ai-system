"""
Integration tests for data normalization and quality system

Tests the complete data quality workflow including normalization,
quality monitoring, reconciliation, lineage tracking, and alerting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from scrollintel.core.data_normalizer import (
    DataNormalizer, DataSchema, SchemaField, SchemaMapping, 
    DataType, TransformationType
)
from scrollintel.core.data_quality_monitor import (
    DataQualityMonitor, QualityRule, QualityRuleType, QualitySeverity
)
from scrollintel.core.data_reconciliation import (
    DataReconciliationEngine, DataSource, ReconciliationRule, 
    ConflictResolutionStrategy
)
from scrollintel.core.data_lineage import (
    DataLineageTracker, DataAsset, LineageEventType, DataClassification
)
from scrollintel.core.data_quality_alerting import (
    DataQualityAlertingSystem, AlertRule, AlertChannel, AlertFrequency,
    ReportSchedule, ReportFormat
)


class TestDataNormalizationIntegration:
    """Test data normalization functionality"""
    
    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()
    
    @pytest.fixture
    def sample_schemas(self):
        # Source schema (ERP system)
        source_schema = DataSchema(
            name="erp_customers",
            version="1.0",
            fields=[
                SchemaField("customer_id", DataType.STRING),
                SchemaField("customer_name", DataType.STRING),
                SchemaField("email_address", DataType.STRING),
                SchemaField("phone_number", DataType.STRING),
                SchemaField("registration_date", DataType.DATETIME)
            ]
        )
        
        # Target schema (normalized)
        target_schema = DataSchema(
            name="unified_customers",
            version="1.0",
            fields=[
                SchemaField("id", DataType.STRING),
                SchemaField("name", DataType.STRING),
                SchemaField("email", DataType.STRING),
                SchemaField("phone", DataType.STRING),
                SchemaField("created_at", DataType.DATETIME),
                SchemaField("full_contact", DataType.STRING)
            ]
        )
        
        return source_schema, target_schema
    
    @pytest.fixture
    def sample_mappings(self):
        return [
            SchemaMapping("customer_id", "id", TransformationType.DIRECT_MAPPING),
            SchemaMapping("customer_name", "name", TransformationType.DIRECT_MAPPING),
            SchemaMapping("email_address", "email", TransformationType.DIRECT_MAPPING),
            SchemaMapping("phone_number", "phone", TransformationType.DIRECT_MAPPING),
            SchemaMapping("registration_date", "created_at", TransformationType.DIRECT_MAPPING),
            SchemaMapping("customer_name", "full_contact", TransformationType.CONCATENATION, {
                "source_fields": ["customer_name", "email_address"],
                "separator": " - "
            })
        ]
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame([
            {
                "customer_id": "CUST001",
                "customer_name": "John Doe",
                "email_address": "john@example.com",
                "phone_number": "+1234567890",
                "registration_date": "2023-01-15"
            },
            {
                "customer_id": "CUST002",
                "customer_name": "Jane Smith",
                "email_address": "jane@example.com",
                "phone_number": "+1987654321",
                "registration_date": "2023-02-20"
            }
        ])
    
    def test_schema_registration(self, normalizer, sample_schemas):
        """Test schema registration"""
        source_schema, target_schema = sample_schemas
        
        assert normalizer.register_schema(source_schema)
        assert normalizer.register_schema(target_schema)
        
        assert "erp_customers" in normalizer.schemas
        assert "unified_customers" in normalizer.schemas
    
    def test_mapping_registration(self, normalizer, sample_schemas, sample_mappings):
        """Test mapping registration"""
        source_schema, target_schema = sample_schemas
        
        normalizer.register_schema(source_schema)
        normalizer.register_schema(target_schema)
        
        assert normalizer.register_mapping("erp_customers", "unified_customers", sample_mappings)
        
        mapping_key = "erp_customers->unified_customers"
        assert mapping_key in normalizer.mappings
        assert len(normalizer.mappings[mapping_key]) == 6
    
    def test_data_normalization(self, normalizer, sample_schemas, sample_mappings, sample_data):
        """Test complete data normalization process"""
        source_schema, target_schema = sample_schemas
        
        # Setup
        normalizer.register_schema(source_schema)
        normalizer.register_schema(target_schema)
        normalizer.register_mapping("erp_customers", "unified_customers", sample_mappings)
        
        # Normalize data
        result = normalizer.normalize_data(sample_data, "erp_customers", "unified_customers")
        
        assert result.success
        assert len(result.normalized_data) == 2
        assert "id" in result.normalized_data.columns
        assert "full_contact" in result.normalized_data.columns
        
        # Check concatenation worked
        first_row = result.normalized_data.iloc[0]
        assert first_row["full_contact"] == "John Doe - john@example.com"
        
        # Check quality metrics
        assert "transformation_success_rate" in result.quality_metrics
        assert result.quality_metrics["transformation_success_rate"] == 1.0
    
    def test_schema_compatibility_validation(self, normalizer, sample_schemas, sample_mappings):
        """Test schema compatibility validation"""
        source_schema, target_schema = sample_schemas
        
        normalizer.register_schema(source_schema)
        normalizer.register_schema(target_schema)
        normalizer.register_mapping("erp_customers", "unified_customers", sample_mappings)
        
        compatibility = normalizer.validate_schema_compatibility("erp_customers", "unified_customers")
        
        assert compatibility["compatible"]
        assert compatibility["mapping_coverage"] == 1.0
        assert len(compatibility["errors"]) == 0


class TestDataQualityMonitoringIntegration:
    """Test data quality monitoring functionality"""
    
    @pytest.fixture
    def quality_monitor(self):
        return DataQualityMonitor()
    
    @pytest.fixture
    def sample_quality_rules(self):
        return [
            QualityRule(
                id="completeness_rule",
                name="Email Completeness",
                description="Email field should be 95% complete",
                rule_type=QualityRuleType.COMPLETENESS,
                severity=QualitySeverity.HIGH,
                target_fields=["email"],
                parameters={"min_completeness": 0.95}
            ),
            QualityRule(
                id="uniqueness_rule",
                name="Customer ID Uniqueness",
                description="Customer IDs should be unique",
                rule_type=QualityRuleType.UNIQUENESS,
                severity=QualitySeverity.CRITICAL,
                target_fields=["customer_id"]
            ),
            QualityRule(
                id="format_rule",
                name="Email Format",
                description="Email should match valid format",
                rule_type=QualityRuleType.FORMAT,
                severity=QualitySeverity.MEDIUM,
                target_fields=["email"],
                parameters={"email_pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
            )
        ]
    
    @pytest.fixture
    def quality_test_data(self):
        return pd.DataFrame([
            {"customer_id": "CUST001", "email": "john@example.com", "age": 25},
            {"customer_id": "CUST002", "email": "jane@example.com", "age": 30},
            {"customer_id": "CUST003", "email": None, "age": 35},  # Missing email
            {"customer_id": "CUST001", "email": "duplicate@example.com", "age": 40},  # Duplicate ID
            {"customer_id": "CUST005", "email": "invalid-email", "age": 45}  # Invalid email format
        ])
    
    def test_quality_rule_registration(self, quality_monitor, sample_quality_rules):
        """Test quality rule registration"""
        for rule in sample_quality_rules:
            assert quality_monitor.register_quality_rule(rule)
        
        assert len(quality_monitor.rules) == 3
        assert "completeness_rule" in quality_monitor.rules
    
    def test_data_quality_assessment(self, quality_monitor, sample_quality_rules, quality_test_data):
        """Test comprehensive data quality assessment"""
        # Register rules
        for rule in sample_quality_rules:
            quality_monitor.register_quality_rule(rule)
        
        # Assess data quality
        report = quality_monitor.assess_data_quality(quality_test_data, "test_dataset")
        
        assert report.dataset_name == "test_dataset"
        assert report.total_records == 5
        assert len(report.issues) > 0
        
        # Check for expected issues
        issue_types = [issue.rule_name for issue in report.issues]
        assert "Email Completeness" in issue_types
        assert "Customer ID Uniqueness" in issue_types
        
        # Check dimension scores
        assert "completeness" in report.dimension_scores
        assert "uniqueness" in report.dimension_scores
        
        # Overall score should be less than 100 due to issues
        assert report.overall_score < 100
    
    def test_quality_trends_tracking(self, quality_monitor, sample_quality_rules, quality_test_data):
        """Test quality trends over time"""
        # Register rules
        for rule in sample_quality_rules:
            quality_monitor.register_quality_rule(rule)
        
        # Generate multiple reports
        for i in range(3):
            report = quality_monitor.assess_data_quality(quality_test_data, "trend_dataset")
            # Simulate time passing
            report.assessment_timestamp = datetime.utcnow() - timedelta(days=i)
        
        # Get trends
        trends = quality_monitor.get_quality_trends("trend_dataset", days=7)
        
        assert trends["dataset_name"] == "trend_dataset"
        assert trends["assessments_count"] == 3
        assert "score_trend" in trends
        assert "dimension_trends" in trends


class TestDataReconciliationIntegration:
    """Test data reconciliation functionality"""
    
    @pytest.fixture
    def reconciliation_engine(self):
        return DataReconciliationEngine()
    
    @pytest.fixture
    def sample_data_sources(self):
        return [
            DataSource(
                id="erp_system",
                name="ERP System",
                priority=3,
                reliability_score=0.9,
                last_updated=datetime.utcnow()
            ),
            DataSource(
                id="crm_system",
                name="CRM System",
                priority=2,
                reliability_score=0.8,
                last_updated=datetime.utcnow() - timedelta(hours=1)
            ),
            DataSource(
                id="web_analytics",
                name="Web Analytics",
                priority=1,
                reliability_score=0.7,
                last_updated=datetime.utcnow() - timedelta(hours=2)
            )
        ]
    
    @pytest.fixture
    def conflicting_datasets(self):
        # ERP data
        erp_data = pd.DataFrame([
            {"customer_id": "CUST001", "name": "John Doe", "email": "john.doe@company.com", "status": "active"},
            {"customer_id": "CUST002", "name": "Jane Smith", "email": "jane.smith@company.com", "status": "active"}
        ])
        
        # CRM data (some conflicts)
        crm_data = pd.DataFrame([
            {"customer_id": "CUST001", "name": "John D.", "email": "john.doe@company.com", "status": "premium"},
            {"customer_id": "CUST002", "name": "Jane Smith", "email": "j.smith@company.com", "status": "active"},
            {"customer_id": "CUST003", "name": "Bob Johnson", "email": "bob@company.com", "status": "inactive"}
        ])
        
        # Web Analytics data
        web_data = pd.DataFrame([
            {"customer_id": "CUST001", "name": "John Doe", "email": "john.doe@company.com", "status": "engaged"},
            {"customer_id": "CUST003", "name": "Robert Johnson", "email": "bob@company.com", "status": "visitor"}
        ])
        
        return {
            "erp_system": erp_data,
            "crm_system": crm_data,
            "web_analytics": web_data
        }
    
    @pytest.fixture
    def reconciliation_rules(self):
        return [
            ReconciliationRule(
                field_name="name",
                strategy=ConflictResolutionStrategy.HIGHEST_PRIORITY
            ),
            ReconciliationRule(
                field_name="email",
                strategy=ConflictResolutionStrategy.LATEST_TIMESTAMP
            ),
            ReconciliationRule(
                field_name="status",
                strategy=ConflictResolutionStrategy.CUSTOM_RULE,
                parameters={"custom_logic": "prefer_non_null"}
            )
        ]
    
    def test_data_source_registration(self, reconciliation_engine, sample_data_sources):
        """Test data source registration"""
        for source in sample_data_sources:
            assert reconciliation_engine.register_data_source(source)
        
        assert len(reconciliation_engine.data_sources) == 3
        assert "erp_system" in reconciliation_engine.data_sources
    
    def test_reconciliation_rule_registration(self, reconciliation_engine, reconciliation_rules):
        """Test reconciliation rule registration"""
        for rule in reconciliation_rules:
            assert reconciliation_engine.register_reconciliation_rule(rule)
        
        assert len(reconciliation_engine.reconciliation_rules) == 3
        assert "name" in reconciliation_engine.reconciliation_rules
    
    def test_data_reconciliation_process(self, reconciliation_engine, sample_data_sources, 
                                       conflicting_datasets, reconciliation_rules):
        """Test complete data reconciliation process"""
        # Setup
        for source in sample_data_sources:
            reconciliation_engine.register_data_source(source)
        
        for rule in reconciliation_rules:
            reconciliation_engine.register_reconciliation_rule(rule)
        
        # Reconcile data
        result = reconciliation_engine.reconcile_data(conflicting_datasets, "customer_id")
        
        assert result.success
        assert result.conflicts_detected > 0
        assert result.conflicts_resolved > 0
        assert len(result.reconciled_data) >= 3  # Should have all unique customers
        
        # Check that conflicts were resolved
        reconciled_customer_001 = result.reconciled_data[
            result.reconciled_data["customer_id"] == "CUST001"
        ].iloc[0]
        
        # Name should come from highest priority source (ERP)
        assert reconciled_customer_001["name"] == "John Doe"
        
        # Check quality metrics
        assert "source_count" in result.quality_metrics
        assert result.quality_metrics["source_count"] == 3
    
    def test_conflict_summary_analysis(self, reconciliation_engine, sample_data_sources, 
                                     conflicting_datasets, reconciliation_rules):
        """Test conflict summary and analysis"""
        # Setup and reconcile
        for source in sample_data_sources:
            reconciliation_engine.register_data_source(source)
        
        for rule in reconciliation_rules:
            reconciliation_engine.register_reconciliation_rule(rule)
        
        reconciliation_engine.reconcile_data(conflicting_datasets, "customer_id")
        
        # Get conflict summary
        summary = reconciliation_engine.get_conflict_summary(days=1)
        
        assert "total_conflicts" in summary
        assert "conflicts_by_field" in summary
        assert "strategy_usage" in summary
        assert summary["total_conflicts"] > 0


class TestDataLineageIntegration:
    """Test data lineage tracking functionality"""
    
    @pytest.fixture
    def lineage_tracker(self):
        return DataLineageTracker()
    
    @pytest.fixture
    def sample_assets(self):
        return [
            DataAsset(
                id="erp_customers",
                name="ERP Customer Table",
                type="table",
                source_system="SAP_ERP",
                schema_info={"fields": ["customer_id", "name", "email"]},
                classification=DataClassification.INTERNAL,
                owner="data_team"
            ),
            DataAsset(
                id="crm_contacts",
                name="CRM Contacts",
                type="table",
                source_system="Salesforce",
                schema_info={"fields": ["contact_id", "full_name", "email_address"]},
                classification=DataClassification.CONFIDENTIAL,
                owner="sales_team"
            ),
            DataAsset(
                id="unified_customers",
                name="Unified Customer View",
                type="view",
                source_system="Analytics_Platform",
                schema_info={"fields": ["id", "name", "email", "source"]},
                classification=DataClassification.INTERNAL,
                owner="analytics_team"
            )
        ]
    
    def test_asset_registration(self, lineage_tracker, sample_assets):
        """Test data asset registration"""
        for asset in sample_assets:
            assert lineage_tracker.register_data_asset(asset)
        
        assert len(lineage_tracker.assets) == 3
        assert "erp_customers" in lineage_tracker.assets
        assert lineage_tracker.lineage_graph.number_of_nodes() == 3
    
    def test_lineage_event_tracking(self, lineage_tracker, sample_assets):
        """Test lineage event tracking"""
        # Register assets
        for asset in sample_assets:
            lineage_tracker.register_data_asset(asset)
        
        # Track transformation event
        event_id = lineage_tracker.track_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_assets=["erp_customers", "crm_contacts"],
            target_assets=["unified_customers"],
            transformation_details={
                "transformation_type": "merge",
                "join_key": "email",
                "business_rules": ["deduplicate", "normalize_names"]
            },
            user_id="data_engineer_1",
            system_id="airflow_dag_123"
        )
        
        assert event_id != ""
        assert len(lineage_tracker.events) == 1
        assert lineage_tracker.lineage_graph.number_of_edges() == 2
    
    def test_upstream_lineage_analysis(self, lineage_tracker, sample_assets):
        """Test upstream lineage analysis"""
        # Setup
        for asset in sample_assets:
            lineage_tracker.register_data_asset(asset)
        
        lineage_tracker.track_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_assets=["erp_customers", "crm_contacts"],
            target_assets=["unified_customers"],
            transformation_details={"type": "merge"},
            user_id="data_engineer",
            system_id="etl_system"
        )
        
        # Get upstream lineage
        upstream = lineage_tracker.get_upstream_lineage("unified_customers")
        
        assert "error" not in upstream
        assert upstream["total_upstream_count"] == 2
        assert len(upstream["upstream_assets"]) == 2
        
        # Check that both source assets are included
        upstream_ids = [asset["id"] for asset in upstream["upstream_assets"]]
        assert "erp_customers" in upstream_ids
        assert "crm_contacts" in upstream_ids
    
    def test_impact_analysis(self, lineage_tracker, sample_assets):
        """Test impact analysis"""
        # Setup
        for asset in sample_assets:
            lineage_tracker.register_data_asset(asset)
        
        lineage_tracker.track_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_assets=["erp_customers"],
            target_assets=["unified_customers"],
            transformation_details={"type": "transform"},
            user_id="data_engineer",
            system_id="etl_system"
        )
        
        # Analyze impact
        impact = lineage_tracker.get_impact_analysis("erp_customers")
        
        assert "error" not in impact
        assert impact["total_impacted_assets"] == 1
        assert "impact_by_classification" in impact
        assert "impact_by_system" in impact
    
    def test_audit_report_generation(self, lineage_tracker, sample_assets):
        """Test audit report generation"""
        # Setup
        for asset in sample_assets:
            lineage_tracker.register_data_asset(asset)
        
        # Track multiple events
        for i in range(3):
            lineage_tracker.track_lineage_event(
                event_type=LineageEventType.DATA_INGESTION,
                source_assets=[],
                target_assets=["erp_customers"],
                transformation_details={"batch_id": f"batch_{i}"},
                user_id=f"user_{i}",
                system_id="ingestion_system"
            )
        
        # Generate audit report
        report = lineage_tracker.generate_audit_report()
        
        assert "error" not in report
        assert report["summary"]["total_events"] == 3
        assert report["summary"]["unique_assets_involved"] >= 1
        assert "event_breakdown" in report
        assert "user_activity" in report


class TestDataQualityAlertingIntegration:
    """Test data quality alerting functionality"""
    
    @pytest.fixture
    def alerting_system(self):
        return DataQualityAlertingSystem()
    
    @pytest.fixture
    def sample_alert_rules(self):
        return [
            AlertRule(
                id="critical_quality_alert",
                name="Critical Quality Issues",
                description="Alert when critical quality issues are found",
                trigger_conditions={
                    "min_quality_score": 80,
                    "max_critical_issues": 0
                },
                severity_threshold=QualitySeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                recipients=["admin@company.com", "data-team@company.com"],
                frequency=AlertFrequency.IMMEDIATE
            ),
            AlertRule(
                id="daily_quality_summary",
                name="Daily Quality Summary",
                description="Daily summary of quality metrics",
                trigger_conditions={
                    "min_quality_score": 95
                },
                severity_threshold=QualitySeverity.MEDIUM,
                channels=[AlertChannel.EMAIL],
                recipients=["manager@company.com"],
                frequency=AlertFrequency.DAILY
            )
        ]
    
    @pytest.fixture
    def sample_report_schedule(self):
        return ReportSchedule(
            id="weekly_quality_report",
            name="Weekly Quality Report",
            description="Comprehensive weekly quality report",
            datasets=["customers", "orders", "products"],
            report_format=ReportFormat.HTML,
            frequency=AlertFrequency.WEEKLY,
            recipients=["executives@company.com"]
        )
    
    @pytest.fixture
    def poor_quality_data(self):
        """Data with quality issues to trigger alerts"""
        return pd.DataFrame([
            {"id": "1", "email": "valid@example.com", "age": 25},
            {"id": "2", "email": None, "age": 30},  # Missing email
            {"id": "3", "email": "invalid-email", "age": -5},  # Invalid email and age
            {"id": "1", "email": "duplicate@example.com", "age": 40},  # Duplicate ID
            {"id": "5", "email": "", "age": None}  # Empty email, missing age
        ])
    
    def test_alert_rule_registration(self, alerting_system, sample_alert_rules):
        """Test alert rule registration"""
        for rule in sample_alert_rules:
            assert alerting_system.register_alert_rule(rule)
        
        assert len(alerting_system.alert_rules) == 2
        assert "critical_quality_alert" in alerting_system.alert_rules
    
    def test_report_schedule_registration(self, alerting_system, sample_report_schedule):
        """Test report schedule registration"""
        assert alerting_system.register_report_schedule(sample_report_schedule)
        assert len(alerting_system.report_schedules) == 1
        assert "weekly_quality_report" in alerting_system.report_schedules
    
    @pytest.mark.asyncio
    async def test_monitoring_and_alerting_workflow(self, alerting_system, sample_alert_rules, 
                                                  poor_quality_data):
        """Test complete monitoring and alerting workflow"""
        # Setup quality rules for the monitor
        quality_rules = [
            QualityRule(
                id="email_completeness",
                name="Email Completeness Check",
                description="Check email completeness",
                rule_type=QualityRuleType.COMPLETENESS,
                severity=QualitySeverity.CRITICAL,
                target_fields=["email"],
                parameters={"min_completeness": 0.8}
            ),
            QualityRule(
                id="id_uniqueness",
                name="ID Uniqueness Check",
                description="Check ID uniqueness",
                rule_type=QualityRuleType.UNIQUENESS,
                severity=QualitySeverity.CRITICAL,
                target_fields=["id"]
            )
        ]
        
        for rule in quality_rules:
            alerting_system.quality_monitor.register_quality_rule(rule)
        
        # Register alert rules
        for rule in sample_alert_rules:
            alerting_system.register_alert_rule(rule)
        
        # Mock notification handlers to avoid actual sending
        with patch.object(alerting_system, '_send_email_alert', new_callable=AsyncMock) as mock_email, \
             patch.object(alerting_system, '_send_dashboard_alert', new_callable=AsyncMock) as mock_dashboard:
            
            # Monitor and alert
            result = await alerting_system.monitor_and_alert("test_dataset", poor_quality_data)
            
            assert "quality_report" in result
            assert result["alerts_triggered"] > 0
            
            # Check that alerts were created
            assert len(alerting_system.alert_history) > 0
            
            # Verify notification methods were called
            assert mock_email.called or mock_dashboard.called
    
    @pytest.mark.asyncio
    async def test_scheduled_report_generation(self, alerting_system, sample_report_schedule):
        """Test scheduled report generation"""
        # Register schedule
        alerting_system.register_report_schedule(sample_report_schedule)
        
        # Create some mock quality history
        from scrollintel.core.data_quality_monitor import QualityReport
        
        mock_report = QualityReport(
            dataset_name="customers",
            assessment_timestamp=datetime.utcnow(),
            total_records=1000,
            overall_score=85.5,
            dimension_scores={"completeness": 90, "uniqueness": 80},
            issues=[],
            metrics={}
        )
        
        alerting_system.quality_monitor.quality_history.append(mock_report)
        
        # Mock email sending
        with patch.object(alerting_system, '_send_report_email', new_callable=AsyncMock) as mock_send:
            # Force report generation by setting last_generated to past
            sample_report_schedule.last_generated = datetime.utcnow() - timedelta(days=8)
            
            result = await alerting_system.generate_scheduled_reports()
            
            assert result["reports_generated"] == 1
            assert len(result["report_details"]) == 1
            assert mock_send.called
    
    def test_alert_acknowledgment(self, alerting_system):
        """Test alert acknowledgment"""
        # Create a mock alert
        from scrollintel.core.data_quality_alerting import Alert
        
        alert = Alert(
            id="test_alert_123",
            rule_id="test_rule",
            dataset_name="test_dataset",
            severity=QualitySeverity.HIGH,
            title="Test Alert",
            message="Test alert message",
            issues=[]
        )
        
        alerting_system.alert_history.append(alert)
        
        # Acknowledge alert
        assert alerting_system.acknowledge_alert("test_alert_123", "user123")
        
        # Check acknowledgment
        acknowledged_alert = alerting_system.alert_history[0]
        assert acknowledged_alert.acknowledged
        assert acknowledged_alert.acknowledged_by == "user123"
        assert acknowledged_alert.acknowledged_at is not None
    
    def test_alert_summary_analysis(self, alerting_system):
        """Test alert summary analysis"""
        # Create mock alerts
        from scrollintel.core.data_quality_alerting import Alert
        
        alerts = [
            Alert(
                id=f"alert_{i}",
                rule_id="test_rule",
                dataset_name="test_dataset",
                severity=QualitySeverity.HIGH if i % 2 == 0 else QualitySeverity.MEDIUM,
                title=f"Alert {i}",
                message=f"Alert message {i}",
                issues=[],
                triggered_at=datetime.utcnow() - timedelta(hours=i)
            )
            for i in range(5)
        ]
        
        # Acknowledge some alerts
        alerts[0].acknowledged = True
        alerts[2].acknowledged = True
        
        alerting_system.alert_history.extend(alerts)
        
        # Get summary
        summary = alerting_system.get_alert_summary(days=1)
        
        assert summary["total_alerts"] == 5
        assert "alerts_by_severity" in summary
        assert summary["alerts_by_severity"]["high"] == 3
        assert summary["alerts_by_severity"]["medium"] == 2
        assert summary["acknowledgment_rate"] == 0.4  # 2 out of 5 acknowledged


class TestEndToEndDataQualityWorkflow:
    """Test complete end-to-end data quality workflow"""
    
    @pytest.fixture
    def complete_system(self):
        """Setup complete integrated system"""
        return {
            "normalizer": DataNormalizer(),
            "quality_monitor": DataQualityMonitor(),
            "reconciliation_engine": DataReconciliationEngine(),
            "lineage_tracker": DataLineageTracker(),
            "alerting_system": DataQualityAlertingSystem()
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, complete_system):
        """Test complete end-to-end workflow"""
        normalizer = complete_system["normalizer"]
        quality_monitor = complete_system["quality_monitor"]
        reconciliation_engine = complete_system["reconciliation_engine"]
        lineage_tracker = complete_system["lineage_tracker"]
        alerting_system = complete_system["alerting_system"]
        
        # 1. Setup schemas and normalization
        source_schema = DataSchema(
            name="raw_data",
            version="1.0",
            fields=[
                SchemaField("id", DataType.STRING),
                SchemaField("name", DataType.STRING),
                SchemaField("email", DataType.STRING)
            ]
        )
        
        target_schema = DataSchema(
            name="clean_data",
            version="1.0",
            fields=[
                SchemaField("customer_id", DataType.STRING),
                SchemaField("customer_name", DataType.STRING),
                SchemaField("email_address", DataType.STRING)
            ]
        )
        
        mappings = [
            SchemaMapping("id", "customer_id", TransformationType.DIRECT_MAPPING),
            SchemaMapping("name", "customer_name", TransformationType.DIRECT_MAPPING),
            SchemaMapping("email", "email_address", TransformationType.DIRECT_MAPPING)
        ]
        
        normalizer.register_schema(source_schema)
        normalizer.register_schema(target_schema)
        normalizer.register_mapping("raw_data", "clean_data", mappings)
        
        # 2. Setup quality rules
        quality_rules = [
            QualityRule(
                id="completeness_check",
                name="Data Completeness",
                description="Check data completeness",
                rule_type=QualityRuleType.COMPLETENESS,
                severity=QualitySeverity.HIGH,
                target_fields=["email_address"],
                parameters={"min_completeness": 0.9}
            )
        ]
        
        for rule in quality_rules:
            quality_monitor.register_quality_rule(rule)
        
        # 3. Setup lineage tracking
        raw_asset = DataAsset(
            id="raw_data_table",
            name="Raw Data Table",
            type="table",
            source_system="source_db",
            schema_info={"fields": ["id", "name", "email"]},
            classification=DataClassification.INTERNAL,
            owner="data_team"
        )
        
        clean_asset = DataAsset(
            id="clean_data_table",
            name="Clean Data Table",
            type="table",
            source_system="analytics_db",
            schema_info={"fields": ["customer_id", "customer_name", "email_address"]},
            classification=DataClassification.INTERNAL,
            owner="data_team"
        )
        
        lineage_tracker.register_data_asset(raw_asset)
        lineage_tracker.register_data_asset(clean_asset)
        
        # 4. Setup alerting
        alert_rule = AlertRule(
            id="quality_alert",
            name="Quality Alert",
            description="Alert on quality issues",
            trigger_conditions={"min_quality_score": 85},
            severity_threshold=QualitySeverity.MEDIUM,
            channels=[AlertChannel.DASHBOARD],
            recipients=["admin@company.com"],
            frequency=AlertFrequency.IMMEDIATE
        )
        
        alerting_system.register_alert_rule(alert_rule)
        
        # 5. Process data through complete workflow
        raw_data = pd.DataFrame([
            {"id": "1", "name": "John Doe", "email": "john@example.com"},
            {"id": "2", "name": "Jane Smith", "email": None},  # Missing email
            {"id": "3", "name": "Bob Johnson", "email": "bob@example.com"}
        ])
        
        # Normalize data
        normalization_result = normalizer.normalize_data(raw_data, "raw_data", "clean_data")
        assert normalization_result.success
        
        normalized_data = normalization_result.normalized_data
        
        # Track lineage
        lineage_tracker.track_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_assets=["raw_data_table"],
            target_assets=["clean_data_table"],
            transformation_details={"operation": "normalization"},
            user_id="etl_process",
            system_id="data_pipeline"
        )
        
        # Monitor quality and alert
        with patch.object(alerting_system, '_send_dashboard_alert', new_callable=AsyncMock):
            monitoring_result = await alerting_system.monitor_and_alert("clean_data", normalized_data)
        
        # Verify complete workflow
        assert normalization_result.success
        assert "quality_report" in monitoring_result
        assert len(lineage_tracker.events) == 1
        assert lineage_tracker.lineage_graph.number_of_edges() == 1
        
        # Check that quality issues were detected (missing email)
        quality_report = monitoring_result["quality_report"]
        assert len(quality_report.issues) > 0
        
        # Verify lineage can be traced
        upstream = lineage_tracker.get_upstream_lineage("clean_data_table")
        assert upstream["total_upstream_count"] == 1
        assert upstream["upstream_assets"][0]["id"] == "raw_data_table"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])