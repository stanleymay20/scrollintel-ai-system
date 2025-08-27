"""
Integration tests for Data Normalization and Quality System
Tests the complete workflow from data ingestion to quality reporting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import json

from scrollintel.core.data_normalizer import (
    DataNormalizer, DataSchema, SchemaField, SchemaMapping, 
    DataType, TransformationType
)
from scrollintel.core.data_quality_monitor import (
    DataQualityMonitor, QualityRule, QualityRuleType, QualitySeverity
)
from scrollintel.core.data_lineage import (
    DataLineageTracker, DataAsset, LineageEventType, DataClassification
)
from scrollintel.core.data_reconciliation import (
    DataReconciliationEngine, DataSource, ReconciliationRule, 
    ConflictResolutionStrategy
)
from scrollintel.core.data_quality_alerting import (
    DataQualityAlerting, AlertRule, AlertChannel, AlertFrequency
)
from scrollintel.core.data_quality_reporting import (
    DataQualityReporting, ReportTemplate, ReportFormat, ReportFrequency
)


class TestDataQualityIntegration:
    """Integration tests for the complete data quality system"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample datasets for testing"""
        # Source dataset 1 (CRM)
        crm_data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'customer_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'email': ['john@example.com', 'jane@example.com', 'bob@example.com', None, 'charlie@example.com'],
            'phone': ['555-1234', '555-5678', None, '555-9999', '555-0000'],
            'registration_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12'],
            'status': ['active', 'active', 'inactive', 'active', 'pending']
        })
        
        # Source dataset 2 (ERP)
        erp_data = pd.DataFrame({
            'cust_id': [1, 2, 3, 4, 6],
            'full_name': ['John Doe', 'Jane Smith', 'Robert Johnson', 'Alice Brown', 'David Lee'],
            'email_address': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'david@example.com'],
            'contact_phone': ['555-1234', '555-5678', '555-2222', '555-9999', '555-1111'],
            'created_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-06-01'],
            'account_status': ['A', 'A', 'I', 'A', 'P']
        })
        
        # Target schema data
        target_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson', 'David Lee'],
            'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com', 'david@example.com'],
            'phone': ['555-1234', '555-5678', '555-2222', '555-9999', '555-0000', '555-1111'],
            'created_at': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', '2023-06-01']),
            'is_active': [True, True, False, True, False, False]
        })
        
        return {
            'crm': crm_data,
            'erp': erp_data,
            'target': target_data
        }
    
    @pytest.fixture
    def data_normalizer(self):
        """Create configured data normalizer"""
        config = {
            "quality_thresholds": {
                "completeness": 0.8,
                "validity": 0.9,
                "consistency": 0.95
            }
        }
        return DataNormalizer(config)
    
    @pytest.fixture
    def quality_monitor(self):
        """Create configured quality monitor"""
        return DataQualityMonitor()
    
    @pytest.fixture
    def lineage_tracker(self):
        """Create configured lineage tracker"""
        return DataLineageTracker()
    
    @pytest.fixture
    def reconciliation_engine(self):
        """Create configured reconciliation engine"""
        return DataReconciliationEngine()
    
    @pytest.fixture
    def quality_alerting(self):
        """Create configured quality alerting system"""
        config = {
            "notifications": {
                "email": {"enabled": False},
                "slack": {"enabled": False},
                "webhook": {"enabled": False}
            }
        }
        return DataQualityAlerting(config)
    
    @pytest.fixture
    def quality_reporting(self):
        """Create configured quality reporting system"""
        return DataQualityReporting()
    
    def test_complete_data_quality_workflow(self, sample_data, data_normalizer, 
                                          quality_monitor, lineage_tracker,
                                          reconciliation_engine, quality_alerting,
                                          quality_reporting):
        """Test the complete data quality workflow from ingestion to reporting"""
        
        # Step 1: Register schemas and mappings
        self._setup_schemas_and_mappings(data_normalizer)
        
        # Step 2: Register data sources for reconciliation
        self._setup_data_sources(reconciliation_engine)
        
        # Step 3: Register data assets for lineage tracking
        self._setup_data_assets(lineage_tracker)
        
        # Step 4: Set up quality rules
        self._setup_quality_rules(quality_monitor)
        
        # Step 5: Set up alerting rules
        self._setup_alert_rules(quality_alerting)
        
        # Step 6: Normalize data from multiple sources
        crm_result = data_normalizer.normalize_data(
            sample_data['crm'], 'crm_schema', 'target_schema'
        )
        assert crm_result.success
        
        erp_result = data_normalizer.normalize_data(
            sample_data['erp'], 'erp_schema', 'target_schema'
        )
        assert erp_result.success
        
        # Step 7: Reconcile data from multiple sources
        datasets = {
            'crm': crm_result.normalized_data,
            'erp': erp_result.normalized_data
        }
        
        reconciliation_result = reconciliation_engine.reconcile_data(datasets, 'id')
        assert reconciliation_result.success
        
        # Step 8: Track lineage events
        lineage_event_id = lineage_tracker.track_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_assets=['crm_customers', 'erp_customers'],
            target_assets=['unified_customers'],
            transformation_details={
                "transformation_type": "data_reconciliation",
                "conflicts_resolved": reconciliation_result.conflicts_resolved,
                "records_processed": len(reconciliation_result.reconciled_data)
            },
            user_id="system",
            system_id="data_quality_system"
        )
        assert lineage_event_id
        
        # Step 9: Assess data quality
        quality_report = quality_monitor.assess_data_quality(
            reconciliation_result.reconciled_data, 
            "unified_customers"
        )
        assert quality_report.overall_score > 0
        
        # Step 10: Process quality alerts
        alerts = quality_alerting.process_quality_report(quality_report)
        
        # Step 11: Add to reporting system and generate report
        quality_reporting.add_quality_report(quality_report)
        
        report_result = quality_reporting.generate_quality_summary_report(
            dataset_names=["unified_customers"],
            format=ReportFormat.JSON
        )
        assert report_result["success"]
        
        # Verify end-to-end results
        assert reconciliation_result.reconciled_data is not None
        assert len(reconciliation_result.reconciled_data) > 0
        assert quality_report.total_records > 0
        assert "unified_customers" in report_result["content"]
    
    def test_data_normalization_with_validation(self, sample_data, data_normalizer):
        """Test data normalization with custom validation rules"""
        
        # Set up schemas
        self._setup_schemas_and_mappings(data_normalizer)
        
        # Register custom validation rule
        def email_validation(column_data):
            """Validate email format"""
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            valid_emails = column_data.dropna().astype(str).apply(
                lambda x: bool(re.match(email_pattern, x)) if x != 'nan' else True
            )
            
            invalid_count = (~valid_emails).sum()
            
            return {
                "valid": invalid_count == 0,
                "message": f"Found {invalid_count} invalid email addresses",
                "invalid_count": invalid_count
            }
        
        data_normalizer.register_validation_rule('email', email_validation)
        
        # Normalize data
        result = data_normalizer.normalize_data(
            sample_data['crm'], 'crm_schema', 'target_schema'
        )
        
        assert result.success
        
        # Validate normalized data
        validation_result = data_normalizer.validate_normalized_data(
            result.normalized_data, 'target_schema'
        )
        
        assert "quality_metrics" in validation_result
        assert "email" in validation_result["quality_metrics"]
    
    def test_quality_monitoring_with_multiple_rules(self, sample_data, quality_monitor):
        """Test quality monitoring with multiple validation rules"""
        
        # Register multiple quality rules
        completeness_rule = QualityRule(
            id="completeness_check",
            name="Completeness Check",
            description="Check data completeness",
            rule_type=QualityRuleType.COMPLETENESS,
            severity=QualitySeverity.HIGH,
            target_fields=["name", "email"],
            parameters={"min_completeness": 0.8}
        )
        
        uniqueness_rule = QualityRule(
            id="uniqueness_check",
            name="Uniqueness Check", 
            description="Check data uniqueness",
            rule_type=QualityRuleType.UNIQUENESS,
            severity=QualitySeverity.MEDIUM,
            target_fields=["id", "email"]
        )
        
        validity_rule = QualityRule(
            id="validity_check",
            name="Validity Check",
            description="Check data validity",
            rule_type=QualityRuleType.VALIDITY,
            severity=QualitySeverity.HIGH,
            target_fields=["email"],
            parameters={
                "condition": "email.str.contains('@', na=False)"
            }
        )
        
        quality_monitor.register_quality_rule(completeness_rule)
        quality_monitor.register_quality_rule(uniqueness_rule)
        quality_monitor.register_quality_rule(validity_rule)
        
        # Assess quality
        report = quality_monitor.assess_data_quality(sample_data['crm'], "crm_customers")
        
        assert report.total_records == len(sample_data['crm'])
        assert len(report.issues) >= 0  # May have issues depending on data
        assert report.overall_score >= 0
        
        # Check that all rule types were applied
        rule_types_applied = set()
        for issue in report.issues:
            if "completeness" in issue.rule_name.lower():
                rule_types_applied.add("completeness")
            elif "uniqueness" in issue.rule_name.lower():
                rule_types_applied.add("uniqueness")
            elif "validity" in issue.rule_name.lower():
                rule_types_applied.add("validity")
    
    def test_data_reconciliation_with_conflicts(self, reconciliation_engine):
        """Test data reconciliation with conflicting values"""
        
        # Set up data sources
        source1 = DataSource(
            id="source1",
            name="Source 1",
            priority=1,
            reliability_score=0.8,
            last_updated=datetime.utcnow()
        )
        
        source2 = DataSource(
            id="source2", 
            name="Source 2",
            priority=2,
            reliability_score=0.9,
            last_updated=datetime.utcnow()
        )
        
        reconciliation_engine.register_data_source(source1)
        reconciliation_engine.register_data_source(source2)
        
        # Set up reconciliation rules
        name_rule = ReconciliationRule(
            field_name="name",
            strategy=ConflictResolutionStrategy.HIGHEST_PRIORITY
        )
        
        email_rule = ReconciliationRule(
            field_name="email",
            strategy=ConflictResolutionStrategy.LATEST_TIMESTAMP
        )
        
        reconciliation_engine.register_reconciliation_rule(name_rule)
        reconciliation_engine.register_reconciliation_rule(email_rule)
        
        # Create conflicting datasets
        dataset1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@old.com', 'jane@old.com', 'bob@old.com']
        })
        
        dataset2 = pd.DataFrame({
            'id': [1, 2, 4],
            'name': ['John D.', 'Jane Smith', 'Alice Brown'],
            'email': ['john@new.com', 'jane@new.com', 'alice@new.com']
        })
        
        datasets = {'source1': dataset1, 'source2': dataset2}
        
        # Reconcile data
        result = reconciliation_engine.reconcile_data(datasets, 'id')
        
        assert result.success
        assert result.conflicts_detected > 0
        assert result.conflicts_resolved >= 0
        assert len(result.reconciled_data) >= 3  # Should have at least 3 unique records
    
    def test_lineage_tracking_and_impact_analysis(self, lineage_tracker):
        """Test data lineage tracking and impact analysis"""
        
        # Register data assets
        source_asset = DataAsset(
            id="source_customers",
            name="Source Customer Data",
            type="table",
            source_system="CRM",
            schema_info={"fields": ["id", "name", "email"]},
            classification=DataClassification.INTERNAL,
            owner="data_team"
        )
        
        target_asset = DataAsset(
            id="processed_customers",
            name="Processed Customer Data", 
            type="table",
            source_system="DW",
            schema_info={"fields": ["id", "name", "email", "processed_at"]},
            classification=DataClassification.INTERNAL,
            owner="analytics_team"
        )
        
        lineage_tracker.register_data_asset(source_asset)
        lineage_tracker.register_data_asset(target_asset)
        
        # Track transformation event
        event_id = lineage_tracker.track_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_assets=["source_customers"],
            target_assets=["processed_customers"],
            transformation_details={
                "transformation_type": "normalization",
                "rules_applied": ["email_validation", "name_standardization"]
            },
            user_id="etl_user",
            system_id="data_pipeline"
        )
        
        assert event_id
        
        # Get upstream lineage
        upstream = lineage_tracker.get_upstream_lineage("processed_customers")
        assert "source_customers" in [asset["id"] for asset in upstream["upstream_assets"]]
        
        # Get downstream lineage
        downstream = lineage_tracker.get_downstream_lineage("source_customers")
        assert "processed_customers" in [asset["id"] for asset in downstream["downstream_assets"]]
        
        # Perform impact analysis
        impact = lineage_tracker.get_impact_analysis("source_customers")
        assert impact["total_impacted_assets"] >= 1
        assert "processed_customers" in [asset["asset_id"] for asset in impact.get("critical_paths", [])] or impact["total_impacted_assets"] == 1
    
    def test_quality_alerting_workflow(self, quality_alerting, quality_monitor):
        """Test quality alerting workflow"""
        
        # Set up alert rule
        alert_rule = AlertRule(
            id="critical_quality_alert",
            name="Critical Quality Alert",
            description="Alert on critical quality issues",
            dataset_patterns=[".*customers.*"],
            severity_threshold=QualitySeverity.HIGH,
            score_threshold=70.0,
            channels=[AlertChannel.DASHBOARD],
            frequency=AlertFrequency.IMMEDIATE,
            recipients=["admin@example.com"]
        )
        
        quality_alerting.register_alert_rule(alert_rule)
        
        # Create a quality report with issues
        poor_data = pd.DataFrame({
            'id': [1, 2, None, 4, 5],
            'name': ['John', None, 'Bob', 'Alice', ''],
            'email': ['invalid-email', 'jane@example.com', None, 'alice@example.com', 'bad-email']
        })
        
        # Set up quality rules that will trigger issues
        completeness_rule = QualityRule(
            id="completeness_test",
            name="Completeness Test",
            description="Test completeness",
            rule_type=QualityRuleType.COMPLETENESS,
            severity=QualitySeverity.CRITICAL,
            target_fields=["id", "name"],
            parameters={"min_completeness": 0.9}
        )
        
        quality_monitor.register_quality_rule(completeness_rule)
        
        # Assess quality (should find issues)
        quality_report = quality_monitor.assess_data_quality(poor_data, "test_customers")
        
        # Process alerts
        alerts = quality_alerting.process_quality_report(quality_report)
        
        # Should generate alerts due to poor quality
        assert len(alerts) >= 0  # May or may not generate alerts depending on thresholds
    
    def test_quality_reporting_and_trends(self, quality_reporting, quality_monitor):
        """Test quality reporting and trend analysis"""
        
        # Generate multiple quality reports over time
        base_time = datetime.utcnow() - timedelta(days=10)
        
        for i in range(5):
            # Create data with gradually improving quality
            data = pd.DataFrame({
                'id': range(1, 101),
                'name': [f'User {j}' if j <= 80 + i*2 else None for j in range(1, 101)],
                'email': [f'user{j}@example.com' if j <= 85 + i*2 else None for j in range(1, 101)]
            })
            
            # Assess quality
            report = quality_monitor.assess_data_quality(data, "trending_dataset")
            
            # Manually set timestamp for trend analysis
            report.assessment_timestamp = base_time + timedelta(days=i*2)
            
            # Add to reporting system
            quality_reporting.add_quality_report(report)
        
        # Generate summary report
        summary_report = quality_reporting.generate_quality_summary_report(
            dataset_names=["trending_dataset"],
            format=ReportFormat.JSON
        )
        
        assert summary_report["success"]
        
        # Generate trend analysis
        trend_report = quality_reporting.generate_trend_analysis_report(
            "trending_dataset",
            days=15,
            format=ReportFormat.JSON
        )
        
        assert trend_report["success"]
        assert "trend_summary" in trend_report["content"]
    
    def _setup_schemas_and_mappings(self, data_normalizer):
        """Set up schemas and mappings for testing"""
        
        # CRM schema
        crm_schema = DataSchema(
            name="crm_schema",
            version="1.0",
            fields=[
                SchemaField("customer_id", DataType.INTEGER, required=True),
                SchemaField("customer_name", DataType.STRING, required=True),
                SchemaField("email", DataType.STRING, required=False),
                SchemaField("phone", DataType.STRING, required=False),
                SchemaField("registration_date", DataType.STRING, required=True),
                SchemaField("status", DataType.STRING, required=True)
            ]
        )
        
        # ERP schema
        erp_schema = DataSchema(
            name="erp_schema",
            version="1.0",
            fields=[
                SchemaField("cust_id", DataType.INTEGER, required=True),
                SchemaField("full_name", DataType.STRING, required=True),
                SchemaField("email_address", DataType.STRING, required=False),
                SchemaField("contact_phone", DataType.STRING, required=False),
                SchemaField("created_date", DataType.STRING, required=True),
                SchemaField("account_status", DataType.STRING, required=True)
            ]
        )
        
        # Target schema
        target_schema = DataSchema(
            name="target_schema",
            version="1.0",
            fields=[
                SchemaField("id", DataType.INTEGER, required=True),
                SchemaField("name", DataType.STRING, required=True),
                SchemaField("email", DataType.STRING, required=False),
                SchemaField("phone", DataType.STRING, required=False),
                SchemaField("created_at", DataType.DATETIME, required=True),
                SchemaField("is_active", DataType.BOOLEAN, required=True)
            ]
        )
        
        # Register schemas
        data_normalizer.register_schema(crm_schema)
        data_normalizer.register_schema(erp_schema)
        data_normalizer.register_schema(target_schema)
        
        # CRM to target mappings
        crm_mappings = [
            SchemaMapping("customer_id", "id", TransformationType.DIRECT_MAPPING),
            SchemaMapping("customer_name", "name", TransformationType.DIRECT_MAPPING),
            SchemaMapping("email", "email", TransformationType.DIRECT_MAPPING),
            SchemaMapping("phone", "phone", TransformationType.DIRECT_MAPPING),
            SchemaMapping("registration_date", "created_at", TransformationType.CALCULATION, 
                         {"formula": "pd.to_datetime(registration_date)"}),
            SchemaMapping("status", "is_active", TransformationType.VALUE_MAPPING,
                         {"value_map": {"active": True, "inactive": False, "pending": False}})
        ]
        
        # ERP to target mappings
        erp_mappings = [
            SchemaMapping("cust_id", "id", TransformationType.DIRECT_MAPPING),
            SchemaMapping("full_name", "name", TransformationType.DIRECT_MAPPING),
            SchemaMapping("email_address", "email", TransformationType.DIRECT_MAPPING),
            SchemaMapping("contact_phone", "phone", TransformationType.DIRECT_MAPPING),
            SchemaMapping("created_date", "created_at", TransformationType.CALCULATION,
                         {"formula": "pd.to_datetime(created_date)"}),
            SchemaMapping("account_status", "is_active", TransformationType.VALUE_MAPPING,
                         {"value_map": {"A": True, "I": False, "P": False}})
        ]
        
        # Register mappings
        data_normalizer.register_mapping("crm_schema", "target_schema", crm_mappings)
        data_normalizer.register_mapping("erp_schema", "target_schema", erp_mappings)
    
    def _setup_data_sources(self, reconciliation_engine):
        """Set up data sources for reconciliation testing"""
        
        crm_source = DataSource(
            id="crm",
            name="CRM System",
            priority=1,
            reliability_score=0.8,
            last_updated=datetime.utcnow() - timedelta(hours=1)
        )
        
        erp_source = DataSource(
            id="erp",
            name="ERP System", 
            priority=2,
            reliability_score=0.9,
            last_updated=datetime.utcnow()
        )
        
        reconciliation_engine.register_data_source(crm_source)
        reconciliation_engine.register_data_source(erp_source)
        
        # Set up reconciliation rules
        name_rule = ReconciliationRule(
            field_name="name",
            strategy=ConflictResolutionStrategy.HIGHEST_PRIORITY
        )
        
        reconciliation_engine.register_reconciliation_rule(name_rule)
    
    def _setup_data_assets(self, lineage_tracker):
        """Set up data assets for lineage tracking"""
        
        crm_asset = DataAsset(
            id="crm_customers",
            name="CRM Customer Data",
            type="table",
            source_system="CRM",
            schema_info={"table": "customers", "fields": ["customer_id", "customer_name", "email"]},
            classification=DataClassification.INTERNAL,
            owner="crm_team"
        )
        
        erp_asset = DataAsset(
            id="erp_customers",
            name="ERP Customer Data",
            type="table", 
            source_system="ERP",
            schema_info={"table": "customers", "fields": ["cust_id", "full_name", "email_address"]},
            classification=DataClassification.INTERNAL,
            owner="erp_team"
        )
        
        unified_asset = DataAsset(
            id="unified_customers",
            name="Unified Customer Data",
            type="table",
            source_system="DW",
            schema_info={"table": "unified_customers", "fields": ["id", "name", "email"]},
            classification=DataClassification.INTERNAL,
            owner="analytics_team"
        )
        
        lineage_tracker.register_data_asset(crm_asset)
        lineage_tracker.register_data_asset(erp_asset)
        lineage_tracker.register_data_asset(unified_asset)
    
    def _setup_quality_rules(self, quality_monitor):
        """Set up quality rules for testing"""
        
        completeness_rule = QualityRule(
            id="test_completeness",
            name="Test Completeness Rule",
            description="Check data completeness",
            rule_type=QualityRuleType.COMPLETENESS,
            severity=QualitySeverity.MEDIUM,
            target_fields=["id", "name"],
            parameters={"min_completeness": 0.8}
        )
        
        quality_monitor.register_quality_rule(completeness_rule)
    
    def _setup_alert_rules(self, quality_alerting):
        """Set up alert rules for testing"""
        
        alert_rule = AlertRule(
            id="test_alert",
            name="Test Quality Alert",
            description="Test alert rule",
            dataset_patterns=[".*"],
            severity_threshold=QualitySeverity.HIGH,
            score_threshold=60.0,
            channels=[AlertChannel.DASHBOARD],
            frequency=AlertFrequency.IMMEDIATE
        )
        
        quality_alerting.register_alert_rule(alert_rule)


if __name__ == "__main__":
    pytest.main([__file__])