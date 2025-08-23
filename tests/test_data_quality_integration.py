"""
Integration tests for data quality monitoring workflows
Tests the complete data quality monitoring system end-to-end
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import asyncio
from unittest.mock import Mock, patch

from scrollintel.models.data_quality_models import (
    Base, QualityRule, QualityReport, DataAnomaly, DataProfile, QualityAlert,
    QualityRuleType, Severity, QualityStatus
)
from scrollintel.engines.data_quality_monitor import DataQualityMonitor
from scrollintel.engines.anomaly_detector import AnomalyDetector
from scrollintel.engines.quality_alerting import QualityAlertManager, RealTimeQualityMonitor, AlertConfig, AlertChannel
from scrollintel.engines.data_profiler import DataProfiler, ProfilingConfig

@pytest.fixture
def db_session():
    """Create test database session"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    data = pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'User_{i}' for i in range(1, 1001)],
        'email': [f'user{i}@example.com' if i % 10 != 0 else None for i in range(1, 1001)],
        'age': np.random.normal(35, 10, 1000).astype(int),
        'salary': np.random.normal(50000, 15000, 1000),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 1000),
        'join_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'is_active': np.random.choice([True, False], 1000, p=[0.8, 0.2])
    })
    
    # Introduce some quality issues
    data.loc[50:60, 'email'] = 'invalid_email'  # Invalid emails
    data.loc[100:110, 'age'] = -5  # Invalid ages
    data.loc[200:205, 'salary'] = 1000000  # Outlier salaries
    
    return data

@pytest.fixture
def quality_monitor(db_session):
    """Create data quality monitor instance"""
    return DataQualityMonitor(db_session)

@pytest.fixture
def anomaly_detector(db_session):
    """Create anomaly detector instance"""
    return AnomalyDetector(db_session)

@pytest.fixture
def data_profiler(db_session):
    """Create data profiler instance"""
    return DataProfiler(db_session)

@pytest.fixture
def alert_manager(db_session):
    """Create alert manager instance"""
    config = AlertConfig(
        channels=[AlertChannel.DASHBOARD],
        recipients=['test@example.com'],
        severity_threshold=Severity.MEDIUM
    )
    return QualityAlertManager(db_session, config)

class TestDataQualityWorkflow:
    """Test complete data quality monitoring workflow"""
    
    def test_complete_quality_workflow(self, db_session, sample_data, quality_monitor, 
                                     anomaly_detector, data_profiler):
        """Test complete end-to-end quality monitoring workflow"""
        table_name = "test_users"
        pipeline_id = "test_pipeline_001"
        
        # Step 1: Create data profiles
        profiles = data_profiler.create_comprehensive_profile(sample_data, table_name, pipeline_id)
        
        assert len(profiles) == len(sample_data.columns)
        assert all(p.table_name == table_name for p in profiles)
        
        # Step 2: Establish quality baseline and get recommended rules
        baseline = data_profiler.establish_quality_baseline(profiles, table_name, pipeline_id)
        
        assert baseline.completeness_baseline > 0
        assert len(baseline.recommended_rules) > 0
        
        # Step 3: Create quality rules from recommendations
        rule_ids = []
        for rule_config in baseline.recommended_rules[:5]:  # Test first 5 rules
            rule_id = quality_monitor.create_quality_rule(rule_config)
            rule_ids.append(rule_id)
        
        assert len(rule_ids) == 5
        
        # Step 4: Validate data against rules
        rules = db_session.query(QualityRule).filter(QualityRule.id.in_(rule_ids)).all()
        reports = quality_monitor.validate_data_batch(sample_data, rules, "exec_001")
        
        assert len(reports) == len(rules)
        assert all(isinstance(r, QualityReport) for r in reports)
        
        # Step 5: Detect anomalies
        email_profile = next(p for p in profiles if p.column_name == 'email')
        anomalies = anomaly_detector.detect_anomalies(sample_data, email_profile)
        
        # Should detect some anomalies due to invalid emails we introduced
        assert len(anomalies) > 0
        
        # Step 6: Check quality metrics
        time_range = (datetime.utcnow() - timedelta(hours=1), datetime.utcnow())
        metrics = quality_monitor.get_quality_metrics(pipeline_id, time_range)
        
        assert metrics['total_checks'] > 0
        assert 'average_score' in metrics
    
    def test_data_profiling_comprehensive(self, db_session, sample_data, data_profiler):
        """Test comprehensive data profiling"""
        table_name = "test_comprehensive"
        
        profiles = data_profiler.create_comprehensive_profile(sample_data, table_name)
        
        # Check each column type is profiled correctly
        id_profile = next(p for p in profiles if p.column_name == 'id')
        assert id_profile.unique_count == len(sample_data)  # Should be unique
        assert id_profile.completeness_score == 100.0  # No nulls
        
        email_profile = next(p for p in profiles if p.column_name == 'email')
        assert email_profile.completeness_score < 100.0  # Has nulls
        assert 'email' in email_profile.format_patterns  # Should detect email pattern
        
        age_profile = next(p for p in profiles if p.column_name == 'age')
        assert age_profile.mean_value is not None  # Numeric statistics
        assert age_profile.std_deviation is not None
        
        department_profile = next(p for p in profiles if p.column_name == 'department')
        assert department_profile.most_frequent_values is not None  # Categorical analysis
    
    def test_quality_rule_validation(self, db_session, sample_data, quality_monitor):
        """Test quality rule validation with different rule types"""
        
        # Test completeness rule
        completeness_rule = QualityRule(
            name="Email completeness",
            rule_type=QualityRuleType.COMPLETENESS,
            target_table="test_users",
            target_column="email",
            threshold_value=0.95,
            severity=Severity.MEDIUM
        )
        db_session.add(completeness_rule)
        db_session.commit()
        
        reports = quality_monitor.validate_data_batch(sample_data, [completeness_rule])
        assert len(reports) == 1
        assert reports[0].status == QualityStatus.FAILED  # Should fail due to nulls
        
        # Test validity rule
        validity_rule = QualityRule(
            name="Age validity",
            rule_type=QualityRuleType.VALIDITY,
            target_table="test_users",
            target_column="age",
            conditions={"min_value": 0, "max_value": 120},
            threshold_value=0.95,
            severity=Severity.HIGH
        )
        db_session.add(validity_rule)
        db_session.commit()
        
        reports = quality_monitor.validate_data_batch(sample_data, [validity_rule])
        assert len(reports) == 1
        assert reports[0].status == QualityStatus.FAILED  # Should fail due to negative ages
        
        # Test uniqueness rule
        uniqueness_rule = QualityRule(
            name="ID uniqueness",
            rule_type=QualityRuleType.UNIQUENESS,
            target_table="test_users",
            target_column="id",
            threshold_value=1.0,
            severity=Severity.CRITICAL
        )
        db_session.add(uniqueness_rule)
        db_session.commit()
        
        reports = quality_monitor.validate_data_batch(sample_data, [uniqueness_rule])
        assert len(reports) == 1
        assert reports[0].status == QualityStatus.PASSED  # Should pass - IDs are unique
    
    def test_anomaly_detection_methods(self, db_session, sample_data, anomaly_detector, data_profiler):
        """Test different anomaly detection methods"""
        
        # Create profile for salary column (has outliers)
        salary_profile = data_profiler._profile_column(sample_data, "test_users", "salary")
        db_session.add(salary_profile)
        db_session.commit()
        
        # Test statistical anomaly detection
        anomalies = anomaly_detector._detect_statistical_anomalies(sample_data, "salary", salary_profile)
        assert len(anomalies) > 0  # Should detect outlier salaries
        
        # Test isolation forest detection
        anomalies_iso = anomaly_detector._detect_isolation_forest_anomalies(sample_data, "salary", salary_profile)
        assert len(anomalies_iso) > 0  # Should detect anomalies
        
        # Test data drift detection
        # Create modified data with different distribution
        modified_data = sample_data.copy()
        modified_data['salary'] = modified_data['salary'] * 2  # Double all salaries
        
        drift_result = anomaly_detector.detect_data_drift(modified_data, salary_profile, "salary")
        assert drift_result['drift_detected'] == True
        assert drift_result['drift_type'] in ['mean_shift', 'distribution_shift']
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, db_session, sample_data, alert_manager):
        """Test real-time quality monitoring and alerting"""
        
        # Create a quality rule that will fail
        rule = QualityRule(
            name="Test alert rule",
            rule_type=QualityRuleType.COMPLETENESS,
            target_table="test_users",
            target_column="email",
            threshold_value=0.99,  # High threshold to trigger failure
            severity=Severity.HIGH,
            target_pipeline_id="test_pipeline"
        )
        db_session.add(rule)
        db_session.commit()
        
        # Create a failing quality report
        report = QualityReport(
            rule_id=rule.id,
            status=QualityStatus.FAILED,
            score=85.0,  # Below threshold
            records_checked=1000,
            records_failed=150,
            error_message="Completeness below threshold"
        )
        db_session.add(report)
        db_session.commit()
        
        # Process the report through alert manager
        await alert_manager.process_quality_report(report)
        
        # Check that alert was created
        alerts = db_session.query(QualityAlert).filter_by(quality_report_id=report.id).all()
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.HIGH
    
    def test_quality_metrics_calculation(self, db_session, quality_monitor):
        """Test quality metrics calculation"""
        
        # Create test pipeline and rules
        pipeline_id = "metrics_test_pipeline"
        
        rule1 = QualityRule(
            name="Test rule 1",
            rule_type=QualityRuleType.COMPLETENESS,
            target_pipeline_id=pipeline_id,
            severity=Severity.MEDIUM
        )
        rule2 = QualityRule(
            name="Test rule 2",
            rule_type=QualityRuleType.VALIDITY,
            target_pipeline_id=pipeline_id,
            severity=Severity.HIGH
        )
        
        db_session.add_all([rule1, rule2])
        db_session.commit()
        
        # Create test reports
        reports = [
            QualityReport(rule_id=rule1.id, status=QualityStatus.PASSED, score=95.0),
            QualityReport(rule_id=rule1.id, status=QualityStatus.FAILED, score=75.0),
            QualityReport(rule_id=rule2.id, status=QualityStatus.PASSED, score=98.0),
            QualityReport(rule_id=rule2.id, status=QualityStatus.PASSED, score=92.0)
        ]
        
        db_session.add_all(reports)
        db_session.commit()
        
        # Calculate metrics
        time_range = (datetime.utcnow() - timedelta(hours=1), datetime.utcnow())
        metrics = quality_monitor.get_quality_metrics(pipeline_id, time_range)
        
        assert metrics['total_checks'] == 4
        assert metrics['passed_checks'] == 3
        assert metrics['failed_checks'] == 1
        assert metrics['average_score'] == (95.0 + 75.0 + 98.0 + 92.0) / 4
    
    def test_profile_baseline_update(self, db_session, sample_data, data_profiler):
        """Test updating profile baselines with new data"""
        
        table_name = "baseline_test"
        column_name = "age"
        
        # Create initial profile
        initial_profile = data_profiler._profile_column(sample_data, table_name, column_name)
        db_session.add(initial_profile)
        db_session.commit()
        
        initial_mean = initial_profile.mean_value
        
        # Create new data with different distribution
        new_data = sample_data.copy()
        new_data['age'] = new_data['age'] + 10  # Age everyone by 10 years
        
        # Update profile
        updated_profile = data_profiler.update_profile_baseline(table_name, column_name, new_data)
        
        # Check that profile was updated (blended)
        assert updated_profile.mean_value != initial_mean
        assert updated_profile.mean_value > initial_mean  # Should be higher due to aging
    
    def test_recommended_rules_generation(self, db_session, sample_data, data_profiler):
        """Test generation of recommended quality rules"""
        
        table_name = "rules_test"
        pipeline_id = "rules_pipeline"
        
        profiles = data_profiler.create_comprehensive_profile(sample_data, table_name, pipeline_id)
        baseline = data_profiler.establish_quality_baseline(profiles, table_name, pipeline_id)
        
        rules = baseline.recommended_rules
        
        # Should have rules for different types
        rule_types = [rule['rule_type'] for rule in rules]
        assert QualityRuleType.COMPLETENESS.value in rule_types
        assert QualityRuleType.UNIQUENESS.value in rule_types  # For ID column
        assert QualityRuleType.STATISTICAL.value in rule_types  # For numeric columns
        
        # Check rule structure
        for rule in rules:
            assert 'name' in rule
            assert 'rule_type' in rule
            assert 'target_table' in rule
            assert 'target_column' in rule
            assert 'severity' in rule
            assert 'description' in rule
    
    def test_error_handling_and_recovery(self, db_session, quality_monitor):
        """Test error handling in quality monitoring"""
        
        # Create rule with invalid column
        invalid_rule = QualityRule(
            name="Invalid column rule",
            rule_type=QualityRuleType.COMPLETENESS,
            target_column="nonexistent_column",
            threshold_value=0.95
        )
        db_session.add(invalid_rule)
        db_session.commit()
        
        # Create sample data without the column
        test_data = pd.DataFrame({'valid_column': [1, 2, 3, 4, 5]})
        
        # Should handle error gracefully
        reports = quality_monitor.validate_data_batch(test_data, [invalid_rule])
        
        assert len(reports) == 1
        assert reports[0].status == QualityStatus.FAILED
        assert "not found" in reports[0].error_message.lower()
    
    def test_performance_with_large_dataset(self, db_session, quality_monitor, data_profiler):
        """Test performance with larger datasets"""
        
        # Create larger dataset
        large_data = pd.DataFrame({
            'id': range(10000),
            'value': np.random.normal(100, 20, 10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        # Profile the data (should use sampling)
        start_time = datetime.utcnow()
        profiles = data_profiler.create_comprehensive_profile(large_data, "large_test")
        profile_time = (datetime.utcnow() - start_time).total_seconds()
        
        assert len(profiles) == 3
        assert profile_time < 10.0  # Should complete within 10 seconds
        
        # Create and test quality rule
        rule = QualityRule(
            name="Large dataset rule",
            rule_type=QualityRuleType.COMPLETENESS,
            target_column="value",
            threshold_value=0.99
        )
        db_session.add(rule)
        db_session.commit()
        
        start_time = datetime.utcnow()
        reports = quality_monitor.validate_data_batch(large_data, [rule])
        validation_time = (datetime.utcnow() - start_time).total_seconds()
        
        assert len(reports) == 1
        assert validation_time < 5.0  # Should complete within 5 seconds

@pytest.mark.asyncio
class TestRealTimeMonitoring:
    """Test real-time monitoring capabilities"""
    
    async def test_monitoring_lifecycle(self, db_session, alert_manager):
        """Test complete monitoring lifecycle"""
        
        monitor = RealTimeQualityMonitor(db_session, alert_manager)
        
        # Start monitoring
        await monitor.start_monitoring(['test_pipeline'])
        
        status = monitor.get_monitoring_status()
        assert status['is_running'] == True
        assert len(status['active_tasks']) > 0
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        status = monitor.get_monitoring_status()
        assert status['is_running'] == False
    
    async def test_alert_escalation(self, db_session, alert_manager):
        """Test alert escalation for unacknowledged alerts"""
        
        # Create critical alert
        alert = QualityAlert(
            alert_type="test_alert",
            severity=Severity.CRITICAL,
            message="Test critical alert",
            created_at=datetime.utcnow() - timedelta(hours=2)  # Old alert
        )
        db_session.add(alert)
        db_session.commit()
        
        monitor = RealTimeQualityMonitor(db_session, alert_manager)
        
        # Process alerts (should escalate)
        await monitor._process_alerts()
        
        # Check for escalated alert
        escalated_alerts = db_session.query(QualityAlert).filter_by(alert_type="escalated_alert").all()
        assert len(escalated_alerts) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])