"""Tests for usage tracking and compliance reporting system."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import uuid

from ai_data_readiness.engines.usage_tracker import UsageTracker, UsageTrackerError
from ai_data_readiness.engines.compliance_reporter import ComplianceReporter, ComplianceReporterError
from ai_data_readiness.models.governance_models import (
    AuditEvent, AuditEventType, UsageMetrics, ComplianceReport,
    DataClassification, PolicyType
)


class TestUsageTracker:
    """Test cases for UsageTracker."""
    
    @pytest.fixture
    def usage_tracker(self):
        """Create UsageTracker instance."""
        return UsageTracker()
    
    @pytest.fixture
    def mock_audit_events(self):
        """Create mock audit events."""
        events = []
        base_time = datetime.utcnow() - timedelta(days=7)
        
        for i in range(100):
            event = Mock()
            event.id = str(uuid.uuid4())
            event.event_type = AuditEventType.DATA_ACCESS.value
            event.user_id = f"user_{i % 10}"  # 10 different users
            event.resource_id = f"resource_{i % 5}"  # 5 different resources
            event.resource_type = "dataset"
            event.action = "read"
            event.timestamp = base_time + timedelta(hours=i)
            event.success = True
            event.details = {}
            events.append(event)
        
        return events
    
    def test_track_data_access(self, usage_tracker):
        """Test tracking data access."""
        with patch('ai_data_readiness.engines.usage_tracker.get_db_session') as mock_session:
            mock_session.return_value.__enter__.return_value = Mock()
            
            result = usage_tracker.track_data_access(
                user_id="test_user",
                resource_id="test_resource",
                resource_type="dataset",
                action="read"
            )
            
            assert result is True
    
    @patch('ai_data_readiness.engines.usage_tracker.get_db_session')
    def test_get_usage_analytics(self, mock_session, usage_tracker, mock_audit_events):
        """Test getting usage analytics."""
        # Set up the mock chain properly
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        # Mock the query chain
        mock_query = Mock()
        mock_session_instance.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = mock_audit_events
        
        analytics = usage_tracker.get_usage_analytics(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            aggregation_level="daily"
        )
        
        assert 'summary' in analytics
        assert 'trends' in analytics
        assert 'user_analytics' in analytics
        assert 'resource_analytics' in analytics
        assert 'access_patterns' in analytics
        assert 'performance_metrics' in analytics
        
        # Check summary statistics
        assert analytics['summary']['total_events'] == 100
        assert analytics['summary']['unique_users'] == 10
        assert analytics['summary']['unique_resources'] == 5
    
    @patch('ai_data_readiness.engines.usage_tracker.get_db_session')
    def test_get_user_activity_report(self, mock_session, usage_tracker, mock_audit_events):
        """Test getting user activity report."""
        # Mock user query
        mock_user = Mock()
        mock_user.id = "test_user"
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.department = "IT"
        mock_user.role = "analyst"
        
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        # Mock user query
        mock_session_instance.query.return_value.filter.return_value.first.return_value = mock_user
        
        # Mock events query
        user_events = [event for event in mock_audit_events if event.user_id == "user_1"]
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = user_events
        
        report = usage_tracker.get_user_activity_report(
            user_id="test_user",
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            include_details=True
        )
        
        assert 'user_info' in report
        assert 'activity_summary' in report
        assert 'resource_usage' in report
        assert 'behavioral_patterns' in report
        assert 'compliance_metrics' in report
        assert 'risk_indicators' in report
        assert 'detailed_events' in report
        
        assert report['user_info']['username'] == "testuser"
    
    @patch('ai_data_readiness.engines.usage_tracker.get_db_session')
    def test_get_resource_usage_report(self, mock_session, usage_tracker, mock_audit_events):
        """Test getting resource usage report."""
        # Mock resource query
        mock_resource = Mock()
        mock_resource.dataset_id = "test_resource"
        mock_resource.name = "Test Dataset"
        mock_resource.classification = DataClassification.INTERNAL
        mock_resource.owner = "owner_user"
        
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        # Mock resource query
        mock_session_instance.query.return_value.filter.return_value.first.return_value = mock_resource
        
        # Mock events query
        resource_events = [event for event in mock_audit_events if event.resource_id == "resource_1"]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = resource_events
        
        report = usage_tracker.get_resource_usage_report(
            resource_id="test_resource",
            resource_type="dataset",
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            include_user_breakdown=True
        )
        
        assert 'resource_info' in report
        assert 'usage_summary' in report
        assert 'access_patterns' in report
        assert 'performance_metrics' in report
        assert 'security_metrics' in report
        assert 'user_breakdown' in report
        
        assert report['resource_info']['name'] == "Test Dataset"
    
    @patch('ai_data_readiness.engines.usage_tracker.get_db_session')
    def test_get_system_usage_overview(self, mock_session, usage_tracker, mock_audit_events):
        """Test getting system usage overview."""
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        # Mock queries
        mock_session_instance.query.return_value.all.return_value = mock_audit_events
        mock_session_instance.query.return_value.filter.return_value.count.return_value = 50  # total users
        mock_session_instance.query.return_value.count.return_value = 25  # total resources
        
        overview = usage_tracker.get_system_usage_overview(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )
        
        assert 'system_metrics' in overview
        assert 'activity_summary' in overview
        assert 'top_users' in overview
        assert 'top_resources' in overview
        assert 'security_overview' in overview
        assert 'compliance_overview' in overview
        
        assert overview['system_metrics']['total_users'] == 50
        assert overview['system_metrics']['total_resources'] == 25
    
    @patch('ai_data_readiness.engines.usage_tracker.get_db_session')
    def test_generate_usage_trends(self, mock_session, usage_tracker, mock_audit_events):
        """Test generating usage trends."""
        mock_session.return_value.__enter__.return_value.query.return_value.filter.return_value.all.return_value = mock_audit_events
        
        trends = usage_tracker.generate_usage_trends(
            metric_type="access_count",
            time_period="7d",
            granularity="daily"
        )
        
        assert 'metric_type' in trends
        assert 'time_period' in trends
        assert 'granularity' in trends
        assert 'trends' in trends
        assert 'summary' in trends
        
        assert trends['metric_type'] == "access_count"
        assert trends['time_period'] == "7d"
        assert len(trends['trends']) > 0
    
    def test_calculate_usage_summary(self, usage_tracker, mock_audit_events):
        """Test calculating usage summary."""
        summary = usage_tracker._calculate_usage_summary(mock_audit_events)
        
        assert summary['total_events'] == 100
        assert summary['unique_users'] == 10
        assert summary['unique_resources'] == 5
        assert summary['success_rate'] == 1.0  # All events are successful
    
    def test_calculate_access_patterns(self, usage_tracker, mock_audit_events):
        """Test calculating access patterns."""
        patterns = usage_tracker._calculate_access_patterns(mock_audit_events)
        
        assert 'hourly_distribution' in patterns
        assert 'daily_distribution' in patterns
        assert len(patterns['hourly_distribution']) > 0
        assert len(patterns['daily_distribution']) > 0
    
    def test_error_handling(self, usage_tracker):
        """Test error handling in usage tracker."""
        with patch('ai_data_readiness.engines.usage_tracker.get_db_session') as mock_session:
            mock_session.side_effect = Exception("Database error")
            
            with pytest.raises(UsageTrackerError):
                usage_tracker.get_usage_analytics()


class TestComplianceReporter:
    """Test cases for ComplianceReporter."""
    
    @pytest.fixture
    def compliance_reporter(self):
        """Create ComplianceReporter instance."""
        return ComplianceReporter()
    
    @pytest.fixture
    def mock_compliance_data(self):
        """Create mock compliance data."""
        return {
            'datasets': [
                Mock(
                    dataset_id="dataset_1",
                    name="Test Dataset 1",
                    classification=DataClassification.CONFIDENTIAL,
                    business_glossary_terms=["finance", "customer"],
                    description="Test dataset description"
                ),
                Mock(
                    dataset_id="dataset_2",
                    name="Test Dataset 2",
                    classification=None,
                    business_glossary_terms=[],
                    description=""
                )
            ],
            'audit_events': [
                Mock(
                    action="consent_granted",
                    timestamp=datetime.utcnow(),
                    event_type=AuditEventType.USER_ACTION.value
                ),
                Mock(
                    action="data_deleted",
                    timestamp=datetime.utcnow(),
                    event_type=AuditEventType.DATA_MODIFICATION.value
                )
            ],
            'access_controls': [
                Mock(
                    resource_id="dataset_1",
                    is_active=True
                )
            ]
        }
    
    def test_supported_frameworks(self, compliance_reporter):
        """Test supported compliance frameworks."""
        frameworks = compliance_reporter.compliance_frameworks
        
        assert 'GDPR' in frameworks
        assert 'CCPA' in frameworks
        assert 'SOX' in frameworks
        assert 'HIPAA' in frameworks
        
        # Check GDPR requirements
        gdpr_requirements = frameworks['GDPR']['requirements']
        assert 'data_minimization' in gdpr_requirements
        assert 'consent_management' in gdpr_requirements
        assert 'right_to_erasure' in gdpr_requirements
    
    @patch('ai_data_readiness.engines.compliance_reporter.get_db_session')
    def test_generate_compliance_report(self, mock_session, compliance_reporter, mock_compliance_data):
        """Test generating compliance report."""
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        # Mock database queries
        mock_session_instance.query.return_value.all.return_value = mock_compliance_data['datasets']
        mock_session_instance.query.return_value.filter.return_value.count.return_value = 1
        mock_session_instance.add = Mock()
        mock_session_instance.commit = Mock()
        mock_session_instance.refresh = Mock()
        
        # Mock the saved report
        mock_report = Mock()
        mock_report.id = str(uuid.uuid4())
        mock_session_instance.refresh.side_effect = lambda x: setattr(x, 'id', mock_report.id)
        
        report = compliance_reporter.generate_compliance_report(
            framework="GDPR",
            scope=["dataset_1", "dataset_2"],
            generated_by="test_user"
        )
        
        assert isinstance(report, ComplianceReport)
        assert report.report_type == "GDPR"
        assert report.generated_by == "test_user"
        assert report.compliance_score >= 0
        assert report.compliance_score <= 100
    
    @patch('ai_data_readiness.engines.compliance_reporter.get_db_session')
    def test_get_compliance_dashboard(self, mock_session, compliance_reporter):
        """Test getting compliance dashboard."""
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        # Mock latest reports
        mock_report = Mock()
        mock_report.compliance_score = 85.0
        mock_report.generated_at = datetime.utcnow()
        mock_report.violations = []
        
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_report
        mock_session_instance.query.return_value.filter.return_value.all.return_value = [mock_report]
        
        dashboard = compliance_reporter.get_compliance_dashboard(
            frameworks=["GDPR", "CCPA"],
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )
        
        assert 'frameworks' in dashboard
        assert 'overall_metrics' in dashboard
        assert 'trending' in dashboard
        assert 'alerts' in dashboard
        
        assert 'GDPR' in dashboard['frameworks']
        assert 'CCPA' in dashboard['frameworks']
    
    @patch('ai_data_readiness.engines.compliance_reporter.get_db_session')
    def test_validate_data_classification_compliance(self, mock_session, compliance_reporter):
        """Test validating data classification compliance."""
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.dataset_id = "test_dataset"
        mock_dataset.classification = DataClassification.CONFIDENTIAL
        mock_dataset.description = "Test dataset"
        mock_dataset.retention_policy = "7_years"
        
        mock_session_instance.query.return_value.filter.return_value.first.return_value = mock_dataset
        mock_session_instance.query.return_value.filter.return_value.count.return_value = 1
        
        validation_result = compliance_reporter.validate_data_classification_compliance(
            dataset_id="test_dataset"
        )
        
        assert 'dataset_id' in validation_result
        assert 'classification' in validation_result
        assert 'compliance_status' in validation_result
        assert 'issues' in validation_result
        assert 'recommendations' in validation_result
        
        assert validation_result['dataset_id'] == "test_dataset"
        assert validation_result['classification'] == "confidential"
    
    @patch('ai_data_readiness.engines.compliance_reporter.get_db_session')
    def test_audit_access_compliance(self, mock_session, compliance_reporter, mock_compliance_data):
        """Test auditing access compliance."""
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        # Mock access events
        access_events = []
        for i in range(50):
            event = Mock()
            event.user_id = f"user_{i % 5}"
            event.resource_id = f"resource_{i % 3}"
            event.resource_type = "dataset"
            event.action = "read"
            event.timestamp = datetime.utcnow() - timedelta(hours=i)
            event.success = i % 10 != 0  # 10% failure rate
            access_events.append(event)
        
        mock_session_instance.query.return_value.filter.return_value.all.return_value = access_events
        
        audit_result = compliance_reporter.audit_access_compliance(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )
        
        assert 'total_access_events' in audit_result
        assert 'compliance_violations' in audit_result
        assert 'access_patterns' in audit_result
        assert 'risk_indicators' in audit_result
        assert 'recommendations' in audit_result
        
        assert audit_result['total_access_events'] == 50
    
    @patch('ai_data_readiness.engines.compliance_reporter.get_db_session')
    def test_generate_audit_trail_report(self, mock_session, compliance_reporter, mock_compliance_data):
        """Test generating audit trail report."""
        mock_session_instance = Mock()
        mock_session.return_value.__enter__.return_value = mock_session_instance
        
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_compliance_data['audit_events']
        
        report = compliance_reporter.generate_audit_trail_report(
            resource_id="test_resource",
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )
        
        assert 'filters' in report
        assert 'summary' in report
        assert 'timeline' in report
        assert 'detailed_events' in report
        assert 'compliance_notes' in report
        
        assert report['filters']['resource_id'] == "test_resource"
    
    def test_assess_gdpr_requirement(self, compliance_reporter):
        """Test assessing GDPR requirements."""
        with patch('ai_data_readiness.engines.compliance_reporter.get_db_session') as mock_session:
            mock_session_instance = Mock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock datasets without business terms
            mock_datasets = [
                Mock(dataset_id="dataset_1", business_glossary_terms=[]),
                Mock(dataset_id="dataset_2", business_glossary_terms=["finance"])
            ]
            mock_session_instance.query.return_value.all.return_value = mock_datasets
            
            score, violations, recommendations = compliance_reporter._assess_gdpr_requirement(
                mock_session_instance,
                "data_minimization",
                None,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow()
            )
            
            assert score <= 100
            assert len(violations) >= 0
            assert isinstance(recommendations, list)
    
    def test_assess_ccpa_requirement(self, compliance_reporter):
        """Test assessing CCPA requirements."""
        with patch('ai_data_readiness.engines.compliance_reporter.get_db_session') as mock_session:
            mock_session_instance = Mock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock no consumer rights events
            mock_session_instance.query.return_value.filter.return_value.count.return_value = 0
            
            score, violations, recommendations = compliance_reporter._assess_ccpa_requirement(
                mock_session_instance,
                "consumer_rights",
                None,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow()
            )
            
            assert score <= 100
            assert len(violations) > 0  # Should have violations for no consumer rights events
    
    def test_assess_sox_requirement(self, compliance_reporter):
        """Test assessing SOX requirements."""
        with patch('ai_data_readiness.engines.compliance_reporter.get_db_session') as mock_session:
            mock_session_instance = Mock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock audit events
            mock_session_instance.query.return_value.filter.return_value.count.return_value = 100
            
            score, violations, recommendations = compliance_reporter._assess_sox_requirement(
                mock_session_instance,
                "audit_trails",
                None,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow()
            )
            
            assert score == 100  # Should be compliant with audit events present
            assert len(violations) == 0
    
    def test_assess_hipaa_requirement(self, compliance_reporter):
        """Test assessing HIPAA requirements."""
        with patch('ai_data_readiness.engines.compliance_reporter.get_db_session') as mock_session:
            mock_session_instance = Mock()
            mock_session.return_value.__enter__.return_value = mock_session_instance
            
            # Mock PHI datasets
            phi_dataset = Mock()
            phi_dataset.dataset_id = "phi_dataset"
            phi_dataset.tags = ["PHI"]
            phi_dataset.classification = DataClassification.INTERNAL  # Should be RESTRICTED
            
            mock_session_instance.query.return_value.filter.return_value.all.return_value = [phi_dataset]
            
            score, violations, recommendations = compliance_reporter._assess_hipaa_requirement(
                mock_session_instance,
                "phi_protection",
                None,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow()
            )
            
            assert score < 100  # Should have violations for improper PHI classification
            assert len(violations) > 0
    
    def test_analyze_access_patterns(self, compliance_reporter):
        """Test analyzing access patterns."""
        # Create mock events with various patterns
        events = []
        for i in range(24):  # 24 hours of events
            event = Mock()
            event.timestamp = datetime.utcnow().replace(hour=i, minute=0, second=0, microsecond=0)
            event.user_id = f"user_{i % 3}"
            event.resource_id = f"resource_{i % 2}"
            event.resource_type = "dataset"
            event.action = "read"
            events.append(event)
        
        patterns = compliance_reporter._analyze_access_patterns(events)
        
        assert 'hourly_distribution' in patterns
        assert 'top_users' in patterns
        assert 'top_resources' in patterns
        assert 'unusual_access_count' in patterns
        assert 'unusual_access_details' in patterns
        
        # Should have 24 different hours
        assert len(patterns['hourly_distribution']) == 24
        # Should have unusual access times (before 6 AM and after 10 PM)
        assert patterns['unusual_access_count'] > 0
    
    def test_identify_access_violations(self, compliance_reporter):
        """Test identifying access violations."""
        with patch('ai_data_readiness.engines.compliance_reporter.get_db_session') as mock_session:
            mock_session_instance = Mock()
            
            # Create events with high failure rate
            events = []
            for i in range(100):
                event = Mock()
                event.user_id = f"user_{i % 5}"
                event.success = i % 5 != 0  # 20% failure rate
                events.append(event)
            
            violations = compliance_reporter._identify_access_violations(
                mock_session_instance, events
            )
            
            assert len(violations) > 0
            # Should identify high failure rate
            failure_violations = [v for v in violations if v['type'] == 'high_failure_rate']
            assert len(failure_violations) > 0
    
    def test_error_handling(self, compliance_reporter):
        """Test error handling in compliance reporter."""
        with patch('ai_data_readiness.engines.compliance_reporter.get_db_session') as mock_session:
            mock_session.side_effect = Exception("Database error")
            
            with pytest.raises(ComplianceReporterError):
                compliance_reporter.generate_compliance_report(
                    framework="GDPR",
                    generated_by="test_user"
                )
    
    def test_unsupported_framework(self, compliance_reporter):
        """Test handling unsupported compliance framework."""
        with pytest.raises(ComplianceReporterError, match="Unsupported compliance framework"):
            compliance_reporter.generate_compliance_report(
                framework="INVALID_FRAMEWORK",
                generated_by="test_user"
            )


class TestIntegration:
    """Integration tests for usage tracking and compliance reporting."""
    
    @pytest.fixture
    def usage_tracker(self):
        return UsageTracker()
    
    @pytest.fixture
    def compliance_reporter(self):
        return ComplianceReporter()
    
    @patch('ai_data_readiness.engines.usage_tracker.get_db_session')
    @patch('ai_data_readiness.engines.compliance_reporter.get_db_session')
    def test_usage_tracking_compliance_integration(self, mock_compliance_session, mock_usage_session, usage_tracker, compliance_reporter):
        """Test integration between usage tracking and compliance reporting."""
        # Mock usage tracking data
        mock_usage_session_instance = Mock()
        mock_usage_session.return_value.__enter__.return_value = mock_usage_session_instance
        
        # Mock compliance data
        mock_compliance_session_instance = Mock()
        mock_compliance_session.return_value.__enter__.return_value = mock_compliance_session_instance
        
        # Create mock audit events
        audit_events = []
        for i in range(50):
            event = Mock()
            event.id = str(uuid.uuid4())
            event.event_type = AuditEventType.DATA_ACCESS.value
            event.user_id = f"user_{i % 5}"
            event.resource_id = f"resource_{i % 3}"
            event.resource_type = "dataset"
            event.action = "read"
            event.timestamp = datetime.utcnow() - timedelta(hours=i)
            event.success = True
            event.details = {}
            audit_events.append(event)
        
        mock_usage_session_instance.query.return_value.filter.return_value.all.return_value = audit_events
        mock_compliance_session_instance.query.return_value.filter.return_value.all.return_value = audit_events
        
        # Mock other required data for compliance
        mock_compliance_session_instance.query.return_value.all.return_value = []
        mock_compliance_session_instance.query.return_value.filter.return_value.count.return_value = 0
        mock_compliance_session_instance.add = Mock()
        mock_compliance_session_instance.commit = Mock()
        mock_compliance_session_instance.refresh = Mock()
        
        # Test usage analytics
        analytics = usage_tracker.get_usage_analytics(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )
        
        # Test compliance audit using the same data
        audit_result = compliance_reporter.audit_access_compliance(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )
        
        # Verify both systems can work with the same audit data
        assert analytics['summary']['total_events'] == 50
        assert audit_result['total_access_events'] == 50
        
        # Both should identify the same users and resources
        assert analytics['summary']['unique_users'] == 5
        assert analytics['summary']['unique_resources'] == 3


if __name__ == "__main__":
    pytest.main([__file__])