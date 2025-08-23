"""
Tests for Security Monitoring and Analytics Dashboard
Comprehensive test suite for security monitoring functionality
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from security.monitoring.security_dashboard import (
    SecurityDashboard, SecurityAnalytics, SecurityMetric, 
    SecuritySeverity, ThreatIntelligence, ThreatType
)
from security.monitoring.threat_intelligence_correlator import (
    ThreatIntelligenceCorrelator, ThreatIndicator, IndicatorType, ThreatSource
)
from security.monitoring.predictive_analytics import (
    SecurityPredictiveAnalytics, SecurityPrediction, PredictionType, RiskLevel
)
from security.monitoring.forensic_analyzer import (
    ForensicAnalyzer, DigitalEvidence, EvidenceType, IncidentReconstruction
)
from security.monitoring.security_benchmarking import (
    SecurityBenchmarkingSystem, BenchmarkMetric, BenchmarkFramework, MaturityLevel
)

class TestSecurityDashboard:
    """Test cases for SecurityDashboard"""
    
    @pytest_asyncio.fixture
    async def dashboard(self):
        """Create dashboard instance for testing"""
        dashboard = SecurityDashboard("sqlite:///:memory:")
        await dashboard.initialize()
        return dashboard
        
    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, dashboard):
        """Test dashboard initialization"""
        assert dashboard is not None
        assert len(dashboard.threat_feeds) > 0
        
    @pytest.mark.asyncio
    async def test_collect_security_metrics(self, dashboard):
        """Test security metrics collection"""
        metrics = dashboard.collect_security_metrics()
        
        assert len(metrics) > 0
        assert all(isinstance(metric, SecurityMetric) for metric in metrics)
        assert all(hasattr(metric, 'metric_id') for metric in metrics)
        assert all(hasattr(metric, 'value') for metric in metrics)
        assert all(hasattr(metric, 'severity') for metric in metrics)
        
    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, dashboard):
        """Test executive summary generation"""
        summary = dashboard.generate_executive_summary()
        
        assert 'timestamp' in summary
        assert 'overall_risk_score' in summary
        assert 'risk_level' in summary
        assert 'threat_landscape' in summary
        assert 'compliance_status' in summary
        assert 'incident_summary' in summary
        assert 'key_metrics' in summary
        assert 'recommendations' in summary
        
        # Validate risk score is within expected range
        assert 0 <= summary['overall_risk_score'] <= 100
        
    @pytest.mark.asyncio
    async def test_risk_score_calculation(self, dashboard):
        """Test risk score calculation logic"""
        # Create test metrics with known severities
        test_metrics = [
            SecurityMetric(
                metric_id="test1",
                name="Test Metric 1",
                value=5.0,
                unit="count",
                timestamp=datetime.now(),
                severity=SecuritySeverity.HIGH,
                category="test",
                description="Test metric"
            ),
            SecurityMetric(
                metric_id="test2", 
                name="Test Metric 2",
                value=2.0,
                unit="count",
                timestamp=datetime.now(),
                severity=SecuritySeverity.MEDIUM,
                category="test",
                description="Test metric"
            )
        ]
        
        risk_score = dashboard._calculate_risk_score(test_metrics)
        assert 0 <= risk_score <= 100
        
    @pytest.mark.asyncio
    async def test_dashboard_data_structure(self, dashboard):
        """Test dashboard data structure completeness"""
        dashboard_data = dashboard.get_dashboard_data()
        
        required_keys = [
            'executive_summary',
            'real_time_metrics', 
            'threat_intelligence',
            'security_trends',
            'compliance_dashboard',
            'incident_management'
        ]
        
        for key in required_keys:
            assert key in dashboard_data

class TestThreatIntelligenceCorrelator:
    """Test cases for ThreatIntelligenceCorrelator"""
    
    @pytest_asyncio.fixture
    async def correlator(self):
        """Create correlator instance for testing"""
        correlator = ThreatIntelligenceCorrelator("sqlite:///:memory:")
        await correlator.initialize()
        return correlator
        
    @pytest.mark.asyncio
    async def test_correlator_initialization(self, correlator):
        """Test correlator initialization"""
        assert correlator is not None
        assert len(correlator.feed_configs) > 0
        
    @pytest.mark.asyncio
    async def test_threat_intelligence_collection(self, correlator):
        """Test threat intelligence collection"""
        collection_results = await correlator.collect_threat_intelligence()
        
        assert isinstance(collection_results, dict)
        assert len(collection_results) > 0
        
        # Check that each feed has a result
        for feed_name, result in collection_results.items():
            assert 'status' in result
            assert result['status'] in ['success', 'error']
            
    @pytest.mark.asyncio
    async def test_indicator_correlation(self, correlator):
        """Test indicator correlation functionality"""
        # First collect some intelligence
        await correlator.collect_threat_intelligence()
        
        # Test correlation with sample indicators
        test_indicators = [
            "192.168.1.100",
            "malicious-domain.com",
            "d41d8cd98f00b204e9800998ecf8427e"
        ]
        
        correlations = correlator.correlate_indicators(test_indicators)
        
        # Should return correlation results (may be empty if no matches)
        assert isinstance(correlations, list)
        
    @pytest.mark.asyncio
    async def test_threat_intelligence_summary(self, correlator):
        """Test threat intelligence summary generation"""
        summary = correlator.get_threat_intelligence_summary()
        
        required_keys = [
            'feed_status',
            'total_indicators',
            'indicator_breakdown',
            'recent_correlations',
            'threat_landscape',
            'feed_health'
        ]
        
        for key in required_keys:
            assert key in summary
            
    @pytest.mark.asyncio
    async def test_mitre_attack_collection(self, correlator):
        """Test MITRE ATT&CK data collection"""
        indicators = await correlator._collect_mitre_attack()
        
        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert all(isinstance(indicator, ThreatIndicator) for indicator in indicators)
        assert all(indicator.source == ThreatSource.MITRE_ATTACK for indicator in indicators)

class TestSecurityPredictiveAnalytics:
    """Test cases for SecurityPredictiveAnalytics"""
    
    @pytest_asyncio.fixture
    async def analytics(self):
        """Create analytics instance for testing"""
        analytics = SecurityPredictiveAnalytics("sqlite:///:memory:")
        await analytics.initialize()
        return analytics
        
    @pytest.mark.asyncio
    async def test_analytics_initialization(self, analytics):
        """Test analytics initialization"""
        assert analytics is not None
        assert not analytics.historical_data.empty
        assert len(analytics.feature_columns) > 0
        
    @pytest.mark.asyncio
    async def test_incident_prediction(self, analytics):
        """Test security incident prediction"""
        predictions = analytics.predict_security_incidents(time_horizon=7)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 7  # One prediction per day
        assert all(isinstance(pred, SecurityPrediction) for pred in predictions)
        assert all(pred.prediction_type == PredictionType.SECURITY_INCIDENT for pred in predictions)
        assert all(0 <= pred.predicted_probability <= 1 for pred in predictions)
        
    @pytest.mark.asyncio
    async def test_trend_analysis(self, analytics):
        """Test security trend analysis"""
        metrics = ["failed_logins", "network_anomalies", "malware_detections"]
        trends = analytics.analyze_security_trends(metrics, days_back=30)
        
        assert isinstance(trends, list)
        assert len(trends) <= len(metrics)  # May be fewer if metrics don't exist
        
        for trend in trends:
            assert hasattr(trend, 'trend_direction')
            assert trend.trend_direction in ['increasing', 'decreasing', 'stable']
            assert hasattr(trend, 'trend_strength')
            assert 0 <= trend.trend_strength <= 1
            
    @pytest.mark.asyncio
    async def test_risk_forecasting(self, analytics):
        """Test risk forecasting functionality"""
        categories = ["cyber_attacks", "data_breaches", "compliance_violations"]
        forecasts = analytics.generate_risk_forecast(categories, time_horizon=30)
        
        assert isinstance(forecasts, list)
        assert len(forecasts) == len(categories)
        
        for forecast in forecasts:
            assert hasattr(forecast, 'risk_category')
            assert forecast.risk_category in categories
            assert hasattr(forecast, 'current_risk_score')
            assert 0 <= forecast.current_risk_score <= 100
            assert len(forecast.forecasted_risk_scores) == 30
            
    @pytest.mark.asyncio
    async def test_analytics_summary(self, analytics):
        """Test analytics summary generation"""
        summary = analytics.get_analytics_summary()
        
        required_keys = [
            'model_status',
            'data_summary',
            'recent_predictions',
            'trend_analysis',
            'risk_forecasts'
        ]
        
        for key in required_keys:
            assert key in summary

class TestForensicAnalyzer:
    """Test cases for ForensicAnalyzer"""
    
    @pytest_asyncio.fixture
    async def analyzer(self):
        """Create analyzer instance for testing"""
        analyzer = ForensicAnalyzer("test_evidence_store")
        await analyzer.initialize()
        return analyzer
        
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.evidence_store_path.exists()
        
    @pytest.mark.asyncio
    async def test_evidence_collection(self, analyzer):
        """Test evidence collection functionality"""
        incident_id = "test_incident_001"
        source_systems = ["web_server_01", "database_01", "workstation_01"]
        
        evidence_items = await analyzer.collect_evidence(incident_id, source_systems)
        
        assert isinstance(evidence_items, list)
        assert len(evidence_items) > 0
        assert all(isinstance(item, DigitalEvidence) for item in evidence_items)
        assert all(item.integrity_verified for item in evidence_items)
        
    @pytest.mark.asyncio
    async def test_evidence_analysis(self, analyzer):
        """Test evidence analysis functionality"""
        # Create sample evidence
        sample_evidence = [
            DigitalEvidence(
                evidence_id="test_evidence_001",
                evidence_type=EvidenceType.LOG_FILE,
                source_system="test_system",
                collection_timestamp=datetime.now(),
                file_path="/var/log/test.log",
                file_hash="test_hash",
                file_size=1024,
                metadata={},
                chain_of_custody=[],
                integrity_verified=True
            )
        ]
        
        analysis_results = await analyzer.analyze_evidence(sample_evidence)
        
        required_keys = [
            'evidence_summary',
            'timeline_analysis',
            'artifact_analysis',
            'correlation_analysis',
            'ioc_extraction'
        ]
        
        for key in required_keys:
            assert key in analysis_results
            
    @pytest.mark.asyncio
    async def test_incident_reconstruction(self, analyzer):
        """Test incident reconstruction functionality"""
        incident_id = "test_incident_002"
        evidence_items = []  # Empty for testing
        
        reconstruction = await analyzer.reconstruct_incident(incident_id, evidence_items)
        
        assert isinstance(reconstruction, IncidentReconstruction)
        assert reconstruction.incident_id == incident_id
        assert hasattr(reconstruction, 'attack_vector')
        assert hasattr(reconstruction, 'confidence_level')
        assert 0 <= reconstruction.confidence_level <= 1

class TestSecurityBenchmarkingSystem:
    """Test cases for SecurityBenchmarkingSystem"""
    
    @pytest_asyncio.fixture
    async def benchmarking(self):
        """Create benchmarking instance for testing"""
        benchmarking = SecurityBenchmarkingSystem()
        await benchmarking.initialize()
        return benchmarking
        
    @pytest.mark.asyncio
    async def test_benchmarking_initialization(self, benchmarking):
        """Test benchmarking system initialization"""
        assert benchmarking is not None
        assert len(benchmarking.benchmark_data) > 0
        assert len(benchmarking.industry_standards) > 0
        
    @pytest.mark.asyncio
    async def test_security_posture_assessment(self, benchmarking):
        """Test security posture assessment"""
        test_metrics = {
            "mean_time_to_detection": 48.0,
            "mean_time_to_response": 72.0,
            "vulnerability_remediation_time": 45.0
        }
        
        benchmark_metrics = await benchmarking.assess_security_posture(test_metrics)
        
        assert isinstance(benchmark_metrics, list)
        assert len(benchmark_metrics) > 0
        assert all(isinstance(metric, BenchmarkMetric) for metric in benchmark_metrics)
        
        for metric in benchmark_metrics:
            assert hasattr(metric, 'percentile_rank')
            assert 0 <= metric.percentile_rank <= 100
            assert hasattr(metric, 'maturity_level')
            assert isinstance(metric.maturity_level, MaturityLevel)
            
    @pytest.mark.asyncio
    async def test_compliance_assessment(self, benchmarking):
        """Test compliance assessment functionality"""
        framework = BenchmarkFramework.NIST_CSF
        test_controls = {
            "identify": 0.85,
            "protect": 0.78,
            "detect": 0.82,
            "respond": 0.75,
            "recover": 0.70
        }
        
        assessment = await benchmarking.perform_compliance_assessment(framework, test_controls)
        
        assert hasattr(assessment, 'framework')
        assert assessment.framework == framework
        assert hasattr(assessment, 'overall_score')
        assert 0 <= assessment.overall_score <= 1
        assert hasattr(assessment, 'compliance_percentage')
        assert 0 <= assessment.compliance_percentage <= 100
        
    @pytest.mark.asyncio
    async def test_peer_comparison(self, benchmarking):
        """Test peer comparison functionality"""
        industry_sector = "technology"
        organization_size = "medium"
        test_metrics = {
            "security_maturity": 3.5,
            "compliance_score": 85.0
        }
        
        comparison = await benchmarking.compare_with_peers(
            industry_sector, 
            organization_size, 
            test_metrics
        )
        
        assert hasattr(comparison, 'industry_sector')
        assert comparison.industry_sector == industry_sector
        assert hasattr(comparison, 'organization_size')
        assert comparison.organization_size == organization_size
        assert hasattr(comparison, 'competitive_position')
        
    @pytest.mark.asyncio
    async def test_improvement_roadmap(self, benchmarking):
        """Test improvement roadmap generation"""
        # Create sample data
        sample_metrics = []
        sample_assessments = []
        
        roadmap = await benchmarking.generate_improvement_roadmap(
            sample_metrics, 
            sample_assessments
        )
        
        required_keys = [
            'roadmap_id',
            'created_date',
            'total_improvements',
            'estimated_duration',
            'phases',
            'success_metrics',
            'resource_requirements'
        ]
        
        for key in required_keys:
            assert key in roadmap

class TestSecurityMonitoringIntegration:
    """Integration tests for security monitoring system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        # Initialize components
        dashboard = SecurityDashboard("sqlite:///:memory:")
        await dashboard.initialize()
        
        # Collect metrics
        metrics = dashboard.collect_security_metrics()
        assert len(metrics) > 0
        
        # Generate executive summary
        summary = dashboard.generate_executive_summary()
        assert 'overall_risk_score' in summary
        
        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()
        assert 'executive_summary' in dashboard_data
        
    @pytest.mark.asyncio
    async def test_threat_intelligence_integration(self):
        """Test threat intelligence integration"""
        correlator = ThreatIntelligenceCorrelator("sqlite:///:memory:")
        await correlator.initialize()
        
        # Collect intelligence
        collection_results = await correlator.collect_threat_intelligence()
        assert isinstance(collection_results, dict)
        
        # Test correlation
        test_indicators = ["192.168.1.100", "malicious-domain.com"]
        correlations = correlator.correlate_indicators(test_indicators)
        assert isinstance(correlations, list)
        
    @pytest.mark.asyncio
    async def test_predictive_analytics_integration(self):
        """Test predictive analytics integration"""
        analytics = SecurityPredictiveAnalytics("sqlite:///:memory:")
        await analytics.initialize()
        
        # Generate predictions
        predictions = analytics.predict_security_incidents(7)
        assert len(predictions) == 7
        
        # Analyze trends
        trends = analytics.analyze_security_trends(["failed_logins"], 30)
        assert isinstance(trends, list)
        
    @pytest.mark.asyncio
    async def test_benchmarking_integration(self):
        """Test benchmarking system integration"""
        benchmarking = SecurityBenchmarkingSystem()
        await benchmarking.initialize()
        
        # Assess posture
        test_metrics = {"mean_time_to_detection": 48.0}
        benchmark_metrics = await benchmarking.assess_security_posture(test_metrics)
        assert isinstance(benchmark_metrics, list)
        
        # Perform compliance assessment
        assessment = await benchmarking.perform_compliance_assessment(
            BenchmarkFramework.NIST_CSF,
            {"identify": 0.85}
        )
        assert hasattr(assessment, 'overall_score')

if __name__ == "__main__":
    pytest.main([__file__])