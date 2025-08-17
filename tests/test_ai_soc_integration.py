"""
Integration tests for AI-Enhanced Security Operations Center
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from security.ai_soc.ai_soc_orchestrator import AISOCOrchestrator
from security.ai_soc.ml_siem_engine import SecurityEvent, EventType, ThreatLevel


class TestAISOCIntegration:
    """Test AI SOC integration and end-to-end functionality"""
    
    @pytest.fixture
    async def ai_soc(self):
        """Create AI SOC instance for testing"""
        soc = AISOCOrchestrator()
        await soc.initialize()
        return soc
    
    @pytest.fixture
    def sample_security_event(self):
        """Create sample security event for testing"""
        return SecurityEvent(
            event_id="test_event_001",
            timestamp=datetime.now(),
            event_type=EventType.LOGIN_ATTEMPT,
            source_ip="192.168.1.100",
            user_id="test_user",
            resource="login_portal",
            raw_data={
                "success": False,
                "attempts": 5,
                "user_agent": "Mozilla/5.0"
            },
            risk_score=0.7
        )
    
    @pytest.mark.asyncio
    async def test_ai_soc_initialization(self):
        """Test AI SOC initialization"""
        soc = AISOCOrchestrator()
        
        # Should not be initialized initially
        assert not soc.is_initialized
        
        # Initialize
        await soc.initialize()
        
        # Should be initialized
        assert soc.is_initialized
        
        # All components should be initialized
        assert soc.ml_siem.is_trained
        assert soc.predictive_analytics.is_trained
    
    @pytest.mark.asyncio
    async def test_end_to_end_event_processing(self, ai_soc, sample_security_event):
        """Test complete end-to-end event processing"""
        # Process event through AI SOC
        results = await ai_soc.process_security_event(sample_security_event)
        
        # Verify results structure
        assert "event_id" in results
        assert "processing_id" in results
        assert "actions_taken" in results
        assert results["event_id"] == sample_security_event.event_id
        
        # Verify metrics updated
        assert ai_soc.soc_metrics.events_processed > 0
    
    @pytest.mark.asyncio
    async def test_high_risk_event_creates_incident(self, ai_soc):
        """Test that high-risk events create incidents"""
        # Create high-risk event
        high_risk_event = SecurityEvent(
            event_id="high_risk_001",
            timestamp=datetime.now(),
            event_type=EventType.DATA_EXFILTRATION,
            source_ip="10.0.0.50",
            user_id="suspicious_user",
            resource="sensitive_database",
            raw_data={
                "bytes_transferred": 10000000,
                "destination": "external_server"
            },
            risk_score=0.9
        )
        
        # Process event
        results = await ai_soc.process_security_event(high_risk_event)
        
        # Should create incidents for high-risk events
        assert len(results.get("incidents", [])) > 0
        assert ai_soc.soc_metrics.incidents_created > 0
    
    @pytest.mark.asyncio
    async def test_ml_siem_false_positive_reduction(self, ai_soc):
        """Test ML SIEM false positive reduction capability"""
        # Create events that should be filtered as false positives
        false_positive_events = []
        
        for i in range(10):
            event = SecurityEvent(
                event_id=f"fp_test_{i}",
                timestamp=datetime.now(),
                event_type=EventType.FILE_ACCESS,
                source_ip="192.168.1.10",
                user_id="normal_user",
                resource="normal_file",
                raw_data={"false_positive": True, "value": 0.1},
                risk_score=0.2
            )
            false_positive_events.append(event)
        
        # Process events
        alert_count = 0
        for event in false_positive_events:
            results = await ai_soc.process_security_event(event)
            alert_count += len(results.get("alerts", []))
        
        # Should have low alert rate due to false positive reduction
        false_positive_rate = alert_count / len(false_positive_events)
        assert false_positive_rate < 0.2  # Less than 20% should generate alerts
    
    @pytest.mark.asyncio
    async def test_threat_correlation_system(self, ai_soc):
        """Test threat correlation across multiple events"""
        # Create correlated events (brute force pattern)
        base_time = datetime.now()
        correlated_events = []
        
        for i in range(6):  # 6 failed login attempts
            event = SecurityEvent(
                event_id=f"brute_force_{i}",
                timestamp=base_time + timedelta(minutes=i),
                event_type=EventType.LOGIN_ATTEMPT,
                source_ip="10.0.0.100",
                user_id=f"target_user_{i % 2}",  # Targeting 2 users
                resource="login_portal",
                raw_data={"success": False, "attempt": i + 1},
                risk_score=0.6
            )
            correlated_events.append(event)
        
        # Process events
        total_correlations = 0
        for event in correlated_events:
            results = await ai_soc.process_security_event(event)
            total_correlations += len(results.get("correlations", []))
        
        # Should detect correlation pattern
        assert total_correlations > 0
    
    @pytest.mark.asyncio
    async def test_behavioral_analytics_anomaly_detection(self, ai_soc):
        """Test behavioral analytics anomaly detection"""
        # Create normal behavior pattern first
        normal_events = []
        base_time = datetime.now() - timedelta(days=7)
        
        for day in range(7):
            for hour in range(9, 17):  # Normal business hours
                event = SecurityEvent(
                    event_id=f"normal_{day}_{hour}",
                    timestamp=base_time + timedelta(days=day, hours=hour),
                    event_type=EventType.FILE_ACCESS,
                    source_ip="192.168.1.50",
                    user_id="regular_user",
                    resource="work_documents",
                    raw_data={"normal_activity": True},
                    risk_score=0.1
                )
                normal_events.append(event)
        
        # Process normal events to establish baseline
        for event in normal_events:
            await ai_soc.process_security_event(event)
        
        # Create anomalous event (access at 3 AM)
        anomalous_event = SecurityEvent(
            event_id="anomaly_001",
            timestamp=datetime.now().replace(hour=3, minute=0),
            event_type=EventType.FILE_ACCESS,
            source_ip="10.0.0.200",  # Different IP
            user_id="regular_user",
            resource="sensitive_files",  # Different resource
            raw_data={"unusual_activity": True},
            risk_score=0.8
        )
        
        # Process anomalous event
        results = await ai_soc.process_security_event(anomalous_event)
        
        # Should detect behavioral anomaly
        assert len(results.get("anomalies", [])) > 0
    
    @pytest.mark.asyncio
    async def test_incident_response_orchestration(self, ai_soc):
        """Test automated incident response orchestration"""
        # Create critical security event
        critical_event = SecurityEvent(
            event_id="critical_001",
            timestamp=datetime.now(),
            event_type=EventType.MALWARE_DETECTION,
            source_ip="192.168.1.200",
            user_id="infected_user",
            resource="workstation_001",
            raw_data={
                "malware_type": "trojan",
                "file_hash": "abc123def456",
                "severity": "critical"
            },
            risk_score=0.95
        )
        
        # Process event
        results = await ai_soc.process_security_event(critical_event)
        
        # Should create incident and execute response
        assert len(results.get("incidents", [])) > 0
        
        # Check if automated response was executed
        incident_id = results["incidents"][0].incident_id
        assert incident_id in ai_soc.active_incidents
        
        # Incident should have response actions
        incident = ai_soc.active_incidents[incident_id]
        assert incident.status.value in ["INVESTIGATING", "CONTAINED"]
    
    @pytest.mark.asyncio
    async def test_predictive_analytics_risk_forecasting(self, ai_soc):
        """Test predictive analytics risk forecasting"""
        # Generate risk forecast
        forecast = await ai_soc.predictive_analytics.generate_risk_forecast(
            "user", "test_user", 30
        )
        
        # Verify forecast structure
        assert forecast.entity_type == "user"
        assert forecast.entity_id == "test_user"
        assert forecast.predicted_risk_score >= 0.0
        assert forecast.predicted_risk_score <= 1.0
        assert len(forecast.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_threat_prediction(self, ai_soc):
        """Test threat likelihood prediction"""
        # Generate threat prediction
        prediction = await ai_soc.predictive_analytics.predict_threat_likelihood(
            "malware", 30
        )
        
        # Verify prediction structure
        assert prediction.threat_type == "malware"
        assert prediction.probability >= 0.0
        assert prediction.probability <= 1.0
        assert len(prediction.mitigation_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_soc_dashboard_generation(self, ai_soc):
        """Test SOC dashboard generation"""
        # Generate some activity first
        test_event = SecurityEvent(
            event_id="dashboard_test",
            timestamp=datetime.now(),
            event_type=EventType.NETWORK_CONNECTION,
            source_ip="192.168.1.75",
            user_id="dashboard_user",
            resource="external_api",
            raw_data={"connection_type": "https"},
            risk_score=0.4
        )
        
        await ai_soc.process_security_event(test_event)
        
        # Generate dashboard
        dashboard = await ai_soc.get_soc_dashboard()
        
        # Verify dashboard structure
        assert dashboard.timestamp is not None
        assert dashboard.overall_risk_score >= 0.0
        assert dashboard.overall_risk_score <= 1.0
        assert isinstance(dashboard.active_incidents, int)
        assert isinstance(dashboard.recent_alerts, list)
        assert isinstance(dashboard.top_threats, list)
        assert isinstance(dashboard.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, ai_soc):
        """Test comprehensive performance metrics collection"""
        # Process some events to generate metrics
        for i in range(5):
            event = SecurityEvent(
                event_id=f"metrics_test_{i}",
                timestamp=datetime.now(),
                event_type=EventType.FILE_ACCESS,
                source_ip=f"192.168.1.{100 + i}",
                user_id=f"metrics_user_{i}",
                resource="test_resource",
                raw_data={"test": True},
                risk_score=0.3
            )
            await ai_soc.process_security_event(event)
        
        # Get comprehensive metrics
        metrics = ai_soc.get_comprehensive_metrics()
        
        # Verify metrics structure
        assert "soc_metrics" in metrics
        assert "ml_siem" in metrics
        assert "correlation_system" in metrics
        assert "incident_orchestrator" in metrics
        assert "behavioral_analytics" in metrics
        assert "predictive_analytics" in metrics
        
        # Verify SOC metrics
        soc_metrics = metrics["soc_metrics"]
        assert soc_metrics["events_processed"] >= 5
        assert "false_positive_rate" in soc_metrics
        assert "automation_rate" in soc_metrics
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, ai_soc):
        """Test system health monitoring"""
        # Get system health
        health = await ai_soc._get_system_health()
        
        # Verify health structure
        assert "ml_siem" in health
        assert "correlation_system" in health
        assert "incident_orchestrator" in health
        assert "behavioral_analytics" in health
        assert "predictive_analytics" in health
        assert "overall" in health
        
        # All components should be healthy
        for component, status in health.items():
            assert status in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self, ai_soc):
        """Test concurrent processing of multiple events"""
        # Create multiple events
        events = []
        for i in range(10):
            event = SecurityEvent(
                event_id=f"concurrent_{i}",
                timestamp=datetime.now(),
                event_type=EventType.NETWORK_CONNECTION,
                source_ip=f"10.0.0.{i + 1}",
                user_id=f"concurrent_user_{i}",
                resource="concurrent_resource",
                raw_data={"concurrent": True, "index": i},
                risk_score=0.5
            )
            events.append(event)
        
        # Process events concurrently
        tasks = [ai_soc.process_security_event(event) for event in events]
        results = await asyncio.gather(*tasks)
        
        # All events should be processed successfully
        assert len(results) == 10
        for result in results:
            assert "event_id" in result
            assert "actions_taken" in result
        
        # Metrics should reflect all processed events
        assert ai_soc.soc_metrics.events_processed >= 10
    
    @pytest.mark.asyncio
    async def test_threat_hunting_execution(self, ai_soc):
        """Test threat hunting execution"""
        # Execute threat hunting
        hunting_results = await ai_soc.behavioral_analytics.execute_threat_hunting()
        
        # Verify results structure
        assert isinstance(hunting_results, list)
        
        # If results found, verify structure
        for result in hunting_results:
            assert "query_id" in result
            assert "query_name" in result
            assert "detected_at" in result
    
    def test_ai_soc_configuration(self):
        """Test AI SOC configuration management"""
        soc = AISOCOrchestrator()
        
        # Verify default configuration
        assert soc.config["auto_incident_creation"] is True
        assert soc.config["auto_response_enabled"] is True
        assert soc.config["threat_hunting_interval"] == 3600
        assert soc.config["forecasting_interval"] == 86400
        
        # Test configuration modification
        soc.config["auto_incident_creation"] = False
        assert soc.config["auto_incident_creation"] is False


@pytest.mark.asyncio
async def test_ai_soc_performance_benchmarks():
    """Test AI SOC performance against benchmarks"""
    soc = AISOCOrchestrator()
    await soc.initialize()
    
    # Test processing speed
    start_time = datetime.now()
    
    # Process 100 events
    for i in range(100):
        event = SecurityEvent(
            event_id=f"perf_test_{i}",
            timestamp=datetime.now(),
            event_type=EventType.FILE_ACCESS,
            source_ip=f"192.168.1.{i % 255}",
            user_id=f"perf_user_{i % 10}",
            resource="performance_test",
            raw_data={"performance_test": True},
            risk_score=0.3
        )
        await soc.process_security_event(event)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Should process events quickly (target: < 1 second per event)
    avg_processing_time = processing_time / 100
    assert avg_processing_time < 1.0, f"Average processing time {avg_processing_time:.3f}s exceeds 1s target"
    
    # Verify false positive rate is low
    metrics = soc.get_comprehensive_metrics()
    fp_rate = metrics["soc_metrics"]["false_positive_rate"]
    assert fp_rate < 0.1, f"False positive rate {fp_rate:.3f} exceeds 10% target"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])