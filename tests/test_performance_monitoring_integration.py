"""
Integration Tests for Performance Monitoring System

This module tests the complete integration of performance monitoring
components including API routes, engine, and models.
"""

import pytest
import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from scrollintel.engines.performance_monitoring_engine import PerformanceMonitoringEngine
from scrollintel.models.performance_monitoring_models import InterventionType, SupportType


@pytest.fixture
def client():
    """Create test client"""
    from scrollintel.api.routes.performance_monitoring_routes import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_crisis_id():
    """Sample crisis ID for testing"""
    return "integration_test_crisis"


@pytest.fixture
def sample_team_members():
    """Sample team members for testing"""
    return ["member_001", "member_002", "member_003"]


class TestPerformanceMonitoringIntegration:
    """Integration test cases for Performance Monitoring System"""
    
    def test_track_team_performance_endpoint(self, client, sample_crisis_id, sample_team_members):
        """Test team performance tracking endpoint"""
        response = client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=sample_team_members
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["crisis_id"] == sample_crisis_id
        assert data["team_id"] == f"crisis_team_{sample_crisis_id}"
        assert len(data["member_performances"]) == len(sample_team_members)
        assert "overall_performance_score" in data
        assert "team_efficiency" in data
        assert "collaboration_index" in data
        assert "stress_level_avg" in data
        assert "last_updated" in data
    
    def test_track_team_performance_empty_members(self, client, sample_crisis_id):
        """Test tracking performance with empty team members"""
        response = client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=[]
        )
        
        assert response.status_code == 400
        assert "Team members list cannot be empty" in response.json()["detail"]
    
    def test_get_team_performance_endpoint(self, client, sample_crisis_id, sample_team_members):
        """Test get team performance endpoint"""
        # First track performance
        client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=sample_team_members
        )
        
        # Then get performance
        response = client.get(f"/api/v1/performance-monitoring/performance/{sample_crisis_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["crisis_id"] == sample_crisis_id
    
    def test_get_team_performance_not_found(self, client):
        """Test get team performance for non-existent crisis"""
        response = client.get("/api/v1/performance-monitoring/performance/nonexistent")
        
        assert response.status_code == 404
        assert "No performance data found" in response.json()["detail"]
    
    def test_identify_performance_issues_endpoint(self, client, sample_crisis_id, sample_team_members):
        """Test performance issues identification endpoint"""
        # First track performance
        client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=sample_team_members
        )
        
        # Then identify issues
        response = client.get(f"/api/v1/performance-monitoring/issues/{sample_crisis_id}")
        
        assert response.status_code == 200
        issues = response.json()
        assert isinstance(issues, list)
    
    def test_implement_intervention_endpoint(self, client, sample_crisis_id):
        """Test intervention implementation endpoint"""
        member_id = "test_member"
        intervention_type = InterventionType.COACHING.value
        
        response = client.post(
            f"/api/v1/performance-monitoring/intervention/{sample_crisis_id}/{member_id}",
            params={"intervention_type": intervention_type}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["member_id"] == member_id
        assert data["crisis_id"] == sample_crisis_id
        assert data["intervention_type"] == intervention_type
        assert "description" in data
        assert "expected_outcome" in data
    
    def test_provide_support_endpoint(self, client, sample_crisis_id):
        """Test support provision endpoint"""
        member_id = "test_member"
        support_type = SupportType.TECHNICAL_SUPPORT.value
        provider = "test_provider"
        
        response = client.post(
            f"/api/v1/performance-monitoring/support/{sample_crisis_id}/{member_id}",
            params={"support_type": support_type, "provider": provider}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["member_id"] == member_id
        assert data["crisis_id"] == sample_crisis_id
        assert data["support_type"] == support_type
        assert data["provider"] == provider
        assert "description" in data
    
    def test_get_performance_optimization_endpoint(self, client, sample_crisis_id, sample_team_members):
        """Test performance optimization endpoint"""
        # First track performance
        client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=sample_team_members
        )
        
        # Then get optimizations
        response = client.get(f"/api/v1/performance-monitoring/optimization/{sample_crisis_id}")
        
        assert response.status_code == 200
        optimizations = response.json()
        assert isinstance(optimizations, list)
    
    def test_get_performance_alerts_endpoint(self, client, sample_crisis_id, sample_team_members):
        """Test performance alerts endpoint"""
        # First track performance
        client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=sample_team_members
        )
        
        # Then get alerts
        response = client.get(f"/api/v1/performance-monitoring/alerts/{sample_crisis_id}")
        
        assert response.status_code == 200
        alerts = response.json()
        assert isinstance(alerts, list)
    
    def test_acknowledge_alert_endpoint(self, client, sample_crisis_id, sample_team_members):
        """Test alert acknowledgment endpoint"""
        # First track performance and generate alerts
        client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=sample_team_members
        )
        
        alerts_response = client.get(f"/api/v1/performance-monitoring/alerts/{sample_crisis_id}")
        alerts = alerts_response.json()
        
        if alerts:
            alert_id = alerts[0]["alert_id"]
            response = client.post(
                f"/api/v1/performance-monitoring/alerts/{sample_crisis_id}/{alert_id}/acknowledge"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "Alert acknowledged successfully" in data["message"]
            assert "acknowledged_at" in data
    
    def test_acknowledge_alert_not_found(self, client, sample_crisis_id):
        """Test acknowledging non-existent alert"""
        response = client.post(
            f"/api/v1/performance-monitoring/alerts/{sample_crisis_id}/nonexistent_alert/acknowledge"
        )
        
        assert response.status_code == 404
    
    def test_generate_performance_report_endpoint(self, client, sample_crisis_id, sample_team_members):
        """Test performance report generation endpoint"""
        # First track performance
        client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=sample_team_members
        )
        
        # Then generate report
        response = client.get(
            f"/api/v1/performance-monitoring/report/{sample_crisis_id}",
            params={"time_period_hours": 24}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["crisis_id"] == sample_crisis_id
        assert data["report_type"] == "COMPREHENSIVE_PERFORMANCE"
        assert "team_overview" in data
        assert "key_insights" in data
        assert "performance_trends" in data
        assert "recommendations" in data
        assert "success_metrics" in data
    
    def test_generate_performance_report_invalid_time_period(self, client, sample_crisis_id):
        """Test report generation with invalid time period"""
        response = client.get(
            f"/api/v1/performance-monitoring/report/{sample_crisis_id}",
            params={"time_period_hours": 200}  # Too long
        )
        
        assert response.status_code == 400
        assert "Time period must be between 1 and 168 hours" in response.json()["detail"]
    
    def test_get_intervention_history_endpoint(self, client, sample_crisis_id):
        """Test intervention history endpoint"""
        # First implement an intervention
        client.post(
            f"/api/v1/performance-monitoring/intervention/{sample_crisis_id}/test_member",
            params={"intervention_type": InterventionType.COACHING.value}
        )
        
        # Then get history
        response = client.get(f"/api/v1/performance-monitoring/interventions/{sample_crisis_id}")
        
        assert response.status_code == 200
        interventions = response.json()
        assert isinstance(interventions, list)
        assert len(interventions) >= 1
    
    def test_get_support_provisions_endpoint(self, client, sample_crisis_id):
        """Test support provisions endpoint"""
        # First provide support
        client.post(
            f"/api/v1/performance-monitoring/support/{sample_crisis_id}/test_member",
            params={"support_type": SupportType.TECHNICAL_SUPPORT.value, "provider": "test_provider"}
        )
        
        # Then get support provisions
        response = client.get(f"/api/v1/performance-monitoring/support/{sample_crisis_id}")
        
        assert response.status_code == 200
        support_provisions = response.json()
        assert isinstance(support_provisions, list)
        assert len(support_provisions) >= 1
    
    def test_update_intervention_effectiveness_endpoint(self, client, sample_crisis_id):
        """Test intervention effectiveness update endpoint"""
        # First implement an intervention
        intervention_response = client.post(
            f"/api/v1/performance-monitoring/intervention/{sample_crisis_id}/test_member",
            params={"intervention_type": InterventionType.COACHING.value}
        )
        
        intervention_id = intervention_response.json()["intervention_id"]
        
        # Then update effectiveness
        response = client.put(
            f"/api/v1/performance-monitoring/intervention/{intervention_id}/effectiveness",
            params={"effectiveness_score": 8.5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "Intervention effectiveness updated successfully" in data["message"]
        assert data["effectiveness_score"] == 8.5
    
    def test_update_intervention_effectiveness_not_found(self, client):
        """Test updating effectiveness for non-existent intervention"""
        response = client.put(
            "/api/v1/performance-monitoring/intervention/nonexistent/effectiveness",
            params={"effectiveness_score": 8.5}
        )
        
        assert response.status_code == 404
    
    def test_update_support_feedback_endpoint(self, client, sample_crisis_id):
        """Test support feedback update endpoint"""
        # First provide support
        support_response = client.post(
            f"/api/v1/performance-monitoring/support/{sample_crisis_id}/test_member",
            params={"support_type": SupportType.TECHNICAL_SUPPORT.value, "provider": "test_provider"}
        )
        
        support_id = support_response.json()["support_id"]
        
        # Then update feedback
        response = client.put(
            f"/api/v1/performance-monitoring/support/{support_id}/feedback",
            params={"effectiveness_rating": 9.0, "member_feedback": "Very helpful"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "Support feedback updated successfully" in data["message"]
        assert data["effectiveness_rating"] == 9.0
    
    def test_update_support_feedback_not_found(self, client):
        """Test updating feedback for non-existent support"""
        response = client.put(
            "/api/v1/performance-monitoring/support/nonexistent/feedback",
            params={"effectiveness_rating": 9.0}
        )
        
        assert response.status_code == 404
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/performance-monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "performance-monitoring"
        assert "timestamp" in data
        assert "active_crises" in data
        assert "total_alerts" in data
        assert "total_interventions" in data
        assert "total_support_provisions" in data
    
    def test_complete_workflow_integration(self, client, sample_crisis_id, sample_team_members):
        """Test complete performance monitoring workflow"""
        # Step 1: Track team performance
        track_response = client.post(
            f"/api/v1/performance-monitoring/track-team/{sample_crisis_id}",
            json=sample_team_members
        )
        assert track_response.status_code == 200
        
        # Step 2: Identify issues
        issues_response = client.get(f"/api/v1/performance-monitoring/issues/{sample_crisis_id}")
        assert issues_response.status_code == 200
        
        # Step 3: Implement intervention
        intervention_response = client.post(
            f"/api/v1/performance-monitoring/intervention/{sample_crisis_id}/member_001",
            params={"intervention_type": InterventionType.COACHING.value}
        )
        assert intervention_response.status_code == 200
        
        # Step 4: Provide support
        support_response = client.post(
            f"/api/v1/performance-monitoring/support/{sample_crisis_id}/member_001",
            params={"support_type": SupportType.TECHNICAL_SUPPORT.value, "provider": "senior_engineer"}
        )
        assert support_response.status_code == 200
        
        # Step 5: Generate alerts
        alerts_response = client.get(f"/api/v1/performance-monitoring/alerts/{sample_crisis_id}")
        assert alerts_response.status_code == 200
        
        # Step 6: Get optimizations
        opt_response = client.get(f"/api/v1/performance-monitoring/optimization/{sample_crisis_id}")
        assert opt_response.status_code == 200
        
        # Step 7: Generate report
        report_response = client.get(f"/api/v1/performance-monitoring/report/{sample_crisis_id}")
        assert report_response.status_code == 200
        
        # Step 8: Check health
        health_response = client.get("/api/v1/performance-monitoring/health")
        assert health_response.status_code == 200
        
        # Verify workflow completion
        health_data = health_response.json()
        assert health_data["active_crises"] >= 1
        assert health_data["total_interventions"] >= 1
        assert health_data["total_support_provisions"] >= 1
    
    def test_concurrent_performance_tracking(self, client, sample_team_members):
        """Test concurrent performance tracking for multiple crises"""
        crisis_ids = ["concurrent_crisis_1", "concurrent_crisis_2", "concurrent_crisis_3"]
        
        # Track performance for multiple crises concurrently
        responses = []
        for crisis_id in crisis_ids:
            response = client.post(
                f"/api/v1/performance-monitoring/track-team/{crisis_id}",
                json=sample_team_members
            )
            responses.append(response)
        
        # Verify all requests succeeded
        for response in responses:
            assert response.status_code == 200
        
        # Verify each crisis has separate data
        for i, crisis_id in enumerate(crisis_ids):
            data = responses[i].json()
            assert data["crisis_id"] == crisis_id
            assert data["team_id"] == f"crisis_team_{crisis_id}"
    
    def test_error_handling_integration(self, client):
        """Test error handling across the integration"""
        # Test various error scenarios
        
        # Invalid crisis ID format
        response = client.get("/api/v1/performance-monitoring/performance/")
        assert response.status_code == 404
        
        # Invalid intervention type
        response = client.post(
            "/api/v1/performance-monitoring/intervention/test_crisis/test_member",
            params={"intervention_type": "invalid_type"}
        )
        assert response.status_code == 422  # Validation error
        
        # Invalid support type
        response = client.post(
            "/api/v1/performance-monitoring/support/test_crisis/test_member",
            params={"support_type": "invalid_type", "provider": "test"}
        )
        assert response.status_code == 422  # Validation error