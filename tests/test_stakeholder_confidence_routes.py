"""
Tests for stakeholder confidence management API routes.
Tests REST API endpoints for confidence monitoring and management.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from scrollintel.api.routes.stakeholder_confidence_routes import router
from scrollintel.models.stakeholder_confidence_models import (
    StakeholderProfile, StakeholderFeedback, StakeholderType, ConfidenceLevel
)
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestStakeholderConfidenceRoutes:
    """Test cases for stakeholder confidence management API routes"""
    
    def test_monitor_stakeholder_confidence(self):
        """Test stakeholder confidence monitoring endpoint"""
        # Test data
        request_data = {
            "crisis_id": "crisis_001",
            "stakeholder_ids": ["stakeholder_001", "stakeholder_002"]
        }
        
        # Make request
        response = client.post(
            "/api/v1/stakeholder-confidence/monitor",
            params=request_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, dict)
        # Should have confidence data for each stakeholder
        for stakeholder_id in request_data["stakeholder_ids"]:
            if stakeholder_id in data:
                metrics = data[stakeholder_id]
                assert "stakeholder_id" in metrics
                assert "confidence_level" in metrics
                assert "trust_score" in metrics
                assert "measurement_time" in metrics
    
    def test_assess_overall_confidence(self):
        """Test overall confidence assessment endpoint"""
        # Make request
        response = client.post(
            "/api/v1/stakeholder-confidence/assess",
            params={"crisis_id": "crisis_001"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "assessment_id" in data
        assert "crisis_id" in data
        assert data["crisis_id"] == "crisis_001"
        assert "overall_confidence_score" in data
        assert "stakeholder_breakdown" in data
        assert "risk_areas" in data
        assert "improvement_opportunities" in data
        assert "recommended_actions" in data
        assert "assessment_time" in data
        assert "next_assessment_date" in data
        
        # Verify score is valid
        assert 0.0 <= data["overall_confidence_score"] <= 1.0
    
    def test_build_confidence_strategy(self):
        """Test confidence building strategy endpoint"""
        # Test data
        request_data = {
            "stakeholder_type": "investor",
            "current_confidence": "low",
            "target_confidence": "high"
        }
        
        # Make request
        response = client.post(
            "/api/v1/stakeholder-confidence/strategy/build",
            params=request_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "strategy_id" in data
        assert "stakeholder_type" in data
        assert data["stakeholder_type"] == request_data["stakeholder_type"]
        assert "target_confidence_level" in data
        assert data["target_confidence_level"] == request_data["target_confidence"]
        assert "communication_approach" in data
        assert "key_messages" in data
        assert "engagement_tactics" in data
        assert "timeline" in data
        assert "success_metrics" in data
        assert "resource_requirements" in data
        assert "risk_mitigation" in data
        
        # Verify lists are not empty
        assert len(data["key_messages"]) > 0
        assert len(data["engagement_tactics"]) > 0
        assert len(data["success_metrics"]) > 0
    
    def test_maintain_stakeholder_trust(self):
        """Test stakeholder trust maintenance endpoint"""
        # First create a stakeholder profile
        profile_data = {
            "stakeholder_id": "stakeholder_001",
            "name": "Test Stakeholder",
            "stakeholder_type": "investor",
            "influence_level": "high",
            "communication_preferences": ["email"],
            "historical_confidence": [0.8, 0.7],
            "key_concerns": ["financial_impact"],
            "relationship_strength": 0.8,
            "contact_information": {"email": "test@example.com"}
        }
        
        profile_response = client.post(
            "/api/v1/stakeholder-confidence/profiles",
            json=profile_data
        )
        assert profile_response.status_code == 200
        
        # Test trust maintenance
        crisis_context = {
            "crisis_type": "system_outage",
            "severity": "high"
        }
        
        response = client.post(
            "/api/v1/stakeholder-confidence/trust/maintain",
            params={"stakeholder_id": "stakeholder_001"},
            json=crisis_context
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Verify each action
        for action in data:
            assert "action_id" in action
            assert "stakeholder_id" in action
            assert action["stakeholder_id"] == "stakeholder_001"
            assert "action_type" in action
            assert "description" in action
            assert "priority" in action
            assert action["priority"] in ["high", "medium", "low"]
            assert "implementation_steps" in action
            assert "required_resources" in action
            assert "success_criteria" in action
            assert "timeline" in action
    
    def test_create_communication_plan(self):
        """Test communication plan creation endpoint"""
        # Test data
        request_data = {
            "crisis_id": "crisis_001",
            "stakeholder_segments": ["investor", "customer"]
        }
        
        # Make request
        response = client.post(
            "/api/v1/stakeholder-confidence/communication/plan",
            params=request_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "plan_id" in data
        assert "stakeholder_segments" in data
        assert len(data["stakeholder_segments"]) == 2
        assert "key_messages" in data
        assert "communication_channels" in data
        assert "frequency" in data
        assert "tone_and_style" in data
        assert "approval_workflow" in data
        assert "feedback_mechanisms" in data
        assert "escalation_triggers" in data
        assert "effectiveness_metrics" in data
        
        # Verify key messages for each segment
        key_messages = data["key_messages"]
        for segment in request_data["stakeholder_segments"]:
            assert segment in key_messages
            assert isinstance(key_messages[segment], str)
    
    def test_process_stakeholder_feedback(self):
        """Test stakeholder feedback processing endpoint"""
        # Test data
        feedback_data = {
            "feedback_id": "feedback_001",
            "stakeholder_id": "stakeholder_001",
            "feedback_type": "concern",
            "content": "Concerned about the crisis impact",
            "sentiment": "negative",
            "urgency_level": "high",
            "received_time": datetime.now().isoformat(),
            "response_required": True
        }
        
        # Make request
        response = client.post(
            "/api/v1/stakeholder-confidence/feedback/process",
            json=feedback_data
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "feedback_id" in data
        assert data["feedback_id"] == feedback_data["feedback_id"]
        assert "analysis" in data
        assert "response_strategy" in data
        assert "follow_up_actions" in data
        assert "processing_time" in data
        
        # Verify analysis
        analysis = data["analysis"]
        assert "sentiment_score" in analysis
        assert "urgency_level" in analysis
        assert "requires_escalation" in analysis
        
        # Verify response strategy
        response_strategy = data["response_strategy"]
        assert "response_type" in response_strategy
        assert "response_timeline" in response_strategy
        
        # Verify follow-up actions
        follow_up_actions = data["follow_up_actions"]
        assert isinstance(follow_up_actions, list)
        assert len(follow_up_actions) > 0
    
    def test_stakeholder_profile_crud(self):
        """Test stakeholder profile CRUD operations"""
        # Create profile
        profile_data = {
            "stakeholder_id": "stakeholder_test",
            "name": "Test Stakeholder",
            "stakeholder_type": "customer",
            "influence_level": "medium",
            "communication_preferences": ["email", "phone"],
            "historical_confidence": [0.7, 0.8],
            "key_concerns": ["service_quality"],
            "relationship_strength": 0.7,
            "contact_information": {"email": "test@customer.com"}
        }
        
        # Create
        create_response = client.post(
            "/api/v1/stakeholder-confidence/profiles",
            json=profile_data
        )
        assert create_response.status_code == 200
        created_profile = create_response.json()
        assert created_profile["stakeholder_id"] == profile_data["stakeholder_id"]
        
        # Read
        get_response = client.get(
            f"/api/v1/stakeholder-confidence/profiles/{profile_data['stakeholder_id']}"
        )
        assert get_response.status_code == 200
        retrieved_profile = get_response.json()
        assert retrieved_profile["stakeholder_id"] == profile_data["stakeholder_id"]
        assert retrieved_profile["name"] == profile_data["name"]
    
    def test_get_confidence_metrics(self):
        """Test confidence metrics retrieval endpoint"""
        # First monitor confidence to generate metrics
        monitor_response = client.post(
            "/api/v1/stakeholder-confidence/monitor",
            params={
                "crisis_id": "crisis_001",
                "stakeholder_ids": ["stakeholder_metrics_test"]
            }
        )
        
        # Get metrics
        response = client.get(
            "/api/v1/stakeholder-confidence/metrics/stakeholder_metrics_test"
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # May be empty if stakeholder doesn't exist, but should be valid list
    
    def test_get_active_alerts(self):
        """Test active alerts retrieval endpoint"""
        response = client.get("/api/v1/stakeholder-confidence/alerts")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # Verify alert structure if any alerts exist
        for alert in data:
            assert "alert_id" in alert
            assert "stakeholder_id" in alert
            assert "alert_type" in alert
            assert "severity" in alert
            assert "description" in alert
            assert "trigger_time" in alert
    
    def test_resolve_confidence_alert(self):
        """Test confidence alert resolution endpoint"""
        # This test assumes there might not be any alerts, so we test the not found case
        response = client.post(
            "/api/v1/stakeholder-confidence/alerts/nonexistent_alert/resolve"
        )
        
        # Should return 404 for non-existent alert
        assert response.status_code == 404
        assert "Alert not found" in response.json()["detail"]
    
    def test_get_confidence_assessments(self):
        """Test confidence assessments retrieval endpoint"""
        response = client.get("/api/v1/stakeholder-confidence/assessments")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # Verify assessment structure if any assessments exist
        for assessment in data:
            assert "assessment_id" in assessment
            assert "crisis_id" in assessment
            assert "overall_confidence_score" in assessment
            assert "assessment_time" in assessment
    
    def test_get_stakeholder_feedback(self):
        """Test stakeholder feedback retrieval endpoint"""
        response = client.get("/api/v1/stakeholder-confidence/feedback")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # Verify feedback structure if any feedback exists
        for feedback in data:
            assert "feedback_id" in feedback
            assert "stakeholder_id" in feedback
            assert "feedback_type" in feedback
            assert "content" in feedback
    
    def test_get_confidence_strategies(self):
        """Test confidence strategies retrieval endpoint"""
        response = client.get("/api/v1/stakeholder-confidence/strategies")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # Verify strategy structure if any strategies exist
        for strategy in data:
            assert "strategy_id" in strategy
            assert "stakeholder_type" in strategy
            assert "target_confidence_level" in strategy
    
    def test_get_trust_actions(self):
        """Test trust actions retrieval endpoint"""
        response = client.get(
            "/api/v1/stakeholder-confidence/actions/stakeholder_001"
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # Verify action structure if any actions exist
        for action in data:
            assert "action_id" in action
            assert "stakeholder_id" in action
            assert "action_type" in action
            assert "description" in action
    
    def test_complete_trust_action(self):
        """Test trust action completion endpoint"""
        # This test assumes there might not be any actions, so we test the not found case
        response = client.post(
            "/api/v1/stakeholder-confidence/actions/nonexistent_action/complete"
        )
        
        # Should return 404 for non-existent action
        assert response.status_code == 404
        assert "Trust maintenance action not found" in response.json()["detail"]
    
    def test_get_confidence_dashboard(self):
        """Test confidence dashboard endpoint"""
        response = client.get("/api/v1/stakeholder-confidence/dashboard")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "total_stakeholders" in data
        assert "active_alerts" in data
        assert "pending_feedback" in data
        assert "confidence_trends" in data
        assert "risk_areas" in data
        assert "improvement_opportunities" in data
        
        # Verify data types
        assert isinstance(data["total_stakeholders"], int)
        assert isinstance(data["active_alerts"], int)
        assert isinstance(data["pending_feedback"], int)
        assert isinstance(data["confidence_trends"], dict)
        assert isinstance(data["risk_areas"], list)
        assert isinstance(data["improvement_opportunities"], list)
    
    def test_invalid_stakeholder_type(self):
        """Test handling of invalid stakeholder type"""
        request_data = {
            "stakeholder_type": "invalid_type",
            "current_confidence": "low",
            "target_confidence": "high"
        }
        
        response = client.post(
            "/api/v1/stakeholder-confidence/strategy/build",
            params=request_data
        )
        
        # Should return 422 for invalid enum value
        assert response.status_code == 422
    
    def test_invalid_confidence_level(self):
        """Test handling of invalid confidence level"""
        request_data = {
            "stakeholder_type": "investor",
            "current_confidence": "invalid_level",
            "target_confidence": "high"
        }
        
        response = client.post(
            "/api/v1/stakeholder-confidence/strategy/build",
            params=request_data
        )
        
        # Should return 422 for invalid enum value
        assert response.status_code == 422
    
    def test_missing_required_parameters(self):
        """Test handling of missing required parameters"""
        # Test monitor endpoint without stakeholder_ids
        response = client.post(
            "/api/v1/stakeholder-confidence/monitor",
            params={"crisis_id": "crisis_001"}
        )
        
        # Should return 422 for missing required parameter
        assert response.status_code == 422
    
    def test_nonexistent_stakeholder_profile(self):
        """Test handling of non-existent stakeholder profile"""
        response = client.get(
            "/api/v1/stakeholder-confidence/profiles/nonexistent_stakeholder"
        )
        
        # Should return 404 for non-existent profile
        assert response.status_code == 404
        assert "Stakeholder profile not found" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])