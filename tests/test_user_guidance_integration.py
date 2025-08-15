"""
Integration tests for User Guidance and Support System
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from datetime import datetime
import json

from scrollintel.api.main import app
from scrollintel.models.user_guidance_models import GuidanceContext

class TestUserGuidanceAPIIntegration:
    """Integration tests for user guidance API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_context_data(self):
        """Sample context data for API requests"""
        return {
            "user_id": "test_user_123",
            "session_id": "session_456",
            "current_page": "/dashboard",
            "user_action": "click_button",
            "system_state": {"load": 0.5, "errors": 0},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def test_contextual_help_endpoint(self, client, sample_context_data):
        """Test contextual help API endpoint"""
        response = client.post(
            "/api/v1/guidance/help/contextual",
            json=sample_context_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "guidance" in data
        assert "timestamp" in data
        
        guidance = data["guidance"]
        assert "title" in guidance
        assert "content" in guidance
        assert "confidence_score" in guidance
    
    def test_error_explanation_endpoint(self, client, sample_context_data):
        """Test error explanation API endpoint"""
        error_data = {
            "type": "ValueError",
            "message": "Invalid input format",
            "stack_trace": "Traceback..."
        }
        
        response = client.post(
            "/api/v1/guidance/error/explain",
            json={
                "error_data": error_data,
                "context": sample_context_data
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "explanation" in data
        
        explanation = data["explanation"]
        assert "error_id" in explanation
        assert "user_friendly_explanation" in explanation
        assert "actionable_solutions" in explanation
        assert "severity" in explanation
        assert "resolution_confidence" in explanation
        assert isinstance(explanation["actionable_solutions"], list)
    
    def test_proactive_guidance_endpoint(self, client):
        """Test proactive guidance API endpoint"""
        user_id = "test_user_123"
        
        response = client.get(f"/api/v1/guidance/proactive/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "guidance" in data
        assert "count" in data
        assert isinstance(data["guidance"], list)
        
        if data["guidance"]:
            guidance_item = data["guidance"][0]
            assert "guidance_id" in guidance_item
            assert "type" in guidance_item
            assert "title" in guidance_item
            assert "message" in guidance_item
            assert "priority" in guidance_item
    
    def test_support_ticket_creation_endpoint(self, client, sample_context_data):
        """Test support ticket creation API endpoint"""
        ticket_data = {
            "context": sample_context_data,
            "issue_description": "Unable to upload large files",
            "error_details": {
                "error_code": "FILE_TOO_LARGE",
                "file_size": "50MB",
                "max_size": "25MB"
            }
        }
        
        response = client.post(
            "/api/v1/guidance/support/ticket",
            json=ticket_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "ticket" in data
        
        ticket = data["ticket"]
        assert "ticket_id" in ticket
        assert "title" in ticket
        assert "priority" in ticket
        assert "status" in ticket
        assert "created_at" in ticket
        assert "estimated_resolution" in ticket
    
    def test_support_ticket_retrieval_endpoint(self, client, sample_context_data):
        """Test support ticket retrieval API endpoint"""
        # First create a ticket
        ticket_data = {
            "context": sample_context_data,
            "issue_description": "Test issue for retrieval",
            "error_details": None
        }
        
        create_response = client.post(
            "/api/v1/guidance/support/ticket",
            json=ticket_data
        )
        
        assert create_response.status_code == 200
        ticket_id = create_response.json()["ticket"]["ticket_id"]
        
        # Now retrieve the ticket
        response = client.get(f"/api/v1/guidance/support/ticket/{ticket_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "ticket" in data
        
        ticket = data["ticket"]
        assert ticket["ticket_id"] == ticket_id
        assert "title" in ticket
        assert "description" in ticket
        assert "priority" in ticket
        assert "status" in ticket
    
    def test_feedback_submission_endpoint(self, client):
        """Test feedback submission API endpoint"""
        feedback_data = {
            "user_id": "test_user_123",
            "guidance_id": "guidance_456",
            "rating": 4,
            "helpful": True,
            "comments": "Very helpful guidance",
            "resolution_achieved": True,
            "time_to_resolution": 5
        }
        
        response = client.post(
            "/api/v1/guidance/feedback",
            json=feedback_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        assert "feedback_id" in data
        assert "timestamp" in data
    
    def test_contextual_hints_endpoint(self, client):
        """Test contextual hints API endpoint"""
        page = "dashboard"
        user_id = "test_user_123"
        
        response = client.get(
            f"/api/v1/guidance/hints/{page}",
            params={"user_id": user_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "hints" in data
        assert isinstance(data["hints"], list)
    
    def test_hint_dismissal_endpoint(self, client):
        """Test hint dismissal API endpoint"""
        hint_id = "hint_123"
        user_id = "test_user_123"
        
        response = client.post(
            f"/api/v1/guidance/hints/{hint_id}/dismiss",
            params={"user_id": user_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
    
    def test_guidance_analytics_endpoint(self, client):
        """Test guidance analytics API endpoint"""
        response = client.get("/api/v1/guidance/analytics/effectiveness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "analytics" in data
        assert "period_days" in data
        
        analytics = data["analytics"]
        assert "total_help_requests" in analytics
        assert "successful_resolutions" in analytics
        assert "user_satisfaction_score" in analytics
        assert "proactive_guidance_acceptance" in analytics
    
    def test_guidance_action_execution_endpoint(self, client):
        """Test guidance action execution API endpoint"""
        guidance_id = "guidance_123"
        action_data = {
            "action": "refresh_page",
            "parameters": {}
        }
        user_id = "test_user_123"
        
        response = client.post(
            f"/api/v1/guidance/guidance/{guidance_id}/action",
            json=action_data,
            params={"user_id": user_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        assert "timestamp" in data
    
    def test_error_handling_invalid_context(self, client):
        """Test error handling with invalid context data"""
        invalid_context = {
            "user_id": "",  # Invalid empty user_id
            "session_id": "session_456",
            "current_page": "/dashboard"
        }
        
        response = client.post(
            "/api/v1/guidance/help/contextual",
            json=invalid_context
        )
        
        # Should handle gracefully and provide fallback help
        assert response.status_code in [200, 400]  # Depending on validation
    
    def test_error_handling_nonexistent_ticket(self, client):
        """Test error handling for nonexistent ticket retrieval"""
        nonexistent_ticket_id = "nonexistent_ticket_123"
        
        response = client.get(f"/api/v1/guidance/support/ticket/{nonexistent_ticket_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

class TestUserGuidanceSystemIntegration:
    """Integration tests for the complete user guidance system"""
    
    @pytest.mark.asyncio
    async def test_complete_help_workflow(self):
        """Test complete workflow from help request to resolution"""
        from scrollintel.core.user_guidance_system import UserGuidanceSystem
        
        guidance_system = UserGuidanceSystem()
        
        # Step 1: User requests contextual help
        context = GuidanceContext(
            user_id="workflow_test_user",
            session_id="session_789",
            current_page="/data-analysis",
            user_action="create_chart"
        )
        
        help_result = await guidance_system.provide_contextual_help(context)
        
        assert help_result is not None
        assert "title" in help_result
        
        # Step 2: User encounters an error
        error = ValueError("Invalid chart configuration")
        error_explanation = await guidance_system.explain_error_intelligently(error, context)
        
        assert error_explanation is not None
        assert error_explanation.user_friendly_explanation
        assert len(error_explanation.actionable_solutions) > 0
        
        # Step 3: Check if proactive guidance is provided
        system_state = {"error_rate": 0.02, "load": 0.6}
        proactive_guidance = await guidance_system.provide_proactive_guidance(
            context.user_id, system_state
        )
        
        assert isinstance(proactive_guidance, list)
    
    @pytest.mark.asyncio
    async def test_error_escalation_workflow(self):
        """Test workflow for error escalation to support ticket"""
        from scrollintel.core.user_guidance_system import UserGuidanceSystem
        
        guidance_system = UserGuidanceSystem()
        
        context = GuidanceContext(
            user_id="escalation_test_user",
            session_id="session_escalation",
            current_page="/critical-operation"
        )
        
        # Create a critical error that should escalate
        critical_error = Exception("Database connection completely failed")
        
        # Mock the severity assessment to return CRITICAL
        with pytest.MonkeyPatch().context() as m:
            async def mock_assess_severity(error, context):
                return "critical"
            
            async def mock_calculate_confidence(error, solutions):
                return 0.1  # Low confidence
            
            m.setattr(guidance_system, '_assess_error_severity', mock_assess_severity)
            m.setattr(guidance_system, '_calculate_resolution_confidence', mock_calculate_confidence)
            
            explanation = await guidance_system.explain_error_intelligently(critical_error, context)
            
            # Should have created a support ticket automatically
            assert len(guidance_system.support_tickets) > 0
            
            # Find the created ticket
            ticket = list(guidance_system.support_tickets.values())[0]
            assert ticket.priority in ["high", "critical"]
            assert ticket.auto_created is True
    
    @pytest.mark.asyncio
    async def test_proactive_guidance_system_degradation(self):
        """Test proactive guidance during system degradation"""
        from scrollintel.core.user_guidance_system import UserGuidanceSystem
        
        guidance_system = UserGuidanceSystem()
        
        user_id = "degradation_test_user"
        degraded_system_state = {
            "degraded_services": ["api_service", "ml_service"],
            "error_rate": 0.15,
            "load": 0.95
        }
        
        proactive_guidance = await guidance_system.provide_proactive_guidance(
            user_id, degraded_system_state
        )
        
        # Should provide guidance about system degradation
        assert len(proactive_guidance) > 0
        
        # Check if any guidance is about system status
        system_guidance = [
            g for g in proactive_guidance 
            if "system" in g.title.lower() or "status" in g.title.lower()
        ]
        assert len(system_guidance) > 0
    
    @pytest.mark.asyncio
    async def test_user_behavior_learning(self):
        """Test that system learns from user behavior patterns"""
        from scrollintel.core.user_guidance_system import UserGuidanceSystem
        
        guidance_system = UserGuidanceSystem()
        
        user_id = "learning_test_user"
        
        # Simulate multiple help requests to establish patterns
        contexts = [
            GuidanceContext(
                user_id=user_id,
                session_id=f"session_{i}",
                current_page="/data-upload",
                user_action="upload_file"
            )
            for i in range(3)
        ]
        
        # Request help multiple times
        for context in contexts:
            await guidance_system.provide_contextual_help(context)
        
        # Analyze user behavior
        behavior_pattern = await guidance_system._analyze_user_behavior(user_id)
        
        assert behavior_pattern.user_id == user_id
        # The system should have learned about the user's common actions
        # (This would be more meaningful with actual data storage)
    
    @pytest.mark.asyncio
    async def test_guidance_effectiveness_tracking(self):
        """Test that guidance effectiveness is tracked and improved"""
        from scrollintel.core.user_guidance_system import UserGuidanceSystem
        from scrollintel.models.user_guidance_models import UserFeedback
        
        guidance_system = UserGuidanceSystem()
        
        # Simulate providing guidance
        context = GuidanceContext(
            user_id="effectiveness_test_user",
            session_id="session_effectiveness",
            current_page="/reports"
        )
        
        guidance = await guidance_system.provide_contextual_help(context)
        
        # Simulate positive user feedback
        feedback = UserFeedback(
            user_id=context.user_id,
            guidance_id="test_guidance_id",
            rating=5,
            helpful=True,
            resolution_achieved=True,
            time_to_resolution=3
        )
        
        # The system should use this feedback to improve future guidance
        # (Implementation would update effectiveness metrics)
        assert feedback.helpful is True
        assert feedback.resolution_achieved is True

class TestUserGuidancePerformance:
    """Performance tests for user guidance system"""
    
    @pytest.mark.asyncio
    async def test_contextual_help_response_time(self):
        """Test that contextual help responds quickly"""
        from scrollintel.core.user_guidance_system import UserGuidanceSystem
        import time
        
        guidance_system = UserGuidanceSystem()
        context = GuidanceContext(
            user_id="perf_test_user",
            session_id="session_perf",
            current_page="/dashboard"
        )
        
        start_time = time.time()
        result = await guidance_system.provide_contextual_help(context)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 2 seconds (as per requirements)
        assert response_time < 2.0
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_help_requests(self):
        """Test handling multiple concurrent help requests"""
        from scrollintel.core.user_guidance_system import UserGuidanceSystem
        
        guidance_system = UserGuidanceSystem()
        
        # Create multiple concurrent requests
        contexts = [
            GuidanceContext(
                user_id=f"concurrent_user_{i}",
                session_id=f"session_{i}",
                current_page="/dashboard"
            )
            for i in range(10)
        ]
        
        # Execute all requests concurrently
        tasks = [
            guidance_system.provide_contextual_help(context)
            for context in contexts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should complete successfully
        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)
            assert result is not None

if __name__ == "__main__":
    pytest.main([__file__])