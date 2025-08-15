"""
Integration Tests for Board Executive Mastery System
Tests the complete API integration and system functionality
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from scrollintel.api.routes.board_executive_mastery_routes import router
from scrollintel.core.board_executive_mastery_system import (
    BoardExecutiveMasterySystem,
    BoardExecutiveMasteryConfig
)

# Create test app
app = FastAPI()
app.include_router(router)

class TestBoardExecutiveMasteryIntegration:
    """Integration tests for board executive mastery system"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_board_request(self):
        """Create sample board mastery request"""
        return {
            "board_info": {
                "id": "test_board",
                "company_name": "Test Corp",
                "members": [
                    {
                        "id": "member_1",
                        "name": "John Doe",
                        "role": "chair"
                    }
                ]
            },
            "communication_context": {
                "engagement_type": "board_meeting",
                "key_messages": ["Q3 Results", "Strategic Update"]
            },
            "strategic_context": {
                "current_strategy": {"focus": "growth"}
            }
        }
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return {"id": "test_user", "email": "test@example.com"}
    
    def test_create_engagement_plan_success(self, client, sample_board_request, mock_user):
        """Test successful engagement plan creation"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user):
            response = client.post(
                "/api/v1/board-executive-mastery/create-engagement-plan",
                json=sample_board_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "engagement_plan" in data
            assert data["message"] == "Board engagement plan created successfully"
    
    def test_create_engagement_plan_validation_error(self, client, mock_user):
        """Test engagement plan creation with validation error"""
        invalid_request = {"invalid": "data"}
        
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user):
            response = client.post(
                "/api/v1/board-executive-mastery/create-engagement-plan",
                json=invalid_request
            )
            
            assert response.status_code == 400
    
    def test_execute_interaction_success(self, client, mock_user):
        """Test successful board interaction execution"""
        interaction_request = {
            "engagement_id": "test_engagement",
            "interaction_context": {
                "current_topic": "Q3 Results",
                "board_mood": "positive"
            }
        }
        
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system method
            mock_strategy = Mock()
            mock_strategy.engagement_id = "test_engagement"
            mock_strategy.confidence_level = 0.85
            mock_system.execute_board_interaction.return_value = mock_strategy
            
            response = client.post(
                "/api/v1/board-executive-mastery/execute-interaction",
                json=interaction_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "interaction_strategy" in data
    
    def test_validate_mastery_success(self, client, mock_user):
        """Test successful mastery validation"""
        validation_request = {
            "engagement_id": "test_engagement",
            "validation_context": {
                "positive_board_feedback": True,
                "executive_endorsement": True
            }
        }
        
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system method
            mock_metrics = Mock()
            mock_metrics.engagement_id = "test_engagement"
            mock_metrics.overall_mastery_score = 0.92
            mock_metrics.meets_success_criteria = True
            mock_system.validate_board_mastery_effectiveness.return_value = mock_metrics
            
            response = client.post(
                "/api/v1/board-executive-mastery/validate-mastery",
                json=validation_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "mastery_metrics" in data
    
    def test_optimize_mastery_success(self, client, mock_user):
        """Test successful mastery optimization"""
        optimization_request = {
            "engagement_id": "test_engagement",
            "optimization_context": {
                "focus_areas": ["communication", "trust_building"]
            }
        }
        
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system method
            mock_plan = Mock()
            mock_plan.id = "test_engagement"
            mock_plan.last_updated = datetime.now()
            mock_system.optimize_board_executive_mastery.return_value = mock_plan
            
            response = client.post(
                "/api/v1/board-executive-mastery/optimize-mastery",
                json=optimization_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "engagement_plan" in data
    
    def test_get_engagement_plan_success(self, client, mock_user):
        """Test successful engagement plan retrieval"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system data
            mock_plan = Mock()
            mock_plan.id = "test_engagement"
            mock_plan.board_id = "test_board"
            mock_system.active_engagements = {"test_engagement": mock_plan}
            
            response = client.get("/api/v1/board-executive-mastery/engagement-plan/test_engagement")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "engagement_plan" in data
    
    def test_get_engagement_plan_not_found(self, client, mock_user):
        """Test engagement plan not found"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock empty system data
            mock_system.active_engagements = {}
            
            response = client.get("/api/v1/board-executive-mastery/engagement-plan/nonexistent")
            
            assert response.status_code == 404
    
    def test_get_mastery_metrics_success(self, client, mock_user):
        """Test successful mastery metrics retrieval"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system data
            mock_metrics = Mock()
            mock_metrics.engagement_id = "test_engagement"
            mock_metrics.overall_mastery_score = 0.88
            mock_system.performance_metrics = {"test_engagement": mock_metrics}
            
            response = client.get("/api/v1/board-executive-mastery/mastery-metrics/test_engagement")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "mastery_metrics" in data
    
    def test_get_active_engagements_success(self, client, mock_user):
        """Test successful active engagements retrieval"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system data
            mock_plan = Mock()
            mock_plan.board_id = "test_board"
            mock_plan.created_at = datetime.now()
            
            mock_metrics = Mock()
            mock_metrics.overall_mastery_score = 0.85
            
            mock_system.active_engagements = {"test_engagement": mock_plan}
            mock_system.performance_metrics = {"test_engagement": mock_metrics}
            
            response = client.get("/api/v1/board-executive-mastery/active-engagements")
            
            assert response.status_code == 200
            data = response.json()
            assert "active_engagements" in data
            assert "total_count" in data
            assert data["total_count"] == 1
    
    def test_get_system_status_success(self, client, mock_user):
        """Test successful system status retrieval"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system status
            mock_status = {
                "system_status": "operational",
                "active_engagements": 2,
                "total_validations": 5,
                "performance_averages": {
                    "board_confidence": 0.88,
                    "executive_trust": 0.85,
                    "strategic_alignment": 0.92,
                    "overall_mastery": 0.88
                },
                "system_health": {
                    "board_dynamics_engine": {"status": "operational"},
                    "executive_communication_engine": {"status": "operational"}
                },
                "configuration": {
                    "real_time_adaptation": True,
                    "predictive_analytics": True,
                    "continuous_learning": True
                },
                "timestamp": datetime.now().isoformat()
            }
            mock_system.get_mastery_system_status.return_value = mock_status
            
            response = client.get("/api/v1/board-executive-mastery/system-status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["system_status"] == "operational"
            assert "performance_averages" in data
            assert "system_health" in data
            assert "configuration" in data
    
    def test_delete_engagement_plan_success(self, client, mock_user):
        """Test successful engagement plan deletion"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system data
            mock_plan = Mock()
            mock_system.active_engagements = {"test_engagement": mock_plan}
            mock_system.performance_metrics = {"test_engagement": Mock()}
            
            response = client.delete("/api/v1/board-executive-mastery/engagement-plan/test_engagement")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "deleted successfully" in data["message"]
    
    def test_batch_validate_success(self, client, mock_user):
        """Test successful batch validation"""
        batch_request = {
            "engagement_ids": ["engagement_1", "engagement_2"],
            "validation_context": {
                "positive_board_feedback": True
            }
        }
        
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system method
            mock_metrics = Mock()
            mock_metrics.engagement_id = "test_engagement"
            mock_metrics.overall_mastery_score = 0.90
            mock_system.validate_board_mastery_effectiveness.return_value = mock_metrics
            
            response = client.post(
                "/api/v1/board-executive-mastery/batch-validate",
                json=batch_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "validation_results" in data
            assert "total_processed" in data
            assert "successful_validations" in data
    
    def test_get_performance_analytics_success(self, client, mock_user):
        """Test successful performance analytics retrieval"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock the system data
            mock_metrics_1 = Mock()
            mock_metrics_1.board_confidence_score = 0.85
            mock_metrics_1.executive_trust_score = 0.80
            mock_metrics_1.strategic_alignment_score = 0.90
            mock_metrics_1.communication_effectiveness_score = 0.88
            mock_metrics_1.stakeholder_influence_score = 0.82
            mock_metrics_1.overall_mastery_score = 0.85
            mock_metrics_1.meets_success_criteria = True
            
            mock_metrics_2 = Mock()
            mock_metrics_2.board_confidence_score = 0.90
            mock_metrics_2.executive_trust_score = 0.85
            mock_metrics_2.strategic_alignment_score = 0.95
            mock_metrics_2.communication_effectiveness_score = 0.92
            mock_metrics_2.stakeholder_influence_score = 0.88
            mock_metrics_2.overall_mastery_score = 0.90
            mock_metrics_2.meets_success_criteria = True
            
            mock_system.performance_metrics = {
                "engagement_1": mock_metrics_1,
                "engagement_2": mock_metrics_2
            }
            
            response = client.get("/api/v1/board-executive-mastery/performance-analytics")
            
            assert response.status_code == 200
            data = response.json()
            assert "analytics" in data
            analytics = data["analytics"]
            assert analytics["total_engagements"] == 2
            assert analytics["successful_engagements"] == 2
            assert analytics["success_rate"] == 1.0
            assert "average_scores" in analytics
            assert "score_distribution" in analytics
            assert "performance_thresholds" in analytics
    
    def test_unauthorized_access(self, client):
        """Test unauthorized access to protected endpoints"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', side_effect=Exception("Unauthorized")):
            response = client.get("/api/v1/board-executive-mastery/system-status")
            assert response.status_code == 500  # FastAPI converts exceptions to 500
    
    def test_system_error_handling(self, client, mock_user, sample_board_request):
        """Test system error handling"""
        with patch('scrollintel.api.routes.board_executive_mastery_routes.get_current_user', return_value=mock_user), \
             patch('scrollintel.api.routes.board_executive_mastery_routes.mastery_system') as mock_system:
            
            # Mock system error
            mock_system.create_comprehensive_engagement_plan.side_effect = Exception("System error")
            
            response = client.post(
                "/api/v1/board-executive-mastery/create-engagement-plan",
                json=sample_board_request
            )
            
            assert response.status_code == 500

class TestBoardExecutiveMasterySystemIntegration:
    """Integration tests for the core system functionality"""
    
    @pytest.fixture
    def mastery_system(self):
        """Create test mastery system"""
        config = BoardExecutiveMasteryConfig(
            enable_real_time_adaptation=True,
            enable_predictive_analytics=True,
            enable_continuous_learning=True
        )
        return BoardExecutiveMasterySystem(config)
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self, mastery_system):
        """Test complete workflow from plan creation to optimization"""
        # Create mock request
        mock_request = Mock()
        mock_request.board_info = Mock()
        mock_request.board_info.id = "test_board"
        mock_request.executives = []
        mock_request.communication_context = Mock()
        mock_request.presentation_requirements = Mock()
        mock_request.strategic_context = Mock()
        mock_request.meeting_context = Mock()
        mock_request.credibility_context = Mock()
        
        # Step 1: Create engagement plan
        engagement_plan = await mastery_system.create_comprehensive_engagement_plan(mock_request)
        assert engagement_plan is not None
        assert engagement_plan.board_id == "test_board"
        
        # Step 2: Execute interaction
        interaction_context = {"test": "context"}
        interaction_strategy = await mastery_system.execute_board_interaction(
            engagement_plan.id, interaction_context
        )
        assert interaction_strategy is not None
        assert interaction_strategy.engagement_id == engagement_plan.id
        
        # Step 3: Validate mastery
        validation_context = {"positive_feedback": True}
        mastery_metrics = await mastery_system.validate_board_mastery_effectiveness(
            engagement_plan.id, validation_context
        )
        assert mastery_metrics is not None
        assert mastery_metrics.engagement_id == engagement_plan.id
        
        # Step 4: Optimize mastery
        optimization_context = {"focus": "improvement"}
        optimized_plan = await mastery_system.optimize_board_executive_mastery(
            engagement_plan.id, optimization_context
        )
        assert optimized_plan is not None
        assert optimized_plan.id == engagement_plan.id
        
        # Step 5: Get system status
        system_status = await mastery_system.get_mastery_system_status()
        assert system_status is not None
        assert system_status["system_status"] == "operational"
        assert system_status["active_engagements"] == 1
        assert system_status["total_validations"] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_engagements(self, mastery_system):
        """Test handling multiple concurrent engagements"""
        # Create multiple mock requests
        requests = []
        for i in range(3):
            mock_request = Mock()
            mock_request.board_info = Mock()
            mock_request.board_info.id = f"test_board_{i}"
            mock_request.executives = []
            mock_request.communication_context = Mock()
            mock_request.presentation_requirements = Mock()
            mock_request.strategic_context = Mock()
            mock_request.meeting_context = Mock()
            mock_request.credibility_context = Mock()
            requests.append(mock_request)
        
        # Create engagement plans concurrently
        tasks = [
            mastery_system.create_comprehensive_engagement_plan(req)
            for req in requests
        ]
        engagement_plans = await asyncio.gather(*tasks)
        
        # Verify all plans were created
        assert len(engagement_plans) == 3
        assert len(mastery_system.active_engagements) == 3
        
        # Verify each plan has unique ID and correct board ID
        for i, plan in enumerate(engagement_plans):
            assert plan.board_id == f"test_board_{i}"
            assert plan.id in mastery_system.active_engagements
    
    @pytest.mark.asyncio
    async def test_system_resilience(self, mastery_system):
        """Test system resilience to errors"""
        # Test with invalid engagement ID
        with pytest.raises(ValueError):
            await mastery_system.execute_board_interaction("invalid_id", {})
        
        with pytest.raises(ValueError):
            await mastery_system.validate_board_mastery_effectiveness("invalid_id", {})
        
        with pytest.raises(ValueError):
            await mastery_system.optimize_board_executive_mastery("invalid_id", {})
        
        # Verify system status remains operational
        system_status = await mastery_system.get_mastery_system_status()
        assert system_status["system_status"] == "operational"
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        config = BoardExecutiveMasteryConfig(
            enable_real_time_adaptation=True,
            enable_predictive_analytics=False,
            enable_continuous_learning=True,
            board_confidence_threshold=0.75,
            executive_trust_threshold=0.70,
            strategic_alignment_threshold=0.80
        )
        
        system = BoardExecutiveMasterySystem(config)
        assert system.config.board_confidence_threshold == 0.75
        assert system.config.executive_trust_threshold == 0.70
        assert system.config.strategic_alignment_threshold == 0.80
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, mastery_system):
        """Test performance metrics tracking and analytics"""
        # Create engagement and validate multiple times
        mock_request = Mock()
        mock_request.board_info = Mock()
        mock_request.board_info.id = "test_board"
        mock_request.executives = []
        mock_request.communication_context = Mock()
        mock_request.presentation_requirements = Mock()
        mock_request.strategic_context = Mock()
        mock_request.meeting_context = Mock()
        mock_request.credibility_context = Mock()
        
        engagement_plan = await mastery_system.create_comprehensive_engagement_plan(mock_request)
        
        # Validate mastery multiple times with different contexts
        validation_contexts = [
            {"positive_feedback": True, "score_modifier": 0.1},
            {"positive_feedback": True, "score_modifier": 0.2},
            {"positive_feedback": False, "score_modifier": -0.1}
        ]
        
        for context in validation_contexts:
            await mastery_system.validate_board_mastery_effectiveness(
                engagement_plan.id, context
            )
        
        # Check that only the latest metrics are stored (not all validations)
        assert len(mastery_system.performance_metrics) == 1
        assert engagement_plan.id in mastery_system.performance_metrics
        
        # Verify system status reflects the metrics
        system_status = await mastery_system.get_mastery_system_status()
        assert system_status["total_validations"] == 1
        assert "performance_averages" in system_status

if __name__ == "__main__":
    pytest.main([__file__])