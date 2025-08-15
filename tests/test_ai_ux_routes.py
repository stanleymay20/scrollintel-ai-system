"""
Tests for AI UX Optimization API Routes

This module contains tests for the REST API endpoints that provide
AI-powered user experience optimization functionality.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from scrollintel.api.routes.ai_ux_routes import router
from scrollintel.engines.ai_ux_optimizer import (
    AIUXOptimizer, UserBehaviorPattern, DegradationStrategy,
    FailurePrediction, UserBehaviorAnalysis, PersonalizedDegradation, InterfaceOptimization,
    PredictionType
)

# Create test client
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestAIUXRoutes:
    """Test cases for AI UX optimization API routes"""
    
    @pytest.fixture
    def mock_optimizer(self):
        """Mock AI UX Optimizer for testing"""
        optimizer = Mock(spec=AIUXOptimizer)
        optimizer.user_profiles = {}
        return optimizer
    
    @pytest.fixture
    def sample_user_interaction(self):
        """Sample user interaction request"""
        return {
            "user_id": "test_user_123",
            "session_id": "session_456",
            "action_type": "click",
            "feature_used": "dashboard",
            "page_visited": "/dashboard",
            "duration": 2.5,
            "success": True,
            "error_encountered": None,
            "help_requested": False,
            "metadata": {"browser": "chrome", "device": "desktop"}
        }
    
    @pytest.fixture
    def sample_system_metrics(self):
        """Sample system metrics request"""
        return {
            "cpu_usage": 0.7,
            "memory_usage": 0.6,
            "disk_usage": 0.4,
            "network_latency": 150.0,
            "error_rate": 0.02,
            "response_time": 800.0,
            "active_users": 150,
            "request_rate": 25.0,
            "system_load": 0.65,
            "additional_metrics": {"custom_metric": 42}
        }
    
    @pytest.fixture
    def sample_user_feedback(self):
        """Sample user feedback request"""
        return {
            "user_id": "test_user_123",
            "optimization_type": "interface_optimization",
            "satisfaction_score": 4.5,
            "feedback_text": "The interface improvements made it much easier to use",
            "improvement_suggestions": ["Add more keyboard shortcuts", "Improve mobile layout"],
            "would_recommend": True
        }
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/ai-ux/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "message" in data
    
    @patch('scrollintel.api.routes.ai_ux_routes.get_database_session')
    def test_record_user_interaction(self, mock_db_session, sample_user_interaction):
        """Test recording user interaction"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value = mock_session
        
        response = client.post(
            "/api/v1/ai-ux/interactions",
            json=sample_user_interaction
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data
        
        # Verify database interaction
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @patch('scrollintel.api.routes.ai_ux_routes.get_database_session')
    def test_record_system_metrics(self, mock_db_session, sample_system_metrics):
        """Test recording system metrics"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value = mock_session
        
        response = client.post(
            "/api/v1/ai-ux/system-metrics",
            json=sample_system_metrics
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data
        
        # Verify database interaction
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer')
    def test_get_failure_predictions(self, mock_optimizer):
        """Test getting failure predictions"""
        # Mock failure predictions
        mock_prediction = FailurePrediction(
            prediction_type=PredictionType.FAILURE_RISK,
            probability=0.7,
            confidence=0.8,
            time_to_failure=15,
            contributing_factors=["High CPU usage", "Elevated error rate"],
            recommended_actions=["Scale up resources", "Enable circuit breakers"],
            timestamp=datetime.now()
        )
        
        mock_optimizer.predict_failures = AsyncMock(return_value=[mock_prediction])
        
        response = client.get("/api/v1/ai-ux/failure-predictions")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        
        prediction_data = data[0]
        assert prediction_data["prediction_type"] == "failure_risk"
        assert prediction_data["probability"] == 0.7
        assert prediction_data["confidence"] == 0.8
        assert prediction_data["time_to_failure"] == 15
        assert len(prediction_data["contributing_factors"]) == 2
        assert len(prediction_data["recommended_actions"]) == 2
    
    @patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer')
    def test_get_user_behavior_analysis(self, mock_optimizer):
        """Test getting user behavior analysis"""
        user_id = "test_user_123"
        
        # Mock user profile
        mock_profile = UserBehaviorAnalysis(
            user_id=user_id,
            behavior_pattern=UserBehaviorPattern.POWER_USER,
            engagement_score=0.8,
            frustration_indicators=[],
            preferred_features=["dashboard", "analytics"],
            usage_patterns={"avg_session_duration": 25},
            assistance_needs=["Advanced features"],
            timestamp=datetime.now()
        )
        
        mock_optimizer.user_profiles = {user_id: mock_profile}
        
        response = client.get(f"/api/v1/ai-ux/user-behavior/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == user_id
        assert data["behavior_pattern"] == "power_user"
        assert data["engagement_score"] == 0.8
        assert data["preferred_features"] == ["dashboard", "analytics"]
        assert data["assistance_needs"] == ["Advanced features"]
    
    @patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer')
    def test_get_user_behavior_analysis_not_found(self, mock_optimizer):
        """Test getting user behavior analysis for non-existent user"""
        mock_optimizer.user_profiles = {}
        
        response = client.get("/api/v1/ai-ux/user-behavior/nonexistent_user")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer')
    def test_analyze_user_behavior(self, mock_optimizer):
        """Test analyzing user behavior"""
        user_id = "test_user_123"
        interaction_data = {
            "session_duration": 20,
            "clicks_per_minute": 8,
            "pages_visited": 5,
            "errors_encountered": 1
        }
        
        # Mock behavior analysis
        mock_analysis = UserBehaviorAnalysis(
            user_id=user_id,
            behavior_pattern=UserBehaviorPattern.CASUAL_USER,
            engagement_score=0.6,
            frustration_indicators=["Some errors"],
            preferred_features=["search"],
            usage_patterns={"avg_session_duration": 20},
            assistance_needs=["Error prevention guidance"],
            timestamp=datetime.now()
        )
        
        mock_optimizer.analyze_user_behavior = AsyncMock(return_value=mock_analysis)
        
        response = client.post(
            f"/api/v1/ai-ux/user-behavior/{user_id}/analyze",
            json=interaction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == user_id
        assert data["behavior_pattern"] == "casual_user"
        assert data["engagement_score"] == 0.6
        assert "Some errors" in data["frustration_indicators"]
    
    @patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer')
    def test_get_personalized_degradation(self, mock_optimizer):
        """Test getting personalized degradation strategy"""
        user_id = "test_user_123"
        
        # Mock degradation strategy
        mock_degradation = PersonalizedDegradation(
            user_id=user_id,
            strategy=DegradationStrategy.MODERATE,
            feature_priorities={"dashboard": 9, "search": 7},
            acceptable_delays={"page_load": 3.0, "search": 2.0},
            fallback_preferences={"search": "cached_results"},
            communication_style="informative",
            timestamp=datetime.now()
        )
        
        mock_optimizer.create_personalized_degradation = AsyncMock(return_value=mock_degradation)
        
        response = client.get(f"/api/v1/ai-ux/personalized-degradation/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == user_id
        assert data["strategy"] == "moderate"
        assert data["feature_priorities"]["dashboard"] == 9
        assert data["acceptable_delays"]["page_load"] == 3.0
        assert data["communication_style"] == "informative"
    
    @patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer')
    def test_get_interface_optimization(self, mock_optimizer):
        """Test getting interface optimization"""
        user_id = "test_user_123"
        
        # Mock interface optimization
        mock_optimization = InterfaceOptimization(
            user_id=user_id,
            layout_preferences={"density": "compact", "sidebar": "expanded"},
            interaction_patterns={"keyboard_usage": 0.8, "click_frequency": 12.0},
            performance_requirements={"page_load_time": 2.0, "interaction_response": 0.3},
            accessibility_needs=["High contrast mode"],
            optimization_suggestions=["Enable compact view", "Show keyboard shortcuts"],
            timestamp=datetime.now()
        )
        
        mock_optimizer.optimize_interface = AsyncMock(return_value=mock_optimization)
        
        response = client.get(f"/api/v1/ai-ux/interface-optimization/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == user_id
        assert data["layout_preferences"]["density"] == "compact"
        assert data["interaction_patterns"]["keyboard_usage"] == 0.8
        assert "High contrast mode" in data["accessibility_needs"]
        assert "Enable compact view" in data["optimization_suggestions"]
    
    @patch('scrollintel.api.routes.ai_ux_routes.get_database_session')
    def test_submit_user_feedback(self, mock_db_session, sample_user_feedback):
        """Test submitting user feedback"""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value = mock_session
        
        response = client.post(
            "/api/v1/ai-ux/feedback",
            json=sample_user_feedback
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data
        
        # Verify database interaction
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    def test_submit_user_feedback_validation_error(self):
        """Test submitting invalid user feedback"""
        invalid_feedback = {
            "user_id": "test_user_123",
            "optimization_type": "interface_optimization",
            "satisfaction_score": 6.0,  # Invalid: should be 1-5
            "would_recommend": True
        }
        
        response = client.post(
            "/api/v1/ai-ux/feedback",
            json=invalid_feedback
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer')
    def test_get_optimization_metrics(self, mock_optimizer):
        """Test getting optimization metrics"""
        # Mock metrics
        mock_metrics = {
            "total_users_analyzed": 150,
            "behavior_patterns": {
                "power_user": 30,
                "casual_user": 80,
                "struggling_user": 25,
                "new_user": 15
            },
            "average_engagement_score": 0.65,
            "common_frustration_indicators": {
                "Multiple errors in session": 45,
                "Frequent help requests": 32,
                "Rapid page switching": 28
            }
        }
        
        mock_optimizer.get_optimization_metrics = AsyncMock(return_value=mock_metrics)
        
        response = client.get("/api/v1/ai-ux/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_users_analyzed"] == 150
        assert "behavior_patterns" in data
        assert data["average_engagement_score"] == 0.65
        assert "common_frustration_indicators" in data
    
    def test_train_ai_models(self):
        """Test training AI models"""
        training_data = {
            "failure_data": [
                {
                    "metrics": {"cpu_usage": 0.9, "error_rate": 0.05},
                    "failure_occurred": 1
                }
            ],
            "behavior_data": [
                {
                    "user_id": "user1",
                    "interaction_data": {"session_duration": 30, "errors": 0}
                }
            ]
        }
        
        response = client.post(
            "/api/v1/ai-ux/train-models",
            json=training_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "training started" in data["message"].lower()
    
    @patch('scrollintel.api.routes.ai_ux_routes.get_database_session')
    def test_get_user_session_analysis(self, mock_db_session):
        """Test getting user session analysis"""
        user_id = "test_user_123"
        
        # Mock database session and interactions
        mock_session = Mock()
        mock_db_session.return_value = mock_session
        
        # Mock interaction data
        mock_interactions = [
            Mock(
                user_id=user_id,
                session_id="session1",
                success=True,
                timestamp=datetime.now(),
                duration=2.5,
                error_encountered=None,
                help_requested=False
            ),
            Mock(
                user_id=user_id,
                session_id="session1",
                success=True,
                timestamp=datetime.now(),
                duration=1.8,
                error_encountered=None,
                help_requested=False
            )
        ]
        
        mock_session.query.return_value.filter.return_value.all.return_value = mock_interactions
        
        response = client.get(f"/api/v1/ai-ux/user-sessions/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == user_id
        assert "total_interactions" in data
        assert "total_sessions" in data
        assert "sessions" in data
    
    @patch('scrollintel.api.routes.ai_ux_routes.get_database_session')
    def test_get_user_session_analysis_no_data(self, mock_db_session):
        """Test getting user session analysis with no data"""
        user_id = "nonexistent_user"
        
        # Mock database session with no interactions
        mock_session = Mock()
        mock_db_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.all.return_value = []
        
        response = client.get(f"/api/v1/ai-ux/user-sessions/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "No interactions found" in data["message"]
    
    def test_invalid_user_interaction_request(self):
        """Test invalid user interaction request"""
        invalid_interaction = {
            "user_id": "",  # Empty user_id
            "session_id": "session_456",
            "action_type": "click"
        }
        
        response = client.post(
            "/api/v1/ai-ux/interactions",
            json=invalid_interaction
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_system_metrics_request(self):
        """Test invalid system metrics request"""
        invalid_metrics = {
            "cpu_usage": 1.5,  # Invalid: should be 0-1
            "memory_usage": -0.1,  # Invalid: should be >= 0
            "error_rate": 2.0  # Invalid: should be 0-1
        }
        
        response = client.post(
            "/api/v1/ai-ux/system-metrics",
            json=invalid_metrics
        )
        
        assert response.status_code == 422  # Validation error

class TestAIUXRoutesIntegration:
    """Integration tests for AI UX routes"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete AI UX optimization workflow through API"""
        user_id = "integration_test_user"
        
        # Step 1: Record user interaction
        interaction_data = {
            "user_id": user_id,
            "session_id": "session_123",
            "action_type": "page_view",
            "feature_used": "dashboard",
            "page_visited": "/dashboard",
            "duration": 5.2,
            "success": True,
            "help_requested": False,
            "metadata": {"device": "desktop"}
        }
        
        with patch('scrollintel.api.routes.ai_ux_routes.get_database_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            response = client.post("/api/v1/ai-ux/interactions", json=interaction_data)
            assert response.status_code == 200
        
        # Step 2: Record system metrics
        metrics_data = {
            "cpu_usage": 0.75,
            "memory_usage": 0.68,
            "error_rate": 0.025,
            "response_time": 950.0,
            "active_users": 180
        }
        
        with patch('scrollintel.api.routes.ai_ux_routes.get_database_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            response = client.post("/api/v1/ai-ux/system-metrics", json=metrics_data)
            assert response.status_code == 200
        
        # Step 3: Analyze user behavior
        behavior_data = {
            "session_duration": 25,
            "clicks_per_minute": 10,
            "pages_visited": 6,
            "errors_encountered": 1,
            "features_used": 4
        }
        
        with patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer') as mock_optimizer:
            mock_analysis = UserBehaviorAnalysis(
                user_id=user_id,
                behavior_pattern=UserBehaviorPattern.CASUAL_USER,
                engagement_score=0.7,
                frustration_indicators=[],
                preferred_features=["dashboard"],
                usage_patterns={"avg_session_duration": 25},
                assistance_needs=[],
                timestamp=datetime.now()
            )
            mock_optimizer.analyze_user_behavior = AsyncMock(return_value=mock_analysis)
            
            response = client.post(
                f"/api/v1/ai-ux/user-behavior/{user_id}/analyze",
                json=behavior_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["behavior_pattern"] == "casual_user"
        
        # Step 4: Get personalized degradation
        with patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer') as mock_optimizer:
            mock_degradation = PersonalizedDegradation(
                user_id=user_id,
                strategy=DegradationStrategy.MODERATE,
                feature_priorities={"dashboard": 8},
                acceptable_delays={"page_load": 3.0},
                fallback_preferences={"search": "cached_results"},
                communication_style="informative",
                timestamp=datetime.now()
            )
            mock_optimizer.create_personalized_degradation = AsyncMock(return_value=mock_degradation)
            
            response = client.get(f"/api/v1/ai-ux/personalized-degradation/{user_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["strategy"] == "moderate"
        
        # Step 5: Get interface optimization
        with patch('scrollintel.api.routes.ai_ux_routes.ai_ux_optimizer') as mock_optimizer:
            mock_optimization = InterfaceOptimization(
                user_id=user_id,
                layout_preferences={"density": "comfortable"},
                interaction_patterns={"keyboard_usage": 0.4},
                performance_requirements={"page_load_time": 3.0},
                accessibility_needs=[],
                optimization_suggestions=["Add progress indicators"],
                timestamp=datetime.now()
            )
            mock_optimizer.optimize_interface = AsyncMock(return_value=mock_optimization)
            
            response = client.get(f"/api/v1/ai-ux/interface-optimization/{user_id}")
            assert response.status_code == 200
            data = response.json()
            assert "Add progress indicators" in data["optimization_suggestions"]
        
        # Step 6: Submit feedback
        feedback_data = {
            "user_id": user_id,
            "optimization_type": "interface_optimization",
            "satisfaction_score": 4.0,
            "feedback_text": "Great improvements!",
            "would_recommend": True
        }
        
        with patch('scrollintel.api.routes.ai_ux_routes.get_database_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            response = client.post("/api/v1/ai-ux/feedback", json=feedback_data)
            assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])