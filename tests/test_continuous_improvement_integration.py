"""
Integration Tests for Continuous Improvement System

This module tests the complete continuous improvement workflow including
API endpoints, database operations, and business logic integration.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, Mock

from scrollintel.api.main import app
from scrollintel.models.continuous_improvement_models import (
    Base, UserFeedback, ABTest, ABTestResult, ModelRetrainingJob, FeatureEnhancement,
    FeedbackType, FeedbackPriority, ABTestStatus, ModelRetrainingStatus,
    FeatureEnhancementStatus
)
from scrollintel.core.database import get_db_session

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_continuous_improvement.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db_session] = override_get_db

class TestContinuousImprovementIntegration:
    """Integration test suite for continuous improvement system"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_database(self):
        """Set up test database"""
        Base.metadata.create_all(bind=engine)
        yield
        Base.metadata.drop_all(bind=engine)
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth(self):
        """Mock authentication for testing"""
        def mock_get_current_user():
            return {"user_id": "test_user", "permissions": ["admin"]}
        
        def mock_require_permissions(permissions):
            def wrapper():
                return {"user_id": "test_user", "permissions": permissions}
            return wrapper
        
        with patch("scrollintel.api.routes.continuous_improvement_routes.get_current_user", mock_get_current_user):
            with patch("scrollintel.api.routes.continuous_improvement_routes.require_permissions", mock_require_permissions):
                yield
    
    @pytest.fixture
    def sample_feedback_payload(self):
        """Sample feedback payload for API testing"""
        return {
            "feedback_type": "user_satisfaction",
            "priority": "medium",
            "title": "Dashboard performance issue",
            "description": "Dashboard loads slowly with large datasets",
            "context": {"page": "dashboard", "dataset_size": "large"},
            "satisfaction_rating": 6,
            "feature_area": "dashboard"
        }
    
    @pytest.fixture
    def sample_ab_test_payload(self):
        """Sample A/B test payload for API testing"""
        return {
            "name": "dashboard_optimization_test",
            "description": "Test dashboard loading optimizations",
            "hypothesis": "Optimized dashboard will improve user satisfaction",
            "feature_area": "dashboard",
            "control_config": {"optimization": False},
            "variant_configs": [{"optimization": True}],
            "traffic_allocation": {"control": 0.5, "variant_1": 0.5},
            "primary_metric": "satisfaction_rating",
            "secondary_metrics": ["session_duration", "bounce_rate"],
            "minimum_sample_size": 100,
            "confidence_level": 0.95
        }
    
    def test_create_feedback_endpoint(self, client, mock_auth, sample_feedback_payload):
        """Test feedback creation endpoint"""
        response = client.post(
            "/api/v1/continuous-improvement/feedback",
            json=sample_feedback_payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == sample_feedback_payload["title"]
        assert data["feedback_type"] == sample_feedback_payload["feedback_type"]
        assert data["user_id"] == "test_user"
        assert "id" in data
        assert "created_at" in data
    
    def test_get_feedback_endpoint(self, client, mock_auth, sample_feedback_payload):
        """Test feedback retrieval endpoint"""
        # Create feedback first
        create_response = client.post(
            "/api/v1/continuous-improvement/feedback",
            json=sample_feedback_payload
        )
        assert create_response.status_code == 200
        
        # Get feedback
        response = client.get("/api/v1/continuous-improvement/feedback")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["title"] == sample_feedback_payload["title"]
    
    def test_get_feedback_with_filters(self, client, mock_auth, sample_feedback_payload):
        """Test feedback retrieval with filters"""
        # Create feedback first
        client.post(
            "/api/v1/continuous-improvement/feedback",
            json=sample_feedback_payload
        )
        
        # Test filtering by feedback type
        response = client.get(
            "/api/v1/continuous-improvement/feedback",
            params={"feedback_type": "user_satisfaction"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert all(item["feedback_type"] == "user_satisfaction" for item in data)
        
        # Test filtering by feature area
        response = client.get(
            "/api/v1/continuous-improvement/feedback",
            params={"feature_area": "dashboard"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert all(item["feature_area"] == "dashboard" for item in data)
    
    def test_create_ab_test_endpoint(self, client, mock_auth, sample_ab_test_payload):
        """Test A/B test creation endpoint"""
        response = client.post(
            "/api/v1/continuous-improvement/ab-tests",
            json=sample_ab_test_payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == sample_ab_test_payload["name"]
        assert data["hypothesis"] == sample_ab_test_payload["hypothesis"]
        assert data["status"] == "draft"
        assert "id" in data
        assert "created_at" in data
    
    def test_start_ab_test_endpoint(self, client, mock_auth, sample_ab_test_payload):
        """Test A/B test start endpoint"""
        # Create A/B test first
        create_response = client.post(
            "/api/v1/continuous-improvement/ab-tests",
            json=sample_ab_test_payload
        )
        test_id = create_response.json()["id"]
        
        # Mock daily users estimation
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._estimate_daily_users", return_value=500):
            with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._initialize_ab_test_monitoring", return_value=None):
                # Start the test
                response = client.post(f"/api/v1/continuous-improvement/ab-tests/{test_id}/start")
        
        assert response.status_code == 200
        data = response.json()
        assert "started successfully" in data["message"]
    
    def test_record_ab_test_result_endpoint(self, client, mock_auth, sample_ab_test_payload):
        """Test A/B test result recording endpoint"""
        # Create and start A/B test
        create_response = client.post(
            "/api/v1/continuous-improvement/ab-tests",
            json=sample_ab_test_payload
        )
        test_id = create_response.json()["id"]
        
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._estimate_daily_users", return_value=500):
            with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._initialize_ab_test_monitoring", return_value=None):
                client.post(f"/api/v1/continuous-improvement/ab-tests/{test_id}/start")
        
        # Record test result
        metrics = {
            "satisfaction_rating": 8.5,
            "session_duration": 300,
            "conversion_event": True,
            "business_value_generated": 150.0
        }
        
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._check_ab_test_completion", return_value=None):
            response = client.post(
                f"/api/v1/continuous-improvement/ab-tests/{test_id}/results",
                params={"variant_name": "variant_1"},
                json=metrics
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "recorded successfully" in data["message"]
        assert "result_id" in data
    
    def test_analyze_ab_test_endpoint(self, client, mock_auth, sample_ab_test_payload):
        """Test A/B test analysis endpoint"""
        # Create A/B test
        create_response = client.post(
            "/api/v1/continuous-improvement/ab-tests",
            json=sample_ab_test_payload
        )
        test_id = create_response.json()["id"]
        
        # Mock analysis results
        mock_analysis = {
            "status": "completed",
            "sample_size": 150,
            "statistical_results": {
                "variant_1": {
                    "sample_size": 75,
                    "mean": 8.0,
                    "lift": 0.143,
                    "p_value": 0.01,
                    "significant": True
                }
            },
            "business_impact": {"revenue_increase": 15000},
            "recommendations": ["Deploy variant_1 to all users"]
        }
        
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine.analyze_ab_test_results", return_value=mock_analysis):
            response = client.get(f"/api/v1/continuous-improvement/ab-tests/{test_id}/analysis")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "statistical_results" in data
        assert "business_impact" in data
    
    def test_schedule_model_retraining_endpoint(self, client, mock_auth):
        """Test model retraining scheduling endpoint"""
        retraining_payload = {
            "model_name": "user_satisfaction_predictor",
            "model_version": "v2.0",
            "agent_type": "bi_agent",
            "training_config": {
                "algorithm": "random_forest",
                "hyperparameters": {"n_estimators": 100}
            },
            "data_sources": [{"type": "feedback", "table": "user_feedback"}],
            "performance_threshold": 0.85,
            "scheduled_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        # Mock validation and baseline metrics
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._validate_model_config", return_value=None):
            with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._get_model_baseline_metrics", return_value={"accuracy": 0.82}):
                with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._schedule_retraining_execution", return_value=None):
                    response = client.post(
                        "/api/v1/continuous-improvement/model-retraining",
                        json=retraining_payload
                    )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == retraining_payload["model_name"]
        assert data["status"] == "scheduled"
        assert "id" in data
    
    def test_create_feature_enhancement_endpoint(self, client, mock_auth):
        """Test feature enhancement creation endpoint"""
        enhancement_payload = {
            "title": "Add real-time notifications",
            "description": "Implement real-time notifications for critical alerts",
            "feature_area": "notifications",
            "enhancement_type": "new_feature",
            "priority": "high",
            "complexity_score": 7,
            "business_value_score": 8.5,
            "user_impact_score": 9.0,
            "technical_feasibility_score": 7.5,
            "estimated_effort_hours": 120,
            "expected_roi": 2.5,
            "requirements": ["Real-time push notifications", "Email notifications"],
            "acceptance_criteria": ["Notifications delivered within 5 seconds"]
        }
        
        # Mock priority calculation and review trigger
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._calculate_enhancement_priority", return_value=8.0):
            with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._trigger_enhancement_review", return_value=None):
                response = client.post(
                    "/api/v1/continuous-improvement/feature-enhancements",
                    json=enhancement_payload
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == enhancement_payload["title"]
        assert data["status"] == "submitted"
        assert data["requester_id"] == "test_user"
        assert "id" in data
    
    def test_get_improvement_recommendations_endpoint(self, client, mock_auth):
        """Test improvement recommendations endpoint"""
        mock_recommendations = [
            {
                "recommendation_id": "rec_001",
                "category": "performance",
                "title": "Optimize dashboard queries",
                "description": "Implement query caching for dashboard",
                "priority": "high",
                "expected_impact": {"performance_improvement": 0.3},
                "implementation_effort": 40,
                "confidence_score": 0.85,
                "supporting_data": {"slow_queries": 15},
                "recommended_actions": ["Add query caching", "Optimize SQL"],
                "timeline": "2 weeks",
                "success_metrics": ["Query response time", "User satisfaction"]
            }
        ]
        
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine.generate_improvement_recommendations", return_value=mock_recommendations):
            response = client.get("/api/v1/continuous-improvement/recommendations")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["category"] == "performance"
        assert data[0]["title"] == "Optimize dashboard queries"
    
    def test_get_improvement_metrics_endpoint(self, client, mock_auth):
        """Test improvement metrics endpoint"""
        mock_metrics = {
            "feedback_metrics": {"total_feedback": 150, "avg_satisfaction": 7.2},
            "ab_test_metrics": {"active_tests": 3, "completed_tests": 12},
            "model_performance_metrics": {"avg_accuracy": 0.85, "models_retrained": 5},
            "feature_adoption_metrics": {"adoption_rate": 0.78, "new_features": 8},
            "business_impact_metrics": {"total_savings": 50000, "revenue_increase": 25000},
            "user_satisfaction_trends": [{"date": "2024-01-01", "satisfaction": 7.5}],
            "system_reliability_trends": [{"date": "2024-01-01", "uptime": 99.9}]
        }
        
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine.get_improvement_metrics", return_value=Mock(**mock_metrics)):
            response = client.get("/api/v1/continuous-improvement/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "feedback_metrics" in data
        assert "ab_test_metrics" in data
        assert "model_performance_metrics" in data
        assert "business_impact_metrics" in data
    
    def test_improvement_dashboard_endpoint(self, client, mock_auth):
        """Test improvement dashboard endpoint"""
        mock_metrics = Mock()
        mock_metrics.feedback_metrics = {"total_feedback": 150}
        mock_metrics.ab_test_metrics = {"active_tests": 3}
        
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine.get_improvement_metrics", return_value=mock_metrics):
            response = client.get("/api/v1/continuous-improvement/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "recent_feedback_count" in data
        assert "active_ab_tests" in data
        assert "recent_enhancements" in data
        assert "time_window_days" in data
    
    def test_resolve_feedback_endpoint(self, client, mock_auth, sample_feedback_payload):
        """Test feedback resolution endpoint"""
        # Create feedback first
        create_response = client.post(
            "/api/v1/continuous-improvement/feedback",
            json=sample_feedback_payload
        )
        feedback_id = create_response.json()["id"]
        
        # Resolve feedback
        response = client.post(
            f"/api/v1/continuous-improvement/feedback/{feedback_id}/resolve",
            params={"resolution_notes": "Issue resolved by optimizing database queries"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "resolved successfully" in data["message"]
    
    def test_update_enhancement_status_endpoint(self, client, mock_auth):
        """Test feature enhancement status update endpoint"""
        # Create enhancement first
        enhancement_payload = {
            "title": "Test enhancement",
            "description": "Test description",
            "feature_area": "test",
            "enhancement_type": "improvement",
            "priority": "medium",
            "complexity_score": 5,
            "business_value_score": 7.0,
            "user_impact_score": 6.0,
            "technical_feasibility_score": 8.0,
            "requirements": ["Test requirement"],
            "acceptance_criteria": ["Test criteria"]
        }
        
        with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._calculate_enhancement_priority", return_value=7.0):
            with patch("scrollintel.engines.continuous_improvement_engine.ContinuousImprovementEngine._trigger_enhancement_review", return_value=None):
                create_response = client.post(
                    "/api/v1/continuous-improvement/feature-enhancements",
                    json=enhancement_payload
                )
        
        enhancement_id = create_response.json()["id"]
        
        # Update status
        response = client.put(
            f"/api/v1/continuous-improvement/feature-enhancements/{enhancement_id}/status",
            params={
                "status": "approved",
                "notes": "Enhancement approved for development"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "status updated" in data["message"]
    
    def test_error_handling(self, client, mock_auth):
        """Test error handling in API endpoints"""
        # Test non-existent feedback
        response = client.post(
            "/api/v1/continuous-improvement/feedback/99999/resolve",
            params={"resolution_notes": "Test"}
        )
        assert response.status_code == 404
        
        # Test non-existent A/B test
        response = client.post("/api/v1/continuous-improvement/ab-tests/99999/start")
        assert response.status_code == 400
        
        # Test invalid A/B test configuration
        invalid_config = {
            "name": "test",
            "hypothesis": "test",
            "feature_area": "test",
            "control_config": {},
            "variant_configs": [{}],
            "traffic_allocation": {"control": 0.3, "variant_1": 0.5},  # Doesn't sum to 1.0
            "primary_metric": "test"
        }
        
        response = client.post(
            "/api/v1/continuous-improvement/ab-tests",
            json=invalid_config
        )
        assert response.status_code == 500  # Should be caught by validation

if __name__ == "__main__":
    pytest.main([__file__])