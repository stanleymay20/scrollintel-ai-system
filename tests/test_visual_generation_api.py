"""
Tests for Visual Generation API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import json

from scrollintel.api.gateway import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_auth_user():
    """Mock authenticated user"""
    return {
        "id": "test_user_123",
        "username": "testuser",
        "permissions": ["visual_generation"],
        "preferences": {}
    }


@pytest.fixture
def mock_generation_result():
    """Mock generation result"""
    return Mock(
        id="gen_123",
        status=Mock(value="completed"),
        content_urls=["https://example.com/image.jpg"],
        generation_time=5.2,
        cost=0.0,
        model_used="stable_diffusion_xl",
        quality_metrics=Mock(overall_score=0.95),
        error_message=None,
        metadata={"resolution": [1024, 1024]}
    )


class TestVisualGenerationAPI:
    """Test visual generation API endpoints"""
    
    @patch('scrollintel.api.routes.visual_generation_routes.get_authenticated_user')
    @patch('scrollintel.api.routes.visual_generation_routes.get_visual_engine')
    @patch('scrollintel.api.routes.visual_generation_routes.visual_rate_limiter')
    def test_generate_image_success(self, mock_rate_limiter, mock_engine, mock_auth, client, mock_auth_user, mock_generation_result):
        """Test successful image generation"""
        # Setup mocks
        mock_auth.return_value = mock_auth_user
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=Mock(allowed=True))
        
        mock_engine_instance = AsyncMock()
        mock_engine_instance.generate_image.return_value = mock_generation_result
        mock_engine.return_value = mock_engine_instance
        
        # Test request
        request_data = {
            "prompt": "A beautiful landscape",
            "resolution": [1024, 1024],
            "style": "photorealistic",
            "quality": "high",
            "num_images": 1
        }
        
        response = client.post(
            "/api/v1/visual/generate/image",
            json=request_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result_id"] == "gen_123"
        assert data["status"] == "completed"
        assert len(data["content_urls"]) == 1
        assert "scrollintel_advantages" in data
    
    @patch('scrollintel.api.routes.visual_generation_routes.get_authenticated_user')
    @patch('scrollintel.api.routes.visual_generation_routes.get_visual_engine')
    @patch('scrollintel.api.routes.visual_generation_routes.visual_rate_limiter')
    def test_generate_video_success(self, mock_rate_limiter, mock_engine, mock_auth, client, mock_auth_user, mock_generation_result):
        """Test successful video generation"""
        # Setup mocks
        mock_auth.return_value = mock_auth_user
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=Mock(allowed=True))
        
        mock_engine_instance = AsyncMock()
        mock_engine_instance.generate_video.return_value = mock_generation_result
        mock_engine.return_value = mock_engine_instance
        
        # Test request
        request_data = {
            "prompt": "A person walking in a park",
            "duration": 5.0,
            "resolution": [1920, 1080],
            "fps": 30,
            "style": "photorealistic",
            "quality": "high",
            "humanoid_generation": True,
            "physics_simulation": True
        }
        
        response = client.post(
            "/api/v1/visual/generate/video",
            json=request_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result_id"] == "gen_123"
        assert "scrollintel_advantages" in data
        assert "4K 60fps + Physics + Humanoids" in data["scrollintel_advantages"]["features"]
    
    def test_generate_image_validation_error(self, client):
        """Test image generation with validation errors"""
        # Invalid resolution
        request_data = {
            "prompt": "Test",
            "resolution": [100, 100],  # Too small
            "style": "photorealistic",
            "quality": "high"
        }
        
        response = client.post(
            "/api/v1/visual/generate/image",
            json=request_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_video_validation_error(self, client):
        """Test video generation with validation errors"""
        # Invalid style
        request_data = {
            "prompt": "Test video",
            "duration": 5.0,
            "resolution": [1920, 1080],
            "style": "invalid_style",  # Invalid style
            "quality": "high"
        }
        
        response = client.post(
            "/api/v1/visual/generate/video",
            json=request_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('scrollintel.api.routes.visual_generation_routes.get_visual_engine')
    def test_get_model_capabilities(self, mock_engine, client):
        """Test getting model capabilities"""
        mock_engine_instance = AsyncMock()
        mock_engine_instance.get_model_capabilities.return_value = {
            "stable_diffusion_xl": {"max_resolution": [2048, 2048]},
            "dalle3": {"max_resolution": [1024, 1024]}
        }
        mock_engine.return_value = mock_engine_instance
        
        with patch('scrollintel.api.routes.visual_generation_routes.get_production_config') as mock_config:
            mock_config.return_value.get_competitive_advantages.return_value = {
                "cost_advantage": "FREE local generation"
            }
            
            response = client.get("/api/v1/visual/models/capabilities")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "model_capabilities" in data
            assert "scrollintel_advantages" in data
            assert "competitive_summary" in data
    
    @patch('scrollintel.api.routes.visual_generation_routes.get_visual_engine')
    def test_get_system_status(self, mock_engine, client):
        """Test getting system status"""
        mock_engine_instance = Mock()
        mock_engine_instance.get_system_status.return_value = {
            "status": "operational",
            "models_loaded": 3,
            "gpu_usage": 45.2
        }
        mock_engine.return_value = mock_engine_instance
        
        with patch('scrollintel.api.routes.visual_generation_routes.get_production_config') as mock_config:
            mock_config.return_value.validate_production_readiness.return_value = {
                "ready": True,
                "checks_passed": 10
            }
            
            response = client.get("/api/v1/visual/system/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "system_status" in data
            assert "production_readiness" in data
            assert "scrollintel_info" in data
    
    def test_get_api_documentation(self, client):
        """Test getting API documentation"""
        response = client.get("/api/v1/visual/docs/api-guide")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "api_documentation" in data
        
        docs = data["api_documentation"]
        assert "overview" in docs
        assert "endpoints" in docs
        assert "competitive_comparison" in docs
        assert "getting_started" in docs
        assert "best_practices" in docs
    
    @patch('scrollintel.api.routes.visual_generation_routes.get_visual_engine')
    def test_estimate_cost(self, mock_engine, client):
        """Test cost estimation"""
        mock_engine_instance = AsyncMock()
        mock_engine_instance.estimate_cost.return_value = 0.0
        mock_engine_instance.estimate_time.return_value = 5.2
        mock_engine.return_value = mock_engine_instance
        
        response = client.get(
            "/api/v1/visual/estimate/cost",
            params={
                "prompt": "Test image",
                "content_type": "image",
                "resolution": "1024x1024"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["estimated_cost"] == 0.0
        assert data["estimated_time"] == 5.2
        assert "scrollintel_advantage" in data
        assert "cost_comparison" in data
    
    def test_get_competitive_analysis(self, client):
        """Test getting competitive analysis"""
        with patch('scrollintel.api.routes.visual_generation_routes.get_production_config') as mock_config:
            mock_config.return_value.get_competitive_advantages.return_value = {
                "cost_advantage": "FREE vs paid competitors",
                "quality_advantage": "98% vs 75% industry average"
            }
            
            response = client.get("/api/v1/visual/competitive/analysis")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "competitive_analysis" in data
            assert "market_position" in data
            assert "roi_analysis" in data
            assert data["recommendation"] == "ScrollIntel is objectively superior to InVideo and all competitors"


class TestVisualGenerationValidation:
    """Test request validation for visual generation"""
    
    def test_image_request_validation(self):
        """Test ImageGenerationRequest validation"""
        from scrollintel.api.routes.visual_generation_routes import ImageGenerationRequest
        
        # Valid request
        valid_request = ImageGenerationRequest(
            prompt="A beautiful landscape",
            resolution=(1024, 1024),
            style="photorealistic",
            quality="high"
        )
        assert valid_request.prompt == "A beautiful landscape"
        assert valid_request.resolution == (1024, 1024)
        
        # Test validation errors
        with pytest.raises(ValueError, match="Resolution must be a tuple"):
            ImageGenerationRequest(
                prompt="Test",
                resolution="invalid",
                style="photorealistic",
                quality="high"
            )
        
        with pytest.raises(ValueError, match="Minimum resolution is 256x256"):
            ImageGenerationRequest(
                prompt="Test",
                resolution=(100, 100),
                style="photorealistic",
                quality="high"
            )
        
        with pytest.raises(ValueError, match="Style must be one of"):
            ImageGenerationRequest(
                prompt="Test",
                resolution=(1024, 1024),
                style="invalid_style",
                quality="high"
            )
    
    def test_video_request_validation(self):
        """Test VideoGenerationRequest validation"""
        from scrollintel.api.routes.visual_generation_routes import VideoGenerationRequest
        
        # Valid request
        valid_request = VideoGenerationRequest(
            prompt="A person walking",
            duration=5.0,
            resolution=(1920, 1080),
            fps=30,
            style="photorealistic",
            quality="high"
        )
        assert valid_request.prompt == "A person walking"
        assert valid_request.duration == 5.0
        
        # Test validation errors
        with pytest.raises(ValueError, match="Minimum video resolution is 480x360"):
            VideoGenerationRequest(
                prompt="Test",
                duration=5.0,
                resolution=(320, 240),
                fps=30,
                style="photorealistic",
                quality="high"
            )
        
        with pytest.raises(ValueError, match="Style must be one of"):
            VideoGenerationRequest(
                prompt="Test",
                duration=5.0,
                resolution=(1920, 1080),
                fps=30,
                style="invalid_style",
                quality="high"
            )


if __name__ == "__main__":
    pytest.main([__file__])