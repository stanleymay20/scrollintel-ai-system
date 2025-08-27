"""
Test visual generation API integration with main ScrollIntel API
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json

from scrollintel.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_auth():
    """Mock authentication"""
    with patch("scrollintel.security.auth.get_current_user") as mock:
        mock.return_value = {
            "id": "test_user",
            "email": "test@example.com",
            "role": "user",
            "permissions": ["visual_generation", "api_access"]
        }
        yield mock


@pytest.fixture
def mock_visual_engine():
    """Mock visual generation engine"""
    with patch("scrollintel.api.routes.visual_generation_routes.get_visual_engine") as mock:
        engine = AsyncMock()
        engine.generate_image.return_value = AsyncMock(
            id="test_result_123",
            status=AsyncMock(value="completed"),
            content_urls=["http://example.com/image.jpg"],
            generation_time=2.5,
            cost=0.0,
            model_used="stable_diffusion_xl",
            quality_metrics=AsyncMock(overall_score=0.95),
            error_message=None
        )
        engine.generate_video.return_value = AsyncMock(
            id="test_video_123",
            status=AsyncMock(value="completed"),
            content_urls=["http://example.com/video.mp4"],
            generation_time=15.2,
            cost=0.0,
            model_used="proprietary_video_engine",
            quality_metrics=AsyncMock(overall_score=0.98),
            metadata={"resolution": "1920x1080", "fps": 30},
            error_message=None
        )
        mock.return_value = engine
        yield engine


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter"""
    with patch("scrollintel.api.routes.visual_generation_routes.visual_rate_limiter") as mock:
        mock.check_rate_limit = AsyncMock()
        yield mock


def test_visual_generation_routes_included(client):
    """Test that visual generation routes are included in main API"""
    # Test that the visual generation endpoints are accessible
    response = client.get("/api/v1/visual/system/status")
    # Should get 401 without auth, not 404
    assert response.status_code in [401, 422]  # 422 for missing auth header


def test_image_generation_endpoint_integration(client, mock_auth, mock_visual_engine, mock_rate_limiter):
    """Test image generation endpoint integration"""
    with patch("scrollintel.api.routes.visual_generation_routes.apply_visual_rate_limit") as mock_rate:
        mock_rate.return_value = {"id": "test_user", "permissions": ["visual_generation"]}
        
        response = client.post(
            "/api/v1/visual/generate/image",
            json={
                "prompt": "A beautiful landscape",
                "resolution": [1024, 1024],
                "num_images": 1,
                "style": "photorealistic",
                "quality": "high"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result_id" in data
        assert "scrollintel_advantages" in data


def test_video_generation_endpoint_integration(client, mock_auth, mock_visual_engine, mock_rate_limiter):
    """Test video generation endpoint integration"""
    with patch("scrollintel.api.routes.visual_generation_routes.apply_visual_rate_limit") as mock_rate:
        mock_rate.return_value = {"id": "test_user", "permissions": ["visual_generation"]}
        
        response = client.post(
            "/api/v1/visual/generate/video",
            json={
                "prompt": "A person walking in a park",
                "duration": 5.0,
                "resolution": [1920, 1080],
                "fps": 30,
                "style": "photorealistic",
                "quality": "high",
                "humanoid_generation": True,
                "physics_simulation": True
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result_id" in data
        assert "scrollintel_advantages" in data


def test_authentication_required(client):
    """Test that authentication is required for visual generation endpoints"""
    response = client.post(
        "/api/v1/visual/generate/image",
        json={
            "prompt": "Test prompt",
            "resolution": [512, 512]
        }
    )
    
    # Should require authentication
    assert response.status_code in [401, 422]


def test_permissions_required(client, mock_auth, mock_visual_engine):
    """Test that proper permissions are required"""
    # Mock user without visual generation permissions
    with patch("scrollintel.security.auth.get_current_user") as mock:
        mock.return_value = {
            "id": "test_user",
            "email": "test@example.com",
            "role": "user",
            "permissions": ["basic_access"]  # No visual_generation permission
        }
        
        response = client.post(
            "/api/v1/visual/generate/image",
            json={
                "prompt": "Test prompt",
                "resolution": [512, 512]
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Should be forbidden due to lack of permissions
        assert response.status_code == 403


def test_system_status_endpoint(client, mock_auth, mock_visual_engine):
    """Test system status endpoint integration"""
    mock_visual_engine.get_system_status.return_value = {
        "status": "healthy",
        "models_loaded": 5,
        "queue_length": 0
    }
    
    with patch("scrollintel.api.routes.visual_generation_routes.get_production_config") as mock_config:
        mock_config.return_value.validate_production_readiness.return_value = {
            "ready": True,
            "issues": []
        }
        
        response = client.get(
            "/api/v1/visual/system/status",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "system_status" in data
        assert "scrollintel_info" in data


def test_model_capabilities_endpoint(client, mock_auth, mock_visual_engine):
    """Test model capabilities endpoint integration"""
    mock_visual_engine.get_model_capabilities.return_value = {
        "image_models": ["stable_diffusion_xl", "dalle3"],
        "video_models": ["proprietary_engine"],
        "enhancement_models": ["real_esrgan", "gfpgan"]
    }
    
    with patch("scrollintel.api.routes.visual_generation_routes.get_production_config") as mock_config:
        mock_config.return_value.get_competitive_advantages.return_value = {
            "cost_advantage": "FREE local generation",
            "quality_advantage": "98% vs 75% industry average",
            "performance_advantage": "10x faster"
        }
        
        response = client.get(
            "/api/v1/visual/models/capabilities",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "model_capabilities" in data
        assert "scrollintel_advantages" in data
        assert "competitive_summary" in data


def test_cost_estimation_endpoint(client, mock_auth, mock_visual_engine):
    """Test cost estimation endpoint integration"""
    mock_visual_engine.estimate_cost.return_value = 0.0
    mock_visual_engine.estimate_time.return_value = 3.5
    
    response = client.get(
        "/api/v1/visual/estimate/cost?prompt=test&content_type=image&resolution=1024x1024",
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["estimated_cost"] == 0.0
    assert "scrollintel_advantage" in data
    assert "cost_comparison" in data


def test_user_generations_endpoint(client, mock_auth, mock_visual_engine):
    """Test user generations history endpoint"""
    mock_visual_engine.get_user_generations.return_value = [
        AsyncMock(
            id="gen_1",
            prompt="Test prompt 1",
            content_type="image",
            status=AsyncMock(value="completed"),
            created_at=AsyncMock(isoformat=lambda: "2024-01-01T00:00:00"),
            generation_time=2.5,
            cost=0.0,
            model_used="stable_diffusion_xl",
            content_urls=["http://example.com/image1.jpg"]
        )
    ]
    
    response = client.get(
        "/api/v1/visual/user/generations",
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "generations" in data
    assert len(data["generations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__])