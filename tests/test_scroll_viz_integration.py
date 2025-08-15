"""
Integration tests for ScrollViz API routes.
"""

import pytest
import json
import io
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

from scrollintel.api.gateway import app
from scrollintel.engines.scroll_viz_engine import ChartType, ExportFormat


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    return Mock(id="test_user_123", email="test@example.com", role="analyst")


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "category": ["A", "B", "C", "D"],
        "sales": [100, 200, 150, 300],
        "profit": [20, 40, 30, 60],
        "region": ["North", "South", "East", "West"]
    }


@pytest.fixture
def time_series_data():
    """Time series data for testing."""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    return {
        "date": [d.isoformat() for d in dates],
        "value": np.random.normal(100, 20, 30).tolist()
    }


class TestScrollVizRoutes:
    """Test ScrollViz API routes."""
    
    @patch('scrollintel.security.auth.get_current_user')
    @patch('scrollintel.api.routes.scroll_viz_routes.get_viz_engine')
    def test_recommend_chart_type(self, mock_engine, mock_auth, client, mock_user, sample_data):
        """Test chart type recommendation endpoint."""
        mock_auth.return_value = mock_user
        
        # Mock engine response
        mock_viz_engine = AsyncMock()
        mock_viz_engine.execute.return_value = {
            "data_types": ["categorical", "numerical"],
            "recommended_type": "bar",
            "alternative_types": ["pie", "column"],
            "reasoning": "Bar charts are ideal for comparing categorical data"
        }
        mock_engine.return_value = mock_viz_engine
        
        response = client.post(
            "/api/v1/viz/recommend-chart-type",
            json={
                "data": sample_data,
                "columns": ["category", "sales"]
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "data" in result
        assert result["data"]["recommended_type"] == "bar"
        assert "alternative_types" in result["data"]
    
    @patch('scrollintel.security.auth.get_current_user')
    @patch('scrollintel.api.routes.scroll_viz_routes.get_viz_engine')
    def test_generate_chart(self, mock_engine, mock_auth, client, mock_user, sample_data):
        """Test chart generation endpoint."""
        mock_auth.return_value = mock_user
        
        # Mock engine response
        mock_viz_engine = AsyncMock()
        mock_viz_engine.execute.return_value = {
            "chart_type": "bar",
            "plotly_json": '{"data": [], "layout": {"title": {"text": "Test Chart"}}}',
            "html": "<html>chart content</html>",
            "config": {"chart_type": "bar"}
        }
        mock_engine.return_value = mock_viz_engine
        
        response = client.post(
            "/api/v1/viz/generate-chart",
            json={
                "data": sample_data,
                "chart_type": "bar",
                "x_column": "category",
                "y_column": "sales",
                "title": "Sales by Category"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["data"]["chart_type"] == "bar"
        assert "plotly_json" in result["data"]
        assert "html" in result["data"]
    
    def test_get_chart_types(self, client):
        """Test get supported chart types endpoint."""
        response = client.get("/api/v1/viz/chart-types")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "chart_types" in result["data"]
        assert result["data"]["total_count"] > 0
        
        # Check that common chart types are included
        chart_types = [ct["type"] for ct in result["data"]["chart_types"]]
        assert "bar" in chart_types
        assert "line" in chart_types
        assert "scatter" in chart_types
        assert "pie" in chart_types
    
    def test_get_export_formats(self, client):
        """Test get supported export formats endpoint."""
        response = client.get("/api/v1/viz/export-formats")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "export_formats" in result["data"]
        assert result["data"]["total_count"] > 0
        
        # Check that common formats are included
        formats = [ef["format"] for ef in result["data"]["export_formats"]]
        assert "png" in formats
        assert "svg" in formats
        assert "pdf" in formats
        assert "html" in formats
        assert "json" in formats