"""
Tests for visualization API routes.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.api.routes.visualization_routes import router
from scrollintel.models.visualization_models import (
    ChartConfiguration, ChartType, ExportFormat, DataFilter, FilterOperator
)


@pytest.fixture
def client():
    """Create a test client."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"category": "A", "sales": 1000, "date": "2024-01-01"},
        {"category": "B", "sales": 1200, "date": "2024-01-02"},
        {"category": "C", "sales": 800, "date": "2024-01-03"}
    ]


@pytest.fixture
def chart_config():
    """Sample chart configuration."""
    return {
        "chart_type": "bar",
        "title": "Test Chart",
        "x_axis": "category",
        "y_axis": "sales",
        "color_scheme": "default",
        "width": 800,
        "height": 400,
        "interactive": True,
        "show_legend": True,
        "show_grid": True,
        "show_tooltip": True
    }


class TestVisualizationRoutes:
    """Test cases for visualization API routes."""
    
    @patch('scrollintel.api.routes.visualization_routes.viz_engine')
    def test_create_chart_success(self, mock_viz_engine, client, sample_data, chart_config):
        """Test successful chart creation."""
        # Mock the visualization engine response
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = Mock()
        mock_response.data.id = "test-chart-1"
        mock_response.data.name = "Test Chart"
        mock_response.message = "Chart created successfully"
        
        mock_viz_engine.create_visualization = AsyncMock(return_value=mock_response)
        
        response = client.post(
            "/api/visualization/charts/create",
            json={
                "data": sample_data,
                "config": chart_config
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["message"] == "Chart created successfully"
        
        # Verify the engine was called correctly
        mock_viz_engine.create_visualization.assert_called_once()
    
    @patch('scrollintel.api.routes.visualization_routes.viz_engine')
    def test_create_chart_with_filters(self, mock_viz_engine, client, sample_data, chart_config):
        """Test chart creation with filters."""
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = Mock()
        mock_response.message = "Chart created successfully"
        
        mock_viz_engine.create_visualization = AsyncMock(return_value=mock_response)
        
        filters = [
            {
                "field": "category",
                "operator": "equals",
                "value": "A"
            }
        ]
        
        response = client.post(
            "/api/visualization/charts/create",
            json={
                "data": sample_data,
                "config": chart_config,
                "filters": filters
            }
        )
        
        assert response.status_code == 200
        
        # Verify filters were passed to the engine
        call_args = mock_viz_engine.create_visualization.call_args
        assert call_args[0][2] is not None  # filters parameter
    
    @patch('scrollintel.api.routes.visualization_routes.viz_engine')
    def test_create_interactive_chart(self, mock_viz_engine, client, sample_data, chart_config):
        """Test interactive chart creation."""
        mock_viz_engine.create_interactive_chart = AsyncMock(return_value={
            "success": True,
            "chart": {"id": "test-chart"},
            "interactiveConfig": {"zoom": True}
        })
        
        interactive_features = {
            "zoom": True,
            "pan": True,
            "brush": False,
            "crossfilter": False,
            "drill_down": {
                "enabled": True,
                "levels": ["category", "region"],
                "default_level": 0,
                "show_breadcrumbs": True
            }
        }
        
        response = client.post(
            "/api/visualization/charts/interactive",
            json={
                "data": sample_data,
                "config": chart_config,
                "interactive_features": interactive_features
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "chart" in result
        assert "interactiveConfig" in result
    
    @patch('scrollintel.api.routes.visualization_routes.viz_engine')
    def test_get_chart_suggestions(self, mock_viz_engine, client, sample_data):
        """Test chart suggestions endpoint."""
        mock_suggestions = [
            {
                "type": "bar",
                "title": "Sales by Category",
                "x_axis": "category",
                "y_axis": "sales",
                "confidence": 0.9
            },
            {
                "type": "pie",
                "title": "Sales Distribution",
                "x_axis": "category",
                "y_axis": "sales",
                "confidence": 0.7
            }
        ]
        
        mock_viz_engine.get_chart_suggestions.return_value = mock_suggestions
        
        response = client.post(
            "/api/visualization/charts/suggestions",
            json=sample_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "suggestions" in result
        assert len(result["suggestions"]) == 2
        assert result["count"] == 2
    
    @patch('scrollintel.api.routes.visualization_routes.viz_engine')
    def test_get_chart_statistics(self, mock_viz_engine, client, sample_data, chart_config):
        """Test chart statistics endpoint."""
        mock_stats = {
            "total_records": 3,
            "columns": ["category", "sales", "date"],
            "numeric_stats": {
                "sales": {"mean": 1000, "std": 200, "min": 800, "max": 1200}
            },
            "categorical_stats": {
                "category": {"unique_values": 3, "most_common": {"A": 1}}
            }
        }
        
        mock_viz_engine.calculate_chart_statistics.return_value = mock_stats
        
        response = client.post(
            "/api/visualization/charts/statistics",
            json={
                "data": sample_data,
                "config": chart_config
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "statistics" in result
        assert result["statistics"]["total_records"] == 3
    
    @patch('scrollintel.api.routes.visualization_routes.export_engine')
    def test_export_visualization(self, mock_export_engine, client):
        """Test visualization export."""
        mock_export_engine.export_visualization = AsyncMock(return_value={
            "success": True,
            "filename": "export_20240101_120000.pdf",
            "filepath": "/tmp/export_20240101_120000.pdf",
            "content_type": "application/pdf",
            "size": 1024
        })
        
        export_request = {
            "format": "pdf",
            "chart_ids": ["chart-1", "chart-2"],
            "include_data": True,
            "include_metadata": True,
            "custom_title": "Test Export"
        }
        
        response = client.post(
            "/api/visualization/export",
            json=export_request
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "export_id" in result
        assert "download_url" in result
        assert result["content_type"] == "application/pdf"
        assert result["size"] == 1024
    
    @patch('scrollintel.api.routes.visualization_routes.export_engine')
    def test_export_visualization_failure(self, mock_export_engine, client):
        """Test visualization export failure."""
        mock_export_engine.export_visualization = AsyncMock(return_value={
            "success": False,
            "error": "Export failed"
        })
        
        export_request = {
            "format": "pdf",
            "chart_ids": ["chart-1"]
        }
        
        response = client.post(
            "/api/visualization/export",
            json=export_request
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "Export failed" in result["detail"]
    
    @patch('scrollintel.api.routes.visualization_routes.dashboard_engine')
    def test_create_dashboard(self, mock_dashboard_engine, client):
        """Test dashboard creation."""
        mock_dashboard_engine.create_dashboard = AsyncMock(return_value={
            "success": True,
            "dashboard": {
                "id": "dashboard-1",
                "name": "Test Dashboard"
            },
            "message": "Dashboard created successfully"
        })
        
        response = client.post(
            "/api/visualization/dashboards/create",
            params={
                "name": "Test Dashboard",
                "description": "A test dashboard",
                "template": "executive"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["message"] == "Dashboard created successfully"
    
    @patch('scrollintel.api.routes.visualization_routes.dashboard_engine')
    def test_update_dashboard_layout(self, mock_dashboard_engine, client):
        """Test dashboard layout update."""
        mock_dashboard_engine.update_dashboard_layout = AsyncMock(return_value={
            "success": True,
            "dashboard": {"id": "dashboard-1"},
            "message": "Layout updated successfully"
        })
        
        layout = [
            {"i": "chart-1", "x": 0, "y": 0, "w": 6, "h": 4},
            {"i": "chart-2", "x": 6, "y": 0, "w": 6, "h": 4}
        ]
        
        response = client.put(
            "/api/visualization/dashboards/dashboard-1/layout",
            json=layout
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
    
    @patch('scrollintel.api.routes.visualization_routes.dashboard_engine')
    def test_add_chart_to_dashboard(self, mock_dashboard_engine, client):
        """Test adding chart to dashboard."""
        mock_dashboard_engine.add_chart_to_dashboard = AsyncMock(return_value={
            "success": True,
            "layout_item": {
                "i": "chart-test-1",
                "x": 0,
                "y": 0,
                "w": 6,
                "h": 4
            },
            "message": "Chart added successfully"
        })
        
        response = client.post(
            "/api/visualization/dashboards/dashboard-1/charts",
            params={
                "chart_id": "test-1"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "layout_item" in result
    
    @patch('scrollintel.api.routes.visualization_routes.dashboard_engine')
    def test_apply_dashboard_filters(self, mock_dashboard_engine, client):
        """Test applying filters to dashboard."""
        mock_dashboard_engine.apply_dashboard_filters = AsyncMock(return_value={
            "success": True,
            "filters_applied": 2,
            "message": "Filters applied successfully"
        })
        
        filters = [
            {
                "field": "category",
                "operator": "equals",
                "value": "A"
            },
            {
                "field": "sales",
                "operator": "greater_than",
                "value": 1000
            }
        ]
        
        response = client.post(
            "/api/visualization/dashboards/dashboard-1/filters",
            json=filters
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["filters_applied"] == 2
    
    @patch('scrollintel.api.routes.visualization_routes.dashboard_engine')
    def test_get_dashboard_templates(self, mock_dashboard_engine, client):
        """Test getting dashboard templates."""
        mock_templates = [
            {
                "id": "executive",
                "name": "Executive Dashboard",
                "description": "High-level KPIs",
                "widget_count": 6
            },
            {
                "id": "analytical",
                "name": "Analytical Dashboard",
                "description": "Detailed analysis",
                "widget_count": 5
            }
        ]
        
        mock_dashboard_engine.get_dashboard_templates.return_value = mock_templates
        
        response = client.get("/api/visualization/dashboards/templates")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "templates" in result
        assert len(result["templates"]) == 2
        assert result["count"] == 2
    
    @patch('scrollintel.api.routes.visualization_routes.viz_engine')
    def test_filter_data(self, mock_viz_engine, client, sample_data):
        """Test data filtering endpoint."""
        filtered_data = [{"category": "A", "sales": 1000, "date": "2024-01-01"}]
        mock_viz_engine._apply_filters.return_value = filtered_data
        
        filters = [
            {
                "field": "category",
                "operator": "equals",
                "value": "A"
            }
        ]
        
        response = client.post(
            "/api/visualization/data/filter",
            json={
                "data": sample_data,
                "filters": filters
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "data" in result
        assert result["original_count"] == 3
        assert result["filtered_count"] == 1
        assert result["filters_applied"] == 1
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/visualization/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert result["service"] == "visualization"
        assert "features" in result
        
        features = result["features"]
        assert features["chart_creation"] is True
        assert features["interactive_charts"] is True
        assert features["export_pdf"] is True
        assert features["export_excel"] is True
        assert features["dashboard_customization"] is True
        assert features["print_layouts"] is True
    
    @patch('scrollintel.api.routes.visualization_routes.viz_engine')
    def test_create_chart_error_handling(self, mock_viz_engine, client, sample_data, chart_config):
        """Test error handling in chart creation."""
        mock_viz_engine.create_visualization = AsyncMock(side_effect=Exception("Test error"))
        
        response = client.post(
            "/api/visualization/charts/create",
            json={
                "data": sample_data,
                "config": chart_config
            }
        )
        
        assert response.status_code == 500
        result = response.json()
        assert "Failed to create chart" in result["detail"]
    
    @patch('scrollintel.api.routes.visualization_routes.viz_engine')
    def test_interactive_chart_failure(self, mock_viz_engine, client, sample_data, chart_config):
        """Test interactive chart creation failure."""
        mock_viz_engine.create_interactive_chart = AsyncMock(return_value={
            "success": False,
            "error": "Interactive chart creation failed"
        })
        
        interactive_features = {
            "zoom": True,
            "pan": True
        }
        
        response = client.post(
            "/api/visualization/charts/interactive",
            json={
                "data": sample_data,
                "config": chart_config,
                "interactive_features": interactive_features
            }
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "Interactive chart creation failed" in result["detail"]
    
    @patch('os.path.exists')
    def test_download_export_file_not_found(self, mock_exists, client):
        """Test download when file doesn't exist."""
        mock_exists.return_value = False
        
        response = client.get("/api/visualization/download/nonexistent.pdf")
        
        assert response.status_code == 404
        result = response.json()
        assert result["detail"] == "File not found"
    
    @patch('scrollintel.api.routes.visualization_routes.dashboard_engine')
    def test_dashboard_operation_failure(self, mock_dashboard_engine, client):
        """Test dashboard operation failure."""
        mock_dashboard_engine.create_dashboard = AsyncMock(return_value={
            "success": False,
            "error": "Dashboard creation failed"
        })
        
        response = client.post(
            "/api/visualization/dashboards/create",
            params={"name": "Test Dashboard"}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "Dashboard creation failed" in result["detail"]