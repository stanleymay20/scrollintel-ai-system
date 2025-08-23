"""
Integration tests for Enterprise User Interface
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from scrollintel.api.main import app
from scrollintel.models.database import get_db
from scrollintel.models.enterprise_ui_models import (
    UserProfile, DashboardConfig, NaturalLanguageQuery,
    VisualizationDefinition, UserAlert, SystemMetric
)

client = TestClient(app)

class TestEnterpriseUIIntegration:
    """Test Enterprise UI integration"""
    
    def setup_method(self):
        """Set up test data"""
        self.test_user_id = "test_user_123"
        self.test_profile_id = "test_profile_123"
        
    def test_dashboard_data_executive(self):
        """Test executive dashboard data retrieval"""
        response = client.get(
            f"/api/enterprise-ui/dashboard/executive?time_range=24h",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["role"] == "executive"
        assert "metrics" in data
        assert "alerts" in data
        assert "insights" in data
        assert len(data["metrics"]) > 0
        
        # Check executive-specific metrics
        metric_titles = [m["title"] for m in data["metrics"]]
        assert "Business Value Generated" in metric_titles
        assert "Cost Savings" in metric_titles
        
    def test_dashboard_data_analyst(self):
        """Test analyst dashboard data retrieval"""
        response = client.get(
            f"/api/enterprise-ui/dashboard/analyst?time_range=7d",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["role"] == "analyst"
        metric_titles = [m["title"] for m in data["metrics"]]
        assert "Data Processing Rate" in metric_titles
        assert "Model Accuracy" in metric_titles
        
    def test_dashboard_data_technical(self):
        """Test technical dashboard data retrieval"""
        response = client.get(
            f"/api/enterprise-ui/dashboard/technical?time_range=1h",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["role"] == "technical"
        metric_titles = [m["title"] for m in data["metrics"]]
        assert "CPU Usage" in metric_titles
        assert "Memory Usage" in metric_titles
        
    def test_natural_language_query_revenue(self):
        """Test natural language query about revenue"""
        query_data = {
            "query": "What is our current revenue and how has it changed?",
            "context": {"role": "executive"}
        }
        
        response = client.post(
            "/api/enterprise-ui/natural-language/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "id" in result
        assert result["query"] == query_data["query"]
        assert "revenue" in result["response"].lower()
        assert result["confidence"] > 0.8
        assert result["processing_time"] > 0
        
    def test_natural_language_query_performance(self):
        """Test natural language query about system performance"""
        query_data = {
            "query": "How is our system performing today?",
            "context": {"role": "technical"}
        }
        
        response = client.post(
            "/api/enterprise-ui/natural-language/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "performance" in result["response"].lower()
        assert "data" in result
        assert result["confidence"] > 0.8
        
    def test_natural_language_query_agents(self):
        """Test natural language query about agents"""
        query_data = {
            "query": "Which agents are performing best this month?",
            "context": {"role": "analyst"}
        }
        
        response = client.post(
            "/api/enterprise-ui/natural-language/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "agent" in result["response"].lower()
        assert "data" in result
        assert "top_agents" in result["data"]
        
    def test_visualizations_retrieval(self):
        """Test visualization data retrieval"""
        response = client.get(
            "/api/enterprise-ui/visualizations",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        visualizations = response.json()
        
        assert isinstance(visualizations, list)
        assert len(visualizations) > 0
        
        # Check visualization structure
        viz = visualizations[0]
        assert "id" in viz
        assert "title" in viz
        assert "type" in viz
        assert "data" in viz
        assert "config" in viz
        assert "last_updated" in viz
        assert "is_real_time" in viz
        
    def test_visualizations_refresh(self):
        """Test visualization data refresh"""
        response = client.post(
            "/api/enterprise-ui/visualizations/refresh",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        visualizations = response.json()
        
        assert isinstance(visualizations, list)
        assert len(visualizations) > 0
        
    def test_user_alerts_retrieval(self):
        """Test user alerts retrieval"""
        response = client.get(
            "/api/enterprise-ui/alerts?limit=5",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        alerts = response.json()
        
        assert isinstance(alerts, list)
        assert len(alerts) <= 5
        
        if alerts:
            alert = alerts[0]
            assert "id" in alert
            assert "title" in alert
            assert "message" in alert
            assert "severity" in alert
            assert "timestamp" in alert
            assert "read" in alert
            
    def test_user_alerts_unread_only(self):
        """Test retrieving only unread alerts"""
        response = client.get(
            "/api/enterprise-ui/alerts?unread_only=true",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        alerts = response.json()
        
        # All returned alerts should be unread
        for alert in alerts:
            assert alert["read"] == False
            
    def test_mark_alert_read(self):
        """Test marking an alert as read"""
        # First get alerts to find one to mark as read
        response = client.get(
            "/api/enterprise-ui/alerts",
            headers={"Authorization": "Bearer test_token"}
        )
        alerts = response.json()
        
        if alerts:
            alert_id = alerts[0]["id"]
            
            response = client.put(
                f"/api/enterprise-ui/alerts/{alert_id}/read",
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success"
            
    def test_system_status(self):
        """Test system status retrieval"""
        response = client.get(
            "/api/enterprise-ui/system-status",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        status = response.json()
        
        assert "overall_status" in status
        assert "services" in status
        assert "uptime" in status
        assert "last_updated" in status
        
        # Check service status structure
        services = status["services"]
        assert "api" in services
        assert "database" in services
        assert "agents" in services
        
    def test_invalid_role_dashboard(self):
        """Test dashboard request with invalid role"""
        response = client.get(
            "/api/enterprise-ui/dashboard/invalid_role",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 400
        
    def test_empty_natural_language_query(self):
        """Test natural language query with empty query"""
        query_data = {"query": ""}
        
        response = client.post(
            "/api/enterprise-ui/natural-language/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Should handle empty queries gracefully
        assert response.status_code in [200, 400]
        
    def test_query_processing_time(self):
        """Test that query processing time is reasonable"""
        query_data = {
            "query": "Show me the latest business metrics",
            "context": {"role": "executive"}
        }
        
        response = client.post(
            "/api/enterprise-ui/natural-language/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Processing time should be reasonable (less than 5 seconds)
        assert result["processing_time"] < 5000
        
    def test_dashboard_metrics_structure(self):
        """Test that dashboard metrics have proper structure"""
        response = client.get(
            "/api/enterprise-ui/dashboard/executive",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        for metric in data["metrics"]:
            assert "id" in metric
            assert "title" in metric
            assert "value" in metric
            assert "change" in metric
            assert "trend" in metric
            assert "icon" in metric
            assert "color" in metric
            
            # Trend should be valid
            assert metric["trend"] in ["up", "down", "stable"]
            
    def test_visualization_data_structure(self):
        """Test that visualization data has proper structure"""
        response = client.get(
            "/api/enterprise-ui/visualizations",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        visualizations = response.json()
        
        for viz in visualizations:
            assert "id" in viz
            assert "title" in viz
            assert "type" in viz
            assert "data" in viz
            assert "config" in viz
            
            # Type should be valid
            valid_types = ["bar", "line", "pie", "area", "scatter", "heatmap", "gauge"]
            assert viz["type"] in valid_types
            
            # Data should be a list
            assert isinstance(viz["data"], list)
            
            # Config should be a dict
            assert isinstance(viz["config"], dict)

class TestEnterpriseUIPerformance:
    """Test Enterprise UI performance"""
    
    def test_dashboard_response_time(self):
        """Test dashboard response time"""
        import time
        
        start_time = time.time()
        response = client.get(
            "/api/enterprise-ui/dashboard/executive",
            headers={"Authorization": "Bearer test_token"}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be fast (less than 2 seconds)
        response_time = end_time - start_time
        assert response_time < 2.0
        
    def test_query_response_time(self):
        """Test natural language query response time"""
        import time
        
        query_data = {
            "query": "What are our key performance indicators?",
            "context": {"role": "analyst"}
        }
        
        start_time = time.time()
        response = client.post(
            "/api/enterprise-ui/natural-language/query",
            json=query_data,
            headers={"Authorization": "Bearer test_token"}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Query processing should be reasonably fast
        response_time = end_time - start_time
        assert response_time < 3.0
        
    def test_concurrent_dashboard_requests(self):
        """Test handling concurrent dashboard requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get(
                "/api/enterprise-ui/dashboard/executive",
                headers={"Authorization": "Bearer test_token"}
            )
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])