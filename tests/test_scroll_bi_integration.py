"""
Integration tests for ScrollBI agent - dashboard creation and real-time updates.

Tests the complete workflow of dashboard creation, real-time updates,
alert system, and sharing functionality.
"""

import pytest
import asyncio
import pandas as pd
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from scrollintel.agents.scroll_bi_agent import ScrollBIAgent, DashboardType, AlertType, SharePermission
from scrollintel.core.interfaces import AgentRequest, AgentType, ResponseStatus
from scrollintel.engines.scroll_viz_engine import ScrollVizEngine


class TestScrollBIIntegration:
    """Integration tests for ScrollBI agent."""
    
    @pytest.fixture
    def bi_agent(self):
        """Create ScrollBI agent instance."""
        return ScrollBIAgent()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'revenue': [1000 + i * 10 + (i % 7) * 50 for i in range(100)],
            'customers': [50 + i * 2 + (i % 5) * 10 for i in range(100)],
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'region': ['North', 'South', 'East', 'West'] * 25
        })
    
    @pytest.fixture
    def dashboard_request(self):
        """Create dashboard creation request."""
        return AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Create an executive dashboard with revenue trends and customer metrics",
            context={
                "dashboard_config": {
                    "name": "Executive Dashboard",
                    "description": "High-level business metrics",
                    "dashboard_type": DashboardType.EXECUTIVE.value,
                    "real_time_enabled": True
                }
            },
            priority=1,
            created_at=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_dashboard_creation_workflow(self, bi_agent, sample_data, dashboard_request):
        """Test complete dashboard creation workflow."""
        # Add sample data to request context
        dashboard_request.context["dataset"] = sample_data
        
        # Process dashboard creation request
        response = await bi_agent.process_request(dashboard_request)
        
        # Verify response
        assert response.status == ResponseStatus.SUCCESS
        assert "Dashboard Overview" in response.content
        assert "Executive Dashboard" in response.content
        assert "Visualizations Created" in response.content
        assert "Real-time Configuration" in response.content
        assert response.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_instant_dashboard_generation(self, bi_agent, sample_data):
        """Test instant dashboard generation from data schema."""
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Build a sales dashboard instantly",
            context={
                "dataset": sample_data,
                "dashboard_config": {
                    "name": "Sales Dashboard",
                    "dashboard_type": DashboardType.SALES.value,
                    "layout": "grid"
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Sales Dashboard" in response.content
        assert "Dashboard Configuration" in response.content
        assert "json" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_real_time_dashboard_setup(self, bi_agent):
        """Test real-time dashboard updates configuration."""
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Set up real-time updates for my dashboard",
            context={
                "dashboard_id": "dashboard-123",
                "update_interval": 30,
                "websocket_config": {
                    "endpoint": "ws://localhost:8000/ws"
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Real-time Dashboard Configuration" in response.content
        assert "WebSocket Configuration" in response.content
        assert "Data Streaming Setup" in response.content
        assert "Connection Monitoring" in response.content
    
    @pytest.mark.asyncio
    async def test_alert_system_setup(self, bi_agent):
        """Test alert system configuration."""
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Set up alerts for revenue threshold and customer churn",
            context={
                "dashboard_id": "dashboard-123",
                "alert_config": {
                    "thresholds": {
                        "revenue": 100000,
                        "conversion_rate": 0.05
                    },
                    "notification_channels": ["email", "slack"],
                    "email_recipients": ["admin@company.com"]
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Alert System Configuration" in response.content
        assert "Alert Rules Created" in response.content
        assert "Notification Channels" in response.content
        assert "revenue" in response.content.lower()
        assert "conversion_rate" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_dashboard_sharing_management(self, bi_agent):
        """Test dashboard sharing and permission management."""
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Share dashboard with team members with different permissions",
            context={
                "dashboard_id": "dashboard-123",
                "sharing_config": {
                    "users": [
                        {"email": "viewer@company.com", "permission": SharePermission.VIEW_ONLY.value},
                        {"email": "editor@company.com", "permission": SharePermission.EDIT.value}
                    ],
                    "public_access": False,
                    "expiration_days": 30
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Dashboard Sharing Configuration" in response.content
        assert "Share Links Created" in response.content
        assert "Permission Matrix" in response.content
        assert "Access Controls" in response.content
    
    @pytest.mark.asyncio
    async def test_bi_query_analysis(self, bi_agent):
        """Test BI query analysis and dashboard recommendations."""
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Analyze this BI query and recommend dashboard layout",
            context={
                "bi_query": "SELECT date, SUM(revenue) as total_revenue, COUNT(customers) as customer_count FROM sales GROUP BY date ORDER BY date",
                "data_context": {
                    "tables": ["sales", "customers"],
                    "metrics": ["revenue", "customer_count"]
                },
                "user_preferences": {
                    "layout": "grid",
                    "theme": "dark"
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "BI Query Analysis Report" in response.content
        assert "Query Analysis" in response.content
        assert "Dashboard Recommendations" in response.content
        assert "Chart Suggestions" in response.content
        assert "Layout Options" in response.content
    
    @pytest.mark.asyncio
    async def test_dashboard_optimization(self, bi_agent):
        """Test dashboard performance optimization."""
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Optimize my dashboard performance",
            context={
                "dashboard_id": "dashboard-123",
                "performance_data": {
                    "load_time": 5.2,
                    "chart_render_time": 2.1,
                    "data_fetch_time": 1.8
                },
                "usage_analytics": {
                    "daily_users": 150,
                    "peak_concurrent_users": 25,
                    "most_used_charts": ["revenue_trend", "kpi_cards"]
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Dashboard Optimization Report" in response.content
        assert "Performance Analysis" in response.content
        assert "Optimization Recommendations" in response.content
        assert "User Experience Recommendations" in response.content
    
    @pytest.mark.asyncio
    async def test_multiple_data_sources(self, bi_agent, sample_data):
        """Test dashboard creation with multiple data sources."""
        # Create additional data source
        financial_data = pd.DataFrame({
            'month': pd.date_range('2024-01-01', periods=12, freq='M'),
            'profit': [10000 + i * 1000 for i in range(12)],
            'expenses': [8000 + i * 800 for i in range(12)]
        })
        
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Create financial dashboard with multiple data sources",
            context={
                "dataset": sample_data,
                "additional_datasets": {
                    "financial": financial_data
                },
                "dashboard_config": {
                    "name": "Financial Dashboard",
                    "dashboard_type": DashboardType.FINANCIAL.value
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Financial Dashboard" in response.content
        assert "Data Sources" in response.content
    
    @pytest.mark.asyncio
    async def test_dashboard_templates(self, bi_agent, sample_data):
        """Test different dashboard templates."""
        templates = [
            (DashboardType.EXECUTIVE, "executive summary"),
            (DashboardType.SALES, "sales performance"),
            (DashboardType.OPERATIONAL, "operational metrics"),
            (DashboardType.MARKETING, "marketing analytics")
        ]
        
        for dashboard_type, prompt_text in templates:
            request = AgentRequest(
                id=f"req-{uuid4()}",
                user_id=f"user-{uuid4()}",
                agent_id="scroll-bi",
                prompt=f"Create {prompt_text} dashboard",
                context={
                    "dataset": sample_data,
                    "dashboard_config": {
                        "dashboard_type": dashboard_type.value
                    }
                },
                priority=1,
                created_at=datetime.now()
            )
            
            response = await bi_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert dashboard_type.value in response.content.lower() or prompt_text in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, bi_agent):
        """Test error handling in dashboard creation."""
        # Test with invalid data
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Create dashboard",
            context={
                "dataset": "invalid_data",  # Invalid data type
                "dashboard_config": {}
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        # Should handle error gracefully
        assert response.status == ResponseStatus.ERROR
        assert "Error" in response.content
        assert response.error_message is not None
    
    @pytest.mark.asyncio
    async def test_websocket_configuration(self, bi_agent):
        """Test WebSocket configuration for real-time updates."""
        dashboard_id = "test-dashboard-123"
        context = {
            "update_interval": 15,
            "max_connections": 50
        }
        
        # Test WebSocket configuration
        websocket_config = await bi_agent._configure_websockets(dashboard_id, context)
        
        assert "endpoint" in websocket_config
        assert dashboard_id in websocket_config["endpoint"]
        assert websocket_config["max_connections"] == 100  # Default value
        assert websocket_config["heartbeat_interval"] == 30
    
    @pytest.mark.asyncio
    async def test_alert_rule_creation(self, bi_agent):
        """Test alert rule creation and management."""
        requirements = {
            "thresholds": {
                "revenue": 100000,
                "conversion_rate": 0.05,
                "customer_satisfaction": 4.0
            },
            "notification_channels": ["email", "slack"]
        }
        
        alert_rules = await bi_agent._create_alert_rules(requirements)
        
        assert len(alert_rules) == 3
        assert all(rule.is_active for rule in alert_rules)
        assert all(rule.notification_channels == ["email", "slack"] for rule in alert_rules)
        
        # Check specific rules
        revenue_rule = next((r for r in alert_rules if r.metric_name == "revenue"), None)
        assert revenue_rule is not None
        assert revenue_rule.threshold_value == 100000
        assert revenue_rule.alert_type == AlertType.THRESHOLD_EXCEEDED
    
    @pytest.mark.asyncio
    async def test_dashboard_specification_generation(self, bi_agent, sample_data):
        """Test dashboard specification generation."""
        config = {
            "name": "Test Dashboard",
            "description": "Test dashboard description",
            "dashboard_type": DashboardType.ANALYTICAL.value,
            "layout": "grid",
            "theme": "light",
            "refresh_interval": 60,
            "auto_refresh": True,
            "real_time_enabled": False,
            "charts": [
                {"type": "bar", "title": "Revenue by Category"},
                {"type": "line", "title": "Revenue Trend"}
            ],
            "filters": {},
            "alert_config": {}
        }
        
        data_sources = [{
            "id": "primary_dataset",
            "name": "Primary Dataset",
            "type": "dataframe",
            "data": sample_data,
            "schema": bi_agent._analyze_dataframe_schema(sample_data)
        }]
        
        dashboard_spec = await bi_agent._generate_dashboard_spec(config, data_sources)
        
        assert dashboard_spec["name"] == "Test Dashboard"
        assert dashboard_spec["dashboard_type"] == DashboardType.ANALYTICAL.value
        assert dashboard_spec["layout"] == "grid"
        assert len(dashboard_spec["charts"]) == 2
        assert dashboard_spec["data_sources"] == ["primary_dataset"]
    
    @pytest.mark.asyncio
    async def test_visualization_creation(self, bi_agent, sample_data):
        """Test visualization creation using ScrollViz engine."""
        dashboard_spec = {
            "id": "test-dashboard",
            "charts": [
                {
                    "id": "chart-1",
                    "type": "bar",
                    "title": "Revenue by Category",
                    "position": {"x": 0, "y": 0, "w": 6, "h": 4},
                    "data_source_id": "primary_dataset",
                    "config": {
                        "x_column": "category",
                        "y_column": "revenue"
                    }
                }
            ]
        }
        
        data_sources = [{
            "id": "primary_dataset",
            "name": "Primary Dataset",
            "type": "dataframe",
            "data": sample_data,
            "schema": bi_agent._analyze_dataframe_schema(sample_data)
        }]
        
        visualizations = await bi_agent._create_dashboard_visualizations(dashboard_spec, data_sources)
        
        assert len(visualizations) == 1
        assert visualizations[0]["id"] == "chart-1"
        assert visualizations[0]["type"] == "bar"
        assert visualizations[0]["title"] == "Revenue by Category"
        assert "config" in visualizations[0]
    
    @pytest.mark.asyncio
    async def test_data_schema_analysis(self, bi_agent, sample_data):
        """Test data schema analysis for dashboard creation."""
        schema = bi_agent._analyze_dataframe_schema(sample_data)
        
        assert schema["date"] == "datetime"
        assert schema["revenue"] == "numerical"
        assert schema["customers"] == "numerical"
        assert schema["category"] == "categorical"
        assert schema["region"] == "categorical"
    
    @pytest.mark.asyncio
    async def test_performance_recommendations(self, bi_agent, sample_data):
        """Test performance optimization recommendations."""
        dashboard_spec = {
            "charts": [{"type": "bar"} for _ in range(15)],  # Many charts
            "refresh_interval": 1  # Very frequent refresh
        }
        
        data_sources = [{
            "data": sample_data,
            "schema": bi_agent._analyze_dataframe_schema(sample_data)
        }]
        
        recommendations = await bi_agent._generate_performance_recommendations(dashboard_spec, data_sources)
        
        assert "reducing the number of charts" in recommendations or "lazy loading" in recommendations
        assert "frequent refresh intervals" in recommendations
    
    @pytest.mark.asyncio
    async def test_health_check(self, bi_agent):
        """Test agent health check."""
        is_healthy = await bi_agent.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_capabilities(self, bi_agent):
        """Test agent capabilities."""
        capabilities = bi_agent.get_capabilities()
        
        capability_names = [cap.name for cap in capabilities]
        assert "instant_dashboard_creation" in capability_names
        assert "real_time_dashboard_updates" in capability_names
        assert "threshold_alerts" in capability_names
        assert "dashboard_sharing" in capability_names
        assert "bi_query_analysis" in capability_names
        assert "dashboard_optimization" in capability_names
    
    @pytest.mark.asyncio
    async def test_concurrent_dashboard_creation(self, bi_agent, sample_data):
        """Test concurrent dashboard creation requests."""
        requests = []
        for i in range(3):
            request = AgentRequest(
                id=f"req-{uuid4()}",
                user_id=f"user-{uuid4()}",
                agent_id="scroll-bi",
                prompt=f"Create dashboard {i+1}",
                context={
                    "dataset": sample_data,
                    "dashboard_config": {
                        "name": f"Dashboard {i+1}",
                        "dashboard_type": DashboardType.CUSTOM.value
                    }
                },
                priority=1,
                created_at=datetime.now()
            )
            requests.append(request)
        
        # Process requests concurrently
        tasks = [bi_agent.process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses are successful
        assert all(response.status == ResponseStatus.SUCCESS for response in responses)
        assert all("Dashboard Overview" in response.content for response in responses)
    
    @pytest.mark.asyncio
    async def test_dashboard_with_filters(self, bi_agent, sample_data):
        """Test dashboard creation with filters."""
        request = AgentRequest(
            id=f"req-{uuid4()}",
            user_id=f"user-{uuid4()}",
            agent_id="scroll-bi",
            prompt="Create dashboard with date and category filters",
            context={
                "dataset": sample_data,
                "dashboard_config": {
                    "name": "Filtered Dashboard",
                    "filters": {
                        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
                        "category": {"values": ["A", "B", "C"], "multi_select": True}
                    }
                }
            },
            priority=1,
            created_at=datetime.now()
        )
        
        response = await bi_agent.process_request(request)
        
        assert response.status == ResponseStatus.SUCCESS
        assert "Filtered Dashboard" in response.content
        assert "filter" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_ai_assistance_fallback(self, bi_agent):
        """Test AI assistance fallback when OpenAI is not available."""
        with patch('scrollintel.agents.scroll_bi_agent.HAS_OPENAI', False):
            request = AgentRequest(
                id=f"req-{uuid4()}",
                user_id=f"user-{uuid4()}",
                agent_id="scroll-bi",
                prompt="Help me with dashboard best practices",
                context={},
                priority=1,
                created_at=datetime.now()
            )
            
            response = await bi_agent.process_request(request)
            
            assert response.status == ResponseStatus.SUCCESS
            assert "AI assistance is not available" in response.content or "Best Practices" in response.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])