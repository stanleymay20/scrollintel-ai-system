"""
Integration tests for ScrollIntel Real-Time Monitoring API
Tests the complete monitoring system including API routes
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from scrollintel.api.routes.real_time_monitoring_routes import router
from scrollintel.core.real_time_monitoring import (
    AgentPerformanceMetrics,
    BusinessImpactMetrics,
    SystemHealthMetrics
)

# Create a test FastAPI app
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)

client = TestClient(app)

class TestRealTimeMonitoringAPI:
    """Test real-time monitoring API endpoints"""
    
    def test_get_agent_performance_metrics(self):
        """Test agent performance metrics endpoint"""
        # Mock agent metrics
        mock_agent_metrics = [
            AgentPerformanceMetrics(
                agent_id="test-agent-1",
                agent_type="data_scientist",
                status="active",
                cpu_usage=45.0,
                memory_usage=60.0,
                request_count=100,
                success_rate=95.5,
                avg_response_time=1.2,
                error_count=5,
                last_activity=datetime.utcnow(),
                uptime_seconds=3600.0,
                throughput_per_minute=50.0,
                business_value_generated=5000.0,
                cost_savings=2500.0
            )
        ]
        
        with patch('scrollintel.api.routes.real_time_monitoring_routes.real_time_monitor') as mock_monitor:
            mock_monitor.get_all_agent_metrics.return_value = mock_agent_metrics
            
            response = client.get("/api/v1/monitoring/agents")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "data" in data
            assert "agents" in data["data"]
            assert "summary" in data["data"]
            
            agents = data["data"]["agents"]
            assert len(agents) == 1
            
            agent = agents[0]
            assert agent["agent_id"] == "test-agent-1"
            assert agent["agent_type"] == "data_scientist"
            assert agent["status"] == "active"
            assert agent["performance"]["success_rate"] == 95.5
            assert agent["business_metrics"]["business_value_generated"] == 5000.0
            
            summary = data["data"]["summary"]
            assert summary["total_agents"] == 1
            assert summary["active_agents"] == 1
    
    def test_get_agent_details(self):
        """Test individual agent details endpoint"""
        agent_id = "test-agent-1"
        
        mock_agent_metrics = AgentPerformanceMetrics(
            agent_id=agent_id,
            agent_type="ml_engineer",
            status="active",
            cpu_usage=55.0,
            memory_usage=70.0,
            request_count=150,
            success_rate=97.2,
            avg_response_time=0.8,
            error_count=3,
            last_activity=datetime.utcnow(),
            uptime_seconds=7200.0,
            throughput_per_minute=75.0,
            business_value_generated=8000.0,
            cost_savings=4000.0
        )
        
        mock_history = [
            {
                "timestamp": datetime.utcnow(),
                "metrics": {
                    "success_rate": 97.2,
                    "avg_response_time": 0.8,
                    "business_value_generated": 8000.0
                }
            }
        ]
        
        with patch('scrollintel.api.routes.real_time_monitoring_routes.real_time_monitor') as mock_monitor:
            mock_monitor.get_agent_metrics.return_value = mock_agent_metrics
            mock_monitor.get_agent_history.return_value = mock_history
            
            response = client.get(f"/api/v1/monitoring/agents/{agent_id}")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["agent_id"] == agent_id
            
            current_metrics = data["data"]["current_metrics"]
            assert current_metrics["agent_type"] == "ml_engineer"
            assert current_metrics["success_rate"] == 97.2
            assert current_metrics["business_value_generated"] == 8000.0
            
            assert len(data["data"]["historical_data"]) == 1
    
    def test_get_agent_details_not_found(self):
        """Test agent details endpoint with non-existent agent"""
        with patch('scrollintel.api.routes.real_time_monitoring_routes.real_time_monitor') as mock_monitor:
            mock_monitor.get_agent_metrics.return_value = None
            
            response = client.get("/api/v1/monitoring/agents/non-existent-agent")
            
            assert response.status_code == 404
            data = response.json()
            assert "Agent non-existent-agent not found" in data["detail"]
    
    @patch('scrollintel.api.routes.real_time_monitoring_routes.business_impact_tracker')
    def test_get_business_impact_metrics(self, mock_tracker):
        """Test business impact metrics endpoint"""
        mock_business_metrics = BusinessImpactMetrics(
            timestamp=datetime.utcnow(),
            total_roi=250.0,
            cost_savings_24h=5000.0,
            cost_savings_7d=25000.0,
            cost_savings_30d=100000.0,
            revenue_impact=50000.0,
            productivity_gain=25.0,
            decision_accuracy_improvement=30.0,
            time_to_insight_reduction=60.0,
            user_satisfaction_score=92.5,
            competitive_advantage_score=88.0
        )
        
        mock_tracker.calculate_roi_metrics = AsyncMock(return_value=mock_business_metrics)
        
        response = client.get("/api/v1/monitoring/business-impact")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["total_roi"] == 250.0
        assert data["data"]["cost_savings"]["30_days"] == 100000.0
        assert data["data"]["revenue_impact"] == 50000.0
        assert data["data"]["productivity_gain"] == 25.0
        assert data["data"]["user_satisfaction_score"] == 92.5
    
    @patch('scrollintel.api.routes.real_time_monitoring_routes.executive_reporting')
    def test_get_system_health(self, mock_reporting):
        """Test system health endpoint"""
        mock_dashboard = {
            "system_health": {
                "overall_health_score": 95.5,
                "uptime_percentage": 99.95,
                "performance_score": 92.0,
                "security_score": 96.5,
                "agent_health_score": 94.2
            }
        }
        
        mock_reporting.generate_executive_dashboard = AsyncMock(return_value=mock_dashboard)
        
        response = client.get("/api/v1/monitoring/system-health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["overall_health_score"] == 95.5
        assert data["data"]["uptime_percentage"] == 99.95
        assert data["data"]["performance_score"] == 92.0
    
    @patch('scrollintel.api.routes.real_time_monitoring_routes.alert_manager')
    def test_get_active_alerts(self, mock_alert_manager):
        """Test active alerts endpoint"""
        from scrollintel.core.alerting import Alert, AlertSeverity, AlertStatus
        
        mock_active_alerts = [
            Alert(
                id="alert-1",
                name="High CPU Usage",
                description="CPU usage above threshold",
                severity=AlertSeverity.WARNING,
                status=AlertStatus.ACTIVE,
                metric_name="cpu_usage",
                current_value=85.0,
                threshold=80.0,
                timestamp=datetime.utcnow(),
                tags={"component": "system"}
            )
        ]
        
        mock_alert_history = [
            Alert(
                id="alert-2",
                name="Memory Usage",
                description="Memory usage resolved",
                severity=AlertSeverity.WARNING,
                status=AlertStatus.RESOLVED,
                metric_name="memory_usage",
                current_value=70.0,
                threshold=85.0,
                timestamp=datetime.utcnow(),
                resolved_at=datetime.utcnow(),
                tags={}
            )
        ]
        
        mock_alert_manager.get_active_alerts.return_value = mock_active_alerts
        mock_alert_manager.get_alert_history.return_value = mock_alert_history
        
        response = client.get("/api/v1/monitoring/alerts")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["data"]["active_alerts"]) == 1
        assert len(data["data"]["alert_history"]) == 1
        
        active_alert = data["data"]["active_alerts"][0]
        assert active_alert["name"] == "High CPU Usage"
        assert active_alert["severity"] == "warning"
        assert active_alert["current_value"] == 85.0
        
        statistics = data["data"]["statistics"]
        assert statistics["total_active"] == 1
        assert statistics["warning_alerts"] == 1
    
    @patch('scrollintel.api.routes.real_time_monitoring_routes.alert_manager')
    def test_acknowledge_alert(self, mock_alert_manager):
        """Test alert acknowledgment endpoint"""
        alert_id = "alert-1"
        user_id = "test-user"
        
        response = client.post(
            f"/api/v1/monitoring/alerts/{alert_id}/acknowledge?user_id={user_id}"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert f"Alert {alert_id} acknowledged successfully" in data["message"]
        
        # Verify the alert manager was called
        mock_alert_manager.acknowledge_alert.assert_called_once_with(alert_id, user_id)
    
    @patch('scrollintel.api.routes.real_time_monitoring_routes.real_time_monitor')
    def test_update_agent_metrics(self, mock_monitor):
        """Test agent metrics update endpoint"""
        agent_id = "test-agent-1"
        metrics_data = {
            "cpu_usage": 55.0,
            "memory_usage": 70.0,
            "success_rate": 96.5,
            "avg_response_time": 1.1,
            "business_value_generated": 6000.0
        }
        
        mock_monitor.update_agent_metrics = AsyncMock()
        
        response = client.post(
            f"/api/v1/monitoring/agents/{agent_id}/metrics",
            json=metrics_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert f"Metrics updated for agent {agent_id}" in data["message"]
        
        # Verify the monitor was called with correct data
        mock_monitor.update_agent_metrics.assert_called_once_with(agent_id, metrics_data)
    
    @patch('scrollintel.api.routes.real_time_monitoring_routes.analytics_engine')
    def test_get_analytics_summary(self, mock_analytics):
        """Test analytics summary endpoint"""
        from scrollintel.core.analytics import AnalyticsMetrics
        
        mock_analytics_summary = AnalyticsMetrics(
            total_users=1000,
            active_users_24h=250,
            active_users_7d=600,
            active_users_30d=900,
            total_sessions=5000,
            avg_session_duration=15.5,
            bounce_rate=25.0,
            top_events=[{"event": "page_view", "count": 10000}],
            top_pages=[{"page": "/dashboard", "count": 5000}],
            user_retention={"day_1": 80.0, "day_7": 60.0}
        )
        
        mock_agent_usage = {
            "agent_usage": [
                {
                    "agent_type": "data_scientist",
                    "requests": 500,
                    "avg_duration": 2.5,
                    "success_rate": 95.0
                }
            ]
        }
        
        mock_analytics.get_analytics_summary = AsyncMock(return_value=mock_analytics_summary)
        mock_analytics.get_agent_usage_stats = AsyncMock(return_value=mock_agent_usage)
        
        response = client.get("/api/v1/monitoring/analytics/summary?days=30")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["period_days"] == 30
        
        user_analytics = data["data"]["user_analytics"]
        assert user_analytics["total_users"] == 1000
        assert user_analytics["active_users_24h"] == 250
        
        agent_usage = data["data"]["agent_usage"]
        assert len(agent_usage["agent_usage"]) == 1
        assert agent_usage["agent_usage"][0]["agent_type"] == "data_scientist"
    
    @patch('scrollintel.api.routes.real_time_monitoring_routes.executive_reporting')
    def test_generate_executive_report(self, mock_reporting):
        """Test executive report generation endpoint"""
        mock_dashboard = {
            "executive_summary": {
                "total_roi": 300.0,
                "monthly_cost_savings": 120000.0,
                "system_health": 96.0
            },
            "business_impact": {
                "total_roi": 300.0,
                "cost_savings_30d": 120000.0
            },
            "system_health": {
                "overall_health_score": 96.0
            },
            "agent_performance": {
                "total_agents": 10,
                "active_agents": 9
            },
            "key_performance_indicators": {
                "requests_per_minute": 150.0
            },
            "competitive_positioning": {
                "advantage_score": 90.0
            }
        }
        
        mock_reporting.generate_executive_dashboard = AsyncMock(return_value=mock_dashboard)
        
        response = client.get("/api/v1/monitoring/reports/executive?format=json&period=monthly")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        
        report = data["data"]
        assert report["report_metadata"]["format"] == "json"
        assert report["report_metadata"]["period"] == "monthly"
        assert report["executive_summary"]["total_roi"] == 300.0
        assert len(report["recommendations"]) > 0
        assert "risk_assessment" in report
    
    def test_monitoring_health_check(self):
        """Test monitoring system health check endpoint"""
        response = client.get("/api/v1/monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert "components" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        
        components = data["components"]
        expected_components = [
            "real_time_monitor",
            "business_impact_tracker",
            "executive_reporting",
            "automated_alerting",
            "metrics_collector",
            "alert_manager"
        ]
        
        for component in expected_components:
            assert component in components
            assert components[component] == "healthy"

class TestMonitoringSystemIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow from agent registration to reporting"""
        from scrollintel.core.real_time_monitoring import real_time_monitor, business_impact_tracker
        
        # 1. Register an agent
        agent_id = "integration-test-agent"
        agent_type = "test_agent"
        
        await real_time_monitor.register_agent(agent_id, agent_type)
        
        # 2. Update agent metrics
        metrics = {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "success_rate": 95.0,
            "avg_response_time": 1.5,
            "request_count": 100,
            "business_value_generated": 5000.0,
            "cost_savings": 2500.0
        }
        
        await real_time_monitor.update_agent_metrics(agent_id, metrics)
        
        # 3. Verify agent metrics are stored
        agent_metrics = real_time_monitor.get_agent_metrics(agent_id)
        assert agent_metrics is not None
        assert agent_metrics.success_rate == 95.0
        assert agent_metrics.business_value_generated == 5000.0
        
        # 4. Test API endpoints
        with patch('scrollintel.api.routes.real_time_monitoring_routes.real_time_monitor', real_time_monitor):
            response = client.get("/api/v1/monitoring/agents")
            assert response.status_code == 200
            
            data = response.json()
            agents = data["data"]["agents"]
            
            # Find our test agent
            test_agent = next((a for a in agents if a["agent_id"] == agent_id), None)
            assert test_agent is not None
            assert test_agent["performance"]["success_rate"] == 95.0
        
        # 5. Test business impact calculation (with mocked dependencies)
        with patch('scrollintel.core.real_time_monitoring.analytics_engine') as mock_analytics:
            mock_analytics.get_agent_usage_stats.return_value = {'agent_usage': []}
            mock_analytics.get_analytics_summary.return_value = Mock(
                total_users=100,
                active_users_30d=80
            )
            
            business_metrics = await business_impact_tracker.calculate_roi_metrics()
            assert isinstance(business_metrics, BusinessImpactMetrics)
            assert business_metrics.total_roi >= 0
    
    def test_api_error_handling(self):
        """Test API error handling"""
        # Test with invalid agent ID format
        response = client.get("/api/v1/monitoring/agents/")
        assert response.status_code == 404
        
        # Test analytics with invalid days parameter
        response = client.get("/api/v1/monitoring/analytics/summary?days=0")
        assert response.status_code == 422  # Validation error
        
        # Test executive report with invalid format
        response = client.get("/api/v1/monitoring/reports/executive?format=invalid")
        assert response.status_code == 422  # Validation error

if __name__ == "__main__":
    pytest.main([__file__, "-v"])