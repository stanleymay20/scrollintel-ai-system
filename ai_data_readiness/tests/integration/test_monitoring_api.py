"""Integration tests for monitoring API endpoints."""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from ai_data_readiness.api.app import app
from ai_data_readiness.core.platform_monitor import SystemMetrics, PlatformMetrics
from ai_data_readiness.core.resource_optimizer import ResourceUsage
from ai_data_readiness.models.monitoring_models import AlertSeverity


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_healthy_metrics():
    """Mock healthy system metrics."""
    system_metrics = SystemMetrics(
        timestamp=datetime.utcnow(),
        cpu_percent=50.0,
        memory_percent=60.0,
        memory_used_gb=8.0,
        memory_available_gb=8.0,
        disk_usage_percent=70.0,
        disk_free_gb=100.0,
        network_bytes_sent=1000000,
        network_bytes_recv=2000000,
        active_connections=25
    )
    
    platform_metrics = PlatformMetrics(
        timestamp=datetime.utcnow(),
        active_datasets=10,
        processing_datasets=2,
        failed_datasets=0,
        total_data_processed_gb=150.0,
        avg_processing_time_seconds=45.0,
        quality_assessments_completed=8,
        bias_analyses_completed=5,
        api_requests_per_minute=120.0,
        error_rate_percent=2.0
    )
    
    resource_usage = ResourceUsage(
        timestamp=datetime.utcnow(),
        cpu_cores_used=4.0,
        memory_mb_used=8192.0,
        disk_io_mb_per_sec=10.0,
        network_io_mb_per_sec=5.0,
        active_threads=20,
        active_processes=100
    )
    
    return system_metrics, platform_metrics, resource_usage


class TestHealthEndpoints:
    """Test health monitoring endpoints."""
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_health_status_healthy(self, mock_get_monitor, client):
        """Test health status endpoint when system is healthy."""
        mock_monitor = Mock()
        mock_monitor.get_health_status.return_value = {
            'status': 'healthy',
            'message': 'All systems operating normally'
        }
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'message' in data
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_health_status_warning(self, mock_get_monitor, client):
        """Test health status endpoint when system has warnings."""
        mock_monitor = Mock()
        mock_monitor.get_health_status.return_value = {
            'status': 'warning',
            'issues': ['High CPU usage: 85.0%']
        }
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data['status'] == 'warning'
        assert 'issues' in data
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_health_status_error(self, mock_get_monitor, client):
        """Test health status endpoint error handling."""
        mock_monitor = Mock()
        mock_monitor.get_health_status.side_effect = Exception("Monitor error")
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/health")
        
        assert response.status_code == 500


class TestMetricsEndpoints:
    """Test metrics endpoints."""
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_resource_optimizer')
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_current_metrics(self, mock_get_monitor, mock_get_optimizer, 
                                client, mock_healthy_metrics):
        """Test current metrics endpoint."""
        system_metrics, platform_metrics, resource_usage = mock_healthy_metrics
        
        mock_monitor = Mock()
        mock_monitor.get_current_system_metrics.return_value = system_metrics
        mock_monitor.get_current_platform_metrics.return_value = platform_metrics
        mock_get_monitor.return_value = mock_monitor
        
        mock_optimizer = Mock()
        mock_optimizer.get_current_resource_usage.return_value = resource_usage
        mock_optimizer.get_resource_efficiency_metrics.return_value = {
            'cpu_efficiency': 0.75,
            'memory_efficiency': 0.60
        }
        mock_get_optimizer.return_value = mock_optimizer
        
        response = client.get("/monitoring/metrics/current")
        
        assert response.status_code == 200
        data = response.json()
        assert 'timestamp' in data
        assert 'system_metrics' in data
        assert 'platform_metrics' in data
        assert 'resource_usage' in data
        assert 'efficiency_metrics' in data
        
        # Verify metric values
        assert data['system_metrics']['cpu_percent'] == 50.0
        assert data['platform_metrics']['active_datasets'] == 10
        assert data['resource_usage']['cpu_cores_used'] == 4.0
        assert data['efficiency_metrics']['cpu_efficiency'] == 0.75
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_resource_optimizer')
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_metrics_history(self, mock_get_monitor, mock_get_optimizer, client):
        """Test metrics history endpoint."""
        mock_monitor = Mock()
        mock_monitor.get_metrics_history.return_value = {
            'system_metrics': [{'timestamp': '2024-01-01T00:00:00', 'cpu_percent': 50.0}],
            'platform_metrics': [{'timestamp': '2024-01-01T00:00:00', 'active_datasets': 10}],
            'performance_metrics': []
        }
        mock_get_monitor.return_value = mock_monitor
        
        mock_optimizer = Mock()
        mock_optimizer.get_resource_history.return_value = [
            {'timestamp': '2024-01-01T00:00:00', 'cpu_cores_used': 4.0}
        ]
        mock_get_optimizer.return_value = mock_optimizer
        
        response = client.get("/monitoring/metrics/history?hours=24")
        
        assert response.status_code == 200
        data = response.json()
        assert data['period_hours'] == 24
        assert 'system_metrics' in data
        assert 'platform_metrics' in data
        assert 'resource_usage' in data
        
        # Test with metric type filter
        response = client.get("/monitoring/metrics/history?hours=12&metric_type=system_metrics")
        assert response.status_code == 200
        data = response.json()
        assert 'system_metrics' in data
        assert 'platform_metrics' not in data
    
    def test_get_metrics_history_validation(self, client):
        """Test metrics history endpoint validation."""
        # Test invalid hours parameter
        response = client.get("/monitoring/metrics/history?hours=200")
        assert response.status_code == 422  # Validation error


class TestAlertsEndpoints:
    """Test alerts endpoints."""
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_alerts_healthy_system(self, mock_get_monitor, client):
        """Test alerts endpoint when system is healthy."""
        mock_monitor = Mock()
        mock_monitor.get_health_status.return_value = {
            'status': 'healthy',
            'message': 'All systems operating normally'
        }
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/alerts")
        
        assert response.status_code == 200
        data = response.json()
        assert 'alerts' in data
        assert 'total_count' in data
        assert 'active_count' in data
        assert 'critical_count' in data
        assert data['total_count'] == 0
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_alerts_with_issues(self, mock_get_monitor, client):
        """Test alerts endpoint when system has issues."""
        mock_monitor = Mock()
        mock_monitor.get_health_status.return_value = {
            'status': 'critical',
            'issues': ['High CPU usage: 95.0%', 'High memory usage: 90.0%']
        }
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/alerts")
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_count'] == 2
        assert data['critical_count'] == 2
        assert len(data['alerts']) == 2
        
        # Verify alert structure
        alert = data['alerts'][0]
        assert 'severity' in alert
        assert 'title' in alert
        assert 'description' in alert
        assert 'created_at' in alert
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_alerts_with_filters(self, mock_get_monitor, client):
        """Test alerts endpoint with filters."""
        mock_monitor = Mock()
        mock_monitor.get_health_status.return_value = {
            'status': 'warning',
            'issues': ['High CPU usage: 85.0%']
        }
        mock_get_monitor.return_value = mock_monitor
        
        # Test severity filter
        response = client.get("/monitoring/alerts?severity=warning")
        assert response.status_code == 200
        data = response.json()
        assert all(alert['severity'] == 'warning' for alert in data['alerts'])
        
        # Test limit
        response = client.get("/monitoring/alerts?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data['alerts']) <= 1
    
    def test_acknowledge_alert(self, client):
        """Test alert acknowledgment endpoint."""
        response = client.post("/monitoring/alerts/test-alert-123/acknowledge?user=test_user")
        
        assert response.status_code == 200
        data = response.json()
        assert data['alert_id'] == 'test-alert-123'
        assert data['acknowledged_by'] == 'test_user'
        assert 'acknowledged_at' in data
        assert data['status'] == 'acknowledged'


class TestPerformanceEndpoints:
    """Test performance monitoring endpoints."""
    
    def test_get_performance_benchmarks(self, client):
        """Test performance benchmarks endpoint."""
        response = client.get("/monitoring/performance/benchmarks")
        
        assert response.status_code == 200
        data = response.json()
        assert 'benchmarks' in data
        assert 'summary' in data
        assert 'period_days' in data
        
        # Test with filters
        response = client.get("/monitoring/performance/benchmarks?operation_type=data_ingestion&days=30")
        assert response.status_code == 200
        data = response.json()
        assert data['period_days'] == 30


class TestOptimizationEndpoints:
    """Test optimization endpoints."""
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_resource_optimizer')
    def test_get_optimization_recommendations(self, mock_get_optimizer, client):
        """Test optimization recommendations endpoint."""
        mock_optimizer = Mock()
        mock_optimizer.get_optimization_recommendations.return_value = [
            {
                'category': 'cpu',
                'priority': 'medium',
                'title': 'High CPU Usage',
                'description': 'CPU usage at 85%',
                'action': 'Consider scaling up',
                'estimated_improvement': '20% performance gain',
                'implementation_effort': 'Low'
            },
            {
                'category': 'memory',
                'priority': 'high',
                'title': 'Memory Pressure',
                'description': 'Memory usage at 90%',
                'action': 'Increase memory allocation',
                'estimated_improvement': '30% performance gain',
                'implementation_effort': 'Medium'
            }
        ]
        mock_get_optimizer.return_value = mock_optimizer
        
        response = client.get("/monitoring/optimization/recommendations")
        
        assert response.status_code == 200
        data = response.json()
        assert 'recommendations' in data
        assert 'total_count' in data
        assert 'categories' in data
        assert data['total_count'] == 2
        assert 'cpu' in data['categories']
        assert 'memory' in data['categories']
        
        # Test category filter
        response = client.get("/monitoring/optimization/recommendations?category=cpu")
        assert response.status_code == 200
        mock_optimizer.get_optimization_recommendations.assert_called_with(category='cpu')


class TestCapacityPlanningEndpoints:
    """Test capacity planning endpoints."""
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_resource_optimizer')
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_capacity_planning(self, mock_get_monitor, mock_get_optimizer, 
                                  client, mock_healthy_metrics):
        """Test capacity planning endpoint."""
        system_metrics, platform_metrics, resource_usage = mock_healthy_metrics
        
        mock_monitor = Mock()
        mock_monitor.get_current_system_metrics.return_value = system_metrics
        mock_get_monitor.return_value = mock_monitor
        
        mock_optimizer = Mock()
        mock_optimizer.get_resource_efficiency_metrics.return_value = {
            'cpu_efficiency': 0.75,
            'memory_efficiency': 0.60
        }
        mock_get_optimizer.return_value = mock_optimizer
        
        response = client.get("/monitoring/capacity/planning")
        
        assert response.status_code == 200
        data = response.json()
        assert 'capacity_plans' in data
        assert 'time_horizon_days' in data
        assert 'generated_at' in data
        assert len(data['capacity_plans']) == 4  # cpu, memory, storage, network
        
        # Verify capacity plan structure
        plan = data['capacity_plans'][0]
        assert 'component' in plan
        assert 'current_utilization' in plan
        assert 'projected_utilization' in plan
        assert 'recommendation' in plan
        assert 'confidence_level' in plan
        
        # Test component filter
        response = client.get("/monitoring/capacity/planning?component=cpu&time_horizon_days=60")
        assert response.status_code == 200
        data = response.json()
        assert len(data['capacity_plans']) == 1
        assert data['capacity_plans'][0]['component'] == 'cpu'
        assert data['time_horizon_days'] == 60
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_get_capacity_planning_no_metrics(self, mock_get_monitor, client):
        """Test capacity planning endpoint when no metrics available."""
        mock_monitor = Mock()
        mock_monitor.get_current_system_metrics.return_value = None
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/capacity/planning")
        
        assert response.status_code == 404


class TestReportingEndpoints:
    """Test reporting endpoints."""
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_resource_optimizer')
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_generate_monitoring_report(self, mock_get_monitor, mock_get_optimizer, client):
        """Test monitoring report generation."""
        mock_monitor = Mock()
        mock_monitor.get_metrics_history.return_value = {
            'system_metrics': [
                {'cpu_percent': 50.0, 'memory_percent': 60.0},
                {'cpu_percent': 55.0, 'memory_percent': 65.0}
            ],
            'platform_metrics': [
                {'active_datasets': 10, 'avg_processing_time_seconds': 30.0, 'failed_datasets': 0},
                {'active_datasets': 12, 'avg_processing_time_seconds': 35.0, 'failed_datasets': 1}
            ]
        }
        mock_get_monitor.return_value = mock_monitor
        
        mock_optimizer = Mock()
        mock_optimizer.get_optimization_recommendations.return_value = [
            {'title': 'Optimize CPU usage', 'category': 'cpu'},
            {'title': 'Reduce memory footprint', 'category': 'memory'}
        ]
        mock_get_optimizer.return_value = mock_optimizer
        
        response = client.get("/monitoring/reports/generate?report_type=daily")
        
        assert response.status_code == 200
        data = response.json()
        assert data['report_type'] == 'daily'
        assert 'period_start' in data
        assert 'period_end' in data
        assert 'generated_at' in data
        assert 'total_datasets_processed' in data
        assert 'avg_processing_time_seconds' in data
        assert 'avg_cpu_utilization' in data
        assert 'optimization_recommendations' in data
        
        # Test different report types
        for report_type in ['weekly', 'monthly']:
            response = client.get(f"/monitoring/reports/generate?report_type={report_type}")
            assert response.status_code == 200
            data = response.json()
            assert data['report_type'] == report_type
        
        # Test invalid report type
        response = client.get("/monitoring/reports/generate?report_type=invalid")
        assert response.status_code == 400
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_generate_monitoring_report_insufficient_data(self, mock_get_monitor, client):
        """Test report generation with insufficient data."""
        mock_monitor = Mock()
        mock_monitor.get_metrics_history.return_value = {
            'system_metrics': [],
            'platform_metrics': []
        }
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/reports/generate")
        
        assert response.status_code == 404


class TestControlEndpoints:
    """Test monitoring control endpoints."""
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_resource_optimizer')
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_start_monitoring(self, mock_get_monitor, mock_get_optimizer, client):
        """Test start monitoring endpoint."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        
        response = client.post("/monitoring/monitoring/start?interval_seconds=30")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'started'
        assert data['monitoring_interval_seconds'] == 30
        assert data['optimization_interval_seconds'] == 150
        assert 'started_at' in data
        
        mock_monitor.start_monitoring.assert_called_once_with(interval_seconds=30)
        mock_optimizer.start_optimization.assert_called_once_with(interval_seconds=150)
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_resource_optimizer')
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_stop_monitoring(self, mock_get_monitor, mock_get_optimizer, client):
        """Test stop monitoring endpoint."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        
        response = client.post("/monitoring/monitoring/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'stopped'
        assert 'stopped_at' in data
        
        mock_monitor.stop_monitoring.assert_called_once()
        mock_optimizer.stop_optimization.assert_called_once()


class TestUtilityEndpoints:
    """Test utility endpoints."""
    
    def test_get_metric_definitions(self, client):
        """Test metric definitions endpoint."""
        response = client.get("/monitoring/metrics/definitions")
        
        assert response.status_code == 200
        data = response.json()
        assert 'metrics' in data
        assert 'total_count' in data
        assert 'categories' in data
        assert data['total_count'] > 0
        assert 'system' in data['categories']
        assert 'platform' in data['categories']
        assert 'performance' in data['categories']
        
        # Verify metric definition structure
        metric = data['metrics'][0]
        assert 'name' in metric
        assert 'type' in metric
        assert 'description' in metric
        assert 'unit' in metric
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_export_metrics(self, mock_get_monitor, client):
        """Test metrics export endpoint."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/monitoring/export?hours=12&format=json")
        
        assert response.status_code == 200
        data = response.json()
        assert 'export_file' in data
        assert data['format'] == 'json'
        assert data['hours'] == 12
        assert 'exported_at' in data
        
        mock_monitor.export_metrics.assert_called_once()
        
        # Test unsupported format
        response = client.get("/monitoring/monitoring/export?format=csv")
        assert response.status_code == 400


class TestErrorHandling:
    """Test error handling in monitoring endpoints."""
    
    @patch('ai_data_readiness.api.routes.monitoring_routes.get_platform_monitor')
    def test_endpoint_error_handling(self, mock_get_monitor, client):
        """Test that endpoints handle errors gracefully."""
        mock_monitor = Mock()
        mock_monitor.get_health_status.side_effect = Exception("Test error")
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/monitoring/health")
        assert response.status_code == 500
        
        response = client.get("/monitoring/metrics/current")
        assert response.status_code == 500