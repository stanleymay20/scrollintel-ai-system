"""
Tests for visual generation monitoring system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from scrollintel.engines.visual_generation.utils.metrics_collector import (
    MetricsCollector, SystemMetrics, GenerationMetrics, AlertRule
)
from scrollintel.engines.visual_generation.utils.alerting_system import (
    AlertingSystem, NotificationChannel, AlertNotification
)
from scrollintel.engines.visual_generation.utils.monitoring_dashboard import (
    MonitoringDashboard, create_monitoring_app
)
from scrollintel.engines.visual_generation.config import InfrastructureConfig


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = InfrastructureConfig()
    config.cache_enabled = True
    config.redis_url = None  # Use local for tests
    config.semantic_similarity_enabled = False
    return config


@pytest.fixture
def metrics_collector(mock_config):
    """Create metrics collector for testing."""
    return MetricsCollector(mock_config)


@pytest.fixture
def alerting_system(mock_config):
    """Create alerting system for testing."""
    return AlertingSystem(mock_config)


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, metrics_collector):
        """Test system metrics collection."""
        # Mock psutil functions
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network:
            
            # Setup mocks
            mock_memory.return_value = Mock(percent=60.0, available=8*1024**3)
            mock_disk.return_value = Mock(used=50*1024**3, total=100*1024**3, free=50*1024**3)
            mock_network.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000)
            
            # Collect metrics
            system_metrics = await metrics_collector._collect_system_metrics()
            
            assert system_metrics.cpu_usage_percent == 75.0
            assert system_metrics.memory_usage_percent == 60.0
            assert system_metrics.disk_usage_percent == 50.0
            assert system_metrics.network_bytes_sent == 1000000
            assert system_metrics.network_bytes_recv == 2000000
    
    @pytest.mark.asyncio
    async def test_generation_metrics_collection(self, metrics_collector):
        """Test generation metrics collection."""
        # Mock Redis client
        metrics_collector.redis_client = AsyncMock()
        metrics_collector.redis_client.get.side_effect = lambda key: {
            'metrics:total_requests': '100',
            'metrics:completed_requests': '95',
            'metrics:failed_requests': '5'
        }.get(key, '0')
        metrics_collector.redis_client.llen.return_value = 10
        metrics_collector.redis_client.lrange.return_value = ['5.0', '3.2', '7.1']
        metrics_collector.redis_client.keys.return_value = ['metrics:model_usage:dalle3']
        
        generation_metrics = await metrics_collector._collect_generation_metrics()
        
        assert generation_metrics.total_requests == 100
        assert generation_metrics.completed_requests == 95
        assert generation_metrics.failed_requests == 5
        assert generation_metrics.queue_length == 10
        assert generation_metrics.success_rate == 0.95
        assert generation_metrics.error_rate == 0.05
    
    @pytest.mark.asyncio
    async def test_request_tracking(self, metrics_collector):
        """Test request start and completion tracking."""
        request_id = "test-request-1"
        model_name = "test-model"
        
        # Record request start
        await metrics_collector.record_request_start(request_id, model_name)
        
        assert request_id in metrics_collector.request_start_times
        assert metrics_collector.model_usage_counters[model_name] == 1
        
        # Record request completion
        await metrics_collector.record_request_completion(request_id, model_name, True, 0.9)
        
        assert request_id not in metrics_collector.request_start_times
    
    @pytest.mark.asyncio
    async def test_alert_rule_evaluation(self, metrics_collector):
        """Test alert rule evaluation."""
        # Add test alert rule
        test_rule = AlertRule(
            name="Test High CPU",
            metric_path="system.cpu_usage_percent",
            threshold=80.0,
            comparison="gt",
            duration=60,
            severity="warning"
        )
        metrics_collector.alert_rules.append(test_rule)
        
        # Create test metrics
        system_metrics = SystemMetrics(cpu_usage_percent=85.0)
        generation_metrics = GenerationMetrics()
        
        # Test metric value extraction
        metric_value = metrics_collector._get_metric_value(
            "system.cpu_usage_percent", 
            system_metrics, 
            generation_metrics
        )
        assert metric_value == 85.0
        
        # Test condition evaluation
        condition_met = metrics_collector._evaluate_condition(85.0, 80.0, "gt")
        assert condition_met is True
        
        condition_not_met = metrics_collector._evaluate_condition(75.0, 80.0, "gt")
        assert condition_not_met is False
    
    @pytest.mark.asyncio
    async def test_metrics_cleanup(self, metrics_collector):
        """Test metrics history cleanup."""
        # Add old metrics
        old_time = time.time() - 90000  # 25 hours ago
        old_system_metrics = SystemMetrics(timestamp=old_time)
        old_generation_metrics = GenerationMetrics(timestamp=old_time)
        
        metrics_collector.system_metrics_history.append(old_system_metrics)
        metrics_collector.generation_metrics_history.append(old_generation_metrics)
        
        # Add recent metrics
        recent_system_metrics = SystemMetrics()
        recent_generation_metrics = GenerationMetrics()
        
        metrics_collector.system_metrics_history.append(recent_system_metrics)
        metrics_collector.generation_metrics_history.append(recent_generation_metrics)
        
        # Cleanup old metrics
        await metrics_collector._cleanup_old_metrics()
        
        # Should only have recent metrics
        assert len(metrics_collector.system_metrics_history) == 1
        assert len(metrics_collector.generation_metrics_history) == 1
        assert metrics_collector.system_metrics_history[0].timestamp > old_time
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, metrics_collector):
        """Test monitoring start and stop."""
        # Start monitoring
        await metrics_collector.start_monitoring()
        
        assert metrics_collector.collection_task is not None
        assert metrics_collector.alert_task is not None
        
        # Stop monitoring
        await metrics_collector.stop_monitoring()
        
        assert metrics_collector.collection_task is None
        assert metrics_collector.alert_task is None


class TestAlertingSystem:
    """Test alerting system functionality."""
    
    @pytest.mark.asyncio
    async def test_alert_notification_creation(self, alerting_system):
        """Test alert notification creation and sending."""
        # Mock HTTP session
        alerting_system.http_session = AsyncMock()
        
        # Add test notification channel
        test_channel = NotificationChannel(
            name='test_webhook',
            type='webhook',
            config={'webhook_url': 'https://example.com/webhook'},
            severity_filter=['critical', 'warning']
        )
        alerting_system.notification_channels.append(test_channel)
        
        # Create test alert
        alert_data = {
            'rule_name': 'Test Alert',
            'severity': 'warning',
            'metric_path': 'test.metric',
            'current_value': 90.0,
            'threshold': 80.0,
            'timestamp': time.time(),
            'message': 'Test alert message'
        }
        
        # Send alert
        await alerting_system.send_alert(alert_data)
        
        # Verify alert was stored
        assert len(alerting_system.alert_history) == 1
        assert alerting_system.alert_history[0].rule_name == 'Test Alert'
    
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, alerting_system):
        """Test alert deduplication."""
        # Create test alert
        alert_data = {
            'rule_name': 'Duplicate Alert',
            'severity': 'warning',
            'metric_path': 'test.metric',
            'current_value': 90.0,
            'threshold': 80.0,
            'timestamp': time.time(),
            'message': 'Test alert message'
        }
        
        # Send first alert
        await alerting_system.send_alert(alert_data)
        assert len(alerting_system.alert_history) == 1
        
        # Send duplicate alert immediately
        alert_data['timestamp'] = time.time()
        await alerting_system.send_alert(alert_data)
        
        # Should still only have one alert due to deduplication
        assert len(alerting_system.alert_history) == 1
    
    @pytest.mark.asyncio
    async def test_slack_notification(self, alerting_system):
        """Test Slack notification formatting."""
        # Mock HTTP session
        mock_response = AsyncMock()
        mock_response.status = 200
        alerting_system.http_session = AsyncMock()
        alerting_system.http_session.post.return_value.__aenter__.return_value = mock_response
        
        # Create test notification
        notification = AlertNotification(
            alert_id='test-alert',
            rule_name='Test Alert',
            severity='critical',
            metric_path='test.metric',
            current_value=95.0,
            threshold=80.0,
            timestamp=time.time(),
            message='Test critical alert'
        )
        
        # Create Slack channel
        slack_channel = NotificationChannel(
            name='slack',
            type='slack',
            config={'webhook_url': 'https://hooks.slack.com/test'},
            severity_filter=['critical']
        )
        
        # Send notification
        await alerting_system._send_slack_notification(notification, slack_channel)
        
        # Verify HTTP call was made
        alerting_system.http_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_email_notification_formatting(self, alerting_system):
        """Test email notification HTML formatting."""
        notification = AlertNotification(
            alert_id='test-alert',
            rule_name='Test Alert',
            severity='warning',
            metric_path='system.cpu_usage_percent',
            current_value=85.0,
            threshold=80.0,
            timestamp=time.time(),
            message='CPU usage is high'
        )
        
        html_body = alerting_system._create_email_body(notification)
        
        # Verify HTML contains expected content
        assert 'Test Alert' in html_body
        assert 'WARNING' in html_body
        assert 'system.cpu_usage_percent' in html_body
        assert '85.0' in html_body
        assert '80.0' in html_body
        assert 'CPU usage is high' in html_body
    
    @pytest.mark.asyncio
    async def test_notification_channel_management(self, alerting_system):
        """Test notification channel management."""
        # Add channel
        test_channel = NotificationChannel(
            name='test_channel',
            type='webhook',
            config={'webhook_url': 'https://example.com'},
            enabled=True
        )
        
        alerting_system.add_notification_channel(test_channel)
        assert len(alerting_system.notification_channels) > 0
        
        # Disable channel
        alerting_system.disable_channel('test_channel')
        channel = next(c for c in alerting_system.notification_channels if c.name == 'test_channel')
        assert channel.enabled is False
        
        # Enable channel
        alerting_system.enable_channel('test_channel')
        channel = next(c for c in alerting_system.notification_channels if c.name == 'test_channel')
        assert channel.enabled is True
        
        # Remove channel
        alerting_system.remove_notification_channel('test_channel')
        test_channels = [c for c in alerting_system.notification_channels if c.name == 'test_channel']
        assert len(test_channels) == 0
    
    @pytest.mark.asyncio
    async def test_alert_statistics(self, alerting_system):
        """Test alert statistics calculation."""
        # Add test alerts
        current_time = time.time()
        test_alerts = [
            {
                'alert_id': 'alert1',
                'rule_name': 'High CPU',
                'severity': 'critical',
                'timestamp': current_time - 3600  # 1 hour ago
            },
            {
                'alert_id': 'alert2',
                'rule_name': 'High Memory',
                'severity': 'warning',
                'timestamp': current_time - 1800  # 30 minutes ago
            },
            {
                'alert_id': 'alert3',
                'rule_name': 'High CPU',
                'severity': 'critical',
                'timestamp': current_time - 900   # 15 minutes ago
            }
        ]
        
        # Mock get_recent_alerts to return test data
        alerting_system.get_recent_alerts = AsyncMock(return_value=test_alerts)
        
        # Get statistics
        stats = await alerting_system.get_alert_statistics(24)
        
        assert stats['total_alerts'] == 3
        assert stats['by_severity']['critical'] == 2
        assert stats['by_severity']['warning'] == 1
        assert stats['by_rule']['High CPU'] == 2
        assert stats['by_rule']['High Memory'] == 1
        assert stats['alert_rate'] == 3 / 24  # 3 alerts in 24 hours


class TestMonitoringDashboard:
    """Test monitoring dashboard functionality."""
    
    @pytest.fixture
    def monitoring_app(self, mock_config, metrics_collector, alerting_system):
        """Create monitoring app for testing."""
        dashboard = MonitoringDashboard(
            config=mock_config,
            metrics_collector=metrics_collector,
            alerting_system=alerting_system
        )
        return TestClient(dashboard.app)
    
    def test_health_check_endpoint(self, monitoring_app):
        """Test health check endpoint."""
        response = monitoring_app.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'version' in data
        assert 'uptime_seconds' in data
        assert 'components' in data
    
    def test_liveness_probe(self, monitoring_app):
        """Test Kubernetes liveness probe."""
        response = monitoring_app.get("/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'alive'
        assert 'timestamp' in data
    
    def test_readiness_probe(self, monitoring_app):
        """Test Kubernetes readiness probe."""
        response = monitoring_app.get("/health/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
    
    def test_metrics_endpoint(self, monitoring_app):
        """Test metrics endpoint."""
        response = monitoring_app.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert 'timestamp' in data
        assert 'system' in data
        assert 'generation' in data
    
    def test_metrics_history_endpoint(self, monitoring_app):
        """Test metrics history endpoint."""
        response = monitoring_app.get("/metrics/history?hours=1")
        assert response.status_code == 200
        
        data = response.json()
        assert 'time_period_hours' in data
        assert 'system_metrics' in data
        assert 'generation_metrics' in data
        assert data['time_period_hours'] == 1
    
    def test_alerts_endpoint(self, monitoring_app):
        """Test alerts endpoint."""
        response = monitoring_app.get("/alerts")
        assert response.status_code == 200
        
        data = response.json()
        assert 'total_alerts' in data
        assert 'active_alerts' in data
        assert 'alert_statistics' in data
    
    def test_dashboard_html_endpoint(self, monitoring_app):
        """Test dashboard HTML endpoint."""
        response = monitoring_app.get("/dashboard")
        assert response.status_code == 200
        assert 'text/html' in response.headers['content-type']
        
        # Check for key HTML elements
        html_content = response.text
        assert '<title>ScrollIntel Visual Generation Monitoring</title>' in html_content
        assert 'System Health' in html_content
        assert 'CPU Usage' in html_content
    
    def test_dashboard_data_endpoint(self, monitoring_app):
        """Test dashboard data endpoint."""
        response = monitoring_app.get("/dashboard/data")
        assert response.status_code == 200
        
        data = response.json()
        assert 'timestamp' in data
        assert 'health' in data
        assert 'metrics' in data
        assert 'alerts' in data
        assert 'uptime_seconds' in data


class TestMonitoringIntegration:
    """Test integration between monitoring components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self, mock_config):
        """Test complete monitoring flow from metrics to alerts to dashboard."""
        # Initialize components
        metrics_collector = MetricsCollector(mock_config)
        alerting_system = AlertingSystem(mock_config)
        
        await alerting_system.initialize()
        
        # Add alert callback to connect metrics to alerting
        async def alert_callback(alert_data):
            await alerting_system.send_alert(alert_data)
        
        metrics_collector.add_alert_callback(alert_callback)
        
        # Create monitoring dashboard
        dashboard = MonitoringDashboard(
            config=mock_config,
            metrics_collector=metrics_collector,
            alerting_system=alerting_system
        )
        
        # Simulate high CPU usage that should trigger alert
        with patch.object(metrics_collector, '_collect_system_metrics') as mock_collect:
            mock_collect.return_value = SystemMetrics(cpu_usage_percent=95.0)
            
            # Collect metrics
            system_metrics = await metrics_collector._collect_system_metrics()
            assert system_metrics.cpu_usage_percent == 95.0
            
            # Check if alert would be triggered
            high_cpu_rule = next(
                (rule for rule in metrics_collector.alert_rules if 'CPU' in rule.name),
                None
            )
            
            if high_cpu_rule:
                condition_met = metrics_collector._evaluate_condition(
                    95.0, high_cpu_rule.threshold, high_cpu_rule.comparison
                )
                assert condition_met is True
        
        # Test dashboard health check
        health_check = await dashboard._perform_health_check()
        assert health_check.status in ['healthy', 'degraded', 'unhealthy']
        
        # Cleanup
        await alerting_system.cleanup()
        await metrics_collector.cleanup()
    
    @pytest.mark.asyncio
    async def test_monitoring_performance_under_load(self, mock_config):
        """Test monitoring system performance under load."""
        metrics_collector = MetricsCollector(mock_config)
        
        # Simulate multiple concurrent metric collections
        async def collect_metrics_task():
            return await metrics_collector._collect_system_metrics()
        
        # Run multiple collections concurrently
        start_time = time.time()
        tasks = [collect_metrics_task() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify all collections completed
        assert len(results) == 10
        assert all(isinstance(result, SystemMetrics) for result in results)
        
        # Verify performance (should complete within reasonable time)
        total_time = end_time - start_time
        assert total_time < 5.0  # Should complete within 5 seconds
        
        await metrics_collector.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])