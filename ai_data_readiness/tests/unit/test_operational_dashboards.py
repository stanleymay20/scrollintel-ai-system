"""Tests for operational dashboards and alerting system."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from ...core.operational_dashboard import (
    OperationalDashboardGenerator, DashboardWidget, DashboardLayout,
    get_dashboard_generator
)
from ...core.alerting_system import (
    AlertingSystem, AlertRule, AlertContact, AlertIncident,
    AlertChannel, AlertSeverity, EscalationLevel,
    get_alerting_system
)
from ...core.capacity_planner import (
    CapacityPlanner, ResourceComponent, ResourceForecast,
    CapacityRecommendation, CapacityPlanningReport,
    get_capacity_planner
)
from ...models.monitoring_models import Alert, MonitoringDashboard


class TestOperationalDashboardGenerator:
    """Test operational dashboard generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = OperationalDashboardGenerator()
    
    def test_dashboard_templates_available(self):
        """Test that dashboard templates are available."""
        templates = self.generator.dashboard_templates
        
        assert len(templates) > 0
        assert 'system_overview' in templates
        assert 'platform_health' in templates
        assert 'performance_monitoring' in templates
        assert 'capacity_planning' in templates
        assert 'alert_management' in templates
        assert 'executive_summary' in templates
    
    def test_create_dashboard_from_template(self):
        """Test creating dashboard from template."""
        dashboard = self.generator.create_dashboard('system_overview')
        
        assert isinstance(dashboard, MonitoringDashboard)
        assert dashboard.name == "System Overview"
        assert len(dashboard.metrics) > 0
        assert dashboard.refresh_interval_seconds > 0
        assert 'widgets' in dashboard.layout
    
    def test_create_dashboard_with_custom_config(self):
        """Test creating dashboard with custom configuration."""
        custom_config = {
            'refresh_interval': 120,
            'theme': 'dark'
        }
        
        dashboard = self.generator.create_dashboard('system_overview', custom_config)
        
        assert isinstance(dashboard, MonitoringDashboard)
        assert dashboard.name == "System Overview"
    
    def test_create_dashboard_invalid_template(self):
        """Test creating dashboard with invalid template."""
        with pytest.raises(ValueError):
            self.generator.create_dashboard('invalid_template')
    
    @patch('ai_data_readiness.core.operational_dashboard.get_platform_monitor')
    def test_collect_dashboard_data(self, mock_monitor):
        """Test collecting dashboard data."""
        # Mock monitor
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        # Mock metrics
        mock_system_metrics = Mock()
        mock_system_metrics.to_dict.return_value = {
            'cpu_percent': 45.2,
            'memory_percent': 67.8,
            'disk_usage_percent': 23.1
        }
        
        mock_platform_metrics = Mock()
        mock_platform_metrics.to_dict.return_value = {
            'active_datasets': 15,
            'processing_datasets': 3,
            'error_rate_percent': 1.2
        }
        
        mock_monitor_instance.get_metrics_history.return_value = {
            'system_metrics': [mock_system_metrics.to_dict()],
            'platform_metrics': [mock_platform_metrics.to_dict()]
        }
        mock_monitor_instance.get_current_system_metrics.return_value = mock_system_metrics
        mock_monitor_instance.get_current_platform_metrics.return_value = mock_platform_metrics
        mock_monitor_instance.get_health_status.return_value = {'status': 'healthy'}
        
        # Test data collection
        layout = self.generator.dashboard_templates['system_overview']
        data = self.generator._collect_dashboard_data(layout, 24)
        
        assert 'system_metrics' in data
        assert 'platform_metrics' in data
        assert 'health_status' in data
        assert data['health_status']['status'] == 'healthy'
    
    def test_generate_dashboard_html(self):
        """Test generating dashboard HTML."""
        dashboard = self.generator.create_dashboard('system_overview')
        
        with patch.object(self.generator, '_collect_dashboard_data') as mock_collect:
            mock_collect.return_value = {
                'system_metrics': {'current': {}, 'history': []},
                'platform_metrics': {'current': {}, 'history': []},
                'health_status': {'status': 'healthy'}
            }
            
            html = self.generator.generate_dashboard_html(dashboard, 24)
            
            assert isinstance(html, str)
            assert 'System Overview' in html
            assert 'dashboard-grid' in html
            assert 'plotly' in html.lower()
    
    def test_export_dashboard_config(self, tmp_path):
        """Test exporting dashboard configuration."""
        dashboard = self.generator.create_dashboard('system_overview')
        filepath = tmp_path / "dashboard_config.json"
        
        self.generator.export_dashboard_config(dashboard, str(filepath))
        
        assert filepath.exists()
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        assert config['name'] == dashboard.name
        assert 'layout' in config
    
    def test_import_dashboard_config(self, tmp_path):
        """Test importing dashboard configuration."""
        # Create a test config file
        config = {
            'id': 'test_dashboard',
            'name': 'Test Dashboard',
            'description': 'Test description',
            'metrics': ['cpu_percent', 'memory_percent'],
            'refresh_interval_seconds': 60,
            'layout': {'widgets': []}
        }
        
        filepath = tmp_path / "test_config.json"
        with open(filepath, 'w') as f:
            json.dump(config, f)
        
        dashboard = self.generator.import_dashboard_config(str(filepath))
        
        assert isinstance(dashboard, MonitoringDashboard)
        assert dashboard.name == 'Test Dashboard'
        assert dashboard.refresh_interval_seconds == 60


class TestAlertingSystem:
    """Test alerting system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alerting = AlertingSystem()
    
    def test_default_alert_rules_loaded(self):
        """Test that default alert rules are loaded."""
        rules = self.alerting.get_alert_rules()
        
        assert len(rules) > 0
        
        # Check for specific default rules
        rule_names = [rule['name'] for rule in rules]
        assert 'High CPU Usage' in rule_names
        assert 'High Memory Usage' in rule_names
        assert 'High Error Rate' in rule_names
    
    def test_default_contacts_loaded(self):
        """Test that default contacts are loaded."""
        contacts = self.alerting.get_contacts()
        
        assert len(contacts) > 0
        
        # Check escalation levels
        levels = [contact['escalation_level'] for contact in contacts]
        assert 'level_1' in levels
        assert 'level_2' in levels
    
    def test_add_alert_rule(self):
        """Test adding a new alert rule."""
        rule = AlertRule(
            id='test_rule',
            name='Test Rule',
            description='Test alert rule',
            metric_name='test_metric',
            condition='>',
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL]
        )
        
        self.alerting.add_alert_rule(rule)
        
        rules = self.alerting.get_alert_rules()
        rule_ids = [r['id'] for r in rules]
        assert 'test_rule' in rule_ids
    
    def test_remove_alert_rule(self):
        """Test removing an alert rule."""
        # Add a rule first
        rule = AlertRule(
            id='test_rule_remove',
            name='Test Rule Remove',
            description='Test alert rule for removal',
            metric_name='test_metric',
            condition='>',
            threshold=80.0,
            severity=AlertSeverity.WARNING
        )
        
        self.alerting.add_alert_rule(rule)
        
        # Verify it exists
        rules = self.alerting.get_alert_rules()
        rule_ids = [r['id'] for r in rules]
        assert 'test_rule_remove' in rule_ids
        
        # Remove it
        self.alerting.remove_alert_rule('test_rule_remove')
        
        # Verify it's gone
        rules = self.alerting.get_alert_rules()
        rule_ids = [r['id'] for r in rules]
        assert 'test_rule_remove' not in rule_ids
    
    def test_add_contact(self):
        """Test adding a new contact."""
        contact = AlertContact(
            id='test_contact',
            name='Test Contact',
            email='test@example.com',
            escalation_level=EscalationLevel.L1,
            channels=[AlertChannel.EMAIL]
        )
        
        self.alerting.add_contact(contact)
        
        contacts = self.alerting.get_contacts()
        contact_ids = [c['id'] for c in contacts]
        assert 'test_contact' in contact_ids
    
    def test_check_condition(self):
        """Test condition checking logic."""
        assert self.alerting._check_condition(85, '>', 80) == True
        assert self.alerting._check_condition(75, '>', 80) == False
        assert self.alerting._check_condition(75, '<', 80) == True
        assert self.alerting._check_condition(80, '>=', 80) == True
        assert self.alerting._check_condition(80, '<=', 80) == True
        assert self.alerting._check_condition(80, '==', 80) == True
        assert self.alerting._check_condition(80, '!=', 75) == True
    
    @patch('ai_data_readiness.core.alerting_system.get_platform_monitor')
    def test_evaluate_alert_rule(self, mock_monitor):
        """Test evaluating alert rules."""
        # Mock monitor
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        # Create test rule
        rule = AlertRule(
            id='test_eval_rule',
            name='Test Eval Rule',
            description='Test evaluation',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration_minutes=0  # No duration requirement for test
        )
        
        # Test with condition met
        metrics = {'cpu_percent': 85.0}
        self.alerting._evaluate_alert_rule(rule, metrics)
        
        # Check that metric state was recorded
        assert rule.id in self.alerting.metric_states
        assert len(self.alerting.metric_states[rule.id]) > 0
    
    def test_acknowledge_incident(self):
        """Test acknowledging an incident."""
        # Create a test incident
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title='Test Alert',
            description='Test alert description'
        )
        
        incident = AlertIncident(
            id='test_incident',
            alert_rule_id='test_rule',
            alert=alert
        )
        
        self.alerting.active_incidents['test_incident'] = incident
        
        # Acknowledge the incident
        self.alerting.acknowledge_incident('test_incident', 'test_user')
        
        # Check that it was acknowledged
        assert incident.acknowledged_by == 'test_user'
        assert incident.acknowledged_at is not None
        assert incident.is_acknowledged == True
    
    def test_resolve_incident(self):
        """Test resolving an incident."""
        # Create a test incident
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title='Test Alert',
            description='Test alert description'
        )
        
        incident = AlertIncident(
            id='test_incident_resolve',
            alert_rule_id='test_rule',
            alert=alert
        )
        
        self.alerting.active_incidents['test_incident_resolve'] = incident
        
        # Resolve the incident
        self.alerting.resolve_incident('test_incident_resolve', 'test_user')
        
        # Check that it was resolved and moved to history
        assert 'test_incident_resolve' not in self.alerting.active_incidents
        assert len(self.alerting.incident_history) > 0
    
    def test_get_alerting_statistics(self):
        """Test getting alerting statistics."""
        stats = self.alerting.get_alerting_statistics()
        
        assert 'total_incidents' in stats
        assert 'active_incidents' in stats
        assert 'mttr_minutes' in stats
        assert 'alerts_per_hour' in stats
        assert 'alert_rules_count' in stats
        assert 'contacts_count' in stats
        assert 'alerting_active' in stats
        
        assert isinstance(stats['total_incidents'], int)
        assert isinstance(stats['alerting_active'], bool)


class TestCapacityPlanner:
    """Test capacity planner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = CapacityPlanner()
    
    def test_capacity_thresholds_defined(self):
        """Test that capacity thresholds are defined."""
        thresholds = self.planner.capacity_thresholds
        
        assert len(thresholds) > 0
        assert ResourceComponent.CPU in thresholds
        assert ResourceComponent.MEMORY in thresholds
        assert ResourceComponent.STORAGE in thresholds
        
        # Check threshold structure
        cpu_thresholds = thresholds[ResourceComponent.CPU]
        assert 'warning' in cpu_thresholds
        assert 'critical' in cpu_thresholds
        assert cpu_thresholds['warning'] < cpu_thresholds['critical']
    
    def test_cost_estimates_defined(self):
        """Test that cost estimates are defined."""
        costs = self.planner.cost_estimates
        
        assert len(costs) > 0
        assert ResourceComponent.CPU in costs
        assert ResourceComponent.MEMORY in costs
        
        # Check cost structure
        cpu_cost = costs[ResourceComponent.CPU]
        assert 'unit' in cpu_cost
        assert 'cost_per_unit' in cpu_cost
        assert isinstance(cpu_cost['cost_per_unit'], (int, float))
    
    @patch('ai_data_readiness.core.capacity_planner.get_platform_monitor')
    @patch('ai_data_readiness.core.capacity_planner.get_resource_optimizer')
    def test_collect_historical_data(self, mock_optimizer, mock_monitor):
        """Test collecting historical data."""
        # Mock monitor
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        mock_monitor_instance.get_metrics_history.return_value = {
            'system_metrics': [
                {
                    'timestamp': '2024-01-01T12:00:00',
                    'cpu_percent': 45.2,
                    'memory_percent': 67.8,
                    'disk_usage_percent': 23.1,
                    'network_bytes_sent': 1000000,
                    'network_bytes_recv': 2000000
                }
            ],
            'platform_metrics': [
                {
                    'timestamp': '2024-01-01T12:00:00',
                    'processing_datasets': 5
                }
            ]
        }
        
        # Mock optimizer
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance
        mock_optimizer_instance.get_resource_history.return_value = {}
        
        # Test data collection
        data = self.planner._collect_historical_data(days=7)
        
        assert isinstance(data, dict)
        assert ResourceComponent.CPU in data
        assert ResourceComponent.MEMORY in data
        assert ResourceComponent.STORAGE in data
    
    @patch('ai_data_readiness.core.capacity_planner.get_platform_monitor')
    def test_get_current_capacity_status(self, mock_monitor):
        """Test getting current capacity status."""
        # Mock monitor
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        # Mock system metrics
        mock_system_metrics = Mock()
        mock_system_metrics.cpu_percent = 45.2
        mock_system_metrics.memory_percent = 67.8
        mock_system_metrics.disk_usage_percent = 23.1
        
        # Mock platform metrics
        mock_platform_metrics = Mock()
        mock_platform_metrics.processing_datasets = 5
        
        mock_monitor_instance.get_current_system_metrics.return_value = mock_system_metrics
        mock_monitor_instance.get_current_platform_metrics.return_value = mock_platform_metrics
        
        # Test status retrieval
        status = self.planner.get_current_capacity_status()
        
        assert isinstance(status, dict)
        assert 'cpu' in status
        assert 'memory' in status
        assert 'storage' in status
        assert 'overall' in status
        
        # Check status structure
        cpu_status = status['cpu']
        assert 'utilization' in cpu_status
        assert 'status' in cpu_status
        assert 'headroom_percent' in cpu_status
    
    def test_get_utilization_status(self):
        """Test utilization status calculation."""
        thresholds = {'warning': 0.7, 'critical': 0.85}
        
        assert self.planner._get_utilization_status(0.5, thresholds) == 'healthy'
        assert self.planner._get_utilization_status(0.75, thresholds) == 'warning'
        assert self.planner._get_utilization_status(0.9, thresholds) == 'critical'
    
    def test_generate_recommendations(self):
        """Test generating capacity recommendations."""
        # Create test forecasts
        forecasts = [
            ResourceForecast(
                component=ResourceComponent.CPU,
                current_utilization=0.6,
                forecasted_values=[70, 75, 80, 85, 90],  # Exceeds critical threshold
                forecast_dates=[datetime.utcnow() + timedelta(days=i) for i in range(5)],
                confidence_intervals=[(65, 75), (70, 80), (75, 85), (80, 90), (85, 95)],
                method_used=None,
                accuracy_score=0.8,
                trend_direction='increasing'
            )
        ]
        
        recommendations = self.planner._generate_recommendations(forecasts)
        
        assert len(recommendations) > 0
        
        rec = recommendations[0]
        assert rec.component == ResourceComponent.CPU
        assert rec.priority in ['high', 'medium', 'low']
        assert rec.capacity_increase_percent > 0
        assert rec.estimated_cost is not None
    
    @patch('ai_data_readiness.core.capacity_planner.get_platform_monitor')
    @patch('ai_data_readiness.core.capacity_planner.get_resource_optimizer')
    def test_generate_capacity_plan(self, mock_optimizer, mock_monitor):
        """Test generating complete capacity plan."""
        # Mock dependencies
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance
        
        # Mock historical data collection
        with patch.object(self.planner, '_collect_historical_data') as mock_collect:
            mock_collect.return_value = {
                ResourceComponent.CPU: {
                    'timestamps': [datetime.utcnow() - timedelta(hours=i) for i in range(24)],
                    'values': [50 + i for i in range(24)]  # Increasing trend
                }
            }
            
            # Test plan generation
            report = self.planner.generate_capacity_plan(time_horizon_days=30)
            
            assert isinstance(report, CapacityPlanningReport)
            assert report.time_horizon_days == 30
            assert isinstance(report.forecasts, list)
            assert isinstance(report.recommendations, list)
            assert isinstance(report.risk_assessment, dict)
            assert isinstance(report.cost_analysis, dict)
            assert isinstance(report.executive_summary, str)
    
    def test_export_capacity_plan(self, tmp_path):
        """Test exporting capacity plan."""
        # Create a test report
        report = CapacityPlanningReport(
            time_horizon_days=30,
            executive_summary="Test summary"
        )
        
        filepath = tmp_path / "capacity_plan.json"
        
        self.planner.export_capacity_plan(report, str(filepath))
        
        assert filepath.exists()
        
        with open(filepath, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data['time_horizon_days'] == 30
        assert exported_data['executive_summary'] == "Test summary"


class TestGlobalInstances:
    """Test global instance functions."""
    
    def test_get_dashboard_generator(self):
        """Test getting global dashboard generator instance."""
        generator1 = get_dashboard_generator()
        generator2 = get_dashboard_generator()
        
        assert generator1 is generator2  # Should be same instance
        assert isinstance(generator1, OperationalDashboardGenerator)
    
    def test_get_alerting_system(self):
        """Test getting global alerting system instance."""
        alerting1 = get_alerting_system()
        alerting2 = get_alerting_system()
        
        assert alerting1 is alerting2  # Should be same instance
        assert isinstance(alerting1, AlertingSystem)
    
    def test_get_capacity_planner(self):
        """Test getting global capacity planner instance."""
        planner1 = get_capacity_planner()
        planner2 = get_capacity_planner()
        
        assert planner1 is planner2  # Should be same instance
        assert isinstance(planner1, CapacityPlanner)


class TestIntegration:
    """Integration tests for operational systems."""
    
    @patch('ai_data_readiness.core.operational_dashboard.get_platform_monitor')
    @patch('ai_data_readiness.core.operational_dashboard.get_resource_optimizer')
    def test_dashboard_with_alerting_integration(self, mock_optimizer, mock_monitor):
        """Test dashboard integration with alerting system."""
        # Set up mocks
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance
        
        # Mock health status with alerts
        mock_monitor_instance.get_health_status.return_value = {
            'status': 'warning',
            'issues': ['High CPU usage: 85.0%']
        }
        mock_monitor_instance.get_metrics_history.return_value = {
            'system_metrics': [],
            'platform_metrics': [],
            'performance_metrics': []
        }
        mock_monitor_instance.get_current_system_metrics.return_value = None
        mock_monitor_instance.get_current_platform_metrics.return_value = None
        mock_optimizer_instance.get_resource_history.return_value = {}
        
        # Create dashboard and alerting system
        generator = get_dashboard_generator()
        alerting = get_alerting_system()
        
        # Create dashboard
        dashboard = generator.create_dashboard('alert_management')
        
        # Get dashboard data (should include alerts)
        data = generator.get_dashboard_data(dashboard, 24)
        
        assert 'alerts' in data
        assert isinstance(data['alerts'], list)
    
    def test_capacity_planning_with_alerting(self):
        """Test capacity planning integration with alerting."""
        planner = get_capacity_planner()
        alerting = get_alerting_system()
        
        # Get current capacity status
        with patch.object(planner, 'get_current_capacity_status') as mock_status:
            mock_status.return_value = {
                'cpu': {'utilization': 0.9, 'status': 'critical'},
                'memory': {'utilization': 0.7, 'status': 'warning'},
                'overall': {'status': 'critical'}
            }
            
            status = planner.get_current_capacity_status()
            
            # Check that critical status would trigger alerts
            assert status['overall']['status'] == 'critical'
            assert status['cpu']['status'] == 'critical'
            
            # Verify alerting system has rules for capacity
            rules = alerting.get_alert_rules()
            cpu_rules = [r for r in rules if 'cpu' in r['metric_name'].lower()]
            assert len(cpu_rules) > 0