"""
Unit Tests for Hyperscale Monitoring Engine

This module provides comprehensive unit tests for the hyperscale monitoring engine
components including metrics collection, predictive analytics, and incident response.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.hyperscale_monitoring_engine import HyperscaleMonitoringEngine
from scrollintel.models.hyperscale_monitoring_models import (
    GlobalMetrics, RegionalMetrics, PredictiveAlert, SystemIncident,
    ExecutiveDashboardMetrics, CapacityForecast, GlobalInfrastructureHealth,
    MonitoringDashboard, SeverityLevel, SystemStatus, IncidentStatus
)


class TestHyperscaleMonitoringEngine:
    """Unit tests for hyperscale monitoring engine"""
    
    @pytest.fixture
    def monitoring_engine(self):
        """Create monitoring engine instance for testing"""
        return HyperscaleMonitoringEngine()
    
    @pytest.fixture
    def sample_global_metrics(self):
        """Create sample global metrics for testing"""
        return GlobalMetrics(
            timestamp=datetime.utcnow(),
            total_requests_per_second=2500000,
            active_users=1200000000,
            global_latency_p99=150.0,
            global_latency_p95=85.0,
            global_latency_p50=45.0,
            error_rate=0.001,
            availability=99.99,
            throughput=2500000,
            cpu_utilization=65.0,
            memory_utilization=70.0,
            disk_utilization=45.0,
            network_utilization=60.0
        )
    
    @pytest.fixture
    def sample_regional_metrics(self):
        """Create sample regional metrics for testing"""
        return RegionalMetrics(
            region="us-east-1",
            timestamp=datetime.utcnow(),
            requests_per_second=250000,
            active_users=120000000,
            latency_p99=120.0,
            latency_p95=75.0,
            error_rate=0.0008,
            availability=99.995,
            server_count=50000,
            load_balancer_health=98.5,
            database_connections=25000,
            cache_hit_rate=94.2
        )
    
    @pytest.mark.asyncio
    async def test_collect_global_metrics(self, monitoring_engine):
        """Test global metrics collection"""
        metrics = await monitoring_engine.collect_global_metrics()
        
        assert isinstance(metrics, GlobalMetrics)
        assert metrics.total_requests_per_second > 0
        assert metrics.active_users > 0
        assert 0 <= metrics.error_rate <= 1
        assert 0 <= metrics.availability <= 100
        assert isinstance(metrics.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_collect_regional_metrics(self, monitoring_engine):
        """Test regional metrics collection"""
        regions = ["us-east-1", "eu-west-1"]
        metrics = await monitoring_engine.collect_regional_metrics(regions)
        
        assert len(metrics) == len(regions)
        for metric in metrics:
            assert isinstance(metric, RegionalMetrics)
            assert metric.region in regions
            assert metric.requests_per_second > 0
            assert metric.active_users > 0
    
    @pytest.mark.asyncio
    async def test_analyze_predictive_failures_high_cpu(self, monitoring_engine):
        """Test predictive failure analysis for high CPU"""
        # Create metrics with high CPU utilization
        high_cpu_metrics = GlobalMetrics(
            timestamp=datetime.utcnow(),
            total_requests_per_second=2500000,
            active_users=1200000000,
            global_latency_p99=150.0,
            global_latency_p95=85.0,
            global_latency_p50=45.0,
            error_rate=0.001,
            availability=99.99,
            throughput=2500000,
            cpu_utilization=85.0,  # High CPU
            memory_utilization=70.0,
            disk_utilization=45.0,
            network_utilization=60.0
        )
        
        alerts = await monitoring_engine.analyze_predictive_failures(high_cpu_metrics)
        
        # Should generate CPU saturation alert
        cpu_alerts = [a for a in alerts if "cpu" in a.alert_type.lower()]
        assert len(cpu_alerts) > 0
        
        cpu_alert = cpu_alerts[0]
        assert cpu_alert.severity == SeverityLevel.HIGH
        assert cpu_alert.confidence > 0
        assert "compute" in " ".join(cpu_alert.affected_systems).lower()
    
    @pytest.mark.asyncio
    async def test_analyze_predictive_failures_high_memory(self, monitoring_engine):
        """Test predictive failure analysis for high memory"""
        # Create metrics with high memory utilization
        high_memory_metrics = GlobalMetrics(
            timestamp=datetime.utcnow(),
            total_requests_per_second=2500000,
            active_users=1200000000,
            global_latency_p99=150.0,
            global_latency_p95=85.0,
            global_latency_p50=45.0,
            error_rate=0.001,
            availability=99.99,
            throughput=2500000,
            cpu_utilization=65.0,
            memory_utilization=90.0,  # High memory
            disk_utilization=45.0,
            network_utilization=60.0
        )
        
        alerts = await monitoring_engine.analyze_predictive_failures(high_memory_metrics)
        
        # Should generate memory pressure alert
        memory_alerts = [a for a in alerts if "memory" in a.alert_type.lower()]
        assert len(memory_alerts) > 0
        
        memory_alert = memory_alerts[0]
        assert memory_alert.severity == SeverityLevel.CRITICAL
        assert memory_alert.confidence > 0
    
    @pytest.mark.asyncio
    async def test_analyze_predictive_failures_high_latency(self, monitoring_engine):
        """Test predictive failure analysis for high latency"""
        # Create metrics with high latency
        high_latency_metrics = GlobalMetrics(
            timestamp=datetime.utcnow(),
            total_requests_per_second=2500000,
            active_users=1200000000,
            global_latency_p99=250.0,  # High latency
            global_latency_p95=180.0,
            global_latency_p50=120.0,
            error_rate=0.001,
            availability=99.99,
            throughput=2500000,
            cpu_utilization=65.0,
            memory_utilization=70.0,
            disk_utilization=45.0,
            network_utilization=60.0
        )
        
        alerts = await monitoring_engine.analyze_predictive_failures(high_latency_metrics)
        
        # Should generate latency degradation alert
        latency_alerts = [a for a in alerts if "latency" in a.alert_type.lower()]
        assert len(latency_alerts) > 0
        
        latency_alert = latency_alerts[0]
        assert latency_alert.severity == SeverityLevel.MEDIUM
        assert latency_alert.confidence > 0
    
    @pytest.mark.asyncio
    async def test_create_incident(self, monitoring_engine):
        """Test incident creation from alert"""
        alert = PredictiveAlert(
            id="test_alert",
            timestamp=datetime.utcnow(),
            alert_type="cpu_saturation_prediction",
            severity=SeverityLevel.CRITICAL,
            predicted_failure_time=datetime.utcnow() + timedelta(minutes=10),
            confidence=0.95,
            affected_systems=["compute_cluster"],
            recommended_actions=["Scale up instances"],
            description="Test alert"
        )
        
        incident = await monitoring_engine.create_incident(alert)
        
        assert isinstance(incident, SystemIncident)
        assert incident.severity == SeverityLevel.CRITICAL
        assert incident.status == IncidentStatus.OPEN
        assert len(incident.affected_services) > 0
        assert len(incident.resolution_steps) > 0
        assert incident.id in monitoring_engine.active_incidents
    
    @pytest.mark.asyncio
    async def test_execute_automated_response(self, monitoring_engine):
        """Test automated response execution"""
        incident = SystemIncident(
            id="test_incident",
            title="Test Incident",
            description="Test incident for automated response",
            severity=SeverityLevel.HIGH,
            status=IncidentStatus.OPEN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resolved_at=None,
            affected_services=["api_gateway"],
            affected_regions=["us-east-1"],
            impact_assessment="Test impact",
            root_cause=None,
            resolution_steps=["Scale up instances", "Enable throttling"],
            estimated_users_affected=1000000
        )
        
        responses = await monitoring_engine.execute_automated_response(incident)
        
        assert len(responses) == len(incident.resolution_steps)
        for response in responses:
            assert response.incident_id == incident.id
            assert response.success is True
            assert response.status == "completed"
    
    @pytest.mark.asyncio
    async def test_generate_executive_dashboard_metrics(self, monitoring_engine):
        """Test executive dashboard metrics generation"""
        with patch.object(monitoring_engine, 'collect_global_metrics') as mock_collect:
            mock_collect.return_value = GlobalMetrics(
                timestamp=datetime.utcnow(),
                total_requests_per_second=2500000,
                active_users=1200000000,
                global_latency_p99=150.0,
                global_latency_p95=85.0,
                global_latency_p50=45.0,
                error_rate=0.001,
                availability=99.99,
                throughput=2500000,
                cpu_utilization=65.0,
                memory_utilization=70.0,
                disk_utilization=45.0,
                network_utilization=60.0
            )
            
            exec_metrics = await monitoring_engine.generate_executive_dashboard_metrics()
            
            assert isinstance(exec_metrics, ExecutiveDashboardMetrics)
            assert exec_metrics.total_active_users == 1200000000
            assert exec_metrics.system_availability == 99.99
            assert exec_metrics.global_system_health in [
                SystemStatus.HEALTHY, SystemStatus.DEGRADED, 
                SystemStatus.CRITICAL, SystemStatus.DOWN
            ]
    
    def test_calculate_system_health_healthy(self, monitoring_engine):
        """Test system health calculation for healthy system"""
        metrics = GlobalMetrics(
            timestamp=datetime.utcnow(),
            total_requests_per_second=2500000,
            active_users=1200000000,
            global_latency_p99=150.0,
            global_latency_p95=85.0,
            global_latency_p50=45.0,
            error_rate=0.0005,  # Low error rate
            availability=99.995,  # High availability
            throughput=2500000,
            cpu_utilization=65.0,
            memory_utilization=70.0,
            disk_utilization=45.0,
            network_utilization=60.0
        )
        
        health = monitoring_engine._calculate_system_health(metrics)
        assert health == SystemStatus.HEALTHY
    
    def test_calculate_system_health_degraded(self, monitoring_engine):
        """Test system health calculation for degraded system"""
        metrics = GlobalMetrics(
            timestamp=datetime.utcnow(),
            total_requests_per_second=2500000,
            active_users=1200000000,
            global_latency_p99=150.0,
            global_latency_p95=85.0,
            global_latency_p50=45.0,
            error_rate=0.005,  # Higher error rate
            availability=99.95,  # Lower availability
            throughput=2500000,
            cpu_utilization=65.0,
            memory_utilization=70.0,
            disk_utilization=45.0,
            network_utilization=60.0
        )
        
        health = monitoring_engine._calculate_system_health(metrics)
        assert health == SystemStatus.DEGRADED
    
    def test_calculate_performance_score(self, monitoring_engine):
        """Test performance score calculation"""
        metrics = GlobalMetrics(
            timestamp=datetime.utcnow(),
            total_requests_per_second=2500000,
            active_users=1200000000,
            global_latency_p99=100.0,  # Good latency
            global_latency_p95=85.0,
            global_latency_p50=45.0,
            error_rate=0.001,  # Low error rate
            availability=99.99,  # High availability
            throughput=2500000,
            cpu_utilization=65.0,
            memory_utilization=70.0,
            disk_utilization=45.0,
            network_utilization=60.0
        )
        
        score = monitoring_engine._calculate_performance_score(metrics)
        assert 0 <= score <= 100
        assert score > 90  # Should be high for good metrics
    
    @pytest.mark.asyncio
    async def test_generate_capacity_forecast(self, monitoring_engine):
        """Test capacity forecasting"""
        with patch.object(monitoring_engine, 'collect_global_metrics') as mock_collect:
            mock_collect.return_value = GlobalMetrics(
                timestamp=datetime.utcnow(),
                total_requests_per_second=2500000,
                active_users=1200000000,
                global_latency_p99=150.0,
                global_latency_p95=85.0,
                global_latency_p50=45.0,
                error_rate=0.001,
                availability=99.99,
                throughput=2500000,
                cpu_utilization=65.0,
                memory_utilization=70.0,
                disk_utilization=45.0,
                network_utilization=60.0
            )
            
            forecast = await monitoring_engine.generate_capacity_forecast(30)
            
            assert isinstance(forecast, CapacityForecast)
            assert forecast.forecast_horizon_days == 30
            assert forecast.predicted_user_growth >= 0
            assert forecast.predicted_traffic_growth >= 0
            assert forecast.required_server_capacity > 0
            assert forecast.estimated_cost > 0
            assert len(forecast.scaling_recommendations) > 0
            assert len(forecast.risk_factors) > 0
    
    @pytest.mark.asyncio
    async def test_get_global_infrastructure_health(self, monitoring_engine):
        """Test global infrastructure health assessment"""
        with patch.object(monitoring_engine, 'collect_global_metrics') as mock_global, \
             patch.object(monitoring_engine, 'collect_regional_metrics') as mock_regional, \
             patch.object(monitoring_engine, 'analyze_predictive_failures') as mock_alerts:
            
            mock_global.return_value = GlobalMetrics(
                timestamp=datetime.utcnow(),
                total_requests_per_second=2500000,
                active_users=1200000000,
                global_latency_p99=150.0,
                global_latency_p95=85.0,
                global_latency_p50=45.0,
                error_rate=0.001,
                availability=99.99,
                throughput=2500000,
                cpu_utilization=65.0,
                memory_utilization=70.0,
                disk_utilization=45.0,
                network_utilization=60.0
            )
            
            mock_regional.return_value = [
                RegionalMetrics(
                    region="us-east-1",
                    timestamp=datetime.utcnow(),
                    requests_per_second=250000,
                    active_users=120000000,
                    latency_p99=120.0,
                    latency_p95=75.0,
                    error_rate=0.0008,
                    availability=99.995,
                    server_count=50000,
                    load_balancer_health=98.5,
                    database_connections=25000,
                    cache_hit_rate=94.2
                )
            ]
            
            mock_alerts.return_value = []
            
            health = await monitoring_engine.get_global_infrastructure_health()
            
            assert isinstance(health, GlobalInfrastructureHealth)
            assert 0 <= health.overall_health_score <= 100
            assert len(health.regional_health) > 0
            assert len(health.service_health) > 0
            assert health.critical_alerts >= 0
            assert health.active_incidents >= 0
    
    @pytest.mark.asyncio
    async def test_create_monitoring_dashboard(self, monitoring_engine):
        """Test monitoring dashboard creation"""
        dashboard_types = ["executive", "operational", "technical"]
        
        for dashboard_type in dashboard_types:
            dashboard = await monitoring_engine.create_monitoring_dashboard(dashboard_type)
            
            assert isinstance(dashboard, MonitoringDashboard)
            assert dashboard.dashboard_type == dashboard_type
            assert len(dashboard.widgets) > 0
            assert dashboard.refresh_interval > 0
            assert len(dashboard.access_permissions) > 0
    
    @pytest.mark.asyncio
    async def test_run_monitoring_cycle(self, monitoring_engine):
        """Test complete monitoring cycle execution"""
        with patch.object(monitoring_engine, 'collect_global_metrics') as mock_collect, \
             patch.object(monitoring_engine, 'analyze_predictive_failures') as mock_analyze, \
             patch.object(monitoring_engine, 'generate_executive_dashboard_metrics') as mock_exec:
            
            # Mock return values
            mock_collect.return_value = GlobalMetrics(
                timestamp=datetime.utcnow(),
                total_requests_per_second=2500000,
                active_users=1200000000,
                global_latency_p99=150.0,
                global_latency_p95=85.0,
                global_latency_p50=45.0,
                error_rate=0.001,
                availability=99.99,
                throughput=2500000,
                cpu_utilization=65.0,
                memory_utilization=70.0,
                disk_utilization=45.0,
                network_utilization=60.0
            )
            
            mock_analyze.return_value = []
            
            mock_exec.return_value = ExecutiveDashboardMetrics(
                timestamp=datetime.utcnow(),
                global_system_health=SystemStatus.HEALTHY,
                total_active_users=1200000000,
                revenue_impact=0.0,
                customer_satisfaction_score=98.5,
                system_availability=99.99,
                performance_score=95.0,
                security_incidents=0,
                cost_efficiency=87.3,
                innovation_velocity=92.1,
                competitive_advantage_score=94.7
            )
            
            # Run monitoring cycle
            await monitoring_engine.run_monitoring_cycle()
            
            # Verify all components were called
            mock_collect.assert_called_once()
            mock_analyze.assert_called_once()
            mock_exec.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_metrics_collection(self, monitoring_engine):
        """Test error handling in metrics collection"""
        with patch.object(monitoring_engine, 'logger') as mock_logger:
            # This should not raise an exception even if there are internal errors
            try:
                metrics = await monitoring_engine.collect_global_metrics()
                assert isinstance(metrics, GlobalMetrics)
            except Exception as e:
                # If an exception is raised, verify it's logged
                mock_logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_monitoring_operations(self, monitoring_engine):
        """Test concurrent monitoring operations"""
        import asyncio
        
        # Run multiple monitoring operations concurrently
        tasks = [
            monitoring_engine.collect_global_metrics(),
            monitoring_engine.collect_regional_metrics(["us-east-1"]),
            monitoring_engine.generate_capacity_forecast(30)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all operations completed (either successfully or with handled exceptions)
        assert len(results) == 3
        
        # Check that at least the global metrics collection succeeded
        assert isinstance(results[0], GlobalMetrics)


class TestHyperscaleMonitoringModels:
    """Unit tests for hyperscale monitoring data models"""
    
    def test_global_metrics_creation(self):
        """Test GlobalMetrics model creation"""
        metrics = GlobalMetrics(
            timestamp=datetime.utcnow(),
            total_requests_per_second=2500000,
            active_users=1200000000,
            global_latency_p99=150.0,
            global_latency_p95=85.0,
            global_latency_p50=45.0,
            error_rate=0.001,
            availability=99.99,
            throughput=2500000,
            cpu_utilization=65.0,
            memory_utilization=70.0,
            disk_utilization=45.0,
            network_utilization=60.0
        )
        
        assert metrics.total_requests_per_second == 2500000
        assert metrics.active_users == 1200000000
        assert metrics.error_rate == 0.001
        assert metrics.availability == 99.99
    
    def test_predictive_alert_creation(self):
        """Test PredictiveAlert model creation"""
        alert = PredictiveAlert(
            id="test_alert",
            timestamp=datetime.utcnow(),
            alert_type="cpu_saturation_prediction",
            severity=SeverityLevel.CRITICAL,
            predicted_failure_time=datetime.utcnow() + timedelta(minutes=10),
            confidence=0.95,
            affected_systems=["compute_cluster"],
            recommended_actions=["Scale up instances"],
            description="Test alert"
        )
        
        assert alert.id == "test_alert"
        assert alert.severity == SeverityLevel.CRITICAL
        assert alert.confidence == 0.95
        assert len(alert.affected_systems) == 1
        assert len(alert.recommended_actions) == 1
    
    def test_system_incident_creation(self):
        """Test SystemIncident model creation"""
        incident = SystemIncident(
            id="test_incident",
            title="Test Incident",
            description="Test incident description",
            severity=SeverityLevel.HIGH,
            status=IncidentStatus.OPEN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resolved_at=None,
            affected_services=["api_gateway"],
            affected_regions=["us-east-1"],
            impact_assessment="Test impact",
            root_cause=None,
            resolution_steps=["Scale up instances"],
            estimated_users_affected=1000000
        )
        
        assert incident.id == "test_incident"
        assert incident.severity == SeverityLevel.HIGH
        assert incident.status == IncidentStatus.OPEN
        assert incident.estimated_users_affected == 1000000
    
    def test_executive_dashboard_metrics_creation(self):
        """Test ExecutiveDashboardMetrics model creation"""
        metrics = ExecutiveDashboardMetrics(
            timestamp=datetime.utcnow(),
            global_system_health=SystemStatus.HEALTHY,
            total_active_users=1200000000,
            revenue_impact=0.0,
            customer_satisfaction_score=98.5,
            system_availability=99.99,
            performance_score=95.0,
            security_incidents=0,
            cost_efficiency=87.3,
            innovation_velocity=92.1,
            competitive_advantage_score=94.7
        )
        
        assert metrics.global_system_health == SystemStatus.HEALTHY
        assert metrics.total_active_users == 1200000000
        assert metrics.customer_satisfaction_score == 98.5
        assert metrics.performance_score == 95.0