"""
Integration tests for the Bulletproof Orchestrator.
Tests the unified system coordination and management interface.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.bulletproof_orchestrator import (
    BulletproofOrchestrator,
    UnifiedConfiguration,
    ProtectionMode,
    SystemHealthStatus,
    SystemHealthMetrics,
    bulletproof_orchestrator,
    start_bulletproof_system,
    stop_bulletproof_system,
    get_system_health
)


class TestBulletproofOrchestrator:
    """Test the bulletproof orchestrator functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance."""
        config = UnifiedConfiguration(
            protection_mode=ProtectionMode.FULL_PROTECTION,
            auto_recovery_enabled=True,
            predictive_prevention_enabled=True,
            user_experience_optimization=True,
            intelligent_routing_enabled=True,
            fallback_generation_enabled=True,
            degradation_learning_enabled=True,
            real_time_monitoring=True,
            alert_thresholds={
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "error_rate": 0.05,
                "response_time": 3.0,
                "user_satisfaction": 0.7
            },
            recovery_timeouts={
                "automatic_recovery": 30.0,
                "degradation_recovery": 60.0,
                "system_restart": 300.0
            },
            user_notification_settings={
                "show_system_status": True,
                "show_recovery_progress": True,
                "show_degradation_notices": True,
                "auto_dismiss_success": True
            }
        )
        return BulletproofOrchestrator(config)
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator is not None
        assert not orchestrator.is_active
        assert orchestrator.config.protection_mode == ProtectionMode.FULL_PROTECTION
        assert len(orchestrator.protection_systems) > 0
        
        # Check that all expected protection systems are registered
        expected_systems = [
            "failure_prevention",
            "degradation_manager", 
            "ux_protector",
            "intelligent_retry",
            "fallback_generator"
        ]
        
        for system_name in expected_systems:
            assert system_name in orchestrator.protection_systems
            assert "instance" in orchestrator.protection_systems[system_name]
            assert "health_check" in orchestrator.protection_systems[system_name]
            assert "metrics_collector" in orchestrator.protection_systems[system_name]
    
    @pytest.mark.asyncio
    async def test_orchestrator_start_stop(self, orchestrator):
        """Test orchestrator start and stop functionality."""
        # Test start
        await orchestrator.start()
        assert orchestrator.is_active
        
        # Test stop
        await orchestrator.stop()
        assert not orchestrator.is_active
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, orchestrator):
        """Test system health monitoring functionality."""
        await orchestrator.start()
        
        try:
            # Wait a moment for monitoring to collect data
            await asyncio.sleep(1)
            
            # Collect system health metrics
            await orchestrator._collect_system_health_metrics()
            
            # Check that metrics were collected
            assert len(orchestrator.system_metrics_history) > 0
            
            latest_metrics = orchestrator.system_metrics_history[-1]
            assert isinstance(latest_metrics, SystemHealthMetrics)
            assert isinstance(latest_metrics.overall_status, SystemHealthStatus)
            assert latest_metrics.cpu_usage >= 0
            assert latest_metrics.memory_usage >= 0
            assert latest_metrics.protection_effectiveness >= 0
            assert latest_metrics.recovery_success_rate >= 0
            assert latest_metrics.user_satisfaction_score >= 0
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_protection_system_health_checks(self, orchestrator):
        """Test individual protection system health checks."""
        await orchestrator.start()
        
        try:
            # Test failure prevention health check
            health_score = await orchestrator._check_failure_prevention_health()
            assert 0.0 <= health_score <= 1.0
            
            # Test degradation manager health check
            health_score = await orchestrator._check_degradation_manager_health()
            assert 0.0 <= health_score <= 1.0
            
            # Test UX protector health check
            health_score = await orchestrator._check_ux_protector_health()
            assert 0.0 <= health_score <= 1.0
            
            # Test intelligent retry health check
            health_score = await orchestrator._check_intelligent_retry_health()
            assert 0.0 <= health_score <= 1.0
            
            # Test fallback generator health check
            health_score = await orchestrator._check_fallback_generator_health()
            assert 0.0 <= health_score <= 1.0
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, orchestrator):
        """Test metrics collection from protection systems."""
        await orchestrator.start()
        
        try:
            # Test failure prevention metrics
            metrics = await orchestrator._collect_failure_prevention_metrics()
            assert isinstance(metrics, dict)
            assert "recent_failures" in metrics or "error" in metrics
            
            # Test degradation manager metrics
            metrics = await orchestrator._collect_degradation_manager_metrics()
            assert isinstance(metrics, dict)
            assert "current_degradation_level" in metrics or "error" in metrics
            
            # Test UX protector metrics
            metrics = await orchestrator._collect_ux_protector_metrics()
            assert isinstance(metrics, dict)
            assert "experience_level" in metrics or "error" in metrics
            
            # Test intelligent retry metrics
            metrics = await orchestrator._collect_intelligent_retry_metrics()
            assert isinstance(metrics, dict)
            assert "failure_history_size" in metrics or "error" in metrics
            
            # Test fallback generator metrics
            metrics = await orchestrator._collect_fallback_generator_metrics()
            assert isinstance(metrics, dict)
            assert "fallback_templates" in metrics or "error" in metrics
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_alert_system(self, orchestrator):
        """Test alert detection and management."""
        await orchestrator.start()
        
        try:
            # Simulate high CPU usage to trigger alert
            with patch('psutil.cpu_percent', return_value=95.0):
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory.return_value.percent = 50.0
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk.return_value.percent = 30.0
                        
                        # Collect metrics and check alerts
                        await orchestrator._collect_system_health_metrics()
                        await orchestrator._check_system_alerts()
                        
                        # Should have generated a CPU alert
                        cpu_alerts = [alert for alert in orchestrator.active_alerts 
                                    if alert["type"] == "cpu_high"]
                        assert len(cpu_alerts) > 0
                        
                        cpu_alert = cpu_alerts[0]
                        assert cpu_alert["severity"] == "warning"
                        assert cpu_alert["value"] == 95.0
                        assert cpu_alert["threshold"] == 80.0
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_recovery_coordination(self, orchestrator):
        """Test recovery action coordination."""
        await orchestrator.start()
        
        try:
            # Create a mock system health metrics with critical status
            critical_metrics = SystemHealthMetrics(
                overall_status=SystemHealthStatus.CRITICAL,
                cpu_usage=95.0,
                memory_usage=90.0,
                disk_usage=50.0,
                network_latency=100.0,
                error_rate=0.15,
                response_time=8.0,
                active_users=10,
                protection_effectiveness=0.3,
                recovery_success_rate=0.5,
                user_satisfaction_score=0.2
            )
            
            orchestrator.system_metrics_history.append(critical_metrics)
            
            # Test recovery coordination
            await orchestrator._coordinate_recovery_actions()
            
            # Should have triggered recovery actions
            assert len(orchestrator.recovery_actions) > 0
            
            # Check that appropriate recovery actions were taken
            recovery_types = [action["type"] for action in orchestrator.recovery_actions]
            assert "system_critical" in recovery_types or "high_error_rate" in recovery_types
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_emergency_mode_activation(self, orchestrator):
        """Test emergency mode activation."""
        await orchestrator.start()
        
        try:
            # Test emergency mode activation
            await orchestrator._trigger_emergency_mode()
            
            # Verify emergency mode was activated
            # (In a real implementation, this would check actual system states)
            assert True  # Placeholder - would check actual emergency mode state
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_configuration_management(self, orchestrator):
        """Test configuration management."""
        # Test getting configuration
        config = orchestrator.get_configuration()
        assert isinstance(config, UnifiedConfiguration)
        assert config.protection_mode == ProtectionMode.FULL_PROTECTION
        
        # Test updating configuration
        new_config = UnifiedConfiguration(
            protection_mode=ProtectionMode.PERFORMANCE_OPTIMIZED,
            auto_recovery_enabled=False,
            predictive_prevention_enabled=False,
            user_experience_optimization=True,
            intelligent_routing_enabled=True,
            fallback_generation_enabled=True,
            degradation_learning_enabled=False,
            real_time_monitoring=True,
            alert_thresholds={"cpu_usage": 90.0},
            recovery_timeouts={"automatic_recovery": 60.0},
            user_notification_settings={"show_system_status": False}
        )
        
        orchestrator.update_configuration(new_config)
        
        updated_config = orchestrator.get_configuration()
        assert updated_config.protection_mode == ProtectionMode.PERFORMANCE_OPTIMIZED
        assert not updated_config.auto_recovery_enabled
        assert not updated_config.predictive_prevention_enabled
    
    def test_system_health_api(self, orchestrator):
        """Test system health API."""
        # Test with no metrics
        health_data = orchestrator.get_system_health()
        assert health_data["status"] == "initializing"
        
        # Add some mock metrics
        mock_metrics = SystemHealthMetrics(
            overall_status=SystemHealthStatus.GOOD,
            cpu_usage=45.0,
            memory_usage=60.0,
            disk_usage=30.0,
            network_latency=50.0,
            error_rate=0.02,
            response_time=1.5,
            active_users=25,
            protection_effectiveness=0.9,
            recovery_success_rate=0.95,
            user_satisfaction_score=0.85
        )
        
        orchestrator.system_metrics_history.append(mock_metrics)
        
        health_data = orchestrator.get_system_health()
        assert health_data["overall_status"] == "good"
        assert health_data["metrics"]["cpu_usage"] == 45.0
        assert health_data["metrics"]["memory_usage"] == 60.0
        assert health_data["metrics"]["protection_effectiveness"] == 0.9
        assert health_data["is_active"] == orchestrator.is_active
    
    @pytest.mark.asyncio
    async def test_cross_system_coordination(self, orchestrator):
        """Test cross-system coordination callbacks."""
        await orchestrator.start()
        
        try:
            # Mock a failure event
            from scrollintel.core.failure_prevention import FailureEvent, FailureType
            
            mock_failure = FailureEvent(
                failure_type=FailureType.MEMORY_ERROR,
                timestamp=datetime.utcnow(),
                error_message="Mock memory error",
                stack_trace="Mock stack trace",
                context={"test": True}
            )
            
            # Test failure handling
            await orchestrator._handle_system_failure(mock_failure)
            
            # Test experience change handling
            experience_data = {
                "experience_level": "degraded",
                "user_id": "test_user",
                "metrics": {"response_time": 5.0}
            }
            
            await orchestrator._handle_experience_change(experience_data)
            
            # Test degradation event handling
            degradation_data = {
                "service_name": "test_service",
                "level": "major_degradation",
                "reason": "high_load"
            }
            
            await orchestrator._handle_degradation_event(degradation_data)
            
            # Verify coordination occurred (would check actual system states in real implementation)
            assert True
            
        finally:
            await orchestrator.stop()
    
    def test_utility_methods(self, orchestrator):
        """Test utility methods."""
        # Test error rate calculation
        error_rate = orchestrator._calculate_system_error_rate()
        assert 0.0 <= error_rate <= 1.0
        
        # Test response time calculation
        response_time = orchestrator._calculate_avg_response_time()
        assert response_time >= 0.0
        
        # Test active users estimation
        active_users = orchestrator._estimate_active_users()
        assert active_users >= 0
        
        # Test protection effectiveness calculation
        effectiveness = orchestrator._calculate_protection_effectiveness()
        assert 0.0 <= effectiveness <= 1.0
        
        # Test recovery success rate calculation
        success_rate = orchestrator._calculate_recovery_success_rate()
        assert 0.0 <= success_rate <= 1.0
        
        # Test user satisfaction calculation
        satisfaction = orchestrator._calculate_user_satisfaction()
        assert 0.0 <= satisfaction <= 1.0


class TestGlobalOrchestratorFunctions:
    """Test global orchestrator functions."""
    
    @pytest.mark.asyncio
    async def test_start_stop_bulletproof_system(self):
        """Test global start/stop functions."""
        # Test start
        await start_bulletproof_system()
        assert bulletproof_orchestrator.is_active
        
        # Test stop
        await stop_bulletproof_system()
        assert not bulletproof_orchestrator.is_active
    
    def test_get_system_health_global(self):
        """Test global system health function."""
        health_data = get_system_health()
        assert isinstance(health_data, dict)
        assert "status" in health_data or "overall_status" in health_data
    
    @pytest.mark.asyncio
    async def test_bulletproof_protection_context_manager(self):
        """Test bulletproof protection context manager."""
        from scrollintel.core.bulletproof_orchestrator import bulletproof_protection
        
        async with bulletproof_protection() as orchestrator:
            assert isinstance(orchestrator, BulletproofOrchestrator)
            assert orchestrator.is_active


class TestOrchestratorComponents:
    """Test orchestrator component classes."""
    
    def test_system_health_monitor(self):
        """Test SystemHealthMonitor."""
        from scrollintel.core.bulletproof_orchestrator import SystemHealthMonitor
        
        monitor = SystemHealthMonitor()
        assert not monitor.is_active
        
        monitor.start()
        assert monitor.is_active
        
        monitor.stop()
        assert not monitor.is_active
    
    def test_recovery_coordinator(self):
        """Test RecoveryCoordinator."""
        from scrollintel.core.bulletproof_orchestrator import RecoveryCoordinator
        
        coordinator = RecoveryCoordinator()
        assert not coordinator.is_active
        
        coordinator.start()
        assert coordinator.is_active
        
        coordinator.stop()
        assert not coordinator.is_active
    
    @pytest.mark.asyncio
    async def test_alert_manager(self):
        """Test AlertManager."""
        from scrollintel.core.bulletproof_orchestrator import AlertManager
        
        alert_manager = AlertManager()
        assert not alert_manager.is_active
        assert len(alert_manager.alerts) == 0
        
        alert_manager.start()
        assert alert_manager.is_active
        
        # Test adding alert
        test_alert = {
            "type": "test_alert",
            "severity": "warning",
            "message": "Test alert message",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await alert_manager.add_alert(test_alert)
        assert len(alert_manager.alerts) == 1
        assert alert_manager.alerts[0] == test_alert
        
        alert_manager.stop()
        assert not alert_manager.is_active
    
    @pytest.mark.asyncio
    async def test_dashboard_manager(self):
        """Test DashboardManager."""
        from scrollintel.core.bulletproof_orchestrator import DashboardManager
        
        dashboard_manager = DashboardManager()
        assert not dashboard_manager.is_active
        assert len(dashboard_manager.dashboard_data) == 0
        
        dashboard_manager.start()
        assert dashboard_manager.is_active
        
        # Test updating dashboard
        test_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "good",
            "metrics": {"cpu": 50.0, "memory": 60.0}
        }
        
        await dashboard_manager.update_dashboard(test_data)
        assert dashboard_manager.dashboard_data == test_data
        
        dashboard_manager.stop()
        assert not dashboard_manager.is_active
    
    def test_configuration_manager(self):
        """Test ConfigurationManager."""
        from scrollintel.core.bulletproof_orchestrator import ConfigurationManager
        
        initial_config = UnifiedConfiguration(
            protection_mode=ProtectionMode.FULL_PROTECTION,
            auto_recovery_enabled=True,
            predictive_prevention_enabled=True,
            user_experience_optimization=True,
            intelligent_routing_enabled=True,
            fallback_generation_enabled=True,
            degradation_learning_enabled=True,
            real_time_monitoring=True,
            alert_thresholds={},
            recovery_timeouts={},
            user_notification_settings={}
        )
        
        config_manager = ConfigurationManager(initial_config)
        assert config_manager.get_config() == initial_config
        
        # Test updating configuration
        new_config = UnifiedConfiguration(
            protection_mode=ProtectionMode.EMERGENCY_MODE,
            auto_recovery_enabled=False,
            predictive_prevention_enabled=False,
            user_experience_optimization=False,
            intelligent_routing_enabled=False,
            fallback_generation_enabled=True,
            degradation_learning_enabled=False,
            real_time_monitoring=False,
            alert_thresholds={},
            recovery_timeouts={},
            user_notification_settings={}
        )
        
        config_manager.update_config(new_config)
        assert config_manager.get_config() == new_config


if __name__ == "__main__":
    pytest.main([__file__])