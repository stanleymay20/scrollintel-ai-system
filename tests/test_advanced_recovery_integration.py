"""
Integration tests for Advanced Recovery and Self-Healing System.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from scrollintel.core.advanced_recovery_system import advanced_recovery_system
from scrollintel.api.routes.advanced_recovery_routes import router


class TestAdvancedRecoveryIntegration:
    """Integration tests for the advanced recovery system."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    def test_get_system_status_endpoint(self, client):
        """Test system status endpoint."""
        response = client.get("/api/advanced-recovery/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "auto_recovery_enabled" in data
        assert "monitoring_active" in data
        assert "dependency_health" in data
        assert "maintenance_tasks" in data
        assert "performance_thresholds" in data
    
    def test_check_node_health_endpoint(self, client):
        """Test node health check endpoint."""
        with patch.object(advanced_recovery_system, 'perform_health_check', return_value=0.8):
            response = client.get("/api/advanced-recovery/health/database")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["node_name"] == "database"
            assert data["health_score"] == 0.8
            assert "status" in data
            assert "timestamp" in data
    
    def test_trigger_autonomous_repair_endpoint(self, client):
        """Test autonomous repair trigger endpoint."""
        response = client.post("/api/advanced-recovery/repair/ai_services")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert data["node_name"] == "ai_services"
        assert "timestamp" in data
    
    def test_get_dependency_status_endpoint(self, client):
        """Test dependency status endpoint."""
        with patch.object(advanced_recovery_system, 'intelligent_dependency_management') as mock_dep:
            mock_dep.return_value = {
                "database": {"health_score": 0.9, "status": "healthy"},
                "ai_services": {"health_score": 0.8, "status": "healthy"}
            }
            
            response = client.get("/api/advanced-recovery/dependencies")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "database" in data
            assert "ai_services" in data
    
    def test_get_performance_status_endpoint(self, client):
        """Test performance status endpoint."""
        with patch.object(advanced_recovery_system, 'self_optimizing_performance_tuning') as mock_perf:
            mock_perf.return_value = {
                "current_metrics": {"cpu_usage": 45.0, "memory_usage": 60.0},
                "optimizations_applied": {},
                "performance_trend": {"cpu_trend": "stable"}
            }
            
            response = client.get("/api/advanced-recovery/performance")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "current_metrics" in data
            assert "optimizations_applied" in data
            assert "performance_trend" in data
    
    def test_trigger_performance_optimization_endpoint(self, client):
        """Test performance optimization trigger endpoint."""
        response = client.post("/api/advanced-recovery/optimize")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "timestamp" in data
    
    def test_get_maintenance_status_endpoint(self, client):
        """Test maintenance status endpoint."""
        with patch.object(advanced_recovery_system, 'predictive_maintenance') as mock_maint:
            mock_maint.return_value = {
                "scheduled_tasks": [],
                "completed_tasks": [],
                "predicted_issues": [],
                "preventive_actions": []
            }
            
            response = client.get("/api/advanced-recovery/maintenance")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "scheduled_tasks" in data
            assert "completed_tasks" in data
            assert "predicted_issues" in data
            assert "preventive_actions" in data
    
    def test_schedule_maintenance_endpoint(self, client):
        """Test maintenance scheduling endpoint."""
        response = client.post("/api/advanced-recovery/maintenance/schedule")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "timestamp" in data
    
    def test_toggle_auto_recovery_endpoint(self, client):
        """Test auto-recovery toggle endpoint."""
        # Enable auto-recovery
        response = client.post("/api/advanced-recovery/config/auto-recovery?enabled=true")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["auto_recovery_enabled"] is True
        assert "message" in data
        
        # Disable auto-recovery
        response = client.post("/api/advanced-recovery/config/auto-recovery?enabled=false")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["auto_recovery_enabled"] is False
    
    def test_get_performance_history_endpoint(self, client):
        """Test performance history endpoint."""
        response = client.get("/api/advanced-recovery/metrics/performance-history?limit=50")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics" in data
        assert "total_samples" in data
        assert "returned_samples" in data
        assert isinstance(data["metrics"], list)
    
    def test_get_recovery_patterns_endpoint(self, client):
        """Test recovery patterns endpoint."""
        response = client.get("/api/advanced-recovery/recovery-patterns")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "recovery_patterns" in data
        assert "optimization_rules" in data
        assert "timestamp" in data
        
        # Check that recovery patterns are properly formatted
        patterns = data["recovery_patterns"]
        assert isinstance(patterns, dict)
        assert len(patterns) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_recovery_scenario(self):
        """Test end-to-end recovery scenario."""
        # Simulate a failing database
        database_node = advanced_recovery_system.dependency_graph['database']
        original_health = database_node.health_score
        database_node.health_score = 0.2  # Simulate failure
        database_node.failure_count = 3
        
        try:
            # Trigger dependency management (should detect failure)
            with patch.object(advanced_recovery_system, 'perform_health_check', return_value=0.2), \
                 patch.object(advanced_recovery_system, 'autonomous_system_repair', return_value=True):
                
                status = await advanced_recovery_system.intelligent_dependency_management()
                
                # Should detect the failing database
                assert 'database' in status
                assert status['database']['health_score'] == 0.2
                
                # Should attempt recovery
                assert status['database'].get('recovery_attempted') is True
                
        finally:
            # Restore original state
            database_node.health_score = original_health
            database_node.failure_count = 0
    
    @pytest.mark.asyncio
    async def test_performance_optimization_scenario(self):
        """Test performance optimization scenario."""
        # Add performance history showing degradation
        from scrollintel.core.advanced_recovery_system import PerformanceMetrics
        
        # Clear existing history
        advanced_recovery_system.performance_history.clear()
        
        # Add metrics showing increasing resource usage
        for i in range(15):
            metrics = PerformanceMetrics(
                cpu_usage=60 + i * 2,  # Increasing CPU
                memory_usage=50 + i * 1.5,  # Increasing memory
                disk_io=10.0,
                network_io=5.0,
                response_time=1.0 + i * 0.1,  # Increasing response time
                throughput=100.0,
                error_rate=0.01
            )
            advanced_recovery_system.performance_history.append(metrics)
        
        # Mock optimization functions
        with patch.object(advanced_recovery_system, '_optimize_cpu_usage', return_value=True), \
             patch.object(advanced_recovery_system, '_optimize_memory_usage', return_value=True), \
             patch.object(advanced_recovery_system, '_collect_performance_metrics') as mock_collect:
            
            # Mock current high resource usage
            mock_collect.return_value = PerformanceMetrics(
                cpu_usage=85.0, memory_usage=80.0, disk_io=15.0,
                network_io=8.0, response_time=2.5, throughput=80.0, error_rate=0.03
            )
            
            result = await advanced_recovery_system.self_optimizing_performance_tuning()
            
            assert 'current_metrics' in result
            assert 'optimizations_applied' in result
            assert 'performance_trend' in result
            
            # Should apply optimizations due to high resource usage
            optimizations = result['optimizations_applied']
            assert len(optimizations) > 0
    
    @pytest.mark.asyncio
    async def test_predictive_maintenance_scenario(self):
        """Test predictive maintenance scenario."""
        # Add performance history showing concerning trends
        from scrollintel.core.advanced_recovery_system import PerformanceMetrics
        
        # Clear existing history
        advanced_recovery_system.performance_history.clear()
        
        # Add metrics showing concerning trends
        for i in range(25):
            metrics = PerformanceMetrics(
                cpu_usage=70 + i * 1.5,  # Rapidly increasing
                memory_usage=60 + i * 1.2,  # Increasing
                disk_io=10.0,
                network_io=5.0,
                response_time=1.0 + i * 0.08,  # Increasing
                throughput=100.0,
                error_rate=0.01
            )
            advanced_recovery_system.performance_history.append(metrics)
        
        # Mock disk usage to trigger disk space prediction
        with patch('psutil.disk_usage') as mock_disk, \
             patch.object(advanced_recovery_system, '_execute_maintenance_task', return_value=True):
            
            mock_disk.return_value.percent = 88.0  # High disk usage
            
            result = await advanced_recovery_system.predictive_maintenance()
            
            assert 'scheduled_tasks' in result
            assert 'predicted_issues' in result
            assert 'preventive_actions' in result
            
            # Should predict issues based on trends
            issues = result['predicted_issues']
            assert len(issues) > 0
            
            # Should schedule maintenance tasks
            tasks = result['scheduled_tasks']
            assert len(tasks) > 0
    
    @pytest.mark.asyncio
    async def test_cascade_failure_prevention_scenario(self):
        """Test cascade failure prevention scenario."""
        # Simulate database failure
        database_node = advanced_recovery_system.dependency_graph['database']
        original_health = database_node.health_score
        database_node.health_score = 0.1  # Critical failure
        
        try:
            with patch.object(advanced_recovery_system, 'perform_health_check') as mock_health, \
                 patch.object(advanced_recovery_system, '_strengthen_node') as mock_strengthen:
                
                # Mock health checks - database failing, others healthy but at risk
                def health_side_effect(node_name):
                    if node_name == 'database':
                        return 0.1
                    else:
                        return 0.7  # Healthy but could be affected
                
                mock_health.side_effect = health_side_effect
                
                # Run dependency management
                await advanced_recovery_system.intelligent_dependency_management()
                
                # Should strengthen nodes that depend on the failing database
                mock_strengthen.assert_called()
                
        finally:
            # Restore original state
            database_node.health_score = original_health
    
    def test_error_handling_in_endpoints(self, client):
        """Test error handling in API endpoints."""
        # Test with invalid node name
        response = client.get("/api/advanced-recovery/health/invalid_node")
        
        # Should handle gracefully (might return 200 with 0 health score or 500)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["node_name"] == "invalid_node"
            assert data["health_score"] == 0.0
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self):
        """Test monitoring system integration."""
        # Ensure monitoring is active
        if not advanced_recovery_system.maintenance_scheduler_active:
            advanced_recovery_system._start_monitoring()
        
        # Wait a short time for monitoring to run
        await asyncio.sleep(1)
        
        # Check that system status reflects monitoring activity
        status = advanced_recovery_system.get_system_status()
        assert status["monitoring_active"] is True
        
        # Stop monitoring for cleanup
        advanced_recovery_system.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_recovery_system_resilience(self):
        """Test that recovery system itself is resilient to errors."""
        # Test with mocked exceptions in various components
        with patch.object(advanced_recovery_system, 'perform_health_check', side_effect=Exception("Test error")):
            # Should handle health check errors gracefully
            try:
                await advanced_recovery_system.intelligent_dependency_management()
                # Should not raise exception
            except Exception as e:
                pytest.fail(f"Recovery system should handle errors gracefully: {e}")
        
        with patch.object(advanced_recovery_system, '_collect_performance_metrics', side_effect=Exception("Test error")):
            # Should handle performance collection errors gracefully
            try:
                await advanced_recovery_system.self_optimizing_performance_tuning()
                # Should not raise exception
            except Exception as e:
                pytest.fail(f"Recovery system should handle errors gracefully: {e}")
    
    def test_configuration_persistence(self):
        """Test that configuration changes persist."""
        original_enabled = advanced_recovery_system.auto_recovery_enabled
        
        try:
            # Change configuration
            advanced_recovery_system.auto_recovery_enabled = False
            
            # Check that change is reflected in status
            status = advanced_recovery_system.get_system_status()
            assert status["auto_recovery_enabled"] is False
            
            # Change back
            advanced_recovery_system.auto_recovery_enabled = True
            status = advanced_recovery_system.get_system_status()
            assert status["auto_recovery_enabled"] is True
            
        finally:
            # Restore original state
            advanced_recovery_system.auto_recovery_enabled = original_enabled
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent recovery operations."""
        # Run multiple operations concurrently
        tasks = [
            advanced_recovery_system.perform_health_check('database'),
            advanced_recovery_system.perform_health_check('ai_services'),
            advanced_recovery_system.perform_health_check('file_system'),
            advanced_recovery_system._collect_performance_metrics()
        ]
        
        # Should complete without errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception), f"Concurrent operation failed: {result}"
    
    def test_system_status_completeness(self):
        """Test that system status includes all expected information."""
        status = advanced_recovery_system.get_system_status()
        
        # Check all required fields are present
        required_fields = [
            'timestamp', 'auto_recovery_enabled', 'monitoring_active',
            'dependency_health', 'maintenance_tasks', 'performance_thresholds'
        ]
        
        for field in required_fields:
            assert field in status, f"Missing required field: {field}"
        
        # Check dependency health includes all nodes
        dependency_health = status['dependency_health']
        expected_nodes = ['database', 'file_system', 'ai_services', 'visualization_engine', 'web_server']
        
        for node in expected_nodes:
            assert node in dependency_health, f"Missing dependency health for: {node}"
            
            node_health = dependency_health[node]
            assert 'health_score' in node_health
            assert 'failure_count' in node_health
            assert 'recovery_attempts' in node_health
            assert 'last_check' in node_health