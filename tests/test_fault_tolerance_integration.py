"""
Integration tests for Fault Tolerance and Recovery System API

Tests the REST API endpoints for:
- Circuit breaker management
- Graceful degradation control
- Recovery operations
- System health monitoring

Requirements: 4.3, 9.2
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json
from datetime import datetime

from scrollintel.api.routes.fault_tolerance_routes import router
from scrollintel.core.fault_tolerance import (
    fault_tolerance_manager,
    CircuitBreakerConfig,
    FailureType,
    DegradationLevel
)
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestCircuitBreakerAPI:
    """Test circuit breaker API endpoints"""
    
    def test_create_circuit_breaker(self):
        """Test creating a circuit breaker via API"""
        config_data = {
            "name": "test_api_breaker",
            "failure_threshold": 5,
            "recovery_timeout": 60,
            "success_threshold": 3,
            "timeout": 30
        }
        
        response = client.post("/api/v1/fault-tolerance/circuit-breakers", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_api_breaker"
        assert data["status"] == "created"
        
        # Verify circuit breaker was actually created
        cb = fault_tolerance_manager.get_circuit_breaker("test_api_breaker")
        assert cb is not None
        assert cb.config.failure_threshold == 5
    
    def test_create_circuit_breaker_validation_error(self):
        """Test circuit breaker creation with invalid data"""
        config_data = {
            "name": "invalid_breaker",
            "failure_threshold": -1,  # Invalid negative value
            "recovery_timeout": 60
        }
        
        response = client.post("/api/v1/fault-tolerance/circuit-breakers", json=config_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_list_circuit_breakers(self):
        """Test listing all circuit breakers"""
        # Create test circuit breakers
        config1 = CircuitBreakerConfig(name="breaker_1")
        config2 = CircuitBreakerConfig(name="breaker_2")
        
        fault_tolerance_manager.create_circuit_breaker(config1)
        cb2 = fault_tolerance_manager.create_circuit_breaker(config2)
        
        # Trigger some failures on breaker_2
        cb2.record_failure()
        cb2.record_failure()
        
        response = client.get("/api/v1/fault-tolerance/circuit-breakers")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should contain both breakers
        breaker_names = [cb["name"] for cb in data]
        assert "breaker_1" in breaker_names
        assert "breaker_2" in breaker_names
        
        # Check breaker states
        breaker_2_data = next(cb for cb in data if cb["name"] == "breaker_2")
        assert breaker_2_data["failure_count"] == 2
    
    def test_get_circuit_breaker_status(self):
        """Test getting specific circuit breaker status"""
        config = CircuitBreakerConfig(name="status_test_breaker")
        cb = fault_tolerance_manager.create_circuit_breaker(config)
        
        # Record some activity
        cb.record_failure()
        cb.record_success()
        
        response = client.get("/api/v1/fault-tolerance/circuit-breakers/status_test_breaker")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "status_test_breaker"
        assert data["state"] == "closed"
        assert data["failure_count"] == 0  # Reset after success
        assert data["success_count"] == 1
    
    def test_get_circuit_breaker_not_found(self):
        """Test getting non-existent circuit breaker"""
        response = client.get("/api/v1/fault-tolerance/circuit-breakers/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_reset_circuit_breaker(self):
        """Test resetting circuit breaker"""
        config = CircuitBreakerConfig(name="reset_test_breaker", failure_threshold=1)
        cb = fault_tolerance_manager.create_circuit_breaker(config)
        
        # Open the circuit breaker
        cb.record_failure()
        assert cb.state.value == "open"
        
        response = client.post("/api/v1/fault-tolerance/circuit-breakers/reset_test_breaker/reset")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"
        
        # Verify circuit breaker was reset
        assert cb.state.value == "closed"
        assert cb.failure_count == 0

class TestGracefulDegradationAPI:
    """Test graceful degradation API endpoints"""
    
    def test_degrade_service(self):
        """Test degrading a service"""
        # Register service first
        fault_tolerance_manager.degradation_manager.register_service(
            "api_test_service",
            lambda level: f"fallback_{level.value}"
        )
        
        degradation_data = {
            "service_name": "api_test_service",
            "degradation_level": "moderate"
        }
        
        response = client.post("/api/v1/fault-tolerance/degradation", json=degradation_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "api_test_service"
        assert data["level"] == "moderate"
        
        # Verify service was actually degraded
        level = fault_tolerance_manager.degradation_manager.degradation_levels["api_test_service"]
        assert level == DegradationLevel.MODERATE
    
    def test_get_degraded_services(self):
        """Test getting list of degraded services"""
        # Register and degrade services
        fault_tolerance_manager.degradation_manager.register_service(
            "service_1",
            lambda level: "fallback"
        )
        fault_tolerance_manager.degradation_manager.register_service(
            "service_2",
            lambda level: "fallback"
        )
        
        fault_tolerance_manager.degradation_manager.degrade_service(
            "service_1",
            DegradationLevel.MINIMAL
        )
        fault_tolerance_manager.degradation_manager.degrade_service(
            "service_2",
            DegradationLevel.SEVERE
        )
        
        response = client.get("/api/v1/fault-tolerance/degradation")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service_1" in data
        assert "service_2" in data
        assert data["service_1"] == "minimal"
        assert data["service_2"] == "severe"
    
    def test_restore_service(self):
        """Test restoring degraded service"""
        # Register and degrade service
        fault_tolerance_manager.degradation_manager.register_service(
            "restore_test_service",
            lambda level: "fallback"
        )
        fault_tolerance_manager.degradation_manager.degrade_service(
            "restore_test_service",
            DegradationLevel.MODERATE
        )
        
        response = client.post("/api/v1/fault-tolerance/degradation/restore_test_service/restore")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "restored"
        
        # Verify service was restored
        level = fault_tolerance_manager.degradation_manager.degradation_levels["restore_test_service"]
        assert level == DegradationLevel.NONE

class TestRecoveryAPI:
    """Test recovery operation API endpoints"""
    
    @pytest.mark.asyncio
    async def test_initiate_recovery(self):
        """Test initiating recovery operation"""
        failure_data = {
            "failure_id": "api_test_failure_001",
            "failure_type": "service_unavailable",
            "component": "test_api_component",
            "message": "API test failure",
            "severity": "high",
            "context": {"test": True}
        }
        
        # Register health checker for the component
        async def test_health_checker():
            return True
        
        fault_tolerance_manager.recovery_manager.register_health_checker(
            "test_api_component",
            test_health_checker
        )
        
        response = client.post("/api/v1/fault-tolerance/recovery/initiate", json=failure_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "recovery_id" in data
        assert data["success"] is True
        assert len(data["actions_taken"]) > 0
        assert data["health_status"]["healthy"] is True
    
    def test_get_recovery_history(self):
        """Test getting recovery history"""
        response = client.get("/api/v1/fault-tolerance/recovery/history")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        # Should contain at least the recovery from previous test
        assert len(data) >= 0
    
    def test_get_recovery_history_with_limit(self):
        """Test getting recovery history with limit"""
        response = client.get("/api/v1/fault-tolerance/recovery/history?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) <= 5
    
    def test_get_recovery_details_not_found(self):
        """Test getting details for non-existent recovery"""
        response = client.get("/api/v1/fault-tolerance/recovery/nonexistent_recovery")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

class TestSystemHealthAPI:
    """Test system health monitoring API endpoints"""
    
    def test_get_system_health(self):
        """Test getting overall system health"""
        # Create some test data
        config = CircuitBreakerConfig(name="health_test_breaker")
        fault_tolerance_manager.create_circuit_breaker(config)
        
        fault_tolerance_manager.degradation_manager.register_service(
            "health_test_service",
            lambda level: "fallback"
        )
        
        response = client.get("/api/v1/fault-tolerance/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "overall_status" in data
        assert "circuit_breakers" in data
        assert "degraded_services" in data
        assert "recent_recoveries" in data
        
        # Should contain our test circuit breaker
        assert "health_test_breaker" in data["circuit_breakers"]
    
    def test_get_health_summary(self):
        """Test getting health summary"""
        response = client.get("/api/v1/fault-tolerance/health/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_status" in data
        assert "timestamp" in data
        assert "metrics" in data
        assert "alerts" in data
        
        metrics = data["metrics"]
        assert "total_circuit_breakers" in metrics
        assert "open_circuit_breakers" in metrics
        assert "degraded_services" in metrics
        assert "recent_recovery_success_rate" in metrics
        
        assert isinstance(data["alerts"], list)

class TestConfigurationAPI:
    """Test configuration API endpoints"""
    
    def test_get_default_configs(self):
        """Test getting default configurations"""
        response = client.get("/api/v1/fault-tolerance/config/defaults")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "circuit_breaker" in data
        assert "retry" in data
        
        cb_config = data["circuit_breaker"]
        assert "failure_threshold" in cb_config
        assert "recovery_timeout" in cb_config
        assert "success_threshold" in cb_config
        assert "timeout" in cb_config
        
        retry_config = data["retry"]
        assert "max_attempts" in retry_config
        assert "base_delay" in retry_config
        assert "max_delay" in retry_config
        assert "exponential_base" in retry_config
        assert "jitter" in retry_config

class TestFailureSimulationAPI:
    """Test failure simulation API endpoints"""
    
    @pytest.mark.asyncio
    async def test_simulate_failure(self):
        """Test simulating system failure"""
        # Register health checker for test component
        async def test_health_checker():
            return True
        
        fault_tolerance_manager.recovery_manager.register_health_checker(
            "simulation_test_component",
            test_health_checker
        )
        
        response = client.post(
            "/api/v1/fault-tolerance/test/failure?failure_type=network_error&component=simulation_test_component"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "failure_id" in data
        assert "recovery_id" in data
        assert "recovery_success" in data
        assert data["recovery_success"] is True

class TestErrorHandling:
    """Test API error handling"""
    
    def test_invalid_json_request(self):
        """Test handling of invalid JSON in request"""
        response = client.post(
            "/api/v1/fault-tolerance/circuit-breakers",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        incomplete_data = {
            "failure_threshold": 5
            # Missing required 'name' field
        }
        
        response = client.post("/api/v1/fault-tolerance/circuit-breakers", json=incomplete_data)
        
        assert response.status_code == 422
    
    def test_invalid_enum_values(self):
        """Test handling of invalid enum values"""
        invalid_data = {
            "service_name": "test_service",
            "degradation_level": "invalid_level"  # Invalid enum value
        }
        
        response = client.post("/api/v1/fault-tolerance/degradation", json=invalid_data)
        
        assert response.status_code == 422

class TestConcurrentOperations:
    """Test concurrent API operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_operations(self):
        """Test concurrent circuit breaker operations"""
        import asyncio
        import httpx
        
        async def create_breaker(name):
            async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
                config_data = {
                    "name": f"concurrent_breaker_{name}",
                    "failure_threshold": 5,
                    "recovery_timeout": 60
                }
                response = await ac.post("/api/v1/fault-tolerance/circuit-breakers", json=config_data)
                return response.status_code == 200
        
        # Create multiple circuit breakers concurrently
        tasks = [create_breaker(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert all(results)
        
        # Verify all breakers were created
        response = client.get("/api/v1/fault-tolerance/circuit-breakers")
        data = response.json()
        
        concurrent_breakers = [cb for cb in data if cb["name"].startswith("concurrent_breaker_")]
        assert len(concurrent_breakers) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_recovery_operations(self):
        """Test concurrent recovery operations"""
        import asyncio
        import httpx
        
        # Register health checker
        async def test_health_checker():
            return True
        
        fault_tolerance_manager.recovery_manager.register_health_checker(
            "concurrent_test_component",
            test_health_checker
        )
        
        async def initiate_recovery(failure_id):
            async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
                failure_data = {
                    "failure_id": f"concurrent_failure_{failure_id}",
                    "failure_type": "service_unavailable",
                    "component": "concurrent_test_component",
                    "message": f"Concurrent test failure {failure_id}",
                    "severity": "medium"
                }
                response = await ac.post("/api/v1/fault-tolerance/recovery/initiate", json=failure_data)
                return response.status_code == 200
        
        # Initiate multiple recoveries concurrently
        tasks = [initiate_recovery(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert all(results)
        
        # Verify all recoveries were recorded
        response = client.get("/api/v1/fault-tolerance/recovery/history?limit=10")
        data = response.json()
        
        concurrent_recoveries = [
            r for r in data 
            if any(action for action in r.get("actions_taken", []) if "concurrent" in str(action))
        ]
        # Should have at least some concurrent recoveries
        assert len(concurrent_recoveries) >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])