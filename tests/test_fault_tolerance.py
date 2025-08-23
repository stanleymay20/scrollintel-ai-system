"""
Comprehensive tests for Fault Tolerance and Recovery System

Tests all components:
- Circuit breaker pattern implementation
- Retry logic with exponential backoff
- Graceful degradation system
- Automated recovery procedures

Requirements: 4.3, 9.2
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from scrollintel.core.fault_tolerance import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    RetryManager,
    RetryConfig,
    GracefulDegradationManager,
    DegradationLevel,
    AutomatedRecoveryManager,
    FaultToleranceManager,
    SystemFailure,
    FailureType,
    RecoveryResult
)

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        config = CircuitBreakerConfig(
            name="test_breaker",
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2
        )
        
        cb = CircuitBreaker(config)
        
        assert cb.config.name == "test_breaker"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert not cb.is_open()
    
    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opens after failure threshold"""
        config = CircuitBreakerConfig(name="test", failure_threshold=2)
        cb = CircuitBreaker(config)
        
        # First failure
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 2
        assert cb.is_open()
    
    def test_circuit_breaker_recovery_timeout(self):
        """Test circuit breaker recovery after timeout"""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            recovery_timeout=1  # 1 second
        )
        cb = CircuitBreaker(config)
        
        # Trigger failure to open circuit
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Should still be open immediately
        assert cb.is_open()
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should move to half-open
        assert not cb.is_open()
        assert cb.state == CircuitBreakerState.HALF_OPEN
    
    def test_circuit_breaker_half_open_success(self):
        """Test circuit breaker closes after successful operations in half-open"""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            success_threshold=2
        )
        cb = CircuitBreaker(config)
        
        # Open circuit
        cb.record_failure()
        cb.state = CircuitBreakerState.HALF_OPEN
        
        # First success
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Second success - should close circuit
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker reopens on failure in half-open state"""
        config = CircuitBreakerConfig(name="test", failure_threshold=1)
        cb = CircuitBreaker(config)
        
        # Move to half-open
        cb.state = CircuitBreakerState.HALF_OPEN
        
        # Failure should reopen circuit
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

class TestRetryManager:
    """Test retry logic functionality"""
    
    def test_calculate_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=False
        )
        
        # Test exponential progression
        assert RetryManager.calculate_delay(1, config) == 1.0
        assert RetryManager.calculate_delay(2, config) == 2.0
        assert RetryManager.calculate_delay(3, config) == 4.0
        assert RetryManager.calculate_delay(4, config) == 8.0
    
    def test_calculate_delay_max_limit(self):
        """Test delay respects maximum limit"""
        config = RetryConfig(
            base_delay=10.0,
            exponential_base=3.0,
            max_delay=20.0,
            jitter=False
        )
        
        # Should be capped at max_delay
        delay = RetryManager.calculate_delay(5, config)
        assert delay <= 20.0
    
    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter"""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # With jitter, delays should vary slightly
        delays = [RetryManager.calculate_delay(2, config) for _ in range(10)]
        
        # All delays should be around 2.0 but with some variation
        assert all(1.8 <= delay <= 2.2 for delay in delays)
        assert len(set(delays)) > 1  # Should have variation
    
    def test_is_retryable_exception(self):
        """Test exception retryability checking"""
        config = RetryConfig(
            retryable_exceptions=[ConnectionError, TimeoutError]
        )
        
        assert RetryManager.is_retryable_exception(ConnectionError(), config)
        assert RetryManager.is_retryable_exception(TimeoutError(), config)
        assert not RetryManager.is_retryable_exception(ValueError(), config)

class TestGracefulDegradationManager:
    """Test graceful degradation functionality"""
    
    def test_service_registration(self):
        """Test service registration with fallback strategy"""
        manager = GracefulDegradationManager()
        
        def fallback_strategy(level, *args, **kwargs):
            return f"fallback_response_{level.value}"
        
        manager.register_service("test_service", fallback_strategy)
        
        assert "test_service" in manager.degradation_levels
        assert manager.degradation_levels["test_service"] == DegradationLevel.NONE
        assert manager.service_health["test_service"] is True
    
    def test_service_degradation(self):
        """Test service degradation levels"""
        manager = GracefulDegradationManager()
        
        def fallback_strategy(level, *args, **kwargs):
            return f"fallback_{level.value}"
        
        manager.register_service("test_service", fallback_strategy)
        
        # Degrade service
        manager.degrade_service("test_service", DegradationLevel.MODERATE)
        
        assert manager.degradation_levels["test_service"] == DegradationLevel.MODERATE
        assert manager.service_health["test_service"] is False
    
    def test_fallback_response(self):
        """Test fallback response generation"""
        manager = GracefulDegradationManager()
        
        def fallback_strategy(level, *args, **kwargs):
            if level == DegradationLevel.MODERATE:
                return "limited_functionality"
            elif level == DegradationLevel.SEVERE:
                return "basic_functionality"
            return "full_functionality"
        
        manager.register_service("test_service", fallback_strategy)
        
        # Test different degradation levels
        manager.degrade_service("test_service", DegradationLevel.MODERATE)
        response = manager.get_fallback_response("test_service")
        assert response == "limited_functionality"
        
        manager.degrade_service("test_service", DegradationLevel.SEVERE)
        response = manager.get_fallback_response("test_service")
        assert response == "basic_functionality"

class TestAutomatedRecoveryManager:
    """Test automated recovery functionality"""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create recovery manager for testing"""
        return AutomatedRecoveryManager()
    
    @pytest.fixture
    def sample_failure(self):
        """Create sample system failure"""
        return SystemFailure(
            failure_id="test_failure_001",
            failure_type=FailureType.SERVICE_UNAVAILABLE,
            component="test_service",
            message="Service is unavailable",
            timestamp=datetime.now(),
            severity="high"
        )
    
    def test_recovery_strategy_registration(self, recovery_manager):
        """Test recovery strategy registration"""
        async def test_strategy(failure):
            return ["test_action_1", "test_action_2"]
        
        recovery_manager.register_recovery_strategy(
            FailureType.SERVICE_UNAVAILABLE,
            test_strategy
        )
        
        assert FailureType.SERVICE_UNAVAILABLE in recovery_manager.recovery_strategies
    
    def test_health_checker_registration(self, recovery_manager):
        """Test health checker registration"""
        async def test_health_checker():
            return True
        
        recovery_manager.register_health_checker("test_component", test_health_checker)
        
        assert "test_component" in recovery_manager.health_checkers
    
    @pytest.mark.asyncio
    async def test_initiate_recovery_success(self, recovery_manager, sample_failure):
        """Test successful recovery initiation"""
        # Register recovery strategy
        async def recovery_strategy(failure):
            return ["restarted_service", "verified_health"]
        
        recovery_manager.register_recovery_strategy(
            FailureType.SERVICE_UNAVAILABLE,
            recovery_strategy
        )
        
        # Register health checker
        async def health_checker():
            return True
        
        recovery_manager.register_health_checker("test_service", health_checker)
        
        # Initiate recovery
        result = await recovery_manager.initiate_recovery(sample_failure)
        
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert "restarted_service" in result.actions_taken
        assert result.health_status["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_initiate_recovery_failure(self, recovery_manager, sample_failure):
        """Test recovery failure handling"""
        # Register failing health checker
        async def failing_health_checker():
            return False
        
        recovery_manager.register_health_checker("test_service", failing_health_checker)
        
        # Initiate recovery
        result = await recovery_manager.initiate_recovery(sample_failure)
        
        assert isinstance(result, RecoveryResult)
        assert result.success is False
        assert result.health_status["healthy"] is False
    
    @pytest.mark.asyncio
    async def test_recovery_history_tracking(self, recovery_manager, sample_failure):
        """Test recovery history is properly tracked"""
        initial_count = len(recovery_manager.recovery_history)
        
        # Initiate recovery
        await recovery_manager.initiate_recovery(sample_failure)
        
        assert len(recovery_manager.recovery_history) == initial_count + 1
        
        # Check latest recovery record
        latest_recovery = recovery_manager.recovery_history[-1]
        assert latest_recovery.recovery_id.startswith("recovery_")

class TestFaultToleranceManager:
    """Test main fault tolerance manager"""
    
    @pytest.fixture
    def ft_manager(self):
        """Create fault tolerance manager for testing"""
        return FaultToleranceManager()
    
    def test_circuit_breaker_creation(self, ft_manager):
        """Test circuit breaker creation and retrieval"""
        config = CircuitBreakerConfig(name="test_cb")
        
        cb = ft_manager.create_circuit_breaker(config)
        
        assert cb is not None
        assert ft_manager.get_circuit_breaker("test_cb") is cb
    
    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_success(self, ft_manager):
        """Test successful operation with circuit breaker"""
        config = CircuitBreakerConfig(name="test_operation")
        
        async def successful_operation():
            return "success"
        
        result = await ft_manager.execute_with_circuit_breaker(
            successful_operation,
            config
        )
        
        assert result == "success"
        
        # Circuit breaker should remain closed
        cb = ft_manager.get_circuit_breaker("test_operation")
        assert cb.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_failure(self, ft_manager):
        """Test operation failure with circuit breaker"""
        config = CircuitBreakerConfig(
            name="failing_operation",
            failure_threshold=1
        )
        
        async def failing_operation():
            raise Exception("Operation failed")
        
        # First failure should open circuit
        with pytest.raises(Exception):
            await ft_manager.execute_with_circuit_breaker(
                failing_operation,
                config
            )
        
        cb = ft_manager.get_circuit_breaker("failing_operation")
        assert cb.state == CircuitBreakerState.OPEN
        
        # Second attempt should fail immediately due to open circuit
        with pytest.raises(Exception, match="Circuit breaker .* is open"):
            await ft_manager.execute_with_circuit_breaker(
                failing_operation,
                config
            )
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, ft_manager):
        """Test successful operation with retry"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        
        call_count = 0
        
        async def eventually_successful_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await ft_manager.execute_with_retry(
            eventually_successful_operation,
            config
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_max_attempts(self, ft_manager):
        """Test retry respects maximum attempts"""
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        
        call_count = 0
        
        async def always_failing_operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            await ft_manager.execute_with_retry(
                always_failing_operation,
                config
            )
        
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_fault_tolerance_combined(self, ft_manager):
        """Test combined fault tolerance features"""
        cb_config = CircuitBreakerConfig(name="combined_test")
        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)
        
        call_count = 0
        
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First attempt fails")
            return "success"
        
        result = await ft_manager.execute_with_fault_tolerance(
            flaky_operation,
            circuit_breaker_config=cb_config,
            retry_config=retry_config
        )
        
        assert result == "success"
        assert call_count == 2
        
        # Circuit breaker should remain closed after successful retry
        cb = ft_manager.get_circuit_breaker("combined_test")
        assert cb.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, ft_manager):
        """Test operation with fallback"""
        async def failing_operation():
            raise Exception("Operation failed")
        
        async def fallback_operation():
            return "fallback_result"
        
        result = await ft_manager.execute_with_fault_tolerance(
            failing_operation,
            fallback=fallback_operation
        )
        
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_handle_system_failure(self, ft_manager):
        """Test system failure handling"""
        failure = SystemFailure(
            failure_id="test_failure",
            failure_type=FailureType.SERVICE_UNAVAILABLE,
            component="test_component",
            message="Test failure",
            timestamp=datetime.now(),
            severity="high"
        )
        
        # Register service for degradation
        ft_manager.degradation_manager.register_service(
            "test_component",
            lambda level: f"degraded_{level.value}"
        )
        
        result = await ft_manager.handle_system_failure(failure)
        
        assert isinstance(result, RecoveryResult)
        assert result.recovery_id.startswith("recovery_")
    
    def test_get_system_health(self, ft_manager):
        """Test system health status retrieval"""
        # Create some circuit breakers
        config1 = CircuitBreakerConfig(name="cb1")
        config2 = CircuitBreakerConfig(name="cb2")
        
        ft_manager.create_circuit_breaker(config1)
        cb2 = ft_manager.create_circuit_breaker(config2)
        
        # Open one circuit breaker
        cb2.record_failure()
        cb2.record_failure()
        cb2.record_failure()
        cb2.record_failure()
        cb2.record_failure()
        
        # Register and degrade a service
        ft_manager.degradation_manager.register_service(
            "test_service",
            lambda level: "fallback"
        )
        ft_manager.degradation_manager.degrade_service(
            "test_service",
            DegradationLevel.MODERATE
        )
        
        health = ft_manager.get_system_health()
        
        assert health["overall_status"] == "degraded"
        assert "cb1" in health["circuit_breakers"]
        assert "cb2" in health["circuit_breakers"]
        assert health["circuit_breakers"]["cb1"]["state"] == "closed"
        assert health["circuit_breakers"]["cb2"]["state"] == "open"
        assert "test_service" in health["degraded_services"]
        assert health["degraded_services"]["test_service"] == "moderate"

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure_scenario(self):
        """Test handling database connection failures"""
        ft_manager = FaultToleranceManager()
        
        # Configure circuit breaker for database operations
        db_cb_config = CircuitBreakerConfig(
            name="database_operations",
            failure_threshold=3,
            recovery_timeout=30
        )
        
        # Configure retry for transient failures
        db_retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            retryable_exceptions=[ConnectionError]
        )
        
        # Register database service for degradation
        ft_manager.degradation_manager.register_service(
            "database",
            lambda level: {"status": "degraded", "level": level.value}
        )
        
        # Simulate database operation
        connection_attempts = 0
        
        async def database_operation():
            nonlocal connection_attempts
            connection_attempts += 1
            
            if connection_attempts <= 2:
                raise ConnectionError("Database connection failed")
            return {"data": "success"}
        
        # Execute with fault tolerance
        result = await ft_manager.execute_with_fault_tolerance(
            database_operation,
            circuit_breaker_config=db_cb_config,
            retry_config=db_retry_config
        )
        
        assert result == {"data": "success"}
        assert connection_attempts == 3
        
        # Circuit breaker should be closed after success
        cb = ft_manager.get_circuit_breaker("database_operations")
        assert cb.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_api_service_outage_scenario(self):
        """Test handling external API service outage"""
        ft_manager = FaultToleranceManager()
        
        # Configure for API service
        api_cb_config = CircuitBreakerConfig(
            name="external_api",
            failure_threshold=2,
            recovery_timeout=60
        )
        
        # Register API service with fallback
        ft_manager.degradation_manager.register_service(
            "external_api",
            lambda level: {"data": "cached_response", "degraded": True}
        )
        
        # Simulate API failures
        async def api_call():
            raise Exception("API service unavailable")
        
        async def fallback_response():
            return ft_manager.degradation_manager.get_fallback_response("external_api")
        
        # First few calls should fail and open circuit
        for _ in range(3):
            try:
                await ft_manager.execute_with_circuit_breaker(api_call, api_cb_config)
            except Exception:
                pass
        
        # Circuit should be open now
        cb = ft_manager.get_circuit_breaker("external_api")
        assert cb.state == CircuitBreakerState.OPEN
        
        # Use fallback for subsequent calls
        result = await ft_manager.execute_with_fault_tolerance(
            api_call,
            circuit_breaker_config=api_cb_config,
            fallback=fallback_response
        )
        
        assert result["degraded"] is True
        assert result["data"] == "cached_response"
    
    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures"""
        ft_manager = FaultToleranceManager()
        
        # Simulate cascading failure
        failure1 = SystemFailure(
            failure_id="cascade_1",
            failure_type=FailureType.SERVICE_UNAVAILABLE,
            component="service_a",
            message="Service A failed",
            timestamp=datetime.now(),
            severity="high"
        )
        
        failure2 = SystemFailure(
            failure_id="cascade_2",
            failure_type=FailureType.RESOURCE_EXHAUSTED,
            component="service_b",
            message="Service B overloaded due to Service A failure",
            timestamp=datetime.now(),
            severity="critical"
        )
        
        # Register services for degradation
        ft_manager.degradation_manager.register_service(
            "service_a",
            lambda level: "service_a_fallback"
        )
        ft_manager.degradation_manager.register_service(
            "service_b",
            lambda level: "service_b_fallback"
        )
        
        # Handle cascading failures
        result1 = await ft_manager.handle_system_failure(failure1)
        result2 = await ft_manager.handle_system_failure(failure2)
        
        # Both recoveries should be tracked
        assert len(ft_manager.recovery_manager.recovery_history) >= 2
        
        # Check that recoveries were attempted
        assert result1.recovery_id.startswith("recovery_")
        assert result2.recovery_id.startswith("recovery_")
        
        # Verify recovery history contains our failures
        recovery_ids = [r.recovery_id for r in ft_manager.recovery_manager.recovery_history]
        assert result1.recovery_id in recovery_ids
        assert result2.recovery_id in recovery_ids

if __name__ == "__main__":
    pytest.main([__file__, "-v"])