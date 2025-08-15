"""
Comprehensive tests for the error handling system.
Tests error handling, recovery mechanisms, monitoring, and alerting.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from scrollintel.core.error_handling import (
    ErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory,
    RetryConfig, CircuitBreakerConfig, CircuitBreaker,
    with_error_handling, with_retry, with_circuit_breaker
)
from scrollintel.core.error_middleware import (
    ErrorHandlingMiddleware, ExternalServiceErrorHandler
)
from scrollintel.core.error_monitoring import (
    ErrorMonitor, ErrorMetrics, AlertManager, AlertRule, AlertLevel
)
from scrollintel.core.user_messages import (
    UserMessageGenerator, get_user_friendly_error
)
from scrollintel.core.interfaces import (
    AgentError, SecurityError, EngineError, DataError,
    ValidationError, ExternalServiceError
)


class TestErrorHandler:
    """Test the main error handler functionality."""
    
    @pytest.fixture
    def error_handler(self):
        return ErrorHandler()
    
    @pytest.fixture
    def error_context(self):
        return ErrorContext(
            error_id="test-error-123",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="test_agent",
            operation="test_operation",
            user_id="user123",
            session_id="session123"
        )
    
    def test_error_classification(self, error_handler):
        """Test error classification into categories."""
        # Test agent error
        agent_error = AgentError("Agent failed")
        category = error_handler._classify_error(agent_error)
        assert category == ErrorCategory.AGENT
        
        # Test security error
        security_error = SecurityError("Access denied")
        category = error_handler._classify_error(security_error)
        assert category == ErrorCategory.SECURITY
        
        # Test validation error
        validation_error = ValidationError("Invalid input")
        category = error_handler._classify_error(validation_error)
        assert category == ErrorCategory.VALIDATION
    
    def test_severity_determination(self, error_handler, error_context):
        """Test error severity determination."""
        # Critical error
        security_error = SecurityError("Security breach")
        severity = error_handler._determine_severity(security_error, error_context)
        assert severity == ErrorSeverity.CRITICAL
        
        # High severity error
        agent_error = AgentError("Agent crashed")
        severity = error_handler._determine_severity(agent_error, error_context)
        assert severity == ErrorSeverity.HIGH
        
        # Medium severity error
        data_error = DataError("Data format issue")
        severity = error_handler._determine_severity(data_error, error_context)
        assert severity == ErrorSeverity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_error_handling_flow(self, error_handler, error_context):
        """Test complete error handling flow."""
        test_error = AgentError("Test agent error")
        
        # Mock fallback handler
        async def mock_fallback(error, context):
            return {"message": "Fallback executed", "data": {"status": "fallback"}}
        
        error_handler.register_fallback("test_agent", mock_fallback)
        
        # Handle error
        result = await error_handler.handle_error(test_error, error_context)
        
        # Verify result
        assert result["success"] is True
        assert result["fallback_used"] is True
        assert "Fallback executed" in result["result"]["message"]
    
    def test_error_rate_tracking(self, error_handler, error_context):
        """Test error rate tracking functionality."""
        component = "test_component"
        
        # Record multiple errors
        for _ in range(5):
            error_handler._track_error_rate(component)
            time.sleep(0.1)
        
        # Check error rate
        error_rate = error_handler.get_error_rate(component)
        assert error_rate > 0
        
        # Check error statistics
        stats = error_handler.get_error_statistics()
        assert component in stats["components"]
        assert stats["components"][component]["error_count"] == 5


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker_config(self):
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0
        )
    
    @pytest.fixture
    def circuit_breaker(self, circuit_breaker_config):
        return CircuitBreaker(circuit_breaker_config)
    
    def test_circuit_breaker_states(self, circuit_breaker):
        """Test circuit breaker state transitions."""
        # Initially closed
        assert circuit_breaker.can_execute() is True
        
        # Record failures to open circuit
        for _ in range(3):
            circuit_breaker.record_failure()
        
        # Should be open now
        assert circuit_breaker.can_execute() is False
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should be half-open now
        assert circuit_breaker.can_execute() is True
        
        # Record success to close circuit
        for _ in range(3):
            circuit_breaker.record_success()
        
        # Should be closed again
        assert circuit_breaker.can_execute() is True
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        call_count = 0
        
        @with_circuit_breaker("test_service", CircuitBreakerConfig(failure_threshold=2))
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ExternalServiceError("Service failed")
            return "success"
        
        # First two calls should fail and open circuit
        with pytest.raises(ExternalServiceError):
            asyncio.run(test_function())
        
        with pytest.raises(ExternalServiceError):
            asyncio.run(test_function())
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(ExternalServiceError):
            asyncio.run(test_function())


class TestRetryMechanism:
    """Test retry mechanisms."""
    
    @pytest.fixture
    def retry_config(self):
        return RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            exponential_base=2.0
        )
    
    @pytest.mark.asyncio
    async def test_retry_decorator(self, retry_config):
        """Test retry decorator functionality."""
        call_count = 0
        
        @with_retry(retry_config)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert call_count == 3
    
    def test_delay_calculation(self):
        """Test retry delay calculation."""
        error_handler = ErrorHandler()
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        # Test exponential backoff
        delay1 = error_handler._calculate_delay(0, config)
        delay2 = error_handler._calculate_delay(1, config)
        delay3 = error_handler._calculate_delay(2, config)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0


class TestErrorMonitoring:
    """Test error monitoring and metrics."""
    
    @pytest.fixture
    def error_metrics(self):
        return ErrorMetrics(window_minutes=1)
    
    @pytest.fixture
    def error_context(self):
        return ErrorContext(
            error_id="test-error",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="test_component",
            operation="test_operation"
        )
    
    def test_error_recording(self, error_metrics, error_context):
        """Test error recording and metrics calculation."""
        # Record errors
        for _ in range(5):
            error_metrics.record_error(error_context)
        
        # Check metrics
        error_rate = error_metrics.get_error_rate("test_component")
        assert error_rate == 5.0  # 5 errors per minute
        
        breakdown = error_metrics.get_error_breakdown("test_component")
        assert breakdown["total_errors"] == 5
        assert breakdown["error_rate"] == 5.0
    
    def test_success_recording(self, error_metrics):
        """Test success recording and metrics."""
        component = "test_component"
        
        # Record successes
        for i in range(10):
            error_metrics.record_success(component, 0.5 + i * 0.1)
        
        # Check metrics
        avg_response_time = error_metrics.get_average_response_time(component)
        assert 0.9 < avg_response_time < 1.0  # Should be around 0.95
        
        success_rate = error_metrics.get_success_rate(component)
        assert success_rate == 1.0  # 100% success rate
    
    def test_component_health_tracking(self, error_metrics, error_context):
        """Test component health status tracking."""
        component = "test_component"
        
        # Initially should be healthy
        health = error_metrics.get_component_health(component)
        assert health in ["healthy", "unknown"]
        
        # Record many errors to make unhealthy
        for _ in range(15):
            error_metrics.record_error(error_context)
        
        health = error_metrics.get_component_health(component)
        assert health == "unhealthy"


class TestAlertManager:
    """Test alert management functionality."""
    
    @pytest.fixture
    def alert_manager(self):
        return AlertManager()
    
    @pytest.fixture
    def error_metrics(self):
        metrics = ErrorMetrics(window_minutes=1)
        # Add some test data
        context = ErrorContext(
            error_id="test",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="test_component",
            operation="test"
        )
        for _ in range(15):  # High error rate
            metrics.record_error(context)
        return metrics
    
    def test_alert_rule_evaluation(self, alert_manager, error_metrics):
        """Test alert rule evaluation."""
        # Check if high error rate rule triggers
        alert_manager.check_alerts(error_metrics)
        
        # Should have active alerts
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) > 0
        
        # Check alert details
        alert = active_alerts[0]
        assert alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]
        assert "test_component" in alert.component
    
    def test_custom_alert_rule(self, alert_manager, error_metrics):
        """Test custom alert rule creation."""
        custom_rule = AlertRule(
            name="custom_test_rule",
            condition="error_rate > 5",
            threshold=5.0,
            alert_level=AlertLevel.WARNING
        )
        
        alert_manager.add_rule(custom_rule)
        alert_manager.check_alerts(error_metrics)
        
        # Should trigger custom rule
        active_alerts = alert_manager.get_active_alerts()
        custom_alerts = [a for a in active_alerts if a.rule_name == "custom_test_rule"]
        assert len(custom_alerts) > 0
    
    def test_alert_resolution(self, alert_manager, error_metrics):
        """Test alert resolution."""
        # Trigger alerts
        alert_manager.check_alerts(error_metrics)
        active_alerts = alert_manager.get_active_alerts()
        
        if active_alerts:
            alert_id = active_alerts[0].id
            
            # Resolve alert
            alert_manager.resolve_alert(alert_id)
            
            # Check it's no longer active
            remaining_alerts = alert_manager.get_active_alerts()
            assert not any(a.id == alert_id for a in remaining_alerts)
            
            # Check it's in history
            history = alert_manager.get_alert_history()
            assert any(a.id == alert_id and a.resolved for a in history)


class TestUserMessages:
    """Test user-friendly message generation."""
    
    @pytest.fixture
    def message_generator(self):
        return UserMessageGenerator()
    
    def test_error_message_generation(self, message_generator):
        """Test error message generation."""
        message = message_generator.generate_user_message(
            error_category=ErrorCategory.AGENT,
            error_severity=ErrorSeverity.HIGH,
            component="scroll_cto",
            operation="generate_architecture"
        )
        
        assert message["type"] == "error"
        assert "AI CTO" in message["message"]
        assert len(message["recovery_actions"]) > 0
        assert message["severity"] == "high"
    
    def test_success_message_generation(self, message_generator):
        """Test success message generation."""
        message = message_generator.generate_success_message(
            operation="data_analysis",
            component="scroll_data_scientist",
            fallback_used=True
        )
        
        assert message["type"] == "success"
        assert message["fallback_used"] is True
        assert "backup systems" in message["message"]
    
    def test_maintenance_message_generation(self, message_generator):
        """Test maintenance message generation."""
        message = message_generator.generate_maintenance_message(
            component="scroll_ml_engineer",
            estimated_duration="30 minutes"
        )
        
        assert message["type"] == "info"
        assert "maintenance" in message["message"]
        assert "30 minutes" in message["message"]


class TestErrorMiddleware:
    """Test error handling middleware."""
    
    @pytest.fixture
    def middleware(self):
        return ErrorHandlingMiddleware(None, enable_detailed_errors=True)
    
    def test_error_context_creation(self, middleware):
        """Test error context creation from request."""
        # Mock request
        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/agents/scroll-cto"
        mock_request.query_params = {}
        mock_request.headers = {"User-Agent": "test-agent"}
        mock_request.client.host = "127.0.0.1"
        mock_request.state = Mock()
        
        error = AgentError("Test error")
        context = middleware._create_error_context(mock_request, error, "req-123")
        
        assert context.component == "api"
        assert context.operation == "POST /api/agents/scroll-cto"
        assert context.request_id == "req-123"
        assert context.metadata["method"] == "POST"
    
    def test_http_response_creation(self, middleware):
        """Test HTTP response creation from error response."""
        error_response = {
            "success": False,
            "error": {
                "type": "agent_unavailable",
                "message": "Agent is down"
            }
        }
        
        context = ErrorContext(
            error_id="test",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="test_agent",
            operation="test"
        )
        
        response = middleware._create_http_response(error_response, context)
        
        assert response.status_code == 422  # Unprocessable Entity for agent errors
        content = response.body.decode()
        assert "agent_unavailable" in content


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_error_handling_flow(self):
        """Test complete error handling from error to resolution."""
        # Initialize components
        error_handler = ErrorHandler()
        error_monitor = ErrorMonitor()
        
        # Start monitoring
        await error_monitor.start_monitoring()
        
        try:
            # Simulate error scenario
            context = ErrorContext(
                error_id="integration-test",
                timestamp=time.time(),
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.AGENT,
                component="test_agent",
                operation="test_operation"
            )
            
            # Record error
            error_monitor.record_error(context)
            
            # Check metrics
            metrics = error_monitor.get_metrics()
            assert "test_agent" in metrics["components"]
            
            # Wait for alert check
            await asyncio.sleep(1.1)
            
            # Check for alerts
            alerts = error_monitor.get_active_alerts()
            # May or may not have alerts depending on thresholds
            
        finally:
            await error_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_fallback_and_recovery_scenario(self):
        """Test fallback and recovery scenario."""
        error_handler = ErrorHandler()
        
        # Register fallback
        async def test_fallback(error, context):
            return {"message": "Fallback successful", "data": {"status": "ok"}}
        
        error_handler.register_fallback("test_service", test_fallback)
        
        # Create error context
        context = ErrorContext(
            error_id="fallback-test",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="test_service",
            operation="test_operation"
        )
        
        # Handle error
        result = await error_handler.handle_error(AgentError("Service failed"), context)
        
        # Verify fallback was used
        assert result["success"] is True
        assert result["fallback_used"] is True
        assert result["result"]["message"] == "Fallback successful"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])