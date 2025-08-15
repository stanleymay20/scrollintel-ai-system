"""
Integration tests for comprehensive error handling and recovery system.
Tests error scenarios, recovery mechanisms, and user-friendly responses.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from scrollintel.core.error_handling import (
    ErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory,
    RetryConfig, CircuitBreakerConfig, CircuitBreaker,
    with_error_handling, with_retry, with_circuit_breaker
)
from scrollintel.core.error_middleware import (
    ErrorHandlingMiddleware, ExternalServiceErrorHandler,
    handle_agent_error, handle_validation_error, handle_external_service_error
)
from scrollintel.core.interfaces import (
    AgentError, SecurityError, EngineError, DataError,
    ValidationError, ExternalServiceError
)
from scrollintel.agents.error_handling import (
    AgentErrorHandler, with_agent_error_handling,
    AIServiceFallbackManager, AgentRecoveryStrategies
)


class TestErrorHandler:
    """Test the core error handler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        self.context = ErrorContext(
            error_id="test-error-123",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AGENT,
            component="test_agent",
            operation="test_operation"
        )
    
    @pytest.mark.asyncio
    async def test_error_classification(self):
        """Test error classification into categories."""
        # Test agent error classification
        agent_error = AgentError("Agent failed")
        category = self.error_handler._classify_error(agent_error)
        assert category == ErrorCategory.AGENT
        
        # Test security error classification
        security_error = SecurityError("Access denied")
        category = self.error_handler._classify_error(security_error)
        assert category == ErrorCategory.SECURITY
        
        # Test external service error classification
        external_error = ExternalServiceError("API failed")
        category = self.error_handler._classify_error(external_error)
        assert category == ErrorCategory.EXTERNAL_SERVICE
    
    @pytest.mark.asyncio
    async def test_severity_determination(self):
        """Test error severity determination."""
        # Test critical severity for security errors
        security_error = SecurityError("Access denied")
        severity = self.error_handler._determine_severity(security_error, self.context)
        assert severity == ErrorSeverity.CRITICAL
        
        # Test high severity for agent errors
        agent_error = AgentError("Agent failed")
        severity = self.error_handler._determine_severity(agent_error, self.context)
        assert severity == ErrorSeverity.HIGH
        
        # Test medium severity for data errors
        data_error = DataError("Invalid data")
        severity = self.error_handler._determine_severity(data_error, self.context)
        assert severity == ErrorSeverity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_recovery_strategy_determination(self):
        """Test recovery strategy determination."""
        # Test fail fast for security errors
        self.context.category = ErrorCategory.SECURITY
        strategy = self.error_handler._determine_recovery_strategy(None, self.context)
        assert strategy.value == "fail_fast"
        
        # Test circuit breaker for external services
        self.context.category = ErrorCategory.EXTERNAL_SERVICE
        strategy = self.error_handler._determine_recovery_strategy(None, self.context)
        assert strategy.value == "circuit_breaker"
        
        # Test fallback for agents
        self.context.category = ErrorCategory.AGENT
        strategy = self.error_handler._determine_recovery_strategy(None, self.context)
        assert strategy.value == "fallback"
    
    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        """Test fallback handler execution."""
        # Register a test fallback handler
        async def test_fallback(error, context):
            return {"message": "Fallback executed", "data": {"status": "ok"}}
        
        self.error_handler.register_fallback("test_agent", test_fallback)
        
        # Test fallback execution
        result = await self.error_handler._execute_fallback(
            Exception("Test error"), self.context
        )
        
        assert result["success"] is True
        assert result["fallback_used"] is True
        assert result["result"]["message"] == "Fallback executed"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation."""
        # Register a test degradation handler
        async def test_degradation(error, context):
            return {"message": "Degraded mode", "features": ["basic"]}
        
        self.error_handler.register_degradation("test_agent", test_degradation)
        
        # Test degradation execution
        result = await self.error_handler._graceful_degradation(
            Exception("Test error"), self.context
        )
        
        assert result["success"] is True
        assert result["degraded"] is True
        assert result["result"]["message"] == "Degraded mode"


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0
        )
        self.circuit_breaker = CircuitBreaker(self.config)
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        assert self.circuit_breaker.can_execute() is True
        assert self.circuit_breaker.state.value == "closed"
    
    def test_circuit_breaker_failure_recording(self):
        """Test failure recording and state transitions."""
        # Record failures up to threshold
        for i in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()
        
        # Should be open now
        assert self.circuit_breaker.state.value == "open"
        assert self.circuit_breaker.can_execute() is False
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        # Force circuit breaker to open state
        for i in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()
        
        assert self.circuit_breaker.can_execute() is False
        
        # Wait for recovery timeout
        time.sleep(self.config.recovery_timeout + 0.1)
        
        # Should allow execution in half-open state
        assert self.circuit_breaker.can_execute() is True
        assert self.circuit_breaker.state.value == "half_open"
    
    def test_circuit_breaker_success_recovery(self):
        """Test circuit breaker recovery with successful calls."""
        # Force to half-open state
        for i in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()
        time.sleep(self.config.recovery_timeout + 0.1)
        self.circuit_breaker.can_execute()  # Transition to half-open
        
        # Record successful calls
        for i in range(3):  # Need 3 successes to close
            self.circuit_breaker.record_success()
        
        assert self.circuit_breaker.state.value == "closed"
        assert self.circuit_breaker.failure_count == 0


class TestRetryMechanism:
    """Test retry mechanisms with exponential backoff."""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.01))
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await flaky_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_decorator_failure(self):
        """Test retry decorator with persistent failure."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.01))
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            await failing_function()
        
        assert call_count == 3
    
    def test_delay_calculation(self):
        """Test delay calculation for different backoff strategies."""
        error_handler = ErrorHandler()
        
        # Test exponential backoff
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        delay1 = error_handler._calculate_delay(0, config)
        delay2 = error_handler._calculate_delay(1, config)
        delay3 = error_handler._calculate_delay(2, config)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0
        
        # Test linear backoff
        config = RetryConfig(base_delay=1.0, backoff_strategy="linear", jitter=False)
        delay1 = error_handler._calculate_delay(0, config)
        delay2 = error_handler._calculate_delay(1, config)
        delay3 = error_handler._calculate_delay(2, config)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 3.0


class TestErrorMiddleware:
    """Test error handling middleware."""
    
    def setup_method(self):
        """Set up test FastAPI app with error middleware."""
        self.app = FastAPI()
        self.app.add_middleware(ErrorHandlingMiddleware, enable_detailed_errors=True)
        
        @self.app.get("/test-agent-error")
        async def test_agent_error():
            raise AgentError("Test agent error")
        
        @self.app.get("/test-security-error")
        async def test_security_error():
            raise SecurityError("Test security error")
        
        @self.app.get("/test-validation-error")
        async def test_validation_error():
            raise ValidationError("Test validation error")
        
        @self.app.get("/test-external-service-error")
        async def test_external_service_error():
            raise ExternalServiceError("Test external service error")
        
        @self.app.get("/test-success")
        async def test_success():
            return {"message": "success"}
        
        self.client = TestClient(self.app)
    
    def test_agent_error_handling(self):
        """Test agent error handling through middleware."""
        response = self.client.get("/test-agent-error")
        
        assert response.status_code == 422  # Unprocessable Entity for agent errors
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["category"] == "agent"
        assert "error_id" in data
        assert "timestamp" in data
    
    def test_security_error_handling(self):
        """Test security error handling through middleware."""
        response = self.client.get("/test-security-error")
        
        assert response.status_code == 401  # Unauthorized for security errors
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["category"] == "security"
    
    def test_validation_error_handling(self):
        """Test validation error handling through middleware."""
        response = self.client.get("/test-validation-error")
        
        assert response.status_code == 400  # Bad Request for validation errors
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["category"] == "validation"
    
    def test_external_service_error_handling(self):
        """Test external service error handling through middleware."""
        response = self.client.get("/test-external-service-error")
        
        assert response.status_code == 503  # Service Unavailable for external service errors
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["category"] == "external_service"
    
    def test_successful_request(self):
        """Test successful request handling."""
        response = self.client.get("/test-success")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "success"


class TestAgentErrorHandling:
    """Test agent-specific error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent_error_handler = AgentErrorHandler("test_agent")
    
    @pytest.mark.asyncio
    async def test_agent_unavailable_with_fallback(self):
        """Test agent unavailable scenario with fallback response."""
        # Register fallback response
        fallback_response = {"message": "Cached response", "data": {"status": "ok"}}
        self.agent_error_handler.register_fallback_response("test_operation", fallback_response)
        
        result = await self.agent_error_handler.handle_agent_unavailable("test_operation")
        
        assert result["success"] is True
        assert result["fallback_used"] is True
        assert result["data"] == fallback_response
    
    @pytest.mark.asyncio
    async def test_agent_unavailable_with_degraded_capabilities(self):
        """Test agent unavailable scenario with degraded capabilities."""
        # Register degraded capabilities
        capabilities = ["basic_chat", "simple_queries"]
        self.agent_error_handler.register_degraded_capabilities("test_operation", capabilities)
        
        result = await self.agent_error_handler.handle_agent_unavailable("test_operation")
        
        assert result["success"] is True
        assert result["degraded"] is True
        assert result["data"]["available_capabilities"] == capabilities
    
    @pytest.mark.asyncio
    async def test_agent_unavailable_no_fallback(self):
        """Test agent unavailable scenario without fallback."""
        result = await self.agent_error_handler.handle_agent_unavailable("unknown_operation")
        
        assert result["success"] is False
        assert result["error"]["type"] == "agent_unavailable"
    
    @pytest.mark.asyncio
    async def test_ai_service_error_handling(self):
        """Test AI service error handling."""
        error = ExternalServiceError("OpenAI API failed")
        result = await self.agent_error_handler.handle_ai_service_error(
            "openai", error, "text_generation"
        )
        
        # The handler should try alternative services and succeed
        assert result["success"] is True
        assert result["fallback_used"] is True
        assert result["alternative_service"] == "anthropic"
    
    @pytest.mark.asyncio
    async def test_agent_error_handling_decorator(self):
        """Test agent error handling decorator."""
        class TestAgent:
            @with_agent_error_handling("test_agent", "test_operation")
            async def test_method(self):
                raise AgentError("Test agent error")
        
        agent = TestAgent()
        result = await agent.test_method()
        
        assert result["success"] is False
        assert result["error"]["type"] == "agent_unavailable"


class TestAIServiceFallbackManager:
    """Test AI service fallback management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fallback_manager = AIServiceFallbackManager()
    
    @pytest.mark.asyncio
    async def test_get_fallback_service(self):
        """Test getting fallback service."""
        # Test OpenAI fallback to Anthropic for text generation
        fallback = await self.fallback_manager.get_fallback_service("openai", "text_generation")
        assert fallback == "anthropic"
        
        # Test Anthropic fallback to HuggingFace for text generation
        fallback = await self.fallback_manager.get_fallback_service("anthropic", "text_generation")
        assert fallback == "huggingface"
        
        # Test Pinecone fallback to Supabase for vector search
        fallback = await self.fallback_manager.get_fallback_service("pinecone", "vector_search")
        assert fallback == "supabase"
    
    @pytest.mark.asyncio
    async def test_service_availability_check(self):
        """Test service availability checking."""
        # Test service availability (should return True by default)
        available = await self.fallback_manager.is_service_available("openai")
        assert available is True
        
        # Test with circuit breaker in open state
        from scrollintel.core.error_handling import error_handler
        circuit_breaker = error_handler.get_circuit_breaker("test_service")
        
        # Force circuit breaker to open state
        for i in range(5):
            circuit_breaker.record_failure()
        
        available = await self.fallback_manager.is_service_available("test_service")
        assert available is False


class TestAgentRecoveryStrategies:
    """Test agent-specific recovery strategies."""
    
    @pytest.mark.asyncio
    async def test_cto_agent_fallback(self):
        """Test CTO agent fallback strategy."""
        context = ErrorContext(
            error_id="test-error",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="scroll_cto",
            operation="architecture_recommendation"
        )
        
        result = await AgentRecoveryStrategies.cto_agent_fallback(
            Exception("Test error"), context
        )
        
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0
        assert result["confidence"] == "medium"
    
    @pytest.mark.asyncio
    async def test_data_scientist_fallback(self):
        """Test data scientist agent fallback strategy."""
        context = ErrorContext(
            error_id="test-error",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="scroll_data_scientist",
            operation="data_analysis"
        )
        
        result = await AgentRecoveryStrategies.data_scientist_fallback(
            Exception("Test error"), context
        )
        
        assert "analysis" in result
        assert "recommendations" in result["analysis"]
        assert result["confidence"] == "low"
    
    @pytest.mark.asyncio
    async def test_ml_engineer_fallback(self):
        """Test ML engineer agent fallback strategy."""
        context = ErrorContext(
            error_id="test-error",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="scroll_ml_engineer",
            operation="model_training"
        )
        
        result = await AgentRecoveryStrategies.ml_engineer_fallback(
            Exception("Test error"), context
        )
        
        assert "pipeline" in result
        assert "steps" in result["pipeline"]
        assert "recommended_models" in result["pipeline"]
        assert result["confidence"] == "medium"
    
    @pytest.mark.asyncio
    async def test_ai_engineer_fallback(self):
        """Test AI engineer agent fallback strategy."""
        context = ErrorContext(
            error_id="test-error",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            component="scroll_ai_engineer",
            operation="ai_processing"
        )
        
        result = await AgentRecoveryStrategies.ai_engineer_fallback(
            Exception("Test error"), context
        )
        
        assert "available_features" in result
        assert "unavailable_features" in result
        assert len(result["available_features"]) > 0
        assert len(result["unavailable_features"]) > 0


class TestEndToEndErrorScenarios:
    """Test end-to-end error scenarios and recovery."""
    
    def setup_method(self):
        """Set up test FastAPI app with full error handling."""
        self.app = FastAPI()
        self.app.add_middleware(ErrorHandlingMiddleware, enable_detailed_errors=True)
        
        @self.app.get("/agent/{agent_name}/execute")
        async def execute_agent(agent_name: str):
            if agent_name == "failing_agent":
                raise AgentError("Agent is temporarily unavailable")
            elif agent_name == "external_service_agent":
                raise ExternalServiceError("External API failed")
            elif agent_name == "security_agent":
                raise SecurityError("Access denied")
            else:
                return {"message": f"Agent {agent_name} executed successfully"}
        
        self.client = TestClient(self.app)
    
    def test_agent_failure_recovery(self):
        """Test agent failure with recovery."""
        response = self.client.get("/agent/failing_agent/execute")
        
        # Should return error but with recovery information
        assert response.status_code == 422
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert "recovery_actions" in data["error"]
        assert data["error"]["category"] == "agent"
    
    def test_external_service_failure_recovery(self):
        """Test external service failure with recovery."""
        response = self.client.get("/agent/external_service_agent/execute")
        
        # Should return service unavailable with recovery information
        assert response.status_code == 503
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["category"] == "external_service"
        assert "recovery_actions" in data["error"]
    
    def test_security_failure_fast_fail(self):
        """Test security failure with fast fail."""
        response = self.client.get("/agent/security_agent/execute")
        
        # Should return unauthorized with no recovery
        assert response.status_code == 401
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["category"] == "security"
    
    def test_successful_agent_execution(self):
        """Test successful agent execution."""
        response = self.client.get("/agent/working_agent/execute")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Agent working_agent executed successfully"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])