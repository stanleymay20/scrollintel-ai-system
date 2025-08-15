"""
Demo script for comprehensive error handling and recovery system.
Shows different error scenarios and recovery mechanisms.
"""

import asyncio
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scrollintel.core.error_handling import (
    ErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory,
    with_error_handling, with_retry, with_circuit_breaker,
    RetryConfig, CircuitBreakerConfig
)
from scrollintel.core.error_middleware import ErrorHandlingMiddleware
from scrollintel.core.interfaces import (
    AgentError, SecurityError, EngineError, DataError,
    ValidationError, ExternalServiceError
)
from scrollintel.agents.error_handling import (
    AgentErrorHandler, with_agent_error_handling
)


async def demo_basic_error_handling():
    """Demo basic error handling functionality."""
    print("=== Demo: Basic Error Handling ===")
    
    error_handler = ErrorHandler()
    
    # Test different error types
    errors_to_test = [
        (AgentError("Agent is busy"), "Agent Error"),
        (SecurityError("Access denied"), "Security Error"),
        (ExternalServiceError("API timeout"), "External Service Error"),
        (ValidationError("Invalid input"), "Validation Error"),
        (DataError("Corrupted data"), "Data Error")
    ]
    
    for error, error_name in errors_to_test:
        print(f"\n--- Testing {error_name} ---")
        
        context = ErrorContext(
            error_id=f"demo-{int(time.time())}",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.UNKNOWN,
            component="demo_component",
            operation="demo_operation"
        )
        
        result = await error_handler.handle_error(error, context)
        
        print(f"Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"Recovery used: {result.get('fallback_used', False) or result.get('degraded', False)}")
        else:
            error_info = result.get('error', {})
            print(f"Error type: {error_info.get('type', 'unknown')}")
            print(f"User message: {error_info.get('user_message', 'No message')}")


async def demo_retry_mechanism():
    """Demo retry mechanism with exponential backoff."""
    print("\n=== Demo: Retry Mechanism ===")
    
    attempt_count = 0
    
    @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
    async def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        print(f"Attempt {attempt_count}")
        
        if attempt_count < 3:
            raise ExternalServiceError("Temporary failure")
        
        return "Success after retries!"
    
    try:
        result = await flaky_operation()
        print(f"Result: {result}")
        print(f"Total attempts: {attempt_count}")
    except Exception as e:
        print(f"Failed after all retries: {e}")


async def demo_circuit_breaker():
    """Demo circuit breaker pattern."""
    print("\n=== Demo: Circuit Breaker ===")
    
    @with_circuit_breaker("demo_service", CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0))
    async def unreliable_service():
        # Simulate service failure
        raise ExternalServiceError("Service is down")
    
    # Test circuit breaker behavior
    for i in range(5):
        try:
            await unreliable_service()
        except ExternalServiceError as e:
            if "Circuit breaker open" in str(e):
                print(f"Call {i+1}: Circuit breaker is open - {e}")
            else:
                print(f"Call {i+1}: Service failed - {e}")
        
        # Small delay between calls
        await asyncio.sleep(0.1)
    
    # Wait for recovery timeout
    print("Waiting for circuit breaker recovery...")
    await asyncio.sleep(1.1)
    
    try:
        await unreliable_service()
    except ExternalServiceError as e:
        print(f"After recovery timeout: {e}")


async def demo_agent_error_handling():
    """Demo agent-specific error handling."""
    print("\n=== Demo: Agent Error Handling ===")
    
    class DemoAgent:
        def __init__(self):
            self.error_handler = AgentErrorHandler("demo_agent")
            
            # Register fallback responses
            self.error_handler.register_fallback_response(
                "chat", 
                {"message": "I'm using a cached response while my main systems are offline."}
            )
            
            # Register degraded capabilities
            self.error_handler.register_degraded_capabilities(
                "analysis",
                ["basic_stats", "simple_charts"]
            )
        
        @with_agent_error_handling("demo_agent", "chat")
        async def chat(self, message: str):
            # Simulate agent failure
            raise AgentError("Agent is temporarily unavailable")
        
        @with_agent_error_handling("demo_agent", "analysis")
        async def analyze_data(self, data):
            # Simulate agent failure
            raise AgentError("Analysis engine is down")
    
    agent = DemoAgent()
    
    # Test chat with fallback
    print("--- Testing chat with fallback ---")
    result = await agent.chat("Hello")
    print(f"Success: {result.get('success', False)}")
    if result.get('success'):
        print(f"Fallback used: {result.get('fallback_used', False)}")
    
    # Test analysis with degraded capabilities
    print("\n--- Testing analysis with degraded capabilities ---")
    result = await agent.analyze_data({"values": [1, 2, 3]})
    print(f"Success: {result.get('success', False)}")
    if result.get('success'):
        print(f"Degraded: {result.get('degraded', False)}")
        if result.get('degraded'):
            data = result.get('data', {})
            print(f"Available capabilities: {data.get('available_capabilities', [])}")


def demo_fastapi_error_middleware():
    """Demo FastAPI error middleware."""
    print("\n=== Demo: FastAPI Error Middleware ===")
    
    # Create FastAPI app with error middleware
    app = FastAPI()
    app.add_middleware(ErrorHandlingMiddleware, enable_detailed_errors=True)
    
    @app.get("/test-agent-error")
    async def test_agent_error():
        raise AgentError("Demo agent is unavailable")
    
    @app.get("/test-security-error")
    async def test_security_error():
        raise SecurityError("Demo access denied")
    
    @app.get("/test-validation-error")
    async def test_validation_error():
        raise ValidationError("Demo invalid input")
    
    @app.get("/test-success")
    async def test_success():
        return {"message": "Demo success"}
    
    # Test the endpoints
    client = TestClient(app)
    
    endpoints_to_test = [
        ("/test-agent-error", "Agent Error"),
        ("/test-security-error", "Security Error"),
        ("/test-validation-error", "Validation Error"),
        ("/test-success", "Success")
    ]
    
    for endpoint, description in endpoints_to_test:
        print(f"\n--- Testing {description} ---")
        response = client.get(endpoint)
        print(f"Status Code: {response.status_code}")
        
        data = response.json()
        print(f"Success: {data.get('success', False)}")
        
        if not data.get('success', False):
            error = data.get('error', {})
            print(f"Error Category: {error.get('category', 'unknown')}")
            print(f"User Message: {error.get('message', 'No message')}")
            recovery_actions = error.get('recovery_actions', [])
            if recovery_actions:
                print(f"Recovery Actions: {recovery_actions[:2]}")  # Show first 2 actions
        else:
            print(f"Message: {data.get('message', 'No message')}")


async def main():
    """Run all error handling demos."""
    print("ScrollIntel™ Error Handling and Recovery System Demo")
    print("=" * 60)
    
    try:
        await demo_basic_error_handling()
        await demo_retry_mechanism()
        await demo_circuit_breaker()
        await demo_agent_error_handling()
        demo_fastapi_error_middleware()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Comprehensive error classification and handling")
        print("✓ Retry mechanisms with exponential backoff")
        print("✓ Circuit breaker pattern for external services")
        print("✓ Agent-specific error handling with fallbacks")
        print("✓ User-friendly error messages with recovery guidance")
        print("✓ FastAPI middleware integration")
        print("✓ Graceful degradation and fallback strategies")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())