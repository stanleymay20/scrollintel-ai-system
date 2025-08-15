"""
Error handling middleware for FastAPI routes.
Provides comprehensive error handling, recovery, and user-friendly responses.
"""

import time
import uuid
import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .error_handling import (
    error_handler, ErrorContext, ErrorSeverity, ErrorCategory,
    with_error_handling, RetryConfig, CircuitBreakerConfig
)
from .interfaces import (
    AgentError, SecurityError, EngineError, DataError,
    ValidationError, ExternalServiceError, ConfigurationError
)
from .user_messages import user_message_generator


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling across all API endpoints."""
    
    def __init__(self, app, enable_detailed_errors: bool = False):
        super().__init__(app)
        self.enable_detailed_errors = enable_detailed_errors
        self.logger = logging.getLogger(__name__)
        
        # Register fallback handlers
        self._register_fallback_handlers()
        
        # Register degradation handlers
        self._register_degradation_handlers()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive error handling."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            
            # Log successful request
            processing_time = time.time() - start_time
            self.logger.info(
                f"Request {request_id} completed successfully in {processing_time:.3f}s - "
                f"{request.method} {request.url.path}"
            )
            
            return response
            
        except Exception as error:
            # Create error context
            context = self._create_error_context(request, error, request_id)
            
            # Handle error with recovery
            error_response = await error_handler.handle_error(error, context)
            
            # Convert to HTTP response
            return self._create_http_response(error_response, context)
    
    def _create_error_context(self, request: Request, error: Exception, request_id: str) -> ErrorContext:
        """Create error context from request and error."""
        # Extract user information from request state if available
        security_context = getattr(request.state, 'security_context', None)
        user_id = security_context.user_id if security_context else None
        session_id = security_context.session_id if security_context else None
        
        # Determine component from URL path
        component = self._extract_component_from_path(request.url.path)
        
        return ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=time.time(),
            severity=ErrorSeverity.LOW,  # Will be updated by error handler
            category=ErrorCategory.UNKNOWN,  # Will be updated by error handler
            component=component,
            operation=f"{request.method} {request.url.path}",
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            metadata={
                "method": request.method,
                "path": str(request.url.path),
                "query_params": dict(request.query_params),
                "user_agent": request.headers.get("User-Agent", ""),
                "client_ip": self._get_client_ip(request)
            }
        )
    
    def _extract_component_from_path(self, path: str) -> str:
        """Extract component name from URL path."""
        path_parts = path.strip('/').split('/')
        if len(path_parts) > 0:
            return path_parts[0]
        return "unknown"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    def _create_http_response(self, error_response: Dict[str, Any], context: ErrorContext) -> JSONResponse:
        """Create HTTP response from error response."""
        if error_response.get("success", False):
            # Successful recovery
            status_code = status.HTTP_200_OK
            if error_response.get("fallback_used") or error_response.get("degraded"):
                status_code = status.HTTP_206_PARTIAL_CONTENT
        else:
            # Failed recovery
            error_info = error_response.get("error", {})
            error_type = error_info.get("type", "unknown_error")
            
            # Map error categories to HTTP status codes
            if context.category == ErrorCategory.SECURITY:
                status_code = status.HTTP_401_UNAUTHORIZED
            elif context.category == ErrorCategory.VALIDATION:
                status_code = status.HTTP_400_BAD_REQUEST
            elif context.category == ErrorCategory.DATA:
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
            elif context.category == ErrorCategory.AGENT:
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
            elif context.category == ErrorCategory.EXTERNAL_SERVICE:
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            elif error_type == "critical_error":
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            elif error_type == "service_unavailable":
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        # Prepare response content
        response_content = {
            "success": error_response.get("success", False),
            "timestamp": context.timestamp,
            "request_id": context.request_id,
            "error_id": context.error_id
        }
        
        if error_response.get("success", False):
            # Successful recovery response - generate user-friendly success message
            success_message = user_message_generator.generate_success_message(
                operation=context.operation,
                component=context.component,
                fallback_used=error_response.get("fallback_used", False),
                degraded=error_response.get("degraded", False),
                context=context.metadata
            )
            
            response_content.update({
                "message": success_message["message"],
                "title": success_message["title"],
                "result": error_response.get("result"),
                "fallback_used": error_response.get("fallback_used", False),
                "degraded": error_response.get("degraded", False),
                "user_message": success_message
            })
        else:
            # Error response - generate user-friendly error message
            user_message = user_message_generator.generate_user_message(
                error_category=context.category,
                error_severity=context.severity,
                component=context.component,
                operation=context.operation,
                context=context.metadata
            )
            
            error_info = error_response.get("error", {})
            response_content.update({
                "error": {
                    "type": error_info.get("type", "unknown_error"),
                    "title": user_message["title"],
                    "message": user_message["message"],
                    "recovery_actions": user_message["recovery_actions"],
                    "technical_explanation": user_message.get("technical_explanation") if self.enable_detailed_errors else None,
                    "severity": context.severity.value,
                    "category": context.category.value
                },
                "user_message": user_message
            })
            
            # Add detailed error information in development
            if self.enable_detailed_errors:
                response_content["error"]["detailed_message"] = error_info.get("message")
                response_content["error"]["component"] = context.component
                response_content["error"]["operation"] = context.operation
        
        return JSONResponse(
            status_code=status_code,
            content=response_content
        )
    
    def _register_fallback_handlers(self):
        """Register fallback handlers for different components."""
        
        async def agent_fallback(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            """Fallback handler for agent errors."""
            return {
                "message": "Using cached response or simplified processing",
                "data": {"status": "fallback_active", "component": context.component}
            }
        
        async def engine_fallback(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            """Fallback handler for engine errors."""
            return {
                "message": "Using backup processing engine",
                "data": {"status": "backup_engine_active", "component": context.component}
            }
        
        async def external_service_fallback(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            """Fallback handler for external service errors."""
            return {
                "message": "Using cached data or alternative service",
                "data": {"status": "alternative_service_active", "component": context.component}
            }
        
        # Register fallback handlers
        error_handler.register_fallback("agents", agent_fallback)
        error_handler.register_fallback("automodel", engine_fallback)
        error_handler.register_fallback("scroll-qa", engine_fallback)
        error_handler.register_fallback("scroll-viz", engine_fallback)
        error_handler.register_fallback("scroll-forecast", engine_fallback)
        error_handler.register_fallback("files", external_service_fallback)
        error_handler.register_fallback("vault", external_service_fallback)
    
    def _register_degradation_handlers(self):
        """Register degradation handlers for graceful service degradation."""
        
        async def agent_degradation(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            """Degradation handler for agent services."""
            return {
                "message": "Agent service running with limited capabilities",
                "data": {
                    "status": "degraded",
                    "available_features": ["basic_chat", "simple_queries"],
                    "unavailable_features": ["advanced_analysis", "model_training"]
                }
            }
        
        async def ml_degradation(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            """Degradation handler for ML services."""
            return {
                "message": "ML service running with basic models only",
                "data": {
                    "status": "degraded",
                    "available_models": ["linear_regression", "decision_tree"],
                    "unavailable_models": ["neural_networks", "ensemble_methods"]
                }
            }
        
        async def viz_degradation(error: Exception, context: ErrorContext) -> Dict[str, Any]:
            """Degradation handler for visualization services."""
            return {
                "message": "Visualization service using basic charts only",
                "data": {
                    "status": "degraded",
                    "available_charts": ["bar", "line", "pie"],
                    "unavailable_charts": ["3d", "interactive", "animated"]
                }
            }
        
        # Register degradation handlers
        error_handler.register_degradation("agents", agent_degradation)
        error_handler.register_degradation("automodel", ml_degradation)
        error_handler.register_degradation("scroll-viz", viz_degradation)


class ExternalServiceErrorHandler:
    """Specialized error handler for external service integrations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configure circuit breakers for external services
        self.openai_circuit_breaker = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0
        )
        
        self.anthropic_circuit_breaker = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0
        )
        
        self.pinecone_circuit_breaker = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0
        )
    
    @with_error_handling(
        component="openai",
        operation="api_call",
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
    )
    async def call_openai_api(self, *args, **kwargs):
        """Call OpenAI API with error handling."""
        # This would be implemented by the actual OpenAI integration
        pass
    
    @with_error_handling(
        component="anthropic",
        operation="api_call",
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
    )
    async def call_anthropic_api(self, *args, **kwargs):
        """Call Anthropic API with error handling."""
        # This would be implemented by the actual Anthropic integration
        pass
    
    @with_error_handling(
        component="pinecone",
        operation="vector_operation",
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)
    )
    async def call_pinecone_api(self, *args, **kwargs):
        """Call Pinecone API with error handling."""
        # This would be implemented by the actual Pinecone integration
        pass


# Global external service error handler
external_service_handler = ExternalServiceErrorHandler()


def create_error_handling_middleware(enable_detailed_errors: bool = False) -> ErrorHandlingMiddleware:
    """Create error handling middleware with configuration."""
    return ErrorHandlingMiddleware(None, enable_detailed_errors)


# Utility functions for route-level error handling
def handle_agent_error(error: Exception, agent_name: str, operation: str) -> JSONResponse:
    """Handle agent-specific errors."""
    context = ErrorContext(
        error_id=str(uuid.uuid4()),
        timestamp=time.time(),
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.AGENT,
        component=agent_name,
        operation=operation
    )
    
    user_message = f"The {agent_name} agent is temporarily unavailable. Please try again in a few moments."
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "success": False,
            "error": {
                "type": "agent_unavailable",
                "message": user_message,
                "recovery_action": "Try again in a few moments or use a different agent",
                "error_id": context.error_id
            },
            "timestamp": context.timestamp
        }
    )


def handle_validation_error(error: Exception, field_name: str = None) -> JSONResponse:
    """Handle validation errors with user-friendly messages."""
    context = ErrorContext(
        error_id=str(uuid.uuid4()),
        timestamp=time.time(),
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.VALIDATION,
        component="validation",
        operation="input_validation"
    )
    
    if field_name:
        user_message = f"Invalid value for {field_name}. Please check your input and try again."
    else:
        user_message = "Please check your input and try again."
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": {
                "type": "validation_error",
                "message": user_message,
                "recovery_action": "Correct the input and resubmit",
                "error_id": context.error_id,
                "field": field_name
            },
            "timestamp": context.timestamp
        }
    )


def handle_external_service_error(error: Exception, service_name: str) -> JSONResponse:
    """Handle external service errors."""
    context = ErrorContext(
        error_id=str(uuid.uuid4()),
        timestamp=time.time(),
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.EXTERNAL_SERVICE,
        component=service_name,
        operation="external_api_call"
    )
    
    user_message = f"The {service_name} service is temporarily unavailable. We're working to restore full functionality."
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "success": False,
            "error": {
                "type": "service_unavailable",
                "message": user_message,
                "recovery_action": "Please try again later",
                "error_id": context.error_id,
                "service": service_name
            },
            "timestamp": context.timestamp
        }
    )