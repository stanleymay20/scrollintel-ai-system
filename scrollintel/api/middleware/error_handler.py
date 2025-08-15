"""
Error Handling Middleware for Visual Generation API

Provides comprehensive error handling with:
- Standardized error responses
- Error logging and tracking
- Rate limit error handling
- Authentication error handling
- Validation error handling
- Custom error codes
"""

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
import time
from typing import Dict, Any, Optional
from datetime import datetime
import json

from ...core.rate_limiter import RateLimitExceeded
from ...security.auth import AuthenticationError, AuthorizationError
from ...engines.visual_generation.exceptions import (
    VisualGenerationError, ModelError, ResourceError, SafetyError
)

logger = logging.getLogger(__name__)


class ErrorResponse:
    """Standardized error response format."""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.retry_after = retry_after
        self.request_id = request_id
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        response = {
            "success": False,
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
                "timestamp": self.timestamp.isoformat()
            }
        }
        
        if self.request_id:
            response["request_id"] = self.request_id
        
        if self.retry_after:
            response["retry_after"] = self.retry_after
        
        return response
    
    def to_json_response(self) -> JSONResponse:
        """Convert to FastAPI JSONResponse."""
        headers = {}
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        
        return JSONResponse(
            status_code=self.status_code,
            content=self.to_dict(),
            headers=headers
        )


class ErrorTracker:
    """Track and analyze errors for monitoring."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_history: list = []
        self.max_history = 1000
    
    def track_error(
        self,
        error_code: str,
        message: str,
        request_path: str,
        user_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Track an error occurrence."""
        try:
            # Increment error count
            self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
            
            # Add to history
            error_entry = {
                "error_code": error_code,
                "message": message,
                "request_path": request_path,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "additional_data": additional_data or {}
            }
            
            self.error_history.append(error_entry)
            
            # Maintain history size
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history:]
            
            # Log error
            logger.error(f"Error tracked: {error_code} - {message}", extra={
                "error_code": error_code,
                "request_path": request_path,
                "user_id": user_id,
                "additional_data": additional_data
            })
            
        except Exception as e:
            logger.error(f"Failed to track error: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts.copy(),
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }


# Global error tracker
error_tracker = ErrorTracker()


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and handle errors."""
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', None)
        
        try:
            response = await call_next(request)
            return response
            
        except RateLimitExceeded as e:
            # Rate limit exceeded
            error_response = ErrorResponse(
                error_code="RATE_LIMIT_EXCEEDED",
                message=e.message,
                details={
                    "retry_after": e.retry_after,
                    "scrollintel_advantage": "Upgrade to Pro for higher limits - still cheaper than competitors!"
                },
                status_code=429,
                retry_after=e.retry_after,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "RATE_LIMIT_EXCEEDED",
                e.message,
                request.url.path,
                getattr(request.state, 'user_id', None)
            )
            
            return error_response.to_json_response()
            
        except AuthenticationError as e:
            # Authentication failed
            error_response = ErrorResponse(
                error_code="AUTHENTICATION_FAILED",
                message=str(e),
                details={
                    "help": "Get your FREE API key at https://scrollintel.com/api-keys",
                    "scrollintel_advantage": "FREE API access vs paid competitors"
                },
                status_code=401,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "AUTHENTICATION_FAILED",
                str(e),
                request.url.path
            )
            
            return error_response.to_json_response()
            
        except AuthorizationError as e:
            # Authorization failed
            error_response = ErrorResponse(
                error_code="AUTHORIZATION_FAILED",
                message=str(e),
                details={
                    "help": "Upgrade your plan for additional permissions",
                    "scrollintel_advantage": "Even our paid plans cost less than competitors"
                },
                status_code=403,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "AUTHORIZATION_FAILED",
                str(e),
                request.url.path,
                getattr(request.state, 'user_id', None)
            )
            
            return error_response.to_json_response()
            
        except SafetyError as e:
            # Content safety violation
            error_response = ErrorResponse(
                error_code="CONTENT_SAFETY_VIOLATION",
                message=str(e),
                details={
                    "help": "Please modify your prompt to comply with content policy",
                    "policy_url": "https://scrollintel.com/content-policy",
                    "scrollintel_advantage": "Advanced safety filters protect you and comply with regulations"
                },
                status_code=400,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "CONTENT_SAFETY_VIOLATION",
                str(e),
                request.url.path,
                getattr(request.state, 'user_id', None),
                {"prompt_length": len(request.query_params.get("prompt", ""))}
            )
            
            return error_response.to_json_response()
            
        except ModelError as e:
            # Model execution error
            error_response = ErrorResponse(
                error_code="MODEL_ERROR",
                message=str(e),
                details={
                    "help": "Model temporarily unavailable, trying alternative models",
                    "scrollintel_advantage": "Multiple model fallbacks ensure high availability"
                },
                status_code=503,
                retry_after=60,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "MODEL_ERROR",
                str(e),
                request.url.path,
                getattr(request.state, 'user_id', None)
            )
            
            return error_response.to_json_response()
            
        except ResourceError as e:
            # Resource unavailable
            error_response = ErrorResponse(
                error_code="RESOURCE_UNAVAILABLE",
                message=str(e),
                details={
                    "help": "High demand detected, request queued for processing",
                    "scrollintel_advantage": "FREE queuing vs paid priority on other platforms"
                },
                status_code=503,
                retry_after=120,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "RESOURCE_UNAVAILABLE",
                str(e),
                request.url.path,
                getattr(request.state, 'user_id', None)
            )
            
            return error_response.to_json_response()
            
        except VisualGenerationError as e:
            # General visual generation error
            error_response = ErrorResponse(
                error_code="GENERATION_ERROR",
                message=str(e),
                details={
                    "help": "Generation failed, please try again or modify your request",
                    "scrollintel_advantage": "Advanced error recovery and retry mechanisms"
                },
                status_code=500,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "GENERATION_ERROR",
                str(e),
                request.url.path,
                getattr(request.state, 'user_id', None)
            )
            
            return error_response.to_json_response()
            
        except RequestValidationError as e:
            # Pydantic validation error
            error_details = []
            for error in e.errors():
                error_details.append({
                    "field": " -> ".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })
            
            error_response = ErrorResponse(
                error_code="VALIDATION_ERROR",
                message="Request validation failed",
                details={
                    "validation_errors": error_details,
                    "help": "Please check your request parameters",
                    "scrollintel_advantage": "Comprehensive validation prevents wasted API calls"
                },
                status_code=422,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "VALIDATION_ERROR",
                "Request validation failed",
                request.url.path,
                getattr(request.state, 'user_id', None),
                {"validation_errors": error_details}
            )
            
            return error_response.to_json_response()
            
        except HTTPException as e:
            # FastAPI HTTP exception
            error_response = ErrorResponse(
                error_code="HTTP_ERROR",
                message=e.detail,
                status_code=e.status_code,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "HTTP_ERROR",
                e.detail,
                request.url.path,
                getattr(request.state, 'user_id', None),
                {"status_code": e.status_code}
            )
            
            return error_response.to_json_response()
            
        except StarletteHTTPException as e:
            # Starlette HTTP exception
            error_response = ErrorResponse(
                error_code="HTTP_ERROR",
                message=str(e.detail),
                status_code=e.status_code,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "HTTP_ERROR",
                str(e.detail),
                request.url.path,
                getattr(request.state, 'user_id', None),
                {"status_code": e.status_code}
            )
            
            return error_response.to_json_response()
            
        except Exception as e:
            # Unexpected error
            processing_time = time.time() - start_time
            
            # Log full traceback for debugging
            logger.error(f"Unexpected error in {request.url.path}: {str(e)}", extra={
                "traceback": traceback.format_exc(),
                "processing_time": processing_time,
                "request_id": request_id
            })
            
            error_response = ErrorResponse(
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                details={
                    "help": "Please try again or contact support if the issue persists",
                    "support_email": "support@scrollintel.com",
                    "scrollintel_advantage": "24/7 support included with all plans"
                },
                status_code=500,
                request_id=request_id
            )
            
            error_tracker.track_error(
                "INTERNAL_ERROR",
                str(e),
                request.url.path,
                getattr(request.state, 'user_id', None),
                {
                    "processing_time": processing_time,
                    "error_type": type(e).__name__
                }
            )
            
            return error_response.to_json_response()


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    retry_after: Optional[int] = None
) -> JSONResponse:
    """Helper function to create standardized error responses."""
    error_response = ErrorResponse(
        error_code=error_code,
        message=message,
        status_code=status_code,
        details=details,
        retry_after=retry_after
    )
    
    return error_response.to_json_response()


def get_error_stats() -> Dict[str, Any]:
    """Get current error statistics."""
    return error_tracker.get_error_stats()


# Custom exception handlers for FastAPI
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceptions."""
    return create_error_response(
        error_code="RATE_LIMIT_EXCEEDED",
        message=exc.message,
        status_code=429,
        details={
            "retry_after": exc.retry_after,
            "scrollintel_advantage": "Upgrade to Pro for higher limits - still cheaper than competitors!"
        },
        retry_after=exc.retry_after
    )


async def auth_exception_handler(request: Request, exc: AuthenticationError):
    """Handle authentication exceptions."""
    return create_error_response(
        error_code="AUTHENTICATION_FAILED",
        message=str(exc),
        status_code=401,
        details={
            "help": "Get your FREE API key at https://scrollintel.com/api-keys",
            "scrollintel_advantage": "FREE API access vs paid competitors"
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation exceptions."""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return create_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        status_code=422,
        details={
            "validation_errors": error_details,
            "help": "Please check your request parameters",
            "scrollintel_advantage": "Comprehensive validation prevents wasted API calls"
        }
    )


__all__ = [
    'ErrorHandlingMiddleware', 'ErrorResponse', 'ErrorTracker', 'error_tracker',
    'create_error_response', 'get_error_stats', 'rate_limit_exception_handler',
    'auth_exception_handler', 'validation_exception_handler'
]