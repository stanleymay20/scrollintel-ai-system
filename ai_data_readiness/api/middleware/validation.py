"""Request validation middleware."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request validation."""
        # Log request details
        logger.info(f"{request.method} {request.url.path}")
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if content_type and not content_type.startswith("application/json"):
                if not content_type.startswith("multipart/form-data"):
                    return Response(
                        content='{"error": {"code": 400, "message": "Content-Type must be application/json"}}',
                        status_code=400,
                        media_type="application/json"
                    )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response


def validate_json_payload(payload: Dict[str, Any], required_fields: list = None) -> Dict[str, Any]:
    """Validate JSON payload structure."""
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    
    if required_fields:
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    return payload


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input."""
    if not isinstance(value, str):
        raise ValueError("Value must be a string")
    
    # Remove potentially dangerous characters
    sanitized = value.strip()
    
    # Limit length
    if len(sanitized) > max_length:
        raise ValueError(f"String length exceeds maximum of {max_length} characters")
    
    return sanitized


def validate_dataset_id(dataset_id: str) -> str:
    """Validate dataset ID format."""
    if not dataset_id or not isinstance(dataset_id, str):
        raise ValueError("Dataset ID must be a non-empty string")
    
    # Basic UUID format validation
    if len(dataset_id) != 36 or dataset_id.count('-') != 4:
        raise ValueError("Dataset ID must be a valid UUID format")
    
    return dataset_id


def validate_pagination_params(page: int = 1, size: int = 20) -> tuple:
    """Validate pagination parameters."""
    if page < 1:
        raise ValueError("Page number must be greater than 0")
    
    if size < 1 or size > 100:
        raise ValueError("Page size must be between 1 and 100")
    
    return page, size


def validate_score_range(score: float, field_name: str = "score") -> float:
    """Validate score is within valid range (0-1)."""
    if not isinstance(score, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    
    if not 0 <= score <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    
    return float(score)