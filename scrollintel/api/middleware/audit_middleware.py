"""
Audit Middleware for ScrollIntel API

Automatically logs all API requests and responses for audit trail compliance.
"""

import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from ...core.audit_system import audit_system, AuditAction
from ...core.logging_config import get_logger

logger = get_logger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log API requests for audit compliance"""
    
    def __init__(self, app, excluded_paths: list = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log audit information"""
        
        # Skip audit logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Extract request information
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        headers = dict(request.headers)
        client_ip = self._get_client_ip(request)
        user_agent = headers.get("user-agent", "Unknown")
        
        # Get user information if available
        user_id = getattr(request.state, 'user_id', None)
        user_email = getattr(request.state, 'user_email', None)
        session_id = getattr(request.state, 'session_id', None)
        
        # Read request body for POST/PUT requests (if not too large)
        request_body = None
        if method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) < 10000:  # Only log small request bodies
                    request_body = body.decode('utf-8')
            except Exception:
                request_body = "[Unable to read request body]"
        
        # Process the request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Extract response information
        status_code = response.status_code
        success = 200 <= status_code < 400
        
        # Read response body for logging (if not streaming and not too large)
        response_body = None
        if not isinstance(response, StreamingResponse) and hasattr(response, 'body'):
            try:
                if len(response.body) < 5000:  # Only log small response bodies
                    response_body = response.body.decode('utf-8')
            except Exception:
                response_body = "[Unable to read response body]"
        
        # Determine audit action based on HTTP method and path
        audit_action = self._determine_audit_action(method, path)
        
        # Determine resource type and ID from path
        resource_type, resource_id = self._extract_resource_info(path)
        
        # Prepare audit details
        audit_details = {
            "http_method": method,
            "path": path,
            "query_params": query_params,
            "status_code": status_code,
            "response_time": round(response_time, 3),
            "client_ip": client_ip,
            "user_agent": user_agent
        }
        
        # Add request/response bodies if available and not sensitive
        if request_body and not self._is_sensitive_data(path, request_body):
            audit_details["request_body"] = request_body
        
        if response_body and not self._is_sensitive_data(path, response_body):
            audit_details["response_body"] = response_body
        
        # Add error information for failed requests
        if not success:
            audit_details["error"] = True
            if response_body:
                try:
                    error_data = json.loads(response_body)
                    audit_details["error_message"] = error_data.get("message", "Unknown error")
                except:
                    audit_details["error_message"] = "Failed to parse error response"
        
        # Log the audit event
        try:
            await audit_system.log_event(
                action=audit_action,
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id,
                user_email=user_email,
                session_id=session_id,
                ip_address=client_ip,
                user_agent=user_agent,
                details=audit_details,
                success=success,
                error_message=audit_details.get("error_message")
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers first (for load balancers/proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _determine_audit_action(self, method: str, path: str) -> AuditAction:
        """Determine appropriate audit action based on HTTP method and path"""
        
        # Authentication endpoints
        if "/auth/" in path:
            if "login" in path:
                return AuditAction.LOGIN_SUCCESS
            elif "logout" in path:
                return AuditAction.LOGOUT
            elif "register" in path:
                return AuditAction.USER_CREATE
            else:
                return AuditAction.API_REQUEST
        
        # User management endpoints
        elif "/users/" in path:
            if method == "POST":
                return AuditAction.USER_CREATE
            elif method in ["PUT", "PATCH"]:
                return AuditAction.USER_UPDATE
            elif method == "DELETE":
                return AuditAction.USER_DELETE
            else:
                return AuditAction.API_REQUEST
        
        # Data endpoints
        elif "/data/" in path or "/upload/" in path:
            if method == "POST":
                return AuditAction.DATA_UPLOAD
            elif method == "DELETE":
                return AuditAction.DATA_DELETE
            elif method == "GET" and "export" in path:
                return AuditAction.DATA_EXPORT
            else:
                return AuditAction.DATA_VIEW
        
        # Dashboard endpoints
        elif "/dashboard/" in path:
            if method == "POST":
                return AuditAction.DASHBOARD_CREATE
            elif method in ["PUT", "PATCH"]:
                return AuditAction.DASHBOARD_UPDATE
            elif method == "DELETE":
                return AuditAction.DASHBOARD_DELETE
            elif "share" in path:
                return AuditAction.DASHBOARD_SHARE
            else:
                return AuditAction.DASHBOARD_VIEW
        
        # Agent endpoints
        elif "/agent/" in path:
            if method == "POST" and "execute" in path:
                return AuditAction.AGENT_EXECUTE
            elif method == "POST":
                return AuditAction.AGENT_CREATE
            elif method in ["PUT", "PATCH"]:
                return AuditAction.AGENT_UPDATE
            elif method == "DELETE":
                return AuditAction.AGENT_DELETE
            else:
                return AuditAction.API_REQUEST
        
        # Model endpoints
        elif "/model/" in path:
            if method == "POST" and "train" in path:
                return AuditAction.MODEL_TRAIN
            elif method == "POST" and "predict" in path:
                return AuditAction.MODEL_PREDICT
            elif method == "POST" and "deploy" in path:
                return AuditAction.MODEL_DEPLOY
            elif method == "POST":
                return AuditAction.MODEL_CREATE
            elif method == "DELETE":
                return AuditAction.MODEL_DELETE
            else:
                return AuditAction.API_REQUEST
        
        # API key endpoints
        elif "/api-key/" in path:
            if method == "POST":
                return AuditAction.API_KEY_CREATE
            elif method == "DELETE":
                return AuditAction.API_KEY_DELETE
            else:
                return AuditAction.API_REQUEST
        
        # Default to generic API request
        else:
            return AuditAction.API_REQUEST
    
    def _extract_resource_info(self, path: str) -> tuple:
        """Extract resource type and ID from API path"""
        
        path_parts = path.strip("/").split("/")
        
        # Skip 'api' prefix if present
        if path_parts and path_parts[0] == "api":
            path_parts = path_parts[1:]
        
        if not path_parts:
            return "api", None
        
        resource_type = path_parts[0]
        resource_id = None
        
        # Look for ID in path (usually second part or after specific keywords)
        if len(path_parts) > 1:
            # Check if second part looks like an ID
            potential_id = path_parts[1]
            if (potential_id.isdigit() or 
                len(potential_id) == 36 or  # UUID length
                potential_id.startswith(("id_", "uuid_"))):
                resource_id = potential_id
            
            # Check for ID after specific keywords
            for i, part in enumerate(path_parts):
                if part in ["id", "uuid"] and i + 1 < len(path_parts):
                    resource_id = path_parts[i + 1]
                    break
        
        return resource_type, resource_id
    
    def _is_sensitive_data(self, path: str, data: str) -> bool:
        """Check if data contains sensitive information that shouldn't be logged"""
        
        sensitive_keywords = [
            "password", "token", "secret", "key", "credential",
            "ssn", "social_security", "credit_card", "card_number",
            "cvv", "pin", "private_key", "api_key"
        ]
        
        # Check path for sensitive endpoints
        if any(keyword in path.lower() for keyword in sensitive_keywords):
            return True
        
        # Check data content for sensitive patterns
        data_lower = data.lower()
        if any(keyword in data_lower for keyword in sensitive_keywords):
            return True
        
        # Check for common sensitive patterns
        import re
        
        # Credit card pattern
        if re.search(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', data):
            return True
        
        # SSN pattern
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', data):
            return True
        
        # Email in password context
        if "password" in data_lower and "@" in data:
            return True
        
        return False


class ComplianceAuditMiddleware(BaseHTTPMiddleware):
    """Enhanced audit middleware with compliance-specific logging"""
    
    def __init__(self, app, compliance_frameworks: list = None):
        super().__init__(app)
        self.compliance_frameworks = compliance_frameworks or ["gdpr", "sox", "iso27001"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with compliance-specific audit logging"""
        
        # Check if this request requires compliance logging
        requires_compliance = self._requires_compliance_logging(request)
        
        if not requires_compliance:
            return await call_next(request)
        
        # Enhanced audit logging for compliance
        start_time = time.time()
        
        # Extract detailed request information
        audit_context = {
            "compliance_frameworks": self.compliance_frameworks,
            "data_classification": self._classify_data(request),
            "access_justification": request.headers.get("x-access-justification"),
            "business_purpose": request.headers.get("x-business-purpose"),
            "retention_period": request.headers.get("x-retention-period")
        }
        
        # Process request
        response = await call_next(request)
        
        # Log compliance-specific audit event
        response_time = time.time() - start_time
        
        try:
            await audit_system.log_event(
                action=AuditAction.COMPLIANCE_EXPORT,
                resource_type="compliance_audit",
                details={
                    **audit_context,
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "compliance_check": True
                },
                user_id=getattr(request.state, 'user_id', None),
                user_email=getattr(request.state, 'user_email', None),
                ip_address=request.client.host if hasattr(request.client, 'host') else None,
                success=200 <= response.status_code < 400
            )
        except Exception as e:
            logger.error(f"Failed to log compliance audit event: {e}")
        
        return response
    
    def _requires_compliance_logging(self, request: Request) -> bool:
        """Determine if request requires compliance-specific logging"""
        
        compliance_paths = [
            "/api/users/",
            "/api/data/",
            "/api/export/",
            "/api/reports/",
            "/api/audit/"
        ]
        
        return any(request.url.path.startswith(path) for path in compliance_paths)
    
    def _classify_data(self, request: Request) -> str:
        """Classify data sensitivity level based on request"""
        
        path = request.url.path.lower()
        
        if any(term in path for term in ["personal", "user", "profile"]):
            return "personal_data"
        elif any(term in path for term in ["financial", "billing", "payment"]):
            return "financial_data"
        elif any(term in path for term in ["health", "medical"]):
            return "health_data"
        else:
            return "general_data"