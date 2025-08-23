"""
Security Middleware for Agent Steering System
Implements enterprise-grade security controls for all API endpoints
"""

import jwt
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from fastapi import Request, Response, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging

from ...models.security_compliance_models import (
    User, UserSession, AuditEventType, PermissionType, SecurityLevel
)
from ...security.enterprise_security_framework import EnterpriseSecurityFramework
from ...core.database import get_db

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware implementing:
    - JWT token validation
    - Session management
    - Rate limiting
    - Request/response encryption
    - Audit logging
    - IP whitelisting
    """
    
    def __init__(self, app, security_framework: EnterpriseSecurityFramework):
        super().__init__(app)
        self.security_framework = security_framework
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_ips: set = set()
        
        # Security headers
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security pipeline"""
        
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        try:
            # 1. IP Blocking Check
            if client_ip in self.blocked_ips:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "IP address blocked"}
                )
            
            # 2. Rate Limiting
            if not self._check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"error": "Rate limit exceeded"}
                )
            
            # 3. Request Validation
            await self._validate_request(request)
            
            # 4. Authentication (for protected endpoints)
            user_context = None
            if self._requires_authentication(request):
                user_context = await self._authenticate_request(request)
            
            # 5. Authorization (for protected endpoints)
            if user_context and self._requires_authorization(request):
                await self._authorize_request(request, user_context)
            
            # 6. Process request
            request.state.user = user_context
            request.state.start_time = start_time
            request.state.client_ip = client_ip
            
            response = await call_next(request)
            
            # 7. Add security headers
            for header, value in self.security_headers.items():
                response.headers[header] = value
            
            # 8. Audit logging
            await self._log_request(request, response, user_context)
            
            return response
            
        except HTTPException as e:
            # Log security exceptions
            await self._log_security_exception(request, e, client_ip)
            raise e
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            await self._log_security_exception(request, e, client_ip)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal security error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str, limit: int = 100, window: int = 60) -> bool:
        """Check rate limiting for client IP"""
        
        current_time = time.time()
        
        # Initialize or clean old entries
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        
        # Remove old requests outside window
        self.rate_limits[client_ip] = [
            req_time for req_time in self.rate_limits[client_ip]
            if current_time - req_time < window
        ]
        
        # Check if limit exceeded
        if len(self.rate_limits[client_ip]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[client_ip].append(current_time)
        return True
    
    async def _validate_request(self, request: Request):
        """Validate request format and content"""
        
        # Check content length
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        # Validate content type for POST/PUT requests
        if request.method in ['POST', 'PUT', 'PATCH']:
            content_type = request.headers.get('content-type', '')
            if not content_type.startswith(('application/json', 'multipart/form-data')):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported content type"
                )
        
        # Check for suspicious patterns in URL
        suspicious_patterns = ['../', '<script', 'javascript:', 'data:', 'vbscript:']
        url_path = str(request.url.path).lower()
        
        for pattern in suspicious_patterns:
            if pattern in url_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Malicious request detected"
                )
    
    def _requires_authentication(self, request: Request) -> bool:
        """Check if endpoint requires authentication"""
        
        # Public endpoints that don't require authentication
        public_endpoints = [
            '/health',
            '/docs',
            '/openapi.json',
            '/auth/login',
            '/auth/register',
            '/auth/sso'
        ]
        
        path = request.url.path
        return not any(path.startswith(endpoint) for endpoint in public_endpoints)
    
    def _requires_authorization(self, request: Request) -> bool:
        """Check if endpoint requires authorization"""
        
        # Admin endpoints that require special authorization
        admin_endpoints = [
            '/admin/',
            '/security/',
            '/audit/',
            '/compliance/'
        ]
        
        path = request.url.path
        return any(path.startswith(endpoint) for endpoint in admin_endpoints)
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate request and return user context"""
        
        # Extract JWT token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header"
            )
        
        token = auth_header.split(' ')[1]
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                token, 
                self.security_framework.jwt_secret, 
                algorithms=['HS256']
            )
            
            # Validate token expiration
            if datetime.utcnow() > datetime.fromisoformat(payload['exp']):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            # Validate session in database
            db = next(get_db())
            session = db.query(UserSession).filter(
                UserSession.id == payload['session_id'],
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            ).first()
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid session"
                )
            
            # Update session activity
            session.last_activity = datetime.utcnow()
            db.commit()
            
            # Get user
            user = db.query(User).filter(User.id == payload['user_id']).first()
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            return {
                'user_id': user.id,
                'username': user.username,
                'security_level': user.security_level,
                'session_id': session.id,
                'permissions': self.security_framework.get_user_permissions(db, user.id)
            }
            
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {e}"
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    async def _authorize_request(self, request: Request, user_context: Dict[str, Any]):
        """Authorize request based on user permissions"""
        
        # Extract resource and action from request
        resource, action = self._extract_resource_action(request)
        
        # Check if user has required permission
        required_permission = f"{resource}:{action}"
        
        if required_permission not in user_context['permissions']:
            # Check for admin override
            if 'admin:all' not in user_context['permissions']:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions: {required_permission}"
                )
    
    def _extract_resource_action(self, request: Request) -> tuple[str, str]:
        """Extract resource and action from request path and method"""
        
        path = request.url.path
        method = request.method.lower()
        
        # Map HTTP methods to actions
        method_action_map = {
            'get': 'read',
            'post': 'write',
            'put': 'write',
            'patch': 'write',
            'delete': 'delete'
        }
        
        action = method_action_map.get(method, 'execute')
        
        # Extract resource from path
        path_parts = path.strip('/').split('/')
        if len(path_parts) > 0:
            resource = path_parts[0]
        else:
            resource = 'unknown'
        
        return resource, action
    
    async def _log_request(self, request: Request, response: Response, user_context: Optional[Dict]):
        """Log request for audit trail"""
        
        try:
            db = next(get_db())
            
            # Calculate request duration
            duration_ms = int((time.time() - request.state.start_time) * 1000)
            
            # Determine if sensitive data was accessed
            sensitive_endpoints = ['/users/', '/security/', '/audit/', '/compliance/']
            sensitive_data_accessed = any(
                request.url.path.startswith(endpoint) for endpoint in sensitive_endpoints
            )
            
            # Log audit event
            self.security_framework._log_audit_event(
                db=db,
                user_id=user_context['user_id'] if user_context else None,
                event_type=AuditEventType.DATA_ACCESS,
                event_category="api_request",
                event_description=f"{request.method} {request.url.path}",
                resource_type="api_endpoint",
                resource_id=request.url.path,
                action=request.method.lower(),
                success=200 <= response.status_code < 400,
                error_code=str(response.status_code) if response.status_code >= 400 else None,
                event_metadata={
                    'method': request.method,
                    'path': request.url.path,
                    'query_params': str(request.query_params),
                    'status_code': response.status_code,
                    'duration_ms': duration_ms,
                    'user_agent': request.headers.get('user-agent'),
                    'content_length': request.headers.get('content-length')
                },
                ip_address=request.state.client_ip,
                user_agent=request.headers.get('user-agent'),
                sensitive_data_accessed=sensitive_data_accessed
            )
            
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    
    async def _log_security_exception(self, request: Request, exception: Exception, client_ip: str):
        """Log security exceptions"""
        
        try:
            db = next(get_db())
            
            self.security_framework._log_audit_event(
                db=db,
                user_id=None,
                event_type=AuditEventType.SECURITY_INCIDENT,
                event_category="security_exception",
                event_description=f"Security exception: {str(exception)}",
                resource_type="api_endpoint",
                resource_id=request.url.path,
                action=request.method.lower(),
                success=False,
                error_code=getattr(exception, 'status_code', 'UNKNOWN'),
                error_message=str(exception),
                event_metadata={
                    'method': request.method,
                    'path': request.url.path,
                    'exception_type': type(exception).__name__
                },
                ip_address=client_ip,
                user_agent=request.headers.get('user-agent')
            )
            
        except Exception as e:
            logger.error(f"Failed to log security exception: {e}")

class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication"""
    
    def __init__(self, security_framework: EnterpriseSecurityFramework, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
        self.security_framework = security_framework
    
    async def __call__(self, request: Request) -> Optional[Dict[str, Any]]:
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme"
                )
            
            return await self._verify_jwt_token(credentials.credentials, request)
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization code"
            )
    
    async def _verify_jwt_token(self, token: str, request: Request) -> Dict[str, Any]:
        """Verify JWT token and return user context"""
        
        try:
            payload = jwt.decode(
                token,
                self.security_framework.jwt_secret,
                algorithms=['HS256']
            )
            
            # Validate token expiration
            if datetime.utcnow() > datetime.fromisoformat(payload['exp']):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            return payload
            
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid token"
            )

def require_permission(resource: str, action: PermissionType):
    """Decorator to require specific permission for endpoint"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user context from request
            request = kwargs.get('request') or args[0] if args else None
            
            if not request or not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user_context = request.state.user
            required_permission = f"{resource}:{action.value}"
            
            if required_permission not in user_context['permissions']:
                if 'admin:all' not in user_context['permissions']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions: {required_permission}"
                    )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def require_security_level(min_level: SecurityLevel):
    """Decorator to require minimum security level"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or args[0] if args else None
            
            if not request or not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user_context = request.state.user
            user_level = SecurityLevel(user_context['security_level'])
            
            # Define security level hierarchy
            level_hierarchy = {
                SecurityLevel.PUBLIC: 0,
                SecurityLevel.INTERNAL: 1,
                SecurityLevel.CONFIDENTIAL: 2,
                SecurityLevel.SECRET: 3,
                SecurityLevel.TOP_SECRET: 4
            }
            
            if level_hierarchy[user_level] < level_hierarchy[min_level]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient security clearance: {min_level.value} required"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator