"""
Secure API Gateway Implementation
Rate limiting, authentication, authorization, and security controls
"""

import asyncio
import json
import time
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
# import redis
# import aioredis
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import ipaddress

class AuthenticationMethod(Enum):
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    MUTUAL_TLS = "mutual_tls"
    BASIC_AUTH = "basic_auth"

class RateLimitType(Enum):
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    BANDWIDTH_PER_MINUTE = "bandwidth_per_minute"
    CONCURRENT_REQUESTS = "concurrent_requests"

@dataclass
class RateLimitRule:
    name: str
    limit_type: RateLimitType
    limit_value: int
    window_size: int  # in seconds
    scope: str  # 'global', 'per_ip', 'per_user', 'per_api_key'
    burst_allowance: int = 0
    penalty_duration: int = 300  # 5 minutes default

@dataclass
class APIEndpoint:
    path: str
    methods: List[str]
    auth_required: bool
    auth_methods: List[AuthenticationMethod]
    rate_limits: List[RateLimitRule]
    permissions: List[str]
    ip_whitelist: Optional[List[str]] = None
    ip_blacklist: Optional[List[str]] = None
    request_size_limit: int = 10 * 1024 * 1024  # 10MB default
    response_size_limit: int = 50 * 1024 * 1024  # 50MB default

@dataclass
class APIKey:
    key_id: str
    key_hash: str
    name: str
    permissions: List[str]
    rate_limits: List[RateLimitRule]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RequestContext:
    request_id: str
    source_ip: str
    user_id: Optional[str]
    api_key_id: Optional[str]
    endpoint: str
    method: str
    timestamp: datetime
    request_size: int
    headers: Dict[str, str]
    authenticated: bool = False
    permissions: List[str] = field(default_factory=list)

class SecureAPIGateway:
    """Enterprise-grade secure API gateway"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        
        # Authentication and authorization
        self.api_keys: Dict[str, APIKey] = {}
        self.jwt_secret = config.get('jwt_secret', 'your-secret-key')
        self.jwt_algorithm = config.get('jwt_algorithm', 'HS256')
        
        # Rate limiting
        self.rate_limit_storage = defaultdict(lambda: defaultdict(deque))
        self.blocked_ips = {}
        self.blocked_api_keys = {}
        
        # Security settings
        self.security_headers = config.get('security_headers', {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        })
        
        # API endpoints configuration
        self.endpoints: Dict[str, APIEndpoint] = {}
        self._load_endpoint_configurations()
        
        # Monitoring and analytics
        self.request_metrics = defaultdict(int)
        self.error_metrics = defaultdict(int)
        self.response_times = deque(maxlen=10000)
        
        # Initialize Redis for distributed rate limiting (disabled for demo)
        # asyncio.create_task(self._initialize_redis())
    
    async def _initialize_redis(self):
        """Initialize Redis connection for distributed rate limiting"""
        try:
            # redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            # self.redis_client = await aioredis.from_url(redis_url)
            print("Redis disabled for demo - using in-memory rate limiting")
            self.redis_client = None
        except Exception as e:
            print(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    def _load_endpoint_configurations(self):
        """Load API endpoint configurations"""
        # Default rate limits
        default_rate_limits = [
            RateLimitRule(
                name="default_per_minute",
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit_value=100,
                window_size=60,
                scope="per_ip"
            ),
            RateLimitRule(
                name="default_per_hour",
                limit_type=RateLimitType.REQUESTS_PER_HOUR,
                limit_value=1000,
                window_size=3600,
                scope="per_ip"
            )
        ]
        
        # Example endpoint configurations
        self.endpoints = {
            "/api/v1/auth/login": APIEndpoint(
                path="/api/v1/auth/login",
                methods=["POST"],
                auth_required=False,
                auth_methods=[],
                rate_limits=[
                    RateLimitRule(
                        name="login_rate_limit",
                        limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                        limit_value=5,
                        window_size=60,
                        scope="per_ip",
                        penalty_duration=900  # 15 minutes
                    )
                ],
                permissions=[],
                request_size_limit=1024  # 1KB for login requests
            ),
            
            "/api/v1/data/*": APIEndpoint(
                path="/api/v1/data/*",
                methods=["GET", "POST", "PUT", "DELETE"],
                auth_required=True,
                auth_methods=[AuthenticationMethod.API_KEY, AuthenticationMethod.JWT_TOKEN],
                rate_limits=default_rate_limits + [
                    RateLimitRule(
                        name="data_bandwidth",
                        limit_type=RateLimitType.BANDWIDTH_PER_MINUTE,
                        limit_value=100 * 1024 * 1024,  # 100MB per minute
                        window_size=60,
                        scope="per_api_key"
                    )
                ],
                permissions=["data.read", "data.write"],
                request_size_limit=50 * 1024 * 1024  # 50MB
            ),
            
            "/api/v1/admin/*": APIEndpoint(
                path="/api/v1/admin/*",
                methods=["GET", "POST", "PUT", "DELETE"],
                auth_required=True,
                auth_methods=[AuthenticationMethod.JWT_TOKEN],
                rate_limits=[
                    RateLimitRule(
                        name="admin_strict",
                        limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                        limit_value=20,
                        window_size=60,
                        scope="per_user"
                    )
                ],
                permissions=["admin.read", "admin.write"],
                ip_whitelist=self.config.get('admin_ip_whitelist', [])
            )
        }
    
    async def process_request(self, request: Request) -> RequestContext:
        """Process incoming request through security pipeline"""
        
        # Create request context
        context = RequestContext(
            request_id=self._generate_request_id(),
            source_ip=self._get_client_ip(request),
            user_id=None,
            api_key_id=None,
            endpoint=str(request.url.path),
            method=request.method,
            timestamp=datetime.now(),
            request_size=int(request.headers.get('content-length', 0)),
            headers=dict(request.headers)
        )
        
        # Find matching endpoint configuration
        endpoint_config = self._find_endpoint_config(context.endpoint, context.method)
        if not endpoint_config:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        
        # IP-based access control
        await self._check_ip_access(context, endpoint_config)
        
        # Rate limiting
        await self._check_rate_limits(context, endpoint_config)
        
        # Request size validation
        if context.request_size > endpoint_config.request_size_limit:
            raise HTTPException(status_code=413, detail="Request too large")
        
        # Authentication
        if endpoint_config.auth_required:
            await self._authenticate_request(request, context, endpoint_config)
        
        # Authorization
        if endpoint_config.permissions:
            await self._authorize_request(context, endpoint_config)
        
        # Update metrics
        self._update_request_metrics(context)
        
        return context
    
    def _find_endpoint_config(self, path: str, method: str) -> Optional[APIEndpoint]:
        """Find matching endpoint configuration"""
        # Exact match first
        if path in self.endpoints:
            endpoint = self.endpoints[path]
            if method in endpoint.methods:
                return endpoint
        
        # Wildcard matching
        for endpoint_path, endpoint in self.endpoints.items():
            if endpoint_path.endswith('/*'):
                prefix = endpoint_path[:-2]
                if path.startswith(prefix) and method in endpoint.methods:
                    return endpoint
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address considering proxies"""
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else 'unknown'
    
    async def _check_ip_access(self, context: RequestContext, endpoint: APIEndpoint):
        """Check IP-based access control"""
        
        # Check if IP is blocked
        if context.source_ip in self.blocked_ips:
            block_info = self.blocked_ips[context.source_ip]
            if datetime.now() < block_info['blocked_until']:
                raise HTTPException(status_code=403, detail="IP address blocked")
            else:
                # Remove expired block
                del self.blocked_ips[context.source_ip]
        
        # Check IP whitelist
        if endpoint.ip_whitelist:
            if not self._ip_in_list(context.source_ip, endpoint.ip_whitelist):
                raise HTTPException(status_code=403, detail="IP not whitelisted")
        
        # Check IP blacklist
        if endpoint.ip_blacklist:
            if self._ip_in_list(context.source_ip, endpoint.ip_blacklist):
                raise HTTPException(status_code=403, detail="IP blacklisted")
    
    def _ip_in_list(self, ip: str, ip_list: List[str]) -> bool:
        """Check if IP is in the given list (supports CIDR notation)"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            for ip_range in ip_list:
                if '/' in ip_range:
                    # CIDR notation
                    if ip_addr in ipaddress.ip_network(ip_range, strict=False):
                        return True
                else:
                    # Single IP
                    if ip_addr == ipaddress.ip_address(ip_range):
                        return True
        except ValueError:
            pass
        
        return False
    
    async def _check_rate_limits(self, context: RequestContext, endpoint: APIEndpoint):
        """Check rate limiting rules"""
        
        for rate_limit in endpoint.rate_limits:
            # Determine the key for rate limiting
            if rate_limit.scope == "global":
                key = "global"
            elif rate_limit.scope == "per_ip":
                key = f"ip:{context.source_ip}"
            elif rate_limit.scope == "per_user":
                key = f"user:{context.user_id}" if context.user_id else f"ip:{context.source_ip}"
            elif rate_limit.scope == "per_api_key":
                key = f"api_key:{context.api_key_id}" if context.api_key_id else f"ip:{context.source_ip}"
            else:
                key = f"ip:{context.source_ip}"
            
            # Check rate limit
            if await self._is_rate_limited(key, rate_limit, context):
                # Apply penalty
                await self._apply_rate_limit_penalty(context, rate_limit)
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    async def _is_rate_limited(self, key: str, rate_limit: RateLimitRule, context: RequestContext) -> bool:
        """Check if request exceeds rate limit"""
        
        current_time = time.time()
        window_start = current_time - rate_limit.window_size
        
        if self.redis_client:
            # Use Redis for distributed rate limiting
            return await self._redis_rate_limit_check(key, rate_limit, current_time)
        else:
            # Use in-memory rate limiting
            return self._memory_rate_limit_check(key, rate_limit, current_time, context)
    
    async def _redis_rate_limit_check(self, key: str, rate_limit: RateLimitRule, current_time: float) -> bool:
        """Redis-based distributed rate limiting"""
        
        redis_key = f"rate_limit:{rate_limit.name}:{key}"
        window_start = current_time - rate_limit.window_size
        
        # Remove old entries
        await self.redis_client.zremrangebyscore(redis_key, 0, window_start)
        
        # Count current requests
        current_count = await self.redis_client.zcard(redis_key)
        
        if current_count >= rate_limit.limit_value:
            return True
        
        # Add current request
        await self.redis_client.zadd(redis_key, {str(current_time): current_time})
        await self.redis_client.expire(redis_key, rate_limit.window_size)
        
        return False
    
    def _memory_rate_limit_check(self, key: str, rate_limit: RateLimitRule, 
                                current_time: float, context: RequestContext) -> bool:
        """In-memory rate limiting"""
        
        window_start = current_time - rate_limit.window_size
        
        # Clean old entries
        requests = self.rate_limit_storage[rate_limit.name][key]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if bandwidth limiting
        if rate_limit.limit_type == RateLimitType.BANDWIDTH_PER_MINUTE:
            total_bandwidth = sum(
                getattr(req, 'size', 0) for req in requests
                if hasattr(req, 'size')
            )
            if total_bandwidth + context.request_size > rate_limit.limit_value:
                return True
        else:
            # Request count limiting
            if len(requests) >= rate_limit.limit_value:
                return True
        
        # Add current request
        if rate_limit.limit_type == RateLimitType.BANDWIDTH_PER_MINUTE:
            # Store request with size for bandwidth tracking
            class RequestRecord:
                def __init__(self, timestamp, size):
                    self.timestamp = timestamp
                    self.size = size
                def __float__(self):
                    return self.timestamp
            
            requests.append(RequestRecord(current_time, context.request_size))
        else:
            requests.append(current_time)
        
        return False
    
    async def _apply_rate_limit_penalty(self, context: RequestContext, rate_limit: RateLimitRule):
        """Apply penalty for rate limit violation"""
        
        if rate_limit.penalty_duration > 0:
            penalty_until = datetime.now() + timedelta(seconds=rate_limit.penalty_duration)
            
            if context.api_key_id:
                self.blocked_api_keys[context.api_key_id] = {
                    'blocked_until': penalty_until,
                    'reason': f'Rate limit violation: {rate_limit.name}'
                }
            else:
                self.blocked_ips[context.source_ip] = {
                    'blocked_until': penalty_until,
                    'reason': f'Rate limit violation: {rate_limit.name}'
                }
    
    async def _authenticate_request(self, request: Request, context: RequestContext, endpoint: APIEndpoint):
        """Authenticate the request"""
        
        authenticated = False
        
        for auth_method in endpoint.auth_methods:
            try:
                if auth_method == AuthenticationMethod.API_KEY:
                    if await self._authenticate_api_key(request, context):
                        authenticated = True
                        break
                
                elif auth_method == AuthenticationMethod.JWT_TOKEN:
                    if await self._authenticate_jwt(request, context):
                        authenticated = True
                        break
                
                elif auth_method == AuthenticationMethod.OAUTH2:
                    if await self._authenticate_oauth2(request, context):
                        authenticated = True
                        break
                
            except Exception as e:
                print(f"Authentication error with {auth_method}: {e}")
                continue
        
        if not authenticated:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        context.authenticated = True
    
    async def _authenticate_api_key(self, request: Request, context: RequestContext) -> bool:
        """Authenticate using API key"""
        
        # Check for API key in header
        api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            return False
        
        # Hash the API key for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Find matching API key
        for key_id, api_key_obj in self.api_keys.items():
            if api_key_obj.key_hash == key_hash and api_key_obj.is_active:
                # Check expiration
                if api_key_obj.expires_at and datetime.now() > api_key_obj.expires_at:
                    continue
                
                # Update usage
                api_key_obj.last_used = datetime.now()
                api_key_obj.usage_count += 1
                
                # Set context
                context.api_key_id = key_id
                context.permissions = api_key_obj.permissions
                
                return True
        
        return False
    
    async def _authenticate_jwt(self, request: Request, context: RequestContext) -> bool:
        """Authenticate using JWT token"""
        
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return False
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Extract user information
            context.user_id = payload.get('sub')
            context.permissions = payload.get('permissions', [])
            
            # Check token expiration
            exp = payload.get('exp')
            if exp and datetime.fromtimestamp(exp) < datetime.now():
                return False
            
            return True
            
        except jwt.InvalidTokenError:
            return False
    
    async def _authenticate_oauth2(self, request: Request, context: RequestContext) -> bool:
        """Authenticate using OAuth2"""
        # Implement OAuth2 authentication logic
        # This would typically involve validating the token with the OAuth2 provider
        return False
    
    async def _authorize_request(self, context: RequestContext, endpoint: APIEndpoint):
        """Authorize the request based on permissions"""
        
        required_permissions = set(endpoint.permissions)
        user_permissions = set(context.permissions)
        
        if not required_permissions.issubset(user_permissions):
            missing_permissions = required_permissions - user_permissions
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Missing: {', '.join(missing_permissions)}"
            )
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return hashlib.md5(f"{time.time()}{id(self)}".encode()).hexdigest()
    
    def _update_request_metrics(self, context: RequestContext):
        """Update request metrics"""
        self.request_metrics[f"{context.method}:{context.endpoint}"] += 1
        self.request_metrics["total_requests"] += 1
        
        if context.authenticated:
            self.request_metrics["authenticated_requests"] += 1
        else:
            self.request_metrics["anonymous_requests"] += 1
    
    def add_api_key(self, name: str, permissions: List[str], 
                   expires_at: Optional[datetime] = None) -> str:
        """Add new API key"""
        
        # Generate API key
        import secrets
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()
        
        # Create API key object
        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            rate_limits=[],  # Use endpoint defaults
            created_at=datetime.now(),
            expires_at=expires_at,
            last_used=None,
            usage_count=0,
            is_active=True
        )
        
        self.api_keys[key_id] = api_key_obj
        
        return api_key  # Return the actual key (only time it's shown)
    
    def revoke_api_key(self, key_id: str):
        """Revoke an API key"""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers to add to responses"""
        return self.security_headers.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get API gateway metrics"""
        return {
            'request_metrics': dict(self.request_metrics),
            'error_metrics': dict(self.error_metrics),
            'active_api_keys': len([k for k in self.api_keys.values() if k.is_active]),
            'blocked_ips': len(self.blocked_ips),
            'blocked_api_keys': len(self.blocked_api_keys),
            'avg_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0
        }

class APIGatewayMiddleware:
    """FastAPI middleware for API gateway"""
    
    def __init__(self, gateway: SecureAPIGateway):
        self.gateway = gateway
    
    async def __call__(self, request: Request, call_next):
        """Process request through API gateway"""
        
        start_time = time.time()
        
        try:
            # Process request through security pipeline
            context = await self.gateway.process_request(request)
            
            # Add context to request state
            request.state.security_context = context
            
            # Process the request
            response = await call_next(request)
            
            # Add security headers
            for header, value in self.gateway.get_security_headers().items():
                response.headers[header] = value
            
            # Add request ID header
            response.headers['X-Request-ID'] = context.request_id
            
            # Update response time metrics
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.gateway.response_times.append(response_time)
            
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log unexpected errors
            print(f"API Gateway error: {e}")
            self.gateway.error_metrics["internal_errors"] += 1
            raise HTTPException(status_code=500, detail="Internal server error")

def create_secure_api_gateway(config: Dict[str, Any]) -> SecureAPIGateway:
    """Factory function to create configured API gateway"""
    
    gateway = SecureAPIGateway(config)
    
    # Add default admin API key if configured
    admin_key = config.get('admin_api_key')
    if admin_key:
        gateway.api_keys['admin'] = APIKey(
            key_id='admin',
            key_hash=hashlib.sha256(admin_key.encode()).hexdigest(),
            name='Admin Key',
            permissions=['admin.read', 'admin.write', 'data.read', 'data.write'],
            rate_limits=[],
            created_at=datetime.now(),
            expires_at=None,
            last_used=None,
            usage_count=0,
            is_active=True
        )
    
    return gateway