"""Rate limiting middleware."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict, deque
from typing import Dict, Deque
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent API abuse."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times: Dict[str, Deque[float]] = defaultdict(deque)
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health"]:
            return await call_next(request)
        
        # Check rate limit
        current_time = time.time()
        if self._is_rate_limited(client_ip, current_time):
            return Response(
                content='{"error": {"code": 429, "message": "Rate limit exceeded. Please try again later."}}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self._record_request(client_ip, current_time)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit."""
        request_times = self.request_times[client_ip]
        
        # Remove old requests (older than 1 minute)
        cutoff_time = current_time - 60
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        # Check if limit exceeded
        return len(request_times) >= self.requests_per_minute
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record a request for rate limiting."""
        self.request_times[client_ip].append(current_time)
        
        # Clean up old entries to prevent memory leaks
        if len(self.request_times) > 1000:  # Arbitrary limit
            # Remove oldest client entries
            oldest_clients = sorted(
                self.request_times.keys(),
                key=lambda k: self.request_times[k][0] if self.request_times[k] else 0
            )[:100]
            
            for client in oldest_clients:
                if not self.request_times[client]:
                    del self.request_times[client]