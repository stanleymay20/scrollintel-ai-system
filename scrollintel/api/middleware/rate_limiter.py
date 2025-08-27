"""
Rate Limiting Middleware for Advanced Analytics Dashboard API

This module provides rate limiting functionality to prevent API abuse
and ensure fair usage across different API key tiers.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis
import json

logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    """Rate limit types"""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    BANDWIDTH_PER_MINUTE = "bandwidth_per_minute"
    BANDWIDTH_PER_HOUR = "bandwidth_per_hour"

@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit: int
    window: int  # seconds
    type: RateLimitType
    burst_limit: Optional[int] = None  # Allow burst up to this limit

@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    limit: int
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None

class RateLimitStore:
    """Abstract rate limit store"""
    
    async def get_usage(self, key: str, window: int) -> int:
        """Get current usage for key within window"""
        raise NotImplementedError
    
    async def increment_usage(self, key: str, window: int, amount: int = 1) -> int:
        """Increment usage and return new total"""
        raise NotImplementedError
    
    async def get_bandwidth_usage(self, key: str, window: int) -> int:
        """Get current bandwidth usage for key within window"""
        raise NotImplementedError
    
    async def increment_bandwidth(self, key: str, window: int, bytes_used: int) -> int:
        """Increment bandwidth usage and return new total"""
        raise NotImplementedError

class MemoryRateLimitStore(RateLimitStore):
    """In-memory rate limit store (for development/testing)"""
    
    def __init__(self):
        self.usage_data = defaultdict(lambda: defaultdict(deque))
        self.bandwidth_data = defaultdict(lambda: defaultdict(deque))
        self.lock = asyncio.Lock()
    
    async def get_usage(self, key: str, window: int) -> int:
        """Get current usage for key within window"""
        async with self.lock:
            now = time.time()
            cutoff = now - window
            
            # Clean old entries
            usage_queue = self.usage_data[key][window]
            while usage_queue and usage_queue[0] < cutoff:
                usage_queue.popleft()
            
            return len(usage_queue)
    
    async def increment_usage(self, key: str, window: int, amount: int = 1) -> int:
        """Increment usage and return new total"""
        async with self.lock:
            now = time.time()
            cutoff = now - window
            
            # Clean old entries
            usage_queue = self.usage_data[key][window]
            while usage_queue and usage_queue[0] < cutoff:
                usage_queue.popleft()
            
            # Add new entries
            for _ in range(amount):
                usage_queue.append(now)
            
            return len(usage_queue)
    
    async def get_bandwidth_usage(self, key: str, window: int) -> int:
        """Get current bandwidth usage for key within window"""
        async with self.lock:
            now = time.time()
            cutoff = now - window
            
            # Clean old entries and sum bandwidth
            bandwidth_queue = self.bandwidth_data[key][window]
            total_bandwidth = 0
            
            # Remove old entries
            while bandwidth_queue and bandwidth_queue[0][0] < cutoff:
                bandwidth_queue.popleft()
            
            # Sum remaining entries
            for timestamp, bytes_used in bandwidth_queue:
                total_bandwidth += bytes_used
            
            return total_bandwidth
    
    async def increment_bandwidth(self, key: str, window: int, bytes_used: int) -> int:
        """Increment bandwidth usage and return new total"""
        async with self.lock:
            now = time.time()
            cutoff = now - window
            
            # Clean old entries
            bandwidth_queue = self.bandwidth_data[key][window]
            while bandwidth_queue and bandwidth_queue[0][0] < cutoff:
                bandwidth_queue.popleft()
            
            # Add new entry
            bandwidth_queue.append((now, bytes_used))
            
            # Calculate total
            total_bandwidth = sum(entry[1] for entry in bandwidth_queue)
            return total_bandwidth

class RedisRateLimitStore(RateLimitStore):
    """Redis-based rate limit store (for production)"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.redis_url)
    
    async def cleanup(self):
        """Cleanup Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get_usage(self, key: str, window: int) -> int:
        """Get current usage for key within window"""
        if not self.redis_client:
            await self.initialize()
        
        now = time.time()
        cutoff = now - window
        
        # Use sorted set to store timestamps
        redis_key = f"rate_limit:requests:{key}:{window}"
        
        # Remove old entries
        await self.redis_client.zremrangebyscore(redis_key, 0, cutoff)
        
        # Count remaining entries
        count = await self.redis_client.zcard(redis_key)
        return count
    
    async def increment_usage(self, key: str, window: int, amount: int = 1) -> int:
        """Increment usage and return new total"""
        if not self.redis_client:
            await self.initialize()
        
        now = time.time()
        cutoff = now - window
        
        redis_key = f"rate_limit:requests:{key}:{window}"
        
        # Use pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(redis_key, 0, cutoff)
        
        # Add new entries
        for i in range(amount):
            pipe.zadd(redis_key, {f"{now}:{i}": now})
        
        # Set expiration
        pipe.expire(redis_key, window + 60)  # Extra buffer
        
        # Count total
        pipe.zcard(redis_key)
        
        results = await pipe.execute()
        return results[-1]  # Return count from last operation
    
    async def get_bandwidth_usage(self, key: str, window: int) -> int:
        """Get current bandwidth usage for key within window"""
        if not self.redis_client:
            await self.initialize()
        
        now = time.time()
        cutoff = now - window
        
        redis_key = f"rate_limit:bandwidth:{key}:{window}"
        
        # Remove old entries
        await self.redis_client.zremrangebyscore(redis_key, 0, cutoff)
        
        # Sum bandwidth from remaining entries
        entries = await self.redis_client.zrangebyscore(redis_key, cutoff, now, withscores=True)
        total_bandwidth = sum(int(member.decode().split(':')[1]) for member, score in entries)
        
        return total_bandwidth
    
    async def increment_bandwidth(self, key: str, window: int, bytes_used: int) -> int:
        """Increment bandwidth usage and return new total"""
        if not self.redis_client:
            await self.initialize()
        
        now = time.time()
        cutoff = now - window
        
        redis_key = f"rate_limit:bandwidth:{key}:{window}"
        
        # Use pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(redis_key, 0, cutoff)
        
        # Add new entry
        pipe.zadd(redis_key, {f"{now}:{bytes_used}": now})
        
        # Set expiration
        pipe.expire(redis_key, window + 60)
        
        # Get all entries to calculate total
        pipe.zrangebyscore(redis_key, cutoff, now)
        
        results = await pipe.execute()
        entries = results[-1]
        
        # Calculate total bandwidth
        total_bandwidth = sum(int(entry.decode().split(':')[1]) for entry in entries)
        return total_bandwidth

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, rate_limits: Dict[str, Dict[str, Any]], store: Optional[RateLimitStore] = None):
        super().__init__(app)
        self.rate_limits = self._parse_rate_limits(rate_limits)
        self.store = store or MemoryRateLimitStore()
        
        # Paths exempt from rate limiting
        self.exempt_paths = {
            "/health",
            "/api/info",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    def _parse_rate_limits(self, rate_limits: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, RateLimit]]:
        """Parse rate limit configuration"""
        parsed_limits = {}
        
        for tier, limits in rate_limits.items():
            parsed_limits[tier] = {}
            
            # Parse request limits
            if "requests" in limits and "window" in limits:
                parsed_limits[tier]["requests"] = RateLimit(
                    limit=limits["requests"],
                    window=limits["window"],
                    type=RateLimitType.REQUESTS_PER_MINUTE,
                    burst_limit=limits.get("burst_limit")
                )
            
            # Parse bandwidth limits
            if "bandwidth" in limits:
                parsed_limits[tier]["bandwidth"] = RateLimit(
                    limit=limits["bandwidth"],
                    window=limits.get("bandwidth_window", 60),
                    type=RateLimitType.BANDWIDTH_PER_MINUTE
                )
        
        return parsed_limits
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting middleware"""
        # Skip rate limiting for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Get API key info from request state (set by auth middleware)
        api_key_info = getattr(request.state, 'api_key', None)
        
        if not api_key_info:
            # No API key info, skip rate limiting
            return await call_next(request)
        
        # Determine rate limit tier
        tier = self._get_rate_limit_tier(api_key_info)
        
        if tier not in self.rate_limits:
            # No rate limits configured for this tier
            return await call_next(request)
        
        # Check rate limits
        rate_limit_status = await self._check_rate_limits(api_key_info.id, tier, request)
        
        if not rate_limit_status:
            # Rate limit exceeded
            return self._create_rate_limit_response(rate_limit_status)
        
        # Process request
        response = await call_next(request)
        
        # Record usage after successful request
        await self._record_usage(api_key_info.id, tier, request, response)
        
        # Add rate limit headers to response
        self._add_rate_limit_headers(response, rate_limit_status)
        
        return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from rate limiting"""
        return path in self.exempt_paths
    
    def _get_rate_limit_tier(self, api_key_info) -> str:
        """Determine rate limit tier for API key"""
        # This would typically be based on the API key's subscription tier
        if hasattr(api_key_info, 'tier'):
            return api_key_info.tier
        elif hasattr(api_key_info, 'is_admin') and api_key_info.is_admin:
            return "enterprise"
        else:
            return "default"
    
    async def _check_rate_limits(self, api_key_id: str, tier: str, request: Request) -> Optional[RateLimitStatus]:
        """Check if request is within rate limits"""
        tier_limits = self.rate_limits.get(tier, {})
        
        # Check request rate limits
        if "requests" in tier_limits:
            request_limit = tier_limits["requests"]
            current_usage = await self.store.get_usage(
                f"{api_key_id}:requests",
                request_limit.window
            )
            
            if current_usage >= request_limit.limit:
                return RateLimitStatus(
                    limit=request_limit.limit,
                    remaining=0,
                    reset_time=int(time.time() + request_limit.window),
                    retry_after=request_limit.window
                )
        
        # Check bandwidth limits (if applicable)
        if "bandwidth" in tier_limits:
            bandwidth_limit = tier_limits["bandwidth"]
            current_bandwidth = await self.store.get_bandwidth_usage(
                f"{api_key_id}:bandwidth",
                bandwidth_limit.window
            )
            
            if current_bandwidth >= bandwidth_limit.limit:
                return RateLimitStatus(
                    limit=bandwidth_limit.limit,
                    remaining=0,
                    reset_time=int(time.time() + bandwidth_limit.window),
                    retry_after=bandwidth_limit.window
                )
        
        # Calculate remaining requests
        request_limit = tier_limits.get("requests")
        if request_limit:
            current_usage = await self.store.get_usage(
                f"{api_key_id}:requests",
                request_limit.window
            )
            
            return RateLimitStatus(
                limit=request_limit.limit,
                remaining=max(0, request_limit.limit - current_usage),
                reset_time=int(time.time() + request_limit.window)
            )
        
        return RateLimitStatus(limit=0, remaining=0, reset_time=0)
    
    async def _record_usage(self, api_key_id: str, tier: str, request: Request, response: Response):
        """Record API usage for rate limiting"""
        tier_limits = self.rate_limits.get(tier, {})
        
        # Record request
        if "requests" in tier_limits:
            request_limit = tier_limits["requests"]
            await self.store.increment_usage(
                f"{api_key_id}:requests",
                request_limit.window
            )
        
        # Record bandwidth usage
        if "bandwidth" in tier_limits:
            bandwidth_limit = tier_limits["bandwidth"]
            
            # Calculate request size
            request_size = int(request.headers.get("content-length", 0))
            
            # Calculate response size (approximate)
            response_size = 0
            if hasattr(response, 'body'):
                response_size = len(response.body)
            elif hasattr(response, 'content'):
                response_size = len(response.content)
            
            total_bytes = request_size + response_size
            
            if total_bytes > 0:
                await self.store.increment_bandwidth(
                    f"{api_key_id}:bandwidth",
                    bandwidth_limit.window,
                    total_bytes
                )
    
    def _create_rate_limit_response(self, status: RateLimitStatus) -> JSONResponse:
        """Create rate limit exceeded response"""
        headers = {
            "X-RateLimit-Limit": str(status.limit),
            "X-RateLimit-Remaining": str(status.remaining),
            "X-RateLimit-Reset": str(status.reset_time)
        }
        
        if status.retry_after:
            headers["Retry-After"] = str(status.retry_after)
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"API rate limit of {status.limit} requests exceeded. Try again in {status.retry_after} seconds.",
                "limit": status.limit,
                "remaining": status.remaining,
                "reset_time": status.reset_time
            },
            headers=headers
        )
    
    def _add_rate_limit_headers(self, response: Response, status: RateLimitStatus):
        """Add rate limit headers to response"""
        response.headers["X-RateLimit-Limit"] = str(status.limit)
        response.headers["X-RateLimit-Remaining"] = str(status.remaining)
        response.headers["X-RateLimit-Reset"] = str(status.reset_time)

# Utility functions

def create_rate_limit_config(
    default_requests: int = 100,
    default_window: int = 60,
    premium_requests: int = 1000,
    premium_window: int = 60,
    enterprise_requests: int = 10000,
    enterprise_window: int = 60
) -> Dict[str, Dict[str, Any]]:
    """Create standard rate limit configuration"""
    return {
        "default": {
            "requests": default_requests,
            "window": default_window
        },
        "premium": {
            "requests": premium_requests,
            "window": premium_window,
            "bandwidth": 10 * 1024 * 1024,  # 10MB per minute
            "bandwidth_window": 60
        },
        "enterprise": {
            "requests": enterprise_requests,
            "window": enterprise_window,
            "bandwidth": 100 * 1024 * 1024,  # 100MB per minute
            "bandwidth_window": 60
        }
    }

async def get_rate_limit_status(api_key_id: str, tier: str, store: RateLimitStore, rate_limits: Dict) -> RateLimitStatus:
    """Get current rate limit status for an API key"""
    tier_limits = rate_limits.get(tier, {})
    
    if "requests" in tier_limits:
        request_limit = tier_limits["requests"]
        current_usage = await store.get_usage(
            f"{api_key_id}:requests",
            request_limit.window
        )
        
        return RateLimitStatus(
            limit=request_limit.limit,
            remaining=max(0, request_limit.limit - current_usage),
            reset_time=int(time.time() + request_limit.window)
        )
    
    return RateLimitStatus(limit=0, remaining=0, reset_time=0)