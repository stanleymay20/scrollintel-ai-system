"""
Advanced Rate Limiting System for ScrollIntel API
Provides intelligent rate limiting with user-based quotas and burst handling.
"""

import time
import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import redis
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # seconds
    redis_key_prefix: str = "rate_limit"
    enable_redis: bool = True


@dataclass
class RateLimitStatus:
    """Status of rate limiting for a user"""
    allowed: bool
    remaining_requests: int
    reset_time: float
    retry_after: Optional[int] = None
    current_usage: int = 0
    limit_type: str = "requests_per_minute"


class RateLimiter:
    """
    Advanced rate limiter with multiple time windows and burst protection.
    Supports both in-memory and Redis-based storage for distributed systems.
    """
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000,
                 requests_per_day: int = 10000,
                 burst_limit: int = 10,
                 redis_url: Optional[str] = None):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour
            requests_per_day: Maximum requests per day
            burst_limit: Maximum burst requests in short time
            redis_url: Redis URL for distributed rate limiting
        """
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            requests_per_day=requests_per_day,
            burst_limit=burst_limit
        )
        
        # In-memory storage for rate limiting
        self.request_counts: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                'minute': deque(),
                'hour': deque(),
                'day': deque(),
                'burst': deque()
            }
        )
        
        # Redis client for distributed rate limiting
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.config.enable_redis = True
                logger.info("Redis rate limiting enabled")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory rate limiting.")
                self.config.enable_redis = False
        else:
            self.config.enable_redis = False
    
    async def check_rate_limit(self, user_id: str, ip_address: Optional[str] = None) -> RateLimitStatus:
        """
        Check if request is allowed under rate limits.
        
        Args:
            user_id: User identifier
            ip_address: Client IP address for additional limiting
            
        Returns:
            RateLimitStatus indicating if request is allowed
        """
        current_time = time.time()
        
        # Use Redis if available, otherwise use in-memory
        if self.config.enable_redis and self.redis_client:
            return await self._check_rate_limit_redis(user_id, current_time, ip_address)
        else:
            return await self._check_rate_limit_memory(user_id, current_time, ip_address)
    
    async def _check_rate_limit_redis(self, user_id: str, current_time: float, ip_address: Optional[str]) -> RateLimitStatus:
        """Check rate limits using Redis storage"""
        try:
            pipe = self.redis_client.pipeline()
            
            # Keys for different time windows
            minute_key = f"{self.config.redis_key_prefix}:{user_id}:minute"
            hour_key = f"{self.config.redis_key_prefix}:{user_id}:hour"
            day_key = f"{self.config.redis_key_prefix}:{user_id}:day"
            burst_key = f"{self.config.redis_key_prefix}:{user_id}:burst"
            
            # Get current counts
            pipe.zcount(minute_key, current_time - 60, current_time)
            pipe.zcount(hour_key, current_time - 3600, current_time)
            pipe.zcount(day_key, current_time - 86400, current_time)
            pipe.zcount(burst_key, current_time - 10, current_time)
            
            counts = pipe.execute()
            minute_count, hour_count, day_count, burst_count = counts
            
            # Check limits
            if burst_count >= self.config.burst_limit:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=current_time + 10,
                    retry_after=10,
                    current_usage=burst_count,
                    limit_type="burst_limit"
                )
            
            if minute_count >= self.config.requests_per_minute:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=current_time + 60,
                    retry_after=60,
                    current_usage=minute_count,
                    limit_type="requests_per_minute"
                )
            
            if hour_count >= self.config.requests_per_hour:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=current_time + 3600,
                    retry_after=3600,
                    current_usage=hour_count,
                    limit_type="requests_per_hour"
                )
            
            if day_count >= self.config.requests_per_day:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=current_time + 86400,
                    retry_after=86400,
                    current_usage=day_count,
                    limit_type="requests_per_day"
                )
            
            # Record the request
            request_id = f"{current_time}:{user_id}"
            pipe.zadd(minute_key, {request_id: current_time})
            pipe.zadd(hour_key, {request_id: current_time})
            pipe.zadd(day_key, {request_id: current_time})
            pipe.zadd(burst_key, {request_id: current_time})
            
            # Set expiration times
            pipe.expire(minute_key, 120)  # 2 minutes
            pipe.expire(hour_key, 7200)   # 2 hours
            pipe.expire(day_key, 172800)  # 2 days
            pipe.expire(burst_key, 60)    # 1 minute
            
            # Clean up old entries
            pipe.zremrangebyscore(minute_key, 0, current_time - 60)
            pipe.zremrangebyscore(hour_key, 0, current_time - 3600)
            pipe.zremrangebyscore(day_key, 0, current_time - 86400)
            pipe.zremrangebyscore(burst_key, 0, current_time - 10)
            
            pipe.execute()
            
            return RateLimitStatus(
                allowed=True,
                remaining_requests=self.config.requests_per_minute - minute_count - 1,
                reset_time=current_time + 60,
                current_usage=minute_count + 1,
                limit_type="requests_per_minute"
            )
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fallback to in-memory
            return await self._check_rate_limit_memory(user_id, current_time, ip_address)
    
    async def _check_rate_limit_memory(self, user_id: str, current_time: float, ip_address: Optional[str]) -> RateLimitStatus:
        """Check rate limits using in-memory storage"""
        user_requests = self.request_counts[user_id]
        
        # Clean up old requests
        self._cleanup_old_requests(user_requests, current_time)
        
        # Check burst limit (last 10 seconds)
        burst_count = len(user_requests['burst'])
        if burst_count >= self.config.burst_limit:
            return RateLimitStatus(
                allowed=False,
                remaining_requests=0,
                reset_time=current_time + 10,
                retry_after=10,
                current_usage=burst_count,
                limit_type="burst_limit"
            )
        
        # Check minute limit
        minute_count = len(user_requests['minute'])
        if minute_count >= self.config.requests_per_minute:
            return RateLimitStatus(
                allowed=False,
                remaining_requests=0,
                reset_time=current_time + 60,
                retry_after=60,
                current_usage=minute_count,
                limit_type="requests_per_minute"
            )
        
        # Check hour limit
        hour_count = len(user_requests['hour'])
        if hour_count >= self.config.requests_per_hour:
            return RateLimitStatus(
                allowed=False,
                remaining_requests=0,
                reset_time=current_time + 3600,
                retry_after=3600,
                current_usage=hour_count,
                limit_type="requests_per_hour"
            )
        
        # Check day limit
        day_count = len(user_requests['day'])
        if day_count >= self.config.requests_per_day:
            return RateLimitStatus(
                allowed=False,
                remaining_requests=0,
                reset_time=current_time + 86400,
                retry_after=86400,
                current_usage=day_count,
                limit_type="requests_per_day"
            )
        
        # Record the request
        user_requests['minute'].append(current_time)
        user_requests['hour'].append(current_time)
        user_requests['day'].append(current_time)
        user_requests['burst'].append(current_time)
        
        return RateLimitStatus(
            allowed=True,
            remaining_requests=self.config.requests_per_minute - minute_count - 1,
            reset_time=current_time + 60,
            current_usage=minute_count + 1,
            limit_type="requests_per_minute"
        )
    
    def _cleanup_old_requests(self, user_requests: Dict[str, deque], current_time: float):
        """Clean up old requests from in-memory storage"""
        # Clean minute window (60 seconds)
        while user_requests['minute'] and user_requests['minute'][0] < current_time - 60:
            user_requests['minute'].popleft()
        
        # Clean hour window (3600 seconds)
        while user_requests['hour'] and user_requests['hour'][0] < current_time - 3600:
            user_requests['hour'].popleft()
        
        # Clean day window (86400 seconds)
        while user_requests['day'] and user_requests['day'][0] < current_time - 86400:
            user_requests['day'].popleft()
        
        # Clean burst window (10 seconds)
        while user_requests['burst'] and user_requests['burst'][0] < current_time - 10:
            user_requests['burst'].popleft()
    
    async def get_user_usage(self, user_id: str) -> Dict[str, Any]:
        """Get current usage statistics for a user"""
        current_time = time.time()
        
        if self.config.enable_redis and self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                
                minute_key = f"{self.config.redis_key_prefix}:{user_id}:minute"
                hour_key = f"{self.config.redis_key_prefix}:{user_id}:hour"
                day_key = f"{self.config.redis_key_prefix}:{user_id}:day"
                
                pipe.zcount(minute_key, current_time - 60, current_time)
                pipe.zcount(hour_key, current_time - 3600, current_time)
                pipe.zcount(day_key, current_time - 86400, current_time)
                
                counts = pipe.execute()
                minute_count, hour_count, day_count = counts
                
            except Exception as e:
                logger.error(f"Redis usage query error: {e}")
                # Fallback to in-memory
                user_requests = self.request_counts[user_id]
                self._cleanup_old_requests(user_requests, current_time)
                minute_count = len(user_requests['minute'])
                hour_count = len(user_requests['hour'])
                day_count = len(user_requests['day'])
        else:
            user_requests = self.request_counts[user_id]
            self._cleanup_old_requests(user_requests, current_time)
            minute_count = len(user_requests['minute'])
            hour_count = len(user_requests['hour'])
            day_count = len(user_requests['day'])
        
        return {
            "user_id": user_id,
            "current_usage": {
                "requests_per_minute": minute_count,
                "requests_per_hour": hour_count,
                "requests_per_day": day_count
            },
            "limits": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
                "burst_limit": self.config.burst_limit
            },
            "remaining": {
                "requests_per_minute": max(0, self.config.requests_per_minute - minute_count),
                "requests_per_hour": max(0, self.config.requests_per_hour - hour_count),
                "requests_per_day": max(0, self.config.requests_per_day - day_count)
            },
            "reset_times": {
                "minute": current_time + 60,
                "hour": current_time + 3600,
                "day": current_time + 86400
            }
        }
    
    async def reset_user_limits(self, user_id: str) -> bool:
        """Reset rate limits for a specific user (admin function)"""
        try:
            if self.config.enable_redis and self.redis_client:
                keys = [
                    f"{self.config.redis_key_prefix}:{user_id}:minute",
                    f"{self.config.redis_key_prefix}:{user_id}:hour",
                    f"{self.config.redis_key_prefix}:{user_id}:day",
                    f"{self.config.redis_key_prefix}:{user_id}:burst"
                ]
                self.redis_client.delete(*keys)
            else:
                if user_id in self.request_counts:
                    del self.request_counts[user_id]
            
            logger.info(f"Reset rate limits for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset rate limits for user {user_id}: {e}")
            return False


class RateLimitException(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, status: RateLimitStatus):
        self.status = status
        super().__init__(f"Rate limit exceeded: {status.limit_type}")


# Global rate limiter instances
default_rate_limiter = RateLimiter()
visual_generation_rate_limiter = RateLimiter(
    requests_per_minute=30,
    requests_per_hour=200,
    requests_per_day=1000,
    burst_limit=5
)