"""
API Key Management Service for ScrollIntel Launch MVP.
Handles API key generation, validation, and lifecycle management.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..models.api_key_models import APIKey, APIUsage, APIQuota, RateLimitRecord
from ..models.database import User
from ..core.config import get_settings


class APIKeyManager:
    """Service for managing API keys and access control."""
    
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        rate_limit_per_minute: int = 60,
        rate_limit_per_hour: int = 1000,
        rate_limit_per_day: int = 10000,
        quota_requests_per_month: Optional[int] = None,
        expires_in_days: Optional[int] = None
    ) -> Tuple[APIKey, str]:
        """
        Create a new API key for a user.
        
        Returns:
            Tuple of (APIKey object, raw key string)
        """
        # Generate secure API key
        raw_key, key_hash = APIKey.generate_key()
        key_prefix = raw_key[:8]
        
        # Set expiration if specified
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Create API key record
        api_key = APIKey(
            user_id=user_id,
            name=name,
            description=description,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=permissions or [],
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_hour=rate_limit_per_hour,
            rate_limit_per_day=rate_limit_per_day,
            quota_requests_per_month=quota_requests_per_month,
            expires_at=expires_at
        )
        
        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)
        
        return api_key, raw_key
    
    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return the APIKey object if valid.
        
        Args:
            raw_key: The raw API key string
            
        Returns:
            APIKey object if valid, None otherwise
        """
        if not raw_key or not raw_key.startswith('sk-'):
            return None
        
        key_hash = APIKey.hash_key(raw_key)
        
        api_key = self.db.query(APIKey).filter(
            and_(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True
            )
        ).first()
        
        if not api_key:
            return None
        
        # Check if expired
        if api_key.is_expired():
            return None
        
        # Update last used timestamp
        api_key.last_used = datetime.utcnow()
        self.db.commit()
        
        return api_key
    
    def check_rate_limit(self, api_key: APIKey) -> Dict[str, Any]:
        """
        Check if API key is within rate limits.
        
        Returns:
            Dict with rate limit status and remaining requests
        """
        now = datetime.utcnow()
        
        # Check minute limit
        minute_start = now.replace(second=0, microsecond=0)
        minute_record = self._get_or_create_rate_limit_record(
            api_key.id, minute_start, 60
        )
        
        # Check hour limit
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        hour_record = self._get_or_create_rate_limit_record(
            api_key.id, hour_start, 3600
        )
        
        # Check day limit
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        day_record = self._get_or_create_rate_limit_record(
            api_key.id, day_start, 86400
        )
        
        # Check limits
        minute_exceeded = minute_record.request_count >= api_key.rate_limit_per_minute
        hour_exceeded = hour_record.request_count >= api_key.rate_limit_per_hour
        day_exceeded = day_record.request_count >= api_key.rate_limit_per_day
        
        return {
            'allowed': not (minute_exceeded or hour_exceeded or day_exceeded),
            'minute': {
                'limit': api_key.rate_limit_per_minute,
                'used': minute_record.request_count,
                'remaining': max(0, api_key.rate_limit_per_minute - minute_record.request_count),
                'reset_at': minute_start + timedelta(minutes=1)
            },
            'hour': {
                'limit': api_key.rate_limit_per_hour,
                'used': hour_record.request_count,
                'remaining': max(0, api_key.rate_limit_per_hour - hour_record.request_count),
                'reset_at': hour_start + timedelta(hours=1)
            },
            'day': {
                'limit': api_key.rate_limit_per_day,
                'used': day_record.request_count,
                'remaining': max(0, api_key.rate_limit_per_day - day_record.request_count),
                'reset_at': day_start + timedelta(days=1)
            }
        }
    
    def record_api_usage(
        self,
        api_key: APIKey,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        request_size_bytes: Optional[int] = None,
        response_size_bytes: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> APIUsage:
        """
        Record API usage for tracking and billing.
        """
        # Create usage record
        usage = APIUsage(
            api_key_id=api_key.id,
            user_id=api_key.user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            ip_address=ip_address,
            user_agent=user_agent,
            request_metadata=request_metadata or {},
            error_message=error_message
        )
        
        self.db.add(usage)
        
        # Update rate limit records
        self._increment_rate_limit_counters(api_key.id)
        
        # Update quota if exists
        self._update_quota_usage(api_key, response_time_ms, 
                               request_size_bytes, response_size_bytes)
        
        self.db.commit()
        return usage
    
    def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        return self.db.query(APIKey).filter(
            APIKey.user_id == user_id
        ).order_by(APIKey.created_at.desc()).all()
    
    def get_api_key_by_id(self, api_key_id: str, user_id: str) -> Optional[APIKey]:
        """Get API key by ID for a specific user."""
        return self.db.query(APIKey).filter(
            and_(
                APIKey.id == api_key_id,
                APIKey.user_id == user_id
            )
        ).first()
    
    def update_api_key(
        self,
        api_key_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        rate_limit_per_minute: Optional[int] = None,
        rate_limit_per_hour: Optional[int] = None,
        rate_limit_per_day: Optional[int] = None,
        is_active: Optional[bool] = None
    ) -> Optional[APIKey]:
        """Update an existing API key."""
        api_key = self.get_api_key_by_id(api_key_id, user_id)
        if not api_key:
            return None
        
        if name is not None:
            api_key.name = name
        if description is not None:
            api_key.description = description
        if permissions is not None:
            api_key.permissions = permissions
        if rate_limit_per_minute is not None:
            api_key.rate_limit_per_minute = rate_limit_per_minute
        if rate_limit_per_hour is not None:
            api_key.rate_limit_per_hour = rate_limit_per_hour
        if rate_limit_per_day is not None:
            api_key.rate_limit_per_day = rate_limit_per_day
        if is_active is not None:
            api_key.is_active = is_active
        
        api_key.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(api_key)
        
        return api_key
    
    def delete_api_key(self, api_key_id: str, user_id: str) -> bool:
        """Delete an API key."""
        api_key = self.get_api_key_by_id(api_key_id, user_id)
        if not api_key:
            return False
        
        self.db.delete(api_key)
        self.db.commit()
        return True
    
    def get_usage_analytics(
        self,
        api_key_id: str,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get usage analytics for an API key."""
        api_key = self.get_api_key_by_id(api_key_id, user_id)
        if not api_key:
            return {}
        
        # Default to last 30 days
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Query usage data
        usage_query = self.db.query(APIUsage).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        )
        
        # Basic statistics
        total_requests = usage_query.count()
        successful_requests = usage_query.filter(
            APIUsage.status_code < 400
        ).count()
        
        # Average response time
        avg_response_time = self.db.query(
            func.avg(APIUsage.response_time_ms)
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        ).scalar() or 0
        
        # Top endpoints
        top_endpoints = self.db.query(
            APIUsage.endpoint,
            func.count(APIUsage.id).label('count')
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date
            )
        ).group_by(APIUsage.endpoint).order_by(
            func.count(APIUsage.id).desc()
        ).limit(10).all()
        
        # Error rate by status code
        error_stats = self.db.query(
            APIUsage.status_code,
            func.count(APIUsage.id).label('count')
        ).filter(
            and_(
                APIUsage.api_key_id == api_key_id,
                APIUsage.timestamp >= start_date,
                APIUsage.timestamp <= end_date,
                APIUsage.status_code >= 400
            )
        ).group_by(APIUsage.status_code).all()
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'error_rate': (total_requests - successful_requests) / max(total_requests, 1) * 100,
            'average_response_time_ms': round(avg_response_time, 2),
            'top_endpoints': [
                {'endpoint': endpoint, 'count': count}
                for endpoint, count in top_endpoints
            ],
            'error_breakdown': [
                {'status_code': status_code, 'count': count}
                for status_code, count in error_stats
            ]
        }
    
    def _get_or_create_rate_limit_record(
        self,
        api_key_id: str,
        window_start: datetime,
        window_duration_seconds: int
    ) -> RateLimitRecord:
        """Get or create a rate limit record for a specific window."""
        record = self.db.query(RateLimitRecord).filter(
            and_(
                RateLimitRecord.api_key_id == api_key_id,
                RateLimitRecord.window_start == window_start,
                RateLimitRecord.window_duration_seconds == window_duration_seconds
            )
        ).first()
        
        if not record:
            record = RateLimitRecord(
                api_key_id=api_key_id,
                window_start=window_start,
                window_duration_seconds=window_duration_seconds,
                first_request_at=datetime.utcnow(),
                last_request_at=datetime.utcnow()
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
        
        return record
    
    def _increment_rate_limit_counters(self, api_key_id: str):
        """Increment rate limit counters for all active windows."""
        now = datetime.utcnow()
        
        # Update minute counter
        minute_start = now.replace(second=0, microsecond=0)
        self._increment_counter(api_key_id, minute_start, 60)
        
        # Update hour counter
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        self._increment_counter(api_key_id, hour_start, 3600)
        
        # Update day counter
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self._increment_counter(api_key_id, day_start, 86400)
    
    def _increment_counter(
        self,
        api_key_id: str,
        window_start: datetime,
        window_duration_seconds: int
    ):
        """Increment counter for a specific rate limit window."""
        record = self._get_or_create_rate_limit_record(
            api_key_id, window_start, window_duration_seconds
        )
        record.request_count += 1
        record.last_request_at = datetime.utcnow()
    
    def _update_quota_usage(
        self,
        api_key: APIKey,
        response_time_ms: float,
        request_size_bytes: Optional[int],
        response_size_bytes: Optional[int]
    ):
        """Update quota usage for the current period."""
        # Get current month period
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate next month start
        if now.month == 12:
            period_end = period_start.replace(year=now.year + 1, month=1)
        else:
            period_end = period_start.replace(month=now.month + 1)
        
        # Get or create quota record
        quota = self.db.query(APIQuota).filter(
            and_(
                APIQuota.api_key_id == api_key.id,
                APIQuota.period_start == period_start
            )
        ).first()
        
        if not quota:
            quota = APIQuota(
                api_key_id=api_key.id,
                user_id=api_key.user_id,
                period_start=period_start,
                period_end=period_end,
                requests_limit=api_key.quota_requests_per_month
            )
            self.db.add(quota)
        
        # Update usage
        quota.requests_count += 1
        quota.compute_time_seconds += response_time_ms / 1000.0
        
        if request_size_bytes:
            quota.data_transfer_bytes += request_size_bytes
        if response_size_bytes:
            quota.data_transfer_bytes += response_size_bytes
        
        # Check if quota exceeded
        if not quota.is_exceeded and (
            quota.is_requests_exceeded() or
            quota.is_data_transfer_exceeded() or
            quota.is_compute_time_exceeded() or
            quota.is_cost_exceeded()
        ):
            quota.is_exceeded = True
            quota.exceeded_at = datetime.utcnow()