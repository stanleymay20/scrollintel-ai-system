"""
API Key Management Models for ScrollIntel Launch MVP.
Handles API key generation, management, and usage tracking.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import uuid
import secrets
import hashlib
from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, JSON, 
    ForeignKey, Float, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID
from .database import Base, JSONType


class APIKey(Base):
    """API Key model for secure API access management."""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)  # User-friendly name
    description = Column(Text, nullable=True)
    key_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 hash of the key
    key_prefix = Column(String(10), nullable=False)  # First 8 chars for display
    permissions = Column(JSONType, nullable=False, default=list)  # List of allowed endpoints/actions
    rate_limit_per_minute = Column(Integer, nullable=False, default=60)
    rate_limit_per_hour = Column(Integer, nullable=False, default=1000)
    rate_limit_per_day = Column(Integer, nullable=False, default=10000)
    quota_requests_per_month = Column(Integer, nullable=True)  # None = unlimited
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    usage_records = relationship("APIUsage", back_populates="api_key", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_key_user', 'user_id'),
        Index('idx_api_key_hash', 'key_hash'),
        Index('idx_api_key_active', 'is_active'),
        Index('idx_api_key_expires', 'expires_at'),
        Index('idx_api_key_last_used', 'last_used'),
    )
    
    @validates('permissions')
    def validate_permissions(self, key, permissions):
        """Validate permissions is a list."""
        if not isinstance(permissions, list):
            raise ValueError("Permissions must be a list")
        return permissions
    
    @validates('rate_limit_per_minute')
    def validate_rate_limit_per_minute(self, key, rate_limit):
        """Validate rate limit is positive."""
        if rate_limit <= 0:
            raise ValueError("Rate limit per minute must be positive")
        return rate_limit
    
    @validates('rate_limit_per_hour')
    def validate_rate_limit_per_hour(self, key, rate_limit):
        """Validate rate limit is positive."""
        if rate_limit <= 0:
            raise ValueError("Rate limit per hour must be positive")
        return rate_limit
    
    @validates('rate_limit_per_day')
    def validate_rate_limit_per_day(self, key, rate_limit):
        """Validate rate limit is positive."""
        if rate_limit <= 0:
            raise ValueError("Rate limit per day must be positive")
        return rate_limit
    
    @classmethod
    def generate_key(cls) -> tuple[str, str]:
        """Generate a new API key and its hash."""
        # Generate a secure random key
        key = f"sk-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash
    
    @classmethod
    def hash_key(cls, key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if the API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def get_display_key(self) -> str:
        """Get a display-safe version of the key."""
        return f"{self.key_prefix}...{self.key_hash[-4:]}"
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.name}, user_id={self.user_id}, active={self.is_active})>"


class APIUsage(Base):
    """API Usage tracking model for monitoring and billing."""
    
    __tablename__ = "api_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)  # GET, POST, PUT, DELETE
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=False)
    request_size_bytes = Column(Integer, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(String(500), nullable=True)
    request_metadata = Column(JSONType, nullable=False, default=dict)
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    api_key = relationship("APIKey", back_populates="usage_records")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_usage_key', 'api_key_id'),
        Index('idx_api_usage_user', 'user_id'),
        Index('idx_api_usage_endpoint', 'endpoint'),
        Index('idx_api_usage_timestamp', 'timestamp'),
        Index('idx_api_usage_status', 'status_code'),
        Index('idx_api_usage_method', 'method'),
    )
    
    @validates('request_metadata')
    def validate_request_metadata(self, key, request_metadata):
        """Validate request_metadata is a dict."""
        if not isinstance(request_metadata, dict):
            raise ValueError("Request metadata must be a dictionary")
        return request_metadata
    
    @validates('method')
    def validate_method(self, key, method):
        """Validate HTTP method."""
        allowed_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if method not in allowed_methods:
            raise ValueError(f"Method must be one of: {allowed_methods}")
        return method
    
    def __repr__(self):
        return f"<APIUsage(id={self.id}, endpoint={self.endpoint}, method={self.method}, status={self.status_code})>"


class APIQuota(Base):
    """API Quota tracking model for usage limits and billing."""
    
    __tablename__ = "api_quotas"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    requests_count = Column(Integer, nullable=False, default=0)
    requests_limit = Column(Integer, nullable=True)  # None = unlimited
    data_transfer_bytes = Column(Integer, nullable=False, default=0)
    data_transfer_limit_bytes = Column(Integer, nullable=True)  # None = unlimited
    compute_time_seconds = Column(Float, nullable=False, default=0.0)
    compute_time_limit_seconds = Column(Float, nullable=True)  # None = unlimited
    cost_usd = Column(Float, nullable=False, default=0.0)
    cost_limit_usd = Column(Float, nullable=True)  # None = unlimited
    is_exceeded = Column(Boolean, default=False, nullable=False)
    exceeded_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    api_key = relationship("APIKey")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_quota_key', 'api_key_id'),
        Index('idx_api_quota_user', 'user_id'),
        Index('idx_api_quota_period', 'period_start', 'period_end'),
        Index('idx_api_quota_exceeded', 'is_exceeded'),
        UniqueConstraint('api_key_id', 'period_start', name='uq_api_quota_key_period'),
    )
    
    def is_requests_exceeded(self) -> bool:
        """Check if request quota is exceeded."""
        if self.requests_limit is None:
            return False
        return self.requests_count >= self.requests_limit
    
    def is_data_transfer_exceeded(self) -> bool:
        """Check if data transfer quota is exceeded."""
        if self.data_transfer_limit_bytes is None:
            return False
        return self.data_transfer_bytes >= self.data_transfer_limit_bytes
    
    def is_compute_time_exceeded(self) -> bool:
        """Check if compute time quota is exceeded."""
        if self.compute_time_limit_seconds is None:
            return False
        return self.compute_time_seconds >= self.compute_time_limit_seconds
    
    def is_cost_exceeded(self) -> bool:
        """Check if cost quota is exceeded."""
        if self.cost_limit_usd is None:
            return False
        return self.cost_usd >= self.cost_limit_usd
    
    def get_usage_percentage(self, metric: str) -> float:
        """Get usage percentage for a specific metric."""
        if metric == 'requests':
            if self.requests_limit is None:
                return 0.0
            return (self.requests_count / self.requests_limit) * 100
        elif metric == 'data_transfer':
            if self.data_transfer_limit_bytes is None:
                return 0.0
            return (self.data_transfer_bytes / self.data_transfer_limit_bytes) * 100
        elif metric == 'compute_time':
            if self.compute_time_limit_seconds is None:
                return 0.0
            return (self.compute_time_seconds / self.compute_time_limit_seconds) * 100
        elif metric == 'cost':
            if self.cost_limit_usd is None:
                return 0.0
            return (self.cost_usd / self.cost_limit_usd) * 100
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def __repr__(self):
        return f"<APIQuota(id={self.id}, api_key_id={self.api_key_id}, requests={self.requests_count}/{self.requests_limit})>"


class RateLimitRecord(Base):
    """Rate limiting tracking model for API throttling."""
    
    __tablename__ = "rate_limit_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=False)
    window_start = Column(DateTime, nullable=False)
    window_duration_seconds = Column(Integer, nullable=False)  # 60, 3600, 86400 for minute/hour/day
    request_count = Column(Integer, nullable=False, default=0)
    limit_exceeded = Column(Boolean, default=False, nullable=False)
    first_request_at = Column(DateTime, nullable=False)
    last_request_at = Column(DateTime, nullable=False)
    
    # Relationships
    api_key = relationship("APIKey")
    
    # Indexes
    __table_args__ = (
        Index('idx_rate_limit_key', 'api_key_id'),
        Index('idx_rate_limit_window', 'window_start', 'window_duration_seconds'),
        Index('idx_rate_limit_exceeded', 'limit_exceeded'),
        UniqueConstraint('api_key_id', 'window_start', 'window_duration_seconds', 
                        name='uq_rate_limit_key_window'),
    )
    
    def is_window_active(self) -> bool:
        """Check if the rate limit window is still active."""
        window_end = self.window_start + timedelta(seconds=self.window_duration_seconds)
        return datetime.utcnow() < window_end
    
    def __repr__(self):
        return f"<RateLimitRecord(id={self.id}, api_key_id={self.api_key_id}, count={self.request_count})>"