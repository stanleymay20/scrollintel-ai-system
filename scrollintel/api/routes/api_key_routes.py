"""
API Key Management Routes for ScrollIntel Launch MVP.
Provides REST endpoints for API key CRUD operations and usage analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ...core.api_key_manager import APIKeyManager
from ...models.database import get_db
from ...security.auth import get_current_user, get_current_active_user
from ...models.database import User


router = APIRouter(prefix="/api/v1/keys", tags=["API Keys"])
security = HTTPBearer()


# Pydantic models for request/response
class APIKeyCreate(BaseModel):
    """Request model for creating an API key."""
    name: str = Field(..., min_length=1, max_length=255, description="User-friendly name for the API key")
    description: Optional[str] = Field(None, max_length=1000, description="Optional description")
    permissions: Optional[List[str]] = Field(default=[], description="List of allowed permissions")
    rate_limit_per_minute: int = Field(60, ge=1, le=10000, description="Requests per minute limit")
    rate_limit_per_hour: int = Field(1000, ge=1, le=100000, description="Requests per hour limit")
    rate_limit_per_day: int = Field(10000, ge=1, le=1000000, description="Requests per day limit")
    quota_requests_per_month: Optional[int] = Field(None, ge=1, description="Monthly request quota")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")


class APIKeyUpdate(BaseModel):
    """Request model for updating an API key."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    permissions: Optional[List[str]] = Field(None)
    rate_limit_per_minute: Optional[int] = Field(None, ge=1, le=10000)
    rate_limit_per_hour: Optional[int] = Field(None, ge=1, le=100000)
    rate_limit_per_day: Optional[int] = Field(None, ge=1, le=1000000)
    is_active: Optional[bool] = Field(None)


class APIKeyResponse(BaseModel):
    """Response model for API key information."""
    id: str
    name: str
    description: Optional[str]
    key_display: str
    permissions: List[str]
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    rate_limit_per_day: int
    quota_requests_per_month: Optional[int]
    is_active: bool
    last_used: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class APIKeyCreateResponse(BaseModel):
    """Response model for API key creation."""
    api_key: APIKeyResponse
    key: str  # Only returned once during creation


class UsageAnalytics(BaseModel):
    """Response model for usage analytics."""
    total_requests: int
    successful_requests: int
    error_rate: float
    average_response_time_ms: float
    top_endpoints: List[Dict[str, Any]]
    error_breakdown: List[Dict[str, Any]]


class RateLimitStatus(BaseModel):
    """Response model for rate limit status."""
    allowed: bool
    minute: Dict[str, Any]
    hour: Dict[str, Any]
    day: Dict[str, Any]


@router.post("/", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new API key for the authenticated user.
    
    The API key will be returned only once during creation.
    Store it securely as it cannot be retrieved again.
    """
    try:
        manager = APIKeyManager(db)
        api_key, raw_key = manager.create_api_key(
            user_id=str(current_user.id),
            name=key_data.name,
            description=key_data.description,
            permissions=key_data.permissions,
            rate_limit_per_minute=key_data.rate_limit_per_minute,
            rate_limit_per_hour=key_data.rate_limit_per_hour,
            rate_limit_per_day=key_data.rate_limit_per_day,
            quota_requests_per_month=key_data.quota_requests_per_month,
            expires_in_days=key_data.expires_in_days
        )
        
        return APIKeyCreateResponse(
            api_key=APIKeyResponse(
                id=str(api_key.id),
                name=api_key.name,
                description=api_key.description,
                key_display=api_key.get_display_key(),
                permissions=api_key.permissions,
                rate_limit_per_minute=api_key.rate_limit_per_minute,
                rate_limit_per_hour=api_key.rate_limit_per_hour,
                rate_limit_per_day=api_key.rate_limit_per_day,
                quota_requests_per_month=api_key.quota_requests_per_month,
                is_active=api_key.is_active,
                last_used=api_key.last_used,
                expires_at=api_key.expires_at,
                created_at=api_key.created_at,
                updated_at=api_key.updated_at
            ),
            key=raw_key
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create API key: {str(e)}"
        )


@router.get("/", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List all API keys for the authenticated user.
    """
    manager = APIKeyManager(db)
    api_keys = manager.get_user_api_keys(str(current_user.id))
    
    return [
        APIKeyResponse(
            id=str(key.id),
            name=key.name,
            description=key.description,
            key_display=key.get_display_key(),
            permissions=key.permissions,
            rate_limit_per_minute=key.rate_limit_per_minute,
            rate_limit_per_hour=key.rate_limit_per_hour,
            rate_limit_per_day=key.rate_limit_per_day,
            quota_requests_per_month=key.quota_requests_per_month,
            is_active=key.is_active,
            last_used=key.last_used,
            expires_at=key.expires_at,
            created_at=key.created_at,
            updated_at=key.updated_at
        )
        for key in api_keys
    ]


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific API key.
    """
    manager = APIKeyManager(db)
    api_key = manager.get_api_key_by_id(key_id, str(current_user.id))
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    return APIKeyResponse(
        id=str(api_key.id),
        name=api_key.name,
        description=api_key.description,
        key_display=api_key.get_display_key(),
        permissions=api_key.permissions,
        rate_limit_per_minute=api_key.rate_limit_per_minute,
        rate_limit_per_hour=api_key.rate_limit_per_hour,
        rate_limit_per_day=api_key.rate_limit_per_day,
        quota_requests_per_month=api_key.quota_requests_per_month,
        is_active=api_key.is_active,
        last_used=api_key.last_used,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        updated_at=api_key.updated_at
    )


@router.put("/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: str,
    key_data: APIKeyUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing API key.
    """
    manager = APIKeyManager(db)
    api_key = manager.update_api_key(
        api_key_id=key_id,
        user_id=str(current_user.id),
        name=key_data.name,
        description=key_data.description,
        permissions=key_data.permissions,
        rate_limit_per_minute=key_data.rate_limit_per_minute,
        rate_limit_per_hour=key_data.rate_limit_per_hour,
        rate_limit_per_day=key_data.rate_limit_per_day,
        is_active=key_data.is_active
    )
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    return APIKeyResponse(
        id=str(api_key.id),
        name=api_key.name,
        description=api_key.description,
        key_display=api_key.get_display_key(),
        permissions=api_key.permissions,
        rate_limit_per_minute=api_key.rate_limit_per_minute,
        rate_limit_per_hour=api_key.rate_limit_per_hour,
        rate_limit_per_day=api_key.rate_limit_per_day,
        quota_requests_per_month=api_key.quota_requests_per_month,
        is_active=api_key.is_active,
        last_used=api_key.last_used,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        updated_at=api_key.updated_at
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete an API key.
    """
    manager = APIKeyManager(db)
    success = manager.delete_api_key(key_id, str(current_user.id))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )


@router.get("/{key_id}/usage", response_model=UsageAnalytics)
async def get_api_key_usage(
    key_id: str,
    start_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for analytics"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get usage analytics for an API key.
    """
    manager = APIKeyManager(db)
    analytics = manager.get_usage_analytics(
        api_key_id=key_id,
        user_id=str(current_user.id),
        start_date=start_date,
        end_date=end_date
    )
    
    if not analytics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    return UsageAnalytics(**analytics)


@router.get("/{key_id}/rate-limit", response_model=RateLimitStatus)
async def get_rate_limit_status(
    key_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get current rate limit status for an API key.
    """
    manager = APIKeyManager(db)
    api_key = manager.get_api_key_by_id(key_id, str(current_user.id))
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    rate_limit_status = manager.check_rate_limit(api_key)
    return RateLimitStatus(**rate_limit_status)


# API Key Authentication Dependency
async def get_api_key_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Authenticate user via API key.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    manager = APIKeyManager(db)
    api_key = manager.validate_api_key(credentials.credentials)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check rate limits
    rate_limit_status = manager.check_rate_limit(api_key)
    if not rate_limit_status['allowed']:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit-Minute": str(api_key.rate_limit_per_minute),
                "X-RateLimit-Remaining-Minute": str(rate_limit_status['minute']['remaining']),
                "X-RateLimit-Reset-Minute": rate_limit_status['minute']['reset_at'].isoformat(),
            }
        )
    
    # Get user
    user = db.query(User).filter(User.id == api_key.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account inactive"
        )
    
    return user