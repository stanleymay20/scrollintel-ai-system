"""API routes for usage tracking functionality."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import logging

from ...engines.usage_tracker import UsageTracker, UsageTrackerError
from ...engines.audit_logger import AuditLogger
from ...models.governance_models import UsageMetrics
from ..middleware.auth import get_current_user
from ..models.responses import APIResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/usage", tags=["usage-tracking"])


# Request/Response Models
class UsageAnalyticsRequest(BaseModel):
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    user_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    aggregation_level: str = Field(default="daily", regex="^(hourly|daily|weekly|monthly)$")


class UserActivityRequest(BaseModel):
    user_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_details: bool = False


class ResourceUsageRequest(BaseModel):
    resource_id: str
    resource_type: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_user_breakdown: bool = True


class UsageTrendsRequest(BaseModel):
    metric_type: str = Field(default="access_count", regex="^(access_count|unique_users|data_volume)$")
    resource_type: Optional[str] = None
    time_period: str = Field(default="30d", regex="^(7d|30d|90d|1y)$")
    granularity: str = Field(default="daily", regex="^(hourly|daily|weekly)$")


class TrackAccessRequest(BaseModel):
    user_id: str
    resource_id: str
    resource_type: str
    action: str = "read"
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


# Initialize services
usage_tracker = UsageTracker()
audit_logger = AuditLogger()


@router.post("/track-access")
async def track_data_access(
    request: TrackAccessRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Track data access event."""
    try:
        # Log the access event first
        audit_logger.log_data_access(
            user_id=request.user_id,
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            action=request.action,
            details=request.metadata,
            ip_address=request.ip_address,
            user_agent=request.user_agent,
            session_id=request.session_id
        )
        
        # Track usage metrics
        success = usage_tracker.track_data_access(
            user_id=request.user_id,
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            action=request.action,
            metadata=request.metadata,
            session_id=request.session_id,
            ip_address=request.ip_address,
            user_agent=request.user_agent
        )
        
        return APIResponse(
            success=success,
            message="Data access tracked successfully" if success else "Failed to track data access",
            data={"tracked": success}
        )
        
    except Exception as e:
        logger.error(f"Error tracking data access: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track data access: {str(e)}")


@router.post("/analytics")
async def get_usage_analytics(
    request: UsageAnalyticsRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get comprehensive usage analytics."""
    try:
        analytics = usage_tracker.get_usage_analytics(
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date,
            aggregation_level=request.aggregation_level
        )
        
        return APIResponse(
            success=True,
            message="Usage analytics retrieved successfully",
            data=analytics
        )
        
    except UsageTrackerError as e:
        logger.error(f"Usage tracker error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting usage analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage analytics: {str(e)}")


@router.post("/user-activity")
async def get_user_activity_report(
    request: UserActivityRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get detailed user activity report."""
    try:
        report = usage_tracker.get_user_activity_report(
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date,
            include_details=request.include_details
        )
        
        return APIResponse(
            success=True,
            message="User activity report retrieved successfully",
            data=report
        )
        
    except UsageTrackerError as e:
        logger.error(f"Usage tracker error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting user activity report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user activity report: {str(e)}")


@router.post("/resource-usage")
async def get_resource_usage_report(
    request: ResourceUsageRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get detailed resource usage report."""
    try:
        report = usage_tracker.get_resource_usage_report(
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            start_date=request.start_date,
            end_date=request.end_date,
            include_user_breakdown=request.include_user_breakdown
        )
        
        return APIResponse(
            success=True,
            message="Resource usage report retrieved successfully",
            data=report
        )
        
    except UsageTrackerError as e:
        logger.error(f"Usage tracker error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting resource usage report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource usage report: {str(e)}")


@router.get("/system-overview")
async def get_system_usage_overview(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get system-wide usage overview."""
    try:
        overview = usage_tracker.get_system_usage_overview(
            start_date=start_date,
            end_date=end_date
        )
        
        return APIResponse(
            success=True,
            message="System usage overview retrieved successfully",
            data=overview
        )
        
    except UsageTrackerError as e:
        logger.error(f"Usage tracker error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting system usage overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system usage overview: {str(e)}")


@router.post("/trends")
async def generate_usage_trends(
    request: UsageTrendsRequest,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Generate usage trends for visualization."""
    try:
        trends = usage_tracker.generate_usage_trends(
            metric_type=request.metric_type,
            resource_type=request.resource_type,
            time_period=request.time_period,
            granularity=request.granularity
        )
        
        return APIResponse(
            success=True,
            message="Usage trends generated successfully",
            data=trends
        )
        
    except UsageTrackerError as e:
        logger.error(f"Usage tracker error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating usage trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate usage trends: {str(e)}")


@router.get("/audit-trail")
async def get_audit_trail(
    resource_id: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(1000, ge=1, le=10000),
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get audit trail with filters."""
    try:
        from ...models.governance_models import AuditEventType
        
        # Convert event_type string to enum if provided
        event_type_enum = None
        if event_type:
            try:
                event_type_enum = AuditEventType(event_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
        audit_trail = audit_logger.get_audit_trail(
            resource_id=resource_id,
            resource_type=resource_type,
            user_id=user_id,
            event_type=event_type_enum,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        # Convert to serializable format
        trail_data = [
            {
                'id': event.id,
                'event_type': event.event_type.value,
                'user_id': event.user_id,
                'resource_id': event.resource_id,
                'resource_type': event.resource_type,
                'action': event.action,
                'details': event.details,
                'ip_address': event.ip_address,
                'user_agent': event.user_agent,
                'session_id': event.session_id,
                'timestamp': event.timestamp,
                'success': event.success,
                'error_message': event.error_message
            }
            for event in audit_trail
        ]
        
        return APIResponse(
            success=True,
            message="Audit trail retrieved successfully",
            data={
                'events': trail_data,
                'total_events': len(trail_data),
                'filters': {
                    'resource_id': resource_id,
                    'resource_type': resource_type,
                    'user_id': user_id,
                    'event_type': event_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'limit': limit
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audit trail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get audit trail: {str(e)}")


@router.get("/user/{user_id}/summary")
async def get_user_activity_summary(
    user_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get user activity summary."""
    try:
        summary = audit_logger.get_user_activity_summary(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return APIResponse(
            success=True,
            message="User activity summary retrieved successfully",
            data=summary
        )
        
    except Exception as e:
        logger.error(f"Error getting user activity summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user activity summary: {str(e)}")


@router.get("/resource/{resource_id}/summary")
async def get_resource_access_summary(
    resource_id: str,
    resource_type: str = Query(...),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Get resource access summary."""
    try:
        summary = audit_logger.get_resource_access_summary(
            resource_id=resource_id,
            resource_type=resource_type,
            start_date=start_date,
            end_date=end_date
        )
        
        return APIResponse(
            success=True,
            message="Resource access summary retrieved successfully",
            data=summary
        )
        
    except Exception as e:
        logger.error(f"Error getting resource access summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource access summary: {str(e)}")


@router.post("/metrics/update")
async def update_usage_metrics(
    resource_id: str,
    resource_type: str,
    period_start: datetime,
    period_end: datetime,
    current_user: dict = Depends(get_current_user)
) -> APIResponse:
    """Update usage metrics for a resource."""
    try:
        metrics = audit_logger.update_usage_metrics(
            resource_id=resource_id,
            resource_type=resource_type,
            period_start=period_start,
            period_end=period_end
        )
        
        # Convert to serializable format
        metrics_data = {
            'resource_id': metrics.resource_id,
            'resource_type': metrics.resource_type,
            'access_count': metrics.access_count,
            'unique_users': metrics.unique_users,
            'last_accessed': metrics.last_accessed,
            'most_frequent_user': metrics.most_frequent_user,
            'access_patterns': metrics.access_patterns,
            'performance_metrics': metrics.performance_metrics,
            'period_start': metrics.period_start,
            'period_end': metrics.period_end
        }
        
        return APIResponse(
            success=True,
            message="Usage metrics updated successfully",
            data=metrics_data
        )
        
    except Exception as e:
        logger.error(f"Error updating usage metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update usage metrics: {str(e)}")


@router.get("/health")
async def health_check() -> APIResponse:
    """Health check endpoint for usage tracking service."""
    try:
        # Test basic functionality
        test_analytics = usage_tracker.get_usage_analytics(
            start_date=datetime.utcnow() - timedelta(days=1),
            end_date=datetime.utcnow()
        )
        
        return APIResponse(
            success=True,
            message="Usage tracking service is healthy",
            data={
                'service': 'usage_tracking',
                'status': 'healthy',
                'timestamp': datetime.utcnow()
            }
        )
        
    except Exception as e:
        logger.error(f"Usage tracking health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Usage tracking service unhealthy: {str(e)}")