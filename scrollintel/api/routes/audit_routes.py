"""
Audit Log API Routes for ScrollIntel Launch MVP

Provides comprehensive audit log viewing, searching, and compliance reporting endpoints.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from uuid import UUID

from ...core.audit_system import audit_system, AuditAction, ComplianceLevel, RetentionPolicy
from ...security.auth import get_current_user
from ...models.database import User
from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/audit", tags=["audit"])


class AuditLogSearchRequest(BaseModel):
    """Audit log search parameters"""
    user_id: Optional[str] = None
    action: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    success: Optional[bool] = None
    ip_address: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class AuditLogResponse(BaseModel):
    """Audit log response model"""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    user_email: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    success: bool
    error_message: Optional[str]


class AuditLogSearchResponse(BaseModel):
    """Audit log search response"""
    logs: List[AuditLogResponse]
    total_count: int
    page_info: Dict[str, Any]


class ComplianceReportRequest(BaseModel):
    """Compliance report generation request"""
    start_date: datetime
    end_date: datetime
    report_type: str = "comprehensive"


class ComplianceReportResponse(BaseModel):
    """Compliance report response"""
    id: str
    report_type: str
    generated_at: datetime
    date_range_start: datetime
    date_range_end: datetime
    total_events: int
    successful_events: int
    failed_events: int
    security_events: int
    compliance_violations: List[Dict[str, Any]]
    user_activity_summary: Dict[str, Any]
    resource_access_summary: Dict[str, Any]
    recommendations: List[str]


class AuditExportRequest(BaseModel):
    """Audit log export request"""
    start_date: datetime
    end_date: datetime
    format: str = Field(default="json", pattern="^(json|csv)$")
    user_id: Optional[str] = None
    action: Optional[str] = None
    resource_type: Optional[str] = None


@router.get("/logs/search", response_model=AuditLogSearchResponse)
async def search_audit_logs(
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    success: Optional[bool] = Query(None),
    ip_address: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """Search audit logs with comprehensive filtering"""
    
    # Check permissions
    await require_permission(current_user, "audit.logs.view")
    
    try:
        # Log the search request
        await audit_system.log_user_action(
            user_id=str(current_user.id),
            user_email=current_user.email,
            action=AuditAction.AUDIT_LOG_EXPORT,
            resource_type="audit_logs",
            details={
                "search_filters": {
                    "user_id": user_id,
                    "action": action,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "success": success,
                    "ip_address": ip_address
                },
                "pagination": {"limit": limit, "offset": offset}
            }
        )
        
        # Search audit logs
        logs = await audit_system.search_audit_logs(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            start_date=start_date,
            end_date=end_date,
            success=success,
            ip_address=ip_address,
            limit=limit,
            offset=offset
        )
        
        # Convert to response format
        audit_logs = [
            AuditLogResponse(
                id=log["id"],
                timestamp=datetime.fromisoformat(log["timestamp"]),
                user_id=log["user_id"],
                user_email=log["user_email"],
                action=log["action"],
                resource_type=log["resource_type"],
                resource_id=log["resource_id"],
                details=log["details"],
                ip_address=log["ip_address"],
                user_agent=log["user_agent"],
                session_id=log["session_id"],
                success=log["success"],
                error_message=log["error_message"]
            )
            for log in logs
        ]
        
        return AuditLogSearchResponse(
            logs=audit_logs,
            total_count=len(audit_logs),
            page_info={
                "limit": limit,
                "offset": offset,
                "has_more": len(audit_logs) == limit
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching audit logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to search audit logs")


@router.get("/logs/{log_id}", response_model=AuditLogResponse)
async def get_audit_log(
    log_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get specific audit log by ID"""
    
    await require_permission(current_user, "audit.logs.view")
    
    try:
        logs = await audit_system.search_audit_logs(limit=1)
        
        # Find the specific log (simplified - in practice you'd query by ID)
        log = next((l for l in logs if l["id"] == log_id), None)
        
        if not log:
            raise HTTPException(status_code=404, detail="Audit log not found")
        
        return AuditLogResponse(
            id=log["id"],
            timestamp=datetime.fromisoformat(log["timestamp"]),
            user_id=log["user_id"],
            user_email=log["user_email"],
            action=log["action"],
            resource_type=log["resource_type"],
            resource_id=log["resource_id"],
            details=log["details"],
            ip_address=log["ip_address"],
            user_agent=log["user_agent"],
            session_id=log["session_id"],
            success=log["success"],
            error_message=log["error_message"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audit log {log_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit log")


@router.get("/user/{user_id}/activity", response_model=List[AuditLogResponse])
async def get_user_activity(
    user_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user)
):
    """Get recent activity for a specific user"""
    
    await require_permission(current_user, "audit.logs.view")
    
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        logs = await audit_system.search_audit_logs(
            user_id=user_id,
            start_date=start_date,
            limit=1000
        )
        
        return [
            AuditLogResponse(
                id=log["id"],
                timestamp=datetime.fromisoformat(log["timestamp"]),
                user_id=log["user_id"],
                user_email=log["user_email"],
                action=log["action"],
                resource_type=log["resource_type"],
                resource_id=log["resource_id"],
                details=log["details"],
                ip_address=log["ip_address"],
                user_agent=log["user_agent"],
                session_id=log["session_id"],
                success=log["success"],
                error_message=log["error_message"]
            )
            for log in logs
        ]
        
    except Exception as e:
        logger.error(f"Error retrieving user activity for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user activity")


@router.post("/compliance/report", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate comprehensive compliance report"""
    
    await require_permission(current_user, "compliance.reports.generate")
    
    try:
        # Validate date range
        if request.end_date <= request.start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        if (request.end_date - request.start_date).days > 365:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 365 days")
        
        # Log the report generation
        await audit_system.log_user_action(
            user_id=str(current_user.id),
            user_email=current_user.email,
            action=AuditAction.COMPLIANCE_REPORT_GENERATE,
            resource_type="compliance_report",
            details={
                "report_type": request.report_type,
                "date_range_start": request.start_date.isoformat(),
                "date_range_end": request.end_date.isoformat()
            }
        )
        
        # Generate compliance report
        report = await audit_system.generate_compliance_report(
            start_date=request.start_date,
            end_date=request.end_date,
            report_type=request.report_type
        )
        
        return ComplianceReportResponse(
            id=report.id,
            report_type=report.report_type,
            generated_at=report.generated_at,
            date_range_start=report.date_range_start,
            date_range_end=report.date_range_end,
            total_events=report.total_events,
            successful_events=report.successful_events,
            failed_events=report.failed_events,
            security_events=report.security_events,
            compliance_violations=report.compliance_violations,
            user_activity_summary=report.user_activity_summary,
            resource_access_summary=report.resource_access_summary,
            recommendations=report.recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")


@router.post("/export")
async def export_audit_logs(
    request: AuditExportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Export audit logs for compliance purposes"""
    
    await require_permission(current_user, "audit.logs.export")
    
    try:
        # Validate date range
        if request.end_date <= request.start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        if (request.end_date - request.start_date).days > 90:
            raise HTTPException(status_code=400, detail="Export date range cannot exceed 90 days")
        
        # Export audit logs
        file_path = await audit_system.export_audit_logs(
            start_date=request.start_date,
            end_date=request.end_date,
            format=request.format,
            user_id=request.user_id,
            action=request.action,
            resource_type=request.resource_type
        )
        
        # Log the export
        await audit_system.log_user_action(
            user_id=str(current_user.id),
            user_email=current_user.email,
            action=AuditAction.AUDIT_LOG_EXPORT,
            resource_type="audit_logs",
            details={
                "export_format": request.format,
                "date_range_start": request.start_date.isoformat(),
                "date_range_end": request.end_date.isoformat(),
                "filters": {
                    "user_id": request.user_id,
                    "action": request.action,
                    "resource_type": request.resource_type
                },
                "file_path": file_path
            }
        )
        
        return FileResponse(
            path=file_path,
            filename=f"audit_logs_export.{request.format}",
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting audit logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to export audit logs")


@router.get("/statistics")
async def get_audit_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user)
):
    """Get audit log statistics and metrics"""
    
    await require_permission(current_user, "audit.logs.view")
    
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        end_date = datetime.utcnow()
        
        # Generate a mini compliance report for statistics
        report = await audit_system.generate_compliance_report(
            start_date=start_date,
            end_date=end_date,
            report_type="statistics"
        )
        
        return {
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "event_summary": {
                "total_events": report.total_events,
                "successful_events": report.successful_events,
                "failed_events": report.failed_events,
                "success_rate": (report.successful_events / report.total_events * 100) if report.total_events > 0 else 0
            },
            "security_summary": {
                "security_events": report.security_events,
                "compliance_violations": len(report.compliance_violations),
                "violation_rate": (len(report.compliance_violations) / report.total_events * 100) if report.total_events > 0 else 0
            },
            "activity_summary": {
                "active_users": len(report.user_activity_summary),
                "resource_types_accessed": len(report.resource_access_summary),
                "top_resources": dict(sorted(report.resource_access_summary.items(), key=lambda x: x[1], reverse=True)[:5])
            },
            "recommendations": report.recommendations
        }
        
    except Exception as e:
        logger.error(f"Error retrieving audit statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit statistics")


@router.post("/retention/cleanup")
async def cleanup_expired_logs(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Clean up expired audit logs based on retention policies"""
    
    await require_permission(current_user, "audit.logs.manage")
    
    try:
        # Run cleanup in background
        background_tasks.add_task(audit_system.cleanup_expired_logs)
        
        # Log the cleanup request
        await audit_system.log_user_action(
            user_id=str(current_user.id),
            user_email=current_user.email,
            action=AuditAction.DATA_RETENTION_CLEANUP,
            resource_type="audit_logs",
            details={"cleanup_requested_at": datetime.utcnow().isoformat()}
        )
        
        return {"message": "Audit log cleanup initiated", "status": "processing"}
        
    except Exception as e:
        logger.error(f"Error initiating audit log cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate cleanup")


@router.get("/actions")
async def get_available_actions(
    current_user: User = Depends(get_current_user)
):
    """Get list of available audit actions for filtering"""
    
    await require_permission(current_user, "audit.logs.view")
    
    return {
        "actions": [action.value for action in AuditAction],
        "categories": {
            "authentication": [a.value for a in AuditAction if a.value.startswith("auth.")],
            "user_management": [a.value for a in AuditAction if a.value.startswith("user.")],
            "data_operations": [a.value for a in AuditAction if a.value.startswith("data.")],
            "system_operations": [a.value for a in AuditAction if a.value.startswith("system.")],
            "compliance": [a.value for a in AuditAction if a.value.startswith("compliance.")]
        }
    }