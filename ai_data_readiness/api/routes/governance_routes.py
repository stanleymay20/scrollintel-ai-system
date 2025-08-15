"""API routes for data governance functionality."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

from ...engines.data_catalog import DataCatalog
from ...engines.policy_engine import PolicyEngine
from ...engines.audit_logger import AuditLogger
from ...engines.usage_tracker import UsageTracker
from ...engines.compliance_reporter import ComplianceReporter
from ...models.governance_models import (
    DataClassification, PolicyType, AccessLevel, AuditEventType
)


router = APIRouter(prefix="/governance", tags=["governance"])


# Request/Response models
class DataCatalogEntryRequest(BaseModel):
    name: str
    description: str
    owner: str
    classification: DataClassification = DataClassification.INTERNAL
    steward: Optional[str] = None
    tags: Optional[List[str]] = None
    business_terms: Optional[List[str]] = None


class DataCatalogEntryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    classification: Optional[DataClassification] = None
    steward: Optional[str] = None
    tags: Optional[List[str]] = None
    business_terms: Optional[List[str]] = None
    retention_policy: Optional[str] = None
    compliance_requirements: Optional[List[str]] = None


class PolicyRequest(BaseModel):
    name: str
    description: str
    policy_type: PolicyType
    rules: List[Dict[str, Any]]
    conditions: Optional[Dict[str, Any]] = None
    enforcement_level: str = "strict"
    applicable_resources: Optional[List[str]] = None


class AccessGrantRequest(BaseModel):
    user_id: str
    resource_id: str
    resource_type: str
    access_level: AccessLevel
    expires_at: Optional[datetime] = None
    conditions: Optional[Dict[str, Any]] = None


class AuditEventRequest(BaseModel):
    event_type: AuditEventType
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    action: str
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


# Dependency injection
def get_data_catalog() -> DataCatalog:
    return DataCatalog()


def get_policy_engine() -> PolicyEngine:
    return PolicyEngine()


def get_audit_logger() -> AuditLogger:
    return AuditLogger()


def get_usage_tracker() -> UsageTracker:
    return UsageTracker()


def get_compliance_reporter() -> ComplianceReporter:
    return ComplianceReporter()


# Data Catalog endpoints
@router.post("/catalog/register/{dataset_id}")
async def register_dataset_in_catalog(
    dataset_id: str,
    request: DataCatalogEntryRequest,
    catalog: DataCatalog = Depends(get_data_catalog)
):
    """Register a dataset in the data catalog."""
    try:
        entry = catalog.register_dataset(
            dataset_id=dataset_id,
            name=request.name,
            description=request.description,
            owner=request.owner,
            classification=request.classification,
            steward=request.steward,
            tags=request.tags,
            business_terms=request.business_terms
        )
        return entry
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/catalog/{dataset_id}")
async def get_catalog_entry(
    dataset_id: str,
    catalog: DataCatalog = Depends(get_data_catalog)
):
    """Get a catalog entry by dataset ID."""
    try:
        entry = catalog.get_catalog_entry(dataset_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Catalog entry not found")
        return entry
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/catalog/{dataset_id}")
async def update_catalog_entry(
    dataset_id: str,
    request: DataCatalogEntryUpdate,
    catalog: DataCatalog = Depends(get_data_catalog)
):
    """Update a catalog entry."""
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}
        entry = catalog.update_catalog_entry(dataset_id, updates)
        return entry
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/catalog/search")
async def search_catalog(
    query: Optional[str] = Query(None),
    classification: Optional[DataClassification] = Query(None),
    tags: Optional[List[str]] = Query(None),
    owner: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    catalog: DataCatalog = Depends(get_data_catalog)
):
    """Search the data catalog."""
    try:
        entries = catalog.search_catalog(
            query=query,
            classification=classification,
            tags=tags,
            owner=owner,
            limit=limit
        )
        return entries
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/catalog/{dataset_id}/quality-metrics")
async def update_quality_metrics(
    dataset_id: str,
    quality_metrics: Dict[str, float],
    catalog: DataCatalog = Depends(get_data_catalog)
):
    """Update quality metrics for a catalog entry."""
    try:
        catalog.update_quality_metrics(dataset_id, quality_metrics)
        return {"message": "Quality metrics updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/catalog/metrics")
async def get_governance_metrics(
    catalog: DataCatalog = Depends(get_data_catalog)
):
    """Get governance metrics."""
    try:
        metrics = catalog.get_governance_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Policy Management endpoints
@router.post("/policies")
async def create_policy(
    request: PolicyRequest,
    created_by: str = Query(..., description="User ID creating the policy"),
    policy_engine: PolicyEngine = Depends(get_policy_engine)
):
    """Create a new governance policy."""
    try:
        policy = policy_engine.create_policy(
            name=request.name,
            description=request.description,
            policy_type=request.policy_type,
            rules=request.rules,
            created_by=created_by,
            conditions=request.conditions,
            enforcement_level=request.enforcement_level,
            applicable_resources=request.applicable_resources
        )
        return policy
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/policies/{policy_id}/activate")
async def activate_policy(
    policy_id: str,
    approved_by: str = Query(..., description="User ID approving the policy"),
    policy_engine: PolicyEngine = Depends(get_policy_engine)
):
    """Activate a governance policy."""
    try:
        policy = policy_engine.activate_policy(policy_id, approved_by)
        return policy
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/policies/enforce")
async def enforce_policy(
    user_id: str = Query(...),
    resource_id: str = Query(...),
    resource_type: str = Query(...),
    action: str = Query(...),
    context: Optional[Dict[str, Any]] = None,
    policy_engine: PolicyEngine = Depends(get_policy_engine)
):
    """Enforce policies for a user action."""
    try:
        allowed, violations = policy_engine.enforce_policy(
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            context=context
        )
        return {
            "allowed": allowed,
            "violations": violations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Access Control endpoints
@router.post("/access/grant")
async def grant_access(
    request: AccessGrantRequest,
    granted_by: str = Query(..., description="User ID granting access"),
    policy_engine: PolicyEngine = Depends(get_policy_engine)
):
    """Grant access to a resource."""
    try:
        access_entry = policy_engine.grant_access(
            user_id=request.user_id,
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            access_level=request.access_level,
            granted_by=granted_by,
            expires_at=request.expires_at,
            conditions=request.conditions
        )
        return access_entry
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/access/revoke")
async def revoke_access(
    user_id: str = Query(...),
    resource_id: str = Query(...),
    resource_type: str = Query(...),
    revoked_by: str = Query(..., description="User ID revoking access"),
    policy_engine: PolicyEngine = Depends(get_policy_engine)
):
    """Revoke access to a resource."""
    try:
        success = policy_engine.revoke_access(
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            revoked_by=revoked_by
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/access/permissions/{user_id}")
async def get_user_permissions(
    user_id: str,
    resource_type: Optional[str] = Query(None),
    policy_engine: PolicyEngine = Depends(get_policy_engine)
):
    """Get all permissions for a user."""
    try:
        permissions = policy_engine.get_user_permissions(
            user_id=user_id,
            resource_type=resource_type
        )
        return permissions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/access/check")
async def check_access_permission(
    user_id: str = Query(...),
    resource_id: str = Query(...),
    resource_type: str = Query(...),
    required_access: AccessLevel = Query(...),
    policy_engine: PolicyEngine = Depends(get_policy_engine)
):
    """Check if user has required access permission."""
    try:
        has_access = policy_engine.check_access_permission(
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            required_access=required_access
        )
        return {"has_access": has_access}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Audit endpoints
@router.post("/audit/log")
async def log_audit_event(
    request: AuditEventRequest,
    user_id: str = Query(..., description="User ID performing the action"),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Log an audit event."""
    try:
        if request.event_type == AuditEventType.DATA_ACCESS:
            event = audit_logger.log_data_access(
                user_id=user_id,
                resource_id=request.resource_id,
                resource_type=request.resource_type,
                action=request.action,
                details=request.details,
                ip_address=request.ip_address,
                user_agent=request.user_agent,
                session_id=request.session_id
            )
        elif request.event_type == AuditEventType.DATA_MODIFICATION:
            event = audit_logger.log_data_modification(
                user_id=user_id,
                resource_id=request.resource_id,
                resource_type=request.resource_type,
                action=request.action,
                details=request.details,
                ip_address=request.ip_address,
                user_agent=request.user_agent,
                session_id=request.session_id
            )
        else:
            event = audit_logger.log_user_action(
                user_id=user_id,
                action=request.action,
                details=request.details,
                resource_id=request.resource_id,
                resource_type=request.resource_type,
                ip_address=request.ip_address,
                user_agent=request.user_agent,
                session_id=request.session_id
            )
        
        return event
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/audit/trail")
async def get_audit_trail(
    resource_id: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    event_type: Optional[AuditEventType] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(1000, le=10000),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Get audit trail with filters."""
    try:
        events = audit_logger.get_audit_trail(
            resource_id=resource_id,
            resource_type=resource_type,
            user_id=user_id,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        return events
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/audit/user-activity/{user_id}")
async def get_user_activity_summary(
    user_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Get user activity summary."""
    try:
        summary = audit_logger.get_user_activity_summary(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/audit/resource-access/{resource_id}")
async def get_resource_access_summary(
    resource_id: str,
    resource_type: str = Query(...),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Get resource access summary."""
    try:
        summary = audit_logger.get_resource_access_summary(
            resource_id=resource_id,
            resource_type=resource_type,
            start_date=start_date,
            end_date=end_date
        )
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/audit/usage-metrics/{resource_id}")
async def update_usage_metrics(
    resource_id: str,
    resource_type: str = Query(...),
    period_start: datetime = Query(...),
    period_end: datetime = Query(...),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Update usage metrics for a resource."""
    try:
        metrics = audit_logger.update_usage_metrics(
            resource_id=resource_id,
            resource_type=resource_type,
            period_start=period_start,
            period_end=period_end
        )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Us
age Tracking endpoints
@router.post("/usage/track")
async def track_data_access(
    user_id: str = Query(...),
    resource_id: str = Query(...),
    resource_type: str = Query(...),
    action: str = Query(...),
    details: Optional[Dict[str, Any]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    usage_tracker: UsageTracker = Depends(get_usage_tracker)
):
    """Track data access event."""
    try:
        usage_tracker.track_data_access(
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            details=details,
            session_context=session_context
        )
        return {"message": "Data access tracked successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/usage/report")
async def generate_usage_report(
    resource_id: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    report_type: str = Query("summary", regex="^(summary|detailed|trends)$"),
    usage_tracker: UsageTracker = Depends(get_usage_tracker)
):
    """Generate usage report."""
    try:
        report = usage_tracker.generate_usage_report(
            resource_id=resource_id,
            resource_type=resource_type,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            report_type=report_type
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/usage/patterns/{user_id}")
async def get_user_access_patterns(
    user_id: str,
    days_back: int = Query(30, ge=1, le=365),
    usage_tracker: UsageTracker = Depends(get_usage_tracker)
):
    """Get user access patterns analysis."""
    try:
        patterns = usage_tracker.get_user_access_patterns(
            user_id=user_id,
            days_back=days_back
        )
        return patterns
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/usage/popularity")
async def get_resource_popularity_metrics(
    resource_type: Optional[str] = Query(None),
    days_back: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100),
    usage_tracker: UsageTracker = Depends(get_usage_tracker)
):
    """Get resource popularity metrics."""
    try:
        metrics = usage_tracker.get_resource_popularity_metrics(
            resource_type=resource_type,
            days_back=days_back,
            limit=limit
        )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/usage/anomalies")
async def detect_anomalous_usage(
    user_id: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    days_back: int = Query(30, ge=1, le=365),
    sensitivity: float = Query(2.0, ge=1.0, le=5.0),
    usage_tracker: UsageTracker = Depends(get_usage_tracker)
):
    """Detect anomalous usage patterns."""
    try:
        anomalies = usage_tracker.detect_anomalous_usage(
            user_id=user_id,
            resource_id=resource_id,
            days_back=days_back,
            sensitivity=sensitivity
        )
        return anomalies
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Compliance Reporting endpoints
@router.post("/compliance/report/{compliance_type}")
async def generate_compliance_report(
    compliance_type: str,
    scope: Optional[List[str]] = None,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    generated_by: str = Query(..., description="User ID generating the report"),
    compliance_reporter: ComplianceReporter = Depends(get_compliance_reporter)
):
    """Generate compliance report."""
    try:
        report = compliance_reporter.generate_compliance_report(
            compliance_type=compliance_type,
            scope=scope,
            start_date=start_date,
            end_date=end_date,
            generated_by=generated_by
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/compliance/dashboard")
async def get_compliance_dashboard(
    compliance_types: Optional[List[str]] = Query(None),
    compliance_reporter: ComplianceReporter = Depends(get_compliance_reporter)
):
    """Get compliance dashboard."""
    try:
        dashboard = compliance_reporter.get_compliance_dashboard(
            compliance_types=compliance_types
        )
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/compliance/violations/{compliance_type}")
async def get_violation_details(
    compliance_type: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    compliance_reporter: ComplianceReporter = Depends(get_compliance_reporter)
):
    """Get detailed violation information."""
    try:
        violations = compliance_reporter.get_violation_details(
            compliance_type=compliance_type,
            start_date=start_date,
            end_date=end_date
        )
        return violations
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compliance/remediation/{compliance_type}")
async def generate_remediation_plan(
    compliance_type: str,
    violations: List[Dict[str, Any]],
    compliance_reporter: ComplianceReporter = Depends(get_compliance_reporter)
):
    """Generate remediation plan for compliance violations."""
    try:
        plan = compliance_reporter.generate_remediation_plan(
            compliance_type=compliance_type,
            violations=violations
        )
        return plan
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/compliance/usage-report/{compliance_type}")
async def generate_compliance_usage_report(
    compliance_type: str,
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    usage_tracker: UsageTracker = Depends(get_usage_tracker)
):
    """Generate usage report for compliance requirements."""
    try:
        report = usage_tracker.generate_compliance_usage_report(
            compliance_type=compliance_type,
            start_date=start_date,
            end_date=end_date
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))