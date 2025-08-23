"""
API Routes for Data Lineage and Compliance System

This module provides REST API endpoints for data lineage tracking,
compliance management, and audit reporting.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import logging

from scrollintel.engines.lineage_tracker import LineageTracker
from scrollintel.engines.compliance_engine import ComplianceEngine
from scrollintel.engines.audit_reporter import AuditReporter, AuditReportConfig
from scrollintel.engines.data_governance import DataGovernanceEngine, DataClassification, PrivacyLevel
from scrollintel.models.lineage_models import (
    LineageEventRequest, LineageQueryRequest, ComplianceRuleRequest,
    ComplianceViolationResponse, ComplianceStatus, AuditTrailRequest
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/lineage-compliance", tags=["Data Lineage & Compliance"])

# Dependency injection
def get_lineage_tracker() -> LineageTracker:
    return LineageTracker()

def get_compliance_engine() -> ComplianceEngine:
    return ComplianceEngine()

def get_audit_reporter() -> AuditReporter:
    return AuditReporter()

def get_governance_engine() -> DataGovernanceEngine:
    return DataGovernanceEngine()


# Data Lineage Endpoints

@router.post("/lineage/track")
async def track_lineage_event(
    request: LineageEventRequest,
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    tracker: LineageTracker = Depends(get_lineage_tracker)
) -> Dict[str, str]:
    """Track a data lineage event"""
    try:
        lineage_id = tracker.track_lineage_event(request, user_id, session_id)
        return {"lineage_id": lineage_id, "status": "tracked"}
    except Exception as e:
        logger.error(f"Error tracking lineage event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lineage/query")
async def query_lineage_history(
    request: LineageQueryRequest,
    tracker: LineageTracker = Depends(get_lineage_tracker)
) -> List[Dict[str, Any]]:
    """Query data lineage history"""
    try:
        return tracker.get_lineage_history(request)
    except Exception as e:
        logger.error(f"Error querying lineage history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lineage/graph/{dataset_id}")
async def get_lineage_graph(
    dataset_id: str,
    depth: int = Query(5, ge=1, le=10),
    direction: str = Query("both", regex="^(upstream|downstream|both)$"),
    tracker: LineageTracker = Depends(get_lineage_tracker)
) -> Dict[str, Any]:
    """Get data lineage graph for a dataset"""
    try:
        return tracker.get_data_lineage_graph(dataset_id, depth, direction)
    except Exception as e:
        logger.error(f"Error getting lineage graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lineage/impact-analysis/{dataset_id}")
async def get_impact_analysis(
    dataset_id: str,
    change_type: str = Query("schema_change"),
    tracker: LineageTracker = Depends(get_lineage_tracker)
) -> Dict[str, Any]:
    """Analyze impact of changes to a dataset"""
    try:
        return tracker.get_impact_analysis(dataset_id, change_type)
    except Exception as e:
        logger.error(f"Error performing impact analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Compliance Management Endpoints

@router.post("/compliance/rules")
async def create_compliance_rule(
    request: ComplianceRuleRequest,
    created_by: str = Query(...),
    engine: ComplianceEngine = Depends(get_compliance_engine)
) -> Dict[str, str]:
    """Create a new compliance rule"""
    try:
        rule_id = engine.create_compliance_rule(request, created_by)
        return {"rule_id": rule_id, "status": "created"}
    except Exception as e:
        logger.error(f"Error creating compliance rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compliance/evaluate")
async def evaluate_compliance(
    pipeline_id: str = Body(...),
    dataset_id: Optional[str] = Body(None),
    operation_type: str = Body("data_processing"),
    context: Optional[Dict[str, Any]] = Body(None),
    engine: ComplianceEngine = Depends(get_compliance_engine)
) -> List[Dict[str, Any]]:
    """Evaluate compliance for a pipeline operation"""
    try:
        return engine.evaluate_compliance(pipeline_id, dataset_id, operation_type, context)
    except Exception as e:
        logger.error(f"Error evaluating compliance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/violations")
async def get_compliance_violations(
    pipeline_id: Optional[str] = Query(None),
    rule_id: Optional[str] = Query(None),
    status: Optional[ComplianceStatus] = Query(None),
    severity: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    engine: ComplianceEngine = Depends(get_compliance_engine)
) -> List[ComplianceViolationResponse]:
    """Get compliance violations"""
    try:
        return engine.get_compliance_violations(
            pipeline_id, rule_id, status, severity, start_date, end_date, limit
        )
    except Exception as e:
        logger.error(f"Error getting compliance violations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/compliance/violations/{violation_id}/resolve")
async def resolve_violation(
    violation_id: str,
    resolution_notes: str = Body(...),
    resolved_by: str = Body(...),
    engine: ComplianceEngine = Depends(get_compliance_engine)
) -> Dict[str, str]:
    """Resolve a compliance violation"""
    try:
        success = engine.resolve_violation(violation_id, resolution_notes, resolved_by)
        return {"violation_id": violation_id, "status": "resolved" if success else "failed"}
    except Exception as e:
        logger.error(f"Error resolving violation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/report")
async def get_compliance_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    include_details: bool = Query(False),
    engine: ComplianceEngine = Depends(get_compliance_engine)
) -> Dict[str, Any]:
    """Generate compliance report"""
    try:
        return engine.get_compliance_report(start_date, end_date, include_details)
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Governance Endpoints

@router.post("/governance/classify")
async def classify_data(
    dataset_id: str = Body(...),
    data_sample: Dict[str, Any] = Body(...),
    schema_info: Optional[Dict[str, Any]] = Body(None),
    engine: DataGovernanceEngine = Depends(get_governance_engine)
) -> Dict[str, Any]:
    """Automatically classify data"""
    try:
        return engine.classify_data(dataset_id, data_sample, schema_info)
    except Exception as e:
        logger.error(f"Error classifying data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/enforce-policy")
async def enforce_governance_policy(
    policy_id: str = Body(...),
    operation: str = Body(...),
    data_context: Dict[str, Any] = Body(...),
    user_context: Dict[str, Any] = Body(...),
    engine: DataGovernanceEngine = Depends(get_governance_engine)
) -> Dict[str, Any]:
    """Enforce data governance policy"""
    try:
        return engine.enforce_data_governance_policy(
            policy_id, operation, data_context, user_context
        )
    except Exception as e:
        logger.error(f"Error enforcing governance policy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/apply-privacy-controls")
async def apply_privacy_controls(
    dataset_id: str = Body(...),
    data: Dict[str, Any] = Body(...),
    privacy_level: PrivacyLevel = Body(...),
    user_context: Dict[str, Any] = Body(...),
    engine: DataGovernanceEngine = Depends(get_governance_engine)
) -> Dict[str, Any]:
    """Apply privacy controls to data"""
    try:
        return engine.apply_privacy_controls(dataset_id, data, privacy_level, user_context)
    except Exception as e:
        logger.error(f"Error applying privacy controls: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance/retention-compliance/{dataset_id}")
async def check_retention_compliance(
    dataset_id: str,
    data_age_days: int = Query(...),
    classification: DataClassification = Query(...),
    engine: DataGovernanceEngine = Depends(get_governance_engine)
) -> Dict[str, Any]:
    """Check data retention compliance"""
    try:
        data_age = timedelta(days=data_age_days)
        return engine.check_data_retention_compliance(dataset_id, data_age, classification)
    except Exception as e:
        logger.error(f"Error checking retention compliance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/anonymize")
async def anonymize_data(
    data: Dict[str, Any] = Body(...),
    anonymization_level: str = Body("standard"),
    preserve_utility: bool = Body(True),
    engine: DataGovernanceEngine = Depends(get_governance_engine)
) -> Dict[str, Any]:
    """Anonymize data for privacy protection"""
    try:
        return engine.anonymize_data(data, anonymization_level, preserve_utility)
    except Exception as e:
        logger.error(f"Error anonymizing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Audit Trail Endpoints

@router.get("/audit/trail")
async def get_audit_trail(
    entity_type: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    tracker: LineageTracker = Depends(get_lineage_tracker)
) -> List[Dict[str, Any]]:
    """Get audit trail entries"""
    try:
        return tracker.get_audit_trail(
            entity_type, entity_id, user_id, start_date, end_date, limit
        )
    except Exception as e:
        logger.error(f"Error getting audit trail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit/report")
async def generate_audit_report(
    config: AuditReportConfig,
    reporter: AuditReporter = Depends(get_audit_reporter)
) -> Dict[str, Any]:
    """Generate comprehensive audit report"""
    try:
        return reporter.generate_audit_report(config)
    except Exception as e:
        logger.error(f"Error generating audit report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audit/compliance-report")
async def generate_compliance_audit_report(
    regulation_type: str = Body(...),
    start_date: datetime = Body(...),
    end_date: datetime = Body(...),
    reporter: AuditReporter = Depends(get_audit_reporter)
) -> Dict[str, Any]:
    """Generate compliance-specific audit report"""
    try:
        return reporter.generate_compliance_audit_report(
            regulation_type, start_date, end_date
        )
    except Exception as e:
        logger.error(f"Error generating compliance audit report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/export")
async def export_audit_trail(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    format: str = Query("csv", regex="^(csv|json)$"),
    entity_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    reporter: AuditReporter = Depends(get_audit_reporter)
):
    """Export audit trail data"""
    try:
        filters = {}
        if entity_type:
            filters["entity_type"] = entity_type
        if user_id:
            filters["user_id"] = user_id
        if action:
            filters["action"] = action
        
        exported_data = reporter.export_audit_trail(
            start_date, end_date, format, filters
        )
        
        # Determine content type and filename
        if format == "csv":
            content_type = "text/csv"
            filename = f"audit_trail_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        else:
            content_type = "application/json"
            filename = f"audit_trail_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        
        # Create streaming response
        def generate():
            yield exported_data
        
        return StreamingResponse(
            io.StringIO(exported_data),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error exporting audit trail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/data-processing-history/{dataset_id}")
async def get_data_processing_history(
    dataset_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    reporter: AuditReporter = Depends(get_audit_reporter)
) -> Dict[str, Any]:
    """Get complete processing history for a dataset"""
    try:
        return reporter.get_data_processing_history(dataset_id, start_date, end_date)
    except Exception as e:
        logger.error(f"Error getting data processing history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check Endpoint

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "lineage-compliance"}


# Utility Endpoints

@router.get("/schemas/lineage-event")
async def get_lineage_event_schema() -> Dict[str, Any]:
    """Get schema for lineage event requests"""
    return LineageEventRequest.schema()


@router.get("/schemas/compliance-rule")
async def get_compliance_rule_schema() -> Dict[str, Any]:
    """Get schema for compliance rule requests"""
    return ComplianceRuleRequest.schema()


@router.get("/enums/data-classification")
async def get_data_classification_values() -> List[str]:
    """Get available data classification values"""
    return [classification.value for classification in DataClassification]


@router.get("/enums/privacy-level")
async def get_privacy_level_values() -> List[str]:
    """Get available privacy level values"""
    return [level.value for level in PrivacyLevel]