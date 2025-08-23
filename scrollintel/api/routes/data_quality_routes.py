"""
API Routes for Data Quality Monitoring System
Provides REST endpoints for quality monitoring, profiling, and alerting
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd
import json

from ...core.database import get_db
from ...models.data_quality_models import (
    QualityRule, QualityReport, DataAnomaly, DataProfile, QualityAlert,
    QualityRuleType, Severity, QualityStatus
)
from ...engines.data_quality_monitor import DataQualityMonitor
from ...engines.anomaly_detector import AnomalyDetector
from ...engines.quality_alerting import QualityAlertManager, RealTimeQualityMonitor, AlertConfig, AlertChannel
from ...engines.data_profiler import DataProfiler, ProfilingConfig

router = APIRouter(prefix="/api/v1/data-quality", tags=["Data Quality"])

# Pydantic models for API
class QualityRuleCreate(BaseModel):
    name: str
    description: Optional[str] = None
    rule_type: QualityRuleType
    severity: Severity = Severity.MEDIUM
    target_table: str
    target_column: str
    target_pipeline_id: Optional[str] = None
    threshold_value: Optional[float] = None
    expected_value: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, Any]]] = None

class QualityRuleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[Severity] = None
    threshold_value: Optional[float] = None
    expected_value: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, Any]]] = None
    is_active: Optional[bool] = None

class DataValidationRequest(BaseModel):
    data: List[Dict[str, Any]]
    rule_ids: List[str]
    pipeline_execution_id: Optional[str] = None

class ProfilingRequest(BaseModel):
    data: List[Dict[str, Any]]
    table_name: str
    pipeline_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class AnomalyDetectionRequest(BaseModel):
    data: List[Dict[str, Any]]
    profile_id: str
    detection_methods: Optional[List[str]] = None

class AlertConfigUpdate(BaseModel):
    channels: List[AlertChannel]
    recipients: List[str]
    severity_threshold: Severity
    cooldown_minutes: int = 15
    escalation_minutes: int = 60

class QualityMetricsResponse(BaseModel):
    total_checks: int
    passed_checks: int
    failed_checks: int
    average_score: float
    quality_trend: List[Dict[str, Any]]

# Quality Rules Management
@router.post("/rules", response_model=Dict[str, str])
async def create_quality_rule(
    rule_data: QualityRuleCreate,
    db: Session = Depends(get_db)
):
    """Create a new quality rule"""
    try:
        monitor = DataQualityMonitor(db)
        rule_id = monitor.create_quality_rule(rule_data.dict())
        return {"rule_id": rule_id, "message": "Quality rule created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/rules", response_model=List[Dict[str, Any]])
async def list_quality_rules(
    pipeline_id: Optional[str] = Query(None),
    table_name: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    """List quality rules with optional filtering"""
    query = db.query(QualityRule)
    
    if pipeline_id:
        query = query.filter(QualityRule.target_pipeline_id == pipeline_id)
    if table_name:
        query = query.filter(QualityRule.target_table == table_name)
    if is_active is not None:
        query = query.filter(QualityRule.is_active == is_active)
    
    rules = query.all()
    
    return [
        {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "rule_type": rule.rule_type.value,
            "severity": rule.severity.value,
            "target_table": rule.target_table,
            "target_column": rule.target_column,
            "target_pipeline_id": rule.target_pipeline_id,
            "threshold_value": rule.threshold_value,
            "conditions": rule.conditions,
            "actions": rule.actions,
            "is_active": rule.is_active,
            "created_at": rule.created_at.isoformat(),
            "updated_at": rule.updated_at.isoformat()
        }
        for rule in rules
    ]

@router.get("/rules/{rule_id}", response_model=Dict[str, Any])
async def get_quality_rule(rule_id: str, db: Session = Depends(get_db)):
    """Get a specific quality rule"""
    rule = db.query(QualityRule).filter(QualityRule.id == rule_id).first()
    
    if not rule:
        raise HTTPException(status_code=404, detail="Quality rule not found")
    
    return {
        "id": rule.id,
        "name": rule.name,
        "description": rule.description,
        "rule_type": rule.rule_type.value,
        "severity": rule.severity.value,
        "target_table": rule.target_table,
        "target_column": rule.target_column,
        "target_pipeline_id": rule.target_pipeline_id,
        "threshold_value": rule.threshold_value,
        "expected_value": rule.expected_value,
        "conditions": rule.conditions,
        "actions": rule.actions,
        "is_active": rule.is_active,
        "created_at": rule.created_at.isoformat(),
        "updated_at": rule.updated_at.isoformat()
    }

@router.put("/rules/{rule_id}", response_model=Dict[str, str])
async def update_quality_rule(
    rule_id: str,
    rule_update: QualityRuleUpdate,
    db: Session = Depends(get_db)
):
    """Update a quality rule"""
    rule = db.query(QualityRule).filter(QualityRule.id == rule_id).first()
    
    if not rule:
        raise HTTPException(status_code=404, detail="Quality rule not found")
    
    # Update fields
    update_data = rule_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(rule, field, value)
    
    rule.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Quality rule updated successfully"}

@router.delete("/rules/{rule_id}", response_model=Dict[str, str])
async def delete_quality_rule(rule_id: str, db: Session = Depends(get_db)):
    """Delete a quality rule"""
    rule = db.query(QualityRule).filter(QualityRule.id == rule_id).first()
    
    if not rule:
        raise HTTPException(status_code=404, detail="Quality rule not found")
    
    db.delete(rule)
    db.commit()
    
    return {"message": "Quality rule deleted successfully"}

# Data Validation
@router.post("/validate", response_model=List[Dict[str, Any]])
async def validate_data(
    validation_request: DataValidationRequest,
    db: Session = Depends(get_db)
):
    """Validate data against quality rules"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(validation_request.data)
        
        # Get rules
        rules = db.query(QualityRule).filter(
            QualityRule.id.in_(validation_request.rule_ids)
        ).all()
        
        if not rules:
            raise HTTPException(status_code=404, detail="No valid rules found")
        
        # Validate data
        monitor = DataQualityMonitor(db)
        reports = monitor.validate_data_batch(
            df, 
            rules, 
            validation_request.pipeline_execution_id
        )
        
        return [
            {
                "id": report.id,
                "rule_id": report.rule_id,
                "status": report.status.value,
                "score": report.score,
                "records_checked": report.records_checked,
                "records_failed": report.records_failed,
                "error_message": report.error_message,
                "error_details": report.error_details,
                "sample_failures": report.sample_failures,
                "execution_time_ms": report.execution_time_ms,
                "check_timestamp": report.check_timestamp.isoformat()
            }
            for report in reports
        ]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Data Profiling
@router.post("/profile", response_model=List[Dict[str, Any]])
async def create_data_profile(
    profiling_request: ProfilingRequest,
    db: Session = Depends(get_db)
):
    """Create comprehensive data profiles"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(profiling_request.data)
        
        # Create profiler with config
        config = ProfilingConfig(**(profiling_request.config or {}))
        profiler = DataProfiler(db, config)
        
        # Create profiles
        profiles = profiler.create_comprehensive_profile(
            df,
            profiling_request.table_name,
            profiling_request.pipeline_id
        )
        
        return [
            {
                "id": profile.id,
                "table_name": profile.table_name,
                "column_name": profile.column_name,
                "data_type": profile.data_type,
                "record_count": profile.record_count,
                "null_count": profile.null_count,
                "unique_count": profile.unique_count,
                "min_value": profile.min_value,
                "max_value": profile.max_value,
                "mean_value": profile.mean_value,
                "median_value": profile.median_value,
                "std_deviation": profile.std_deviation,
                "most_frequent_values": profile.most_frequent_values,
                "value_distribution": profile.value_distribution,
                "common_patterns": profile.common_patterns,
                "format_patterns": profile.format_patterns,
                "completeness_score": profile.completeness_score,
                "consistency_score": profile.consistency_score,
                "validity_score": profile.validity_score,
                "created_at": profile.created_at.isoformat()
            }
            for profile in profiles
        ]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/profiles", response_model=List[Dict[str, Any]])
async def list_data_profiles(
    table_name: Optional[str] = Query(None),
    column_name: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """List data profiles with optional filtering"""
    query = db.query(DataProfile)
    
    if table_name:
        query = query.filter(DataProfile.table_name == table_name)
    if column_name:
        query = query.filter(DataProfile.column_name == column_name)
    
    profiles = query.order_by(DataProfile.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": profile.id,
            "table_name": profile.table_name,
            "column_name": profile.column_name,
            "data_type": profile.data_type,
            "record_count": profile.record_count,
            "completeness_score": profile.completeness_score,
            "validity_score": profile.validity_score,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat()
        }
        for profile in profiles
    ]

@router.post("/profiles/{profile_id}/baseline", response_model=Dict[str, Any])
async def establish_baseline(
    profile_id: str,
    profiling_request: ProfilingRequest,
    db: Session = Depends(get_db)
):
    """Establish quality baseline and get recommended rules"""
    try:
        # Get profile
        profile = db.query(DataProfile).filter(DataProfile.id == profile_id).first()
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Get all profiles for the table
        profiles = db.query(DataProfile).filter(
            DataProfile.table_name == profile.table_name
        ).all()
        
        # Establish baseline
        profiler = DataProfiler(db)
        baseline = profiler.establish_quality_baseline(
            profiles,
            profile.table_name,
            profiling_request.pipeline_id
        )
        
        return {
            "completeness_baseline": baseline.completeness_baseline,
            "validity_baseline": baseline.validity_baseline,
            "consistency_baseline": baseline.consistency_baseline,
            "uniqueness_baseline": baseline.uniqueness_baseline,
            "statistical_baseline": baseline.statistical_baseline,
            "pattern_baseline": baseline.pattern_baseline,
            "recommended_rules": baseline.recommended_rules
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Anomaly Detection
@router.post("/anomalies/detect", response_model=List[Dict[str, Any]])
async def detect_anomalies(
    detection_request: AnomalyDetectionRequest,
    db: Session = Depends(get_db)
):
    """Detect anomalies in data"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(detection_request.data)
        
        # Get profile
        profile = db.query(DataProfile).filter(
            DataProfile.id == detection_request.profile_id
        ).first()
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Detect anomalies
        detector = AnomalyDetector(db)
        anomalies = detector.detect_anomalies(
            df,
            profile,
            detection_request.detection_methods
        )
        
        return [
            {
                "id": anomaly.get("id"),
                "table_name": anomaly.get("table_name"),
                "column_name": anomaly.get("column_name"),
                "record_id": anomaly.get("record_id"),
                "anomaly_type": anomaly.get("anomaly_type"),
                "confidence_score": anomaly.get("confidence_score"),
                "severity": anomaly.get("severity"),
                "expected_value": anomaly.get("expected_value"),
                "actual_value": anomaly.get("actual_value"),
                "deviation_score": anomaly.get("deviation_score"),
                "z_score": anomaly.get("z_score")
            }
            for anomaly in anomalies
        ]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/anomalies", response_model=List[Dict[str, Any]])
async def list_anomalies(
    table_name: Optional[str] = Query(None),
    column_name: Optional[str] = Query(None),
    severity: Optional[Severity] = Query(None),
    is_resolved: Optional[bool] = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """List detected anomalies with filtering"""
    query = db.query(DataAnomaly)
    
    if table_name:
        query = query.filter(DataAnomaly.table_name == table_name)
    if column_name:
        query = query.filter(DataAnomaly.column_name == column_name)
    if severity:
        query = query.filter(DataAnomaly.severity == severity)
    if is_resolved is not None:
        query = query.filter(DataAnomaly.is_resolved == is_resolved)
    
    anomalies = query.order_by(DataAnomaly.detected_at.desc()).limit(limit).all()
    
    return [
        {
            "id": anomaly.id,
            "table_name": anomaly.table_name,
            "column_name": anomaly.column_name,
            "record_id": anomaly.record_id,
            "anomaly_type": anomaly.anomaly_type,
            "confidence_score": anomaly.confidence_score,
            "severity": anomaly.severity.value,
            "expected_value": anomaly.expected_value,
            "actual_value": anomaly.actual_value,
            "deviation_score": anomaly.deviation_score,
            "z_score": anomaly.z_score,
            "detected_at": anomaly.detected_at.isoformat(),
            "is_resolved": anomaly.is_resolved,
            "resolved_at": anomaly.resolved_at.isoformat() if anomaly.resolved_at else None
        }
        for anomaly in anomalies
    ]

@router.put("/anomalies/{anomaly_id}/resolve", response_model=Dict[str, str])
async def resolve_anomaly(anomaly_id: str, db: Session = Depends(get_db)):
    """Mark an anomaly as resolved"""
    anomaly = db.query(DataAnomaly).filter(DataAnomaly.id == anomaly_id).first()
    
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    
    anomaly.is_resolved = True
    anomaly.resolved_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Anomaly marked as resolved"}

# Quality Metrics and Reporting
@router.get("/metrics/{pipeline_id}", response_model=QualityMetricsResponse)
async def get_quality_metrics(
    pipeline_id: str,
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    db: Session = Depends(get_db)
):
    """Get quality metrics for a pipeline"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        monitor = DataQualityMonitor(db)
        metrics = monitor.get_quality_metrics(pipeline_id, (start_time, end_time))
        
        return QualityMetricsResponse(**metrics)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/reports", response_model=List[Dict[str, Any]])
async def list_quality_reports(
    pipeline_id: Optional[str] = Query(None),
    status: Optional[QualityStatus] = Query(None),
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """List quality reports with filtering"""
    query = db.query(QualityReport).join(QualityRule)
    
    if pipeline_id:
        query = query.filter(QualityRule.target_pipeline_id == pipeline_id)
    if status:
        query = query.filter(QualityReport.status == status)
    
    # Time filter
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    query = query.filter(QualityReport.check_timestamp > cutoff_time)
    
    reports = query.order_by(QualityReport.check_timestamp.desc()).limit(limit).all()
    
    return [
        {
            "id": report.id,
            "rule_id": report.rule_id,
            "rule_name": report.rule.name,
            "status": report.status.value,
            "score": report.score,
            "records_checked": report.records_checked,
            "records_failed": report.records_failed,
            "error_message": report.error_message,
            "execution_time_ms": report.execution_time_ms,
            "check_timestamp": report.check_timestamp.isoformat()
        }
        for report in reports
    ]

# Alerting and Monitoring
@router.get("/alerts", response_model=List[Dict[str, Any]])
async def list_quality_alerts(
    severity: Optional[Severity] = Query(None),
    is_acknowledged: Optional[bool] = Query(None),
    is_resolved: Optional[bool] = Query(None),
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """List quality alerts with filtering"""
    query = db.query(QualityAlert)
    
    if severity:
        query = query.filter(QualityAlert.severity == severity)
    if is_acknowledged is not None:
        query = query.filter(QualityAlert.is_acknowledged == is_acknowledged)
    if is_resolved is not None:
        query = query.filter(QualityAlert.is_resolved == is_resolved)
    
    # Time filter
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    query = query.filter(QualityAlert.created_at > cutoff_time)
    
    alerts = query.order_by(QualityAlert.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": alert.id,
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "message": alert.message,
            "pipeline_id": alert.pipeline_id,
            "table_name": alert.table_name,
            "column_name": alert.column_name,
            "is_acknowledged": alert.is_acknowledged,
            "is_resolved": alert.is_resolved,
            "created_at": alert.created_at.isoformat(),
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
        }
        for alert in alerts
    ]

@router.put("/alerts/{alert_id}/acknowledge", response_model=Dict[str, str])
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str,
    db: Session = Depends(get_db)
):
    """Acknowledge a quality alert"""
    alert = db.query(QualityAlert).filter(QualityAlert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.is_acknowledged = True
    alert.acknowledged_by = acknowledged_by
    alert.acknowledged_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Alert acknowledged successfully"}

@router.put("/alerts/{alert_id}/resolve", response_model=Dict[str, str])
async def resolve_alert(
    alert_id: str,
    resolved_by: str,
    resolution_notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Resolve a quality alert"""
    alert = db.query(QualityAlert).filter(QualityAlert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.is_resolved = True
    alert.resolved_by = resolved_by
    alert.resolved_at = datetime.utcnow()
    alert.resolution_notes = resolution_notes
    db.commit()
    
    return {"message": "Alert resolved successfully"}

# Health and Status
@router.get("/health", response_model=Dict[str, Any])
async def get_system_health(db: Session = Depends(get_db)):
    """Get data quality system health status"""
    try:
        # Count active rules
        active_rules = db.query(QualityRule).filter(QualityRule.is_active == True).count()
        
        # Count recent reports
        recent_cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_reports = db.query(QualityReport).filter(
            QualityReport.check_timestamp > recent_cutoff
        ).count()
        
        # Count unresolved alerts
        unresolved_alerts = db.query(QualityAlert).filter(
            QualityAlert.is_resolved == False
        ).count()
        
        # Count active anomalies
        active_anomalies = db.query(DataAnomaly).filter(
            DataAnomaly.is_resolved == False
        ).count()
        
        return {
            "status": "healthy",
            "active_rules": active_rules,
            "recent_reports": recent_reports,
            "unresolved_alerts": unresolved_alerts,
            "active_anomalies": active_anomalies,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }