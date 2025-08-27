"""
API routes for Data Quality System
Provides REST endpoints for data normalization, quality monitoring, and reporting
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import json
import logging
from pydantic import BaseModel, Field

from scrollintel.core.data_normalizer import (
    DataNormalizer, DataSchema, SchemaField, SchemaMapping,
    DataType, TransformationType
)
from scrollintel.core.data_quality_monitor import (
    DataQualityMonitor, QualityRule, QualityRuleType, QualitySeverity
)
from scrollintel.core.data_lineage import (
    DataLineageTracker, DataAsset, LineageEventType, DataClassification
)
from scrollintel.core.data_reconciliation import (
    DataReconciliationEngine, DataSource, ReconciliationRule,
    ConflictResolutionStrategy
)
from scrollintel.core.data_quality_alerting import (
    DataQualityAlerting, AlertRule, AlertChannel, AlertFrequency
)
from scrollintel.core.data_quality_reporting import (
    DataQualityReporting, ReportTemplate, ReportFormat, ReportFrequency
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data-quality", tags=["data-quality"])

# Global instances (in production, these would be dependency-injected)
data_normalizer = DataNormalizer()
quality_monitor = DataQualityMonitor()
lineage_tracker = DataLineageTracker()
reconciliation_engine = DataReconciliationEngine()
quality_alerting = DataQualityAlerting()
quality_reporting = DataQualityReporting()


# Pydantic models for API requests/responses
class SchemaFieldModel(BaseModel):
    name: str
    data_type: str
    required: bool = True
    default_value: Any = None
    validation_rules: List[str] = []
    description: str = ""


class DataSchemaModel(BaseModel):
    name: str
    version: str
    fields: List[SchemaFieldModel]
    metadata: Dict[str, Any] = {}


class SchemaMappingModel(BaseModel):
    source_field: str
    target_field: str
    transformation_type: str
    transformation_config: Dict[str, Any] = {}
    validation_rules: List[str] = []


class QualityRuleModel(BaseModel):
    id: str
    name: str
    description: str
    rule_type: str
    severity: str
    target_fields: List[str]
    parameters: Dict[str, Any] = {}
    enabled: bool = True


class DataSourceModel(BaseModel):
    id: str
    name: str
    priority: int
    reliability_score: float
    metadata: Dict[str, Any] = {}


class DataAssetModel(BaseModel):
    id: str
    name: str
    type: str
    source_system: str
    schema_info: Dict[str, Any]
    classification: str
    owner: str
    metadata: Dict[str, Any] = {}


class AlertRuleModel(BaseModel):
    id: str
    name: str
    description: str
    dataset_patterns: List[str]
    severity_threshold: str
    score_threshold: float
    channels: List[str]
    frequency: str
    enabled: bool = True
    recipients: List[str] = []
    custom_conditions: Dict[str, Any] = {}


# Schema Management Endpoints
@router.post("/schemas")
async def register_schema(schema: DataSchemaModel):
    """Register a new data schema"""
    try:
        # Convert Pydantic model to domain model
        fields = []
        for field_model in schema.fields:
            field = SchemaField(
                name=field_model.name,
                data_type=DataType(field_model.data_type),
                required=field_model.required,
                default_value=field_model.default_value,
                validation_rules=field_model.validation_rules,
                description=field_model.description
            )
            fields.append(field)
        
        data_schema = DataSchema(
            name=schema.name,
            version=schema.version,
            fields=fields,
            metadata=schema.metadata
        )
        
        success = data_normalizer.register_schema(data_schema)
        
        if success:
            return {"success": True, "message": f"Schema {schema.name} registered successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to register schema")
            
    except Exception as e:
        logger.error(f"Failed to register schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schemas")
async def list_schemas():
    """List all registered schemas"""
    try:
        schemas = []
        for schema_name, schema in data_normalizer.schemas.items():
            schema_info = data_normalizer.get_schema_info(schema_name)
            if schema_info:
                schemas.append(schema_info)
        
        return {"schemas": schemas}
        
    except Exception as e:
        logger.error(f"Failed to list schemas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schemas/{schema_name}")
async def get_schema(schema_name: str):
    """Get details of a specific schema"""
    try:
        schema_info = data_normalizer.get_schema_info(schema_name)
        
        if not schema_info:
            raise HTTPException(status_code=404, detail=f"Schema {schema_name} not found")
        
        return schema_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schemas/{source_schema}/mappings/{target_schema}")
async def register_mapping(source_schema: str, target_schema: str, mappings: List[SchemaMappingModel]):
    """Register field mappings between schemas"""
    try:
        # Convert Pydantic models to domain models
        schema_mappings = []
        for mapping_model in mappings:
            mapping = SchemaMapping(
                source_field=mapping_model.source_field,
                target_field=mapping_model.target_field,
                transformation_type=TransformationType(mapping_model.transformation_type),
                transformation_config=mapping_model.transformation_config,
                validation_rules=mapping_model.validation_rules
            )
            schema_mappings.append(mapping)
        
        success = data_normalizer.register_mapping(source_schema, target_schema, schema_mappings)
        
        if success:
            return {"success": True, "message": f"Mappings registered for {source_schema} -> {target_schema}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to register mappings")
            
    except Exception as e:
        logger.error(f"Failed to register mappings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Normalization Endpoints
@router.post("/normalize")
async def normalize_data(
    source_schema: str,
    target_schema: str,
    file: UploadFile = File(...)
):
    """Normalize uploaded data from source schema to target schema"""
    try:
        # Read uploaded file
        if file.content_type == "text/csv":
            content = await file.read()
            data = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        elif file.content_type == "application/json":
            content = await file.read()
            json_data = json.loads(content.decode('utf-8'))
            data = pd.DataFrame(json_data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        # Normalize data
        result = data_normalizer.normalize_data(data, source_schema, target_schema)
        
        if result.success:
            # Convert DataFrame to JSON for response
            normalized_json = result.normalized_data.to_dict(orient='records')
            
            return {
                "success": True,
                "normalized_data": normalized_json,
                "errors": result.errors,
                "warnings": result.warnings,
                "quality_metrics": result.quality_metrics,
                "transformation_log": result.transformation_log
            }
        else:
            return {
                "success": False,
                "errors": result.errors,
                "warnings": result.warnings
            }
            
    except Exception as e:
        logger.error(f"Failed to normalize data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_data(
    schema_name: str,
    file: UploadFile = File(...)
):
    """Validate uploaded data against a schema"""
    try:
        # Read uploaded file
        if file.content_type == "text/csv":
            content = await file.read()
            data = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        elif file.content_type == "application/json":
            content = await file.read()
            json_data = json.loads(content.decode('utf-8'))
            data = pd.DataFrame(json_data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        # Validate data
        validation_result = data_normalizer.validate_normalized_data(data, schema_name)
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Failed to validate data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Quality Monitoring Endpoints
@router.post("/quality-rules")
async def register_quality_rule(rule: QualityRuleModel):
    """Register a new data quality rule"""
    try:
        quality_rule = QualityRule(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            rule_type=QualityRuleType(rule.rule_type),
            severity=QualitySeverity(rule.severity),
            target_fields=rule.target_fields,
            parameters=rule.parameters,
            enabled=rule.enabled
        )
        
        success = quality_monitor.register_quality_rule(quality_rule)
        
        if success:
            return {"success": True, "message": f"Quality rule {rule.name} registered successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to register quality rule")
            
    except Exception as e:
        logger.error(f"Failed to register quality rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-quality")
async def assess_quality(
    dataset_name: str,
    rule_ids: Optional[List[str]] = None,
    file: UploadFile = File(...)
):
    """Assess data quality of uploaded dataset"""
    try:
        # Read uploaded file
        if file.content_type == "text/csv":
            content = await file.read()
            data = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        elif file.content_type == "application/json":
            content = await file.read()
            json_data = json.loads(content.decode('utf-8'))
            data = pd.DataFrame(json_data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        # Assess quality
        report = quality_monitor.assess_data_quality(data, dataset_name, rule_ids)
        
        # Convert report to JSON-serializable format
        issues_json = []
        for issue in report.issues:
            issues_json.append({
                "rule_id": issue.rule_id,
                "rule_name": issue.rule_name,
                "severity": issue.severity.value,
                "field_name": issue.field_name,
                "issue_description": issue.issue_description,
                "affected_records": issue.affected_records,
                "sample_values": issue.sample_values,
                "detected_at": issue.detected_at.isoformat()
            })
        
        return {
            "dataset_name": report.dataset_name,
            "assessment_timestamp": report.assessment_timestamp.isoformat(),
            "total_records": report.total_records,
            "overall_score": report.overall_score,
            "dimension_scores": report.dimension_scores,
            "issues": issues_json,
            "metrics": report.metrics,
            "recommendations": report.recommendations
        }
        
    except Exception as e:
        logger.error(f"Failed to assess quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality-trends/{dataset_name}")
async def get_quality_trends(dataset_name: str, days: int = 30):
    """Get quality trends for a dataset"""
    try:
        trends = quality_monitor.get_quality_trends(dataset_name, days)
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get quality trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Lineage Endpoints
@router.post("/lineage/assets")
async def register_data_asset(asset: DataAssetModel):
    """Register a new data asset for lineage tracking"""
    try:
        data_asset = DataAsset(
            id=asset.id,
            name=asset.name,
            type=asset.type,
            source_system=asset.source_system,
            schema_info=asset.schema_info,
            classification=DataClassification(asset.classification),
            owner=asset.owner,
            metadata=asset.metadata
        )
        
        success = lineage_tracker.register_data_asset(data_asset)
        
        if success:
            return {"success": True, "message": f"Data asset {asset.name} registered successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to register data asset")
            
    except Exception as e:
        logger.error(f"Failed to register data asset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lineage/upstream/{asset_id}")
async def get_upstream_lineage(asset_id: str, max_depth: int = 10):
    """Get upstream lineage for a data asset"""
    try:
        lineage = lineage_tracker.get_upstream_lineage(asset_id, max_depth)
        return lineage
        
    except Exception as e:
        logger.error(f"Failed to get upstream lineage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lineage/downstream/{asset_id}")
async def get_downstream_lineage(asset_id: str, max_depth: int = 10):
    """Get downstream lineage for a data asset"""
    try:
        lineage = lineage_tracker.get_downstream_lineage(asset_id, max_depth)
        return lineage
        
    except Exception as e:
        logger.error(f"Failed to get downstream lineage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lineage/impact/{asset_id}")
async def get_impact_analysis(asset_id: str):
    """Get impact analysis for a data asset"""
    try:
        impact = lineage_tracker.get_impact_analysis(asset_id)
        return impact
        
    except Exception as e:
        logger.error(f"Failed to get impact analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Reconciliation Endpoints
@router.post("/reconciliation/sources")
async def register_data_source(source: DataSourceModel):
    """Register a data source for reconciliation"""
    try:
        data_source = DataSource(
            id=source.id,
            name=source.name,
            priority=source.priority,
            reliability_score=source.reliability_score,
            last_updated=datetime.utcnow(),
            metadata=source.metadata
        )
        
        success = reconciliation_engine.register_data_source(data_source)
        
        if success:
            return {"success": True, "message": f"Data source {source.name} registered successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to register data source")
            
    except Exception as e:
        logger.error(f"Failed to register data source: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reconciliation/reconcile")
async def reconcile_datasets(
    key_field: str,
    files: List[UploadFile] = File(...),
    source_ids: List[str] = []
):
    """Reconcile data from multiple uploaded files"""
    try:
        if len(files) != len(source_ids):
            raise HTTPException(status_code=400, detail="Number of files must match number of source IDs")
        
        datasets = {}
        
        # Read all uploaded files
        for i, file in enumerate(files):
            source_id = source_ids[i]
            
            if file.content_type == "text/csv":
                content = await file.read()
                data = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
            elif file.content_type == "application/json":
                content = await file.read()
                json_data = json.loads(content.decode('utf-8'))
                data = pd.DataFrame(json_data)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format for {file.filename}. Use CSV or JSON.")
            
            datasets[source_id] = data
        
        # Reconcile data
        result = reconciliation_engine.reconcile_data(datasets, key_field)
        
        if result.success:
            # Convert reconciled data to JSON
            reconciled_json = result.reconciled_data.to_dict(orient='records')
            
            # Convert unresolved conflicts to JSON
            unresolved_conflicts_json = []
            for conflict in result.unresolved_conflicts:
                unresolved_conflicts_json.append({
                    "id": conflict.id,
                    "field_name": conflict.field_name,
                    "record_key": conflict.record_key,
                    "conflicting_values": conflict.conflicting_values,
                    "status": conflict.status.value,
                    "detected_at": conflict.detected_at.isoformat()
                })
            
            return {
                "success": True,
                "reconciled_data": reconciled_json,
                "conflicts_detected": result.conflicts_detected,
                "conflicts_resolved": result.conflicts_resolved,
                "unresolved_conflicts": unresolved_conflicts_json,
                "processing_time": result.processing_time,
                "quality_metrics": result.quality_metrics
            }
        else:
            return {
                "success": False,
                "error": "Reconciliation failed",
                "processing_time": result.processing_time
            }
            
    except Exception as e:
        logger.error(f"Failed to reconcile datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Quality Alerting Endpoints
@router.post("/alerts/rules")
async def register_alert_rule(rule: AlertRuleModel):
    """Register a new alert rule"""
    try:
        alert_rule = AlertRule(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            dataset_patterns=rule.dataset_patterns,
            severity_threshold=QualitySeverity(rule.severity_threshold),
            score_threshold=rule.score_threshold,
            channels=[AlertChannel(channel) for channel in rule.channels],
            frequency=AlertFrequency(rule.frequency),
            enabled=rule.enabled,
            recipients=rule.recipients,
            custom_conditions=rule.custom_conditions
        )
        
        success = quality_alerting.register_alert_rule(alert_rule)
        
        if success:
            return {"success": True, "message": f"Alert rule {rule.name} registered successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to register alert rule")
            
    except Exception as e:
        logger.error(f"Failed to register alert rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/statistics")
async def get_alert_statistics(days: int = 30):
    """Get alert statistics for the specified period"""
    try:
        stats = quality_alerting.get_alert_statistics(days)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get alert statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Quality Reporting Endpoints
@router.post("/reports/summary")
async def generate_summary_report(
    dataset_names: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    format: str = "json"
):
    """Generate a quality summary report"""
    try:
        # Parse time range if provided
        time_range = None
        if start_date and end_date:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            time_range = (start_dt, end_dt)
        
        # Generate report
        report_format = ReportFormat(format.lower())
        result = quality_reporting.generate_quality_summary_report(
            dataset_names=dataset_names,
            time_range=time_range,
            format=report_format
        )
        
        if result.get("success"):
            if report_format == ReportFormat.HTML:
                return HTMLResponse(content=result["content"])
            else:
                return JSONResponse(content={"report": result["content"], "metadata": result["metadata"]})
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate report"))
            
    except Exception as e:
        logger.error(f"Failed to generate summary report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/trends/{dataset_name}")
async def generate_trend_report(
    dataset_name: str,
    days: int = 30,
    format: str = "json"
):
    """Generate a trend analysis report for a dataset"""
    try:
        report_format = ReportFormat(format.lower())
        result = quality_reporting.generate_trend_analysis_report(
            dataset_name=dataset_name,
            days=days,
            format=report_format
        )
        
        if result.get("success"):
            if report_format == ReportFormat.HTML:
                return HTMLResponse(content=result["content"])
            else:
                return JSONResponse(content={"report": result["content"]})
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to generate trend report"))
            
    except Exception as e:
        logger.error(f"Failed to generate trend report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Profiling Endpoints
@router.post("/profile")
async def profile_dataset(
    dataset_name: str,
    file: UploadFile = File(...)
):
    """Generate a comprehensive data quality profile for uploaded dataset"""
    try:
        # Read uploaded file
        if file.content_type == "text/csv":
            content = await file.read()
            data = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        elif file.content_type == "application/json":
            content = await file.read()
            json_data = json.loads(content.decode('utf-8'))
            data = pd.DataFrame(json_data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        # Generate profile
        profile = data_normalizer.create_data_quality_profile(data, dataset_name)
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to profile dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for the data quality system"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "data_normalizer": "operational",
                "quality_monitor": "operational", 
                "lineage_tracker": "operational",
                "reconciliation_engine": "operational",
                "quality_alerting": "operational",
                "quality_reporting": "operational"
            },
            "statistics": {
                "registered_schemas": len(data_normalizer.schemas),
                "quality_rules": len(quality_monitor.rules),
                "data_assets": len(lineage_tracker.assets),
                "data_sources": len(reconciliation_engine.data_sources),
                "alert_rules": len(quality_alerting.alert_rules)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))