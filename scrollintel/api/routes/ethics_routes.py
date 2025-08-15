"""
API routes for EthicsEngine - AI bias detection and fairness evaluation
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field

from ...engines.ethics_engine import EthicsEngine, ComplianceFramework
from ...security.auth import get_current_user
from ...security.permissions import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ethics", tags=["ethics"])

# Initialize EthicsEngine
ethics_engine = EthicsEngine()

# Pydantic models for request/response
class BiasDetectionRequest(BaseModel):
    """Request model for bias detection"""
    data: List[Dict[str, Any]] = Field(..., description="Dataset for bias analysis")
    predictions: List[float] = Field(..., description="Model predictions")
    protected_attributes: List[str] = Field(..., description="Protected attribute column names")
    true_labels: Optional[List[int]] = Field(None, description="Ground truth labels")
    prediction_probabilities: Optional[List[float]] = Field(None, description="Prediction probabilities")

class TransparencyReportRequest(BaseModel):
    """Request model for transparency report generation"""
    model_info: Dict[str, Any] = Field(..., description="Model information")
    bias_results: Dict[str, Any] = Field(..., description="Bias detection results")
    performance_metrics: Dict[str, Any] = Field(..., description="Model performance metrics")

class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking"""
    framework: ComplianceFramework = Field(..., description="Compliance framework")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    bias_results: Dict[str, Any] = Field(..., description="Bias detection results")

class FairnessThresholdUpdate(BaseModel):
    """Request model for updating fairness thresholds"""
    thresholds: Dict[str, float] = Field(..., description="New fairness thresholds")

@router.on_event("startup")
async def startup_event():
    """Initialize the EthicsEngine on startup"""
    try:
        await ethics_engine.start()
        logger.info("EthicsEngine started successfully")
    except Exception as e:
        logger.error(f"Failed to start EthicsEngine: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup the EthicsEngine on shutdown"""
    try:
        await ethics_engine.stop()
        logger.info("EthicsEngine stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping EthicsEngine: {e}")

@router.get("/status")
async def get_ethics_engine_status(
    current_user: dict = Depends(get_current_user)
):
    """Get the current status of the EthicsEngine"""
    try:
        status = ethics_engine.get_status()
        return {
            "status": "success",
            "engine_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get EthicsEngine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-bias")
async def detect_bias(
    request: BiasDetectionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Detect bias in model predictions across protected attributes
    """
    try:
        # Convert request data to pandas DataFrame
        data_df = pd.DataFrame(request.data)
        predictions = np.array(request.predictions)
        
        # Optional arrays
        true_labels = np.array(request.true_labels) if request.true_labels else None
        pred_probs = np.array(request.prediction_probabilities) if request.prediction_probabilities else None
        
        # Validate data
        if len(data_df) != len(predictions):
            raise HTTPException(
                status_code=400,
                detail="Data and predictions must have the same length"
            )
        
        # Check if protected attributes exist in data
        missing_attrs = [attr for attr in request.protected_attributes if attr not in data_df.columns]
        if missing_attrs:
            raise HTTPException(
                status_code=400,
                detail=f"Protected attributes not found in data: {missing_attrs}"
            )
        
        # Perform bias detection
        result = await ethics_engine.detect_bias(
            data=data_df,
            predictions=predictions,
            protected_attributes=request.protected_attributes,
            true_labels=true_labels,
            prediction_probabilities=pred_probs
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "bias_analysis": result["results"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to detect bias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-bias-detection")
async def upload_bias_detection(
    file: UploadFile = File(...),
    protected_attributes: str = "",
    predictions_column: str = "predictions",
    true_labels_column: Optional[str] = None,
    probabilities_column: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a file for bias detection analysis
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.json')):
            raise HTTPException(
                status_code=400,
                detail="File must be CSV, Excel, or JSON format"
            )
        
        # Read file content
        content = await file.read()
        
        # Parse file based on extension
        if file.filename.endswith('.csv'):
            data_df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.xlsx'):
            data_df = pd.read_excel(pd.io.common.BytesIO(content))
        elif file.filename.endswith('.json'):
            data_df = pd.read_json(pd.io.common.StringIO(content.decode('utf-8')))
        
        # Parse protected attributes
        protected_attrs = [attr.strip() for attr in protected_attributes.split(',') if attr.strip()]
        if not protected_attrs:
            raise HTTPException(
                status_code=400,
                detail="At least one protected attribute must be specified"
            )
        
        # Validate required columns
        if predictions_column not in data_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Predictions column '{predictions_column}' not found in data"
            )
        
        missing_attrs = [attr for attr in protected_attrs if attr not in data_df.columns]
        if missing_attrs:
            raise HTTPException(
                status_code=400,
                detail=f"Protected attributes not found in data: {missing_attrs}"
            )
        
        # Extract data
        predictions = data_df[predictions_column].values
        true_labels = data_df[true_labels_column].values if true_labels_column and true_labels_column in data_df.columns else None
        pred_probs = data_df[probabilities_column].values if probabilities_column and probabilities_column in data_df.columns else None
        
        # Remove prediction columns from feature data
        feature_data = data_df.drop(columns=[col for col in [predictions_column, true_labels_column, probabilities_column] if col and col in data_df.columns])
        
        # Perform bias detection
        result = await ethics_engine.detect_bias(
            data=feature_data,
            predictions=predictions,
            protected_attributes=protected_attrs,
            true_labels=true_labels,
            prediction_probabilities=pred_probs
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "file_info": {
                "filename": file.filename,
                "rows": len(data_df),
                "columns": len(data_df.columns),
                "protected_attributes": protected_attrs
            },
            "bias_analysis": result["results"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded file for bias detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transparency-report")
async def generate_transparency_report(
    request: TransparencyReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate comprehensive AI transparency report
    """
    try:
        result = await ethics_engine.generate_transparency_report(
            model_info=request.model_info,
            bias_results=request.bias_results,
            performance_metrics=request.performance_metrics
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "transparency_report": result["report"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate transparency report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compliance-check")
async def check_compliance(
    request: ComplianceCheckRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Check compliance with regulatory frameworks
    """
    try:
        result = await ethics_engine.check_regulatory_compliance(
            framework=request.framework,
            model_info=request.model_info,
            bias_results=request.bias_results
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "compliance_check": result["compliance"],
            "framework": request.framework.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit-trail")
async def get_audit_trail(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    event_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve audit trail with optional filtering
    """
    try:
        result = await ethics_engine.get_audit_trail(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "audit_trail": result["audit_trail"],
            "total_entries": result["total_entries"],
            "filters": result["filters_applied"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve audit trail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ethical-guidelines")
async def get_ethical_guidelines(
    current_user: dict = Depends(get_current_user)
):
    """
    Get ethical guidelines and principles
    """
    try:
        result = await ethics_engine.get_ethical_guidelines()
        
        return {
            "status": "success",
            "guidelines": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get ethical guidelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/fairness-thresholds")
async def update_fairness_thresholds(
    request: FairnessThresholdUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update fairness thresholds for bias detection
    """
    try:
        result = await ethics_engine.update_fairness_thresholds(request.thresholds)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "message": result["message"],
            "updated_thresholds": result["updated_thresholds"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update fairness thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-frameworks")
async def get_supported_frameworks(
    current_user: dict = Depends(get_current_user)
):
    """
    Get list of supported compliance frameworks
    """
    try:
        frameworks = [framework.value for framework in ComplianceFramework]
        
        return {
            "status": "success",
            "supported_frameworks": frameworks,
            "framework_descriptions": {
                "gdpr": "General Data Protection Regulation (EU)",
                "ccpa": "California Consumer Privacy Act",
                "hipaa": "Health Insurance Portability and Accountability Act",
                "sox": "Sarbanes-Oxley Act",
                "iso_27001": "ISO/IEC 27001 Information Security Management",
                "nist_ai_rmf": "NIST AI Risk Management Framework",
                "eu_ai_act": "European Union AI Act"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported frameworks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint for EthicsEngine"""
    try:
        is_healthy = await ethics_engine.health_check()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "engine": "EthicsEngine",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "engine": "EthicsEngine",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }