"""
ScrollInsightRadar API Routes

Provides REST API endpoints for automated pattern detection, trend analysis,
anomaly detection, and insight generation across all data sources.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from typing import Dict, List, Any, Optional
import pandas as pd
import io
import logging
from datetime import datetime

from ...engines.scroll_insight_radar import ScrollInsightRadar
from ...models.schemas import (
    InsightRadarResult, PatternDetectionConfig, 
    TrendAnalysis, AnomalyDetection, InsightNotification
)
from ...security.auth import get_current_user
from ...security.permissions import require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/insight-radar", tags=["ScrollInsightRadar"])

# Initialize the ScrollInsightRadar engine
insight_radar = ScrollInsightRadar()

@router.post("/detect-patterns", response_model=Dict[str, Any])
async def detect_patterns(
    file: UploadFile = File(...),
    config: Optional[PatternDetectionConfig] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Detect patterns across uploaded data with comprehensive analysis.
    
    - **file**: CSV or Excel file containing data to analyze
    - **config**: Optional configuration for pattern detection
    
    Returns comprehensive pattern detection results including:
    - Correlation patterns
    - Seasonal patterns  
    - Clustering patterns
    - Distribution patterns
    - Trend analysis
    - Anomaly detection
    - Ranked insights
    - Business impact score
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV and Excel files are supported"
            )
        
        # Read the uploaded file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            data = pd.read_excel(io.BytesIO(contents))
        
        logger.info(f"Processing pattern detection for file: {file.filename}")
        
        # Perform pattern detection
        results = await insight_radar.detect_patterns(data, config)
        
        # Log the analysis
        logger.info(f"Pattern detection completed for user {current_user.get('user_id')}. "
                   f"Found {len(results.get('insights', []))} insights")
        
        return {
            "success": True,
            "message": "Pattern detection completed successfully",
            "file_name": file.filename,
            "analysis_timestamp": datetime.now().isoformat(),
            "user_id": current_user.get("user_id"),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in pattern detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pattern detection failed: {str(e)}")

@router.post("/analyze-trends", response_model=Dict[str, Any])
async def analyze_trends(
    file: UploadFile = File(...),
    config: Optional[PatternDetectionConfig] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Perform detailed trend analysis with statistical significance testing.
    
    - **file**: CSV or Excel file containing time series data
    - **config**: Optional configuration for trend analysis
    
    Returns trend analysis results including:
    - Linear trend detection
    - Statistical significance testing
    - Trend strength and direction
    - Confidence levels
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV and Excel files are supported"
            )
        
        # Read the uploaded file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            data = pd.read_excel(io.BytesIO(contents))
        
        logger.info(f"Processing trend analysis for file: {file.filename}")
        
        # Perform trend analysis
        results = await insight_radar._analyze_trends(data, config or PatternDetectionConfig())
        
        return {
            "success": True,
            "message": "Trend analysis completed successfully",
            "file_name": file.filename,
            "analysis_timestamp": datetime.now().isoformat(),
            "user_id": current_user.get("user_id"),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.post("/detect-anomalies", response_model=Dict[str, Any])
async def detect_anomalies(
    file: UploadFile = File(...),
    config: Optional[PatternDetectionConfig] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Detect anomalies and unusual patterns in the data.
    
    - **file**: CSV or Excel file containing data to analyze
    - **config**: Optional configuration for anomaly detection
    
    Returns anomaly detection results including:
    - Isolation Forest anomalies
    - Statistical anomalies (Z-score, IQR)
    - Anomaly severity scores
    - Column-specific anomaly analysis
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV and Excel files are supported"
            )
        
        # Read the uploaded file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            data = pd.read_excel(io.BytesIO(contents))
        
        logger.info(f"Processing anomaly detection for file: {file.filename}")
        
        # Perform anomaly detection
        results = await insight_radar._detect_anomalies(data, config or PatternDetectionConfig())
        
        return {
            "success": True,
            "message": "Anomaly detection completed successfully",
            "file_name": file.filename,
            "analysis_timestamp": datetime.now().isoformat(),
            "user_id": current_user.get("user_id"),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@router.post("/generate-insights", response_model=Dict[str, Any])
async def generate_insights(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    send_notifications: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate comprehensive insights with automated ranking and notifications.
    
    - **file**: CSV or Excel file containing data to analyze
    - **send_notifications**: Whether to send notifications for high-priority insights
    
    Returns ranked insights with business impact scores and actionable recommendations.
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV and Excel files are supported"
            )
        
        # Read the uploaded file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            data = pd.read_excel(io.BytesIO(contents))
        
        logger.info(f"Generating insights for file: {file.filename}")
        
        # Perform comprehensive pattern detection
        results = await insight_radar.detect_patterns(data)
        
        # Send notifications if requested
        if send_notifications and results.get("insights"):
            background_tasks.add_task(
                insight_radar.send_insight_notification,
                results["insights"],
                current_user.get("user_id")
            )
        
        return {
            "success": True,
            "message": "Insight generation completed successfully",
            "file_name": file.filename,
            "analysis_timestamp": datetime.now().isoformat(),
            "user_id": current_user.get("user_id"),
            "insights": results.get("insights", []),
            "business_impact_score": results.get("business_impact_score", 0.0),
            "total_insights": len(results.get("insights", [])),
            "high_priority_insights": len([i for i in results.get("insights", []) if i.get("priority") == "high"]),
            "notifications_sent": send_notifications
        }
        
    except Exception as e:
        logger.error(f"Error in insight generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def get_health_status():
    """
    Get health status of the ScrollInsightRadar engine.
    
    Returns current health status, capabilities, and version information.
    """
    try:
        health_status = await insight_radar.get_health_status()
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/capabilities", response_model=Dict[str, Any])
async def get_capabilities():
    """
    Get ScrollInsightRadar capabilities and supported analysis types.
    
    Returns list of supported pattern detection and analysis capabilities.
    """
    try:
        return {
            "engine": insight_radar.name,
            "version": insight_radar.version,
            "capabilities": insight_radar.capabilities,
            "supported_file_types": [".csv", ".xlsx", ".xls"],
            "analysis_types": [
                "correlation_patterns",
                "seasonal_patterns", 
                "clustering_patterns",
                "distribution_patterns",
                "trend_analysis",
                "anomaly_detection",
                "insight_ranking",
                "statistical_significance_testing"
            ],
            "notification_support": True,
            "real_time_processing": True
        }
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

@router.post("/batch-analysis", response_model=Dict[str, Any])
async def batch_analysis(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Perform batch analysis across multiple data files.
    
    - **files**: List of CSV or Excel files to analyze
    
    Returns aggregated insights across all uploaded files.
    """
    try:
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 files allowed for batch analysis"
            )
        
        batch_results = []
        all_insights = []
        
        for file in files:
            # Validate file type
            if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
                continue
            
            # Read the file
            contents = await file.read()
            
            if file.filename.endswith('.csv'):
                data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            else:
                data = pd.read_excel(io.BytesIO(contents))
            
            # Perform pattern detection
            results = await insight_radar.detect_patterns(data)
            
            batch_results.append({
                "file_name": file.filename,
                "insights_count": len(results.get("insights", [])),
                "business_impact_score": results.get("business_impact_score", 0.0),
                "patterns_found": len(results.get("patterns", {})),
                "anomalies_found": results.get("anomalies", {}).get("total_anomalies_found", 0)
            })
            
            all_insights.extend(results.get("insights", []))
        
        # Re-rank all insights across files
        all_insights.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
        for i, insight in enumerate(all_insights):
            insight["global_rank"] = i + 1
        
        return {
            "success": True,
            "message": f"Batch analysis completed for {len(batch_results)} files",
            "analysis_timestamp": datetime.now().isoformat(),
            "user_id": current_user.get("user_id"),
            "files_processed": len(batch_results),
            "batch_results": batch_results,
            "aggregated_insights": all_insights[:50],  # Top 50 insights
            "total_insights": len(all_insights),
            "average_impact_score": sum(r["business_impact_score"] for r in batch_results) / len(batch_results) if batch_results else 0
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")