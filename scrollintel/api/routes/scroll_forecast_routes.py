"""
API routes for ScrollForecast Engine.
Provides endpoints for time series forecasting and analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Dict, Any, List, Optional
import pandas as pd
import json
import logging

from ...engines.scroll_forecast_engine import ScrollForecastEngine
from ...security.auth import get_current_user
from ...models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["forecast"])

# Global engine instance
forecast_engine = None


async def get_forecast_engine() -> ScrollForecastEngine:
    """Get or create the ScrollForecast engine instance."""
    global forecast_engine
    if forecast_engine is None:
        forecast_engine = ScrollForecastEngine()
        await forecast_engine.initialize()
    return forecast_engine


@router.post("/create")
async def create_forecast(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    engine: ScrollForecastEngine = Depends(get_forecast_engine)
):
    """
    Create time series forecast.
    
    Request body should contain:
    - data: Time series data (list of dicts or DataFrame-like structure)
    - date_column: Name of the date column (default: "date")
    - value_column: Name of the value column (default: "value")
    - forecast_periods: Number of periods to forecast (default: 30)
    - models: List of models to use (default: auto-select)
    - confidence_level: Confidence level for intervals (default: 0.95)
    - forecast_name: Name for the forecast (optional)
    """
    try:
        # Validate required fields
        if "data" not in request:
            raise HTTPException(status_code=400, detail="Time series data is required")
        
        # Set default values
        forecast_request = {
            "action": "forecast",
            "data": request["data"],
            "date_column": request.get("date_column", "date"),
            "value_column": request.get("value_column", "value"),
            "forecast_periods": request.get("forecast_periods", 30),
            "models": request.get("models", ["auto"]),
            "confidence_level": request.get("confidence_level", 0.95),
            "forecast_name": request.get("forecast_name")
        }
        
        # Process forecast
        result = await engine.process(forecast_request, {})
        
        return {
            "success": True,
            "forecast": result,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Error creating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_time_series(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    engine: ScrollForecastEngine = Depends(get_forecast_engine)
):
    """
    Analyze time series data characteristics.
    
    Request body should contain:
    - data: Time series data
    - date_column: Name of the date column (default: "date")
    - value_column: Name of the value column (default: "value")
    """
    try:
        if "data" not in request:
            raise HTTPException(status_code=400, detail="Time series data is required")
        
        analysis_request = {
            "action": "analyze",
            "data": request["data"],
            "date_column": request.get("date_column", "date"),
            "value_column": request.get("value_column", "value")
        }
        
        result = await engine.process(analysis_request, {})
        
        return {
            "success": True,
            "analysis": result,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Error analyzing time series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decompose")
async def decompose_time_series(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    engine: ScrollForecastEngine = Depends(get_forecast_engine)
):
    """
    Perform time series decomposition.
    
    Request body should contain:
    - data: Time series data
    - date_column: Name of the date column (default: "date")
    - value_column: Name of the value column (default: "value")
    - type: Decomposition type - "additive" or "multiplicative" (default: "additive")
    """
    try:
        if "data" not in request:
            raise HTTPException(status_code=400, detail="Time series data is required")
        
        decompose_request = {
            "action": "decompose",
            "data": request["data"],
            "date_column": request.get("date_column", "date"),
            "value_column": request.get("value_column", "value"),
            "type": request.get("type", "additive")
        }
        
        result = await engine.process(decompose_request, {})
        
        return {
            "success": True,
            "decomposition": result,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Error decomposing time series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_trained_models(
    current_user: User = Depends(get_current_user),
    engine: ScrollForecastEngine = Depends(get_forecast_engine)
):
    """List all trained forecast models."""
    try:
        models = {}
        for name, info in engine.trained_models.items():
            models[name] = {
                "model_type": info.get("model_type"),
                "metrics": info.get("metrics"),
                "trained_at": info.get("trained_at").isoformat() if info.get("trained_at") else None,
                "data_points": info.get("data_analysis", {}).get("data_points")
            }
        
        return {
            "success": True,
            "models": models,
            "total_models": len(models),
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_models(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    engine: ScrollForecastEngine = Depends(get_forecast_engine)
):
    """
    Compare performance of multiple trained models.
    
    Request body should contain:
    - model_names: List of model names to compare (optional, defaults to all models)
    """
    try:
        compare_request = {
            "action": "compare",
            "model_names": request.get("model_names", list(engine.trained_models.keys()))
        }
        
        result = await engine.process(compare_request, {})
        
        return {
            "success": True,
            "comparison": result,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-csv")
async def upload_csv_for_forecast(
    file: UploadFile = File(...),
    date_column: str = "date",
    value_column: str = "value",
    forecast_periods: int = 30,
    current_user: User = Depends(get_current_user),
    engine: ScrollForecastEngine = Depends(get_forecast_engine)
):
    """
    Upload CSV file and create forecast.
    
    Parameters:
    - file: CSV file containing time series data
    - date_column: Name of the date column in the CSV
    - value_column: Name of the value column in the CSV
    - forecast_periods: Number of periods to forecast
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        # Create forecast request
        forecast_request = {
            "action": "forecast",
            "data": data,
            "date_column": date_column,
            "value_column": value_column,
            "forecast_periods": forecast_periods,
            "forecast_name": f"csv_forecast_{file.filename.replace('.csv', '')}"
        }
        
        result = await engine.process(forecast_request, {})
        
        return {
            "success": True,
            "forecast": result,
            "file_info": {
                "filename": file.filename,
                "rows_processed": len(data),
                "columns": list(df.columns)
            },
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_engine_status(
    current_user: User = Depends(get_current_user),
    engine: ScrollForecastEngine = Depends(get_forecast_engine)
):
    """Get ScrollForecast engine status and capabilities."""
    try:
        status = engine.get_status()
        metrics = engine.get_metrics()
        
        return {
            "success": True,
            "status": status,
            "metrics": metrics,
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    engine: ScrollForecastEngine = Depends(get_forecast_engine)
):
    """Health check endpoint for ScrollForecast engine."""
    try:
        health = await engine.health_check()
        
        return {
            "healthy": health,
            "engine": "ScrollForecast",
            "status": engine.status.value if hasattr(engine.status, 'value') else str(engine.status)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "healthy": False,
            "engine": "ScrollForecast",
            "error": str(e)
        }


# Cleanup function for engine
async def cleanup_forecast_engine():
    """Cleanup function to be called on app shutdown."""
    global forecast_engine
    if forecast_engine:
        await forecast_engine.cleanup()
        forecast_engine = None