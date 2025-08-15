"""
API routes for predictive analytics engine.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ...engines.predictive_engine import PredictiveEngine
from ...models.predictive_models import (
    BusinessMetric, Forecast, ScenarioConfig, ScenarioResult,
    RiskPrediction, PredictionUpdate, BusinessContext,
    ForecastModel, MetricCategory
)

router = APIRouter(prefix="/api/predictive", tags=["predictive"])
logger = logging.getLogger(__name__)

# Global predictive engine instance
predictive_engine = PredictiveEngine()


@router.post("/forecast", response_model=Dict[str, Any])
async def generate_forecast(
    metric_id: str,
    horizon_days: int = Query(30, ge=1, le=365),
    model_type: ForecastModel = ForecastModel.ENSEMBLE,
    historical_data: List[Dict[str, Any]] = None
):
    """
    Generate forecast for a business metric.
    
    Args:
        metric_id: ID of the metric to forecast
        horizon_days: Number of days to forecast
        model_type: Forecasting model to use
        historical_data: Historical metric data
    
    Returns:
        Forecast results with predictions and confidence intervals
    """
    try:
        # Convert historical data to BusinessMetric objects
        if not historical_data:
            raise HTTPException(status_code=400, detail="Historical data is required")
        
        metrics = []
        for data in historical_data:
            metric = BusinessMetric(
                id=data.get('id', metric_id),
                name=data.get('name', f'metric_{metric_id}'),
                category=MetricCategory(data.get('category', 'operational')),
                value=float(data['value']),
                unit=data.get('unit', ''),
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                source=data.get('source', 'api'),
                context=data.get('context', {})
            )
            metrics.append(metric)
        
        # Get current metric (last in historical data)
        current_metric = metrics[-1]
        
        # Generate forecast
        forecast = predictive_engine.forecast_metrics(
            metric=current_metric,
            horizon=horizon_days,
            historical_data=metrics,
            model_type=model_type
        )
        
        # Convert to response format
        response = {
            "metric_id": forecast.metric_id,
            "model_type": forecast.model_type.value,
            "horizon_days": forecast.horizon_days,
            "predictions": forecast.predictions,
            "timestamps": [ts.isoformat() for ts in forecast.timestamps],
            "confidence_intervals": {
                "lower": forecast.confidence_lower,
                "upper": forecast.confidence_upper,
                "level": forecast.confidence_level
            },
            "accuracy_score": forecast.accuracy_score,
            "created_at": forecast.created_at.isoformat()
        }
        
        logger.info(f"Generated forecast for metric {metric_id} using {model_type.value}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenario", response_model=Dict[str, Any])
async def model_scenario(scenario_data: Dict[str, Any]):
    """
    Perform scenario modeling and what-if analysis.
    
    Args:
        scenario_data: Scenario configuration and historical data
    
    Returns:
        Scenario analysis results
    """
    try:
        # Parse scenario configuration
        scenario_config = ScenarioConfig(
            id=scenario_data.get('id', ''),
            name=scenario_data['name'],
            description=scenario_data.get('description', ''),
            parameters=scenario_data['parameters'],
            target_metrics=scenario_data['target_metrics'],
            time_horizon=scenario_data.get('time_horizon', 30),
            created_by=scenario_data.get('created_by', 'api_user'),
            created_at=datetime.utcnow()
        )
        
        # Parse historical data
        historical_data = {}
        for metric_id, data_list in scenario_data.get('historical_data', {}).items():
            metrics = []
            for data in data_list:
                metric = BusinessMetric(
                    id=data.get('id', metric_id),
                    name=data.get('name', f'metric_{metric_id}'),
                    category=MetricCategory(data.get('category', 'operational')),
                    value=float(data['value']),
                    unit=data.get('unit', ''),
                    timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                    source=data.get('source', 'api'),
                    context=data.get('context', {})
                )
                metrics.append(metric)
            historical_data[metric_id] = metrics
        
        # Perform scenario modeling
        result = predictive_engine.model_scenario(scenario_config, historical_data)
        
        # Convert to response format
        response = {
            "scenario_id": result.scenario_id,
            "baseline_forecasts": {
                metric_id: {
                    "predictions": forecast.predictions,
                    "timestamps": [ts.isoformat() for ts in forecast.timestamps],
                    "confidence_level": forecast.confidence_level
                }
                for metric_id, forecast in result.baseline_forecast.items()
            },
            "scenario_forecasts": {
                metric_id: {
                    "predictions": forecast.predictions,
                    "timestamps": [ts.isoformat() for ts in forecast.timestamps],
                    "confidence_level": forecast.confidence_level
                }
                for metric_id, forecast in result.scenario_forecast.items()
            },
            "impact_analysis": result.impact_analysis,
            "recommendations": result.recommendations,
            "confidence_score": result.confidence_score,
            "created_at": result.created_at.isoformat()
        }
        
        logger.info(f"Completed scenario modeling for {scenario_config.name}")
        return response
        
    except Exception as e:
        logger.error(f"Error in scenario modeling: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risks", response_model=Dict[str, Any])
async def predict_risks(risk_data: Dict[str, Any]):
    """
    Predict business risks with early warning systems.
    
    Args:
        risk_data: Current metrics, historical data, and business context
    
    Returns:
        Risk predictions and summary
    """
    try:
        # Parse business context
        context_data = risk_data.get('context', {})
        context = BusinessContext(
            industry=context_data.get('industry', 'technology'),
            company_size=context_data.get('company_size', 'medium'),
            market_conditions=context_data.get('market_conditions', {}),
            seasonal_factors=context_data.get('seasonal_factors', {}),
            external_factors=context_data.get('external_factors', {}),
            historical_patterns=context_data.get('historical_patterns', {})
        )
        
        # Parse current metrics
        current_metrics = []
        for data in risk_data.get('current_metrics', []):
            metric = BusinessMetric(
                id=data.get('id', ''),
                name=data.get('name', ''),
                category=MetricCategory(data.get('category', 'operational')),
                value=float(data['value']),
                unit=data.get('unit', ''),
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                source=data.get('source', 'api'),
                context=data.get('context', {})
            )
            current_metrics.append(metric)
        
        # Parse historical data
        historical_data = {}
        for metric_id, data_list in risk_data.get('historical_data', {}).items():
            metrics = []
            for data in data_list:
                metric = BusinessMetric(
                    id=data.get('id', metric_id),
                    name=data.get('name', f'metric_{metric_id}'),
                    category=MetricCategory(data.get('category', 'operational')),
                    value=float(data['value']),
                    unit=data.get('unit', ''),
                    timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                    source=data.get('source', 'api'),
                    context=data.get('context', {})
                )
                metrics.append(metric)
            historical_data[metric_id] = metrics
        
        # Predict risks
        risks = predictive_engine.predict_risks(context, current_metrics, historical_data)
        
        # Generate risk summary
        risk_summary = predictive_engine.get_risk_summary(risks)
        
        # Convert to response format
        response = {
            "risks": [
                {
                    "id": risk.id,
                    "metric_id": risk.metric_id,
                    "risk_type": risk.risk_type,
                    "risk_level": risk.risk_level.value,
                    "probability": risk.probability,
                    "impact_score": risk.impact_score,
                    "description": risk.description,
                    "early_warning_threshold": risk.early_warning_threshold,
                    "mitigation_strategies": risk.mitigation_strategies,
                    "predicted_date": risk.predicted_date.isoformat() if risk.predicted_date else None,
                    "created_at": risk.created_at.isoformat()
                }
                for risk in risks
            ],
            "summary": risk_summary,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Identified {len(risks)} potential risks")
        return response
        
    except Exception as e:
        logger.error(f"Error predicting risks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update", response_model=Dict[str, Any])
async def update_predictions(update_data: Dict[str, Any]):
    """
    Update predictions based on new data.
    
    Args:
        update_data: New metric data
    
    Returns:
        Prediction updates and notifications
    """
    try:
        # Parse new metrics
        new_metrics = []
        for data in update_data.get('new_data', []):
            metric = BusinessMetric(
                id=data.get('id', ''),
                name=data.get('name', ''),
                category=MetricCategory(data.get('category', 'operational')),
                value=float(data['value']),
                unit=data.get('unit', ''),
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                source=data.get('source', 'api'),
                context=data.get('context', {})
            )
            new_metrics.append(metric)
        
        # Update predictions
        updates = predictive_engine.update_predictions(new_metrics)
        
        # Convert to response format
        response = {
            "updates": [
                {
                    "metric_id": update.metric_id,
                    "change_magnitude": update.change_magnitude,
                    "change_reason": update.change_reason,
                    "previous_forecast": {
                        "predictions": update.previous_forecast.predictions,
                        "confidence_level": update.previous_forecast.confidence_level
                    },
                    "updated_forecast": {
                        "predictions": update.updated_forecast.predictions,
                        "confidence_level": update.updated_forecast.confidence_level
                    },
                    "stakeholders_notified": update.stakeholders_notified,
                    "update_timestamp": update.update_timestamp.isoformat()
                }
                for update in updates
            ],
            "total_updates": len(updates),
            "processing_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Generated {len(updates)} prediction updates")
        return response
        
    except Exception as e:
        logger.error(f"Error updating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{metric_id}/performance", response_model=Dict[str, Any])
async def get_model_performance(metric_id: str):
    """
    Get performance metrics for all models for a specific metric.
    
    Args:
        metric_id: ID of the metric
    
    Returns:
        Model performance metrics
    """
    try:
        performance = predictive_engine.get_model_performance(metric_id)
        
        # Convert to response format
        response = {
            "metric_id": metric_id,
            "models": {
                model_type: {
                    "mae": accuracy.mae,
                    "mape": accuracy.mape,
                    "rmse": accuracy.rmse,
                    "r2_score": accuracy.r2_score,
                    "sample_size": accuracy.sample_size,
                    "evaluation_date": accuracy.evaluation_date.isoformat()
                }
                for model_type, accuracy in performance.items()
            },
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for predictive analytics service."""
    return {
        "status": "healthy",
        "service": "predictive_analytics",
        "timestamp": datetime.utcnow().isoformat(),
        "available_models": [model.value for model in ForecastModel]
    }