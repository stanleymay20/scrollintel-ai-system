"""
API routes for Predictive Failure Prevention Engine
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from ...core.predictive_failure_prevention import (
    predictive_engine,
    start_predictive_prevention,
    stop_predictive_prevention,
    get_prevention_status,
    get_health_report,
    PredictionConfidence,
    AnomalyType,
    ScalingAction
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predictive-failure", tags=["Predictive Failure Prevention"])


@router.get("/status")
async def get_engine_status():
    """Get current status of the predictive failure prevention engine"""
    try:
        status = get_prevention_status()
        return {
            "success": True,
            "data": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_engine(background_tasks: BackgroundTasks):
    """Start the predictive failure prevention engine"""
    try:
        if predictive_engine.running:
            return {
                "success": True,
                "message": "Engine is already running",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        background_tasks.add_task(start_predictive_prevention)
        
        return {
            "success": True,
            "message": "Predictive failure prevention engine started",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_engine(background_tasks: BackgroundTasks):
    """Stop the predictive failure prevention engine"""
    try:
        if not predictive_engine.running:
            return {
                "success": True,
                "message": "Engine is already stopped",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        background_tasks.add_task(stop_predictive_prevention)
        
        return {
            "success": True,
            "message": "Predictive failure prevention engine stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-report")
async def get_comprehensive_health_report():
    """Get comprehensive health report with predictions and recommendations"""
    try:
        report = await get_health_report()
        return {
            "success": True,
            "data": report,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting health report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/current")
async def get_current_metrics():
    """Get current system health metrics"""
    try:
        metrics = await predictive_engine.health_monitor.collect_comprehensive_metrics()
        if not metrics:
            raise HTTPException(status_code=503, detail="Unable to collect metrics")
        
        return {
            "success": True,
            "data": {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "disk_usage": metrics.disk_usage,
                "response_time_avg": metrics.response_time_avg,
                "response_time_p95": metrics.response_time_p95,
                "error_rate": metrics.error_rate,
                "request_rate": metrics.request_rate,
                "database_query_time": metrics.database_query_time,
                "user_sessions": metrics.user_sessions,
                "agent_processing_time": metrics.agent_processing_time
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_recent_anomalies(hours: int = 24):
    """Get recent anomalies detected by the system"""
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_anomalies = [
            {
                "anomaly_type": anomaly.anomaly_type.value,
                "confidence": anomaly.confidence.value,
                "timestamp": anomaly.timestamp.isoformat(),
                "affected_metrics": anomaly.affected_metrics,
                "anomaly_score": anomaly.anomaly_score,
                "description": anomaly.description,
                "recommended_actions": anomaly.recommended_actions,
                "predicted_impact": anomaly.predicted_impact,
                "time_to_failure": anomaly.time_to_failure.total_seconds() if anomaly.time_to_failure else None
            }
            for anomaly in predictive_engine.health_monitor.anomalies
            if anomaly.timestamp > cutoff_time
        ]
        
        return {
            "success": True,
            "data": {
                "anomalies": recent_anomalies,
                "count": len(recent_anomalies),
                "time_window_hours": hours
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recent anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions")
async def get_failure_predictions():
    """Get current failure predictions"""
    try:
        # Get current metrics for prediction
        current_metrics = await predictive_engine.health_monitor.collect_comprehensive_metrics()
        if not current_metrics:
            raise HTTPException(status_code=503, detail="Unable to collect metrics for prediction")
        
        predictions = await predictive_engine.failure_predictor.predict_failures(current_metrics)
        
        prediction_data = [
            {
                "failure_type": pred.failure_type.value,
                "confidence": pred.confidence.value,
                "predicted_time": pred.predicted_time.isoformat(),
                "affected_components": pred.affected_components,
                "root_cause_analysis": pred.root_cause_analysis,
                "prevention_actions": pred.prevention_actions,
                "impact_assessment": pred.impact_assessment,
                "probability": pred.probability
            }
            for pred in predictions
        ]
        
        return {
            "success": True,
            "data": {
                "predictions": prediction_data,
                "count": len(prediction_data),
                "based_on_metrics_timestamp": current_metrics.timestamp.isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting failure predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dependencies")
async def get_dependency_health():
    """Get health status of all monitored dependencies"""
    try:
        dependency_status = predictive_engine.dependency_monitor.get_dependency_status()
        
        return {
            "success": True,
            "data": {
                "dependencies": dependency_status,
                "total_dependencies": len(dependency_status),
                "healthy_count": len([d for d in dependency_status.values() if d['status'] == 'healthy']),
                "degraded_count": len([d for d in dependency_status.values() if d['status'] == 'degraded']),
                "failed_count": len([d for d in dependency_status.values() if d['status'] == 'failed'])
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting dependency health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dependencies/{dependency_name}/check")
async def check_specific_dependency(dependency_name: str):
    """Manually trigger health check for a specific dependency"""
    try:
        if dependency_name not in predictive_engine.dependency_monitor.dependencies:
            raise HTTPException(status_code=404, detail=f"Dependency '{dependency_name}' not found")
        
        health = await predictive_engine.dependency_monitor.check_dependency_health(dependency_name)
        
        return {
            "success": True,
            "data": {
                "service_name": health.service_name,
                "endpoint": health.endpoint,
                "status": health.status,
                "response_time": health.response_time,
                "error_rate": health.error_rate,
                "consecutive_failures": health.consecutive_failures,
                "availability": health.availability,
                "last_check": health.last_check.isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking dependency {dependency_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resource-optimizations")
async def get_resource_optimizations():
    """Get current resource optimization recommendations"""
    try:
        # Get current metrics and predictions
        current_metrics = await predictive_engine.health_monitor.collect_comprehensive_metrics()
        if not current_metrics:
            raise HTTPException(status_code=503, detail="Unable to collect metrics")
        
        predictions = await predictive_engine.failure_predictor.predict_failures(current_metrics)
        optimizations = await predictive_engine.resource_scaler.analyze_resource_needs(current_metrics, predictions)
        
        optimization_data = [
            {
                "resource_type": opt.resource_type,
                "current_usage": opt.current_usage,
                "predicted_usage": opt.predicted_usage,
                "recommended_action": opt.recommended_action.value,
                "urgency": opt.urgency.value,
                "estimated_impact": opt.estimated_impact,
                "cost_benefit": opt.cost_benefit
            }
            for opt in optimizations
        ]
        
        return {
            "success": True,
            "data": {
                "optimizations": optimization_data,
                "count": len(optimization_data),
                "urgent_count": len([o for o in optimizations if o.urgency in [PredictionConfidence.HIGH, PredictionConfidence.CRITICAL]]),
                "based_on_metrics_timestamp": current_metrics.timestamp.isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting resource optimizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prevention-history")
async def get_prevention_history(limit: int = 50):
    """Get history of preventive actions taken"""
    try:
        history = predictive_engine.prevention_history[-limit:] if predictive_engine.prevention_history else []
        
        return {
            "success": True,
            "data": {
                "history": [
                    {
                        "timestamp": entry["timestamp"].isoformat(),
                        "actions": entry["actions"],
                        "anomaly_count": entry["anomaly_count"],
                        "prediction_count": entry["prediction_count"],
                        "optimization_count": entry["optimization_count"]
                    }
                    for entry in history
                ],
                "count": len(history),
                "total_actions": sum(len(entry["actions"]) for entry in history)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting prevention history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-performance")
async def get_model_performance():
    """Get performance metrics of prediction models"""
    try:
        return {
            "success": True,
            "data": {
                "trained_models": len(predictive_engine.failure_predictor.models),
                "training_data_points": len(predictive_engine.failure_predictor.failure_history),
                "feature_history_size": len(predictive_engine.failure_predictor.feature_history),
                "anomaly_detector_trained": predictive_engine.health_monitor.is_trained,
                "metrics_history_size": len(predictive_engine.health_monitor.metrics_history),
                "prediction_accuracy": predictive_engine.failure_predictor.prediction_accuracy
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-models")
async def trigger_model_training(background_tasks: BackgroundTasks):
    """Manually trigger model training with current data"""
    try:
        if len(predictive_engine.failure_predictor.failure_history) < 5:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient training data. Need at least 5 failure events."
            )
        
        background_tasks.add_task(predictive_engine.failure_predictor._retrain_models)
        
        return {
            "success": True,
            "message": "Model training initiated",
            "training_data_points": len(predictive_engine.failure_predictor.failure_history),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard-data")
async def get_dashboard_data():
    """Get comprehensive data for the predictive failure prevention dashboard"""
    try:
        # Get all relevant data
        status = get_prevention_status()
        health_report = await get_health_report()
        
        # Get recent metrics for trending
        recent_metrics = list(predictive_engine.health_monitor.metrics_history)[-20:] if predictive_engine.health_monitor.metrics_history else []
        
        metrics_trend = [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "disk_usage": m.disk_usage,
                "response_time_avg": m.response_time_avg,
                "error_rate": m.error_rate
            }
            for m in recent_metrics
        ]
        
        # Get recent anomalies
        recent_anomalies = [
            {
                "type": a.anomaly_type.value,
                "confidence": a.confidence.value,
                "timestamp": a.timestamp.isoformat(),
                "description": a.description
            }
            for a in predictive_engine.health_monitor.anomalies[-10:]
        ]
        
        return {
            "success": True,
            "data": {
                "engine_status": status,
                "current_metrics": health_report.get("current_metrics"),
                "metrics_trend": metrics_trend,
                "recent_anomalies": recent_anomalies,
                "failure_predictions": health_report.get("failure_predictions", []),
                "dependency_health": health_report.get("dependency_health", {}),
                "prevention_actions": health_report.get("prevention_history", [])
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_active_alerts():
    """Get currently active alerts and warnings"""
    try:
        alerts = []
        
        # Check for critical anomalies
        critical_anomalies = [
            a for a in predictive_engine.health_monitor.anomalies
            if a.confidence == PredictionConfidence.CRITICAL and 
               a.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        for anomaly in critical_anomalies:
            alerts.append({
                "type": "critical_anomaly",
                "severity": "critical",
                "title": f"Critical Anomaly: {anomaly.anomaly_type.value}",
                "description": anomaly.description,
                "timestamp": anomaly.timestamp.isoformat(),
                "actions": anomaly.recommended_actions
            })
        
        # Check for failed dependencies
        dependency_status = predictive_engine.dependency_monitor.get_dependency_status()
        failed_deps = [name for name, status in dependency_status.items() if status['status'] == 'failed']
        
        for dep_name in failed_deps:
            alerts.append({
                "type": "dependency_failure",
                "severity": "high",
                "title": f"Dependency Failed: {dep_name}",
                "description": f"Dependency {dep_name} is currently unavailable",
                "timestamp": dependency_status[dep_name]['last_check'],
                "actions": ["Check dependency health", "Trigger failover if available"]
            })
        
        # Check for high-confidence failure predictions
        current_metrics = await predictive_engine.health_monitor.collect_comprehensive_metrics()
        if current_metrics:
            predictions = await predictive_engine.failure_predictor.predict_failures(current_metrics)
            high_conf_predictions = [p for p in predictions if p.confidence in [PredictionConfidence.HIGH, PredictionConfidence.CRITICAL]]
            
            for prediction in high_conf_predictions:
                alerts.append({
                    "type": "failure_prediction",
                    "severity": "high" if prediction.confidence == PredictionConfidence.HIGH else "critical",
                    "title": f"Predicted Failure: {prediction.failure_type.value}",
                    "description": f"Failure predicted at {prediction.predicted_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "actions": prediction.prevention_actions
                })
        
        return {
            "success": True,
            "data": {
                "alerts": alerts,
                "count": len(alerts),
                "critical_count": len([a for a in alerts if a["severity"] == "critical"]),
                "high_count": len([a for a in alerts if a["severity"] == "high"])
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))