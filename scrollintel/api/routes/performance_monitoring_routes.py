"""
API routes for performance monitoring and optimization.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

from ...core.database import get_db
from ...models.performance_models import (
    PerformanceMetrics, ResourceUsage, SLAViolation, PerformanceAlert,
    OptimizationRecommendation, PerformanceTuningConfig,
    PerformanceMetricsCreate, PerformanceMetricsResponse,
    ResourceUsageCreate, SLAViolationCreate, OptimizationRecommendationCreate,
    PerformanceTuningConfigCreate
)
from ...engines.performance_monitoring_engine import PerformanceMonitoringEngine

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])

# Initialize performance monitoring engine
monitoring_engine = PerformanceMonitoringEngine()

@router.post("/monitoring/start")
async def start_monitoring(
    pipeline_id: str,
    execution_id: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Start performance monitoring for a pipeline execution."""
    try:
        result = await monitoring_engine.start_monitoring(pipeline_id, execution_id)
        
        # Start background monitoring task
        background_tasks.add_task(
            _background_monitoring_task,
            result["metrics_id"]
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Performance monitoring started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/monitoring/stop/{metrics_id}")
async def stop_monitoring(metrics_id: int) -> Dict[str, Any]:
    """Stop performance monitoring and generate final report."""
    try:
        result = await monitoring_engine.stop_monitoring(metrics_id)
        
        return {
            "success": True,
            "data": result,
            "message": "Performance monitoring stopped successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@router.get("/metrics/{pipeline_id}")
async def get_pipeline_metrics(
    pipeline_id: str,
    hours: int = Query(24, description="Time range in hours"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get performance metrics for a specific pipeline."""
    try:
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        metrics = db.query(PerformanceMetrics).filter(
            and_(
                PerformanceMetrics.pipeline_id == pipeline_id,
                PerformanceMetrics.start_time >= start_time
            )
        ).order_by(desc(PerformanceMetrics.start_time)).all()
        
        return {
            "success": True,
            "data": {
                "pipeline_id": pipeline_id,
                "time_range_hours": hours,
                "metrics_count": len(metrics),
                "metrics": [
                    {
                        "id": m.id,
                        "execution_id": m.execution_id,
                        "start_time": m.start_time,
                        "end_time": m.end_time,
                        "duration_seconds": m.duration_seconds,
                        "cpu_usage_percent": m.cpu_usage_percent,
                        "memory_usage_mb": m.memory_usage_mb,
                        "records_processed": m.records_processed,
                        "records_per_second": m.records_per_second,
                        "error_count": m.error_count,
                        "error_rate": m.error_rate,
                        "total_cost": m.total_cost
                    } for m in metrics
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/dashboard")
async def get_performance_dashboard(
    pipeline_id: Optional[str] = Query(None, description="Filter by pipeline ID"),
    hours: int = Query(24, description="Time range in hours")
) -> Dict[str, Any]:
    """Get performance dashboard data."""
    try:
        dashboard_data = await monitoring_engine.get_performance_dashboard_data(
            pipeline_id=pipeline_id,
            time_range_hours=hours
        )
        
        return {
            "success": True,
            "data": dashboard_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@router.get("/alerts")
async def get_active_alerts(
    pipeline_id: Optional[str] = Query(None, description="Filter by pipeline ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get active performance alerts."""
    try:
        query = db.query(PerformanceAlert).filter(
            PerformanceAlert.acknowledged == False
        )
        
        if pipeline_id:
            query = query.join(SLAViolation).join(PerformanceMetrics).filter(
                PerformanceMetrics.pipeline_id == pipeline_id
            )
        
        if severity:
            query = query.filter(PerformanceAlert.alert_level == severity)
        
        alerts = query.order_by(desc(PerformanceAlert.created_at)).all()
        
        return {
            "success": True,
            "data": {
                "alerts_count": len(alerts),
                "alerts": [
                    {
                        "id": a.id,
                        "alert_type": a.alert_type,
                        "alert_level": a.alert_level,
                        "alert_message": a.alert_message,
                        "created_at": a.created_at,
                        "escalation_level": a.escalation_level,
                        "notification_sent": a.notification_sent
                    } for a in alerts
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    acknowledged_by: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Acknowledge a performance alert."""
    try:
        alert = db.query(PerformanceAlert).filter(
            PerformanceAlert.id == alert_id
        ).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "success": True,
            "data": {
                "alert_id": alert_id,
                "acknowledged_by": acknowledged_by,
                "acknowledged_at": alert.acknowledged_at
            },
            "message": "Alert acknowledged successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.get("/recommendations/{pipeline_id}")
async def get_optimization_recommendations(
    pipeline_id: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get optimization recommendations for a pipeline."""
    try:
        query = db.query(OptimizationRecommendation).filter(
            OptimizationRecommendation.pipeline_id == pipeline_id
        )
        
        if status:
            query = query.filter(OptimizationRecommendation.status == status)
        
        if priority:
            query = query.filter(OptimizationRecommendation.priority == priority)
        
        recommendations = query.order_by(
            desc(OptimizationRecommendation.created_at)
        ).all()
        
        return {
            "success": True,
            "data": {
                "pipeline_id": pipeline_id,
                "recommendations_count": len(recommendations),
                "recommendations": [
                    {
                        "id": r.id,
                        "recommendation_type": r.recommendation_type,
                        "priority": r.priority,
                        "title": r.title,
                        "description": r.description,
                        "expected_improvement": r.expected_improvement,
                        "estimated_cost_savings": r.estimated_cost_savings,
                        "implementation_effort": r.implementation_effort,
                        "status": r.status,
                        "created_at": r.created_at
                    } for r in recommendations
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.post("/recommendations")
async def create_optimization_recommendation(
    recommendation: OptimizationRecommendationCreate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Create a new optimization recommendation."""
    try:
        db_recommendation = OptimizationRecommendation(**recommendation.dict())
        db.add(db_recommendation)
        db.commit()
        db.refresh(db_recommendation)
        
        return {
            "success": True,
            "data": {
                "id": db_recommendation.id,
                "pipeline_id": db_recommendation.pipeline_id,
                "recommendation_type": db_recommendation.recommendation_type,
                "priority": db_recommendation.priority,
                "title": db_recommendation.title,
                "created_at": db_recommendation.created_at
            },
            "message": "Optimization recommendation created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create recommendation: {str(e)}")

@router.put("/recommendations/{recommendation_id}/status")
async def update_recommendation_status(
    recommendation_id: int,
    status: str,
    implemented_by: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update the status of an optimization recommendation."""
    try:
        recommendation = db.query(OptimizationRecommendation).filter(
            OptimizationRecommendation.id == recommendation_id
        ).first()
        
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        recommendation.status = status
        recommendation.updated_at = datetime.utcnow()
        
        if status == "implemented":
            recommendation.implemented_at = datetime.utcnow()
            recommendation.implemented_by = implemented_by
        
        db.commit()
        
        return {
            "success": True,
            "data": {
                "recommendation_id": recommendation_id,
                "status": status,
                "updated_at": recommendation.updated_at
            },
            "message": "Recommendation status updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update recommendation: {str(e)}")

@router.get("/tuning/{pipeline_id}")
async def get_tuning_config(
    pipeline_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get performance tuning configuration for a pipeline."""
    try:
        config = db.query(PerformanceTuningConfig).filter(
            PerformanceTuningConfig.pipeline_id == pipeline_id,
            PerformanceTuningConfig.is_active == True
        ).first()
        
        if not config:
            raise HTTPException(status_code=404, detail="Tuning configuration not found")
        
        return {
            "success": True,
            "data": {
                "id": config.id,
                "pipeline_id": config.pipeline_id,
                "auto_scaling_enabled": config.auto_scaling_enabled,
                "min_instances": config.min_instances,
                "max_instances": config.max_instances,
                "target_cpu_utilization": config.target_cpu_utilization,
                "target_memory_utilization": config.target_memory_utilization,
                "latency_threshold_ms": config.latency_threshold_ms,
                "cost_optimization_enabled": config.cost_optimization_enabled,
                "last_tuned": config.last_tuned,
                "created_at": config.created_at
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tuning config: {str(e)}")

@router.post("/tuning")
async def create_tuning_config(
    config: PerformanceTuningConfigCreate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Create performance tuning configuration."""
    try:
        # Deactivate existing configs for the pipeline
        db.query(PerformanceTuningConfig).filter(
            PerformanceTuningConfig.pipeline_id == config.pipeline_id
        ).update({"is_active": False})
        
        # Create new config
        db_config = PerformanceTuningConfig(**config.dict())
        db.add(db_config)
        db.commit()
        db.refresh(db_config)
        
        return {
            "success": True,
            "data": {
                "id": db_config.id,
                "pipeline_id": db_config.pipeline_id,
                "created_at": db_config.created_at
            },
            "message": "Tuning configuration created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create tuning config: {str(e)}")

@router.post("/tuning/{pipeline_id}/apply")
async def apply_auto_tuning(pipeline_id: str) -> Dict[str, Any]:
    """Apply automated performance tuning for a pipeline."""
    try:
        result = await monitoring_engine.apply_auto_tuning(pipeline_id)
        
        return {
            "success": True,
            "data": result,
            "message": "Auto-tuning applied successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply auto-tuning: {str(e)}")

@router.get("/cost-analysis/{pipeline_id}")
async def get_cost_analysis(
    pipeline_id: str,
    days: int = Query(7, description="Number of days for cost analysis"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get cost analysis for a pipeline."""
    try:
        start_time = datetime.utcnow() - timedelta(days=days)
        
        metrics = db.query(PerformanceMetrics).filter(
            and_(
                PerformanceMetrics.pipeline_id == pipeline_id,
                PerformanceMetrics.start_time >= start_time,
                PerformanceMetrics.total_cost.isnot(None)
            )
        ).all()
        
        total_cost = sum(m.total_cost or 0 for m in metrics)
        avg_cost_per_execution = total_cost / max(len(metrics), 1)
        
        # Calculate cost trends
        daily_costs = {}
        for metric in metrics:
            date_key = metric.start_time.date().isoformat()
            if date_key not in daily_costs:
                daily_costs[date_key] = 0
            daily_costs[date_key] += metric.total_cost or 0
        
        return {
            "success": True,
            "data": {
                "pipeline_id": pipeline_id,
                "analysis_period_days": days,
                "total_executions": len(metrics),
                "total_cost": round(total_cost, 2),
                "avg_cost_per_execution": round(avg_cost_per_execution, 2),
                "daily_costs": daily_costs,
                "cost_trend": "increasing" if len(daily_costs) > 1 and 
                             list(daily_costs.values())[-1] > list(daily_costs.values())[0] else "stable"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost analysis: {str(e)}")

async def _background_monitoring_task(metrics_id: int):
    """Background task for continuous monitoring."""
    try:
        while monitoring_engine.monitoring_active:
            await monitoring_engine.collect_metrics(metrics_id)
            await asyncio.sleep(monitoring_engine.monitoring_interval)
    except Exception as e:
        logger.error(f"Background monitoring task error: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for performance monitoring service."""
    return {
        "status": "healthy",
        "service": "performance_monitoring",
        "monitoring_active": monitoring_engine.monitoring_active,
        "timestamp": datetime.utcnow()
    }