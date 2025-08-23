"""
Performance Optimization API Routes

Provides REST API endpoints for performance optimization system management
including caching, load balancing, auto-scaling, and resource forecasting.

Requirements: 4.1, 6.1
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field

from scrollintel.core.performance_optimization import (
    PerformanceOptimizationSystem,
    AgentMetrics,
    CacheStrategy,
    LoadBalancingStrategy
)
from scrollintel.core.config import get_redis_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])

# Pydantic models for API
class AgentMetricsRequest(BaseModel):
    agent_id: str
    cpu_usage: float = Field(..., ge=0, le=100)
    memory_usage: float = Field(..., ge=0, le=100)
    response_time: float = Field(..., ge=0)
    throughput: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=1)
    active_connections: int = Field(..., ge=0)
    queue_length: int = Field(..., ge=0)

class CacheRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = None

class LoadBalancingRequest(BaseModel):
    request_context: Dict[str, Any] = {}

class ScalingConfigRequest(BaseModel):
    min_instances: Optional[int] = Field(None, ge=1)
    max_instances: Optional[int] = Field(None, ge=1)
    cooldown_period: Optional[int] = Field(None, ge=60)

class ForecastRequest(BaseModel):
    hours_ahead: int = Field(1, ge=1, le=168)  # Max 1 week

# Global performance optimization system instance
_performance_system: Optional[PerformanceOptimizationSystem] = None

async def get_performance_system() -> PerformanceOptimizationSystem:
    """Get or create performance optimization system instance"""
    global _performance_system
    if _performance_system is None:
        redis_client = get_redis_client()
        _performance_system = PerformanceOptimizationSystem(redis_client)
        await _performance_system.start()
    return _performance_system

@router.get("/health")
async def get_system_health(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get overall system health status"""
    try:
        report = await system.get_performance_report()
        return {
            "status": "success",
            "data": {
                "health": report.get("system_health", {}),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report")
async def get_performance_report(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get comprehensive performance report"""
    try:
        report = await system.get_performance_report()
        return {
            "status": "success",
            "data": report
        }
    except Exception as e:
        logger.error(f"Performance report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache Management Endpoints
@router.post("/cache/set")
async def set_cache_value(
    request: CacheRequest,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Set value in intelligent cache"""
    try:
        success = await system.cache_manager.set(
            request.key, 
            request.value, 
            request.ttl
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Cache value set for key: {request.key}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to set cache value")
            
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/get/{key}")
async def get_cache_value(
    key: str,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get value from intelligent cache"""
    try:
        value = await system.cache_manager.get(key)
        
        if value is not None:
            return {
                "status": "success",
                "data": {
                    "key": key,
                    "value": value,
                    "found": True
                }
            }
        else:
            return {
                "status": "success",
                "data": {
                    "key": key,
                    "value": None,
                    "found": False
                }
            }
            
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def get_cache_stats(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get cache performance statistics"""
    try:
        stats = system.cache_manager.get_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Load Balancing Endpoints
@router.post("/load-balancer/register-agent")
async def register_agent(
    metrics: AgentMetricsRequest,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Register new agent with load balancer"""
    try:
        agent_metrics = AgentMetrics(
            agent_id=metrics.agent_id,
            cpu_usage=metrics.cpu_usage,
            memory_usage=metrics.memory_usage,
            response_time=metrics.response_time,
            throughput=metrics.throughput,
            error_rate=metrics.error_rate,
            active_connections=metrics.active_connections,
            queue_length=metrics.queue_length,
            last_updated=datetime.now()
        )
        
        system.load_balancer.register_agent(metrics.agent_id, agent_metrics)
        
        return {
            "status": "success",
            "message": f"Agent {metrics.agent_id} registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Agent registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-balancer/update-metrics")
async def update_agent_metrics(
    metrics: AgentMetricsRequest,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Update agent performance metrics"""
    try:
        agent_metrics = AgentMetrics(
            agent_id=metrics.agent_id,
            cpu_usage=metrics.cpu_usage,
            memory_usage=metrics.memory_usage,
            response_time=metrics.response_time,
            throughput=metrics.throughput,
            error_rate=metrics.error_rate,
            active_connections=metrics.active_connections,
            queue_length=metrics.queue_length,
            last_updated=datetime.now()
        )
        
        system.load_balancer.update_agent_metrics(metrics.agent_id, agent_metrics)
        
        return {
            "status": "success",
            "message": f"Metrics updated for agent {metrics.agent_id}"
        }
        
    except Exception as e:
        logger.error(f"Metrics update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-balancer/select-agent")
async def select_optimal_agent(
    request: LoadBalancingRequest,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Select optimal agent for request"""
    try:
        selected_agent = await system.load_balancer.select_agent(request.request_context)
        
        if selected_agent:
            return {
                "status": "success",
                "data": {
                    "selected_agent": selected_agent,
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            return {
                "status": "success",
                "data": {
                    "selected_agent": None,
                    "message": "No healthy agents available"
                }
            }
            
    except Exception as e:
        logger.error(f"Agent selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/load-balancer/stats")
async def get_load_balancer_stats(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get load balancer statistics"""
    try:
        stats = system.load_balancer.get_agent_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Load balancer stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-balancer/train")
async def train_load_balancer(
    background_tasks: BackgroundTasks,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Train ML load balancer model"""
    try:
        background_tasks.add_task(system.load_balancer.train_model)
        
        return {
            "status": "success",
            "message": "Load balancer training started in background"
        }
        
    except Exception as e:
        logger.error(f"Load balancer training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Auto-Scaling Endpoints
@router.get("/scaling/status")
async def get_scaling_status(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get current scaling status"""
    try:
        stats = system.resource_manager.get_scaling_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Scaling status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scaling/configure")
async def configure_scaling(
    config: ScalingConfigRequest,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Configure auto-scaling parameters"""
    try:
        if config.min_instances is not None:
            system.resource_manager.min_instances = config.min_instances
        
        if config.max_instances is not None:
            system.resource_manager.max_instances = config.max_instances
            
        if config.cooldown_period is not None:
            system.resource_manager.cooldown_period = config.cooldown_period
        
        return {
            "status": "success",
            "message": "Scaling configuration updated",
            "data": {
                "min_instances": system.resource_manager.min_instances,
                "max_instances": system.resource_manager.max_instances,
                "cooldown_period": system.resource_manager.cooldown_period
            }
        }
        
    except Exception as e:
        logger.error(f"Scaling configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scaling/evaluate")
async def evaluate_scaling_needs(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Manually trigger scaling evaluation"""
    try:
        # Get current metrics
        metrics = await system._collect_system_metrics()
        
        # Evaluate scaling
        scaling_action = await system.resource_manager.evaluate_scaling(metrics)
        
        if scaling_action:
            return {
                "status": "success",
                "data": {
                    "scaling_action": scaling_action,
                    "message": "Scaling action executed"
                }
            }
        else:
            return {
                "status": "success",
                "data": {
                    "scaling_action": None,
                    "message": "No scaling action needed"
                }
            }
            
    except Exception as e:
        logger.error(f"Scaling evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Forecasting Endpoints
@router.post("/forecasting/predict")
async def predict_resource_demand(
    request: ForecastRequest,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Predict future resource demand"""
    try:
        forecasts = await system.forecaster.forecast_demand(request.hours_ahead)
        
        forecast_data = [
            {
                "timestamp": f.timestamp.isoformat(),
                "predicted_cpu": f.predicted_cpu,
                "predicted_memory": f.predicted_memory,
                "predicted_requests": f.predicted_requests,
                "confidence": f.confidence,
                "scaling_recommendation": f.scaling_recommendation
            }
            for f in forecasts
        ]
        
        return {
            "status": "success",
            "data": {
                "forecasts": forecast_data,
                "hours_ahead": request.hours_ahead,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecasting/stats")
async def get_forecasting_stats(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get forecasting model statistics"""
    try:
        stats = system.forecaster.get_forecasting_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Forecasting stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecasting/train")
async def train_forecasting_models(
    background_tasks: BackgroundTasks,
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Train forecasting models"""
    try:
        background_tasks.add_task(system.forecaster.train_forecasting_models)
        
        return {
            "status": "success",
            "message": "Forecasting model training started in background"
        }
        
    except Exception as e:
        logger.error(f"Forecasting training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Control Endpoints
@router.post("/system/start")
async def start_performance_system(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Start performance optimization system"""
    try:
        if not system._running:
            await system.start()
        
        return {
            "status": "success",
            "message": "Performance optimization system started"
        }
        
    except Exception as e:
        logger.error(f"System start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/stop")
async def stop_performance_system(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Stop performance optimization system"""
    try:
        await system.stop()
        
        return {
            "status": "success",
            "message": "Performance optimization system stopped"
        }
        
    except Exception as e:
        logger.error(f"System stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/current")
async def get_current_metrics(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get current system metrics"""
    try:
        metrics = await system._collect_system_metrics()
        return {
            "status": "success",
            "data": {
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Current metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization/recommendations")
async def get_optimization_recommendations(
    system: PerformanceOptimizationSystem = Depends(get_performance_system)
):
    """Get performance optimization recommendations"""
    try:
        # Get current performance report
        report = await system.get_performance_report()
        
        recommendations = []
        
        # Analyze cache performance
        cache_stats = report.get("cache_stats", {})
        hit_rate = cache_stats.get("hit_rate", 0)
        
        if hit_rate < 0.7:
            recommendations.append({
                "type": "cache",
                "priority": "high",
                "message": f"Cache hit rate is low ({hit_rate:.2%}). Consider increasing cache size or adjusting TTL values.",
                "action": "optimize_cache_strategy"
            })
        
        # Analyze load balancing
        lb_stats = report.get("load_balancer_stats", {})
        healthy_agents = lb_stats.get("healthy_agents", 0)
        total_agents = lb_stats.get("total_agents", 0)
        
        if total_agents > 0 and (healthy_agents / total_agents) < 0.8:
            recommendations.append({
                "type": "load_balancing",
                "priority": "high",
                "message": f"Only {healthy_agents}/{total_agents} agents are healthy. Check agent health and consider scaling.",
                "action": "investigate_agent_health"
            })
        
        # Analyze forecasts for proactive recommendations
        forecasts = report.get("forecasts", [])
        if forecasts:
            next_forecast = forecasts[0]
            if next_forecast.get("predicted_cpu", 0) > 80:
                recommendations.append({
                    "type": "scaling",
                    "priority": "medium",
                    "message": "High CPU usage predicted in the next hour. Consider proactive scaling.",
                    "action": "prepare_scale_up"
                })
        
        return {
            "status": "success",
            "data": {
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))