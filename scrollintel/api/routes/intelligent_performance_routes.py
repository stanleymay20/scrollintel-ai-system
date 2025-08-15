"""
API routes for intelligent performance optimization.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from ...core.intelligent_performance_optimizer import (
    get_intelligent_optimizer,
    optimize_performance_for_request,
    predict_system_load,
    DeviceCapability,
    OptimizationStrategy
)
from ...core.never_fail_decorators import never_fail_api_endpoint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])


@router.post("/optimize")
@never_fail_api_endpoint(
    fallback_response={"optimized": False, "message": "Optimization service unavailable"},
    degradation_service="performance_optimization"
)
async def optimize_for_device(request: Request, 
                            user_id: Optional[str] = None,
                            client_hints: Optional[Dict[str, Any]] = None):
    """
    Optimize performance settings for the requesting device.
    """
    user_agent = request.headers.get("user-agent", "")
    
    result = await optimize_performance_for_request(
        user_agent=user_agent,
        client_hints=client_hints or {},
        user_id=user_id
    )
    
    return result


@router.get("/predict-load")
@never_fail_api_endpoint(
    fallback_response={"predicted": False, "message": "Load prediction unavailable"},
    degradation_service="performance_optimization"
)
async def get_load_prediction(horizon_minutes: int = 30):
    """
    Get system load prediction and resource allocation recommendations.
    """
    if horizon_minutes < 1 or horizon_minutes > 1440:  # 1 minute to 24 hours
        raise HTTPException(status_code=400, detail="Invalid prediction horizon")
    
    result = await predict_system_load(horizon_minutes)
    return result


@router.get("/cache/stats")
@never_fail_api_endpoint(
    fallback_response={"available": False, "message": "Cache stats unavailable"},
    degradation_service="performance_optimization"
)
async def get_cache_statistics():
    """
    Get intelligent cache performance statistics.
    """
    optimizer = get_intelligent_optimizer()
    cache_stats = optimizer.cache_manager.get_cache_stats()
    
    return {
        "available": True,
        "cache_stats": cache_stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/cache/optimize")
@never_fail_api_endpoint(
    fallback_response={"optimized": False, "message": "Cache optimization unavailable"},
    degradation_service="performance_optimization"
)
async def optimize_cache_strategy(user_id: Optional[str] = None,
                                context: Optional[Dict[str, Any]] = None):
    """
    Optimize caching strategy with predictive pre-loading.
    """
    optimizer = get_intelligent_optimizer()
    
    cache_optimization = await optimizer.optimize_caching_strategy(
        user_id=user_id,
        context=context or {}
    )
    
    return {
        "optimized": True,
        "optimization": cache_optimization,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/features/adapt")
@never_fail_api_endpoint(
    fallback_response={"adapted": False, "message": "Feature adaptation unavailable"},
    degradation_service="performance_optimization"
)
async def adapt_features_for_device(request: Request,
                                  feature_set: str = "dashboard",
                                  user_id: Optional[str] = None):
    """
    Adapt features based on device capabilities and current performance.
    """
    optimizer = get_intelligent_optimizer()
    user_agent = request.headers.get("user-agent", "")
    
    # Detect device capabilities
    device_profile = await optimizer.device_detector.detect_device_capabilities(user_agent)
    
    # Get current performance
    current_performance = await optimizer._get_current_performance()
    
    # Adapt features
    enabled_features = await optimizer.enhancement_manager.adapt_features_for_device(
        feature_set, device_profile, current_performance
    )
    
    return {
        "adapted": True,
        "feature_set": feature_set,
        "device_type": device_profile.device_type.value,
        "performance_score": device_profile.performance_score,
        "enabled_features": enabled_features,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/feedback/performance")
@never_fail_api_endpoint(
    fallback_response={"recorded": False, "message": "Feedback recording unavailable"},
    degradation_service="performance_optimization"
)
async def record_performance_feedback(feature_set: str,
                                    feature: str,
                                    device_capability: str,
                                    performance_score: float,
                                    user_id: Optional[str] = None):
    """
    Record performance feedback for machine learning optimization.
    """
    if not (0.0 <= performance_score <= 1.0):
        raise HTTPException(status_code=400, detail="Performance score must be between 0.0 and 1.0")
    
    try:
        device_cap = DeviceCapability(device_capability)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid device capability")
    
    optimizer = get_intelligent_optimizer()
    
    await optimizer.enhancement_manager.record_feature_performance(
        feature_set, feature, device_cap, performance_score
    )
    
    return {
        "recorded": True,
        "feature_set": feature_set,
        "feature": feature,
        "device_capability": device_capability,
        "performance_score": performance_score,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/stats")
@never_fail_api_endpoint(
    fallback_response={"available": False, "message": "Statistics unavailable"},
    degradation_service="performance_optimization"
)
async def get_optimization_statistics():
    """
    Get comprehensive optimization statistics and metrics.
    """
    optimizer = get_intelligent_optimizer()
    stats = optimizer.get_optimization_stats()
    
    return {
        "available": True,
        "statistics": stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/strategy/set")
@never_fail_api_endpoint(
    fallback_response={"updated": False, "message": "Strategy update unavailable"},
    degradation_service="performance_optimization"
)
async def set_optimization_strategy(strategy: str):
    """
    Set the optimization strategy (aggressive, balanced, conservative, adaptive).
    """
    try:
        optimization_strategy = OptimizationStrategy(strategy)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid optimization strategy")
    
    optimizer = get_intelligent_optimizer()
    optimizer.current_strategy = optimization_strategy
    
    return {
        "updated": True,
        "strategy": strategy,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/device/profile")
@never_fail_api_endpoint(
    fallback_response={"detected": False, "message": "Device detection unavailable"},
    degradation_service="performance_optimization"
)
async def get_device_profile(request: Request, client_hints: Optional[Dict[str, Any]] = None):
    """
    Get detailed device capability profile.
    """
    optimizer = get_intelligent_optimizer()
    user_agent = request.headers.get("user-agent", "")
    
    device_profile = await optimizer.device_detector.detect_device_capabilities(
        user_agent, client_hints or {}
    )
    
    return {
        "detected": True,
        "device_profile": {
            "device_id": device_profile.device_id,
            "device_type": device_profile.device_type.value,
            "cpu_cores": device_profile.cpu_cores,
            "memory_gb": device_profile.memory_gb,
            "network_speed": device_profile.network_speed,
            "screen_resolution": device_profile.screen_resolution,
            "supports_webgl": device_profile.supports_webgl,
            "supports_webworkers": device_profile.supports_webworkers,
            "is_mobile": device_profile.is_mobile,
            "performance_score": device_profile.performance_score,
            "last_updated": device_profile.last_updated.isoformat()
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/resources/allocation")
@never_fail_api_endpoint(
    fallback_response={"available": False, "message": "Resource allocation unavailable"},
    degradation_service="performance_optimization"
)
async def get_current_resource_allocation():
    """
    Get current resource allocation status.
    """
    optimizer = get_intelligent_optimizer()
    
    # Get latest resource allocation
    latest_allocation = None
    if optimizer.resource_allocations:
        latest_key = max(optimizer.resource_allocations.keys())
        latest_allocation = optimizer.resource_allocations[latest_key]
    
    return {
        "available": True,
        "current_allocation": latest_allocation,
        "allocation_history_count": len(optimizer.resource_allocations),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/preload/schedule")
@never_fail_api_endpoint(
    fallback_response={"scheduled": False, "message": "Preload scheduling unavailable"},
    degradation_service="performance_optimization"
)
async def schedule_cache_preload(cache_keys: List[str], 
                               user_id: Optional[str] = None,
                               priority: float = 1.0):
    """
    Schedule cache items for predictive preloading.
    """
    if not cache_keys:
        raise HTTPException(status_code=400, detail="No cache keys provided")
    
    if not (0.0 <= priority <= 10.0):
        raise HTTPException(status_code=400, detail="Priority must be between 0.0 and 10.0")
    
    optimizer = get_intelligent_optimizer()
    
    # Add items to preload queue
    scheduled_count = 0
    for key in cache_keys[:10]:  # Limit to 10 items
        try:
            await optimizer.cache_manager.preload_queue.put(key)
            scheduled_count += 1
        except Exception as e:
            logger.warning(f"Failed to schedule preload for key {key}: {e}")
    
    return {
        "scheduled": scheduled_count > 0,
        "scheduled_count": scheduled_count,
        "total_requested": len(cache_keys),
        "user_id": user_id,
        "priority": priority,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health")
@never_fail_api_endpoint(
    fallback_response={"healthy": False, "message": "Health check unavailable"},
    degradation_service="performance_optimization"
)
async def get_optimization_health():
    """
    Get health status of the optimization system.
    """
    optimizer = get_intelligent_optimizer()
    
    # Check various components
    health_status = {
        "device_detector": len(optimizer.device_detector.device_profiles) > 0,
        "load_predictor": len(optimizer.load_predictor.historical_metrics) > 0,
        "cache_manager": len(optimizer.cache_manager.cache) >= 0,
        "enhancement_manager": len(optimizer.enhancement_manager.enhancement_configs) > 0
    }
    
    overall_health = all(health_status.values())
    
    return {
        "healthy": overall_health,
        "components": health_status,
        "current_strategy": optimizer.current_strategy.value,
        "optimization_active": True,
        "timestamp": datetime.utcnow().isoformat()
    }