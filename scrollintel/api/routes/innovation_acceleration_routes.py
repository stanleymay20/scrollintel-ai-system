"""
Innovation Acceleration System API Routes

This module provides REST API endpoints for the innovation acceleration system,
including bottleneck identification, timeline optimization, and acceleration strategies.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional
from datetime import datetime
import logging

from ...engines.innovation_acceleration_system import (
    InnovationAccelerationSystem, AccelerationStrategy, BottleneckAnalysis,
    TimelineOptimization, AccelerationResult, AccelerationType, BottleneckType
)
from ...engines.innovation_pipeline_optimizer import InnovationPipelineOptimizer
from ...models.innovation_pipeline_models import InnovationPipelineItem

router = APIRouter(prefix="/api/v1/innovation-acceleration", tags=["Innovation Acceleration"])
logger = logging.getLogger(__name__)

# Global acceleration system instance
_acceleration_system: Optional[InnovationAccelerationSystem] = None
_pipeline_optimizer: Optional[InnovationPipelineOptimizer] = None


def get_acceleration_system() -> InnovationAccelerationSystem:
    """Get or create acceleration system instance"""
    global _acceleration_system
    if _acceleration_system is None:
        _acceleration_system = InnovationAccelerationSystem()
    return _acceleration_system


def get_pipeline_optimizer() -> InnovationPipelineOptimizer:
    """Get or create pipeline optimizer instance"""
    global _pipeline_optimizer
    if _pipeline_optimizer is None:
        from ...engines.innovation_pipeline_optimizer import InnovationPipelineOptimizer
        _pipeline_optimizer = InnovationPipelineOptimizer()
    return _pipeline_optimizer


@router.post("/accelerate", response_model=dict)
async def accelerate_innovation_development(
    background_tasks: BackgroundTasks,
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system),
    pipeline_optimizer: InnovationPipelineOptimizer = Depends(get_pipeline_optimizer)
):
    """Accelerate innovation development across the pipeline"""
    try:
        # Get current pipeline items
        pipeline_items = pipeline_optimizer.pipeline_items
        
        if not pipeline_items:
            raise HTTPException(
                status_code=400,
                detail="No innovations in pipeline to accelerate"
            )
        
        # Run acceleration
        result = await acceleration_system.accelerate_innovation_development(pipeline_items)
        
        return {
            "success": True,
            "acceleration_id": result.acceleration_id,
            "timestamp": result.timestamp.isoformat(),
            "innovations_accelerated": result.innovations_accelerated,
            "total_time_saved": result.total_time_saved,
            "bottlenecks_resolved": result.bottlenecks_resolved,
            "performance_improvement": result.performance_improvement,
            "cost_efficiency": result.cost_efficiency,
            "strategies_applied": len(result.acceleration_strategies_applied),
            "timeline_optimizations": len(result.timeline_optimizations)
        }
    
    except Exception as e:
        logger.error(f"Error accelerating innovation development: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bottlenecks", response_model=List[dict])
async def identify_bottlenecks(
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system),
    pipeline_optimizer: InnovationPipelineOptimizer = Depends(get_pipeline_optimizer)
):
    """Identify bottlenecks in the innovation pipeline"""
    try:
        pipeline_items = pipeline_optimizer.pipeline_items
        bottlenecks = await acceleration_system.identify_bottlenecks(pipeline_items)
        
        result = []
        for bottleneck in bottlenecks:
            result.append({
                "bottleneck_id": bottleneck.bottleneck_id,
                "innovation_id": bottleneck.innovation_id,
                "bottleneck_type": bottleneck.bottleneck_type.value,
                "affected_stage": bottleneck.affected_stage.value,
                "severity": bottleneck.severity,
                "estimated_delay": bottleneck.estimated_delay,
                "root_causes": bottleneck.root_causes,
                "impact_on_downstream": bottleneck.impact_on_downstream,
                "resolution_strategies_count": len(bottleneck.resolution_strategies)
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error identifying bottlenecks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bottlenecks/{innovation_id}", response_model=List[dict])
async def get_innovation_bottlenecks(
    innovation_id: str,
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system),
    pipeline_optimizer: InnovationPipelineOptimizer = Depends(get_pipeline_optimizer)
):
    """Get bottlenecks for a specific innovation"""
    try:
        if innovation_id not in pipeline_optimizer.pipeline_items:
            raise HTTPException(status_code=404, detail="Innovation not found")
        
        pipeline_items = pipeline_optimizer.pipeline_items
        all_bottlenecks = await acceleration_system.identify_bottlenecks(pipeline_items)
        
        # Filter bottlenecks for specific innovation
        innovation_bottlenecks = [
            bottleneck for bottleneck in all_bottlenecks
            if bottleneck.innovation_id == innovation_id
        ]
        
        result = []
        for bottleneck in innovation_bottlenecks:
            strategies = []
            for strategy in bottleneck.resolution_strategies:
                strategies.append({
                    "id": strategy.id,
                    "acceleration_type": strategy.acceleration_type.value,
                    "expected_time_reduction": strategy.expected_time_reduction,
                    "resource_cost": strategy.resource_cost,
                    "success_probability": strategy.success_probability,
                    "risk_factor": strategy.risk_factor,
                    "prerequisites": strategy.prerequisites,
                    "side_effects": strategy.side_effects
                })
            
            result.append({
                "bottleneck_id": bottleneck.bottleneck_id,
                "bottleneck_type": bottleneck.bottleneck_type.value,
                "affected_stage": bottleneck.affected_stage.value,
                "severity": bottleneck.severity,
                "estimated_delay": bottleneck.estimated_delay,
                "root_causes": bottleneck.root_causes,
                "resolution_strategies": strategies
            })
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting innovation bottlenecks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-timeline/{innovation_id}", response_model=dict)
async def optimize_innovation_timeline(
    innovation_id: str,
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system),
    pipeline_optimizer: InnovationPipelineOptimizer = Depends(get_pipeline_optimizer)
):
    """Optimize timeline for a specific innovation"""
    try:
        if innovation_id not in pipeline_optimizer.pipeline_items:
            raise HTTPException(status_code=404, detail="Innovation not found")
        
        innovation = pipeline_optimizer.pipeline_items[innovation_id]
        optimization = await acceleration_system.optimize_innovation_timeline(innovation)
        
        strategies = []
        for strategy in optimization.acceleration_strategies:
            strategies.append({
                "id": strategy.id,
                "acceleration_type": strategy.acceleration_type.value,
                "target_stage": strategy.target_stage.value,
                "expected_time_reduction": strategy.expected_time_reduction,
                "resource_cost": strategy.resource_cost,
                "success_probability": strategy.success_probability,
                "risk_factor": strategy.risk_factor,
                "prerequisites": strategy.prerequisites,
                "side_effects": strategy.side_effects
            })
        
        return {
            "optimization_id": optimization.optimization_id,
            "innovation_id": optimization.innovation_id,
            "original_timeline": optimization.original_timeline,
            "optimized_timeline": optimization.optimized_timeline,
            "time_savings": optimization.time_savings,
            "time_savings_percentage": (optimization.time_savings / optimization.original_timeline * 100) if optimization.original_timeline > 0 else 0,
            "confidence_level": optimization.confidence_level,
            "risk_assessment": optimization.risk_assessment,
            "acceleration_strategies": strategies
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies", response_model=List[dict])
async def get_acceleration_strategies(
    innovation_id: Optional[str] = None,
    acceleration_type: Optional[str] = None,
    min_success_probability: Optional[float] = None,
    max_risk_factor: Optional[float] = None,
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system)
):
    """Get acceleration strategies with optional filtering"""
    try:
        # Get active accelerations
        strategies = list(acceleration_system.active_accelerations.values())
        
        # Apply filters
        if innovation_id:
            strategies = [s for s in strategies if s.innovation_id == innovation_id]
        
        if acceleration_type:
            try:
                accel_type = AccelerationType(acceleration_type)
                strategies = [s for s in strategies if s.acceleration_type == accel_type]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid acceleration type: {acceleration_type}")
        
        if min_success_probability is not None:
            strategies = [s for s in strategies if s.success_probability >= min_success_probability]
        
        if max_risk_factor is not None:
            strategies = [s for s in strategies if s.risk_factor <= max_risk_factor]
        
        result = []
        for strategy in strategies:
            result.append({
                "id": strategy.id,
                "innovation_id": strategy.innovation_id,
                "acceleration_type": strategy.acceleration_type.value,
                "target_stage": strategy.target_stage.value,
                "expected_time_reduction": strategy.expected_time_reduction,
                "resource_cost": strategy.resource_cost,
                "success_probability": strategy.success_probability,
                "risk_factor": strategy.risk_factor,
                "prerequisites": strategy.prerequisites,
                "side_effects": strategy.side_effects
            })
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting acceleration strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=dict)
async def get_acceleration_metrics(
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system)
):
    """Get acceleration performance metrics"""
    try:
        metrics = acceleration_system.get_acceleration_metrics()
        bottleneck_patterns = acceleration_system.get_bottleneck_patterns()
        
        return {
            "performance_metrics": metrics,
            "bottleneck_patterns": bottleneck_patterns,
            "active_accelerations": len(acceleration_system.active_accelerations),
            "acceleration_history_count": len(acceleration_system.acceleration_history),
            "bottleneck_history_count": len(acceleration_system.bottleneck_history)
        }
    
    except Exception as e:
        logger.error(f"Error getting acceleration metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[dict])
async def get_acceleration_history(
    limit: int = 10,
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system)
):
    """Get acceleration history"""
    try:
        history = acceleration_system.acceleration_history[-limit:]
        
        result = []
        for acceleration in history:
            result.append({
                "acceleration_id": acceleration.acceleration_id,
                "timestamp": acceleration.timestamp.isoformat(),
                "innovations_accelerated": acceleration.innovations_accelerated,
                "total_time_saved": acceleration.total_time_saved,
                "bottlenecks_resolved": acceleration.bottlenecks_resolved,
                "performance_improvement": acceleration.performance_improvement,
                "cost_efficiency": acceleration.cost_efficiency,
                "strategies_applied": len(acceleration.acceleration_strategies_applied),
                "timeline_optimizations": len(acceleration.timeline_optimizations)
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting acceleration history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bottleneck-patterns", response_model=dict)
async def get_bottleneck_patterns(
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system)
):
    """Get bottleneck pattern analysis"""
    try:
        patterns = acceleration_system.get_bottleneck_patterns()
        
        # Convert to more detailed format
        result = {
            "patterns": {},
            "total_bottlenecks": sum(patterns.values()),
            "most_common_bottleneck": max(patterns.items(), key=lambda x: x[1])[0] if patterns else None
        }
        
        for bottleneck_type, count in patterns.items():
            result["patterns"][bottleneck_type] = {
                "count": count,
                "percentage": (count / sum(patterns.values()) * 100) if sum(patterns.values()) > 0 else 0
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting bottleneck patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/acceleration-types", response_model=List[dict])
async def get_acceleration_types():
    """Get available acceleration types"""
    try:
        types = []
        for accel_type in AccelerationType:
            types.append({
                "type": accel_type.value,
                "name": accel_type.name,
                "description": _get_acceleration_type_description(accel_type)
            })
        
        return types
    
    except Exception as e:
        logger.error(f"Error getting acceleration types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bottleneck-types", response_model=List[dict])
async def get_bottleneck_types():
    """Get available bottleneck types"""
    try:
        types = []
        for bottleneck_type in BottleneckType:
            types.append({
                "type": bottleneck_type.value,
                "name": bottleneck_type.name,
                "description": _get_bottleneck_type_description(bottleneck_type)
            })
        
        return types
    
    except Exception as e:
        logger.error(f"Error getting bottleneck types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-strategy/{strategy_id}", response_model=dict)
async def apply_acceleration_strategy(
    strategy_id: str,
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system),
    pipeline_optimizer: InnovationPipelineOptimizer = Depends(get_pipeline_optimizer)
):
    """Apply a specific acceleration strategy"""
    try:
        if strategy_id not in acceleration_system.active_accelerations:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy = acceleration_system.active_accelerations[strategy_id]
        
        if strategy.innovation_id not in pipeline_optimizer.pipeline_items:
            raise HTTPException(status_code=404, detail="Innovation not found")
        
        innovation = pipeline_optimizer.pipeline_items[strategy.innovation_id]
        
        # Apply strategy
        await acceleration_system._apply_strategy(strategy, innovation)
        
        return {
            "success": True,
            "message": f"Applied {strategy.acceleration_type.value} strategy to {innovation.innovation_id}",
            "strategy_id": strategy_id,
            "innovation_id": strategy.innovation_id,
            "expected_time_reduction": strategy.expected_time_reduction,
            "resource_cost": strategy.resource_cost
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying acceleration strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/strategies/{strategy_id}", response_model=dict)
async def remove_acceleration_strategy(
    strategy_id: str,
    acceleration_system: InnovationAccelerationSystem = Depends(get_acceleration_system)
):
    """Remove an acceleration strategy"""
    try:
        if strategy_id not in acceleration_system.active_accelerations:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy = acceleration_system.active_accelerations.pop(strategy_id)
        
        return {
            "success": True,
            "message": f"Removed acceleration strategy {strategy_id}",
            "strategy_id": strategy_id,
            "innovation_id": strategy.innovation_id,
            "acceleration_type": strategy.acceleration_type.value
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing acceleration strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_acceleration_type_description(accel_type: AccelerationType) -> str:
    """Get description for acceleration type"""
    descriptions = {
        AccelerationType.PARALLEL_PROCESSING: "Execute multiple tasks simultaneously to reduce overall time",
        AccelerationType.RESOURCE_BOOST: "Allocate additional resources to accelerate development",
        AccelerationType.FAST_TRACK: "Bypass normal processes for high-priority innovations",
        AccelerationType.BOTTLENECK_BYPASS: "Find alternative paths around identified bottlenecks",
        AccelerationType.AUTOMATED_TRANSITION: "Automate stage transitions to reduce manual delays",
        AccelerationType.PREDICTIVE_SCALING: "Preemptively scale resources based on predicted needs"
    }
    return descriptions.get(accel_type, "Unknown acceleration type")


def _get_bottleneck_type_description(bottleneck_type: BottleneckType) -> str:
    """Get description for bottleneck type"""
    descriptions = {
        BottleneckType.RESOURCE_CONSTRAINT: "Limited availability of required resources",
        BottleneckType.CAPACITY_LIMIT: "Stage capacity exceeded, causing queuing delays",
        BottleneckType.DEPENDENCY_BLOCK: "Waiting for dependent innovations to complete",
        BottleneckType.QUALITY_GATE: "Quality standards not met, preventing progression",
        BottleneckType.APPROVAL_DELAY: "Waiting for approvals or decision-making",
        BottleneckType.SKILL_GAP: "Lack of required skills or expertise",
        BottleneckType.TECHNOLOGY_LIMITATION: "Technical constraints limiting progress"
    }
    return descriptions.get(bottleneck_type, "Unknown bottleneck type")