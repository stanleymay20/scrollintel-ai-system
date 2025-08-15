"""
Innovation Pipeline Management API Routes

This module provides REST API endpoints for managing innovation pipelines,
including optimization, monitoring, and resource allocation.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional
from datetime import datetime
import logging

from ...engines.innovation_pipeline_optimizer import InnovationPipelineOptimizer
from ...models.innovation_pipeline_models import (
    InnovationPipelineItem, PipelineOptimizationConfig, PipelineOptimizationResult,
    PipelinePerformanceReport, InnovationPriority, ResourceType, PipelineStage
)
from ...core.config import get_settings

router = APIRouter(prefix="/api/v1/innovation-pipeline", tags=["Innovation Pipeline"])
logger = logging.getLogger(__name__)

# Global optimizer instance
_optimizer_instance: Optional[InnovationPipelineOptimizer] = None


def get_optimizer() -> InnovationPipelineOptimizer:
    """Get or create pipeline optimizer instance"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = InnovationPipelineOptimizer()
    return _optimizer_instance


@router.post("/innovations", response_model=dict)
async def add_innovation_to_pipeline(
    innovation_item: InnovationPipelineItem,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Add a new innovation to the pipeline"""
    try:
        success = await optimizer.add_innovation_to_pipeline(innovation_item)
        
        if success:
            return {
                "success": True,
                "message": f"Innovation {innovation_item.id} added to pipeline",
                "innovation_id": innovation_item.id,
                "stage": innovation_item.current_stage.value
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to add innovation to pipeline - capacity constraints or other issues"
            )
    
    except Exception as e:
        logger.error(f"Error adding innovation to pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=PipelineOptimizationResult)
async def optimize_pipeline(
    background_tasks: BackgroundTasks,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Optimize the innovation pipeline for maximum efficiency"""
    try:
        # Run optimization in background for better performance
        result = await optimizer.optimize_pipeline_flow()
        
        return result
    
    except Exception as e:
        logger.error(f"Error optimizing pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=PipelinePerformanceReport)
async def get_pipeline_performance(
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Get comprehensive pipeline performance report"""
    try:
        report = await optimizer.monitor_pipeline_performance()
        return report
    
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prioritize", response_model=Dict[str, str])
async def prioritize_innovations(
    criteria: Dict[str, float],
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Prioritize innovations based on multiple criteria"""
    try:
        # Validate criteria
        valid_criteria = {'impact', 'success_probability', 'risk', 'resource_efficiency', 'time_sensitivity'}
        if not all(criterion in valid_criteria for criterion in criteria.keys()):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid criteria. Valid options: {valid_criteria}"
            )
        
        # Ensure criteria weights sum to reasonable value
        total_weight = sum(criteria.values())
        if total_weight <= 0 or total_weight > 2.0:
            raise HTTPException(
                status_code=400,
                detail="Criteria weights must sum to a positive value <= 2.0"
            )
        
        priorities = await optimizer.prioritize_innovations(criteria)
        
        # Convert enum values to strings for JSON response
        return {innovation_id: priority.value for innovation_id, priority in priorities.items()}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error prioritizing innovations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/allocate-resources", response_model=Dict[str, List[dict]])
async def allocate_resources(
    allocation_strategy: str = "balanced",
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Allocate resources across pipeline innovations"""
    try:
        # Validate allocation strategy
        valid_strategies = {"balanced", "aggressive", "conservative"}
        if allocation_strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid allocation strategy. Valid options: {valid_strategies}"
            )
        
        allocations = await optimizer.allocate_pipeline_resources(allocation_strategy)
        
        # Convert to JSON-serializable format
        result = {}
        for innovation_id, allocation_list in allocations.items():
            result[innovation_id] = [
                {
                    "id": allocation.id,
                    "resource_type": allocation.resource_type.value,
                    "allocated_amount": allocation.allocated_amount,
                    "used_amount": allocation.used_amount,
                    "allocation_time": allocation.allocation_time.isoformat(),
                    "expected_completion": allocation.expected_completion.isoformat() if allocation.expected_completion else None,
                    "efficiency_score": allocation.efficiency_score
                }
                for allocation in allocation_list
            ]
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error allocating resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/innovations", response_model=List[dict])
async def get_pipeline_innovations(
    stage: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Get innovations in the pipeline with optional filtering"""
    try:
        innovations = list(optimizer.pipeline_items.values())
        
        # Apply filters
        if stage:
            try:
                stage_enum = PipelineStage(stage)
                innovations = [item for item in innovations if item.current_stage == stage_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")
        
        if priority:
            try:
                priority_enum = InnovationPriority(priority)
                innovations = [item for item in innovations if item.priority == priority_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        if status:
            innovations = [item for item in innovations if item.status.value == status]
        
        # Convert to JSON-serializable format
        result = []
        for item in innovations:
            result.append({
                "id": item.id,
                "innovation_id": item.innovation_id,
                "current_stage": item.current_stage.value,
                "priority": item.priority.value,
                "status": item.status.value,
                "created_at": item.created_at.isoformat(),
                "stage_entered_at": item.stage_entered_at.isoformat(),
                "estimated_completion": item.estimated_completion.isoformat() if item.estimated_completion else None,
                "success_probability": item.success_probability,
                "risk_score": item.risk_score,
                "impact_score": item.impact_score,
                "resource_allocations_count": len(item.resource_allocations),
                "dependencies_count": len(item.dependencies),
                "blocking_issues_count": len(item.blocking_issues)
            })
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline innovations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/innovations/{innovation_id}", response_model=dict)
async def get_innovation_details(
    innovation_id: str,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Get detailed information about a specific innovation"""
    try:
        if innovation_id not in optimizer.pipeline_items:
            raise HTTPException(status_code=404, detail="Innovation not found")
        
        item = optimizer.pipeline_items[innovation_id]
        
        return {
            "id": item.id,
            "innovation_id": item.innovation_id,
            "current_stage": item.current_stage.value,
            "priority": item.priority.value,
            "status": item.status.value,
            "created_at": item.created_at.isoformat(),
            "stage_entered_at": item.stage_entered_at.isoformat(),
            "estimated_completion": item.estimated_completion.isoformat() if item.estimated_completion else None,
            "actual_completion": item.actual_completion.isoformat() if item.actual_completion else None,
            "success_probability": item.success_probability,
            "risk_score": item.risk_score,
            "impact_score": item.impact_score,
            "resource_requirements": [
                {
                    "resource_type": req.resource_type.value,
                    "amount": req.amount,
                    "unit": req.unit,
                    "duration": req.duration,
                    "priority": req.priority.value
                }
                for req in item.resource_requirements
            ],
            "resource_allocations": [
                {
                    "id": allocation.id,
                    "resource_type": allocation.resource_type.value,
                    "allocated_amount": allocation.allocated_amount,
                    "used_amount": allocation.used_amount,
                    "allocation_time": allocation.allocation_time.isoformat(),
                    "expected_completion": allocation.expected_completion.isoformat() if allocation.expected_completion else None,
                    "efficiency_score": allocation.efficiency_score
                }
                for allocation in item.resource_allocations
            ],
            "stage_metrics": {
                stage.value: {
                    "throughput": metrics.throughput,
                    "cycle_time": metrics.cycle_time,
                    "success_rate": metrics.success_rate,
                    "resource_utilization": metrics.resource_utilization,
                    "bottleneck_score": metrics.bottleneck_score,
                    "quality_score": metrics.quality_score,
                    "cost_efficiency": metrics.cost_efficiency
                }
                for stage, metrics in item.stage_metrics.items()
            },
            "dependencies": item.dependencies,
            "blocking_issues": item.blocking_issues,
            "metadata": item.metadata
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting innovation details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/innovations/{innovation_id}/stage", response_model=dict)
async def update_innovation_stage(
    innovation_id: str,
    new_stage: str,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Update the stage of a specific innovation"""
    try:
        if innovation_id not in optimizer.pipeline_items:
            raise HTTPException(status_code=404, detail="Innovation not found")
        
        try:
            stage_enum = PipelineStage(new_stage)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {new_stage}")
        
        item = optimizer.pipeline_items[innovation_id]
        old_stage = item.current_stage
        item.current_stage = stage_enum
        item.stage_entered_at = datetime.utcnow()
        
        return {
            "success": True,
            "message": f"Innovation {innovation_id} moved from {old_stage.value} to {new_stage}",
            "innovation_id": innovation_id,
            "old_stage": old_stage.value,
            "new_stage": new_stage,
            "stage_entered_at": item.stage_entered_at.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating innovation stage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/innovations/{innovation_id}/priority", response_model=dict)
async def update_innovation_priority(
    innovation_id: str,
    new_priority: str,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Update the priority of a specific innovation"""
    try:
        if innovation_id not in optimizer.pipeline_items:
            raise HTTPException(status_code=404, detail="Innovation not found")
        
        try:
            priority_enum = InnovationPriority(new_priority)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {new_priority}")
        
        item = optimizer.pipeline_items[innovation_id]
        old_priority = item.priority
        item.priority = priority_enum
        
        return {
            "success": True,
            "message": f"Innovation {innovation_id} priority changed from {old_priority.value} to {new_priority}",
            "innovation_id": innovation_id,
            "old_priority": old_priority.value,
            "new_priority": new_priority
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating innovation priority: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/bottlenecks", response_model=Dict[str, float])
async def get_pipeline_bottlenecks(
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Get current pipeline bottlenecks"""
    try:
        bottlenecks = await optimizer._identify_bottlenecks()
        return {stage.value: severity for stage, severity in bottlenecks.items()}
    
    except Exception as e:
        logger.error(f"Error getting pipeline bottlenecks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/resource-utilization", response_model=Dict[str, float])
async def get_resource_utilization(
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Get current resource utilization"""
    try:
        utilization = await optimizer._calculate_resource_utilization()
        return {resource_type.value: util for resource_type, util in utilization.items()}
    
    except Exception as e:
        logger.error(f"Error getting resource utilization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=dict)
async def get_pipeline_config(
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Get current pipeline configuration"""
    try:
        config = optimizer.config
        return {
            "max_concurrent_innovations": config.max_concurrent_innovations,
            "resource_buffer_percentage": config.resource_buffer_percentage,
            "priority_weights": {
                "critical": config.priority_weight_critical,
                "high": config.priority_weight_high,
                "medium": config.priority_weight_medium,
                "low": config.priority_weight_low
            },
            "optimization_objectives": {
                "maximize_throughput": config.maximize_throughput,
                "minimize_cycle_time": config.minimize_cycle_time,
                "maximize_success_rate": config.maximize_success_rate,
                "minimize_resource_waste": config.minimize_resource_waste
            },
            "thresholds": {
                "bottleneck_threshold": config.bottleneck_threshold,
                "resource_utilization_target": config.resource_utilization_target,
                "quality_threshold": config.quality_threshold
            },
            "rebalancing": {
                "rebalance_frequency_minutes": config.rebalance_frequency_minutes,
                "emergency_rebalance_threshold": config.emergency_rebalance_threshold
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting pipeline config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config", response_model=dict)
async def update_pipeline_config(
    config_updates: dict,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Update pipeline configuration"""
    try:
        # Update configuration fields
        config = optimizer.config
        
        if "max_concurrent_innovations" in config_updates:
            config.max_concurrent_innovations = config_updates["max_concurrent_innovations"]
        
        if "resource_buffer_percentage" in config_updates:
            config.resource_buffer_percentage = config_updates["resource_buffer_percentage"]
        
        if "bottleneck_threshold" in config_updates:
            config.bottleneck_threshold = config_updates["bottleneck_threshold"]
        
        if "resource_utilization_target" in config_updates:
            config.resource_utilization_target = config_updates["resource_utilization_target"]
        
        if "quality_threshold" in config_updates:
            config.quality_threshold = config_updates["quality_threshold"]
        
        return {
            "success": True,
            "message": "Pipeline configuration updated successfully",
            "updated_fields": list(config_updates.keys())
        }
    
    except Exception as e:
        logger.error(f"Error updating pipeline config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/optimizations", response_model=List[dict])
async def get_optimization_history(
    limit: int = 10,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Get optimization history"""
    try:
        history = optimizer.optimization_history[-limit:]
        
        result = []
        for opt in history:
            result.append({
                "optimization_id": opt.optimization_id,
                "timestamp": opt.timestamp.isoformat(),
                "resource_reallocations_count": len(opt.resource_reallocations),
                "priority_adjustments_count": len(opt.priority_adjustments),
                "stage_transitions_count": len(opt.stage_transitions),
                "expected_throughput_improvement": opt.expected_throughput_improvement,
                "expected_cycle_time_reduction": opt.expected_cycle_time_reduction,
                "expected_resource_savings": opt.expected_resource_savings,
                "optimization_score": opt.optimization_score,
                "confidence_level": opt.confidence_level,
                "recommendations_count": len(opt.recommendations),
                "warnings_count": len(opt.warnings)
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting optimization history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/performance", response_model=List[dict])
async def get_performance_history(
    limit: int = 10,
    optimizer: InnovationPipelineOptimizer = Depends(get_optimizer)
):
    """Get performance report history"""
    try:
        history = optimizer.historical_metrics[-limit:]
        
        result = []
        for report in history:
            result.append({
                "report_id": report.report_id,
                "generated_at": report.generated_at.isoformat(),
                "period_start": report.period_start.isoformat(),
                "period_end": report.period_end.isoformat(),
                "total_innovations": report.total_innovations,
                "completed_innovations": report.completed_innovations,
                "failed_innovations": report.failed_innovations,
                "active_innovations": report.active_innovations,
                "overall_throughput": report.overall_throughput,
                "average_cycle_time": report.average_cycle_time,
                "overall_success_rate": report.overall_success_rate,
                "cost_per_innovation": report.cost_per_innovation,
                "identified_bottlenecks": [stage.value for stage in report.identified_bottlenecks],
                "optimization_recommendations_count": len(report.optimization_recommendations),
                "capacity_recommendations_count": len(report.capacity_recommendations),
                "process_improvements_count": len(report.process_improvements)
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))