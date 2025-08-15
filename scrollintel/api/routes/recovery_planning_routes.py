"""
Recovery Planning API Routes for Crisis Leadership Excellence System

This module provides REST API endpoints for post-crisis recovery strategy development,
milestone tracking, and success measurement.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from ...engines.recovery_planning_engine import RecoveryPlanningEngine
from ...models.recovery_planning_models import (
    RecoveryStrategy, RecoveryProgress, RecoveryOptimization
)
from ...models.crisis_detection_models import Crisis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/recovery-planning", tags=["recovery-planning"])

# Global engine instance
recovery_engine = RecoveryPlanningEngine()


@router.post("/strategies", response_model=Dict[str, Any])
async def develop_recovery_strategy(
    crisis_data: Dict[str, Any],
    recovery_objectives: List[str]
):
    """
    Develop comprehensive post-crisis recovery strategy
    
    Args:
        crisis_data: Crisis information requiring recovery
        recovery_objectives: List of recovery objectives
        
    Returns:
        Dict containing recovery strategy details
    """
    try:
        # Convert crisis_data to CrisisModel object (simplified for demo)
        from ...models.crisis_detection_models import CrisisModel
        
        crisis = CrisisModel(
            id=crisis_data.get("id", "demo_crisis"),
            crisis_type=crisis_data.get("crisis_type", "system_failure"),
            severity_level=crisis_data.get("severity_level", "high"),
            status=crisis_data.get("status", "resolved"),
            start_time=crisis_data.get("start_time"),
            affected_areas=crisis_data.get("affected_areas", []),
            stakeholders_impacted=crisis_data.get("stakeholders_impacted", []),
            resolution_time=crisis_data.get("resolution_time")
        )
        
        strategy = recovery_engine.develop_recovery_strategy(crisis, recovery_objectives)
        
        return {
            "success": True,
            "strategy_id": strategy.id,
            "strategy": {
                "id": strategy.id,
                "crisis_id": strategy.crisis_id,
                "strategy_name": strategy.strategy_name,
                "created_at": strategy.created_at.isoformat(),
                "recovery_objectives": strategy.recovery_objectives,
                "success_metrics": strategy.success_metrics,
                "milestones_count": len(strategy.milestones),
                "timeline": {phase.value: str(duration) for phase, duration in strategy.timeline.items()},
                "resource_allocation": strategy.resource_allocation
            },
            "message": "Recovery strategy developed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error developing recovery strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}/progress", response_model=Dict[str, Any])
async def track_recovery_progress(strategy_id: str):
    """
    Track progress of recovery strategy implementation
    
    Args:
        strategy_id: ID of recovery strategy to track
        
    Returns:
        Dict containing recovery progress details
    """
    try:
        progress = recovery_engine.track_recovery_progress(strategy_id)
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "progress": {
                "overall_progress": progress.overall_progress,
                "phase_progress": {phase.value: prog for phase, prog in progress.phase_progress.items()},
                "milestone_completion_rate": progress.milestone_completion_rate,
                "timeline_adherence": progress.timeline_adherence,
                "resource_utilization": progress.resource_utilization,
                "success_metric_achievement": progress.success_metric_achievement,
                "identified_issues": progress.identified_issues,
                "recommended_adjustments": progress.recommended_adjustments,
                "last_updated": progress.last_updated.isoformat()
            },
            "message": "Recovery progress tracked successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking recovery progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}/optimize", response_model=Dict[str, Any])
async def optimize_recovery_strategy(strategy_id: str):
    """
    Generate optimization recommendations for recovery strategy
    
    Args:
        strategy_id: ID of recovery strategy to optimize
        
    Returns:
        Dict containing optimization recommendations
    """
    try:
        optimizations = recovery_engine.optimize_recovery_strategy(strategy_id)
        
        optimization_data = []
        for opt in optimizations:
            optimization_data.append({
                "optimization_type": opt.optimization_type,
                "current_performance": opt.current_performance,
                "target_performance": opt.target_performance,
                "recommended_actions": opt.recommended_actions,
                "expected_impact": opt.expected_impact,
                "implementation_effort": opt.implementation_effort,
                "priority_score": opt.priority_score,
                "created_at": opt.created_at.isoformat()
            })
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "optimizations": optimization_data,
            "optimization_count": len(optimizations),
            "message": "Recovery optimization completed successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing recovery strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}", response_model=Dict[str, Any])
async def get_recovery_strategy(strategy_id: str):
    """
    Get detailed recovery strategy information
    
    Args:
        strategy_id: ID of recovery strategy to retrieve
        
    Returns:
        Dict containing complete recovery strategy details
    """
    try:
        if strategy_id not in recovery_engine.recovery_strategies:
            raise HTTPException(status_code=404, detail="Recovery strategy not found")
        
        strategy = recovery_engine.recovery_strategies[strategy_id]
        
        milestones_data = []
        for milestone in strategy.milestones:
            milestones_data.append({
                "id": milestone.id,
                "name": milestone.name,
                "description": milestone.description,
                "phase": milestone.phase.value,
                "priority": milestone.priority.value,
                "target_date": milestone.target_date.isoformat(),
                "completion_date": milestone.completion_date.isoformat() if milestone.completion_date else None,
                "status": milestone.status.value,
                "progress_percentage": milestone.progress_percentage,
                "success_criteria": milestone.success_criteria,
                "dependencies": milestone.dependencies,
                "assigned_team": milestone.assigned_team,
                "resources_required": milestone.resources_required,
                "risk_factors": milestone.risk_factors
            })
        
        return {
            "success": True,
            "strategy": {
                "id": strategy.id,
                "crisis_id": strategy.crisis_id,
                "strategy_name": strategy.strategy_name,
                "created_at": strategy.created_at.isoformat(),
                "updated_at": strategy.updated_at.isoformat(),
                "recovery_objectives": strategy.recovery_objectives,
                "success_metrics": strategy.success_metrics,
                "milestones": milestones_data,
                "resource_allocation": strategy.resource_allocation,
                "timeline": {phase.value: str(duration) for phase, duration in strategy.timeline.items()},
                "stakeholder_communication_plan": strategy.stakeholder_communication_plan,
                "risk_mitigation_measures": strategy.risk_mitigation_measures,
                "contingency_plans": strategy.contingency_plans
            },
            "message": "Recovery strategy retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving recovery strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/strategies/{strategy_id}/milestones/{milestone_id}/progress")
async def update_milestone_progress(
    strategy_id: str,
    milestone_id: str,
    progress_data: Dict[str, Any]
):
    """
    Update progress for a specific recovery milestone
    
    Args:
        strategy_id: ID of recovery strategy
        milestone_id: ID of milestone to update
        progress_data: Progress update information
        
    Returns:
        Dict containing update confirmation
    """
    try:
        if strategy_id not in recovery_engine.recovery_strategies:
            raise HTTPException(status_code=404, detail="Recovery strategy not found")
        
        strategy = recovery_engine.recovery_strategies[strategy_id]
        milestone = None
        
        for m in strategy.milestones:
            if m.id == milestone_id:
                milestone = m
                break
        
        if not milestone:
            raise HTTPException(status_code=404, detail="Milestone not found")
        
        # Update milestone progress
        if "progress_percentage" in progress_data:
            milestone.progress_percentage = progress_data["progress_percentage"]
        
        if "status" in progress_data:
            from ...models.recovery_planning_models import RecoveryStatus
            milestone.status = RecoveryStatus(progress_data["status"])
        
        if "completion_date" in progress_data and progress_data["completion_date"]:
            from datetime import datetime
            milestone.completion_date = datetime.fromisoformat(progress_data["completion_date"])
        
        # Update strategy timestamp
        from datetime import datetime
        strategy.updated_at = datetime.now()
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "milestone_id": milestone_id,
            "updated_progress": milestone.progress_percentage,
            "updated_status": milestone.status.value,
            "message": "Milestone progress updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating milestone progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies", response_model=Dict[str, Any])
async def list_recovery_strategies():
    """
    List all recovery strategies
    
    Returns:
        Dict containing list of recovery strategies
    """
    try:
        strategies_list = []
        
        for strategy_id, strategy in recovery_engine.recovery_strategies.items():
            strategies_list.append({
                "id": strategy.id,
                "crisis_id": strategy.crisis_id,
                "strategy_name": strategy.strategy_name,
                "created_at": strategy.created_at.isoformat(),
                "milestones_count": len(strategy.milestones),
                "objectives_count": len(strategy.recovery_objectives)
            })
        
        return {
            "success": True,
            "strategies": strategies_list,
            "total_count": len(strategies_list),
            "message": "Recovery strategies listed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error listing recovery strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/strategies/{strategy_id}")
async def delete_recovery_strategy(strategy_id: str):
    """
    Delete a recovery strategy
    
    Args:
        strategy_id: ID of recovery strategy to delete
        
    Returns:
        Dict containing deletion confirmation
    """
    try:
        if strategy_id not in recovery_engine.recovery_strategies:
            raise HTTPException(status_code=404, detail="Recovery strategy not found")
        
        del recovery_engine.recovery_strategies[strategy_id]
        
        # Also remove associated progress and optimizations
        if strategy_id in recovery_engine.recovery_progress:
            del recovery_engine.recovery_progress[strategy_id]
        
        if strategy_id in recovery_engine.optimization_recommendations:
            del recovery_engine.optimization_recommendations[strategy_id]
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "message": "Recovery strategy deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting recovery strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))