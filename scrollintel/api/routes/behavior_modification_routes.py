"""
API Routes for Behavior Modification Engine

Provides REST endpoints for behavior modification functionality
including strategy development, intervention creation, and progress tracking.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from datetime import datetime
import logging

from ...engines.behavior_modification_engine import BehaviorModificationEngine
from ...models.behavior_modification_models import (
    BehaviorModificationStrategy, ModificationIntervention, BehaviorChangeProgress,
    ModificationTechnique, ModificationStatus, ProgressLevel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/behavior-modification", tags=["behavior-modification"])

# Global engine instance
modification_engine = BehaviorModificationEngine()


@router.post("/strategies/develop")
async def develop_modification_strategy(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Develop systematic behavior change strategy
    
    Requirements: 3.2, 3.3 - Create systematic behavior change strategy development
    """
    try:
        target_behavior = request_data.get("target_behavior", "")
        current_state = request_data.get("current_state", {})
        desired_outcome = request_data.get("desired_outcome", "")
        constraints = request_data.get("constraints", {})
        stakeholders = request_data.get("stakeholders", [])
        
        if not target_behavior or not desired_outcome:
            raise HTTPException(
                status_code=400, 
                detail="target_behavior and desired_outcome are required"
            )
        
        strategy = modification_engine.develop_modification_strategy(
            target_behavior=target_behavior,
            current_state=current_state,
            desired_outcome=desired_outcome,
            constraints=constraints,
            stakeholders=stakeholders
        )
        
        return {
            "success": True,
            "strategy_id": strategy.id,
            "strategy": {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "target_behavior": strategy.target_behavior,
                "desired_outcome": strategy.desired_outcome,
                "techniques": [t.value for t in strategy.techniques],
                "timeline_weeks": strategy.timeline_weeks,
                "success_criteria": strategy.success_criteria,
                "resources_required": strategy.resources_required,
                "stakeholders": strategy.stakeholders,
                "risk_factors": strategy.risk_factors,
                "mitigation_strategies": strategy.mitigation_strategies,
                "created_date": strategy.created_date.isoformat(),
                "created_by": strategy.created_by
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error developing modification strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Strategy development failed: {str(e)}")


@router.post("/strategies/{strategy_id}/interventions")
async def create_modification_interventions(
    strategy_id: str,
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create specific interventions for behavior modification strategy
    
    Requirements: 3.2, 3.3 - Implement behavior modification technique selection and application
    """
    try:
        strategy = modification_engine.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        participants = request_data.get("participants", [])
        facilitators = request_data.get("facilitators", [])
        
        if not participants:
            raise HTTPException(status_code=400, detail="participants list is required")
        
        interventions = modification_engine.create_modification_interventions(
            strategy=strategy,
            participants=participants,
            facilitators=facilitators
        )
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "interventions_created": len(interventions),
            "interventions": [
                {
                    "id": intervention.id,
                    "technique": intervention.technique.value,
                    "name": intervention.intervention_name,
                    "description": intervention.description,
                    "target_participants": intervention.target_participants,
                    "duration_days": intervention.duration_days,
                    "frequency": intervention.frequency,
                    "status": intervention.status.value,
                    "assigned_facilitator": intervention.assigned_facilitator
                }
                for intervention in interventions
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating interventions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Intervention creation failed: {str(e)}")


@router.post("/progress/track")
async def track_behavior_change_progress(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Track progress of behavior change efforts
    
    Requirements: 3.2, 3.3 - Build behavior change progress tracking and optimization
    """
    try:
        strategy_id = request_data.get("strategy_id", "")
        participant_id = request_data.get("participant_id", "")
        current_measurement = request_data.get("current_measurement")
        baseline_measurement = request_data.get("baseline_measurement")
        target_measurement = request_data.get("target_measurement")
        
        if not strategy_id or not participant_id or current_measurement is None:
            raise HTTPException(
                status_code=400,
                detail="strategy_id, participant_id, and current_measurement are required"
            )
        
        if not 0.0 <= current_measurement <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="current_measurement must be between 0.0 and 1.0"
            )
        
        progress = modification_engine.track_behavior_change_progress(
            strategy_id=strategy_id,
            participant_id=participant_id,
            current_measurement=current_measurement,
            baseline_measurement=baseline_measurement,
            target_measurement=target_measurement
        )
        
        return {
            "success": True,
            "progress_id": progress.id,
            "progress": {
                "id": progress.id,
                "strategy_id": progress.strategy_id,
                "participant_id": progress.participant_id,
                "baseline_measurement": progress.baseline_measurement,
                "current_measurement": progress.current_measurement,
                "target_measurement": progress.target_measurement,
                "progress_level": progress.progress_level.value,
                "improvement_rate": progress.improvement_rate,
                "milestones_achieved": progress.milestones_achieved,
                "challenges_encountered": progress.challenges_encountered,
                "adjustments_made": progress.adjustments_made,
                "last_updated": progress.last_updated.isoformat(),
                "next_review_date": progress.next_review_date.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Progress tracking failed: {str(e)}")


@router.post("/strategies/{strategy_id}/optimize")
async def optimize_modification_strategy(
    strategy_id: str,
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Optimize behavior modification strategy based on progress and effectiveness data
    
    Requirements: 3.2, 3.3 - Build behavior change progress tracking and optimization
    """
    try:
        strategy = modification_engine.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Get progress data (simplified - in real implementation would query database)
        progress_data = []  # Would be populated from actual progress tracking
        effectiveness_data = request_data.get("effectiveness_data", {})
        
        optimization = modification_engine.optimize_modification_strategy(
            strategy_id=strategy_id,
            progress_data=progress_data,
            effectiveness_data=effectiveness_data
        )
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "optimization": {
                "current_effectiveness": optimization.current_effectiveness,
                "optimization_opportunities": optimization.optimization_opportunities,
                "recommended_adjustments": optimization.recommended_adjustments,
                "expected_improvement": optimization.expected_improvement,
                "implementation_effort": optimization.implementation_effort,
                "risk_level": optimization.risk_level,
                "priority_score": optimization.priority_score,
                "analysis_date": optimization.analysis_date.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Strategy optimization failed: {str(e)}")


@router.get("/strategies/{strategy_id}")
async def get_modification_strategy(strategy_id: str) -> Dict[str, Any]:
    """Get behavior modification strategy details"""
    try:
        strategy = modification_engine.get_strategy(strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "success": True,
            "strategy": {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "target_behavior": strategy.target_behavior,
                "desired_outcome": strategy.desired_outcome,
                "techniques": [t.value for t in strategy.techniques],
                "timeline_weeks": strategy.timeline_weeks,
                "success_criteria": strategy.success_criteria,
                "resources_required": strategy.resources_required,
                "stakeholders": strategy.stakeholders,
                "risk_factors": strategy.risk_factors,
                "mitigation_strategies": strategy.mitigation_strategies,
                "created_date": strategy.created_date.isoformat(),
                "created_by": strategy.created_by
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve strategy: {str(e)}")


@router.get("/organization/{organization_id}/strategies")
async def get_organization_strategies(organization_id: str) -> Dict[str, Any]:
    """Get all behavior modification strategies for an organization"""
    try:
        strategies = modification_engine.get_organization_strategies(organization_id)
        
        return {
            "success": True,
            "organization_id": organization_id,
            "strategies_count": len(strategies),
            "strategies": [
                {
                    "id": strategy.id,
                    "name": strategy.name,
                    "target_behavior": strategy.target_behavior,
                    "desired_outcome": strategy.desired_outcome,
                    "techniques_count": len(strategy.techniques),
                    "timeline_weeks": strategy.timeline_weeks,
                    "created_date": strategy.created_date.isoformat(),
                    "created_by": strategy.created_by
                }
                for strategy in strategies
            ]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving organization strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve strategies: {str(e)}")


@router.post("/organization/{organization_id}/metrics")
async def calculate_modification_metrics(organization_id: str) -> Dict[str, Any]:
    """Calculate comprehensive behavior modification metrics for an organization"""
    try:
        metrics = modification_engine.calculate_modification_metrics(organization_id)
        
        return {
            "success": True,
            "organization_id": organization_id,
            "metrics": {
                "total_strategies": metrics.total_strategies,
                "active_interventions": metrics.active_interventions,
                "participants_engaged": metrics.participants_engaged,
                "overall_success_rate": metrics.overall_success_rate,
                "average_improvement_rate": metrics.average_improvement_rate,
                "time_to_change_average": metrics.time_to_change_average,
                "participant_satisfaction_average": metrics.participant_satisfaction_average,
                "cost_per_participant": metrics.cost_per_participant,
                "roi_achieved": metrics.roi_achieved,
                "sustainability_index": metrics.sustainability_index,
                "calculated_date": metrics.calculated_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate metrics: {str(e)}")


@router.get("/techniques")
async def get_modification_techniques() -> Dict[str, Any]:
    """Get available behavior modification techniques"""
    return {
        "success": True,
        "techniques": [
            {
                "value": technique.value,
                "name": technique.value.replace("_", " ").title(),
                "description": f"Apply {technique.value.replace('_', ' ')} for behavior modification"
            }
            for technique in ModificationTechnique
        ]
    }


@router.get("/progress-levels")
async def get_progress_levels() -> Dict[str, Any]:
    """Get available progress levels"""
    return {
        "success": True,
        "progress_levels": [
            {
                "value": level.value,
                "name": level.value.replace("_", " ").title()
            }
            for level in ProgressLevel
        ]
    }


@router.post("/validate/strategy")
async def validate_strategy_data(strategy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate behavior modification strategy data"""
    try:
        required_fields = ["target_behavior", "desired_outcome"]
        missing_fields = [field for field in required_fields if not strategy_data.get(field)]
        
        if missing_fields:
            return {
                "success": True,
                "valid": False,
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }
        
        # Validate timeline if provided
        timeline_weeks = strategy_data.get("timeline_weeks")
        if timeline_weeks is not None and (not isinstance(timeline_weeks, int) or timeline_weeks <= 0):
            return {
                "success": True,
                "valid": False,
                "message": "timeline_weeks must be a positive integer"
            }
        
        return {
            "success": True,
            "valid": True,
            "message": "Strategy data is valid"
        }
        
    except Exception as e:
        return {
            "success": True,
            "valid": False,
            "message": f"Validation failed: {str(e)}"
        }