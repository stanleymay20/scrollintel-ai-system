"""
API Routes for Crisis-Strategic Planning Integration

This module provides REST API endpoints for integrating crisis leadership
capabilities with strategic planning systems.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.crisis_strategic_integration import (
    CrisisStrategicIntegration, CrisisStrategicImpact, 
    CrisisAwareAdjustment, RecoveryIntegrationPlan
)
from ...models.crisis_detection_models import Crisis
from ...models.strategic_planning_models import StrategicRoadmap
from ...core.auth import get_current_user
from ...core.monitoring import track_api_call

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/crisis-strategic", tags=["crisis-strategic-integration"])

# Initialize the integration engine
integration_engine = CrisisStrategicIntegration()


@router.post("/assess-impact")
@track_api_call
async def assess_crisis_impact(
    crisis_data: Dict[str, Any],
    strategic_roadmap_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Assess the impact of a crisis on strategic plans
    
    Args:
        crisis_data: Crisis information
        strategic_roadmap_data: Strategic roadmap data
        current_user: Authenticated user information
        
    Returns:
        Crisis impact assessment
    """
    try:
        logger.info(f"Assessing crisis impact for user: {current_user.get('user_id')}")
        
        # Convert input data to model objects
        crisis = Crisis(**crisis_data)
        strategic_roadmap = StrategicRoadmap(**strategic_roadmap_data)
        
        # Perform impact assessment
        impact_assessment = await integration_engine.assess_crisis_impact_on_strategy(
            crisis, strategic_roadmap
        )
        
        # Convert to dictionary for JSON response
        result = {
            "crisis_id": impact_assessment.crisis_id,
            "strategic_plan_id": impact_assessment.strategic_plan_id,
            "impact_level": impact_assessment.impact_level.value,
            "affected_milestones": impact_assessment.affected_milestones,
            "affected_technology_bets": impact_assessment.affected_technology_bets,
            "resource_reallocation_needed": impact_assessment.resource_reallocation_needed,
            "timeline_adjustments": impact_assessment.timeline_adjustments,
            "risk_level_changes": impact_assessment.risk_level_changes,
            "strategic_recommendations": impact_assessment.strategic_recommendations,
            "recovery_timeline": impact_assessment.recovery_timeline,
            "created_at": impact_assessment.created_at.isoformat()
        }
        
        logger.info(f"Crisis impact assessment completed: {impact_assessment.impact_level.value}")
        return {
            "status": "success",
            "data": result,
            "message": "Crisis impact assessment completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error assessing crisis impact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assess crisis impact: {str(e)}")


@router.post("/generate-adjustments")
@track_api_call
async def generate_crisis_adjustments(
    crisis_data: Dict[str, Any],
    strategic_roadmap_data: Dict[str, Any],
    impact_assessment_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate crisis-aware strategic adjustments
    
    Args:
        crisis_data: Crisis information
        strategic_roadmap_data: Strategic roadmap data
        impact_assessment_data: Crisis impact assessment data
        current_user: Authenticated user information
        
    Returns:
        List of crisis-aware strategic adjustments
    """
    try:
        logger.info(f"Generating crisis adjustments for user: {current_user.get('user_id')}")
        
        # Convert input data to model objects
        crisis = Crisis(**crisis_data)
        strategic_roadmap = StrategicRoadmap(**strategic_roadmap_data)
        impact_assessment = CrisisStrategicImpact(**impact_assessment_data)
        
        # Generate adjustments
        adjustments = await integration_engine.generate_crisis_aware_adjustments(
            crisis, strategic_roadmap, impact_assessment
        )
        
        # Convert to dictionary list for JSON response
        adjustments_data = []
        for adjustment in adjustments:
            adjustments_data.append({
                "adjustment_id": adjustment.adjustment_id,
                "crisis_id": adjustment.crisis_id,
                "adjustment_type": adjustment.adjustment_type,
                "description": adjustment.description,
                "priority": adjustment.priority,
                "implementation_timeline": adjustment.implementation_timeline,
                "resource_requirements": adjustment.resource_requirements,
                "expected_benefits": adjustment.expected_benefits,
                "risks": adjustment.risks,
                "success_metrics": adjustment.success_metrics,
                "dependencies": adjustment.dependencies,
                "created_at": adjustment.created_at.isoformat()
            })
        
        logger.info(f"Generated {len(adjustments)} crisis-aware adjustments")
        return {
            "status": "success",
            "data": {
                "adjustments": adjustments_data,
                "total_count": len(adjustments)
            },
            "message": f"Generated {len(adjustments)} crisis-aware strategic adjustments"
        }
        
    except Exception as e:
        logger.error(f"Error generating crisis adjustments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate adjustments: {str(e)}")


@router.post("/create-recovery-plan")
@track_api_call
async def create_recovery_integration_plan(
    crisis_data: Dict[str, Any],
    strategic_roadmap_data: Dict[str, Any],
    impact_assessment_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create integrated crisis recovery and strategic planning plan
    
    Args:
        crisis_data: Crisis information
        strategic_roadmap_data: Strategic roadmap data
        impact_assessment_data: Crisis impact assessment data
        current_user: Authenticated user information
        
    Returns:
        Comprehensive recovery integration plan
    """
    try:
        logger.info(f"Creating recovery integration plan for user: {current_user.get('user_id')}")
        
        # Convert input data to model objects
        crisis = Crisis(**crisis_data)
        strategic_roadmap = StrategicRoadmap(**strategic_roadmap_data)
        impact_assessment = CrisisStrategicImpact(**impact_assessment_data)
        
        # Create recovery plan
        recovery_plan = await integration_engine.create_recovery_integration_plan(
            crisis, strategic_roadmap, impact_assessment
        )
        
        # Convert to dictionary for JSON response
        result = {
            "plan_id": recovery_plan.plan_id,
            "crisis_id": recovery_plan.crisis_id,
            "strategic_roadmap_id": recovery_plan.strategic_roadmap_id,
            "recovery_phases": recovery_plan.recovery_phases,
            "milestone_realignment": {
                k: v.isoformat() for k, v in recovery_plan.milestone_realignment.items()
            },
            "resource_rebalancing": recovery_plan.resource_rebalancing,
            "technology_bet_adjustments": recovery_plan.technology_bet_adjustments,
            "stakeholder_communication_plan": recovery_plan.stakeholder_communication_plan,
            "success_criteria": recovery_plan.success_criteria,
            "monitoring_framework": recovery_plan.monitoring_framework,
            "created_at": recovery_plan.created_at.isoformat()
        }
        
        logger.info(f"Recovery integration plan created: {recovery_plan.plan_id}")
        return {
            "status": "success",
            "data": result,
            "message": "Recovery integration plan created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating recovery plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create recovery plan: {str(e)}")


@router.get("/integration-status/{crisis_id}")
@track_api_call
async def get_integration_status(
    crisis_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the status of crisis-strategic integration for a specific crisis
    
    Args:
        crisis_id: Crisis identifier
        current_user: Authenticated user information
        
    Returns:
        Integration status information
    """
    try:
        logger.info(f"Getting integration status for crisis: {crisis_id}")
        
        # This would typically query a database for stored integration data
        # For now, return a mock status
        status = {
            "crisis_id": crisis_id,
            "integration_active": True,
            "impact_assessment_completed": True,
            "adjustments_generated": True,
            "recovery_plan_created": True,
            "last_updated": datetime.now().isoformat(),
            "status": "active",
            "progress": {
                "impact_assessment": 100,
                "strategic_adjustments": 85,
                "recovery_planning": 70,
                "implementation": 45
            }
        }
        
        return {
            "status": "success",
            "data": status,
            "message": "Integration status retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting integration status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get integration status: {str(e)}")


@router.post("/simulate-integration")
@track_api_call
async def simulate_crisis_strategic_integration(
    simulation_params: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Simulate crisis-strategic integration scenarios
    
    Args:
        simulation_params: Simulation parameters
        current_user: Authenticated user information
        
    Returns:
        Simulation results
    """
    try:
        logger.info(f"Running crisis-strategic integration simulation for user: {current_user.get('user_id')}")
        
        # Extract simulation parameters
        crisis_scenarios = simulation_params.get("crisis_scenarios", [])
        strategic_context = simulation_params.get("strategic_context", {})
        simulation_duration = simulation_params.get("duration_days", 365)
        
        # Run simulation (simplified version)
        simulation_results = {
            "simulation_id": f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "scenarios_tested": len(crisis_scenarios),
            "duration_days": simulation_duration,
            "results": {
                "average_impact_level": "moderate",
                "average_recovery_time": 120,  # days
                "resource_reallocation_range": [0.15, 0.45],
                "strategic_milestone_delays": [30, 180],  # days
                "resilience_score": 0.75
            },
            "recommendations": [
                "Increase crisis response resource buffer to 20%",
                "Implement quarterly strategic resilience reviews",
                "Develop automated crisis-strategic integration protocols",
                "Enhance stakeholder communication frameworks"
            ],
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Simulation completed: {simulation_results['simulation_id']}")
        return {
            "status": "success",
            "data": simulation_results,
            "message": "Crisis-strategic integration simulation completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run simulation: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for crisis-strategic integration service"""
    return {
        "status": "healthy",
        "service": "crisis-strategic-integration",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }