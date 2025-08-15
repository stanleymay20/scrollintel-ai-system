"""
API Routes for Breakthrough Innovation Integration System
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging

from ...engines.breakthrough_innovation_integration import BreakthroughInnovationIntegration
from ...models.breakthrough_innovation_integration_models import (
    BreakthroughInnovation, InnovationSynergy, CrossPollinationOpportunity,
    InnovationAccelerationPlan, BreakthroughValidationResult, IntegrationMetrics
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/breakthrough-innovation-integration", tags=["breakthrough-innovation-integration"])

# Global integration engine instance
integration_engine = BreakthroughInnovationIntegration()

@router.post("/synergies/identify")
async def identify_innovation_synergies(
    lab_components: List[str],
    breakthrough_innovations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Identify synergies between innovation lab components and breakthrough innovations
    """
    try:
        # Convert dict representations to BreakthroughInnovation objects
        innovations = []
        for innovation_data in breakthrough_innovations:
            innovation = BreakthroughInnovation(**innovation_data)
            innovations.append(innovation)
        
        synergies = await integration_engine.identify_innovation_synergies(
            lab_components, innovations
        )
        
        return {
            "status": "success",
            "synergies_identified": len(synergies),
            "synergies": [
                {
                    "id": synergy.id,
                    "innovation_lab_component": synergy.innovation_lab_component,
                    "breakthrough_innovation_id": synergy.breakthrough_innovation_id,
                    "synergy_type": synergy.synergy_type.value,
                    "synergy_strength": synergy.synergy_strength,
                    "enhancement_potential": synergy.enhancement_potential,
                    "implementation_effort": synergy.implementation_effort,
                    "expected_benefits": synergy.expected_benefits,
                    "integration_requirements": synergy.integration_requirements,
                    "validation_metrics": synergy.validation_metrics,
                    "created_at": synergy.created_at.isoformat()
                }
                for synergy in synergies
            ]
        }
        
    except Exception as e:
        logger.error(f"Error identifying innovation synergies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cross-pollination/implement")
async def implement_cross_pollination(
    source_innovation: Dict[str, Any],
    target_research_areas: List[str]
) -> Dict[str, Any]:
    """
    Implement innovation cross-pollination between breakthrough innovations and research areas
    """
    try:
        innovation = BreakthroughInnovation(**source_innovation)
        
        opportunities = await integration_engine.implement_cross_pollination(
            innovation, target_research_areas
        )
        
        return {
            "status": "success",
            "opportunities_created": len(opportunities),
            "cross_pollination_opportunities": [
                {
                    "id": opp.id,
                    "source_innovation": opp.source_innovation,
                    "target_research_area": opp.target_research_area,
                    "pollination_type": opp.pollination_type,
                    "enhancement_potential": opp.enhancement_potential,
                    "feasibility_score": opp.feasibility_score,
                    "expected_outcomes": opp.expected_outcomes,
                    "integration_pathway": opp.integration_pathway,
                    "resource_requirements": opp.resource_requirements,
                    "timeline_estimate": opp.timeline_estimate,
                    "success_indicators": opp.success_indicators
                }
                for opp in opportunities
            ]
        }
        
    except Exception as e:
        logger.error(f"Error implementing cross-pollination: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synergies/exploit")
async def identify_synergy_exploitation(
    synergy_ids: List[str]
) -> Dict[str, Any]:
    """
    Identify and create plans for exploiting innovation synergies
    """
    try:
        # Get synergies by IDs
        synergies = []
        for synergy_id in synergy_ids:
            if synergy_id in integration_engine.active_synergies:
                synergies.append(integration_engine.active_synergies[synergy_id])
        
        if not synergies:
            raise HTTPException(status_code=404, detail="No valid synergies found")
        
        acceleration_plans = await integration_engine.identify_synergy_exploitation(synergies)
        
        return {
            "status": "success",
            "acceleration_plans_created": len(acceleration_plans),
            "acceleration_plans": [
                {
                    "id": plan.id,
                    "target_innovation": plan.target_innovation,
                    "acceleration_strategies": plan.acceleration_strategies,
                    "resource_optimization": plan.resource_optimization,
                    "timeline_compression": plan.timeline_compression,
                    "risk_mitigation": plan.risk_mitigation,
                    "success_metrics": plan.success_metrics,
                    "implementation_steps": plan.implementation_steps,
                    "monitoring_framework": plan.monitoring_framework
                }
                for plan in acceleration_plans
            ]
        }
        
    except Exception as e:
        logger.error(f"Error identifying synergy exploitation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validation/breakthrough")
async def validate_breakthrough_integration(
    innovation: Dict[str, Any],
    integration_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate breakthrough innovation integration with lab systems
    """
    try:
        breakthrough_innovation = BreakthroughInnovation(**innovation)
        
        validation_result = await integration_engine.validate_breakthrough_integration(
            breakthrough_innovation, integration_context
        )
        
        return {
            "status": "success",
            "validation_result": {
                "innovation_id": validation_result.innovation_id,
                "validation_score": validation_result.validation_score,
                "feasibility_assessment": validation_result.feasibility_assessment,
                "impact_prediction": validation_result.impact_prediction,
                "risk_analysis": validation_result.risk_analysis,
                "implementation_pathway": validation_result.implementation_pathway,
                "resource_requirements": validation_result.resource_requirements,
                "success_probability": validation_result.success_probability,
                "recommendations": validation_result.recommendations,
                "validation_timestamp": validation_result.validation_timestamp.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error validating breakthrough integration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimization/performance")
async def optimize_integration_performance(
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Optimize the performance of breakthrough innovation integration
    """
    try:
        optimization_results = await integration_engine.optimize_integration_performance()
        
        return {
            "status": "success",
            "optimization_results": optimization_results,
            "message": "Integration performance optimization completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error optimizing integration performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_integration_status() -> Dict[str, Any]:
    """
    Get current integration status and metrics
    """
    try:
        status = integration_engine.get_integration_status()
        
        return {
            "status": "success",
            "integration_status": status
        }
        
    except Exception as e:
        logger.error(f"Error getting integration status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/synergies")
async def get_active_synergies() -> Dict[str, Any]:
    """
    Get all active innovation synergies
    """
    try:
        synergies = list(integration_engine.active_synergies.values())
        
        return {
            "status": "success",
            "total_synergies": len(synergies),
            "synergies": [
                {
                    "id": synergy.id,
                    "innovation_lab_component": synergy.innovation_lab_component,
                    "breakthrough_innovation_id": synergy.breakthrough_innovation_id,
                    "synergy_type": synergy.synergy_type.value,
                    "synergy_strength": synergy.synergy_strength,
                    "enhancement_potential": synergy.enhancement_potential,
                    "implementation_effort": synergy.implementation_effort,
                    "expected_benefits": synergy.expected_benefits,
                    "created_at": synergy.created_at.isoformat()
                }
                for synergy in synergies
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting active synergies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cross-pollination")
async def get_cross_pollination_opportunities() -> Dict[str, Any]:
    """
    Get all cross-pollination opportunities
    """
    try:
        opportunities = list(integration_engine.cross_pollination_opportunities.values())
        
        return {
            "status": "success",
            "total_opportunities": len(opportunities),
            "opportunities": [
                {
                    "id": opp.id,
                    "source_innovation": opp.source_innovation,
                    "target_research_area": opp.target_research_area,
                    "pollination_type": opp.pollination_type,
                    "enhancement_potential": opp.enhancement_potential,
                    "feasibility_score": opp.feasibility_score,
                    "expected_outcomes": opp.expected_outcomes,
                    "timeline_estimate": opp.timeline_estimate
                }
                for opp in opportunities
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting cross-pollination opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/acceleration-plans")
async def get_acceleration_plans() -> Dict[str, Any]:
    """
    Get all innovation acceleration plans
    """
    try:
        plans = list(integration_engine.acceleration_plans.values())
        
        return {
            "status": "success",
            "total_plans": len(plans),
            "acceleration_plans": [
                {
                    "id": plan.id,
                    "target_innovation": plan.target_innovation,
                    "acceleration_strategies": plan.acceleration_strategies,
                    "timeline_compression": plan.timeline_compression,
                    "success_metrics": plan.success_metrics,
                    "implementation_steps": plan.implementation_steps
                }
                for plan in plans
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting acceleration plans: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/synergies/{synergy_id}")
async def remove_synergy(synergy_id: str) -> Dict[str, Any]:
    """
    Remove a specific innovation synergy
    """
    try:
        if synergy_id in integration_engine.active_synergies:
            del integration_engine.active_synergies[synergy_id]
            return {
                "status": "success",
                "message": f"Synergy {synergy_id} removed successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Synergy not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing synergy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cross-pollination/{opportunity_id}")
async def remove_cross_pollination_opportunity(opportunity_id: str) -> Dict[str, Any]:
    """
    Remove a specific cross-pollination opportunity
    """
    try:
        if opportunity_id in integration_engine.cross_pollination_opportunities:
            del integration_engine.cross_pollination_opportunities[opportunity_id]
            return {
                "status": "success",
                "message": f"Cross-pollination opportunity {opportunity_id} removed successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Cross-pollination opportunity not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing cross-pollination opportunity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))