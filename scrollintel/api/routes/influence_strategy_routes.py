"""
API Routes for Influence Strategy Development System
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.influence_strategy_engine import InfluenceStrategyEngine
from ...models.influence_strategy_models import (
    InfluenceStrategy, InfluenceObjective, InfluenceContext,
    InfluenceExecution, InfluenceEffectivenessMetrics, InfluenceOptimization
)
from ...models.stakeholder_influence_models import Stakeholder
from ...core.auth import get_current_user

router = APIRouter(prefix="/api/v1/influence-strategy", tags=["influence-strategy"])
logger = logging.getLogger(__name__)


@router.post("/develop", response_model=Dict[str, Any])
async def develop_influence_strategy(
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Develop targeted influence strategy for building support and consensus"""
    try:
        engine = InfluenceStrategyEngine()
        
        # Extract request parameters
        objective = InfluenceObjective(request.get("objective"))
        context = InfluenceContext(request.get("context"))
        stakeholder_data = request.get("target_stakeholders", [])
        constraints = request.get("constraints")
        
        # Convert stakeholder data to Stakeholder objects
        stakeholders = []
        for data in stakeholder_data:
            stakeholder = Stakeholder(
                id=data.get("id"),
                name=data.get("name"),
                role=data.get("role"),
                influence_level=data.get("influence_level", 0.5),
                decision_making_style=data.get("decision_making_style", "analytical"),
                communication_preferences=data.get("communication_preferences", []),
                motivations=data.get("motivations", []),
                concerns=data.get("concerns", []),
                relationship_strength=data.get("relationship_strength", 0.5),
                contact_info=data.get("contact_info", {}),
                interaction_history=data.get("interaction_history", [])
            )
            stakeholders.append(stakeholder)
        
        # Develop strategy
        strategy = engine.develop_influence_strategy(
            objective=objective,
            target_stakeholders=stakeholders,
            context=context,
            constraints=constraints
        )
        
        return {
            "success": True,
            "strategy": {
                "id": strategy.id,
                "name": strategy.name,
                "objective": strategy.objective.value,
                "context": strategy.context.value,
                "target_count": len(strategy.target_stakeholders),
                "primary_tactics": [
                    {
                        "id": tactic.id,
                        "name": tactic.name,
                        "type": tactic.influence_type.value,
                        "effectiveness_score": tactic.effectiveness_score
                    }
                    for tactic in strategy.primary_tactics
                ],
                "secondary_tactics": [
                    {
                        "id": tactic.id,
                        "name": tactic.name,
                        "type": tactic.influence_type.value,
                        "effectiveness_score": tactic.effectiveness_score
                    }
                    for tactic in strategy.secondary_tactics
                ],
                "expected_effectiveness": strategy.expected_effectiveness,
                "timeline": {k: v.isoformat() for k, v in strategy.timeline.items()},
                "success_metrics": strategy.success_metrics,
                "resource_requirements": strategy.resource_requirements,
                "risk_mitigation": strategy.risk_mitigation
            }
        }
        
    except Exception as e:
        logger.error(f"Error developing influence strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-tactics", response_model=Dict[str, Any])
async def select_optimal_tactics(
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Select and optimize influence tactics"""
    try:
        engine = InfluenceStrategyEngine()
        
        # Extract request parameters
        objective = InfluenceObjective(request.get("objective"))
        context = InfluenceContext(request.get("context"))
        target_data = request.get("targets", [])
        constraints = request.get("constraints")
        
        # Convert to InfluenceTarget objects (simplified)
        from ...models.influence_strategy_models import InfluenceTarget
        targets = []
        for data in target_data:
            target = InfluenceTarget(
                stakeholder_id=data.get("stakeholder_id"),
                name=data.get("name"),
                role=data.get("role"),
                influence_level=data.get("influence_level", 0.5),
                decision_making_style=data.get("decision_making_style", "analytical"),
                communication_preferences=data.get("communication_preferences", []),
                key_motivators=data.get("key_motivators", []),
                concerns=data.get("concerns", []),
                relationship_strength=data.get("relationship_strength", 0.5)
            )
            targets.append(target)
        
        # Select tactics
        primary_tactics, secondary_tactics = engine.select_optimal_tactics(
            objective=objective,
            targets=targets,
            context=context,
            constraints=constraints
        )
        
        return {
            "success": True,
            "tactic_selection": {
                "primary_tactics": [
                    {
                        "id": tactic.id,
                        "name": tactic.name,
                        "type": tactic.influence_type.value,
                        "description": tactic.description,
                        "effectiveness_score": tactic.effectiveness_score,
                        "required_preparation": tactic.required_preparation,
                        "success_indicators": tactic.success_indicators,
                        "risk_factors": tactic.risk_factors
                    }
                    for tactic in primary_tactics
                ],
                "secondary_tactics": [
                    {
                        "id": tactic.id,
                        "name": tactic.name,
                        "type": tactic.influence_type.value,
                        "description": tactic.description,
                        "effectiveness_score": tactic.effectiveness_score,
                        "required_preparation": tactic.required_preparation,
                        "success_indicators": tactic.success_indicators,
                        "risk_factors": tactic.risk_factors
                    }
                    for tactic in secondary_tactics
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error selecting tactics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/measure-effectiveness", response_model=Dict[str, Any])
async def measure_influence_effectiveness(
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Measure and track influence effectiveness"""
    try:
        engine = InfluenceStrategyEngine()
        
        # Extract execution data
        execution_data = request.get("execution")
        strategy_data = request.get("strategy")
        
        # Create execution object (simplified)
        execution = InfluenceExecution(
            id=execution_data.get("id"),
            strategy_id=execution_data.get("strategy_id"),
            execution_date=datetime.fromisoformat(execution_data.get("execution_date")),
            context_details=execution_data.get("context_details", {}),
            tactics_used=execution_data.get("tactics_used", []),
            target_responses=execution_data.get("target_responses", {}),
            immediate_outcomes=execution_data.get("immediate_outcomes", []),
            effectiveness_rating=execution_data.get("effectiveness_rating", 0.5),
            lessons_learned=execution_data.get("lessons_learned", []),
            follow_up_actions=execution_data.get("follow_up_actions", [])
        )
        
        # Create strategy object (simplified)
        from ...models.influence_strategy_models import InfluenceTarget
        targets = []
        for target_data in strategy_data.get("target_stakeholders", []):
            target = InfluenceTarget(
                stakeholder_id=target_data.get("stakeholder_id"),
                name=target_data.get("name"),
                role=target_data.get("role"),
                influence_level=target_data.get("influence_level", 0.5),
                decision_making_style=target_data.get("decision_making_style", "analytical"),
                communication_preferences=target_data.get("communication_preferences", []),
                key_motivators=target_data.get("key_motivators", []),
                concerns=target_data.get("concerns", []),
                relationship_strength=target_data.get("relationship_strength", 0.5)
            )
            targets.append(target)
        
        strategy = InfluenceStrategy(
            id=strategy_data.get("id"),
            name=strategy_data.get("name"),
            objective=InfluenceObjective(strategy_data.get("objective")),
            target_stakeholders=targets,
            primary_tactics=[],  # Simplified
            secondary_tactics=[],  # Simplified
            context=InfluenceContext(strategy_data.get("context")),
            timeline={},
            success_metrics=strategy_data.get("success_metrics", []),
            risk_mitigation={},
            resource_requirements=[],
            expected_effectiveness=strategy_data.get("expected_effectiveness", 0.5)
        )
        
        # Measure effectiveness
        metrics = engine.measure_influence_effectiveness(execution, strategy)
        
        return {
            "success": True,
            "effectiveness_metrics": {
                "strategy_id": metrics.strategy_id,
                "execution_id": metrics.execution_id,
                "objective_achievement": metrics.objective_achievement,
                "stakeholder_satisfaction": metrics.stakeholder_satisfaction,
                "relationship_impact": metrics.relationship_impact,
                "consensus_level": metrics.consensus_level,
                "support_gained": metrics.support_gained,
                "opposition_reduced": metrics.opposition_reduced,
                "long_term_relationship_health": metrics.long_term_relationship_health,
                "measured_at": metrics.measured_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error measuring influence effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_influence_strategy(
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Optimize influence strategy based on effectiveness data"""
    try:
        engine = InfluenceStrategyEngine()
        
        # Extract strategy and metrics data
        strategy_data = request.get("strategy")
        metrics_data = request.get("effectiveness_metrics", [])
        
        # Create strategy object (simplified)
        strategy = InfluenceStrategy(
            id=strategy_data.get("id"),
            name=strategy_data.get("name"),
            objective=InfluenceObjective(strategy_data.get("objective")),
            target_stakeholders=[],  # Simplified
            primary_tactics=[],  # Simplified
            secondary_tactics=[],  # Simplified
            context=InfluenceContext(strategy_data.get("context")),
            timeline={},
            success_metrics=strategy_data.get("success_metrics", []),
            risk_mitigation={},
            resource_requirements=[],
            expected_effectiveness=strategy_data.get("expected_effectiveness", 0.5)
        )
        
        # Create metrics objects
        metrics = []
        for metric_data in metrics_data:
            metric = InfluenceEffectivenessMetrics(
                strategy_id=metric_data.get("strategy_id"),
                execution_id=metric_data.get("execution_id"),
                objective_achievement=metric_data.get("objective_achievement", 0.5),
                stakeholder_satisfaction=metric_data.get("stakeholder_satisfaction", {}),
                relationship_impact=metric_data.get("relationship_impact", {}),
                consensus_level=metric_data.get("consensus_level", 0.5),
                support_gained=metric_data.get("support_gained", 0.5),
                opposition_reduced=metric_data.get("opposition_reduced", 0.5),
                long_term_relationship_health=metric_data.get("long_term_relationship_health", 0.5),
                measured_at=datetime.fromisoformat(metric_data.get("measured_at"))
            )
            metrics.append(metric)
        
        # Generate optimization
        optimization = engine.optimize_influence_strategy(strategy, metrics)
        
        return {
            "success": True,
            "optimization": {
                "strategy_id": optimization.strategy_id,
                "current_effectiveness": optimization.current_effectiveness,
                "optimization_opportunities": optimization.optimization_opportunities,
                "recommended_tactic_changes": optimization.recommended_tactic_changes,
                "timing_adjustments": optimization.timing_adjustments,
                "context_modifications": optimization.context_modifications,
                "target_approach_refinements": optimization.target_approach_refinements,
                "expected_improvement": optimization.expected_improvement,
                "confidence_level": optimization.confidence_level
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing influence strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tactics/library", response_model=Dict[str, Any])
async def get_tactic_library(
    current_user: dict = Depends(get_current_user)
):
    """Get available influence tactics library"""
    try:
        engine = InfluenceStrategyEngine()
        
        tactics = []
        for tactic in engine.tactic_library:
            tactics.append({
                "id": tactic.id,
                "name": tactic.name,
                "type": tactic.influence_type.value,
                "description": tactic.description,
                "effectiveness_score": tactic.effectiveness_score,
                "context_suitability": [ctx.value for ctx in tactic.context_suitability],
                "target_personality_types": tactic.target_personality_types,
                "required_preparation": tactic.required_preparation,
                "success_indicators": tactic.success_indicators,
                "risk_factors": tactic.risk_factors
            })
        
        return {
            "success": True,
            "tactics": tactics,
            "total_count": len(tactics)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving tactic library: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}/effectiveness", response_model=Dict[str, Any])
async def get_strategy_effectiveness_history(
    strategy_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get effectiveness history for a strategy"""
    try:
        engine = InfluenceStrategyEngine()
        
        # In practice, this would query a database
        history = []
        for execution_id, metrics in engine.effectiveness_history.items():
            if metrics.strategy_id == strategy_id:
                history.append({
                    "execution_id": execution_id,
                    "objective_achievement": metrics.objective_achievement,
                    "consensus_level": metrics.consensus_level,
                    "support_gained": metrics.support_gained,
                    "relationship_health": metrics.long_term_relationship_health,
                    "measured_at": metrics.measured_at.isoformat()
                })
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "effectiveness_history": history,
            "total_executions": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving effectiveness history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))