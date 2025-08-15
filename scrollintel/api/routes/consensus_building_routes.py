"""
Consensus Building API Routes for Board Executive Mastery System

This module provides REST API endpoints for board consensus building strategy development,
tracking, and facilitation.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.consensus_building_engine import ConsensusBuildingEngine
from ...models.consensus_building_models import (
    ConsensusBuilding, BoardMemberProfile, ConsensusPosition, ConsensusBarrier,
    ConsensusStrategy, ConsensusAction, ConsensusMetrics, ConsensusRecommendation,
    ConsensusOptimization, ConsensusVisualization, ConsensusStatus,
    StakeholderPosition, InfluenceLevel, ConsensusStrategyType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/consensus-building", tags=["consensus-building"])

# Global engine instance
consensus_engine = ConsensusBuildingEngine()


# Request/Response Models
from pydantic import BaseModel

class CreateConsensusProcessRequest(BaseModel):
    title: str
    description: str
    decision_topic: str
    target_consensus_level: ConsensusStatus = ConsensusStatus.STRONG_CONSENSUS
    deadline: Optional[datetime] = None
    facilitator_id: str = "system"


class AddBoardMemberRequest(BaseModel):
    name: str
    role: str
    influence_level: InfluenceLevel
    decision_making_style: str = "analytical"
    key_concerns: List[str] = []
    motivations: List[str] = []
    communication_preferences: List[str] = []


class UpdateStakeholderPositionRequest(BaseModel):
    stakeholder_id: str
    stakeholder_name: str
    position: StakeholderPosition
    confidence_level: float = 0.8
    key_concerns: List[str] = []
    requirements_for_support: List[str] = []
    deal_breakers: List[str] = []


class CreateConsensusActionRequest(BaseModel):
    title: str
    description: str
    action_type: str
    target_stakeholders: List[str]
    responsible_party: str
    deadline: datetime
    expected_impact: str


class CreateVisualizationRequest(BaseModel):
    visualization_type: str = "stakeholder_map"


# API Endpoints

@router.post("/processes", response_model=Dict[str, Any])
async def create_consensus_process(request: CreateConsensusProcessRequest):
    """Create a new consensus building process"""
    try:
        process = consensus_engine.create_consensus_building(
            title=request.title,
            description=request.description,
            decision_topic=request.decision_topic,
            target_consensus_level=request.target_consensus_level,
            deadline=request.deadline,
            facilitator_id=request.facilitator_id
        )
        
        return {
            "status": "success",
            "message": "Consensus building process created successfully",
            "process_id": process.id,
            "process": {
                "id": process.id,
                "title": process.title,
                "decision_topic": process.decision_topic,
                "target_consensus_level": process.target_consensus_level.value,
                "current_consensus_level": process.current_consensus_level.value,
                "created_at": process.created_at.isoformat(),
                "deadline": process.deadline.isoformat() if process.deadline else None,
                "consensus_score": process.consensus_score
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating consensus process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processes", response_model=Dict[str, Any])
async def list_consensus_processes():
    """List all consensus building processes"""
    try:
        processes = consensus_engine.list_consensus_processes()
        
        processes_summary = []
        for process in processes:
            processes_summary.append({
                "id": process.id,
                "title": process.title,
                "decision_topic": process.decision_topic,
                "target_consensus_level": process.target_consensus_level.value,
                "current_consensus_level": process.current_consensus_level.value,
                "created_at": process.created_at.isoformat(),
                "deadline": process.deadline.isoformat() if process.deadline else None,
                "consensus_score": process.consensus_score,
                "stakeholder_count": len(process.stakeholder_positions),
                "barrier_count": len(process.barriers),
                "strategy_count": len(process.strategies),
                "success_probability": process.success_probability
            })
        
        return {
            "status": "success",
            "processes": processes_summary,
            "total_count": len(processes)
        }
        
    except Exception as e:
        logger.error(f"Error listing consensus processes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processes/{process_id}", response_model=Dict[str, Any])
async def get_consensus_process(process_id: str):
    """Get detailed consensus building process"""
    try:
        process = consensus_engine.get_consensus_process(process_id)
        
        if not process:
            raise HTTPException(status_code=404, detail="Consensus process not found")
        
        return {
            "status": "success",
            "process": {
                "id": process.id,
                "title": process.title,
                "description": process.description,
                "decision_topic": process.decision_topic,
                "target_consensus_level": process.target_consensus_level.value,
                "current_consensus_level": process.current_consensus_level.value,
                "created_at": process.created_at.isoformat(),
                "deadline": process.deadline.isoformat() if process.deadline else None,
                "consensus_score": process.consensus_score,
                "momentum": process.momentum,
                "success_probability": process.success_probability,
                "board_members": [
                    {
                        "id": member.id,
                        "name": member.name,
                        "role": member.role,
                        "influence_level": member.influence_level.value,
                        "decision_making_style": member.decision_making_style,
                        "key_concerns": member.key_concerns,
                        "motivations": member.motivations
                    } for member in process.board_members
                ],
                "stakeholder_positions": [
                    {
                        "stakeholder_id": pos.stakeholder_id,
                        "stakeholder_name": pos.stakeholder_name,
                        "current_position": pos.current_position.value,
                        "confidence_level": pos.confidence_level,
                        "key_concerns": pos.key_concerns,
                        "requirements_for_support": pos.requirements_for_support,
                        "deal_breakers": pos.deal_breakers,
                        "last_updated": pos.last_updated.isoformat()
                    } for pos in process.stakeholder_positions
                ],
                "barriers": [
                    {
                        "id": barrier.id,
                        "description": barrier.description,
                        "barrier_type": barrier.barrier_type,
                        "severity": barrier.severity,
                        "affected_stakeholders": barrier.affected_stakeholders,
                        "mitigation_strategies": barrier.mitigation_strategies
                    } for barrier in process.barriers
                ],
                "strategies": [
                    {
                        "id": strategy.id,
                        "strategy_type": strategy.strategy_type.value,
                        "description": strategy.description,
                        "target_stakeholders": strategy.target_stakeholders,
                        "success_probability": strategy.success_probability,
                        "estimated_timeline": strategy.estimated_timeline
                    } for strategy in process.strategies
                ],
                "actions": [
                    {
                        "id": action.id,
                        "title": action.title,
                        "action_type": action.action_type,
                        "target_stakeholders": action.target_stakeholders,
                        "responsible_party": action.responsible_party,
                        "deadline": action.deadline.isoformat(),
                        "status": action.status,
                        "expected_impact": action.expected_impact
                    } for action in process.actions
                ],
                "coalition_map": process.coalition_map,
                "key_influencers": process.key_influencers
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting consensus process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processes/{process_id}/board-members", response_model=Dict[str, Any])
async def add_board_member(process_id: str, request: AddBoardMemberRequest):
    """Add board member to consensus process"""
    try:
        member = consensus_engine.add_board_member(
            process_id=process_id,
            name=request.name,
            role=request.role,
            influence_level=request.influence_level,
            decision_making_style=request.decision_making_style,
            key_concerns=request.key_concerns,
            motivations=request.motivations,
            communication_preferences=request.communication_preferences
        )
        
        return {
            "status": "success",
            "message": "Board member added successfully",
            "member_id": member.id,
            "member": {
                "id": member.id,
                "name": member.name,
                "role": member.role,
                "influence_level": member.influence_level.value,
                "decision_making_style": member.decision_making_style
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding board member: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/processes/{process_id}/stakeholder-positions", response_model=Dict[str, Any])
async def update_stakeholder_position(process_id: str, request: UpdateStakeholderPositionRequest):
    """Update stakeholder position in consensus process"""
    try:
        position = consensus_engine.update_stakeholder_position(
            process_id=process_id,
            stakeholder_id=request.stakeholder_id,
            stakeholder_name=request.stakeholder_name,
            position=request.position,
            confidence_level=request.confidence_level,
            key_concerns=request.key_concerns,
            requirements_for_support=request.requirements_for_support,
            deal_breakers=request.deal_breakers
        )
        
        # Get updated metrics
        metrics = consensus_engine.track_consensus_progress(process_id)
        
        return {
            "status": "success",
            "message": "Stakeholder position updated successfully",
            "position": {
                "stakeholder_name": position.stakeholder_name,
                "current_position": position.current_position.value,
                "confidence_level": position.confidence_level,
                "last_updated": position.last_updated.isoformat()
            },
            "consensus_metrics": {
                "consensus_score": metrics.weighted_support_score,
                "support_percentage": metrics.support_percentage,
                "opposition_percentage": metrics.opposition_percentage,
                "neutral_percentage": metrics.neutral_percentage,
                "momentum_direction": metrics.momentum_direction
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating stakeholder position: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processes/{process_id}/barriers", response_model=Dict[str, Any])
async def identify_consensus_barriers(process_id: str, decision_type: str = "strategic_decision"):
    """Identify barriers to consensus"""
    try:
        barriers = consensus_engine.identify_consensus_barriers(
            process_id=process_id,
            decision_type=decision_type
        )
        
        return {
            "status": "success",
            "message": f"Identified {len(barriers)} consensus barriers",
            "barriers": [
                {
                    "id": barrier.id,
                    "description": barrier.description,
                    "barrier_type": barrier.barrier_type,
                    "severity": barrier.severity,
                    "affected_stakeholders": barrier.affected_stakeholders,
                    "mitigation_strategies": barrier.mitigation_strategies,
                    "estimated_resolution_time": barrier.estimated_resolution_time
                } for barrier in barriers
            ]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error identifying consensus barriers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processes/{process_id}/strategies", response_model=Dict[str, Any])
async def develop_consensus_strategy(process_id: str, decision_type: str = "strategic_decision"):
    """Develop strategies for building consensus"""
    try:
        strategies = consensus_engine.develop_consensus_strategy(
            process_id=process_id,
            decision_type=decision_type
        )
        
        return {
            "status": "success",
            "message": f"Developed {len(strategies)} consensus strategies",
            "strategies": [
                {
                    "id": strategy.id,
                    "strategy_type": strategy.strategy_type.value,
                    "description": strategy.description,
                    "target_stakeholders": strategy.target_stakeholders,
                    "tactics": strategy.tactics,
                    "expected_outcomes": strategy.expected_outcomes,
                    "success_probability": strategy.success_probability,
                    "estimated_timeline": strategy.estimated_timeline,
                    "resource_requirements": strategy.resource_requirements,
                    "risks": strategy.risks
                } for strategy in strategies
            ]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error developing consensus strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processes/{process_id}/actions", response_model=Dict[str, Any])
async def create_consensus_action(process_id: str, request: CreateConsensusActionRequest):
    """Create specific consensus building action"""
    try:
        action = consensus_engine.create_consensus_action(
            process_id=process_id,
            title=request.title,
            description=request.description,
            action_type=request.action_type,
            target_stakeholders=request.target_stakeholders,
            responsible_party=request.responsible_party,
            deadline=request.deadline,
            expected_impact=request.expected_impact
        )
        
        return {
            "status": "success",
            "message": "Consensus action created successfully",
            "action_id": action.id,
            "action": {
                "id": action.id,
                "title": action.title,
                "action_type": action.action_type,
                "target_stakeholders": action.target_stakeholders,
                "responsible_party": action.responsible_party,
                "deadline": action.deadline.isoformat(),
                "status": action.status,
                "expected_impact": action.expected_impact
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating consensus action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processes/{process_id}/metrics", response_model=Dict[str, Any])
async def track_consensus_progress(process_id: str):
    """Track and measure consensus building progress"""
    try:
        metrics = consensus_engine.track_consensus_progress(process_id)
        
        return {
            "status": "success",
            "metrics": {
                "id": metrics.id,
                "measurement_date": metrics.measurement_date.isoformat(),
                "support_percentage": metrics.support_percentage,
                "opposition_percentage": metrics.opposition_percentage,
                "neutral_percentage": metrics.neutral_percentage,
                "weighted_support_score": metrics.weighted_support_score,
                "momentum_direction": metrics.momentum_direction,
                "stakeholder_engagement_level": metrics.stakeholder_engagement_level,
                "communication_effectiveness": metrics.communication_effectiveness,
                "trust_level": metrics.trust_level,
                "key_concerns_addressed": metrics.key_concerns_addressed,
                "barriers_resolved": metrics.barriers_resolved,
                "new_barriers_identified": metrics.new_barriers_identified
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking consensus progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processes/{process_id}/recommendation", response_model=Dict[str, Any])
async def generate_consensus_recommendation(process_id: str):
    """Generate recommendations for achieving consensus"""
    try:
        recommendation = consensus_engine.generate_consensus_recommendation(process_id)
        
        return {
            "status": "success",
            "recommendation": {
                "id": recommendation.id,
                "title": recommendation.title,
                "description": recommendation.description,
                "recommended_approach": recommendation.recommended_approach.value,
                "priority_actions": recommendation.priority_actions,
                "key_stakeholders_to_focus": recommendation.key_stakeholders_to_focus,
                "timeline_recommendation": recommendation.timeline_recommendation,
                "communication_strategy": recommendation.communication_strategy,
                "meeting_recommendations": recommendation.meeting_recommendations,
                "negotiation_points": recommendation.negotiation_points,
                "compromise_options": recommendation.compromise_options,
                "potential_risks": recommendation.potential_risks,
                "mitigation_strategies": recommendation.mitigation_strategies,
                "contingency_plans": recommendation.contingency_plans,
                "success_probability": recommendation.success_probability,
                "critical_success_factors": recommendation.critical_success_factors,
                "early_warning_indicators": recommendation.early_warning_indicators
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating consensus recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processes/{process_id}/optimize", response_model=Dict[str, Any])
async def optimize_consensus_process(process_id: str):
    """Generate optimization recommendations for consensus building"""
    try:
        optimization = consensus_engine.optimize_consensus_process(process_id)
        
        return {
            "status": "success",
            "optimization": {
                "id": optimization.id,
                "process_improvements": optimization.process_improvements,
                "communication_enhancements": optimization.communication_enhancements,
                "engagement_strategies": optimization.engagement_strategies,
                "stakeholder_specific_approaches": optimization.stakeholder_specific_approaches,
                "coalition_building_opportunities": optimization.coalition_building_opportunities,
                "influence_leverage_points": optimization.influence_leverage_points,
                "accelerated_timeline_options": optimization.accelerated_timeline_options,
                "parallel_workstream_opportunities": optimization.parallel_workstream_opportunities,
                "quick_wins": optimization.quick_wins,
                "decision_quality_enhancements": optimization.decision_quality_enhancements,
                "information_gaps_to_address": optimization.information_gaps_to_address,
                "expertise_to_bring_in": optimization.expertise_to_bring_in
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing consensus process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processes/{process_id}/visualizations", response_model=Dict[str, Any])
async def create_consensus_visualization(process_id: str, request: CreateVisualizationRequest):
    """Create visualization for consensus building process"""
    try:
        visualization = consensus_engine.create_consensus_visualization(
            process_id=process_id,
            visualization_type=request.visualization_type
        )
        
        return {
            "status": "success",
            "visualization": {
                "id": visualization.id,
                "visualization_type": visualization.visualization_type,
                "title": visualization.title,
                "description": visualization.description,
                "chart_config": visualization.chart_config,
                "executive_summary": visualization.executive_summary
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating consensus visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processes/{process_id}/dashboard", response_model=Dict[str, Any])
async def get_consensus_dashboard(process_id: str):
    """Get comprehensive consensus building dashboard"""
    try:
        process = consensus_engine.get_consensus_process(process_id)
        if not process:
            raise HTTPException(status_code=404, detail="Consensus process not found")
        
        metrics = consensus_engine.track_consensus_progress(process_id)
        
        # Calculate additional dashboard metrics
        total_stakeholders = len(process.stakeholder_positions)
        high_influence_count = len([
            member for member in process.board_members 
            if member.influence_level in [InfluenceLevel.HIGH, InfluenceLevel.CRITICAL]
        ])
        
        completed_actions = len([action for action in process.actions if action.status == "completed"])
        pending_actions = len([action for action in process.actions if action.status in ["planned", "in_progress"]])
        
        # Identify critical stakeholders needing attention
        critical_stakeholders = []
        for pos in process.stakeholder_positions:
            member = next((m for m in process.board_members if m.id == pos.stakeholder_id), None)
            if (member and member.influence_level in [InfluenceLevel.HIGH, InfluenceLevel.CRITICAL] 
                and pos.current_position in [StakeholderPosition.OPPOSE, StakeholderPosition.NEUTRAL, StakeholderPosition.UNDECIDED]):
                critical_stakeholders.append({
                    "name": pos.stakeholder_name,
                    "position": pos.current_position.value,
                    "influence": member.influence_level.value,
                    "key_concerns": pos.key_concerns[:3]  # Top 3 concerns
                })
        
        return {
            "status": "success",
            "dashboard": {
                "process_overview": {
                    "id": process.id,
                    "title": process.title,
                    "decision_topic": process.decision_topic,
                    "current_status": process.current_consensus_level.value,
                    "target_status": process.target_consensus_level.value,
                    "consensus_score": process.consensus_score,
                    "success_probability": process.success_probability,
                    "days_remaining": (process.deadline - datetime.now()).days if process.deadline else None
                },
                "stakeholder_summary": {
                    "total_stakeholders": total_stakeholders,
                    "high_influence_count": high_influence_count,
                    "support_percentage": metrics.support_percentage,
                    "opposition_percentage": metrics.opposition_percentage,
                    "neutral_percentage": metrics.neutral_percentage,
                    "weighted_support_score": metrics.weighted_support_score
                },
                "progress_indicators": {
                    "momentum_direction": metrics.momentum_direction,
                    "engagement_level": metrics.stakeholder_engagement_level,
                    "communication_effectiveness": metrics.communication_effectiveness,
                    "trust_level": metrics.trust_level
                },
                "action_status": {
                    "completed_actions": completed_actions,
                    "pending_actions": pending_actions,
                    "total_actions": len(process.actions)
                },
                "barriers_and_strategies": {
                    "total_barriers": len(process.barriers),
                    "high_severity_barriers": len([b for b in process.barriers if b.severity > 0.7]),
                    "total_strategies": len(process.strategies),
                    "high_probability_strategies": len([s for s in process.strategies if s.success_probability > 0.7])
                },
                "critical_stakeholders": critical_stakeholders,
                "next_steps": [
                    action.title for action in process.actions 
                    if action.status == "planned" and action.deadline > datetime.now()
                ][:5]  # Next 5 upcoming actions
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting consensus dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))