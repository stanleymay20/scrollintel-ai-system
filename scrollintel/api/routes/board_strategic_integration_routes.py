"""
API Routes for Board Strategic Integration

Provides endpoints for integrating board executive mastery with strategic planning systems.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
from datetime import datetime, date
import logging

from ...core.board_strategic_integration import (
    BoardStrategicIntegration,
    BoardStrategicAlignment,
    BoardFeedbackIntegration,
    BoardApprovalTracking
)
from ...models.board_dynamics_models import Board
from ...models.strategic_planning_models import TechnologyVision, StrategicRoadmap, StrategicPivot
from ...security.auth import get_current_user
from ...core.logging_config import get_logger

router = APIRouter(prefix="/api/v1/board-strategic", tags=["board-strategic-integration"])
logger = get_logger(__name__)


@router.post("/create-aligned-plan")
async def create_board_aligned_strategic_plan(
    board_data: Dict[str, Any],
    vision_data: Dict[str, Any],
    horizon: int,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Create strategic planning aligned with board priorities and preferences
    """
    try:
        logger.info(f"Creating board-aligned strategic plan for {horizon} years")
        
        integration_system = BoardStrategicIntegration()
        
        # Convert input data to models
        board = Board(**board_data)
        vision = TechnologyVision(**vision_data)
        
        # Create board-aligned strategic plan
        roadmap = await integration_system.create_board_aligned_strategic_plan(
            board, vision, horizon
        )
        
        # Schedule background alignment assessment
        background_tasks.add_task(
            _assess_and_log_alignment,
            board, roadmap
        )
        
        return {
            "status": "success",
            "message": "Board-aligned strategic plan created successfully",
            "roadmap": {
                "id": roadmap.id,
                "name": roadmap.name,
                "description": roadmap.description,
                "time_horizon": roadmap.time_horizon,
                "milestones_count": len(roadmap.milestones),
                "technology_bets_count": len(roadmap.technology_bets),
                "stakeholders": roadmap.stakeholders
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating board-aligned strategic plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrate-feedback")
async def integrate_board_feedback(
    roadmap_id: str,
    feedback_items: List[Dict[str, Any]],
    current_user: Dict = Depends(get_current_user)
):
    """
    Integrate board feedback into strategic planning
    """
    try:
        logger.info(f"Integrating {len(feedback_items)} feedback items for roadmap {roadmap_id}")
        
        integration_system = BoardStrategicIntegration()
        
        # Convert feedback items to models
        feedback_list = []
        for item in feedback_items:
            feedback = BoardFeedbackIntegration(
                feedback_id=item.get("feedback_id", f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                board_member_id=item["board_member_id"],
                strategic_element=item["strategic_element"],
                feedback_type=item["feedback_type"],
                feedback_content=item["feedback_content"],
                impact_assessment=item.get("impact_assessment", 0.5),
                integration_status="pending",
                created_at=datetime.now()
            )
            feedback_list.append(feedback)
        
        # Get current roadmap (in real implementation, fetch from database)
        # For now, create a mock roadmap
        current_roadmap = await _get_roadmap_by_id(roadmap_id)
        
        # Integrate feedback
        updated_roadmap = await integration_system.integrate_board_feedback(
            feedback_list, current_roadmap
        )
        
        return {
            "status": "success",
            "message": f"Successfully integrated {len(feedback_items)} feedback items",
            "updated_roadmap": {
                "id": updated_roadmap.id,
                "name": updated_roadmap.name,
                "last_updated": updated_roadmap.updated_at.isoformat()
            },
            "integrated_feedback": [
                {
                    "feedback_id": f.feedback_id,
                    "status": f.integration_status,
                    "impact": f.impact_assessment
                }
                for f in feedback_list
            ]
        }
        
    except Exception as e:
        logger.error(f"Error integrating board feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-approval")
async def track_board_approval(
    initiative_id: str,
    board_data: Dict[str, Any],
    voting_record: Dict[str, str],
    current_user: Dict = Depends(get_current_user)
):
    """
    Track board approval status for strategic initiatives
    """
    try:
        logger.info(f"Tracking board approval for initiative {initiative_id}")
        
        integration_system = BoardStrategicIntegration()
        
        # Convert board data to model
        board = Board(**board_data)
        
        # Track approval
        approval_tracking = await integration_system.track_board_approval(
            initiative_id, board, voting_record
        )
        
        return {
            "status": "success",
            "message": "Board approval tracking created successfully",
            "approval_tracking": {
                "initiative_id": approval_tracking.initiative_id,
                "approval_status": approval_tracking.approval_status,
                "vote_summary": f"{len([v for v in voting_record.values() if v == 'approve'])}/{len(voting_record)} approved",
                "conditions_count": len(approval_tracking.approval_conditions),
                "next_review_date": approval_tracking.next_review_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error tracking board approval: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-strategic-adjustment")
async def generate_strategic_adjustment(
    board_data: Dict[str, Any],
    roadmap_id: str,
    market_changes: List[Dict[str, Any]],
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate strategic adjustments based on board input and market changes
    """
    try:
        logger.info(f"Generating strategic adjustment for roadmap {roadmap_id}")
        
        integration_system = BoardStrategicIntegration()
        
        # Convert input data to models
        board = Board(**board_data)
        roadmap = await _get_roadmap_by_id(roadmap_id)
        
        # Generate strategic adjustment
        pivot = await integration_system.generate_board_strategic_adjustment(
            board, roadmap, market_changes
        )
        
        return {
            "status": "success",
            "message": "Strategic adjustment recommendations generated",
            "strategic_pivot": {
                "id": pivot.id,
                "name": pivot.name,
                "description": pivot.description,
                "pivot_type": pivot.pivot_type,
                "timeline": pivot.timeline,
                "trigger_events": pivot.trigger_events,
                "board_approval_required": pivot.board_approval_required,
                "success_metrics": pivot.success_metrics
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating strategic adjustment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alignment-assessment/{board_id}/{roadmap_id}")
async def get_strategic_alignment_assessment(
    board_id: str,
    roadmap_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get strategic alignment assessment between board and roadmap
    """
    try:
        logger.info(f"Getting alignment assessment for board {board_id} and roadmap {roadmap_id}")
        
        integration_system = BoardStrategicIntegration()
        
        # Get board and roadmap (in real implementation, fetch from database)
        board = await _get_board_by_id(board_id)
        roadmap = await _get_roadmap_by_id(roadmap_id)
        
        # Assess alignment
        alignment = await integration_system._assess_strategic_alignment(board, roadmap)
        
        return {
            "status": "success",
            "alignment_assessment": {
                "board_id": alignment.board_id,
                "strategic_plan_id": alignment.strategic_plan_id,
                "alignment_score": alignment.alignment_score,
                "priority_matches_count": len(alignment.priority_matches),
                "concern_areas": alignment.concern_areas,
                "recommendations": alignment.recommendations,
                "last_updated": alignment.last_updated.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting alignment assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/board-priorities/{board_id}")
async def get_board_priorities(
    board_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get consolidated board priorities for strategic planning
    """
    try:
        logger.info(f"Getting board priorities for board {board_id}")
        
        integration_system = BoardStrategicIntegration()
        
        # Get board (in real implementation, fetch from database)
        board = await _get_board_by_id(board_id)
        
        # Extract priorities
        priorities = integration_system._extract_board_priorities(board.members)
        risk_tolerance = integration_system._assess_board_risk_tolerance(board.members)
        
        return {
            "status": "success",
            "board_priorities": {
                "board_id": board_id,
                "consolidated_priorities": priorities,
                "risk_tolerance": risk_tolerance,
                "priority_count": len(priorities),
                "top_priority": priorities[0] if priorities else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting board priorities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions (in real implementation, these would interact with database)

async def _get_board_by_id(board_id: str) -> Board:
    """Get board by ID (mock implementation)"""
    # In real implementation, fetch from database
    from ...models.board_dynamics_models import BoardMember
    from ...engines.board_dynamics_engine import Background, Priority, InfluenceLevel, CommunicationStyle, DecisionPattern
    
    # Mock board data
    mock_member = BoardMember(
        id="member_1",
        name="John Smith",
        background=Background(
            industry_experience=["technology", "finance"],
            functional_expertise=["strategy", "operations"],
            education=["MBA", "Engineering"],
            previous_roles=["CEO", "CTO"],
            years_experience=20
        ),
        expertise_areas=["technology", "strategy"],
        influence_level=InfluenceLevel.HIGH,
        communication_style=CommunicationStyle.ANALYTICAL,
        decision_making_pattern=DecisionPattern.DATA_DRIVEN,
        priorities=[
            Priority(
                area="AI Innovation",
                importance=0.9,
                description="Focus on AI advancement",
                timeline="2-3 years"
            )
        ]
    )
    
    return Board(
        id=board_id,
        name="Mock Board",
        members=[mock_member],
        committees=[],
        governance_structure={},
        meeting_schedule=[],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


async def _get_roadmap_by_id(roadmap_id: str) -> StrategicRoadmap:
    """Get roadmap by ID (mock implementation)"""
    # In real implementation, fetch from database
    from ...models.strategic_planning_models import TechnologyBet, StrategicMilestone, RiskAssessment, SuccessMetric
    from ...models.strategic_planning_models import TechnologyDomain, InvestmentRisk, MarketImpact, CompetitivePosition
    
    # Mock roadmap data
    mock_vision = TechnologyVision(
        title="AI Leadership Vision",
        description="Achieve AI technology leadership",
        time_horizon=5,
        key_technologies=["AI", "ML"],
        market_assumptions={"growth": "high"},
        success_criteria=["market_share"],
        stakeholders=["CTO"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    return StrategicRoadmap(
        id=roadmap_id,
        name="Mock Strategic Roadmap",
        description="Mock roadmap for testing",
        vision=mock_vision,
        time_horizon=5,
        milestones=[],
        technology_bets=[],
        risk_assessments=[],
        success_metrics=[],
        competitive_positioning=CompetitivePosition.LEADER,
        market_assumptions={"growth": "high"},
        resource_allocation={},
        scenario_plans=[],
        review_schedule=[],
        stakeholders=["CTO"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


async def _assess_and_log_alignment(board: Board, roadmap: StrategicRoadmap):
    """Background task to assess and log alignment"""
    try:
        integration_system = BoardStrategicIntegration()
        alignment = await integration_system._assess_strategic_alignment(board, roadmap)
        logger.info(f"Alignment assessment completed: {alignment.alignment_score:.2f}")
    except Exception as e:
        logger.error(f"Error in background alignment assessment: {str(e)}")