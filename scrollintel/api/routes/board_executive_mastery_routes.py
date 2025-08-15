"""
Board Executive Mastery API Routes
Complete API for board and executive engagement mastery
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from ...core.board_executive_mastery_system import (
    BoardExecutiveMasterySystem,
    BoardExecutiveMasteryConfig
)
from ...models.board_executive_mastery_models import (
    BoardExecutiveMasteryRequest,
    BoardExecutiveMasteryResponse,
    BoardEngagementPlan,
    ExecutiveInteractionStrategy,
    BoardMasteryMetrics,
    BOARD_MASTERY_REQUEST_SCHEMA,
    INTERACTION_REQUEST_SCHEMA,
    VALIDATION_REQUEST_SCHEMA
)
from ...core.schema_validation import validate_request_schema
from ...security.auth import get_current_user
from ...core.error_handling import handle_api_error

router = APIRouter(prefix="/api/v1/board-executive-mastery", tags=["Board Executive Mastery"])
logger = logging.getLogger(__name__)

# Initialize board executive mastery system
mastery_config = BoardExecutiveMasteryConfig(
    enable_real_time_adaptation=True,
    enable_predictive_analytics=True,
    enable_continuous_learning=True,
    board_confidence_threshold=0.85,
    executive_trust_threshold=0.80,
    strategic_alignment_threshold=0.90
)
mastery_system = BoardExecutiveMasterySystem(mastery_config)

@router.post("/create-engagement-plan", response_model=BoardExecutiveMasteryResponse)
async def create_engagement_plan(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Create comprehensive board engagement plan"""
    try:
        # Validate request schema
        validate_request_schema(request_data, BOARD_MASTERY_REQUEST_SCHEMA)
        
        # Create request object
        mastery_request = BoardExecutiveMasteryRequest(
            id=f"mastery_req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            board_info=request_data["board_info"],
            executives=request_data.get("executives", []),
            communication_context=request_data["communication_context"],
            presentation_requirements=request_data.get("presentation_requirements", {}),
            strategic_context=request_data["strategic_context"],
            meeting_context=request_data.get("meeting_context", {}),
            credibility_context=request_data.get("credibility_context", {}),
            success_criteria=request_data.get("success_criteria", {}),
            timeline=request_data.get("timeline", {}),
            created_at=datetime.now()
        )
        
        # Create engagement plan
        engagement_plan = await mastery_system.create_comprehensive_engagement_plan(
            mastery_request
        )
        
        # Log successful creation
        logger.info(f"Created board engagement plan: {engagement_plan.id}")
        
        return BoardExecutiveMasteryResponse(
            request_id=mastery_request.id,
            engagement_plan=engagement_plan,
            success=True,
            message="Board engagement plan created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating engagement plan: {str(e)}")
        raise handle_api_error(e, "Failed to create board engagement plan")

@router.post("/execute-interaction", response_model=BoardExecutiveMasteryResponse)
async def execute_board_interaction(
    request_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Execute real-time board interaction with adaptive strategy"""
    try:
        # Validate request schema
        validate_request_schema(request_data, INTERACTION_REQUEST_SCHEMA)
        
        engagement_id = request_data["engagement_id"]
        interaction_context = request_data["interaction_context"]
        
        # Execute interaction
        interaction_strategy = await mastery_system.execute_board_interaction(
            engagement_id,
            interaction_context
        )
        
        logger.info(f"Executed board interaction for engagement: {engagement_id}")
        
        return BoardExecutiveMasteryResponse(
            request_id=f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            interaction_strategy=interaction_strategy,
            success=True,
            message="Board interaction executed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error executing board interaction: {str(e)}")
        raise handle_api_error(e, "Failed to execute board interaction")

@router.post("/validate-mastery", response_model=BoardExecutiveMasteryResponse)
async def validate_board_mastery(
    request_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Validate effectiveness of board executive mastery"""
    try:
        # Validate request schema
        validate_request_schema(request_data, VALIDATION_REQUEST_SCHEMA)
        
        engagement_id = request_data["engagement_id"]
        validation_context = request_data["validation_context"]
        
        # Validate mastery effectiveness
        mastery_metrics = await mastery_system.validate_board_mastery_effectiveness(
            engagement_id,
            validation_context
        )
        
        logger.info(f"Validated board mastery for engagement: {engagement_id}")
        
        return BoardExecutiveMasteryResponse(
            request_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mastery_metrics=mastery_metrics,
            success=True,
            message="Board mastery validation completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error validating board mastery: {str(e)}")
        raise handle_api_error(e, "Failed to validate board mastery")

@router.post("/optimize-mastery", response_model=BoardExecutiveMasteryResponse)
async def optimize_board_mastery(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Optimize board executive mastery based on performance data"""
    try:
        engagement_id = request_data["engagement_id"]
        optimization_context = request_data.get("optimization_context", {})
        
        # Optimize mastery
        optimized_plan = await mastery_system.optimize_board_executive_mastery(
            engagement_id,
            optimization_context
        )
        
        logger.info(f"Optimized board mastery for engagement: {engagement_id}")
        
        return BoardExecutiveMasteryResponse(
            request_id=f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            engagement_plan=optimized_plan,
            success=True,
            message="Board mastery optimization completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error optimizing board mastery: {str(e)}")
        raise handle_api_error(e, "Failed to optimize board mastery")

@router.get("/engagement-plan/{engagement_id}", response_model=BoardExecutiveMasteryResponse)
async def get_engagement_plan(
    engagement_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get specific board engagement plan"""
    try:
        engagement_plan = mastery_system.active_engagements.get(engagement_id)
        
        if not engagement_plan:
            raise HTTPException(
                status_code=404,
                detail=f"Engagement plan not found: {engagement_id}"
            )
        
        return BoardExecutiveMasteryResponse(
            request_id=f"get_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            engagement_plan=engagement_plan,
            success=True,
            message="Engagement plan retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving engagement plan: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve engagement plan")

@router.get("/mastery-metrics/{engagement_id}", response_model=BoardExecutiveMasteryResponse)
async def get_mastery_metrics(
    engagement_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get board mastery metrics for specific engagement"""
    try:
        mastery_metrics = mastery_system.performance_metrics.get(engagement_id)
        
        if not mastery_metrics:
            raise HTTPException(
                status_code=404,
                detail=f"Mastery metrics not found: {engagement_id}"
            )
        
        return BoardExecutiveMasteryResponse(
            request_id=f"get_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mastery_metrics=mastery_metrics,
            success=True,
            message="Mastery metrics retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving mastery metrics: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve mastery metrics")

@router.get("/active-engagements", response_model=Dict[str, Any])
async def get_active_engagements(
    current_user: Dict = Depends(get_current_user)
):
    """Get all active board engagements"""
    try:
        active_engagements = list(mastery_system.active_engagements.keys())
        
        engagement_summaries = []
        for engagement_id in active_engagements:
            plan = mastery_system.active_engagements[engagement_id]
            metrics = mastery_system.performance_metrics.get(engagement_id)
            
            summary = {
                "engagement_id": engagement_id,
                "board_id": plan.board_id,
                "created_at": plan.created_at.isoformat(),
                "has_metrics": metrics is not None,
                "overall_score": metrics.overall_mastery_score if metrics else None
            }
            engagement_summaries.append(summary)
        
        return {
            "active_engagements": engagement_summaries,
            "total_count": len(active_engagements),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving active engagements: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve active engagements")

@router.get("/system-status", response_model=Dict[str, Any])
async def get_system_status(
    current_user: Dict = Depends(get_current_user)
):
    """Get comprehensive board executive mastery system status"""
    try:
        system_status = await mastery_system.get_mastery_system_status()
        
        return system_status
        
    except Exception as e:
        logger.error(f"Error retrieving system status: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve system status")

@router.delete("/engagement-plan/{engagement_id}")
async def delete_engagement_plan(
    engagement_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Delete specific board engagement plan"""
    try:
        if engagement_id not in mastery_system.active_engagements:
            raise HTTPException(
                status_code=404,
                detail=f"Engagement plan not found: {engagement_id}"
            )
        
        # Remove engagement plan and metrics
        del mastery_system.active_engagements[engagement_id]
        if engagement_id in mastery_system.performance_metrics:
            del mastery_system.performance_metrics[engagement_id]
        
        logger.info(f"Deleted engagement plan: {engagement_id}")
        
        return {
            "success": True,
            "message": f"Engagement plan {engagement_id} deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting engagement plan: {str(e)}")
        raise handle_api_error(e, "Failed to delete engagement plan")

@router.post("/batch-validate")
async def batch_validate_mastery(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Batch validate multiple board engagements"""
    try:
        engagement_ids = request_data.get("engagement_ids", [])
        validation_context = request_data.get("validation_context", {})
        
        if not engagement_ids:
            raise HTTPException(
                status_code=400,
                detail="No engagement IDs provided for batch validation"
            )
        
        # Validate each engagement
        validation_results = []
        for engagement_id in engagement_ids:
            try:
                metrics = await mastery_system.validate_board_mastery_effectiveness(
                    engagement_id,
                    validation_context
                )
                validation_results.append({
                    "engagement_id": engagement_id,
                    "success": True,
                    "metrics": metrics
                })
            except Exception as e:
                validation_results.append({
                    "engagement_id": engagement_id,
                    "success": False,
                    "error": str(e)
                })
        
        successful_validations = sum(1 for r in validation_results if r["success"])
        
        logger.info(f"Batch validated {successful_validations}/{len(engagement_ids)} engagements")
        
        return {
            "validation_results": validation_results,
            "total_processed": len(engagement_ids),
            "successful_validations": successful_validations,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch validation: {str(e)}")
        raise handle_api_error(e, "Failed to perform batch validation")

@router.get("/performance-analytics")
async def get_performance_analytics(
    current_user: Dict = Depends(get_current_user)
):
    """Get comprehensive performance analytics for board executive mastery"""
    try:
        all_metrics = list(mastery_system.performance_metrics.values())
        
        if not all_metrics:
            return {
                "analytics": {
                    "total_engagements": 0,
                    "average_scores": {},
                    "success_rate": 0.0,
                    "trends": {}
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate analytics
        total_engagements = len(all_metrics)
        successful_engagements = sum(1 for m in all_metrics if m.meets_success_criteria)
        success_rate = successful_engagements / total_engagements
        
        # Average scores
        avg_board_confidence = sum(m.board_confidence_score for m in all_metrics) / total_engagements
        avg_executive_trust = sum(m.executive_trust_score for m in all_metrics) / total_engagements
        avg_strategic_alignment = sum(m.strategic_alignment_score for m in all_metrics) / total_engagements
        avg_communication_effectiveness = sum(m.communication_effectiveness_score for m in all_metrics) / total_engagements
        avg_stakeholder_influence = sum(m.stakeholder_influence_score for m in all_metrics) / total_engagements
        avg_overall_mastery = sum(m.overall_mastery_score for m in all_metrics) / total_engagements
        
        # Score distribution
        score_ranges = {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.85": 0, "0.85-1.0": 0}
        for metrics in all_metrics:
            score = metrics.overall_mastery_score
            if score < 0.5:
                score_ranges["0.0-0.5"] += 1
            elif score < 0.7:
                score_ranges["0.5-0.7"] += 1
            elif score < 0.85:
                score_ranges["0.7-0.85"] += 1
            else:
                score_ranges["0.85-1.0"] += 1
        
        return {
            "analytics": {
                "total_engagements": total_engagements,
                "successful_engagements": successful_engagements,
                "success_rate": success_rate,
                "average_scores": {
                    "board_confidence": avg_board_confidence,
                    "executive_trust": avg_executive_trust,
                    "strategic_alignment": avg_strategic_alignment,
                    "communication_effectiveness": avg_communication_effectiveness,
                    "stakeholder_influence": avg_stakeholder_influence,
                    "overall_mastery": avg_overall_mastery
                },
                "score_distribution": score_ranges,
                "performance_thresholds": {
                    "board_confidence_threshold": mastery_config.board_confidence_threshold,
                    "executive_trust_threshold": mastery_config.executive_trust_threshold,
                    "strategic_alignment_threshold": mastery_config.strategic_alignment_threshold
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving performance analytics: {str(e)}")
        raise handle_api_error(e, "Failed to retrieve performance analytics")