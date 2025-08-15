"""
Strategic Recommendation API Routes

API endpoints for strategic recommendation development, quality assessment,
optimization, and validation functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from scrollintel.engines.strategic_recommendation_engine import (
    StrategicRecommendationEngine,
    RecommendationType,
    PriorityLevel,
    BoardPriority,
    StrategicRecommendation
)
from scrollintel.models.strategic_recommendation_models import (
    BoardPriorityCreate,
    BoardPriorityResponse,
    StrategicRecommendationCreate,
    StrategicRecommendationResponse,
    RecommendationValidationCreate,
    RecommendationValidationResponse,
    RecommendationFeedbackCreate,
    RecommendationFeedbackResponse,
    RecommendationSummaryResponse,
    RecommendationOptimizationRequest,
    RecommendationOptimizationResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategic-recommendations", tags=["strategic-recommendations"])

# Global engine instance (in production, this would be dependency injected)
recommendation_engine = StrategicRecommendationEngine()

@router.post("/board-priorities", response_model=BoardPriorityResponse)
async def create_board_priority(priority_data: BoardPriorityCreate):
    """Create a new board priority"""
    try:
        priority = BoardPriority(
            id=f"priority_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=priority_data.title,
            description=priority_data.description,
            priority_level=PriorityLevel(priority_data.priority_level),
            impact_areas=[area for area in priority_data.impact_areas],
            target_timeline=priority_data.target_timeline,
            success_metrics=priority_data.success_metrics,
            stakeholders=priority_data.stakeholders
        )
        
        recommendation_engine.add_board_priority(priority)
        
        return BoardPriorityResponse(
            id=priority.id,
            title=priority.title,
            description=priority.description,
            priority_level=priority.priority_level.value,
            impact_areas=[area.value if hasattr(area, 'value') else str(area) for area in priority.impact_areas],
            target_timeline=priority.target_timeline,
            success_metrics=priority.success_metrics,
            stakeholders=priority.stakeholders,
            created_at=priority.created_at,
            updated_at=priority.created_at
        )
        
    except Exception as e:
        logger.error(f"Error creating board priority: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create board priority: {str(e)}")

@router.get("/board-priorities", response_model=List[BoardPriorityResponse])
async def get_board_priorities():
    """Get all board priorities"""
    try:
        priorities = []
        for priority in recommendation_engine.board_priorities:
            priorities.append(BoardPriorityResponse(
                id=priority.id,
                title=priority.title,
                description=priority.description,
                priority_level=priority.priority_level.value,
                impact_areas=[area.value if hasattr(area, 'value') else str(area) for area in priority.impact_areas],
                target_timeline=priority.target_timeline,
                success_metrics=priority.success_metrics,
                stakeholders=priority.stakeholders,
                created_at=priority.created_at,
                updated_at=priority.created_at
            ))
        
        return priorities
        
    except Exception as e:
        logger.error(f"Error retrieving board priorities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve board priorities: {str(e)}")

@router.post("/recommendations", response_model=StrategicRecommendationResponse)
async def create_strategic_recommendation(recommendation_data: StrategicRecommendationCreate):
    """Create a new strategic recommendation"""
    try:
        recommendation = recommendation_engine.create_strategic_recommendation(
            title=recommendation_data.title,
            recommendation_type=RecommendationType(recommendation_data.recommendation_type),
            strategic_context=recommendation_data.strategic_context,
            target_priorities=recommendation_data.board_priorities
        )
        
        return _convert_to_response(recommendation)
        
    except Exception as e:
        logger.error(f"Error creating strategic recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create strategic recommendation: {str(e)}")

@router.get("/recommendations", response_model=List[StrategicRecommendationResponse])
async def get_strategic_recommendations(
    priority_id: Optional[str] = Query(None, description="Filter by board priority ID"),
    min_quality_score: Optional[float] = Query(None, description="Minimum quality score filter"),
    recommendation_type: Optional[str] = Query(None, description="Filter by recommendation type")
):
    """Get strategic recommendations with optional filters"""
    try:
        recommendations = recommendation_engine.recommendations
        
        # Apply filters
        if priority_id:
            recommendations = [r for r in recommendations if priority_id in r.board_priorities]
        
        if min_quality_score is not None:
            recommendations = [r for r in recommendations if r.quality_score >= min_quality_score]
        
        if recommendation_type:
            recommendations = [r for r in recommendations if r.recommendation_type.value == recommendation_type]
        
        return [_convert_to_response(rec) for rec in recommendations]
        
    except Exception as e:
        logger.error(f"Error retrieving strategic recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve strategic recommendations: {str(e)}")

@router.get("/recommendations/{recommendation_id}", response_model=StrategicRecommendationResponse)
async def get_strategic_recommendation(recommendation_id: str):
    """Get a specific strategic recommendation"""
    try:
        recommendation = next(
            (r for r in recommendation_engine.recommendations if r.id == recommendation_id), 
            None
        )
        
        if not recommendation:
            raise HTTPException(status_code=404, detail="Strategic recommendation not found")
        
        return _convert_to_response(recommendation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving strategic recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve strategic recommendation: {str(e)}")

@router.post("/recommendations/{recommendation_id}/optimize", response_model=RecommendationOptimizationResponse)
async def optimize_recommendation(recommendation_id: str, optimization_request: RecommendationOptimizationRequest):
    """Optimize a strategic recommendation to improve quality score"""
    try:
        # Get original score
        original_recommendation = next(
            (r for r in recommendation_engine.recommendations if r.id == recommendation_id), 
            None
        )
        
        if not original_recommendation:
            raise HTTPException(status_code=404, detail="Strategic recommendation not found")
        
        original_score = original_recommendation.quality_score
        
        # Optimize recommendation
        optimized_recommendation = recommendation_engine.optimize_recommendation(recommendation_id)
        
        improvement = optimized_recommendation.quality_score - original_score
        
        return RecommendationOptimizationResponse(
            recommendation_id=recommendation_id,
            original_score=original_score,
            optimized_score=optimized_recommendation.quality_score,
            improvement=improvement,
            optimization_details={
                "strategic_alignment_improved": len(optimized_recommendation.board_priorities) > len(original_recommendation.board_priorities),
                "financial_viability_improved": optimized_recommendation.financial_impact.roi_projection > original_recommendation.financial_impact.roi_projection,
                "implementation_feasibility_improved": optimized_recommendation.risk_assessment.success_probability > original_recommendation.risk_assessment.success_probability,
                "optimization_areas": optimization_request.optimization_areas
            },
            updated_at=optimized_recommendation.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize recommendation: {str(e)}")

@router.post("/recommendations/{recommendation_id}/validate", response_model=RecommendationValidationResponse)
async def validate_recommendation(recommendation_id: str, validation_request: RecommendationValidationCreate):
    """Validate a strategic recommendation against quality standards"""
    try:
        validation_results = recommendation_engine.validate_recommendation(recommendation_id)
        
        return RecommendationValidationResponse(
            recommendation_id=validation_results['recommendation_id'],
            validation_status=validation_results['validation_status'],
            quality_score=validation_results['quality_score'],
            meets_threshold=validation_results['meets_threshold'],
            validation_details=validation_results['validation_details'],
            improvement_recommendations=validation_results['recommendations_for_improvement'],
            validated_by=validation_request.validated_by,
            validated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error validating recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate recommendation: {str(e)}")

@router.get("/recommendations/{recommendation_id}/summary", response_model=RecommendationSummaryResponse)
async def get_recommendation_summary(recommendation_id: str):
    """Get executive summary of a strategic recommendation"""
    try:
        summary = recommendation_engine.generate_recommendation_summary(recommendation_id)
        
        return RecommendationSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"Error generating recommendation summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendation summary: {str(e)}")

@router.get("/recommendations/by-priority/{priority_id}", response_model=List[StrategicRecommendationResponse])
async def get_recommendations_by_priority(priority_id: str):
    """Get all recommendations aligned with a specific board priority"""
    try:
        recommendations = recommendation_engine.get_recommendations_by_priority(priority_id)
        
        return [_convert_to_response(rec) for rec in recommendations]
        
    except Exception as e:
        logger.error(f"Error retrieving recommendations by priority: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recommendations by priority: {str(e)}")

@router.get("/recommendations/high-quality", response_model=List[StrategicRecommendationResponse])
async def get_high_quality_recommendations(min_score: Optional[float] = Query(None, description="Minimum quality score")):
    """Get recommendations that meet quality thresholds"""
    try:
        recommendations = recommendation_engine.get_high_quality_recommendations(min_score)
        
        return [_convert_to_response(rec) for rec in recommendations]
        
    except Exception as e:
        logger.error(f"Error retrieving high quality recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve high quality recommendations: {str(e)}")

@router.get("/analytics/quality-distribution")
async def get_quality_distribution():
    """Get distribution of recommendation quality scores"""
    try:
        recommendations = recommendation_engine.recommendations
        
        if not recommendations:
            return {
                "total_recommendations": 0,
                "quality_distribution": {},
                "average_quality": 0.0
            }
        
        quality_scores = [r.quality_score for r in recommendations]
        
        # Create quality score buckets
        buckets = {
            "excellent (0.9-1.0)": len([s for s in quality_scores if s >= 0.9]),
            "good (0.8-0.89)": len([s for s in quality_scores if 0.8 <= s < 0.9]),
            "fair (0.7-0.79)": len([s for s in quality_scores if 0.7 <= s < 0.8]),
            "poor (0.6-0.69)": len([s for s in quality_scores if 0.6 <= s < 0.7]),
            "very poor (<0.6)": len([s for s in quality_scores if s < 0.6])
        }
        
        return {
            "total_recommendations": len(recommendations),
            "quality_distribution": buckets,
            "average_quality": sum(quality_scores) / len(quality_scores),
            "highest_quality": max(quality_scores),
            "lowest_quality": min(quality_scores)
        }
        
    except Exception as e:
        logger.error(f"Error generating quality distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate quality distribution: {str(e)}")

@router.get("/analytics/priority-alignment")
async def get_priority_alignment_analytics():
    """Get analytics on how well recommendations align with board priorities"""
    try:
        recommendations = recommendation_engine.recommendations
        priorities = recommendation_engine.board_priorities
        
        if not recommendations or not priorities:
            return {
                "total_priorities": len(priorities),
                "total_recommendations": len(recommendations),
                "alignment_analytics": {}
            }
        
        alignment_data = {}
        
        for priority in priorities:
            aligned_recs = [r for r in recommendations if priority.id in r.board_priorities]
            avg_quality = sum(r.quality_score for r in aligned_recs) / len(aligned_recs) if aligned_recs else 0
            
            alignment_data[priority.title] = {
                "priority_id": priority.id,
                "priority_level": priority.priority_level.value,
                "aligned_recommendations": len(aligned_recs),
                "average_quality": avg_quality,
                "recommendation_types": list(set(r.recommendation_type.value for r in aligned_recs))
            }
        
        return {
            "total_priorities": len(priorities),
            "total_recommendations": len(recommendations),
            "alignment_analytics": alignment_data,
            "unaligned_recommendations": len([r for r in recommendations if not r.board_priorities])
        }
        
    except Exception as e:
        logger.error(f"Error generating priority alignment analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate priority alignment analytics: {str(e)}")

def _convert_to_response(recommendation: StrategicRecommendation) -> StrategicRecommendationResponse:
    """Convert engine recommendation to API response model"""
    from scrollintel.models.strategic_recommendation_models import (
        FinancialImpactResponse,
        RiskAssessmentResponse,
        ImplementationPlanResponse
    )
    
    financial_impact = FinancialImpactResponse(
        revenue_impact=recommendation.financial_impact.revenue_impact,
        cost_impact=recommendation.financial_impact.cost_impact,
        roi_projection=recommendation.financial_impact.roi_projection,
        payback_period=recommendation.financial_impact.payback_period,
        confidence_level=recommendation.financial_impact.confidence_level,
        assumptions=recommendation.financial_impact.assumptions
    )
    
    risk_assessment = RiskAssessmentResponse(
        risk_level=recommendation.risk_assessment.risk_level,
        key_risks=recommendation.risk_assessment.key_risks,
        mitigation_strategies=recommendation.risk_assessment.mitigation_strategies,
        success_probability=recommendation.risk_assessment.success_probability,
        contingency_plans=recommendation.risk_assessment.contingency_plans
    )
    
    implementation_plan = ImplementationPlanResponse(
        phases=recommendation.implementation_plan.phases,
        timeline=recommendation.implementation_plan.timeline,
        resource_requirements=recommendation.implementation_plan.resource_requirements,
        dependencies=recommendation.implementation_plan.dependencies,
        milestones=recommendation.implementation_plan.milestones,
        success_criteria=recommendation.implementation_plan.success_criteria
    )
    
    return StrategicRecommendationResponse(
        id=recommendation.id,
        title=recommendation.title,
        recommendation_type=recommendation.recommendation_type.value,
        board_priorities=recommendation.board_priorities,
        strategic_rationale=recommendation.strategic_rationale,
        quality_score=recommendation.quality_score,
        impact_prediction=recommendation.impact_prediction,
        validation_status=recommendation.validation_status,
        financial_impact=financial_impact,
        risk_assessment=risk_assessment,
        implementation_plan=implementation_plan,
        created_at=recommendation.created_at,
        updated_at=recommendation.updated_at
    )