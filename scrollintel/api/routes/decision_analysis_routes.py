"""
Decision Analysis API Routes for Board Executive Mastery System

This module provides REST API endpoints for executive decision analysis and recommendation.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.decision_analysis_engine import DecisionAnalysisEngine
from ...models.decision_analysis_models import (
    DecisionAnalysis, DecisionOption, DecisionCriteria, StakeholderImpact,
    RiskAssessment, DecisionRecommendation, DecisionImpactAssessment,
    DecisionOptimization, DecisionVisualization, DecisionType, DecisionUrgency,
    DecisionComplexity, ImpactLevel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/decision-analysis", tags=["decision-analysis"])

# Global engine instance
decision_engine = DecisionAnalysisEngine()


# Request/Response Models
from pydantic import BaseModel

class CreateDecisionAnalysisRequest(BaseModel):
    title: str
    description: str
    decision_type: DecisionType
    urgency: DecisionUrgency = DecisionUrgency.MEDIUM
    complexity: DecisionComplexity = DecisionComplexity.MODERATE
    deadline: Optional[datetime] = None
    background: str = ""
    current_situation: str = ""
    decision_drivers: List[str] = []
    constraints: List[str] = []
    analyst_id: str = "system"


class AddDecisionOptionRequest(BaseModel):
    title: str
    description: str
    pros: List[str]
    cons: List[str]
    estimated_cost: Optional[float] = None
    estimated_timeline: Optional[str] = None
    risk_level: ImpactLevel = ImpactLevel.MODERATE
    expected_outcome: str = ""
    success_probability: float = 0.5


class ScoreOptionCriteriaRequest(BaseModel):
    criteria_scores: Dict[str, float]


class AddStakeholderImpactRequest(BaseModel):
    stakeholder_id: str
    stakeholder_name: str
    impact_level: ImpactLevel
    impact_description: str
    support_likelihood: float = 0.5
    concerns: List[str] = []
    mitigation_strategies: List[str] = []


class AddRiskAssessmentRequest(BaseModel):
    risk_category: str
    probability: float
    impact: ImpactLevel
    description: str
    mitigation_strategies: List[str] = []
    contingency_plans: List[str] = []


class CreateVisualizationRequest(BaseModel):
    visualization_type: str = "comparison_matrix"


# API Endpoints

@router.post("/analyses", response_model=Dict[str, Any])
async def create_decision_analysis(request: CreateDecisionAnalysisRequest):
    """Create a new decision analysis framework"""
    try:
        analysis = decision_engine.create_decision_analysis(
            title=request.title,
            description=request.description,
            decision_type=request.decision_type,
            urgency=request.urgency,
            complexity=request.complexity,
            deadline=request.deadline,
            background=request.background,
            current_situation=request.current_situation,
            decision_drivers=request.decision_drivers,
            constraints=request.constraints,
            analyst_id=request.analyst_id
        )
        
        return {
            "status": "success",
            "message": "Decision analysis created successfully",
            "analysis_id": analysis.id,
            "analysis": {
                "id": analysis.id,
                "title": analysis.title,
                "decision_type": analysis.decision_type.value,
                "urgency": analysis.urgency.value,
                "complexity": analysis.complexity.value,
                "created_at": analysis.created_at.isoformat(),
                "criteria_count": len(analysis.criteria),
                "template_applied": len(analysis.criteria) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating decision analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses", response_model=Dict[str, Any])
async def list_decision_analyses():
    """List all decision analyses"""
    try:
        analyses = decision_engine.list_decision_analyses()
        
        analyses_summary = []
        for analysis in analyses:
            analyses_summary.append({
                "id": analysis.id,
                "title": analysis.title,
                "decision_type": analysis.decision_type.value,
                "urgency": analysis.urgency.value,
                "complexity": analysis.complexity.value,
                "created_at": analysis.created_at.isoformat(),
                "options_count": len(analysis.options),
                "has_recommendation": analysis.recommended_option_id is not None,
                "quality_score": analysis.analysis_quality_score,
                "confidence_level": analysis.confidence_level
            })
        
        return {
            "status": "success",
            "analyses": analyses_summary,
            "total_count": len(analyses)
        }
        
    except Exception as e:
        logger.error(f"Error listing decision analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{analysis_id}", response_model=Dict[str, Any])
async def get_decision_analysis(analysis_id: str):
    """Get detailed decision analysis"""
    try:
        analysis = decision_engine.get_decision_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Decision analysis not found")
        
        return {
            "status": "success",
            "analysis": {
                "id": analysis.id,
                "title": analysis.title,
                "description": analysis.description,
                "decision_type": analysis.decision_type.value,
                "urgency": analysis.urgency.value,
                "complexity": analysis.complexity.value,
                "created_at": analysis.created_at.isoformat(),
                "deadline": analysis.deadline.isoformat() if analysis.deadline else None,
                "background": analysis.background,
                "current_situation": analysis.current_situation,
                "decision_drivers": analysis.decision_drivers,
                "constraints": analysis.constraints,
                "criteria": [
                    {
                        "id": crit.id,
                        "name": crit.name,
                        "weight": crit.weight,
                        "description": crit.description
                    } for crit in analysis.criteria
                ],
                "options": [
                    {
                        "id": opt.id,
                        "title": opt.title,
                        "description": opt.description,
                        "pros": opt.pros,
                        "cons": opt.cons,
                        "estimated_cost": opt.estimated_cost,
                        "estimated_timeline": opt.estimated_timeline,
                        "risk_level": opt.risk_level.value,
                        "expected_outcome": opt.expected_outcome,
                        "success_probability": opt.success_probability,
                        "criteria_scores": opt.criteria_scores
                    } for opt in analysis.options
                ],
                "stakeholder_impacts": [
                    {
                        "stakeholder_id": si.stakeholder_id,
                        "stakeholder_name": si.stakeholder_name,
                        "impact_level": si.impact_level.value,
                        "impact_description": si.impact_description,
                        "support_likelihood": si.support_likelihood,
                        "concerns": si.concerns,
                        "mitigation_strategies": si.mitigation_strategies
                    } for si in analysis.stakeholder_impacts
                ],
                "risk_assessments": [
                    {
                        "id": risk.id,
                        "risk_category": risk.risk_category,
                        "probability": risk.probability,
                        "impact": risk.impact.value,
                        "description": risk.description,
                        "mitigation_strategies": risk.mitigation_strategies,
                        "contingency_plans": risk.contingency_plans
                    } for risk in analysis.risk_assessments
                ],
                "recommended_option_id": analysis.recommended_option_id,
                "recommendation_rationale": analysis.recommendation_rationale,
                "implementation_plan": analysis.implementation_plan,
                "success_metrics": analysis.success_metrics,
                "confidence_level": analysis.confidence_level,
                "analysis_quality_score": analysis.analysis_quality_score
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decision analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyses/{analysis_id}/options", response_model=Dict[str, Any])
async def add_decision_option(analysis_id: str, request: AddDecisionOptionRequest):
    """Add a decision option to the analysis"""
    try:
        option = decision_engine.add_decision_option(
            analysis_id=analysis_id,
            title=request.title,
            description=request.description,
            pros=request.pros,
            cons=request.cons,
            estimated_cost=request.estimated_cost,
            estimated_timeline=request.estimated_timeline,
            risk_level=request.risk_level,
            expected_outcome=request.expected_outcome,
            success_probability=request.success_probability
        )
        
        # Update quality score
        quality_score = decision_engine.update_analysis_quality_score(analysis_id)
        
        return {
            "status": "success",
            "message": "Decision option added successfully",
            "option_id": option.id,
            "analysis_quality_score": quality_score,
            "option": {
                "id": option.id,
                "title": option.title,
                "description": option.description,
                "success_probability": option.success_probability,
                "risk_level": option.risk_level.value
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding decision option: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/analyses/{analysis_id}/options/{option_id}/scores", response_model=Dict[str, Any])
async def score_option_criteria(analysis_id: str, option_id: str, request: ScoreOptionCriteriaRequest):
    """Score a decision option against criteria"""
    try:
        success = decision_engine.score_option_criteria(
            analysis_id=analysis_id,
            option_id=option_id,
            criteria_scores=request.criteria_scores
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Analysis or option not found")
        
        # Calculate updated scores
        option_scores = decision_engine.calculate_option_scores(analysis_id)
        quality_score = decision_engine.update_analysis_quality_score(analysis_id)
        
        return {
            "status": "success",
            "message": "Option criteria scored successfully",
            "option_scores": option_scores,
            "analysis_quality_score": quality_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring option criteria: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyses/{analysis_id}/stakeholders", response_model=Dict[str, Any])
async def add_stakeholder_impact(analysis_id: str, request: AddStakeholderImpactRequest):
    """Add stakeholder impact analysis"""
    try:
        impact = decision_engine.add_stakeholder_impact(
            analysis_id=analysis_id,
            stakeholder_id=request.stakeholder_id,
            stakeholder_name=request.stakeholder_name,
            impact_level=request.impact_level,
            impact_description=request.impact_description,
            support_likelihood=request.support_likelihood,
            concerns=request.concerns,
            mitigation_strategies=request.mitigation_strategies
        )
        
        quality_score = decision_engine.update_analysis_quality_score(analysis_id)
        
        return {
            "status": "success",
            "message": "Stakeholder impact added successfully",
            "analysis_quality_score": quality_score,
            "stakeholder_impact": {
                "stakeholder_name": impact.stakeholder_name,
                "impact_level": impact.impact_level.value,
                "support_likelihood": impact.support_likelihood
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding stakeholder impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyses/{analysis_id}/risks", response_model=Dict[str, Any])
async def add_risk_assessment(analysis_id: str, request: AddRiskAssessmentRequest):
    """Add risk assessment to decision analysis"""
    try:
        risk = decision_engine.add_risk_assessment(
            analysis_id=analysis_id,
            risk_category=request.risk_category,
            probability=request.probability,
            impact=request.impact,
            description=request.description,
            mitigation_strategies=request.mitigation_strategies,
            contingency_plans=request.contingency_plans
        )
        
        quality_score = decision_engine.update_analysis_quality_score(analysis_id)
        
        return {
            "status": "success",
            "message": "Risk assessment added successfully",
            "risk_id": risk.id,
            "analysis_quality_score": quality_score,
            "risk": {
                "id": risk.id,
                "risk_category": risk.risk_category,
                "probability": risk.probability,
                "impact": risk.impact.value
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyses/{analysis_id}/recommendation", response_model=Dict[str, Any])
async def generate_recommendation(analysis_id: str):
    """Generate executive decision recommendation"""
    try:
        recommendation = decision_engine.generate_recommendation(analysis_id)
        
        return {
            "status": "success",
            "message": "Recommendation generated successfully",
            "recommendation": {
                "id": recommendation.id,
                "title": recommendation.title,
                "executive_summary": recommendation.executive_summary,
                "recommended_action": recommendation.recommended_action,
                "key_benefits": recommendation.key_benefits,
                "critical_risks": recommendation.critical_risks,
                "resource_requirements": recommendation.resource_requirements,
                "timeline": recommendation.timeline,
                "success_probability": recommendation.success_probability,
                "confidence_level": recommendation.confidence_level,
                "next_steps": recommendation.next_steps,
                "approval_requirements": recommendation.approval_requirements
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{analysis_id}/impact-assessment", response_model=Dict[str, Any])
async def assess_decision_impact(analysis_id: str):
    """Assess comprehensive impact of decision"""
    try:
        impact_assessment = decision_engine.assess_decision_impact(analysis_id)
        
        return {
            "status": "success",
            "impact_assessment": {
                "id": impact_assessment.id,
                "financial_impact": impact_assessment.financial_impact,
                "roi_projection": impact_assessment.roi_projection,
                "payback_period": impact_assessment.payback_period,
                "strategic_alignment": impact_assessment.strategic_alignment,
                "competitive_advantage": impact_assessment.competitive_advantage,
                "market_position_impact": impact_assessment.market_position_impact,
                "operational_complexity": impact_assessment.operational_complexity.value,
                "resource_requirements": impact_assessment.resource_requirements,
                "implementation_difficulty": impact_assessment.implementation_difficulty,
                "overall_risk_level": impact_assessment.overall_risk_level.value,
                "board_support_likelihood": impact_assessment.board_support_likelihood,
                "employee_impact": impact_assessment.employee_impact,
                "customer_impact": impact_assessment.customer_impact,
                "investor_impact": impact_assessment.investor_impact
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error assessing decision impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyses/{analysis_id}/options/{option_id}/optimize", response_model=Dict[str, Any])
async def optimize_decision(analysis_id: str, option_id: str):
    """Generate optimization recommendations for a decision option"""
    try:
        optimization = decision_engine.optimize_decision(analysis_id, option_id)
        
        return {
            "status": "success",
            "optimization": {
                "id": optimization.id,
                "optimization_suggestions": optimization.optimization_suggestions,
                "enhanced_benefits": optimization.enhanced_benefits,
                "risk_reductions": optimization.risk_reductions,
                "cost_optimizations": optimization.cost_optimizations,
                "timeline_improvements": optimization.timeline_improvements,
                "optimized_success_probability": optimization.optimized_success_probability,
                "optimization_confidence": optimization.optimization_confidence,
                "implementation_complexity": optimization.implementation_complexity.value
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing decision: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyses/{analysis_id}/visualizations", response_model=Dict[str, Any])
async def create_decision_visualization(analysis_id: str, request: CreateVisualizationRequest):
    """Create visualization for decision analysis"""
    try:
        visualization = decision_engine.create_decision_visualization(
            analysis_id=analysis_id,
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
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{analysis_id}/scores", response_model=Dict[str, Any])
async def get_option_scores(analysis_id: str):
    """Get calculated scores for all options"""
    try:
        option_scores = decision_engine.calculate_option_scores(analysis_id)
        
        if not option_scores:
            raise HTTPException(status_code=404, detail="No options found for scoring")
        
        # Get option details for context
        analysis = decision_engine.get_decision_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        scored_options = []
        for option in analysis.options:
            if option.id in option_scores:
                scored_options.append({
                    "option_id": option.id,
                    "title": option.title,
                    "score": option_scores[option.id],
                    "success_probability": option.success_probability,
                    "risk_level": option.risk_level.value
                })
        
        # Sort by score descending
        scored_options.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "status": "success",
            "option_scores": scored_options,
            "best_option": scored_options[0] if scored_options else None,
            "analysis_quality_score": analysis.analysis_quality_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting option scores: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))