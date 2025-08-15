"""
Risk-Benefit Analysis API Routes for Crisis Leadership Excellence

This module provides REST API endpoints for risk-benefit analysis,
including response option evaluation, mitigation strategy generation,
and benefit optimization.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
import logging
from datetime import datetime

from ...engines.risk_benefit_analyzer import RiskBenefitAnalyzer
from ...models.risk_benefit_models import (
    RiskFactor, BenefitFactor, ResponseOption, MitigationStrategy,
    TradeOffAnalysis, RiskBenefitEvaluation, OptimizationResult,
    RiskLevel, BenefitType, UncertaintyLevel
)
from ...core.auth import get_current_user

router = APIRouter(prefix="/api/v1/risk-benefit", tags=["risk-benefit-analysis"])
logger = logging.getLogger(__name__)


@router.post("/evaluate", response_model=Dict)
async def evaluate_response_options(
    crisis_id: str,
    response_options: List[Dict],
    evaluation_criteria: Dict[str, float],
    risk_tolerance: str = "medium",
    time_pressure: str = "moderate",
    current_user: Dict = Depends(get_current_user)
):
    """
    Evaluate multiple response options using risk-benefit analysis.
    
    Args:
        crisis_id: ID of the crisis being addressed
        response_options: List of response options to evaluate
        evaluation_criteria: Weighted criteria for evaluation
        risk_tolerance: Organization's risk tolerance (low/medium/high)
        time_pressure: Level of time pressure (immediate/urgent/moderate/low)
        
    Returns:
        Complete risk-benefit evaluation with recommendations
    """
    try:
        logger.info(f"User {current_user.get('id')} evaluating response options for crisis {crisis_id}")
        
        # Initialize analyzer
        analyzer = RiskBenefitAnalyzer()
        
        # Convert input data to models
        option_models = []
        for option_data in response_options:
            # Convert risks
            risks = []
            for risk_data in option_data.get('risks', []):
                risk = RiskFactor(
                    name=risk_data.get('name', ''),
                    description=risk_data.get('description', ''),
                    category=risk_data.get('category', ''),
                    probability=risk_data.get('probability', 0.0),
                    impact_severity=RiskLevel(risk_data.get('impact_severity', 'medium')),
                    potential_impact=risk_data.get('potential_impact', ''),
                    time_horizon=risk_data.get('time_horizon', 'short_term'),
                    uncertainty_level=UncertaintyLevel(risk_data.get('uncertainty_level', 'moderate'))
                )
                risks.append(risk)
            
            # Convert benefits
            benefits = []
            for benefit_data in option_data.get('benefits', []):
                benefit = BenefitFactor(
                    name=benefit_data.get('name', ''),
                    description=benefit_data.get('description', ''),
                    benefit_type=BenefitType(benefit_data.get('benefit_type', 'operational')),
                    expected_value=benefit_data.get('expected_value', 0.0),
                    probability_of_realization=benefit_data.get('probability_of_realization', 0.0),
                    time_to_realization=benefit_data.get('time_to_realization', 'short_term'),
                    sustainability=benefit_data.get('sustainability', 'medium_term'),
                    uncertainty_level=UncertaintyLevel(benefit_data.get('uncertainty_level', 'moderate'))
                )
                benefits.append(benefit)
            
            # Create response option
            option = ResponseOption(
                name=option_data.get('name', ''),
                description=option_data.get('description', ''),
                category=option_data.get('category', ''),
                implementation_complexity=option_data.get('implementation_complexity', 'medium'),
                resource_requirements=option_data.get('resource_requirements', {}),
                time_to_implement=option_data.get('time_to_implement', ''),
                risks=risks,
                benefits=benefits,
                dependencies=option_data.get('dependencies', []),
                success_criteria=option_data.get('success_criteria', [])
            )
            option_models.append(option)
        
        # Perform evaluation
        evaluation = analyzer.evaluate_response_options(
            crisis_id=crisis_id,
            response_options=option_models,
            evaluation_criteria=evaluation_criteria,
            risk_tolerance=RiskLevel(risk_tolerance),
            time_pressure=time_pressure
        )
        
        # Convert to response format
        response = {
            "evaluation_id": evaluation.id,
            "crisis_id": evaluation.crisis_id,
            "recommended_option": evaluation.recommended_option,
            "confidence_score": evaluation.confidence_score,
            "uncertainty_factors": evaluation.uncertainty_factors,
            "trade_off_analyses": [
                {
                    "option_a_id": analysis.option_a_id,
                    "option_b_id": analysis.option_b_id,
                    "recommendation": analysis.recommendation,
                    "confidence_level": analysis.confidence_level,
                    "decision_rationale": analysis.decision_rationale
                }
                for analysis in evaluation.trade_off_analyses
            ],
            "mitigation_plan": [
                {
                    "name": strategy.name,
                    "description": strategy.description,
                    "effectiveness_score": strategy.effectiveness_score,
                    "implementation_time": strategy.implementation_time,
                    "success_probability": strategy.success_probability
                }
                for strategy in evaluation.mitigation_plan
            ],
            "monitoring_requirements": evaluation.monitoring_requirements,
            "created_at": evaluation.created_at.isoformat()
        }
        
        logger.info(f"Risk-benefit evaluation completed for crisis {crisis_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in risk-benefit evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/optimize-benefits", response_model=Dict)
async def optimize_benefits(
    evaluation_id: str,
    optimization_objective: str = "maximize_total_value",
    current_user: Dict = Depends(get_current_user)
):
    """
    Optimize benefits for a completed risk-benefit evaluation.
    
    Args:
        evaluation_id: ID of the evaluation to optimize
        optimization_objective: Objective for optimization
        
    Returns:
        Optimization result with enhanced benefits
    """
    try:
        logger.info(f"User {current_user.get('id')} optimizing benefits for evaluation {evaluation_id}")
        
        # Note: In a real implementation, you would retrieve the evaluation from storage
        # For now, we'll create a mock evaluation for demonstration
        
        analyzer = RiskBenefitAnalyzer()
        
        # Create mock evaluation (in real implementation, retrieve from database)
        mock_option = ResponseOption(
            name="Sample Response Option",
            description="Mock option for optimization",
            benefits=[
                BenefitFactor(
                    name="Operational Efficiency",
                    benefit_type=BenefitType.OPERATIONAL,
                    expected_value=0.7,
                    probability_of_realization=0.8
                ),
                BenefitFactor(
                    name="Cost Savings",
                    benefit_type=BenefitType.FINANCIAL,
                    expected_value=0.6,
                    probability_of_realization=0.7
                )
            ]
        )
        
        mock_evaluation = RiskBenefitEvaluation(
            crisis_id="mock_crisis",
            response_options=[mock_option],
            recommended_option=mock_option.id
        )
        
        # Perform optimization
        optimization_result = analyzer.optimize_benefits(
            evaluation=mock_evaluation,
            optimization_objective=optimization_objective
        )
        
        # Convert to response format
        response = {
            "optimization_id": optimization_result.id,
            "evaluation_id": optimization_result.evaluation_id,
            "optimization_objective": optimization_result.optimization_objective,
            "expected_improvement": optimization_result.expected_improvement,
            "optimization_strategies": optimization_result.optimization_strategies,
            "optimized_benefits": [
                {
                    "name": benefit.name,
                    "benefit_type": benefit.benefit_type.value,
                    "expected_value": benefit.expected_value,
                    "probability_of_realization": benefit.probability_of_realization,
                    "optimization_strategies": benefit.optimization_strategies
                }
                for benefit in optimization_result.optimized_benefits
            ],
            "implementation_requirements": optimization_result.implementation_requirements,
            "success_probability": optimization_result.success_probability,
            "created_at": optimization_result.created_at.isoformat()
        }
        
        logger.info(f"Benefit optimization completed for evaluation {evaluation_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in benefit optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/generate-mitigation", response_model=Dict)
async def generate_mitigation_strategies(
    risks: List[Dict],
    risk_tolerance: str = "medium",
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate mitigation strategies for identified risks.
    
    Args:
        risks: List of risk factors to mitigate
        risk_tolerance: Organization's risk tolerance level
        
    Returns:
        Generated mitigation strategies
    """
    try:
        logger.info(f"User {current_user.get('id')} generating mitigation strategies")
        
        # Convert risks to models
        risk_models = []
        for risk_data in risks:
            risk = RiskFactor(
                name=risk_data.get('name', ''),
                description=risk_data.get('description', ''),
                category=risk_data.get('category', ''),
                probability=risk_data.get('probability', 0.0),
                impact_severity=RiskLevel(risk_data.get('impact_severity', 'medium')),
                potential_impact=risk_data.get('potential_impact', ''),
                time_horizon=risk_data.get('time_horizon', 'short_term'),
                uncertainty_level=UncertaintyLevel(risk_data.get('uncertainty_level', 'moderate'))
            )
            risk_models.append(risk)
        
        # Generate mitigation strategies
        analyzer = RiskBenefitAnalyzer()
        strategies = analyzer._generate_mitigation_strategies(risk_models)
        
        # Convert to response format
        response = {
            "mitigation_strategies": [
                {
                    "id": strategy.id,
                    "name": strategy.name,
                    "description": strategy.description,
                    "target_risks": strategy.target_risks,
                    "effectiveness_score": strategy.effectiveness_score,
                    "implementation_cost": strategy.implementation_cost,
                    "implementation_time": strategy.implementation_time,
                    "resource_requirements": strategy.resource_requirements,
                    "success_probability": strategy.success_probability,
                    "created_at": strategy.created_at.isoformat()
                }
                for strategy in strategies
            ],
            "total_strategies": len(strategies),
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Generated {len(strategies)} mitigation strategies")
        return response
        
    except Exception as e:
        logger.error(f"Error generating mitigation strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Strategy generation failed: {str(e)}")


@router.post("/trade-off-analysis", response_model=Dict)
async def perform_trade_off_analysis(
    option_a: Dict,
    option_b: Dict,
    comparison_criteria: List[str],
    current_user: Dict = Depends(get_current_user)
):
    """
    Perform detailed trade-off analysis between two response options.
    
    Args:
        option_a: First response option
        option_b: Second response option
        comparison_criteria: Criteria for comparison
        
    Returns:
        Detailed trade-off analysis
    """
    try:
        logger.info(f"User {current_user.get('id')} performing trade-off analysis")
        
        # Convert options to models (simplified for demonstration)
        def convert_option(option_data):
            risks = [
                RiskFactor(
                    name=risk.get('name', ''),
                    probability=risk.get('probability', 0.0),
                    impact_severity=RiskLevel(risk.get('impact_severity', 'medium'))
                )
                for risk in option_data.get('risks', [])
            ]
            
            benefits = [
                BenefitFactor(
                    name=benefit.get('name', ''),
                    expected_value=benefit.get('expected_value', 0.0),
                    probability_of_realization=benefit.get('probability_of_realization', 0.0)
                )
                for benefit in option_data.get('benefits', [])
            ]
            
            return ResponseOption(
                name=option_data.get('name', ''),
                description=option_data.get('description', ''),
                risks=risks,
                benefits=benefits
            )
        
        option_a_model = convert_option(option_a)
        option_b_model = convert_option(option_b)
        
        # Perform analysis
        analyzer = RiskBenefitAnalyzer()
        
        # Calculate scores for both options
        risk_score_a = analyzer._calculate_risk_score(option_a_model.risks, RiskLevel.MEDIUM)
        benefit_score_a = analyzer._calculate_benefit_score(option_a_model.benefits)
        
        risk_score_b = analyzer._calculate_risk_score(option_b_model.risks, RiskLevel.MEDIUM)
        benefit_score_b = analyzer._calculate_benefit_score(option_b_model.benefits)
        
        # Generate trade-off analysis
        trade_off_factors = {
            'risk_difference': risk_score_a - risk_score_b,
            'benefit_difference': benefit_score_a - benefit_score_b,
            'overall_difference': (benefit_score_a - risk_score_a) - (benefit_score_b - risk_score_b)
        }
        
        # Generate recommendation
        if trade_off_factors['overall_difference'] > 0.1:
            recommendation = f"Strong preference for {option_a_model.name}"
            confidence = 0.8
        elif trade_off_factors['overall_difference'] > 0.05:
            recommendation = f"Moderate preference for {option_a_model.name}"
            confidence = 0.6
        elif trade_off_factors['overall_difference'] < -0.1:
            recommendation = f"Strong preference for {option_b_model.name}"
            confidence = 0.8
        elif trade_off_factors['overall_difference'] < -0.05:
            recommendation = f"Moderate preference for {option_b_model.name}"
            confidence = 0.6
        else:
            recommendation = "Options are closely matched - consider additional factors"
            confidence = 0.4
        
        response = {
            "option_a": {
                "name": option_a_model.name,
                "risk_score": risk_score_a,
                "benefit_score": benefit_score_a,
                "overall_score": benefit_score_a - risk_score_a
            },
            "option_b": {
                "name": option_b_model.name,
                "risk_score": risk_score_b,
                "benefit_score": benefit_score_b,
                "overall_score": benefit_score_b - risk_score_b
            },
            "comparison_criteria": comparison_criteria,
            "trade_off_factors": trade_off_factors,
            "recommendation": recommendation,
            "confidence_level": confidence,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info("Trade-off analysis completed")
        return response
        
    except Exception as e:
        logger.error(f"Error in trade-off analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trade-off analysis failed: {str(e)}")


@router.get("/risk-levels", response_model=Dict)
async def get_risk_levels():
    """Get available risk levels for the system."""
    return {
        "risk_levels": [level.value for level in RiskLevel],
        "descriptions": {
            "critical": "Immediate threat to organization survival",
            "high": "Significant impact on operations or reputation",
            "medium": "Moderate impact requiring attention",
            "low": "Minor impact with manageable consequences",
            "minimal": "Negligible impact on operations"
        }
    }


@router.get("/benefit-types", response_model=Dict)
async def get_benefit_types():
    """Get available benefit types for the system."""
    return {
        "benefit_types": [benefit_type.value for benefit_type in BenefitType],
        "descriptions": {
            "financial": "Direct financial gains or cost savings",
            "operational": "Improvements in operational efficiency",
            "strategic": "Long-term strategic advantages",
            "reputational": "Enhancement of organization reputation",
            "stakeholder": "Improved stakeholder relationships",
            "competitive": "Competitive market advantages"
        }
    }


@router.get("/uncertainty-levels", response_model=Dict)
async def get_uncertainty_levels():
    """Get available uncertainty levels for the system."""
    return {
        "uncertainty_levels": [level.value for level in UncertaintyLevel],
        "descriptions": {
            "very_high": "Extremely uncertain with limited information",
            "high": "High uncertainty requiring careful monitoring",
            "moderate": "Moderate uncertainty with reasonable estimates",
            "low": "Low uncertainty with good information",
            "very_low": "Very low uncertainty with high confidence"
        }
    }