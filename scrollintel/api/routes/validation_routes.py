"""
API routes for Innovation Validation Framework.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ...engines.validation_framework import ValidationFramework
from ...engines.impact_assessment_framework import ImpactAssessmentFramework
from ...engines.success_prediction_system import SuccessPredictionSystem
from ...models.validation_models import (
    Innovation, ValidationRequest, ValidationReport, ValidationMethodology,
    ValidationType, ValidationStatus, ValidationResult, ImpactAssessment,
    SuccessPrediction, SuccessProbability
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/validation", tags=["validation"])

# Global framework instances
validation_framework = ValidationFramework()
impact_assessment_framework = ImpactAssessmentFramework()
success_prediction_system = SuccessPredictionSystem()


async def get_validation_framework() -> ValidationFramework:
    """Dependency to get validation framework instance."""
    if validation_framework.status.value != "ready":
        await validation_framework.start()
    return validation_framework


async def get_impact_assessment_framework() -> ImpactAssessmentFramework:
    """Dependency to get impact assessment framework instance."""
    if impact_assessment_framework.status.value != "ready":
        await impact_assessment_framework.start()
    return impact_assessment_framework


async def get_success_prediction_system() -> SuccessPredictionSystem:
    """Dependency to get success prediction system instance."""
    if success_prediction_system.status.value != "ready":
        await success_prediction_system.start()
    return success_prediction_system


@router.post("/validate", response_model=Dict[str, Any])
async def validate_innovation(
    innovation: Innovation,
    validation_types: Optional[List[ValidationType]] = None,
    framework: ValidationFramework = Depends(get_validation_framework)
):
    """
    Validate an innovation using the validation framework.
    
    Args:
        innovation: Innovation to validate
        validation_types: Types of validation to perform (optional)
        
    Returns:
        Validation report with comprehensive analysis
    """
    try:
        logger.info(f"Starting validation for innovation: {innovation.id}")
        
        # Perform validation
        report = await framework.validate_innovation(innovation, validation_types)
        
        logger.info(f"Validation completed for innovation: {innovation.id}")
        
        return {
            "status": "success",
            "innovation_id": innovation.id,
            "validation_report": {
                "id": report.id,
                "overall_score": report.overall_score,
                "overall_result": report.overall_result.value,
                "confidence_level": report.confidence_level,
                "strengths": report.strengths,
                "weaknesses": report.weaknesses,
                "opportunities": report.opportunities,
                "threats": report.threats,
                "recommendations": report.recommendations,
                "next_steps": report.next_steps,
                "validation_scores": [
                    {
                        "criteria_id": score.criteria_id,
                        "score": score.score,
                        "confidence": score.confidence,
                        "reasoning": score.reasoning
                    }
                    for score in report.validation_scores
                ],
                "created_at": report.created_at.isoformat(),
                "completed_at": report.completed_at.isoformat() if report.completed_at else None
            }
        }
        
    except Exception as e:
        logger.error(f"Validation failed for innovation {innovation.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/request", response_model=Dict[str, Any])
async def create_validation_request(
    request: ValidationRequest,
    background_tasks: BackgroundTasks,
    framework: ValidationFramework = Depends(get_validation_framework)
):
    """
    Create a validation request for processing.
    
    Args:
        request: Validation request details
        background_tasks: Background task processor
        
    Returns:
        Validation request confirmation
    """
    try:
        logger.info(f"Creating validation request: {request.id}")
        
        # Add validation processing to background tasks
        background_tasks.add_task(
            _process_validation_request_background,
            request,
            framework
        )
        
        return {
            "status": "success",
            "message": "Validation request created successfully",
            "request_id": request.id,
            "estimated_completion": "2-4 hours"
        }
        
    except Exception as e:
        logger.error(f"Failed to create validation request: {e}")
        raise HTTPException(status_code=500, detail=f"Request creation failed: {str(e)}")


@router.get("/methodologies", response_model=Dict[str, Any])
async def get_validation_methodologies(
    framework: ValidationFramework = Depends(get_validation_framework)
):
    """
    Get available validation methodologies.
    
    Returns:
        List of available validation methodologies
    """
    try:
        methodologies = []
        for methodology_id, methodology in framework.validation_methodologies.items():
            methodologies.append({
                "id": methodology_id,
                "name": methodology.name,
                "description": methodology.description,
                "validation_types": [vt.value for vt in methodology.validation_types],
                "estimated_duration": methodology.estimated_duration,
                "accuracy_rate": methodology.accuracy_rate,
                "confidence_level": methodology.confidence_level,
                "applicable_domains": methodology.applicable_domains
            })
        
        return {
            "status": "success",
            "methodologies": methodologies
        }
        
    except Exception as e:
        logger.error(f"Failed to get methodologies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get methodologies: {str(e)}")


@router.get("/criteria/{validation_type}", response_model=Dict[str, Any])
async def get_validation_criteria(
    validation_type: ValidationType,
    framework: ValidationFramework = Depends(get_validation_framework)
):
    """
    Get validation criteria for a specific validation type.
    
    Args:
        validation_type: Type of validation
        
    Returns:
        List of validation criteria
    """
    try:
        criteria_list = framework.validation_criteria.get(validation_type, [])
        
        criteria = []
        for criterion in criteria_list:
            criteria.append({
                "id": criterion.id,
                "name": criterion.name,
                "description": criterion.description,
                "weight": criterion.weight,
                "threshold": criterion.threshold,
                "required": criterion.required
            })
        
        return {
            "status": "success",
            "validation_type": validation_type.value,
            "criteria": criteria
        }
        
    except Exception as e:
        logger.error(f"Failed to get criteria for {validation_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get criteria: {str(e)}")


@router.get("/report/{report_id}", response_model=Dict[str, Any])
async def get_validation_report(
    report_id: str,
    framework: ValidationFramework = Depends(get_validation_framework)
):
    """
    Get a specific validation report.
    
    Args:
        report_id: ID of the validation report
        
    Returns:
        Validation report details
    """
    try:
        # Find report in historical validations
        report = None
        for historical_report in framework.historical_validations:
            if historical_report.id == report_id:
                report = historical_report
                break
        
        if not report:
            raise HTTPException(status_code=404, detail="Validation report not found")
        
        return {
            "status": "success",
            "report": {
                "id": report.id,
                "innovation_id": report.innovation_id,
                "request_id": report.request_id,
                "overall_score": report.overall_score,
                "overall_result": report.overall_result.value,
                "confidence_level": report.confidence_level,
                "strengths": report.strengths,
                "weaknesses": report.weaknesses,
                "opportunities": report.opportunities,
                "threats": report.threats,
                "recommendations": report.recommendations,
                "next_steps": report.next_steps,
                "risk_mitigation": report.risk_mitigation,
                "success_factors": report.success_factors,
                "validation_methodology": report.validation_methodology,
                "data_sources": report.data_sources,
                "assumptions": report.assumptions,
                "limitations": report.limitations,
                "validation_scores": [
                    {
                        "criteria_id": score.criteria_id,
                        "score": score.score,
                        "confidence": score.confidence,
                        "reasoning": score.reasoning,
                        "evidence": score.evidence,
                        "recommendations": score.recommendations
                    }
                    for score in report.validation_scores
                ],
                "created_at": report.created_at.isoformat(),
                "completed_at": report.completed_at.isoformat() if report.completed_at else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get validation report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


@router.get("/reports", response_model=Dict[str, Any])
async def get_validation_reports(
    innovation_id: Optional[str] = None,
    status: Optional[ValidationResult] = None,
    limit: int = 50,
    framework: ValidationFramework = Depends(get_validation_framework)
):
    """
    Get validation reports with optional filtering.
    
    Args:
        innovation_id: Filter by innovation ID (optional)
        status: Filter by validation result (optional)
        limit: Maximum number of reports to return
        
    Returns:
        List of validation reports
    """
    try:
        reports = framework.historical_validations
        
        # Apply filters
        if innovation_id:
            reports = [r for r in reports if r.innovation_id == innovation_id]
        
        if status:
            reports = [r for r in reports if r.overall_result == status]
        
        # Limit results
        reports = reports[:limit]
        
        report_summaries = []
        for report in reports:
            report_summaries.append({
                "id": report.id,
                "innovation_id": report.innovation_id,
                "overall_score": report.overall_score,
                "overall_result": report.overall_result.value,
                "confidence_level": report.confidence_level,
                "created_at": report.created_at.isoformat(),
                "completed_at": report.completed_at.isoformat() if report.completed_at else None
            })
        
        return {
            "status": "success",
            "reports": report_summaries,
            "total_count": len(framework.historical_validations),
            "filtered_count": len(reports)
        }
        
    except Exception as e:
        logger.error(f"Failed to get validation reports: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get reports: {str(e)}")


@router.get("/analytics", response_model=Dict[str, Any])
async def get_validation_analytics(
    framework: ValidationFramework = Depends(get_validation_framework)
):
    """
    Get validation analytics and insights.
    
    Returns:
        Validation analytics data
    """
    try:
        reports = framework.historical_validations
        
        if not reports:
            return {
                "status": "success",
                "analytics": {
                    "total_validations": 0,
                    "average_score": 0.0,
                    "success_rate": 0.0,
                    "result_distribution": {},
                    "trend_analysis": "Insufficient data"
                }
            }
        
        # Calculate analytics
        total_validations = len(reports)
        average_score = sum(r.overall_score for r in reports) / total_validations
        
        # Result distribution
        result_counts = {}
        for report in reports:
            result = report.overall_result.value
            result_counts[result] = result_counts.get(result, 0) + 1
        
        # Success rate (approved + conditional approval)
        successful_results = [
            ValidationResult.APPROVED.value,
            ValidationResult.CONDITIONAL_APPROVAL.value
        ]
        successful_count = sum(
            result_counts.get(result, 0) for result in successful_results
        )
        success_rate = successful_count / total_validations
        
        return {
            "status": "success",
            "analytics": {
                "total_validations": total_validations,
                "average_score": round(average_score, 3),
                "success_rate": round(success_rate, 3),
                "result_distribution": result_counts,
                "trend_analysis": "Validation performance tracking available"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get validation analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def get_validation_health(
    framework: ValidationFramework = Depends(get_validation_framework)
):
    """
    Get validation framework health status.
    
    Returns:
        Health status of validation framework
    """
    try:
        status = framework.get_status()
        metrics = framework.get_metrics()
        
        return {
            "status": "success",
            "health": {
                "framework_status": status,
                "performance_metrics": metrics,
                "last_health_check": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get validation health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/impact-assessment", response_model=Dict[str, Any])
async def assess_innovation_impact(
    innovation: Innovation,
    framework: ImpactAssessmentFramework = Depends(get_impact_assessment_framework)
):
    """
    Assess comprehensive impact of an innovation.
    
    Args:
        innovation: Innovation to assess
        
    Returns:
        Comprehensive impact assessment
    """
    try:
        logger.info(f"Starting impact assessment for innovation: {innovation.id}")
        
        # Perform impact assessment
        assessment = await framework.assess_innovation_impact(innovation)
        
        logger.info(f"Impact assessment completed for innovation: {innovation.id}")
        
        return {
            "status": "success",
            "innovation_id": innovation.id,
            "impact_assessment": {
                "id": assessment.id,
                "market_impact": assessment.market_impact.value,
                "technical_impact": assessment.technical_impact.value,
                "business_impact": assessment.business_impact.value,
                "social_impact": assessment.social_impact.value,
                "environmental_impact": assessment.environmental_impact.value,
                "economic_impact": assessment.economic_impact.value,
                "market_size": assessment.market_size,
                "addressable_market": assessment.addressable_market,
                "revenue_potential": assessment.revenue_potential,
                "cost_savings_potential": assessment.cost_savings_potential,
                "job_creation_potential": assessment.job_creation_potential,
                "disruption_potential": assessment.disruption_potential,
                "scalability_factor": assessment.scalability_factor,
                "time_to_market": assessment.time_to_market,
                "competitive_advantage_duration": assessment.competitive_advantage_duration,
                "quantitative_metrics": assessment.quantitative_metrics,
                "qualitative_factors": assessment.qualitative_factors,
                "stakeholder_impact": assessment.stakeholder_impact,
                "created_at": assessment.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Impact assessment failed for innovation {innovation.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Impact assessment failed: {str(e)}")


@router.get("/impact-assessment/{assessment_id}", response_model=Dict[str, Any])
async def get_impact_assessment(
    assessment_id: str,
    framework: ImpactAssessmentFramework = Depends(get_impact_assessment_framework)
):
    """
    Get a specific impact assessment.
    
    Args:
        assessment_id: ID of the impact assessment
        
    Returns:
        Impact assessment details
    """
    try:
        # Find assessment in historical assessments
        assessment = None
        for historical_assessment in framework.historical_assessments:
            if historical_assessment.id == assessment_id:
                assessment = historical_assessment
                break
        
        if not assessment:
            raise HTTPException(status_code=404, detail="Impact assessment not found")
        
        return {
            "status": "success",
            "assessment": {
                "id": assessment.id,
                "innovation_id": assessment.innovation_id,
                "market_impact": assessment.market_impact.value,
                "technical_impact": assessment.technical_impact.value,
                "business_impact": assessment.business_impact.value,
                "social_impact": assessment.social_impact.value,
                "environmental_impact": assessment.environmental_impact.value,
                "economic_impact": assessment.economic_impact.value,
                "market_size": assessment.market_size,
                "addressable_market": assessment.addressable_market,
                "market_penetration_potential": assessment.market_penetration_potential,
                "revenue_potential": assessment.revenue_potential,
                "cost_savings_potential": assessment.cost_savings_potential,
                "job_creation_potential": assessment.job_creation_potential,
                "disruption_potential": assessment.disruption_potential,
                "scalability_factor": assessment.scalability_factor,
                "time_to_market": assessment.time_to_market,
                "competitive_advantage_duration": assessment.competitive_advantage_duration,
                "impact_timeline": assessment.impact_timeline,
                "quantitative_metrics": assessment.quantitative_metrics,
                "qualitative_factors": assessment.qualitative_factors,
                "stakeholder_impact": assessment.stakeholder_impact,
                "created_at": assessment.created_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get impact assessment {assessment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get assessment: {str(e)}")


@router.get("/impact-assessments", response_model=Dict[str, Any])
async def get_impact_assessments(
    innovation_id: Optional[str] = None,
    impact_level: Optional[str] = None,
    limit: int = 50,
    framework: ImpactAssessmentFramework = Depends(get_impact_assessment_framework)
):
    """
    Get impact assessments with optional filtering.
    
    Args:
        innovation_id: Filter by innovation ID (optional)
        impact_level: Filter by impact level (optional)
        limit: Maximum number of assessments to return
        
    Returns:
        List of impact assessments
    """
    try:
        assessments = framework.historical_assessments
        
        # Apply filters
        if innovation_id:
            assessments = [a for a in assessments if a.innovation_id == innovation_id]
        
        if impact_level:
            assessments = [a for a in assessments if a.market_impact.value == impact_level]
        
        # Limit results
        assessments = assessments[:limit]
        
        assessment_summaries = []
        for assessment in assessments:
            assessment_summaries.append({
                "id": assessment.id,
                "innovation_id": assessment.innovation_id,
                "market_impact": assessment.market_impact.value,
                "technical_impact": assessment.technical_impact.value,
                "business_impact": assessment.business_impact.value,
                "revenue_potential": assessment.revenue_potential,
                "disruption_potential": assessment.disruption_potential,
                "created_at": assessment.created_at.isoformat()
            })
        
        return {
            "status": "success",
            "assessments": assessment_summaries,
            "total_count": len(framework.historical_assessments),
            "filtered_count": len(assessments)
        }
        
    except Exception as e:
        logger.error(f"Failed to get impact assessments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get assessments: {str(e)}")


@router.get("/impact-analytics", response_model=Dict[str, Any])
async def get_impact_analytics(
    framework: ImpactAssessmentFramework = Depends(get_impact_assessment_framework)
):
    """
    Get impact assessment analytics and insights.
    
    Returns:
        Impact assessment analytics data
    """
    try:
        assessments = framework.historical_assessments
        
        if not assessments:
            return {
                "status": "success",
                "analytics": {
                    "total_assessments": 0,
                    "average_revenue_potential": 0.0,
                    "average_disruption_potential": 0.0,
                    "impact_distribution": {},
                    "trend_analysis": "Insufficient data"
                }
            }
        
        # Calculate analytics
        total_assessments = len(assessments)
        avg_revenue = sum(a.revenue_potential for a in assessments) / total_assessments
        avg_disruption = sum(a.disruption_potential for a in assessments) / total_assessments
        
        # Impact level distribution
        impact_counts = {}
        for assessment in assessments:
            impact = assessment.market_impact.value
            impact_counts[impact] = impact_counts.get(impact, 0) + 1
        
        return {
            "status": "success",
            "analytics": {
                "total_assessments": total_assessments,
                "average_revenue_potential": round(avg_revenue, 2),
                "average_disruption_potential": round(avg_disruption, 3),
                "impact_distribution": impact_counts,
                "trend_analysis": "Impact assessment tracking available"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get impact analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.post("/success-prediction", response_model=Dict[str, Any])
async def predict_innovation_success(
    innovation: Innovation,
    system: SuccessPredictionSystem = Depends(get_success_prediction_system)
):
    """
    Predict success probability for an innovation.
    
    Args:
        innovation: Innovation to predict success for
        
    Returns:
        Comprehensive success prediction
    """
    try:
        logger.info(f"Starting success prediction for innovation: {innovation.id}")
        
        # Perform success prediction
        prediction = await system.predict_innovation_success(innovation)
        
        logger.info(f"Success prediction completed for innovation: {innovation.id}")
        
        return {
            "status": "success",
            "innovation_id": innovation.id,
            "success_prediction": {
                "id": prediction.id,
                "overall_probability": prediction.overall_probability,
                "probability_category": prediction.probability_category.value,
                "technical_success_probability": prediction.technical_success_probability,
                "market_success_probability": prediction.market_success_probability,
                "financial_success_probability": prediction.financial_success_probability,
                "timeline_success_probability": prediction.timeline_success_probability,
                "key_success_factors": prediction.key_success_factors,
                "critical_risks": prediction.critical_risks,
                "success_scenarios": prediction.success_scenarios,
                "failure_scenarios": prediction.failure_scenarios,
                "mitigation_strategies": prediction.mitigation_strategies,
                "optimization_opportunities": prediction.optimization_opportunities,
                "confidence_intervals": prediction.confidence_intervals,
                "model_accuracy": prediction.model_accuracy,
                "data_quality_score": prediction.data_quality_score,
                "prediction_methodology": prediction.prediction_methodology,
                "created_at": prediction.created_at.isoformat(),
                "expires_at": prediction.expires_at.isoformat() if prediction.expires_at else None
            }
        }
        
    except Exception as e:
        logger.error(f"Success prediction failed for innovation {innovation.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Success prediction failed: {str(e)}")


@router.get("/success-prediction/{prediction_id}", response_model=Dict[str, Any])
async def get_success_prediction(
    prediction_id: str,
    system: SuccessPredictionSystem = Depends(get_success_prediction_system)
):
    """
    Get a specific success prediction.
    
    Args:
        prediction_id: ID of the success prediction
        
    Returns:
        Success prediction details
    """
    try:
        # Find prediction in historical predictions
        prediction = None
        for historical_prediction in system.historical_predictions:
            if historical_prediction.id == prediction_id:
                prediction = historical_prediction
                break
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Success prediction not found")
        
        return {
            "status": "success",
            "prediction": {
                "id": prediction.id,
                "innovation_id": prediction.innovation_id,
                "overall_probability": prediction.overall_probability,
                "probability_category": prediction.probability_category.value,
                "technical_success_probability": prediction.technical_success_probability,
                "market_success_probability": prediction.market_success_probability,
                "financial_success_probability": prediction.financial_success_probability,
                "timeline_success_probability": prediction.timeline_success_probability,
                "key_success_factors": prediction.key_success_factors,
                "critical_risks": prediction.critical_risks,
                "success_scenarios": prediction.success_scenarios,
                "failure_scenarios": prediction.failure_scenarios,
                "mitigation_strategies": prediction.mitigation_strategies,
                "optimization_opportunities": prediction.optimization_opportunities,
                "confidence_intervals": prediction.confidence_intervals,
                "model_accuracy": prediction.model_accuracy,
                "data_quality_score": prediction.data_quality_score,
                "prediction_methodology": prediction.prediction_methodology,
                "created_at": prediction.created_at.isoformat(),
                "expires_at": prediction.expires_at.isoformat() if prediction.expires_at else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get success prediction {prediction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prediction: {str(e)}")


@router.get("/success-predictions", response_model=Dict[str, Any])
async def get_success_predictions(
    innovation_id: Optional[str] = None,
    probability_category: Optional[SuccessProbability] = None,
    limit: int = 50,
    system: SuccessPredictionSystem = Depends(get_success_prediction_system)
):
    """
    Get success predictions with optional filtering.
    
    Args:
        innovation_id: Filter by innovation ID (optional)
        probability_category: Filter by probability category (optional)
        limit: Maximum number of predictions to return
        
    Returns:
        List of success predictions
    """
    try:
        predictions = system.historical_predictions
        
        # Apply filters
        if innovation_id:
            predictions = [p for p in predictions if p.innovation_id == innovation_id]
        
        if probability_category:
            predictions = [p for p in predictions if p.probability_category == probability_category]
        
        # Limit results
        predictions = predictions[:limit]
        
        prediction_summaries = []
        for prediction in predictions:
            prediction_summaries.append({
                "id": prediction.id,
                "innovation_id": prediction.innovation_id,
                "overall_probability": prediction.overall_probability,
                "probability_category": prediction.probability_category.value,
                "model_accuracy": prediction.model_accuracy,
                "data_quality_score": prediction.data_quality_score,
                "created_at": prediction.created_at.isoformat(),
                "expires_at": prediction.expires_at.isoformat() if prediction.expires_at else None
            })
        
        return {
            "status": "success",
            "predictions": prediction_summaries,
            "total_count": len(system.historical_predictions),
            "filtered_count": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get success predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")


@router.get("/success-analytics", response_model=Dict[str, Any])
async def get_success_analytics(
    system: SuccessPredictionSystem = Depends(get_success_prediction_system)
):
    """
    Get success prediction analytics and insights.
    
    Returns:
        Success prediction analytics data
    """
    try:
        predictions = system.historical_predictions
        
        if not predictions:
            return {
                "status": "success",
                "analytics": {
                    "total_predictions": 0,
                    "average_success_probability": 0.0,
                    "probability_distribution": {},
                    "model_performance": {},
                    "trend_analysis": "Insufficient data"
                }
            }
        
        # Calculate analytics
        total_predictions = len(predictions)
        avg_probability = sum(p.overall_probability for p in predictions) / total_predictions
        avg_model_accuracy = sum(p.model_accuracy for p in predictions) / total_predictions
        avg_data_quality = sum(p.data_quality_score for p in predictions) / total_predictions
        
        # Probability category distribution
        category_counts = {}
        for prediction in predictions:
            category = prediction.probability_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "status": "success",
            "analytics": {
                "total_predictions": total_predictions,
                "average_success_probability": round(avg_probability, 3),
                "average_model_accuracy": round(avg_model_accuracy, 3),
                "average_data_quality": round(avg_data_quality, 3),
                "probability_distribution": category_counts,
                "model_performance": {
                    "accuracy": round(avg_model_accuracy, 3),
                    "data_quality": round(avg_data_quality, 3)
                },
                "trend_analysis": "Success prediction tracking available"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get success analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


async def _process_validation_request_background(
    request: ValidationRequest,
    framework: ValidationFramework
):
    """Process validation request in background."""
    try:
        logger.info(f"Processing validation request {request.id} in background")
        
        # This would typically involve more complex processing
        # For now, we'll simulate the process
        await framework.process(request)
        
        logger.info(f"Validation request {request.id} processed successfully")
        
    except Exception as e:
        logger.error(f"Background validation processing failed for {request.id}: {e}")