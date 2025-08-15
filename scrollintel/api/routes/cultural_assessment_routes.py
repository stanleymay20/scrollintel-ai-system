"""
API Routes for Cultural Assessment Engine
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from datetime import datetime

from ...engines.cultural_assessment_engine import CulturalAssessmentEngine
from ...models.cultural_assessment_models import (
    CulturalAssessmentRequest, CulturalAssessmentResult,
    CultureMap, DimensionAnalysis, CulturalDimension,
    Subculture, CulturalHealthMetric
)

router = APIRouter(prefix="/cultural-assessment", tags=["Cultural Assessment"])
logger = logging.getLogger(__name__)

# Initialize the cultural assessment engine
cultural_assessment_engine = CulturalAssessmentEngine()


@router.post("/assess", response_model=Dict[str, Any])
async def conduct_cultural_assessment(request: CulturalAssessmentRequest):
    """
    Conduct comprehensive cultural assessment for an organization
    
    Requirements addressed: 1.1, 1.2
    """
    try:
        logger.info(f"Received cultural assessment request for organization {request.organization_id}")
        
        # Conduct the assessment
        result = cultural_assessment_engine.conduct_cultural_assessment(request)
        
        # Convert result to dictionary for JSON response
        response = {
            "request_id": result.request_id,
            "organization_id": result.organization_id,
            "culture_map": {
                "organization_id": result.culture_map.organization_id,
                "assessment_date": result.culture_map.assessment_date.isoformat(),
                "cultural_dimensions": {dim.value: score for dim, score in result.culture_map.cultural_dimensions.items()},
                "overall_health_score": result.culture_map.overall_health_score,
                "assessment_confidence": result.culture_map.assessment_confidence,
                "subcultures_count": len(result.culture_map.subcultures),
                "health_metrics_count": len(result.culture_map.health_metrics),
                "values_count": len(result.culture_map.values)
            },
            "key_findings": result.key_findings,
            "recommendations": result.recommendations,
            "assessment_summary": result.assessment_summary,
            "confidence_score": result.confidence_score,
            "completion_date": result.completion_date.isoformat()
        }
        
        logger.info(f"Cultural assessment completed successfully for organization {request.organization_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in cultural assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cultural assessment failed: {str(e)}")


@router.get("/culture-map/{organization_id}", response_model=Dict[str, Any])
async def get_culture_map(organization_id: str):
    """
    Get comprehensive culture map for an organization
    
    Requirements addressed: 1.1
    """
    try:
        logger.info(f"Retrieving culture map for organization {organization_id}")
        
        # Create a sample request for demonstration
        request = CulturalAssessmentRequest(
            organization_id=organization_id,
            assessment_type="comprehensive",
            focus_areas=list(CulturalDimension),
            data_sources=["survey", "observation", "interview"],
            timeline="2 weeks",
            stakeholders=["HR", "Leadership", "Employees"]
        )
        
        # Conduct assessment to get culture map
        result = cultural_assessment_engine.conduct_cultural_assessment(request)
        culture_map = result.culture_map
        
        response = {
            "organization_id": culture_map.organization_id,
            "assessment_date": culture_map.assessment_date.isoformat(),
            "cultural_dimensions": {dim.value: score for dim, score in culture_map.cultural_dimensions.items()},
            "values": [
                {
                    "name": value.name,
                    "description": value.description,
                    "importance_score": value.importance_score,
                    "alignment_score": value.alignment_score,
                    "evidence_count": len(value.evidence)
                }
                for value in culture_map.values
            ],
            "behaviors": [
                {
                    "behavior_id": behavior.behavior_id,
                    "description": behavior.description,
                    "frequency": behavior.frequency,
                    "impact_score": behavior.impact_score,
                    "context": behavior.context
                }
                for behavior in culture_map.behaviors
            ],
            "subcultures": [
                {
                    "subculture_id": subculture.subculture_id,
                    "name": subculture.name,
                    "type": subculture.type.value,
                    "members_count": len(subculture.members),
                    "strength": subculture.strength,
                    "influence": subculture.influence
                }
                for subculture in culture_map.subcultures
            ],
            "overall_health_score": culture_map.overall_health_score,
            "assessment_confidence": culture_map.assessment_confidence
        }
        
        logger.info(f"Culture map retrieved successfully for organization {organization_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving culture map: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve culture map: {str(e)}")


@router.get("/dimensions/{organization_id}", response_model=Dict[str, Any])
async def analyze_cultural_dimensions(organization_id: str):
    """
    Analyze cultural dimensions for an organization
    
    Requirements addressed: 1.1
    """
    try:
        logger.info(f"Analyzing cultural dimensions for organization {organization_id}")
        
        # Create assessment request
        request = CulturalAssessmentRequest(
            organization_id=organization_id,
            assessment_type="focused",
            focus_areas=list(CulturalDimension),
            data_sources=["survey"],
            timeline="1 week",
            stakeholders=["HR"]
        )
        
        # Conduct assessment
        result = cultural_assessment_engine.conduct_cultural_assessment(request)
        
        response = {
            "organization_id": organization_id,
            "dimensions": {
                dim.value: {
                    "current_score": score,
                    "dimension_name": dim.value.replace('_', ' ').title(),
                    "status": "strong" if score > 0.7 else "moderate" if score > 0.5 else "needs_improvement"
                }
                for dim, score in result.culture_map.cultural_dimensions.items()
            },
            "dimension_analyses": [
                {
                    "dimension": analysis.dimension.value,
                    "current_score": analysis.current_score,
                    "ideal_score": analysis.ideal_score,
                    "gap_analysis": analysis.gap_analysis,
                    "contributing_factors": analysis.contributing_factors,
                    "improvement_recommendations": analysis.improvement_recommendations,
                    "measurement_confidence": analysis.measurement_confidence
                }
                for analysis in result.dimension_analyses
            ]
        }
        
        logger.info(f"Cultural dimensions analyzed successfully for organization {organization_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing cultural dimensions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze cultural dimensions: {str(e)}")


@router.get("/subcultures/{organization_id}", response_model=Dict[str, Any])
async def identify_subcultures(organization_id: str):
    """
    Identify subcultures within an organization
    
    Requirements addressed: 1.2
    """
    try:
        logger.info(f"Identifying subcultures for organization {organization_id}")
        
        # Create assessment request
        request = CulturalAssessmentRequest(
            organization_id=organization_id,
            assessment_type="comprehensive",
            focus_areas=[CulturalDimension.COLLABORATION_STYLE, CulturalDimension.COMMUNICATION_DIRECTNESS],
            data_sources=["survey", "observation"],
            timeline="2 weeks",
            stakeholders=["HR", "Management"]
        )
        
        # Conduct assessment
        result = cultural_assessment_engine.conduct_cultural_assessment(request)
        subcultures = result.culture_map.subcultures
        
        response = {
            "organization_id": organization_id,
            "subcultures_identified": len(subcultures),
            "subcultures": [
                {
                    "subculture_id": subculture.subculture_id,
                    "name": subculture.name,
                    "type": subculture.type.value,
                    "members_count": len(subculture.members),
                    "strength": subculture.strength,
                    "influence": subculture.influence,
                    "characteristics": {
                        key: f"{len(values)} data points" if isinstance(values, list) else str(values)
                        for key, values in subculture.characteristics.items()
                    }
                }
                for subculture in subcultures
            ],
            "analysis": {
                "strongest_subculture": max(subcultures, key=lambda s: s.strength).name if subcultures else None,
                "most_influential": max(subcultures, key=lambda s: s.influence).name if subcultures else None,
                "diversity_score": len(subcultures) / 10.0  # Normalized diversity score
            }
        }
        
        logger.info(f"Subcultures identified successfully for organization {organization_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error identifying subcultures: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to identify subcultures: {str(e)}")


@router.get("/health-metrics/{organization_id}", response_model=Dict[str, Any])
async def get_cultural_health_metrics(organization_id: str):
    """
    Get cultural health metrics for an organization
    
    Requirements addressed: 1.2
    """
    try:
        logger.info(f"Calculating cultural health metrics for organization {organization_id}")
        
        # Create assessment request
        request = CulturalAssessmentRequest(
            organization_id=organization_id,
            assessment_type="comprehensive",
            focus_areas=list(CulturalDimension),
            data_sources=["survey", "observation", "interview"],
            timeline="2 weeks",
            stakeholders=["HR", "Leadership", "Employees"]
        )
        
        # Conduct assessment
        result = cultural_assessment_engine.conduct_cultural_assessment(request)
        health_metrics = result.culture_map.health_metrics
        
        response = {
            "organization_id": organization_id,
            "overall_health_score": result.culture_map.overall_health_score,
            "assessment_date": result.culture_map.assessment_date.isoformat(),
            "metrics": [
                {
                    "metric_id": metric.metric_id,
                    "name": metric.name,
                    "value": metric.value,
                    "target_value": metric.target_value,
                    "trend": metric.trend,
                    "confidence_level": metric.confidence_level,
                    "status": "on_target" if metric.target_value and metric.value >= metric.target_value else "below_target",
                    "gap": metric.target_value - metric.value if metric.target_value else 0
                }
                for metric in health_metrics
            ],
            "summary": {
                "metrics_count": len(health_metrics),
                "on_target_count": sum(1 for m in health_metrics if m.target_value and m.value >= m.target_value),
                "improving_count": sum(1 for m in health_metrics if m.trend == "improving"),
                "average_confidence": sum(m.confidence_level for m in health_metrics) / len(health_metrics) if health_metrics else 0
            }
        }
        
        logger.info(f"Cultural health metrics calculated successfully for organization {organization_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error calculating cultural health metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate cultural health metrics: {str(e)}")


@router.post("/quick-assessment", response_model=Dict[str, Any])
async def quick_cultural_assessment(organization_id: str, focus_areas: List[str] = None):
    """
    Conduct a quick cultural assessment focusing on specific areas
    
    Requirements addressed: 1.1, 1.2
    """
    try:
        logger.info(f"Conducting quick cultural assessment for organization {organization_id}")
        
        # Convert focus areas to CulturalDimension enums
        if focus_areas:
            focus_dimensions = []
            for area in focus_areas:
                try:
                    focus_dimensions.append(CulturalDimension(area))
                except ValueError:
                    logger.warning(f"Invalid focus area: {area}")
        else:
            focus_dimensions = [
                CulturalDimension.INNOVATION_ORIENTATION,
                CulturalDimension.COLLABORATION_STYLE,
                CulturalDimension.COMMUNICATION_DIRECTNESS
            ]
        
        # Create quick assessment request
        request = CulturalAssessmentRequest(
            organization_id=organization_id,
            assessment_type="quick",
            focus_areas=focus_dimensions,
            data_sources=["survey"],
            timeline="3 days",
            stakeholders=["HR"]
        )
        
        # Conduct assessment
        result = cultural_assessment_engine.conduct_cultural_assessment(request)
        
        response = {
            "organization_id": organization_id,
            "assessment_type": "quick",
            "overall_health_score": result.culture_map.overall_health_score,
            "confidence_score": result.confidence_score,
            "focus_areas": [dim.value for dim in focus_dimensions],
            "key_findings": result.key_findings[:3],  # Top 3 findings
            "priority_recommendations": result.recommendations[:3],  # Top 3 recommendations
            "cultural_dimensions": {
                dim.value: score 
                for dim, score in result.culture_map.cultural_dimensions.items()
                if dim in focus_dimensions
            },
            "completion_date": result.completion_date.isoformat()
        }
        
        logger.info(f"Quick cultural assessment completed for organization {organization_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in quick cultural assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quick cultural assessment failed: {str(e)}")


@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint for cultural assessment service"""
    return {
        "status": "healthy",
        "service": "Cultural Assessment Engine",
        "timestamp": datetime.now().isoformat()
    }