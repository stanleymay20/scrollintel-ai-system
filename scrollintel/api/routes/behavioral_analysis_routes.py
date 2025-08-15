"""
API Routes for Behavioral Analysis Engine

Provides REST endpoints for behavioral analysis functionality
including behavior pattern identification, norm assessment, and culture alignment.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from datetime import datetime
import logging

from ...engines.behavioral_analysis_engine import BehavioralAnalysisEngine
from ...models.behavioral_analysis_models import (
    BehaviorAnalysisResult, BehaviorObservation, BehaviorMetrics,
    BehaviorType, BehaviorFrequency, AlignmentLevel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/behavioral-analysis", tags=["behavioral-analysis"])

# Global engine instance
behavioral_engine = BehavioralAnalysisEngine()


@router.post("/analyze/{organization_id}")
async def analyze_organizational_behaviors(
    organization_id: str,
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze organizational behaviors and identify patterns
    
    Requirements: 3.1, 3.2 - Build current behavioral pattern identification and analysis
    """
    try:
        # Extract observations from request
        observations_data = request_data.get("observations", [])
        cultural_values = request_data.get("cultural_values", [])
        analyst = request_data.get("analyst", "ScrollIntel")
        
        # Convert observations data to BehaviorObservation objects
        observations = []
        for obs_data in observations_data:
            observation = BehaviorObservation(
                id=obs_data.get("id", ""),
                observer_id=obs_data.get("observer_id", ""),
                observed_behavior=obs_data.get("observed_behavior", ""),
                behavior_type=BehaviorType(obs_data.get("behavior_type", "communication")),
                context=obs_data.get("context", {}),
                participants=obs_data.get("participants", []),
                timestamp=datetime.fromisoformat(obs_data.get("timestamp", datetime.now().isoformat())),
                impact_assessment=obs_data.get("impact_assessment", ""),
                cultural_relevance=obs_data.get("cultural_relevance", 0.5)
            )
            observations.append(observation)
        
        # Perform behavioral analysis
        result = behavioral_engine.analyze_organizational_behaviors(
            organization_id=organization_id,
            observations=observations,
            cultural_values=cultural_values,
            analyst=analyst
        )
        
        return {
            "success": True,
            "analysis_id": result.analysis_id,
            "organization_id": result.organization_id,
            "behavior_patterns_count": len(result.behavior_patterns),
            "behavioral_norms_count": len(result.behavioral_norms),
            "culture_alignments_count": len(result.culture_alignments),
            "overall_health_score": result.overall_health_score,
            "key_insights": result.key_insights,
            "recommendations": result.recommendations,
            "analysis_date": result.analysis_date.isoformat(),
            "analyst": result.analyst
        }
        
    except Exception as e:
        logger.error(f"Error in behavioral analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/results/{analysis_id}")
async def get_analysis_result(analysis_id: str) -> Dict[str, Any]:
    """Get detailed behavioral analysis results"""
    try:
        result = behavioral_engine.get_analysis_result(analysis_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        return {
            "success": True,
            "analysis_id": result.analysis_id,
            "organization_id": result.organization_id,
            "behavior_patterns": [
                {
                    "id": pattern.id,
                    "name": pattern.name,
                    "description": pattern.description,
                    "behavior_type": pattern.behavior_type.value,
                    "frequency": pattern.frequency.value,
                    "triggers": pattern.triggers,
                    "outcomes": pattern.outcomes,
                    "participants": pattern.participants,
                    "strength": pattern.strength,
                    "identified_date": pattern.identified_date.isoformat()
                }
                for pattern in result.behavior_patterns
            ],
            "behavioral_norms": [
                {
                    "id": norm.id,
                    "name": norm.name,
                    "description": norm.description,
                    "behavior_type": norm.behavior_type.value,
                    "expected_behaviors": norm.expected_behaviors,
                    "discouraged_behaviors": norm.discouraged_behaviors,
                    "compliance_rate": norm.compliance_rate,
                    "cultural_importance": norm.cultural_importance
                }
                for norm in result.behavioral_norms
            ],
            "culture_alignments": [
                {
                    "id": alignment.id,
                    "behavior_pattern_id": alignment.behavior_pattern_id,
                    "cultural_value": alignment.cultural_value,
                    "alignment_level": alignment.alignment_level.value,
                    "alignment_score": alignment.alignment_score,
                    "supporting_evidence": alignment.supporting_evidence,
                    "conflicting_evidence": alignment.conflicting_evidence,
                    "recommendations": alignment.recommendations
                }
                for alignment in result.culture_alignments
            ],
            "overall_health_score": result.overall_health_score,
            "key_insights": result.key_insights,
            "recommendations": result.recommendations,
            "analysis_date": result.analysis_date.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve result: {str(e)}")


@router.get("/organization/{organization_id}/analyses")
async def get_organization_analyses(organization_id: str) -> Dict[str, Any]:
    """Get all behavioral analyses for an organization"""
    try:
        analyses = behavioral_engine.get_organization_analyses(organization_id)
        
        return {
            "success": True,
            "organization_id": organization_id,
            "analyses_count": len(analyses),
            "analyses": [
                {
                    "analysis_id": analysis.analysis_id,
                    "overall_health_score": analysis.overall_health_score,
                    "behavior_patterns_count": len(analysis.behavior_patterns),
                    "behavioral_norms_count": len(analysis.behavioral_norms),
                    "analysis_date": analysis.analysis_date.isoformat(),
                    "analyst": analysis.analyst
                }
                for analysis in analyses
            ]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving organization analyses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analyses: {str(e)}")


@router.post("/metrics/{analysis_id}")
async def calculate_behavioral_metrics(analysis_id: str) -> Dict[str, Any]:
    """Calculate comprehensive behavioral metrics for an analysis"""
    try:
        result = behavioral_engine.get_analysis_result(analysis_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        metrics = behavioral_engine.calculate_behavior_metrics(result)
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "metrics": {
                "behavior_diversity_index": metrics.behavior_diversity_index,
                "norm_compliance_average": metrics.norm_compliance_average,
                "culture_alignment_score": metrics.culture_alignment_score,
                "behavior_consistency_index": metrics.behavior_consistency_index,
                "positive_behavior_ratio": metrics.positive_behavior_ratio,
                "improvement_trend": metrics.improvement_trend,
                "calculated_date": metrics.calculated_date.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating behavioral metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate metrics: {str(e)}")


@router.get("/behavior-types")
async def get_behavior_types() -> Dict[str, Any]:
    """Get available behavior types for analysis"""
    return {
        "success": True,
        "behavior_types": [
            {
                "value": behavior_type.value,
                "name": behavior_type.value.replace("_", " ").title()
            }
            for behavior_type in BehaviorType
        ]
    }


@router.get("/alignment-levels")
async def get_alignment_levels() -> Dict[str, Any]:
    """Get available alignment levels"""
    return {
        "success": True,
        "alignment_levels": [
            {
                "value": level.value,
                "name": level.value.replace("_", " ").title()
            }
            for level in AlignmentLevel
        ]
    }


@router.post("/observations/validate")
async def validate_behavior_observation(observation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate behavior observation data"""
    try:
        # Attempt to create BehaviorObservation to validate
        observation = BehaviorObservation(
            id=observation_data.get("id", ""),
            observer_id=observation_data.get("observer_id", ""),
            observed_behavior=observation_data.get("observed_behavior", ""),
            behavior_type=BehaviorType(observation_data.get("behavior_type", "communication")),
            context=observation_data.get("context", {}),
            participants=observation_data.get("participants", []),
            timestamp=datetime.fromisoformat(observation_data.get("timestamp", datetime.now().isoformat())),
            impact_assessment=observation_data.get("impact_assessment", ""),
            cultural_relevance=observation_data.get("cultural_relevance", 0.5)
        )
        
        return {
            "success": True,
            "valid": True,
            "message": "Observation data is valid"
        }
        
    except Exception as e:
        return {
            "success": True,
            "valid": False,
            "message": f"Validation failed: {str(e)}"
        }