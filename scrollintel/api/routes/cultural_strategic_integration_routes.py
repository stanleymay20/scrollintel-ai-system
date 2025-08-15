"""
Cultural Strategic Integration API Routes

API endpoints for integrating cultural transformation with strategic planning.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ...engines.cultural_strategic_integration import CulturalStrategicIntegrationEngine
from ...models.cultural_strategic_integration_models import (
    StrategicObjective, CulturalStrategicAlignment, StrategicInitiative,
    CulturalImpactAssessment, CultureAwareDecision, IntegrationReport,
    StrategicObjectiveType, CulturalAlignment, ImpactLevel
)
from ...models.cultural_assessment_models import Culture
from ...core.auth import get_current_user

router = APIRouter(prefix="/api/v1/cultural-strategic-integration", tags=["Cultural Strategic Integration"])
logger = logging.getLogger(__name__)


def get_integration_engine():
    """Dependency to get cultural strategic integration engine"""
    return CulturalStrategicIntegrationEngine()


@router.post("/assess-alignment")
async def assess_cultural_alignment(
    objective_data: Dict[str, Any],
    culture_data: Dict[str, Any],
    engine: CulturalStrategicIntegrationEngine = Depends(get_integration_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Assess cultural alignment with strategic objective
    
    Args:
        objective_data: Strategic objective information
        culture_data: Current culture information
        
    Returns:
        Cultural alignment assessment
    """
    try:
        # Create objective from data
        objective = StrategicObjective(
            id=objective_data.get('id', 'temp_id'),
            name=objective_data['name'],
            description=objective_data['description'],
            objective_type=StrategicObjectiveType(objective_data['objective_type']),
            target_date=datetime.fromisoformat(objective_data['target_date']),
            success_metrics=objective_data.get('success_metrics', []),
            priority_level=objective_data.get('priority_level', 1),
            owner=objective_data.get('owner', 'unknown'),
            cultural_requirements=objective_data.get('cultural_requirements', []),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Create culture from data (simplified)
        culture = Culture(
            organization_id=culture_data.get('organization_id', 'temp_org'),
            cultural_dimensions=culture_data.get('cultural_dimensions', {}),
            values=culture_data.get('values', []),
            behaviors=culture_data.get('behaviors', []),
            norms=culture_data.get('norms', []),
            subcultures=culture_data.get('subcultures', []),
            health_score=culture_data.get('health_score', 0.5),
            assessment_date=datetime.now()
        )
        
        # Assess alignment
        alignment = engine.assess_cultural_alignment(objective, culture)
        
        return {
            "status": "success",
            "alignment": {
                "id": alignment.id,
                "objective_id": alignment.objective_id,
                "alignment_level": alignment.alignment_level.value,
                "alignment_score": alignment.alignment_score,
                "gap_analysis": alignment.gap_analysis,
                "recommendations": alignment.recommendations,
                "assessment_date": alignment.assessment_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error assessing cultural alignment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-impact")
async def assess_cultural_impact(
    initiative_data: Dict[str, Any],
    culture_data: Dict[str, Any],
    engine: CulturalStrategicIntegrationEngine = Depends(get_integration_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Assess cultural impact of strategic initiative
    
    Args:
        initiative_data: Strategic initiative information
        culture_data: Current culture information
        
    Returns:
        Cultural impact assessment
    """
    try:
        # Create initiative from data
        initiative = StrategicInitiative(
            id=initiative_data.get('id', 'temp_id'),
            name=initiative_data['name'],
            description=initiative_data['description'],
            objective_ids=initiative_data.get('objective_ids', []),
            start_date=datetime.fromisoformat(initiative_data['start_date']),
            end_date=datetime.fromisoformat(initiative_data['end_date']),
            budget=initiative_data.get('budget', 0.0),
            team_size=initiative_data.get('team_size', 1),
            cultural_impact_level=ImpactLevel(initiative_data.get('cultural_impact_level', 'medium')),
            cultural_requirements=initiative_data.get('cultural_requirements', []),
            success_criteria=initiative_data.get('success_criteria', []),
            status=initiative_data.get('status', 'planned')
        )
        
        # Create culture from data
        culture = Culture(
            organization_id=culture_data.get('organization_id', 'temp_org'),
            cultural_dimensions=culture_data.get('cultural_dimensions', {}),
            values=culture_data.get('values', []),
            behaviors=culture_data.get('behaviors', []),
            norms=culture_data.get('norms', []),
            subcultures=culture_data.get('subcultures', []),
            health_score=culture_data.get('health_score', 0.5),
            assessment_date=datetime.now()
        )
        
        # Assess impact
        assessment = engine.assess_cultural_impact(initiative, culture)
        
        return {
            "status": "success",
            "assessment": {
                "id": assessment.id,
                "initiative_id": assessment.initiative_id,
                "impact_level": assessment.impact_level.value,
                "impact_score": assessment.impact_score,
                "cultural_enablers": assessment.cultural_enablers,
                "cultural_barriers": assessment.cultural_barriers,
                "mitigation_strategies": assessment.mitigation_strategies,
                "success_probability": assessment.success_probability,
                "assessment_date": assessment.assessment_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error assessing cultural impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/culture-aware-decision")
async def make_culture_aware_decision(
    decision_data: Dict[str, Any],
    engine: CulturalStrategicIntegrationEngine = Depends(get_integration_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Make culture-aware strategic decision
    
    Args:
        decision_data: Decision context and options
        
    Returns:
        Culture-aware decision recommendation
    """
    try:
        decision_context = decision_data['decision_context']
        strategic_options = decision_data['strategic_options']
        culture_data = decision_data['culture_data']
        
        # Create culture from data
        culture = Culture(
            organization_id=culture_data.get('organization_id', 'temp_org'),
            cultural_dimensions=culture_data.get('cultural_dimensions', {}),
            values=culture_data.get('values', []),
            behaviors=culture_data.get('behaviors', []),
            norms=culture_data.get('norms', []),
            subcultures=culture_data.get('subcultures', []),
            health_score=culture_data.get('health_score', 0.5),
            assessment_date=datetime.now()
        )
        
        # Make decision
        decision = engine.make_culture_aware_decision(
            decision_context, strategic_options, culture
        )
        
        return {
            "status": "success",
            "decision": {
                "id": decision.id,
                "decision_context": decision.decision_context,
                "recommended_option": decision.recommended_option,
                "cultural_rationale": decision.cultural_rationale,
                "risk_assessment": decision.risk_assessment,
                "implementation_plan": decision.implementation_plan,
                "decision_date": decision.decision_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error making culture-aware decision: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integration-report")
async def generate_integration_report(
    report_data: Dict[str, Any],
    engine: CulturalStrategicIntegrationEngine = Depends(get_integration_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate cultural-strategic integration report
    
    Args:
        report_data: Objectives, initiatives, and culture data
        
    Returns:
        Comprehensive integration report
    """
    try:
        objectives_data = report_data['objectives']
        initiatives_data = report_data['initiatives']
        culture_data = report_data['culture_data']
        
        # Create objectives
        objectives = []
        for obj_data in objectives_data:
            objective = StrategicObjective(
                id=obj_data.get('id', f'obj_{len(objectives)}'),
                name=obj_data['name'],
                description=obj_data['description'],
                objective_type=StrategicObjectiveType(obj_data['objective_type']),
                target_date=datetime.fromisoformat(obj_data['target_date']),
                success_metrics=obj_data.get('success_metrics', []),
                priority_level=obj_data.get('priority_level', 1),
                owner=obj_data.get('owner', 'unknown'),
                cultural_requirements=obj_data.get('cultural_requirements', []),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            objectives.append(objective)
        
        # Create initiatives
        initiatives = []
        for init_data in initiatives_data:
            initiative = StrategicInitiative(
                id=init_data.get('id', f'init_{len(initiatives)}'),
                name=init_data['name'],
                description=init_data['description'],
                objective_ids=init_data.get('objective_ids', []),
                start_date=datetime.fromisoformat(init_data['start_date']),
                end_date=datetime.fromisoformat(init_data['end_date']),
                budget=init_data.get('budget', 0.0),
                team_size=init_data.get('team_size', 1),
                cultural_impact_level=ImpactLevel(init_data.get('cultural_impact_level', 'medium')),
                cultural_requirements=init_data.get('cultural_requirements', []),
                success_criteria=init_data.get('success_criteria', []),
                status=init_data.get('status', 'planned')
            )
            initiatives.append(initiative)
        
        # Create culture
        culture = Culture(
            organization_id=culture_data.get('organization_id', 'temp_org'),
            cultural_dimensions=culture_data.get('cultural_dimensions', {}),
            values=culture_data.get('values', []),
            behaviors=culture_data.get('behaviors', []),
            norms=culture_data.get('norms', []),
            subcultures=culture_data.get('subcultures', []),
            health_score=culture_data.get('health_score', 0.5),
            assessment_date=datetime.now()
        )
        
        # Generate report
        report = engine.generate_integration_report(objectives, initiatives, culture)
        
        return {
            "status": "success",
            "report": {
                "id": report.id,
                "report_type": report.report_type,
                "reporting_period": report.reporting_period,
                "alignment_summary": report.alignment_summary,
                "recommendations": report.recommendations,
                "success_metrics": report.success_metrics,
                "generated_at": report.generated_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating integration report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alignment-status/{objective_id}")
async def get_alignment_status(
    objective_id: str,
    engine: CulturalStrategicIntegrationEngine = Depends(get_integration_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get cultural alignment status for specific objective
    
    Args:
        objective_id: Strategic objective ID
        
    Returns:
        Current alignment status
    """
    try:
        alignment = engine.alignment_cache.get(objective_id)
        
        if not alignment:
            return {
                "status": "not_found",
                "message": f"No alignment assessment found for objective {objective_id}"
            }
        
        return {
            "status": "success",
            "alignment": {
                "objective_id": alignment.objective_id,
                "alignment_level": alignment.alignment_level.value,
                "alignment_score": alignment.alignment_score,
                "assessment_date": alignment.assessment_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting alignment status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/impact-status/{initiative_id}")
async def get_impact_status(
    initiative_id: str,
    engine: CulturalStrategicIntegrationEngine = Depends(get_integration_engine),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get cultural impact status for specific initiative
    
    Args:
        initiative_id: Strategic initiative ID
        
    Returns:
        Current impact assessment
    """
    try:
        assessment = engine.impact_assessments.get(initiative_id)
        
        if not assessment:
            return {
                "status": "not_found",
                "message": f"No impact assessment found for initiative {initiative_id}"
            }
        
        return {
            "status": "success",
            "assessment": {
                "initiative_id": assessment.initiative_id,
                "impact_level": assessment.impact_level.value,
                "impact_score": assessment.impact_score,
                "success_probability": assessment.success_probability,
                "assessment_date": assessment.assessment_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting impact status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "cultural_strategic_integration",
        "timestamp": datetime.now().isoformat()
    }