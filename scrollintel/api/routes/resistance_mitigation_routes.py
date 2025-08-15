"""
API Routes for Cultural Change Resistance Mitigation Framework
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ...engines.resistance_mitigation_engine import ResistanceMitigationEngine
from ...models.resistance_mitigation_models import (
    MitigationPlan, MitigationExecution, ResistanceResolution,
    MitigationValidation, MitigationTemplate, MitigationMetrics
)
from ...models.resistance_detection_models import ResistanceDetection, ResistanceType, ResistanceSeverity
from ...models.cultural_assessment_models import Organization
from ...models.transformation_roadmap_models import Transformation
from ...core.auth import get_current_user

router = APIRouter(prefix="/api/v1/resistance-mitigation", tags=["resistance-mitigation"])
logger = logging.getLogger(__name__)


@router.post("/create-plan", response_model=MitigationPlan)
async def create_mitigation_plan(
    detection_id: str,
    organization_id: str,
    transformation_id: str,
    constraints: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Create targeted resistance addressing strategies
    
    Args:
        detection_id: ID of resistance detection
        organization_id: ID of organization
        transformation_id: ID of transformation process
        constraints: Resource and timeline constraints
        
    Returns:
        Comprehensive mitigation plan
    """
    try:
        engine = ResistanceMitigationEngine()
        
        # Mock detection, organization, and transformation objects
        detection = ResistanceDetection(
            id=detection_id,
            organization_id=organization_id,
            transformation_id=transformation_id,
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            source=None,
            severity=ResistanceSeverity.MODERATE,
            confidence_score=0.8,
            detected_at=datetime.now(),
            indicators_triggered=["low_participation", "delayed_compliance"],
            affected_areas=["engineering_team", "product_team"],
            potential_impact={"timeline_delay": 0.15},
            detection_method="behavioral_analysis",
            raw_data={}
        )
        
        organization = Organization(
            id=organization_id,
            name="Sample Organization",
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_score=0.8,
            assessment_date=datetime.now()
        )
        
        transformation = Transformation(
            id=transformation_id,
            organization_id=organization_id,
            current_culture=None,
            target_culture=None,
            vision=None,
            roadmap=None,
            interventions=[],
            progress=0.5,
            start_date=datetime.now(),
            target_completion=datetime.now()
        )
        
        mitigation_plan = engine.create_mitigation_plan(
            detection=detection,
            organization=organization,
            transformation=transformation,
            constraints=constraints
        )
        
        logger.info(f"Created mitigation plan {mitigation_plan.id} for detection {detection_id}")
        return mitigation_plan
        
    except Exception as e:
        logger.error(f"Error creating mitigation plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-plan/{plan_id}", response_model=MitigationExecution)
async def execute_mitigation_plan(
    plan_id: str,
    organization_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Execute mitigation plan with intervention coordination
    
    Args:
        plan_id: ID of mitigation plan
        organization_id: ID of organization
        background_tasks: Background task manager
        
    Returns:
        Execution tracking object
    """
    try:
        engine = ResistanceMitigationEngine()
        
        # Mock plan and organization objects
        from ...models.resistance_mitigation_models import MitigationStrategy, MitigationIntervention, InterventionType, MitigationStatus
        
        plan = MitigationPlan(
            id=plan_id,
            detection_id="det_001",
            organization_id=organization_id,
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            severity=ResistanceSeverity.MODERATE,
            strategies=[MitigationStrategy.TRAINING_SUPPORT, MitigationStrategy.STAKEHOLDER_ENGAGEMENT],
            interventions=[
                MitigationIntervention(
                    id="int_001",
                    plan_id=plan_id,
                    intervention_type=InterventionType.TRAINING_SESSION,
                    strategy=MitigationStrategy.TRAINING_SUPPORT,
                    title="Skills Development Training",
                    description="Training to build capabilities",
                    target_audience=["team_alpha"],
                    facilitators=["trainer_1"],
                    duration_hours=6.0,
                    scheduled_date=datetime.now(),
                    completion_date=None,
                    status=MitigationStatus.PLANNED,
                    success_metrics={},
                    actual_results={},
                    participant_feedback=[],
                    effectiveness_score=None,
                    lessons_learned=[],
                    follow_up_actions=[]
                )
            ],
            target_stakeholders=["team_alpha", "team_beta"],
            success_criteria={},
            timeline={},
            resource_requirements={},
            risk_factors=[],
            contingency_plans=[],
            created_at=datetime.now(),
            created_by="system"
        )
        
        organization = Organization(
            id=organization_id,
            name="Sample Organization",
            cultural_dimensions={},
            values=[],
            behaviors=[],
            norms=[],
            subcultures=[],
            health_score=0.8,
            assessment_date=datetime.now()
        )
        
        # Execute plan in background
        background_tasks.add_task(
            _execute_plan_background,
            engine,
            plan,
            organization
        )
        
        # Return initial execution object
        execution = MitigationExecution(
            id=f"exec_{plan_id}",
            plan_id=plan_id,
            execution_phase="initiation",
            start_date=datetime.now(),
            end_date=None,
            status=MitigationStatus.IN_PROGRESS,
            progress_percentage=0.0,
            completed_interventions=[],
            active_interventions=[],
            pending_interventions=[i.id for i in plan.interventions],
            resource_utilization={},
            stakeholder_engagement={},
            interim_results={},
            challenges_encountered=[],
            adjustments_made=[],
            next_steps=[]
        )
        
        logger.info(f"Started execution of mitigation plan {plan_id}")
        return execution
        
    except Exception as e:
        logger.error(f"Error executing mitigation plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-resolution", response_model=ResistanceResolution)
async def track_resistance_resolution(
    detection_id: str,
    plan_id: str,
    execution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Track resolution of resistance instance
    
    Args:
        detection_id: ID of resistance detection
        plan_id: ID of mitigation plan
        execution_id: ID of execution
        
    Returns:
        Resolution tracking with effectiveness assessment
    """
    try:
        engine = ResistanceMitigationEngine()
        
        # Mock objects for demonstration
        detection = ResistanceDetection(
            id=detection_id,
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            source=None,
            severity=ResistanceSeverity.MODERATE,
            confidence_score=0.8,
            detected_at=datetime.now(),
            indicators_triggered=[],
            affected_areas=[],
            potential_impact={},
            detection_method="test",
            raw_data={}
        )
        
        from ...models.resistance_mitigation_models import MitigationStrategy, MitigationIntervention, InterventionType, MitigationStatus
        
        plan = MitigationPlan(
            id=plan_id,
            detection_id=detection_id,
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            severity=ResistanceSeverity.MODERATE,
            strategies=[MitigationStrategy.TRAINING_SUPPORT],
            interventions=[],
            target_stakeholders=[],
            success_criteria={},
            timeline={},
            resource_requirements={},
            risk_factors=[],
            contingency_plans=[],
            created_at=datetime.now(),
            created_by="system"
        )
        
        execution = MitigationExecution(
            id=execution_id,
            plan_id=plan_id,
            execution_phase="completed",
            start_date=datetime.now(),
            end_date=datetime.now(),
            status=MitigationStatus.COMPLETED,
            progress_percentage=100.0,
            completed_interventions=[],
            active_interventions=[],
            pending_interventions=[],
            resource_utilization={},
            stakeholder_engagement={},
            interim_results={},
            challenges_encountered=[],
            adjustments_made=[],
            next_steps=[]
        )
        
        resolution = engine.track_resistance_resolution(
            detection=detection,
            plan=plan,
            execution=execution
        )
        
        logger.info(f"Tracked resistance resolution {resolution.id}")
        return resolution
        
    except Exception as e:
        logger.error(f"Error tracking resistance resolution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-effectiveness/{plan_id}", response_model=MitigationValidation)
async def validate_mitigation_effectiveness(
    plan_id: str,
    execution_id: str,
    post_intervention_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Validate effectiveness of mitigation efforts
    
    Args:
        plan_id: ID of mitigation plan
        execution_id: ID of execution
        post_intervention_data: Data collected after interventions
        
    Returns:
        Validation results with effectiveness assessment
    """
    try:
        engine = ResistanceMitigationEngine()
        
        # Mock objects for demonstration
        from ...models.resistance_mitigation_models import MitigationStrategy, MitigationIntervention, InterventionType, MitigationStatus
        
        plan = MitigationPlan(
            id=plan_id,
            detection_id="det_001",
            organization_id="org_001",
            transformation_id="trans_001",
            resistance_type=ResistanceType.PASSIVE_RESISTANCE,
            severity=ResistanceSeverity.MODERATE,
            strategies=[MitigationStrategy.TRAINING_SUPPORT],
            interventions=[],
            target_stakeholders=[],
            success_criteria={
                "resistance_reduction": 0.7,
                "engagement_improvement": 0.2
            },
            timeline={},
            resource_requirements={},
            risk_factors=[],
            contingency_plans=[],
            created_at=datetime.now(),
            created_by="system"
        )
        
        execution = MitigationExecution(
            id=execution_id,
            plan_id=plan_id,
            execution_phase="completed",
            start_date=datetime.now(),
            end_date=datetime.now(),
            status=MitigationStatus.COMPLETED,
            progress_percentage=100.0,
            completed_interventions=[],
            active_interventions=[],
            pending_interventions=[],
            resource_utilization={},
            stakeholder_engagement={},
            interim_results={},
            challenges_encountered=[],
            adjustments_made=[],
            next_steps=[]
        )
        
        validation = engine.validate_mitigation_effectiveness(
            plan=plan,
            execution=execution,
            post_intervention_data=post_intervention_data
        )
        
        logger.info(f"Validated mitigation effectiveness {validation.id}")
        return validation
        
    except Exception as e:
        logger.error(f"Error validating mitigation effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", response_model=List[MitigationTemplate])
async def get_mitigation_templates(
    resistance_type: Optional[str] = None,
    severity: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Get available mitigation templates
    
    Args:
        resistance_type: Filter by resistance type
        severity: Filter by severity level
        
    Returns:
        List of available mitigation templates
    """
    try:
        engine = ResistanceMitigationEngine()
        
        # Return available templates
        templates = engine.mitigation_templates
        
        # Apply filters if provided
        if resistance_type:
            templates = [t for t in templates if resistance_type in [rt.value for rt in t.resistance_types]]
        
        if severity:
            templates = [t for t in templates if severity in [s.value for s in t.severity_levels]]
        
        logger.info(f"Retrieved {len(templates)} mitigation templates")
        return templates
        
    except Exception as e:
        logger.error(f"Error retrieving mitigation templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans/{organization_id}", response_model=List[MitigationPlan])
async def get_mitigation_plans(
    organization_id: str,
    status: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
    current_user: dict = Depends(get_current_user)
):
    """
    Get mitigation plans for organization
    
    Args:
        organization_id: ID of organization
        status: Filter by plan status
        limit: Maximum number of plans to return
        offset: Number of plans to skip
        
    Returns:
        List of mitigation plans
    """
    try:
        # Mock plans for demonstration
        plans = []
        
        logger.info(f"Retrieved {len(plans)} mitigation plans for org {organization_id}")
        return plans
        
    except Exception as e:
        logger.error(f"Error retrieving mitigation plans: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{plan_id}", response_model=MitigationMetrics)
async def get_mitigation_metrics(
    plan_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get metrics for mitigation plan performance
    
    Args:
        plan_id: ID of mitigation plan
        
    Returns:
        Mitigation performance metrics
    """
    try:
        # Mock metrics for demonstration
        metrics = MitigationMetrics(
            id=f"metrics_{plan_id}",
            plan_id=plan_id,
            measurement_date=datetime.now(),
            resistance_reduction=0.75,
            engagement_improvement=0.22,
            sentiment_change=0.18,
            behavioral_compliance=0.85,
            stakeholder_satisfaction=0.80,
            intervention_effectiveness={"training": 0.85, "communication": 0.78},
            resource_efficiency=0.90,
            timeline_adherence=0.95,
            cost_effectiveness=0.88,
            sustainability_indicators={"long_term_adoption": 0.75, "behavior_persistence": 0.80}
        )
        
        logger.info(f"Retrieved metrics for mitigation plan {plan_id}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving mitigation metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/plans/{plan_id}/adjust")
async def adjust_mitigation_plan(
    plan_id: str,
    adjustments: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Adjust mitigation plan based on interim results
    
    Args:
        plan_id: ID of mitigation plan
        adjustments: Plan adjustments to make
        
    Returns:
        Adjustment confirmation
    """
    try:
        # Mock plan adjustment
        logger.info(f"Adjusted mitigation plan {plan_id}")
        return {
            "message": "Mitigation plan adjusted successfully",
            "plan_id": plan_id,
            "adjustments_applied": list(adjustments.keys()),
            "status": "updated"
        }
        
    except Exception as e:
        logger.error(f"Error adjusting mitigation plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_plan_background(
    engine: ResistanceMitigationEngine,
    plan: MitigationPlan,
    organization: Organization
):
    """Background task for executing mitigation plan"""
    try:
        execution = engine.execute_mitigation_plan(plan, organization)
        logger.info(f"Completed background execution of plan {plan.id}")
        
    except Exception as e:
        logger.error(f"Error in background plan execution: {str(e)}")