"""
Crisis Preparedness API Routes

API endpoints for crisis preparedness assessment and enhancement.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging

from ...engines.crisis_preparedness_engine import CrisisPreparednessEngine
from ...models.crisis_preparedness_models import (
    PreparednessAssessment, CrisisSimulation, SimulationType, TrainingProgram,
    TrainingType, CapabilityDevelopment, CapabilityArea, PreparednessLevel,
    PreparednessReport
)

router = APIRouter(prefix="/api/v1/crisis-preparedness", tags=["crisis-preparedness"])
logger = logging.getLogger(__name__)


def get_preparedness_engine() -> CrisisPreparednessEngine:
    """Dependency to get preparedness engine instance"""
    return CrisisPreparednessEngine()


@router.post("/assess")
async def assess_crisis_preparedness(
    organization_data: Dict[str, Any],
    assessor_id: str,
    engine: CrisisPreparednessEngine = Depends(get_preparedness_engine)
) -> PreparednessAssessment:
    """Conduct comprehensive crisis preparedness assessment"""
    try:
        assessment = engine.assess_crisis_preparedness(
            organization_data=organization_data,
            assessor_id=assessor_id
        )
        
        logger.info(f"Completed preparedness assessment: {assessment.id}")
        return assessment
        
    except Exception as e:
        logger.error(f"Error conducting preparedness assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulation/create")
async def create_crisis_simulation(
    simulation_type: str,
    participants: List[str],
    facilitator_id: str,
    custom_parameters: Optional[Dict[str, Any]] = None,
    engine: CrisisPreparednessEngine = Depends(get_preparedness_engine)
) -> CrisisSimulation:
    """Create crisis simulation exercise"""
    try:
        # Convert string to enum
        sim_type = SimulationType(simulation_type)
        
        simulation = engine.create_crisis_simulation(
            simulation_type=sim_type,
            participants=participants,
            facilitator_id=facilitator_id,
            custom_parameters=custom_parameters
        )
        
        logger.info(f"Created crisis simulation: {simulation.id}")
        return simulation
        
    except ValueError as e:
        logger.error(f"Invalid simulation type: {simulation_type}")
        raise HTTPException(status_code=400, detail=f"Invalid simulation type: {simulation_type}")
    except Exception as e:
        logger.error(f"Error creating crisis simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulation/{simulation_id}/execute")
async def execute_simulation(
    simulation_id: str,
    simulation_data: Dict[str, Any],
    engine: CrisisPreparednessEngine = Depends(get_preparedness_engine)
) -> Dict[str, Any]:
    """Execute crisis simulation and collect results"""
    try:
        # Mock simulation object for demonstration
        simulation = CrisisSimulation(
            id=simulation_id,
            simulation_type=SimulationType.SYSTEM_OUTAGE,
            title=simulation_data.get("title", "Crisis Simulation"),
            description=simulation_data.get("description", "Crisis response simulation"),
            scenario_details=simulation_data.get("scenario_details", "Simulation scenario"),
            complexity_level=simulation_data.get("complexity_level", "Medium"),
            duration_minutes=simulation_data.get("duration_minutes", 120),
            participants=simulation_data.get("participants", []),
            learning_objectives=simulation_data.get("learning_objectives", []),
            success_criteria=simulation_data.get("success_criteria", []),
            facilitator_id=simulation_data.get("facilitator_id", ""),
            scheduled_date=simulation_data.get("scheduled_date", "2024-01-01T00:00:00"),
            actual_start_time=None,
            actual_end_time=None,
            participant_performance={},
            objectives_achieved=[],
            lessons_learned=[],
            improvement_areas=[],
            simulation_status="scheduled",
            feedback_collected=False,
            report_generated=False
        )
        
        results = engine.execute_simulation(simulation)
        
        logger.info(f"Executed simulation: {simulation_id}")
        return {
            "simulation_id": simulation_id,
            "execution_results": results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error executing simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/develop")
async def develop_training_program(
    capability_area: str,
    target_audience: List[str],
    training_type: str,
    engine: CrisisPreparednessEngine = Depends(get_preparedness_engine)
) -> TrainingProgram:
    """Develop crisis response training program"""
    try:
        # Convert strings to enums
        capability = CapabilityArea(capability_area)
        t_type = TrainingType(training_type)
        
        program = engine.develop_training_program(
            capability_area=capability,
            target_audience=target_audience,
            training_type=t_type
        )
        
        logger.info(f"Developed training program: {program.id}")
        return program
        
    except ValueError as e:
        logger.error(f"Invalid parameter: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error developing training program: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capability/develop")
async def create_capability_development_plan(
    capability_area: str,
    current_level: str,
    target_level: str,
    assessment_data: Dict[str, Any],
    engine: CrisisPreparednessEngine = Depends(get_preparedness_engine)
) -> CapabilityDevelopment:
    """Create capability development plan"""
    try:
        # Convert strings to enums
        capability = CapabilityArea(capability_area)
        target = PreparednessLevel(target_level)
        
        # Mock assessment object
        assessment = PreparednessAssessment(
            id=assessment_data.get("id", "mock_assessment"),
            assessment_date="2024-01-01T00:00:00",
            assessor_id=assessment_data.get("assessor_id", ""),
            overall_preparedness_level=PreparednessLevel(current_level),
            overall_score=assessment_data.get("overall_score", 75.0),
            capability_scores={capability: assessment_data.get("capability_score", 70.0)},
            capability_levels={capability: PreparednessLevel(current_level)},
            strengths=assessment_data.get("strengths", []),
            weaknesses=assessment_data.get("weaknesses", []),
            gaps_identified=assessment_data.get("gaps_identified", []),
            high_risk_scenarios=assessment_data.get("high_risk_scenarios", []),
            vulnerability_areas=assessment_data.get("vulnerability_areas", []),
            improvement_priorities=assessment_data.get("improvement_priorities", []),
            recommended_actions=assessment_data.get("recommended_actions", []),
            assessment_methodology=assessment_data.get("assessment_methodology", ""),
            data_sources=assessment_data.get("data_sources", []),
            confidence_level=assessment_data.get("confidence_level", 80.0)
        )
        
        plan = engine.create_capability_development_plan(
            capability_area=capability,
            current_assessment=assessment,
            target_level=target
        )
        
        logger.info(f"Created capability development plan: {plan.id}")
        return plan
        
    except ValueError as e:
        logger.error(f"Invalid parameter: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating capability development plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/generate")
async def generate_preparedness_report(
    assessment_data: Dict[str, Any],
    simulations_data: List[Dict[str, Any]] = [],
    training_data: List[Dict[str, Any]] = [],
    engine: CrisisPreparednessEngine = Depends(get_preparedness_engine)
) -> PreparednessReport:
    """Generate comprehensive preparedness report"""
    try:
        # Mock objects for demonstration
        assessment = PreparednessAssessment(
            id=assessment_data.get("id", "mock_assessment"),
            assessment_date="2024-01-01T00:00:00",
            assessor_id=assessment_data.get("assessor_id", ""),
            overall_preparedness_level=PreparednessLevel.GOOD,
            overall_score=assessment_data.get("overall_score", 80.0),
            capability_scores={},
            capability_levels={},
            strengths=assessment_data.get("strengths", []),
            weaknesses=assessment_data.get("weaknesses", []),
            gaps_identified=assessment_data.get("gaps_identified", []),
            high_risk_scenarios=assessment_data.get("high_risk_scenarios", []),
            vulnerability_areas=assessment_data.get("vulnerability_areas", []),
            improvement_priorities=assessment_data.get("improvement_priorities", []),
            recommended_actions=assessment_data.get("recommended_actions", []),
            assessment_methodology="Mock Assessment",
            data_sources=[],
            confidence_level=80.0
        )
        
        simulations = []
        training_programs = []
        
        report = engine.generate_preparedness_report(
            assessment=assessment,
            simulations=simulations,
            training_programs=training_programs
        )
        
        logger.info(f"Generated preparedness report: {report.id}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating preparedness report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assessment/{assessment_id}")
async def get_assessment(assessment_id: str) -> Dict[str, Any]:
    """Get preparedness assessment by ID"""
    try:
        # Mock response - in real implementation, retrieve from database
        return {
            "id": assessment_id,
            "overall_score": 82.5,
            "overall_level": "good",
            "capabilities_assessed": 7,
            "strengths_count": 4,
            "weaknesses_count": 3,
            "assessment_date": "2024-01-01T00:00:00"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulations")
async def list_simulations(
    simulation_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """List crisis simulations with optional filtering"""
    try:
        # Mock response - in real implementation, query database
        simulations = [
            {
                "id": f"simulation_{i}",
                "type": simulation_type or "system_outage",
                "title": f"Crisis Simulation {i}",
                "status": status or "scheduled",
                "participants_count": 5 + i,
                "scheduled_date": "2024-01-01T00:00:00"
            }
            for i in range(min(limit, 10))
        ]
        
        logger.info(f"Retrieved {len(simulations)} simulations")
        return simulations
        
    except Exception as e:
        logger.error(f"Error listing simulations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-programs")
async def list_training_programs(
    capability_area: Optional[str] = None,
    training_type: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """List training programs with optional filtering"""
    try:
        # Mock response - in real implementation, query database
        programs = [
            {
                "id": f"program_{i}",
                "name": f"Crisis Training Program {i}",
                "capability_area": capability_area or "crisis_detection",
                "training_type": training_type or "tabletop_exercise",
                "duration_hours": 4.0 + i,
                "status": "approved"
            }
            for i in range(min(limit, 10))
        ]
        
        logger.info(f"Retrieved {len(programs)} training programs")
        return programs
        
    except Exception as e:
        logger.error(f"Error listing training programs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capability-plans")
async def list_capability_development_plans(
    capability_area: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """List capability development plans with optional filtering"""
    try:
        # Mock response - in real implementation, query database
        plans = [
            {
                "id": f"plan_{i}",
                "capability_area": capability_area or "crisis_detection",
                "current_level": "adequate",
                "target_level": "good",
                "progress": 25.0 + i * 10,
                "status": status or "in_progress",
                "target_completion": "2024-06-01T00:00:00"
            }
            for i in range(min(limit, 10))
        ]
        
        logger.info(f"Retrieved {len(plans)} capability development plans")
        return plans
        
    except Exception as e:
        logger.error(f"Error listing capability development plans: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/preparedness")
async def get_preparedness_metrics(
    time_period: str = "30d"
) -> Dict[str, Any]:
    """Get crisis preparedness metrics"""
    try:
        # Mock metrics
        metrics = {
            "overall_preparedness_score": 82.3,
            "assessments_completed": 12,
            "simulations_conducted": 8,
            "training_programs_delivered": 15,
            "capability_improvements": 23,
            "high_risk_scenarios_addressed": 6,
            "time_period": time_period
        }
        
        logger.info("Retrieved preparedness metrics")
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving preparedness metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulation/{simulation_id}/feedback")
async def submit_simulation_feedback(
    simulation_id: str,
    feedback_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Submit feedback for simulation exercise"""
    try:
        # Mock feedback processing
        result = {
            "simulation_id": simulation_id,
            "feedback_submitted": True,
            "participant_id": feedback_data.get("participant_id"),
            "overall_rating": feedback_data.get("overall_rating", 0),
            "comments": feedback_data.get("comments", ""),
            "submission_date": "2024-01-01T00:00:00"
        }
        
        logger.info(f"Submitted feedback for simulation {simulation_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error submitting simulation feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capability-plan/{plan_id}/update-progress")
async def update_capability_plan_progress(
    plan_id: str,
    progress_update: Dict[str, Any]
) -> Dict[str, Any]:
    """Update progress on capability development plan"""
    try:
        # Mock progress update
        result = {
            "plan_id": plan_id,
            "previous_progress": progress_update.get("previous_progress", 0),
            "new_progress": progress_update.get("new_progress", 0),
            "completed_actions": progress_update.get("completed_actions", []),
            "updated_by": progress_update.get("updated_by"),
            "update_date": "2024-01-01T00:00:00",
            "notes": progress_update.get("notes", "")
        }
        
        logger.info(f"Updated capability plan {plan_id} progress")
        return result
        
    except Exception as e:
        logger.error(f"Error updating capability plan progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))