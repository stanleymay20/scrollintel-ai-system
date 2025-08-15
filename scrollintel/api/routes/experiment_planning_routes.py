"""
API routes for experiment planning framework.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ...engines.experiment_planner import ExperimentPlanner
from ...engines.methodology_selector import MethodologySelector
from ...engines.timeline_planner import TimelinePlanner, TimelineConstraint
from ...models.experimental_design_models import (
    ExperimentPlan, ValidationStudy, MethodologyRecommendation,
    ExperimentMilestone, ExperimentOptimization, MethodologyType,
    ExperimentType
)

router = APIRouter(prefix="/api/experiment-planning", tags=["experiment-planning"])
logger = logging.getLogger(__name__)

# Initialize components
experiment_planner = ExperimentPlanner()
methodology_selector = MethodologySelector()
timeline_planner = TimelinePlanner()


@router.post("/plan-experiment", response_model=Dict[str, Any])
async def plan_experiment(
    research_question: str,
    domain: str,
    constraints: Optional[Dict[str, Any]] = None
):
    """
    Create comprehensive experiment plan from research question.
    """
    try:
        experiment_plan = experiment_planner.plan_experiment(
            research_question=research_question,
            domain=domain,
            constraints=constraints
        )
        
        return {
            "success": True,
            "experiment_plan": {
                "plan_id": experiment_plan.plan_id,
                "title": experiment_plan.title,
                "research_question": experiment_plan.research_question,
                "experiment_type": experiment_plan.experiment_type.value,
                "methodology": experiment_plan.methodology.value,
                "hypotheses": [
                    {
                        "hypothesis_id": h.hypothesis_id,
                        "statement": h.statement,
                        "null_hypothesis": h.null_hypothesis,
                        "alternative_hypothesis": h.alternative_hypothesis,
                        "variables_involved": h.variables_involved,
                        "confidence_level": h.confidence_level
                    }
                    for h in experiment_plan.hypotheses
                ],
                "variables": [
                    {
                        "name": v.name,
                        "variable_type": v.variable_type,
                        "data_type": v.data_type,
                        "measurement_unit": v.measurement_unit,
                        "description": v.description
                    }
                    for v in experiment_plan.variables
                ],
                "conditions": [
                    {
                        "condition_id": c.condition_id,
                        "name": c.name,
                        "variables": c.variables,
                        "sample_size": c.sample_size,
                        "description": c.description
                    }
                    for c in experiment_plan.conditions
                ],
                "protocol": {
                    "protocol_id": experiment_plan.protocol.protocol_id,
                    "title": experiment_plan.protocol.title,
                    "objective": experiment_plan.protocol.objective,
                    "methodology": experiment_plan.protocol.methodology.value,
                    "procedures": experiment_plan.protocol.procedures,
                    "materials_required": experiment_plan.protocol.materials_required,
                    "safety_considerations": experiment_plan.protocol.safety_considerations,
                    "estimated_duration_days": experiment_plan.protocol.estimated_duration.days
                },
                "resource_requirements": [
                    {
                        "resource_type": r.resource_type,
                        "resource_name": r.resource_name,
                        "quantity_needed": r.quantity_needed,
                        "duration_needed_days": r.duration_needed.days,
                        "cost_estimate": r.cost_estimate
                    }
                    for r in experiment_plan.resource_requirements
                ],
                "timeline": [
                    {
                        "milestone_id": m.milestone_id,
                        "name": m.name,
                        "description": m.description,
                        "target_date": m.target_date.isoformat(),
                        "dependencies": m.dependencies,
                        "deliverables": m.deliverables,
                        "completion_criteria": m.completion_criteria
                    }
                    for m in experiment_plan.timeline
                ],
                "success_criteria": experiment_plan.success_criteria,
                "risk_factors": experiment_plan.risk_factors,
                "mitigation_strategies": experiment_plan.mitigation_strategies,
                "status": experiment_plan.status.value,
                "estimated_completion": experiment_plan.estimated_completion.isoformat() if experiment_plan.estimated_completion else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error planning experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-validation-study", response_model=Dict[str, Any])
async def create_validation_study(
    experiment_plan_id: str,
    validation_requirements: Optional[Dict[str, Any]] = None
):
    """
    Create validation study for experiment plan.
    """
    try:
        # In a real implementation, would retrieve experiment plan from database
        # For now, create a mock experiment plan
        mock_plan = ExperimentPlan(
            plan_id=experiment_plan_id,
            title="Mock Experiment",
            research_question="Mock research question",
            hypotheses=[],
            experiment_type=ExperimentType.CONTROLLED,
            methodology=MethodologyType.QUANTITATIVE,
            variables=[],
            conditions=[],
            protocol=None,
            resource_requirements=[],
            timeline=[],
            success_criteria=[],
            risk_factors=[],
            mitigation_strategies=[]
        )
        
        validation_study = experiment_planner.create_validation_study(
            experiment_plan=mock_plan,
            validation_requirements=validation_requirements
        )
        
        return {
            "success": True,
            "validation_study": {
                "study_id": validation_study.study_id,
                "experiment_plan_id": validation_study.experiment_plan_id,
                "validation_type": validation_study.validation_type,
                "validation_methods": validation_study.validation_methods,
                "validation_criteria": validation_study.validation_criteria,
                "expected_outcomes": validation_study.expected_outcomes,
                "status": validation_study.status.value
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating validation study: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend-methodology", response_model=Dict[str, Any])
async def recommend_methodology(
    research_question: str,
    domain: str,
    constraints: Optional[Dict[str, Any]] = None
):
    """
    Recommend optimal experimental methodologies.
    """
    try:
        recommendations = experiment_planner.recommend_methodology(
            research_question=research_question,
            domain=domain,
            constraints=constraints
        )
        
        return {
            "success": True,
            "recommendations": [
                {
                    "methodology": r.methodology.value,
                    "experiment_type": r.experiment_type.value,
                    "suitability_score": r.suitability_score,
                    "advantages": r.advantages,
                    "disadvantages": r.disadvantages,
                    "estimated_duration_days": r.estimated_duration.days,
                    "confidence_level": r.confidence_level,
                    "resource_requirements": [
                        {
                            "resource_type": res.resource_type,
                            "resource_name": res.resource_name,
                            "quantity_needed": res.quantity_needed,
                            "cost_estimate": res.cost_estimate
                        }
                        for res in r.resource_requirements
                    ]
                }
                for r in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"Error recommending methodology: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-experiment-design", response_model=Dict[str, Any])
async def optimize_experiment_design(
    experiment_plan_id: str,
    optimization_goals: List[str]
):
    """
    Optimize experiment design for specific goals.
    """
    try:
        # Mock experiment plan for demonstration
        mock_plan = ExperimentPlan(
            plan_id=experiment_plan_id,
            title="Mock Experiment",
            research_question="Mock research question",
            hypotheses=[],
            experiment_type=ExperimentType.CONTROLLED,
            methodology=MethodologyType.QUANTITATIVE,
            variables=[],
            conditions=[],
            protocol=None,
            resource_requirements=[],
            timeline=[],
            success_criteria=[],
            risk_factors=[],
            mitigation_strategies=[]
        )
        
        optimization = experiment_planner.optimize_experiment_design(
            experiment_plan=mock_plan,
            optimization_goals=optimization_goals
        )
        
        return {
            "success": True,
            "optimization": {
                "optimization_id": optimization.optimization_id,
                "experiment_plan_id": optimization.experiment_plan_id,
                "optimization_type": optimization.optimization_type,
                "current_metrics": optimization.current_metrics,
                "optimized_metrics": optimization.optimized_metrics,
                "optimization_strategies": optimization.optimization_strategies,
                "trade_offs": optimization.trade_offs,
                "implementation_steps": optimization.implementation_steps,
                "confidence_score": optimization.confidence_score
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing experiment design: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-methodology", response_model=Dict[str, Any])
async def select_methodology(
    research_context: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
    preferences: Optional[Dict[str, Any]] = None
):
    """
    Select optimal experimental methodology.
    """
    try:
        recommendation = methodology_selector.select_optimal_methodology(
            research_context=research_context,
            constraints=constraints,
            preferences=preferences
        )
        
        return {
            "success": True,
            "recommendation": {
                "methodology": recommendation.methodology.value,
                "experiment_type": recommendation.experiment_type.value,
                "suitability_score": recommendation.suitability_score,
                "advantages": recommendation.advantages,
                "disadvantages": recommendation.disadvantages,
                "estimated_duration_days": recommendation.estimated_duration.days,
                "confidence_level": recommendation.confidence_level,
                "resource_requirements": [
                    {
                        "resource_type": r.resource_type,
                        "resource_name": r.resource_name,
                        "quantity_needed": r.quantity_needed,
                        "cost_estimate": r.cost_estimate
                    }
                    for r in recommendation.resource_requirements
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error selecting methodology: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-methodologies", response_model=Dict[str, Any])
async def compare_methodologies(
    research_context: Dict[str, Any],
    methodologies: List[str],
    constraints: Optional[Dict[str, Any]] = None
):
    """
    Compare multiple methodologies for research context.
    """
    try:
        # Convert string methodologies to enum
        methodology_enums = []
        for method in methodologies:
            try:
                methodology_enums.append(MethodologyType(method))
            except ValueError:
                logger.warning(f"Invalid methodology: {method}")
        
        if not methodology_enums:
            raise HTTPException(status_code=400, detail="No valid methodologies provided")
        
        recommendations = methodology_selector.compare_methodologies(
            research_context=research_context,
            methodologies=methodology_enums,
            constraints=constraints
        )
        
        return {
            "success": True,
            "comparisons": [
                {
                    "methodology": r.methodology.value,
                    "experiment_type": r.experiment_type.value,
                    "suitability_score": r.suitability_score,
                    "advantages": r.advantages,
                    "disadvantages": r.disadvantages,
                    "estimated_duration_days": r.estimated_duration.days,
                    "confidence_level": r.confidence_level
                }
                for r in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"Error comparing methodologies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-timeline", response_model=Dict[str, Any])
async def create_timeline(
    protocol_data: Dict[str, Any],
    resources_data: List[Dict[str, Any]],
    experiment_type: str,
    methodology: str,
    constraints: Optional[List[Dict[str, Any]]] = None,
    start_date: Optional[str] = None
):
    """
    Create experiment timeline with milestones.
    """
    try:
        # Convert string enums
        exp_type = ExperimentType(experiment_type)
        method_type = MethodologyType(methodology)
        
        # Parse start date
        start_dt = None
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        
        # Convert constraints
        timeline_constraints = []
        if constraints:
            for constraint_data in constraints:
                constraint = TimelineConstraint(
                    constraint_type=constraint_data.get('constraint_type', 'general'),
                    description=constraint_data.get('description', ''),
                    start_date=datetime.fromisoformat(constraint_data['start_date']) if constraint_data.get('start_date') else None,
                    end_date=datetime.fromisoformat(constraint_data['end_date']) if constraint_data.get('end_date') else None,
                    priority=constraint_data.get('priority', 'medium')
                )
                timeline_constraints.append(constraint)
        
        # Create mock protocol and resources for demonstration
        from ...models.experimental_design_models import ExperimentalProtocol, ResourceRequirement
        
        protocol = ExperimentalProtocol(
            protocol_id="mock_protocol",
            title=protocol_data.get('title', 'Mock Protocol'),
            objective=protocol_data.get('objective', 'Mock objective'),
            methodology=method_type,
            procedures=protocol_data.get('procedures', []),
            materials_required=protocol_data.get('materials_required', []),
            safety_considerations=protocol_data.get('safety_considerations', []),
            quality_controls=protocol_data.get('quality_controls', []),
            data_collection_methods=protocol_data.get('data_collection_methods', []),
            analysis_plan=protocol_data.get('analysis_plan', 'Mock analysis plan'),
            estimated_duration=timedelta(days=protocol_data.get('estimated_duration_days', 56))
        )
        
        resources = []
        for res_data in resources_data:
            resource = ResourceRequirement(
                resource_type=res_data.get('resource_type', 'general'),
                resource_name=res_data.get('resource_name', 'Mock Resource'),
                quantity_needed=res_data.get('quantity_needed', 1),
                duration_needed=timedelta(days=res_data.get('duration_needed_days', 30)),
                cost_estimate=res_data.get('cost_estimate', 1000.0)
            )
            resources.append(resource)
        
        milestones = timeline_planner.create_experiment_timeline(
            protocol=protocol,
            resources=resources,
            experiment_type=exp_type,
            methodology=method_type,
            constraints=timeline_constraints,
            start_date=start_dt
        )
        
        return {
            "success": True,
            "timeline": [
                {
                    "milestone_id": m.milestone_id,
                    "name": m.name,
                    "description": m.description,
                    "target_date": m.target_date.isoformat(),
                    "dependencies": m.dependencies,
                    "deliverables": m.deliverables,
                    "completion_criteria": m.completion_criteria
                }
                for m in milestones
            ]
        }
        
    except Exception as e:
        logger.error(f"Error creating timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-timeline", response_model=Dict[str, Any])
async def optimize_timeline(
    milestones_data: List[Dict[str, Any]],
    optimization_goals: List[str],
    constraints: Optional[List[Dict[str, Any]]] = None
):
    """
    Optimize experiment timeline for specific goals.
    """
    try:
        # Convert milestone data to objects
        milestones = []
        for milestone_data in milestones_data:
            milestone = ExperimentMilestone(
                milestone_id=milestone_data.get('milestone_id', ''),
                name=milestone_data.get('name', ''),
                description=milestone_data.get('description', ''),
                target_date=datetime.fromisoformat(milestone_data['target_date']),
                dependencies=milestone_data.get('dependencies', []),
                deliverables=milestone_data.get('deliverables', []),
                completion_criteria=milestone_data.get('completion_criteria', [])
            )
            milestones.append(milestone)
        
        # Convert constraints
        timeline_constraints = []
        if constraints:
            for constraint_data in constraints:
                constraint = TimelineConstraint(
                    constraint_type=constraint_data.get('constraint_type', 'general'),
                    description=constraint_data.get('description', ''),
                    start_date=datetime.fromisoformat(constraint_data['start_date']) if constraint_data.get('start_date') else None,
                    end_date=datetime.fromisoformat(constraint_data['end_date']) if constraint_data.get('end_date') else None,
                    priority=constraint_data.get('priority', 'medium')
                )
                timeline_constraints.append(constraint)
        
        optimized_milestones, optimization_result = timeline_planner.optimize_timeline(
            milestones=milestones,
            optimization_goals=optimization_goals,
            constraints=timeline_constraints
        )
        
        return {
            "success": True,
            "optimized_timeline": [
                {
                    "milestone_id": m.milestone_id,
                    "name": m.name,
                    "description": m.description,
                    "target_date": m.target_date.isoformat(),
                    "dependencies": m.dependencies,
                    "deliverables": m.deliverables,
                    "completion_criteria": m.completion_criteria
                }
                for m in optimized_milestones
            ],
            "optimization_result": {
                "original_duration_days": optimization_result.original_duration.days,
                "optimized_duration_days": optimization_result.optimized_duration.days,
                "time_savings_days": optimization_result.time_savings.days,
                "optimization_strategies": optimization_result.optimization_strategies,
                "trade_offs": optimization_result.trade_offs,
                "confidence_score": optimization_result.confidence_score
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "experiment-planning"}