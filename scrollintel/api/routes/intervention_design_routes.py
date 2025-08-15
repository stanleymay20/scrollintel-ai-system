"""
Intervention Design API Routes

API endpoints for strategic intervention design, effectiveness prediction,
and sequence optimization.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from ...engines.intervention_design_engine import InterventionDesignEngine
from ...models.intervention_design_models import (
    InterventionDesignRequest, InterventionDesignResult, InterventionDesign,
    EffectivenessPrediction, InterventionSequence, InterventionOptimization
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/intervention-design", tags=["intervention-design"])

# Initialize engine
intervention_engine = InterventionDesignEngine()


@router.post("/design", response_model=InterventionDesignResult)
async def design_interventions(request: InterventionDesignRequest):
    """
    Design comprehensive intervention strategy
    
    Args:
        request: Intervention design request with requirements
        
    Returns:
        Complete intervention design result
    """
    try:
        logger.info(f"Designing interventions for organization {request.organization_id}")
        
        result = intervention_engine.design_interventions(request)
        
        logger.info(f"Successfully designed {len(result.interventions)} interventions")
        return result
        
    except Exception as e:
        logger.error(f"Error designing interventions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-effectiveness", response_model=EffectivenessPrediction)
async def predict_intervention_effectiveness(
    intervention: InterventionDesign,
    context: Dict[str, Any]
):
    """
    Predict effectiveness of a specific intervention
    
    Args:
        intervention: Intervention design to evaluate
        context: Organizational and situational context
        
    Returns:
        Effectiveness prediction
    """
    try:
        logger.info(f"Predicting effectiveness for intervention {intervention.id}")
        
        prediction = intervention_engine.predict_intervention_effectiveness(intervention, context)
        
        logger.info(f"Predicted effectiveness: {prediction.predicted_effectiveness.value}")
        return prediction
        
    except Exception as e:
        logger.error(f"Error predicting intervention effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-sequence", response_model=InterventionSequence)
async def optimize_intervention_sequence(
    interventions: List[InterventionDesign],
    constraints: Dict[str, Any]
):
    """
    Optimize the sequence and coordination of interventions
    
    Args:
        interventions: List of interventions to sequence
        constraints: Sequencing constraints and preferences
        
    Returns:
        Optimized intervention sequence
    """
    try:
        logger.info(f"Optimizing sequence for {len(interventions)} interventions")
        
        sequence = intervention_engine.optimize_intervention_sequence(interventions, constraints)
        
        logger.info(f"Optimized sequence with {sequence.total_duration.days} day duration")
        return sequence
        
    except Exception as e:
        logger.error(f"Error optimizing intervention sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intervention/{intervention_id}")
async def get_intervention(intervention_id: str):
    """
    Retrieve an intervention design by ID
    
    Args:
        intervention_id: ID of the intervention to retrieve
        
    Returns:
        Intervention design details
    """
    try:
        # In a real implementation, this would query a database
        # For now, return a placeholder response
        raise HTTPException(status_code=404, detail="Intervention not found")
        
    except Exception as e:
        logger.error(f"Error retrieving intervention {intervention_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{org_id}/interventions")
async def get_organization_interventions(org_id: str):
    """
    Retrieve all interventions for an organization
    
    Args:
        org_id: Organization ID
        
    Returns:
        List of intervention designs
    """
    try:
        # In a real implementation, this would query a database
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Error retrieving interventions for organization {org_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intervention/{intervention_id}/optimize")
async def optimize_single_intervention(
    intervention_id: str,
    optimization_criteria: Dict[str, Any]
):
    """
    Optimize a single intervention design
    
    Args:
        intervention_id: ID of the intervention to optimize
        optimization_criteria: Criteria for optimization
        
    Returns:
        Optimization recommendations
    """
    try:
        logger.info(f"Optimizing intervention {intervention_id}")
        
        # In a real implementation, would load intervention and apply optimizations
        recommendations = [
            {
                "type": "Effectiveness",
                "current_issue": "Low engagement predicted",
                "recommendation": "Add interactive elements to activities",
                "expected_benefit": "Increased participant engagement",
                "priority": 1
            },
            {
                "type": "Resources",
                "current_issue": "High resource requirements",
                "recommendation": "Leverage existing training materials",
                "expected_benefit": "Reduced resource needs",
                "priority": 2
            }
        ]
        
        return {
            "intervention_id": intervention_id,
            "optimization_recommendations": recommendations,
            "overall_improvement_potential": 0.25,
            "implementation_effort": "Medium"
        }
        
    except Exception as e:
        logger.error(f"Error optimizing intervention {intervention_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/intervention-types")
async def get_intervention_type_templates():
    """
    Get available intervention type templates
    
    Returns:
        List of intervention type templates
    """
    try:
        templates = {
            "training": {
                "name": "Training Intervention",
                "description": "Structured learning programs to develop specific skills and knowledge",
                "typical_activities": [
                    "Needs assessment and gap analysis",
                    "Learning objective definition",
                    "Content development and curation",
                    "Training delivery (classroom, online, or blended)",
                    "Practice sessions and skill application",
                    "Knowledge assessment and validation",
                    "Follow-up coaching and reinforcement"
                ],
                "typical_duration": "2-6 weeks",
                "resource_requirements": {
                    "human": ["Subject matter experts", "Trainers/facilitators", "Participants"],
                    "material": ["Training materials", "Assessment tools", "Reference guides"],
                    "technology": ["Learning management system", "Presentation tools", "Assessment platform"],
                    "space": ["Training rooms", "Online meeting platforms"]
                },
                "success_factors": [
                    "Clear learning objectives",
                    "Engaging content and delivery",
                    "Practical application opportunities",
                    "Regular feedback and assessment",
                    "Post-training reinforcement"
                ],
                "measurement_methods": [
                    "Pre/post knowledge assessments",
                    "Skill demonstration evaluations",
                    "Participant satisfaction surveys",
                    "Behavioral change observation",
                    "Performance improvement metrics"
                ]
            },
            "communication": {
                "name": "Communication Intervention",
                "description": "Strategic communication campaigns to inform, persuade, and engage stakeholders",
                "typical_activities": [
                    "Audience analysis and segmentation",
                    "Message strategy development",
                    "Content creation and adaptation",
                    "Channel selection and optimization",
                    "Campaign launch and execution",
                    "Feedback collection and analysis",
                    "Message refinement and iteration"
                ],
                "typical_duration": "1-4 weeks",
                "resource_requirements": {
                    "human": ["Communication specialists", "Content creators", "Channel managers"],
                    "material": ["Communication materials", "Visual assets", "Message templates"],
                    "technology": ["Communication platforms", "Analytics tools", "Content management systems"],
                    "budget": ["Media costs", "Production costs", "Distribution costs"]
                },
                "success_factors": [
                    "Clear and compelling messaging",
                    "Appropriate channel selection",
                    "Consistent message delivery",
                    "Two-way communication opportunities",
                    "Regular message reinforcement"
                ],
                "measurement_methods": [
                    "Message reach and frequency metrics",
                    "Engagement and interaction rates",
                    "Message comprehension surveys",
                    "Attitude and perception changes",
                    "Behavioral response indicators"
                ]
            },
            "behavioral_nudge": {
                "name": "Behavioral Nudge Intervention",
                "description": "Subtle environmental and contextual changes to encourage desired behaviors",
                "typical_activities": [
                    "Behavior analysis and mapping",
                    "Decision point identification",
                    "Nudge design and testing",
                    "Environmental modification",
                    "Choice architecture optimization",
                    "Behavior monitoring and tracking",
                    "Nudge refinement and scaling"
                ],
                "typical_duration": "2-8 weeks",
                "resource_requirements": {
                    "human": ["Behavioral scientists", "UX designers", "Implementation team"],
                    "material": ["Signage and visual cues", "Digital interfaces", "Environmental modifications"],
                    "technology": ["Behavior tracking tools", "A/B testing platforms", "Analytics systems"],
                    "space": ["Physical environment access", "Digital platform access"]
                },
                "success_factors": [
                    "Deep understanding of target behaviors",
                    "Subtle and non-intrusive design",
                    "Contextually appropriate nudges",
                    "Continuous testing and optimization",
                    "Sustainable implementation"
                ],
                "measurement_methods": [
                    "Behavior frequency tracking",
                    "Choice selection analysis",
                    "A/B testing results",
                    "Long-term behavior sustainability",
                    "Unintended consequence monitoring"
                ]
            },
            "process_change": {
                "name": "Process Change Intervention",
                "description": "Systematic modification of organizational processes to embed cultural values",
                "typical_activities": [
                    "Current process mapping and analysis",
                    "Cultural alignment assessment",
                    "Process redesign and optimization",
                    "Stakeholder consultation and buy-in",
                    "Implementation planning and execution",
                    "Training on new processes",
                    "Monitoring and continuous improvement"
                ],
                "typical_duration": "4-12 weeks",
                "resource_requirements": {
                    "human": ["Process analysts", "Change managers", "Process owners", "End users"],
                    "material": ["Process documentation", "Training materials", "Implementation guides"],
                    "technology": ["Process management tools", "Workflow systems", "Monitoring dashboards"],
                    "governance": ["Change approval processes", "Quality assurance", "Compliance validation"]
                },
                "success_factors": [
                    "Clear process objectives and outcomes",
                    "Stakeholder involvement and buy-in",
                    "Comprehensive training and support",
                    "Effective change management",
                    "Continuous monitoring and improvement"
                ],
                "measurement_methods": [
                    "Process efficiency metrics",
                    "Quality and compliance indicators",
                    "User adoption and satisfaction",
                    "Cultural alignment assessment",
                    "Business impact measurement"
                ]
            },
            "leadership_modeling": {
                "name": "Leadership Modeling Intervention",
                "description": "Strategic leadership behaviors to demonstrate and reinforce cultural values",
                "typical_activities": [
                    "Leadership behavior assessment",
                    "Cultural value alignment mapping",
                    "Behavior change planning",
                    "Leadership coaching and development",
                    "Visible leadership actions",
                    "Storytelling and communication",
                    "Feedback and continuous improvement"
                ],
                "typical_duration": "6-16 weeks",
                "resource_requirements": {
                    "human": ["Executive coaches", "Leadership team", "Communication support"],
                    "material": ["Leadership development materials", "Communication templates", "Feedback tools"],
                    "technology": ["360-degree feedback systems", "Communication platforms", "Progress tracking tools"],
                    "support": ["Executive coaching", "Peer mentoring", "Leadership circles"]
                },
                "success_factors": [
                    "Authentic leadership commitment",
                    "Consistent behavior demonstration",
                    "Clear value-behavior connections",
                    "Regular feedback and adjustment",
                    "Cascading leadership influence"
                ],
                "measurement_methods": [
                    "Leadership behavior assessments",
                    "Employee perception surveys",
                    "Cultural climate indicators",
                    "Engagement and trust metrics",
                    "Organizational performance indicators"
                ]
            }
        }
        
        return templates
        
    except Exception as e:
        logger.error(f"Error retrieving intervention type templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/effectiveness-factors")
async def get_effectiveness_factors():
    """
    Get factors that influence intervention effectiveness
    
    Returns:
        List of effectiveness factors and their impact
    """
    try:
        factors = {
            "organizational_factors": {
                "leadership_support": {
                    "description": "Level of visible leadership commitment and support",
                    "impact_weight": 0.25,
                    "measurement": "Leadership engagement score, resource allocation, public statements"
                },
                "organizational_readiness": {
                    "description": "Organization's capacity and willingness to change",
                    "impact_weight": 0.20,
                    "measurement": "Change readiness assessment, past change success, current stability"
                },
                "cultural_alignment": {
                    "description": "Alignment between intervention and existing culture",
                    "impact_weight": 0.15,
                    "measurement": "Culture assessment, value alignment, norm compatibility"
                },
                "resource_availability": {
                    "description": "Adequacy of resources (time, budget, people) for intervention",
                    "impact_weight": 0.15,
                    "measurement": "Resource allocation, budget approval, time availability"
                }
            },
            "intervention_factors": {
                "design_quality": {
                    "description": "Quality and appropriateness of intervention design",
                    "impact_weight": 0.20,
                    "measurement": "Design review scores, expert evaluation, best practice alignment"
                },
                "participant_engagement": {
                    "description": "Level of participant involvement and enthusiasm",
                    "impact_weight": 0.18,
                    "measurement": "Participation rates, engagement surveys, feedback quality"
                },
                "facilitator_competence": {
                    "description": "Skills and experience of intervention facilitators",
                    "impact_weight": 0.12,
                    "measurement": "Facilitator credentials, experience, participant ratings"
                },
                "timing_appropriateness": {
                    "description": "Appropriateness of intervention timing and sequencing",
                    "impact_weight": 0.10,
                    "measurement": "Timing assessment, competing priorities, organizational calendar"
                }
            },
            "contextual_factors": {
                "external_environment": {
                    "description": "External pressures and opportunities affecting intervention",
                    "impact_weight": 0.08,
                    "measurement": "Market conditions, regulatory changes, competitive pressures"
                },
                "internal_dynamics": {
                    "description": "Internal organizational dynamics and politics",
                    "impact_weight": 0.07,
                    "measurement": "Stakeholder analysis, political climate, internal conflicts"
                },
                "communication_effectiveness": {
                    "description": "Quality and reach of intervention communication",
                    "impact_weight": 0.10,
                    "measurement": "Communication reach, message clarity, feedback quality"
                }
            }
        }
        
        return factors
        
    except Exception as e:
        logger.error(f"Error retrieving effectiveness factors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sequence/analyze-dependencies")
async def analyze_intervention_dependencies(interventions: List[InterventionDesign]):
    """
    Analyze dependencies between interventions
    
    Args:
        interventions: List of interventions to analyze
        
    Returns:
        Dependency analysis results
    """
    try:
        logger.info(f"Analyzing dependencies for {len(interventions)} interventions")
        
        # Mock dependency analysis
        dependencies = {}
        parallel_opportunities = []
        sequential_requirements = []
        
        for i, intervention in enumerate(interventions):
            # Simple rule-based dependency analysis
            if intervention.intervention_type.value == "training":
                if any(other.intervention_type.value == "communication" for other in interventions):
                    dependencies[intervention.id] = [
                        other.id for other in interventions 
                        if other.intervention_type.value == "communication"
                    ]
                    sequential_requirements.append({
                        "intervention": intervention.name,
                        "requires": "Communication intervention must precede training"
                    })
            
            # Identify parallel opportunities
            if intervention.intervention_type.value in ["behavioral_nudge", "environmental_change"]:
                parallel_candidates = [
                    other.id for other in interventions 
                    if other.intervention_type.value in ["behavioral_nudge", "environmental_change"]
                    and other.id != intervention.id
                ]
                if parallel_candidates:
                    parallel_opportunities.append({
                        "primary": intervention.id,
                        "parallel_with": parallel_candidates,
                        "rationale": "Environmental and behavioral interventions can run simultaneously"
                    })
        
        return {
            "total_interventions": len(interventions),
            "dependencies": dependencies,
            "parallel_opportunities": parallel_opportunities,
            "sequential_requirements": sequential_requirements,
            "complexity_score": len(dependencies) / len(interventions) if interventions else 0,
            "optimization_potential": len(parallel_opportunities) / len(interventions) if interventions else 0,
            "recommendations": [
                "Consider running behavioral and environmental interventions in parallel",
                "Ensure communication interventions precede training programs",
                "Monitor resource conflicts between parallel interventions",
                "Plan coordination meetings for dependent interventions"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing intervention dependencies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))