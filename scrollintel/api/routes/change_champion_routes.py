"""
Change Champion API Routes

REST API endpoints for change champion identification, development, and network management.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ...engines.change_champion_development_engine import ChangeChampionDevelopmentEngine
from ...models.change_champion_models import (
    ChangeChampionProfile, ChampionLevel, ChampionRole, ChangeCapability,
    ChampionDevelopmentProgram, ChampionNetwork, ChampionPerformanceMetrics
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/change-champions", tags=["change-champions"])

# Initialize engine
champion_engine = ChangeChampionDevelopmentEngine()


@router.post("/identify", response_model=Dict[str, Any])
async def identify_potential_champions(
    organization_id: str,
    employee_data: List[Dict[str, Any]],
    criteria_type: str = "standard",
    target_count: Optional[int] = None
):
    """Identify potential change champions from employee data"""
    try:
        logger.info(f"Identifying potential champions for organization {organization_id}")
        
        candidates = champion_engine.identify_potential_champions(
            organization_id=organization_id,
            employee_data=employee_data,
            criteria_type=criteria_type,
            target_count=target_count
        )
        
        return {
            "success": True,
            "organization_id": organization_id,
            "criteria_type": criteria_type,
            "candidates_found": len(candidates),
            "candidates": candidates,
            "summary": {
                "average_score": sum(c["champion_score"] for c in candidates) / len(candidates) if candidates else 0,
                "top_score": max(c["champion_score"] for c in candidates) if candidates else 0,
                "departments_represented": len(set(c["department"] for c in candidates)),
                "recommended_levels": {
                    level.value: len([c for c in candidates if c["recommended_level"] == level])
                    for level in ChampionLevel
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error identifying potential champions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profile", response_model=Dict[str, Any])
async def create_champion_profile(
    employee_data: Dict[str, Any],
    champion_assessment: Dict[str, Any]
):
    """Create comprehensive change champion profile"""
    try:
        logger.info(f"Creating champion profile for {employee_data.get('name', 'unknown')}")
        
        profile = champion_engine.create_champion_profile(
            employee_data=employee_data,
            champion_assessment=champion_assessment
        )
        
        return {
            "success": True,
            "profile": {
                "id": profile.id,
                "employee_id": profile.employee_id,
                "name": profile.name,
                "role": profile.role,
                "department": profile.department,
                "organization_id": profile.organization_id,
                "champion_level": profile.champion_level.value,
                "champion_roles": [role.value for role in profile.champion_roles],
                "capabilities": {
                    capability.value: score 
                    for capability, score in profile.capabilities.items()
                },
                "influence_network_size": len(profile.influence_network),
                "credibility_score": profile.credibility_score,
                "engagement_score": profile.engagement_score,
                "availability_score": profile.availability_score,
                "motivation_score": profile.motivation_score,
                "cultural_fit_score": profile.cultural_fit_score,
                "change_experience": profile.change_experience,
                "status": profile.status,
                "joined_date": profile.joined_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating champion profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/development-program", response_model=Dict[str, Any])
async def design_development_program(
    champions_data: List[Dict[str, Any]],
    program_objectives: List[str],
    constraints: Dict[str, Any]
):
    """Design customized development program for champions"""
    try:
        logger.info(f"Designing development program for {len(champions_data)} champions")
        
        # Convert champion data to profiles (simplified for demo)
        champions = []
        for champ_data in champions_data:
            # Create mock profile for demonstration
            capabilities = {
                ChangeCapability.CHANGE_ADVOCACY: champ_data.get("change_advocacy", 60),
                ChangeCapability.COMMUNICATION: champ_data.get("communication", 70),
                ChangeCapability.INFLUENCE_BUILDING: champ_data.get("influence_building", 65)
            }
            
            profile = ChangeChampionProfile(
                id=champ_data["id"],
                employee_id=champ_data["employee_id"],
                name=champ_data["name"],
                role=champ_data["role"],
                department=champ_data["department"],
                organization_id=champ_data["organization_id"],
                champion_level=ChampionLevel(champ_data.get("level", "developing")),
                champion_roles=[ChampionRole.ADVOCATE],
                capabilities=capabilities,
                influence_network=[],
                credibility_score=champ_data.get("credibility_score", 70),
                engagement_score=champ_data.get("engagement_score", 75),
                availability_score=champ_data.get("availability_score", 80),
                motivation_score=champ_data.get("motivation_score", 85),
                cultural_fit_score=champ_data.get("cultural_fit_score", 80),
                change_experience=[],
                training_completed=[],
                certifications=[],
                mentorship_relationships=[],
                success_metrics={}
            )
            champions.append(profile)
        
        program = champion_engine.design_development_program(
            champions=champions,
            program_objectives=program_objectives,
            constraints=constraints
        )
        
        return {
            "success": True,
            "program": {
                "id": program.id,
                "name": program.name,
                "description": program.description,
                "target_level": program.target_level.value,
                "target_roles": [role.value for role in program.target_roles],
                "duration_weeks": program.duration_weeks,
                "learning_modules": [
                    {
                        "id": module.id,
                        "title": module.title,
                        "description": module.description,
                        "target_capabilities": [cap.value for cap in module.target_capabilities],
                        "content_type": module.content_type,
                        "duration_hours": module.duration_hours,
                        "delivery_method": module.delivery_method,
                        "learning_objectives": module.learning_objectives
                    }
                    for module in program.learning_modules
                ],
                "practical_assignments": [
                    {
                        "id": assignment.id,
                        "title": assignment.title,
                        "description": assignment.description,
                        "assignment_type": assignment.assignment_type,
                        "duration_weeks": assignment.duration_weeks,
                        "deliverables": assignment.deliverables,
                        "success_metrics": assignment.success_metrics
                    }
                    for assignment in program.practical_assignments
                ],
                "mentorship_component": program.mentorship_component,
                "peer_learning_groups": program.peer_learning_groups,
                "certification_available": program.certification_available,
                "success_criteria": program.success_criteria,
                "prerequisites": program.prerequisites,
                "resources_required": program.resources_required
            }
        }
        
    except Exception as e:
        logger.error(f"Error designing development program: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/network", response_model=Dict[str, Any])
async def create_champion_network(
    champions_data: List[Dict[str, Any]],
    network_type: str,
    objectives: List[str]
):
    """Create change champion network"""
    try:
        logger.info(f"Creating {network_type} champion network")
        
        # Convert champion data to profiles (simplified for demo)
        champions = []
        for champ_data in champions_data:
            capabilities = {cap: 70 for cap in ChangeCapability}  # Default scores
            
            profile = ChangeChampionProfile(
                id=champ_data["id"],
                employee_id=champ_data["employee_id"],
                name=champ_data["name"],
                role=champ_data["role"],
                department=champ_data["department"],
                organization_id=champ_data["organization_id"],
                champion_level=ChampionLevel(champ_data.get("level", "active")),
                champion_roles=[ChampionRole.ADVOCATE, ChampionRole.FACILITATOR],
                capabilities=capabilities,
                influence_network=[],
                credibility_score=75,
                engagement_score=80,
                availability_score=85,
                motivation_score=90,
                cultural_fit_score=85,
                change_experience=[],
                training_completed=[],
                certifications=[],
                mentorship_relationships=[],
                success_metrics={}
            )
            champions.append(profile)
        
        network = champion_engine.create_champion_network(
            champions=champions,
            network_type=network_type,
            objectives=objectives
        )
        
        return {
            "success": True,
            "network": {
                "id": network.id,
                "name": network.name,
                "organization_id": network.organization_id,
                "network_type": network.network_type,
                "champion_count": len(network.champions),
                "network_lead": network.network_lead,
                "coordinators": network.coordinators,
                "coverage_areas": network.coverage_areas,
                "network_status": network.network_status.value,
                "formation_date": network.formation_date.isoformat(),
                "objectives": network.objectives,
                "success_metrics": network.success_metrics,
                "communication_channels": network.communication_channels,
                "meeting_schedule": network.meeting_schedule,
                "governance_structure": network.governance_structure
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating champion network: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/network/{network_id}/coordination-plan", response_model=Dict[str, Any])
async def plan_network_coordination(
    network_id: str,
    coordination_period: str,
    key_initiatives: List[str]
):
    """Create coordination plan for champion network"""
    try:
        logger.info(f"Planning coordination for network {network_id}")
        
        # Create mock network for demonstration
        mock_network = ChampionNetwork(
            id=network_id,
            name="Demo Champion Network",
            organization_id="demo_org",
            network_type="cross_functional",
            champions=["champ1", "champ2", "champ3"],
            network_lead="champ1",
            coordinators=["champ2"],
            coverage_areas=["Operations", "HR", "IT"],
            network_status=NetworkStatus.PERFORMING,
            formation_date=datetime.now(),
            objectives=["Support organizational transformation", "Build change capability"],
            success_metrics=["Network engagement > 80%"],
            communication_channels=["Monthly meetings", "Slack"],
            meeting_schedule="Monthly",
            governance_structure={"lead": "champ1"},
            performance_metrics={}
        )
        
        plan = champion_engine.plan_network_coordination(
            network=mock_network,
            coordination_period=coordination_period,
            key_initiatives=key_initiatives
        )
        
        return {
            "success": True,
            "coordination_plan": {
                "id": plan.id,
                "network_id": plan.network_id,
                "coordination_period": plan.coordination_period,
                "objectives": plan.objectives,
                "key_initiatives": plan.key_initiatives,
                "resource_allocation": plan.resource_allocation,
                "communication_strategy": plan.communication_strategy,
                "training_schedule": plan.training_schedule,
                "performance_targets": plan.performance_targets,
                "risk_mitigation": plan.risk_mitigation,
                "success_metrics": plan.success_metrics,
                "review_schedule": plan.review_schedule,
                "stakeholder_engagement": plan.stakeholder_engagement
            }
        }
        
    except Exception as e:
        logger.error(f"Error planning network coordination: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance", response_model=Dict[str, Any])
async def measure_champion_performance(
    champion_id: str,
    measurement_period: str,
    performance_data: Dict[str, Any]
):
    """Measure individual change champion performance"""
    try:
        logger.info(f"Measuring performance for champion {champion_id}")
        
        metrics = champion_engine.measure_champion_performance(
            champion_id=champion_id,
            measurement_period=measurement_period,
            performance_data=performance_data
        )
        
        return {
            "success": True,
            "performance_metrics": {
                "champion_id": metrics.champion_id,
                "measurement_period": metrics.measurement_period,
                "change_initiatives_supported": metrics.change_initiatives_supported,
                "training_sessions_delivered": metrics.training_sessions_delivered,
                "employees_influenced": metrics.employees_influenced,
                "resistance_cases_resolved": metrics.resistance_cases_resolved,
                "feedback_sessions_conducted": metrics.feedback_sessions_conducted,
                "network_engagement_score": metrics.network_engagement_score,
                "peer_rating": metrics.peer_rating,
                "manager_rating": metrics.manager_rating,
                "change_success_contribution": metrics.change_success_contribution,
                "knowledge_sharing_score": metrics.knowledge_sharing_score,
                "mentorship_effectiveness": metrics.mentorship_effectiveness,
                "innovation_contributions": metrics.innovation_contributions,
                "cultural_alignment_score": metrics.cultural_alignment_score,
                "overall_performance_score": metrics.overall_performance_score,
                "recognition_received": metrics.recognition_received,
                "development_areas": metrics.development_areas,
                "measurement_date": metrics.measurement_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error measuring champion performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_change_capabilities():
    """Get list of change champion capabilities and descriptions"""
    try:
        capabilities = {
            "change_advocacy": {
                "name": "Change Advocacy",
                "description": "Ability to promote and support organizational change initiatives",
                "key_behaviors": [
                    "Actively promotes change initiatives",
                    "Communicates benefits of change",
                    "Addresses concerns and objections",
                    "Builds enthusiasm for change"
                ]
            },
            "influence_building": {
                "name": "Influence Building",
                "description": "Capacity to influence others without formal authority",
                "key_behaviors": [
                    "Builds trust and credibility",
                    "Uses persuasion effectively",
                    "Leverages relationships for influence",
                    "Adapts influence style to audience"
                ]
            },
            "communication": {
                "name": "Communication",
                "description": "Effective communication across all levels and contexts",
                "key_behaviors": [
                    "Communicates clearly and concisely",
                    "Listens actively and empathetically",
                    "Adapts message to audience",
                    "Facilitates group discussions"
                ]
            },
            "training_delivery": {
                "name": "Training Delivery",
                "description": "Ability to design and deliver effective training programs",
                "key_behaviors": [
                    "Designs engaging training content",
                    "Delivers training effectively",
                    "Adapts to different learning styles",
                    "Evaluates training effectiveness"
                ]
            },
            "resistance_management": {
                "name": "Resistance Management",
                "description": "Skills in identifying and addressing change resistance",
                "key_behaviors": [
                    "Identifies resistance early",
                    "Understands root causes of resistance",
                    "Applies appropriate intervention strategies",
                    "Converts resistance into support"
                ]
            },
            "network_building": {
                "name": "Network Building",
                "description": "Ability to build and maintain professional networks",
                "key_behaviors": [
                    "Builds relationships across organization",
                    "Maintains network connections",
                    "Leverages network for change support",
                    "Facilitates network connections"
                ]
            },
            "feedback_collection": {
                "name": "Feedback Collection",
                "description": "Skills in gathering and analyzing stakeholder feedback",
                "key_behaviors": [
                    "Designs effective feedback mechanisms",
                    "Collects feedback systematically",
                    "Analyzes feedback for insights",
                    "Acts on feedback appropriately"
                ]
            },
            "coaching_mentoring": {
                "name": "Coaching & Mentoring",
                "description": "Ability to coach and mentor others through change",
                "key_behaviors": [
                    "Provides effective coaching",
                    "Mentors others in change skills",
                    "Develops change capability in others",
                    "Creates supportive learning environment"
                ]
            },
            "project_coordination": {
                "name": "Project Coordination",
                "description": "Skills in coordinating change projects and initiatives",
                "key_behaviors": [
                    "Plans and organizes change projects",
                    "Coordinates multiple stakeholders",
                    "Manages project timelines",
                    "Ensures project deliverables"
                ]
            },
            "cultural_sensitivity": {
                "name": "Cultural Sensitivity",
                "description": "Understanding and respect for organizational and cultural diversity",
                "key_behaviors": [
                    "Demonstrates cultural awareness",
                    "Adapts approach to cultural context",
                    "Promotes inclusive practices",
                    "Bridges cultural differences"
                ]
            }
        }
        
        return {
            "success": True,
            "capabilities": capabilities
        }
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/identification-criteria", response_model=Dict[str, Any])
async def get_identification_criteria():
    """Get available change champion identification criteria"""
    try:
        criteria_info = {}
        
        for criteria_type, criteria in champion_engine.identification_criteria.items():
            criteria_info[criteria_type] = {
                "name": criteria.name,
                "description": criteria.description,
                "required_capabilities": [cap.value for cap in criteria.required_capabilities],
                "minimum_scores": {cap.value: score for cap, score in criteria.minimum_scores.items()},
                "influence_requirements": criteria.influence_requirements,
                "experience_requirements": criteria.experience_requirements,
                "role_preferences": criteria.role_preferences,
                "department_coverage": criteria.department_coverage,
                "cultural_factors": criteria.cultural_factors,
                "exclusion_criteria": criteria.exclusion_criteria,
                "weight_factors": criteria.weight_factors
            }
        
        return {
            "success": True,
            "criteria": criteria_info
        }
        
    except Exception as e:
        logger.error(f"Error getting identification criteria: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/development-programs", response_model=Dict[str, Any])
async def get_development_programs():
    """Get available change champion development programs"""
    try:
        programs_info = {}
        
        for program_type, program in champion_engine.development_programs.items():
            programs_info[program_type] = {
                "name": program.name,
                "description": program.description,
                "target_level": program.target_level.value,
                "target_roles": [role.value for role in program.target_roles],
                "duration_weeks": program.duration_weeks,
                "module_count": len(program.learning_modules),
                "assignment_count": len(program.practical_assignments),
                "mentorship_component": program.mentorship_component,
                "peer_learning_groups": program.peer_learning_groups,
                "certification_available": program.certification_available,
                "prerequisites": program.prerequisites,
                "success_criteria": program.success_criteria
            }
        
        return {
            "success": True,
            "programs": programs_info
        }
        
    except Exception as e:
        logger.error(f"Error getting development programs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))