"""
API routes for Role Assignment Engine
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from scrollintel.engines.role_assignment_engine import (
    RoleAssignmentEngine, Person, PersonSkill, RoleType, SkillLevel,
    AssignmentResult, RoleAssignment
)

router = APIRouter(prefix="/api/v1/role-assignment", tags=["role-assignment"])

# Initialize the engine
role_assignment_engine = RoleAssignmentEngine()


# Pydantic models for API
class PersonSkillModel(BaseModel):
    skill_name: str
    level: int = Field(..., ge=1, le=5, description="Skill level from 1 (Novice) to 5 (Master)")
    years_experience: int = Field(..., ge=0)
    recent_performance: float = Field(..., ge=0.0, le=1.0)
    crisis_experience: bool = False


class PersonModel(BaseModel):
    id: str
    name: str
    current_availability: float = Field(..., ge=0.0, le=1.0)
    skills: List[PersonSkillModel]
    preferred_roles: List[str]
    stress_tolerance: float = Field(..., ge=0.0, le=1.0)
    leadership_experience: int = Field(..., ge=0)
    crisis_history: List[str] = []
    current_workload: float = Field(0.0, ge=0.0, le=1.0)


class RoleAssignmentRequest(BaseModel):
    crisis_id: str
    available_people: List[PersonModel]
    required_roles: List[str]
    crisis_severity: float = Field(0.5, ge=0.0, le=1.0)


class RoleAssignmentResponse(BaseModel):
    assignments: List[Dict]
    unassigned_roles: List[str]
    assignment_quality_score: float
    recommendations: List[str]
    backup_assignments: Dict[str, List[str]]


class RoleClarificationRequest(BaseModel):
    assignment_id: str
    person_id: str
    role_type: str


class RoleConfirmationRequest(BaseModel):
    assignment_id: str
    person_confirmation: bool


def convert_person_model(person_model: PersonModel) -> Person:
    """Convert Pydantic model to engine model"""
    skills = [
        PersonSkill(
            skill_name=skill.skill_name,
            level=SkillLevel(skill.level),
            years_experience=skill.years_experience,
            recent_performance=skill.recent_performance,
            crisis_experience=skill.crisis_experience
        )
        for skill in person_model.skills
    ]
    
    preferred_roles = [RoleType(role) for role in person_model.preferred_roles if role in [r.value for r in RoleType]]
    
    return Person(
        id=person_model.id,
        name=person_model.name,
        current_availability=person_model.current_availability,
        skills=skills,
        preferred_roles=preferred_roles,
        stress_tolerance=person_model.stress_tolerance,
        leadership_experience=person_model.leadership_experience,
        crisis_history=person_model.crisis_history,
        current_workload=person_model.current_workload
    )


@router.post("/assign", response_model=RoleAssignmentResponse)
async def assign_roles(request: RoleAssignmentRequest):
    """
    Assign crisis roles to available people based on their strengths and role requirements
    """
    try:
        # Convert Pydantic models to engine models
        people = [convert_person_model(person) for person in request.available_people]
        
        # Convert role strings to RoleType enums
        required_roles = []
        for role_str in request.required_roles:
            try:
                required_roles.append(RoleType(role_str))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid role type: {role_str}"
                )
        
        # Perform role assignment
        result = role_assignment_engine.assign_roles(
            crisis_id=request.crisis_id,
            available_people=people,
            required_roles=required_roles,
            crisis_severity=request.crisis_severity
        )
        
        # Convert result to response format
        assignments_data = []
        for assignment in result.assignments:
            assignments_data.append({
                "person_id": assignment.person_id,
                "role_type": assignment.role_type.value,
                "assignment_confidence": assignment.assignment_confidence,
                "responsibilities": assignment.responsibilities,
                "reporting_structure": assignment.reporting_structure,
                "assignment_time": assignment.assignment_time.isoformat()
            })
        
        backup_assignments_data = {
            role.value: person_ids 
            for role, person_ids in result.backup_assignments.items()
        }
        
        return RoleAssignmentResponse(
            assignments=assignments_data,
            unassigned_roles=[role.value for role in result.unassigned_roles],
            assignment_quality_score=result.assignment_quality_score,
            recommendations=result.recommendations,
            backup_assignments=backup_assignments_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Role assignment failed: {str(e)}")


@router.get("/roles/available")
async def get_available_roles():
    """
    Get list of available crisis role types
    """
    return {
        "roles": [
            {
                "type": role.value,
                "name": role.value.replace('_', ' ').title(),
                "description": f"Crisis role: {role.value.replace('_', ' ')}"
            }
            for role in RoleType
        ]
    }


@router.post("/clarify")
async def get_role_clarity_communication(request: RoleClarificationRequest):
    """
    Get clear role communication and responsibilities for an assignment
    """
    try:
        # Create a mock assignment for demonstration
        # In real implementation, would lookup actual assignment
        role_type = RoleType(request.role_type)
        
        mock_assignment = RoleAssignment(
            person_id=request.person_id,
            role_type=role_type,
            assignment_confidence=0.85,
            responsibilities=[],
            reporting_structure={},
            assignment_time=datetime.now()
        )
        
        clarity_info = role_assignment_engine.get_role_clarity_communication(mock_assignment)
        
        return {
            "assignment_id": request.assignment_id,
            "clarity_communication": clarity_info
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid role type: {request.role_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get role clarity: {str(e)}")


@router.post("/confirm")
async def confirm_role_assignment(request: RoleConfirmationRequest):
    """
    Confirm role assignment with the assigned person
    """
    try:
        confirmation_result = role_assignment_engine.confirm_role_assignment(
            assignment_id=request.assignment_id,
            person_confirmation=request.person_confirmation
        )
        
        return {
            "assignment_id": request.assignment_id,
            "confirmation_result": confirmation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to confirm assignment: {str(e)}")


@router.get("/assignments/history")
async def get_assignment_history():
    """
    Get history of role assignments
    """
    try:
        history_data = []
        for assignment in role_assignment_engine.assignment_history:
            history_data.append({
                "person_id": assignment.person_id,
                "role_type": assignment.role_type.value,
                "assignment_confidence": assignment.assignment_confidence,
                "assignment_time": assignment.assignment_time.isoformat(),
                "responsibilities_count": len(assignment.responsibilities)
            })
        
        return {
            "assignment_history": history_data,
            "total_assignments": len(history_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assignment history: {str(e)}")


@router.get("/optimization/recommendations")
async def get_optimization_recommendations():
    """
    Get recommendations for improving role assignment processes
    """
    return {
        "recommendations": [
            "Maintain updated skill assessments for all personnel",
            "Conduct regular crisis simulation exercises",
            "Cross-train personnel in multiple crisis roles",
            "Establish clear role succession plans",
            "Monitor and improve assignment quality metrics",
            "Develop role-specific performance benchmarks",
            "Create feedback loops for assignment effectiveness"
        ],
        "best_practices": [
            "Assign roles based on demonstrated competencies",
            "Ensure clear communication of responsibilities",
            "Provide backup assignments for critical roles",
            "Monitor workload distribution during crisis",
            "Maintain role flexibility for changing situations"
        ]
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for role assignment service"""
    return {
        "status": "healthy",
        "service": "role_assignment_engine",
        "timestamp": datetime.now().isoformat(),
        "available_roles": len(RoleType),
        "assignment_history_count": len(role_assignment_engine.assignment_history)
    }