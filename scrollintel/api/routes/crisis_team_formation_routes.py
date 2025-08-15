"""
Crisis Team Formation API Routes

FastAPI routes for crisis team formation, optimization, and management.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.crisis_team_formation_engine import CrisisTeamFormationEngine
from ...models.team_coordination_models import (
    CrisisTeam, TeamFormationRequest, Person, SkillMatch,
    TeamRole, AvailabilityStatus, RoleAssignment
)
from ...core.auth import get_current_user
from ...core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/crisis-team-formation", tags=["Crisis Team Formation"])

# Global engine instance
team_formation_engine = CrisisTeamFormationEngine()


@router.post("/teams/form", response_model=Dict[str, Any])
async def form_crisis_team(
    request: TeamFormationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Form a new crisis response team"""
    try:
        logger.info(f"Forming crisis team for crisis {request.crisis_id}")
        
        # Form the crisis team
        crisis_team = await team_formation_engine.form_crisis_team(request)
        
        # Schedule team activation in background
        background_tasks.add_task(activate_team_communication, crisis_team.id)
        
        return {
            "success": True,
            "team_id": crisis_team.id,
            "team": {
                "id": crisis_team.id,
                "name": crisis_team.team_name,
                "crisis_id": crisis_team.crisis_id,
                "crisis_type": crisis_team.crisis_type,
                "team_lead_id": crisis_team.team_lead_id,
                "member_count": len(crisis_team.members),
                "members": crisis_team.members,
                "status": crisis_team.team_status,
                "formation_time": crisis_team.formation_time.isoformat(),
                "communication_channels": crisis_team.communication_channels
            },
            "role_assignments": [
                {
                    "person_id": assignment.person_id,
                    "role": assignment.role.value,
                    "confidence": assignment.assignment_confidence,
                    "responsibilities": assignment.responsibilities,
                    "rationale": assignment.assignment_rationale
                }
                for assignment in crisis_team.role_assignments
            ]
        }
        
    except Exception as e:
        logger.error(f"Error forming crisis team: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to form crisis team: {str(e)}")


@router.get("/teams/{team_id}", response_model=Dict[str, Any])
async def get_crisis_team(
    team_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get crisis team details"""
    try:
        team = team_formation_engine.get_team_by_id(team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Crisis team not found")
        
        return {
            "success": True,
            "team": {
                "id": team.id,
                "name": team.team_name,
                "crisis_id": team.crisis_id,
                "crisis_type": team.crisis_type,
                "team_lead_id": team.team_lead_id,
                "members": team.members,
                "status": team.team_status,
                "formation_time": team.formation_time.isoformat(),
                "activation_time": team.activation_time.isoformat() if team.activation_time else None,
                "communication_channels": team.communication_channels,
                "escalation_contacts": team.escalation_contacts,
                "performance_metrics": team.performance_metrics
            },
            "role_assignments": [
                {
                    "person_id": assignment.person_id,
                    "role": assignment.role.value,
                    "confidence": assignment.assignment_confidence,
                    "responsibilities": assignment.responsibilities,
                    "required_skills": assignment.required_skills,
                    "backup_person_id": assignment.backup_person_id,
                    "rationale": assignment.assignment_rationale,
                    "assigned_at": assignment.assigned_at.isoformat()
                }
                for assignment in team.role_assignments
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving crisis team {team_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve crisis team: {str(e)}")


@router.post("/teams/{team_id}/optimize", response_model=Dict[str, Any])
async def optimize_team_composition(
    team_id: str,
    crisis_type: str,
    current_user: Dict = Depends(get_current_user)
):
    """Optimize existing team composition"""
    try:
        logger.info(f"Optimizing team composition for team {team_id}")
        
        optimized_team = await team_formation_engine.optimize_team_composition(team_id, crisis_type)
        
        return {
            "success": True,
            "message": "Team composition optimized successfully",
            "team_id": optimized_team.id,
            "optimizations_applied": True,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing team {team_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize team: {str(e)}")


@router.post("/skills/match", response_model=Dict[str, Any])
async def match_skills_to_availability(
    required_skills: List[str],
    current_user: Dict = Depends(get_current_user)
):
    """Match required skills to available personnel"""
    try:
        logger.info(f"Matching skills: {required_skills}")
        
        skill_matches = await team_formation_engine.match_skills_to_availability(required_skills)
        
        return {
            "success": True,
            "matches": [
                {
                    "person_id": match.person_id,
                    "role": match.role.value,
                    "skill_match_score": match.skill_match_score,
                    "experience_match_score": match.experience_match_score,
                    "availability_score": match.availability_score,
                    "overall_match_score": match.overall_match_score,
                    "missing_skills": match.missing_skills,
                    "strengths": match.strengths,
                    "rationale": match.match_rationale
                }
                for match in skill_matches[:20]  # Top 20 matches
            ],
            "total_matches": len(skill_matches)
        }
        
    except Exception as e:
        logger.error(f"Error matching skills: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to match skills: {str(e)}")


@router.post("/personnel/add", response_model=Dict[str, Any])
async def add_personnel(
    person_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Add new personnel to the registry"""
    try:
        # Create Person object from data
        person = Person(
            name=person_data.get("name", ""),
            email=person_data.get("email", ""),
            phone=person_data.get("phone", ""),
            department=person_data.get("department", ""),
            title=person_data.get("title", ""),
            timezone=person_data.get("timezone", "UTC"),
            languages=person_data.get("languages", [])
        )
        
        # Add skills if provided
        if "skills" in person_data:
            from ...models.team_coordination_models import Skill, SkillLevel
            for skill_data in person_data["skills"]:
                skill = Skill(
                    name=skill_data["name"],
                    level=SkillLevel(skill_data["level"]),
                    years_experience=skill_data.get("years_experience", 0),
                    certifications=skill_data.get("certifications", [])
                )
                person.skills.append(skill)
        
        # Add to registry
        team_formation_engine.add_person(person)
        
        logger.info(f"Added personnel: {person.name}")
        
        return {
            "success": True,
            "message": "Personnel added successfully",
            "person_id": person.id,
            "name": person.name
        }
        
    except Exception as e:
        logger.error(f"Error adding personnel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add personnel: {str(e)}")


@router.put("/personnel/{person_id}/availability", response_model=Dict[str, Any])
async def update_personnel_availability(
    person_id: str,
    availability_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Update personnel availability status"""
    try:
        status = AvailabilityStatus(availability_data["status"])
        team_formation_engine.update_person_availability(person_id, status)
        
        logger.info(f"Updated availability for person {person_id} to {status.value}")
        
        return {
            "success": True,
            "message": "Availability updated successfully",
            "person_id": person_id,
            "new_status": status.value,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid availability status: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating availability: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update availability: {str(e)}")


@router.post("/teams/{team_id}/activate", response_model=Dict[str, Any])
async def activate_crisis_team(
    team_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Activate crisis team for response"""
    try:
        team = team_formation_engine.get_team_by_id(team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Crisis team not found")
        
        # Update team status
        team.team_status = "active"
        team.activation_time = datetime.utcnow()
        
        logger.info(f"Activated crisis team {team_id}")
        
        return {
            "success": True,
            "message": "Crisis team activated successfully",
            "team_id": team_id,
            "activation_time": team.activation_time.isoformat(),
            "status": team.team_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating team {team_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to activate team: {str(e)}")


@router.post("/teams/{team_id}/deactivate", response_model=Dict[str, Any])
async def deactivate_crisis_team(
    team_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Deactivate crisis team"""
    try:
        team_formation_engine.deactivate_team(team_id)
        
        logger.info(f"Deactivated crisis team {team_id}")
        
        return {
            "success": True,
            "message": "Crisis team deactivated successfully",
            "team_id": team_id,
            "deactivation_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error deactivating team {team_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to deactivate team: {str(e)}")


@router.get("/teams", response_model=Dict[str, Any])
async def list_active_teams(
    current_user: Dict = Depends(get_current_user)
):
    """List all active crisis teams"""
    try:
        active_teams = []
        for team in team_formation_engine.active_teams.values():
            if team.team_status in ["forming", "active", "standby"]:
                active_teams.append({
                    "id": team.id,
                    "name": team.team_name,
                    "crisis_id": team.crisis_id,
                    "crisis_type": team.crisis_type,
                    "status": team.team_status,
                    "member_count": len(team.members),
                    "formation_time": team.formation_time.isoformat(),
                    "activation_time": team.activation_time.isoformat() if team.activation_time else None
                })
        
        return {
            "success": True,
            "teams": active_teams,
            "total_count": len(active_teams)
        }
        
    except Exception as e:
        logger.error(f"Error listing teams: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list teams: {str(e)}")


@router.get("/roles", response_model=Dict[str, Any])
async def get_available_roles():
    """Get list of available team roles"""
    try:
        roles = [
            {
                "value": role.value,
                "name": role.value.replace("_", " ").title(),
                "description": f"Role responsible for {role.value.replace('_', ' ')}"
            }
            for role in TeamRole
        ]
        
        return {
            "success": True,
            "roles": roles
        }
        
    except Exception as e:
        logger.error(f"Error getting roles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get roles: {str(e)}")


async def activate_team_communication(team_id: str):
    """Background task to activate team communication channels"""
    try:
        team = team_formation_engine.get_team_by_id(team_id)
        if team:
            # Simulate communication channel setup
            logger.info(f"Setting up communication channels for team {team_id}")
            # In real implementation, this would:
            # - Create Slack/Teams channels
            # - Setup video conference rooms
            # - Initialize status dashboards
            # - Send team formation notifications
            
    except Exception as e:
        logger.error(f"Error setting up communication for team {team_id}: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for crisis team formation service"""
    return {
        "status": "healthy",
        "service": "crisis_team_formation",
        "timestamp": datetime.utcnow().isoformat(),
        "active_teams": len(team_formation_engine.active_teams),
        "personnel_count": len(team_formation_engine.personnel_registry)
    }