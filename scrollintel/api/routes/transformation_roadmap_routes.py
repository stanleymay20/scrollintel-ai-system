"""
Transformation Roadmap API Routes

API endpoints for transformation roadmap planning, progress tracking, and optimization.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from ...engines.transformation_roadmap_engine import TransformationRoadmapEngine
from ...models.transformation_roadmap_models import (
    RoadmapPlanningRequest, RoadmapPlanningResult, TransformationRoadmap,
    ProgressUpdate, RoadmapOptimization, RoadmapAdjustment
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/transformation-roadmap", tags=["transformation-roadmap"])

# Initialize engine
roadmap_engine = TransformationRoadmapEngine()


@router.post("/create", response_model=RoadmapPlanningResult)
async def create_transformation_roadmap(request: RoadmapPlanningRequest):
    """
    Create a comprehensive transformation roadmap
    
    Args:
        request: Roadmap planning request with requirements
        
    Returns:
        Complete roadmap planning result
    """
    try:
        logger.info(f"Creating transformation roadmap for organization {request.organization_id}")
        
        result = roadmap_engine.create_transformation_roadmap(request)
        
        logger.info(f"Successfully created transformation roadmap {result.roadmap.id}")
        return result
        
    except Exception as e:
        logger.error(f"Error creating transformation roadmap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/roadmap/{roadmap_id}/progress", response_model=TransformationRoadmap)
async def update_milestone_progress(
    roadmap_id: str,
    progress_updates: List[ProgressUpdate]
):
    """
    Update progress on roadmap milestones
    
    Args:
        roadmap_id: ID of the roadmap
        progress_updates: List of progress updates
        
    Returns:
        Updated roadmap with progress
    """
    try:
        logger.info(f"Updating progress for roadmap {roadmap_id}")
        
        updated_roadmap = roadmap_engine.track_milestone_progress(roadmap_id, progress_updates)
        
        logger.info(f"Successfully updated progress for roadmap {roadmap_id}")
        return updated_roadmap
        
    except Exception as e:
        logger.error(f"Error updating milestone progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/roadmap/{roadmap_id}/optimize", response_model=List[RoadmapOptimization])
async def optimize_roadmap(
    roadmap_id: str,
    performance_data: Dict[str, Any]
):
    """
    Optimize roadmap based on performance data and feedback
    
    Args:
        roadmap_id: ID of the roadmap to optimize
        performance_data: Performance and feedback data
        
    Returns:
        List of optimization recommendations
    """
    try:
        logger.info(f"Optimizing roadmap {roadmap_id}")
        
        # In a real implementation, would load roadmap from database
        roadmap = roadmap_engine._get_roadmap_by_id(roadmap_id)
        
        optimizations = roadmap_engine.optimize_roadmap(roadmap, performance_data)
        
        logger.info(f"Generated {len(optimizations)} optimization recommendations")
        return optimizations
        
    except Exception as e:
        logger.error(f"Error optimizing roadmap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/roadmap/{roadmap_id}/adjust", response_model=TransformationRoadmap)
async def adjust_roadmap(
    roadmap_id: str,
    adjustments: List[RoadmapAdjustment]
):
    """
    Apply adjustments to roadmap based on feedback and changing conditions
    
    Args:
        roadmap_id: ID of the roadmap to adjust
        adjustments: List of adjustments to apply
        
    Returns:
        Updated roadmap
    """
    try:
        logger.info(f"Applying adjustments to roadmap {roadmap_id}")
        
        # In a real implementation, would load roadmap from database
        roadmap = roadmap_engine._get_roadmap_by_id(roadmap_id)
        
        updated_roadmap = roadmap_engine.adjust_roadmap(roadmap, adjustments)
        
        logger.info(f"Successfully applied adjustments to roadmap {roadmap_id}")
        return updated_roadmap
        
    except Exception as e:
        logger.error(f"Error adjusting roadmap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roadmap/{roadmap_id}", response_model=TransformationRoadmap)
async def get_roadmap(roadmap_id: str):
    """
    Retrieve a transformation roadmap by ID
    
    Args:
        roadmap_id: ID of the roadmap to retrieve
        
    Returns:
        Transformation roadmap details
    """
    try:
        roadmap = roadmap_engine._get_roadmap_by_id(roadmap_id)
        
        if not roadmap:
            raise HTTPException(status_code=404, detail="Roadmap not found")
        
        return roadmap
        
    except Exception as e:
        logger.error(f"Error retrieving roadmap {roadmap_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{org_id}/roadmaps", response_model=List[TransformationRoadmap])
async def get_organization_roadmaps(org_id: str):
    """
    Retrieve all transformation roadmaps for an organization
    
    Args:
        org_id: Organization ID
        
    Returns:
        List of transformation roadmaps
    """
    try:
        # In a real implementation, this would query a database
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Error retrieving roadmaps for organization {org_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roadmap/{roadmap_id}/critical-path")
async def get_critical_path(roadmap_id: str):
    """
    Get the critical path for a roadmap
    
    Args:
        roadmap_id: ID of the roadmap
        
    Returns:
        Critical path information
    """
    try:
        roadmap = roadmap_engine._get_roadmap_by_id(roadmap_id)
        critical_path = roadmap_engine._calculate_critical_path(roadmap)
        
        # Get milestone details for critical path
        critical_milestones = []
        for milestone_id in critical_path:
            milestone = roadmap_engine._find_milestone_by_id(roadmap, milestone_id)
            if milestone:
                critical_milestones.append({
                    "id": milestone.id,
                    "name": milestone.name,
                    "target_date": milestone.target_date,
                    "status": milestone.status,
                    "progress_percentage": milestone.progress_percentage
                })
        
        return {
            "roadmap_id": roadmap_id,
            "critical_path_length": len(critical_path),
            "critical_milestones": critical_milestones,
            "estimated_completion": roadmap.target_completion_date,
            "risk_factors": [
                "Critical path delays will impact overall timeline",
                "Resource constraints on critical milestones",
                "Dependencies between critical milestones"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting critical path for roadmap {roadmap_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roadmap/{roadmap_id}/progress-report")
async def get_progress_report(roadmap_id: str):
    """
    Get comprehensive progress report for a roadmap
    
    Args:
        roadmap_id: ID of the roadmap
        
    Returns:
        Detailed progress report
    """
    try:
        roadmap = roadmap_engine._get_roadmap_by_id(roadmap_id)
        
        # Calculate progress statistics
        total_milestones = len(roadmap.milestones)
        completed_milestones = len([m for m in roadmap.milestones if m.status.value == "completed"])
        in_progress_milestones = len([m for m in roadmap.milestones if m.status.value == "in_progress"])
        delayed_milestones = len([m for m in roadmap.milestones if m.status.value == "delayed"])
        at_risk_milestones = len([m for m in roadmap.milestones if m.status.value == "at_risk"])
        
        # Calculate phase progress
        phase_progress = {}
        for phase in roadmap.phases:
            phase_milestones = [m for m in roadmap.milestones 
                              if any(phase.phase.value in m.name.lower() for m in roadmap.milestones)]
            if phase_milestones:
                phase_progress[phase.phase.value] = {
                    "total_milestones": len(phase_milestones),
                    "average_progress": sum(m.progress_percentage for m in phase_milestones) / len(phase_milestones)
                }
        
        return {
            "roadmap_id": roadmap_id,
            "overall_progress": roadmap.overall_progress,
            "current_phase": roadmap.current_phase.value,
            "milestone_statistics": {
                "total": total_milestones,
                "completed": completed_milestones,
                "in_progress": in_progress_milestones,
                "delayed": delayed_milestones,
                "at_risk": at_risk_milestones
            },
            "phase_progress": phase_progress,
            "timeline_status": {
                "start_date": roadmap.start_date,
                "target_completion": roadmap.target_completion_date,
                "days_remaining": (roadmap.target_completion_date - roadmap.created_date).days,
                "on_track": delayed_milestones == 0
            },
            "key_insights": [
                f"{completed_milestones}/{total_milestones} milestones completed",
                f"Currently in {roadmap.current_phase.value} phase",
                f"{delayed_milestones} milestones delayed" if delayed_milestones > 0 else "All milestones on track",
                f"{at_risk_milestones} milestones at risk" if at_risk_milestones > 0 else "No milestones at risk"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating progress report for roadmap {roadmap_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/roadmap-phases")
async def get_roadmap_phase_templates():
    """
    Get available roadmap phase templates
    
    Returns:
        List of roadmap phase templates
    """
    try:
        templates = {
            "standard_transformation": {
                "name": "Standard Cultural Transformation",
                "description": "Comprehensive 5-phase transformation approach",
                "phases": [
                    {
                        "name": "Preparation & Foundation",
                        "duration_days": 30,
                        "key_activities": [
                            "Culture assessment",
                            "Stakeholder alignment",
                            "Resource planning",
                            "Communication strategy"
                        ]
                    },
                    {
                        "name": "Launch & Awareness",
                        "duration_days": 45,
                        "key_activities": [
                            "Vision communication",
                            "Initial training",
                            "Quick wins",
                            "Feedback systems"
                        ]
                    },
                    {
                        "name": "Implementation & Integration",
                        "duration_days": 90,
                        "key_activities": [
                            "Behavior change programs",
                            "Process integration",
                            "System updates",
                            "Progress monitoring"
                        ]
                    },
                    {
                        "name": "Reinforcement & Optimization",
                        "duration_days": 60,
                        "key_activities": [
                            "Reinforcement programs",
                            "Optimization initiatives",
                            "Gap analysis",
                            "Sustainability planning"
                        ]
                    },
                    {
                        "name": "Evaluation & Sustainability",
                        "duration_days": 30,
                        "key_activities": [
                            "Success measurement",
                            "Sustainability assessment",
                            "Lessons learned",
                            "Future planning"
                        ]
                    }
                ],
                "total_duration_days": 255,
                "success_factors": [
                    "Strong leadership commitment",
                    "Clear communication",
                    "Employee engagement",
                    "Continuous feedback"
                ]
            },
            "agile_transformation": {
                "name": "Agile Cultural Transformation",
                "description": "Iterative approach with shorter cycles",
                "phases": [
                    {
                        "name": "Sprint 0 - Foundation",
                        "duration_days": 14,
                        "key_activities": [
                            "Rapid assessment",
                            "Team formation",
                            "Initial planning"
                        ]
                    },
                    {
                        "name": "Sprint 1-3 - Core Changes",
                        "duration_days": 42,
                        "key_activities": [
                            "Iterative implementation",
                            "Regular retrospectives",
                            "Continuous adjustment"
                        ]
                    },
                    {
                        "name": "Sprint 4-6 - Reinforcement",
                        "duration_days": 42,
                        "key_activities": [
                            "Habit formation",
                            "Process optimization",
                            "Scaling successful practices"
                        ]
                    }
                ],
                "total_duration_days": 98,
                "success_factors": [
                    "Rapid iteration",
                    "Continuous feedback",
                    "Adaptive planning",
                    "Quick wins"
                ]
            }
        }
        
        return templates
        
    except Exception as e:
        logger.error(f"Error retrieving roadmap phase templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/milestone-types")
async def get_milestone_type_templates():
    """
    Get available milestone type templates
    
    Returns:
        List of milestone type templates
    """
    try:
        templates = {
            "foundation": {
                "name": "Foundation Milestones",
                "description": "Milestones that establish the foundation for transformation",
                "examples": [
                    {
                        "name": "Culture Assessment Complete",
                        "description": "Comprehensive assessment of current culture",
                        "typical_duration_days": 10,
                        "success_criteria": [
                            "Assessment methodology defined",
                            "Data collection completed",
                            "Analysis and insights generated",
                            "Baseline metrics established"
                        ]
                    },
                    {
                        "name": "Transformation Team Established",
                        "description": "Formation and training of transformation team",
                        "typical_duration_days": 15,
                        "success_criteria": [
                            "Team members identified and committed",
                            "Roles and responsibilities defined",
                            "Training completed",
                            "Team charter approved"
                        ]
                    }
                ]
            },
            "awareness": {
                "name": "Awareness Milestones",
                "description": "Milestones focused on building awareness and understanding",
                "examples": [
                    {
                        "name": "Vision Launch Complete",
                        "description": "Successful launch of cultural vision",
                        "typical_duration_days": 7,
                        "success_criteria": [
                            "Vision communicated to all stakeholders",
                            "Awareness levels measured and achieved",
                            "Initial feedback collected",
                            "Engagement metrics established"
                        ]
                    }
                ]
            },
            "adoption": {
                "name": "Adoption Milestones",
                "description": "Milestones focused on behavior adoption and change",
                "examples": [
                    {
                        "name": "Behavior Change Programs Active",
                        "description": "Launch and activation of behavior change initiatives",
                        "typical_duration_days": 30,
                        "success_criteria": [
                            "Programs designed and launched",
                            "Participation rates achieved",
                            "Behavior changes observed",
                            "Progress tracking systems active"
                        ]
                    }
                ]
            }
        }
        
        return templates
        
    except Exception as e:
        logger.error(f"Error retrieving milestone type templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))