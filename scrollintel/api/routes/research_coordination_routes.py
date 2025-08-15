"""
Research Coordination API Routes for Autonomous Innovation Lab

This module provides REST API endpoints for autonomous research project management,
milestone tracking, resource coordination, and research collaboration.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.research_project_manager import ResearchProjectManager
from ...engines.research_collaboration_system import ResearchCollaborationSystem
from ...models.research_coordination_models import (
    ResearchProject, ResearchMilestone, ResearchResource, ResearchCollaboration,
    KnowledgeAsset, ResearchSynergy, ProjectStatus, MilestoneStatus, CollaborationType
)
from ...models.research_coordination_models import ResearchTopic, Hypothesis
from ...core.error_handling import handle_api_error

# Initialize router
router = APIRouter(prefix="/api/v1/research-coordination", tags=["research-coordination"])
logger = logging.getLogger(__name__)

# Initialize engines
project_manager = ResearchProjectManager()
collaboration_system = ResearchCollaborationSystem()


@router.post("/projects", response_model=Dict[str, Any])
async def create_research_project(
    research_topic: Dict[str, Any],
    hypotheses: List[Dict[str, Any]],
    project_type: str = "basic_research",
    priority: int = 5
):
    """
    Create a new autonomous research project
    
    Args:
        research_topic: Research topic information
        hypotheses: List of research hypotheses
        project_type: Type of research project template
        priority: Project priority (1-10)
        
    Returns:
        Created research project information
    """
    try:
        # Convert input to models
        topic = ResearchTopic(**research_topic)
        hyp_list = [Hypothesis(**h) for h in hypotheses]
        
        # Create project
        project = await project_manager.create_research_project(
            research_topic=topic,
            hypotheses=hyp_list,
            project_type=project_type,
            priority=priority
        )
        
        return {
            "success": True,
            "project_id": project.id,
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "status": project.status.value,
                "priority": project.priority,
                "research_domain": project.research_domain,
                "objectives": project.objectives,
                "hypotheses": project.hypotheses,
                "methodology": project.methodology,
                "planned_start": project.planned_start.isoformat() if project.planned_start else None,
                "planned_end": project.planned_end.isoformat() if project.planned_end else None,
                "progress_percentage": project.progress_percentage,
                "milestones_count": len(project.milestones),
                "allocated_resources_count": len(project.allocated_resources)
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating research project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}", response_model=Dict[str, Any])
async def get_project_status(project_id: str):
    """
    Get comprehensive project status
    
    Args:
        project_id: Project identifier
        
    Returns:
        Project status information
    """
    try:
        status = await project_manager.get_project_status(project_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {
            "success": True,
            "project_status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/projects/{project_id}/milestones/{milestone_id}/progress")
async def update_milestone_progress(
    project_id: str,
    milestone_id: str,
    progress: float,
    status: Optional[str] = None
):
    """
    Update milestone progress and status
    
    Args:
        project_id: Project identifier
        milestone_id: Milestone identifier
        progress: Progress percentage (0-100)
        status: Optional milestone status
        
    Returns:
        Update result
    """
    try:
        # Convert status string to enum if provided
        milestone_status = None
        if status:
            try:
                milestone_status = MilestoneStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid milestone status: {status}")
        
        # Update milestone
        success = await project_manager.update_milestone_progress(
            project_id=project_id,
            milestone_id=milestone_id,
            progress=progress,
            status=milestone_status
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Project or milestone not found")
        
        return {
            "success": True,
            "message": f"Milestone progress updated to {progress}%"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating milestone progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/optimize-resources")
async def optimize_project_resources(project_id: str):
    """
    Optimize resource allocation for project
    
    Args:
        project_id: Project identifier
        
    Returns:
        Optimization results
    """
    try:
        result = await project_manager.optimize_resource_allocation(project_id)
        
        if "error" in result:
            if result["error"] == "Project not found":
                raise HTTPException(status_code=404, detail="Project not found")
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "optimization_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing project resources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects", response_model=Dict[str, Any])
async def list_projects(
    status: Optional[str] = Query(None, description="Filter by project status"),
    domain: Optional[str] = Query(None, description="Filter by research domain"),
    priority_min: Optional[int] = Query(None, description="Minimum priority"),
    priority_max: Optional[int] = Query(None, description="Maximum priority")
):
    """
    List research projects with optional filters
    
    Args:
        status: Filter by project status
        domain: Filter by research domain
        priority_min: Minimum priority filter
        priority_max: Maximum priority filter
        
    Returns:
        List of projects
    """
    try:
        projects = list(project_manager.active_projects.values())
        
        # Apply filters
        if status:
            try:
                status_enum = ProjectStatus(status)
                projects = [p for p in projects if p.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid project status: {status}")
        
        if domain:
            projects = [p for p in projects if domain.lower() in p.research_domain.lower()]
        
        if priority_min is not None:
            projects = [p for p in projects if p.priority >= priority_min]
        
        if priority_max is not None:
            projects = [p for p in projects if p.priority <= priority_max]
        
        # Convert to response format
        project_list = []
        for project in projects:
            project_list.append({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "status": project.status.value,
                "priority": project.priority,
                "research_domain": project.research_domain,
                "progress_percentage": project.progress_percentage,
                "planned_start": project.planned_start.isoformat() if project.planned_start else None,
                "planned_end": project.planned_end.isoformat() if project.planned_end else None,
                "milestones_count": len(project.milestones),
                "active_milestones": len(project.get_active_milestones()),
                "overdue_milestones": len(project.get_overdue_milestones())
            })
        
        return {
            "success": True,
            "projects": project_list,
            "total_count": len(project_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing projects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collaborations/identify")
async def identify_collaboration_opportunities(
    project_ids: Optional[List[str]] = None,
    min_synergy_score: float = 0.6
):
    """
    Identify collaboration opportunities between research projects
    
    Args:
        project_ids: Optional list of specific project IDs to analyze
        min_synergy_score: Minimum synergy score threshold
        
    Returns:
        List of identified collaboration opportunities
    """
    try:
        # Get projects to analyze
        if project_ids:
            projects = [project_manager.active_projects[pid] for pid in project_ids 
                       if pid in project_manager.active_projects]
        else:
            projects = list(project_manager.active_projects.values())
        
        if len(projects) < 2:
            return {
                "success": True,
                "synergies": [],
                "message": "Need at least 2 projects to identify collaborations"
            }
        
        # Identify synergies
        synergies = await collaboration_system.identify_collaboration_opportunities(
            projects=projects,
            min_synergy_score=min_synergy_score
        )
        
        # Convert to response format
        synergy_list = []
        for synergy in synergies:
            synergy_list.append({
                "id": synergy.id,
                "project_ids": synergy.project_ids,
                "synergy_type": synergy.synergy_type,
                "overall_score": synergy.overall_score,
                "potential_score": synergy.potential_score,
                "feasibility_score": synergy.feasibility_score,
                "impact_score": synergy.impact_score,
                "complementary_strengths": synergy.complementary_strengths,
                "shared_challenges": synergy.shared_challenges,
                "collaboration_opportunities": synergy.collaboration_opportunities,
                "recommended_actions": synergy.recommended_actions,
                "estimated_benefits": synergy.estimated_benefits,
                "implementation_complexity": synergy.implementation_complexity,
                "is_exploited": synergy.is_exploited
            })
        
        return {
            "success": True,
            "synergies": synergy_list,
            "total_opportunities": len(synergy_list)
        }
        
    except Exception as e:
        logger.error(f"Error identifying collaboration opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collaborations")
async def create_collaboration(
    synergy_id: str,
    collaboration_type: str = "knowledge_sharing"
):
    """
    Create a research collaboration based on identified synergy
    
    Args:
        synergy_id: Identified synergy ID
        collaboration_type: Type of collaboration to establish
        
    Returns:
        Created collaboration information
    """
    try:
        # Get synergy
        if synergy_id not in collaboration_system.identified_synergies:
            raise HTTPException(status_code=404, detail="Synergy not found")
        
        synergy = collaboration_system.identified_synergies[synergy_id]
        
        # Convert collaboration type
        try:
            collab_type = CollaborationType(collaboration_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid collaboration type: {collaboration_type}")
        
        # Create collaboration
        collaboration = await collaboration_system.create_collaboration(
            synergy=synergy,
            collaboration_type=collab_type
        )
        
        return {
            "success": True,
            "collaboration": {
                "id": collaboration.id,
                "collaboration_type": collaboration.collaboration_type.value,
                "primary_project_id": collaboration.primary_project_id,
                "collaborating_project_ids": collaboration.collaborating_project_ids,
                "synergy_score": collaboration.synergy_score,
                "coordination_frequency": collaboration.coordination_frequency,
                "shared_resources": collaboration.shared_resources,
                "shared_knowledge": collaboration.shared_knowledge,
                "joint_objectives": collaboration.joint_objectives,
                "is_active": collaboration.is_active,
                "created_at": collaboration.created_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating collaboration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/share")
async def share_knowledge_asset(
    source_project_id: str,
    target_project_ids: List[str],
    asset: Dict[str, Any]
):
    """
    Share knowledge asset between projects
    
    Args:
        source_project_id: Source project ID
        target_project_ids: Target project IDs
        asset: Knowledge asset information
        
    Returns:
        Sharing result
    """
    try:
        # Create knowledge asset
        knowledge_asset = KnowledgeAsset(**asset)
        
        # Share asset
        success = await collaboration_system.share_knowledge_asset(
            source_project_id=source_project_id,
            asset=knowledge_asset,
            target_project_ids=target_project_ids
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to share knowledge asset")
        
        return {
            "success": True,
            "asset_id": knowledge_asset.id,
            "message": f"Knowledge asset shared with {len(target_project_ids)} projects"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing knowledge asset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collaborations", response_model=Dict[str, Any])
async def list_collaborations(
    active_only: bool = Query(True, description="Show only active collaborations"),
    collaboration_type: Optional[str] = Query(None, description="Filter by collaboration type")
):
    """
    List research collaborations
    
    Args:
        active_only: Show only active collaborations
        collaboration_type: Filter by collaboration type
        
    Returns:
        List of collaborations
    """
    try:
        collaborations = list(collaboration_system.active_collaborations.values())
        
        # Apply filters
        if active_only:
            collaborations = [c for c in collaborations if c.is_active]
        
        if collaboration_type:
            try:
                collab_type = CollaborationType(collaboration_type)
                collaborations = [c for c in collaborations if c.collaboration_type == collab_type]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid collaboration type: {collaboration_type}")
        
        # Convert to response format
        collab_list = []
        for collaboration in collaborations:
            collab_list.append({
                "id": collaboration.id,
                "collaboration_type": collaboration.collaboration_type.value,
                "primary_project_id": collaboration.primary_project_id,
                "collaborating_project_ids": collaboration.collaborating_project_ids,
                "synergy_score": collaboration.synergy_score,
                "knowledge_transfer_rate": collaboration.knowledge_transfer_rate,
                "resource_efficiency_gain": collaboration.resource_efficiency_gain,
                "coordination_frequency": collaboration.coordination_frequency,
                "is_active": collaboration.is_active,
                "created_at": collaboration.created_at.isoformat(),
                "updated_at": collaboration.updated_at.isoformat()
            })
        
        return {
            "success": True,
            "collaborations": collab_list,
            "total_count": len(collab_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing collaborations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=Dict[str, Any])
async def get_coordination_metrics():
    """
    Get comprehensive research coordination metrics
    
    Returns:
        Research coordination performance metrics
    """
    try:
        # Get project management metrics
        project_metrics = await project_manager.get_coordination_metrics()
        
        # Get collaboration metrics
        collaboration_metrics = await collaboration_system.get_collaboration_metrics()
        
        return {
            "success": True,
            "metrics": {
                "project_management": {
                    "total_projects": project_metrics.total_projects,
                    "active_projects": project_metrics.active_projects,
                    "completed_projects": project_metrics.completed_projects,
                    "total_resources": project_metrics.total_resources,
                    "resource_utilization_rate": project_metrics.resource_utilization_rate,
                    "total_milestones": project_metrics.total_milestones,
                    "completed_milestones": project_metrics.completed_milestones,
                    "overdue_milestones": project_metrics.overdue_milestones,
                    "milestone_completion_rate": project_metrics.milestone_completion_rate,
                    "average_project_duration": project_metrics.average_project_duration,
                    "success_rate": project_metrics.success_rate
                },
                "collaboration": collaboration_metrics,
                "overall_coordination_score": (
                    project_metrics.success_rate + 
                    collaboration_metrics.get("collaboration_effectiveness", 0) * 100
                ) / 2 if collaboration_metrics.get("collaboration_effectiveness") else project_metrics.success_rate
            },
            "calculated_at": project_metrics.calculated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting coordination metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-assets", response_model=Dict[str, Any])
async def list_knowledge_assets(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    validation_status: Optional[str] = Query(None, description="Filter by validation status")
):
    """
    List knowledge assets
    
    Args:
        domain: Filter by domain
        asset_type: Filter by asset type
        validation_status: Filter by validation status
        
    Returns:
        List of knowledge assets
    """
    try:
        assets = list(collaboration_system.knowledge_assets.values())
        
        # Apply filters
        if domain:
            assets = [a for a in assets if domain.lower() in a.domain.lower()]
        
        if asset_type:
            assets = [a for a in assets if a.asset_type == asset_type]
        
        if validation_status:
            assets = [a for a in assets if a.validation_status == validation_status]
        
        # Convert to response format
        asset_list = []
        for asset in assets:
            asset_list.append({
                "id": asset.id,
                "title": asset.title,
                "description": asset.description,
                "asset_type": asset.asset_type,
                "source_project_id": asset.source_project_id,
                "domain": asset.domain,
                "keywords": asset.keywords,
                "confidence_score": asset.confidence_score,
                "validation_status": asset.validation_status,
                "access_count": asset.access_count,
                "citation_count": asset.citation_count,
                "reuse_count": asset.reuse_count,
                "created_at": asset.created_at.isoformat()
            })
        
        return {
            "success": True,
            "knowledge_assets": asset_list,
            "total_count": len(asset_list)
        }
        
    except Exception as e:
        logger.error(f"Error listing knowledge assets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))