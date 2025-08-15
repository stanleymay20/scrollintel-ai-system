"""
API routes for Autonomous Innovation Lab system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
import logging

from ...core.autonomous_innovation_lab import AutonomousInnovationLab, LabConfiguration
from ...models.innovation_lab_integration_models import (
    LabStatusResponse, ValidationRequest, ValidationResponse,
    InnovationProjectResponse, LabMetricsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/innovation-lab", tags=["Autonomous Innovation Lab"])

# Global lab instance
lab_instance: Optional[AutonomousInnovationLab] = None

@router.post("/start", response_model=Dict[str, Any])
async def start_innovation_lab(
    config: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None
):
    """Start the autonomous innovation lab"""
    global lab_instance
    
    try:
        # Create lab configuration
        lab_config = LabConfiguration()
        if config:
            for key, value in config.items():
                if hasattr(lab_config, key):
                    setattr(lab_config, key, value)
        
        # Initialize lab
        lab_instance = AutonomousInnovationLab(lab_config)
        
        # Start lab
        success = await lab_instance.start_lab()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start innovation lab")
        
        return {
            "success": True,
            "message": "Autonomous Innovation Lab started successfully",
            "lab_id": "main_lab",
            "status": await lab_instance.get_lab_status()
        }
    
    except Exception as e:
        logger.error(f"Error starting innovation lab: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start lab: {str(e)}")

@router.post("/stop")
async def stop_innovation_lab():
    """Stop the autonomous innovation lab"""
    global lab_instance
    
    try:
        if not lab_instance:
            raise HTTPException(status_code=404, detail="Innovation lab not running")
        
        await lab_instance.stop_lab()
        lab_instance = None
        
        return {
            "success": True,
            "message": "Autonomous Innovation Lab stopped successfully"
        }
    
    except Exception as e:
        logger.error(f"Error stopping innovation lab: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop lab: {str(e)}")

@router.get("/status", response_model=LabStatusResponse)
async def get_lab_status():
    """Get current lab status and metrics"""
    try:
        if not lab_instance:
            return LabStatusResponse(
                status="stopped",
                is_running=False,
                active_projects=0,
                metrics={},
                message="Innovation lab is not running"
            )
        
        status = await lab_instance.get_lab_status()
        
        return LabStatusResponse(
            status=status["status"],
            is_running=status["is_running"],
            active_projects=status["active_projects"],
            metrics=status["metrics"],
            last_validation=status.get("last_validation"),
            research_domains=status.get("research_domains", [])
        )
    
    except Exception as e:
        logger.error(f"Error getting lab status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/validate", response_model=ValidationResponse)
async def validate_lab_capability(request: ValidationRequest):
    """Validate autonomous innovation lab capability across research domains"""
    try:
        if not lab_instance:
            raise HTTPException(status_code=404, detail="Innovation lab not running")
        
        validation_result = await lab_instance.validate_lab_capability(request.domain)
        
        return ValidationResponse(
            overall_success=validation_result["overall_success"],
            domain_results=validation_result.get("domain_results", {}),
            validation_timestamp=validation_result["validation_timestamp"],
            lab_status=validation_result.get("lab_status", {}),
            error=validation_result.get("error")
        )
    
    except Exception as e:
        logger.error(f"Error validating lab capability: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/projects", response_model=List[InnovationProjectResponse])
async def get_active_projects():
    """Get all active innovation projects"""
    try:
        if not lab_instance:
            raise HTTPException(status_code=404, detail="Innovation lab not running")
        
        projects = []
        for project_id, project in lab_instance.active_projects.items():
            projects.append(InnovationProjectResponse(
                id=project.id,
                title=project.title,
                domain=project.domain,
                status=project.status,
                created_at=project.created_at.isoformat(),
                experiment_count=len(project.experiment_plans),
                prototype_count=len(project.prototypes) if hasattr(project, 'prototypes') else 0,
                innovation_count=len(project.validated_innovations) if hasattr(project, 'validated_innovations') else 0
            ))
        
        return projects
    
    except Exception as e:
        logger.error(f"Error getting active projects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")

@router.get("/projects/{project_id}")
async def get_project_details(project_id: str):
    """Get detailed information about a specific project"""
    try:
        if not lab_instance:
            raise HTTPException(status_code=404, detail="Innovation lab not running")
        
        if project_id not in lab_instance.active_projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project = lab_instance.active_projects[project_id]
        
        return {
            "id": project.id,
            "title": project.title,
            "domain": project.domain,
            "status": project.status,
            "created_at": project.created_at.isoformat(),
            "research_topic": str(project.research_topic),
            "hypotheses": [str(h) for h in project.hypotheses],
            "experiments": [
                {
                    "id": getattr(exp, 'id', 'unknown'),
                    "status": getattr(exp, 'status', 'unknown'),
                    "results": getattr(exp, 'results', None)
                }
                for exp in project.experiment_plans
            ],
            "prototypes": [
                {
                    "id": getattr(proto, 'id', 'unknown'),
                    "validated": getattr(proto, 'validated', False)
                }
                for proto in (project.prototypes if hasattr(project, 'prototypes') else [])
            ],
            "innovations": len(project.validated_innovations) if hasattr(project, 'validated_innovations') else 0
        }
    
    except Exception as e:
        logger.error(f"Error getting project details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get project details: {str(e)}")

@router.get("/metrics", response_model=LabMetricsResponse)
async def get_lab_metrics():
    """Get comprehensive lab metrics and performance data"""
    try:
        if not lab_instance:
            raise HTTPException(status_code=404, detail="Innovation lab not running")
        
        status = await lab_instance.get_lab_status()
        metrics = status["metrics"]
        
        # Calculate additional metrics
        total_projects = metrics.get("active_projects", 0) + metrics.get("completed_projects", 0)
        success_rate = metrics.get("success_rate", 0.0)
        
        return LabMetricsResponse(
            total_innovations=metrics.get("total_innovations", 0),
            active_projects=metrics.get("active_projects", 0),
            completed_projects=metrics.get("completed_projects", 0),
            total_projects=total_projects,
            success_rate=success_rate,
            research_domains=status.get("research_domains", []),
            last_updated=status.get("last_validation")
        )
    
    except Exception as e:
        logger.error(f"Error getting lab metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.post("/domains/{domain}/validate")
async def validate_domain_capability(domain: str):
    """Validate capability in a specific research domain"""
    try:
        if not lab_instance:
            raise HTTPException(status_code=404, detail="Innovation lab not running")
        
        validation_result = await lab_instance.validate_lab_capability(domain)
        
        if domain not in validation_result.get("domain_results", {}):
            raise HTTPException(status_code=404, detail=f"Domain {domain} not found in validation results")
        
        return {
            "domain": domain,
            "validation_result": validation_result["domain_results"][domain],
            "timestamp": validation_result["validation_timestamp"]
        }
    
    except Exception as e:
        logger.error(f"Error validating domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Domain validation failed: {str(e)}")

@router.post("/emergency-stop")
async def emergency_stop():
    """Emergency stop of all lab operations"""
    global lab_instance
    
    try:
        if lab_instance:
            await lab_instance.stop_lab()
            lab_instance = None
        
        return {
            "success": True,
            "message": "Emergency stop executed successfully"
        }
    
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        return {
            "success": False,
            "message": f"Emergency stop failed: {str(e)}"
        }

@router.get("/health")
async def health_check():
    """Health check for the innovation lab system"""
    try:
        if not lab_instance:
            return {
                "status": "stopped",
                "healthy": False,
                "message": "Innovation lab is not running"
            }
        
        status = await lab_instance.get_lab_status()
        
        return {
            "status": status["status"],
            "healthy": status["is_running"],
            "active_projects": status["active_projects"],
            "message": "Innovation lab is operational"
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "healthy": False,
            "message": f"Health check failed: {str(e)}"
        }