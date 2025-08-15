"""
API Routes for Infrastructure Redundancy System

This module provides REST API endpoints for managing infrastructure redundancy,
multi-cloud resources, and research acceleration capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from ...core.infrastructure_redundancy import (
    infrastructure_engine, 
    CloudProvider, 
    ResourceType, 
    CloudResource,
    ResearchTask as CoreResearchTask
)
from ...core.multi_cloud_manager import multi_cloud_manager
from ...core.unlimited_compute_provisioner import (
    get_unlimited_compute_provisioner,
    ComputeRequest,
    ComputeWorkloadType,
    ScalingStrategy
)
from ...core.research_acceleration_engine import (
    get_research_acceleration_engine,
    ResearchDomain,
    TaskPriority,
    ResearchTask,
    ResearchProject
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/infrastructure", tags=["infrastructure"])

# Pydantic models for API requests/responses

class ResourceProvisionRequest(BaseModel):
    resource_type: str = Field(..., description="Type of resource to provision")
    required_capacity: Dict[str, Any] = Field(..., description="Required capacity specifications")
    priority: int = Field(default=1, ge=1, le=10, description="Priority level (1-10)")

class ResourceProvisionResponse(BaseModel):
    provisioned_resources: List[str] = Field(..., description="List of provisioned resource IDs")
    total_count: int = Field(..., description="Total number of resources provisioned")
    estimated_cost: float = Field(..., description="Estimated cost per hour")
    provisioning_time: datetime = Field(..., description="Time when resources were provisioned")

class FailoverRequest(BaseModel):
    failed_resource_id: str = Field(..., description="ID of the failed resource")

class FailoverResponse(BaseModel):
    success: bool = Field(..., description="Whether failover was successful")
    backup_resource_id: Optional[str] = Field(None, description="ID of the backup resource")
    failover_time: datetime = Field(..., description="Time when failover was executed")

class SystemStatusResponse(BaseModel):
    total_resources: int = Field(..., description="Total number of resources")
    active_providers: int = Field(..., description="Number of active cloud providers")
    resource_pools: Dict[str, Dict[str, int]] = Field(..., description="Resource pool status")
    failover_chains: int = Field(..., description="Number of failover chains")
    research_tasks: int = Field(..., description="Number of active research tasks")
    system_health: str = Field(..., description="Overall system health status")

class ComputeRequestModel(BaseModel):
    workload_type: str = Field(..., description="Type of compute workload")
    required_resources: Dict[str, Any] = Field(..., description="Required resource specifications")
    priority: int = Field(default=1, ge=1, le=10, description="Priority level")
    deadline: Optional[datetime] = Field(None, description="Deadline for completion")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in hours")
    scaling_strategy: str = Field(default="aggressive", description="Scaling strategy")
    cost_budget: Optional[float] = Field(None, description="Cost budget limit")
    preferred_providers: List[str] = Field(default_factory=list, description="Preferred cloud providers")

class ComputeAllocationResponse(BaseModel):
    request_id: str = Field(..., description="Request ID")
    allocated_resources: List[str] = Field(..., description="List of allocated resource IDs")
    allocation_time: datetime = Field(..., description="Time of allocation")
    estimated_cost: float = Field(..., description="Estimated cost per hour")
    performance_prediction: Dict[str, float] = Field(..., description="Performance predictions")

class ResearchProjectRequest(BaseModel):
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    domain: str = Field(..., description="Research domain")
    principal_investigator: str = Field(..., description="Principal investigator")
    compute_budget: float = Field(default=1000000.0, description="Compute budget")

class ResearchTaskRequest(BaseModel):
    name: str = Field(..., description="Task name")
    domain: str = Field(..., description="Research domain")
    priority: int = Field(default=3, ge=1, le=5, description="Task priority (1=critical, 5=background)")
    estimated_compute_hours: float = Field(..., description="Estimated compute hours")
    memory_requirements_gb: float = Field(..., description="Memory requirements in GB")
    cpu_cores_required: int = Field(..., description="Required CPU cores")
    gpu_required: bool = Field(default=False, description="Whether GPU is required")
    gpu_memory_gb: float = Field(default=0, description="GPU memory requirements in GB")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

class ResearchProjectResponse(BaseModel):
    id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    domain: str = Field(..., description="Research domain")
    principal_investigator: str = Field(..., description="Principal investigator")
    total_compute_budget: float = Field(..., description="Total compute budget")
    created_at: datetime = Field(..., description="Creation time")
    status: str = Field(..., description="Project status")
    progress: float = Field(..., description="Project progress (0-1)")
    task_count: int = Field(..., description="Number of tasks")

class MultiCloudStatusResponse(BaseModel):
    total_providers: int = Field(..., description="Total number of providers")
    active_providers: int = Field(..., description="Number of active providers")
    degraded_providers: int = Field(..., description="Number of degraded providers")
    failed_providers: int = Field(..., description="Number of failed providers")
    provider_details: Dict[str, Dict[str, Any]] = Field(..., description="Provider details")
    failover_rules: int = Field(..., description="Number of failover rules")
    total_regions: int = Field(..., description="Total number of regions")

# API Endpoints

@router.post("/provision-unlimited-resources", response_model=ResourceProvisionResponse)
async def provision_unlimited_resources(request: ResourceProvisionRequest):
    """Provision unlimited computing resources across all cloud providers"""
    try:
        # Convert string to enum
        resource_type = ResourceType(request.resource_type.lower())
        
        # Provision resources
        provisioned_resource_ids = infrastructure_engine.provision_unlimited_resources(
            resource_type=resource_type,
            required_capacity=request.required_capacity,
            priority=request.priority
        )
        
        # Calculate estimated cost
        total_cost = 0.0
        for resource_id in provisioned_resource_ids:
            if resource_id in infrastructure_engine.cloud_resources:
                resource = infrastructure_engine.cloud_resources[resource_id]
                total_cost += resource.cost_per_hour
        
        return ResourceProvisionResponse(
            provisioned_resources=provisioned_resource_ids,
            total_count=len(provisioned_resource_ids),
            estimated_cost=total_cost,
            provisioning_time=datetime.now()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid resource type: {e}")
    except Exception as e:
        logger.error(f"Failed to provision resources: {e}")
        raise HTTPException(status_code=500, detail="Failed to provision resources")

@router.post("/implement-failover", response_model=FailoverResponse)
async def implement_failover_system():
    """Implement automatic failover system for all resources"""
    try:
        failover_chains = infrastructure_engine.implement_failover_system()
        
        return {
            "success": True,
            "failover_chains_created": len(failover_chains),
            "implementation_time": datetime.now(),
            "message": f"Implemented failover system with {len(failover_chains)} failover chains"
        }
        
    except Exception as e:
        logger.error(f"Failed to implement failover system: {e}")
        raise HTTPException(status_code=500, detail="Failed to implement failover system")

@router.post("/execute-failover", response_model=FailoverResponse)
async def execute_failover(request: FailoverRequest):
    """Execute failover for a specific failed resource"""
    try:
        backup_resource_id = infrastructure_engine.execute_failover(request.failed_resource_id)
        
        return FailoverResponse(
            success=backup_resource_id is not None,
            backup_resource_id=backup_resource_id,
            failover_time=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to execute failover: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute failover")

@router.get("/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive infrastructure system status"""
    try:
        status = infrastructure_engine.get_system_status()
        return SystemStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@router.get("/multi-cloud-status", response_model=MultiCloudStatusResponse)
async def get_multi_cloud_status():
    """Get multi-cloud provider status"""
    try:
        status = multi_cloud_manager.get_multi_cloud_status()
        return MultiCloudStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get multi-cloud status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get multi-cloud status")

@router.post("/request-unlimited-compute", response_model=ComputeAllocationResponse)
async def request_unlimited_compute(request: ComputeRequestModel):
    """Request unlimited computing resources with specific requirements"""
    try:
        # Get compute provisioner
        compute_provisioner = get_unlimited_compute_provisioner(multi_cloud_manager)
        
        # Convert request to internal format
        compute_request = ComputeRequest(
            id=f"api-request-{datetime.now().timestamp()}",
            workload_type=ComputeWorkloadType(request.workload_type.lower()),
            required_resources=request.required_resources,
            priority=request.priority,
            deadline=request.deadline,
            estimated_duration=timedelta(hours=request.estimated_duration) if request.estimated_duration else None,
            scaling_strategy=ScalingStrategy(request.scaling_strategy.lower()),
            cost_budget=request.cost_budget,
            preferred_providers=[CloudProvider(p.lower()) for p in request.preferred_providers]
        )
        
        # Request compute resources
        allocation = await compute_provisioner.request_unlimited_compute(compute_request)
        
        return ComputeAllocationResponse(
            request_id=allocation.request_id,
            allocated_resources=[r.id for r in allocation.allocated_resources],
            allocation_time=allocation.allocation_time,
            estimated_cost=allocation.estimated_cost,
            performance_prediction=allocation.performance_prediction
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request parameters: {e}")
    except Exception as e:
        logger.error(f"Failed to request unlimited compute: {e}")
        raise HTTPException(status_code=500, detail="Failed to request unlimited compute")

@router.get("/provisioning-status")
async def get_provisioning_status():
    """Get current provisioning system status"""
    try:
        compute_provisioner = get_unlimited_compute_provisioner(multi_cloud_manager)
        status = compute_provisioner.get_provisioning_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get provisioning status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get provisioning status")

@router.post("/scale-allocation")
async def scale_allocation(request_id: str, scale_factor: float):
    """Scale an existing compute allocation"""
    try:
        compute_provisioner = get_unlimited_compute_provisioner(multi_cloud_manager)
        success = compute_provisioner.scale_allocation(request_id, scale_factor)
        
        return {
            "success": success,
            "request_id": request_id,
            "scale_factor": scale_factor,
            "scaled_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to scale allocation: {e}")
        raise HTTPException(status_code=500, detail="Failed to scale allocation")

@router.post("/research-projects", response_model=ResearchProjectResponse)
async def create_research_project(request: ResearchProjectRequest):
    """Create a new research project for massive parallel processing"""
    try:
        # Get research acceleration engine
        research_engine = get_research_acceleration_engine()
        
        # Create project
        project = await research_engine.create_research_project(
            name=request.name,
            description=request.description,
            domain=ResearchDomain(request.domain.lower()),
            principal_investigator=request.principal_investigator,
            compute_budget=request.compute_budget
        )
        
        return ResearchProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            domain=project.domain.value,
            principal_investigator=project.principal_investigator,
            total_compute_budget=project.total_compute_budget,
            created_at=project.created_at,
            status=project.status,
            progress=project.progress,
            task_count=len(project.tasks)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid project parameters: {e}")
    except Exception as e:
        logger.error(f"Failed to create research project: {e}")
        raise HTTPException(status_code=500, detail="Failed to create research project")

@router.post("/research-projects/{project_id}/tasks")
async def add_research_task(project_id: str, request: ResearchTaskRequest):
    """Add a research task to a project"""
    try:
        # Get research acceleration engine
        research_engine = get_research_acceleration_engine()
        
        # Create task (simplified - in real implementation would need actual computation function)
        task = ResearchTask(
            id=f"task-{datetime.now().timestamp()}",
            name=request.name,
            domain=ResearchDomain(request.domain.lower()),
            priority=TaskPriority(request.priority),
            computation_function=lambda x: f"Result for {x}",  # Placeholder function
            input_data={"task_name": request.name},
            expected_output_type=str,
            estimated_compute_hours=request.estimated_compute_hours,
            memory_requirements_gb=request.memory_requirements_gb,
            cpu_cores_required=request.cpu_cores_required,
            gpu_required=request.gpu_required,
            gpu_memory_gb=request.gpu_memory_gb,
            dependencies=request.dependencies,
            deadline=request.deadline,
            max_retries=request.max_retries
        )
        
        # Add task to project
        success = await research_engine.add_research_task(project_id, task)
        
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {
            "success": True,
            "task_id": task.id,
            "project_id": project_id,
            "added_at": datetime.now()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid task parameters: {e}")
    except Exception as e:
        logger.error(f"Failed to add research task: {e}")
        raise HTTPException(status_code=500, detail="Failed to add research task")

@router.post("/research-projects/{project_id}/execute")
async def execute_massive_parallel_research(project_id: str, background_tasks: BackgroundTasks):
    """Execute research project with massive parallel processing"""
    try:
        # Get research acceleration engine
        research_engine = get_research_acceleration_engine()
        
        # Start execution in background
        background_tasks.add_task(
            research_engine.execute_massive_parallel_research,
            project_id
        )
        
        return {
            "success": True,
            "project_id": project_id,
            "execution_started": datetime.now(),
            "message": "Massive parallel research execution started"
        }
        
    except Exception as e:
        logger.error(f"Failed to execute research project: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute research project")

@router.get("/research-projects")
async def list_research_projects():
    """List all active research projects"""
    try:
        research_engine = get_research_acceleration_engine()
        
        projects = []
        for project in research_engine.active_projects.values():
            projects.append(ResearchProjectResponse(
                id=project.id,
                name=project.name,
                description=project.description,
                domain=project.domain.value,
                principal_investigator=project.principal_investigator,
                total_compute_budget=project.total_compute_budget,
                created_at=project.created_at,
                status=project.status,
                progress=project.progress,
                task_count=len(project.tasks)
            ))
        
        return {"projects": projects, "total_count": len(projects)}
        
    except Exception as e:
        logger.error(f"Failed to list research projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to list research projects")

@router.get("/research-projects/{project_id}")
async def get_research_project(project_id: str):
    """Get detailed information about a research project"""
    try:
        research_engine = get_research_acceleration_engine()
        
        if project_id not in research_engine.active_projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project = research_engine.active_projects[project_id]
        
        # Get task details
        tasks = []
        for task in project.tasks:
            tasks.append({
                "id": task.id,
                "name": task.name,
                "domain": task.domain.value,
                "priority": task.priority.value,
                "status": task.status.value,
                "progress": task.progress,
                "estimated_compute_hours": task.estimated_compute_hours,
                "cpu_cores_required": task.cpu_cores_required,
                "memory_requirements_gb": task.memory_requirements_gb,
                "gpu_required": task.gpu_required,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error_message": task.error_message
            })
        
        return {
            "project": ResearchProjectResponse(
                id=project.id,
                name=project.name,
                description=project.description,
                domain=project.domain.value,
                principal_investigator=project.principal_investigator,
                total_compute_budget=project.total_compute_budget,
                created_at=project.created_at,
                status=project.status,
                progress=project.progress,
                task_count=len(project.tasks)
            ),
            "tasks": tasks,
            "results": project.results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get research project: {e}")
        raise HTTPException(status_code=500, detail="Failed to get research project")

@router.get("/acceleration-status")
async def get_acceleration_status():
    """Get research acceleration engine status"""
    try:
        research_engine = get_research_acceleration_engine()
        status = research_engine.get_acceleration_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get acceleration status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get acceleration status")

@router.get("/health")
async def health_check():
    """Health check endpoint for infrastructure system"""
    try:
        # Check all major components
        infrastructure_status = infrastructure_engine.get_system_status()
        multi_cloud_status = multi_cloud_manager.get_multi_cloud_status()
        
        compute_provisioner = get_unlimited_compute_provisioner(multi_cloud_manager)
        provisioning_status = compute_provisioner.get_provisioning_status()
        
        research_engine = get_research_acceleration_engine()
        acceleration_status = research_engine.get_acceleration_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "infrastructure_engine": infrastructure_status["system_health"],
                "multi_cloud_manager": "operational",
                "compute_provisioner": provisioning_status["system_status"],
                "research_acceleration": acceleration_status["system_status"]
            },
            "summary": {
                "total_resources": infrastructure_status["total_resources"],
                "active_providers": multi_cloud_status["active_providers"],
                "active_allocations": provisioning_status["active_allocations"],
                "active_projects": acceleration_status["active_projects"]
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(),
            "error": str(e)
        }