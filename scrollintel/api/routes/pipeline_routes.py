"""
Data Pipeline Automation - API Routes
FastAPI routes for pipeline management and operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

from scrollintel.core.database import get_db
from scrollintel.core.pipeline_builder import PipelineBuilder, DataSourceConfig, TransformConfig
from scrollintel.models.pipeline_models import Pipeline, PipelineNode, PipelineStatus, NodeType
from scrollintel.core.pipeline_orchestrator import (
    get_orchestrator, ScheduleConfig, ScheduleType, ResourceType
)

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


# Pydantic models for API
class PipelineCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    created_by: Optional[str] = None


class PipelineUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[PipelineStatus] = None


class NodeCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    node_type: NodeType
    component_type: str
    position_x: int = 0
    position_y: int = 0
    config: dict = Field(default_factory=dict)


class ConnectionCreate(BaseModel):
    source_node_id: str
    target_node_id: str
    source_port: str = "output"
    target_port: str = "input"


class PipelineExecutionRequest(BaseModel):
    priority: int = Field(default=5, ge=1, le=10)
    resource_requirements: Optional[dict] = None
    dependencies: Optional[List[str]] = None


class ScheduleRequest(BaseModel):
    schedule_type: str
    interval_seconds: Optional[int] = None
    cron_expression: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    enabled: bool = True
    timezone: str = "UTC"
    priority: int = Field(default=5, ge=1, le=10)
    resource_requirements: Optional[dict] = None
    dependencies: Optional[List[str]] = None


class PipelineResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: PipelineStatus
    validation_status: str
    created_at: str
    updated_at: Optional[str]
    created_by: Optional[str]
    
    class Config:
        from_attributes = True


class NodeResponse(BaseModel):
    id: str
    name: str
    node_type: NodeType
    component_type: str
    position_x: int
    position_y: int
    config: dict
    is_valid: bool
    validation_errors: List[str]
    
    class Config:
        from_attributes = True


@router.post("/", response_model=PipelineResponse)
async def create_pipeline(
    pipeline_data: PipelineCreate,
    db: Session = Depends(get_db)
):
    """Create a new pipeline"""
    try:
        builder = PipelineBuilder(db)
        pipeline = builder.create_pipeline(
            name=pipeline_data.name,
            description=pipeline_data.description or "",
            created_by=pipeline_data.created_by or ""
        )
        
        return PipelineResponse(
            id=pipeline.id,
            name=pipeline.name,
            description=pipeline.description,
            status=pipeline.status,
            validation_status=pipeline.validation_status.value,
            created_at=pipeline.created_at.isoformat(),
            updated_at=pipeline.updated_at.isoformat() if pipeline.updated_at else None,
            created_by=pipeline.created_by
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pipeline: {str(e)}"
        )


@router.get("/", response_model=List[PipelineResponse])
async def list_pipelines(
    status_filter: Optional[PipelineStatus] = None,
    created_by: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all pipelines with optional filtering"""
    try:
        builder = PipelineBuilder(db)
        pipelines = builder.list_pipelines(status=status_filter, created_by=created_by)
        
        return [
            PipelineResponse(
                id=p.id,
                name=p.name,
                description=p.description,
                status=p.status,
                validation_status=p.validation_status.value,
                created_at=p.created_at.isoformat(),
                updated_at=p.updated_at.isoformat() if p.updated_at else None,
                created_by=p.created_by
            )
            for p in pipelines
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list pipelines: {str(e)}"
        )


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific pipeline by ID"""
    try:
        builder = PipelineBuilder(db)
        pipeline = builder.get_pipeline(pipeline_id)
        
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline not found"
            )
        
        return PipelineResponse(
            id=pipeline.id,
            name=pipeline.name,
            description=pipeline.description,
            status=pipeline.status,
            validation_status=pipeline.validation_status.value,
            created_at=pipeline.created_at.isoformat(),
            updated_at=pipeline.updated_at.isoformat() if pipeline.updated_at else None,
            created_by=pipeline.created_by
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline: {str(e)}"
        )


@router.put("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: str,
    pipeline_data: PipelineUpdate,
    db: Session = Depends(get_db)
):
    """Update a pipeline"""
    try:
        builder = PipelineBuilder(db)
        
        update_data = {k: v for k, v in pipeline_data.dict().items() if v is not None}
        pipeline = builder.update_pipeline(pipeline_id, **update_data)
        
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline not found"
            )
        
        return PipelineResponse(
            id=pipeline.id,
            name=pipeline.name,
            description=pipeline.description,
            status=pipeline.status,
            validation_status=pipeline.validation_status.value,
            created_at=pipeline.created_at.isoformat(),
            updated_at=pipeline.updated_at.isoformat() if pipeline.updated_at else None,
            created_by=pipeline.created_by
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update pipeline: {str(e)}"
        )


@router.delete("/{pipeline_id}")
async def delete_pipeline(
    pipeline_id: str,
    db: Session = Depends(get_db)
):
    """Delete a pipeline"""
    try:
        builder = PipelineBuilder(db)
        success = builder.delete_pipeline(pipeline_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline not found"
            )
        
        return {"message": "Pipeline deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete pipeline: {str(e)}"
        )


@router.post("/{pipeline_id}/nodes", response_model=NodeResponse)
async def add_node(
    pipeline_id: str,
    node_data: NodeCreate,
    db: Session = Depends(get_db)
):
    """Add a node to a pipeline"""
    try:
        builder = PipelineBuilder(db)
        
        if node_data.node_type == NodeType.DATA_SOURCE:
            source_config = DataSourceConfig(
                source_type=node_data.component_type,
                connection_params=node_data.config
            )
            node = builder.add_data_source(
                pipeline_id=pipeline_id,
                source_config=source_config,
                name=node_data.name,
                position=(node_data.position_x, node_data.position_y)
            )
        elif node_data.node_type == NodeType.TRANSFORMATION:
            transform_config = TransformConfig(
                transform_type=node_data.component_type,
                parameters=node_data.config
            )
            node = builder.add_transformation(
                pipeline_id=pipeline_id,
                transform_config=transform_config,
                name=node_data.name,
                position=(node_data.position_x, node_data.position_y)
            )
        else:
            # For other node types, create directly
            from scrollintel.models.pipeline_models import PipelineNode
            node = PipelineNode(
                pipeline_id=pipeline_id,
                name=node_data.name,
                node_type=node_data.node_type,
                component_type=node_data.component_type,
                position_x=node_data.position_x,
                position_y=node_data.position_y,
                config=node_data.config
            )
            db.add(node)
            db.commit()
            db.refresh(node)
        
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline not found"
            )
        
        return NodeResponse(
            id=node.id,
            name=node.name,
            node_type=node.node_type,
            component_type=node.component_type,
            position_x=node.position_x,
            position_y=node.position_y,
            config=node.config,
            is_valid=node.is_valid,
            validation_errors=node.validation_errors
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add node: {str(e)}"
        )


@router.get("/{pipeline_id}/nodes", response_model=List[NodeResponse])
async def get_pipeline_nodes(
    pipeline_id: str,
    db: Session = Depends(get_db)
):
    """Get all nodes in a pipeline"""
    try:
        nodes = db.query(PipelineNode).filter(PipelineNode.pipeline_id == pipeline_id).all()
        
        return [
            NodeResponse(
                id=node.id,
                name=node.name,
                node_type=node.node_type,
                component_type=node.component_type,
                position_x=node.position_x,
                position_y=node.position_y,
                config=node.config,
                is_valid=node.is_valid,
                validation_errors=node.validation_errors
            )
            for node in nodes
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get nodes: {str(e)}"
        )


@router.post("/{pipeline_id}/connections")
async def add_connection(
    pipeline_id: str,
    connection_data: ConnectionCreate,
    db: Session = Depends(get_db)
):
    """Add a connection between nodes"""
    try:
        builder = PipelineBuilder(db)
        connection = builder.connect_nodes(
            source_node_id=connection_data.source_node_id,
            target_node_id=connection_data.target_node_id,
            source_port=connection_data.source_port,
            target_port=connection_data.target_port
        )
        
        if not connection:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create connection"
            )
        
        return {
            "id": connection.id,
            "source_node_id": connection.source_node_id,
            "target_node_id": connection.target_node_id,
            "source_port": connection.source_port,
            "target_port": connection.target_port
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add connection: {str(e)}"
        )


@router.post("/{pipeline_id}/validate")
async def validate_pipeline(
    pipeline_id: str,
    db: Session = Depends(get_db)
):
    """Validate a pipeline"""
    try:
        builder = PipelineBuilder(db)
        result = builder.validate_pipeline(pipeline_id)
        
        return {
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate pipeline: {str(e)}"
        )


@router.post("/{pipeline_id}/execute")
async def execute_pipeline_now(
    pipeline_id: str,
    execution_request: PipelineExecutionRequest,
    db: Session = Depends(get_db)
):
    """Execute a pipeline immediately"""
    try:
        builder = PipelineBuilder(db)
        pipeline = builder.get_pipeline(pipeline_id)
        
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline not found"
            )
        
        # Validate before execution
        validation_result = builder.validate_pipeline(pipeline_id)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Pipeline validation failed: {validation_result.errors}"
            )
        
        # Convert resource requirements
        resource_requirements = {}
        if execution_request.resource_requirements:
            for key, value in execution_request.resource_requirements.items():
                try:
                    resource_type = ResourceType(key.lower())
                    resource_requirements[resource_type] = value
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid resource type: {key}"
                    )
        
        orchestrator = get_orchestrator()
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=execution_request.priority,
            resource_requirements=resource_requirements
        )
        
        return {
            "execution_id": execution_id,
            "pipeline_id": pipeline_id,
            "status": "scheduled",
            "priority": execution_request.priority,
            "resource_requirements": execution_request.resource_requirements
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute pipeline: {str(e)}"
        )


@router.post("/{pipeline_id}/schedule")
async def schedule_pipeline(
    pipeline_id: str,
    schedule_request: ScheduleRequest,
    db: Session = Depends(get_db)
):
    """Schedule a pipeline for execution"""
    try:
        builder = PipelineBuilder(db)
        pipeline = builder.get_pipeline(pipeline_id)
        
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pipeline not found"
            )
        
        # Create schedule config
        try:
            schedule_type = ScheduleType(schedule_request.schedule_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid schedule type: {schedule_request.schedule_type}"
            )
        
        from datetime import datetime
        start_time = None
        end_time = None
        
        if schedule_request.start_time:
            start_time = datetime.fromisoformat(schedule_request.start_time.replace('Z', '+00:00'))
        if schedule_request.end_time:
            end_time = datetime.fromisoformat(schedule_request.end_time.replace('Z', '+00:00'))
        
        schedule_config = ScheduleConfig(
            schedule_type=schedule_type,
            interval_seconds=schedule_request.interval_seconds,
            cron_expression=schedule_request.cron_expression,
            start_time=start_time,
            end_time=end_time,
            enabled=schedule_request.enabled,
            timezone=schedule_request.timezone
        )
        
        # Convert resource requirements
        resource_requirements = {}
        if schedule_request.resource_requirements:
            for key, value in schedule_request.resource_requirements.items():
                try:
                    resource_type = ResourceType(key.lower())
                    resource_requirements[resource_type] = value
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid resource type: {key}"
                    )
        
        orchestrator = get_orchestrator()
        execution_id = orchestrator.schedule_pipeline(
            pipeline_id=pipeline_id,
            schedule_config=schedule_config,
            priority=schedule_request.priority,
            dependencies=schedule_request.dependencies,
            resource_requirements=resource_requirements
        )
        
        return {
            "execution_id": execution_id,
            "pipeline_id": pipeline_id,
            "schedule_config": schedule_request.dict(),
            "status": "scheduled"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule pipeline: {str(e)}"
        )


@router.get("/{pipeline_id}/executions")
async def get_pipeline_executions(
    pipeline_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get execution history for a pipeline"""
    try:
        from scrollintel.models.pipeline_models import PipelineExecution
        
        executions = db.query(PipelineExecution).filter(
            PipelineExecution.pipeline_id == pipeline_id
        ).order_by(PipelineExecution.start_time.desc()).offset(skip).limit(limit).all()
        
        return [
            {
                "id": execution.id,
                "pipeline_id": execution.pipeline_id,
                "status": execution.status,
                "start_time": execution.start_time.isoformat(),
                "end_time": execution.end_time.isoformat() if execution.end_time else None,
                "records_processed": execution.records_processed,
                "execution_metrics": execution.execution_metrics,
                "error_details": execution.error_details
            }
            for execution in executions
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get executions: {str(e)}"
        )


@router.get("/executions/{execution_id}/status")
async def get_execution_status(execution_id: str):
    """Get status of a specific execution"""
    try:
        orchestrator = get_orchestrator()
        status_info = orchestrator.get_execution_status(execution_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found"
            )
        
        return status_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution status: {str(e)}"
        )


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel a pending or running execution"""
    try:
        orchestrator = get_orchestrator()
        success = orchestrator.cancel_execution(execution_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found or cannot be cancelled"
            )
        
        return {"message": "Execution cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel execution: {str(e)}"
        )


@router.post("/executions/{execution_id}/pause")
async def pause_execution(execution_id: str):
    """Pause a running execution"""
    try:
        orchestrator = get_orchestrator()
        success = orchestrator.pause_execution(execution_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found or cannot be paused"
            )
        
        return {"message": "Execution paused successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause execution: {str(e)}"
        )


@router.post("/executions/{execution_id}/resume")
async def resume_execution(execution_id: str):
    """Resume a paused execution"""
    try:
        orchestrator = get_orchestrator()
        success = orchestrator.resume_execution(execution_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found or cannot be resumed"
            )
        
        return {"message": "Execution resumed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume execution: {str(e)}"
        )


@router.get("/{pipeline_id}/dependencies")
async def get_pipeline_dependencies(pipeline_id: str):
    """Get dependency chain for a pipeline"""
    try:
        orchestrator = get_orchestrator()
        dependencies = orchestrator.get_pipeline_dependencies(pipeline_id)
        return {"dependencies": dependencies}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dependencies: {str(e)}"
        )


@router.get("/orchestrator/metrics")
async def get_orchestrator_metrics():
    """Get orchestrator performance metrics"""
    try:
        orchestrator = get_orchestrator()
        return orchestrator.get_orchestrator_metrics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/orchestrator/resource-utilization")
async def get_resource_utilization():
    """Get current resource utilization"""
    try:
        orchestrator = get_orchestrator()
        return orchestrator.get_resource_utilization()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource utilization: {str(e)}"
        )


@router.get("/templates/components")
async def get_component_templates(
    node_type: Optional[NodeType] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get available component templates"""
    try:
        builder = PipelineBuilder(db)
        templates = builder.get_component_templates(node_type=node_type, category=category)
        
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "node_type": t.node_type.value,
                "component_type": t.component_type,
                "default_config": t.default_config,
                "icon": t.icon,
                "color": t.color
            }
            for t in templates
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get component templates: {str(e)}"
        )