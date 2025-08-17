"""
API routes for pipeline management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from scrollintel.models.database import get_db
from scrollintel.core.pipeline_builder import PipelineBuilder
from scrollintel.models.pipeline_models import PipelineStatus, NodeType

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])

# Pydantic models for request/response
class PipelineCreate(BaseModel):
    name: str
    description: str = ""
    created_by: str = ""

class PipelineUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[PipelineStatus] = None

class NodeCreate(BaseModel):
    name: str
    node_type: NodeType
    config: Dict[str, Any] = {}
    position_x: int = 0
    position_y: int = 0

class ConnectionCreate(BaseModel):
    source_node_id: str
    target_node_id: str
    config: Dict[str, Any] = {}

class NodePositionUpdate(BaseModel):
    x: int
    y: int

@router.post("/")
async def create_pipeline(pipeline: PipelineCreate, db: Session = Depends(get_db)):
    """Create a new pipeline"""
    try:
        builder = PipelineBuilder(db)
        new_pipeline = builder.create_pipeline(
            name=pipeline.name,
            description=pipeline.description,
            created_by=pipeline.created_by
        )
        return {
            "id": new_pipeline.id,
            "name": new_pipeline.name,
            "description": new_pipeline.description,
            "status": new_pipeline.status.value,
            "created_at": new_pipeline.created_at,
            "created_by": new_pipeline.created_by
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_pipelines(created_by: Optional[str] = None, db: Session = Depends(get_db)):
    """List all pipelines"""
    try:
        builder = PipelineBuilder(db)
        pipelines = builder.list_pipelines(created_by=created_by)
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "status": p.status.value,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
                "created_by": p.created_by,
                "node_count": len(p.nodes),
                "connection_count": len(p.connections)
            }
            for p in pipelines
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str, db: Session = Depends(get_db)):
    """Get a specific pipeline with all nodes and connections"""
    try:
        builder = PipelineBuilder(db)
        pipeline = builder.get_pipeline(pipeline_id)
        
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        nodes = builder.get_pipeline_nodes(pipeline_id)
        connections = builder.get_pipeline_connections(pipeline_id)
        
        return {
            "id": pipeline.id,
            "name": pipeline.name,
            "description": pipeline.description,
            "status": pipeline.status.value,
            "config": pipeline.config,
            "created_at": pipeline.created_at,
            "updated_at": pipeline.updated_at,
            "created_by": pipeline.created_by,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "node_type": n.node_type.value,
                    "config": n.config,
                    "position": {"x": n.position_x, "y": n.position_y},
                    "created_at": n.created_at
                }
                for n in nodes
            ],
            "connections": [
                {
                    "id": c.id,
                    "source_node_id": c.source_node_id,
                    "target_node_id": c.target_node_id,
                    "config": c.config,
                    "created_at": c.created_at
                }
                for c in connections
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{pipeline_id}")
async def update_pipeline(pipeline_id: str, pipeline: PipelineUpdate, db: Session = Depends(get_db)):
    """Update pipeline properties"""
    try:
        builder = PipelineBuilder(db)
        update_data = {k: v for k, v in pipeline.dict().items() if v is not None}
        
        updated_pipeline = builder.update_pipeline(pipeline_id, **update_data)
        
        return {
            "id": updated_pipeline.id,
            "name": updated_pipeline.name,
            "description": updated_pipeline.description,
            "status": updated_pipeline.status.value,
            "updated_at": updated_pipeline.updated_at
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str, db: Session = Depends(get_db)):
    """Delete a pipeline"""
    try:
        builder = PipelineBuilder(db)
        success = builder.delete_pipeline(pipeline_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {"message": "Pipeline deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{pipeline_id}/nodes")
async def add_node(pipeline_id: str, node: NodeCreate, db: Session = Depends(get_db)):
    """Add a node to the pipeline"""
    try:
        builder = PipelineBuilder(db)
        
        node_config = {
            "name": node.name,
            "position_x": node.position_x,
            "position_y": node.position_y,
            **node.config
        }
        
        if node.node_type == NodeType.DATA_SOURCE:
            new_node = builder.add_data_source(pipeline_id, node_config)
        elif node.node_type == NodeType.TRANSFORMATION:
            new_node = builder.add_transformation(pipeline_id, node_config)
        elif node.node_type == NodeType.DATA_TARGET:
            new_node = builder.add_data_target(pipeline_id, node_config)
        else:
            # For validation or other node types, use add_transformation as base
            new_node = builder.add_transformation(pipeline_id, node_config)
        
        return {
            "id": new_node.id,
            "name": new_node.name,
            "node_type": new_node.node_type.value,
            "config": new_node.config,
            "position": {"x": new_node.position_x, "y": new_node.position_y},
            "created_at": new_node.created_at
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{pipeline_id}/connections")
async def add_connection(pipeline_id: str, connection: ConnectionCreate, db: Session = Depends(get_db)):
    """Add a connection between nodes"""
    try:
        builder = PipelineBuilder(db)
        new_connection = builder.connect_nodes(
            source_node_id=connection.source_node_id,
            target_node_id=connection.target_node_id,
            config=connection.config
        )
        
        return {
            "id": new_connection.id,
            "source_node_id": new_connection.source_node_id,
            "target_node_id": new_connection.target_node_id,
            "config": new_connection.config,
            "created_at": new_connection.created_at
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{pipeline_id}/nodes/{node_id}/position")
async def update_node_position(pipeline_id: str, node_id: str, position: NodePositionUpdate, db: Session = Depends(get_db)):
    """Update node position"""
    try:
        builder = PipelineBuilder(db)
        updated_node = builder.update_node_position(node_id, position.x, position.y)
        
        return {
            "id": updated_node.id,
            "position": {"x": updated_node.position_x, "y": updated_node.position_y}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{pipeline_id}/validate")
async def validate_pipeline(pipeline_id: str, db: Session = Depends(get_db)):
    """Validate pipeline structure and configuration"""
    try:
        builder = PipelineBuilder(db)
        result = builder.validate_pipeline(pipeline_id)
        
        return {
            "pipeline_id": pipeline_id,
            "is_valid": result["is_valid"],
            "errors": result["errors"],
            "warnings": result["warnings"],
            "validation_id": result["validation_id"],
            "validated_at": "now"  # You might want to add timestamp to validation result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pipeline_id}/nodes")
async def get_pipeline_nodes(pipeline_id: str, db: Session = Depends(get_db)):
    """Get all nodes for a pipeline"""
    try:
        builder = PipelineBuilder(db)
        nodes = builder.get_pipeline_nodes(pipeline_id)
        
        return [
            {
                "id": n.id,
                "name": n.name,
                "node_type": n.node_type.value,
                "config": n.config,
                "position": {"x": n.position_x, "y": n.position_y},
                "created_at": n.created_at
            }
            for n in nodes
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{pipeline_id}/connections")
async def get_pipeline_connections(pipeline_id: str, db: Session = Depends(get_db)):
    """Get all connections for a pipeline"""
    try:
        builder = PipelineBuilder(db)
        connections = builder.get_pipeline_connections(pipeline_id)
        
        return [
            {
                "id": c.id,
                "source_node_id": c.source_node_id,
                "target_node_id": c.target_node_id,
                "config": c.config,
                "created_at": c.created_at
            }
            for c in connections
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))