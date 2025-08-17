"""
PipelineBuilder class with CRUD operations for data pipeline management
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
import uuid
from datetime import datetime

from scrollintel.models.pipeline_models import (
    Pipeline, PipelineNode, PipelineConnection, DataSourceConfig, 
    ValidationResult, PipelineStatus, NodeType
)

class ValidationError(Exception):
    """Custom exception for pipeline validation errors"""
    pass

class PipelineBuilder:
    """Main class for building and managing data pipelines"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_pipeline(self, name: str, description: str = "", created_by: str = "") -> Pipeline:
        """Create a new pipeline"""
        pipeline = Pipeline(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_by=created_by,
            status=PipelineStatus.DRAFT
        )
        
        self.db.add(pipeline)
        self.db.commit()
        self.db.refresh(pipeline)
        
        return pipeline
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get a pipeline by ID"""
        return self.db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
    
    def list_pipelines(self, created_by: str = None) -> List[Pipeline]:
        """List all pipelines, optionally filtered by creator"""
        query = self.db.query(Pipeline)
        if created_by:
            query = query.filter(Pipeline.created_by == created_by)
        return query.all()
    
    def update_pipeline(self, pipeline_id: str, **kwargs) -> Pipeline:
        """Update pipeline properties"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        for key, value in kwargs.items():
            if hasattr(pipeline, key):
                setattr(pipeline, key, value)
        
        pipeline.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(pipeline)
        
        return pipeline
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline and all its nodes/connections"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return False
        
        self.db.delete(pipeline)
        self.db.commit()
        return True
    
    def add_data_source(self, pipeline_id: str, source_config: Dict[str, Any]) -> PipelineNode:
        """Add a data source node to the pipeline"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        node = PipelineNode(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline_id,
            name=source_config.get("name", "Data Source"),
            node_type=NodeType.DATA_SOURCE,
            config=source_config,
            position_x=source_config.get("position_x", 0),
            position_y=source_config.get("position_y", 0)
        )
        
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        
        return node
    
    def add_transformation(self, pipeline_id: str, transform_config: Dict[str, Any]) -> PipelineNode:
        """Add a transformation node to the pipeline"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        node = PipelineNode(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline_id,
            name=transform_config.get("name", "Transformation"),
            node_type=NodeType.TRANSFORMATION,
            config=transform_config,
            position_x=transform_config.get("position_x", 0),
            position_y=transform_config.get("position_y", 0)
        )
        
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        
        return node
    
    def add_data_target(self, pipeline_id: str, target_config: Dict[str, Any]) -> PipelineNode:
        """Add a data target node to the pipeline"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        node = PipelineNode(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline_id,
            name=target_config.get("name", "Data Target"),
            node_type=NodeType.DATA_TARGET,
            config=target_config,
            position_x=target_config.get("position_x", 0),
            position_y=target_config.get("position_y", 0)
        )
        
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        
        return node
    
    def connect_nodes(self, source_node_id: str, target_node_id: str, config: Dict[str, Any] = None) -> PipelineConnection:
        """Connect two nodes in the pipeline"""
        source_node = self.db.query(PipelineNode).filter(PipelineNode.id == source_node_id).first()
        target_node = self.db.query(PipelineNode).filter(PipelineNode.id == target_node_id).first()
        
        if not source_node or not target_node:
            raise ValueError("Source or target node not found")
        
        if source_node.pipeline_id != target_node.pipeline_id:
            raise ValueError("Nodes must be in the same pipeline")
        
        # Check for existing connection
        existing = self.db.query(PipelineConnection).filter(
            and_(
                PipelineConnection.source_node_id == source_node_id,
                PipelineConnection.target_node_id == target_node_id
            )
        ).first()
        
        if existing:
            raise ValueError("Connection already exists")
        
        connection = PipelineConnection(
            id=str(uuid.uuid4()),
            pipeline_id=source_node.pipeline_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            config=config or {}
        )
        
        self.db.add(connection)
        self.db.commit()
        self.db.refresh(connection)
        
        return connection
    
    def validate_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Validate pipeline structure and configuration"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        errors = []
        warnings = []
        
        # Check if pipeline has nodes
        if not pipeline.nodes:
            errors.append("Pipeline must have at least one node")
        
        # Check for data sources
        data_sources = [n for n in pipeline.nodes if n.node_type == NodeType.DATA_SOURCE]
        if not data_sources:
            errors.append("Pipeline must have at least one data source")
        
        # Check for data targets
        data_targets = [n for n in pipeline.nodes if n.node_type == NodeType.DATA_TARGET]
        if not data_targets:
            warnings.append("Pipeline should have at least one data target")
        
        # Check for disconnected nodes
        connected_nodes = set()
        for conn in pipeline.connections:
            connected_nodes.add(conn.source_node_id)
            connected_nodes.add(conn.target_node_id)
        
        disconnected = [n for n in pipeline.nodes if n.id not in connected_nodes and len(pipeline.nodes) > 1]
        if disconnected:
            warnings.append(f"Found {len(disconnected)} disconnected nodes")
        
        # Check for circular dependencies
        if self._has_circular_dependency(pipeline):
            errors.append("Pipeline contains circular dependencies")
        
        is_valid = len(errors) == 0
        
        # Save validation result
        validation_result = ValidationResult(
            id=str(uuid.uuid4()),
            pipeline_id=pipeline_id,
            validation_type="structure",
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
        
        self.db.add(validation_result)
        self.db.commit()
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "validation_id": validation_result.id
        }
    
    def _has_circular_dependency(self, pipeline: Pipeline) -> bool:
        """Check for circular dependencies in the pipeline"""
        # Build adjacency list
        graph = {}
        for node in pipeline.nodes:
            graph[node.id] = []
        
        for conn in pipeline.connections:
            if conn.source_node_id in graph:
                graph[conn.source_node_id].append(conn.target_node_id)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id):
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in graph.get(node_id, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in graph:
            if node_id not in visited:
                if has_cycle(node_id):
                    return True
        
        return False
    
    def get_pipeline_nodes(self, pipeline_id: str) -> List[PipelineNode]:
        """Get all nodes for a pipeline"""
        return self.db.query(PipelineNode).filter(PipelineNode.pipeline_id == pipeline_id).all()
    
    def get_pipeline_connections(self, pipeline_id: str) -> List[PipelineConnection]:
        """Get all connections for a pipeline"""
        return self.db.query(PipelineConnection).filter(PipelineConnection.pipeline_id == pipeline_id).all()
    
    def update_node_position(self, node_id: str, x: int, y: int) -> PipelineNode:
        """Update node position for visual layout"""
        node = self.db.query(PipelineNode).filter(PipelineNode.id == node_id).first()
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        node.position_x = x
        node.position_y = y
        self.db.commit()
        self.db.refresh(node)
        
        return node