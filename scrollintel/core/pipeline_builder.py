"""
Data Pipeline Automation - Pipeline Builder
Core pipeline building functionality with CRUD operations and validation.
"""

from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import json
import uuid
from datetime import datetime, timezone

from scrollintel.models.pipeline_models import (
    Pipeline, PipelineNode, PipelineConnection, ComponentTemplate,
    NodeType, PipelineStatus, ValidationStatus
)


class ValidationResult:
    """Pipeline validation result"""
    def __init__(self, is_valid: bool = True, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)


class DataSourceConfig:
    """Data source configuration"""
    def __init__(self, source_type: str, connection_params: Dict[str, Any], schema: Dict[str, Any] = None):
        self.source_type = source_type
        self.connection_params = connection_params
        self.schema = schema or {}


class TransformConfig:
    """Transformation configuration"""
    def __init__(self, transform_type: str, parameters: Dict[str, Any], input_schema: Dict[str, Any] = None):
        self.transform_type = transform_type
        self.parameters = parameters
        self.input_schema = input_schema or {}


class PipelineBuilder:
    """Main pipeline builder class with CRUD operations"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_pipeline(self, name: str, description: str = "", created_by: str = "") -> Pipeline:
        """Create a new pipeline"""
        pipeline = Pipeline(
            name=name,
            description=description,
            created_by=created_by,
            status=PipelineStatus.DRAFT,
            validation_status=ValidationStatus.PENDING
        )
        
        self.db.add(pipeline)
        self.db.commit()
        self.db.refresh(pipeline)
        
        return pipeline
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID"""
        return self.db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
    
    def list_pipelines(self, status: PipelineStatus = None, created_by: str = None) -> List[Pipeline]:
        """List pipelines with optional filtering"""
        query = self.db.query(Pipeline)
        
        if status:
            query = query.filter(Pipeline.status == status)
        if created_by:
            query = query.filter(Pipeline.created_by == created_by)
        
        return query.order_by(Pipeline.created_at.desc()).all()
    
    def update_pipeline(self, pipeline_id: str, **kwargs) -> Optional[Pipeline]:
        """Update pipeline properties"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return None
        
        for key, value in kwargs.items():
            if hasattr(pipeline, key):
                setattr(pipeline, key, value)
        
        pipeline.updated_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(pipeline)
        
        return pipeline
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete pipeline and all associated nodes/connections"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return False
        
        self.db.delete(pipeline)
        self.db.commit()
        return True
    
    def add_data_source(self, pipeline_id: str, source_config: DataSourceConfig, 
                       name: str = "", position: Tuple[int, int] = (0, 0)) -> Optional[PipelineNode]:
        """Add a data source node to the pipeline"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return None
        
        node_name = name or f"{source_config.source_type}_source"
        
        node = PipelineNode(
            pipeline_id=pipeline_id,
            name=node_name,
            node_type=NodeType.DATA_SOURCE,
            component_type=source_config.source_type,
            position_x=position[0],
            position_y=position[1],
            config={
                "connection_params": source_config.connection_params,
                "source_type": source_config.source_type
            },
            output_schema=source_config.schema
        )
        
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        
        # Trigger pipeline validation
        self._update_pipeline_validation_status(pipeline_id)
        
        return node
    
    def add_transformation(self, pipeline_id: str, transform_config: TransformConfig,
                          name: str = "", position: Tuple[int, int] = (0, 0)) -> Optional[PipelineNode]:
        """Add a transformation node to the pipeline"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return None
        
        node_name = name or f"{transform_config.transform_type}_transform"
        
        node = PipelineNode(
            pipeline_id=pipeline_id,
            name=node_name,
            node_type=NodeType.TRANSFORMATION,
            component_type=transform_config.transform_type,
            position_x=position[0],
            position_y=position[1],
            config={
                "parameters": transform_config.parameters,
                "transform_type": transform_config.transform_type
            },
            input_schema=transform_config.input_schema
        )
        
        self.db.add(node)
        self.db.commit()
        self.db.refresh(node)
        
        # Trigger pipeline validation
        self._update_pipeline_validation_status(pipeline_id)
        
        return node
    
    def connect_nodes(self, source_node_id: str, target_node_id: str,
                     source_port: str = "output", target_port: str = "input") -> Optional[PipelineConnection]:
        """Create a connection between two nodes"""
        source_node = self.db.query(PipelineNode).filter(PipelineNode.id == source_node_id).first()
        target_node = self.db.query(PipelineNode).filter(PipelineNode.id == target_node_id).first()
        
        if not source_node or not target_node:
            return None
        
        if source_node.pipeline_id != target_node.pipeline_id:
            return None  # Nodes must be in the same pipeline
        
        # Check if connection already exists
        existing = self.db.query(PipelineConnection).filter(
            and_(
                PipelineConnection.source_node_id == source_node_id,
                PipelineConnection.target_node_id == target_node_id
            )
        ).first()
        
        if existing:
            return existing
        
        connection = PipelineConnection(
            pipeline_id=source_node.pipeline_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_port=source_port,
            target_port=target_port
        )
        
        self.db.add(connection)
        self.db.commit()
        self.db.refresh(connection)
        
        # Trigger pipeline validation
        self._update_pipeline_validation_status(source_node.pipeline_id)
        
        return connection
    
    def validate_pipeline(self, pipeline_id: str) -> ValidationResult:
        """Validate pipeline structure and configuration"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            result = ValidationResult(False)
            result.add_error("Pipeline not found")
            return result
        
        result = ValidationResult()
        
        # Get all nodes and connections
        nodes = self.db.query(PipelineNode).filter(PipelineNode.pipeline_id == pipeline_id).all()
        connections = self.db.query(PipelineConnection).filter(PipelineConnection.pipeline_id == pipeline_id).all()
        
        # Validate basic structure
        if not nodes:
            result.add_error("Pipeline must have at least one node")
            return result
        
        # Check for data sources
        source_nodes = [n for n in nodes if n.node_type == NodeType.DATA_SOURCE]
        if not source_nodes:
            result.add_error("Pipeline must have at least one data source")
        
        # Validate node configurations
        for node in nodes:
            node_validation = self._validate_node(node)
            if not node_validation.is_valid:
                result.errors.extend([f"Node {node.name}: {error}" for error in node_validation.errors])
                result.is_valid = False
            result.warnings.extend([f"Node {node.name}: {warning}" for warning in node_validation.warnings])
        
        # Validate connections
        for connection in connections:
            conn_validation = self._validate_connection(connection, nodes)
            if not conn_validation.is_valid:
                result.errors.extend([f"Connection {connection.id}: {error}" for error in conn_validation.errors])
                result.is_valid = False
            result.warnings.extend([f"Connection {connection.id}: {warning}" for warning in conn_validation.warnings])
        
        # Check for cycles
        if self._has_cycles(nodes, connections):
            result.add_error("Pipeline contains cycles")
        
        # Check for disconnected nodes
        disconnected = self._find_disconnected_nodes(nodes, connections)
        if disconnected:
            result.add_warning(f"Disconnected nodes found: {[n.name for n in disconnected]}")
        
        return result
    
    def get_component_templates(self, node_type: NodeType = None, category: str = None) -> List[ComponentTemplate]:
        """Get available component templates"""
        query = self.db.query(ComponentTemplate).filter(ComponentTemplate.is_active == True)
        
        if node_type:
            query = query.filter(ComponentTemplate.node_type == node_type)
        if category:
            query = query.filter(ComponentTemplate.category == category)
        
        return query.order_by(ComponentTemplate.name).all()
    
    def create_component_template(self, name: str, node_type: NodeType, component_type: str,
                                description: str = "", category: str = "", 
                                default_config: Dict[str, Any] = None) -> ComponentTemplate:
        """Create a new component template"""
        template = ComponentTemplate(
            name=name,
            description=description,
            category=category,
            node_type=node_type,
            component_type=component_type,
            default_config=default_config or {}
        )
        
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        
        return template
    
    def _validate_node(self, node: PipelineNode) -> ValidationResult:
        """Validate individual node configuration"""
        result = ValidationResult()
        
        # Check required configuration
        if not node.config:
            result.add_error("Node configuration is missing")
            return result
        
        # Validate based on node type
        if node.node_type == NodeType.DATA_SOURCE:
            if "source_type" not in node.config:
                result.add_error("Data source type is required")
            if "connection_params" not in node.config:
                result.add_error("Connection parameters are required")
            else:
                # Validate connection parameters based on source type
                conn_params = node.config["connection_params"]
                source_type = node.config.get("source_type", "")
                
                if source_type in ["postgresql", "mysql", "sqlserver", "oracle"]:
                    required_params = ["host", "database"]
                    for param in required_params:
                        if not conn_params.get(param):
                            result.add_error(f"Required parameter '{param}' is missing for {source_type} source")
                
                elif source_type in ["csv", "json", "parquet", "excel"]:
                    if not conn_params.get("file_path"):
                        result.add_error("File path is required for file-based sources")
                
                elif source_type in ["rest_api", "graphql"]:
                    if not conn_params.get("url"):
                        result.add_error("URL is required for API sources")
        
        elif node.node_type == NodeType.TRANSFORMATION:
            if "transform_type" not in node.config:
                result.add_error("Transformation type is required")
            if "parameters" not in node.config:
                result.add_error("Transformation parameters are required")
        
        return result
    
    def _validate_connection(self, connection: PipelineConnection, nodes: List[PipelineNode]) -> ValidationResult:
        """Validate individual connection"""
        result = ValidationResult()
        
        # Find source and target nodes
        source_node = next((n for n in nodes if n.id == connection.source_node_id), None)
        target_node = next((n for n in nodes if n.id == connection.target_node_id), None)
        
        if not source_node:
            result.add_error("Source node not found")
        if not target_node:
            result.add_error("Target node not found")
        
        if source_node and target_node:
            # Check schema compatibility (basic check)
            if source_node.output_schema and target_node.input_schema:
                # This is a simplified schema check - in practice, you'd want more sophisticated validation
                if not self._schemas_compatible(source_node.output_schema, target_node.input_schema):
                    result.add_warning("Schema compatibility issues detected")
        
        return result
    
    def _schemas_compatible(self, output_schema: Dict[str, Any], input_schema: Dict[str, Any]) -> bool:
        """Check if output schema is compatible with input schema"""
        # Simplified compatibility check
        # In practice, this would be much more sophisticated
        return True  # Placeholder implementation
    
    def _has_cycles(self, nodes: List[PipelineNode], connections: List[PipelineConnection]) -> bool:
        """Check if pipeline has cycles using DFS"""
        # Build adjacency list
        graph = {node.id: [] for node in nodes}
        for conn in connections:
            graph[conn.source_node_id].append(conn.target_node_id)
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in nodes:
            if node.id not in visited:
                if dfs(node.id):
                    return True
        
        return False
    
    def _find_disconnected_nodes(self, nodes: List[PipelineNode], connections: List[PipelineConnection]) -> List[PipelineNode]:
        """Find nodes that are not connected to any other nodes"""
        connected_nodes = set()
        
        for conn in connections:
            connected_nodes.add(conn.source_node_id)
            connected_nodes.add(conn.target_node_id)
        
        return [node for node in nodes if node.id not in connected_nodes and len(nodes) > 1]
    
    def _update_pipeline_validation_status(self, pipeline_id: str):
        """Update pipeline validation status after changes"""
        validation_result = self.validate_pipeline(pipeline_id)
        
        if validation_result.is_valid:
            status = ValidationStatus.VALID
        elif validation_result.errors:
            status = ValidationStatus.INVALID
        else:
            status = ValidationStatus.WARNING
        
        self.update_pipeline(pipeline_id, validation_status=status)