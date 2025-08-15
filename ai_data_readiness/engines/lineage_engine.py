"""
Lineage Engine for AI Data Readiness Platform

This module provides comprehensive data lineage tracking, dataset versioning,
and model-to-dataset linking capabilities for complete data transformation tracking.
"""

import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import logging

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """Types of data transformations that can be tracked"""
    INGESTION = "ingestion"
    CLEANING = "cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    JOINING = "joining"
    NORMALIZATION = "normalization"
    ENCODING = "encoding"
    SPLITTING = "splitting"
    VALIDATION = "validation"


class LineageNodeType(Enum):
    """Types of nodes in the lineage graph"""
    DATASET = "dataset"
    TRANSFORMATION = "transformation"
    MODEL = "model"
    FEATURE = "feature"
    SOURCE = "source"


@dataclass
class LineageNode:
    """Represents a node in the data lineage graph"""
    id: str
    name: str
    node_type: LineageNodeType
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['node_type'] = self.node_type.value
        return data


@dataclass
class LineageEdge:
    """Represents a relationship between nodes in the lineage graph"""
    id: str
    source_node_id: str
    target_node_id: str
    transformation_type: TransformationType
    transformation_details: Dict[str, Any]
    created_at: datetime
    created_by: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['transformation_type'] = self.transformation_type.value
        return data


@dataclass
class DatasetVersion:
    """Represents a version of a dataset with change attribution"""
    version_id: str
    dataset_id: str
    version_number: str
    parent_version_id: Optional[str]
    changes: List[Dict[str, Any]]
    created_at: datetime
    created_by: str
    commit_message: str
    schema_hash: str
    data_hash: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class ModelDatasetLink:
    """Represents a link between a model and its training/validation datasets"""
    link_id: str
    model_id: str
    model_version: str
    dataset_id: str
    dataset_version: str
    usage_type: str  # 'training', 'validation', 'test', 'inference'
    created_at: datetime
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


class LineageEngine:
    """
    Engine for tracking complete data transformation lineage, dataset versioning,
    and model-to-dataset linking capabilities.
    """
    
    def __init__(self, storage_backend: Optional[Any] = None):
        """
        Initialize the LineageEngine
        
        Args:
            storage_backend: Backend storage system for persistence
        """
        self.storage_backend = storage_backend
        self.lineage_graph: Dict[str, LineageNode] = {}
        self.lineage_edges: Dict[str, LineageEdge] = {}
        self.dataset_versions: Dict[str, List[DatasetVersion]] = {}
        self.model_dataset_links: Dict[str, List[ModelDatasetLink]] = {}
        
    def create_dataset_node(
        self,
        dataset_id: str,
        name: str,
        metadata: Dict[str, Any],
        version: str = "1.0.0"
    ) -> LineageNode:
        """
        Create a new dataset node in the lineage graph
        
        Args:
            dataset_id: Unique identifier for the dataset
            name: Human-readable name for the dataset
            metadata: Additional metadata about the dataset
            version: Version of the dataset
            
        Returns:
            Created LineageNode
        """
        try:
            # Generate checksum for dataset metadata
            checksum = self._generate_checksum(metadata)
            
            node = LineageNode(
                id=dataset_id,
                name=name,
                node_type=LineageNodeType.DATASET,
                metadata=metadata,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version=version,
                checksum=checksum
            )
            
            self.lineage_graph[dataset_id] = node
            
            # Create initial version
            self._create_dataset_version(
                dataset_id=dataset_id,
                version_number=version,
                changes=[{"type": "initial_creation", "description": "Initial dataset creation"}],
                created_by="system",
                commit_message="Initial dataset creation",
                metadata=metadata
            )
            
            logger.info(f"Created dataset node: {dataset_id}")
            return node
            
        except Exception as e:
            logger.error(f"Error creating dataset node {dataset_id}: {str(e)}")
            raise
    
    def create_transformation_node(
        self,
        transformation_id: str,
        name: str,
        transformation_type: TransformationType,
        parameters: Dict[str, Any]
    ) -> LineageNode:
        """
        Create a transformation node in the lineage graph
        
        Args:
            transformation_id: Unique identifier for the transformation
            name: Human-readable name for the transformation
            transformation_type: Type of transformation
            parameters: Parameters used in the transformation
            
        Returns:
            Created LineageNode
        """
        try:
            metadata = {
                "transformation_type": transformation_type.value,
                "parameters": parameters,
                "execution_time": datetime.now().isoformat()
            }
            
            node = LineageNode(
                id=transformation_id,
                name=name,
                node_type=LineageNodeType.TRANSFORMATION,
                metadata=metadata,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version="1.0.0"
            )
            
            self.lineage_graph[transformation_id] = node
            logger.info(f"Created transformation node: {transformation_id}")
            return node
            
        except Exception as e:
            logger.error(f"Error creating transformation node {transformation_id}: {str(e)}")
            raise
    
    def add_transformation_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        transformation_type: TransformationType,
        transformation_details: Dict[str, Any],
        created_by: str
    ) -> LineageEdge:
        """
        Add a transformation edge between two nodes
        
        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            transformation_type: Type of transformation
            transformation_details: Details about the transformation
            created_by: User who created the transformation
            
        Returns:
            Created LineageEdge
        """
        try:
            edge_id = f"{source_node_id}_{target_node_id}_{uuid.uuid4().hex[:8]}"
            
            edge = LineageEdge(
                id=edge_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                transformation_type=transformation_type,
                transformation_details=transformation_details,
                created_at=datetime.now(),
                created_by=created_by
            )
            
            self.lineage_edges[edge_id] = edge
            logger.info(f"Added transformation edge: {edge_id}")
            return edge
            
        except Exception as e:
            logger.error(f"Error adding transformation edge: {str(e)}")
            raise
    
    def create_dataset_version(
        self,
        dataset_id: str,
        changes: List[Dict[str, Any]],
        created_by: str,
        commit_message: str,
        parent_version_id: Optional[str] = None
    ) -> DatasetVersion:
        """
        Create a new version of a dataset with change attribution
        
        Args:
            dataset_id: ID of the dataset
            changes: List of changes made in this version
            created_by: User who created the version
            commit_message: Description of the changes
            parent_version_id: ID of the parent version
            
        Returns:
            Created DatasetVersion
        """
        try:
            # Get current versions for this dataset
            current_versions = self.dataset_versions.get(dataset_id, [])
            
            # Generate new version number
            if not current_versions:
                version_number = "1.0.0"
            else:
                latest_version = max(current_versions, key=lambda v: v.created_at)
                major, minor, patch = map(int, latest_version.version_number.split('.'))
                version_number = f"{major}.{minor}.{patch + 1}"
            
            # Get dataset metadata for hashing
            dataset_node = self.lineage_graph.get(dataset_id)
            if not dataset_node:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            schema_hash = self._generate_checksum(dataset_node.metadata.get('schema', {}))
            data_hash = self._generate_checksum(changes)
            
            version = DatasetVersion(
                version_id=f"{dataset_id}_v{version_number}",
                dataset_id=dataset_id,
                version_number=version_number,
                parent_version_id=parent_version_id,
                changes=changes,
                created_at=datetime.now(),
                created_by=created_by,
                commit_message=commit_message,
                schema_hash=schema_hash,
                data_hash=data_hash,
                metadata=dataset_node.metadata
            )
            
            if dataset_id not in self.dataset_versions:
                self.dataset_versions[dataset_id] = []
            
            self.dataset_versions[dataset_id].append(version)
            
            # Update dataset node version
            dataset_node.version = version_number
            dataset_node.updated_at = datetime.now()
            
            logger.info(f"Created dataset version: {version.version_id}")
            return version
            
        except Exception as e:
            logger.error(f"Error creating dataset version for {dataset_id}: {str(e)}")
            raise
    
    def _create_dataset_version(
        self,
        dataset_id: str,
        version_number: str,
        changes: List[Dict[str, Any]],
        created_by: str,
        commit_message: str,
        metadata: Dict[str, Any]
    ) -> DatasetVersion:
        """Internal method to create dataset version"""
        schema_hash = self._generate_checksum(metadata.get('schema', {}))
        data_hash = self._generate_checksum(changes)
        
        version = DatasetVersion(
            version_id=f"{dataset_id}_v{version_number}",
            dataset_id=dataset_id,
            version_number=version_number,
            parent_version_id=None,
            changes=changes,
            created_at=datetime.now(),
            created_by=created_by,
            commit_message=commit_message,
            schema_hash=schema_hash,
            data_hash=data_hash,
            metadata=metadata
        )
        
        if dataset_id not in self.dataset_versions:
            self.dataset_versions[dataset_id] = []
        
        self.dataset_versions[dataset_id].append(version)
        return version
    
    def link_model_to_dataset(
        self,
        model_id: str,
        model_version: str,
        dataset_id: str,
        dataset_version: str,
        usage_type: str,
        performance_metrics: Dict[str, float]
    ) -> ModelDatasetLink:
        """
        Create a link between a model and dataset
        
        Args:
            model_id: ID of the model
            model_version: Version of the model
            dataset_id: ID of the dataset
            dataset_version: Version of the dataset
            usage_type: How the dataset was used ('training', 'validation', 'test', 'inference')
            performance_metrics: Performance metrics achieved with this dataset
            
        Returns:
            Created ModelDatasetLink
        """
        try:
            link_id = f"{model_id}_{dataset_id}_{uuid.uuid4().hex[:8]}"
            
            link = ModelDatasetLink(
                link_id=link_id,
                model_id=model_id,
                model_version=model_version,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                usage_type=usage_type,
                created_at=datetime.now(),
                performance_metrics=performance_metrics
            )
            
            if model_id not in self.model_dataset_links:
                self.model_dataset_links[model_id] = []
            
            self.model_dataset_links[model_id].append(link)
            logger.info(f"Created model-dataset link: {link_id}")
            return link
            
        except Exception as e:
            logger.error(f"Error linking model {model_id} to dataset {dataset_id}: {str(e)}")
            raise
    
    def get_dataset_lineage(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get complete lineage for a dataset
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary containing complete lineage information
        """
        try:
            if dataset_id not in self.lineage_graph:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Get all upstream and downstream nodes
            upstream_nodes = self._get_upstream_nodes(dataset_id)
            downstream_nodes = self._get_downstream_nodes(dataset_id)
            
            # Get all related edges
            related_edges = self._get_related_edges(dataset_id)
            
            # Get dataset versions
            versions = self.dataset_versions.get(dataset_id, [])
            
            # Get model links
            model_links = []
            for model_id, links in self.model_dataset_links.items():
                model_links.extend([link for link in links if link.dataset_id == dataset_id])
            
            lineage = {
                "dataset": self.lineage_graph[dataset_id].to_dict(),
                "upstream_nodes": [node.to_dict() for node in upstream_nodes],
                "downstream_nodes": [node.to_dict() for node in downstream_nodes],
                "transformations": [edge.to_dict() for edge in related_edges],
                "versions": [version.to_dict() for version in versions],
                "model_links": [link.to_dict() for link in model_links]
            }
            
            return lineage
            
        except Exception as e:
            logger.error(f"Error getting lineage for dataset {dataset_id}: {str(e)}")
            raise
    
    def get_model_datasets(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get all datasets linked to a model
        
        Args:
            model_id: ID of the model
            
        Returns:
            List of dataset information linked to the model
        """
        try:
            links = self.model_dataset_links.get(model_id, [])
            
            model_datasets = []
            for link in links:
                dataset_node = self.lineage_graph.get(link.dataset_id)
                if dataset_node:
                    dataset_info = {
                        "link": link.to_dict(),
                        "dataset": dataset_node.to_dict(),
                        "versions": [v.to_dict() for v in self.dataset_versions.get(link.dataset_id, [])]
                    }
                    model_datasets.append(dataset_info)
            
            return model_datasets
            
        except Exception as e:
            logger.error(f"Error getting datasets for model {model_id}: {str(e)}")
            raise
    
    def track_transformation(
        self,
        source_dataset_id: str,
        target_dataset_id: str,
        transformation_type: TransformationType,
        transformation_details: Dict[str, Any],
        created_by: str
    ) -> str:
        """
        Track a complete transformation from source to target dataset
        
        Args:
            source_dataset_id: ID of the source dataset
            target_dataset_id: ID of the target dataset
            transformation_type: Type of transformation
            transformation_details: Details about the transformation
            created_by: User who performed the transformation
            
        Returns:
            ID of the transformation edge
        """
        try:
            # Create transformation edge
            edge = self.add_transformation_edge(
                source_node_id=source_dataset_id,
                target_node_id=target_dataset_id,
                transformation_type=transformation_type,
                transformation_details=transformation_details,
                created_by=created_by
            )
            
            # Create new version for target dataset
            changes = [{
                "type": "transformation",
                "transformation_type": transformation_type.value,
                "source_dataset": source_dataset_id,
                "details": transformation_details
            }]
            
            self.create_dataset_version(
                dataset_id=target_dataset_id,
                changes=changes,
                created_by=created_by,
                commit_message=f"Applied {transformation_type.value} transformation"
            )
            
            return edge.id
            
        except Exception as e:
            logger.error(f"Error tracking transformation: {str(e)}")
            raise
    
    def _get_upstream_nodes(self, node_id: str) -> List[LineageNode]:
        """Get all upstream nodes for a given node"""
        upstream_nodes = []
        visited = set()
        
        def traverse_upstream(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            
            for edge in self.lineage_edges.values():
                if edge.target_node_id == current_id:
                    source_node = self.lineage_graph.get(edge.source_node_id)
                    if source_node:
                        upstream_nodes.append(source_node)
                        traverse_upstream(edge.source_node_id)
        
        traverse_upstream(node_id)
        return upstream_nodes
    
    def _get_downstream_nodes(self, node_id: str) -> List[LineageNode]:
        """Get all downstream nodes for a given node"""
        downstream_nodes = []
        visited = set()
        
        def traverse_downstream(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            
            for edge in self.lineage_edges.values():
                if edge.source_node_id == current_id:
                    target_node = self.lineage_graph.get(edge.target_node_id)
                    if target_node:
                        downstream_nodes.append(target_node)
                        traverse_downstream(edge.target_node_id)
        
        traverse_downstream(node_id)
        return downstream_nodes
    
    def _get_related_edges(self, node_id: str) -> List[LineageEdge]:
        """Get all edges related to a node"""
        related_edges = []
        
        for edge in self.lineage_edges.values():
            if edge.source_node_id == node_id or edge.target_node_id == node_id:
                related_edges.append(edge)
        
        return related_edges
    
    def _generate_checksum(self, data: Any) -> str:
        """Generate checksum for data"""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(data).encode()).hexdigest()
    
    def get_lineage_statistics(self) -> Dict[str, Any]:
        """Get statistics about the lineage graph"""
        try:
            stats = {
                "total_nodes": len(self.lineage_graph),
                "total_edges": len(self.lineage_edges),
                "total_dataset_versions": sum(len(versions) for versions in self.dataset_versions.values()),
                "total_model_links": sum(len(links) for links in self.model_dataset_links.values()),
                "node_types": {},
                "transformation_types": {}
            }
            
            # Count node types
            for node in self.lineage_graph.values():
                node_type = node.node_type.value
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            
            # Count transformation types
            for edge in self.lineage_edges.values():
                trans_type = edge.transformation_type.value
                stats["transformation_types"][trans_type] = stats["transformation_types"].get(trans_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting lineage statistics: {str(e)}")
            raise