"""
Tests for LineageEngine in AI Data Readiness Platform
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import json

from ai_data_readiness.engines.lineage_engine import (
    LineageEngine, 
    TransformationType, 
    LineageNodeType,
    LineageNode,
    LineageEdge,
    DatasetVersion,
    ModelDatasetLink
)


class TestLineageEngine:
    """Test cases for LineageEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = LineageEngine()
        
        # Create test datasets
        self.dataset1_metadata = {
            "schema": {"columns": ["id", "name", "value"]},
            "size": 1000,
            "format": "csv"
        }
        
        self.dataset2_metadata = {
            "schema": {"columns": ["id", "name", "processed_value"]},
            "size": 1000,
            "format": "parquet"
        }
    
    def test_create_dataset_node(self):
        """Test creating a dataset node"""
        node = self.engine.create_dataset_node(
            dataset_id="dataset_1",
            name="Test Dataset 1",
            metadata=self.dataset1_metadata,
            version="1.0.0"
        )
        
        assert node.id == "dataset_1"
        assert node.name == "Test Dataset 1"
        assert node.node_type == LineageNodeType.DATASET
        assert node.version == "1.0.0"
        assert node.checksum is not None
        assert "dataset_1" in self.engine.lineage_graph
        assert "dataset_1" in self.engine.dataset_versions
        assert len(self.engine.dataset_versions["dataset_1"]) == 1
    
    def test_create_transformation_node(self):
        """Test creating a transformation node"""
        parameters = {"method": "normalize", "scale": "standard"}
        
        node = self.engine.create_transformation_node(
            transformation_id="transform_1",
            name="Normalization Transform",
            transformation_type=TransformationType.NORMALIZATION,
            parameters=parameters
        )
        
        assert node.id == "transform_1"
        assert node.name == "Normalization Transform"
        assert node.node_type == LineageNodeType.TRANSFORMATION
        assert node.metadata["transformation_type"] == "normalization"
        assert node.metadata["parameters"] == parameters
        assert "transform_1" in self.engine.lineage_graph
    
    def test_add_transformation_edge(self):
        """Test adding a transformation edge"""
        # Create source and target nodes
        self.engine.create_dataset_node("source", "Source Dataset", self.dataset1_metadata)
        self.engine.create_dataset_node("target", "Target Dataset", self.dataset2_metadata)
        
        transformation_details = {"operation": "column_rename", "mapping": {"value": "processed_value"}}
        
        edge = self.engine.add_transformation_edge(
            source_node_id="source",
            target_node_id="target",
            transformation_type=TransformationType.CLEANING,
            transformation_details=transformation_details,
            created_by="test_user"
        )
        
        assert edge.source_node_id == "source"
        assert edge.target_node_id == "target"
        assert edge.transformation_type == TransformationType.CLEANING
        assert edge.transformation_details == transformation_details
        assert edge.created_by == "test_user"
        assert edge.id in self.engine.lineage_edges
    
    def test_create_dataset_version(self):
        """Test creating a dataset version"""
        # Create initial dataset
        self.engine.create_dataset_node("dataset_1", "Test Dataset", self.dataset1_metadata)
        
        changes = [
            {"type": "column_added", "column": "new_column", "description": "Added new column"}
        ]
        
        version = self.engine.create_dataset_version(
            dataset_id="dataset_1",
            changes=changes,
            created_by="test_user",
            commit_message="Added new column for analysis"
        )
        
        assert version.dataset_id == "dataset_1"
        assert version.version_number == "1.0.1"  # Should increment from initial 1.0.0
        assert version.changes == changes
        assert version.created_by == "test_user"
        assert version.commit_message == "Added new column for analysis"
        assert len(self.engine.dataset_versions["dataset_1"]) == 2  # Initial + new version
    
    def test_link_model_to_dataset(self):
        """Test linking a model to a dataset"""
        # Create dataset
        self.engine.create_dataset_node("dataset_1", "Training Dataset", self.dataset1_metadata)
        
        performance_metrics = {"accuracy": 0.95, "precision": 0.92, "recall": 0.88}
        
        link = self.engine.link_model_to_dataset(
            model_id="model_1",
            model_version="2.1.0",
            dataset_id="dataset_1",
            dataset_version="1.0.0",
            usage_type="training",
            performance_metrics=performance_metrics
        )
        
        assert link.model_id == "model_1"
        assert link.model_version == "2.1.0"
        assert link.dataset_id == "dataset_1"
        assert link.dataset_version == "1.0.0"
        assert link.usage_type == "training"
        assert link.performance_metrics == performance_metrics
        assert "model_1" in self.engine.model_dataset_links
    
    def test_get_dataset_lineage(self):
        """Test getting complete dataset lineage"""
        # Create a lineage chain: source -> transform -> target
        self.engine.create_dataset_node("source", "Source Dataset", self.dataset1_metadata)
        self.engine.create_dataset_node("target", "Target Dataset", self.dataset2_metadata)
        
        self.engine.add_transformation_edge(
            source_node_id="source",
            target_node_id="target",
            transformation_type=TransformationType.FEATURE_ENGINEERING,
            transformation_details={"operation": "feature_creation"},
            created_by="test_user"
        )
        
        # Link model to target dataset
        self.engine.link_model_to_dataset(
            model_id="model_1",
            model_version="1.0.0",
            dataset_id="target",
            dataset_version="1.0.0",
            usage_type="training",
            performance_metrics={"accuracy": 0.9}
        )
        
        lineage = self.engine.get_dataset_lineage("target")
        
        assert "dataset" in lineage
        assert "upstream_nodes" in lineage
        assert "downstream_nodes" in lineage
        assert "transformations" in lineage
        assert "versions" in lineage
        assert "model_links" in lineage
        
        assert lineage["dataset"]["id"] == "target"
        assert len(lineage["upstream_nodes"]) == 1
        assert lineage["upstream_nodes"][0]["id"] == "source"
        assert len(lineage["transformations"]) == 1
        assert len(lineage["model_links"]) == 1
    
    def test_get_model_datasets(self):
        """Test getting all datasets linked to a model"""
        # Create datasets
        self.engine.create_dataset_node("train_data", "Training Dataset", self.dataset1_metadata)
        self.engine.create_dataset_node("val_data", "Validation Dataset", self.dataset2_metadata)
        
        # Link model to both datasets
        self.engine.link_model_to_dataset(
            model_id="model_1",
            model_version="1.0.0",
            dataset_id="train_data",
            dataset_version="1.0.0",
            usage_type="training",
            performance_metrics={"accuracy": 0.9}
        )
        
        self.engine.link_model_to_dataset(
            model_id="model_1",
            model_version="1.0.0",
            dataset_id="val_data",
            dataset_version="1.0.0",
            usage_type="validation",
            performance_metrics={"accuracy": 0.85}
        )
        
        model_datasets = self.engine.get_model_datasets("model_1")
        
        assert len(model_datasets) == 2
        dataset_ids = [item["dataset"]["id"] for item in model_datasets]
        assert "train_data" in dataset_ids
        assert "val_data" in dataset_ids
    
    def test_track_transformation(self):
        """Test tracking a complete transformation"""
        # Create source and target datasets
        self.engine.create_dataset_node("source", "Source Dataset", self.dataset1_metadata)
        self.engine.create_dataset_node("target", "Target Dataset", self.dataset2_metadata)
        
        transformation_details = {
            "operation": "aggregation",
            "group_by": ["id"],
            "aggregations": {"value": "mean"}
        }
        
        edge_id = self.engine.track_transformation(
            source_dataset_id="source",
            target_dataset_id="target",
            transformation_type=TransformationType.AGGREGATION,
            transformation_details=transformation_details,
            created_by="test_user"
        )
        
        assert edge_id in self.engine.lineage_edges
        
        # Check that target dataset has a new version
        target_versions = self.engine.dataset_versions["target"]
        assert len(target_versions) == 2  # Initial + transformation version
        
        latest_version = max(target_versions, key=lambda v: v.created_at)
        assert latest_version.changes[0]["type"] == "transformation"
        assert latest_version.changes[0]["transformation_type"] == "aggregation"
    
    def test_get_lineage_statistics(self):
        """Test getting lineage statistics"""
        # Create some nodes and edges
        self.engine.create_dataset_node("dataset_1", "Dataset 1", self.dataset1_metadata)
        self.engine.create_dataset_node("dataset_2", "Dataset 2", self.dataset2_metadata)
        self.engine.create_transformation_node("transform_1", "Transform 1", TransformationType.CLEANING, {})
        
        self.engine.add_transformation_edge(
            source_node_id="dataset_1",
            target_node_id="dataset_2",
            transformation_type=TransformationType.CLEANING,
            transformation_details={},
            created_by="test_user"
        )
        
        self.engine.link_model_to_dataset(
            model_id="model_1",
            model_version="1.0.0",
            dataset_id="dataset_1",
            dataset_version="1.0.0",
            usage_type="training",
            performance_metrics={}
        )
        
        stats = self.engine.get_lineage_statistics()
        
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 1
        assert stats["total_dataset_versions"] == 2  # One for each dataset
        assert stats["total_model_links"] == 1
        assert "dataset" in stats["node_types"]
        assert "transformation" in stats["node_types"]
        assert "cleaning" in stats["transformation_types"]
    
    def test_upstream_nodes_traversal(self):
        """Test upstream nodes traversal"""
        # Create a chain: A -> B -> C
        self.engine.create_dataset_node("A", "Dataset A", self.dataset1_metadata)
        self.engine.create_dataset_node("B", "Dataset B", self.dataset1_metadata)
        self.engine.create_dataset_node("C", "Dataset C", self.dataset1_metadata)
        
        self.engine.add_transformation_edge("A", "B", TransformationType.CLEANING, {}, "user")
        self.engine.add_transformation_edge("B", "C", TransformationType.FEATURE_ENGINEERING, {}, "user")
        
        upstream_nodes = self.engine._get_upstream_nodes("C")
        upstream_ids = [node.id for node in upstream_nodes]
        
        assert "A" in upstream_ids
        assert "B" in upstream_ids
        assert len(upstream_nodes) == 2
    
    def test_downstream_nodes_traversal(self):
        """Test downstream nodes traversal"""
        # Create a chain: A -> B -> C
        self.engine.create_dataset_node("A", "Dataset A", self.dataset1_metadata)
        self.engine.create_dataset_node("B", "Dataset B", self.dataset1_metadata)
        self.engine.create_dataset_node("C", "Dataset C", self.dataset1_metadata)
        
        self.engine.add_transformation_edge("A", "B", TransformationType.CLEANING, {}, "user")
        self.engine.add_transformation_edge("B", "C", TransformationType.FEATURE_ENGINEERING, {}, "user")
        
        downstream_nodes = self.engine._get_downstream_nodes("A")
        downstream_ids = [node.id for node in downstream_nodes]
        
        assert "B" in downstream_ids
        assert "C" in downstream_ids
        assert len(downstream_nodes) == 2
    
    def test_checksum_generation(self):
        """Test checksum generation for data integrity"""
        data1 = {"key": "value", "number": 123}
        data2 = {"number": 123, "key": "value"}  # Same data, different order
        data3 = {"key": "different", "number": 123}
        
        checksum1 = self.engine._generate_checksum(data1)
        checksum2 = self.engine._generate_checksum(data2)
        checksum3 = self.engine._generate_checksum(data3)
        
        assert checksum1 == checksum2  # Same data should have same checksum
        assert checksum1 != checksum3  # Different data should have different checksum
        assert len(checksum1) == 64  # SHA256 produces 64-character hex string
    
    def test_error_handling_invalid_dataset(self):
        """Test error handling for invalid dataset operations"""
        with pytest.raises(ValueError, match="Dataset nonexistent not found"):
            self.engine.get_dataset_lineage("nonexistent")
    
    def test_version_numbering(self):
        """Test proper version numbering for datasets"""
        self.engine.create_dataset_node("dataset_1", "Test Dataset", self.dataset1_metadata)
        
        # Create multiple versions
        for i in range(3):
            self.engine.create_dataset_version(
                dataset_id="dataset_1",
                changes=[{"type": "update", "iteration": i}],
                created_by="test_user",
                commit_message=f"Update {i}"
            )
        
        versions = self.engine.dataset_versions["dataset_1"]
        version_numbers = [v.version_number for v in versions]
        
        assert "1.0.0" in version_numbers  # Initial version
        assert "1.0.1" in version_numbers
        assert "1.0.2" in version_numbers
        assert "1.0.3" in version_numbers
        assert len(versions) == 4  # Initial + 3 updates
    
    def test_complex_lineage_scenario(self):
        """Test a complex lineage scenario with multiple transformations and models"""
        # Create datasets
        datasets = ["raw_data", "cleaned_data", "features", "train_set", "test_set"]
        for dataset_id in datasets:
            self.engine.create_dataset_node(dataset_id, f"Dataset {dataset_id}", self.dataset1_metadata)
        
        # Create transformation chain
        transformations = [
            ("raw_data", "cleaned_data", TransformationType.CLEANING),
            ("cleaned_data", "features", TransformationType.FEATURE_ENGINEERING),
            ("features", "train_set", TransformationType.SPLITTING),
            ("features", "test_set", TransformationType.SPLITTING)
        ]
        
        for source, target, trans_type in transformations:
            self.engine.add_transformation_edge(source, target, trans_type, {}, "user")
        
        # Link models to datasets
        self.engine.link_model_to_dataset("model_v1", "1.0.0", "train_set", "1.0.0", "training", {"accuracy": 0.8})
        self.engine.link_model_to_dataset("model_v1", "1.0.0", "test_set", "1.0.0", "testing", {"accuracy": 0.75})
        self.engine.link_model_to_dataset("model_v2", "2.0.0", "train_set", "1.0.0", "training", {"accuracy": 0.9})
        
        # Test lineage for train_set
        lineage = self.engine.get_dataset_lineage("train_set")
        
        assert len(lineage["upstream_nodes"]) == 3  # raw_data, cleaned_data, features
        assert len(lineage["model_links"]) == 2  # Two model versions
        
        # Test model datasets
        model_v1_datasets = self.engine.get_model_datasets("model_v1")
        model_v2_datasets = self.engine.get_model_datasets("model_v2")
        
        assert len(model_v1_datasets) == 2  # train_set and test_set
        assert len(model_v2_datasets) == 1  # only train_set
        
        # Test statistics
        stats = self.engine.get_lineage_statistics()
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] == 4
        assert stats["total_model_links"] == 3


if __name__ == "__main__":
    pytest.main([__file__])