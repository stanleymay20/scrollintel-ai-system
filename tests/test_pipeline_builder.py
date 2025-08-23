"""
Unit tests for Pipeline Builder functionality
Tests CRUD operations, validation, and pipeline management.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch
import uuid

from scrollintel.models.pipeline_models import (
    Base, Pipeline, PipelineNode, PipelineConnection, ComponentTemplate,
    NodeType, PipelineStatus, ValidationStatus
)
from scrollintel.core.pipeline_builder import (
    PipelineBuilder, DataSourceConfig, TransformConfig, ValidationResult
)


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def pipeline_builder(db_session):
    """Create PipelineBuilder instance with test database"""
    return PipelineBuilder(db_session)


@pytest.fixture
def sample_pipeline(pipeline_builder):
    """Create a sample pipeline for testing"""
    return pipeline_builder.create_pipeline(
        name="Test Pipeline",
        description="A test pipeline",
        created_by="test_user"
    )


class TestPipelineBuilder:
    """Test PipelineBuilder CRUD operations"""
    
    def test_create_pipeline(self, pipeline_builder):
        """Test pipeline creation"""
        pipeline = pipeline_builder.create_pipeline(
            name="Test Pipeline",
            description="Test Description",
            created_by="test_user"
        )
        
        assert pipeline.id is not None
        assert pipeline.name == "Test Pipeline"
        assert pipeline.description == "Test Description"
        assert pipeline.created_by == "test_user"
        assert pipeline.status == PipelineStatus.DRAFT
        assert pipeline.validation_status == ValidationStatus.PENDING
        assert pipeline.created_at is not None
    
    def test_get_pipeline(self, pipeline_builder, sample_pipeline):
        """Test pipeline retrieval"""
        retrieved = pipeline_builder.get_pipeline(sample_pipeline.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_pipeline.id
        assert retrieved.name == sample_pipeline.name
    
    def test_get_nonexistent_pipeline(self, pipeline_builder):
        """Test retrieving non-existent pipeline"""
        result = pipeline_builder.get_pipeline("nonexistent-id")
        assert result is None
    
    def test_list_pipelines(self, pipeline_builder):
        """Test pipeline listing"""
        # Create multiple pipelines
        p1 = pipeline_builder.create_pipeline("Pipeline 1", created_by="user1")
        p2 = pipeline_builder.create_pipeline("Pipeline 2", created_by="user2")
        
        # List all pipelines
        all_pipelines = pipeline_builder.list_pipelines()
        assert len(all_pipelines) == 2
        
        # Filter by user
        user1_pipelines = pipeline_builder.list_pipelines(created_by="user1")
        assert len(user1_pipelines) == 1
        assert user1_pipelines[0].id == p1.id
    
    def test_update_pipeline(self, pipeline_builder, sample_pipeline):
        """Test pipeline updates"""
        updated = pipeline_builder.update_pipeline(
            sample_pipeline.id,
            name="Updated Name",
            description="Updated Description"
        )
        
        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.description == "Updated Description"
        assert updated.updated_at is not None
    
    def test_delete_pipeline(self, pipeline_builder, sample_pipeline):
        """Test pipeline deletion"""
        success = pipeline_builder.delete_pipeline(sample_pipeline.id)
        assert success is True
        
        # Verify deletion
        retrieved = pipeline_builder.get_pipeline(sample_pipeline.id)
        assert retrieved is None


class TestDataSourceNodes:
    """Test data source node operations"""
    
    def test_add_postgresql_source(self, pipeline_builder, sample_pipeline):
        """Test adding PostgreSQL data source"""
        config = DataSourceConfig(
            source_type="postgresql",
            connection_params={
                "host": "localhost",
                "port": 5432,
                "database": "testdb",
                "username": "user",
                "password": "pass"
            },
            schema={"columns": ["id", "name", "email"]}
        )
        
        node = pipeline_builder.add_data_source(
            pipeline_id=sample_pipeline.id,
            source_config=config,
            name="PostgreSQL Source",
            position=(100, 100)
        )
        
        assert node is not None
        assert node.name == "PostgreSQL Source"
        assert node.node_type == NodeType.DATA_SOURCE
        assert node.component_type == "postgresql"
        assert node.position_x == 100
        assert node.position_y == 100
        assert node.config["source_type"] == "postgresql"
        assert "connection_params" in node.config
    
    def test_add_csv_source(self, pipeline_builder, sample_pipeline):
        """Test adding CSV data source"""
        config = DataSourceConfig(
            source_type="csv",
            connection_params={
                "file_path": "/path/to/data.csv",
                "delimiter": ",",
                "header": True
            }
        )
        
        node = pipeline_builder.add_data_source(
            pipeline_id=sample_pipeline.id,
            source_config=config,
            name="CSV Source"
        )
        
        assert node is not None
        assert node.component_type == "csv"
        assert node.config["connection_params"]["file_path"] == "/path/to/data.csv"
    
    def test_add_source_to_nonexistent_pipeline(self, pipeline_builder):
        """Test adding source to non-existent pipeline"""
        config = DataSourceConfig(
            source_type="postgresql",
            connection_params={"host": "localhost"}
        )
        
        node = pipeline_builder.add_data_source(
            pipeline_id="nonexistent-id",
            source_config=config
        )
        
        assert node is None


class TestTransformationNodes:
    """Test transformation node operations"""
    
    def test_add_filter_transformation(self, pipeline_builder, sample_pipeline):
        """Test adding filter transformation"""
        config = TransformConfig(
            transform_type="filter",
            parameters={
                "condition": "age > 18",
                "columns": ["age"]
            }
        )
        
        node = pipeline_builder.add_transformation(
            pipeline_id=sample_pipeline.id,
            transform_config=config,
            name="Age Filter",
            position=(200, 200)
        )
        
        assert node is not None
        assert node.name == "Age Filter"
        assert node.node_type == NodeType.TRANSFORMATION
        assert node.component_type == "filter"
        assert node.config["parameters"]["condition"] == "age > 18"
    
    def test_add_aggregation_transformation(self, pipeline_builder, sample_pipeline):
        """Test adding aggregation transformation"""
        config = TransformConfig(
            transform_type="aggregate",
            parameters={
                "group_by": ["department"],
                "aggregations": {
                    "salary": "avg",
                    "count": "count"
                }
            }
        )
        
        node = pipeline_builder.add_transformation(
            pipeline_id=sample_pipeline.id,
            transform_config=config,
            name="Department Aggregation"
        )
        
        assert node is not None
        assert node.component_type == "aggregate"
        assert "group_by" in node.config["parameters"]


class TestNodeConnections:
    """Test node connection operations"""
    
    def test_connect_nodes(self, pipeline_builder, sample_pipeline):
        """Test connecting two nodes"""
        # Create source node
        source_config = DataSourceConfig(
            source_type="csv",
            connection_params={"file_path": "data.csv"}
        )
        source_node = pipeline_builder.add_data_source(
            sample_pipeline.id, source_config, "Source"
        )
        
        # Create transformation node
        transform_config = TransformConfig(
            transform_type="filter",
            parameters={"condition": "active = true"}
        )
        transform_node = pipeline_builder.add_transformation(
            sample_pipeline.id, transform_config, "Filter"
        )
        
        # Connect nodes
        connection = pipeline_builder.connect_nodes(
            source_node.id, transform_node.id
        )
        
        assert connection is not None
        assert connection.source_node_id == source_node.id
        assert connection.target_node_id == transform_node.id
        assert connection.pipeline_id == sample_pipeline.id
    
    def test_connect_nonexistent_nodes(self, pipeline_builder):
        """Test connecting non-existent nodes"""
        connection = pipeline_builder.connect_nodes(
            "nonexistent-source", "nonexistent-target"
        )
        assert connection is None
    
    def test_duplicate_connection(self, pipeline_builder, sample_pipeline):
        """Test creating duplicate connection"""
        # Create nodes
        source_config = DataSourceConfig("csv", {"file_path": "data.csv"})
        source_node = pipeline_builder.add_data_source(sample_pipeline.id, source_config)
        
        transform_config = TransformConfig("filter", {"condition": "true"})
        transform_node = pipeline_builder.add_transformation(sample_pipeline.id, transform_config)
        
        # Create first connection
        conn1 = pipeline_builder.connect_nodes(source_node.id, transform_node.id)
        
        # Try to create duplicate
        conn2 = pipeline_builder.connect_nodes(source_node.id, transform_node.id)
        
        assert conn1 is not None
        assert conn2 is not None
        assert conn1.id == conn2.id  # Should return existing connection


class TestPipelineValidation:
    """Test pipeline validation functionality"""
    
    def test_validate_empty_pipeline(self, pipeline_builder, sample_pipeline):
        """Test validating empty pipeline"""
        result = pipeline_builder.validate_pipeline(sample_pipeline.id)
        
        assert not result.is_valid
        assert "must have at least one node" in str(result.errors)
    
    def test_validate_pipeline_without_sources(self, pipeline_builder, sample_pipeline):
        """Test validating pipeline without data sources"""
        # Add only transformation node
        transform_config = TransformConfig("filter", {"condition": "true"})
        pipeline_builder.add_transformation(sample_pipeline.id, transform_config)
        
        result = pipeline_builder.validate_pipeline(sample_pipeline.id)
        
        assert not result.is_valid
        assert "must have at least one data source" in str(result.errors)
    
    def test_validate_valid_pipeline(self, pipeline_builder, sample_pipeline):
        """Test validating valid pipeline"""
        # Add source
        source_config = DataSourceConfig("csv", {"file_path": "data.csv"})
        source_node = pipeline_builder.add_data_source(sample_pipeline.id, source_config)
        
        # Add transformation
        transform_config = TransformConfig("filter", {"condition": "active = true"})
        transform_node = pipeline_builder.add_transformation(sample_pipeline.id, transform_config)
        
        # Connect nodes
        pipeline_builder.connect_nodes(source_node.id, transform_node.id)
        
        result = pipeline_builder.validate_pipeline(sample_pipeline.id)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_pipeline_with_cycles(self, pipeline_builder, sample_pipeline):
        """Test detecting cycles in pipeline"""
        # Create nodes
        source_config = DataSourceConfig("csv", {"file_path": "data.csv"})
        node1 = pipeline_builder.add_data_source(sample_pipeline.id, source_config, "Node1")
        
        transform_config1 = TransformConfig("filter", {"condition": "true"})
        node2 = pipeline_builder.add_transformation(sample_pipeline.id, transform_config1, "Node2")
        
        transform_config2 = TransformConfig("map", {"mappings": {}})
        node3 = pipeline_builder.add_transformation(sample_pipeline.id, transform_config2, "Node3")
        
        # Create cycle: node1 -> node2 -> node3 -> node2
        pipeline_builder.connect_nodes(node1.id, node2.id)
        pipeline_builder.connect_nodes(node2.id, node3.id)
        pipeline_builder.connect_nodes(node3.id, node2.id)  # Creates cycle
        
        result = pipeline_builder.validate_pipeline(sample_pipeline.id)
        
        assert not result.is_valid
        assert "contains cycles" in str(result.errors)
    
    def test_validate_nonexistent_pipeline(self, pipeline_builder):
        """Test validating non-existent pipeline"""
        result = pipeline_builder.validate_pipeline("nonexistent-id")
        
        assert not result.is_valid
        assert "Pipeline not found" in str(result.errors)


class TestComponentTemplates:
    """Test component template operations"""
    
    def test_create_component_template(self, pipeline_builder):
        """Test creating component template"""
        template = pipeline_builder.create_component_template(
            name="PostgreSQL Source",
            node_type=NodeType.DATA_SOURCE,
            component_type="postgresql",
            description="PostgreSQL database source",
            category="Databases",
            default_config={
                "host": "localhost",
                "port": 5432
            }
        )
        
        assert template is not None
        assert template.name == "PostgreSQL Source"
        assert template.node_type == NodeType.DATA_SOURCE
        assert template.component_type == "postgresql"
        assert template.default_config["port"] == 5432
    
    def test_get_component_templates(self, pipeline_builder):
        """Test retrieving component templates"""
        # Create templates
        pipeline_builder.create_component_template(
            "PostgreSQL", NodeType.DATA_SOURCE, "postgresql", category="Databases"
        )
        pipeline_builder.create_component_template(
            "Filter", NodeType.TRANSFORMATION, "filter", category="Transformations"
        )
        
        # Get all templates
        all_templates = pipeline_builder.get_component_templates()
        assert len(all_templates) == 2
        
        # Filter by node type
        source_templates = pipeline_builder.get_component_templates(node_type=NodeType.DATA_SOURCE)
        assert len(source_templates) == 1
        assert source_templates[0].component_type == "postgresql"
        
        # Filter by category
        db_templates = pipeline_builder.get_component_templates(category="Databases")
        assert len(db_templates) == 1


class TestValidationHelpers:
    """Test validation helper methods"""
    
    def test_validation_result(self):
        """Test ValidationResult class"""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        
        result.add_error("Test error")
        assert result.is_valid is False
        assert "Test error" in result.errors
        
        result.add_warning("Test warning")
        assert "Test warning" in result.warnings
    
    def test_node_validation(self, pipeline_builder, sample_pipeline):
        """Test individual node validation"""
        # Create node with missing configuration
        source_config = DataSourceConfig("postgresql", {})  # Missing required params
        node = pipeline_builder.add_data_source(sample_pipeline.id, source_config)
        
        # Validate the node directly
        validation_result = pipeline_builder._validate_node(node)
        
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])