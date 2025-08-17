"""
Unit tests for the PipelineBuilder class
"""
import pytest
from unittest.mock import Mock, MagicMock
from sqlalchemy.orm import Session
import uuid

from scrollintel.core.pipeline_builder import PipelineBuilder, ValidationError
from scrollintel.models.pipeline_models import (
    Pipeline, PipelineNode, PipelineConnection, 
    PipelineStatus, NodeType
)

class TestPipelineBuilder:
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = Mock()
        session.refresh = Mock()
        session.delete = Mock()
        session.query = Mock()
        return session
    
    @pytest.fixture
    def pipeline_builder(self, mock_db_session):
        """Create PipelineBuilder instance with mock session"""
        return PipelineBuilder(mock_db_session)
    
    def test_create_pipeline(self, pipeline_builder, mock_db_session):
        """Test creating a new pipeline"""
        # Act
        result = pipeline_builder.create_pipeline(
            name="Test Pipeline",
            description="Test Description",
            created_by="test_user"
        )
        
        # Assert
        assert result.name == "Test Pipeline"
        assert result.description == "Test Description"
        assert result.created_by == "test_user"
        assert result.status == PipelineStatus.DRAFT
        assert result.id is not None
        
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    def test_get_pipeline(self, pipeline_builder, mock_db_session):
        """Test getting a pipeline by ID"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        # Act
        result = pipeline_builder.get_pipeline("test-id")
        
        # Assert
        assert result == mock_pipeline
        mock_db_session.query.assert_called_with(Pipeline)
    
    def test_get_pipeline_not_found(self, pipeline_builder, mock_db_session):
        """Test getting a non-existent pipeline"""
        # Arrange
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Act
        result = pipeline_builder.get_pipeline("non-existent-id")
        
        # Assert
        assert result is None
    
    def test_list_pipelines(self, pipeline_builder, mock_db_session):
        """Test listing all pipelines"""
        # Arrange
        mock_pipelines = [Mock(spec=Pipeline), Mock(spec=Pipeline)]
        mock_db_session.query.return_value.all.return_value = mock_pipelines
        
        # Act
        result = pipeline_builder.list_pipelines()
        
        # Assert
        assert result == mock_pipelines
        mock_db_session.query.assert_called_with(Pipeline)
    
    def test_list_pipelines_filtered_by_creator(self, pipeline_builder, mock_db_session):
        """Test listing pipelines filtered by creator"""
        # Arrange
        mock_pipelines = [Mock(spec=Pipeline)]
        mock_query = mock_db_session.query.return_value
        mock_query.filter.return_value.all.return_value = mock_pipelines
        
        # Act
        result = pipeline_builder.list_pipelines(created_by="test_user")
        
        # Assert
        assert result == mock_pipelines
        mock_query.filter.assert_called_once()
    
    def test_update_pipeline(self, pipeline_builder, mock_db_session):
        """Test updating a pipeline"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.name = "Old Name"
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        # Act
        result = pipeline_builder.update_pipeline("test-id", name="New Name", description="New Description")
        
        # Assert
        assert mock_pipeline.name == "New Name"
        assert mock_pipeline.description == "New Description"
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    def test_update_pipeline_not_found(self, pipeline_builder, mock_db_session):
        """Test updating a non-existent pipeline"""
        # Arrange
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Pipeline test-id not found"):
            pipeline_builder.update_pipeline("test-id", name="New Name")
    
    def test_delete_pipeline(self, pipeline_builder, mock_db_session):
        """Test deleting a pipeline"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        # Act
        result = pipeline_builder.delete_pipeline("test-id")
        
        # Assert
        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_pipeline)
        mock_db_session.commit.assert_called_once()
    
    def test_delete_pipeline_not_found(self, pipeline_builder, mock_db_session):
        """Test deleting a non-existent pipeline"""
        # Arrange
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Act
        result = pipeline_builder.delete_pipeline("non-existent-id")
        
        # Assert
        assert result is False
    
    def test_add_data_source(self, pipeline_builder, mock_db_session):
        """Test adding a data source node"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        source_config = {
            "name": "Test Source",
            "sourceType": "database",
            "position_x": 100,
            "position_y": 200
        }
        
        # Act
        result = pipeline_builder.add_data_source("pipeline-id", source_config)
        
        # Assert
        assert result.name == "Test Source"
        assert result.node_type == NodeType.DATA_SOURCE
        assert result.position_x == 100
        assert result.position_y == 200
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    def test_add_data_source_pipeline_not_found(self, pipeline_builder, mock_db_session):
        """Test adding data source to non-existent pipeline"""
        # Arrange
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Pipeline test-id not found"):
            pipeline_builder.add_data_source("test-id", {})
    
    def test_add_transformation(self, pipeline_builder, mock_db_session):
        """Test adding a transformation node"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        transform_config = {
            "name": "Test Transform",
            "transformationType": "filter"
        }
        
        # Act
        result = pipeline_builder.add_transformation("pipeline-id", transform_config)
        
        # Assert
        assert result.name == "Test Transform"
        assert result.node_type == NodeType.TRANSFORMATION
        mock_db_session.add.assert_called_once()
    
    def test_connect_nodes(self, pipeline_builder, mock_db_session):
        """Test connecting two nodes"""
        # Arrange
        mock_source_node = Mock(spec=PipelineNode)
        mock_source_node.pipeline_id = "pipeline-id"
        mock_target_node = Mock(spec=PipelineNode)
        mock_target_node.pipeline_id = "pipeline-id"
        
        mock_db_session.query.return_value.filter.return_value.first.side_effect = [
            mock_source_node, mock_target_node, None  # None for existing connection check
        ]
        
        # Act
        result = pipeline_builder.connect_nodes("source-id", "target-id")
        
        # Assert
        assert result.source_node_id == "source-id"
        assert result.target_node_id == "target-id"
        assert result.pipeline_id == "pipeline-id"
        mock_db_session.add.assert_called_once()
    
    def test_connect_nodes_different_pipelines(self, pipeline_builder, mock_db_session):
        """Test connecting nodes from different pipelines"""
        # Arrange
        mock_source_node = Mock(spec=PipelineNode)
        mock_source_node.pipeline_id = "pipeline-1"
        mock_target_node = Mock(spec=PipelineNode)
        mock_target_node.pipeline_id = "pipeline-2"
        
        mock_db_session.query.return_value.filter.return_value.first.side_effect = [
            mock_source_node, mock_target_node
        ]
        
        # Act & Assert
        with pytest.raises(ValueError, match="Nodes must be in the same pipeline"):
            pipeline_builder.connect_nodes("source-id", "target-id")
    
    def test_connect_nodes_already_connected(self, pipeline_builder, mock_db_session):
        """Test connecting nodes that are already connected"""
        # Arrange
        mock_source_node = Mock(spec=PipelineNode)
        mock_source_node.pipeline_id = "pipeline-id"
        mock_target_node = Mock(spec=PipelineNode)
        mock_target_node.pipeline_id = "pipeline-id"
        mock_existing_connection = Mock(spec=PipelineConnection)
        
        mock_db_session.query.return_value.filter.return_value.first.side_effect = [
            mock_source_node, mock_target_node, mock_existing_connection
        ]
        
        # Act & Assert
        with pytest.raises(ValueError, match="Connection already exists"):
            pipeline_builder.connect_nodes("source-id", "target-id")
    
    def test_validate_pipeline_valid(self, pipeline_builder, mock_db_session):
        """Test validating a valid pipeline"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        mock_source_node = Mock(spec=PipelineNode)
        mock_source_node.id = "source-id"
        mock_source_node.node_type = NodeType.DATA_SOURCE
        mock_target_node = Mock(spec=PipelineNode)
        mock_target_node.id = "target-id"
        mock_target_node.node_type = NodeType.DATA_TARGET
        mock_pipeline.nodes = [mock_source_node, mock_target_node]
        
        mock_connection = Mock(spec=PipelineConnection)
        mock_connection.source_node_id = "source-id"
        mock_connection.target_node_id = "target-id"
        mock_pipeline.connections = [mock_connection]
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        # Act
        result = pipeline_builder.validate_pipeline("pipeline-id")
        
        # Assert
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        mock_db_session.add.assert_called_once()  # ValidationResult saved
    
    def test_validate_pipeline_no_nodes(self, pipeline_builder, mock_db_session):
        """Test validating a pipeline with no nodes"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.nodes = []
        mock_pipeline.connections = []
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        # Act
        result = pipeline_builder.validate_pipeline("pipeline-id")
        
        # Assert
        assert result["is_valid"] is False
        assert "Pipeline must have at least one node" in result["errors"]
    
    def test_validate_pipeline_no_data_source(self, pipeline_builder, mock_db_session):
        """Test validating a pipeline with no data source"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        mock_node = Mock(spec=PipelineNode)
        mock_node.node_type = NodeType.TRANSFORMATION
        mock_pipeline.nodes = [mock_node]
        mock_pipeline.connections = []
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_pipeline
        
        # Act
        result = pipeline_builder.validate_pipeline("pipeline-id")
        
        # Assert
        assert result["is_valid"] is False
        assert "Pipeline must have at least one data source" in result["errors"]
    
    def test_has_circular_dependency(self, pipeline_builder, mock_db_session):
        """Test circular dependency detection"""
        # Arrange
        mock_pipeline = Mock(spec=Pipeline)
        
        # Create nodes
        node1 = Mock(spec=PipelineNode)
        node1.id = "node1"
        node2 = Mock(spec=PipelineNode)
        node2.id = "node2"
        node3 = Mock(spec=PipelineNode)
        node3.id = "node3"
        mock_pipeline.nodes = [node1, node2, node3]
        
        # Create circular connections: node1 -> node2 -> node3 -> node1
        conn1 = Mock(spec=PipelineConnection)
        conn1.source_node_id = "node1"
        conn1.target_node_id = "node2"
        conn2 = Mock(spec=PipelineConnection)
        conn2.source_node_id = "node2"
        conn2.target_node_id = "node3"
        conn3 = Mock(spec=PipelineConnection)
        conn3.source_node_id = "node3"
        conn3.target_node_id = "node1"
        mock_pipeline.connections = [conn1, conn2, conn3]
        
        # Act
        result = pipeline_builder._has_circular_dependency(mock_pipeline)
        
        # Assert
        assert result is True
    
    def test_update_node_position(self, pipeline_builder, mock_db_session):
        """Test updating node position"""
        # Arrange
        mock_node = Mock(spec=PipelineNode)
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_node
        
        # Act
        result = pipeline_builder.update_node_position("node-id", 150, 250)
        
        # Assert
        assert mock_node.position_x == 150
        assert mock_node.position_y == 250
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    def test_update_node_position_not_found(self, pipeline_builder, mock_db_session):
        """Test updating position of non-existent node"""
        # Arrange
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Node node-id not found"):
            pipeline_builder.update_node_position("node-id", 150, 250)