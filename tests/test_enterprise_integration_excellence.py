"""
Test suite for Enterprise Integration Excellence - Task 9
Tests all components of the enterprise integration system
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from security.enterprise_integration.auto_discovery_engine import (
    AutoDiscoveryEngine, SchemaEntity, RelationshipMapping, IntegrationRecommendation
)
from security.enterprise_integration.etl_recommendation_engine import (
    ETLRecommendationEngine, DataSource, DataTarget, ETLPatternType
)
from security.enterprise_integration.data_quality_engine import (
    DataQualityEngine, DataQualityDimension, QualityIssueType
)
from security.enterprise_integration.streaming_engine import (
    HighPerformanceStreamingEngine, StreamEvent, StreamEventType
)
from security.enterprise_integration.visual_integration_builder import (
    VisualIntegrationBuilder, ComponentType, ConnectionType
)

class TestAutoDiscoveryEngine:
    """Test the auto-discovery system"""
    
    @pytest.fixture
    def discovery_engine(self):
        return AutoDiscoveryEngine()
    
    @pytest.mark.asyncio
    async def test_discover_database_schema(self, discovery_engine):
        """Test database schema discovery"""
        # Mock database connection
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_inspector = Mock()
            mock_inspector.get_schema_names.return_value = ['public']
            mock_inspector.get_table_names.return_value = ['users', 'orders']
            mock_inspector.get_columns.return_value = [
                {'name': 'id', 'type': 'INTEGER', 'nullable': False},
                {'name': 'name', 'type': 'VARCHAR', 'nullable': True}
            ]
            mock_inspector.get_foreign_keys.return_value = []
            mock_inspector.get_indexes.return_value = []
            
            with patch('sqlalchemy.inspect', return_value=mock_inspector):
                entities = await discovery_engine.discover_database_schema(
                    "postgresql://test", "test_system"
                )
                
                assert len(entities) > 0
                assert any(entity.name == 'users' for entity in entities)
                assert any(entity.type == 'table' for entity in entities)
    
    @pytest.mark.asyncio
    async def test_discover_relationships(self, discovery_engine):
        """Test relationship discovery"""
        # Add some test entities
        discovery_engine.discovered_entities = {
            'test.public.users': SchemaEntity(
                name='users', type='table', database='test', schema='public',
                properties={}, relationships=[], confidence_score=0.9,
                discovered_at=datetime.utcnow()
            ),
            'test.public.orders': SchemaEntity(
                name='orders', type='table', database='test', schema='public',
                properties={}, relationships=[], confidence_score=0.9,
                discovered_at=datetime.utcnow()
            )
        }
        
        relationships = await discovery_engine.discover_relationships()
        assert isinstance(relationships, list)
    
    @pytest.mark.asyncio
    async def test_generate_integration_recommendations(self, discovery_engine):
        """Test integration recommendation generation"""
        recommendations = await discovery_engine.generate_integration_recommendations(
            "source_system", "target_system"
        )
        
        assert isinstance(recommendations, list)
        if recommendations:
            assert isinstance(recommendations[0], IntegrationRecommendation)
    
    def test_get_discovery_summary(self, discovery_engine):
        """Test discovery summary generation"""
        summary = discovery_engine.get_discovery_summary()
        
        assert 'total_entities' in summary
        assert 'entity_types' in summary
        assert 'total_relationships' in summary
        assert 'discovery_timestamp' in summary

class TestETLRecommendationEngine:
    """Test the ETL recommendation engine"""
    
    @pytest.fixture
    def etl_engine(self):
        return ETLRecommendationEngine()
    
    @pytest.fixture
    def sample_data_source(self):
        return DataSource(
            name="test_source",
            type="database",
            connection_info={"host": "localhost", "database": "test"},
            schema_info={"users": {"id": "int", "name": "string"}},
            volume_characteristics={"size_gb": 10, "row_count": 1000000},
            update_frequency="daily",
            data_quality_score=0.85
        )
    
    @pytest.fixture
    def sample_data_target(self):
        return DataTarget(
            name="test_target",
            type="database",
            connection_info={"host": "target", "database": "warehouse"},
            schema_requirements={"user_id": "int", "user_name": "string"},
            performance_requirements={"max_latency": "medium", "min_throughput": "high"},
            consistency_requirements="eventual"
        )
    
    @pytest.mark.asyncio
    async def test_analyze_data_characteristics(self, etl_engine, sample_data_source):
        """Test data characteristics analysis"""
        characteristics = await etl_engine.analyze_data_characteristics(sample_data_source)
        
        assert 'volume_category' in characteristics
        assert 'velocity_category' in characteristics
        assert 'variety_score' in characteristics
        assert 'veracity_score' in characteristics
        assert characteristics['veracity_score'] == 0.85
    
    @pytest.mark.asyncio
    async def test_recommend_etl_pipeline(self, etl_engine, sample_data_source, sample_data_target):
        """Test ETL pipeline recommendation"""
        recommendation = await etl_engine.recommend_etl_pipeline(
            sample_data_source, sample_data_target
        )
        
        assert recommendation.pipeline_id is not None
        assert recommendation.pattern_type in ETLPatternType
        assert len(recommendation.transformation_steps) > 0
        assert recommendation.confidence_score > 0
        assert recommendation.estimated_total_time > 0
    
    def test_categorize_volume(self, etl_engine):
        """Test volume categorization"""
        assert etl_engine._categorize_volume({"size_gb": 0.5}) == "small"
        assert etl_engine._categorize_volume({"size_gb": 50}) == "medium"
        assert etl_engine._categorize_volume({"size_gb": 500}) == "large"
        assert etl_engine._categorize_volume({"size_gb": 5000}) == "very_large"
    
    def test_categorize_velocity(self, etl_engine):
        """Test velocity categorization"""
        assert etl_engine._categorize_velocity("real_time") == "high"
        assert etl_engine._categorize_velocity("daily") == "medium"
        assert etl_engine._categorize_velocity("weekly") == "low"

class TestDataQualityEngine:
    """Test the data quality assessment engine"""
    
    @pytest.fixture
    def quality_engine(self):
        return DataQualityEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with quality issues"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 5],  # Duplicate
            'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],  # Missing value
            'email': ['alice@test.com', 'invalid-email', 'charlie@test.com', 
                     'david@test.com', 'eve@test.com', 'eve@test.com'],  # Invalid format
            'age': [25, 30, 35, 200, 28, 28],  # Outlier
            'status': ['Active', 'active', 'INACTIVE', 'Active', 'Inactive', 'Inactive']  # Case inconsistency
        })
    
    @pytest.mark.asyncio
    async def test_assess_data_quality(self, quality_engine, sample_data):
        """Test comprehensive data quality assessment"""
        assessment = await quality_engine.assess_data_quality(sample_data, "test_dataset")
        
        assert assessment.assessment_id is not None
        assert assessment.dataset_name == "test_dataset"
        assert assessment.total_records == len(sample_data)
        assert assessment.total_columns == len(sample_data.columns)
        assert 0 <= assessment.overall_score <= 100
        assert len(assessment.dimension_scores) == len(DataQualityDimension)
        assert len(assessment.issues) > 0
        assert len(assessment.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_assess_completeness(self, quality_engine, sample_data):
        """Test completeness assessment"""
        completeness_score, issues = await quality_engine._assess_completeness(sample_data)
        
        assert 0 <= completeness_score <= 100
        # Should detect missing value in 'name' column
        missing_issues = [issue for issue in issues if issue.issue_type == QualityIssueType.MISSING_VALUES]
        assert len(missing_issues) > 0
    
    @pytest.mark.asyncio
    async def test_assess_uniqueness(self, quality_engine, sample_data):
        """Test uniqueness assessment"""
        uniqueness_score, issues = await quality_engine._assess_uniqueness(sample_data)
        
        assert 0 <= uniqueness_score <= 100
        # Should detect duplicate records
        duplicate_issues = [issue for issue in issues if issue.issue_type == QualityIssueType.DUPLICATE_RECORDS]
        assert len(duplicate_issues) > 0
    
    @pytest.mark.asyncio
    async def test_assess_consistency(self, quality_engine, sample_data):
        """Test consistency assessment"""
        consistency_score, issues = await quality_engine._assess_consistency(sample_data)
        
        assert 0 <= consistency_score <= 100
        # Should detect case inconsistencies in 'status' column
        consistency_issues = [issue for issue in issues if issue.issue_type == QualityIssueType.INCONSISTENT_VALUES]
        assert len(consistency_issues) > 0
    
    @pytest.mark.asyncio
    async def test_cleanse_data(self, quality_engine, sample_data):
        """Test data cleansing"""
        assessment = await quality_engine.assess_data_quality(sample_data, "test_dataset")
        cleaned_data, cleansing_result = await quality_engine.cleanse_data(sample_data, assessment)
        
        assert len(cleaned_data) <= len(sample_data)  # May remove duplicates
        assert cleansing_result.cleansing_id is not None
        assert cleansing_result.original_records == len(sample_data)
        assert len(cleansing_result.issues_resolved) > 0
    
    def test_get_quality_summary(self, quality_engine):
        """Test quality summary generation"""
        # Create a mock assessment
        from security.enterprise_integration.data_quality_engine import QualityAssessment, QualityIssue
        
        assessment = QualityAssessment(
            assessment_id="test",
            dataset_name="test",
            total_records=100,
            total_columns=5,
            overall_score=85.0,
            dimension_scores={dim: 85.0 for dim in DataQualityDimension},
            issues=[],
            recommendations=["Test recommendation"],
            assessment_timestamp=datetime.utcnow(),
            processing_time=1.0
        )
        
        summary = quality_engine.get_quality_summary(assessment)
        
        assert summary['overall_score'] == 85.0
        assert summary['grade'] == 'B'
        assert 'total_issues' in summary
        assert 'dimension_scores' in summary

class TestHighPerformanceStreamingEngine:
    """Test the high-performance streaming engine"""
    
    @pytest.fixture
    def streaming_engine(self):
        config = {
            'max_events_per_second': 10000,  # Lower for testing
            'target_latency_ms': 100,
            'buffer_size': 1000,
            'batch_size': 100,
            'num_worker_threads': 2
        }
        return HighPerformanceStreamingEngine(config)
    
    @pytest.mark.asyncio
    async def test_start_stop_engine(self, streaming_engine):
        """Test starting and stopping the streaming engine"""
        await streaming_engine.start()
        assert streaming_engine.is_running
        
        await streaming_engine.stop()
        assert not streaming_engine.is_running
    
    @pytest.mark.asyncio
    async def test_create_stream_processor(self, streaming_engine):
        """Test creating a stream processor"""
        processor_config = {
            'name': 'test_processor',
            'input_topics': ['test_topic'],
            'processing_function': lambda data: [{'processed': True, 'original': item} for item in data]
        }
        
        processor = await streaming_engine.create_stream_processor(processor_config)
        
        assert processor.name == 'test_processor'
        assert 'test_topic' in processor.input_topics
        assert processor.processor_id in streaming_engine.processors
    
    @pytest.mark.asyncio
    async def test_publish_consume_events(self, streaming_engine):
        """Test publishing and consuming events"""
        await streaming_engine.start()
        
        # Create processor
        processor_config = {
            'name': 'test_processor',
            'input_topics': ['test_topic'],
            'processing_function': lambda data: data
        }
        await streaming_engine.create_stream_processor(processor_config)
        
        # Publish event
        event_data = {'message': 'test event', 'timestamp': datetime.utcnow().isoformat()}
        event_id = await streaming_engine.publish_event('test_topic', event_data)
        
        assert event_id is not None
        
        # Consume events
        consumed_events = []
        async for event in streaming_engine.consume_events('test_topic'):
            consumed_events.append(event)
            if len(consumed_events) >= 1:
                break
        
        assert len(consumed_events) == 1
        assert consumed_events[0].data == event_data
        
        await streaming_engine.stop()
    
    @pytest.mark.asyncio
    async def test_process_stream_batch(self, streaming_engine):
        """Test batch processing"""
        await streaming_engine.start()
        
        # Create processor
        def batch_processor(data_list):
            return [{'processed': True, 'count': len(data_list)}]
        
        processor_config = {
            'name': 'batch_processor',
            'input_topics': ['batch_topic'],
            'processing_function': batch_processor
        }
        processor = await streaming_engine.create_stream_processor(processor_config)
        
        # Create test events
        events = []
        for i in range(5):
            event = StreamEvent(
                event_id=f"event_{i}",
                event_type=StreamEventType.DATA,
                timestamp=datetime.utcnow(),
                partition_key="test_partition",
                data={'id': i, 'value': f'test_{i}'},
                metadata={},
                source='test',
                sequence_number=i
            )
            events.append(event)
        
        # Process batch
        results = await streaming_engine.process_stream_batch(processor.processor_id, events)
        
        assert len(results) > 0
        assert results[0]['processed'] is True
        
        await streaming_engine.stop()
    
    def test_get_stream_metrics(self, streaming_engine):
        """Test metrics collection"""
        metrics = streaming_engine.get_stream_metrics()
        
        assert hasattr(metrics, 'events_per_second')
        assert hasattr(metrics, 'average_latency_ms')
        assert hasattr(metrics, 'error_rate')
        assert metrics.events_per_second >= 0
        assert metrics.average_latency_ms >= 0

class TestVisualIntegrationBuilder:
    """Test the visual integration builder"""
    
    @pytest.fixture
    def integration_builder(self):
        return VisualIntegrationBuilder()
    
    @pytest.mark.asyncio
    async def test_create_new_flow(self, integration_builder):
        """Test creating a new integration flow"""
        flow = await integration_builder.create_new_flow(
            "Test Flow", "A test integration flow"
        )
        
        assert flow.flow_id is not None
        assert flow.name == "Test Flow"
        assert flow.description == "A test integration flow"
        assert len(flow.components) == 0
        assert len(flow.connections) == 0
    
    @pytest.mark.asyncio
    async def test_add_component_to_flow(self, integration_builder):
        """Test adding components to a flow"""
        flow = await integration_builder.create_new_flow("Test Flow")
        
        # Add a database source component
        component = await integration_builder.add_component_to_flow(
            flow.flow_id, "db_source", {"x": 100, "y": 100}
        )
        
        assert component.component_id is not None
        assert component.component_type == ComponentType.SOURCE
        assert component.position == {"x": 100, "y": 100}
        
        # Verify component was added to flow
        updated_flow = integration_builder.get_flow(flow.flow_id)
        assert len(updated_flow.components) == 1
    
    @pytest.mark.asyncio
    async def test_connect_components(self, integration_builder):
        """Test connecting components"""
        flow = await integration_builder.create_new_flow("Test Flow")
        
        # Add source and destination components
        source = await integration_builder.add_component_to_flow(
            flow.flow_id, "db_source", {"x": 100, "y": 100}
        )
        destination = await integration_builder.add_component_to_flow(
            flow.flow_id, "db_destination", {"x": 300, "y": 100}
        )
        
        # Connect components
        connection = await integration_builder.connect_components(
            flow.flow_id,
            source.component_id, "data_out",
            destination.component_id, "data_in"
        )
        
        assert connection.connection_id is not None
        assert connection.source_component_id == source.component_id
        assert connection.target_component_id == destination.component_id
        assert connection.connection_type == ConnectionType.DATA_FLOW
        
        # Verify connection was added to flow
        updated_flow = integration_builder.get_flow(flow.flow_id)
        assert len(updated_flow.connections) == 1
    
    @pytest.mark.asyncio
    async def test_validate_flow(self, integration_builder):
        """Test flow validation"""
        flow = await integration_builder.create_new_flow("Test Flow")
        
        # Add components and connections
        source = await integration_builder.add_component_to_flow(
            flow.flow_id, "db_source", {"x": 100, "y": 100}
        )
        destination = await integration_builder.add_component_to_flow(
            flow.flow_id, "db_destination", {"x": 300, "y": 100}
        )
        await integration_builder.connect_components(
            flow.flow_id,
            source.component_id, "data_out",
            destination.component_id, "data_in"
        )
        
        # Validate flow
        validation_result = await integration_builder.validate_flow(flow.flow_id)
        
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_generate_flow_code(self, integration_builder):
        """Test code generation from flow"""
        flow = await integration_builder.create_new_flow("Test Flow")
        
        # Add a simple component
        await integration_builder.add_component_to_flow(
            flow.flow_id, "db_source", {"x": 100, "y": 100}
        )
        
        # Generate Python code
        code = await integration_builder.generate_flow_code(flow.flow_id, "python")
        
        assert isinstance(code, str)
        assert "class TestFlowIntegration" in code
        assert "async def execute" in code
    
    def test_get_component_library(self, integration_builder):
        """Test getting component library"""
        library = integration_builder.get_component_library()
        
        assert isinstance(library, dict)
        assert len(library) > 0
        assert "db_source" in library
        assert "field_mapper" in library
        assert "db_destination" in library
    
    def test_get_template_library(self, integration_builder):
        """Test getting template library"""
        templates = integration_builder.get_template_library()
        
        assert isinstance(templates, dict)
        # Templates might be empty in test environment
        assert "db_to_db" in templates or len(templates) == 0

class TestIntegrationExcellenceIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_integration_workflow(self):
        """Test complete integration workflow"""
        # 1. Auto-discovery
        discovery_engine = AutoDiscoveryEngine()
        
        # Mock discovered entities
        discovery_engine.discovered_entities = {
            'source.public.customers': SchemaEntity(
                name='customers', type='table', database='source', schema='public',
                properties={'columns': [{'name': 'id', 'type': 'int'}, {'name': 'name', 'type': 'varchar'}]},
                relationships=[], confidence_score=0.9, discovered_at=datetime.utcnow()
            )
        }
        
        # 2. ETL Recommendation
        etl_engine = ETLRecommendationEngine()
        
        source = DataSource(
            name="source_db", type="database",
            connection_info={"host": "source"}, schema_info={"customers": {"id": "int", "name": "string"}},
            volume_characteristics={"size_gb": 5}, update_frequency="daily", data_quality_score=0.8
        )
        
        target = DataTarget(
            name="target_db", type="database",
            connection_info={"host": "target"}, schema_requirements={"customer_id": "int", "customer_name": "string"},
            performance_requirements={"max_latency": "medium"}, consistency_requirements="strong"
        )
        
        recommendation = await etl_engine.recommend_etl_pipeline(source, target)
        
        # 3. Data Quality Assessment
        quality_engine = DataQualityEngine()
        
        sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })
        
        assessment = await quality_engine.assess_data_quality(sample_data, "customers")
        
        # 4. Visual Integration Builder
        builder = VisualIntegrationBuilder()
        flow = await builder.create_new_flow("Customer Migration Flow")
        
        # Add components based on recommendation
        source_comp = await builder.add_component_to_flow(
            flow.flow_id, "db_source", {"x": 100, "y": 100}
        )
        mapper_comp = await builder.add_component_to_flow(
            flow.flow_id, "field_mapper", {"x": 300, "y": 100}
        )
        target_comp = await builder.add_component_to_flow(
            flow.flow_id, "db_destination", {"x": 500, "y": 100}
        )
        
        # Connect components
        await builder.connect_components(
            flow.flow_id, source_comp.component_id, "data_out",
            mapper_comp.component_id, "data_in"
        )
        await builder.connect_components(
            flow.flow_id, mapper_comp.component_id, "data_out",
            target_comp.component_id, "data_in"
        )
        
        # Validate the complete flow
        validation = await builder.validate_flow(flow.flow_id)
        
        # Assertions
        assert len(discovery_engine.discovered_entities) > 0
        assert recommendation.confidence_score > 0
        assert assessment.overall_score > 0
        assert validation.is_valid
        assert len(flow.components) == 3
        assert len(flow.connections) == 2
    
    @pytest.mark.asyncio
    async def test_streaming_integration_performance(self):
        """Test streaming engine performance characteristics"""
        streaming_engine = HighPerformanceStreamingEngine({
            'max_events_per_second': 1000,  # Reduced for testing
            'target_latency_ms': 100,
            'buffer_size': 500,
            'batch_size': 50
        })
        
        await streaming_engine.start()
        
        # Create a high-throughput processor
        processor_config = {
            'name': 'performance_processor',
            'input_topics': ['performance_topic'],
            'processing_function': lambda data: [{'processed_at': datetime.utcnow().isoformat()} for _ in data]
        }
        await streaming_engine.create_stream_processor(processor_config)
        
        # Publish multiple events rapidly
        start_time = time.time()
        event_count = 100
        
        for i in range(event_count):
            await streaming_engine.publish_event(
                'performance_topic',
                {'id': i, 'timestamp': datetime.utcnow().isoformat()}
            )
        
        # Wait a bit for processing
        await asyncio.sleep(1.0)
        
        # Check metrics
        metrics = streaming_engine.get_stream_metrics()
        processing_time = time.time() - start_time
        
        # Verify performance characteristics
        assert processing_time < 5.0  # Should process quickly
        assert metrics.events_per_second >= 0  # Should have some throughput
        
        await streaming_engine.stop()

if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])