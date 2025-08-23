"""
Demo script for Data Pipeline Automation System
Demonstrates the visual pipeline builder foundation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from scrollintel.models.pipeline_models import Base
from scrollintel.core.pipeline_builder import PipelineBuilder, DataSourceConfig, TransformConfig
from scrollintel.core.pipeline_templates import initialize_pipeline_templates


def setup_database():
    """Setup in-memory database for demo"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def demo_pipeline_creation():
    """Demonstrate pipeline creation and management"""
    print("=== Data Pipeline Automation Demo ===\n")
    
    # Setup database
    db = setup_database()
    builder = PipelineBuilder(db)
    
    print("1. Creating a new data pipeline...")
    pipeline = builder.create_pipeline(
        name="Customer Analytics Pipeline",
        description="Process customer data for analytics dashboard",
        created_by="demo_user"
    )
    print(f"   ✓ Created pipeline: {pipeline.name} (ID: {pipeline.id})")
    
    print("\n2. Adding data source nodes...")
    
    # Add PostgreSQL source
    postgres_config = DataSourceConfig(
        source_type="postgresql",
        connection_params={
            "host": "localhost",
            "port": 5432,
            "database": "customers_db",
            "username": "analytics_user",
            "table": "customers"
        },
        schema={
            "columns": ["id", "name", "email", "signup_date", "plan_type"]
        }
    )
    
    postgres_node = builder.add_data_source(
        pipeline_id=pipeline.id,
        source_config=postgres_config,
        name="Customer Database",
        position=(100, 100)
    )
    print(f"   ✓ Added PostgreSQL source: {postgres_node.name}")
    
    # Add CSV source
    csv_config = DataSourceConfig(
        source_type="csv",
        connection_params={
            "file_path": "/data/customer_events.csv",
            "delimiter": ",",
            "header": True
        },
        schema={
            "columns": ["customer_id", "event_type", "timestamp", "value"]
        }
    )
    
    csv_node = builder.add_data_source(
        pipeline_id=pipeline.id,
        source_config=csv_config,
        name="Customer Events",
        position=(100, 300)
    )
    print(f"   ✓ Added CSV source: {csv_node.name}")
    
    print("\n3. Adding transformation nodes...")
    
    # Add filter transformation
    filter_config = TransformConfig(
        transform_type="filter",
        parameters={
            "condition": "plan_type IN ('premium', 'enterprise')",
            "columns": ["plan_type"]
        }
    )
    
    filter_node = builder.add_transformation(
        pipeline_id=pipeline.id,
        transform_config=filter_config,
        name="Premium Customers Filter",
        position=(300, 100)
    )
    print(f"   ✓ Added filter transformation: {filter_node.name}")
    
    # Add aggregation transformation
    agg_config = TransformConfig(
        transform_type="aggregate",
        parameters={
            "group_by": ["plan_type"],
            "aggregations": {
                "customer_count": "count",
                "avg_events": "avg"
            }
        }
    )
    
    agg_node = builder.add_transformation(
        pipeline_id=pipeline.id,
        transform_config=agg_config,
        name="Plan Analytics",
        position=(500, 200)
    )
    print(f"   ✓ Added aggregation transformation: {agg_node.name}")
    
    print("\n4. Connecting pipeline nodes...")
    
    # Connect PostgreSQL -> Filter
    conn1 = builder.connect_nodes(postgres_node.id, filter_node.id)
    print(f"   ✓ Connected {postgres_node.name} -> {filter_node.name}")
    
    # Connect CSV -> Aggregation (simulating a join scenario)
    conn2 = builder.connect_nodes(csv_node.id, agg_node.id)
    print(f"   ✓ Connected {csv_node.name} -> {agg_node.name}")
    
    # Connect Filter -> Aggregation
    conn3 = builder.connect_nodes(filter_node.id, agg_node.id)
    print(f"   ✓ Connected {filter_node.name} -> {agg_node.name}")
    
    print("\n5. Validating pipeline...")
    validation_result = builder.validate_pipeline(pipeline.id)
    
    if validation_result.is_valid:
        print("   ✓ Pipeline validation passed!")
    else:
        print("   ✗ Pipeline validation failed:")
        for error in validation_result.errors:
            print(f"     - Error: {error}")
    
    if validation_result.warnings:
        print("   ⚠ Validation warnings:")
        for warning in validation_result.warnings:
            print(f"     - Warning: {warning}")
    
    print("\n6. Pipeline summary:")
    print(f"   - Pipeline ID: {pipeline.id}")
    print(f"   - Status: {pipeline.status.value}")
    print(f"   - Validation Status: {pipeline.validation_status.value}")
    print(f"   - Nodes: {len(pipeline.nodes)}")
    print(f"   - Connections: {len(pipeline.connections)}")
    
    # List all nodes
    print("\n   Nodes:")
    for node in pipeline.nodes:
        print(f"     - {node.name} ({node.node_type.value}:{node.component_type})")
    
    # List all connections
    print("\n   Connections:")
    for conn in pipeline.connections:
        source_name = next(n.name for n in pipeline.nodes if n.id == conn.source_node_id)
        target_name = next(n.name for n in pipeline.nodes if n.id == conn.target_node_id)
        print(f"     - {source_name} -> {target_name}")
    
    return pipeline, builder


def demo_component_templates():
    """Demonstrate component template functionality"""
    print("\n=== Component Templates Demo ===\n")
    
    db = setup_database()
    
    print("1. Initializing default component templates...")
    templates = initialize_pipeline_templates(db)
    print(f"   ✓ Loaded {len(templates)} component templates")
    
    print("\n2. Available templates by category:")
    
    # Group templates by category
    categories = {}
    for template in templates:
        category = template.category or "Other"
        if category not in categories:
            categories[category] = []
        categories[category].append(template)
    
    for category, category_templates in categories.items():
        print(f"\n   {category}:")
        for template in category_templates:
            print(f"     - {template.name} ({template.component_type})")
            print(f"       {template.description}")
    
    return templates


def demo_pipeline_operations():
    """Demonstrate various pipeline operations"""
    print("\n=== Pipeline Operations Demo ===\n")
    
    db = setup_database()
    builder = PipelineBuilder(db)
    
    print("1. Creating multiple pipelines...")
    pipelines = []
    for i in range(3):
        pipeline = builder.create_pipeline(
            name=f"Pipeline {i+1}",
            description=f"Test pipeline number {i+1}",
            created_by="demo_user"
        )
        pipelines.append(pipeline)
        print(f"   ✓ Created: {pipeline.name}")
    
    print("\n2. Listing all pipelines...")
    all_pipelines = builder.list_pipelines()
    for pipeline in all_pipelines:
        print(f"   - {pipeline.name} (Status: {pipeline.status.value})")
    
    print("\n3. Updating pipeline...")
    updated = builder.update_pipeline(
        pipelines[0].id,
        name="Updated Pipeline Name",
        description="This pipeline has been updated"
    )
    print(f"   ✓ Updated pipeline: {updated.name}")
    
    print("\n4. Testing validation on empty pipeline...")
    empty_validation = builder.validate_pipeline(pipelines[1].id)
    print(f"   - Valid: {empty_validation.is_valid}")
    print(f"   - Errors: {len(empty_validation.errors)}")
    if empty_validation.errors:
        print(f"     First error: {empty_validation.errors[0]}")
    
    print("\n5. Deleting a pipeline...")
    success = builder.delete_pipeline(pipelines[2].id)
    print(f"   ✓ Deletion successful: {success}")
    
    # Verify deletion
    remaining = builder.list_pipelines()
    print(f"   - Remaining pipelines: {len(remaining)}")


if __name__ == "__main__":
    try:
        # Run all demos
        demo_pipeline_creation()
        demo_component_templates()
        demo_pipeline_operations()
        
        print("\n=== Demo completed successfully! ===")
        print("\nKey features demonstrated:")
        print("✓ Pipeline creation and management")
        print("✓ Data source node configuration")
        print("✓ Transformation node setup")
        print("✓ Node connections and relationships")
        print("✓ Pipeline validation and error checking")
        print("✓ Component template system")
        print("✓ CRUD operations for pipelines")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()