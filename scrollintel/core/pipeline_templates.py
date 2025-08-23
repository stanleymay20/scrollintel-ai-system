"""
Data Pipeline Automation - Component Templates
Default component templates for the pipeline builder.
"""

from sqlalchemy.orm import Session
from scrollintel.core.pipeline_builder import PipelineBuilder
from scrollintel.models.pipeline_models import NodeType


def create_default_templates(db_session: Session):
    """Create default component templates"""
    builder = PipelineBuilder(db_session)
    
    # Data Source Templates
    data_source_templates = [
        {
            "name": "PostgreSQL Database",
            "node_type": NodeType.DATA_SOURCE,
            "component_type": "postgresql",
            "description": "Connect to PostgreSQL database tables and views",
            "category": "Databases",
            "default_config": {
                "host": "localhost",
                "port": 5432,
                "database": "",
                "username": "",
                "password": "",
                "table": "",
                "query": ""
            }
        },
        {
            "name": "MySQL Database",
            "node_type": NodeType.DATA_SOURCE,
            "component_type": "mysql",
            "description": "Connect to MySQL database tables and views",
            "category": "Databases",
            "default_config": {
                "host": "localhost",
                "port": 3306,
                "database": "",
                "username": "",
                "password": "",
                "table": ""
            }
        },
        {
            "name": "CSV File",
            "node_type": NodeType.DATA_SOURCE,
            "component_type": "csv",
            "description": "Read data from CSV files with configurable delimiters",
            "category": "Files",
            "default_config": {
                "file_path": "",
                "delimiter": ",",
                "header": True,
                "encoding": "utf-8"
            }
        },
        {
            "name": "JSON File",
            "node_type": NodeType.DATA_SOURCE,
            "component_type": "json",
            "description": "Read structured data from JSON files",
            "category": "Files",
            "default_config": {
                "file_path": "",
                "json_path": "$",
                "encoding": "utf-8"
            }
        },
        {
            "name": "REST API",
            "node_type": NodeType.DATA_SOURCE,
            "component_type": "rest_api",
            "description": "Fetch data from REST API endpoints",
            "category": "APIs",
            "default_config": {
                "url": "",
                "method": "GET",
                "headers": {},
                "params": {},
                "auth_type": "none"
            }
        },
        {
            "name": "Kafka Stream",
            "node_type": NodeType.DATA_SOURCE,
            "component_type": "kafka",
            "description": "Stream data from Kafka topics in real-time",
            "category": "Streaming",
            "default_config": {
                "bootstrap_servers": "localhost:9092",
                "topic": "",
                "group_id": "",
                "auto_offset_reset": "latest"
            }
        }
    ]
    
    # Transformation Templates
    transformation_templates = [
        {
            "name": "Filter Rows",
            "node_type": NodeType.TRANSFORMATION,
            "component_type": "filter",
            "description": "Filter rows based on conditions and expressions",
            "category": "Data Cleaning",
            "default_config": {
                "condition": "",
                "columns": [],
                "keep_nulls": False
            }
        },
        {
            "name": "Map Columns",
            "node_type": NodeType.TRANSFORMATION,
            "component_type": "map",
            "description": "Transform column values using expressions and functions",
            "category": "Data Transformation",
            "default_config": {
                "mappings": {},
                "drop_original": False
            }
        },
        {
            "name": "Aggregate Data",
            "node_type": NodeType.TRANSFORMATION,
            "component_type": "aggregate",
            "description": "Group and aggregate data with various functions",
            "category": "Analytics",
            "default_config": {
                "group_by": [],
                "aggregations": {},
                "having": ""
            }
        },
        {
            "name": "Join Tables",
            "node_type": NodeType.TRANSFORMATION,
            "component_type": "join",
            "description": "Join multiple data streams on specified keys",
            "category": "Data Integration",
            "default_config": {
                "join_type": "inner",
                "left_on": [],
                "right_on": [],
                "suffixes": ["_left", "_right"]
            }
        },
        {
            "name": "Sort Data",
            "node_type": NodeType.TRANSFORMATION,
            "component_type": "sort",
            "description": "Sort data by one or more columns",
            "category": "Data Organization",
            "default_config": {
                "columns": [],
                "ascending": True,
                "na_position": "last"
            }
        },
        {
            "name": "Remove Duplicates",
            "node_type": NodeType.TRANSFORMATION,
            "component_type": "deduplicate",
            "description": "Remove duplicate rows based on specified columns",
            "category": "Data Cleaning",
            "default_config": {
                "columns": [],
                "keep": "first"
            }
        },
        {
            "name": "Pivot Table",
            "node_type": NodeType.TRANSFORMATION,
            "component_type": "pivot",
            "description": "Pivot data from long to wide format",
            "category": "Data Reshaping",
            "default_config": {
                "index": [],
                "columns": [],
                "values": [],
                "aggfunc": "sum"
            }
        },
        {
            "name": "Data Validation",
            "node_type": NodeType.VALIDATION,
            "component_type": "validate",
            "description": "Validate data quality and constraints",
            "category": "Quality Control",
            "default_config": {
                "rules": [],
                "on_error": "raise",
                "report_path": ""
            }
        }
    ]
    
    # Data Sink Templates
    data_sink_templates = [
        {
            "name": "PostgreSQL Table",
            "node_type": NodeType.DATA_SINK,
            "component_type": "postgresql_sink",
            "description": "Write data to PostgreSQL database tables",
            "category": "Databases",
            "default_config": {
                "host": "localhost",
                "port": 5432,
                "database": "",
                "username": "",
                "password": "",
                "table": "",
                "if_exists": "append"
            }
        },
        {
            "name": "CSV Export",
            "node_type": NodeType.DATA_SINK,
            "component_type": "csv_export",
            "description": "Export data to CSV files",
            "category": "Files",
            "default_config": {
                "file_path": "",
                "delimiter": ",",
                "header": True,
                "encoding": "utf-8"
            }
        },
        {
            "name": "JSON Export",
            "node_type": NodeType.DATA_SINK,
            "component_type": "json_export",
            "description": "Export data to JSON files",
            "category": "Files",
            "default_config": {
                "file_path": "",
                "orient": "records",
                "encoding": "utf-8"
            }
        },
        {
            "name": "API Webhook",
            "node_type": NodeType.DATA_SINK,
            "component_type": "api_sink",
            "description": "Send data to API endpoints via HTTP requests",
            "category": "APIs",
            "default_config": {
                "url": "",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "batch_size": 100
            }
        }
    ]
    
    # Create all templates
    all_templates = data_source_templates + transformation_templates + data_sink_templates
    
    created_templates = []
    for template_data in all_templates:
        try:
            template = builder.create_component_template(**template_data)
            created_templates.append(template)
        except Exception as e:
            print(f"Failed to create template {template_data['name']}: {e}")
    
    return created_templates


def get_template_by_type(db_session: Session, component_type: str):
    """Get a specific template by component type"""
    builder = PipelineBuilder(db_session)
    templates = builder.get_component_templates()
    
    for template in templates:
        if template.component_type == component_type:
            return template
    
    return None


def initialize_pipeline_templates(db_session: Session):
    """Initialize default templates if they don't exist"""
    builder = PipelineBuilder(db_session)
    existing_templates = builder.get_component_templates()
    
    if not existing_templates:
        print("Creating default pipeline component templates...")
        created = create_default_templates(db_session)
        print(f"Created {len(created)} component templates")
    else:
        print(f"Found {len(existing_templates)} existing component templates")
    
    return existing_templates or create_default_templates(db_session)