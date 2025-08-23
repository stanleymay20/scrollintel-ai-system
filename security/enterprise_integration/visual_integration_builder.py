"""
Visual No-Code Integration Builder for Legacy System Connectivity
Provides drag-and-drop interface for building complex integrations
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    SOURCE = "source"
    TRANSFORM = "transform"
    DESTINATION = "destination"
    CONDITION = "condition"
    LOOP = "loop"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"

class ConnectionType(Enum):
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    ERROR_FLOW = "error_flow"

@dataclass
class ComponentPort:
    """Represents an input/output port on a component"""
    port_id: str
    name: str
    data_type: str
    required: bool
    description: str
    schema: Dict[str, Any]

@dataclass
class IntegrationComponent:
    """Represents a visual integration component"""
    component_id: str
    name: str
    component_type: ComponentType
    description: str
    icon: str
    input_ports: List[ComponentPort]
    output_ports: List[ComponentPort]
    configuration: Dict[str, Any]
    position: Dict[str, float]  # x, y coordinates
    properties: Dict[str, Any]

@dataclass
class ComponentConnection:
    """Represents a connection between components"""
    connection_id: str
    source_component_id: str
    source_port_id: str
    target_component_id: str
    target_port_id: str
    connection_type: ConnectionType
    data_mapping: Dict[str, Any]
    conditions: List[str]

@dataclass
class IntegrationFlow:
    """Represents a complete integration flow"""
    flow_id: str
    name: str
    description: str
    version: str
    components: List[IntegrationComponent]
    connections: List[ComponentConnection]
    global_variables: Dict[str, Any]
    error_handling: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class FlowValidationResult:
    """Result of flow validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class VisualIntegrationBuilder:
    """
    Visual no-code integration builder for legacy system connectivity
    Provides drag-and-drop interface for building complex integrations
    """
    
    def __init__(self):
        self.component_library = self._initialize_component_library()
        self.template_library = self._initialize_template_library()
        self.active_flows: Dict[str, IntegrationFlow] = {}
        
    def _initialize_component_library(self) -> Dict[str, IntegrationComponent]:
        """Initialize the library of available components"""
        components = {}
        
        # Source Components
        components.update(self._create_source_components())
        
        # Transform Components
        components.update(self._create_transform_components())
        
        # Destination Components
        components.update(self._create_destination_components())
        
        # Control Flow Components
        components.update(self._create_control_flow_components())
        
        # Utility Components
        components.update(self._create_utility_components())
        
        return components
    
    def _create_source_components(self) -> Dict[str, IntegrationComponent]:
        """Create source components for data extraction"""
        components = {}
        
        # Database Source
        components["db_source"] = IntegrationComponent(
            component_id="db_source",
            name="Database Source",
            component_type=ComponentType.SOURCE,
            description="Extract data from relational databases",
            icon="database",
            input_ports=[],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Data Output",
                    data_type="dataset",
                    required=True,
                    description="Extracted data records",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "connection_string": "",
                "query": "SELECT * FROM table_name",
                "batch_size": 1000,
                "timeout": 30
            },
            position={"x": 0, "y": 0},
            properties={"color": "#4CAF50", "category": "sources"}
        )
        
        # File Source
        components["file_source"] = IntegrationComponent(
            component_id="file_source",
            name="File Source",
            component_type=ComponentType.SOURCE,
            description="Read data from files (CSV, JSON, XML)",
            icon="file",
            input_ports=[],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Data Output",
                    data_type="dataset",
                    required=True,
                    description="File data records",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "file_path": "",
                "file_type": "csv",
                "delimiter": ",",
                "encoding": "utf-8",
                "header_row": True
            },
            position={"x": 0, "y": 0},
            properties={"color": "#2196F3", "category": "sources"}
        )
        
        # API Source
        components["api_source"] = IntegrationComponent(
            component_id="api_source",
            name="API Source",
            component_type=ComponentType.SOURCE,
            description="Fetch data from REST APIs",
            icon="api",
            input_ports=[],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Data Output",
                    data_type="dataset",
                    required=True,
                    description="API response data",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "endpoint": "",
                "method": "GET",
                "headers": {},
                "authentication": {"type": "none"},
                "pagination": {"enabled": False}
            },
            position={"x": 0, "y": 0},
            properties={"color": "#FF9800", "category": "sources"}
        )
        
        # Legacy System Source
        components["legacy_source"] = IntegrationComponent(
            component_id="legacy_source",
            name="Legacy System Source",
            component_type=ComponentType.SOURCE,
            description="Connect to legacy systems (COBOL, Mainframe, AS/400)",
            icon="legacy",
            input_ports=[],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Data Output",
                    data_type="dataset",
                    required=True,
                    description="Legacy system data",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "system_type": "mainframe",
                "connection_params": {},
                "data_format": "fixed_width",
                "record_layout": {}
            },
            position={"x": 0, "y": 0},
            properties={"color": "#795548", "category": "sources"}
        )
        
        return components
    
    def _create_transform_components(self) -> Dict[str, IntegrationComponent]:
        """Create transformation components"""
        components = {}
        
        # Field Mapper
        components["field_mapper"] = IntegrationComponent(
            component_id="field_mapper",
            name="Field Mapper",
            component_type=ComponentType.TRANSFORM,
            description="Map and transform fields between schemas",
            icon="map",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Input data to transform",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Data Output",
                    data_type="dataset",
                    required=True,
                    description="Transformed data",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "field_mappings": [],
                "default_values": {},
                "transformation_rules": []
            },
            position={"x": 0, "y": 0},
            properties={"color": "#9C27B0", "category": "transforms"}
        )
        
        # Data Filter
        components["data_filter"] = IntegrationComponent(
            component_id="data_filter",
            name="Data Filter",
            component_type=ComponentType.TRANSFORM,
            description="Filter data based on conditions",
            icon="filter",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Input data to filter",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Filtered Data",
                    data_type="dataset",
                    required=True,
                    description="Filtered data records",
                    schema={"type": "array", "items": {"type": "object"}}
                ),
                ComponentPort(
                    port_id="rejected_out",
                    name="Rejected Data",
                    data_type="dataset",
                    required=False,
                    description="Records that didn't match filter",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "filter_conditions": [],
                "logic_operator": "AND",
                "output_rejected": False
            },
            position={"x": 0, "y": 0},
            properties={"color": "#607D8B", "category": "transforms"}
        )
        
        # Data Aggregator
        components["data_aggregator"] = IntegrationComponent(
            component_id="data_aggregator",
            name="Data Aggregator",
            component_type=ComponentType.AGGREGATION,
            description="Aggregate data using various functions",
            icon="aggregate",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Input data to aggregate",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Aggregated Data",
                    data_type="dataset",
                    required=True,
                    description="Aggregated results",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "group_by_fields": [],
                "aggregation_functions": [],
                "having_conditions": []
            },
            position={"x": 0, "y": 0},
            properties={"color": "#3F51B5", "category": "transforms"}
        )
        
        # Data Joiner
        components["data_joiner"] = IntegrationComponent(
            component_id="data_joiner",
            name="Data Joiner",
            component_type=ComponentType.TRANSFORM,
            description="Join data from multiple sources",
            icon="join",
            input_ports=[
                ComponentPort(
                    port_id="left_data_in",
                    name="Left Data Input",
                    data_type="dataset",
                    required=True,
                    description="Left dataset for join",
                    schema={"type": "array", "items": {"type": "object"}}
                ),
                ComponentPort(
                    port_id="right_data_in",
                    name="Right Data Input",
                    data_type="dataset",
                    required=True,
                    description="Right dataset for join",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Joined Data",
                    data_type="dataset",
                    required=True,
                    description="Joined dataset",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "join_type": "inner",
                "join_conditions": [],
                "output_fields": []
            },
            position={"x": 0, "y": 0},
            properties={"color": "#E91E63", "category": "transforms"}
        )
        
        return components
    
    def _create_destination_components(self) -> Dict[str, IntegrationComponent]:
        """Create destination components for data loading"""
        components = {}
        
        # Database Destination
        components["db_destination"] = IntegrationComponent(
            component_id="db_destination",
            name="Database Destination",
            component_type=ComponentType.DESTINATION,
            description="Load data into relational databases",
            icon="database",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Data to load into database",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="result_out",
                    name="Load Result",
                    data_type="result",
                    required=True,
                    description="Load operation result",
                    schema={"type": "object", "properties": {"status": {"type": "string"}, "records_loaded": {"type": "number"}}}
                )
            ],
            configuration={
                "connection_string": "",
                "table_name": "",
                "load_mode": "insert",
                "batch_size": 1000,
                "create_table": False
            },
            position={"x": 0, "y": 0},
            properties={"color": "#4CAF50", "category": "destinations"}
        )
        
        # File Destination
        components["file_destination"] = IntegrationComponent(
            component_id="file_destination",
            name="File Destination",
            component_type=ComponentType.DESTINATION,
            description="Write data to files (CSV, JSON, XML)",
            icon="file",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Data to write to file",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="result_out",
                    name="Write Result",
                    data_type="result",
                    required=True,
                    description="File write result",
                    schema={"type": "object", "properties": {"status": {"type": "string"}, "file_path": {"type": "string"}}}
                )
            ],
            configuration={
                "file_path": "",
                "file_type": "csv",
                "delimiter": ",",
                "encoding": "utf-8",
                "include_header": True,
                "overwrite": True
            },
            position={"x": 0, "y": 0},
            properties={"color": "#2196F3", "category": "destinations"}
        )
        
        # API Destination
        components["api_destination"] = IntegrationComponent(
            component_id="api_destination",
            name="API Destination",
            component_type=ComponentType.DESTINATION,
            description="Send data to REST APIs",
            icon="api",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Data to send to API",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="result_out",
                    name="API Result",
                    data_type="result",
                    required=True,
                    description="API call result",
                    schema={"type": "object", "properties": {"status": {"type": "string"}, "response": {"type": "object"}}}
                )
            ],
            configuration={
                "endpoint": "",
                "method": "POST",
                "headers": {},
                "authentication": {"type": "none"},
                "batch_requests": False,
                "retry_policy": {"max_retries": 3, "backoff": "exponential"}
            },
            position={"x": 0, "y": 0},
            properties={"color": "#FF9800", "category": "destinations"}
        )
        
        return components
    
    def _create_control_flow_components(self) -> Dict[str, IntegrationComponent]:
        """Create control flow components"""
        components = {}
        
        # Conditional Branch
        components["conditional"] = IntegrationComponent(
            component_id="conditional",
            name="Conditional Branch",
            component_type=ComponentType.CONDITION,
            description="Route data based on conditions",
            icon="branch",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Input data to evaluate",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="true_out",
                    name="True Branch",
                    data_type="dataset",
                    required=False,
                    description="Data when condition is true",
                    schema={"type": "array", "items": {"type": "object"}}
                ),
                ComponentPort(
                    port_id="false_out",
                    name="False Branch",
                    data_type="dataset",
                    required=False,
                    description="Data when condition is false",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "condition": "",
                "evaluation_mode": "record_level"
            },
            position={"x": 0, "y": 0},
            properties={"color": "#FFC107", "category": "control"}
        )
        
        # Loop Iterator
        components["loop"] = IntegrationComponent(
            component_id="loop",
            name="Loop Iterator",
            component_type=ComponentType.LOOP,
            description="Iterate over data collections",
            icon="loop",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Data collection to iterate",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="item_out",
                    name="Current Item",
                    data_type="record",
                    required=True,
                    description="Current iteration item",
                    schema={"type": "object"}
                ),
                ComponentPort(
                    port_id="complete_out",
                    name="Loop Complete",
                    data_type="signal",
                    required=False,
                    description="Signal when loop completes",
                    schema={"type": "object", "properties": {"status": {"type": "string"}}}
                )
            ],
            configuration={
                "batch_size": 1,
                "parallel_execution": False,
                "error_handling": "continue"
            },
            position={"x": 0, "y": 0},
            properties={"color": "#673AB7", "category": "control"}
        )
        
        return components
    
    def _create_utility_components(self) -> Dict[str, IntegrationComponent]:
        """Create utility components"""
        components = {}
        
        # Data Validator
        components["validator"] = IntegrationComponent(
            component_id="validator",
            name="Data Validator",
            component_type=ComponentType.VALIDATION,
            description="Validate data quality and format",
            icon="check",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Data to validate",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="valid_out",
                    name="Valid Data",
                    data_type="dataset",
                    required=True,
                    description="Data that passed validation",
                    schema={"type": "array", "items": {"type": "object"}}
                ),
                ComponentPort(
                    port_id="invalid_out",
                    name="Invalid Data",
                    data_type="dataset",
                    required=False,
                    description="Data that failed validation",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "validation_rules": [],
                "stop_on_error": False,
                "output_invalid": True
            },
            position={"x": 0, "y": 0},
            properties={"color": "#4CAF50", "category": "utilities"}
        )
        
        # Data Enricher
        components["enricher"] = IntegrationComponent(
            component_id="enricher",
            name="Data Enricher",
            component_type=ComponentType.ENRICHMENT,
            description="Enrich data with additional information",
            icon="enrich",
            input_ports=[
                ComponentPort(
                    port_id="data_in",
                    name="Data Input",
                    data_type="dataset",
                    required=True,
                    description="Data to enrich",
                    schema={"type": "array", "items": {"type": "object"}}
                ),
                ComponentPort(
                    port_id="lookup_in",
                    name="Lookup Data",
                    data_type="dataset",
                    required=False,
                    description="Reference data for enrichment",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            output_ports=[
                ComponentPort(
                    port_id="data_out",
                    name="Enriched Data",
                    data_type="dataset",
                    required=True,
                    description="Data with additional fields",
                    schema={"type": "array", "items": {"type": "object"}}
                )
            ],
            configuration={
                "enrichment_rules": [],
                "lookup_keys": [],
                "default_values": {}
            },
            position={"x": 0, "y": 0},
            properties={"color": "#00BCD4", "category": "utilities"}
        )
        
        return components
    
    def _initialize_template_library(self) -> Dict[str, IntegrationFlow]:
        """Initialize library of integration flow templates"""
        templates = {}
        
        # Database to Database Template
        templates["db_to_db"] = self._create_db_to_db_template()
        
        # File to API Template
        templates["file_to_api"] = self._create_file_to_api_template()
        
        # Legacy System Migration Template
        templates["legacy_migration"] = self._create_legacy_migration_template()
        
        # Real-time Sync Template
        templates["realtime_sync"] = self._create_realtime_sync_template()
        
        return templates
    
    def _create_db_to_db_template(self) -> IntegrationFlow:
        """Create database to database integration template"""
        components = [
            IntegrationComponent(
                component_id="source_db",
                name="Source Database",
                component_type=ComponentType.SOURCE,
                description="Source database connection",
                icon="database",
                input_ports=[],
                output_ports=[ComponentPort("data_out", "Data Output", "dataset", True, "Source data", {})],
                configuration={"connection_string": "", "query": "SELECT * FROM source_table"},
                position={"x": 100, "y": 100},
                properties={"color": "#4CAF50"}
            ),
            IntegrationComponent(
                component_id="field_mapper",
                name="Field Mapper",
                component_type=ComponentType.TRANSFORM,
                description="Map fields between schemas",
                icon="map",
                input_ports=[ComponentPort("data_in", "Data Input", "dataset", True, "Input data", {})],
                output_ports=[ComponentPort("data_out", "Data Output", "dataset", True, "Mapped data", {})],
                configuration={"field_mappings": []},
                position={"x": 300, "y": 100},
                properties={"color": "#9C27B0"}
            ),
            IntegrationComponent(
                component_id="target_db",
                name="Target Database",
                component_type=ComponentType.DESTINATION,
                description="Target database connection",
                icon="database",
                input_ports=[ComponentPort("data_in", "Data Input", "dataset", True, "Data to load", {})],
                output_ports=[ComponentPort("result_out", "Load Result", "result", True, "Load result", {})],
                configuration={"connection_string": "", "table_name": "target_table", "load_mode": "insert"},
                position={"x": 500, "y": 100},
                properties={"color": "#4CAF50"}
            )
        ]
        
        connections = [
            ComponentConnection(
                connection_id="conn_1",
                source_component_id="source_db",
                source_port_id="data_out",
                target_component_id="field_mapper",
                target_port_id="data_in",
                connection_type=ConnectionType.DATA_FLOW,
                data_mapping={},
                conditions=[]
            ),
            ComponentConnection(
                connection_id="conn_2",
                source_component_id="field_mapper",
                source_port_id="data_out",
                target_component_id="target_db",
                target_port_id="data_in",
                connection_type=ConnectionType.DATA_FLOW,
                data_mapping={},
                conditions=[]
            )
        ]
        
        return IntegrationFlow(
            flow_id="template_db_to_db",
            name="Database to Database Integration",
            description="Template for migrating data between databases",
            version="1.0.0",
            components=components,
            connections=connections,
            global_variables={},
            error_handling={"strategy": "stop_on_error"},
            metadata={"category": "templates", "complexity": "simple"},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def _create_file_to_api_template(self) -> IntegrationFlow:
        """Create file to API integration template"""
        # Implementation similar to db_to_db but with file source and API destination
        pass
    
    def _create_legacy_migration_template(self) -> IntegrationFlow:
        """Create legacy system migration template"""
        # Implementation for legacy system integration
        pass
    
    def _create_realtime_sync_template(self) -> IntegrationFlow:
        """Create real-time synchronization template"""
        # Implementation for real-time data sync
        pass
    
    async def create_new_flow(self, name: str, description: str = "") -> IntegrationFlow:
        """Create a new integration flow"""
        flow_id = str(uuid.uuid4())
        
        flow = IntegrationFlow(
            flow_id=flow_id,
            name=name,
            description=description,
            version="1.0.0",
            components=[],
            connections=[],
            global_variables={},
            error_handling={"strategy": "continue_on_error"},
            metadata={"created_by": "visual_builder"},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.active_flows[flow_id] = flow
        logger.info(f"Created new integration flow: {flow_id}")
        
        return flow
    
    async def add_component_to_flow(self, flow_id: str, component_type: str, 
                                  position: Dict[str, float], 
                                  configuration: Dict[str, Any] = None) -> IntegrationComponent:
        """Add a component to an integration flow"""
        if flow_id not in self.active_flows:
            raise ValueError(f"Flow {flow_id} not found")
        
        if component_type not in self.component_library:
            raise ValueError(f"Component type {component_type} not found")
        
        # Create a copy of the component template
        template = self.component_library[component_type]
        component_id = f"{component_type}_{str(uuid.uuid4())[:8]}"
        
        component = IntegrationComponent(
            component_id=component_id,
            name=template.name,
            component_type=template.component_type,
            description=template.description,
            icon=template.icon,
            input_ports=template.input_ports.copy(),
            output_ports=template.output_ports.copy(),
            configuration={**template.configuration, **(configuration or {})},
            position=position,
            properties=template.properties.copy()
        )
        
        # Add to flow
        flow = self.active_flows[flow_id]
        flow.components.append(component)
        flow.updated_at = datetime.utcnow()
        
        logger.info(f"Added component {component_id} to flow {flow_id}")
        return component
    
    async def connect_components(self, flow_id: str, 
                               source_component_id: str, source_port_id: str,
                               target_component_id: str, target_port_id: str,
                               connection_type: ConnectionType = ConnectionType.DATA_FLOW,
                               data_mapping: Dict[str, Any] = None) -> ComponentConnection:
        """Connect two components in a flow"""
        if flow_id not in self.active_flows:
            raise ValueError(f"Flow {flow_id} not found")
        
        flow = self.active_flows[flow_id]
        
        # Validate components exist
        source_component = next((c for c in flow.components if c.component_id == source_component_id), None)
        target_component = next((c for c in flow.components if c.component_id == target_component_id), None)
        
        if not source_component:
            raise ValueError(f"Source component {source_component_id} not found")
        if not target_component:
            raise ValueError(f"Target component {target_component_id} not found")
        
        # Validate ports exist
        source_port = next((p for p in source_component.output_ports if p.port_id == source_port_id), None)
        target_port = next((p for p in target_component.input_ports if p.port_id == target_port_id), None)
        
        if not source_port:
            raise ValueError(f"Source port {source_port_id} not found")
        if not target_port:
            raise ValueError(f"Target port {target_port_id} not found")
        
        # Create connection
        connection_id = str(uuid.uuid4())
        connection = ComponentConnection(
            connection_id=connection_id,
            source_component_id=source_component_id,
            source_port_id=source_port_id,
            target_component_id=target_component_id,
            target_port_id=target_port_id,
            connection_type=connection_type,
            data_mapping=data_mapping or {},
            conditions=[]
        )
        
        flow.connections.append(connection)
        flow.updated_at = datetime.utcnow()
        
        logger.info(f"Connected {source_component_id}.{source_port_id} to {target_component_id}.{target_port_id}")
        return connection
    
    async def validate_flow(self, flow_id: str) -> FlowValidationResult:
        """Validate an integration flow"""
        if flow_id not in self.active_flows:
            raise ValueError(f"Flow {flow_id} not found")
        
        flow = self.active_flows[flow_id]
        errors = []
        warnings = []
        suggestions = []
        
        # Check for orphaned components
        connected_components = set()
        for connection in flow.connections:
            connected_components.add(connection.source_component_id)
            connected_components.add(connection.target_component_id)
        
        for component in flow.components:
            if component.component_id not in connected_components:
                if component.component_type == ComponentType.SOURCE:
                    warnings.append(f"Source component '{component.name}' is not connected to any destination")
                elif component.component_type == ComponentType.DESTINATION:
                    warnings.append(f"Destination component '{component.name}' has no input connections")
                else:
                    warnings.append(f"Component '{component.name}' is not connected")
        
        # Check for required port connections
        for component in flow.components:
            for port in component.input_ports:
                if port.required:
                    connected = any(conn.target_component_id == component.component_id and 
                                  conn.target_port_id == port.port_id 
                                  for conn in flow.connections)
                    if not connected:
                        errors.append(f"Required input port '{port.name}' on component '{component.name}' is not connected")
        
        # Check for circular dependencies
        if self._has_circular_dependency(flow):
            errors.append("Flow contains circular dependencies")
        
        # Check for data type compatibility
        for connection in flow.connections:
            source_component = next(c for c in flow.components if c.component_id == connection.source_component_id)
            target_component = next(c for c in flow.components if c.component_id == connection.target_component_id)
            
            source_port = next(p for p in source_component.output_ports if p.port_id == connection.source_port_id)
            target_port = next(p for p in target_component.input_ports if p.port_id == connection.target_port_id)
            
            if source_port.data_type != target_port.data_type:
                warnings.append(f"Data type mismatch: {source_port.data_type} -> {target_port.data_type}")
        
        # Generate suggestions
        if len(flow.components) == 0:
            suggestions.append("Add components to your flow to get started")
        elif len([c for c in flow.components if c.component_type == ComponentType.SOURCE]) == 0:
            suggestions.append("Add a source component to extract data")
        elif len([c for c in flow.components if c.component_type == ComponentType.DESTINATION]) == 0:
            suggestions.append("Add a destination component to load data")
        
        # Performance suggestions
        if len(flow.components) > 10:
            suggestions.append("Consider breaking complex flows into smaller, reusable sub-flows")
        
        is_valid = len(errors) == 0
        
        return FlowValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _has_circular_dependency(self, flow: IntegrationFlow) -> bool:
        """Check if flow has circular dependencies"""
        # Build adjacency list
        graph = {}
        for component in flow.components:
            graph[component.component_id] = []
        
        for connection in flow.connections:
            if connection.connection_type == ConnectionType.DATA_FLOW:
                graph[connection.source_component_id].append(connection.target_component_id)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for component_id in graph:
            if component_id not in visited:
                if has_cycle(component_id):
                    return True
        
        return False
    
    async def generate_flow_code(self, flow_id: str, target_platform: str = "python") -> str:
        """Generate executable code from visual flow"""
        if flow_id not in self.active_flows:
            raise ValueError(f"Flow {flow_id} not found")
        
        flow = self.active_flows[flow_id]
        
        if target_platform.lower() == "python":
            return await self._generate_python_code(flow)
        elif target_platform.lower() == "sql":
            return await self._generate_sql_code(flow)
        elif target_platform.lower() == "yaml":
            return await self._generate_yaml_config(flow)
        else:
            raise ValueError(f"Unsupported target platform: {target_platform}")
    
    async def _generate_python_code(self, flow: IntegrationFlow) -> str:
        """Generate Python code from flow"""
        code = f'''
"""
Generated integration flow: {flow.name}
Description: {flow.description}
Generated at: {datetime.utcnow().isoformat()}
"""

import pandas as pd
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class {flow.name.replace(' ', '')}Integration:
    def __init__(self):
        self.flow_id = "{flow.flow_id}"
        self.flow_name = "{flow.name}"
        
    async def execute(self):
        """Execute the integration flow"""
        try:
            logger.info(f"Starting integration flow: {{self.flow_name}}")
            
            # Component execution order based on dependencies
'''
        
        # Add component execution code
        execution_order = self._get_execution_order(flow)
        
        for component_id in execution_order:
            component = next(c for c in flow.components if c.component_id == component_id)
            code += f'''
            # Execute {component.name}
            {component_id}_result = await self.execute_{component.component_type.value}(
                component_id="{component_id}",
                configuration={json.dumps(component.configuration, indent=16)}
            )
'''
        
        code += '''
            logger.info(f"Integration flow completed successfully: {self.flow_name}")
            return {"status": "success", "flow_id": self.flow_id}
            
        except Exception as e:
            logger.error(f"Integration flow failed: {self.flow_name}, Error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def execute_source(self, component_id: str, configuration: dict):
        """Execute source component"""
        # Implementation depends on source type
        pass
    
    async def execute_transform(self, component_id: str, configuration: dict):
        """Execute transform component"""
        # Implementation depends on transform type
        pass
    
    async def execute_destination(self, component_id: str, configuration: dict):
        """Execute destination component"""
        # Implementation depends on destination type
        pass

if __name__ == "__main__":
    import asyncio
    
    integration = ''' + flow.name.replace(' ', '') + '''Integration()
    result = asyncio.run(integration.execute())
    print(f"Integration result: {result}")
'''
        
        return code
    
    def _get_execution_order(self, flow: IntegrationFlow) -> List[str]:
        """Get component execution order based on dependencies"""
        # Topological sort of components
        in_degree = {c.component_id: 0 for c in flow.components}
        
        # Calculate in-degrees
        for connection in flow.connections:
            if connection.connection_type == ConnectionType.DATA_FLOW:
                in_degree[connection.target_component_id] += 1
        
        # Find components with no dependencies
        queue = [comp_id for comp_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees of dependent components
            for connection in flow.connections:
                if (connection.source_component_id == current and 
                    connection.connection_type == ConnectionType.DATA_FLOW):
                    in_degree[connection.target_component_id] -= 1
                    if in_degree[connection.target_component_id] == 0:
                        queue.append(connection.target_component_id)
        
        return execution_order
    
    async def _generate_sql_code(self, flow: IntegrationFlow) -> str:
        """Generate SQL code from flow (for database-centric flows)"""
        # Implementation for SQL generation
        return "-- SQL code generation not implemented yet"
    
    async def _generate_yaml_config(self, flow: IntegrationFlow) -> str:
        """Generate YAML configuration from flow"""
        # Implementation for YAML config generation
        return "# YAML config generation not implemented yet"
    
    def get_component_library(self) -> Dict[str, IntegrationComponent]:
        """Get available component library"""
        return self.component_library.copy()
    
    def get_template_library(self) -> Dict[str, IntegrationFlow]:
        """Get available template library"""
        return self.template_library.copy()
    
    def get_flow(self, flow_id: str) -> Optional[IntegrationFlow]:
        """Get a specific flow"""
        return self.active_flows.get(flow_id)
    
    def list_flows(self) -> List[IntegrationFlow]:
        """List all active flows"""
        return list(self.active_flows.values())
    
    async def save_flow(self, flow_id: str, file_path: str):
        """Save flow to file"""
        if flow_id not in self.active_flows:
            raise ValueError(f"Flow {flow_id} not found")
        
        flow = self.active_flows[flow_id]
        flow_data = asdict(flow)
        
        with open(file_path, 'w') as f:
            json.dump(flow_data, f, indent=2, default=str)
        
        logger.info(f"Saved flow {flow_id} to {file_path}")
    
    async def load_flow(self, file_path: str) -> IntegrationFlow:
        """Load flow from file"""
        with open(file_path, 'r') as f:
            flow_data = json.load(f)
        
        # Convert back to dataclass instances
        flow = IntegrationFlow(**flow_data)
        self.active_flows[flow.flow_id] = flow
        
        logger.info(f"Loaded flow {flow.flow_id} from {file_path}")
        return flow