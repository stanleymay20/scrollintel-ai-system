"""
Interactive Report Builder with Custom Visualizations

This module provides an interactive report builder that allows users to create
custom reports with drag-and-drop functionality and custom visualizations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of report components"""
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    METRIC = "metric"
    IMAGE = "image"
    DIVIDER = "divider"
    HEADER = "header"
    FOOTER = "footer"


class ChartType(Enum):
    """Types of charts available"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TREEMAP = "treemap"
    WATERFALL = "waterfall"


class LayoutType(Enum):
    """Layout types for report sections"""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    GRID = "grid"
    CUSTOM = "custom"


@dataclass
class ComponentConfig:
    """Configuration for a report component"""
    component_id: str
    component_type: ComponentType
    title: str
    position: Dict[str, int]  # x, y, width, height
    data_source: Optional[str] = None
    styling: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    filters: Optional[List[Dict]] = None
    interactions: Optional[List[Dict]] = None


@dataclass
class ChartConfig:
    """Configuration for chart components"""
    chart_type: ChartType
    data_columns: Dict[str, str]  # axis mappings
    aggregation: Optional[str] = None
    color_scheme: Optional[str] = None
    show_legend: bool = True
    show_grid: bool = True
    animation: bool = True
    custom_options: Optional[Dict[str, Any]] = None


@dataclass
class ReportLayout:
    """Layout configuration for a report"""
    layout_type: LayoutType
    sections: List[Dict[str, Any]]
    global_styling: Dict[str, Any]
    responsive: bool = True
    print_friendly: bool = True


@dataclass
class InteractiveReport:
    """Interactive report definition"""
    report_id: str
    name: str
    description: str
    layout: ReportLayout
    components: List[ComponentConfig]
    data_sources: Dict[str, Any]
    filters: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: str
    version: int = 1
    is_template: bool = False
    tags: List[str] = None


class InteractiveReportBuilder:
    """
    Interactive report builder with drag-and-drop functionality
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reports = {}
        self.templates = {}
        self.data_sources = {}
        self.component_library = self._initialize_component_library()
        
    def _initialize_component_library(self) -> Dict[str, Dict]:
        """Initialize the component library with available components"""
        return {
            "text": {
                "name": "Text Block",
                "description": "Rich text content with formatting",
                "properties": {
                    "content": {"type": "string", "default": ""},
                    "font_size": {"type": "number", "default": 14},
                    "font_weight": {"type": "string", "default": "normal"},
                    "text_align": {"type": "string", "default": "left"},
                    "color": {"type": "string", "default": "#000000"}
                }
            },
            "table": {
                "name": "Data Table",
                "description": "Tabular data display with sorting and filtering",
                "properties": {
                    "columns": {"type": "array", "default": []},
                    "sortable": {"type": "boolean", "default": True},
                    "filterable": {"type": "boolean", "default": True},
                    "paginated": {"type": "boolean", "default": True},
                    "page_size": {"type": "number", "default": 10}
                }
            },
            "chart": {
                "name": "Chart Visualization",
                "description": "Interactive charts and graphs",
                "properties": {
                    "chart_type": {"type": "string", "default": "line"},
                    "x_axis": {"type": "string", "default": ""},
                    "y_axis": {"type": "string", "default": ""},
                    "color_column": {"type": "string", "default": ""},
                    "aggregation": {"type": "string", "default": "sum"}
                }
            },
            "metric": {
                "name": "Key Metric",
                "description": "Single metric display with trend indicators",
                "properties": {
                    "value_column": {"type": "string", "default": ""},
                    "format": {"type": "string", "default": "number"},
                    "show_trend": {"type": "boolean", "default": True},
                    "trend_period": {"type": "string", "default": "previous_period"}
                }
            }
        }
    
    async def create_report(
        self,
        name: str,
        description: str,
        created_by: str,
        layout_type: LayoutType = LayoutType.SINGLE_COLUMN,
        template_id: Optional[str] = None
    ) -> str:
        """
        Create a new interactive report
        
        Args:
            name: Report name
            description: Report description
            created_by: User creating the report
            layout_type: Initial layout type
            template_id: Optional template to base report on
            
        Returns:
            Report ID
        """
        try:
            report_id = str(uuid.uuid4())
            
            # Create layout
            layout = ReportLayout(
                layout_type=layout_type,
                sections=[],
                global_styling={
                    "font_family": "Arial, sans-serif",
                    "background_color": "#ffffff",
                    "padding": "20px",
                    "margin": "0px"
                },
                responsive=True,
                print_friendly=True
            )
            
            # Initialize from template if provided
            components = []
            data_sources = {}
            filters = []
            parameters = {}
            
            if template_id and template_id in self.templates:
                template = self.templates[template_id]
                components = template.components.copy()
                data_sources = template.data_sources.copy()
                filters = template.filters.copy()
                parameters = template.parameters.copy()
                layout = template.layout
            
            # Create report
            report = InteractiveReport(
                report_id=report_id,
                name=name,
                description=description,
                layout=layout,
                components=components,
                data_sources=data_sources,
                filters=filters,
                parameters=parameters,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by=created_by,
                tags=[]
            )
            
            self.reports[report_id] = report
            
            self.logger.info(f"Created interactive report: {report_id}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Error creating report: {str(e)}")
            raise
    
    async def add_component(
        self,
        report_id: str,
        component_type: ComponentType,
        title: str,
        position: Dict[str, int],
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a component to a report
        
        Args:
            report_id: Report ID
            component_type: Type of component to add
            title: Component title
            position: Position and size (x, y, width, height)
            properties: Component-specific properties
            
        Returns:
            Component ID
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            component_id = str(uuid.uuid4())
            
            # Create component configuration
            component = ComponentConfig(
                component_id=component_id,
                component_type=component_type,
                title=title,
                position=position,
                properties=properties or {},
                styling={},
                filters=[],
                interactions=[]
            )
            
            # Add to report
            self.reports[report_id].components.append(component)
            self.reports[report_id].updated_at = datetime.utcnow()
            
            self.logger.info(f"Added component {component_id} to report {report_id}")
            return component_id
            
        except Exception as e:
            self.logger.error(f"Error adding component: {str(e)}")
            raise
    
    async def update_component(
        self,
        report_id: str,
        component_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a component in a report
        
        Args:
            report_id: Report ID
            component_id: Component ID
            updates: Updates to apply
            
        Returns:
            Success status
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            report = self.reports[report_id]
            
            # Find component
            component = None
            for comp in report.components:
                if comp.component_id == component_id:
                    component = comp
                    break
            
            if not component:
                raise ValueError(f"Component {component_id} not found")
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(component, key):
                    setattr(component, key, value)
            
            report.updated_at = datetime.utcnow()
            
            self.logger.info(f"Updated component {component_id} in report {report_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating component: {str(e)}")
            raise
    
    async def remove_component(self, report_id: str, component_id: str) -> bool:
        """Remove a component from a report"""
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            report = self.reports[report_id]
            
            # Remove component
            original_count = len(report.components)
            report.components = [c for c in report.components if c.component_id != component_id]
            
            if len(report.components) == original_count:
                raise ValueError(f"Component {component_id} not found")
            
            report.updated_at = datetime.utcnow()
            
            self.logger.info(f"Removed component {component_id} from report {report_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing component: {str(e)}")
            raise
    
    async def create_chart_component(
        self,
        report_id: str,
        title: str,
        position: Dict[str, int],
        chart_config: ChartConfig,
        data_source: str
    ) -> str:
        """
        Create a chart component with specific configuration
        
        Args:
            report_id: Report ID
            title: Chart title
            position: Position and size
            chart_config: Chart configuration
            data_source: Data source identifier
            
        Returns:
            Component ID
        """
        try:
            properties = {
                "chart_type": chart_config.chart_type.value,
                "data_columns": chart_config.data_columns,
                "aggregation": chart_config.aggregation,
                "color_scheme": chart_config.color_scheme,
                "show_legend": chart_config.show_legend,
                "show_grid": chart_config.show_grid,
                "animation": chart_config.animation,
                "custom_options": chart_config.custom_options or {}
            }
            
            component_id = await self.add_component(
                report_id=report_id,
                component_type=ComponentType.CHART,
                title=title,
                position=position,
                properties=properties
            )
            
            # Set data source
            await self.update_component(
                report_id=report_id,
                component_id=component_id,
                updates={"data_source": data_source}
            )
            
            return component_id
            
        except Exception as e:
            self.logger.error(f"Error creating chart component: {str(e)}")
            raise
    
    async def add_data_source(
        self,
        report_id: str,
        source_name: str,
        source_config: Dict[str, Any]
    ) -> bool:
        """
        Add a data source to a report
        
        Args:
            report_id: Report ID
            source_name: Data source name
            source_config: Data source configuration
            
        Returns:
            Success status
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            report = self.reports[report_id]
            report.data_sources[source_name] = source_config
            report.updated_at = datetime.utcnow()
            
            self.logger.info(f"Added data source {source_name} to report {report_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding data source: {str(e)}")
            raise
    
    async def add_filter(
        self,
        report_id: str,
        filter_config: Dict[str, Any]
    ) -> bool:
        """
        Add a filter to a report
        
        Args:
            report_id: Report ID
            filter_config: Filter configuration
            
        Returns:
            Success status
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            report = self.reports[report_id]
            report.filters.append(filter_config)
            report.updated_at = datetime.utcnow()
            
            self.logger.info(f"Added filter to report {report_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding filter: {str(e)}")
            raise
    
    async def update_layout(
        self,
        report_id: str,
        layout_updates: Dict[str, Any]
    ) -> bool:
        """
        Update report layout
        
        Args:
            report_id: Report ID
            layout_updates: Layout updates
            
        Returns:
            Success status
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            report = self.reports[report_id]
            layout = report.layout
            
            # Apply updates
            for key, value in layout_updates.items():
                if hasattr(layout, key):
                    setattr(layout, key, value)
            
            report.updated_at = datetime.utcnow()
            
            self.logger.info(f"Updated layout for report {report_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating layout: {str(e)}")
            raise
    
    async def generate_report_html(self, report_id: str) -> str:
        """
        Generate HTML representation of the report
        
        Args:
            report_id: Report ID
            
        Returns:
            HTML content
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            report = self.reports[report_id]
            
            # Generate HTML
            html = self._generate_html_structure(report)
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error generating report HTML: {str(e)}")
            raise
    
    async def generate_report_json(self, report_id: str) -> Dict[str, Any]:
        """
        Generate JSON representation of the report
        
        Args:
            report_id: Report ID
            
        Returns:
            JSON data
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            report = self.reports[report_id]
            
            # Convert to JSON-serializable format
            report_data = {
                "report_id": report.report_id,
                "name": report.name,
                "description": report.description,
                "layout": {
                    "layout_type": report.layout.layout_type.value,
                    "sections": report.layout.sections,
                    "global_styling": report.layout.global_styling,
                    "responsive": report.layout.responsive,
                    "print_friendly": report.layout.print_friendly
                },
                "components": [
                    {
                        "component_id": comp.component_id,
                        "component_type": comp.component_type.value,
                        "title": comp.title,
                        "position": comp.position,
                        "data_source": comp.data_source,
                        "styling": comp.styling,
                        "properties": comp.properties,
                        "filters": comp.filters,
                        "interactions": comp.interactions
                    }
                    for comp in report.components
                ],
                "data_sources": report.data_sources,
                "filters": report.filters,
                "parameters": report.parameters,
                "created_at": report.created_at.isoformat(),
                "updated_at": report.updated_at.isoformat(),
                "created_by": report.created_by,
                "version": report.version,
                "is_template": report.is_template,
                "tags": report.tags or []
            }
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating report JSON: {str(e)}")
            raise
    
    async def save_as_template(
        self,
        report_id: str,
        template_name: str,
        template_description: str
    ) -> str:
        """
        Save a report as a template
        
        Args:
            report_id: Report ID
            template_name: Template name
            template_description: Template description
            
        Returns:
            Template ID
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            report = self.reports[report_id]
            template_id = str(uuid.uuid4())
            
            # Create template from report
            template = InteractiveReport(
                report_id=template_id,
                name=template_name,
                description=template_description,
                layout=report.layout,
                components=report.components.copy(),
                data_sources={},  # Templates don't include actual data
                filters=report.filters.copy(),
                parameters=report.parameters.copy(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by=report.created_by,
                is_template=True,
                tags=["template"]
            )
            
            self.templates[template_id] = template
            
            self.logger.info(f"Saved report {report_id} as template {template_id}")
            return template_id
            
        except Exception as e:
            self.logger.error(f"Error saving template: {str(e)}")
            raise
    
    async def clone_report(self, report_id: str, new_name: str, created_by: str) -> str:
        """
        Clone an existing report
        
        Args:
            report_id: Source report ID
            new_name: New report name
            created_by: User creating the clone
            
        Returns:
            New report ID
        """
        try:
            if report_id not in self.reports:
                raise ValueError(f"Report {report_id} not found")
            
            source_report = self.reports[report_id]
            new_report_id = str(uuid.uuid4())
            
            # Clone report
            cloned_report = InteractiveReport(
                report_id=new_report_id,
                name=new_name,
                description=f"Clone of {source_report.name}",
                layout=source_report.layout,
                components=source_report.components.copy(),
                data_sources=source_report.data_sources.copy(),
                filters=source_report.filters.copy(),
                parameters=source_report.parameters.copy(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by=created_by,
                tags=source_report.tags.copy() if source_report.tags else []
            )
            
            # Update component IDs
            for component in cloned_report.components:
                component.component_id = str(uuid.uuid4())
            
            self.reports[new_report_id] = cloned_report
            
            self.logger.info(f"Cloned report {report_id} to {new_report_id}")
            return new_report_id
            
        except Exception as e:
            self.logger.error(f"Error cloning report: {str(e)}")
            raise
    
    def _generate_html_structure(self, report: InteractiveReport) -> str:
        """Generate HTML structure for the report"""
        try:
            # Basic HTML template
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report.name}</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    {self._generate_css_styles(report)}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div class="report-container">
                    <header class="report-header">
                        <h1>{report.name}</h1>
                        <p>{report.description}</p>
                        <div class="report-meta">
                            Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
                        </div>
                    </header>
                    
                    <main class="report-content">
                        {self._generate_components_html(report)}
                    </main>
                    
                    <footer class="report-footer">
                        <p>Created with ScrollIntel Interactive Report Builder</p>
                    </footer>
                </div>
                
                <script>
                    {self._generate_javascript(report)}
                </script>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error generating HTML structure: {str(e)}")
            return "<html><body><h1>Error generating report</h1></body></html>"
    
    def _generate_css_styles(self, report: InteractiveReport) -> str:
        """Generate CSS styles for the report"""
        global_styling = report.layout.global_styling
        
        css = f"""
        body {{
            font-family: {global_styling.get('font_family', 'Arial, sans-serif')};
            background-color: {global_styling.get('background_color', '#ffffff')};
            margin: {global_styling.get('margin', '0px')};
            padding: {global_styling.get('padding', '20px')};
        }}
        
        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .report-header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
        }}
        
        .report-header h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .report-meta {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .report-content {{
            position: relative;
        }}
        
        .component {{
            position: absolute;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .component-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        
        .chart-container {{
            width: 100%;
            height: 300px;
        }}
        
        .table-container {{
            overflow-x: auto;
        }}
        
        .table-container table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .table-container th,
        .table-container td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        
        .table-container th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
            text-align: center;
        }}
        
        .metric-trend {{
            font-size: 0.9em;
            color: #666;
            text-align: center;
        }}
        
        .report-footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media print {{
            .component {{
                position: static !important;
                margin-bottom: 20px;
                page-break-inside: avoid;
            }}
        }}
        """
        
        return css
    
    def _generate_components_html(self, report: InteractiveReport) -> str:
        """Generate HTML for report components"""
        components_html = ""
        
        # Calculate container height based on components
        max_y = 0
        for component in report.components:
            component_bottom = component.position.get('y', 0) + component.position.get('height', 200)
            max_y = max(max_y, component_bottom)
        
        components_html += f'<div style="position: relative; height: {max_y + 50}px;">'
        
        for component in report.components:
            component_html = self._generate_component_html(component)
            components_html += component_html
        
        components_html += '</div>'
        
        return components_html
    
    def _generate_component_html(self, component: ComponentConfig) -> str:
        """Generate HTML for a single component"""
        position = component.position
        style = f"""
        left: {position.get('x', 0)}px;
        top: {position.get('y', 0)}px;
        width: {position.get('width', 300)}px;
        height: {position.get('height', 200)}px;
        """
        
        component_html = f'<div class="component" id="{component.component_id}" style="{style}">'
        component_html += f'<div class="component-title">{component.title}</div>'
        
        if component.component_type == ComponentType.TEXT:
            content = component.properties.get('content', '')
            component_html += f'<div class="text-content">{content}</div>'
        
        elif component.component_type == ComponentType.TABLE:
            component_html += '<div class="table-container">'
            component_html += '<table><thead><tr>'
            
            columns = component.properties.get('columns', [])
            for col in columns:
                component_html += f'<th>{col}</th>'
            
            component_html += '</tr></thead><tbody>'
            component_html += '<tr><td colspan="100%">Data will be loaded dynamically</td></tr>'
            component_html += '</tbody></table></div>'
        
        elif component.component_type == ComponentType.CHART:
            chart_id = f"chart_{component.component_id}"
            component_html += f'<div id="{chart_id}" class="chart-container"></div>'
        
        elif component.component_type == ComponentType.METRIC:
            component_html += '<div class="metric-value">$1,234,567</div>'
            component_html += '<div class="metric-trend">↑ 15% from last period</div>'
        
        component_html += '</div>'
        
        return component_html
    
    def _generate_javascript(self, report: InteractiveReport) -> str:
        """Generate JavaScript for interactive functionality"""
        js = """
        // Initialize charts and interactive elements
        document.addEventListener('DOMContentLoaded', function() {
        """
        
        # Generate chart initialization code
        for component in report.components:
            if component.component_type == ComponentType.CHART:
                chart_js = self._generate_chart_javascript(component)
                js += chart_js
        
        js += """
        });
        
        // Utility functions
        function formatNumber(num) {
            return new Intl.NumberFormat().format(num);
        }
        
        function formatCurrency(num) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(num);
        }
        """
        
        return js
    
    def _generate_chart_javascript(self, component: ComponentConfig) -> str:
        """Generate JavaScript for chart components"""
        chart_id = f"chart_{component.component_id}"
        chart_type = component.properties.get('chart_type', 'line')
        
        # Sample data for demonstration
        sample_data = self._get_sample_chart_data(chart_type)
        
        js = f"""
        // Initialize {chart_type} chart
        var {chart_id}_data = {json.dumps(sample_data['data'])};
        var {chart_id}_layout = {json.dumps(sample_data['layout'])};
        var {chart_id}_config = {json.dumps(sample_data['config'])};
        
        Plotly.newPlot('{chart_id}', {chart_id}_data, {chart_id}_layout, {chart_id}_config);
        """
        
        return js
    
    def _get_sample_chart_data(self, chart_type: str) -> Dict[str, Any]:
        """Get sample data for chart types"""
        if chart_type == 'line':
            return {
                'data': [{
                    'x': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    'y': [20, 14, 23, 25, 22, 16],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Sample Data'
                }],
                'layout': {
                    'title': 'Sample Line Chart',
                    'xaxis': {'title': 'Month'},
                    'yaxis': {'title': 'Value'}
                },
                'config': {'responsive': True}
            }
        
        elif chart_type == 'bar':
            return {
                'data': [{
                    'x': ['Product A', 'Product B', 'Product C', 'Product D'],
                    'y': [20, 14, 23, 25],
                    'type': 'bar',
                    'name': 'Sales'
                }],
                'layout': {
                    'title': 'Sample Bar Chart',
                    'xaxis': {'title': 'Products'},
                    'yaxis': {'title': 'Sales'}
                },
                'config': {'responsive': True}
            }
        
        elif chart_type == 'pie':
            return {
                'data': [{
                    'values': [19, 26, 55],
                    'labels': ['Residential', 'Non-Residential', 'Utility'],
                    'type': 'pie'
                }],
                'layout': {
                    'title': 'Sample Pie Chart'
                },
                'config': {'responsive': True}
            }
        
        else:
            # Default to line chart
            return self._get_sample_chart_data('line')
    
    async def get_report(self, report_id: str) -> Optional[InteractiveReport]:
        """Get a report by ID"""
        return self.reports.get(report_id)
    
    async def list_reports(self, created_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all reports with optional filtering"""
        reports = []
        
        for report_id, report in self.reports.items():
            if created_by and report.created_by != created_by:
                continue
            
            reports.append({
                "report_id": report_id,
                "name": report.name,
                "description": report.description,
                "created_by": report.created_by,
                "created_at": report.created_at.isoformat(),
                "updated_at": report.updated_at.isoformat(),
                "component_count": len(report.components),
                "is_template": report.is_template,
                "tags": report.tags or []
            })
        
        return sorted(reports, key=lambda x: x["updated_at"], reverse=True)
    
    async def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        templates = []
        
        for template_id, template in self.templates.items():
            templates.append({
                "template_id": template_id,
                "name": template.name,
                "description": template.description,
                "component_count": len(template.components),
                "created_at": template.created_at.isoformat(),
                "tags": template.tags or []
            })
        
        return sorted(templates, key=lambda x: x["name"])
    
    async def delete_report(self, report_id: str) -> bool:
        """Delete a report"""
        try:
            if report_id in self.reports:
                del self.reports[report_id]
                self.logger.info(f"Deleted report {report_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting report: {str(e)}")
            raise
    
    def get_component_library(self) -> Dict[str, Dict]:
        """Get the component library"""
        return self.component_library
    
    def get_supported_chart_types(self) -> List[str]:
        """Get list of supported chart types"""
        return [chart_type.value for chart_type in ChartType]
    
    def get_supported_layout_types(self) -> List[str]:
        """Get list of supported layout types"""
        return [layout_type.value for layout_type in LayoutType]