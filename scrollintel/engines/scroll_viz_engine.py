"""
ScrollViz Engine - Automated visualization generation with chart type recommendation.

This engine provides intelligent chart type selection, visualization generation using
Plotly, Recharts, and Vega-Lite, interactive dashboard creation, and export capabilities.
"""

import json
import base64
import io
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import logging

from .base_engine import BaseEngine, EngineCapability, EngineStatus

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    AREA = "area"
    BUBBLE = "bubble"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    FUNNEL = "funnel"
    WATERFALL = "waterfall"
    CANDLESTICK = "candlestick"
    RADAR = "radar"
    PARALLEL_COORDINATES = "parallel_coordinates"
    SANKEY = "sankey"


class DataType(str, Enum):
    """Data types for chart recommendation."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"


class ExportFormat(str, Enum):
    """Supported export formats."""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


class VisualizationTemplate:
    """Template for common visualization patterns."""
    
    def __init__(self, name: str, chart_type: ChartType, description: str, 
                 data_requirements: Dict[str, Any], config: Dict[str, Any]):
        self.name = name
        self.chart_type = chart_type
        self.description = description
        self.data_requirements = data_requirements
        self.config = config


class ScrollVizEngine(BaseEngine):
    """ScrollViz engine for automated visualization generation."""
    
    def __init__(self):
        super().__init__(
            engine_id="scroll_viz",
            name="ScrollViz Visualization Engine",
            capabilities=[EngineCapability.VISUALIZATION]
        )
        self.templates = {}
        self.chart_recommendations = {}
        
    async def initialize(self) -> None:
        """Initialize the ScrollViz engine."""
        try:
            # Initialize visualization templates
            self._initialize_templates()
            
            # Initialize chart recommendation rules
            self._initialize_chart_recommendations()
            
            # Set up Plotly configuration
            pio.templates.default = "plotly_white"
            
            logger.info("ScrollViz engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ScrollViz engine: {e}")
            raise
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process visualization request."""
        try:
            params = parameters or {}
            action = params.get("action", "generate_chart")
            
            if action == "generate_chart":
                return await self._generate_chart(input_data, params)
            elif action == "recommend_chart_type":
                return await self._recommend_chart_type(input_data, params)
            elif action == "create_dashboard":
                return await self._create_dashboard(input_data, params)
            elif action == "export_visualization":
                return await self._export_visualization(input_data, params)
            elif action == "get_templates":
                return await self._get_templates(params)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error processing visualization request: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.templates.clear()
        self.chart_recommendations.clear()
        logger.info("ScrollViz engine cleaned up")
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "healthy": self.status == EngineStatus.READY,
            "status": self.status.value,
            "templates_loaded": len(self.templates),
            "supported_chart_types": [ct.value for ct in ChartType],
            "supported_export_formats": [ef.value for ef in ExportFormat]
        }
    
    def _initialize_templates(self) -> None:
        """Initialize visualization templates."""
        templates = [
            VisualizationTemplate(
                name="sales_dashboard",
                chart_type=ChartType.BAR,
                description="Sales performance dashboard with bar charts",
                data_requirements={"x": "categorical", "y": "numerical"},
                config={"color_scheme": "blues", "show_values": True}
            ),
            VisualizationTemplate(
                name="time_series_trend",
                chart_type=ChartType.LINE,
                description="Time series trend analysis",
                data_requirements={"x": "datetime", "y": "numerical"},
                config={"smooth_line": True, "show_markers": False}
            ),
            VisualizationTemplate(
                name="correlation_heatmap",
                chart_type=ChartType.HEATMAP,
                description="Correlation matrix heatmap",
                data_requirements={"data": "numerical_matrix"},
                config={"color_scale": "RdBu", "show_values": True}
            ),
            VisualizationTemplate(
                name="distribution_histogram",
                chart_type=ChartType.HISTOGRAM,
                description="Data distribution histogram",
                data_requirements={"x": "numerical"},
                config={"bins": "auto", "show_kde": True}
            ),
            VisualizationTemplate(
                name="category_pie",
                chart_type=ChartType.PIE,
                description="Category distribution pie chart",
                data_requirements={"labels": "categorical", "values": "numerical"},
                config={"show_percentages": True, "hole": 0.3}
            )
        ]
        
        for template in templates:
            self.templates[template.name] = template
    
    def _initialize_chart_recommendations(self) -> None:
        """Initialize chart type recommendation rules."""
        self.chart_recommendations = {
            # Single variable recommendations
            ("numerical",): [ChartType.HISTOGRAM, ChartType.BOX, ChartType.VIOLIN],
            ("categorical",): [ChartType.BAR, ChartType.PIE],
            ("datetime",): [ChartType.LINE],
            
            # Two variable recommendations
            ("numerical", "numerical"): [ChartType.SCATTER, ChartType.LINE],
            ("categorical", "numerical"): [ChartType.BAR, ChartType.BOX, ChartType.VIOLIN],
            ("datetime", "numerical"): [ChartType.LINE, ChartType.AREA],
            ("categorical", "categorical"): [ChartType.HEATMAP, ChartType.SUNBURST],
            
            # Three variable recommendations
            ("numerical", "numerical", "numerical"): [ChartType.BUBBLE, ChartType.SCATTER],
            ("categorical", "numerical", "numerical"): [ChartType.BUBBLE, ChartType.SCATTER],
            ("datetime", "numerical", "categorical"): [ChartType.LINE, ChartType.AREA],
        }
    
    async def _generate_chart(self, data: Union[pd.DataFrame, Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a chart based on data and parameters."""
        try:
            # Convert data to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            chart_type = params.get("chart_type")
            if not chart_type:
                # Auto-recommend chart type
                recommendation = await self._recommend_chart_type(df, params)
                chart_type = recommendation["recommended_type"]
            
            # Generate the chart
            fig = await self._create_plotly_chart(df, chart_type, params)
            
            # Convert to various formats
            result = {
                "chart_type": chart_type,
                "plotly_json": fig.to_json(),
                "html": fig.to_html(include_plotlyjs=True),
                "config": params
            }
            
            # Add Recharts config if requested
            if params.get("include_recharts", False):
                result["recharts_config"] = self._generate_recharts_config(df, chart_type, params)
            
            # Add Vega-Lite spec if requested
            if params.get("include_vega", False):
                result["vega_lite_spec"] = self._generate_vega_lite_spec(df, chart_type, params)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            raise
    
    async def _recommend_chart_type(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend chart type based on data characteristics."""
        try:
            # Analyze data types
            data_types = []
            columns = params.get("columns", data.columns.tolist())
            
            for col in columns[:3]:  # Limit to first 3 columns for recommendation
                if col in data.columns:
                    dtype = self._analyze_column_type(data[col])
                    data_types.append(dtype)
            
            # Get recommendation
            data_signature = tuple(data_types)
            recommendations = self.chart_recommendations.get(data_signature, [ChartType.BAR])
            
            # Additional context-based recommendations
            context_recommendations = self._get_context_recommendations(data, params)
            
            return {
                "data_types": data_types,
                "recommended_type": recommendations[0].value,
                "alternative_types": [r.value for r in recommendations[1:3]],
                "context_recommendations": context_recommendations,
                "reasoning": self._explain_recommendation(data_signature, recommendations[0])
            }
            
        except Exception as e:
            logger.error(f"Error recommending chart type: {e}")
            raise
    
    async def _create_dashboard(self, data: Union[List[Dict], Dict, pd.DataFrame], params: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive dashboard with multiple visualizations."""
        try:
            dashboard_config = params.get("dashboard_config", {})
            charts = []
            
            if isinstance(data, (dict, pd.DataFrame)):
                # Single dataset, create multiple views
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                    
                chart_configs = dashboard_config.get("charts", [])
                
                for chart_config in chart_configs:
                    chart_result = await self._generate_chart(df, chart_config)
                    charts.append(chart_result)
            
            elif isinstance(data, list):
                # Multiple datasets or chart configs
                for item in data:
                    if "data" in item and "config" in item:
                        chart_result = await self._generate_chart(item["data"], item["config"])
                        charts.append(chart_result)
            
            # Create dashboard layout
            dashboard = {
                "id": f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "title": dashboard_config.get("title", "ScrollIntel Dashboard"),
                "description": dashboard_config.get("description", ""),
                "layout": dashboard_config.get("layout", "grid"),
                "charts": charts,
                "real_time_config": dashboard_config.get("real_time", {}),
                "created_at": datetime.now().isoformat()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    async def _export_visualization(self, chart_data: Dict, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export visualization in various formats."""
        try:
            export_format = params.get("format", ExportFormat.PNG)
            
            # Recreate figure from JSON
            fig = go.Figure(json.loads(chart_data["plotly_json"]))
            
            result = {"format": export_format}
            
            if export_format == ExportFormat.PNG:
                img_bytes = fig.to_image(format="png", width=params.get("width", 800), 
                                       height=params.get("height", 600))
                result["data"] = base64.b64encode(img_bytes).decode()
                result["mime_type"] = "image/png"
                
            elif export_format == ExportFormat.SVG:
                svg_str = fig.to_image(format="svg", width=params.get("width", 800), 
                                     height=params.get("height", 600))
                result["data"] = svg_str.decode() if isinstance(svg_str, bytes) else svg_str
                result["mime_type"] = "image/svg+xml"
                
            elif export_format == ExportFormat.PDF:
                pdf_bytes = fig.to_image(format="pdf", width=params.get("width", 800), 
                                       height=params.get("height", 600))
                result["data"] = base64.b64encode(pdf_bytes).decode()
                result["mime_type"] = "application/pdf"
                
            elif export_format == ExportFormat.HTML:
                result["data"] = fig.to_html(include_plotlyjs=True)
                result["mime_type"] = "text/html"
                
            elif export_format == ExportFormat.JSON:
                result["data"] = chart_data["plotly_json"]
                result["mime_type"] = "application/json"
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting visualization: {e}")
            raise
    
    async def _get_templates(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get available visualization templates."""
        template_list = []
        for name, template in self.templates.items():
            template_list.append({
                "name": name,
                "chart_type": template.chart_type.value,
                "description": template.description,
                "data_requirements": template.data_requirements,
                "config": template.config
            })
        
        return {
            "templates": template_list,
            "total_count": len(template_list)
        }
    
    async def _create_plotly_chart(self, df: pd.DataFrame, chart_type: str, params: Dict[str, Any]) -> go.Figure:
        """Create Plotly chart based on type and parameters."""
        x_col = params.get("x_column", df.columns[0] if len(df.columns) > 0 else None)
        y_col = params.get("y_column", df.columns[1] if len(df.columns) > 1 else None)
        color_col = params.get("color_column")
        size_col = params.get("size_column")
        
        title = params.get("title", f"{chart_type.title()} Chart")
        
        if chart_type == ChartType.BAR.value:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
            
        elif chart_type == ChartType.LINE.value:
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
            
        elif chart_type == ChartType.SCATTER.value:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, title=title)
            
        elif chart_type == ChartType.PIE.value:
            fig = px.pie(df, names=x_col, values=y_col, title=title)
            
        elif chart_type == ChartType.HISTOGRAM.value:
            fig = px.histogram(df, x=x_col, color=color_col, title=title)
            
        elif chart_type == ChartType.BOX.value:
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
            
        elif chart_type == ChartType.VIOLIN.value:
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, title=title)
            
        elif chart_type == ChartType.HEATMAP.value:
            # For heatmap, use correlation matrix if not specified
            if params.get("correlation", False):
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr_matrix, text_auto=True, title=title)
            else:
                fig = px.density_heatmap(df, x=x_col, y=y_col, title=title)
                
        elif chart_type == ChartType.AREA.value:
            fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title)
            
        elif chart_type == ChartType.BUBBLE.value:
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col, title=title)
            
        else:
            # Default to bar chart
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        
        # Apply styling
        theme = params.get("theme", "plotly_white")
        fig.update_layout(
            showlegend=params.get("show_legend", True),
            width=params.get("width", 800),
            height=params.get("height", 600)
        )
        
        # Apply theme if it's a valid plotly template
        try:
            fig.update_layout(template=theme)
        except Exception:
            # If theme is not valid, use default
            fig.update_layout(template="plotly_white")
        
        return fig
    
    def _analyze_column_type(self, series: pd.Series) -> str:
        """Analyze the data type of a pandas Series."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME.value
        elif pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERICAL.value
        elif pd.api.types.is_bool_dtype(series):
            return DataType.BOOLEAN.value
        elif series.dtype == 'object':
            # Check if it's categorical (limited unique values)
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1 or series.nunique() < 20:
                return DataType.CATEGORICAL.value
            else:
                return DataType.TEXT.value
        else:
            return DataType.CATEGORICAL.value
    
    def _get_context_recommendations(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[str]:
        """Get context-based chart recommendations."""
        recommendations = []
        
        # Time series detection
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            recommendations.append("Consider line or area charts for time series data")
        
        # High cardinality categorical data
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 20:
                recommendations.append(f"Column '{col}' has high cardinality, consider aggregation")
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            recommendations.append("Consider correlation heatmap for multiple numeric variables")
        
        return recommendations
    
    def _explain_recommendation(self, data_signature: Tuple, recommended_type: ChartType) -> str:
        """Explain why a chart type was recommended."""
        explanations = {
            ChartType.BAR: "Bar charts are ideal for comparing categorical data",
            ChartType.LINE: "Line charts are perfect for showing trends over time",
            ChartType.SCATTER: "Scatter plots reveal relationships between numeric variables",
            ChartType.PIE: "Pie charts show proportions of a whole",
            ChartType.HISTOGRAM: "Histograms display the distribution of numeric data",
            ChartType.HEATMAP: "Heatmaps visualize patterns in matrix data"
        }
        
        base_explanation = explanations.get(recommended_type, "This chart type suits your data structure")
        return f"{base_explanation}. Data signature: {data_signature}"
    
    def _generate_recharts_config(self, df: pd.DataFrame, chart_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Recharts configuration."""
        x_col = params.get("x_column", df.columns[0] if len(df.columns) > 0 else None)
        y_col = params.get("y_column", df.columns[1] if len(df.columns) > 1 else None)
        
        config = {
            "type": chart_type,
            "data": df.to_dict("records"),
            "xAxis": {"dataKey": x_col, "type": "category"},
            "yAxis": {"type": "number"},
            "series": [{"dataKey": y_col, "type": chart_type}],
            "responsive": True,
            "legend": {"show": params.get("show_legend", True)}
        }
        
        return config
    
    def _generate_vega_lite_spec(self, df: pd.DataFrame, chart_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Vega-Lite specification."""
        x_col = params.get("x_column", df.columns[0] if len(df.columns) > 0 else None)
        y_col = params.get("y_column", df.columns[1] if len(df.columns) > 1 else None)
        
        # Map chart types to Vega-Lite marks
        mark_mapping = {
            ChartType.BAR.value: "bar",
            ChartType.LINE.value: "line",
            ChartType.SCATTER.value: "circle",
            ChartType.AREA.value: "area",
            ChartType.PIE.value: "arc"
        }
        
        mark = mark_mapping.get(chart_type, "bar")
        
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": f"{chart_type.title()} chart generated by ScrollViz",
            "data": {"values": df.to_dict("records")},
            "mark": mark,
            "encoding": {
                "x": {"field": x_col, "type": "ordinal" if chart_type == "bar" else "quantitative"},
                "y": {"field": y_col, "type": "quantitative"}
            },
            "width": params.get("width", 400),
            "height": params.get("height", 300)
        }
        
        return spec