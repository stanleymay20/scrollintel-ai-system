"""
Advanced data visualization engine for ScrollIntel.
Handles chart generation, interactive features, and data processing.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from io import BytesIO
import base64

from ..models.visualization_models import (
    ChartConfiguration, ChartType, VisualizationData, 
    DataFilter, FilterOperator, DrillDownConfig,
    InteractiveFeatures, VisualizationResponse
)

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """Advanced visualization engine with interactive features."""
    
    def __init__(self):
        self.color_schemes = {
            "default": ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#00ff00"],
            "professional": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "modern": ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe"],
            "corporate": ["#2c3e50", "#3498db", "#e74c3c", "#f39c12", "#27ae60"],
            "scrollintel": ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b"]
        }
        
    async def create_visualization(
        self, 
        data: List[Dict[str, Any]], 
        config: ChartConfiguration,
        filters: Optional[List[DataFilter]] = None
    ) -> VisualizationResponse:
        """Create a visualization from data and configuration."""
        try:
            # Apply filters if provided
            if filters:
                data = self._apply_filters(data, filters)
            
            # Process data based on chart configuration
            processed_data = self._process_data(data, config)
            
            # Generate chart configuration
            chart_config = self._generate_chart_config(processed_data, config)
            
            # Create visualization data object
            viz_data = VisualizationData(
                id=f"viz_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                name=config.title,
                data=processed_data,
                chart_config=config,
                metadata={
                    "data_points": len(processed_data),
                    "chart_type": config.chart_type,
                    "created_at": datetime.utcnow().isoformat(),
                    "filters_applied": len(filters) if filters else 0
                }
            )
            
            return VisualizationResponse(
                success=True,
                data=viz_data,
                message="Visualization created successfully"
            )
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return VisualizationResponse(
                success=False,
                message="Failed to create visualization",
                errors=[str(e)]
            )
    
    def _apply_filters(self, data: List[Dict[str, Any]], filters: List[DataFilter]) -> List[Dict[str, Any]]:
        """Apply filters to data."""
        df = pd.DataFrame(data)
        
        for filter_config in filters:
            field = filter_config.field
            operator = filter_config.operator
            value = filter_config.value
            
            if field not in df.columns:
                continue
                
            if operator == FilterOperator.EQUALS:
                df = df[df[field] == value]
            elif operator == FilterOperator.NOT_EQUALS:
                df = df[df[field] != value]
            elif operator == FilterOperator.GREATER_THAN:
                df = df[df[field] > value]
            elif operator == FilterOperator.LESS_THAN:
                df = df[df[field] < value]
            elif operator == FilterOperator.GREATER_EQUAL:
                df = df[df[field] >= value]
            elif operator == FilterOperator.LESS_EQUAL:
                df = df[df[field] <= value]
            elif operator == FilterOperator.CONTAINS:
                df = df[df[field].astype(str).str.contains(str(value), na=False)]
            elif operator == FilterOperator.NOT_CONTAINS:
                df = df[~df[field].astype(str).str.contains(str(value), na=False)]
            elif operator == FilterOperator.IN:
                df = df[df[field].isin(value)]
            elif operator == FilterOperator.NOT_IN:
                df = df[~df[field].isin(value)]
            elif operator == FilterOperator.BETWEEN and isinstance(value, list) and len(value) == 2:
                df = df[(df[field] >= value[0]) & (df[field] <= value[1])]
        
        return df.to_dict('records')
    
    def _process_data(self, data: List[Dict[str, Any]], config: ChartConfiguration) -> List[Dict[str, Any]]:
        """Process data based on chart configuration."""
        df = pd.DataFrame(data)
        
        if df.empty:
            return []
        
        # Handle grouping
        if config.group_by and config.group_by in df.columns:
            if config.aggregation:
                if config.aggregation == "sum":
                    df = df.groupby(config.group_by).sum().reset_index()
                elif config.aggregation == "avg":
                    df = df.groupby(config.group_by).mean().reset_index()
                elif config.aggregation == "count":
                    df = df.groupby(config.group_by).size().reset_index(name='count')
                elif config.aggregation == "max":
                    df = df.groupby(config.group_by).max().reset_index()
                elif config.aggregation == "min":
                    df = df.groupby(config.group_by).min().reset_index()
        
        # Handle specific chart type processing
        if config.chart_type == ChartType.PIE:
            df = self._process_pie_data(df, config)
        elif config.chart_type == ChartType.HISTOGRAM:
            df = self._process_histogram_data(df, config)
        elif config.chart_type == ChartType.HEATMAP:
            df = self._process_heatmap_data(df, config)
        
        return df.to_dict('records')
    
    def _process_pie_data(self, df: pd.DataFrame, config: ChartConfiguration) -> pd.DataFrame:
        """Process data for pie charts."""
        if config.group_by and config.group_by in df.columns:
            y_field = config.y_axis if isinstance(config.y_axis, str) else config.y_axis[0]
            if y_field in df.columns:
                return df.groupby(config.group_by)[y_field].sum().reset_index()
        return df
    
    def _process_histogram_data(self, df: pd.DataFrame, config: ChartConfiguration) -> pd.DataFrame:
        """Process data for histograms."""
        x_field = config.x_axis
        if x_field in df.columns:
            # Create bins for histogram
            bins = 20  # Default number of bins
            df['bin'] = pd.cut(df[x_field], bins=bins)
            return df.groupby('bin').size().reset_index(name='count')
        return df
    
    def _process_heatmap_data(self, df: pd.DataFrame, config: ChartConfiguration) -> pd.DataFrame:
        """Process data for heatmaps."""
        # For heatmaps, we need x, y, and value columns
        return df
    
    def _generate_chart_config(self, data: List[Dict[str, Any]], config: ChartConfiguration) -> Dict[str, Any]:
        """Generate chart configuration for frontend."""
        colors = self.color_schemes.get(config.color_scheme, self.color_schemes["default"])
        if config.custom_colors:
            colors = config.custom_colors
        
        chart_config = {
            "type": config.chart_type,
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "colors": colors,
            "interactive": config.interactive,
            "showLegend": config.show_legend,
            "showGrid": config.show_grid,
            "showTooltip": config.show_tooltip,
            "xAxis": {
                "dataKey": config.x_axis,
                "label": config.x_axis.replace('_', ' ').title()
            },
            "yAxis": {
                "dataKey": config.y_axis if isinstance(config.y_axis, str) else config.y_axis[0],
                "label": (config.y_axis if isinstance(config.y_axis, str) else config.y_axis[0]).replace('_', ' ').title()
            }
        }
        
        # Add chart-specific configurations
        if config.chart_type == ChartType.LINE:
            chart_config["smooth"] = True
            chart_config["strokeWidth"] = 2
        elif config.chart_type == ChartType.BAR:
            chart_config["barSize"] = 20
        elif config.chart_type == ChartType.PIE:
            chart_config["innerRadius"] = 0
            chart_config["outerRadius"] = 80
        
        return chart_config
    
    async def create_interactive_chart(
        self, 
        data: List[Dict[str, Any]], 
        config: ChartConfiguration,
        interactive_features: InteractiveFeatures
    ) -> Dict[str, Any]:
        """Create an interactive chart with advanced features."""
        try:
            # Base chart creation
            viz_response = await self.create_visualization(data, config)
            
            if not viz_response.success:
                return {"success": False, "error": "Failed to create base chart"}
            
            # Add interactive features
            chart_config = viz_response.data.chart_config
            
            interactive_config = {
                "zoom": interactive_features.zoom,
                "pan": interactive_features.pan,
                "brush": interactive_features.brush,
                "crossfilter": interactive_features.crossfilter,
                "clickActions": interactive_features.click_actions or []
            }
            
            # Add drill-down configuration
            if interactive_features.drill_down:
                interactive_config["drillDown"] = {
                    "enabled": interactive_features.drill_down.enabled,
                    "levels": interactive_features.drill_down.levels,
                    "defaultLevel": interactive_features.drill_down.default_level,
                    "showBreadcrumbs": interactive_features.drill_down.show_breadcrumbs
                }
            
            return {
                "success": True,
                "chart": viz_response.data,
                "interactiveConfig": interactive_config
            }
            
        except Exception as e:
            logger.error(f"Error creating interactive chart: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_chart_suggestions(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest appropriate chart types based on data characteristics."""
        if not data:
            return []
        
        df = pd.DataFrame(data)
        suggestions = []
        
        # Analyze data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Suggest charts based on data characteristics
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": ChartType.SCATTER,
                "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
                "x_axis": numeric_cols[0],
                "y_axis": numeric_cols[1],
                "confidence": 0.8
            })
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append({
                "type": ChartType.BAR,
                "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
                "x_axis": categorical_cols[0],
                "y_axis": numeric_cols[0],
                "confidence": 0.9
            })
            
            suggestions.append({
                "type": ChartType.PIE,
                "title": f"Distribution of {numeric_cols[0]}",
                "x_axis": categorical_cols[0],
                "y_axis": numeric_cols[0],
                "confidence": 0.7
            })
        
        if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append({
                "type": ChartType.LINE,
                "title": f"{numeric_cols[0]} over time",
                "x_axis": datetime_cols[0],
                "y_axis": numeric_cols[0],
                "confidence": 0.95
            })
        
        if len(numeric_cols) >= 1:
            suggestions.append({
                "type": ChartType.HISTOGRAM,
                "title": f"Distribution of {numeric_cols[0]}",
                "x_axis": numeric_cols[0],
                "y_axis": "count",
                "confidence": 0.6
            })
        
        return sorted(suggestions, key=lambda x: x["confidence"], reverse=True)
    
    def calculate_chart_statistics(self, data: List[Dict[str, Any]], config: ChartConfiguration) -> Dict[str, Any]:
        """Calculate statistics for chart data."""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        stats = {
            "total_records": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict()
        }
        
        # Calculate statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        # Calculate statistics for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats["categorical_stats"] = {}
            for col in categorical_cols:
                stats["categorical_stats"][col] = {
                    "unique_values": df[col].nunique(),
                    "most_common": df[col].value_counts().head().to_dict()
                }
        
        return stats