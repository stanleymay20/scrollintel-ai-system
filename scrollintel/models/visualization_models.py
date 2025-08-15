"""
Data visualization and export models for ScrollIntel.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class ChartType(str, Enum):
    """Supported chart types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    FUNNEL = "funnel"
    GAUGE = "gauge"


class ExportFormat(str, Enum):
    """Supported export formats."""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    PNG = "png"
    SVG = "svg"
    JSON = "json"


class FilterOperator(str, Enum):
    """Filter operators for data filtering."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"


class DataFilter(BaseModel):
    """Data filter configuration."""
    field: str
    operator: FilterOperator
    value: Union[str, int, float, List[Any]]
    label: Optional[str] = None


class ChartConfiguration(BaseModel):
    """Chart configuration model."""
    chart_type: ChartType
    title: str
    x_axis: str
    y_axis: Union[str, List[str]]
    color_scheme: Optional[str] = "default"
    width: Optional[int] = 800
    height: Optional[int] = 400
    interactive: bool = True
    show_legend: bool = True
    show_grid: bool = True
    show_tooltip: bool = True
    custom_colors: Optional[List[str]] = None
    aggregation: Optional[str] = None  # sum, avg, count, etc.
    group_by: Optional[str] = None
    filters: Optional[List[DataFilter]] = []


class DashboardLayout(BaseModel):
    """Dashboard layout configuration."""
    id: str
    name: str
    description: Optional[str] = None
    layout: List[Dict[str, Any]]  # Grid layout configuration
    charts: List[str]  # Chart IDs
    filters: Optional[List[DataFilter]] = []
    refresh_interval: Optional[int] = 300  # seconds
    is_public: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ExportRequest(BaseModel):
    """Export request model."""
    format: ExportFormat
    chart_ids: Optional[List[str]] = None
    dashboard_id: Optional[str] = None
    include_data: bool = True
    include_metadata: bool = True
    filters: Optional[List[DataFilter]] = []
    custom_title: Optional[str] = None
    custom_description: Optional[str] = None


class VisualizationData(BaseModel):
    """Visualization data model."""
    id: str
    name: str
    description: Optional[str] = None
    data: List[Dict[str, Any]]
    chart_config: ChartConfiguration
    metadata: Optional[Dict[str, Any]] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DrillDownConfig(BaseModel):
    """Drill-down configuration."""
    enabled: bool = True
    levels: List[str]  # Field names for drill-down levels
    default_level: int = 0
    show_breadcrumbs: bool = True


class InteractiveFeatures(BaseModel):
    """Interactive features configuration."""
    zoom: bool = True
    pan: bool = True
    brush: bool = False  # For selecting data ranges
    crossfilter: bool = False  # For cross-chart filtering
    drill_down: Optional[DrillDownConfig] = None
    click_actions: Optional[List[str]] = []  # Custom click actions


class PrintLayout(BaseModel):
    """Print-friendly layout configuration."""
    page_size: str = "A4"  # A4, Letter, etc.
    orientation: str = "portrait"  # portrait, landscape
    margins: Dict[str, float] = {"top": 1, "bottom": 1, "left": 1, "right": 1}
    header: Optional[str] = None
    footer: Optional[str] = None
    charts_per_page: int = 2
    include_summary: bool = True
    include_filters: bool = True


class VisualizationTemplate(BaseModel):
    """Visualization template model."""
    id: str
    name: str
    description: Optional[str] = None
    category: str
    chart_configs: List[ChartConfiguration]
    dashboard_layout: Optional[DashboardLayout] = None
    sample_data: Optional[List[Dict[str, Any]]] = []
    is_public: bool = True
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DataSource(BaseModel):
    """Data source configuration."""
    id: str
    name: str
    type: str  # file, database, api, etc.
    connection_string: Optional[str] = None
    query: Optional[str] = None
    file_path: Optional[str] = None
    refresh_rate: Optional[int] = None  # minutes
    last_updated: Optional[datetime] = None
    schema: Optional[Dict[str, str]] = {}  # field_name: data_type


class VisualizationResponse(BaseModel):
    """API response for visualization requests."""
    success: bool
    data: Optional[VisualizationData] = None
    charts: Optional[List[VisualizationData]] = None
    dashboard: Optional[DashboardLayout] = None
    export_url: Optional[str] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = []


class ChartAnalytics(BaseModel):
    """Chart analytics and usage tracking."""
    chart_id: str
    views: int = 0
    exports: int = 0
    interactions: int = 0
    avg_view_time: float = 0.0
    last_viewed: Optional[datetime] = None
    popular_filters: List[str] = []
    user_feedback: Optional[float] = None  # 1-5 rating