"""Operational dashboard generator for AI Data Readiness Platform."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

from .platform_monitor import get_platform_monitor
from .resource_optimizer import get_resource_optimizer
from ..models.monitoring_models import (
    MonitoringDashboard, Alert, AlertSeverity, HealthStatus,
    MetricDefinition, ALL_METRICS
)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    id: str
    title: str
    widget_type: str  # chart, metric, alert, table
    data_source: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height
    refresh_interval: int = 60  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'widget_type': self.widget_type,
            'data_source': self.data_source,
            'config': self.config,
            'position': self.position,
            'refresh_interval': self.refresh_interval
        }


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    name: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    grid_columns: int = 12
    grid_rows: int = 8
    auto_refresh: bool = True
    refresh_interval: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'widgets': [w.to_dict() for w in self.widgets],
            'grid_columns': self.grid_columns,
            'grid_rows': self.grid_rows,
            'auto_refresh': self.auto_refresh,
            'refresh_interval': self.refresh_interval
        }


class OperationalDashboardGenerator:
    """Generator for operational monitoring dashboards."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitor = get_platform_monitor()
        self.optimizer = get_resource_optimizer()
        
        # Dashboard templates
        self.dashboard_templates = {
            'system_overview': self._create_system_overview_template(),
            'platform_health': self._create_platform_health_template(),
            'performance_monitoring': self._create_performance_monitoring_template(),
            'capacity_planning': self._create_capacity_planning_template(),
            'alert_management': self._create_alert_management_template(),
            'executive_summary': self._create_executive_summary_template()
        }
    
    def create_dashboard(self, template_name: str, 
                        custom_config: Optional[Dict[str, Any]] = None) -> MonitoringDashboard:
        """Create a dashboard from a template."""
        if template_name not in self.dashboard_templates:
            raise ValueError(f"Unknown dashboard template: {template_name}")
        
        template = self.dashboard_templates[template_name]
        
        # Apply custom configuration if provided
        if custom_config:
            template = self._apply_custom_config(template, custom_config)
        
        # Create monitoring dashboard
        dashboard = MonitoringDashboard(
            name=template.name,
            description=template.description,
            metrics=[w.data_source for w in template.widgets if w.widget_type == 'metric'],
            refresh_interval_seconds=template.refresh_interval,
            layout=template.to_dict()
        )
        
        return dashboard
    
    def generate_dashboard_html(self, dashboard: MonitoringDashboard, 
                               time_range_hours: int = 24) -> str:
        """Generate HTML dashboard with real-time data."""
        try:
            # Get dashboard layout
            layout = DashboardLayout(**dashboard.layout)
            
            # Generate HTML structure
            html_content = self._generate_html_structure(layout)
            
            # Add real-time data
            dashboard_data = self._collect_dashboard_data(layout, time_range_hours)
            
            # Generate charts and widgets
            charts_html = self._generate_charts_html(layout, dashboard_data)
            
            # Combine everything
            full_html = self._combine_html_components(html_content, charts_html, dashboard_data)
            
            return full_html
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard HTML: {e}")
            raise
    
    def get_dashboard_data(self, dashboard: MonitoringDashboard, 
                          time_range_hours: int = 24) -> Dict[str, Any]:
        """Get real-time data for dashboard."""
        layout = DashboardLayout(**dashboard.layout)
        return self._collect_dashboard_data(layout, time_range_hours)
    
    def _create_system_overview_template(self) -> DashboardLayout:
        """Create system overview dashboard template."""
        widgets = [
            DashboardWidget(
                id="cpu_usage",
                title="CPU Usage",
                widget_type="chart",
                data_source="system_metrics",
                config={"chart_type": "line", "metric": "cpu_percent"},
                position={"x": 0, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                id="memory_usage",
                title="Memory Usage",
                widget_type="chart",
                data_source="system_metrics",
                config={"chart_type": "line", "metric": "memory_percent"},
                position={"x": 3, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                id="disk_usage",
                title="Disk Usage",
                widget_type="chart",
                data_source="system_metrics",
                config={"chart_type": "gauge", "metric": "disk_usage_percent"},
                position={"x": 6, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                id="network_activity",
                title="Network Activity",
                widget_type="chart",
                data_source="system_metrics",
                config={"chart_type": "area", "metrics": ["network_bytes_sent", "network_bytes_recv"]},
                position={"x": 9, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                id="system_health",
                title="System Health Status",
                widget_type="metric",
                data_source="health_status",
                config={"display_type": "status_card"},
                position={"x": 0, "y": 2, "width": 6, "height": 1}
            ),
            DashboardWidget(
                id="active_connections",
                title="Active Connections",
                widget_type="metric",
                data_source="system_metrics",
                config={"metric": "active_connections", "display_type": "number"},
                position={"x": 6, "y": 2, "width": 3, "height": 1}
            ),
            DashboardWidget(
                id="uptime",
                title="System Uptime",
                widget_type="metric",
                data_source="system_info",
                config={"metric": "uptime", "display_type": "duration"},
                position={"x": 9, "y": 2, "width": 3, "height": 1}
            )
        ]
        
        return DashboardLayout(
            name="System Overview",
            description="Comprehensive system resource monitoring",
            widgets=widgets,
            refresh_interval=30
        )
    
    def _create_platform_health_template(self) -> DashboardLayout:
        """Create platform health dashboard template."""
        widgets = [
            DashboardWidget(
                id="processing_pipeline",
                title="Data Processing Pipeline",
                widget_type="chart",
                data_source="platform_metrics",
                config={"chart_type": "funnel", "metrics": ["active_datasets", "processing_datasets", "failed_datasets"]},
                position={"x": 0, "y": 0, "width": 4, "height": 3}
            ),
            DashboardWidget(
                id="quality_scores",
                title="Data Quality Trends",
                widget_type="chart",
                data_source="platform_metrics",
                config={"chart_type": "line", "metric": "avg_quality_score"},
                position={"x": 4, "y": 0, "width": 4, "height": 3}
            ),
            DashboardWidget(
                id="error_rate",
                title="Error Rate",
                widget_type="chart",
                data_source="platform_metrics",
                config={"chart_type": "line", "metric": "error_rate_percent"},
                position={"x": 8, "y": 0, "width": 4, "height": 3}
            ),
            DashboardWidget(
                id="api_performance",
                title="API Performance",
                widget_type="chart",
                data_source="platform_metrics",
                config={"chart_type": "histogram", "metric": "api_requests_per_minute"},
                position={"x": 0, "y": 3, "width": 6, "height": 2}
            ),
            DashboardWidget(
                id="processing_time",
                title="Avg Processing Time",
                widget_type="metric",
                data_source="platform_metrics",
                config={"metric": "avg_processing_time_seconds", "display_type": "duration"},
                position={"x": 6, "y": 3, "width": 3, "height": 1}
            ),
            DashboardWidget(
                id="data_volume",
                title="Data Processed (GB)",
                widget_type="metric",
                data_source="platform_metrics",
                config={"metric": "total_data_processed_gb", "display_type": "number"},
                position={"x": 9, "y": 3, "width": 3, "height": 1}
            )
        ]
        
        return DashboardLayout(
            name="Platform Health",
            description="AI Data Readiness Platform health monitoring",
            widgets=widgets,
            refresh_interval=60
        )
    
    def _create_performance_monitoring_template(self) -> DashboardLayout:
        """Create performance monitoring dashboard template."""
        widgets = [
            DashboardWidget(
                id="throughput_trends",
                title="Throughput Trends",
                widget_type="chart",
                data_source="performance_metrics",
                config={"chart_type": "line", "metric": "throughput_ops_per_sec"},
                position={"x": 0, "y": 0, "width": 6, "height": 3}
            ),
            DashboardWidget(
                id="response_times",
                title="Response Time Distribution",
                widget_type="chart",
                data_source="performance_metrics",
                config={"chart_type": "box", "metric": "duration_seconds"},
                position={"x": 6, "y": 0, "width": 6, "height": 3}
            ),
            DashboardWidget(
                id="operation_success_rate",
                title="Operation Success Rate",
                widget_type="chart",
                data_source="performance_metrics",
                config={"chart_type": "pie", "metric": "success_rate"},
                position={"x": 0, "y": 3, "width": 4, "height": 2}
            ),
            DashboardWidget(
                id="concurrent_operations",
                title="Concurrent Operations",
                widget_type="chart",
                data_source="performance_metrics",
                config={"chart_type": "area", "metric": "concurrent_count"},
                position={"x": 4, "y": 3, "width": 4, "height": 2}
            ),
            DashboardWidget(
                id="queue_depth",
                title="Processing Queue",
                widget_type="metric",
                data_source="performance_metrics",
                config={"metric": "queue_depth", "display_type": "number"},
                position={"x": 8, "y": 3, "width": 2, "height": 1}
            ),
            DashboardWidget(
                id="avg_latency",
                title="Avg Latency (ms)",
                widget_type="metric",
                data_source="performance_metrics",
                config={"metric": "avg_latency_ms", "display_type": "number"},
                position={"x": 10, "y": 3, "width": 2, "height": 1}
            )
        ]
        
        return DashboardLayout(
            name="Performance Monitoring",
            description="Detailed performance metrics and trends",
            widgets=widgets,
            refresh_interval=30
        )
    
    def _create_capacity_planning_template(self) -> DashboardLayout:
        """Create capacity planning dashboard template."""
        widgets = [
            DashboardWidget(
                id="resource_utilization",
                title="Resource Utilization Trends",
                widget_type="chart",
                data_source="capacity_metrics",
                config={"chart_type": "multi_line", "metrics": ["cpu_utilization", "memory_utilization", "disk_utilization"]},
                position={"x": 0, "y": 0, "width": 8, "height": 3}
            ),
            DashboardWidget(
                id="capacity_forecast",
                title="Capacity Forecast",
                widget_type="chart",
                data_source="capacity_planning",
                config={"chart_type": "forecast", "time_horizon_days": 30},
                position={"x": 8, "y": 0, "width": 4, "height": 3}
            ),
            DashboardWidget(
                id="scaling_recommendations",
                title="Scaling Recommendations",
                widget_type="table",
                data_source="optimization_recommendations",
                config={"columns": ["component", "current_util", "projected_util", "recommendation"]},
                position={"x": 0, "y": 3, "width": 8, "height": 2}
            ),
            DashboardWidget(
                id="cost_projection",
                title="Cost Impact",
                widget_type="metric",
                data_source="capacity_planning",
                config={"metric": "estimated_cost_impact", "display_type": "currency"},
                position={"x": 8, "y": 3, "width": 4, "height": 1}
            )
        ]
        
        return DashboardLayout(
            name="Capacity Planning",
            description="Resource capacity planning and optimization",
            widgets=widgets,
            refresh_interval=300  # 5 minutes
        )
    
    def _create_alert_management_template(self) -> DashboardLayout:
        """Create alert management dashboard template."""
        widgets = [
            DashboardWidget(
                id="alert_overview",
                title="Alert Overview",
                widget_type="chart",
                data_source="alerts",
                config={"chart_type": "donut", "group_by": "severity"},
                position={"x": 0, "y": 0, "width": 4, "height": 3}
            ),
            DashboardWidget(
                id="alert_timeline",
                title="Alert Timeline",
                widget_type="chart",
                data_source="alerts",
                config={"chart_type": "timeline", "time_range_hours": 24},
                position={"x": 4, "y": 0, "width": 8, "height": 3}
            ),
            DashboardWidget(
                id="active_alerts",
                title="Active Alerts",
                widget_type="table",
                data_source="alerts",
                config={"filter": "active", "columns": ["severity", "title", "created_at", "actions"]},
                position={"x": 0, "y": 3, "width": 12, "height": 3}
            ),
            DashboardWidget(
                id="mttr",
                title="Mean Time to Resolution",
                widget_type="metric",
                data_source="alert_metrics",
                config={"metric": "mttr_minutes", "display_type": "duration"},
                position={"x": 0, "y": 6, "width": 3, "height": 1}
            ),
            DashboardWidget(
                id="alert_rate",
                title="Alert Rate (per hour)",
                widget_type="metric",
                data_source="alert_metrics",
                config={"metric": "alerts_per_hour", "display_type": "number"},
                position={"x": 3, "y": 6, "width": 3, "height": 1}
            )
        ]
        
        return DashboardLayout(
            name="Alert Management",
            description="Alert monitoring and incident management",
            widgets=widgets,
            refresh_interval=60
        )
    
    def _create_executive_summary_template(self) -> DashboardLayout:
        """Create executive summary dashboard template."""
        widgets = [
            DashboardWidget(
                id="kpi_overview",
                title="Key Performance Indicators",
                widget_type="metric",
                data_source="executive_metrics",
                config={"display_type": "kpi_grid", "metrics": ["uptime", "success_rate", "avg_quality", "cost_efficiency"]},
                position={"x": 0, "y": 0, "width": 12, "height": 2}
            ),
            DashboardWidget(
                id="business_impact",
                title="Business Impact Metrics",
                widget_type="chart",
                data_source="business_metrics",
                config={"chart_type": "multi_bar", "metrics": ["datasets_processed", "quality_improvements", "cost_savings"]},
                position={"x": 0, "y": 2, "width": 8, "height": 3}
            ),
            DashboardWidget(
                id="health_summary",
                title="Platform Health Summary",
                widget_type="metric",
                data_source="health_status",
                config={"display_type": "health_card"},
                position={"x": 8, "y": 2, "width": 4, "height": 3}
            ),
            DashboardWidget(
                id="trend_analysis",
                title="30-Day Trend Analysis",
                widget_type="chart",
                data_source="trend_metrics",
                config={"chart_type": "trend_lines", "time_range_days": 30},
                position={"x": 0, "y": 5, "width": 12, "height": 3}
            )
        ]
        
        return DashboardLayout(
            name="Executive Summary",
            description="High-level executive dashboard with key metrics",
            widgets=widgets,
            refresh_interval=300  # 5 minutes
        )
    
    def _collect_dashboard_data(self, layout: DashboardLayout, 
                               time_range_hours: int) -> Dict[str, Any]:
        """Collect data for all dashboard widgets."""
        data = {}
        
        try:
            # Get metrics history
            metrics_history = self.monitor.get_metrics_history(hours=time_range_hours)
            resource_history = self.optimizer.get_resource_history(hours=time_range_hours)
            
            # Current metrics
            current_system = self.monitor.get_current_system_metrics()
            current_platform = self.monitor.get_current_platform_metrics()
            health_status = self.monitor.get_health_status()
            
            # Organize data by source
            data['system_metrics'] = {
                'current': current_system.to_dict() if current_system else {},
                'history': metrics_history.get('system_metrics', [])
            }
            
            data['platform_metrics'] = {
                'current': current_platform.to_dict() if current_platform else {},
                'history': metrics_history.get('platform_metrics', [])
            }
            
            data['performance_metrics'] = {
                'history': metrics_history.get('performance_metrics', [])
            }
            
            data['health_status'] = health_status
            
            # Generate sample alerts for demonstration
            data['alerts'] = self._generate_sample_alerts()
            
            # Executive metrics
            data['executive_metrics'] = self._calculate_executive_metrics(
                current_system, current_platform, health_status
            )
            
            data['capacity_metrics'] = resource_history
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error collecting dashboard data: {e}")
            return {}
    
    def _generate_sample_alerts(self) -> List[Dict[str, Any]]:
        """Generate sample alerts for dashboard."""
        alerts = []
        
        # Get current health status to generate relevant alerts
        health_status = self.monitor.get_health_status()
        
        if health_status['status'] in ['warning', 'critical']:
            for issue in health_status.get('issues', []):
                severity = AlertSeverity.WARNING if health_status['status'] == 'warning' else AlertSeverity.CRITICAL
                alert = Alert(
                    severity=severity,
                    title="System Health Alert",
                    description=issue,
                    created_at=datetime.utcnow()
                )
                alerts.append(alert.to_dict())
        
        return alerts
    
    def _calculate_executive_metrics(self, system_metrics, platform_metrics, health_status) -> Dict[str, Any]:
        """Calculate executive-level metrics."""
        if not system_metrics or not platform_metrics:
            return {}
        
        return {
            'uptime': '99.9%',  # Placeholder
            'success_rate': f"{100 - platform_metrics.get('error_rate_percent', 0):.1f}%",
            'avg_quality': '85.2%',  # Placeholder
            'cost_efficiency': '$0.15/GB',  # Placeholder
            'datasets_processed': platform_metrics.get('active_datasets', 0),
            'quality_improvements': '12%',  # Placeholder
            'cost_savings': '$2,450'  # Placeholder
        }
    
    def _generate_html_structure(self, layout: DashboardLayout) -> str:
        """Generate basic HTML structure for dashboard."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{layout.name} - AI Data Readiness Platform</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <style>
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat({layout.grid_columns}, 1fr);
                    grid-template-rows: repeat({layout.grid_rows}, 1fr);
                    gap: 1rem;
                    height: 100vh;
                    padding: 1rem;
                }}
                .widget {{
                    background: white;
                    border-radius: 0.5rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    padding: 1rem;
                    overflow: hidden;
                }}
                .widget-title {{
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                    color: #374151;
                }}
                .metric-value {{
                    font-size: 2rem;
                    font-weight: 700;
                    color: #1f2937;
                }}
                .metric-label {{
                    font-size: 0.875rem;
                    color: #6b7280;
                }}
                .status-healthy {{ color: #10b981; }}
                .status-warning {{ color: #f59e0b; }}
                .status-critical {{ color: #ef4444; }}
            </style>
        </head>
        <body class="bg-gray-100">
            <div class="dashboard-header bg-white shadow-sm p-4 mb-4">
                <h1 class="text-2xl font-bold text-gray-900">{layout.name}</h1>
                <p class="text-gray-600">{layout.description}</p>
                <div class="text-sm text-gray-500 mt-2">
                    Last updated: <span id="last-updated">{{timestamp}}</span>
                    | Auto-refresh: {layout.refresh_interval}s
                </div>
            </div>
            <div class="dashboard-grid">
                {{widgets_html}}
            </div>
            <script>
                // Auto-refresh functionality
                setInterval(function() {{
                    location.reload();
                }}, {layout.refresh_interval * 1000});
                
                // Update timestamp
                document.getElementById('last-updated').textContent = new Date().toLocaleString();
            </script>
        </body>
        </html>
        """
    
    def _generate_charts_html(self, layout: DashboardLayout, data: Dict[str, Any]) -> str:
        """Generate HTML for all charts and widgets."""
        widgets_html = ""
        
        for widget in layout.widgets:
            widget_html = self._generate_widget_html(widget, data)
            widgets_html += widget_html
        
        return widgets_html
    
    def _generate_widget_html(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate HTML for a single widget."""
        # Handle both DashboardWidget objects and dict representations
        if isinstance(widget, dict):
            position = widget.get('position', {})
            widget_id = widget.get('id', 'unknown')
            title = widget.get('title', 'Unknown Widget')
            widget_type = widget.get('widget_type', 'unknown')
            data_source = widget.get('data_source', '')
            config = widget.get('config', {})
        else:
            position = widget.position
            widget_id = widget.id
            title = widget.title
            widget_type = widget.widget_type
            data_source = widget.data_source
            config = widget.config
        grid_style = f"""
            grid-column: {position.get('x', 0) + 1} / span {position.get('width', 1)};
            grid-row: {position.get('y', 0) + 1} / span {position.get('height', 1)};
        """
        
        if widget_type == "chart":
            content = self._generate_chart_content_dict(widget_id, config, data_source, data)
        elif widget_type == "metric":
            content = self._generate_metric_content_dict(config, data_source, data)
        elif widget_type == "table":
            content = self._generate_table_content_dict(config, data_source, data)
        else:
            content = f"<p>Widget type '{widget_type}' not implemented</p>"
        
        return f"""
        <div class="widget" style="{grid_style}">
            <div class="widget-title">{title}</div>
            {content}
        </div>
        """
    
    def _generate_chart_content(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate chart content for widget."""
        widget_data = data.get(widget.data_source, {})
        chart_id = f"chart_{widget.id}"
        
        # Simple placeholder chart
        return f"""
        <div id="{chart_id}" style="height: 100%;"></div>
        <script>
            var data = [{{"x": [1,2,3,4], "y": [10,11,12,13], "type": "scatter"}}];
            var layout = {{"margin": {{"t": 0, "r": 0, "b": 30, "l": 40}}}};
            Plotly.newPlot('{chart_id}', data, layout, {{"responsive": true}});
        </script>
        """
    
    def _generate_chart_content_dict(self, widget_id: str, config: Dict[str, Any], 
                                   data_source: str, data: Dict[str, Any]) -> str:
        """Generate chart content for widget from dict."""
        widget_data = data.get(data_source, {})
        chart_id = f"chart_{widget_id}"
        
        # Simple placeholder chart
        return f"""
        <div id="{chart_id}" style="height: 100%;"></div>
        <script>
            var data = [{{"x": [1,2,3,4], "y": [10,11,12,13], "type": "scatter"}}];
            var layout = {{"margin": {{"t": 0, "r": 0, "b": 30, "l": 40}}}};
            Plotly.newPlot('{chart_id}', data, layout, {{"responsive": true}});
        </script>
        """
    
    def _generate_metric_content(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate metric content for widget."""
        widget_data = data.get(widget.data_source, {})
        
        if widget.data_source == "health_status":
            status = widget_data.get('status', 'unknown')
            status_class = f"status-{status}"
            return f"""
            <div class="text-center">
                <div class="metric-value {status_class}">{status.upper()}</div>
                <div class="metric-label">Platform Status</div>
            </div>
            """
        else:
            # Generic metric display
            metric_name = widget.config.get('metric', 'unknown')
            current_data = widget_data.get('current', {})
            value = current_data.get(metric_name, 'N/A')
            
            return f"""
            <div class="text-center">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{metric_name.replace('_', ' ').title()}</div>
            </div>
            """
    
    def _generate_metric_content_dict(self, config: Dict[str, Any], 
                                    data_source: str, data: Dict[str, Any]) -> str:
        """Generate metric content for widget from dict."""
        widget_data = data.get(data_source, {})
        
        if data_source == "health_status":
            status = widget_data.get('status', 'unknown')
            status_class = f"status-{status}"
            return f"""
            <div class="text-center">
                <div class="metric-value {status_class}">{status.upper()}</div>
                <div class="metric-label">Platform Status</div>
            </div>
            """
        else:
            # Generic metric display
            metric_name = config.get('metric', 'unknown')
            current_data = widget_data.get('current', {})
            value = current_data.get(metric_name, 'N/A')
            
            return f"""
            <div class="text-center">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{metric_name.replace('_', ' ').title()}</div>
            </div>
            """
    
    def _generate_table_content(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate table content for widget."""
        return """
        <div class="overflow-auto">
            <table class="min-w-full text-sm">
                <thead>
                    <tr class="border-b">
                        <th class="text-left p-2">Item</th>
                        <th class="text-left p-2">Value</th>
                        <th class="text-left p-2">Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="border-b">
                        <td class="p-2">Sample Item</td>
                        <td class="p-2">Sample Value</td>
                        <td class="p-2">OK</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
    
    def _generate_table_content_dict(self, config: Dict[str, Any], 
                                   data_source: str, data: Dict[str, Any]) -> str:
        """Generate table content for widget from dict."""
        return """
        <div class="overflow-auto">
            <table class="min-w-full text-sm">
                <thead>
                    <tr class="border-b">
                        <th class="text-left p-2">Item</th>
                        <th class="text-left p-2">Value</th>
                        <th class="text-left p-2">Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="border-b">
                        <td class="p-2">Sample Item</td>
                        <td class="p-2">Sample Value</td>
                        <td class="p-2">OK</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
    
    def _combine_html_components(self, html_structure: str, charts_html: str, 
                                data: Dict[str, Any]) -> str:
        """Combine all HTML components into final dashboard."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        return html_structure.replace("{{widgets_html}}", charts_html).replace("{{timestamp}}", timestamp)
    
    def _apply_custom_config(self, template: DashboardLayout, 
                           custom_config: Dict[str, Any]) -> DashboardLayout:
        """Apply custom configuration to dashboard template."""
        # This would apply custom configurations like colors, thresholds, etc.
        # For now, just return the template as-is
        return template
    
    def export_dashboard_config(self, dashboard: MonitoringDashboard, 
                               filepath: str):
        """Export dashboard configuration to file."""
        config = dashboard.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Dashboard configuration exported to {filepath}")
    
    def import_dashboard_config(self, filepath: str) -> MonitoringDashboard:
        """Import dashboard configuration from file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return MonitoringDashboard(**config)


# Global dashboard generator instance
_dashboard_generator = None


def get_dashboard_generator() -> OperationalDashboardGenerator:
    """Get global dashboard generator instance."""
    global _dashboard_generator
    if _dashboard_generator is None:
        _dashboard_generator = OperationalDashboardGenerator()
    return _dashboard_generator