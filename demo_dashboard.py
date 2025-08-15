#!/usr/bin/env python3
"""
Demo script for Advanced Analytics Dashboard System.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.core.dashboard_manager import (
    DashboardManager, DashboardConfig, SharePermissions, TimeRange
)
from scrollintel.core.dashboard_templates import DashboardTemplates
from scrollintel.models.dashboard_models import ExecutiveRole, WidgetType


class MockDashboard:
    """Mock dashboard for demonstration."""
    def __init__(self, id: str, name: str, type: str, role: str = None):
        self.id = id
        self.name = name
        self.type = type
        self.role = role
        self.config = {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.widgets = []


class MockWidget:
    """Mock widget for demonstration."""
    def __init__(self, id: str, name: str, type: str):
        self.id = id
        self.name = name
        self.type = type
        self.position_x = 0
        self.position_y = 0
        self.width = 4
        self.height = 2
        self.config = {}
        self.data_source = "mock_source"
        self.updated_at = datetime.utcnow()
        self.is_active = True


class MockBusinessMetric:
    """Mock business metric for demonstration."""
    def __init__(self, name: str, category: str, value: float, unit: str):
        self.id = f"metric_{name}"
        self.name = name
        self.category = category
        self.value = value
        self.unit = unit
        self.timestamp = datetime.utcnow()
        self.source = "mock_source"
        self.context = {"trend": "up"}


def demo_dashboard_config():
    """Demonstrate dashboard configuration."""
    print("=== Dashboard Configuration Demo ===")
    
    config = DashboardConfig(
        layout={"grid_columns": 12, "grid_rows": 8},
        theme="executive",
        auto_refresh=True,
        refresh_interval=300
    )
    
    print(f"Dashboard Config: {json.dumps(config.to_dict(), indent=2)}")
    print()


def demo_dashboard_templates():
    """Demonstrate dashboard templates."""
    print("=== Dashboard Templates Demo ===")
    
    # Get all templates
    templates = DashboardTemplates.get_all_templates()
    
    print(f"Available templates: {list(templates.keys())}")
    print()
    
    # Show CTO template details
    cto_template = DashboardTemplates.get_cto_template()
    print("CTO Template:")
    print(f"  Name: {cto_template['name']}")
    print(f"  Description: {cto_template['description']}")
    print(f"  Widgets: {len(cto_template['widgets'])}")
    
    for i, widget in enumerate(cto_template['widgets']):
        print(f"    Widget {i+1}: {widget['name']} ({widget['type']})")
    print()


def demo_executive_roles():
    """Demonstrate executive roles and their default widgets."""
    print("=== Executive Roles Demo ===")
    
    roles = [ExecutiveRole.CTO, ExecutiveRole.CFO, ExecutiveRole.CEO]
    
    for role in roles:
        print(f"{role.value.upper()} Role:")
        template = DashboardTemplates.get_template_by_role(role.value)
        if template:
            print(f"  Template: {template['name']}")
            print(f"  Widgets: {len(template.get('widgets', []))}")
        print()


def demo_widget_types():
    """Demonstrate widget types."""
    print("=== Widget Types Demo ===")
    
    widget_types = [WidgetType.KPI, WidgetType.CHART, WidgetType.METRIC, WidgetType.TABLE]
    
    for widget_type in widget_types:
        print(f"Widget Type: {widget_type.value}")
        
        # Create mock widget
        widget = MockWidget(f"widget_{widget_type.value}", f"Sample {widget_type.value}", widget_type.value)
        print(f"  Sample Widget: {widget.name}")
        print(f"  Position: ({widget.position_x}, {widget.position_y})")
        print(f"  Size: {widget.width}x{widget.height}")
        print()


def demo_business_metrics():
    """Demonstrate business metrics."""
    print("=== Business Metrics Demo ===")
    
    metrics = [
        MockBusinessMetric("tech_roi", "financial", 18.5, "%"),
        MockBusinessMetric("ai_investment_return", "financial", 250000, "USD"),
        MockBusinessMetric("system_uptime", "operational", 99.8, "%"),
        MockBusinessMetric("deployment_frequency", "operational", 12, "per week"),
        MockBusinessMetric("customer_satisfaction", "business", 4.2, "stars"),
    ]
    
    for metric in metrics:
        print(f"Metric: {metric.name}")
        print(f"  Category: {metric.category}")
        print(f"  Value: {metric.value} {metric.unit}")
        print(f"  Trend: {metric.context.get('trend', 'neutral')}")
        print(f"  Timestamp: {metric.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print()


def demo_time_range():
    """Demonstrate time range functionality."""
    print("=== Time Range Demo ===")
    
    # Create different time ranges
    ranges = [
        ("Last 7 days", datetime.utcnow() - timedelta(days=7), datetime.utcnow()),
        ("Last 30 days", datetime.utcnow() - timedelta(days=30), datetime.utcnow()),
        ("This month", datetime.utcnow().replace(day=1), datetime.utcnow()),
    ]
    
    for name, start, end in ranges:
        time_range = TimeRange(start, end)
        print(f"{name}:")
        print(f"  Start: {time_range.start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End: {time_range.end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {(time_range.end - time_range.start).days} days")
        print()


def demo_share_permissions():
    """Demonstrate share permissions."""
    print("=== Share Permissions Demo ===")
    
    permissions = SharePermissions(
        users=["user1@company.com", "user2@company.com", "manager@company.com"],
        permission_type="view",
        expires_in_days=30
    )
    
    print(f"Share with users: {permissions.users}")
    print(f"Permission type: {permissions.permission_type}")
    print(f"Expires in: {permissions.expires_in_days} days")
    print()


def demo_dashboard_data_structure():
    """Demonstrate dashboard data structure."""
    print("=== Dashboard Data Structure Demo ===")
    
    # Create mock dashboard
    dashboard = MockDashboard("dash_123", "CTO Executive Dashboard", "executive", "cto")
    
    # Create mock widgets
    widgets = [
        MockWidget("widget_1", "Technology ROI", WidgetType.KPI.value),
        MockWidget("widget_2", "AI Performance", WidgetType.CHART.value),
        MockWidget("widget_3", "System Health", WidgetType.METRIC.value),
    ]
    
    # Create mock metrics
    metrics = [
        MockBusinessMetric("tech_roi", "financial", 18.5, "%"),
        MockBusinessMetric("ai_accuracy", "operational", 94.2, "%"),
        MockBusinessMetric("system_uptime", "operational", 99.8, "%"),
    ]
    
    # Simulate dashboard data
    dashboard_data = {
        "dashboard": {
            "id": dashboard.id,
            "name": dashboard.name,
            "type": dashboard.type,
            "role": dashboard.role,
            "updated_at": dashboard.updated_at.isoformat()
        },
        "widgets_data": {
            widget.id: {
                "id": widget.id,
                "name": widget.name,
                "type": widget.type,
                "position": {"x": widget.position_x, "y": widget.position_y},
                "size": {"width": widget.width, "height": widget.height},
                "config": widget.config,
                "data_source": widget.data_source,
                "last_updated": widget.updated_at.isoformat()
            }
            for widget in widgets
        },
        "metrics": [
            {
                "id": metric.id,
                "name": metric.name,
                "category": metric.category,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "source": metric.source,
                "context": metric.context
            }
            for metric in metrics
        ],
        "last_updated": datetime.utcnow().isoformat()
    }
    
    print("Dashboard Data Structure:")
    print(json.dumps(dashboard_data, indent=2))
    print()


def demo_websocket_messages():
    """Demonstrate WebSocket message formats."""
    print("=== WebSocket Messages Demo ===")
    
    # Dashboard update message
    dashboard_update = {
        "type": "dashboard_update",
        "data": {
            "dashboard": {"id": "dash_123", "name": "CTO Dashboard"},
            "widgets": {},
            "metrics": []
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Metrics update message
    metrics_update = {
        "type": "metrics_update",
        "data": [
            {
                "id": "metric_1",
                "name": "tech_roi",
                "value": 19.2,
                "unit": "%",
                "timestamp": datetime.utcnow().isoformat()
            }
        ],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Alert message
    alert_message = {
        "type": "alert",
        "data": {
            "id": "alert_1",
            "type": "warning",
            "title": "ROI Below Target",
            "message": "Technology ROI has dropped below 15% target",
            "dashboard_id": "dash_123"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    messages = [
        ("Dashboard Update", dashboard_update),
        ("Metrics Update", metrics_update),
        ("Alert", alert_message)
    ]
    
    for name, message in messages:
        print(f"{name} Message:")
        print(json.dumps(message, indent=2))
        print()


def main():
    """Run all dashboard demos."""
    print("🚀 Advanced Analytics Dashboard System Demo")
    print("=" * 50)
    print()
    
    try:
        demo_dashboard_config()
        demo_dashboard_templates()
        demo_executive_roles()
        demo_widget_types()
        demo_business_metrics()
        demo_time_range()
        demo_share_permissions()
        demo_dashboard_data_structure()
        demo_websocket_messages()
        
        print("✅ All dashboard demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()