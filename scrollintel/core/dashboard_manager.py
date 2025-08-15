"""
Dashboard Manager for Advanced Analytics Dashboard System.
Handles CRUD operations, templates, and dashboard management.
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from datetime import datetime, timedelta
import json

from ..models.dashboard_models import (
    AnalyticsDashboard, Widget, DashboardPermission, DashboardTemplate, 
    BusinessMetric, DashboardType, ExecutiveRole, WidgetType
)
from ..models.database_utils import get_sync_db
from ..core.config import get_settings


class DashboardConfig:
    def __init__(self, layout: Dict[str, Any] = None, theme: str = "default", 
                 auto_refresh: bool = True, refresh_interval: int = 300):
        self.layout = layout or {}
        self.theme = theme
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout": self.layout,
            "theme": self.theme,
            "auto_refresh": self.auto_refresh,
            "refresh_interval": self.refresh_interval
        }


class SharePermissions:
    def __init__(self, users: List[str], permission_type: str = "view", 
                 expires_in_days: int = None):
        self.users = users
        self.permission_type = permission_type
        self.expires_in_days = expires_in_days


class DashboardData:
    def __init__(self, dashboard: AnalyticsDashboard, widgets_data: Dict[str, Any], 
                 metrics: List[BusinessMetric]):
        self.dashboard = dashboard
        self.widgets_data = widgets_data
        self.metrics = metrics
        self.last_updated = datetime.utcnow()


class TimeRange:
    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end


class ShareLink:
    def __init__(self, url: str, expires_at: datetime = None):
        self.url = url
        self.expires_at = expires_at


class DashboardManager:
    """Manages dashboard creation, updates, and access control."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def create_executive_dashboard(self, role: ExecutiveRole, config: DashboardConfig, 
                                 owner_id: str, name: str = None) -> AnalyticsDashboard:
        """Create a new executive dashboard for a specific role."""
        with get_sync_db() as db:
            # Generate default name if not provided
            if not name:
                name = f"{role.value.replace('_', ' ').title()} Executive Dashboard"
            
            # Create dashboard
            dashboard = AnalyticsDashboard(
                name=name,
                type=DashboardType.EXECUTIVE.value,
                owner_id=owner_id,
                role=role.value,
                config=config.to_dict(),
                description=f"Executive dashboard for {role.value} role"
            )
            
            db.add(dashboard)
            db.flush()  # Get the ID
            
            # Add default widgets based on role
            default_widgets = self._get_default_widgets_for_role(role, dashboard.id)
            for widget_config in default_widgets:
                widget = Widget(**widget_config)
                db.add(widget)
            
            db.commit()
            db.refresh(dashboard)
            return dashboard
    
    def create_dashboard_from_template(self, template_id: str, owner_id: str, 
                                     name: str = None) -> AnalyticsDashboard:
        """Create a dashboard from a template."""
        with get_sync_db() as db:
            template = db.query(DashboardTemplate).filter(
                DashboardTemplate.id == template_id
            ).first()
            
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Create dashboard from template
            dashboard = AnalyticsDashboard(
                name=name or f"{template.name} - Copy",
                type=DashboardType.CUSTOM.value,
                owner_id=owner_id,
                config=template.template_config,
                description=f"Dashboard created from template: {template.name}"
            )
            
            db.add(dashboard)
            db.flush()
            
            # Create widgets from template
            template_widgets = template.template_config.get("widgets", [])
            for widget_config in template_widgets:
                widget = Widget(
                    dashboard_id=dashboard.id,
                    **widget_config
                )
                db.add(widget)
            
            db.commit()
            db.refresh(dashboard)
            return dashboard
    
    def update_dashboard_metrics(self, dashboard_id: str, metrics: List[Dict[str, Any]]) -> bool:
        """Update dashboard with new metrics data."""
        with get_sync_db() as db:
            dashboard = db.query(AnalyticsDashboard).filter(AnalyticsDashboard.id == dashboard_id).first()
            if not dashboard:
                return False
            
            # Update metrics
            for metric_data in metrics:
                metric = BusinessMetric(
                    dashboard_id=dashboard_id,
                    **metric_data
                )
                db.add(metric)
            
            # Update dashboard timestamp
            dashboard.updated_at = datetime.utcnow()
            
            db.commit()
            return True
    
    def get_dashboard_data(self, dashboard_id: str, time_range: TimeRange = None) -> Optional[DashboardData]:
        """Retrieve dashboard data with widgets and metrics."""
        with get_sync_db() as db:
            dashboard = db.query(AnalyticsDashboard).filter(AnalyticsDashboard.id == dashboard_id).first()
            if not dashboard:
                return None
            
            # Get widgets data
            widgets_data = {}
            for widget in dashboard.widgets:
                if widget.is_active:
                    widgets_data[widget.id] = {
                        "id": widget.id,
                        "name": widget.name,
                        "type": widget.type,
                        "position": {"x": widget.position_x, "y": widget.position_y},
                        "size": {"width": widget.width, "height": widget.height},
                        "config": widget.config,
                        "data_source": widget.data_source,
                        "last_updated": widget.updated_at
                    }
            
            # Get metrics within time range
            metrics_query = db.query(BusinessMetric).filter(
                BusinessMetric.dashboard_id == dashboard_id
            )
            
            if time_range:
                metrics_query = metrics_query.filter(
                    and_(
                        BusinessMetric.timestamp >= time_range.start,
                        BusinessMetric.timestamp <= time_range.end
                    )
                )
            
            metrics = metrics_query.order_by(BusinessMetric.timestamp.desc()).all()
            
            return DashboardData(dashboard, widgets_data, metrics)
    
    def share_dashboard(self, dashboard_id: str, permissions: SharePermissions, 
                       shared_by: str) -> ShareLink:
        """Share dashboard with specified users and permissions."""
        with get_sync_db() as db:
            dashboard = db.query(AnalyticsDashboard).filter(AnalyticsDashboard.id == dashboard_id).first()
            if not dashboard:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            # Calculate expiration date
            expires_at = None
            if permissions.expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=permissions.expires_in_days)
            
            # Create permissions for each user
            for user_id in permissions.users:
                permission = DashboardPermission(
                    dashboard_id=dashboard_id,
                    user_id=user_id,
                    permission_type=permissions.permission_type,
                    granted_by=shared_by,
                    expires_at=expires_at
                )
                db.add(permission)
            
            db.commit()
            
            # Generate share link
            share_url = f"{self.settings.base_url}/dashboard/{dashboard_id}/shared"
            return ShareLink(share_url, expires_at)
    
    def get_dashboards_for_user(self, user_id: str, role: str = None) -> List[AnalyticsDashboard]:
        """Get all dashboards accessible to a user."""
        with get_sync_db() as db:
            query = db.query(AnalyticsDashboard).filter(
                or_(
                    AnalyticsDashboard.owner_id == user_id,
                    AnalyticsDashboard.id.in_(
                        db.query(DashboardPermission.dashboard_id).filter(
                            and_(
                                DashboardPermission.user_id == user_id,
                                or_(
                                    DashboardPermission.expires_at.is_(None),
                                    DashboardPermission.expires_at > datetime.utcnow()
                                )
                            )
                        )
                    )
                )
            ).filter(AnalyticsDashboard.is_active == True)
            
            if role:
                query = query.filter(AnalyticsDashboard.role == role)
            
            return query.order_by(Dashboard.updated_at.desc()).all()
    
    def create_dashboard_template(self, name: str, category: str, role: str,
                                description: str, template_config: Dict[str, Any],
                                created_by: str) -> DashboardTemplate:
        """Create a new dashboard template."""
        with get_sync_db() as db:
            template = DashboardTemplate(
                name=name,
                category=category,
                role=role,
                description=description,
                template_config=template_config,
                created_by=created_by
            )
            
            db.add(template)
            db.commit()
            db.refresh(template)
            return template
    
    def get_templates_for_role(self, role: str) -> List[DashboardTemplate]:
        """Get available templates for a specific role."""
        with get_sync_db() as db:
            return db.query(DashboardTemplate).filter(
                or_(
                    DashboardTemplate.role == role,
                    DashboardTemplate.is_public == True
                )
            ).order_by(DashboardTemplate.created_at.desc()).all()
    
    def _get_default_widgets_for_role(self, role: ExecutiveRole, dashboard_id: str) -> List[Dict[str, Any]]:
        """Get default widgets configuration for executive roles."""
        base_widgets = [
            {
                "dashboard_id": dashboard_id,
                "name": "Key Performance Indicators",
                "type": WidgetType.KPI.value,
                "position_x": 0,
                "position_y": 0,
                "width": 12,
                "height": 2,
                "config": {"metrics": ["revenue", "growth_rate", "efficiency"]},
                "data_source": "business_metrics"
            },
            {
                "dashboard_id": dashboard_id,
                "name": "Trend Analysis",
                "type": WidgetType.CHART.value,
                "position_x": 0,
                "position_y": 2,
                "width": 8,
                "height": 4,
                "config": {"chart_type": "line", "time_range": "30d"},
                "data_source": "time_series_data"
            }
        ]
        
        # Role-specific widgets
        role_specific = {
            ExecutiveRole.CTO: [
                {
                    "dashboard_id": dashboard_id,
                    "name": "Technology ROI",
                    "type": WidgetType.METRIC.value,
                    "position_x": 8,
                    "position_y": 2,
                    "width": 4,
                    "height": 2,
                    "config": {"metric": "tech_roi"},
                    "data_source": "roi_calculator"
                },
                {
                    "dashboard_id": dashboard_id,
                    "name": "AI Initiative Status",
                    "type": WidgetType.TABLE.value,
                    "position_x": 8,
                    "position_y": 4,
                    "width": 4,
                    "height": 2,
                    "config": {"columns": ["project", "status", "roi"]},
                    "data_source": "ai_projects"
                }
            ],
            ExecutiveRole.CFO: [
                {
                    "dashboard_id": dashboard_id,
                    "name": "Financial Impact",
                    "type": WidgetType.CHART.value,
                    "position_x": 8,
                    "position_y": 2,
                    "width": 4,
                    "height": 4,
                    "config": {"chart_type": "bar", "metric": "cost_savings"},
                    "data_source": "financial_data"
                }
            ]
        }
        
        return base_widgets + role_specific.get(role, [])