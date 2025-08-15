"""
Template Engine for customizable dashboard templates with industry-specific configurations.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from scrollintel.models.database import get_db
from scrollintel.models.dashboard_models import Dashboard, Widget


class IndustryType(Enum):
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    EDUCATION = "education"
    GOVERNMENT = "government"
    GENERIC = "generic"


class TemplateCategory(Enum):
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    CUSTOM = "custom"


@dataclass
class TemplateWidget:
    id: str
    type: str
    title: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any]
    data_source: str
    refresh_interval: int = 300  # seconds


@dataclass
class DashboardTemplate:
    id: str
    name: str
    description: str
    industry: IndustryType
    category: TemplateCategory
    widgets: List[TemplateWidget]
    layout_config: Dict[str, Any]
    permissions: List[str]
    version: str = "1.0.0"
    created_by: str = "system"
    created_at: datetime = None
    is_public: bool = True
    tags: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.tags is None:
            self.tags = []


class TemplateEngine:
    """Engine for managing customizable dashboard templates."""
    
    def __init__(self):
        self.templates: Dict[str, DashboardTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default industry-specific templates."""
        # Technology Executive Dashboard
        tech_exec_widgets = [
            TemplateWidget(
                id="tech_kpis",
                type="metrics_grid",
                title="Technology KPIs",
                position={"x": 0, "y": 0, "width": 6, "height": 4},
                config={
                    "metrics": ["system_uptime", "deployment_frequency", "lead_time", "mttr"],
                    "display_type": "cards"
                },
                data_source="tech_metrics"
            ),
            TemplateWidget(
                id="roi_chart",
                type="line_chart",
                title="Technology ROI Trends",
                position={"x": 6, "y": 0, "width": 6, "height": 4},
                config={
                    "metrics": ["roi_percentage", "cost_savings"],
                    "time_range": "6M"
                },
                data_source="roi_data"
            ),
            TemplateWidget(
                id="project_status",
                type="kanban_board",
                title="Project Portfolio Status",
                position={"x": 0, "y": 4, "width": 12, "height": 6},
                config={
                    "columns": ["planning", "development", "testing", "deployment"],
                    "show_metrics": True
                },
                data_source="project_data"
            )
        ]
        
        tech_template = DashboardTemplate(
            id="tech_executive_v1",
            name="Technology Executive Dashboard",
            description="Comprehensive dashboard for technology executives with ROI tracking and project oversight",
            industry=IndustryType.TECHNOLOGY,
            category=TemplateCategory.EXECUTIVE,
            widgets=tech_exec_widgets,
            layout_config={"grid_size": 12, "row_height": 60},
            permissions=["executive", "cto", "admin"],
            tags=["executive", "technology", "roi", "projects"]
        )
        
        # Finance Executive Dashboard
        finance_exec_widgets = [
            TemplateWidget(
                id="financial_kpis",
                type="metrics_grid",
                title="Financial KPIs",
                position={"x": 0, "y": 0, "width": 8, "height": 3},
                config={
                    "metrics": ["revenue", "profit_margin", "cash_flow", "burn_rate"],
                    "display_type": "cards_with_trends"
                },
                data_source="financial_metrics"
            ),
            TemplateWidget(
                id="budget_variance",
                type="bar_chart",
                title="Budget vs Actual",
                position={"x": 8, "y": 0, "width": 4, "height": 3},
                config={
                    "comparison_type": "budget_actual",
                    "departments": ["technology", "marketing", "operations"]
                },
                data_source="budget_data"
            ),
            TemplateWidget(
                id="roi_breakdown",
                type="pie_chart",
                title="ROI by Investment Category",
                position={"x": 0, "y": 3, "width": 6, "height": 4},
                config={
                    "categories": ["technology", "marketing", "infrastructure", "talent"],
                    "show_percentages": True
                },
                data_source="roi_breakdown"
            ),
            TemplateWidget(
                id="forecast_trends",
                type="line_chart",
                title="Revenue Forecast",
                position={"x": 6, "y": 3, "width": 6, "height": 4},
                config={
                    "forecast_horizon": "12M",
                    "confidence_intervals": True
                },
                data_source="forecast_data"
            )
        ]
        
        finance_template = DashboardTemplate(
            id="finance_executive_v1",
            name="Finance Executive Dashboard",
            description="Financial oversight dashboard with budget tracking and ROI analysis",
            industry=IndustryType.FINANCE,
            category=TemplateCategory.EXECUTIVE,
            widgets=finance_exec_widgets,
            layout_config={"grid_size": 12, "row_height": 80},
            permissions=["executive", "cfo", "finance_manager", "admin"],
            tags=["executive", "finance", "budget", "roi", "forecast"]
        )
        
        # Operational Dashboard
        ops_widgets = [
            TemplateWidget(
                id="system_health",
                type="status_grid",
                title="System Health Overview",
                position={"x": 0, "y": 0, "width": 4, "height": 3},
                config={
                    "systems": ["api", "database", "cache", "queue"],
                    "health_indicators": ["status", "response_time", "error_rate"]
                },
                data_source="system_metrics"
            ),
            TemplateWidget(
                id="performance_metrics",
                type="gauge_chart",
                title="Performance Metrics",
                position={"x": 4, "y": 0, "width": 4, "height": 3},
                config={
                    "metrics": ["cpu_usage", "memory_usage", "disk_usage"],
                    "thresholds": {"warning": 70, "critical": 90}
                },
                data_source="performance_data"
            ),
            TemplateWidget(
                id="alert_summary",
                type="alert_list",
                title="Active Alerts",
                position={"x": 8, "y": 0, "width": 4, "height": 3},
                config={
                    "severity_levels": ["critical", "warning", "info"],
                    "max_items": 10
                },
                data_source="alert_data"
            ),
            TemplateWidget(
                id="throughput_trends",
                type="area_chart",
                title="System Throughput",
                position={"x": 0, "y": 3, "width": 12, "height": 4},
                config={
                    "metrics": ["requests_per_second", "transactions_per_minute"],
                    "time_range": "24H"
                },
                data_source="throughput_data"
            )
        ]
        
        ops_template = DashboardTemplate(
            id="operational_v1",
            name="Operational Dashboard",
            description="Real-time operational monitoring with system health and performance metrics",
            industry=IndustryType.GENERIC,
            category=TemplateCategory.OPERATIONAL,
            widgets=ops_widgets,
            layout_config={"grid_size": 12, "row_height": 70},
            permissions=["operations", "devops", "admin"],
            tags=["operational", "monitoring", "performance", "alerts"]
        )
        
        # Store templates
        self.templates[tech_template.id] = tech_template
        self.templates[finance_template.id] = finance_template
        self.templates[ops_template.id] = ops_template
    
    def get_templates_by_industry(self, industry: IndustryType) -> List[DashboardTemplate]:
        """Get all templates for a specific industry."""
        return [
            template for template in self.templates.values()
            if template.industry == industry or template.industry == IndustryType.GENERIC
        ]
    
    def get_templates_by_category(self, category: TemplateCategory) -> List[DashboardTemplate]:
        """Get all templates for a specific category."""
        return [
            template for template in self.templates.values()
            if template.category == category
        ]
    
    def get_template(self, template_id: str) -> Optional[DashboardTemplate]:
        """Get a specific template by ID."""
        return self.templates.get(template_id)
    
    def create_custom_template(
        self,
        name: str,
        description: str,
        industry: IndustryType,
        category: TemplateCategory,
        widgets: List[TemplateWidget],
        layout_config: Dict[str, Any],
        permissions: List[str],
        created_by: str,
        tags: List[str] = None
    ) -> DashboardTemplate:
        """Create a new custom template."""
        template_id = f"custom_{uuid.uuid4().hex[:8]}"
        
        template = DashboardTemplate(
            id=template_id,
            name=name,
            description=description,
            industry=industry,
            category=category,
            widgets=widgets,
            layout_config=layout_config,
            permissions=permissions,
            created_by=created_by,
            tags=tags or [],
            is_public=False
        )
        
        self.templates[template_id] = template
        return template
    
    def clone_template(self, template_id: str, new_name: str, created_by: str) -> Optional[DashboardTemplate]:
        """Clone an existing template with a new name."""
        original = self.get_template(template_id)
        if not original:
            return None
        
        new_id = f"clone_{uuid.uuid4().hex[:8]}"
        cloned_template = DashboardTemplate(
            id=new_id,
            name=new_name,
            description=f"Cloned from {original.name}",
            industry=original.industry,
            category=original.category,
            widgets=original.widgets.copy(),
            layout_config=original.layout_config.copy(),
            permissions=original.permissions.copy(),
            created_by=created_by,
            tags=original.tags.copy(),
            is_public=False
        )
        
        self.templates[new_id] = cloned_template
        return cloned_template
    
    def search_templates(self, query: str, industry: Optional[IndustryType] = None) -> List[DashboardTemplate]:
        """Search templates by name, description, or tags."""
        results = []
        query_lower = query.lower()
        
        for template in self.templates.values():
            if industry and template.industry != industry and template.industry != IndustryType.GENERIC:
                continue
            
            if (query_lower in template.name.lower() or 
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results
    
    def get_template_preview(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a preview of template layout and widgets."""
        template = self.get_template(template_id)
        if not template:
            return None
        
        return {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "industry": template.industry.value,
            "category": template.category.value,
            "widget_count": len(template.widgets),
            "layout_preview": {
                "grid_size": template.layout_config.get("grid_size", 12),
                "widgets": [
                    {
                        "type": widget.type,
                        "title": widget.title,
                        "position": widget.position
                    }
                    for widget in template.widgets
                ]
            },
            "tags": template.tags
        }
    
    def export_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Export template as JSON for sharing."""
        template = self.get_template(template_id)
        if not template:
            return None
        
        return asdict(template)
    
    def import_template(self, template_data: Dict[str, Any], imported_by: str) -> DashboardTemplate:
        """Import template from JSON data."""
        # Generate new ID to avoid conflicts
        new_id = f"imported_{uuid.uuid4().hex[:8]}"
        template_data["id"] = new_id
        template_data["created_by"] = imported_by
        template_data["created_at"] = datetime.utcnow()
        template_data["is_public"] = False
        
        # Convert widgets
        widgets = []
        for widget_data in template_data["widgets"]:
            widget = TemplateWidget(**widget_data)
            widgets.append(widget)
        template_data["widgets"] = widgets
        
        # Convert enums
        template_data["industry"] = IndustryType(template_data["industry"])
        template_data["category"] = TemplateCategory(template_data["category"])
        
        template = DashboardTemplate(**template_data)
        self.templates[new_id] = template
        return template


# Global template engine instance
template_engine = TemplateEngine()