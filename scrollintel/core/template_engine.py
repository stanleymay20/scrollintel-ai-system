"""
Template Engine for Advanced Analytics Dashboard System

Provides industry-specific dashboard templates with customization capabilities.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import uuid
from enum import Enum

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
    COMPLIANCE = "compliance"

@dataclass
class WidgetConfig:
    """Configuration for dashboard widgets"""
    id: str
    type: str  # chart, metric, table, etc.
    title: str
    position: Dict[str, int]  # x, y, width, height
    data_source: str
    visualization_config: Dict[str, Any]
    filters: List[Dict[str, Any]]
    refresh_interval: int = 300  # seconds

@dataclass
class DashboardTemplate:
    """Dashboard template definition"""
    id: str
    name: str
    description: str
    industry: IndustryType
    category: TemplateCategory
    widgets: List[WidgetConfig]
    layout_config: Dict[str, Any]
    default_filters: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str = "1.0.0"
    is_public: bool = False
    author: Optional[str] = None
    tags: List[str] = None

class TemplateEngine:
    """Engine for managing dashboard templates"""
    
    def __init__(self):
        self.templates: Dict[str, DashboardTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize industry-specific default templates"""
        
        # Technology Executive Dashboard
        tech_exec_widgets = [
            WidgetConfig(
                id="tech_roi_metric",
                type="metric_card",
                title="AI/ML ROI",
                position={"x": 0, "y": 0, "width": 3, "height": 2},
                data_source="roi_calculator",
                visualization_config={
                    "metric_type": "percentage",
                    "format": "currency",
                    "trend_indicator": True
                },
                filters=[{"field": "project_type", "value": "ai_ml"}]
            ),
            WidgetConfig(
                id="tech_deployment_status",
                type="status_chart",
                title="Deployment Pipeline Status",
                position={"x": 3, "y": 0, "width": 6, "height": 4},
                data_source="deployment_tracker",
                visualization_config={
                    "chart_type": "pipeline_flow",
                    "color_scheme": "status_based"
                },
                filters=[]
            ),
            WidgetConfig(
                id="tech_performance_trends",
                type="line_chart",
                title="System Performance Trends",
                position={"x": 0, "y": 2, "width": 9, "height": 3},
                data_source="performance_monitor",
                visualization_config={
                    "chart_type": "multi_line",
                    "metrics": ["response_time", "throughput", "error_rate"]
                },
                filters=[{"field": "time_range", "value": "last_30_days"}]
            )
        ]
        
        tech_template = DashboardTemplate(
            id="tech_executive_v1",
            name="Technology Executive Dashboard",
            description="Comprehensive executive view for technology organizations",
            industry=IndustryType.TECHNOLOGY,
            category=TemplateCategory.EXECUTIVE,
            widgets=tech_exec_widgets,
            layout_config={
                "grid_size": 12,
                "row_height": 60,
                "margin": [10, 10],
                "responsive_breakpoints": {"lg": 1200, "md": 996, "sm": 768}
            },
            default_filters=[
                {"field": "organization", "value": "current"},
                {"field": "time_range", "value": "last_quarter"}
            ],
            metadata={
                "target_roles": ["CTO", "VP Engineering", "Tech Director"],
                "update_frequency": "real_time",
                "data_sources": ["roi_calculator", "deployment_tracker", "performance_monitor"]
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["executive", "technology", "roi", "performance"]
        )
        
        # Finance Executive Dashboard
        finance_exec_widgets = [
            WidgetConfig(
                id="finance_cost_overview",
                type="metric_grid",
                title="Cost Overview",
                position={"x": 0, "y": 0, "width": 4, "height": 3},
                data_source="cost_tracker",
                visualization_config={
                    "metrics": ["total_cost", "cost_per_project", "budget_variance"],
                    "format": "currency"
                },
                filters=[]
            ),
            WidgetConfig(
                id="finance_roi_breakdown",
                type="waterfall_chart",
                title="ROI Breakdown by Initiative",
                position={"x": 4, "y": 0, "width": 8, "height": 4},
                data_source="roi_calculator",
                visualization_config={
                    "chart_type": "waterfall",
                    "breakdown_by": "initiative_type"
                },
                filters=[{"field": "status", "value": "active"}]
            )
        ]
        
        finance_template = DashboardTemplate(
            id="finance_executive_v1",
            name="Finance Executive Dashboard",
            description="Financial performance and ROI tracking for executives",
            industry=IndustryType.FINANCE,
            category=TemplateCategory.EXECUTIVE,
            widgets=finance_exec_widgets,
            layout_config={
                "grid_size": 12,
                "row_height": 60,
                "margin": [10, 10]
            },
            default_filters=[
                {"field": "fiscal_year", "value": "current"},
                {"field": "department", "value": "all"}
            ],
            metadata={
                "target_roles": ["CFO", "Finance Director", "Budget Manager"],
                "update_frequency": "daily",
                "compliance_requirements": ["SOX", "GAAP"]
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["executive", "finance", "roi", "budgeting"]
        )
        
        # Store templates
        self.templates[tech_template.id] = tech_template
        self.templates[finance_template.id] = finance_template
    
    def get_template(self, template_id: str) -> Optional[DashboardTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self, 
                      industry: Optional[IndustryType] = None,
                      category: Optional[TemplateCategory] = None,
                      tags: Optional[List[str]] = None) -> List[DashboardTemplate]:
        """List templates with optional filtering"""
        templates = list(self.templates.values())
        
        if industry:
            templates = [t for t in templates if t.industry == industry]
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if tags:
            templates = [t for t in templates if any(tag in (t.tags or []) for tag in tags)]
        
        return templates
    
    def create_template(self, template: DashboardTemplate) -> str:
        """Create new template"""
        if not template.id:
            template.id = str(uuid.uuid4())
        
        template.created_at = datetime.now()
        template.updated_at = datetime.now()
        
        self.templates[template.id] = template
        return template.id
    
    def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing template"""
        if template_id not in self.templates:
            return False
        
        template = self.templates[template_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        template.updated_at = datetime.now()
        return True
    
    def clone_template(self, template_id: str, new_name: str) -> Optional[str]:
        """Clone existing template with new name"""
        original = self.get_template(template_id)
        if not original:
            return None
        
        # Create clone
        cloned = DashboardTemplate(
            id=str(uuid.uuid4()),
            name=new_name,
            description=f"Cloned from {original.name}",
            industry=original.industry,
            category=original.category,
            widgets=original.widgets.copy(),
            layout_config=original.layout_config.copy(),
            default_filters=original.default_filters.copy(),
            metadata=original.metadata.copy(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0.0",
            tags=(original.tags or []).copy()
        )
        
        self.templates[cloned.id] = cloned
        return cloned.id
    
    def delete_template(self, template_id: str) -> bool:
        """Delete template"""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False
    
    def export_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Export template as JSON-serializable dict"""
        template = self.get_template(template_id)
        if not template:
            return None
        
        return {
            **asdict(template),
            'created_at': template.created_at.isoformat(),
            'updated_at': template.updated_at.isoformat(),
            'industry': template.industry.value,
            'category': template.category.value
        }
    
    def import_template(self, template_data: Dict[str, Any]) -> Optional[str]:
        """Import template from dict"""
        try:
            # Parse dates
            template_data['created_at'] = datetime.fromisoformat(template_data['created_at'])
            template_data['updated_at'] = datetime.fromisoformat(template_data['updated_at'])
            
            # Convert enums
            template_data['industry'] = IndustryType(template_data['industry'])
            template_data['category'] = TemplateCategory(template_data['category'])
            
            # Convert widgets
            widgets = []
            for widget_data in template_data['widgets']:
                widget = WidgetConfig(**widget_data)
                widgets.append(widget)
            template_data['widgets'] = widgets
            
            template = DashboardTemplate(**template_data)
            return self.create_template(template)
            
        except Exception as e:
            print(f"Error importing template: {e}")
            return None
    
    def get_industry_templates(self, industry: IndustryType) -> List[DashboardTemplate]:
        """Get all templates for specific industry"""
        return [t for t in self.templates.values() if t.industry == industry]
    
    def search_templates(self, query: str) -> List[DashboardTemplate]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query_lower in template.name.lower() or 
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in (template.tags or []))):
                results.append(template)
        
        return results