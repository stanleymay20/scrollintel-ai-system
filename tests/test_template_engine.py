"""
Tests for the template engine functionality.
"""
import pytest
from datetime import datetime
from scrollintel.core.template_engine import (
    TemplateEngine, 
    DashboardTemplate, 
    TemplateWidget,
    IndustryType,
    TemplateCategory
)


class TestTemplateEngine:
    """Test cases for TemplateEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TemplateEngine()
    
    def test_engine_initialization(self):
        """Test that engine initializes with default templates."""
        assert len(self.engine.templates) > 0
        
        # Check for default templates
        template_ids = list(self.engine.templates.keys())
        assert "tech_executive_v1" in template_ids
        assert "finance_executive_v1" in template_ids
        assert "operational_v1" in template_ids
    
    def test_get_templates_by_industry(self):
        """Test filtering templates by industry."""
        tech_templates = self.engine.get_templates_by_industry(IndustryType.TECHNOLOGY)
        assert len(tech_templates) > 0
        
        # Should include technology-specific and generic templates
        tech_template = next((t for t in tech_templates if t.id == "tech_executive_v1"), None)
        assert tech_template is not None
        assert tech_template.industry == IndustryType.TECHNOLOGY
        
        # Should include generic templates
        ops_template = next((t for t in tech_templates if t.id == "operational_v1"), None)
        assert ops_template is not None
        assert ops_template.industry == IndustryType.GENERIC
    
    def test_get_templates_by_category(self):
        """Test filtering templates by category."""
        exec_templates = self.engine.get_templates_by_category(TemplateCategory.EXECUTIVE)
        assert len(exec_templates) >= 2  # tech and finance executive templates
        
        for template in exec_templates:
            assert template.category == TemplateCategory.EXECUTIVE
    
    def test_get_template(self):
        """Test retrieving specific template."""
        template = self.engine.get_template("tech_executive_v1")
        assert template is not None
        assert template.name == "Technology Executive Dashboard"
        assert template.industry == IndustryType.TECHNOLOGY
        assert len(template.widgets) > 0
        
        # Test non-existent template
        assert self.engine.get_template("non_existent") is None
    
    def test_create_custom_template(self):
        """Test creating custom template."""
        widgets = [
            TemplateWidget(
                id="custom_widget",
                type="metrics_grid",
                title="Custom Metrics",
                position={"x": 0, "y": 0, "width": 6, "height": 4},
                config={"metrics": ["custom_metric"]},
                data_source="custom_source"
            )
        ]
        
        template = self.engine.create_custom_template(
            name="Custom Dashboard",
            description="A custom dashboard template",
            industry=IndustryType.TECHNOLOGY,
            category=TemplateCategory.CUSTOM,
            widgets=widgets,
            layout_config={"grid_size": 12, "row_height": 60},
            permissions=["user"],
            created_by="test_user",
            tags=["custom", "test"]
        )
        
        assert template.name == "Custom Dashboard"
        assert template.industry == IndustryType.TECHNOLOGY
        assert template.category == TemplateCategory.CUSTOM
        assert len(template.widgets) == 1
        assert template.created_by == "test_user"
        assert not template.is_public  # Custom templates are private by default
        
        # Template should be stored in engine
        stored_template = self.engine.get_template(template.id)
        assert stored_template is not None
        assert stored_template.name == "Custom Dashboard"
    
    def test_clone_template(self):
        """Test cloning existing template."""
        original_id = "tech_executive_v1"
        cloned = self.engine.clone_template(original_id, "Cloned Tech Dashboard", "test_user")
        
        assert cloned is not None
        assert cloned.name == "Cloned Tech Dashboard"
        assert cloned.created_by == "test_user"
        assert cloned.id != original_id
        assert not cloned.is_public
        
        # Should have same widgets as original
        original = self.engine.get_template(original_id)
        assert len(cloned.widgets) == len(original.widgets)
        
        # Test cloning non-existent template
        assert self.engine.clone_template("non_existent", "Clone", "user") is None
    
    def test_search_templates(self):
        """Test template search functionality."""
        # Search by name
        results = self.engine.search_templates("Technology")
        assert len(results) > 0
        assert any("Technology" in t.name for t in results)
        
        # Search by description
        results = self.engine.search_templates("executive")
        assert len(results) > 0
        
        # Search by tag
        results = self.engine.search_templates("roi")
        assert len(results) > 0
        
        # Search with industry filter
        results = self.engine.search_templates("Dashboard", IndustryType.FINANCE)
        finance_results = [t for t in results if t.industry == IndustryType.FINANCE]
        assert len(finance_results) > 0
    
    def test_get_template_preview(self):
        """Test template preview generation."""
        preview = self.engine.get_template_preview("tech_executive_v1")
        
        assert preview is not None
        assert preview["id"] == "tech_executive_v1"
        assert preview["name"] == "Technology Executive Dashboard"
        assert preview["industry"] == "technology"
        assert preview["category"] == "executive"
        assert "widget_count" in preview
        assert "layout_preview" in preview
        assert "widgets" in preview["layout_preview"]
        
        # Test non-existent template
        assert self.engine.get_template_preview("non_existent") is None
    
    def test_export_template(self):
        """Test template export functionality."""
        exported = self.engine.export_template("tech_executive_v1")
        
        assert exported is not None
        assert exported["id"] == "tech_executive_v1"
        assert exported["name"] == "Technology Executive Dashboard"
        assert "widgets" in exported
        assert "layout_config" in exported
        assert "permissions" in exported
        
        # Test non-existent template
        assert self.engine.export_template("non_existent") is None
    
    def test_import_template(self):
        """Test template import functionality."""
        # First export a template
        original_data = self.engine.export_template("tech_executive_v1")
        assert original_data is not None
        
        # Modify the data for import
        original_data["name"] = "Imported Tech Dashboard"
        
        # Import the template
        imported = self.engine.import_template(original_data, "import_user")
        
        assert imported.name == "Imported Tech Dashboard"
        assert imported.created_by == "import_user"
        assert imported.id != "tech_executive_v1"  # Should have new ID
        assert not imported.is_public
        
        # Should be stored in engine
        stored = self.engine.get_template(imported.id)
        assert stored is not None
        assert stored.name == "Imported Tech Dashboard"


class TestTemplateWidget:
    """Test cases for TemplateWidget."""
    
    def test_widget_creation(self):
        """Test widget creation with all properties."""
        widget = TemplateWidget(
            id="test_widget",
            type="line_chart",
            title="Test Chart",
            position={"x": 2, "y": 1, "width": 8, "height": 6},
            config={"time_range": "24h", "metrics": ["cpu", "memory"]},
            data_source="metrics_api",
            refresh_interval=60
        )
        
        assert widget.id == "test_widget"
        assert widget.type == "line_chart"
        assert widget.title == "Test Chart"
        assert widget.position["width"] == 8
        assert widget.config["time_range"] == "24h"
        assert widget.data_source == "metrics_api"
        assert widget.refresh_interval == 60
    
    def test_widget_default_refresh_interval(self):
        """Test widget default refresh interval."""
        widget = TemplateWidget(
            id="test_widget",
            type="metrics_grid",
            title="Test Metrics",
            position={"x": 0, "y": 0, "width": 4, "height": 3},
            config={},
            data_source="test_source"
        )
        
        assert widget.refresh_interval == 300  # Default 5 minutes


class TestDashboardTemplate:
    """Test cases for DashboardTemplate."""
    
    def test_template_creation(self):
        """Test template creation with all properties."""
        widgets = [
            TemplateWidget(
                id="widget1",
                type="metrics_grid",
                title="Metrics",
                position={"x": 0, "y": 0, "width": 6, "height": 4},
                config={},
                data_source="metrics"
            )
        ]
        
        template = DashboardTemplate(
            id="test_template",
            name="Test Dashboard",
            description="A test dashboard",
            industry=IndustryType.TECHNOLOGY,
            category=TemplateCategory.OPERATIONAL,
            widgets=widgets,
            layout_config={"grid_size": 12, "row_height": 60},
            permissions=["admin", "user"],
            version="2.0.0",
            created_by="test_user",
            tags=["test", "dashboard"]
        )
        
        assert template.id == "test_template"
        assert template.name == "Test Dashboard"
        assert template.industry == IndustryType.TECHNOLOGY
        assert template.category == TemplateCategory.OPERATIONAL
        assert len(template.widgets) == 1
        assert template.version == "2.0.0"
        assert template.created_by == "test_user"
        assert "test" in template.tags
    
    def test_template_defaults(self):
        """Test template default values."""
        template = DashboardTemplate(
            id="test_template",
            name="Test Dashboard",
            description="A test dashboard",
            industry=IndustryType.GENERIC,
            category=TemplateCategory.CUSTOM,
            widgets=[],
            layout_config={},
            permissions=[]
        )
        
        assert template.version == "1.0.0"
        assert template.created_by == "system"
        assert template.created_at is not None
        assert template.is_public is True
        assert template.tags == []
    
    def test_template_created_at_auto_set(self):
        """Test that created_at is automatically set."""
        before_creation = datetime.utcnow()
        
        template = DashboardTemplate(
            id="test_template",
            name="Test Dashboard",
            description="A test dashboard",
            industry=IndustryType.GENERIC,
            category=TemplateCategory.CUSTOM,
            widgets=[],
            layout_config={},
            permissions=[]
        )
        
        after_creation = datetime.utcnow()
        
        assert before_creation <= template.created_at <= after_creation


class TestIndustryAndCategoryEnums:
    """Test cases for industry and category enums."""
    
    def test_industry_types(self):
        """Test industry type enum values."""
        assert IndustryType.TECHNOLOGY.value == "technology"
        assert IndustryType.FINANCE.value == "finance"
        assert IndustryType.HEALTHCARE.value == "healthcare"
        assert IndustryType.GENERIC.value == "generic"
    
    def test_template_categories(self):
        """Test template category enum values."""
        assert TemplateCategory.EXECUTIVE.value == "executive"
        assert TemplateCategory.OPERATIONAL.value == "operational"
        assert TemplateCategory.FINANCIAL.value == "financial"
        assert TemplateCategory.TECHNICAL.value == "technical"
        assert TemplateCategory.CUSTOM.value == "custom"


if __name__ == "__main__":
    pytest.main([__file__])