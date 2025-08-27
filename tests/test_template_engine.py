"""
Unit tests for Template Engine
"""

import pytest
from datetime import datetime
from scrollintel.core.template_engine import (
    TemplateEngine, 
    DashboardTemplate, 
    WidgetConfig,
    IndustryType,
    TemplateCategory
)


class TestTemplateEngine:
    """Test cases for TemplateEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = TemplateEngine()
        
        # Sample widget config
        self.sample_widget = WidgetConfig(
            id="test_widget",
            type="metric_card",
            title="Test Widget",
            position={"x": 0, "y": 0, "width": 4, "height": 2},
            data_source="test_source",
            visualization_config={"format": "currency"},
            filters=[{"field": "status", "value": "active"}]
        )
        
        # Sample template
        self.sample_template = DashboardTemplate(
            id="test_template",
            name="Test Template",
            description="Test template description",
            industry=IndustryType.TECHNOLOGY,
            category=TemplateCategory.EXECUTIVE,
            widgets=[self.sample_widget],
            layout_config={"grid_size": 12, "row_height": 60, "margin": [10, 10]},
            default_filters=[],
            metadata={"test": True},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["test", "sample"]
        )
    
    def test_initialization(self):
        """Test engine initialization with default templates"""
        assert len(self.engine.templates) >= 2  # Should have default templates
        
        # Check that default templates exist
        tech_templates = [t for t in self.engine.templates.values() 
                         if t.industry == IndustryType.TECHNOLOGY]
        assert len(tech_templates) >= 1
        
        finance_templates = [t for t in self.engine.templates.values() 
                           if t.industry == IndustryType.FINANCE]
        assert len(finance_templates) >= 1
    
    def test_get_template(self):
        """Test getting template by ID"""
        # Create and store template
        template_id = self.engine.create_template(self.sample_template)
        
        # Retrieve template
        retrieved = self.engine.get_template(template_id)
        assert retrieved is not None
        assert retrieved.name == "Test Template"
        assert retrieved.industry == IndustryType.TECHNOLOGY
        
        # Test non-existent template
        assert self.engine.get_template("non_existent") is None
    
    def test_create_template(self):
        """Test creating new template"""
        initial_count = len(self.engine.templates)
        
        # Create template
        template_id = self.engine.create_template(self.sample_template)
        
        assert template_id is not None
        assert len(self.engine.templates) == initial_count + 1
        
        # Verify template was stored correctly
        stored_template = self.engine.templates[template_id]
        assert stored_template.name == "Test Template"
        assert stored_template.created_at is not None
        assert stored_template.updated_at is not None
    
    def test_create_template_with_auto_id(self):
        """Test creating template with auto-generated ID"""
        template_without_id = DashboardTemplate(
            id="",  # Empty ID should be auto-generated
            name="Auto ID Template",
            description="Template with auto-generated ID",
            industry=IndustryType.GENERIC,
            category=TemplateCategory.OPERATIONAL,
            widgets=[],
            layout_config={"grid_size": 12},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        template_id = self.engine.create_template(template_without_id)
        
        assert template_id != ""
        assert template_id in self.engine.templates
        assert self.engine.templates[template_id].id == template_id
    
    def test_update_template(self):
        """Test updating existing template"""
        # Create template
        template_id = self.engine.create_template(self.sample_template)
        original_updated_at = self.engine.templates[template_id].updated_at
        
        # Update template
        updates = {
            "name": "Updated Template Name",
            "description": "Updated description"
        }
        
        success = self.engine.update_template(template_id, updates)
        assert success is True
        
        # Verify updates
        updated_template = self.engine.templates[template_id]
        assert updated_template.name == "Updated Template Name"
        assert updated_template.description == "Updated description"
        assert updated_template.updated_at > original_updated_at
        
        # Test updating non-existent template
        assert self.engine.update_template("non_existent", updates) is False
    
    def test_clone_template(self):
        """Test cloning existing template"""
        # Create original template
        original_id = self.engine.create_template(self.sample_template)
        initial_count = len(self.engine.templates)
        
        # Clone template
        cloned_id = self.engine.clone_template(original_id, "Cloned Template")
        
        assert cloned_id is not None
        assert cloned_id != original_id
        assert len(self.engine.templates) == initial_count + 1
        
        # Verify clone properties
        original = self.engine.templates[original_id]
        cloned = self.engine.templates[cloned_id]
        
        assert cloned.name == "Cloned Template"
        assert cloned.description == f"Cloned from {original.name}"
        assert cloned.industry == original.industry
        assert cloned.category == original.category
        assert len(cloned.widgets) == len(original.widgets)
        assert cloned.version == "1.0.0"  # Reset version
        
        # Test cloning non-existent template
        assert self.engine.clone_template("non_existent", "Clone") is None
    
    def test_delete_template(self):
        """Test deleting template"""
        # Create template
        template_id = self.engine.create_template(self.sample_template)
        initial_count = len(self.engine.templates)
        
        # Delete template
        success = self.engine.delete_template(template_id)
        assert success is True
        assert len(self.engine.templates) == initial_count - 1
        assert template_id not in self.engine.templates
        
        # Test deleting non-existent template
        assert self.engine.delete_template("non_existent") is False
    
    def test_list_templates(self):
        """Test listing templates with filters"""
        # Create templates with different properties
        tech_template = self.sample_template
        tech_id = self.engine.create_template(tech_template)
        
        finance_template = DashboardTemplate(
            id="finance_template",
            name="Finance Template",
            description="Finance dashboard",
            industry=IndustryType.FINANCE,
            category=TemplateCategory.FINANCIAL,
            widgets=[],
            layout_config={"grid_size": 12},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["finance", "executive"]
        )
        finance_id = self.engine.create_template(finance_template)
        
        # Test listing all templates
        all_templates = self.engine.list_templates()
        assert len(all_templates) >= 2
        
        # Test filtering by industry
        tech_templates = self.engine.list_templates(industry=IndustryType.TECHNOLOGY)
        assert len(tech_templates) >= 1
        assert all(t.industry == IndustryType.TECHNOLOGY for t in tech_templates)
        
        finance_templates = self.engine.list_templates(industry=IndustryType.FINANCE)
        assert len(finance_templates) >= 1
        assert all(t.industry == IndustryType.FINANCE for t in finance_templates)
        
        # Test filtering by category
        executive_templates = self.engine.list_templates(category=TemplateCategory.EXECUTIVE)
        assert len(executive_templates) >= 1
        
        # Test filtering by tags
        finance_tagged = self.engine.list_templates(tags=["finance"])
        assert len(finance_tagged) >= 1
        assert all("finance" in (t.tags or []) for t in finance_tagged)
    
    def test_export_template(self):
        """Test exporting template to dict"""
        # Create template
        template_id = self.engine.create_template(self.sample_template)
        
        # Export template
        exported = self.engine.export_template(template_id)
        
        assert exported is not None
        assert exported["name"] == "Test Template"
        assert exported["industry"] == IndustryType.TECHNOLOGY.value
        assert exported["category"] == TemplateCategory.EXECUTIVE.value
        assert "created_at" in exported
        assert "updated_at" in exported
        assert len(exported["widgets"]) == 1
        
        # Test exporting non-existent template
        assert self.engine.export_template("non_existent") is None
    
    def test_import_template(self):
        """Test importing template from dict"""
        # Create export data
        export_data = {
            "id": "imported_template",
            "name": "Imported Template",
            "description": "Imported from export",
            "industry": IndustryType.HEALTHCARE.value,
            "category": TemplateCategory.OPERATIONAL.value,
            "widgets": [{
                "id": "imported_widget",
                "type": "chart",
                "title": "Imported Widget",
                "position": {"x": 0, "y": 0, "width": 6, "height": 4},
                "data_source": "imported_source",
                "visualization_config": {},
                "filters": []
            }],
            "layout_config": {"grid_size": 12},
            "default_filters": [],
            "metadata": {"imported": True},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "2.0.0",
            "tags": ["imported", "test"]
        }
        
        # Import template
        template_id = self.engine.import_template(export_data)
        
        assert template_id is not None
        assert template_id in self.engine.templates
        
        imported_template = self.engine.templates[template_id]
        assert imported_template.name == "Imported Template"
        assert imported_template.industry == IndustryType.HEALTHCARE
        assert imported_template.category == TemplateCategory.OPERATIONAL
        assert len(imported_template.widgets) == 1
        assert imported_template.version == "2.0.0"
    
    def test_import_template_invalid_data(self):
        """Test importing template with invalid data"""
        invalid_data = {
            "name": "Invalid Template",
            # Missing required fields
        }
        
        template_id = self.engine.import_template(invalid_data)
        assert template_id is None
    
    def test_get_industry_templates(self):
        """Test getting templates by industry"""
        # Create templates for different industries
        tech_template = self.sample_template
        self.engine.create_template(tech_template)
        
        healthcare_template = DashboardTemplate(
            id="healthcare_template",
            name="Healthcare Template",
            description="Healthcare dashboard",
            industry=IndustryType.HEALTHCARE,
            category=TemplateCategory.OPERATIONAL,
            widgets=[],
            layout_config={"grid_size": 12},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.engine.create_template(healthcare_template)
        
        # Get technology templates
        tech_templates = self.engine.get_industry_templates(IndustryType.TECHNOLOGY)
        assert len(tech_templates) >= 1
        assert all(t.industry == IndustryType.TECHNOLOGY for t in tech_templates)
        
        # Get healthcare templates
        healthcare_templates = self.engine.get_industry_templates(IndustryType.HEALTHCARE)
        assert len(healthcare_templates) >= 1
        assert all(t.industry == IndustryType.HEALTHCARE for t in healthcare_templates)
    
    def test_search_templates(self):
        """Test searching templates by query"""
        # Create searchable templates
        template1 = DashboardTemplate(
            id="search_template_1",
            name="Executive Dashboard",
            description="Dashboard for executives",
            industry=IndustryType.TECHNOLOGY,
            category=TemplateCategory.EXECUTIVE,
            widgets=[],
            layout_config={"grid_size": 12},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["executive", "management"]
        )
        
        template2 = DashboardTemplate(
            id="search_template_2",
            name="Operations Monitor",
            description="Operational metrics dashboard",
            industry=IndustryType.MANUFACTURING,
            category=TemplateCategory.OPERATIONAL,
            widgets=[],
            layout_config={"grid_size": 12},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["operations", "monitoring"]
        )
        
        self.engine.create_template(template1)
        self.engine.create_template(template2)
        
        # Search by name
        results = self.engine.search_templates("Executive")
        assert len(results) >= 1
        assert any("Executive" in t.name for t in results)
        
        # Search by description
        results = self.engine.search_templates("operational")
        assert len(results) >= 1
        assert any("operational" in t.description.lower() for t in results)
        
        # Search by tags
        results = self.engine.search_templates("management")
        assert len(results) >= 1
        assert any("management" in (t.tags or []) for t in results)
        
        # Search with no matches
        results = self.engine.search_templates("nonexistent")
        assert len(results) == 0


class TestWidgetConfig:
    """Test cases for WidgetConfig"""
    
    def test_widget_config_creation(self):
        """Test creating widget configuration"""
        widget = WidgetConfig(
            id="test_widget",
            type="line_chart",
            title="Test Chart",
            position={"x": 2, "y": 1, "width": 6, "height": 4},
            data_source="metrics_api",
            visualization_config={
                "chart_type": "line",
                "color_scheme": "blue"
            },
            filters=[
                {"field": "date_range", "value": "last_30_days"},
                {"field": "status", "value": "active"}
            ],
            refresh_interval=600
        )
        
        assert widget.id == "test_widget"
        assert widget.type == "line_chart"
        assert widget.title == "Test Chart"
        assert widget.position["width"] == 6
        assert widget.data_source == "metrics_api"
        assert widget.visualization_config["chart_type"] == "line"
        assert len(widget.filters) == 2
        assert widget.refresh_interval == 600


class TestDashboardTemplate:
    """Test cases for DashboardTemplate"""
    
    def test_template_creation(self):
        """Test creating dashboard template"""
        widget = WidgetConfig(
            id="widget_1",
            type="metric_card",
            title="Revenue",
            position={"x": 0, "y": 0, "width": 3, "height": 2},
            data_source="finance_api",
            visualization_config={"format": "currency"},
            filters=[]
        )
        
        template = DashboardTemplate(
            id="revenue_dashboard",
            name="Revenue Dashboard",
            description="Track revenue metrics",
            industry=IndustryType.FINANCE,
            category=TemplateCategory.FINANCIAL,
            widgets=[widget],
            layout_config={
                "grid_size": 12,
                "row_height": 60,
                "margin": [10, 10],
                "responsive_breakpoints": {"lg": 1200, "md": 996}
            },
            default_filters=[{"field": "fiscal_year", "value": "2024"}],
            metadata={"version": "1.0", "author": "finance_team"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0.0",
            tags=["finance", "revenue", "executive"]
        )
        
        assert template.id == "revenue_dashboard"
        assert template.name == "Revenue Dashboard"
        assert template.industry == IndustryType.FINANCE
        assert template.category == TemplateCategory.FINANCIAL
        assert len(template.widgets) == 1
        assert template.layout_config["grid_size"] == 12
        assert len(template.default_filters) == 1
        assert template.version == "1.0.0"
        assert "finance" in template.tags


if __name__ == "__main__":
    pytest.main([__file__])