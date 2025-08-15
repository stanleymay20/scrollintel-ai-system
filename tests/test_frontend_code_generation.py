"""
Tests for Frontend Code Generation System

This module contains comprehensive tests for the frontend code generation
engines, including component generation, form creation, and dashboard building.
"""

import pytest
from datetime import datetime
from typing import List, Dict, Any

from scrollintel.models.frontend_generation_models import (
    UIComponent, ComponentLibrary, UIRequirement, FormSpecification,
    DashboardSpecification, FrontendApplication, GenerationContext,
    ComponentType, StyleFramework, AccessibilityLevel, FormField,
    DashboardWidget, ComponentProperty, ComponentStyle
)
from scrollintel.engines.frontend_code_generator import FrontendCodeGenerator
from scrollintel.engines.react_component_generator import ReactComponentGenerator
from scrollintel.engines.form_generator import FormGenerator
from scrollintel.engines.dashboard_generator import DashboardGenerator


class TestFrontendCodeGenerator:
    """Test the main frontend code generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = FrontendCodeGenerator()
        self.sample_requirement = UIRequirement(
            id="test_req_1",
            description="User login form with email and password",
            component_type=ComponentType.FORM,
            responsive=True,
            accessibility_level=AccessibilityLevel.AA,
            data_binding="user_auth",
            validation_rules=["required", "email"],
            interactions=["submit", "reset"]
        )
        self.sample_context = GenerationContext(
            requirements=[self.sample_requirement],
            style_framework=StyleFramework.TAILWIND,
            accessibility_level=AccessibilityLevel.AA
        )
    
    def test_generate_component(self):
        """Test basic component generation"""
        component = self.generator.generate_component(self.sample_requirement, self.sample_context)
        
        assert isinstance(component, UIComponent)
        assert component.name is not None
        assert component.type == ComponentType.FORM
        assert component.code is not None
        assert len(component.code) > 0
        assert "React" in component.code
        assert "export default" in component.code
    
    def test_generate_component_with_responsive_design(self):
        """Test component generation with responsive design"""
        requirement = UIRequirement(
            id="responsive_test",
            description="Responsive data table",
            component_type=ComponentType.TABLE,
            responsive=True,
            accessibility_level=AccessibilityLevel.AA
        )
        
        component = self.generator.generate_component(requirement, self.sample_context)
        
        assert "responsive" in component.code.lower() or (component.styles and any("md:" in cls for cls in component.styles.classes))
        # Check that responsive design features are present
        assert component.styles is not None
    
    def test_generate_component_with_accessibility(self):
        """Test component generation with accessibility features"""
        requirement = UIRequirement(
            id="a11y_test",
            description="Accessible button component",
            component_type=ComponentType.BUTTON,
            accessibility_level=AccessibilityLevel.AAA
        )
        
        component = self.generator.generate_component(requirement, self.sample_context)
        
        assert component.accessibility is not None
        assert component.accessibility.keyboard_navigation
        assert component.accessibility.screen_reader_support
        assert "aria-" in component.code or "role=" in component.code
    
    def test_generate_form_component(self):
        """Test form component generation"""
        form_spec = FormSpecification(
            id="test_form",
            name="ContactForm",
            fields=[
                FormField(name="name", type="text", label="Full Name", required=True),
                FormField(name="email", type="email", label="Email", required=True),
                FormField(name="message", type="textarea", label="Message")
            ],
            submit_endpoint="/api/contact",
            validation_mode="onChange"
        )
        
        component = self.generator.generate_form(form_spec, self.sample_context)
        
        assert component.type == ComponentType.FORM
        assert "useForm" in component.code
        assert "yup" in component.code
        assert "/api/contact" in component.code
        assert "react-hook-form" in component.dependencies
    
    def test_generate_dashboard_component(self):
        """Test dashboard component generation"""
        dashboard_spec = DashboardSpecification(
            id="test_dashboard",
            name="SalesDashboard",
            widgets=[
                DashboardWidget(
                    id="sales_chart",
                    name="Sales Chart",
                    type="chart",
                    data_source="sales_data",
                    chart_type="line"
                ),
                DashboardWidget(
                    id="kpi_widget",
                    name="Revenue KPI",
                    type="kpi",
                    data_source="revenue_data"
                )
            ],
            refresh_interval=30
        )
        
        component = self.generator.generate_dashboard(dashboard_spec, self.sample_context)
        
        assert component.type == ComponentType.DASHBOARD
        assert "react-grid-layout" in component.dependencies
        assert "recharts" in component.dependencies
        assert "SalesDashboard" in component.code
    
    def test_generate_application(self):
        """Test complete application generation"""
        requirements = [
            UIRequirement(
                id="req_1",
                description="User registration form",
                component_type=ComponentType.FORM
            ),
            UIRequirement(
                id="req_2",
                description="User dashboard",
                component_type=ComponentType.DASHBOARD
            )
        ]
        
        application = self.generator.generate_application(requirements, self.sample_context)
        
        assert isinstance(application, FrontendApplication)
        assert len(application.components) == 2
        assert len(application.pages) > 0
        assert len(application.routing) > 0
        assert len(application.dependencies) > 0


class TestReactComponentGenerator:
    """Test the React-specific component generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = ReactComponentGenerator(StyleFramework.TAILWIND)
        self.sample_requirement = UIRequirement(
            id="react_test",
            description="Interactive data table with sorting",
            component_type=ComponentType.TABLE,
            responsive=True,
            accessibility_level=AccessibilityLevel.AA
        )
    
    def test_generate_responsive_component(self):
        """Test responsive React component generation"""
        component = self.generator.generate_responsive_component(self.sample_requirement)
        
        assert component.language == "tsx"
        assert "React.FC" in component.code
        assert "interface" in component.code
        assert "Props" in component.code
        assert len(component.properties) > 0
    
    def test_generate_form_with_validation(self):
        """Test React form generation with validation"""
        form_spec = FormSpecification(
            id="react_form",
            name="UserProfileForm",
            fields=[
                FormField(
                    name="username",
                    type="text",
                    label="Username",
                    required=True,
                    validation_rules=["min:3", "max:20"]
                ),
                FormField(
                    name="email",
                    type="email",
                    label="Email Address",
                    required=True,
                    validation_rules=["email"]
                ),
                FormField(
                    name="age",
                    type="number",
                    label="Age",
                    validation_rules=["min:18", "max:120"]
                )
            ]
        )
        
        component = self.generator.generate_form_with_validation(form_spec)
        
        assert "useForm" in component.code
        assert "yupResolver" in component.code
        assert "validationSchema" in component.code
        assert "min(3" in component.code  # Username min validation
        assert "email(" in component.code  # Email validation
        assert "react-hook-form" in component.dependencies
    
    def test_generate_dashboard_component(self):
        """Test React dashboard generation"""
        dashboard_spec = DashboardSpecification(
            id="react_dashboard",
            name="AnalyticsDashboard",
            widgets=[
                DashboardWidget(
                    id="chart_1",
                    name="Revenue Chart",
                    type="chart",
                    data_source="revenue",
                    chart_type="line",
                    size={"width": 6, "height": 4}
                ),
                DashboardWidget(
                    id="table_1",
                    name="Top Products",
                    type="table",
                    data_source="products",
                    size={"width": 6, "height": 4}
                )
            ],
            responsive=True
        )
        
        component = self.generator.generate_dashboard_component(dashboard_spec)
        
        assert "ResponsiveGridLayout" in component.code
        assert "useState" in component.code
        assert "useEffect" in component.code
        assert "RevenueChartWidget" in component.code
        assert "TopProductsWidget" in component.code
    
    def test_accessibility_compliance(self):
        """Test accessibility compliance in generated components"""
        requirement = UIRequirement(
            id="a11y_test",
            description="Accessible navigation menu",
            component_type=ComponentType.NAVIGATION,
            accessibility_level=AccessibilityLevel.AAA
        )
        
        component = self.generator.generate_responsive_component(requirement)
        
        assert component.accessibility.keyboard_navigation
        assert component.accessibility.screen_reader_support
        assert component.accessibility.color_contrast_ratio >= 7.0  # AAA level
        assert "handleKeyDown" in component.code
        assert "aria-" in component.code or "role=" in component.code


class TestFormGenerator:
    """Test the form generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = FormGenerator(StyleFramework.TAILWIND)
    
    def test_generate_simple_form(self):
        """Test simple form generation"""
        form_spec = FormSpecification(
            id="simple_form",
            name="ContactForm",
            fields=[
                FormField(name="name", type="text", label="Name", required=True),
                FormField(name="email", type="email", label="Email", required=True),
                FormField(name="message", type="textarea", label="Message")
            ]
        )
        
        component = self.generator.generate_form_with_validation(form_spec)
        
        assert component.type == ComponentType.FORM
        assert "ContactForm" in component.code
        assert "register" in component.code
        assert "handleSubmit" in component.code
        assert "errors.name" in component.code
        assert "errors.email" in component.code
    
    def test_generate_form_with_complex_validation(self):
        """Test form generation with complex validation rules"""
        fields = [
            FormField(
                name="password",
                type="password",
                label="Password",
                required=True,
                validation_rules=["min:8", "pattern:^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)"]
            ),
            FormField(
                name="phone",
                type="tel",
                label="Phone Number",
                validation_rules=["phone"]
            )
        ]
        
        form_spec = FormSpecification(
            id="complex_form",
            name="RegistrationForm",
            fields=fields
        )
        
        component = self.generator.generate_form_with_validation(form_spec)
        
        assert "min(8" in component.code
        assert "matches(" in component.code
        assert "phone" in component.code.lower()
    
    def test_generate_form_with_select_and_radio(self):
        """Test form generation with select and radio fields"""
        fields = [
            FormField(
                name="country",
                type="select",
                label="Country",
                required=True,
                options=["USA", "Canada", "UK", "Australia"]
            ),
            FormField(
                name="gender",
                type="radio",
                label="Gender",
                options=["Male", "Female", "Other"]
            )
        ]
        
        form_spec = FormSpecification(
            id="select_form",
            name="ProfileForm",
            fields=fields
        )
        
        component = self.generator.generate_form_with_validation(form_spec)
        
        assert "<select" in component.code
        assert "USA" in component.code
        assert 'type="radio"' in component.code
        assert "fieldset" in component.code
    
    def test_generate_multi_step_form(self):
        """Test multi-step form generation"""
        steps = [
            {"title": "Personal Information", "fields": ["name", "email"]},
            {"title": "Address", "fields": ["street", "city", "zip"]},
            {"title": "Preferences", "fields": ["newsletter", "notifications"]}
        ]
        
        component = self.generator.generate_multi_step_form(steps)
        
        assert "MultiStepForm" in component.code
        assert "useState" in component.code
        assert "currentStep" in component.code
        assert "nextStep" in component.code
        assert "prevStep" in component.code
        assert "progress-indicator" in component.code


class TestDashboardGenerator:
    """Test the dashboard generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = DashboardGenerator(StyleFramework.TAILWIND)
    
    def test_generate_basic_dashboard(self):
        """Test basic dashboard generation"""
        dashboard_spec = DashboardSpecification(
            id="basic_dashboard",
            name="BasicDashboard",
            widgets=[
                DashboardWidget(
                    id="widget_1",
                    name="Sales Chart",
                    type="chart",
                    data_source="sales",
                    chart_type="line"
                )
            ]
        )
        
        component = self.generator.generate_dashboard(dashboard_spec)
        
        assert component.type == ComponentType.DASHBOARD
        assert "BasicDashboard" in component.code
        assert "ResponsiveGridLayout" in component.code
        assert "SalesChartWidget" in component.code
    
    def test_generate_chart_components(self):
        """Test chart component generation"""
        widget = DashboardWidget(
            id="test_chart",
            name="Revenue Chart",
            type="chart",
            data_source="revenue",
            chart_type="bar"
        )
        
        chart_code = self.generator.generate_chart_component(widget)
        
        assert "RevenueChartWidget" in chart_code
        assert "BarChart" in chart_code
        assert "ResponsiveContainer" in chart_code
        assert "XAxis" in chart_code
        assert "YAxis" in chart_code
    
    def test_generate_kpi_widget(self):
        """Test KPI widget generation"""
        widget = DashboardWidget(
            id="kpi_test",
            name="Total Revenue",
            type="kpi",
            data_source="revenue_total"
        )
        
        kpi_code = self.generator.generate_kpi_widget(widget)
        
        assert "TotalRevenueWidget" in kpi_code
        assert "useState" in kpi_code
        assert "useEffect" in kpi_code
        assert "trendDirection" in kpi_code
        assert "percentageChange" in kpi_code
    
    def test_generate_data_table_widget(self):
        """Test data table widget generation"""
        widget = DashboardWidget(
            id="table_test",
            name="Customer List",
            type="table",
            data_source="customers",
            filters=["status", "region"]
        )
        
        table_code = self.generator.generate_data_table_widget(widget)
        
        assert "CustomerListWidget" in table_code
        assert "filteredData" in table_code
        assert "sortConfig" in table_code
        assert "handleSort" in table_code
        assert "<table" in table_code


class TestFrontendGenerationIntegration:
    """Integration tests for the complete frontend generation system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.frontend_generator = FrontendCodeGenerator()
        self.react_generator = ReactComponentGenerator()
        self.form_generator = FormGenerator()
        self.dashboard_generator = DashboardGenerator()
    
    def test_end_to_end_application_generation(self):
        """Test complete application generation workflow"""
        # Define application requirements
        requirements = [
            UIRequirement(
                id="login_form",
                description="User login form with email and password",
                component_type=ComponentType.FORM,
                responsive=True,
                accessibility_level=AccessibilityLevel.AA,
                validation_rules=["required", "email"]
            ),
            UIRequirement(
                id="user_dashboard",
                description="User dashboard with charts and KPIs",
                component_type=ComponentType.DASHBOARD,
                responsive=True,
                data_binding="user_data"
            ),
            UIRequirement(
                id="data_table",
                description="Sortable data table with pagination",
                component_type=ComponentType.TABLE,
                responsive=True,
                interactions=["sort", "paginate", "filter"]
            )
        ]
        
        context = GenerationContext(
            requirements=requirements,
            style_framework=StyleFramework.TAILWIND,
            accessibility_level=AccessibilityLevel.AA
        )
        
        # Generate complete application
        application = self.frontend_generator.generate_application(requirements, context)
        
        # Verify application structure
        assert isinstance(application, FrontendApplication)
        assert len(application.components) == 3
        assert len(application.pages) >= 3
        assert len(application.routing) >= 3
        
        # Verify each component
        form_component = next(c for c in application.components if c.type == ComponentType.FORM)
        dashboard_component = next(c for c in application.components if c.type == ComponentType.DASHBOARD)
        table_component = next(c for c in application.components if c.type == ComponentType.TABLE)
        
        assert "useForm" in form_component.code
        assert "ResponsiveGridLayout" in dashboard_component.code
        assert "useTable" in table_component.code or "table" in table_component.code.lower()
        
        # Verify dependencies
        all_deps = set(application.dependencies)
        assert "react" in all_deps
        assert "react-hook-form" in all_deps
        assert "recharts" in all_deps
        assert "react-grid-layout" in all_deps
    
    def test_component_library_consistency(self):
        """Test that all generators use consistent component library"""
        # Get component library from main generator
        main_library = self.frontend_generator.component_library
        
        # Verify library structure
        assert main_library.id is not None
        assert main_library.name is not None
        assert len(main_library.templates) > 0
        
        # Verify templates have required structure
        for template in main_library.templates:
            assert template.id is not None
            assert template.name is not None
            assert template.type is not None
            assert template.template_code is not None
            assert len(template.template_code) > 0
    
    def test_accessibility_compliance_across_generators(self):
        """Test that all generators produce accessible code"""
        requirement = UIRequirement(
            id="a11y_test",
            description="Accessible component test",
            component_type=ComponentType.FORM,
            accessibility_level=AccessibilityLevel.AA
        )
        
        context = GenerationContext(
            requirements=[requirement],
            accessibility_level=AccessibilityLevel.AA
        )
        
        # Test main generator
        main_component = self.frontend_generator.generate_component(requirement, context)
        assert main_component.accessibility is not None
        assert main_component.accessibility.keyboard_navigation
        
        # Test React generator
        react_component = self.react_generator.generate_responsive_component(requirement)
        assert react_component.accessibility is not None
        assert react_component.accessibility.screen_reader_support
        
        # Test form generator
        form_spec = FormSpecification(
            id="a11y_form",
            name="AccessibleForm",
            fields=[FormField(name="test", type="text", label="Test Field")]
        )
        form_component = self.form_generator.generate_form_with_validation(form_spec)
        assert "aria-describedby" in form_component.code
        assert "role=" in form_component.code or "aria-" in form_component.code
    
    def test_responsive_design_consistency(self):
        """Test that all generators produce responsive code"""
        requirement = UIRequirement(
            id="responsive_test",
            description="Responsive component test",
            component_type=ComponentType.DASHBOARD,
            responsive=True
        )
        
        context = GenerationContext(
            requirements=[requirement],
            responsive_breakpoints=["sm", "md", "lg", "xl"]
        )
        
        # Generate dashboard
        dashboard_spec = DashboardSpecification(
            id="responsive_dashboard",
            name="ResponsiveDashboard",
            widgets=[
                DashboardWidget(
                    id="test_widget",
                    name="Test Widget",
                    type="chart",
                    data_source="test_data"
                )
            ],
            responsive=True
        )
        
        component = self.dashboard_generator.generate_dashboard(dashboard_spec)
        
        # Verify responsive features
        assert "ResponsiveGridLayout" in component.code
        assert "breakpoints" in component.code
        assert "cols" in component.code
        
        # Check for responsive classes
        responsive_indicators = ["sm:", "md:", "lg:", "xl:", "responsive"]
        has_responsive = any(indicator in component.code for indicator in responsive_indicators)
        assert has_responsive


if __name__ == "__main__":
    pytest.main([__file__])