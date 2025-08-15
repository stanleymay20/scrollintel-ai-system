"""
Demo: Frontend Code Generation System

This script demonstrates the capabilities of the frontend code generation system,
including React component generation, form creation, and dashboard building.
"""

import json
from datetime import datetime

from scrollintel.models.frontend_generation_models import (
    UIRequirement, FormSpecification, DashboardSpecification,
    ComponentType, StyleFramework, AccessibilityLevel,
    FormField, DashboardWidget, GenerationContext
)
from scrollintel.engines.frontend_code_generator import FrontendCodeGenerator
from scrollintel.engines.react_component_generator import ReactComponentGenerator
from scrollintel.engines.form_generator import FormGenerator
from scrollintel.engines.dashboard_generator import DashboardGenerator


def demo_basic_component_generation():
    """Demo basic component generation"""
    print("=" * 60)
    print("DEMO: Basic Component Generation")
    print("=" * 60)
    
    generator = FrontendCodeGenerator()
    
    # Create a UI requirement
    requirement = UIRequirement(
        id="demo_button",
        description="Interactive submit button with loading state",
        component_type=ComponentType.BUTTON,
        responsive=True,
        accessibility_level=AccessibilityLevel.AA,
        interactions=["click", "hover", "focus"]
    )
    
    # Create generation context
    context = GenerationContext(
        requirements=[requirement],
        style_framework=StyleFramework.TAILWIND,
        accessibility_level=AccessibilityLevel.AA
    )
    
    # Generate component
    component = generator.generate_component(requirement, context)
    
    print(f"Generated Component: {component.name}")
    print(f"Type: {component.type.value}")
    print(f"Dependencies: {', '.join(component.dependencies)}")
    print(f"Test Cases: {len(component.test_cases)}")
    print("\nGenerated Code:")
    print("-" * 40)
    print(component.code)
    print()


def demo_form_generation():
    """Demo form generation with validation"""
    print("=" * 60)
    print("DEMO: Form Generation with Validation")
    print("=" * 60)
    
    form_generator = FormGenerator(StyleFramework.TAILWIND)
    
    # Create form specification
    form_spec = FormSpecification(
        id="user_registration_form",
        name="UserRegistrationForm",
        fields=[
            FormField(
                name="firstName",
                type="text",
                label="First Name",
                required=True,
                placeholder="Enter your first name",
                validation_rules=["min:2", "max:50"]
            ),
            FormField(
                name="lastName",
                type="text",
                label="Last Name",
                required=True,
                placeholder="Enter your last name",
                validation_rules=["min:2", "max:50"]
            ),
            FormField(
                name="email",
                type="email",
                label="Email Address",
                required=True,
                placeholder="Enter your email",
                validation_rules=["email"]
            ),
            FormField(
                name="password",
                type="password",
                label="Password",
                required=True,
                placeholder="Create a password",
                validation_rules=["min:8", "pattern:^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)"]
            ),
            FormField(
                name="country",
                type="select",
                label="Country",
                required=True,
                options=["United States", "Canada", "United Kingdom", "Australia", "Germany"]
            ),
            FormField(
                name="newsletter",
                type="checkbox",
                label="Subscribe to newsletter"
            )
        ],
        submit_endpoint="/api/users/register",
        validation_mode="onChange",
        layout="vertical"
    )
    
    # Generate form component
    form_component = form_generator.generate_form_with_validation(form_spec)
    
    print(f"Generated Form: {form_component.name}")
    print(f"Fields: {len(form_spec.fields)}")
    print(f"Validation Mode: {form_spec.validation_mode}")
    print(f"Submit Endpoint: {form_spec.submit_endpoint}")
    print(f"Dependencies: {', '.join(form_component.dependencies)}")
    print("\nGenerated Code (first 1000 chars):")
    print("-" * 40)
    print(form_component.code[:1000] + "..." if len(form_component.code) > 1000 else form_component.code)
    print()


def demo_dashboard_generation():
    """Demo dashboard generation with widgets"""
    print("=" * 60)
    print("DEMO: Dashboard Generation with Data Visualization")
    print("=" * 60)
    
    dashboard_generator = DashboardGenerator(StyleFramework.TAILWIND)
    
    # Create dashboard specification
    dashboard_spec = DashboardSpecification(
        id="analytics_dashboard",
        name="AnalyticsDashboard",
        layout="grid",
        widgets=[
            DashboardWidget(
                id="revenue_chart",
                name="Revenue Trends",
                type="chart",
                data_source="revenue_data",
                chart_type="line",
                size={"width": 8, "height": 4},
                refresh_interval=30
            ),
            DashboardWidget(
                id="conversion_kpi",
                name="Conversion Rate",
                type="kpi",
                data_source="conversion_data",
                size={"width": 4, "height": 2}
            ),
            DashboardWidget(
                id="top_products",
                name="Top Products",
                type="table",
                data_source="products_data",
                size={"width": 6, "height": 4},
                filters=["category", "date_range"]
            ),
            DashboardWidget(
                id="user_activity",
                name="User Activity",
                type="chart",
                data_source="activity_data",
                chart_type="bar",
                size={"width": 6, "height": 4}
            )
        ],
        filters=["date_range", "region", "product_category"],
        refresh_interval=60,
        responsive=True
    )
    
    # Generate dashboard component
    dashboard_component = dashboard_generator.generate_dashboard(dashboard_spec)
    
    print(f"Generated Dashboard: {dashboard_component.name}")
    print(f"Widgets: {len(dashboard_spec.widgets)}")
    print(f"Filters: {', '.join(dashboard_spec.filters)}")
    print(f"Refresh Interval: {dashboard_spec.refresh_interval}s")
    print(f"Dependencies: {', '.join(dashboard_component.dependencies)}")
    print("\nWidget Details:")
    for widget in dashboard_spec.widgets:
        print(f"  - {widget.name} ({widget.type}): {widget.size}")
    print("\nGenerated Code (first 1500 chars):")
    print("-" * 40)
    print(dashboard_component.code[:1500] + "..." if len(dashboard_component.code) > 1500 else dashboard_component.code)
    print()


def demo_react_component_generation():
    """Demo React-specific component generation"""
    print("=" * 60)
    print("DEMO: React Component Generation with TypeScript")
    print("=" * 60)
    
    react_generator = ReactComponentGenerator(StyleFramework.TAILWIND)
    
    # Create UI requirement for a data table
    requirement = UIRequirement(
        id="data_table",
        description="Sortable data table with pagination and search",
        component_type=ComponentType.TABLE,
        responsive=True,
        accessibility_level=AccessibilityLevel.AA,
        data_binding="table_data",
        interactions=["sort", "paginate", "search", "filter"]
    )
    
    # Generate React component
    component = react_generator.generate_responsive_component(requirement)
    
    print(f"Generated React Component: {component.name}")
    print(f"Language: {component.language}")
    print(f"Properties: {len(component.properties)}")
    print(f"Accessibility Features:")
    if component.accessibility:
        print(f"  - Keyboard Navigation: {component.accessibility.keyboard_navigation}")
        print(f"  - Screen Reader Support: {component.accessibility.screen_reader_support}")
        print(f"  - Color Contrast Ratio: {component.accessibility.color_contrast_ratio}")
    print(f"Responsive Classes: {component.styles.classes if component.styles else 'None'}")
    print("\nGenerated Code (first 1200 chars):")
    print("-" * 40)
    print(component.code[:1200] + "..." if len(component.code) > 1200 else component.code)
    print()


def demo_complete_application_generation():
    """Demo complete application generation"""
    print("=" * 60)
    print("DEMO: Complete Application Generation")
    print("=" * 60)
    
    generator = FrontendCodeGenerator()
    
    # Define application requirements
    requirements = [
        UIRequirement(
            id="login_form",
            description="User login form with email and password validation",
            component_type=ComponentType.FORM,
            responsive=True,
            accessibility_level=AccessibilityLevel.AA,
            validation_rules=["required", "email"],
            interactions=["submit", "reset"]
        ),
        UIRequirement(
            id="dashboard",
            description="Main dashboard with charts and KPIs",
            component_type=ComponentType.DASHBOARD,
            responsive=True,
            data_binding="dashboard_data",
            interactions=["refresh", "filter"]
        ),
        UIRequirement(
            id="user_profile",
            description="User profile management form",
            component_type=ComponentType.FORM,
            responsive=True,
            accessibility_level=AccessibilityLevel.AA,
            validation_rules=["required"],
            interactions=["save", "cancel"]
        ),
        UIRequirement(
            id="data_table",
            description="Data table with sorting and pagination",
            component_type=ComponentType.TABLE,
            responsive=True,
            data_binding="table_data",
            interactions=["sort", "paginate"]
        )
    ]
    
    # Create generation context
    context = GenerationContext(
        requirements=requirements,
        style_framework=StyleFramework.TAILWIND,
        accessibility_level=AccessibilityLevel.AA,
        responsive_breakpoints=["sm", "md", "lg", "xl"]
    )
    
    # Generate complete application
    application = generator.generate_application(requirements, context)
    
    print(f"Generated Application: {application.name}")
    print(f"Components: {len(application.components)}")
    print(f"Pages: {len(application.pages)}")
    print(f"Routes: {len(application.routing)}")
    print(f"Total Dependencies: {len(application.dependencies)}")
    
    print("\nApplication Structure:")
    for i, component in enumerate(application.components, 1):
        print(f"  {i}. {component.name} ({component.type.value})")
    
    print("\nRouting Configuration:")
    for route, component in application.routing.items():
        print(f"  {route} -> {component}")
    
    print(f"\nDependencies: {', '.join(sorted(application.dependencies))}")
    print()


def demo_accessibility_compliance():
    """Demo accessibility compliance features"""
    print("=" * 60)
    print("DEMO: Accessibility Compliance (WCAG AA/AAA)")
    print("=" * 60)
    
    generator = FrontendCodeGenerator()
    
    # Test different accessibility levels
    accessibility_levels = [
        (AccessibilityLevel.BASIC, "Basic"),
        (AccessibilityLevel.AA, "WCAG AA"),
        (AccessibilityLevel.AAA, "WCAG AAA")
    ]
    
    for level, level_name in accessibility_levels:
        print(f"\n{level_name} Compliance:")
        print("-" * 30)
        
        requirement = UIRequirement(
            id=f"accessible_form_{level.value}",
            description="Contact form with accessibility features",
            component_type=ComponentType.FORM,
            responsive=True,
            accessibility_level=level
        )
        
        context = GenerationContext(
            requirements=[requirement],
            accessibility_level=level
        )
        
        component = generator.generate_component(requirement, context)
        
        if component.accessibility:
            print(f"  Keyboard Navigation: {component.accessibility.keyboard_navigation}")
            print(f"  Screen Reader Support: {component.accessibility.screen_reader_support}")
            print(f"  Color Contrast Ratio: {component.accessibility.color_contrast_ratio}")
            print(f"  Focus Management: {component.accessibility.focus_management}")
            print(f"  ARIA Labels: {len(component.accessibility.aria_labels)} defined")
        
        # Check for accessibility features in code
        accessibility_features = []
        if "aria-" in component.code:
            accessibility_features.append("ARIA attributes")
        if "role=" in component.code:
            accessibility_features.append("ARIA roles")
        if "focus:" in component.code:
            accessibility_features.append("Focus styles")
        if "tabindex" in component.code.lower():
            accessibility_features.append("Tab navigation")
        
        print(f"  Code Features: {', '.join(accessibility_features) if accessibility_features else 'Basic HTML semantics'}")


def main():
    """Run all frontend code generation demos"""
    print("Frontend Code Generation System Demo")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Run all demos
        demo_basic_component_generation()
        demo_form_generation()
        demo_dashboard_generation()
        demo_react_component_generation()
        demo_complete_application_generation()
        demo_accessibility_compliance()
        
        print("=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()