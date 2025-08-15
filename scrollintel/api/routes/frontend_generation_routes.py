"""
Frontend Code Generation API Routes

This module provides REST API endpoints for frontend code generation,
including component generation, form creation, and dashboard building.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

from ...engines.frontend_code_generator import FrontendCodeGenerator
from ...engines.react_component_generator import ReactComponentGenerator
from ...engines.form_generator import FormGenerator
from ...engines.dashboard_generator import DashboardGenerator
from ...models.frontend_generation_models import (
    UIComponent, ComponentLibrary, UIRequirement, FormSpecification,
    DashboardSpecification, FrontendApplication, GenerationContext,
    ComponentType, StyleFramework, AccessibilityLevel, FormField,
    DashboardWidget
)

router = APIRouter(prefix="/api/frontend-generation", tags=["Frontend Generation"])

# Initialize generators
frontend_generator = FrontendCodeGenerator()
react_generator = ReactComponentGenerator()
form_generator = FormGenerator()
dashboard_generator = DashboardGenerator()


# Request/Response Models
class ComponentGenerationRequest(BaseModel):
    description: str
    component_type: ComponentType
    responsive: bool = True
    accessibility_level: AccessibilityLevel = AccessibilityLevel.AA
    style_framework: StyleFramework = StyleFramework.TAILWIND
    data_binding: Optional[str] = None
    validation_rules: List[str] = []
    interactions: List[str] = []


class FormGenerationRequest(BaseModel):
    name: str
    fields: List[Dict[str, Any]]
    submit_endpoint: Optional[str] = None
    validation_mode: str = "onChange"
    layout: str = "vertical"


class DashboardGenerationRequest(BaseModel):
    name: str
    layout: str = "grid"
    widgets: List[Dict[str, Any]]
    filters: List[str] = []
    refresh_interval: int = 30
    responsive: bool = True


class ApplicationGenerationRequest(BaseModel):
    name: str
    description: str
    requirements: List[Dict[str, Any]]
    style_framework: StyleFramework = StyleFramework.TAILWIND
    accessibility_level: AccessibilityLevel = AccessibilityLevel.AA


class ComponentResponse(BaseModel):
    id: str
    name: str
    type: ComponentType
    code: str
    dependencies: List[str]
    properties: List[Dict[str, Any]]
    test_cases: List[str]


class ApplicationResponse(BaseModel):
    id: str
    name: str
    description: str
    components: List[ComponentResponse]
    pages: List[str]
    routing: Dict[str, str]
    dependencies: List[str]


@router.post("/component", response_model=ComponentResponse)
async def generate_component(request: ComponentGenerationRequest):
    """Generate a single UI component from requirements"""
    try:
        # Create UI requirement
        requirement = UIRequirement(
            id=f"req_{hash(request.description)}",
            description=request.description,
            component_type=request.component_type,
            responsive=request.responsive,
            accessibility_level=request.accessibility_level,
            data_binding=request.data_binding,
            validation_rules=request.validation_rules,
            interactions=request.interactions
        )
        
        # Create generation context
        context = GenerationContext(
            requirements=[requirement],
            style_framework=request.style_framework,
            accessibility_level=request.accessibility_level
        )
        
        # Generate component
        component = frontend_generator.generate_component(requirement, context)
        
        return ComponentResponse(
            id=component.id,
            name=component.name,
            type=component.type,
            code=component.code,
            dependencies=component.dependencies,
            properties=[{
                "name": prop.name,
                "type": prop.type,
                "required": prop.required,
                "description": prop.description
            } for prop in component.properties],
            test_cases=component.test_cases
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Component generation failed: {str(e)}")


@router.post("/react-component", response_model=ComponentResponse)
async def generate_react_component(request: ComponentGenerationRequest):
    """Generate a React component with responsive design and accessibility"""
    try:
        # Create UI requirement
        requirement = UIRequirement(
            id=f"react_req_{hash(request.description)}",
            description=request.description,
            component_type=request.component_type,
            responsive=request.responsive,
            accessibility_level=request.accessibility_level,
            data_binding=request.data_binding,
            validation_rules=request.validation_rules,
            interactions=request.interactions
        )
        
        # Generate React component
        react_generator.style_framework = request.style_framework
        component = react_generator.generate_responsive_component(requirement)
        
        return ComponentResponse(
            id=component.id,
            name=component.name,
            type=component.type,
            code=component.code,
            dependencies=component.dependencies,
            properties=[{
                "name": prop.name,
                "type": prop.type,
                "required": prop.required,
                "description": prop.description
            } for prop in component.properties],
            test_cases=component.test_cases
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"React component generation failed: {str(e)}")


@router.post("/form", response_model=ComponentResponse)
async def generate_form(request: FormGenerationRequest):
    """Generate a form component with validation and data binding"""
    try:
        # Convert fields to FormField objects
        fields = []
        for field_data in request.fields:
            field = FormField(
                name=field_data.get("name", ""),
                type=field_data.get("type", "text"),
                label=field_data.get("label", ""),
                required=field_data.get("required", False),
                placeholder=field_data.get("placeholder"),
                validation_rules=field_data.get("validation_rules", []),
                options=field_data.get("options"),
                default_value=field_data.get("default_value")
            )
            fields.append(field)
        
        # Create form specification
        form_spec = FormSpecification(
            id=f"form_{hash(request.name)}",
            name=request.name,
            fields=fields,
            submit_endpoint=request.submit_endpoint,
            validation_mode=request.validation_mode,
            layout=request.layout
        )
        
        # Generate form component
        component = form_generator.generate_form_with_validation(form_spec)
        
        return ComponentResponse(
            id=component.id,
            name=component.name,
            type=component.type,
            code=component.code,
            dependencies=component.dependencies,
            properties=[{
                "name": prop.name,
                "type": prop.type,
                "required": prop.required,
                "description": prop.description
            } for prop in component.properties],
            test_cases=component.test_cases
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Form generation failed: {str(e)}")


@router.post("/dashboard", response_model=ComponentResponse)
async def generate_dashboard(request: DashboardGenerationRequest):
    """Generate a dashboard component with data visualization"""
    try:
        # Convert widgets to DashboardWidget objects
        widgets = []
        for widget_data in request.widgets:
            widget = DashboardWidget(
                id=widget_data.get("id", f"widget_{len(widgets)}"),
                name=widget_data.get("name", "Widget"),
                type=widget_data.get("type", "chart"),
                data_source=widget_data.get("data_source", ""),
                chart_type=widget_data.get("chart_type"),
                filters=widget_data.get("filters", []),
                refresh_interval=widget_data.get("refresh_interval"),
                size=widget_data.get("size", {"width": 4, "height": 3})
            )
            widgets.append(widget)
        
        # Create dashboard specification
        dashboard_spec = DashboardSpecification(
            id=f"dashboard_{hash(request.name)}",
            name=request.name,
            layout=request.layout,
            widgets=widgets,
            filters=request.filters,
            refresh_interval=request.refresh_interval,
            responsive=request.responsive
        )
        
        # Generate dashboard component
        component = dashboard_generator.generate_dashboard(dashboard_spec)
        
        return ComponentResponse(
            id=component.id,
            name=component.name,
            type=component.type,
            code=component.code,
            dependencies=component.dependencies,
            properties=[{
                "name": prop.name,
                "type": prop.type,
                "required": prop.required,
                "description": prop.description
            } for prop in component.properties],
            test_cases=component.test_cases
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")


@router.post("/application", response_model=ApplicationResponse)
async def generate_application(request: ApplicationGenerationRequest):
    """Generate a complete frontend application"""
    try:
        # Convert requirements to UIRequirement objects
        requirements = []
        for req_data in request.requirements:
            requirement = UIRequirement(
                id=f"req_{len(requirements)}",
                description=req_data.get("description", ""),
                component_type=ComponentType(req_data.get("component_type", "form")),
                responsive=req_data.get("responsive", True),
                accessibility_level=request.accessibility_level,
                data_binding=req_data.get("data_binding"),
                validation_rules=req_data.get("validation_rules", []),
                interactions=req_data.get("interactions", [])
            )
            requirements.append(requirement)
        
        # Create generation context
        context = GenerationContext(
            requirements=requirements,
            style_framework=request.style_framework,
            accessibility_level=request.accessibility_level
        )
        
        # Generate application
        application = frontend_generator.generate_application(requirements, context)
        
        # Convert components to response format
        component_responses = []
        for component in application.components:
            component_response = ComponentResponse(
                id=component.id,
                name=component.name,
                type=component.type,
                code=component.code,
                dependencies=component.dependencies,
                properties=[{
                    "name": prop.name,
                    "type": prop.type,
                    "required": prop.required,
                    "description": prop.description
                } for prop in component.properties],
                test_cases=component.test_cases
            )
            component_responses.append(component_response)
        
        return ApplicationResponse(
            id=application.id,
            name=application.name,
            description=application.description,
            components=component_responses,
            pages=application.pages,
            routing=application.routing,
            dependencies=application.dependencies
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Application generation failed: {str(e)}")


@router.get("/component-library", response_model=Dict[str, Any])
async def get_component_library():
    """Get the available component library and templates"""
    try:
        library = frontend_generator.component_library
        
        return {
            "id": library.id,
            "name": library.name,
            "description": library.description,
            "version": library.version,
            "framework": library.framework.value,
            "templates": [
                {
                    "id": template.id,
                    "name": template.name,
                    "type": template.type.value,
                    "required_props": template.required_props,
                    "optional_props": template.optional_props
                }
                for template in library.templates
            ],
            "components": [
                {
                    "id": component.id,
                    "name": component.name,
                    "type": component.type.value,
                    "description": component.description
                }
                for component in library.components
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get component library: {str(e)}")


@router.post("/validate-component")
async def validate_component_code(code: str, component_type: ComponentType):
    """Validate generated component code"""
    try:
        # Basic validation checks
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check for React imports
        if "import React" not in code:
            validation_results["errors"].append("Missing React import")
            validation_results["valid"] = False
        
        # Check for component export
        if "export default" not in code:
            validation_results["errors"].append("Missing default export")
            validation_results["valid"] = False
        
        # Check for TypeScript interface
        if "interface" not in code and "Props" in code:
            validation_results["warnings"].append("Consider adding TypeScript interface for props")
        
        # Check for accessibility attributes
        if "aria-" not in code and component_type in [ComponentType.FORM, ComponentType.BUTTON]:
            validation_results["suggestions"].append("Consider adding ARIA attributes for better accessibility")
        
        # Check for responsive classes
        if "responsive" not in code and "md:" not in code:
            validation_results["suggestions"].append("Consider adding responsive design classes")
        
        return validation_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code validation failed: {str(e)}")


@router.get("/supported-frameworks")
async def get_supported_frameworks():
    """Get list of supported styling frameworks"""
    return {
        "frameworks": [
            {
                "name": framework.value,
                "display_name": framework.value.replace("_", " ").title(),
                "description": f"Support for {framework.value} styling framework"
            }
            for framework in StyleFramework
        ]
    }


@router.get("/component-types")
async def get_component_types():
    """Get list of supported component types"""
    return {
        "types": [
            {
                "name": comp_type.value,
                "display_name": comp_type.value.replace("_", " ").title(),
                "description": f"{comp_type.value.title()} component type"
            }
            for comp_type in ComponentType
        ]
    }