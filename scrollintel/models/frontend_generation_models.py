"""
Frontend Code Generation Models

This module defines the data models for frontend code generation,
including UI components, component libraries, and frontend specifications.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ComponentType(Enum):
    """Types of UI components that can be generated"""
    FORM = "form"
    DASHBOARD = "dashboard"
    TABLE = "table"
    CHART = "chart"
    MODAL = "modal"
    NAVIGATION = "navigation"
    CARD = "card"
    BUTTON = "button"
    INPUT = "input"
    LAYOUT = "layout"


class StyleFramework(Enum):
    """Supported CSS/styling frameworks"""
    TAILWIND = "tailwind"
    BOOTSTRAP = "bootstrap"
    MATERIAL_UI = "material-ui"
    CHAKRA_UI = "chakra-ui"
    STYLED_COMPONENTS = "styled-components"


class AccessibilityLevel(Enum):
    """Accessibility compliance levels"""
    BASIC = "basic"
    AA = "aa"
    AAA = "aaa"


@dataclass
class UIRequirement:
    """Represents a UI requirement from natural language"""
    id: str
    description: str
    component_type: ComponentType
    priority: int = 1
    accessibility_level: AccessibilityLevel = AccessibilityLevel.AA
    responsive: bool = True
    data_binding: Optional[str] = None
    validation_rules: List[str] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)


@dataclass
class ComponentProperty:
    """Represents a property of a UI component"""
    name: str
    type: str
    required: bool = False
    default_value: Optional[Any] = None
    description: Optional[str] = None
    validation: Optional[str] = None


@dataclass
class ComponentStyle:
    """Represents styling information for a component"""
    framework: StyleFramework
    classes: List[str] = field(default_factory=list)
    custom_css: Optional[str] = None
    responsive_breakpoints: Dict[str, str] = field(default_factory=dict)
    theme_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class ComponentAccessibility:
    """Represents accessibility features of a component"""
    aria_labels: Dict[str, str] = field(default_factory=dict)
    keyboard_navigation: bool = True
    screen_reader_support: bool = True
    color_contrast_ratio: float = 4.5
    focus_management: bool = True


@dataclass
class UIComponent:
    """Represents a generated UI component"""
    id: str
    name: str
    type: ComponentType
    description: str
    code: str
    language: str = "tsx"
    properties: List[ComponentProperty] = field(default_factory=list)
    styles: Optional[ComponentStyle] = None
    accessibility: Optional[ComponentAccessibility] = None
    dependencies: List[str] = field(default_factory=list)
    api_integrations: List[str] = field(default_factory=list)
    test_cases: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComponentTemplate:
    """Template for generating components"""
    id: str
    name: str
    type: ComponentType
    template_code: str
    placeholder_mappings: Dict[str, str] = field(default_factory=dict)
    required_props: List[str] = field(default_factory=list)
    optional_props: List[str] = field(default_factory=list)
    style_variants: List[str] = field(default_factory=list)


@dataclass
class ComponentLibrary:
    """Collection of reusable UI components and templates"""
    id: str
    name: str
    description: str
    version: str
    framework: StyleFramework
    components: List[UIComponent] = field(default_factory=list)
    templates: List[ComponentTemplate] = field(default_factory=list)
    design_tokens: Dict[str, Any] = field(default_factory=dict)
    theme_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class FormField:
    """Represents a form field specification"""
    name: str
    type: str
    label: str
    required: bool = False
    placeholder: Optional[str] = None
    validation_rules: List[str] = field(default_factory=list)
    options: Optional[List[str]] = None
    default_value: Optional[Any] = None


@dataclass
class FormSpecification:
    """Specification for generating forms"""
    id: str
    name: str
    fields: List[FormField]
    submit_endpoint: Optional[str] = None
    validation_mode: str = "onChange"
    layout: str = "vertical"
    styling: Optional[ComponentStyle] = None


@dataclass
class DashboardWidget:
    """Represents a dashboard widget"""
    id: str
    name: str
    type: str
    data_source: str
    chart_type: Optional[str] = None
    filters: List[str] = field(default_factory=list)
    refresh_interval: Optional[int] = None
    size: Dict[str, int] = field(default_factory=lambda: {"width": 4, "height": 3})


@dataclass
class DashboardSpecification:
    """Specification for generating dashboards"""
    id: str
    name: str
    layout: str = "grid"
    widgets: List[DashboardWidget] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    refresh_interval: int = 30
    responsive: bool = True


@dataclass
class FrontendApplication:
    """Represents a complete frontend application"""
    id: str
    name: str
    description: str
    framework: str = "react"
    components: List[UIComponent] = field(default_factory=list)
    pages: List[str] = field(default_factory=list)
    routing: Dict[str, str] = field(default_factory=dict)
    api_endpoints: List[str] = field(default_factory=list)
    styling: ComponentStyle = None
    build_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class GenerationContext:
    """Context information for code generation"""
    requirements: List[UIRequirement]
    api_spec: Optional[Dict[str, Any]] = None
    design_system: Optional[ComponentLibrary] = None
    target_framework: str = "react"
    style_framework: StyleFramework = StyleFramework.TAILWIND
    accessibility_level: AccessibilityLevel = AccessibilityLevel.AA
    responsive_breakpoints: List[str] = field(default_factory=lambda: ["sm", "md", "lg", "xl"])
    browser_support: List[str] = field(default_factory=lambda: ["chrome", "firefox", "safari", "edge"])