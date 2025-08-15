"""
Frontend Code Generator Engine

This module provides the main engine for generating frontend code,
including React components, forms, dashboards, and complete applications.
"""

import json
import re
from typing import List, Dict, Optional, Any
from datetime import datetime

from ..models.frontend_generation_models import (
    UIComponent, ComponentLibrary, ComponentTemplate, ComponentType,
    StyleFramework, AccessibilityLevel, UIRequirement, FormSpecification,
    DashboardSpecification, FrontendApplication, GenerationContext,
    ComponentProperty, ComponentStyle, ComponentAccessibility
)


class FrontendCodeGenerator:
    """Main engine for generating frontend code"""
    
    def __init__(self):
        self.component_library = ComponentLibrary(
            id="default",
            name="Default Component Library",
            description="Default components for code generation",
            version="1.0.0",
            framework=StyleFramework.TAILWIND
        )
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default component templates"""
        # React component templates
        self.component_library.templates = [
            ComponentTemplate(
                id="react-form",
                name="React Form",
                type=ComponentType.FORM,
                template_code=self._get_form_template(),
                placeholder_mappings={
                    "{{COMPONENT_NAME}}": "FormComponent",
                    "{{FIELDS}}": "",
                    "{{VALIDATION}}": "",
                    "{{SUBMIT_HANDLER}}": ""
                }
            ),
            ComponentTemplate(
                id="react-dashboard",
                name="React Dashboard",
                type=ComponentType.DASHBOARD,
                template_code=self._get_dashboard_template(),
                placeholder_mappings={
                    "{{COMPONENT_NAME}}": "Dashboard",
                    "{{WIDGETS}}": "",
                    "{{LAYOUT}}": "grid"
                }
            ),
            ComponentTemplate(
                id="react-table",
                name="React Data Table",
                type=ComponentType.TABLE,
                template_code=self._get_table_template(),
                placeholder_mappings={
                    "{{COMPONENT_NAME}}": "DataTable",
                    "{{COLUMNS}}": "",
                    "{{DATA_SOURCE}}": ""
                }
            ),
            ComponentTemplate(
                id="react-button",
                name="React Button",
                type=ComponentType.BUTTON,
                template_code=self._get_button_template(),
                placeholder_mappings={
                    "{{COMPONENT_NAME}}": "Button",
                    "{{BUTTON_TEXT}}": "Click me",
                    "{{BUTTON_TYPE}}": "button"
                }
            ),
            ComponentTemplate(
                id="react-card",
                name="React Card",
                type=ComponentType.CARD,
                template_code=self._get_card_template(),
                placeholder_mappings={
                    "{{COMPONENT_NAME}}": "Card",
                    "{{CARD_CONTENT}}": ""
                }
            )
        ]
    
    def generate_component(self, requirement: UIRequirement, context: GenerationContext) -> UIComponent:
        """Generate a single UI component from requirements"""
        template = self._get_template_for_type(requirement.component_type)
        if not template:
            raise ValueError(f"No template found for component type: {requirement.component_type}")
        
        # Generate component code
        code = self._generate_component_code(template, requirement, context)
        
        # Generate properties
        properties = self._generate_component_properties(requirement, context)
        
        # Generate styling
        styles = self._generate_component_styles(requirement, context)
        
        # Generate accessibility features
        accessibility = self._generate_accessibility_features(requirement, context)
        
        # Generate dependencies
        dependencies = self._generate_dependencies(requirement, context)
        
        component = UIComponent(
            id=f"{requirement.id}_component",
            name=self._generate_component_name(requirement.description),
            type=requirement.component_type,
            description=requirement.description,
            code=code,
            properties=properties,
            styles=styles,
            accessibility=accessibility,
            dependencies=dependencies,
            api_integrations=self._extract_api_integrations(requirement),
            test_cases=self._generate_test_cases(requirement)
        )
        
        return component
    
    def generate_form(self, form_spec: FormSpecification, context: GenerationContext) -> UIComponent:
        """Generate a form component from specification"""
        template = self._get_template_for_type(ComponentType.FORM)
        
        # Generate form fields code
        fields_code = self._generate_form_fields(form_spec.fields, context)
        
        # Generate validation code
        validation_code = self._generate_form_validation(form_spec.fields)
        
        # Generate submit handler
        submit_handler = self._generate_submit_handler(form_spec)
        
        # Replace template placeholders
        code = template.template_code
        code = code.replace("{{COMPONENT_NAME}}", form_spec.name)
        code = code.replace("{{FIELDS}}", fields_code)
        code = code.replace("{{VALIDATION}}", validation_code)
        code = code.replace("{{SUBMIT_HANDLER}}", submit_handler)
        
        return UIComponent(
            id=form_spec.id,
            name=form_spec.name,
            type=ComponentType.FORM,
            description=f"Generated form: {form_spec.name}",
            code=code,
            dependencies=["react", "react-hook-form", "@hookform/resolvers"]
        )
    
    def generate_dashboard(self, dashboard_spec: DashboardSpecification, context: GenerationContext) -> UIComponent:
        """Generate a dashboard component from specification"""
        template = self._get_template_for_type(ComponentType.DASHBOARD)
        
        # Generate widgets code
        widgets_code = self._generate_dashboard_widgets(dashboard_spec.widgets, context)
        
        # Replace template placeholders
        code = template.template_code
        code = code.replace("{{COMPONENT_NAME}}", dashboard_spec.name)
        code = code.replace("{{WIDGETS}}", widgets_code)
        code = code.replace("{{LAYOUT}}", dashboard_spec.layout)
        
        return UIComponent(
            id=dashboard_spec.id,
            name=dashboard_spec.name,
            type=ComponentType.DASHBOARD,
            description=f"Generated dashboard: {dashboard_spec.name}",
            code=code,
            dependencies=["react", "recharts", "react-grid-layout"]
        )
    
    def generate_application(self, requirements: List[UIRequirement], context: GenerationContext) -> FrontendApplication:
        """Generate a complete frontend application"""
        components = []
        
        # Generate individual components
        for requirement in requirements:
            component = self.generate_component(requirement, context)
            components.append(component)
        
        # Generate routing
        routing = self._generate_routing(components)
        
        # Generate pages
        pages = self._generate_pages(components, context)
        
        # Collect all dependencies
        all_dependencies = set()
        for component in components:
            all_dependencies.update(component.dependencies)
        
        application = FrontendApplication(
            id=f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Generated Application",
            description="Auto-generated frontend application",
            components=components,
            pages=pages,
            routing=routing,
            dependencies=list(all_dependencies),
            styling=ComponentStyle(
                framework=context.style_framework,
                classes=["responsive", "accessible"]
            )
        )
        
        return application
    
    def _get_template_for_type(self, component_type: ComponentType) -> Optional[ComponentTemplate]:
        """Get template for component type"""
        for template in self.component_library.templates:
            if template.type == component_type:
                return template
        return None
    
    def _generate_component_code(self, template: ComponentTemplate, requirement: UIRequirement, context: GenerationContext) -> str:
        """Generate component code from template"""
        code = template.template_code
        
        # Replace basic placeholders
        component_name = self._generate_component_name(requirement.description)
        code = code.replace("{{COMPONENT_NAME}}", component_name)
        code = code.replace("{{BUTTON_TYPE}}", "button")
        code = code.replace("{{BUTTON_TEXT}}", "Click me")
        code = code.replace("{{CARD_CONTENT}}", "")
        
        # Add responsive design
        if requirement.responsive:
            code = self._add_responsive_design(code, context)
        
        # Add accessibility features
        if requirement.accessibility_level != AccessibilityLevel.BASIC:
            code = self._add_accessibility_features(code, requirement.accessibility_level)
        
        return code
    
    def _generate_component_name(self, description: str) -> str:
        """Generate a valid component name from description"""
        # Convert to PascalCase
        words = re.findall(r'\w+', description)
        return ''.join(word.capitalize() for word in words)
    
    def _generate_component_properties(self, requirement: UIRequirement, context: GenerationContext) -> List[ComponentProperty]:
        """Generate component properties"""
        properties = []
        
        # Add data binding property if specified
        if requirement.data_binding:
            properties.append(ComponentProperty(
                name="data",
                type="any",
                required=True,
                description=f"Data for {requirement.data_binding}"
            ))
        
        # Add common properties based on component type
        if requirement.component_type == ComponentType.FORM:
            properties.extend([
                ComponentProperty(name="onSubmit", type="function", required=True),
                ComponentProperty(name="initialValues", type="object", required=False),
                ComponentProperty(name="validationSchema", type="object", required=False)
            ])
        elif requirement.component_type == ComponentType.TABLE:
            properties.extend([
                ComponentProperty(name="data", type="array", required=True),
                ComponentProperty(name="columns", type="array", required=True),
                ComponentProperty(name="onRowClick", type="function", required=False)
            ])
        
        return properties
    
    def _generate_component_styles(self, requirement: UIRequirement, context: GenerationContext) -> ComponentStyle:
        """Generate component styling"""
        classes = []
        
        # Add responsive classes
        if requirement.responsive:
            classes.extend(["responsive", "flex", "flex-col", "md:flex-row"])
        
        # Add accessibility classes
        if requirement.accessibility_level != AccessibilityLevel.BASIC:
            classes.extend(["focus:outline-none", "focus:ring-2", "focus:ring-blue-500"])
        
        return ComponentStyle(
            framework=context.style_framework,
            classes=classes,
            responsive_breakpoints=dict(zip(context.responsive_breakpoints, ["640px", "768px", "1024px", "1280px"]))
        )
    
    def _generate_accessibility_features(self, requirement: UIRequirement, context: GenerationContext) -> ComponentAccessibility:
        """Generate accessibility features"""
        return ComponentAccessibility(
            aria_labels={"main": f"Main {requirement.component_type.value} component"},
            keyboard_navigation=True,
            screen_reader_support=True,
            color_contrast_ratio=4.5 if requirement.accessibility_level == AccessibilityLevel.AA else 7.0,
            focus_management=True
        )
    
    def _generate_dependencies(self, requirement: UIRequirement, context: GenerationContext) -> List[str]:
        """Generate component dependencies"""
        dependencies = ["react"]
        
        # Add framework-specific dependencies
        if context.style_framework == StyleFramework.TAILWIND:
            dependencies.append("tailwindcss")
        elif context.style_framework == StyleFramework.MATERIAL_UI:
            dependencies.extend(["@mui/material", "@emotion/react", "@emotion/styled"])
        
        # Add component-specific dependencies
        if requirement.component_type == ComponentType.FORM:
            dependencies.extend(["react-hook-form", "@hookform/resolvers", "yup"])
        elif requirement.component_type == ComponentType.CHART:
            dependencies.extend(["recharts", "d3"])
        elif requirement.component_type == ComponentType.TABLE:
            dependencies.extend(["@tanstack/react-table"])
        
        return dependencies
    
    def _extract_api_integrations(self, requirement: UIRequirement) -> List[str]:
        """Extract API integrations from requirement"""
        integrations = []
        if requirement.data_binding:
            integrations.append(f"GET /api/{requirement.data_binding}")
        return integrations
    
    def _generate_test_cases(self, requirement: UIRequirement) -> List[str]:
        """Generate test cases for component"""
        return [
            "renders without crashing",
            "handles user interactions correctly",
            "displays data properly",
            "meets accessibility requirements"
        ]
    
    def _get_form_template(self) -> str:
        """Get React form template"""
        return '''import React from 'react';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';

interface {{COMPONENT_NAME}}Props {
  onSubmit: (data: any) => void;
  initialValues?: any;
}

const {{COMPONENT_NAME}}: React.FC<{{COMPONENT_NAME}}Props> = ({ onSubmit, initialValues }) => {
  const { register, handleSubmit, formState: { errors } } = useForm({
    resolver: yupResolver(validationSchema),
    defaultValues: initialValues
  });

  {{VALIDATION}}

  {{SUBMIT_HANDLER}}

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      {{FIELDS}}
      <button
        type="submit"
        className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        Submit
      </button>
    </form>
  );
};

export default {{COMPONENT_NAME}};'''
    
    def _get_dashboard_template(self) -> str:
        """Get React dashboard template"""
        return '''import React from 'react';
import { Grid } from 'react-grid-layout';

interface {{COMPONENT_NAME}}Props {
  data?: any;
  layout?: string;
}

const {{COMPONENT_NAME}}: React.FC<{{COMPONENT_NAME}}Props> = ({ data, layout = "{{LAYOUT}}" }) => {
  return (
    <div className="dashboard-container p-4">
      <h1 className="text-2xl font-bold mb-6">{{COMPONENT_NAME}}</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {{WIDGETS}}
      </div>
    </div>
  );
};

export default {{COMPONENT_NAME}};'''
    
    def _get_table_template(self) -> str:
        """Get React table template"""
        return '''import React from 'react';
import { useTable, useSortBy, usePagination } from '@tanstack/react-table';

interface {{COMPONENT_NAME}}Props {
  data: any[];
  columns: any[];
  onRowClick?: (row: any) => void;
}

const {{COMPONENT_NAME}}: React.FC<{{COMPONENT_NAME}}Props> = ({ data, columns, onRowClick }) => {
  const table = useTable(
    { columns, data },
    useSortBy,
    usePagination
  );

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full bg-white border border-gray-200">
        <thead className="bg-gray-50">
          {table.getHeaderGroups().map(headerGroup => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map(header => (
                <th
                  key={header.id}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={header.column.getToggleSortingHandler()}
                >
                  {header.render('Header')}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {table.getRowModel().rows.map(row => (
            <tr
              key={row.id}
              className="hover:bg-gray-50 cursor-pointer"
              onClick={() => onRowClick?.(row.original)}
            >
              {row.getVisibleCells().map(cell => (
                <td key={cell.id} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {cell.render('Cell')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default {{COMPONENT_NAME}};'''
    
    def _get_button_template(self) -> str:
        """Get React button template"""
        return '''import React from 'react';

interface {{COMPONENT_NAME}}Props {
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
  disabled?: boolean;
  children?: React.ReactNode;
  className?: string;
}

const {{COMPONENT_NAME}}: React.FC<{{COMPONENT_NAME}}Props> = ({
  onClick,
  type = '{{BUTTON_TYPE}}',
  disabled = false,
  children = '{{BUTTON_TEXT}}',
  className
}) => {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed ${className || ''}`}
    >
      {children}
    </button>
  );
};

export default {{COMPONENT_NAME}};'''
    
    def _get_card_template(self) -> str:
        """Get React card template"""
        return '''import React from 'react';

interface {{COMPONENT_NAME}}Props {
  title?: string;
  children?: React.ReactNode;
  className?: string;
}

const {{COMPONENT_NAME}}: React.FC<{{COMPONENT_NAME}}Props> = ({
  title,
  children,
  className
}) => {
  return (
    <div className={`bg-white shadow rounded-lg p-6 ${className || ''}`}>
      {title && (
        <div className="card-header mb-4">
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
        </div>
      )}
      <div className="card-body">
        {{CARD_CONTENT}}
        {children}
      </div>
    </div>
  );
};

export default {{COMPONENT_NAME}};'''
    
    def _generate_form_fields(self, fields, context: GenerationContext) -> str:
        """Generate form fields code"""
        field_code = []
        for field in fields:
            field_html = f'''
      <div className="form-field">
        <label htmlFor="{field.name}" className="block text-sm font-medium text-gray-700">
          {field.label}
        </label>
        <input
          id="{field.name}"
          type="{field.type}"
          placeholder="{field.placeholder or ''}"
          {{...register("{field.name}", {{ required: {str(field.required).lower()} }})}}
          className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
        />
        {{errors.{field.name} && <span className="text-red-500 text-sm">{{errors.{field.name}.message}}</span>}}
      </div>'''
            field_code.append(field_html)
        return '\n'.join(field_code)
    
    def _generate_form_validation(self, fields) -> str:
        """Generate form validation schema"""
        return "const validationSchema = yup.object().shape({});"
    
    def _generate_submit_handler(self, form_spec: FormSpecification) -> str:
        """Generate form submit handler"""
        if form_spec.submit_endpoint:
            return f'''
  const handleFormSubmit = async (data: any) => {{
    try {{
      const response = await fetch('{form_spec.submit_endpoint}', {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json',
        }},
        body: JSON.stringify(data),
      }});
      
      if (response.ok) {{
        console.log('Form submitted successfully');
      }} else {{
        console.error('Form submission failed');
      }}
    }} catch (error) {{
      console.error('Network error:', error);
    }}
  }};'''
        else:
            return '''
  const handleFormSubmit = (data: any) => {
    console.log('Form data:', data);
  };'''
    
    def _generate_dashboard_widgets(self, widgets, context: GenerationContext) -> str:
        """Generate dashboard widgets code"""
        widget_code = []
        for widget in widgets:
            widget_html = f'''
        <div className="widget bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">{widget.name}</h3>
          <div className="widget-content">
            {{/* Widget content for {widget.type} */}}
          </div>
        </div>'''
            widget_code.append(widget_html)
        return '\n'.join(widget_code)
    
    def _generate_routing(self, components: List[UIComponent]) -> Dict[str, str]:
        """Generate routing configuration"""
        routing = {}
        for component in components:
            route_path = f"/{component.name.lower().replace(' ', '-')}"
            routing[route_path] = component.name
        return routing
    
    def _generate_pages(self, components: List[UIComponent], context: GenerationContext) -> List[str]:
        """Generate page components"""
        pages = []
        for component in components:
            page_name = f"{component.name}Page"
            pages.append(page_name)
        return pages
    
    def _add_responsive_design(self, code: str, context: GenerationContext) -> str:
        """Add responsive design features to code"""
        # Add responsive classes and breakpoints
        return code
    
    def _add_accessibility_features(self, code: str, level: AccessibilityLevel) -> str:
        """Add accessibility features to code"""
        if level == AccessibilityLevel.BASIC:
            return code
        
        # Add ARIA attributes and accessibility features
        if "<button" in code:
            # Add ARIA attributes to buttons
            code = code.replace(
                "<button",
                '<button\n      aria-label="Button"\n      role="button"'
            )
        
        if "<input" in code:
            # Add ARIA attributes to inputs
            code = code.replace(
                "<input",
                '<input\n      aria-describedby="field-description"'
            )
        
        if "<form" in code:
            # Add ARIA attributes to forms
            code = code.replace(
                "<form",
                '<form\n      role="form"\n      aria-label="Form"'
            )
        
        return code