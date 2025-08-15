"""
React Component Generator

This module provides specialized functionality for generating React components
with responsive design, accessibility compliance, and modern best practices.
"""

import re
from typing import List, Dict, Optional, Any
from datetime import datetime

from ..models.frontend_generation_models import (
    UIComponent, ComponentType, StyleFramework, AccessibilityLevel,
    UIRequirement, ComponentProperty, ComponentStyle, ComponentAccessibility,
    FormField, FormSpecification, DashboardWidget, DashboardSpecification
)


class ReactComponentGenerator:
    """Specialized generator for React components"""
    
    def __init__(self, style_framework: StyleFramework = StyleFramework.TAILWIND):
        self.style_framework = style_framework
        self.component_counter = 0
    
    def generate_responsive_component(self, requirement: UIRequirement) -> UIComponent:
        """Generate a responsive React component"""
        component_name = self._generate_component_name(requirement.description)
        
        # Generate base component structure
        code = self._generate_base_component(component_name, requirement)
        
        # Add responsive design
        code = self._add_responsive_features(code, requirement)
        
        # Add accessibility features
        code = self._add_accessibility_compliance(code, requirement.accessibility_level)
        
        # Generate properties
        properties = self._generate_component_props(requirement)
        
        # Generate styling
        styles = self._generate_responsive_styles(requirement)
        
        # Generate accessibility metadata
        accessibility = self._generate_accessibility_metadata(requirement)
        
        return UIComponent(
            id=f"react_component_{self.component_counter}",
            name=component_name,
            type=requirement.component_type,
            description=requirement.description,
            code=code,
            properties=properties,
            styles=styles,
            accessibility=accessibility,
            dependencies=self._get_react_dependencies(requirement),
            language="tsx"
        )
    
    def generate_form_with_validation(self, form_spec: FormSpecification) -> UIComponent:
        """Generate a React form with validation and data binding"""
        component_name = self._sanitize_component_name(form_spec.name)
        
        # Generate form component code
        code = self._generate_form_component(component_name, form_spec)
        
        # Add validation logic
        code = self._add_form_validation(code, form_spec.fields)
        
        # Add data binding
        code = self._add_data_binding(code, form_spec)
        
        return UIComponent(
            id=form_spec.id,
            name=component_name,
            type=ComponentType.FORM,
            description=f"React form component: {form_spec.name}",
            code=code,
            dependencies=["react", "react-hook-form", "@hookform/resolvers", "yup"],
            language="tsx"
        )
    
    def generate_dashboard_component(self, dashboard_spec: DashboardSpecification) -> UIComponent:
        """Generate a React dashboard with data visualization"""
        component_name = self._sanitize_component_name(dashboard_spec.name)
        
        # Generate dashboard component code
        code = self._generate_dashboard_component(component_name, dashboard_spec)
        
        # Add data visualization widgets
        code = self._add_visualization_widgets(code, dashboard_spec.widgets)
        
        # Add responsive grid layout
        code = self._add_responsive_grid(code, dashboard_spec)
        
        return UIComponent(
            id=dashboard_spec.id,
            name=component_name,
            type=ComponentType.DASHBOARD,
            description=f"React dashboard component: {dashboard_spec.name}",
            code=code,
            dependencies=["react", "recharts", "react-grid-layout", "@types/react-grid-layout"],
            language="tsx"
        )
    
    def _generate_component_name(self, description: str) -> str:
        """Generate a valid React component name"""
        # Extract meaningful words and convert to PascalCase
        words = re.findall(r'\b[a-zA-Z]+\b', description)
        if not words:
            self.component_counter += 1
            return f"GeneratedComponent{self.component_counter}"
        
        name = ''.join(word.capitalize() for word in words[:3])  # Limit to 3 words
        return self._sanitize_component_name(name)
    
    def _sanitize_component_name(self, name: str) -> str:
        """Sanitize component name for React"""
        # Remove invalid characters and ensure it starts with uppercase
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        if not sanitized or not sanitized[0].isupper():
            sanitized = 'Component' + sanitized
        return sanitized
    
    def _generate_base_component(self, name: str, requirement: UIRequirement) -> str:
        """Generate base React component structure"""
        props_interface = self._generate_props_interface(name, requirement)
        
        return f'''import React from 'react';
{self._get_additional_imports(requirement)}

{props_interface}

const {name}: React.FC<{name}Props> = (props) => {{
  {self._generate_component_logic(requirement)}

  return (
    <div className="{self._get_base_classes(requirement)}">
      {self._generate_component_content(requirement)}
    </div>
  );
}};

export default {name};'''
    
    def _generate_props_interface(self, name: str, requirement: UIRequirement) -> str:
        """Generate TypeScript props interface"""
        props = []
        
        # Add common props
        props.append("  className?: string;")
        
        # Add data binding props
        if requirement.data_binding:
            props.append(f"  data?: any;")
            props.append(f"  onDataChange?: (data: any) => void;")
        
        # Add component-specific props
        if requirement.component_type == ComponentType.FORM:
            props.extend([
                "  onSubmit?: (data: any) => void;",
                "  initialValues?: Record<string, any>;",
                "  validationSchema?: any;"
            ])
        elif requirement.component_type == ComponentType.TABLE:
            props.extend([
                "  data: any[];",
                "  columns: any[];",
                "  onRowClick?: (row: any) => void;"
            ])
        elif requirement.component_type == ComponentType.DASHBOARD:
            props.extend([
                "  widgets?: any[];",
                "  layout?: 'grid' | 'flex';",
                "  refreshInterval?: number;"
            ])
        
        props_str = '\n'.join(props)
        return f'''interface {name}Props {{
{props_str}
}}'''
    
    def _generate_component_logic(self, requirement: UIRequirement) -> str:
        """Generate component logic and hooks"""
        logic = []
        
        # Add state management
        if requirement.data_binding:
            logic.append("  const [data, setData] = React.useState(props.data);")
        
        # Add effect hooks for data fetching
        if requirement.component_type in [ComponentType.DASHBOARD, ComponentType.TABLE]:
            logic.append("""  React.useEffect(() => {
    // Fetch data or setup subscriptions
    if (props.data) {
      setData(props.data);
    }
  }, [props.data]);""")
        
        # Add form-specific logic
        if requirement.component_type == ComponentType.FORM:
            logic.append("""  const handleSubmit = (formData: any) => {
    props.onSubmit?.(formData);
  };""")
        
        return '\n'.join(logic)
    
    def _generate_component_content(self, requirement: UIRequirement) -> str:
        """Generate component JSX content"""
        if requirement.component_type == ComponentType.FORM:
            return self._generate_form_content()
        elif requirement.component_type == ComponentType.TABLE:
            return self._generate_table_content()
        elif requirement.component_type == ComponentType.DASHBOARD:
            return self._generate_dashboard_content()
        elif requirement.component_type == ComponentType.CARD:
            return self._generate_card_content()
        else:
            return f'''
      <h2 className="text-xl font-semibold mb-4">{requirement.description}</h2>
      <div className="content">
        {{/* Component content will be generated here */}}
      </div>'''
    
    def _generate_form_content(self) -> str:
        """Generate form JSX content"""
        return '''
      <form onSubmit={handleSubmit} className="space-y-4">
        {{/* Form fields will be generated here */}}
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          Submit
        </button>
      </form>'''
    
    def _generate_table_content(self) -> str:
        """Generate table JSX content"""
        return '''
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            {{/* Table headers will be generated here */}}
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {{/* Table rows will be generated here */}}
          </tbody>
        </table>
      </div>'''
    
    def _generate_dashboard_content(self) -> str:
        """Generate dashboard JSX content"""
        return '''
      <div className="dashboard-header mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {{/* Dashboard widgets will be generated here */}}
      </div>'''
    
    def _generate_card_content(self) -> str:
        """Generate card JSX content"""
        return '''
      <div className="bg-white shadow rounded-lg p-6">
        <div className="card-header mb-4">
          <h3 className="text-lg font-medium text-gray-900">Card Title</h3>
        </div>
        <div className="card-body">
          {{/* Card content will be generated here */}}
        </div>
      </div>'''
    
    def _add_responsive_features(self, code: str, requirement: UIRequirement) -> str:
        """Add responsive design features"""
        if not requirement.responsive:
            return code
        
        # Add responsive classes and breakpoint handling
        responsive_code = '''
  // Responsive breakpoint handling
  const [isMobile, setIsMobile] = React.useState(false);
  
  React.useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);'''
        
        # Insert responsive logic after imports
        import_end = code.find('const ')
        if import_end > 0:
            code = code[:import_end] + responsive_code + '\n\n' + code[import_end:]
        
        return code
    
    def _add_accessibility_compliance(self, code: str, level: AccessibilityLevel) -> str:
        """Add accessibility compliance features"""
        if level == AccessibilityLevel.BASIC:
            return code
        
        # Add ARIA attributes and keyboard navigation
        accessibility_features = {
            AccessibilityLevel.AA: [
                'aria-label',
                'role',
                'tabIndex',
                'onKeyDown'
            ],
            AccessibilityLevel.AAA: [
                'aria-label',
                'aria-describedby',
                'role',
                'tabIndex',
                'onKeyDown',
                'aria-live',
                'aria-expanded'
            ]
        }
        
        features = accessibility_features.get(level, [])
        
        # Add keyboard navigation handler
        if 'onKeyDown' in features:
            keyboard_handler = '''
  const handleKeyDown = (event: React.KeyboardEvent) => {
    // Handle keyboard navigation
    if (event.key === 'Enter' || event.key === ' ') {
      // Handle activation
    }
  };'''
            
            # Insert keyboard handler
            logic_start = code.find('return (')
            if logic_start > 0:
                code = code[:logic_start] + keyboard_handler + '\n\n  ' + code[logic_start:]
        
        return code
    
    def _generate_component_props(self, requirement: UIRequirement) -> List[ComponentProperty]:
        """Generate component properties"""
        props = [
            ComponentProperty(
                name="className",
                type="string",
                required=False,
                description="Additional CSS classes"
            )
        ]
        
        if requirement.data_binding:
            props.append(ComponentProperty(
                name="data",
                type="any",
                required=False,
                description=f"Data for {requirement.data_binding}"
            ))
        
        return props
    
    def _generate_responsive_styles(self, requirement: UIRequirement) -> ComponentStyle:
        """Generate responsive styling"""
        classes = ["component-base"]
        
        if requirement.responsive:
            classes.extend([
                "w-full",
                "max-w-none",
                "sm:max-w-sm",
                "md:max-w-md",
                "lg:max-w-lg",
                "xl:max-w-xl"
            ])
        
        return ComponentStyle(
            framework=self.style_framework,
            classes=classes,
            responsive_breakpoints={
                "sm": "640px",
                "md": "768px",
                "lg": "1024px",
                "xl": "1280px"
            }
        )
    
    def _generate_accessibility_metadata(self, requirement: UIRequirement) -> ComponentAccessibility:
        """Generate accessibility metadata"""
        return ComponentAccessibility(
            aria_labels={
                "main": f"{requirement.component_type.value} component",
                "description": requirement.description
            },
            keyboard_navigation=True,
            screen_reader_support=True,
            color_contrast_ratio=4.5 if requirement.accessibility_level == AccessibilityLevel.AA else 7.0,
            focus_management=True
        )
    
    def _get_react_dependencies(self, requirement: UIRequirement) -> List[str]:
        """Get React-specific dependencies"""
        deps = ["react", "@types/react"]
        
        if self.style_framework == StyleFramework.TAILWIND:
            deps.append("tailwindcss")
        elif self.style_framework == StyleFramework.MATERIAL_UI:
            deps.extend(["@mui/material", "@emotion/react", "@emotion/styled"])
        
        # Add component-specific dependencies
        if requirement.component_type == ComponentType.FORM:
            deps.extend(["react-hook-form", "@hookform/resolvers", "yup"])
        elif requirement.component_type == ComponentType.CHART:
            deps.extend(["recharts", "d3"])
        elif requirement.component_type == ComponentType.TABLE:
            deps.extend(["@tanstack/react-table"])
        elif requirement.component_type == ComponentType.DASHBOARD:
            deps.extend(["react-grid-layout", "@types/react-grid-layout"])
        
        return deps
    
    def _get_additional_imports(self, requirement: UIRequirement) -> str:
        """Get additional imports for component"""
        imports = []
        
        if requirement.component_type == ComponentType.FORM:
            imports.append("import { useForm } from 'react-hook-form';")
        elif requirement.component_type == ComponentType.CHART:
            imports.append("import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';")
        elif requirement.component_type == ComponentType.DASHBOARD:
            imports.append("import { Responsive, WidthProvider } from 'react-grid-layout';")
        
        return '\n'.join(imports)
    
    def _get_base_classes(self, requirement: UIRequirement) -> str:
        """Get base CSS classes for component"""
        classes = ["component-container"]
        
        if requirement.responsive:
            classes.extend(["responsive", "w-full"])
        
        if requirement.accessibility_level != AccessibilityLevel.BASIC:
            classes.extend(["focus:outline-none", "focus:ring-2", "focus:ring-blue-500"])
        
        return " ".join(classes)
    
    def _generate_form_component(self, name: str, form_spec: FormSpecification) -> str:
        """Generate complete form component"""
        fields_code = self._generate_form_fields_code(form_spec.fields)
        validation_schema = self._generate_validation_schema(form_spec.fields)
        
        return f'''import React from 'react';
import {{ useForm }} from 'react-hook-form';
import {{ yupResolver }} from '@hookform/resolvers/yup';
import * as yup from 'yup';

interface {name}Props {{
  onSubmit?: (data: any) => void;
  initialValues?: Record<string, any>;
  className?: string;
}}

{validation_schema}

const {name}: React.FC<{name}Props> = ({{ onSubmit, initialValues, className }}) => {{
  const {{ register, handleSubmit, formState: {{ errors }} }} = useForm({{
    resolver: yupResolver(validationSchema),
    defaultValues: initialValues
  }});

  const onSubmitHandler = (data: any) => {{
    onSubmit?.(data);
  }};

  return (
    <form onSubmit={{handleSubmit(onSubmitHandler)}} className={{`space-y-4 ${{className || ''}}`}}>
      {fields_code}
      <button
        type="submit"
        className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
      >
        Submit
      </button>
    </form>
  );
}};

export default {name};'''
    
    def _generate_form_fields_code(self, fields: List[FormField]) -> str:
        """Generate form fields JSX code"""
        fields_code = []
        
        for field in fields:
            field_jsx = f'''
      <div className="form-field">
        <label htmlFor="{field.name}" className="block text-sm font-medium text-gray-700 mb-1">
          {field.label}
        </label>
        <input
          id="{field.name}"
          type="{field.type}"
          placeholder="{field.placeholder or ''}"
          {{...register("{field.name}")}}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
        />
        {{errors.{field.name} && (
          <span className="text-red-500 text-sm mt-1">{{errors.{field.name}?.message}}</span>
        )}}
      </div>'''
            fields_code.append(field_jsx)
        
        return '\n'.join(fields_code)
    
    def _generate_validation_schema(self, fields: List[FormField]) -> str:
        """Generate Yup validation schema"""
        validations = []
        
        for field in fields:
            validation = f"  {field.name}: yup.string()"
            
            if field.required:
                validation += f".required('{field.label} is required')"
            
            for rule in field.validation_rules:
                if rule.startswith('min:'):
                    min_val = rule.split(':')[1]
                    validation += f".min({min_val}, '{field.label} must be at least {min_val} characters')"
                elif rule.startswith('max:'):
                    max_val = rule.split(':')[1]
                    validation += f".max({max_val}, '{field.label} must be at most {max_val} characters')"
                elif rule == 'email':
                    validation += f".email('Please enter a valid email address')"
            
            validations.append(validation)
        
        validations_str = ',\n'.join(validations)
        return f'''const validationSchema = yup.object().shape({{
{validations_str}
}});'''
    
    def _add_form_validation(self, code: str, fields: List[FormField]) -> str:
        """Add form validation logic"""
        # Validation is already included in the form generation
        return code
    
    def _add_data_binding(self, code: str, form_spec: FormSpecification) -> str:
        """Add data binding logic"""
        if form_spec.submit_endpoint:
            # Add API call logic
            api_code = f'''
  const submitToAPI = async (data: any) => {{
    try {{
      const response = await fetch('{form_spec.submit_endpoint}', {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json',
        }},
        body: JSON.stringify(data),
      }});
      
      if (response.ok) {{
        // Handle success
        console.log('Form submitted successfully');
      }} else {{
        // Handle error
        console.error('Form submission failed');
      }}
    }} catch (error) {{
      console.error('Network error:', error);
    }}
  }};'''
            
            # Insert API code before the return statement
            return_index = code.find('return (')
            if return_index > 0:
                code = code[:return_index] + api_code + '\n\n  ' + code[return_index:]
        
        return code
    
    def _generate_dashboard_component(self, name: str, dashboard_spec: DashboardSpecification) -> str:
        """Generate complete dashboard component"""
        widgets_code = self._generate_widgets_code(dashboard_spec.widgets)
        
        return f'''import React from 'react';
import {{ Responsive, WidthProvider }} from 'react-grid-layout';

const ResponsiveGridLayout = WidthProvider(Responsive);

interface {name}Props {{
  data?: any;
  refreshInterval?: number;
  className?: string;
}}

const {name}: React.FC<{name}Props> = ({{ data, refreshInterval = {dashboard_spec.refresh_interval}, className }}) => {{
  const [dashboardData, setDashboardData] = React.useState(data);

  React.useEffect(() => {{
    const interval = setInterval(() => {{
      // Refresh dashboard data
      // This would typically fetch from an API
    }}, refreshInterval * 1000);

    return () => clearInterval(interval);
  }}, [refreshInterval]);

  return (
    <div className={{`dashboard-container ${{className || ''}}`}}>
      <div className="dashboard-header mb-6">
        <h1 className="text-2xl font-bold text-gray-900">{dashboard_spec.name}</h1>
      </div>
      <ResponsiveGridLayout
        className="layout"
        breakpoints={{{{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}}}
        cols={{{{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}}}
        rowHeight={{60}}
      >
        {widgets_code}
      </ResponsiveGridLayout>
    </div>
  );
}};

export default {name};'''
    
    def _generate_widgets_code(self, widgets: List[DashboardWidget]) -> str:
        """Generate dashboard widgets JSX code"""
        widgets_code = []
        
        for i, widget in enumerate(widgets):
            widget_jsx = f'''
        <div key="{widget.id}" data-grid={{{{ x: {i % 3 * 4}, y: {i // 3 * 3}, w: {widget.size.get('width', 4)}, h: {widget.size.get('height', 3)} }}}}>
          <div className="widget bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-2">{widget.name}</h3>
            <div className="widget-content">
              {self._generate_widget_content(widget)}
            </div>
          </div>
        </div>'''
            widgets_code.append(widget_jsx)
        
        return '\n'.join(widgets_code)
    
    def _generate_widget_content(self, widget: DashboardWidget) -> str:
        """Generate content for a specific widget"""
        if widget.chart_type:
            return f'''
              <div className="chart-container h-48">
                {{/* {widget.chart_type} chart will be rendered here */}}
                <p className="text-gray-500">Chart: {widget.chart_type}</p>
              </div>'''
        else:
            return f'''
              <div className="widget-data">
                <p className="text-gray-600">Data from: {widget.data_source}</p>
              </div>'''
    
    def _add_visualization_widgets(self, code: str, widgets: List[DashboardWidget]) -> str:
        """Add data visualization widgets"""
        # Widgets are already included in the dashboard generation
        return code
    
    def _add_responsive_grid(self, code: str, dashboard_spec: DashboardSpecification) -> str:
        """Add responsive grid layout"""
        # Grid layout is already included in the dashboard generation
        return code