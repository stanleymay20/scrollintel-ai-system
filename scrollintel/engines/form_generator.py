"""
Form Generation Engine

This module provides specialized functionality for generating forms
with validation, data binding, and accessibility compliance.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime

from ..models.frontend_generation_models import (
    UIComponent, ComponentType, FormField, FormSpecification,
    ComponentProperty, ComponentStyle, ComponentAccessibility,
    StyleFramework, AccessibilityLevel
)


class FormGenerator:
    """Specialized generator for form components"""
    
    def __init__(self, style_framework: StyleFramework = StyleFramework.TAILWIND):
        self.style_framework = style_framework
    
    def generate_form_with_validation(self, form_spec: FormSpecification) -> UIComponent:
        """Generate a complete form with validation and data binding"""
        component_name = self._sanitize_name(form_spec.name)
        
        # Generate form component code
        code = self._generate_form_component(component_name, form_spec)
        
        # Generate properties
        properties = self._generate_form_properties(form_spec)
        
        # Generate styling
        styles = self._generate_form_styles(form_spec)
        
        # Generate accessibility features
        accessibility = self._generate_form_accessibility(form_spec)
        
        return UIComponent(
            id=form_spec.id,
            name=component_name,
            type=ComponentType.FORM,
            description=f"Generated form: {form_spec.name}",
            code=code,
            properties=properties,
            styles=styles,
            accessibility=accessibility,
            dependencies=self._get_form_dependencies(),
            language="tsx"
        )
    
    def generate_dynamic_form(self, fields: List[FormField], config: Dict[str, Any]) -> UIComponent:
        """Generate a dynamic form from field definitions"""
        form_spec = FormSpecification(
            id=config.get('id', 'dynamic_form'),
            name=config.get('name', 'Dynamic Form'),
            fields=fields,
            submit_endpoint=config.get('submit_endpoint'),
            validation_mode=config.get('validation_mode', 'onChange'),
            layout=config.get('layout', 'vertical')
        )
        
        return self.generate_form_with_validation(form_spec)
    
    def generate_multi_step_form(self, steps: List[Dict[str, Any]]) -> UIComponent:
        """Generate a multi-step form"""
        component_name = "MultiStepForm"
        
        # Generate multi-step form code
        code = self._generate_multi_step_form(component_name, steps)
        
        return UIComponent(
            id="multi_step_form",
            name=component_name,
            type=ComponentType.FORM,
            description="Multi-step form with navigation",
            code=code,
            dependencies=self._get_form_dependencies() + ["react-hook-form-stepper"],
            language="tsx"
        )
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize form name for component"""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        if not sanitized or not sanitized[0].isupper():
            sanitized = 'Form' + sanitized
        return sanitized
    
    def _generate_form_component(self, name: str, form_spec: FormSpecification) -> str:
        """Generate complete form component code"""
        # Generate validation schema
        validation_schema = self._generate_validation_schema(form_spec.fields)
        
        # Generate form fields
        fields_jsx = self._generate_form_fields_jsx(form_spec.fields, form_spec.layout)
        
        # Generate submit handler
        submit_handler = self._generate_submit_handler(form_spec)
        
        return f'''import React from 'react';
import {{ useForm, Controller }} from 'react-hook-form';
import {{ yupResolver }} from '@hookform/resolvers/yup';
import * as yup from 'yup';

interface {name}Props {{
  onSubmit?: (data: any) => void;
  onError?: (errors: any) => void;
  initialValues?: Record<string, any>;
  disabled?: boolean;
  className?: string;
}}

{validation_schema}

const {name}: React.FC<{name}Props> = ({{
  onSubmit,
  onError,
  initialValues,
  disabled = false,
  className
}}) => {{
  const {{
    register,
    handleSubmit,
    control,
    formState: {{ errors, isSubmitting, isValid }},
    reset,
    watch
  }} = useForm({{
    resolver: yupResolver(validationSchema),
    defaultValues: initialValues,
    mode: '{form_spec.validation_mode}'
  }});

  {submit_handler}

  const onSubmitHandler = async (data: any) => {{
    try {{
      await handleFormSubmit(data);
      onSubmit?.(data);
    }} catch (error) {{
      onError?.(error);
    }}
  }};

  return (
    <div className={{`form-container ${{className || ''}}`}}>
      <form 
        onSubmit={{handleSubmit(onSubmitHandler)}}
        className="space-y-6"
        noValidate
      >
        {fields_jsx}
        
        <div className="form-actions flex justify-end space-x-4">
          <button
            type="button"
            onClick={{() => reset()}}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            disabled={{disabled}}
          >
            Reset
          </button>
          <button
            type="submit"
            disabled={{disabled || isSubmitting || !isValid}}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{isSubmitting ? 'Submitting...' : 'Submit'}}
          </button>
        </div>
      </form>
    </div>
  );
}};

export default {name};'''
    
    def _generate_validation_schema(self, fields: List[FormField]) -> str:
        """Generate Yup validation schema"""
        validations = []
        
        for field in fields:
            validation_parts = []
            
            # Base type validation
            if field.type == 'email':
                validation_parts.append("yup.string().email('Please enter a valid email address')")
            elif field.type == 'number':
                validation_parts.append("yup.number()")
            elif field.type == 'date':
                validation_parts.append("yup.date()")
            elif field.type == 'url':
                validation_parts.append("yup.string().url('Please enter a valid URL')")
            else:
                validation_parts.append("yup.string()")
            
            # Required validation
            if field.required:
                validation_parts.append(f"required('{field.label} is required')")
            
            # Custom validation rules
            for rule in field.validation_rules:
                if rule.startswith('min:'):
                    min_val = rule.split(':')[1]
                    if field.type == 'number':
                        validation_parts.append(f"min({min_val}, '{field.label} must be at least {min_val}')")
                    else:
                        validation_parts.append(f"min({min_val}, '{field.label} must be at least {min_val} characters')")
                elif rule.startswith('max:'):
                    max_val = rule.split(':')[1]
                    if field.type == 'number':
                        validation_parts.append(f"max({max_val}, '{field.label} must be at most {max_val}')")
                    else:
                        validation_parts.append(f"max({max_val}, '{field.label} must be at most {max_val} characters')")
                elif rule == 'phone':
                    validation_parts.append("matches(/^[+]?[1-9]?[0-9]{7,15}$/, 'Please enter a valid phone number')")
                elif rule.startswith('pattern:'):
                    pattern = rule.split(':', 1)[1]
                    validation_parts.append(f"matches(/{pattern}/, 'Invalid format')")
            
            # Join validation parts
            validation = f"  {field.name}: {'.'.join(validation_parts)}"
            validations.append(validation)
        
        validations_str = ',\n'.join(validations)
        return f'''const validationSchema = yup.object().shape({{
{validations_str}
}});'''
    
    def _generate_form_fields_jsx(self, fields: List[FormField], layout: str) -> str:
        """Generate form fields JSX"""
        fields_jsx = []
        
        for field in fields:
            field_jsx = self._generate_field_jsx(field, layout)
            fields_jsx.append(field_jsx)
        
        if layout == 'horizontal':
            return f'''
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {chr(10).join(fields_jsx)}
        </div>'''
        else:
            return chr(10).join(fields_jsx)
    
    def _generate_field_jsx(self, field: FormField, layout: str) -> str:
        """Generate JSX for a single form field"""
        field_id = f"field_{field.name}"
        
        # Base field structure
        if field.type == 'select' and field.options:
            return self._generate_select_field(field, field_id)
        elif field.type == 'textarea':
            return self._generate_textarea_field(field, field_id)
        elif field.type == 'checkbox':
            return self._generate_checkbox_field(field, field_id)
        elif field.type == 'radio' and field.options:
            return self._generate_radio_field(field, field_id)
        elif field.type == 'file':
            return self._generate_file_field(field, field_id)
        else:
            return self._generate_input_field(field, field_id)
    
    def _generate_input_field(self, field: FormField, field_id: str) -> str:
        """Generate input field JSX"""
        return f'''
        <div className="form-field">
          <label 
            htmlFor="{field_id}" 
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            {field.label}
            {' ' if field.required else ''}
            {field.required and '<span className="text-red-500">*</span>' or ''}
          </label>
          <input
            id="{field_id}"
            type="{field.type}"
            placeholder="{field.placeholder or ''}"
            {{...register("{field.name}")}}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
            disabled={{disabled}}
            aria-describedby="{field_id}-error"
          />
          {{errors.{field.name} && (
            <p id="{field_id}-error" className="mt-1 text-sm text-red-600" role="alert">
              {{errors.{field.name}?.message}}
            </p>
          )}}
        </div>'''
    
    def _generate_select_field(self, field: FormField, field_id: str) -> str:
        """Generate select field JSX"""
        options_jsx = []
        if field.options:
            for option in field.options:
                options_jsx.append(f'            <option value="{option}">{option}</option>')
        
        options_str = '\n'.join(options_jsx)
        
        return f'''
        <div className="form-field">
          <label 
            htmlFor="{field_id}" 
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            {field.label}
            {field.required and ' <span className="text-red-500">*</span>' or ''}
          </label>
          <select
            id="{field_id}"
            {{...register("{field.name}")}}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
            disabled={{disabled}}
            aria-describedby="{field_id}-error"
          >
            <option value="">Select {field.label.lower()}</option>
{options_str}
          </select>
          {{errors.{field.name} && (
            <p id="{field_id}-error" className="mt-1 text-sm text-red-600" role="alert">
              {{errors.{field.name}?.message}}
            </p>
          )}}
        </div>'''
    
    def _generate_textarea_field(self, field: FormField, field_id: str) -> str:
        """Generate textarea field JSX"""
        return f'''
        <div className="form-field">
          <label 
            htmlFor="{field_id}" 
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            {field.label}
            {field.required and ' <span className="text-red-500">*</span>' or ''}
          </label>
          <textarea
            id="{field_id}"
            placeholder="{field.placeholder or ''}"
            {{...register("{field.name}")}}
            rows={{4}}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
            disabled={{disabled}}
            aria-describedby="{field_id}-error"
          />
          {{errors.{field.name} && (
            <p id="{field_id}-error" className="mt-1 text-sm text-red-600" role="alert">
              {{errors.{field.name}?.message}}
            </p>
          )}}
        </div>'''
    
    def _generate_checkbox_field(self, field: FormField, field_id: str) -> str:
        """Generate checkbox field JSX"""
        return f'''
        <div className="form-field">
          <div className="flex items-center">
            <input
              id="{field_id}"
              type="checkbox"
              {{...register("{field.name}")}}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded disabled:opacity-50"
              disabled={{disabled}}
              aria-describedby="{field_id}-error"
            />
            <label 
              htmlFor="{field_id}" 
              className="ml-2 block text-sm text-gray-900"
            >
              {field.label}
              {field.required and ' <span className="text-red-500">*</span>' or ''}
            </label>
          </div>
          {{errors.{field.name} && (
            <p id="{field_id}-error" className="mt-1 text-sm text-red-600" role="alert">
              {{errors.{field.name}?.message}}
            </p>
          )}}
        </div>'''
    
    def _generate_radio_field(self, field: FormField, field_id: str) -> str:
        """Generate radio field JSX"""
        options_jsx = []
        if field.options:
            for i, option in enumerate(field.options):
                option_id = f"{field_id}_{i}"
                options_jsx.append(f'''
            <div className="flex items-center">
              <input
                id="{option_id}"
                type="radio"
                value="{option}"
                {{...register("{field.name}")}}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 disabled:opacity-50"
                disabled={{disabled}}
              />
              <label htmlFor="{option_id}" className="ml-2 block text-sm text-gray-900">
                {option}
              </label>
            </div>''')
        
        options_str = '\n'.join(options_jsx)
        
        return f'''
        <div className="form-field">
          <fieldset>
            <legend className="block text-sm font-medium text-gray-700 mb-2">
              {field.label}
              {field.required and ' <span className="text-red-500">*</span>' or ''}
            </legend>
            <div className="space-y-2">
{options_str}
            </div>
          </fieldset>
          {{errors.{field.name} && (
            <p className="mt-1 text-sm text-red-600" role="alert">
              {{errors.{field.name}?.message}}
            </p>
          )}}
        </div>'''
    
    def _generate_file_field(self, field: FormField, field_id: str) -> str:
        """Generate file field JSX"""
        return f'''
        <div className="form-field">
          <label 
            htmlFor="{field_id}" 
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            {field.label}
            {field.required and ' <span className="text-red-500">*</span>' or ''}
          </label>
          <input
            id="{field_id}"
            type="file"
            {{...register("{field.name}")}}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
            disabled={{disabled}}
            aria-describedby="{field_id}-error"
          />
          {{errors.{field.name} && (
            <p id="{field_id}-error" className="mt-1 text-sm text-red-600" role="alert">
              {{errors.{field.name}?.message}}
            </p>
          )}}
        </div>'''
    
    def _generate_submit_handler(self, form_spec: FormSpecification) -> str:
        """Generate form submit handler"""
        if form_spec.submit_endpoint:
            return f'''
  const handleFormSubmit = async (data: any) => {{
    const response = await fetch('{form_spec.submit_endpoint}', {{
      method: 'POST',
      headers: {{
        'Content-Type': 'application/json',
      }},
      body: JSON.stringify(data),
    }});

    if (!response.ok) {{
      throw new Error('Form submission failed');
    }}

    return response.json();
  }};'''
        else:
            return '''
  const handleFormSubmit = async (data: any) => {
    // Handle form submission
    console.log('Form data:', data);
  };'''
    
    def _generate_multi_step_form(self, name: str, steps: List[Dict[str, Any]]) -> str:
        """Generate multi-step form component"""
        steps_jsx = []
        for i, step in enumerate(steps):
            step_jsx = f'''
        {{currentStep === {i} && (
          <div className="step-content">
            <h3 className="text-lg font-medium mb-4">{step.get('title', f'Step {i+1}')}</h3>
            {{/* Step {i+1} fields would be generated here */}}
          </div>
        )}}'''
            steps_jsx.append(step_jsx)
        
        steps_str = '\n'.join(steps_jsx)
        
        return f'''import React, {{ useState }} from 'react';
import {{ useForm }} from 'react-hook-form';

interface {name}Props {{
  onComplete?: (data: any) => void;
  className?: string;
}}

const {name}: React.FC<{name}Props> = ({{ onComplete, className }}) => {{
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState({{}});
  const totalSteps = {len(steps)};

  const {{ register, handleSubmit, formState: {{ errors }} }} = useForm();

  const nextStep = () => {{
    if (currentStep < totalSteps - 1) {{
      setCurrentStep(currentStep + 1);
    }}
  }};

  const prevStep = () => {{
    if (currentStep > 0) {{
      setCurrentStep(currentStep - 1);
    }}
  }};

  const onSubmit = (data: any) => {{
    const updatedData = {{ ...formData, ...data }};
    setFormData(updatedData);

    if (currentStep === totalSteps - 1) {{
      onComplete?.(updatedData);
    }} else {{
      nextStep();
    }}
  }};

  return (
    <div className={{`multi-step-form ${{className || ''}}`}}>
      {{/* Progress indicator */}}
      <div className="progress-indicator mb-8">
        <div className="flex justify-between">
          {{Array.from({{ length: totalSteps }}).map((_, index) => (
            <div
              key={{index}}
              className={{`step-indicator ${{
                index <= currentStep ? 'active' : 'inactive'
              }}`}}
            >
              <div className="step-number">{{index + 1}}</div>
            </div>
          ))}}
        </div>
      </div>

      <form onSubmit={{handleSubmit(onSubmit)}} className="space-y-6">
        {steps_str}

        <div className="form-navigation flex justify-between">
          <button
            type="button"
            onClick={{prevStep}}
            disabled={{currentStep === 0}}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <button
            type="submit"
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            {{currentStep === totalSteps - 1 ? 'Complete' : 'Next'}}
          </button>
        </div>
      </form>
    </div>
  );
}};

export default {name};'''
    
    def _generate_form_properties(self, form_spec: FormSpecification) -> List[ComponentProperty]:
        """Generate form component properties"""
        return [
            ComponentProperty(
                name="onSubmit",
                type="function",
                required=False,
                description="Callback function called when form is submitted"
            ),
            ComponentProperty(
                name="onError",
                type="function",
                required=False,
                description="Callback function called when form submission fails"
            ),
            ComponentProperty(
                name="initialValues",
                type="object",
                required=False,
                description="Initial values for form fields"
            ),
            ComponentProperty(
                name="disabled",
                type="boolean",
                required=False,
                default_value=False,
                description="Whether the form is disabled"
            )
        ]
    
    def _generate_form_styles(self, form_spec: FormSpecification) -> ComponentStyle:
        """Generate form styling"""
        classes = ["form-container"]
        
        if form_spec.layout == 'horizontal':
            classes.extend(["grid", "grid-cols-1", "md:grid-cols-2", "gap-6"])
        else:
            classes.extend(["space-y-6"])
        
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
    
    def _generate_form_accessibility(self, form_spec: FormSpecification) -> ComponentAccessibility:
        """Generate form accessibility features"""
        return ComponentAccessibility(
            aria_labels={
                "form": f"Form: {form_spec.name}",
                "submit": "Submit form",
                "reset": "Reset form"
            },
            keyboard_navigation=True,
            screen_reader_support=True,
            color_contrast_ratio=4.5,
            focus_management=True
        )
    
    def _get_form_dependencies(self) -> List[str]:
        """Get form-specific dependencies"""
        return [
            "react",
            "@types/react",
            "react-hook-form",
            "@hookform/resolvers",
            "yup",
            "@types/yup"
        ]