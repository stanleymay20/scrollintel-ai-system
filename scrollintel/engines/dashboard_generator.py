"""
Dashboard and Data Visualization Generator

This module provides functionality for generating dashboard components
with data visualization, responsive layouts, and interactive widgets.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime

from ..models.frontend_generation_models import (
    UIComponent, ComponentType, DashboardWidget, DashboardSpecification,
    ComponentProperty, ComponentStyle, ComponentAccessibility,
    StyleFramework, AccessibilityLevel
)


class DashboardGenerator:
    """Specialized generator for dashboard and data visualization components"""
    
    def __init__(self, style_framework: StyleFramework = StyleFramework.TAILWIND):
        self.style_framework = style_framework
        self.chart_libraries = {
            'recharts': ['LineChart', 'BarChart', 'PieChart', 'AreaChart', 'ScatterChart'],
            'd3': ['custom_charts'],
            'chart.js': ['Line', 'Bar', 'Pie', 'Doughnut', 'Radar']
        }
    
    def generate_dashboard(self, dashboard_spec: DashboardSpecification) -> UIComponent:
        """Generate a complete dashboard component"""
        component_name = self._sanitize_name(dashboard_spec.name)
        
        # Generate dashboard component code
        code = self._generate_dashboard_component(component_name, dashboard_spec)
        
        # Generate properties
        properties = self._generate_dashboard_properties(dashboard_spec)
        
        # Generate styling
        styles = self._generate_dashboard_styles(dashboard_spec)
        
        # Generate accessibility features
        accessibility = self._generate_dashboard_accessibility(dashboard_spec)
        
        return UIComponent(
            id=dashboard_spec.id,
            name=component_name,
            type=ComponentType.DASHBOARD,
            description=f"Generated dashboard: {dashboard_spec.name}",
            code=code,
            properties=properties,
            styles=styles,
            accessibility=accessibility,
            dependencies=self._get_dashboard_dependencies(dashboard_spec),
            language="tsx"
        )
    
    def generate_chart_component(self, widget: DashboardWidget) -> str:
        """Generate a specific chart component"""
        chart_type = widget.chart_type or 'LineChart'
        
        if chart_type in ['line', 'bar', 'area', 'pie']:
            return self._generate_recharts_component(widget, chart_type)
        else:
            return self._generate_custom_chart_component(widget, chart_type)
    
    def generate_data_table_widget(self, widget: DashboardWidget) -> str:
        """Generate a data table widget"""
        return f'''
const {self._get_widget_name(widget)} = ({{ data, filters }}) => {{
  const [filteredData, setFilteredData] = React.useState(data || []);
  const [sortConfig, setSortConfig] = React.useState(null);

  React.useEffect(() => {{
    let filtered = data || [];
    
    // Apply filters
    if (filters && filters.length > 0) {{
      filtered = filtered.filter(item => {{
        return filters.every(filter => {{
          // Apply filter logic
          return true;
        }});
      }});
    }}

    // Apply sorting
    if (sortConfig) {{
      filtered.sort((a, b) => {{
        if (a[sortConfig.key] < b[sortConfig.key]) {{
          return sortConfig.direction === 'ascending' ? -1 : 1;
        }}
        if (a[sortConfig.key] > b[sortConfig.key]) {{
          return sortConfig.direction === 'ascending' ? 1 : -1;
        }}
        return 0;
      }});
    }}

    setFilteredData(filtered);
  }}, [data, filters, sortConfig]);

  const handleSort = (key: string) => {{
    let direction = 'ascending';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {{
      direction = 'descending';
    }}
    setSortConfig({{ key, direction }});
  }};

  return (
    <div className="data-table-widget">
      <div className="widget-header mb-4">
        <h3 className="text-lg font-semibold">{widget.name}</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {{{{/* Table headers would be generated based on data structure */}}}}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {{filteredData.map((row, index) => (
              <tr key={{index}} className="hover:bg-gray-50">
                {{{{/* Table cells would be generated based on data structure */}}}}
              </tr>
            ))}}
          </tbody>
        </table>
      </div>
    </div>
  );
}};'''
    
    def generate_kpi_widget(self, widget: DashboardWidget) -> str:
        """Generate a KPI (Key Performance Indicator) widget"""
        return f'''
const {self._get_widget_name(widget)} = ({{ data, target, trend }}) => {{
  const [currentValue, setCurrentValue] = React.useState(0);
  const [previousValue, setPreviousValue] = React.useState(0);
  const [trendDirection, setTrendDirection] = React.useState('neutral');

  React.useEffect(() => {{
    if (data && data.length > 0) {{
      const current = data[data.length - 1]?.value || 0;
      const previous = data[data.length - 2]?.value || 0;
      
      setCurrentValue(current);
      setPreviousValue(previous);
      
      if (current > previous) {{
        setTrendDirection('up');
      }} else if (current < previous) {{
        setTrendDirection('down');
      }} else {{
        setTrendDirection('neutral');
      }}
    }}
  }}, [data]);

  const percentageChange = previousValue !== 0 
    ? ((currentValue - previousValue) / previousValue * 100).toFixed(1)
    : 0;

  const getTrendIcon = () => {{
    switch (trendDirection) {{
      case 'up':
        return <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M3.293 9.707a1 1 0 010-1.414l6-6a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L4.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
        </svg>;
      case 'down':
        return <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 10.293a1 1 0 010 1.414l-6 6a1 1 0 01-1.414 0l-6-6a1 1 0 111.414-1.414L9 14.586V3a1 1 0 012 0v11.586l4.293-4.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>;
      default:
        return <svg className="w-4 h-4 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
        </svg>;
    }}
  }};

  return (
    <div className="kpi-widget bg-white p-6 rounded-lg shadow">
      <div className="widget-header mb-2">
        <h3 className="text-sm font-medium text-gray-500">{widget.name}</h3>
      </div>
      <div className="kpi-value">
        <div className="flex items-baseline">
          <p className="text-2xl font-semibold text-gray-900">
            {{typeof currentValue === 'number' ? currentValue.toLocaleString() : currentValue}}
          </p>
          {{target && (
            <p className="ml-2 text-sm text-gray-500">
              / {{target.toLocaleString()}}
            </p>
          )}}
        </div>
        <div className="flex items-center mt-2">
          {{getTrendIcon()}}
          <span className={{`ml-1 text-sm ${{
            trendDirection === 'up' ? 'text-green-600' :
            trendDirection === 'down' ? 'text-red-600' :
            'text-gray-600'
          }}`}}>
            {{percentageChange}}%
          </span>
          <span className="ml-1 text-sm text-gray-500">
            from last period
          </span>
        </div>
      </div>
    </div>
  );
}};'''
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize dashboard name for component"""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        if not sanitized or not sanitized[0].isupper():
            sanitized = 'Dashboard' + sanitized
        return sanitized
    
    def _get_widget_name(self, widget: DashboardWidget) -> str:
        """Get sanitized widget component name"""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', widget.name)
        return sanitized + 'Widget'
    
    def _generate_dashboard_component(self, name: str, dashboard_spec: DashboardSpecification) -> str:
        """Generate complete dashboard component code"""
        # Generate widget components
        widget_components = self._generate_widget_components(dashboard_spec.widgets)
        
        # Generate layout configuration
        layout_config = self._generate_layout_config(dashboard_spec)
        
        # Generate filters
        filters_jsx = self._generate_filters_jsx(dashboard_spec.filters)
        
        return f'''import React, {{ useState, useEffect }} from 'react';
import {{ Responsive, WidthProvider }} from 'react-grid-layout';
import {{
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
}} from 'recharts';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);

interface {name}Props {{
  data?: Record<string, any[]>;
  refreshInterval?: number;
  onWidgetClick?: (widgetId: string, data: any) => void;
  className?: string;
}}

{widget_components}

const {name}: React.FC<{name}Props> = ({{
  data = {{}},
  refreshInterval = {dashboard_spec.refresh_interval},
  onWidgetClick,
  className
}}) => {{
  const [dashboardData, setDashboardData] = useState(data);
  const [filters, setFilters] = useState({{}});
  const [isLoading, setIsLoading] = useState(false);

  // Auto-refresh functionality
  useEffect(() => {{
    if (refreshInterval > 0) {{
      const interval = setInterval(() => {{
        // Refresh dashboard data
        refreshDashboardData();
      }}, refreshInterval * 1000);

      return () => clearInterval(interval);
    }}
  }}, [refreshInterval]);

  const refreshDashboardData = async () => {{
    setIsLoading(true);
    try {{
      // Fetch updated data from API
      // This would be customized based on data sources
      console.log('Refreshing dashboard data...');
    }} catch (error) {{
      console.error('Failed to refresh dashboard data:', error);
    }} finally {{
      setIsLoading(false);
    }}
  }};

  const handleFilterChange = (filterKey: string, value: any) => {{
    setFilters(prev => ({{
      ...prev,
      [filterKey]: value
    }}));
  }};

  {layout_config}

  return (
    <div className={{`dashboard-container ${{className || ''}}`}}>
      {{{{/* Dashboard Header */}}}}
      <div className="dashboard-header mb-6">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">{dashboard_spec.name}</h1>
          <div className="dashboard-actions flex space-x-2">
            <button
              onClick={{refreshDashboardData}}
              disabled={{isLoading}}
              className="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {{isLoading ? 'Refreshing...' : 'Refresh'}}
            </button>
          </div>
        </div>
      </div>

      {{{{/* Dashboard Filters */}}}}
      {filters_jsx}

      {{{{/* Dashboard Grid */}}}}
      <ResponsiveGridLayout
        className="layout"
        layouts={{layouts}}
        breakpoints={{{{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}}}
        cols={{{{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}}}
        rowHeight={{60}}
        isDraggable={{true}}
        isResizable={{true}}
        onLayoutChange={{(layout, layouts) => {{
          // Save layout changes
          console.log('Layout changed:', layout);
        }}}}
      >
        {self._generate_widgets_jsx(dashboard_spec.widgets)}
      </ResponsiveGridLayout>
    </div>
  );
}};

export default {name};'''
    
    def _generate_widget_components(self, widgets: List[DashboardWidget]) -> str:
        """Generate individual widget components"""
        components = []
        
        for widget in widgets:
            if widget.type == 'chart':
                component = self.generate_chart_component(widget)
            elif widget.type == 'table':
                component = self.generate_data_table_widget(widget)
            elif widget.type == 'kpi':
                component = self.generate_kpi_widget(widget)
            else:
                component = self._generate_generic_widget(widget)
            
            components.append(component)
        
        return '\n\n'.join(components)
    
    def _generate_generic_widget(self, widget: DashboardWidget) -> str:
        """Generate a generic widget component"""
        widget_name = self._get_widget_name(widget)
        
        return f'''
const {widget_name} = ({{ data, onWidgetClick }}) => {{
  return (
    <div className="generic-widget bg-white p-4 rounded-lg shadow">
      <div className="widget-header mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{widget.name}</h3>
      </div>
      <div className="widget-content">
        <p className="text-gray-600">Widget type: {widget.type}</p>
        <p className="text-sm text-gray-500">Data source: {widget.data_source}</p>
        {{data && (
          <div className="mt-2">
            <pre className="text-xs bg-gray-100 p-2 rounded">
              {{JSON.stringify(data, null, 2)}}
            </pre>
          </div>
        )}}
      </div>
    </div>
  );
}};'''
    
    def _generate_recharts_component(self, widget: DashboardWidget, chart_type: str) -> str:
        """Generate a Recharts-based chart component"""
        widget_name = self._get_widget_name(widget)
        
        chart_components = {
            'line': 'LineChart',
            'bar': 'BarChart',
            'area': 'AreaChart',
            'pie': 'PieChart'
        }
        
        chart_component = chart_components.get(chart_type, 'LineChart')
        
        if chart_type == 'pie':
            return f'''
const {widget_name} = ({{ data, onWidgetClick }}) => {{
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="chart-widget bg-white p-4 rounded-lg shadow">
      <div className="widget-header mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{widget.name}</h3>
      </div>
      <div className="chart-container" style={{{{ width: '100%', height: 300 }}}}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={{data || []}}
              cx="50%"
              cy="50%"
              labelLine={{false}}
              outerRadius={{80}}
              fill="#8884d8"
              dataKey="value"
            >
              {{(data || []).map((entry, index) => (
                <Cell key={{`cell-${{index}}`}} fill={{COLORS[index % COLORS.length]}} />
              ))}}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}};'''
        else:
            return f'''
const {widget_name} = ({{ data, onWidgetClick }}) => {{
  return (
    <div className="chart-widget bg-white p-4 rounded-lg shadow">
      <div className="widget-header mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{widget.name}</h3>
      </div>
      <div className="chart-container" style={{{{ width: '100%', height: 300 }}}}>
        <ResponsiveContainer>
          <{chart_component} data={{data || []}}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            {self._generate_chart_elements(chart_type)}
          </{chart_component}>
        </ResponsiveContainer>
      </div>
    </div>
  );
}};'''
    
    def _generate_chart_elements(self, chart_type: str) -> str:
        """Generate chart elements based on type"""
        if chart_type == 'line':
            return '<Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />'
        elif chart_type == 'bar':
            return '<Bar dataKey="value" fill="#8884d8" />'
        elif chart_type == 'area':
            return '<Area type="monotone" dataKey="value" stroke="#8884d8" fill="#8884d8" />'
        else:
            return '<Line type="monotone" dataKey="value" stroke="#8884d8" />'
    
    def _generate_custom_chart_component(self, widget: DashboardWidget, chart_type: str) -> str:
        """Generate a custom chart component"""
        widget_name = self._get_widget_name(widget)
        
        return f'''
const {widget_name} = ({{ data, onWidgetClick }}) => {{
  return (
    <div className="custom-chart-widget bg-white p-4 rounded-lg shadow">
      <div className="widget-header mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{widget.name}</h3>
      </div>
      <div className="chart-container" style={{{{ width: '100%', height: 300 }}}}>
        {{/* Custom {chart_type} chart implementation would go here */}}
        <div className="flex items-center justify-center h-full bg-gray-50 rounded">
          <p className="text-gray-500">Custom {chart_type} Chart</p>
        </div>
      </div>
    </div>
  );
}};'''
    
    def _generate_layout_config(self, dashboard_spec: DashboardSpecification) -> str:
        """Generate layout configuration for grid"""
        layouts = {}
        
        # Generate layouts for different breakpoints
        for breakpoint in ['lg', 'md', 'sm', 'xs', 'xxs']:
            layout = []
            cols = {'lg': 12, 'md': 10, 'sm': 6, 'xs': 4, 'xxs': 2}[breakpoint]
            
            for i, widget in enumerate(dashboard_spec.widgets):
                width = min(widget.size.get('width', 4), cols)
                height = widget.size.get('height', 3)
                x = (i * width) % cols
                y = (i * width) // cols * height
                
                layout.append({
                    'i': widget.id,
                    'x': x,
                    'y': y,
                    'w': width,
                    'h': height
                })
            
            layouts[breakpoint] = layout
        
        return f'''
  const layouts = {str(layouts).replace("'", '"')};'''
    
    def _generate_filters_jsx(self, filters: List[str]) -> str:
        """Generate filters JSX"""
        if not filters:
            return ""
        
        filters_jsx = []
        for filter_name in filters:
            filter_jsx = f'''
            <div className="filter-item">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {filter_name.replace('_', ' ').title()}
              </label>
              <select
                onChange={{(e) => handleFilterChange('{filter_name}', e.target.value)}}
                className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">All</option>
                {{/* Filter options would be populated dynamically */}}
              </select>
            </div>'''
            filters_jsx.append(filter_jsx)
        
        filters_str = '\n'.join(filters_jsx)
        
        return f'''
      {{{{/* Dashboard Filters */}}}}
      <div className="dashboard-filters mb-6 p-4 bg-gray-50 rounded-lg">
        <div className="flex flex-wrap gap-4">
          {filters_str}
        </div>
      </div>'''
    
    def _generate_widgets_jsx(self, widgets: List[DashboardWidget]) -> str:
        """Generate widgets JSX for grid layout"""
        widgets_jsx = []
        
        for widget in widgets:
            widget_name = self._get_widget_name(widget)
            widget_jsx = f'''
        <div key="{widget.id}">
          <{widget_name}
            data={{dashboardData['{widget.data_source}']}}
            onWidgetClick={{(data) => onWidgetClick?.('{widget.id}', data)}}
          />
        </div>'''
            widgets_jsx.append(widget_jsx)
        
        return '\n'.join(widgets_jsx)
    
    def _generate_dashboard_properties(self, dashboard_spec: DashboardSpecification) -> List[ComponentProperty]:
        """Generate dashboard component properties"""
        return [
            ComponentProperty(
                name="data",
                type="Record<string, any[]>",
                required=False,
                description="Data for dashboard widgets"
            ),
            ComponentProperty(
                name="refreshInterval",
                type="number",
                required=False,
                default_value=dashboard_spec.refresh_interval,
                description="Auto-refresh interval in seconds"
            ),
            ComponentProperty(
                name="onWidgetClick",
                type="function",
                required=False,
                description="Callback when widget is clicked"
            )
        ]
    
    def _generate_dashboard_styles(self, dashboard_spec: DashboardSpecification) -> ComponentStyle:
        """Generate dashboard styling"""
        classes = ["dashboard-container"]
        
        if dashboard_spec.responsive:
            classes.extend(["responsive", "w-full"])
        
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
    
    def _generate_dashboard_accessibility(self, dashboard_spec: DashboardSpecification) -> ComponentAccessibility:
        """Generate dashboard accessibility features"""
        return ComponentAccessibility(
            aria_labels={
                "dashboard": f"Dashboard: {dashboard_spec.name}",
                "refresh": "Refresh dashboard data",
                "filters": "Dashboard filters"
            },
            keyboard_navigation=True,
            screen_reader_support=True,
            color_contrast_ratio=4.5,
            focus_management=True
        )
    
    def _get_dashboard_dependencies(self, dashboard_spec: DashboardSpecification) -> List[str]:
        """Get dashboard-specific dependencies"""
        deps = [
            "react",
            "@types/react",
            "react-grid-layout",
            "@types/react-grid-layout",
            "recharts"
        ]
        
        if self.style_framework == StyleFramework.TAILWIND:
            deps.append("tailwindcss")
        
        return deps