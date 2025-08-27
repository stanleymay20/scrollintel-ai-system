/**
 * Advanced Dashboard Builder Component
 * 
 * Provides comprehensive drag-and-drop dashboard customization with real-time preview
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Alert, AlertDescription } from '../ui/alert';
import { 
  Settings, Plus, Trash2, Copy, Move, Palette, Layout, BarChart3, PieChart, 
  LineChart, Table, Gauge, Eye, Save, Undo, Redo, Grid, Smartphone, 
  Monitor, Tablet, RefreshCw, Download, Upload, Share2, Lock, Unlock
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';

// Enhanced Types
interface WidgetConfig {
  id: string;
  type: string;
  title: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  data_source: string;
  visualization_config: {
    chart_type?: string;
    color_scheme?: string;
    show_legend?: boolean;
    show_grid?: boolean;
    animation_enabled?: boolean;
    refresh_interval?: number;
    filters?: Array<{
      field: string;
      operator: string;
      value: any;
    }>;
  };
  responsive_config?: {
    mobile?: Partial<WidgetConfig['position']>;
    tablet?: Partial<WidgetConfig['position']>;
    desktop?: Partial<WidgetConfig['position']>;
  };
  permissions?: {
    view_roles: string[];
    edit_roles: string[];
  };
}

interface DashboardTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  widgets: WidgetConfig[];
  layout_config: {
    grid_size: number;
    row_height: number;
    margin: [number, number];
    responsive_breakpoints: {
      mobile: number;
      tablet: number;
      desktop: number;
    };
    auto_layout?: boolean;
  };
  default_filters: Array<{
    field: string;
    operator: string;
    value: any;
  }>;
  theme: {
    primary_color: string;
    secondary_color: string;
    background_color: string;
    text_color: string;
  };
  version: number;
  created_at: string;
  updated_at: string;
  created_by: string;
  is_public: boolean;
  tags: string[];
}

interface DragItem {
  type: string;
  id: string;
  widgetType?: string;
  widget?: WidgetConfig;
}

// Enhanced Widget Types with Categories
const WIDGET_CATEGORIES = {
  metrics: {
    name: 'Metrics & KPIs',
    widgets: [
      { type: 'metric_card', name: 'Metric Card', icon: Gauge, description: 'Single metric display' },
      { type: 'kpi_grid', name: 'KPI Grid', icon: Layout, description: 'Multiple KPIs in grid' },
      { type: 'progress_bar', name: 'Progress Bar', icon: BarChart3, description: 'Progress indicator' },
      { type: 'gauge_chart', name: 'Gauge Chart', icon: Gauge, description: 'Circular gauge' },
    ]
  },
  charts: {
    name: 'Charts & Visualizations',
    widgets: [
      { type: 'line_chart', name: 'Line Chart', icon: LineChart, description: 'Time series data' },
      { type: 'bar_chart', name: 'Bar Chart', icon: BarChart3, description: 'Categorical data' },
      { type: 'pie_chart', name: 'Pie Chart', icon: PieChart, description: 'Proportional data' },
      { type: 'area_chart', name: 'Area Chart', icon: LineChart, description: 'Filled line chart' },
      { type: 'scatter_plot', name: 'Scatter Plot', icon: BarChart3, description: 'Correlation data' },
      { type: 'heatmap', name: 'Heatmap', icon: Grid, description: 'Matrix visualization' },
    ]
  },
  tables: {
    name: 'Tables & Lists',
    widgets: [
      { type: 'data_table', name: 'Data Table', icon: Table, description: 'Tabular data display' },
      { type: 'summary_table', name: 'Summary Table', icon: Table, description: 'Aggregated data' },
      { type: 'ranking_list', name: 'Ranking List', icon: Table, description: 'Ordered list' },
    ]
  },
  advanced: {
    name: 'Advanced Widgets',
    widgets: [
      { type: 'roi_calculator', name: 'ROI Calculator', icon: Gauge, description: 'Interactive ROI analysis' },
      { type: 'forecast_chart', name: 'Forecast Chart', icon: LineChart, description: 'Predictive analytics' },
      { type: 'alert_panel', name: 'Alert Panel', icon: Settings, description: 'System alerts' },
      { type: 'custom_html', name: 'Custom HTML', icon: Layout, description: 'Custom content' },
    ]
  }
};

// Data Source Options
const DATA_SOURCES = [
  { id: 'roi_calculator', name: 'ROI Calculator', category: 'financial' },
  { id: 'performance_monitor', name: 'Performance Monitor', category: 'operational' },
  { id: 'cost_tracker', name: 'Cost Tracker', category: 'financial' },
  { id: 'deployment_tracker', name: 'Deployment Tracker', category: 'operational' },
  { id: 'user_analytics', name: 'User Analytics', category: 'behavioral' },
  { id: 'system_metrics', name: 'System Metrics', category: 'technical' },
  { id: 'business_intelligence', name: 'Business Intelligence', category: 'strategic' },
];

// Responsive Breakpoints
const BREAKPOINTS = {
  mobile: 768,
  tablet: 1024,
  desktop: 1200,
};

// Enhanced Widget Palette Component
const WidgetPalette: React.FC<{
  onWidgetSelect: (widgetType: string) => void;
}> = ({ onWidgetSelect }) => {
  const [selectedCategory, setSelectedCategory] = useState('metrics');
  const [searchTerm, setSearchTerm] = useState('');

  const categories = Object.keys(WIDGET_CATEGORIES);
  const currentCategory = WIDGET_CATEGORIES[selectedCategory as keyof typeof WIDGET_CATEGORIES];

  const filteredWidgets = currentCategory.widgets.filter(widget =>
    widget.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    widget.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const PaletteWidget: React.FC<{ widget: any }> = ({ widget }) => {
    const [{ isDragging }, drag] = useDrag({
      type: 'new-widget',
      item: { type: 'new-widget', widgetType: widget.type },
      collect: (monitor) => ({
        isDragging: monitor.isDragging(),
      }),
    });

    const Icon = widget.icon;

    return (
      <Card
        ref={drag}
        className={`cursor-move transition-all duration-200 ${
          isDragging ? 'opacity-50' : 'opacity-100'
        } hover:shadow-md hover:scale-105`}
        onClick={() => onWidgetSelect(widget.type)}
      >
        <CardContent className="p-3">
          <div className="flex items-center gap-2 mb-2">
            <Icon className="h-5 w-5 text-blue-600" />
            <div className="text-sm font-medium">{widget.name}</div>
          </div>
          <div className="text-xs text-gray-500">{widget.description}</div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-4">
      {/* Search */}
      <div>
        <input
          type="text"
          placeholder="Search widgets..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full px-3 py-2 border rounded-md text-sm"
        />
      </div>

      {/* Category Tabs */}
      <div className="space-y-2">
        {categories.map((category) => (
          <Button
            key={category}
            size="sm"
            variant={selectedCategory === category ? 'default' : 'outline'}
            onClick={() => setSelectedCategory(category)}
            className="w-full justify-start"
          >
            {WIDGET_CATEGORIES[category as keyof typeof WIDGET_CATEGORIES].name}
          </Button>
        ))}
      </div>
      
      {/* Widget Grid */}
      <div className="space-y-2">
        {filteredWidgets.map((widget) => (
          <PaletteWidget key={widget.type} widget={widget} />
        ))}
      </div>
    </div>
  );
};

// Enhanced Drop Zone with Grid Overlay
const DropZone: React.FC<{
  template: DashboardTemplate;
  currentBreakpoint: string;
  onWidgetAdd: (widgetType: string, position: { x: number; y: number }) => void;
  onWidgetMove: (widgetId: string, position: { x: number; y: number }) => void;
  showGrid: boolean;
  children: React.ReactNode;
}> = ({ template, currentBreakpoint, onWidgetAdd, onWidgetMove, showGrid, children }) => {
  const [{ isOver }, drop] = useDrop({
    accept: ['new-widget', 'widget'],
    drop: (item: DragItem, monitor) => {
      const offset = monitor.getClientOffset();
      const dropZoneRect = dropZoneRef.current?.getBoundingClientRect();
      
      if (!offset || !dropZoneRect) return;

      const cellWidth = dropZoneRect.width / template.layout_config.grid_size;
      const x = Math.floor((offset.x - dropZoneRect.left) / cellWidth);
      const y = Math.floor((offset.y - dropZoneRect.top) / template.layout_config.row_height);

      if (item.type === 'new-widget' && item.widgetType) {
        onWidgetAdd(item.widgetType, { x: Math.max(0, x), y: Math.max(0, y) });
      } else if (item.type === 'widget' && item.widget) {
        onWidgetMove(item.widget.id, { x: Math.max(0, x), y: Math.max(0, y) });
      }
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
    }),
  });

  const dropZoneRef = useRef<HTMLDivElement>(null);

  return (
    <div className="relative">
      {/* Grid Overlay */}
      {showGrid && (
        <div
          className="absolute inset-0 pointer-events-none z-10"
          style={{
            backgroundImage: `
              linear-gradient(to right, rgba(0,0,0,0.1) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(0,0,0,0.1) 1px, transparent 1px)
            `,
            backgroundSize: `${100 / template.layout_config.grid_size}% ${template.layout_config.row_height}px`,
          }}
        />
      )}

      {/* Drop Zone */}
      <div
        ref={(node) => {
          drop(node);
          dropZoneRef.current = node;
        }}
        className={`min-h-[600px] border-2 border-dashed transition-colors rounded-lg p-4 ${
          isOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }`}
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${template.layout_config.grid_size}, 1fr)`,
          gap: `${template.layout_config.margin[0]}px`,
          gridAutoRows: `${template.layout_config.row_height}px`,
        }}
      >
        {children}
      </div>
    </div>
  );
};

// Enhanced Draggable Widget with Responsive Indicators
const DraggableWidget: React.FC<{
  widget: WidgetConfig;
  currentBreakpoint: string;
  onSelect: (widget: WidgetConfig) => void;
  onDelete: (widgetId: string) => void;
  onDuplicate: (widget: WidgetConfig) => void;
  isSelected: boolean;
  liveData?: any;
}> = ({ widget, currentBreakpoint, onSelect, onDelete, onDuplicate, isSelected, liveData }) => {
  const [{ isDragging }, drag] = useDrag({
    type: 'widget',
    item: { type: 'widget', id: widget.id, widget },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  const getWidgetIcon = (type: string) => {
    for (const category of Object.values(WIDGET_CATEGORIES)) {
      const widgetType = category.widgets.find(w => w.type === type);
      if (widgetType) return widgetType.icon;
    }
    return BarChart3;
  };

  const Icon = getWidgetIcon(widget.type);

  // Get position for current breakpoint
  const position = widget.responsive_config?.[currentBreakpoint as keyof typeof widget.responsive_config] || widget.position;

  return (
    <Card
      ref={drag}
      className={`cursor-move transition-all duration-200 ${
        isDragging ? 'opacity-50 scale-95' : 'opacity-100 scale-100'
      } ${isSelected ? 'ring-2 ring-blue-500' : ''} hover:shadow-md relative`}
      onClick={() => onSelect(widget)}
      style={{
        gridColumn: `span ${position.width}`,
        gridRow: `span ${position.height}`,
      }}
    >
      {/* Live Data Indicator */}
      {liveData && (
        <div className="absolute top-2 right-2 z-10">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        </div>
      )}

      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Icon className="h-4 w-4" />
            <CardTitle className="text-sm">{widget.title}</CardTitle>
          </div>
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={(e) => {
                e.stopPropagation();
                onDuplicate(widget);
              }}
            >
              <Copy className="h-3 w-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(widget.id);
              }}
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-xs text-muted-foreground space-y-1">
          <Badge variant="secondary">{widget.type}</Badge>
          <div>Source: {widget.data_source}</div>
          {widget.visualization_config.refresh_interval && (
            <div>Refresh: {widget.visualization_config.refresh_interval}s</div>
          )}
        </div>
        
        {/* Live Preview Content */}
        {liveData && (
          <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
            <div className="font-medium">Live Preview</div>
            <div className="text-gray-600">
              {typeof liveData === 'object' ? JSON.stringify(liveData).slice(0, 50) + '...' : liveData}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Enhanced Widget Properties Panel
const WidgetPropertiesPanel: React.FC<{
  widget: WidgetConfig | null;
  currentBreakpoint: string;
  onUpdate: (widgetId: string, updates: Partial<WidgetConfig>) => void;
}> = ({ widget, currentBreakpoint, onUpdate }) => {
  const [localWidget, setLocalWidget] = useState<WidgetConfig | null>(null);

  useEffect(() => {
    setLocalWidget(widget);
  }, [widget]);

  if (!localWidget) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        <Settings className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <div>Select a widget to edit its properties</div>
      </div>
    );
  }

  const handleUpdate = (field: string, value: any) => {
    const updated = { ...localWidget };
    
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      updated[parent as keyof WidgetConfig] = {
        ...(updated[parent as keyof WidgetConfig] as any),
        [child]: value
      };
    } else {
      updated[field as keyof WidgetConfig] = value;
    }
    
    setLocalWidget(updated);
    onUpdate(localWidget.id, updated);
  };

  const currentPosition = localWidget.responsive_config?.[currentBreakpoint as keyof typeof localWidget.responsive_config] || localWidget.position;

  return (
    <div className="space-y-6 p-4 max-h-screen overflow-y-auto">
      {/* Basic Properties */}
      <div className="space-y-4">
        <h3 className="font-semibold text-lg">Widget Properties</h3>
        
        <div>
          <label className="text-sm font-medium">Title</label>
          <input
            type="text"
            value={localWidget.title}
            onChange={(e) => handleUpdate('title', e.target.value)}
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>

        <div>
          <label className="text-sm font-medium">Data Source</label>
          <select
            value={localWidget.data_source}
            onChange={(e) => handleUpdate('data_source', e.target.value)}
            className="w-full mt-1 px-3 py-2 border rounded-md"
          >
            {DATA_SOURCES.map(source => (
              <option key={source.id} value={source.id}>
                {source.name} ({source.category})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Layout Properties */}
      <div className="space-y-4">
        <h4 className="font-medium">Layout ({currentBreakpoint})</h4>
        
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-sm font-medium">Width</label>
            <input
              type="number"
              min="1"
              max="12"
              value={currentPosition.width}
              onChange={(e) => {
                const newPosition = { ...currentPosition, width: parseInt(e.target.value) };
                if (currentBreakpoint === 'desktop') {
                  handleUpdate('position', newPosition);
                } else {
                  handleUpdate(`responsive_config.${currentBreakpoint}`, newPosition);
                }
              }}
              className="w-full mt-1 px-3 py-2 border rounded-md"
            />
          </div>
          <div>
            <label className="text-sm font-medium">Height</label>
            <input
              type="number"
              min="1"
              value={currentPosition.height}
              onChange={(e) => {
                const newPosition = { ...currentPosition, height: parseInt(e.target.value) };
                if (currentBreakpoint === 'desktop') {
                  handleUpdate('position', newPosition);
                } else {
                  handleUpdate(`responsive_config.${currentBreakpoint}`, newPosition);
                }
              }}
              className="w-full mt-1 px-3 py-2 border rounded-md"
            />
          </div>
        </div>
      </div>

      {/* Visualization Config */}
      <div className="space-y-4">
        <h4 className="font-medium">Visualization</h4>
        
        {localWidget.type.includes('chart') && (
          <div>
            <label className="text-sm font-medium">Chart Type</label>
            <select
              value={localWidget.visualization_config.chart_type || 'line'}
              onChange={(e) => handleUpdate('visualization_config.chart_type', e.target.value)}
              className="w-full mt-1 px-3 py-2 border rounded-md"
            >
              <option value="line">Line Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="area">Area Chart</option>
              <option value="pie">Pie Chart</option>
            </select>
          </div>
        )}

        <div>
          <label className="text-sm font-medium">Color Scheme</label>
          <select
            value={localWidget.visualization_config.color_scheme || 'default'}
            onChange={(e) => handleUpdate('visualization_config.color_scheme', e.target.value)}
            className="w-full mt-1 px-3 py-2 border rounded-md"
          >
            <option value="default">Default</option>
            <option value="blue">Blue</option>
            <option value="green">Green</option>
            <option value="red">Red</option>
            <option value="purple">Purple</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="show_legend"
            checked={localWidget.visualization_config.show_legend !== false}
            onChange={(e) => handleUpdate('visualization_config.show_legend', e.target.checked)}
          />
          <label htmlFor="show_legend" className="text-sm">Show Legend</label>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="animation_enabled"
            checked={localWidget.visualization_config.animation_enabled !== false}
            onChange={(e) => handleUpdate('visualization_config.animation_enabled', e.target.checked)}
          />
          <label htmlFor="animation_enabled" className="text-sm">Enable Animations</label>
        </div>

        <div>
          <label className="text-sm font-medium">Refresh Interval (seconds)</label>
          <input
            type="number"
            min="30"
            value={localWidget.visualization_config.refresh_interval || 300}
            onChange={(e) => handleUpdate('visualization_config.refresh_interval', parseInt(e.target.value))}
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>
      </div>

      {/* Permissions */}
      <div className="space-y-4">
        <h4 className="font-medium">Permissions</h4>
        
        <div>
          <label className="text-sm font-medium">View Roles</label>
          <input
            type="text"
            placeholder="admin,manager,analyst"
            value={localWidget.permissions?.view_roles?.join(',') || ''}
            onChange={(e) => handleUpdate('permissions.view_roles', e.target.value.split(',').filter(Boolean))}
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>

        <div>
          <label className="text-sm font-medium">Edit Roles</label>
          <input
            type="text"
            placeholder="admin,manager"
            value={localWidget.permissions?.edit_roles?.join(',') || ''}
            onChange={(e) => handleUpdate('permissions.edit_roles', e.target.value.split(',').filter(Boolean))}
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>
      </div>
    </div>
  );
};

// Template Selection Component
const TemplateSelector: React.FC<{
  templates: DashboardTemplate[];
  onSelect: (template: DashboardTemplate) => void;
  onCreateNew: () => void;
}> = ({ templates, onSelect, onCreateNew }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const categories = ['all', 'executive', 'operational', 'financial', 'technical'];
  
  const filteredTemplates = templates.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Select Template</h3>
        <Button onClick={onCreateNew}>
          <Plus className="h-4 w-4 mr-2" />
          Create New
        </Button>
      </div>

      <div>
        <input
          type="text"
          placeholder="Search templates..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full px-3 py-2 border rounded-md"
        />
      </div>

      <div className="flex gap-2 flex-wrap">
        {categories.map(category => (
          <Button
            key={category}
            size="sm"
            variant={selectedCategory === category ? 'default' : 'outline'}
            onClick={() => setSelectedCategory(category)}
          >
            {category.charAt(0).toUpperCase() + category.slice(1)}
          </Button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredTemplates.map(template => (
          <Card
            key={template.id}
            className="cursor-pointer hover:shadow-md transition-shadow"
            onClick={() => onSelect(template)}
          >
            <CardHeader>
              <div className="flex justify-between items-start">
                <CardTitle className="text-base">{template.name}</CardTitle>
                <Badge variant={template.is_public ? 'default' : 'secondary'}>
                  {template.is_public ? 'Public' : 'Private'}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 mb-2">{template.description}</p>
              <div className="flex justify-between text-xs text-gray-500">
                <span>{template.widgets.length} widgets</span>
                <span>v{template.version}</span>
              </div>
              <div className="flex gap-1 mt-2">
                {template.tags.slice(0, 3).map(tag => (
                  <Badge key={tag} variant="outline" className="text-xs">
                    {tag}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export { WidgetPalette, DropZone, DraggableWidget, WidgetPropertiesPanel, TemplateSelector };
export type { WidgetConfig, DashboardTemplate, DragItem };