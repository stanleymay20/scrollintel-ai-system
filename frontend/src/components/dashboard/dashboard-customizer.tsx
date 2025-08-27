/**
 * Dashboard Customizer Component
 * 
 * Provides drag-and-drop interface for customizing dashboard templates
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { 
  Settings, 
  Plus, 
  Trash2, 
  Copy, 
  Move, 
  Palette,
  Layout,
  BarChart3,
  PieChart,
  LineChart,
  Table,
  Gauge
} from 'lucide-react';

// Types
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
  visualization_config: Record<string, any>;
  filters: Array<Record<string, any>>;
  refresh_interval?: number;
}

interface DashboardTemplate {
  id: string;
  name: string;
  description: string;
  widgets: WidgetConfig[];
  layout_config: {
    grid_size: number;
    row_height: number;
    margin: [number, number];
    responsive_breakpoints?: Record<string, number>;
  };
  default_filters: Array<Record<string, any>>;
}

interface DragItem {
  type: string;
  id: string;
  widgetType?: string;
  widget?: WidgetConfig;
}

// Widget type definitions
const WIDGET_TYPES = [
  { type: 'metric_card', name: 'Metric Card', icon: Gauge, category: 'metrics' },
  { type: 'line_chart', name: 'Line Chart', icon: LineChart, category: 'charts' },
  { type: 'bar_chart', name: 'Bar Chart', icon: BarChart3, category: 'charts' },
  { type: 'pie_chart', name: 'Pie Chart', icon: PieChart, category: 'charts' },
  { type: 'data_table', name: 'Data Table', icon: Table, category: 'tables' },
  { type: 'kpi_grid', name: 'KPI Grid', icon: Layout, category: 'metrics' },
];

// Draggable Widget Component
const DraggableWidget: React.FC<{
  widget: WidgetConfig;
  onSelect: (widget: WidgetConfig) => void;
  onDelete: (widgetId: string) => void;
  onDuplicate: (widget: WidgetConfig) => void;
  isSelected: boolean;
}> = ({ widget, onSelect, onDelete, onDuplicate, isSelected }) => {
  const [{ isDragging }, drag] = useDrag({
    type: 'widget',
    item: { type: 'widget', id: widget.id, widget },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  const getWidgetIcon = (type: string) => {
    const widgetType = WIDGET_TYPES.find(wt => wt.type === type);
    return widgetType?.icon || BarChart3;
  };

  const Icon = getWidgetIcon(widget.type);

  return (
    <Card
      ref={drag}
      className={`cursor-move transition-all duration-200 ${
        isDragging ? 'opacity-50 scale-95' : 'opacity-100 scale-100'
      } ${isSelected ? 'ring-2 ring-blue-500' : ''} hover:shadow-md`}
      onClick={() => onSelect(widget)}
      style={{
        gridColumn: `span ${widget.position.width}`,
        gridRow: `span ${widget.position.height}`,
      }}
    >
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
        <div className="text-xs text-muted-foreground">
          <Badge variant="secondary">{widget.type}</Badge>
          <div className="mt-1">Source: {widget.data_source}</div>
        </div>
      </CardContent>
    </Card>
  );
};

// Widget Palette Component
const WidgetPalette: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState('all');

  const categories = [
    { id: 'all', name: 'All Widgets' },
    { id: 'metrics', name: 'Metrics' },
    { id: 'charts', name: 'Charts' },
    { id: 'tables', name: 'Tables' },
  ];

  const filteredWidgets = selectedCategory === 'all' 
    ? WIDGET_TYPES 
    : WIDGET_TYPES.filter(w => w.category === selectedCategory);

  const PaletteWidget: React.FC<{ widgetType: typeof WIDGET_TYPES[0] }> = ({ widgetType }) => {
    const [{ isDragging }, drag] = useDrag({
      type: 'new-widget',
      item: { type: 'new-widget', widgetType: widgetType.type },
      collect: (monitor) => ({
        isDragging: monitor.isDragging(),
      }),
    });

    const Icon = widgetType.icon;

    return (
      <Card
        ref={drag}
        className={`cursor-move transition-all duration-200 ${
          isDragging ? 'opacity-50' : 'opacity-100'
        } hover:shadow-md`}
      >
        <CardContent className="p-4 text-center">
          <Icon className="h-8 w-8 mx-auto mb-2" />
          <div className="text-sm font-medium">{widgetType.name}</div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        {categories.map((category) => (
          <Button
            key={category.id}
            size="sm"
            variant={selectedCategory === category.id ? 'default' : 'outline'}
            onClick={() => setSelectedCategory(category.id)}
          >
            {category.name}
          </Button>
        ))}
      </div>
      
      <div className="grid grid-cols-2 gap-2">
        {filteredWidgets.map((widgetType) => (
          <PaletteWidget key={widgetType.type} widgetType={widgetType} />
        ))}
      </div>
    </div>
  );
};

// Drop Zone Component
const DropZone: React.FC<{
  template: DashboardTemplate;
  onWidgetAdd: (widgetType: string, position: { x: number; y: number }) => void;
  onWidgetMove: (widgetId: string, position: { x: number; y: number }) => void;
  onWidgetResize: (widgetId: string, size: { width: number; height: number }) => void;
  children: React.ReactNode;
}> = ({ template, onWidgetAdd, onWidgetMove, onWidgetResize, children }) => {
  const [{ isOver }, drop] = useDrop({
    accept: ['new-widget', 'widget'],
    drop: (item: DragItem, monitor) => {
      const offset = monitor.getClientOffset();
      const dropZoneRect = dropZoneRef.current?.getBoundingClientRect();
      
      if (!offset || !dropZoneRect) return;

      const x = Math.floor((offset.x - dropZoneRect.left) / (dropZoneRect.width / template.layout_config.grid_size));
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
    <div
      ref={(node) => {
        drop(node);
        dropZoneRef.current = node;
      }}
      className={`min-h-[600px] border-2 border-dashed transition-colors ${
        isOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
      } rounded-lg p-4`}
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${template.layout_config.grid_size}, 1fr)`,
        gap: `${template.layout_config.margin[0]}px`,
        gridAutoRows: `${template.layout_config.row_height}px`,
      }}
    >
      {children}
    </div>
  );
};

// Widget Properties Panel
const WidgetPropertiesPanel: React.FC<{
  widget: WidgetConfig | null;
  onUpdate: (widgetId: string, updates: Partial<WidgetConfig>) => void;
}> = ({ widget, onUpdate }) => {
  const [localWidget, setLocalWidget] = useState<WidgetConfig | null>(null);

  useEffect(() => {
    setLocalWidget(widget);
  }, [widget]);

  if (!localWidget) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        Select a widget to edit its properties
      </div>
    );
  }

  const handleUpdate = (field: string, value: any) => {
    const updated = { ...localWidget, [field]: value };
    setLocalWidget(updated);
    onUpdate(localWidget.id, { [field]: value });
  };

  return (
    <div className="space-y-4 p-4">
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
          <option value="roi_calculator">ROI Calculator</option>
          <option value="performance_monitor">Performance Monitor</option>
          <option value="cost_tracker">Cost Tracker</option>
          <option value="deployment_tracker">Deployment Tracker</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-sm font-medium">Width</label>
          <input
            type="number"
            min="1"
            max="12"
            value={localWidget.position.width}
            onChange={(e) => handleUpdate('position', {
              ...localWidget.position,
              width: parseInt(e.target.value)
            })}
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>
        <div>
          <label className="text-sm font-medium">Height</label>
          <input
            type="number"
            min="1"
            value={localWidget.position.height}
            onChange={(e) => handleUpdate('position', {
              ...localWidget.position,
              height: parseInt(e.target.value)
            })}
            className="w-full mt-1 px-3 py-2 border rounded-md"
          />
        </div>
      </div>

      <div>
        <label className="text-sm font-medium">Refresh Interval (seconds)</label>
        <input
          type="number"
          min="30"
          value={localWidget.refresh_interval || 300}
          onChange={(e) => handleUpdate('refresh_interval', parseInt(e.target.value))}
          className="w-full mt-1 px-3 py-2 border rounded-md"
        />
      </div>
    </div>
  );
};

// Main Dashboard Customizer Component
export const DashboardCustomizer: React.FC<{
  template: DashboardTemplate;
  onTemplateUpdate: (template: DashboardTemplate) => void;
  onSave: () => void;
  onPreview: () => void;
}> = ({ template, onTemplateUpdate, onSave, onPreview }) => {
  const [selectedWidget, setSelectedWidget] = useState<WidgetConfig | null>(null);
  const [activeTab, setActiveTab] = useState('design');

  const generateWidgetId = () => `widget_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const handleWidgetAdd = useCallback((widgetType: string, position: { x: number; y: number }) => {
    const newWidget: WidgetConfig = {
      id: generateWidgetId(),
      type: widgetType,
      title: `New ${widgetType.replace('_', ' ')}`,
      position: {
        x: position.x,
        y: position.y,
        width: 4,
        height: 3,
      },
      data_source: 'roi_calculator',
      visualization_config: {},
      filters: [],
      refresh_interval: 300,
    };

    const updatedTemplate = {
      ...template,
      widgets: [...template.widgets, newWidget],
    };

    onTemplateUpdate(updatedTemplate);
    setSelectedWidget(newWidget);
  }, [template, onTemplateUpdate]);

  const handleWidgetMove = useCallback((widgetId: string, position: { x: number; y: number }) => {
    const updatedWidgets = template.widgets.map(widget =>
      widget.id === widgetId
        ? { ...widget, position: { ...widget.position, ...position } }
        : widget
    );

    onTemplateUpdate({
      ...template,
      widgets: updatedWidgets,
    });
  }, [template, onTemplateUpdate]);

  const handleWidgetUpdate = useCallback((widgetId: string, updates: Partial<WidgetConfig>) => {
    const updatedWidgets = template.widgets.map(widget =>
      widget.id === widgetId ? { ...widget, ...updates } : widget
    );

    onTemplateUpdate({
      ...template,
      widgets: updatedWidgets,
    });
  }, [template, onTemplateUpdate]);

  const handleWidgetDelete = useCallback((widgetId: string) => {
    const updatedWidgets = template.widgets.filter(widget => widget.id !== widgetId);
    
    onTemplateUpdate({
      ...template,
      widgets: updatedWidgets,
    });

    if (selectedWidget?.id === widgetId) {
      setSelectedWidget(null);
    }
  }, [template, onTemplateUpdate, selectedWidget]);

  const handleWidgetDuplicate = useCallback((widget: WidgetConfig) => {
    const duplicatedWidget: WidgetConfig = {
      ...widget,
      id: generateWidgetId(),
      title: `${widget.title} (Copy)`,
      position: {
        ...widget.position,
        x: Math.min(widget.position.x + 1, template.layout_config.grid_size - widget.position.width),
        y: widget.position.y + 1,
      },
    };

    const updatedTemplate = {
      ...template,
      widgets: [...template.widgets, duplicatedWidget],
    };

    onTemplateUpdate(updatedTemplate);
  }, [template, onTemplateUpdate]);

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="flex h-screen bg-gray-50">
        {/* Left Sidebar - Widget Palette */}
        <div className="w-64 bg-white border-r p-4 overflow-y-auto">
          <h3 className="font-semibold mb-4">Widget Palette</h3>
          <WidgetPalette />
        </div>

        {/* Main Canvas */}
        <div className="flex-1 p-4 overflow-auto">
          <div className="mb-4 flex justify-between items-center">
            <h2 className="text-xl font-semibold">{template.name}</h2>
            <div className="flex gap-2">
              <Button variant="outline" onClick={onPreview}>
                Preview
              </Button>
              <Button onClick={onSave}>
                Save Template
              </Button>
            </div>
          </div>

          <DropZone
            template={template}
            onWidgetAdd={handleWidgetAdd}
            onWidgetMove={handleWidgetMove}
            onWidgetResize={(widgetId, size) => {
              handleWidgetUpdate(widgetId, {
                position: {
                  ...template.widgets.find(w => w.id === widgetId)?.position!,
                  ...size
                }
              });
            }}
          >
            {template.widgets.map((widget) => (
              <DraggableWidget
                key={widget.id}
                widget={widget}
                onSelect={setSelectedWidget}
                onDelete={handleWidgetDelete}
                onDuplicate={handleWidgetDuplicate}
                isSelected={selectedWidget?.id === widget.id}
              />
            ))}
          </DropZone>
        </div>

        {/* Right Sidebar - Properties Panel */}
        <div className="w-80 bg-white border-l">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="design">Design</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
            </TabsList>
            
            <TabsContent value="design">
              <WidgetPropertiesPanel
                widget={selectedWidget}
                onUpdate={handleWidgetUpdate}
              />
            </TabsContent>
            
            <TabsContent value="settings" className="p-4">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Template Name</label>
                  <input
                    type="text"
                    value={template.name}
                    onChange={(e) => onTemplateUpdate({
                      ...template,
                      name: e.target.value
                    })}
                    className="w-full mt-1 px-3 py-2 border rounded-md"
                  />
                </div>
                
                <div>
                  <label className="text-sm font-medium">Description</label>
                  <textarea
                    value={template.description}
                    onChange={(e) => onTemplateUpdate({
                      ...template,
                      description: e.target.value
                    })}
                    className="w-full mt-1 px-3 py-2 border rounded-md"
                    rows={3}
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">Grid Size</label>
                  <select
                    value={template.layout_config.grid_size}
                    onChange={(e) => onTemplateUpdate({
                      ...template,
                      layout_config: {
                        ...template.layout_config,
                        grid_size: parseInt(e.target.value)
                      }
                    })}
                    className="w-full mt-1 px-3 py-2 border rounded-md"
                  >
                    <option value="12">12 Columns</option>
                    <option value="16">16 Columns</option>
                    <option value="24">24 Columns</option>
                  </select>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </DndProvider>
  );
};