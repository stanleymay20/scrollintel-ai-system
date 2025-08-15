'use client';

import React, { useState, useCallback, useRef } from 'react';
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Settings, 
  Save, 
  Undo, 
  Redo, 
  Copy, 
  Trash2, 
  Plus,
  Grid,
  Eye,
  Share2
} from 'lucide-react';

interface WidgetPosition {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface DashboardWidget {
  id: string;
  type: string;
  title: string;
  position: WidgetPosition;
  config: Record<string, any>;
  dataSource: string;
}

interface DashboardTemplate {
  id: string;
  name: string;
  description: string;
  widgets: DashboardWidget[];
  layoutConfig: {
    gridSize: number;
    rowHeight: number;
  };
}

interface DragItem {
  type: string;
  id: string;
  widgetType?: string;
}

const WIDGET_TYPES = [
  { type: 'metrics_grid', name: 'Metrics Grid', icon: '📊' },
  { type: 'line_chart', name: 'Line Chart', icon: '📈' },
  { type: 'bar_chart', name: 'Bar Chart', icon: '📊' },
  { type: 'pie_chart', name: 'Pie Chart', icon: '🥧' },
  { type: 'gauge_chart', name: 'Gauge Chart', icon: '⏲️' },
  { type: 'status_grid', name: 'Status Grid', icon: '🟢' },
  { type: 'alert_list', name: 'Alert List', icon: '🚨' },
  { type: 'kanban_board', name: 'Kanban Board', icon: '📋' },
  { type: 'area_chart', name: 'Area Chart', icon: '📊' },
  { type: 'table', name: 'Data Table', icon: '📋' }
];

const WidgetPalette: React.FC = () => {
  return (
    <div className="w-64 bg-gray-50 border-r p-4">
      <h3 className="font-semibold mb-4">Widget Palette</h3>
      <div className="space-y-2">
        {WIDGET_TYPES.map((widgetType) => (
          <DraggableWidget key={widgetType.type} widgetType={widgetType} />
        ))}
      </div>
    </div>
  );
};

const DraggableWidget: React.FC<{ widgetType: any }> = ({ widgetType }) => {
  const [{ isDragging }, drag] = useDrag({
    type: 'widget',
    item: { type: 'widget', widgetType: widgetType.type },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  return (
    <div
      ref={drag as any}
      className={`p-3 bg-white border rounded-lg cursor-move hover:shadow-md transition-shadow ${
        isDragging ? 'opacity-50' : ''
      }`}
    >
      <div className="flex items-center space-x-2">
        <span className="text-lg">{widgetType.icon}</span>
        <span className="text-sm font-medium">{widgetType.name}</span>
      </div>
    </div>
  );
};

const DroppableCanvas: React.FC<{
  widgets: DashboardWidget[];
  onWidgetAdd: (widgetType: string, position: { x: number; y: number }) => void;
  onWidgetMove: (widgetId: string, position: WidgetPosition) => void;
  onWidgetSelect: (widgetId: string) => void;
  selectedWidget: string | null;
  gridSize: number;
  rowHeight: number;
}> = ({ 
  widgets, 
  onWidgetAdd, 
  onWidgetMove, 
  onWidgetSelect, 
  selectedWidget,
  gridSize,
  rowHeight 
}) => {
  const canvasRef = useRef<HTMLDivElement>(null);

  const [{ isOver }, drop] = useDrop({
    accept: 'widget',
    drop: (item: DragItem, monitor) => {
      if (!canvasRef.current) return;

      const canvasRect = canvasRef.current.getBoundingClientRect();
      const clientOffset = monitor.getClientOffset();
      
      if (clientOffset) {
        const x = Math.floor((clientOffset.x - canvasRect.left) / (canvasRect.width / gridSize));
        const y = Math.floor((clientOffset.y - canvasRect.top) / rowHeight);
        
        if (item.widgetType) {
          onWidgetAdd(item.widgetType, { x: Math.max(0, x), y: Math.max(0, y) });
        }
      }
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
    }),
  });

  return (
    <div
      ref={(node) => {
        drop(node);
        if (canvasRef.current !== node) {
          (canvasRef as any).current = node;
        }
      }}
      className={`flex-1 relative bg-white border-2 border-dashed ${
        isOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300'
      } min-h-[600px] overflow-auto`}
      style={{
        backgroundImage: `
          linear-gradient(to right, #f0f0f0 1px, transparent 1px),
          linear-gradient(to bottom, #f0f0f0 1px, transparent 1px)
        `,
        backgroundSize: `${100 / gridSize}% ${rowHeight}px`,
      }}
    >
      {widgets.map((widget) => (
        <DashboardWidgetComponent
          key={widget.id}
          widget={widget}
          onMove={onWidgetMove}
          onSelect={onWidgetSelect}
          isSelected={selectedWidget === widget.id}
          gridSize={gridSize}
          rowHeight={rowHeight}
        />
      ))}
      {isOver && (
        <div className="absolute inset-0 bg-blue-100 bg-opacity-50 flex items-center justify-center">
          <div className="text-blue-600 font-semibold">Drop widget here</div>
        </div>
      )}
    </div>
  );
};

const DashboardWidgetComponent: React.FC<{
  widget: DashboardWidget;
  onMove: (widgetId: string, position: WidgetPosition) => void;
  onSelect: (widgetId: string) => void;
  isSelected: boolean;
  gridSize: number;
  rowHeight: number;
}> = ({ widget, onMove, onSelect, isSelected, gridSize, rowHeight }) => {
  const [{ isDragging }, drag] = useDrag({
    type: 'existing-widget',
    item: { type: 'existing-widget', id: widget.id },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  const widgetStyle = {
    position: 'absolute' as const,
    left: `${(widget.position.x / gridSize) * 100}%`,
    top: `${widget.position.y * rowHeight}px`,
    width: `${(widget.position.width / gridSize) * 100}%`,
    height: `${widget.position.height * rowHeight}px`,
    zIndex: isDragging ? 1000 : 1,
  };

  return (
    <div
      ref={drag as any}
      style={widgetStyle}
      className={`border-2 rounded-lg p-2 cursor-move ${
        isSelected 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-300 bg-white hover:border-gray-400'
      } ${isDragging ? 'opacity-50' : ''}`}
      onClick={() => onSelect(widget.id)}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-medium text-sm truncate">{widget.title}</h4>
        <Badge variant="secondary" className="text-xs">
          {widget.type}
        </Badge>
      </div>
      <div className="text-xs text-gray-500 mb-2">
        {widget.position.width}x{widget.position.height} grid units
      </div>
      <div className="bg-gray-100 rounded h-full min-h-[60px] flex items-center justify-center">
        <span className="text-gray-400 text-sm">Widget Preview</span>
      </div>
    </div>
  );
};

const WidgetPropertiesPanel: React.FC<{
  widget: DashboardWidget | null;
  onUpdate: (widgetId: string, updates: Partial<DashboardWidget>) => void;
  onDelete: (widgetId: string) => void;
}> = ({ widget, onUpdate, onDelete }) => {
  if (!widget) {
    return (
      <div className="w-80 bg-gray-50 border-l p-4">
        <h3 className="font-semibold mb-4">Properties</h3>
        <p className="text-gray-500 text-sm">Select a widget to edit its properties</p>
      </div>
    );
  }

  return (
    <div className="w-80 bg-gray-50 border-l p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Properties</h3>
        <Button
          variant="destructive"
          size="sm"
          onClick={() => onDelete(widget.id)}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Title</label>
          <input
            type="text"
            value={widget.title}
            onChange={(e) => onUpdate(widget.id, { title: e.target.value })}
            className="w-full px-3 py-2 border rounded-md text-sm"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Type</label>
          <select
            value={widget.type}
            onChange={(e) => onUpdate(widget.id, { type: e.target.value })}
            className="w-full px-3 py-2 border rounded-md text-sm"
          >
            {WIDGET_TYPES.map((type) => (
              <option key={type.type} value={type.type}>
                {type.name}
              </option>
            ))}
          </select>
        </div>
        
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-sm font-medium mb-1">Width</label>
            <input
              type="number"
              min="1"
              max="12"
              value={widget.position.width}
              onChange={(e) => onUpdate(widget.id, {
                position: { ...widget.position, width: parseInt(e.target.value) }
              })}
              className="w-full px-3 py-2 border rounded-md text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Height</label>
            <input
              type="number"
              min="1"
              value={widget.position.height}
              onChange={(e) => onUpdate(widget.id, {
                position: { ...widget.position, height: parseInt(e.target.value) }
              })}
              className="w-full px-3 py-2 border rounded-md text-sm"
            />
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Data Source</label>
          <input
            type="text"
            value={widget.dataSource}
            onChange={(e) => onUpdate(widget.id, { dataSource: e.target.value })}
            className="w-full px-3 py-2 border rounded-md text-sm"
            placeholder="e.g., metrics_api"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Configuration</label>
          <textarea
            value={JSON.stringify(widget.config, null, 2)}
            onChange={(e) => {
              try {
                const config = JSON.parse(e.target.value);
                onUpdate(widget.id, { config });
              } catch (error) {
                // Invalid JSON, ignore
              }
            }}
            className="w-full px-3 py-2 border rounded-md text-sm font-mono"
            rows={6}
            placeholder="JSON configuration"
          />
        </div>
      </div>
    </div>
  );
};

export const DashboardCustomizer: React.FC<{
  template: DashboardTemplate;
  onSave: (template: DashboardTemplate) => void;
  onPreview: () => void;
  onShare: () => void;
}> = ({ template, onSave, onPreview, onShare }) => {
  const [widgets, setWidgets] = useState<DashboardWidget[]>(template.widgets);
  const [selectedWidget, setSelectedWidget] = useState<string | null>(null);
  const [history, setHistory] = useState<DashboardWidget[][]>([template.widgets]);
  const [historyIndex, setHistoryIndex] = useState(0);

  const saveToHistory = useCallback((newWidgets: DashboardWidget[]) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push([...newWidgets]);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  }, [history, historyIndex]);

  const handleWidgetAdd = useCallback((widgetType: string, position: { x: number; y: number }) => {
    const newWidget: DashboardWidget = {
      id: `widget_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: widgetType,
      title: `New ${WIDGET_TYPES.find(t => t.type === widgetType)?.name || 'Widget'}`,
      position: {
        x: position.x,
        y: position.y,
        width: 4,
        height: 3,
      },
      config: {},
      dataSource: '',
    };

    const newWidgets = [...widgets, newWidget];
    setWidgets(newWidgets);
    saveToHistory(newWidgets);
    setSelectedWidget(newWidget.id);
  }, [widgets, saveToHistory]);

  const handleWidgetMove = useCallback((widgetId: string, position: WidgetPosition) => {
    const newWidgets = widgets.map(widget =>
      widget.id === widgetId ? { ...widget, position } : widget
    );
    setWidgets(newWidgets);
    saveToHistory(newWidgets);
  }, [widgets, saveToHistory]);

  const handleWidgetUpdate = useCallback((widgetId: string, updates: Partial<DashboardWidget>) => {
    const newWidgets = widgets.map(widget =>
      widget.id === widgetId ? { ...widget, ...updates } : widget
    );
    setWidgets(newWidgets);
    saveToHistory(newWidgets);
  }, [widgets, saveToHistory]);

  const handleWidgetDelete = useCallback((widgetId: string) => {
    const newWidgets = widgets.filter(widget => widget.id !== widgetId);
    setWidgets(newWidgets);
    saveToHistory(newWidgets);
    setSelectedWidget(null);
  }, [widgets, saveToHistory]);

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      setWidgets([...history[newIndex]]);
    }
  }, [history, historyIndex]);

  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      setWidgets([...history[newIndex]]);
    }
  }, [history, historyIndex]);

  const handleSave = useCallback(() => {
    const updatedTemplate: DashboardTemplate = {
      ...template,
      widgets,
    };
    onSave(updatedTemplate);
  }, [template, widgets, onSave]);

  const selectedWidgetData = selectedWidget 
    ? widgets.find(w => w.id === selectedWidget) || null 
    : null;

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="h-screen flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b px-4 py-2 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <h2 className="font-semibold">Dashboard Customizer</h2>
            <Badge variant="outline">{template.name}</Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleUndo}
              disabled={historyIndex === 0}
            >
              <Undo className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRedo}
              disabled={historyIndex === history.length - 1}
            >
              <Redo className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={onPreview}>
              <Eye className="h-4 w-4 mr-1" />
              Preview
            </Button>
            <Button variant="outline" size="sm" onClick={onShare}>
              <Share2 className="h-4 w-4 mr-1" />
              Share
            </Button>
            <Button onClick={handleSave}>
              <Save className="h-4 w-4 mr-1" />
              Save
            </Button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex">
          <WidgetPalette />
          
          <DroppableCanvas
            widgets={widgets}
            onWidgetAdd={handleWidgetAdd}
            onWidgetMove={handleWidgetMove}
            onWidgetSelect={setSelectedWidget}
            selectedWidget={selectedWidget}
            gridSize={template.layoutConfig.gridSize}
            rowHeight={template.layoutConfig.rowHeight}
          />
          
          <WidgetPropertiesPanel
            widget={selectedWidgetData}
            onUpdate={handleWidgetUpdate}
            onDelete={handleWidgetDelete}
          />
        </div>
      </div>
    </DndProvider>
  );
};

export default DashboardCustomizer;