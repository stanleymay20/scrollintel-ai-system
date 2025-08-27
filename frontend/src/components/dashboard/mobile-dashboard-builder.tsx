/**
 * Mobile Dashboard Builder Component
 * 
 * Specialized interface for creating and customizing mobile dashboard views
 */

import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Alert, AlertDescription } from '../ui/alert';
import { 
  Smartphone, Tablet, Monitor, RotateCcw, Settings, 
  Eye, Save, Share2, Download, Grid, List, 
  ChevronUp, ChevronDown, Plus, Minus
} from 'lucide-react';
import { DashboardTemplate, WidgetConfig } from './dashboard-builder';
import { TemplatePreview } from './template-preview';

interface MobileDashboardBuilderProps {
  template: DashboardTemplate;
  onTemplateUpdate: (template: DashboardTemplate) => void;
  onSave: () => void;
  onPreview: () => void;
}

interface MobileLayoutConfig {
  orientation: 'portrait' | 'landscape';
  stack_widgets: boolean;
  compact_mode: boolean;
  swipe_navigation: boolean;
  collapsible_sections: boolean;
  priority_widgets: string[];
}

// Mobile-specific widget configurations
const MOBILE_WIDGET_CONFIGS = {
  metric_card: {
    min_width: 2,
    max_width: 4,
    min_height: 2,
    max_height: 3,
    recommended: { width: 2, height: 2 }
  },
  kpi_grid: {
    min_width: 4,
    max_width: 4,
    min_height: 3,
    max_height: 4,
    recommended: { width: 4, height: 3 }
  },
  line_chart: {
    min_width: 4,
    max_width: 4,
    min_height: 3,
    max_height: 5,
    recommended: { width: 4, height: 4 }
  },
  bar_chart: {
    min_width: 4,
    max_width: 4,
    min_height: 3,
    max_height: 5,
    recommended: { width: 4, height: 4 }
  },
  pie_chart: {
    min_width: 3,
    max_width: 4,
    min_height: 3,
    max_height: 4,
    recommended: { width: 3, height: 3 }
  },
  data_table: {
    min_width: 4,
    max_width: 4,
    min_height: 4,
    max_height: 6,
    recommended: { width: 4, height: 5 }
  }
};

// Mobile Widget Priority Component
const WidgetPriorityList: React.FC<{
  widgets: WidgetConfig[];
  priorityOrder: string[];
  onPriorityChange: (newOrder: string[]) => void;
}> = ({ widgets, priorityOrder, onPriorityChange }) => {
  const moveWidget = (widgetId: string, direction: 'up' | 'down') => {
    const currentIndex = priorityOrder.indexOf(widgetId);
    if (currentIndex === -1) return;

    const newOrder = [...priorityOrder];
    const targetIndex = direction === 'up' ? currentIndex - 1 : currentIndex + 1;

    if (targetIndex >= 0 && targetIndex < newOrder.length) {
      [newOrder[currentIndex], newOrder[targetIndex]] = [newOrder[targetIndex], newOrder[currentIndex]];
      onPriorityChange(newOrder);
    }
  };

  const getWidgetById = (id: string) => widgets.find(w => w.id === id);

  return (
    <div className="space-y-2">
      <h4 className="font-medium text-sm">Widget Priority (Mobile)</h4>
      <div className="text-xs text-gray-600 mb-3">
        Higher priority widgets appear first on mobile devices
      </div>
      
      {priorityOrder.map((widgetId, index) => {
        const widget = getWidgetById(widgetId);
        if (!widget) return null;

        return (
          <div key={widgetId} className="flex items-center gap-2 p-2 border rounded">
            <div className="flex flex-col">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => moveWidget(widgetId, 'up')}
                disabled={index === 0}
                className="h-6 w-6 p-0"
              >
                <ChevronUp className="h-3 w-3" />
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => moveWidget(widgetId, 'down')}
                disabled={index === priorityOrder.length - 1}
                className="h-6 w-6 p-0"
              >
                <ChevronDown className="h-3 w-3" />
              </Button>
            </div>
            
            <div className="flex-1">
              <div className="font-medium text-sm">{widget.title}</div>
              <div className="text-xs text-gray-500">{widget.type}</div>
            </div>
            
            <Badge variant="outline" className="text-xs">
              #{index + 1}
            </Badge>
          </div>
        );
      })}
    </div>
  );
};

// Mobile Layout Optimizer
const MobileLayoutOptimizer: React.FC<{
  widgets: WidgetConfig[];
  onOptimize: (optimizedWidgets: WidgetConfig[]) => void;
}> = ({ widgets, onOptimize }) => {
  const optimizeForMobile = () => {
    const optimizedWidgets = widgets.map(widget => {
      const config = MOBILE_WIDGET_CONFIGS[widget.type as keyof typeof MOBILE_WIDGET_CONFIGS];
      if (!config) return widget;

      const mobilePosition = {
        x: 0, // Stack vertically on mobile
        y: widget.position.y,
        width: config.recommended.width,
        height: config.recommended.height
      };

      return {
        ...widget,
        responsive_config: {
          ...widget.responsive_config,
          mobile: mobilePosition
        }
      };
    });

    // Sort by priority and reposition
    const sortedWidgets = optimizedWidgets.map((widget, index) => ({
      ...widget,
      responsive_config: {
        ...widget.responsive_config,
        mobile: {
          ...widget.responsive_config?.mobile,
          y: index * (widget.responsive_config?.mobile?.height || 3)
        }
      }
    }));

    onOptimize(sortedWidgets);
  };

  const optimizeForTablet = () => {
    const optimizedWidgets = widgets.map(widget => {
      const config = MOBILE_WIDGET_CONFIGS[widget.type as keyof typeof MOBILE_WIDGET_CONFIGS];
      if (!config) return widget;

      // For tablet, use 2-column layout
      const tabletPosition = {
        x: widget.position.x % 2 === 0 ? 0 : 4,
        y: Math.floor(widget.position.x / 2) * config.recommended.height,
        width: Math.min(config.recommended.width + 1, 4),
        height: config.recommended.height
      };

      return {
        ...widget,
        responsive_config: {
          ...widget.responsive_config,
          tablet: tabletPosition
        }
      };
    });

    onOptimize(optimizedWidgets);
  };

  return (
    <div className="space-y-4">
      <h4 className="font-medium">Layout Optimization</h4>
      
      <div className="grid grid-cols-2 gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={optimizeForMobile}
          className="flex items-center gap-2"
        >
          <Smartphone className="h-4 w-4" />
          Optimize Mobile
        </Button>
        
        <Button
          size="sm"
          variant="outline"
          onClick={optimizeForTablet}
          className="flex items-center gap-2"
        >
          <Tablet className="h-4 w-4" />
          Optimize Tablet
        </Button>
      </div>
      
      <div className="text-xs text-gray-600">
        Auto-optimize widget sizes and positions for mobile and tablet devices
      </div>
    </div>
  );
};

// Mobile Settings Panel
const MobileSettingsPanel: React.FC<{
  mobileConfig: MobileLayoutConfig;
  onConfigChange: (config: MobileLayoutConfig) => void;
}> = ({ mobileConfig, onConfigChange }) => {
  const handleConfigChange = (key: keyof MobileLayoutConfig, value: any) => {
    onConfigChange({
      ...mobileConfig,
      [key]: value
    });
  };

  return (
    <div className="space-y-4">
      <h4 className="font-medium">Mobile Settings</h4>
      
      <div className="space-y-3">
        <div>
          <label className="text-sm font-medium">Orientation</label>
          <select
            value={mobileConfig.orientation}
            onChange={(e) => handleConfigChange('orientation', e.target.value)}
            className="w-full mt-1 px-3 py-2 border rounded-md text-sm"
          >
            <option value="portrait">Portrait</option>
            <option value="landscape">Landscape</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="stack_widgets"
            checked={mobileConfig.stack_widgets}
            onChange={(e) => handleConfigChange('stack_widgets', e.target.checked)}
          />
          <label htmlFor="stack_widgets" className="text-sm">Stack widgets vertically</label>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="compact_mode"
            checked={mobileConfig.compact_mode}
            onChange={(e) => handleConfigChange('compact_mode', e.target.checked)}
          />
          <label htmlFor="compact_mode" className="text-sm">Compact mode</label>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="swipe_navigation"
            checked={mobileConfig.swipe_navigation}
            onChange={(e) => handleConfigChange('swipe_navigation', e.target.checked)}
          />
          <label htmlFor="swipe_navigation" className="text-sm">Swipe navigation</label>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="collapsible_sections"
            checked={mobileConfig.collapsible_sections}
            onChange={(e) => handleConfigChange('collapsible_sections', e.target.checked)}
          />
          <label htmlFor="collapsible_sections" className="text-sm">Collapsible sections</label>
        </div>
      </div>
    </div>
  );
};

// Widget Size Adjuster for Mobile
const MobileWidgetSizeAdjuster: React.FC<{
  widget: WidgetConfig;
  onUpdate: (widgetId: string, updates: Partial<WidgetConfig>) => void;
}> = ({ widget, onUpdate }) => {
  const config = MOBILE_WIDGET_CONFIGS[widget.type as keyof typeof MOBILE_WIDGET_CONFIGS];
  if (!config) return null;

  const mobilePosition = widget.responsive_config?.mobile || widget.position;

  const adjustSize = (dimension: 'width' | 'height', delta: number) => {
    const newValue = mobilePosition[dimension] + delta;
    const minValue = config[`min_${dimension}`];
    const maxValue = config[`max_${dimension}`];

    if (newValue >= minValue && newValue <= maxValue) {
      const newPosition = {
        ...mobilePosition,
        [dimension]: newValue
      };

      onUpdate(widget.id, {
        responsive_config: {
          ...widget.responsive_config,
          mobile: newPosition
        }
      });
    }
  };

  return (
    <div className="space-y-3">
      <h5 className="font-medium text-sm">Mobile Size</h5>
      
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm">Width: {mobilePosition.width}</span>
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="outline"
              onClick={() => adjustSize('width', -1)}
              disabled={mobilePosition.width <= config.min_width}
              className="h-6 w-6 p-0"
            >
              <Minus className="h-3 w-3" />
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => adjustSize('width', 1)}
              disabled={mobilePosition.width >= config.max_width}
              className="h-6 w-6 p-0"
            >
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm">Height: {mobilePosition.height}</span>
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="outline"
              onClick={() => adjustSize('height', -1)}
              disabled={mobilePosition.height <= config.min_height}
              className="h-6 w-6 p-0"
            >
              <Minus className="h-3 w-3" />
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => adjustSize('height', 1)}
              disabled={mobilePosition.height >= config.max_height}
              className="h-6 w-6 p-0"
            >
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>

      <div className="text-xs text-gray-600">
        Recommended: {config.recommended.width}×{config.recommended.height}
      </div>
    </div>
  );
};

// Main Mobile Dashboard Builder Component
export const MobileDashboardBuilder: React.FC<MobileDashboardBuilderProps> = ({
  template,
  onTemplateUpdate,
  onSave,
  onPreview
}) => {
  const [currentBreakpoint, setCurrentBreakpoint] = useState<string>('mobile');
  const [selectedWidget, setSelectedWidget] = useState<WidgetConfig | null>(null);
  const [mobileConfig, setMobileConfig] = useState<MobileLayoutConfig>({
    orientation: 'portrait',
    stack_widgets: true,
    compact_mode: false,
    swipe_navigation: true,
    collapsible_sections: false,
    priority_widgets: template.widgets.map(w => w.id)
  });

  const handleWidgetUpdate = useCallback((widgetId: string, updates: Partial<WidgetConfig>) => {
    const updatedWidgets = template.widgets.map(widget =>
      widget.id === widgetId ? { ...widget, ...updates } : widget
    );

    onTemplateUpdate({
      ...template,
      widgets: updatedWidgets,
    });
  }, [template, onTemplateUpdate]);

  const handlePriorityChange = (newOrder: string[]) => {
    setMobileConfig(prev => ({
      ...prev,
      priority_widgets: newOrder
    }));
  };

  const handleLayoutOptimization = (optimizedWidgets: WidgetConfig[]) => {
    onTemplateUpdate({
      ...template,
      widgets: optimizedWidgets
    });
  };

  const exportMobileConfig = () => {
    const mobileTemplate = {
      ...template,
      mobile_config: mobileConfig,
      widgets: template.widgets.map(widget => ({
        ...widget,
        mobile_priority: mobileConfig.priority_widgets.indexOf(widget.id)
      }))
    };

    const blob = new Blob([JSON.stringify(mobileTemplate, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${template.name}-mobile-config.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Left Sidebar - Mobile Controls */}
      <div className="w-80 bg-white border-r overflow-y-auto">
        <div className="p-4 border-b">
          <h3 className="font-semibold flex items-center gap-2">
            <Smartphone className="h-5 w-5" />
            Mobile Dashboard Builder
          </h3>
        </div>

        <Tabs defaultValue="layout" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="layout">Layout</TabsTrigger>
            <TabsTrigger value="priority">Priority</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="layout" className="p-4 space-y-4">
            <MobileLayoutOptimizer
              widgets={template.widgets}
              onOptimize={handleLayoutOptimization}
            />

            {selectedWidget && (
              <MobileWidgetSizeAdjuster
                widget={selectedWidget}
                onUpdate={handleWidgetUpdate}
              />
            )}
          </TabsContent>

          <TabsContent value="priority" className="p-4">
            <WidgetPriorityList
              widgets={template.widgets}
              priorityOrder={mobileConfig.priority_widgets}
              onPriorityChange={handlePriorityChange}
            />
          </TabsContent>

          <TabsContent value="settings" className="p-4">
            <MobileSettingsPanel
              mobileConfig={mobileConfig}
              onConfigChange={setMobileConfig}
            />
          </TabsContent>
        </Tabs>
      </div>

      {/* Main Preview Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b bg-white flex justify-between items-center">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold">{template.name} - Mobile View</h2>
            
            {/* Breakpoint Selector */}
            <div className="flex border rounded-md">
              {['mobile', 'tablet', 'desktop'].map(breakpoint => {
                const Icon = breakpoint === 'mobile' ? Smartphone : 
                           breakpoint === 'tablet' ? Tablet : Monitor;
                return (
                  <Button
                    key={breakpoint}
                    size="sm"
                    variant={currentBreakpoint === breakpoint ? 'default' : 'ghost'}
                    onClick={() => setCurrentBreakpoint(breakpoint)}
                    className="rounded-none first:rounded-l-md last:rounded-r-md"
                  >
                    <Icon className="h-4 w-4" />
                  </Button>
                );
              })}
            </div>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" onClick={exportMobileConfig}>
              <Download className="h-4 w-4 mr-2" />
              Export Config
            </Button>
            <Button variant="outline" onClick={onPreview}>
              <Eye className="h-4 w-4 mr-2" />
              Preview
            </Button>
            <Button onClick={onSave}>
              <Save className="h-4 w-4 mr-2" />
              Save
            </Button>
          </div>
        </div>

        {/* Preview Content */}
        <div className="flex-1 p-4">
          <div className="max-w-sm mx-auto bg-white rounded-lg shadow-lg overflow-hidden">
            {/* Mobile Device Frame */}
            <div className="bg-gray-900 p-2">
              <div className="bg-white rounded-lg overflow-hidden" style={{ aspectRatio: '9/16' }}>
                <TemplatePreview
                  template={template}
                  currentBreakpoint={currentBreakpoint}
                  onBreakpointChange={setCurrentBreakpoint}
                  isLivePreview={false}
                  className="h-full"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Sidebar - Widget Properties */}
      <div className="w-80 bg-white border-l overflow-y-auto">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Widget Properties</h3>
        </div>

        {selectedWidget ? (
          <div className="p-4 space-y-4">
            <div>
              <h4 className="font-medium">{selectedWidget.title}</h4>
              <Badge variant="outline">{selectedWidget.type}</Badge>
            </div>

            <MobileWidgetSizeAdjuster
              widget={selectedWidget}
              onUpdate={handleWidgetUpdate}
            />

            <div className="pt-4 border-t">
              <h5 className="font-medium text-sm mb-2">Mobile Optimizations</h5>
              <div className="text-xs text-gray-600 space-y-1">
                <div>• Optimized for touch interaction</div>
                <div>• Responsive text sizing</div>
                <div>• Swipe-friendly navigation</div>
                <div>• Battery-efficient updates</div>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-4 text-center text-gray-500">
            <Smartphone className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <div>Select a widget to customize for mobile</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MobileDashboardBuilder;