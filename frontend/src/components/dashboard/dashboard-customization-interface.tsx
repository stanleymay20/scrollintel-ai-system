/**
 * Main Dashboard Customization Interface
 * 
 * Comprehensive interface that combines all dashboard customization features
 */

import React, { useState, useCallback, useEffect } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Alert, AlertDescription } from '../ui/alert';
import { 
  Layout, Eye, Save, Share2, Download, Upload, Settings, 
  Smartphone, Tablet, Monitor, Grid, List, Palette, 
  RefreshCw, Undo, Redo, Copy, Trash2, Plus, 
  AlertTriangle, CheckCircle, Clock, Users
} from 'lucide-react';

import { 
  WidgetPalette, 
  DropZone, 
  DraggableWidget, 
  WidgetPropertiesPanel, 
  TemplateSelector,
  DashboardTemplate, 
  WidgetConfig 
} from './dashboard-builder';
import { TemplatePreview } from './template-preview';
import { MobileDashboardBuilder } from './mobile-dashboard-builder';
import { useWebSocket } from '@/hooks/useWebSocket';

interface DashboardCustomizationInterfaceProps {
  initialTemplate?: DashboardTemplate;
  userId: string;
  userRole: string;
  onSave?: (template: DashboardTemplate) => void;
  onPublish?: (template: DashboardTemplate) => void;
  onShare?: (template: DashboardTemplate) => void;
}

interface HistoryState {
  template: DashboardTemplate;
  timestamp: number;
  action: string;
}

// Default template structure
const createDefaultTemplate = (): DashboardTemplate => ({
  id: `template_${Date.now()}`,
  name: 'New Dashboard',
  description: 'Custom dashboard template',
  category: 'custom',
  widgets: [],
  layout_config: {
    grid_size: 12,
    row_height: 100,
    margin: [10, 10],
    responsive_breakpoints: {
      mobile: 768,
      tablet: 1024,
      desktop: 1200,
    },
    auto_layout: false,
  },
  default_filters: [],
  theme: {
    primary_color: '#3b82f6',
    secondary_color: '#64748b',
    background_color: '#ffffff',
    text_color: '#1f2937',
  },
  version: 1,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  created_by: '',
  is_public: false,
  tags: [],
});

// Available templates
const SAMPLE_TEMPLATES: DashboardTemplate[] = [
  {
    ...createDefaultTemplate(),
    id: 'executive_template',
    name: 'Executive Dashboard',
    description: 'High-level metrics for executives',
    category: 'executive',
    widgets: [
      {
        id: 'roi_overview',
        type: 'kpi_grid',
        title: 'ROI Overview',
        position: { x: 0, y: 0, width: 6, height: 3 },
        data_source: 'roi_calculator',
        visualization_config: {
          chart_type: 'grid',
          color_scheme: 'blue',
          show_legend: false,
          refresh_interval: 300,
        },
      },
      {
        id: 'revenue_trend',
        type: 'line_chart',
        title: 'Revenue Trend',
        position: { x: 6, y: 0, width: 6, height: 3 },
        data_source: 'performance_monitor',
        visualization_config: {
          chart_type: 'line',
          color_scheme: 'green',
          show_legend: true,
          animation_enabled: true,
          refresh_interval: 300,
        },
      },
    ],
    tags: ['executive', 'roi', 'revenue'],
  },
  {
    ...createDefaultTemplate(),
    id: 'operational_template',
    name: 'Operational Dashboard',
    description: 'Day-to-day operational metrics',
    category: 'operational',
    widgets: [
      {
        id: 'system_health',
        type: 'gauge_chart',
        title: 'System Health',
        position: { x: 0, y: 0, width: 4, height: 3 },
        data_source: 'system_metrics',
        visualization_config: {
          chart_type: 'gauge',
          color_scheme: 'default',
          refresh_interval: 60,
        },
      },
      {
        id: 'deployment_status',
        type: 'data_table',
        title: 'Recent Deployments',
        position: { x: 4, y: 0, width: 8, height: 3 },
        data_source: 'deployment_tracker',
        visualization_config: {
          refresh_interval: 120,
        },
      },
    ],
    tags: ['operational', 'system', 'deployments'],
  },
];

export const DashboardCustomizationInterface: React.FC<DashboardCustomizationInterfaceProps> = ({
  initialTemplate,
  userId,
  userRole,
  onSave,
  onPublish,
  onShare
}) => {
  const [currentTemplate, setCurrentTemplate] = useState<DashboardTemplate>(
    initialTemplate || createDefaultTemplate()
  );
  const [selectedWidget, setSelectedWidget] = useState<WidgetConfig | null>(null);
  const [currentBreakpoint, setCurrentBreakpoint] = useState<string>('desktop');
  const [activeTab, setActiveTab] = useState('design');
  const [showGrid, setShowGrid] = useState(true);
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [isMobileBuilder, setIsMobileBuilder] = useState(false);
  const [showTemplateSelector, setShowTemplateSelector] = useState(false);
  
  // History management
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Loading and error states
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const { socket, isConnected } = useWebSocket();

  // Initialize template
  useEffect(() => {
    if (initialTemplate) {
      setCurrentTemplate(initialTemplate);
      addToHistory(initialTemplate, 'Initial load');
    } else {
      const defaultTemplate = createDefaultTemplate();
      defaultTemplate.created_by = userId;
      setCurrentTemplate(defaultTemplate);
      addToHistory(defaultTemplate, 'Created new template');
    }
  }, [initialTemplate, userId]);

  // History management
  const addToHistory = useCallback((template: DashboardTemplate, action: string) => {
    const newHistoryItem: HistoryState = {
      template: JSON.parse(JSON.stringify(template)),
      timestamp: Date.now(),
      action
    };

    setHistory(prev => {
      const newHistory = prev.slice(0, historyIndex + 1);
      newHistory.push(newHistoryItem);
      return newHistory.slice(-50); // Keep last 50 changes
    });
    setHistoryIndex(prev => Math.min(prev + 1, 49));
    setHasUnsavedChanges(true);
  }, [historyIndex]);

  const undo = useCallback(() => {
    if (historyIndex > 0) {
      const previousState = history[historyIndex - 1];
      setCurrentTemplate(previousState.template);
      setHistoryIndex(prev => prev - 1);
      setHasUnsavedChanges(true);
    }
  }, [history, historyIndex]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const nextState = history[historyIndex + 1];
      setCurrentTemplate(nextState.template);
      setHistoryIndex(prev => prev + 1);
      setHasUnsavedChanges(true);
    }
  }, [history, historyIndex]);

  // Template operations
  const handleTemplateUpdate = useCallback((updatedTemplate: DashboardTemplate) => {
    setCurrentTemplate(updatedTemplate);
    addToHistory(updatedTemplate, 'Template updated');
  }, [addToHistory]);

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
      visualization_config: {
        chart_type: widgetType.includes('chart') ? widgetType.replace('_chart', '') : undefined,
        color_scheme: 'default',
        show_legend: true,
        animation_enabled: true,
        refresh_interval: 300,
      },
    };

    const updatedTemplate = {
      ...currentTemplate,
      widgets: [...currentTemplate.widgets, newWidget],
      updated_at: new Date().toISOString(),
    };

    handleTemplateUpdate(updatedTemplate);
    setSelectedWidget(newWidget);
  }, [currentTemplate, handleTemplateUpdate]);

  const handleWidgetMove = useCallback((widgetId: string, position: { x: number; y: number }) => {
    const updatedWidgets = currentTemplate.widgets.map(widget =>
      widget.id === widgetId
        ? { ...widget, position: { ...widget.position, ...position } }
        : widget
    );

    handleTemplateUpdate({
      ...currentTemplate,
      widgets: updatedWidgets,
      updated_at: new Date().toISOString(),
    });
  }, [currentTemplate, handleTemplateUpdate]);

  const handleWidgetUpdate = useCallback((widgetId: string, updates: Partial<WidgetConfig>) => {
    const updatedWidgets = currentTemplate.widgets.map(widget =>
      widget.id === widgetId ? { ...widget, ...updates } : widget
    );

    handleTemplateUpdate({
      ...currentTemplate,
      widgets: updatedWidgets,
      updated_at: new Date().toISOString(),
    });
  }, [currentTemplate, handleTemplateUpdate]);

  const handleWidgetDelete = useCallback((widgetId: string) => {
    const updatedWidgets = currentTemplate.widgets.filter(widget => widget.id !== widgetId);
    
    handleTemplateUpdate({
      ...currentTemplate,
      widgets: updatedWidgets,
      updated_at: new Date().toISOString(),
    });

    if (selectedWidget?.id === widgetId) {
      setSelectedWidget(null);
    }
  }, [currentTemplate, handleTemplateUpdate, selectedWidget]);

  const handleWidgetDuplicate = useCallback((widget: WidgetConfig) => {
    const duplicatedWidget: WidgetConfig = {
      ...widget,
      id: generateWidgetId(),
      title: `${widget.title} (Copy)`,
      position: {
        ...widget.position,
        x: Math.min(widget.position.x + 1, currentTemplate.layout_config.grid_size - widget.position.width),
        y: widget.position.y + 1,
      },
    };

    const updatedTemplate = {
      ...currentTemplate,
      widgets: [...currentTemplate.widgets, duplicatedWidget],
      updated_at: new Date().toISOString(),
    };

    handleTemplateUpdate(updatedTemplate);
  }, [currentTemplate, handleTemplateUpdate]);

  // Save operations
  const handleSave = async () => {
    setSaveStatus('saving');
    setIsLoading(true);

    try {
      if (onSave) {
        await onSave(currentTemplate);
      }
      
      // Simulate API call if no onSave provided
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setSaveStatus('saved');
      setHasUnsavedChanges(false);
      
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (err) {
      setSaveStatus('error');
      setError('Failed to save template');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePublish = async () => {
    if (onPublish) {
      try {
        setIsLoading(true);
        await onPublish({ ...currentTemplate, is_public: true });
        setHasUnsavedChanges(false);
      } catch (err) {
        setError('Failed to publish template');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleShare = () => {
    if (onShare) {
      onShare(currentTemplate);
    } else {
      // Default share behavior - copy link to clipboard
      const shareUrl = `${window.location.origin}/dashboard/template/${currentTemplate.id}`;
      navigator.clipboard.writeText(shareUrl);
      // Show toast notification
    }
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(currentTemplate, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentTemplate.name.replace(/\s+/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedTemplate = JSON.parse(e.target?.result as string);
        setCurrentTemplate(importedTemplate);
        addToHistory(importedTemplate, 'Imported template');
        setShowTemplateSelector(false);
      } catch (err) {
        setError('Invalid template file');
      }
    };
    reader.readAsText(file);
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 's':
            e.preventDefault();
            handleSave();
            break;
          case 'z':
            e.preventDefault();
            if (e.shiftKey) {
              redo();
            } else {
              undo();
            }
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSave, undo, redo]);

  if (showTemplateSelector) {
    return (
      <div className="p-6">
        <TemplateSelector
          templates={SAMPLE_TEMPLATES}
          onSelect={(template) => {
            setCurrentTemplate(template);
            addToHistory(template, 'Selected template');
            setShowTemplateSelector(false);
          }}
          onCreateNew={() => {
            const newTemplate = createDefaultTemplate();
            newTemplate.created_by = userId;
            setCurrentTemplate(newTemplate);
            addToHistory(newTemplate, 'Created new template');
            setShowTemplateSelector(false);
          }}
        />
      </div>
    );
  }

  if (isMobileBuilder) {
    return (
      <MobileDashboardBuilder
        template={currentTemplate}
        onTemplateUpdate={handleTemplateUpdate}
        onSave={handleSave}
        onPreview={() => setIsPreviewMode(true)}
      />
    );
  }

  if (isPreviewMode) {
    return (
      <div className="h-screen">
        <div className="p-4 border-b bg-white flex justify-between items-center">
          <h2 className="text-xl font-semibold">Dashboard Preview</h2>
          <Button onClick={() => setIsPreviewMode(false)}>
            Exit Preview
          </Button>
        </div>
        <TemplatePreview
          template={currentTemplate}
          isLivePreview={isConnected}
          currentBreakpoint={currentBreakpoint}
          onBreakpointChange={setCurrentBreakpoint}
        />
      </div>
    );
  }

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="flex h-screen bg-gray-50">
        {/* Left Sidebar - Widget Palette */}
        <div className="w-64 bg-white border-r overflow-y-auto">
          <div className="p-4 border-b">
            <h3 className="font-semibold mb-4">Widget Palette</h3>
            <WidgetPalette onWidgetSelect={handleWidgetAdd} />
          </div>
        </div>

        {/* Main Canvas */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="p-4 border-b bg-white">
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-4">
                <h2 className="text-xl font-semibold">{currentTemplate.name}</h2>
                
                {hasUnsavedChanges && (
                  <Badge variant="outline" className="text-orange-600">
                    <Clock className="h-3 w-3 mr-1" />
                    Unsaved Changes
                  </Badge>
                )}

                {saveStatus === 'saved' && (
                  <Badge variant="outline" className="text-green-600">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Saved
                  </Badge>
                )}

                {isConnected && (
                  <Badge variant="outline" className="text-blue-600">
                    Live Preview
                  </Badge>
                )}
              </div>

              <div className="flex items-center gap-2">
                {/* History Controls */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={undo}
                  disabled={historyIndex <= 0}
                >
                  <Undo className="h-4 w-4" />
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={redo}
                  disabled={historyIndex >= history.length - 1}
                >
                  <Redo className="h-4 w-4" />
                </Button>

                {/* Breakpoint Controls */}
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

                {/* View Controls */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowGrid(!showGrid)}
                >
                  <Grid className="h-4 w-4" />
                </Button>

                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setIsMobileBuilder(true)}
                >
                  <Smartphone className="h-4 w-4 mr-2" />
                  Mobile Builder
                </Button>

                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setIsPreviewMode(true)}
                >
                  <Eye className="h-4 w-4 mr-2" />
                  Preview
                </Button>

                {/* Action Buttons */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleShare}
                >
                  <Share2 className="h-4 w-4 mr-2" />
                  Share
                </Button>

                <Button
                  onClick={handleSave}
                  disabled={isLoading || saveStatus === 'saving'}
                >
                  {saveStatus === 'saving' ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  Save
                </Button>
              </div>
            </div>

            {/* Error Alert */}
            {error && (
              <Alert className="mb-4">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {error}
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setError(null)}
                    className="ml-2"
                  >
                    Dismiss
                  </Button>
                </AlertDescription>
              </Alert>
            )}
          </div>

          {/* Canvas */}
          <div className="flex-1 p-4 overflow-auto">
            <DropZone
              template={currentTemplate}
              currentBreakpoint={currentBreakpoint}
              onWidgetAdd={handleWidgetAdd}
              onWidgetMove={handleWidgetMove}
              showGrid={showGrid}
            >
              {currentTemplate.widgets.map((widget) => (
                <DraggableWidget
                  key={widget.id}
                  widget={widget}
                  currentBreakpoint={currentBreakpoint}
                  onSelect={setSelectedWidget}
                  onDelete={handleWidgetDelete}
                  onDuplicate={handleWidgetDuplicate}
                  isSelected={selectedWidget?.id === widget.id}
                  liveData={isConnected ? { value: Math.random() * 1000 } : undefined}
                />
              ))}
            </DropZone>
          </div>
        </div>

        {/* Right Sidebar - Properties Panel */}
        <div className="w-80 bg-white border-l">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="design">Design</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
              <TabsTrigger value="export">Export</TabsTrigger>
            </TabsList>
            
            <TabsContent value="design">
              <WidgetPropertiesPanel
                widget={selectedWidget}
                currentBreakpoint={currentBreakpoint}
                onUpdate={handleWidgetUpdate}
              />
            </TabsContent>
            
            <TabsContent value="settings" className="p-4 space-y-4">
              <div>
                <label className="text-sm font-medium">Template Name</label>
                <input
                  type="text"
                  value={currentTemplate.name}
                  onChange={(e) => handleTemplateUpdate({
                    ...currentTemplate,
                    name: e.target.value
                  })}
                  className="w-full mt-1 px-3 py-2 border rounded-md"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Description</label>
                <textarea
                  value={currentTemplate.description}
                  onChange={(e) => handleTemplateUpdate({
                    ...currentTemplate,
                    description: e.target.value
                  })}
                  className="w-full mt-1 px-3 py-2 border rounded-md"
                  rows={3}
                />
              </div>

              <div>
                <label className="text-sm font-medium">Category</label>
                <select
                  value={currentTemplate.category}
                  onChange={(e) => handleTemplateUpdate({
                    ...currentTemplate,
                    category: e.target.value
                  })}
                  className="w-full mt-1 px-3 py-2 border rounded-md"
                >
                  <option value="executive">Executive</option>
                  <option value="operational">Operational</option>
                  <option value="financial">Financial</option>
                  <option value="technical">Technical</option>
                  <option value="custom">Custom</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="is_public"
                  checked={currentTemplate.is_public}
                  onChange={(e) => handleTemplateUpdate({
                    ...currentTemplate,
                    is_public: e.target.checked
                  })}
                />
                <label htmlFor="is_public" className="text-sm">Make template public</label>
              </div>

              <div>
                <Button
                  onClick={handlePublish}
                  disabled={isLoading}
                  className="w-full"
                >
                  <Users className="h-4 w-4 mr-2" />
                  Publish Template
                </Button>
              </div>
            </TabsContent>
            
            <TabsContent value="export" className="p-4 space-y-4">
              <div className="space-y-2">
                <Button
                  onClick={handleExport}
                  variant="outline"
                  className="w-full"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export Template
                </Button>
                
                <div>
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleImport}
                    className="hidden"
                    id="import-template"
                  />
                  <Button
                    onClick={() => document.getElementById('import-template')?.click()}
                    variant="outline"
                    className="w-full"
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Import Template
                  </Button>
                </div>

                <Button
                  onClick={() => setShowTemplateSelector(true)}
                  variant="outline"
                  className="w-full"
                >
                  <Layout className="h-4 w-4 mr-2" />
                  Browse Templates
                </Button>
              </div>

              <div className="pt-4 border-t">
                <h4 className="font-medium mb-2">Template Info</h4>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>Widgets: {currentTemplate.widgets.length}</div>
                  <div>Version: {currentTemplate.version}</div>
                  <div>Created: {new Date(currentTemplate.created_at).toLocaleDateString()}</div>
                  <div>Updated: {new Date(currentTemplate.updated_at).toLocaleDateString()}</div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </DndProvider>
  );
};

export default DashboardCustomizationInterface;