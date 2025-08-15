'use client'

import React, { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Layout, 
  Grid, 
  Plus, 
  Settings, 
  Save, 
  Eye,
  Trash2,
  Copy,
  Move,
  BarChart3,
  PieChart,
  LineChart,
  Table,
  Filter,
  AlertTriangle
} from 'lucide-react'

// Mock react-grid-layout for demonstration
// In a real implementation, you would use react-grid-layout
const GridLayout = ({ children, layout, onLayoutChange, ...props }: any) => (
  <div className="grid grid-cols-12 gap-4 min-h-[600px]" {...props}>
    {children}
  </div>
)

interface DashboardCustomizerProps {
  dashboardId?: string
  initialLayout?: any[]
  charts?: any[]
  onSave?: (layout: any) => void
}

const widgetTypes = [
  { type: 'chart', label: 'Chart', icon: BarChart3, color: 'bg-blue-100 text-blue-700' },
  { type: 'kpi', label: 'KPI Card', icon: BarChart3, color: 'bg-green-100 text-green-700' },
  { type: 'table', label: 'Data Table', icon: Table, color: 'bg-purple-100 text-purple-700' },
  { type: 'filters', label: 'Filter Panel', icon: Filter, color: 'bg-orange-100 text-orange-700' },
  { type: 'status', label: 'Status Widget', icon: AlertTriangle, color: 'bg-red-100 text-red-700' }
]

const dashboardTemplates = [
  {
    id: 'executive',
    name: 'Executive Dashboard',
    description: 'High-level KPIs and metrics',
    layout: [
      { i: 'kpi-1', x: 0, y: 0, w: 3, h: 2, type: 'kpi' },
      { i: 'kpi-2', x: 3, y: 0, w: 3, h: 2, type: 'kpi' },
      { i: 'kpi-3', x: 6, y: 0, w: 3, h: 2, type: 'kpi' },
      { i: 'kpi-4', x: 9, y: 0, w: 3, h: 2, type: 'kpi' },
      { i: 'chart-1', x: 0, y: 2, w: 6, h: 4, type: 'chart' },
      { i: 'chart-2', x: 6, y: 2, w: 6, h: 4, type: 'chart' }
    ]
  },
  {
    id: 'analytical',
    name: 'Analytical Dashboard',
    description: 'Detailed analysis and drill-down',
    layout: [
      { i: 'filters', x: 0, y: 0, w: 3, h: 6, type: 'filters' },
      { i: 'main-chart', x: 3, y: 0, w: 9, h: 4, type: 'chart' },
      { i: 'detail-1', x: 3, y: 4, w: 4, h: 3, type: 'chart' },
      { i: 'detail-2', x: 7, y: 4, w: 5, h: 3, type: 'chart' }
    ]
  },
  {
    id: 'operational',
    name: 'Operational Dashboard',
    description: 'Real-time monitoring',
    layout: [
      { i: 'status-1', x: 0, y: 0, w: 2, h: 2, type: 'status' },
      { i: 'status-2', x: 2, y: 0, w: 2, h: 2, type: 'status' },
      { i: 'status-3', x: 4, y: 0, w: 2, h: 2, type: 'status' },
      { i: 'alerts', x: 6, y: 0, w: 6, h: 2, type: 'alerts' },
      { i: 'trend-1', x: 0, y: 2, w: 6, h: 3, type: 'chart' },
      { i: 'trend-2', x: 6, y: 2, w: 6, h: 3, type: 'chart' }
    ]
  }
]

export function DashboardCustomizer({ 
  dashboardId, 
  initialLayout = [], 
  charts = [], 
  onSave 
}: DashboardCustomizerProps) {
  const [layout, setLayout] = useState(initialLayout)
  const [selectedWidget, setSelectedWidget] = useState<any>(null)
  const [dashboardName, setDashboardName] = useState('My Dashboard')
  const [dashboardDescription, setDashboardDescription] = useState('')
  const [isPreviewMode, setIsPreviewMode] = useState(false)

  const handleLayoutChange = useCallback((newLayout: any[]) => {
    setLayout(newLayout)
  }, [])

  const addWidget = (type: string) => {
    const newWidget = {
      i: `${type}-${Date.now()}`,
      x: 0,
      y: 0,
      w: getDefaultSize(type).w,
      h: getDefaultSize(type).h,
      type: type,
      title: `New ${type}`,
      config: {}
    }
    
    setLayout(prev => [...prev, newWidget])
  }

  const removeWidget = (widgetId: string) => {
    setLayout(prev => prev.filter(item => item.i !== widgetId))
    if (selectedWidget?.i === widgetId) {
      setSelectedWidget(null)
    }
  }

  const duplicateWidget = (widget: any) => {
    const newWidget = {
      ...widget,
      i: `${widget.type}-${Date.now()}`,
      x: widget.x + 1,
      y: widget.y + 1,
      title: `${widget.title} (Copy)`
    }
    
    setLayout(prev => [...prev, newWidget])
  }

  const updateWidget = (widgetId: string, updates: any) => {
    setLayout(prev => prev.map(item => 
      item.i === widgetId ? { ...item, ...updates } : item
    ))
    
    if (selectedWidget?.i === widgetId) {
      setSelectedWidget(prev => ({ ...prev, ...updates }))
    }
  }

  const applyTemplate = (template: any) => {
    setLayout(template.layout)
    setDashboardName(template.name)
    setDashboardDescription(template.description)
  }

  const getDefaultSize = (type: string) => {
    const sizes = {
      chart: { w: 6, h: 4 },
      kpi: { w: 3, h: 2 },
      table: { w: 8, h: 4 },
      filters: { w: 3, h: 6 },
      status: { w: 2, h: 2 },
      alerts: { w: 6, h: 2 }
    }
    return sizes[type as keyof typeof sizes] || { w: 4, h: 3 }
  }

  const renderWidget = (item: any) => {
    const widgetType = widgetTypes.find(w => w.type === item.type)
    const Icon = widgetType?.icon || BarChart3

    return (
      <Card
        key={item.i}
        className={`cursor-pointer transition-all ${
          selectedWidget?.i === item.i ? 'ring-2 ring-blue-500' : ''
        }`}
        style={{
          gridColumn: `span ${item.w}`,
          gridRow: `span ${item.h}`,
          minHeight: `${item.h * 100}px`
        }}
        onClick={() => setSelectedWidget(item)}
      >
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm flex items-center gap-2">
              <Icon className="h-4 w-4" />
              {item.title || `${widgetType?.label} Widget`}
            </CardTitle>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  duplicateWidget(item)
                }}
              >
                <Copy className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  removeWidget(item.i)
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <Icon className="h-8 w-8 mx-auto mb-2" />
              <p className="text-sm">{widgetType?.label}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  const handleSave = () => {
    const dashboardConfig = {
      id: dashboardId,
      name: dashboardName,
      description: dashboardDescription,
      layout: layout,
      updated_at: new Date().toISOString()
    }
    
    onSave?.(dashboardConfig)
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 overflow-y-auto">
        <div className="p-4 border-b">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Layout className="h-5 w-5" />
            Dashboard Builder
          </h2>
        </div>

        <Tabs defaultValue="widgets" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="widgets">Widgets</TabsTrigger>
            <TabsTrigger value="templates">Templates</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="widgets" className="p-4 space-y-4">
            <div>
              <Label className="text-sm font-medium">Add Widgets</Label>
              <div className="grid grid-cols-1 gap-2 mt-2">
                {widgetTypes.map((widget) => {
                  const Icon = widget.icon
                  return (
                    <Button
                      key={widget.type}
                      variant="outline"
                      className="justify-start h-auto p-3"
                      onClick={() => addWidget(widget.type)}
                    >
                      <div className={`p-2 rounded-md mr-3 ${widget.color}`}>
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="text-left">
                        <div className="font-medium">{widget.label}</div>
                      </div>
                    </Button>
                  )
                })}
              </div>
            </div>

            {selectedWidget && (
              <div className="border-t pt-4">
                <Label className="text-sm font-medium">Widget Properties</Label>
                <div className="space-y-3 mt-2">
                  <div>
                    <Label htmlFor="widget-title" className="text-xs">Title</Label>
                    <Input
                      id="widget-title"
                      value={selectedWidget.title || ''}
                      onChange={(e) => updateWidget(selectedWidget.i, { title: e.target.value })}
                      className="h-8"
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <Label className="text-xs">Width</Label>
                      <Input
                        type="number"
                        value={selectedWidget.w}
                        onChange={(e) => updateWidget(selectedWidget.i, { w: parseInt(e.target.value) })}
                        min="1"
                        max="12"
                        className="h-8"
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Height</Label>
                      <Input
                        type="number"
                        value={selectedWidget.h}
                        onChange={(e) => updateWidget(selectedWidget.i, { h: parseInt(e.target.value) })}
                        min="1"
                        max="8"
                        className="h-8"
                      />
                    </div>
                  </div>

                  {selectedWidget.type === 'chart' && (
                    <div>
                      <Label className="text-xs">Chart</Label>
                      <Select
                        value={selectedWidget.chart_id || ''}
                        onValueChange={(value) => updateWidget(selectedWidget.i, { chart_id: value })}
                      >
                        <SelectTrigger className="h-8">
                          <SelectValue placeholder="Select chart" />
                        </SelectTrigger>
                        <SelectContent>
                          {charts.map((chart) => (
                            <SelectItem key={chart.id} value={chart.id}>
                              {chart.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="templates" className="p-4 space-y-4">
            <div>
              <Label className="text-sm font-medium">Dashboard Templates</Label>
              <div className="space-y-2 mt-2">
                {dashboardTemplates.map((template) => (
                  <Card
                    key={template.id}
                    className="cursor-pointer hover:bg-gray-50 transition-colors"
                    onClick={() => applyTemplate(template)}
                  >
                    <CardContent className="p-3">
                      <h4 className="font-medium">{template.name}</h4>
                      <p className="text-sm text-gray-600">{template.description}</p>
                      <Badge variant="outline" className="mt-2">
                        {template.layout.length} widgets
                      </Badge>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="settings" className="p-4 space-y-4">
            <div className="space-y-3">
              <div>
                <Label htmlFor="dashboard-name" className="text-sm">Dashboard Name</Label>
                <Input
                  id="dashboard-name"
                  value={dashboardName}
                  onChange={(e) => setDashboardName(e.target.value)}
                  className="h-8"
                />
              </div>
              
              <div>
                <Label htmlFor="dashboard-description" className="text-sm">Description</Label>
                <Input
                  id="dashboard-description"
                  value={dashboardDescription}
                  onChange={(e) => setDashboardDescription(e.target.value)}
                  className="h-8"
                />
              </div>

              <div className="pt-4 border-t">
                <Label className="text-sm font-medium">Dashboard Stats</Label>
                <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
                  <div>
                    <span className="text-gray-600">Widgets:</span>
                    <span className="ml-2 font-medium">{layout.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Charts:</span>
                    <span className="ml-2 font-medium">
                      {layout.filter(item => item.type === 'chart').length}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-xl font-semibold">{dashboardName}</h1>
              <Badge variant="outline">{layout.length} widgets</Badge>
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                onClick={() => setIsPreviewMode(!isPreviewMode)}
                className="flex items-center gap-2"
              >
                <Eye className="h-4 w-4" />
                {isPreviewMode ? 'Edit' : 'Preview'}
              </Button>
              
              <Button onClick={handleSave} className="flex items-center gap-2">
                <Save className="h-4 w-4" />
                Save Dashboard
              </Button>
            </div>
          </div>
        </div>

        {/* Dashboard Canvas */}
        <div className="flex-1 p-6 overflow-auto">
          {layout.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Grid className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Empty Dashboard</h3>
                <p className="text-gray-600 mb-4">Add widgets from the sidebar to get started</p>
                <Button onClick={() => addWidget('chart')} className="flex items-center gap-2">
                  <Plus className="h-4 w-4" />
                  Add Your First Widget
                </Button>
              </div>
            </div>
          ) : (
            <GridLayout
              layout={layout}
              onLayoutChange={handleLayoutChange}
              isDraggable={!isPreviewMode}
              isResizable={!isPreviewMode}
            >
              {layout.map(renderWidget)}
            </GridLayout>
          )}
        </div>
      </div>
    </div>
  )
}