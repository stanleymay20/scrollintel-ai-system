'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  BarChart3, 
  PieChart, 
  LineChart, 
  Upload, 
  Download, 
  Filter,
  Settings,
  Eye,
  Plus,
  Search,
  Grid,
  List,
  Trash2,
  Edit,
  Share
} from 'lucide-react'

import { ChartBuilder } from './chart-builder'
import { InteractiveChart } from './interactive-chart'
import { ExportDialog } from './export-dialog'
import { DashboardCustomizer } from './dashboard-customizer'

interface VisualizationInterfaceProps {
  data?: any[]
  onDataUpload?: (data: any[]) => void
}

export function VisualizationInterface({ data = [], onDataUpload }: VisualizationInterfaceProps) {
  const [activeTab, setActiveTab] = useState('charts')
  const [charts, setCharts] = useState<any[]>([])
  const [dashboards, setDashboards] = useState<any[]>([])
  const [selectedChart, setSelectedChart] = useState<any>(null)
  const [selectedDashboard, setSelectedDashboard] = useState<any>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState('all')

  useEffect(() => {
    // Load saved charts and dashboards
    loadCharts()
    loadDashboards()
  }, [])

  const loadCharts = async () => {
    // In a real implementation, this would fetch from the API
    const mockCharts = [
      {
        id: '1',
        name: 'Sales Trend',
        type: 'line',
        created_at: '2024-01-15T10:00:00Z',
        data_points: 150,
        config: {
          chart_type: 'line',
          title: 'Sales Trend',
          x_axis: 'date',
          y_axis: 'sales'
        }
      },
      {
        id: '2',
        name: 'Revenue by Category',
        type: 'bar',
        created_at: '2024-01-14T15:30:00Z',
        data_points: 12,
        config: {
          chart_type: 'bar',
          title: 'Revenue by Category',
          x_axis: 'category',
          y_axis: 'revenue'
        }
      }
    ]
    setCharts(mockCharts)
  }

  const loadDashboards = async () => {
    // In a real implementation, this would fetch from the API
    const mockDashboards = [
      {
        id: '1',
        name: 'Executive Overview',
        description: 'High-level business metrics',
        chart_count: 4,
        created_at: '2024-01-10T09:00:00Z'
      }
    ]
    setDashboards(mockDashboards)
  }

  const handleChartCreate = async (chartConfig: any) => {
    try {
      const response = await fetch('/api/visualization/charts/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: data,
          config: chartConfig
        })
      })

      if (response.ok) {
        const result = await response.json()
        if (result.success) {
          const newChart = {
            id: result.data.id,
            name: chartConfig.title,
            type: chartConfig.chart_type,
            created_at: new Date().toISOString(),
            data_points: data.length,
            config: chartConfig,
            data: result.data.data
          }
          setCharts(prev => [newChart, ...prev])
          setSelectedChart(newChart)
        }
      }
    } catch (error) {
      console.error('Error creating chart:', error)
    }
  }

  const handleChartPreview = (chartConfig: any) => {
    const previewChart = {
      id: 'preview',
      name: chartConfig.title,
      type: chartConfig.chart_type,
      config: chartConfig,
      data: data
    }
    setSelectedChart(previewChart)
  }

  const handleDashboardCreate = async (name: string, template?: string) => {
    try {
      const response = await fetch('/api/visualization/dashboards/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          template,
          charts: charts.map(c => c.id)
        })
      })

      if (response.ok) {
        const result = await response.json()
        if (result.success) {
          loadDashboards()
        }
      }
    } catch (error) {
      console.error('Error creating dashboard:', error)
    }
  }

  const handleExport = async (format: string) => {
    if (selectedChart) {
      // Export single chart
      console.log(`Exporting chart ${selectedChart.id} as ${format}`)
    }
  }

  const filteredCharts = charts.filter(chart => {
    const matchesSearch = chart.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesType = filterType === 'all' || chart.type === filterType
    return matchesSearch && matchesType
  })

  const filteredDashboards = dashboards.filter(dashboard =>
    dashboard.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const renderChartCard = (chart: any) => (
    <Card
      key={chart.id}
      className={`cursor-pointer transition-all hover:shadow-md ${
        selectedChart?.id === chart.id ? 'ring-2 ring-blue-500' : ''
      }`}
      onClick={() => setSelectedChart(chart)}
    >
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            {chart.type === 'line' && <LineChart className="h-4 w-4" />}
            {chart.type === 'bar' && <BarChart3 className="h-4 w-4" />}
            {chart.type === 'pie' && <PieChart className="h-4 w-4" />}
            {chart.name}
          </CardTitle>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={(e) => e.stopPropagation()}>
              <Edit className="h-3 w-3" />
            </Button>
            <Button variant="ghost" size="sm" onClick={(e) => e.stopPropagation()}>
              <Share className="h-3 w-3" />
            </Button>
            <Button variant="ghost" size="sm" onClick={(e) => e.stopPropagation()}>
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Type:</span>
            <Badge variant="outline">{chart.type}</Badge>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Data Points:</span>
            <span>{chart.data_points?.toLocaleString()}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Created:</span>
            <span>{new Date(chart.created_at).toLocaleDateString()}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )

  const renderDashboardCard = (dashboard: any) => (
    <Card
      key={dashboard.id}
      className="cursor-pointer transition-all hover:shadow-md"
      onClick={() => setSelectedDashboard(dashboard)}
    >
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Grid className="h-4 w-4" />
            {dashboard.name}
          </CardTitle>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={(e) => e.stopPropagation()}>
              <Edit className="h-3 w-3" />
            </Button>
            <Button variant="ghost" size="sm" onClick={(e) => e.stopPropagation()}>
              <Share className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <p className="text-sm text-gray-600">{dashboard.description}</p>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Charts:</span>
            <Badge variant="outline">{dashboard.chart_count}</Badge>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Created:</span>
            <span>{new Date(dashboard.created_at).toLocaleDateString()}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 overflow-y-auto">
        <div className="p-4 border-b">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Visualization Studio
          </h2>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="charts">Charts</TabsTrigger>
            <TabsTrigger value="dashboards">Dashboards</TabsTrigger>
            <TabsTrigger value="builder">Builder</TabsTrigger>
          </TabsList>

          <div className="p-4">
            {/* Search and Filter */}
            <div className="space-y-3 mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>

              {activeTab === 'charts' && (
                <Select value={filterType} onValueChange={setFilterType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Filter by type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="line">Line Charts</SelectItem>
                    <SelectItem value="bar">Bar Charts</SelectItem>
                    <SelectItem value="pie">Pie Charts</SelectItem>
                    <SelectItem value="scatter">Scatter Plots</SelectItem>
                  </SelectContent>
                </Select>
              )}

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1">
                  <Button
                    variant={viewMode === 'grid' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setViewMode('grid')}
                  >
                    <Grid className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={viewMode === 'list' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setViewMode('list')}
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </div>

                <Button size="sm" onClick={() => {
                  if (activeTab === 'charts') {
                    setActiveTab('builder')
                  } else if (activeTab === 'dashboards') {
                    handleDashboardCreate('New Dashboard')
                  }
                }}>
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <TabsContent value="charts" className="space-y-3 mt-0">
              {filteredCharts.length === 0 ? (
                <div className="text-center py-8">
                  <BarChart3 className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                  <p className="text-gray-600 mb-3">No charts found</p>
                  <Button size="sm" onClick={() => setActiveTab('builder')}>
                    Create Your First Chart
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {filteredCharts.map(renderChartCard)}
                </div>
              )}
            </TabsContent>

            <TabsContent value="dashboards" className="space-y-3 mt-0">
              {filteredDashboards.length === 0 ? (
                <div className="text-center py-8">
                  <Grid className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                  <p className="text-gray-600 mb-3">No dashboards found</p>
                  <Button size="sm" onClick={() => handleDashboardCreate('New Dashboard')}>
                    Create Your First Dashboard
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {filteredDashboards.map(renderDashboardCard)}
                </div>
              )}
            </TabsContent>

            <TabsContent value="builder" className="mt-0">
              {data.length === 0 ? (
                <div className="text-center py-8">
                  <Upload className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                  <p className="text-gray-600 mb-3">Upload data to start building charts</p>
                  <Button size="sm" onClick={() => onDataUpload?.([])}>
                    Upload Data
                  </Button>
                </div>
              ) : (
                <div className="text-sm text-gray-600">
                  <p>Data loaded: {data.length} rows</p>
                  <p>Ready to build charts!</p>
                </div>
              )}
            </TabsContent>
          </div>
        </Tabs>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {selectedChart && (
                <>
                  <h1 className="text-xl font-semibold">{selectedChart.name}</h1>
                  <Badge variant="outline">{selectedChart.type}</Badge>
                </>
              )}
              {selectedDashboard && (
                <>
                  <h1 className="text-xl font-semibold">{selectedDashboard.name}</h1>
                  <Badge variant="outline">{selectedDashboard.chart_count} charts</Badge>
                </>
              )}
              {!selectedChart && !selectedDashboard && activeTab === 'builder' && (
                <h1 className="text-xl font-semibold">Chart Builder</h1>
              )}
            </div>
            
            <div className="flex items-center gap-2">
              {selectedChart && (
                <>
                  <Button variant="outline" size="sm">
                    <Eye className="h-4 w-4 mr-2" />
                    Preview
                  </Button>
                  <ExportDialog
                    trigger={
                      <Button variant="outline" size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Export
                      </Button>
                    }
                    chartIds={[selectedChart.id]}
                    onExport={handleExport}
                  />
                </>
              )}
              
              {selectedDashboard && (
                <Button variant="outline" size="sm">
                  <Settings className="h-4 w-4 mr-2" />
                  Customize
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-auto">
          {activeTab === 'builder' && data.length > 0 && (
            <div className="p-6">
              <ChartBuilder
                data={data}
                onChartCreate={handleChartCreate}
                onPreview={handleChartPreview}
              />
            </div>
          )}

          {selectedChart && activeTab !== 'builder' && (
            <div className="p-6">
              <InteractiveChart
                data={selectedChart.data || data}
                config={selectedChart.config}
                interactive={true}
                onDataClick={(data) => console.log('Data clicked:', data)}
                onExport={handleExport}
              />
            </div>
          )}

          {selectedDashboard && (
            <div className="h-full">
              <DashboardCustomizer
                dashboardId={selectedDashboard.id}
                charts={charts}
                onSave={(config) => console.log('Dashboard saved:', config)}
              />
            </div>
          )}

          {!selectedChart && !selectedDashboard && activeTab !== 'builder' && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <BarChart3 className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {activeTab === 'charts' ? 'Select a Chart' : 'Select a Dashboard'}
                </h3>
                <p className="text-gray-600">
                  {activeTab === 'charts' 
                    ? 'Choose a chart from the sidebar to view and interact with it'
                    : 'Choose a dashboard from the sidebar to customize it'
                  }
                </p>
              </div>
            </div>
          )}

          {activeTab === 'builder' && data.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Upload className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Upload Data</h3>
                <p className="text-gray-600 mb-4">
                  Upload your data to start creating beautiful visualizations
                </p>
                <Button onClick={() => onDataUpload?.([])}>
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Data File
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}