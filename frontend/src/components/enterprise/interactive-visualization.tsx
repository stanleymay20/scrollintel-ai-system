'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  BarChart3, 
  LineChart, 
  PieChart, 
  TrendingUp, 
  Filter, 
  Download, 
  Share, 
  Maximize2,
  Settings,
  RefreshCw,
  Eye,
  EyeOff,
  Layers,
  Grid,
  Zap,
  Target,
  Users,
  DollarSign,
  Clock,
  Database
} from 'lucide-react'

interface VisualizationData {
  id: string
  title: string
  type: 'bar' | 'line' | 'pie' | 'area' | 'scatter' | 'heatmap' | 'gauge'
  data: any[]
  config: VisualizationConfig
  lastUpdated: Date
  isRealTime: boolean
}

interface VisualizationConfig {
  xAxis?: string
  yAxis?: string
  groupBy?: string
  aggregation?: 'sum' | 'avg' | 'count' | 'max' | 'min'
  timeRange?: string
  filters?: Filter[]
  colors?: string[]
  showLegend?: boolean
  showGrid?: boolean
  animate?: boolean
}

interface Filter {
  field: string
  operator: 'equals' | 'contains' | 'greater' | 'less' | 'between'
  value: any
}

interface InteractiveVisualizationProps {
  data?: VisualizationData[]
  onVisualizationChange?: (viz: VisualizationData) => void
  className?: string
}

export function InteractiveVisualization({ data, onVisualizationChange, className }: InteractiveVisualizationProps) {
  const [visualizations, setVisualizations] = useState<VisualizationData[]>([])
  const [selectedViz, setSelectedViz] = useState<VisualizationData | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [activeFilters, setActiveFilters] = useState<Filter[]>([])
  const [refreshInterval, setRefreshInterval] = useState<number>(30000)
  const [isAutoRefresh, setIsAutoRefresh] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Sample visualization data
  const sampleVisualizations: VisualizationData[] = [
    {
      id: '1',
      title: 'Revenue Trends',
      type: 'line',
      data: [
        { month: 'Jan', revenue: 2100000, target: 2000000 },
        { month: 'Feb', revenue: 2300000, target: 2100000 },
        { month: 'Mar', revenue: 2400000, target: 2200000 },
        { month: 'Apr', revenue: 2600000, target: 2300000 },
        { month: 'May', revenue: 2800000, target: 2400000 },
        { month: 'Jun', revenue: 2900000, target: 2500000 }
      ],
      config: {
        xAxis: 'month',
        yAxis: 'revenue',
        showLegend: true,
        showGrid: true,
        animate: true,
        colors: ['#3b82f6', '#10b981']
      },
      lastUpdated: new Date(),
      isRealTime: true
    },
    {
      id: '2',
      title: 'Agent Performance Distribution',
      type: 'bar',
      data: [
        { agent: 'CTO Agent', accuracy: 94.7, requests: 1247 },
        { agent: 'Data Scientist', accuracy: 96.3, requests: 892 },
        { agent: 'BI Agent', accuracy: 91.2, requests: 1156 },
        { agent: 'ML Engineer', accuracy: 93.8, requests: 734 },
        { agent: 'QA Agent', accuracy: 89.5, requests: 623 }
      ],
      config: {
        xAxis: 'agent',
        yAxis: 'accuracy',
        showLegend: false,
        showGrid: true,
        colors: ['#8b5cf6']
      },
      lastUpdated: new Date(),
      isRealTime: false
    },
    {
      id: '3',
      title: 'System Resource Usage',
      type: 'pie',
      data: [
        { resource: 'CPU', usage: 23.4, color: '#3b82f6' },
        { resource: 'Memory', usage: 67.8, color: '#10b981' },
        { resource: 'Storage', usage: 45.2, color: '#f59e0b' },
        { resource: 'Network', usage: 78.9, color: '#ef4444' }
      ],
      config: {
        showLegend: true,
        colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
      },
      lastUpdated: new Date(),
      isRealTime: true
    },
    {
      id: '4',
      title: 'Business Impact Metrics',
      type: 'gauge',
      data: [
        { metric: 'Cost Savings', value: 890000, target: 1000000, unit: '$' },
        { metric: 'Efficiency Gain', value: 67, target: 70, unit: '%' },
        { metric: 'Customer Satisfaction', value: 94.2, target: 95, unit: '%' }
      ],
      config: {
        showLegend: false,
        colors: ['#10b981', '#f59e0b', '#3b82f6']
      },
      lastUpdated: new Date(),
      isRealTime: true
    }
  ]

  useEffect(() => {
    setVisualizations(data || sampleVisualizations)
    if (!selectedViz && (data || sampleVisualizations).length > 0) {
      setSelectedViz((data || sampleVisualizations)[0])
    }
  }, [data])

  useEffect(() => {
    if (!isAutoRefresh) return

    const interval = setInterval(() => {
      refreshVisualizationData()
    }, refreshInterval)

    return () => clearInterval(interval)
  }, [isAutoRefresh, refreshInterval])

  const refreshVisualizationData = async () => {
    try {
      // Simulate API call to refresh data
      const response = await fetch('/api/visualizations/refresh')
      const updatedData = await response.json()
      
      setVisualizations(prev => prev.map(viz => {
        const updated = updatedData.find((u: any) => u.id === viz.id)
        return updated ? { ...viz, ...updated, lastUpdated: new Date() } : viz
      }))
    } catch (error) {
      console.error('Failed to refresh visualization data:', error)
    }
  }

  const renderVisualization = (viz: VisualizationData) => {
    const { type, data, config } = viz

    switch (type) {
      case 'line':
        return renderLineChart(data, config)
      case 'bar':
        return renderBarChart(data, config)
      case 'pie':
        return renderPieChart(data, config)
      case 'gauge':
        return renderGaugeChart(data, config)
      default:
        return renderPlaceholder(type)
    }
  }

  const renderLineChart = (data: any[], config: VisualizationConfig) => (
    <div className="h-64 flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg">
      <div className="text-center">
        <LineChart className="h-16 w-16 mx-auto mb-4 text-blue-600" />
        <p className="text-sm text-blue-800">Line Chart Visualization</p>
        <p className="text-xs text-blue-600 mt-1">{data.length} data points</p>
      </div>
    </div>
  )

  const renderBarChart = (data: any[], config: VisualizationConfig) => (
    <div className="h-64 flex items-center justify-center bg-gradient-to-br from-purple-50 to-violet-100 rounded-lg">
      <div className="text-center">
        <BarChart3 className="h-16 w-16 mx-auto mb-4 text-purple-600" />
        <p className="text-sm text-purple-800">Bar Chart Visualization</p>
        <p className="text-xs text-purple-600 mt-1">{data.length} categories</p>
      </div>
    </div>
  )

  const renderPieChart = (data: any[], config: VisualizationConfig) => (
    <div className="h-64 flex items-center justify-center bg-gradient-to-br from-green-50 to-emerald-100 rounded-lg">
      <div className="text-center">
        <PieChart className="h-16 w-16 mx-auto mb-4 text-green-600" />
        <p className="text-sm text-green-800">Pie Chart Visualization</p>
        <p className="text-xs text-green-600 mt-1">{data.length} segments</p>
      </div>
    </div>
  )

  const renderGaugeChart = (data: any[], config: VisualizationConfig) => (
    <div className="h-64 flex items-center justify-center bg-gradient-to-br from-orange-50 to-amber-100 rounded-lg">
      <div className="text-center">
        <Target className="h-16 w-16 mx-auto mb-4 text-orange-600" />
        <p className="text-sm text-orange-800">Gauge Chart Visualization</p>
        <p className="text-xs text-orange-600 mt-1">{data.length} metrics</p>
      </div>
    </div>
  )

  const renderPlaceholder = (type: string) => (
    <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
      <div className="text-center">
        <Grid className="h-16 w-16 mx-auto mb-4 text-gray-400" />
        <p className="text-sm text-gray-600">{type} Chart</p>
        <p className="text-xs text-gray-500 mt-1">Visualization placeholder</p>
      </div>
    </div>
  )

  const handleVisualizationSelect = (viz: VisualizationData) => {
    setSelectedViz(viz)
    if (onVisualizationChange) {
      onVisualizationChange(viz)
    }
  }

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }

  const exportVisualization = (format: 'png' | 'pdf' | 'csv') => {
    // Implementation for exporting visualization
    console.log(`Exporting ${selectedViz?.title} as ${format}`)
  }

  return (
    <div className={`space-y-6 ${className} ${isFullscreen ? 'fixed inset-0 z-50 bg-background p-6' : ''}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Interactive Visualizations</h2>
          <p className="text-muted-foreground">
            Explore your business data with dynamic, real-time visualizations
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAutoRefresh(!isAutoRefresh)}
          >
            {isAutoRefresh ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            Auto-refresh
          </Button>
          <Button variant="outline" size="sm" onClick={refreshVisualizationData}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toggleFullscreen}>
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Visualization List */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">Available Charts</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {visualizations.map((viz) => (
              <div
                key={viz.id}
                onClick={() => handleVisualizationSelect(viz)}
                className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                  selectedViz?.id === viz.id 
                    ? 'border-primary bg-primary/5' 
                    : 'border-border hover:bg-muted/50'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-sm">{viz.title}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-xs capitalize">
                        {viz.type}
                      </Badge>
                      {viz.isRealTime && (
                        <Badge variant="outline" className="text-xs bg-green-50 text-green-700">
                          Live
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Updated: {viz.lastUpdated.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Main Visualization Area */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                {selectedViz?.type === 'line' && <LineChart className="h-5 w-5" />}
                {selectedViz?.type === 'bar' && <BarChart3 className="h-5 w-5" />}
                {selectedViz?.type === 'pie' && <PieChart className="h-5 w-5" />}
                {selectedViz?.type === 'gauge' && <Target className="h-5 w-5" />}
                {selectedViz?.title || 'Select a Visualization'}
              </CardTitle>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm">
                  <Filter className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="sm">
                  <Settings className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="sm">
                  <Share className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {selectedViz ? (
              <div className="space-y-4">
                {renderVisualization(selectedViz)}
                
                {/* Visualization Controls */}
                <div className="flex items-center justify-between pt-4 border-t">
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span>Data Points: {selectedViz.data.length}</span>
                    <span>Last Updated: {selectedViz.lastUpdated.toLocaleString()}</span>
                    {selectedViz.isRealTime && (
                      <Badge variant="outline" className="bg-green-50 text-green-700">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-1" />
                        Live Data
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Select defaultValue="png">
                      <SelectTrigger className="w-24">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="png">PNG</SelectItem>
                        <SelectItem value="pdf">PDF</SelectItem>
                        <SelectItem value="csv">CSV</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button size="sm" onClick={() => exportVisualization('png')}>
                      Export
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <BarChart3 className="h-16 w-16 mx-auto mb-4" />
                  <p>Select a visualization from the list to view</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Quick Insights Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Quick Insights
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="h-5 w-5 text-blue-600" />
                <h4 className="font-medium text-blue-900">Revenue Growth</h4>
              </div>
              <p className="text-2xl font-bold text-blue-900">+15.3%</p>
              <p className="text-sm text-blue-700">Compared to last quarter</p>
            </div>
            
            <div className="p-4 bg-green-50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Users className="h-5 w-5 text-green-600" />
                <h4 className="font-medium text-green-900">Agent Efficiency</h4>
              </div>
              <p className="text-2xl font-bold text-green-900">94.2%</p>
              <p className="text-sm text-green-700">Average accuracy score</p>
            </div>
            
            <div className="p-4 bg-purple-50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Database className="h-5 w-5 text-purple-600" />
                <h4 className="font-medium text-purple-900">Data Processing</h4>
              </div>
              <p className="text-2xl font-bold text-purple-900">847 GB/hr</p>
              <p className="text-sm text-purple-700">Real-time throughput</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}