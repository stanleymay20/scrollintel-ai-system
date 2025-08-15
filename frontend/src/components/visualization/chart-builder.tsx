'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { 
  BarChart3, 
  LineChart, 
  PieChart, 
  Scatter, 
  TrendingUp,
  Settings,
  Palette,
  Filter,
  Download,
  Eye,
  Lightbulb
} from 'lucide-react'

interface ChartBuilderProps {
  data: any[]
  onChartCreate: (chartConfig: any) => void
  onPreview: (chartConfig: any) => void
}

const chartTypes = [
  { value: 'line', label: 'Line Chart', icon: LineChart },
  { value: 'bar', label: 'Bar Chart', icon: BarChart3 },
  { value: 'pie', label: 'Pie Chart', icon: PieChart },
  { value: 'scatter', label: 'Scatter Plot', icon: Scatter },
  { value: 'area', label: 'Area Chart', icon: TrendingUp },
]

const colorSchemes = [
  { value: 'default', label: 'Default', colors: ['#8884d8', '#82ca9d', '#ffc658'] },
  { value: 'professional', label: 'Professional', colors: ['#1f77b4', '#ff7f0e', '#2ca02c'] },
  { value: 'modern', label: 'Modern', colors: ['#667eea', '#764ba2', '#f093fb'] },
  { value: 'scrollintel', label: 'ScrollIntel', colors: ['#6366f1', '#8b5cf6', '#06b6d4'] },
]

export function ChartBuilder({ data, onChartCreate, onPreview }: ChartBuilderProps) {
  const [chartConfig, setChartConfig] = useState({
    chart_type: 'bar',
    title: '',
    x_axis: '',
    y_axis: '',
    color_scheme: 'default',
    width: 800,
    height: 400,
    interactive: true,
    show_legend: true,
    show_grid: true,
    show_tooltip: true,
    aggregation: '',
    group_by: ''
  })

  const [columns, setColumns] = useState<string[]>([])
  const [suggestions, setSuggestions] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (data && data.length > 0) {
      const cols = Object.keys(data[0])
      setColumns(cols)
      
      // Get chart suggestions
      getSuggestions()
    }
  }, [data])

  const getSuggestions = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/visualization/charts/suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      })
      
      if (response.ok) {
        const result = await response.json()
        setSuggestions(result.suggestions || [])
      }
    } catch (error) {
      console.error('Error getting suggestions:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleConfigChange = (field: string, value: any) => {
    setChartConfig(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const handleSuggestionSelect = (suggestion: any) => {
    setChartConfig(prev => ({
      ...prev,
      chart_type: suggestion.type,
      title: suggestion.title,
      x_axis: suggestion.x_axis,
      y_axis: suggestion.y_axis
    }))
  }

  const handlePreview = () => {
    onPreview(chartConfig)
  }

  const handleCreate = () => {
    onChartCreate(chartConfig)
  }

  const getNumericColumns = () => {
    if (!data || data.length === 0) return []
    
    return columns.filter(col => {
      const sample = data[0][col]
      return typeof sample === 'number' || !isNaN(Number(sample))
    })
  }

  const getCategoricalColumns = () => {
    if (!data || data.length === 0) return []
    
    return columns.filter(col => {
      const sample = data[0][col]
      return typeof sample === 'string' || isNaN(Number(sample))
    })
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Chart Builder
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="style">Style</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
            <TabsTrigger value="suggestions">
              <Lightbulb className="h-4 w-4 mr-1" />
              Suggestions
            </TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="chart-type">Chart Type</Label>
                <Select
                  value={chartConfig.chart_type}
                  onValueChange={(value) => handleConfigChange('chart_type', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select chart type" />
                  </SelectTrigger>
                  <SelectContent>
                    {chartTypes.map((type) => {
                      const Icon = type.icon
                      return (
                        <SelectItem key={type.value} value={type.value}>
                          <div className="flex items-center gap-2">
                            <Icon className="h-4 w-4" />
                            {type.label}
                          </div>
                        </SelectItem>
                      )
                    })}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="title">Chart Title</Label>
                <Input
                  id="title"
                  value={chartConfig.title}
                  onChange={(e) => handleConfigChange('title', e.target.value)}
                  placeholder="Enter chart title"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="x-axis">X-Axis</Label>
                <Select
                  value={chartConfig.x_axis}
                  onValueChange={(value) => handleConfigChange('x_axis', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select X-axis column" />
                  </SelectTrigger>
                  <SelectContent>
                    {columns.map((col) => (
                      <SelectItem key={col} value={col}>
                        {col}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="y-axis">Y-Axis</Label>
                <Select
                  value={chartConfig.y_axis}
                  onValueChange={(value) => handleConfigChange('y_axis', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select Y-axis column" />
                  </SelectTrigger>
                  <SelectContent>
                    {getNumericColumns().map((col) => (
                      <SelectItem key={col} value={col}>
                        {col}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="style" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="color-scheme">Color Scheme</Label>
                <Select
                  value={chartConfig.color_scheme}
                  onValueChange={(value) => handleConfigChange('color_scheme', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select color scheme" />
                  </SelectTrigger>
                  <SelectContent>
                    {colorSchemes.map((scheme) => (
                      <SelectItem key={scheme.value} value={scheme.value}>
                        <div className="flex items-center gap-2">
                          <div className="flex gap-1">
                            {scheme.colors.map((color, idx) => (
                              <div
                                key={idx}
                                className="w-3 h-3 rounded-full"
                                style={{ backgroundColor: color }}
                              />
                            ))}
                          </div>
                          {scheme.label}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="dimensions">Dimensions</Label>
                <div className="flex gap-2">
                  <Input
                    type="number"
                    value={chartConfig.width}
                    onChange={(e) => handleConfigChange('width', parseInt(e.target.value))}
                    placeholder="Width"
                  />
                  <Input
                    type="number"
                    value={chartConfig.height}
                    onChange={(e) => handleConfigChange('height', parseInt(e.target.value))}
                    placeholder="Height"
                  />
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Display Options</Label>
              <div className="flex flex-wrap gap-2">
                <Badge
                  variant={chartConfig.show_legend ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => handleConfigChange('show_legend', !chartConfig.show_legend)}
                >
                  Legend
                </Badge>
                <Badge
                  variant={chartConfig.show_grid ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => handleConfigChange('show_grid', !chartConfig.show_grid)}
                >
                  Grid
                </Badge>
                <Badge
                  variant={chartConfig.show_tooltip ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => handleConfigChange('show_tooltip', !chartConfig.show_tooltip)}
                >
                  Tooltip
                </Badge>
                <Badge
                  variant={chartConfig.interactive ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => handleConfigChange('interactive', !chartConfig.interactive)}
                >
                  Interactive
                </Badge>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="aggregation">Aggregation</Label>
                <Select
                  value={chartConfig.aggregation}
                  onValueChange={(value) => handleConfigChange('aggregation', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select aggregation" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">None</SelectItem>
                    <SelectItem value="sum">Sum</SelectItem>
                    <SelectItem value="avg">Average</SelectItem>
                    <SelectItem value="count">Count</SelectItem>
                    <SelectItem value="max">Maximum</SelectItem>
                    <SelectItem value="min">Minimum</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="group-by">Group By</Label>
                <Select
                  value={chartConfig.group_by}
                  onValueChange={(value) => handleConfigChange('group_by', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select grouping column" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">None</SelectItem>
                    {getCategoricalColumns().map((col) => (
                      <SelectItem key={col} value={col}>
                        {col}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="suggestions" className="space-y-4">
            <div className="space-y-2">
              <Label>AI-Powered Chart Suggestions</Label>
              {loading ? (
                <div className="text-center py-4">Loading suggestions...</div>
              ) : suggestions.length > 0 ? (
                <div className="grid gap-2">
                  {suggestions.map((suggestion, idx) => (
                    <Card
                      key={idx}
                      className="cursor-pointer hover:bg-gray-50 transition-colors"
                      onClick={() => handleSuggestionSelect(suggestion)}
                    >
                      <CardContent className="p-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="font-medium">{suggestion.title}</h4>
                            <p className="text-sm text-gray-600">
                              {suggestion.type} chart with {suggestion.x_axis} vs {suggestion.y_axis}
                            </p>
                          </div>
                          <Badge variant="outline">
                            {Math.round(suggestion.confidence * 100)}% match
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4 text-gray-500">
                  No suggestions available. Upload data to get AI-powered recommendations.
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>

        <div className="flex justify-between mt-6">
          <Button variant="outline" onClick={handlePreview} className="flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Preview
          </Button>
          <Button onClick={handleCreate} className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Create Chart
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}