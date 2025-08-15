'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  ResponsiveContainer,
  LineChart,
  BarChart,
  PieChart,
  ScatterChart,
  AreaChart,
  Line,
  Bar,
  Pie,
  Cell,
  Scatter,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Brush,
  ReferenceLine
} from 'recharts'
import { 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Download, 
  Filter,
  Maximize2,
  Settings,
  TrendingUp,
  Eye,
  ChevronRight
} from 'lucide-react'

interface InteractiveChartProps {
  data: any[]
  config: any
  interactive?: boolean
  onDataClick?: (data: any) => void
  onExport?: (format: string) => void
  className?: string
}

export function InteractiveChart({ 
  data, 
  config, 
  interactive = true, 
  onDataClick,
  onExport,
  className = ""
}: InteractiveChartProps) {
  const [zoomDomain, setZoomDomain] = useState<any>(null)
  const [selectedData, setSelectedData] = useState<any>(null)
  const [drillDownLevel, setDrillDownLevel] = useState(0)
  const [breadcrumbs, setBreadcrumbs] = useState<string[]>([])
  const [filteredData, setFilteredData] = useState(data)
  const chartRef = useRef<any>(null)

  useEffect(() => {
    setFilteredData(data)
  }, [data])

  const handleDataClick = (data: any, index: number) => {
    if (!interactive) return
    
    setSelectedData(data)
    
    // Handle drill-down if configured
    if (config.drillDown?.enabled && config.drillDown.levels) {
      const currentLevel = config.drillDown.levels[drillDownLevel]
      if (currentLevel && data[currentLevel]) {
        handleDrillDown(data[currentLevel])
      }
    }
    
    onDataClick?.(data)
  }

  const handleDrillDown = (value: string) => {
    if (drillDownLevel < config.drillDown.levels.length - 1) {
      const newLevel = drillDownLevel + 1
      const filterField = config.drillDown.levels[drillDownLevel]
      
      // Filter data based on drill-down
      const newFilteredData = data.filter(item => item[filterField] === value)
      setFilteredData(newFilteredData)
      setDrillDownLevel(newLevel)
      setBreadcrumbs([...breadcrumbs, value])
    }
  }

  const handleBreadcrumbClick = (index: number) => {
    if (index === -1) {
      // Reset to top level
      setFilteredData(data)
      setDrillDownLevel(0)
      setBreadcrumbs([])
    } else {
      // Go back to specific level
      const newLevel = index + 1
      setDrillDownLevel(newLevel)
      setBreadcrumbs(breadcrumbs.slice(0, newLevel))
      
      // Re-filter data
      let newData = data
      for (let i = 0; i <= index; i++) {
        const filterField = config.drillDown.levels[i]
        const filterValue = breadcrumbs[i]
        newData = newData.filter(item => item[filterField] === filterValue)
      }
      setFilteredData(newData)
    }
  }

  const handleZoom = (domain: any) => {
    setZoomDomain(domain)
  }

  const resetZoom = () => {
    setZoomDomain(null)
  }

  const handleExport = (format: string) => {
    onExport?.(format)
  }

  const renderChart = () => {
    const chartProps = {
      data: filteredData,
      margin: { top: 20, right: 30, left: 20, bottom: 5 },
      onClick: interactive ? handleDataClick : undefined
    }

    const colors = config.colors || ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00']

    switch (config.type) {
      case 'line':
        return (
          <LineChart {...chartProps}>
            {config.showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis 
              dataKey={config.xAxis.dataKey} 
              domain={zoomDomain?.x}
              label={{ value: config.xAxis.label, position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              domain={zoomDomain?.y}
              label={{ value: config.yAxis.label, angle: -90, position: 'insideLeft' }}
            />
            {config.showTooltip && <Tooltip />}
            {config.showLegend && <Legend />}
            <Line
              type="monotone"
              dataKey={config.yAxis.dataKey}
              stroke={colors[0]}
              strokeWidth={config.strokeWidth || 2}
              dot={{ fill: colors[0], strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: colors[0], strokeWidth: 2 }}
            />
            {interactive && <Brush dataKey={config.xAxis.dataKey} height={30} />}
          </LineChart>
        )

      case 'bar':
        return (
          <BarChart {...chartProps}>
            {config.showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis 
              dataKey={config.xAxis.dataKey}
              label={{ value: config.xAxis.label, position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              label={{ value: config.yAxis.label, angle: -90, position: 'insideLeft' }}
            />
            {config.showTooltip && <Tooltip />}
            {config.showLegend && <Legend />}
            <Bar
              dataKey={config.yAxis.dataKey}
              fill={colors[0]}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        )

      case 'pie':
        return (
          <PieChart {...chartProps}>
            {config.showTooltip && <Tooltip />}
            {config.showLegend && <Legend />}
            <Pie
              data={filteredData}
              cx="50%"
              cy="50%"
              innerRadius={config.innerRadius || 0}
              outerRadius={config.outerRadius || 80}
              paddingAngle={5}
              dataKey={config.yAxis.dataKey}
              nameKey={config.xAxis.dataKey}
            >
              {filteredData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Pie>
          </PieChart>
        )

      case 'scatter':
        return (
          <ScatterChart {...chartProps}>
            {config.showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis 
              dataKey={config.xAxis.dataKey}
              type="number"
              domain={zoomDomain?.x}
              label={{ value: config.xAxis.label, position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              dataKey={config.yAxis.dataKey}
              type="number"
              domain={zoomDomain?.y}
              label={{ value: config.yAxis.label, angle: -90, position: 'insideLeft' }}
            />
            {config.showTooltip && <Tooltip />}
            {config.showLegend && <Legend />}
            <Scatter
              data={filteredData}
              fill={colors[0]}
            />
          </ScatterChart>
        )

      case 'area':
        return (
          <AreaChart {...chartProps}>
            {config.showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis 
              dataKey={config.xAxis.dataKey}
              domain={zoomDomain?.x}
              label={{ value: config.xAxis.label, position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              domain={zoomDomain?.y}
              label={{ value: config.yAxis.label, angle: -90, position: 'insideLeft' }}
            />
            {config.showTooltip && <Tooltip />}
            {config.showLegend && <Legend />}
            <Area
              type="monotone"
              dataKey={config.yAxis.dataKey}
              stroke={colors[0]}
              fill={colors[0]}
              fillOpacity={0.6}
            />
            {interactive && <Brush dataKey={config.xAxis.dataKey} height={30} />}
          </AreaChart>
        )

      default:
        return <div>Unsupported chart type: {config.type}</div>
    }
  }

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            {config.title}
          </CardTitle>
          
          {interactive && (
            <div className="flex items-center gap-2">
              {zoomDomain && (
                <Button variant="outline" size="sm" onClick={resetZoom}>
                  <RotateCcw className="h-4 w-4" />
                </Button>
              )}
              
              <Button variant="outline" size="sm" onClick={() => handleExport('png')}>
                <Download className="h-4 w-4" />
              </Button>
              
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>

        {/* Breadcrumbs for drill-down */}
        {config.drillDown?.enabled && config.drillDown.showBreadcrumbs && breadcrumbs.length > 0 && (
          <div className="flex items-center gap-1 text-sm text-gray-600">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleBreadcrumbClick(-1)}
              className="h-6 px-2"
            >
              All
            </Button>
            {breadcrumbs.map((crumb, index) => (
              <React.Fragment key={index}>
                <ChevronRight className="h-3 w-3" />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleBreadcrumbClick(index)}
                  className="h-6 px-2"
                >
                  {crumb}
                </Button>
              </React.Fragment>
            ))}
          </div>
        )}
      </CardHeader>

      <CardContent>
        <div style={{ width: '100%', height: config.height || 400 }}>
          <ResponsiveContainer>
            {renderChart()}
          </ResponsiveContainer>
        </div>

        {/* Chart Statistics */}
        {selectedData && (
          <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <h4 className="font-medium mb-2">Selected Data Point</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {Object.entries(selectedData).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-gray-600">{key}:</span>
                  <span className="font-medium">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Interactive Features Info */}
        {interactive && (
          <div className="mt-4 flex flex-wrap gap-2">
            <Badge variant="outline" className="text-xs">
              <Eye className="h-3 w-3 mr-1" />
              Click data points
            </Badge>
            {config.type === 'line' || config.type === 'area' ? (
              <Badge variant="outline" className="text-xs">
                <ZoomIn className="h-3 w-3 mr-1" />
                Brush to zoom
              </Badge>
            ) : null}
            {config.drillDown?.enabled && (
              <Badge variant="outline" className="text-xs">
                <Filter className="h-3 w-3 mr-1" />
                Drill-down enabled
              </Badge>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}