/**
 * Template Preview Component
 * 
 * Provides real-time preview of dashboard templates with live data
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { 
  Eye, EyeOff, RefreshCw, Maximize2, Minimize2, 
  Smartphone, Tablet, Monitor, Settings, Play, Pause
} from 'lucide-react';
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell,
  AreaChart, Area, ScatterChart, Scatter
} from 'recharts';
import { DashboardTemplate, WidgetConfig } from './dashboard-builder';
import { useWebSocket } from '@/hooks/useWebSocket';

interface TemplatePreviewProps {
  template: DashboardTemplate;
  isLivePreview?: boolean;
  currentBreakpoint?: string;
  onBreakpointChange?: (breakpoint: string) => void;
  className?: string;
}

interface LiveDataPoint {
  timestamp: string;
  value: number;
  category?: string;
  label?: string;
}

interface WidgetData {
  [widgetId: string]: {
    data: LiveDataPoint[];
    metadata: {
      last_updated: string;
      source: string;
      status: 'loading' | 'success' | 'error';
      error_message?: string;
    };
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d'];

// Mock data generator for preview
const generateMockData = (widgetType: string, dataSource: string): LiveDataPoint[] => {
  const baseData = [];
  const now = new Date();
  
  for (let i = 0; i < 10; i++) {
    const timestamp = new Date(now.getTime() - (9 - i) * 24 * 60 * 60 * 1000);
    baseData.push({
      timestamp: timestamp.toISOString(),
      value: Math.floor(Math.random() * 1000) + 100,
      category: `Category ${i % 3 + 1}`,
      label: `Item ${i + 1}`
    });
  }
  
  return baseData;
};

// Widget Renderer Component
const WidgetRenderer: React.FC<{
  widget: WidgetConfig;
  data: LiveDataPoint[];
  isLoading: boolean;
  error?: string;
  currentBreakpoint: string;
}> = ({ widget, data, isLoading, error, currentBreakpoint }) => {
  const position = widget.responsive_config?.[currentBreakpoint as keyof typeof widget.responsive_config] || widget.position;
  
  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-32">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span className="ml-2">Loading...</span>
        </div>
      );
    }

    if (error) {
      return (
        <Alert className="m-2">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      );
    }

    switch (widget.type) {
      case 'metric_card':
        return renderMetricCard(data[0]);
      case 'kpi_grid':
        return renderKPIGrid(data.slice(0, 6));
      case 'line_chart':
        return renderLineChart(data, widget.visualization_config);
      case 'bar_chart':
        return renderBarChart(data, widget.visualization_config);
      case 'pie_chart':
        return renderPieChart(data, widget.visualization_config);
      case 'area_chart':
        return renderAreaChart(data, widget.visualization_config);
      case 'data_table':
        return renderDataTable(data.slice(0, 5));
      case 'gauge_chart':
        return renderGaugeChart(data[0]);
      case 'progress_bar':
        return renderProgressBar(data[0]);
      default:
        return <div className="p-4 text-center text-gray-500">Widget type: {widget.type}</div>;
    }
  };

  const renderMetricCard = (dataPoint: LiveDataPoint) => {
    if (!dataPoint) return <div>No data</div>;
    
    return (
      <div className="p-4 text-center">
        <div className="text-3xl font-bold text-blue-600">
          {dataPoint.value.toLocaleString()}
        </div>
        <div className="text-sm text-gray-600 mt-1">
          Updated: {new Date(dataPoint.timestamp).toLocaleTimeString()}
        </div>
      </div>
    );
  };

  const renderKPIGrid = (dataPoints: LiveDataPoint[]) => {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 p-4">
        {dataPoints.map((point, index) => (
          <div key={index} className="text-center">
            <div className="text-xl font-bold text-blue-600">
              {point.value.toLocaleString()}
            </div>
            <div className="text-xs text-gray-600">{point.label}</div>
          </div>
        ))}
      </div>
    );
  };

  const renderLineChart = (chartData: LiveDataPoint[], config: any) => {
    const formattedData = chartData.map(d => ({
      ...d,
      date: new Date(d.timestamp).toLocaleDateString()
    }));

    return (
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={formattedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          {config.show_legend !== false && <Legend />}
          <Line 
            type="monotone" 
            dataKey="value" 
            stroke={config.color_scheme === 'green' ? '#00C49F' : '#8884d8'}
            strokeWidth={2}
            animationDuration={config.animation_enabled !== false ? 1000 : 0}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  const renderBarChart = (chartData: LiveDataPoint[], config: any) => {
    return (
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="label" />
          <YAxis />
          <Tooltip />
          {config.show_legend !== false && <Legend />}
          <Bar 
            dataKey="value" 
            fill={config.color_scheme === 'green' ? '#00C49F' : '#8884d8'}
            animationDuration={config.animation_enabled !== false ? 1000 : 0}
          />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderPieChart = (chartData: LiveDataPoint[], config: any) => {
    return (
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ label, percent }) => `${label} ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            animationDuration={config.animation_enabled !== false ? 1000 : 0}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    );
  };

  const renderAreaChart = (chartData: LiveDataPoint[], config: any) => {
    const formattedData = chartData.map(d => ({
      ...d,
      date: new Date(d.timestamp).toLocaleDateString()
    }));

    return (
      <ResponsiveContainer width="100%" height={250}>
        <AreaChart data={formattedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          {config.show_legend !== false && <Legend />}
          <Area 
            type="monotone" 
            dataKey="value" 
            stroke={config.color_scheme === 'green' ? '#00C49F' : '#8884d8'}
            fill={config.color_scheme === 'green' ? '#00C49F' : '#8884d8'}
            fillOpacity={0.6}
            animationDuration={config.animation_enabled !== false ? 1000 : 0}
          />
        </AreaChart>
      </ResponsiveContainer>
    );
  };

  const renderDataTable = (tableData: LiveDataPoint[]) => {
    return (
      <div className="overflow-x-auto p-4">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2">Label</th>
              <th className="text-left p-2">Value</th>
              <th className="text-left p-2">Category</th>
              <th className="text-left p-2">Date</th>
            </tr>
          </thead>
          <tbody>
            {tableData.map((item, index) => (
              <tr key={index} className="border-b">
                <td className="p-2">{item.label}</td>
                <td className="p-2">{item.value.toLocaleString()}</td>
                <td className="p-2">{item.category}</td>
                <td className="p-2">{new Date(item.timestamp).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderGaugeChart = (dataPoint: LiveDataPoint) => {
    if (!dataPoint) return <div>No data</div>;
    
    const percentage = Math.min((dataPoint.value / 1000) * 100, 100);
    
    return (
      <div className="p-4 text-center">
        <div className="relative w-32 h-32 mx-auto">
          <svg className="w-32 h-32 transform -rotate-90">
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              className="text-gray-200"
            />
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              strokeDasharray={`${2 * Math.PI * 56}`}
              strokeDashoffset={`${2 * Math.PI * 56 * (1 - percentage / 100)}`}
              className="text-blue-600"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xl font-bold">{percentage.toFixed(0)}%</span>
          </div>
        </div>
        <div className="mt-2 text-sm text-gray-600">
          {dataPoint.value.toLocaleString()} / 1000
        </div>
      </div>
    );
  };

  const renderProgressBar = (dataPoint: LiveDataPoint) => {
    if (!dataPoint) return <div>No data</div>;
    
    const percentage = Math.min((dataPoint.value / 1000) * 100, 100);
    
    return (
      <div className="p-4">
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium">{dataPoint.label}</span>
          <span className="text-sm text-gray-600">{percentage.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div 
            className="bg-blue-600 h-4 rounded-full transition-all duration-1000"
            style={{ width: `${percentage}%` }}
          />
        </div>
        <div className="mt-2 text-xs text-gray-600">
          {dataPoint.value.toLocaleString()} / 1000
        </div>
      </div>
    );
  };

  return (
    <Card
      className="h-full"
      style={{
        gridColumn: `span ${position.width}`,
        gridRow: `span ${position.height}`,
      }}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">{widget.title}</CardTitle>
          <div className="flex items-center gap-1">
            <Badge variant="outline" className="text-xs">
              {widget.data_source}
            </Badge>
            {widget.visualization_config.refresh_interval && (
              <Badge variant="secondary" className="text-xs">
                {widget.visualization_config.refresh_interval}s
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {renderContent()}
      </CardContent>
    </Card>
  );
};

// Main Template Preview Component
export const TemplatePreview: React.FC<TemplatePreviewProps> = ({
  template,
  isLivePreview = false,
  currentBreakpoint = 'desktop',
  onBreakpointChange,
  className = ''
}) => {
  const [widgetData, setWidgetData] = useState<WidgetData>({});
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const { socket, isConnected } = useWebSocket();

  // Generate initial mock data
  useEffect(() => {
    const initialData: WidgetData = {};
    
    template.widgets.forEach(widget => {
      initialData[widget.id] = {
        data: generateMockData(widget.type, widget.data_source),
        metadata: {
          last_updated: new Date().toISOString(),
          source: widget.data_source,
          status: 'success'
        }
      };
    });
    
    setWidgetData(initialData);
  }, [template.widgets]);

  // WebSocket connection for live data
  useEffect(() => {
    if (!isLivePreview || !socket || isPaused) return;

    const handleLiveData = (message: any) => {
      if (message.type === 'widget_data_update') {
        setWidgetData(prev => ({
          ...prev,
          [message.widget_id]: {
            data: message.data,
            metadata: {
              last_updated: new Date().toISOString(),
              source: message.source,
              status: 'success'
            }
          }
        }));
        setLastUpdate(new Date());
      }
    };

    socket.on('dashboard_preview_data', handleLiveData);

    // Request initial data for all widgets
    template.widgets.forEach(widget => {
      socket.emit('request_widget_data', {
        widget_id: widget.id,
        data_source: widget.data_source,
        config: widget.visualization_config
      });
    });

    return () => {
      socket.off('dashboard_preview_data', handleLiveData);
    };
  }, [socket, isLivePreview, template.widgets, isPaused]);

  // Auto-refresh for mock data when not using live preview
  useEffect(() => {
    if (isLivePreview || isPaused) return;

    const interval = setInterval(() => {
      const updatedData: WidgetData = {};
      
      template.widgets.forEach(widget => {
        updatedData[widget.id] = {
          data: generateMockData(widget.type, widget.data_source),
          metadata: {
            last_updated: new Date().toISOString(),
            source: widget.data_source,
            status: 'success'
          }
        };
      });
      
      setWidgetData(updatedData);
      setLastUpdate(new Date());
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [isLivePreview, template.widgets, isPaused]);

  const handleBreakpointChange = (breakpoint: string) => {
    if (onBreakpointChange) {
      onBreakpointChange(breakpoint);
    }
  };

  const getBreakpointIcon = (breakpoint: string) => {
    switch (breakpoint) {
      case 'mobile': return Smartphone;
      case 'tablet': return Tablet;
      default: return Monitor;
    }
  };

  const getGridColumns = () => {
    switch (currentBreakpoint) {
      case 'mobile': return 4;
      case 'tablet': return 8;
      default: return template.layout_config.grid_size;
    }
  };

  return (
    <div className={`${className} ${isFullscreen ? 'fixed inset-0 z-50 bg-white' : ''}`}>
      {/* Preview Controls */}
      <div className="flex items-center justify-between p-4 border-b bg-gray-50">
        <div className="flex items-center gap-4">
          <h3 className="font-semibold">
            {template.name} Preview
          </h3>
          
          {isLivePreview && (
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm text-gray-600">
                {isConnected ? 'Live' : 'Offline'}
              </span>
            </div>
          )}
          
          <span className="text-xs text-gray-500">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Breakpoint Controls */}
          <div className="flex border rounded-md">
            {['mobile', 'tablet', 'desktop'].map(breakpoint => {
              const Icon = getBreakpointIcon(breakpoint);
              return (
                <Button
                  key={breakpoint}
                  size="sm"
                  variant={currentBreakpoint === breakpoint ? 'default' : 'ghost'}
                  onClick={() => handleBreakpointChange(breakpoint)}
                  className="rounded-none first:rounded-l-md last:rounded-r-md"
                >
                  <Icon className="h-4 w-4" />
                </Button>
              );
            })}
          </div>

          {/* Pause/Play */}
          <Button
            size="sm"
            variant="outline"
            onClick={() => setIsPaused(!isPaused)}
          >
            {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
          </Button>

          {/* Fullscreen Toggle */}
          <Button
            size="sm"
            variant="outline"
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {/* Preview Content */}
      <div 
        className={`p-4 overflow-auto ${isFullscreen ? 'h-full' : 'max-h-96'}`}
        style={{
          backgroundColor: template.theme.background_color,
          color: template.theme.text_color,
        }}
      >
        <div
          className="grid gap-4"
          style={{
            gridTemplateColumns: `repeat(${getGridColumns()}, 1fr)`,
            gridAutoRows: `${template.layout_config.row_height}px`,
          }}
        >
          {template.widgets.map(widget => {
            const data = widgetData[widget.id];
            return (
              <WidgetRenderer
                key={widget.id}
                widget={widget}
                data={data?.data || []}
                isLoading={data?.metadata.status === 'loading'}
                error={data?.metadata.error_message}
                currentBreakpoint={currentBreakpoint}
              />
            );
          })}
        </div>

        {template.widgets.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <Eye className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <div>No widgets in this template</div>
            <div className="text-sm">Add widgets to see the preview</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TemplatePreview;