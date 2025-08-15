"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell 
} from 'recharts';
import { 
  TrendingUp, TrendingDown, DollarSign, Users, 
  Activity, AlertTriangle, RefreshCw, Settings, Wifi, WifiOff 
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';

interface DashboardProps {
  dashboardId: string;
  role: string;
  userId: string;
}

interface Widget {
  id: string;
  name: string;
  type: string;
  position: { x: number; y: number };
  size: { width: number; height: number };
  config: any;
  data_source: string;
  last_updated: string;
}

interface BusinessMetric {
  id: string;
  name: string;
  category: string;
  value: number;
  unit: string;
  timestamp: string;
  source: string;
  context: any;
}

interface DashboardData {
  dashboard: {
    id: string;
    name: string;
    type: string;
    role: string;
    config: any;
  };
  widgets_data: Record<string, Widget>;
  metrics: BusinessMetric[];
  last_updated: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function ExecutiveDashboard({ dashboardId, role, userId }: DashboardProps) {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [alerts, setAlerts] = useState<any[]>([]);

  const fetchDashboardData = useCallback(async () => {
    try {
      const response = await fetch(`/api/dashboard/${dashboardId}/data`);
      if (!response.ok) {
        throw new Error('Failed to fetch dashboard data');
      }
      const data = await response.json();
      setDashboardData(data);
      setLastRefresh(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [dashboardId]);

  // WebSocket connection for real-time updates
  const {
    isConnected: wsConnected,
    isConnecting: wsConnecting,
    error: wsError,
    requestUpdate
  } = useWebSocket({
    url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
    connectionId: `${userId}_${dashboardId}_${Date.now()}`,
    userId,
    dashboardId,
    onMessage: (message) => {
      console.log('Received WebSocket message:', message);
      
      switch (message.type) {
        case 'dashboard_update':
          setDashboardData(message.data);
          setLastRefresh(new Date());
          break;
        case 'metrics_update':
          if (dashboardData) {
            setDashboardData(prev => prev ? {
              ...prev,
              metrics: [...prev.metrics, ...message.data]
            } : null);
          }
          break;
        case 'alert':
          setAlerts(prev => [message.data, ...prev.slice(0, 4)]); // Keep last 5 alerts
          break;
      }
    },
    onConnect: () => {
      console.log('WebSocket connected');
      setError(null);
    },
    onDisconnect: () => {
      console.log('WebSocket disconnected');
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    }
  });

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  useEffect(() => {
    if (autoRefresh && !wsConnected) {
      const interval = setInterval(fetchDashboardData, 300000); // 5 minutes fallback
      return () => clearInterval(interval);
    }
  }, [autoRefresh, wsConnected, fetchDashboardData]);

  const renderKPIWidget = (widget: Widget, metrics: BusinessMetric[]) => {
    const kpiMetrics = metrics.filter(m => 
      widget.config.metrics?.includes(m.name)
    ).slice(0, 3);

    return (
      <div className="grid grid-cols-3 gap-4">
        {kpiMetrics.map((metric, index) => (
          <div key={metric.id} className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {metric.value.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">{metric.unit}</div>
            <div className="text-xs text-gray-500">{metric.name}</div>
          </div>
        ))}
      </div>
    );
  };

  const renderChartWidget = (widget: Widget, metrics: BusinessMetric[]) => {
    const chartData = metrics
      .filter(m => m.category === widget.config.category || !widget.config.category)
      .slice(0, 10)
      .map(m => ({
        name: m.name,
        value: m.value,
        timestamp: new Date(m.timestamp).toLocaleDateString()
      }));

    if (widget.config.chart_type === 'line') {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      );
    }

    if (widget.config.chart_type === 'bar') {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      );
    }

    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
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

  const renderMetricWidget = (widget: Widget, metrics: BusinessMetric[]) => {
    const metric = metrics.find(m => m.name === widget.config.metric);
    if (!metric) return <div>No data available</div>;

    const trend = metric.context?.trend || 'neutral';
    const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Activity;
    const trendColor = trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-gray-600';

    return (
      <div className="flex items-center justify-between">
        <div>
          <div className="text-3xl font-bold">{metric.value.toLocaleString()}</div>
          <div className="text-sm text-gray-600">{metric.unit}</div>
        </div>
        <TrendIcon className={`h-8 w-8 ${trendColor}`} />
      </div>
    );
  };

  const renderTableWidget = (widget: Widget, metrics: BusinessMetric[]) => {
    const tableData = metrics.slice(0, 5);
    const columns = widget.config.columns || ['name', 'value', 'unit'];

    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              {columns.map((col: string) => (
                <th key={col} className="text-left p-2 font-medium">
                  {col.charAt(0).toUpperCase() + col.slice(1)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableData.map((metric) => (
              <tr key={metric.id} className="border-b">
                {columns.map((col: string) => (
                  <td key={col} className="p-2">
                    {col === 'value' ? metric.value.toLocaleString() : (metric as any)[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderWidget = (widget: Widget) => {
    if (!dashboardData) return null;

    const { metrics } = dashboardData;

    let content;
    switch (widget.type) {
      case 'kpi':
        content = renderKPIWidget(widget, metrics);
        break;
      case 'chart':
        content = renderChartWidget(widget, metrics);
        break;
      case 'metric':
        content = renderMetricWidget(widget, metrics);
        break;
      case 'table':
        content = renderTableWidget(widget, metrics);
        break;
      default:
        content = <div>Unsupported widget type: {widget.type}</div>;
    }

    return (
      <Card key={widget.id} className="h-full">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">{widget.name}</CardTitle>
        </CardHeader>
        <CardContent>{content}</CardContent>
      </Card>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading dashboard...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Error loading dashboard: {error}
          <Button onClick={fetchDashboardData} className="ml-2" size="sm">
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (!dashboardData) {
    return <div>No dashboard data available</div>;
  }

  const widgets = Object.values(dashboardData.widgets_data);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">{dashboardData.dashboard.name}</h1>
          <p className="text-gray-600">
            Role: {role.charAt(0).toUpperCase() + role.slice(1)} | 
            Last updated: {lastRefresh.toLocaleTimeString()}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={wsConnected ? "default" : "secondary"}>
            {wsConnected ? (
              <>
                <Wifi className="h-3 w-3 mr-1" />
                Real-time ON
              </>
            ) : (
              <>
                <WifiOff className="h-3 w-3 mr-1" />
                Real-time OFF
              </>
            )}
          </Badge>
          <Badge variant={autoRefresh ? "default" : "secondary"}>
            {autoRefresh ? "Auto-refresh ON" : "Auto-refresh OFF"}
          </Badge>
          <Button
            onClick={() => setAutoRefresh(!autoRefresh)}
            variant="outline"
            size="sm"
          >
            <Settings className="h-4 w-4" />
          </Button>
          <Button 
            onClick={wsConnected ? requestUpdate : fetchDashboardData} 
            size="sm"
            disabled={wsConnecting}
          >
            <RefreshCw className={`h-4 w-4 ${wsConnecting ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.slice(0, 3).map((alert, index) => (
            <Alert key={index} className={
              alert.type === 'critical' ? 'border-red-500' :
              alert.type === 'warning' ? 'border-yellow-500' : 'border-blue-500'
            }>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <strong>{alert.title}:</strong> {alert.message}
              </AlertDescription>
            </Alert>
          ))}
        </div>
      )}

      {/* Dashboard Grid */}
      <div className="grid grid-cols-12 gap-6">
        {widgets.map((widget) => (
          <div
            key={widget.id}
            className={`col-span-${widget.size.width} row-span-${widget.size.height}`}
            style={{
              gridColumn: `span ${widget.size.width}`,
              minHeight: `${widget.size.height * 100}px`
            }}
          >
            {renderWidget(widget)}
          </div>
        ))}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center">
              <DollarSign className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <div className="text-2xl font-bold">
                  ${dashboardData.metrics
                    .filter(m => m.category === 'financial')
                    .reduce((sum, m) => sum + m.value, 0)
                    .toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">Total Revenue</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center">
              <Users className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <div className="text-2xl font-bold">
                  {dashboardData.metrics
                    .filter(m => m.category === 'operational')
                    .length}
                </div>
                <div className="text-sm text-gray-600">Active Projects</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-purple-600" />
              <div className="ml-4">
                <div className="text-2xl font-bold">
                  {Math.round(
                    dashboardData.metrics
                      .filter(m => m.name.includes('efficiency'))
                      .reduce((sum, m) => sum + m.value, 0) / 
                    dashboardData.metrics.filter(m => m.name.includes('efficiency')).length || 1
                  )}%
                </div>
                <div className="text-sm text-gray-600">Avg Efficiency</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-orange-600" />
              <div className="ml-4">
                <div className="text-2xl font-bold">
                  {dashboardData.metrics
                    .filter(m => m.context?.trend === 'up')
                    .length}
                </div>
                <div className="text-sm text-gray-600">Positive Trends</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}