"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area
} from 'recharts';
import { 
  Activity, AlertTriangle, CheckCircle, Clock, DollarSign, 
  TrendingUp, TrendingDown, Zap, Settings, RefreshCw
} from 'lucide-react';

interface PerformanceMetric {
  id: number;
  execution_id: string;
  start_time: string;
  end_time?: string;
  duration_seconds?: number;
  cpu_usage_percent?: number;
  memory_usage_mb?: number;
  records_processed: number;
  records_per_second?: number;
  error_count: number;
  error_rate: number;
  total_cost?: number;
}

interface Alert {
  id: number;
  alert_type: string;
  alert_level: string;
  alert_message: string;
  created_at: string;
  escalation_level: number;
  notification_sent: boolean;
}

interface Recommendation {
  id: number;
  recommendation_type: string;
  priority: string;
  title: string;
  description: string;
  expected_improvement?: number;
  estimated_cost_savings?: number;
  implementation_effort: string;
  status: string;
  created_at: string;
}

interface DashboardData {
  summary: {
    total_executions: number;
    avg_duration_seconds: number;
    avg_cpu_usage: number;
    avg_memory_usage_mb: number;
    total_records_processed: number;
    total_errors: number;
    error_rate: number;
  };
  violations: Array<{
    id: number;
    sla_type: string;
    severity: string;
    threshold: number;
    actual_value: number;
    start_time: string;
    is_resolved: boolean;
  }>;
  active_alerts: Alert[];
}

interface PerformanceMonitoringDashboardProps {
  pipelineId?: string;
  timeRangeHours?: number;
}

export default function PerformanceMonitoringDashboard({ 
  pipelineId, 
  timeRangeHours = 24 
}: PerformanceMonitoringDashboardProps) {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState(timeRangeHours);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    if (pipelineId) {
      fetchMetrics();
      fetchRecommendations();
    }

    // Auto-refresh every 30 seconds
    const interval = autoRefresh ? setInterval(() => {
      fetchDashboardData();
      if (pipelineId) {
        fetchMetrics();
      }
    }, 30000) : null;

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [pipelineId, selectedTimeRange, autoRefresh]);

  const fetchDashboardData = async () => {
    try {
      const params = new URLSearchParams({
        hours: selectedTimeRange.toString()
      });
      if (pipelineId) {
        params.append('pipeline_id', pipelineId);
      }

      const response = await fetch(`/api/v1/performance/dashboard?${params}`);
      const result = await response.json();
      
      if (result.success) {
        setDashboardData(result.data);
      } else {
        setError('Failed to fetch dashboard data');
      }
    } catch (err) {
      setError('Error fetching dashboard data');
      console.error('Dashboard fetch error:', err);
    }
  };

  const fetchMetrics = async () => {
    if (!pipelineId) return;

    try {
      const response = await fetch(`/api/v1/performance/metrics/${pipelineId}?hours=${selectedTimeRange}`);
      const result = await response.json();
      
      if (result.success) {
        setMetrics(result.data.metrics);
      }
    } catch (err) {
      console.error('Metrics fetch error:', err);
    }
  };

  const fetchRecommendations = async () => {
    if (!pipelineId) return;

    try {
      const response = await fetch(`/api/v1/performance/recommendations/${pipelineId}`);
      const result = await response.json();
      
      if (result.success) {
        setRecommendations(result.data.recommendations);
      }
    } catch (err) {
      console.error('Recommendations fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const acknowledgeAlert = async (alertId: number) => {
    try {
      const response = await fetch(`/api/v1/performance/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ acknowledged_by: 'user' })
      });
      
      if (response.ok) {
        fetchDashboardData(); // Refresh to update alerts
      }
    } catch (err) {
      console.error('Error acknowledging alert:', err);
    }
  };

  const applyAutoTuning = async () => {
    if (!pipelineId) return;

    try {
      const response = await fetch(`/api/v1/performance/tuning/${pipelineId}/apply`, {
        method: 'POST'
      });
      
      const result = await response.json();
      if (result.success) {
        alert('Auto-tuning applied successfully!');
        fetchDashboardData();
      }
    } catch (err) {
      console.error('Error applying auto-tuning:', err);
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const formatBytes = (bytes?: number) => {
    if (!bytes) return 'N/A';
    const gb = bytes / 1024;
    return gb > 1 ? `${gb.toFixed(2)} GB` : `${bytes.toFixed(0)} MB`;
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'destructive';
      case 'warning': return 'default';
      case 'info': return 'secondary';
      default: return 'outline';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading performance data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Performance Monitoring</h2>
          <p className="text-muted-foreground">
            {pipelineId ? `Pipeline: ${pipelineId}` : 'All Pipelines'} • Last {selectedTimeRange} hours
          </p>
        </div>
        <div className="flex gap-2">
          <select 
            value={selectedTimeRange} 
            onChange={(e) => setSelectedTimeRange(Number(e.target.value))}
            className="px-3 py-2 border rounded-md"
          >
            <option value={1}>Last Hour</option>
            <option value={6}>Last 6 Hours</option>
            <option value={24}>Last 24 Hours</option>
            <option value={168}>Last Week</option>
          </select>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          {pipelineId && (
            <Button onClick={applyAutoTuning} size="sm">
              <Zap className="h-4 w-4 mr-2" />
              Auto Tune
            </Button>
          )}
        </div>
      </div>

      {/* Summary Cards */}
      {dashboardData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Executions</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dashboardData.summary.total_executions}</div>
              <p className="text-xs text-muted-foreground">
                {dashboardData.summary.total_records_processed.toLocaleString()} records processed
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Duration</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {formatDuration(dashboardData.summary.avg_duration_seconds)}
              </div>
              <p className="text-xs text-muted-foreground">
                Average execution time
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Resource Usage</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {dashboardData.summary.avg_cpu_usage.toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground">
                CPU • {formatBytes(dashboardData.summary.avg_memory_usage_mb)} Memory
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(dashboardData.summary.error_rate * 100).toFixed(2)}%
              </div>
              <p className="text-xs text-muted-foreground">
                {dashboardData.summary.total_errors} total errors
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="metrics" className="space-y-4">
        <TabsList>
          <TabsTrigger value="metrics">Performance Metrics</TabsTrigger>
          <TabsTrigger value="alerts">Active Alerts</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          <TabsTrigger value="cost">Cost Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics" className="space-y-4">
          {metrics.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* CPU Usage Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>CPU Usage Over Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="start_time" 
                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                      />
                      <YAxis />
                      <Tooltip 
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                        formatter={(value: number) => [`${value.toFixed(1)}%`, 'CPU Usage']}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="cpu_usage_percent" 
                        stroke="#8884d8" 
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Memory Usage Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Memory Usage Over Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={metrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="start_time" 
                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                      />
                      <YAxis />
                      <Tooltip 
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                        formatter={(value: number) => [`${formatBytes(value)}`, 'Memory Usage']}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="memory_usage_mb" 
                        stroke="#82ca9d" 
                        fill="#82ca9d"
                        fillOpacity={0.6}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Throughput Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Processing Throughput</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={metrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="start_time" 
                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                      />
                      <YAxis />
                      <Tooltip 
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                        formatter={(value: number) => [`${value?.toFixed(1) || 0} rec/sec`, 'Throughput']}
                      />
                      <Bar dataKey="records_per_second" fill="#ffc658" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Error Rate Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Error Rate Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="start_time" 
                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                      />
                      <YAxis />
                      <Tooltip 
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                        formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Error Rate']}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="error_rate" 
                        stroke="#ff7300" 
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          {dashboardData?.active_alerts && dashboardData.active_alerts.length > 0 ? (
            <div className="space-y-4">
              {dashboardData.active_alerts.map((alert) => (
                <Card key={alert.id}>
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <AlertTriangle className="h-5 w-5" />
                          {alert.alert_type.replace('_', ' ').toUpperCase()}
                          <Badge variant={getSeverityColor(alert.alert_level)}>
                            {alert.alert_level}
                          </Badge>
                        </CardTitle>
                        <CardDescription>
                          Created: {new Date(alert.created_at).toLocaleString()}
                        </CardDescription>
                      </div>
                      <Button 
                        size="sm" 
                        onClick={() => acknowledgeAlert(alert.id)}
                      >
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Acknowledge
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p>{alert.alert_message}</p>
                    {alert.escalation_level > 0 && (
                      <Badge variant="outline" className="mt-2">
                        Escalation Level: {alert.escalation_level}
                      </Badge>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center h-32">
                <div className="text-center">
                  <CheckCircle className="h-8 w-8 text-green-500 mx-auto mb-2" />
                  <p className="text-muted-foreground">No active alerts</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-4">
          {recommendations.length > 0 ? (
            <div className="space-y-4">
              {recommendations.map((rec) => (
                <Card key={rec.id}>
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          {rec.title}
                          <Badge variant={getPriorityColor(rec.priority)}>
                            {rec.priority}
                          </Badge>
                          <Badge variant="outline">
                            {rec.status}
                          </Badge>
                        </CardTitle>
                        <CardDescription>
                          {rec.recommendation_type.replace('_', ' ')} • 
                          Created: {new Date(rec.created_at).toLocaleDateString()}
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="mb-4">{rec.description}</p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      {rec.expected_improvement && (
                        <div>
                          <span className="font-medium">Expected Improvement:</span>
                          <p className="text-green-600">+{rec.expected_improvement}%</p>
                        </div>
                      )}
                      {rec.estimated_cost_savings && (
                        <div>
                          <span className="font-medium">Cost Savings:</span>
                          <p className="text-green-600">${rec.estimated_cost_savings}</p>
                        </div>
                      )}
                      <div>
                        <span className="font-medium">Implementation Effort:</span>
                        <p>{rec.implementation_effort}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center h-32">
                <div className="text-center">
                  <Settings className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                  <p className="text-muted-foreground">No recommendations available</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="cost" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Cost Analysis</CardTitle>
              <CardDescription>
                Cost tracking and optimization insights
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <DollarSign className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">
                  Cost analysis will be available when cost tracking is enabled
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}