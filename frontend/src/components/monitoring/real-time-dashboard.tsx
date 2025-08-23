"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  DollarSign,
  Users,
  Server,
  Zap,
  Target,
  BarChart3,
  RefreshCw
} from 'lucide-react';

interface AgentMetrics {
  response_time: number;
  throughput: number;
  accuracy: number;
  success_rate: number;
  active_tasks: number;
}

interface SystemMetrics {
  overall_health_score: number;
  cpu_utilization: number;
  memory_utilization: number;
  disk_utilization: number;
  active_agents_count: number;
  total_requests_per_second: number;
  average_response_time: number;
  error_rate: number;
  uptime_percentage: number;
}

interface BusinessMetrics {
  roi_percentage: number;
  total_cost_savings: number;
  total_revenue_increase: number;
  productivity_gain_percentage: number;
  customer_satisfaction_score: number;
}

interface AlertData {
  id: string;
  name: string;
  severity: string;
  description: string;
  timestamp: string;
  current_value: number;
  threshold: number;
}

interface DashboardData {
  timestamp: string;
  system: SystemMetrics;
  business: BusinessMetrics;
  agents: Record<string, AgentMetrics>;
  alerts: {
    active: AlertData[];
    count: number;
  };
  business_metrics: BusinessMetrics;
  performance_summary: {
    system_health: number;
    agent_efficiency: number;
    business_value: number;
    operational_status: string;
  };
}

const RealTimeDashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedTab, setSelectedTab] = useState('overview');

  const fetchDashboardData = useCallback(async () => {
    try {
      const response = await fetch('/api/monitoring/realtime/dashboard');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setDashboardData(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch dashboard data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchDashboardData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [autoRefresh, fetchDashboardData]);

  const acknowledgeAlert = async (alertId: string) => {
    try {
      const response = await fetch(`/api/monitoring/realtime/alerts/acknowledge/${alertId}`, {
        method: 'POST',
      });
      if (response.ok) {
        fetchDashboardData(); // Refresh data after acknowledging
      }
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'destructive';
      case 'warning':
        return 'default';
      case 'info':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  const getHealthColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 75) return 'text-yellow-600';
    if (score >= 60) return 'text-orange-600';
    return 'text-red-600';
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-4 w-4 animate-spin" />
          <span>Loading dashboard...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Error loading dashboard: {error}
          <Button 
            variant="outline" 
            size="sm" 
            onClick={fetchDashboardData}
            className="ml-2"
          >
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (!dashboardData) {
    return <div>No data available</div>;
  }

  const { system, business, agents, alerts, performance_summary } = dashboardData;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Real-Time Monitoring Dashboard</h1>
          <p className="text-muted-foreground">
            Last updated: {lastUpdate?.toLocaleTimeString() || 'Never'}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <Activity className="h-4 w-4 mr-2" />
            Auto Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={fetchDashboardData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Status Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Health</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              <span className={getHealthColor(performance_summary.system_health)}>
                {formatPercentage(performance_summary.system_health)}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              {performance_summary.operational_status === 'optimal' ? 'All systems operational' : 'Some issues detected'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Business ROI</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {formatPercentage(performance_summary.business_value)}
            </div>
            <p className="text-xs text-muted-foreground">
              {formatCurrency(business?.total_cost_savings || 0)} saved
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{system?.active_agents_count || 0}</div>
            <p className="text-xs text-muted-foreground">
              Efficiency: {formatPercentage(performance_summary.agent_efficiency)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              <span className={alerts.count > 0 ? 'text-red-600' : 'text-green-600'}>
                {alerts.count}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              {alerts.count === 0 ? 'No active alerts' : `${alerts.count} require attention`}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Dashboard Tabs */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="business">Business Impact</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* System Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>System Performance</CardTitle>
                <CardDescription>Real-time system resource utilization</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>CPU Usage</span>
                    <span>{formatPercentage(system?.cpu_utilization || 0)}</span>
                  </div>
                  <Progress value={system?.cpu_utilization || 0} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Memory Usage</span>
                    <span>{formatPercentage(system?.memory_utilization || 0)}</span>
                  </div>
                  <Progress value={system?.memory_utilization || 0} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Disk Usage</span>
                    <span>{formatPercentage(system?.disk_utilization || 0)}</span>
                  </div>
                  <Progress value={system?.disk_utilization || 0} className="h-2" />
                </div>
                <div className="pt-2 border-t">
                  <div className="flex justify-between text-sm">
                    <span>Uptime</span>
                    <span>{formatPercentage(system?.uptime_percentage || 0)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Requests/sec</span>
                    <span>{(system?.total_requests_per_second || 0).toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Avg Response Time</span>
                    <span>{(system?.average_response_time || 0).toFixed(2)}s</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Business Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Business Impact</CardTitle>
                <CardDescription>Financial and operational metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {formatCurrency(business?.total_cost_savings || 0)}
                    </div>
                    <p className="text-sm text-muted-foreground">Cost Savings</p>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {formatCurrency(business?.total_revenue_increase || 0)}
                    </div>
                    <p className="text-sm text-muted-foreground">Revenue Increase</p>
                  </div>
                </div>
                <div className="pt-2 border-t space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>ROI</span>
                    <span className="font-semibold text-green-600">
                      {formatPercentage(business?.roi_percentage || 0)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Productivity Gain</span>
                    <span>{formatPercentage(business?.productivity_gain_percentage || 0)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Customer Satisfaction</span>
                    <span>{(business?.customer_satisfaction_score || 0).toFixed(1)}/5.0</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="agents" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(agents || {}).map(([agentId, metrics]) => (
              <Card key={agentId}>
                <CardHeader>
                  <CardTitle className="text-lg">{agentId}</CardTitle>
                  <CardDescription>
                    <Badge variant={metrics.success_rate > 0.95 ? "default" : "secondary"}>
                      {formatPercentage(metrics.success_rate * 100)} Success Rate
                    </Badge>
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span>Response Time</span>
                    <span>{metrics.response_time.toFixed(2)}s</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Throughput</span>
                    <span>{metrics.throughput.toFixed(1)}/min</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Accuracy</span>
                    <span>{formatPercentage(metrics.accuracy * 100)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Active Tasks</span>
                    <span>{metrics.active_tasks}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="business" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Financial Impact</CardTitle>
                <CardDescription>Revenue and cost impact analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="h-5 w-5 text-green-600" />
                      <span className="font-medium">Total Value Created</span>
                    </div>
                    <span className="text-lg font-bold text-green-600">
                      {formatCurrency((business?.total_cost_savings || 0) + (business?.total_revenue_increase || 0))}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 border rounded-lg">
                      <div className="text-xl font-bold">{formatCurrency(business?.total_cost_savings || 0)}</div>
                      <p className="text-sm text-muted-foreground">Cost Savings</p>
                    </div>
                    <div className="text-center p-3 border rounded-lg">
                      <div className="text-xl font-bold">{formatCurrency(business?.total_revenue_increase || 0)}</div>
                      <p className="text-sm text-muted-foreground">Revenue Increase</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>Operational efficiency indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Target className="h-5 w-5 text-blue-600" />
                      <span className="font-medium">ROI Achievement</span>
                    </div>
                    <span className="text-lg font-bold text-blue-600">
                      {formatPercentage(business?.roi_percentage || 0)}
                    </span>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Productivity Gain</span>
                      <span>{formatPercentage(business?.productivity_gain_percentage || 0)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Customer Satisfaction</span>
                      <span>{(business?.customer_satisfaction_score || 0).toFixed(1)}/5.0</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          {alerts.active.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center py-8">
                <div className="text-center">
                  <CheckCircle className="h-12 w-12 text-green-600 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold">No Active Alerts</h3>
                  <p className="text-muted-foreground">All systems are operating normally</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {alerts.active.map((alert) => (
                <Card key={alert.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Badge variant={getSeverityColor(alert.severity)}>
                          {alert.severity.toUpperCase()}
                        </Badge>
                        <CardTitle className="text-lg">{alert.name}</CardTitle>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => acknowledgeAlert(alert.id)}
                      >
                        Acknowledge
                      </Button>
                    </div>
                    <CardDescription>{alert.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Current Value:</span> {alert.current_value}
                      </div>
                      <div>
                        <span className="font-medium">Threshold:</span> {alert.threshold}
                      </div>
                      <div>
                        <span className="font-medium">Time:</span> {new Date(alert.timestamp).toLocaleString()}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RealTimeDashboard;