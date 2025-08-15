import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Progress } from '../ui/progress';

interface SystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  disk_percent: number;
  uptime_seconds: number;
  timestamp: string;
}

interface ServiceStatus {
  status: string;
  uptime_percentage: number;
  response_time: number;
  last_check: string;
}

interface AlertData {
  id: string;
  severity: string;
  component: string;
  message: string;
  timestamp: string;
  resolved: boolean;
}

interface MonitoringData {
  system_metrics: SystemMetrics;
  services: Record<string, ServiceStatus>;
  alerts: AlertData[];
  overall_status: string;
  performance_stats: {
    request_rate: number;
    error_rate: number;
    avg_response_time: number;
    active_users: number;
  };
}

const ComprehensiveMonitoringDashboard: React.FC = () => {
  const [monitoringData, setMonitoringData] = useState<MonitoringData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    fetchMonitoringData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchMonitoringData, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const fetchMonitoringData = async () => {
    try {
      const [metricsResponse, statusResponse] = await Promise.all([
        fetch('/api/monitoring/metrics'),
        fetch('/api/status')
      ]);

      if (!metricsResponse.ok || !statusResponse.ok) {
        throw new Error('Failed to fetch monitoring data');
      }

      const [metricsData, statusData] = await Promise.all([
        metricsResponse.json(),
        statusResponse.json()
      ]);

      // Combine data from both endpoints
      const combinedData: MonitoringData = {
        system_metrics: metricsData.system_metrics || {
          cpu_percent: 0,
          memory_percent: 0,
          disk_percent: 0,
          uptime_seconds: 0,
          timestamp: new Date().toISOString()
        },
        services: statusData.services || {},
        alerts: metricsData.alerts || [],
        overall_status: statusData.overall_status || 'unknown',
        performance_stats: metricsData.performance_stats || {
          request_rate: 0,
          error_rate: 0,
          avg_response_time: 0,
          active_users: 0
        }
      };

      setMonitoringData(combinedData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational':
        return 'text-green-600 bg-green-100';
      case 'degraded':
        return 'text-yellow-600 bg-yellow-100';
      case 'partial_outage':
        return 'text-orange-600 bg-orange-100';
      case 'major_outage':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-100';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100';
      case 'info':
        return 'text-blue-600 bg-blue-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <Alert>
          <AlertDescription>
            Error loading monitoring data: {error}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!monitoringData) {
    return null;
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">
          System Monitoring
        </h1>
        <div className="flex items-center space-x-4">
          <Badge className={getStatusColor(monitoringData.overall_status)}>
            {monitoringData.overall_status.toUpperCase()}
          </Badge>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-1 rounded text-sm ${
              autoRefresh 
                ? 'bg-green-100 text-green-800' 
                : 'bg-gray-100 text-gray-800'
            }`}
          >
            Auto Refresh: {autoRefresh ? 'ON' : 'OFF'}
          </button>
          <button
            onClick={fetchMonitoringData}
            className="px-3 py-1 bg-blue-100 text-blue-800 rounded text-sm hover:bg-blue-200"
          >
            Refresh Now
          </button>
        </div>
      </div>

      {/* System Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              CPU Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold mb-2">
              {monitoringData.system_metrics.cpu_percent.toFixed(1)}%
            </div>
            <Progress 
              value={monitoringData.system_metrics.cpu_percent} 
              className="h-2"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Memory Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold mb-2">
              {monitoringData.system_metrics.memory_percent.toFixed(1)}%
            </div>
            <Progress 
              value={monitoringData.system_metrics.memory_percent} 
              className="h-2"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Disk Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold mb-2">
              {monitoringData.system_metrics.disk_percent.toFixed(1)}%
            </div>
            <Progress 
              value={monitoringData.system_metrics.disk_percent} 
              className="h-2"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              System Uptime
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatUptime(monitoringData.system_metrics.uptime_seconds)}
            </div>
            <div className="text-sm text-gray-500">
              Since last restart
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Request Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {monitoringData.performance_stats.request_rate.toFixed(1)}
            </div>
            <div className="text-sm text-gray-500">requests/sec</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Error Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {monitoringData.performance_stats.error_rate.toFixed(2)}%
            </div>
            <div className="text-sm text-gray-500">of requests</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Avg Response Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(monitoringData.performance_stats.avg_response_time * 1000).toFixed(0)}ms
            </div>
            <div className="text-sm text-gray-500">average</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Active Users
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {monitoringData.performance_stats.active_users}
            </div>
            <div className="text-sm text-gray-500">current sessions</div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Tabs */}
      <Tabs defaultValue="services" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="services">Services</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="logs">Recent Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="services">
          <Card>
            <CardHeader>
              <CardTitle>Service Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(monitoringData.services).map(([serviceName, service]) => (
                  <div key={serviceName} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        service.status === 'operational' ? 'bg-green-500' :
                        service.status === 'degraded' ? 'bg-yellow-500' :
                        service.status === 'partial_outage' ? 'bg-orange-500' :
                        'bg-red-500'
                      }`}></div>
                      <div>
                        <h4 className="font-medium">{serviceName}</h4>
                        <p className="text-sm text-gray-600 capitalize">
                          {service.status.replace('_', ' ')}
                        </p>
                      </div>
                    </div>
                    <div className="text-right text-sm">
                      <div>Uptime: {service.uptime_percentage.toFixed(2)}%</div>
                      <div>Response: {service.response_time.toFixed(0)}ms</div>
                      <div className="text-gray-500">
                        Last check: {formatDate(service.last_check)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts">
          <Card>
            <CardHeader>
              <CardTitle>Active Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              {monitoringData.alerts.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No active alerts
                </div>
              ) : (
                <div className="space-y-3">
                  {monitoringData.alerts.map((alert) => (
                    <div key={alert.id} className="p-4 border rounded-lg">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <Badge className={getSeverityColor(alert.severity)}>
                              {alert.severity.toUpperCase()}
                            </Badge>
                            <Badge variant="outline">
                              {alert.component}
                            </Badge>
                            {alert.resolved && (
                              <Badge className="bg-green-100 text-green-800">
                                RESOLVED
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm font-medium mb-1">
                            {alert.message}
                          </p>
                          <p className="text-xs text-gray-500">
                            {formatDate(alert.timestamp)}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="logs">
          <Card>
            <CardHeader>
              <CardTitle>Recent System Logs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-gray-500">
                Log viewer would be implemented here
                <br />
                <span className="text-sm">
                  Connect to log aggregation system for real-time logs
                </span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Footer */}
      <div className="text-center text-sm text-gray-500">
        Last updated: {formatDate(monitoringData.system_metrics.timestamp)}
      </div>
    </div>
  );
};

export default ComprehensiveMonitoringDashboard;