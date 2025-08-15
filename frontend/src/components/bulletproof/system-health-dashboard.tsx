"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Activity, 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Clock,
  Cpu,
  HardDrive,
  Wifi,
  Users,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Settings,
  Zap
} from 'lucide-react';

interface SystemHealthMetrics {
  overall_status: string;
  timestamp: string;
  metrics: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_latency: number;
    error_rate: number;
    response_time: number;
    active_users: number;
    protection_effectiveness: number;
    recovery_success_rate: number;
    user_satisfaction_score: number;
  };
  active_alerts: number;
  recent_recovery_actions: number;
  uptime: string;
  protection_mode: string;
  is_active: boolean;
}

interface ProtectionSystem {
  system_name: string;
  is_active: boolean;
  health_score: number;
  last_action: string | null;
  last_action_time: string | null;
  metrics: Record<string, any>;
  alerts: string[];
}

interface Alert {
  type: string;
  severity: string;
  message: string;
  value: number;
  threshold: number;
  timestamp: string;
}

interface RecoveryAction {
  type: string;
  timestamp: string;
  metrics_snapshot: Record<string, number>;
  actions_taken: string[];
  success: boolean;
  error?: string;
}

const SystemHealthDashboard: React.FC = () => {
  const [healthData, setHealthData] = useState<SystemHealthMetrics | null>(null);
  const [protectionSystems, setProtectionSystems] = useState<ProtectionSystem[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [recoveryActions, setRecoveryActions] = useState<RecoveryAction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchSystemHealth = async () => {
    try {
      const response = await fetch('/api/bulletproof/health');
      if (!response.ok) throw new Error('Failed to fetch system health');
      const data = await response.json();
      setHealthData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/bulletproof/status');
      if (!response.ok) throw new Error('Failed to fetch system status');
      const data = await response.json();
      if (data.success && data.protection_systems) {
        setProtectionSystems(data.protection_systems);
      }
    } catch (err) {
      console.error('Failed to fetch system status:', err);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch('/api/bulletproof/alerts');
      if (!response.ok) throw new Error('Failed to fetch alerts');
      const data = await response.json();
      setAlerts(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Failed to fetch alerts:', err);
    }
  };

  const fetchRecoveryActions = async () => {
    try {
      const response = await fetch('/api/bulletproof/recovery-actions?hours=24');
      if (!response.ok) throw new Error('Failed to fetch recovery actions');
      const data = await response.json();
      setRecoveryActions(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Failed to fetch recovery actions:', err);
    }
  };

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      await Promise.all([
        fetchSystemHealth(),
        fetchSystemStatus(),
        fetchAlerts(),
        fetchRecoveryActions()
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllData();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchAllData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'excellent': return 'text-green-600 bg-green-50';
      case 'good': return 'text-blue-600 bg-blue-50';
      case 'degraded': return 'text-yellow-600 bg-yellow-50';
      case 'critical': return 'text-orange-600 bg-orange-50';
      case 'emergency': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'excellent': return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'good': return <CheckCircle className="h-5 w-5 text-blue-600" />;
      case 'degraded': return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
      case 'critical': return <AlertTriangle className="h-5 w-5 text-orange-600" />;
      case 'emergency': return <XCircle className="h-5 w-5 text-red-600" />;
      default: return <Activity className="h-5 w-5 text-gray-600" />;
    }
  };

  const getMetricTrend = (value: number, threshold: number) => {
    if (value > threshold * 1.1) return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (value < threshold * 0.9) return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Minus className="h-4 w-4 text-gray-500" />;
  };

  const formatUptime = (uptime: string) => {
    // Parse uptime string and format it nicely
    return uptime || '0:00:00';
  };

  const activateEmergencyMode = async () => {
    try {
      const response = await fetch('/api/bulletproof/emergency/activate', {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to activate emergency mode');
      await fetchAllData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to activate emergency mode');
    }
  };

  const deactivateEmergencyMode = async () => {
    try {
      const response = await fetch('/api/bulletproof/emergency/deactivate', {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to deactivate emergency mode');
      await fetchAllData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to deactivate emergency mode');
    }
  };

  if (loading && !healthData) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
        <span className="ml-2 text-lg">Loading system health...</span>
      </div>
    );
  }

  if (error && !healthData) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Failed to load system health data: {error}
          <Button onClick={fetchAllData} className="ml-2" size="sm">
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">System Health Dashboard</h1>
          <p className="text-gray-600">Real-time monitoring of bulletproof protection systems</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          <Button onClick={fetchAllData} size="sm" variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh Now
          </Button>
        </div>
      </div>

      {/* Overall Status */}
      {healthData && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {getStatusIcon(healthData.overall_status)}
                <CardTitle>System Status</CardTitle>
              </div>
              <Badge className={getStatusColor(healthData.overall_status)}>
                {healthData.overall_status.toUpperCase()}
              </Badge>
            </div>
            <CardDescription>
              System uptime: {formatUptime(healthData.uptime)} | 
              Protection mode: {healthData.protection_mode} |
              Last updated: {new Date(healthData.timestamp).toLocaleTimeString()}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{healthData.active_alerts}</div>
                <div className="text-sm text-gray-600">Active Alerts</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{healthData.recent_recovery_actions}</div>
                <div className="text-sm text-gray-600">Recent Recoveries</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {(healthData.metrics.protection_effectiveness * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">Protection Effectiveness</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {(healthData.metrics.user_satisfaction_score * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">User Satisfaction</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Emergency Controls */}
      {healthData && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Zap className="h-5 w-5 mr-2" />
              Emergency Controls
            </CardTitle>
            <CardDescription>
              Emergency mode provides maximum protection with reduced functionality
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4">
              {healthData.protection_mode === 'emergency_mode' ? (
                <Button onClick={deactivateEmergencyMode} variant="outline">
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Deactivate Emergency Mode
                </Button>
              ) : (
                <Button onClick={activateEmergencyMode} variant="destructive">
                  <AlertTriangle className="h-4 w-4 mr-2" />
                  Activate Emergency Mode
                </Button>
              )}
              <Badge variant={healthData.protection_mode === 'emergency_mode' ? 'destructive' : 'default'}>
                {healthData.protection_mode.replace('_', ' ').toUpperCase()}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tabs for detailed information */}
      <Tabs defaultValue="metrics" className="space-y-4">
        <TabsList>
          <TabsTrigger value="metrics">System Metrics</TabsTrigger>
          <TabsTrigger value="protection">Protection Systems</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="recovery">Recovery Actions</TabsTrigger>
        </TabsList>

        {/* System Metrics Tab */}
        <TabsContent value="metrics" className="space-y-4">
          {healthData && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* CPU Usage */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{healthData.metrics.cpu_usage.toFixed(1)}%</div>
                  <Progress value={healthData.metrics.cpu_usage} className="mt-2" />
                  <div className="flex items-center mt-2 text-xs text-muted-foreground">
                    {getMetricTrend(healthData.metrics.cpu_usage, 80)}
                    <span className="ml-1">Threshold: 80%</span>
                  </div>
                </CardContent>
              </Card>

              {/* Memory Usage */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
                  <HardDrive className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{healthData.metrics.memory_usage.toFixed(1)}%</div>
                  <Progress value={healthData.metrics.memory_usage} className="mt-2" />
                  <div className="flex items-center mt-2 text-xs text-muted-foreground">
                    {getMetricTrend(healthData.metrics.memory_usage, 85)}
                    <span className="ml-1">Threshold: 85%</span>
                  </div>
                </CardContent>
              </Card>

              {/* Response Time */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Response Time</CardTitle>
                  <Clock className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{healthData.metrics.response_time.toFixed(2)}s</div>
                  <div className="flex items-center mt-2 text-xs text-muted-foreground">
                    {getMetricTrend(healthData.metrics.response_time, 3.0)}
                    <span className="ml-1">Target: &lt; 3.0s</span>
                  </div>
                </CardContent>
              </Card>

              {/* Error Rate */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
                  <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{(healthData.metrics.error_rate * 100).toFixed(2)}%</div>
                  <div className="flex items-center mt-2 text-xs text-muted-foreground">
                    {getMetricTrend(healthData.metrics.error_rate, 0.05)}
                    <span className="ml-1">Target: &lt; 5%</span>
                  </div>
                </CardContent>
              </Card>

              {/* Active Users */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Users</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{healthData.metrics.active_users}</div>
                  <div className="text-xs text-muted-foreground mt-2">
                    Last 30 minutes
                  </div>
                </CardContent>
              </Card>

              {/* Network Latency */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Network Latency</CardTitle>
                  <Wifi className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{healthData.metrics.network_latency.toFixed(0)}ms</div>
                  <div className="text-xs text-muted-foreground mt-2">
                    Average latency
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Protection Systems Tab */}
        <TabsContent value="protection" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {protectionSystems.map((system) => (
              <Card key={system.system_name}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg capitalize">
                      {system.system_name.replace(/_/g, ' ')}
                    </CardTitle>
                    <div className="flex items-center space-x-2">
                      <Badge variant={system.is_active ? 'default' : 'secondary'}>
                        {system.is_active ? 'Active' : 'Inactive'}
                      </Badge>
                      <div className="flex items-center">
                        <Shield className="h-4 w-4 mr-1" />
                        <span className="text-sm font-medium">
                          {(system.health_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <Progress value={system.health_score * 100} />
                    {system.last_action && (
                      <div className="text-sm text-gray-600">
                        Last action: {system.last_action}
                        {system.last_action_time && (
                          <span className="ml-2">
                            ({new Date(system.last_action_time).toLocaleTimeString()})
                          </span>
                        )}
                      </div>
                    )}
                    {system.alerts.length > 0 && (
                      <div className="space-y-1">
                        {system.alerts.map((alert, index) => (
                          <Alert key={index} className="py-2">
                            <AlertTriangle className="h-4 w-4" />
                            <AlertDescription className="text-sm">{alert}</AlertDescription>
                          </Alert>
                        ))}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-4">
          {alerts.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center py-8">
                <div className="text-center">
                  <CheckCircle className="h-12 w-12 text-green-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900">No Active Alerts</h3>
                  <p className="text-gray-600">All systems are operating normally</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {alerts.map((alert, index) => (
                <Alert key={index} className={
                  alert.severity === 'critical' ? 'border-red-200 bg-red-50' :
                  alert.severity === 'warning' ? 'border-yellow-200 bg-yellow-50' :
                  'border-blue-200 bg-blue-50'
                }>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">{alert.message}</div>
                        <div className="text-sm text-gray-600 mt-1">
                          Value: {alert.value} | Threshold: {alert.threshold} |
                          Time: {new Date(alert.timestamp).toLocaleString()}
                        </div>
                      </div>
                      <Badge variant={
                        alert.severity === 'critical' ? 'destructive' :
                        alert.severity === 'warning' ? 'default' : 'secondary'
                      }>
                        {alert.severity.toUpperCase()}
                      </Badge>
                    </div>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Recovery Actions Tab */}
        <TabsContent value="recovery" className="space-y-4">
          {recoveryActions.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center py-8">
                <div className="text-center">
                  <Activity className="h-12 w-12 text-blue-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900">No Recent Recovery Actions</h3>
                  <p className="text-gray-600">System has been stable</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {recoveryActions.map((action, index) => (
                <Card key={index}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg capitalize">
                        {action.type.replace(/_/g, ' ')}
                      </CardTitle>
                      <div className="flex items-center space-x-2">
                        <Badge variant={action.success ? 'default' : 'destructive'}>
                          {action.success ? 'Success' : 'Failed'}
                        </Badge>
                        <span className="text-sm text-gray-600">
                          {new Date(action.timestamp).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <h4 className="font-medium mb-2">Actions Taken:</h4>
                        <ul className="list-disc list-inside space-y-1">
                          {action.actions_taken.map((actionTaken, actionIndex) => (
                            <li key={actionIndex} className="text-sm text-gray-700">
                              {actionTaken.replace(/_/g, ' ')}
                            </li>
                          ))}
                        </ul>
                      </div>
                      {action.error && (
                        <Alert className="border-red-200 bg-red-50">
                          <XCircle className="h-4 w-4" />
                          <AlertDescription>
                            <strong>Error:</strong> {action.error}
                          </AlertDescription>
                        </Alert>
                      )}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="font-medium">CPU:</span> {action.metrics_snapshot.cpu_usage?.toFixed(1)}%
                        </div>
                        <div>
                          <span className="font-medium">Memory:</span> {action.metrics_snapshot.memory_usage?.toFixed(1)}%
                        </div>
                        <div>
                          <span className="font-medium">Error Rate:</span> {(action.metrics_snapshot.error_rate * 100)?.toFixed(2)}%
                        </div>
                        <div>
                          <span className="font-medium">Response Time:</span> {action.metrics_snapshot.response_time?.toFixed(2)}s
                        </div>
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

export default SystemHealthDashboard;