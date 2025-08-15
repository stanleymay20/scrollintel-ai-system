'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Cpu, 
  Database, 
  MemoryStick, 
  Server, 
  Users, 
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Bell,
  BellOff
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'

interface MonitoringDashboardProps {
  refreshInterval?: number
}

interface SystemMetrics {
  cpu_percent: number
  memory_percent: number
  disk_percent: number
  active_connections: number
  request_rate: number
  error_rate: number
  avg_response_time: number
  agent_count: number
}

interface Alert {
  id: string
  name: string
  description: string
  severity: 'info' | 'warning' | 'critical' | 'emergency'
  status: 'active' | 'resolved' | 'acknowledged' | 'suppressed'
  metric_name: string
  current_value: number
  threshold: number
  timestamp: string
}

interface AnalyticsSummary {
  total_users: number
  active_users_24h: number
  active_users_7d: number
  active_users_30d: number
  total_sessions: number
  avg_session_duration: number
  bounce_rate: number
  top_events: Array<{ event: string; count: number }>
  top_pages: Array<{ page: string; count: number }>
}

export function MonitoringDashboard({ refreshInterval = 30000 }: MonitoringDashboardProps) {
  const [dashboardData, setDashboardData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/api/monitoring/dashboard')
      if (!response.ok) {
        throw new Error('Failed to fetch dashboard data')
      }
      const data = await response.json()
      setDashboardData(data)
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDashboardData()
  }, [])

  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(fetchDashboardData, refreshInterval)
    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'emergency':
        return 'bg-red-500'
      case 'warning':
        return 'bg-yellow-500'
      case 'info':
        return 'bg-blue-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getMetricTrend = (current: number, threshold: number) => {
    if (current > threshold * 1.1) return { icon: TrendingUp, color: 'text-red-500' }
    if (current < threshold * 0.9) return { icon: TrendingDown, color: 'text-green-500' }
    return { icon: Minus, color: 'text-yellow-500' }
  }

  const formatBytes = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    if (bytes === 0) return '0 Bytes'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i]
  }

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    return `${hours}h ${minutes}m ${secs}s`
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading monitoring dashboard...</span>
      </div>
    )
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center text-red-500">
            <AlertTriangle className="h-5 w-5 mr-2" />
            <span>Error loading dashboard: {error}</span>
          </div>
          <Button onClick={fetchDashboardData} className="mt-4">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    )
  }

  const systemMetrics = dashboardData?.system?.current
  const alerts = dashboardData?.alerts?.active || []
  const analytics = dashboardData?.analytics
  const agentStats = dashboardData?.agents?.agent_usage || []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Monitoring</h1>
          <p className="text-muted-foreground">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? <Bell className="h-4 w-4" /> : <BellOff className="h-4 w-4" />}
            {autoRefresh ? 'Auto-refresh On' : 'Auto-refresh Off'}
          </Button>
          <Button onClick={fetchDashboardData} size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Alert Summary */}
      {alerts.length > 0 && (
        <Card className="border-red-200 bg-red-50">
          <CardHeader>
            <CardTitle className="flex items-center text-red-700">
              <AlertTriangle className="h-5 w-5 mr-2" />
              Active Alerts ({alerts.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {alerts.slice(0, 3).map((alert: Alert) => (
                <div key={alert.id} className="flex items-center justify-between p-2 bg-white rounded">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${getSeverityColor(alert.severity)}`} />
                    <span className="font-medium">{alert.name}</span>
                    <Badge variant="outline">{alert.severity}</Badge>
                  </div>
                  <span className="text-sm text-muted-foreground">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              ))}
              {alerts.length > 3 && (
                <p className="text-sm text-muted-foreground">
                  And {alerts.length - 3} more alerts...
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="system">System Resources</TabsTrigger>
          <TabsTrigger value="agents">Agent Performance</TabsTrigger>
          <TabsTrigger value="analytics">User Analytics</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* System Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
                <Cpu className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemMetrics?.cpu_percent?.toFixed(1) || 0}%
                </div>
                <Progress value={systemMetrics?.cpu_percent || 0} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
                <MemoryStick className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemMetrics?.memory_percent?.toFixed(1) || 0}%
                </div>
                <Progress value={systemMetrics?.memory_percent || 0} className="mt-2" />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
                <Server className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemMetrics?.agent_count || 0}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Agents running
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Response Time</CardTitle>
                <Clock className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {systemMetrics?.avg_response_time?.toFixed(0) || 0}ms
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Average response
                </p>
              </CardContent>
            </Card>
          </div>

          {/* System Metrics Chart */}
          {dashboardData?.system?.history && (
            <Card>
              <CardHeader>
                <CardTitle>System Performance Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={dashboardData.system.history.slice(-24)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="cpu_percent" 
                      stroke="#8884d8" 
                      name="CPU %"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="memory_percent" 
                      stroke="#82ca9d" 
                      name="Memory %"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="disk_percent" 
                      stroke="#ffc658" 
                      name="Disk %"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          {/* Detailed System Resources */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Resource Usage</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm">
                    <span>CPU Usage</span>
                    <span>{systemMetrics?.cpu_percent?.toFixed(1) || 0}%</span>
                  </div>
                  <Progress value={systemMetrics?.cpu_percent || 0} className="mt-1" />
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span>Memory Usage</span>
                    <span>{systemMetrics?.memory_percent?.toFixed(1) || 0}%</span>
                  </div>
                  <Progress value={systemMetrics?.memory_percent || 0} className="mt-1" />
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span>Disk Usage</span>
                    <span>{systemMetrics?.disk_percent?.toFixed(1) || 0}%</span>
                  </div>
                  <Progress value={systemMetrics?.disk_percent || 0} className="mt-1" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Database & Cache</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-sm">Database Connections</span>
                  <span className="font-medium">
                    {dashboardData?.database?.active_connections || 0} / {dashboardData?.database?.max_connections || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Cache Hit Rate</span>
                  <span className="font-medium">
                    {dashboardData?.redis?.hit_rate?.toFixed(1) || 0}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Redis Memory</span>
                  <span className="font-medium">
                    {formatBytes(dashboardData?.redis?.used_memory || 0)}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="agents" className="space-y-4">
          {/* Agent Performance */}
          <Card>
            <CardHeader>
              <CardTitle>Agent Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {agentStats.map((agent: any) => (
                  <div key={agent.agent_type} className="flex items-center justify-between p-3 border rounded">
                    <div>
                      <h4 className="font-medium">{agent.agent_type}</h4>
                      <p className="text-sm text-muted-foreground">
                        {agent.requests} requests • {agent.avg_duration}s avg
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium">
                        {agent.success_rate}% success
                      </div>
                      <Badge variant={agent.success_rate > 95 ? "default" : "destructive"}>
                        {agent.successful_requests}/{agent.requests}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          {/* User Analytics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {analytics?.total_users?.toLocaleString() || 0}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active (24h)</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {analytics?.active_users_24h?.toLocaleString() || 0}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active (7d)</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {analytics?.active_users_7d?.toLocaleString() || 0}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Sessions</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {analytics?.total_sessions?.toLocaleString() || 0}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Top Events and Pages */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Top Events</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {analytics?.top_events?.slice(0, 5).map((event: any, index: number) => (
                    <div key={event.event} className="flex justify-between">
                      <span className="text-sm">{event.event}</span>
                      <span className="font-medium">{event.count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top Pages</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {analytics?.top_pages?.slice(0, 5).map((page: any, index: number) => (
                    <div key={page.page} className="flex justify-between">
                      <span className="text-sm truncate">{page.page}</span>
                      <span className="font-medium">{page.count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          {/* Alerts Management */}
          <Card>
            <CardHeader>
              <CardTitle>Alert Management</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.length === 0 ? (
                  <div className="text-center py-8">
                    <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                    <h3 className="text-lg font-medium">All Clear!</h3>
                    <p className="text-muted-foreground">No active alerts at this time.</p>
                  </div>
                ) : (
                  alerts.map((alert: Alert) => (
                    <div key={alert.id} className="border rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-3">
                          <div className={`w-3 h-3 rounded-full mt-1 ${getSeverityColor(alert.severity)}`} />
                          <div>
                            <h4 className="font-medium">{alert.name}</h4>
                            <p className="text-sm text-muted-foreground">{alert.description}</p>
                            <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                              <span>Metric: {alert.metric_name}</span>
                              <span>Current: {alert.current_value}</span>
                              <span>Threshold: {alert.threshold}</span>
                              <span>{new Date(alert.timestamp).toLocaleString()}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <Badge variant="outline">{alert.severity}</Badge>
                          <Badge variant={alert.status === 'active' ? 'destructive' : 'default'}>
                            {alert.status}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}