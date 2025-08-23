'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  BarChart3, 
  Users, 
  TrendingUp, 
  AlertTriangle, 
  Shield, 
  Database,
  Cpu,
  Network,
  DollarSign,
  Clock,
  CheckCircle,
  XCircle,
  Activity,
  Settings,
  Eye,
  Target
} from 'lucide-react'

interface UserRole {
  id: string
  name: string
  permissions: string[]
  dashboardConfig: DashboardConfig
}

interface DashboardConfig {
  widgets: Widget[]
  layout: 'executive' | 'analyst' | 'technical'
  refreshInterval: number
}

interface Widget {
  id: string
  type: 'metric' | 'chart' | 'table' | 'alert' | 'status'
  title: string
  data: any
  size: 'small' | 'medium' | 'large'
  priority: number
}

interface RoleBasedDashboardProps {
  userRole: 'executive' | 'analyst' | 'technical'
  userId: string
  onRoleChange?: (role: string) => void
}

export function RoleBasedDashboard({ userRole, userId, onRoleChange }: RoleBasedDashboardProps) {
  const [dashboardData, setDashboardData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [alerts, setAlerts] = useState<any[]>([])

  useEffect(() => {
    loadDashboardData()
    const interval = setInterval(loadDashboardData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [userRole, selectedTimeRange])

  const loadDashboardData = async () => {
    try {
      setIsLoading(true)
      // Simulate API call - replace with actual API
      const response = await fetch(`/api/dashboard/${userRole}?timeRange=${selectedTimeRange}`)
      const data = await response.json()
      setDashboardData(data)
      setAlerts(data.alerts || [])
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const renderExecutiveDashboard = () => (
    <div className="space-y-6">
      {/* Executive Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Business Value Generated</p>
                <p className="text-2xl font-bold text-green-600">$2.4M</p>
                <p className="text-xs text-muted-foreground">+15% from last month</p>
              </div>
              <DollarSign className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Cost Savings</p>
                <p className="text-2xl font-bold text-blue-600">$890K</p>
                <p className="text-xs text-muted-foreground">+8% efficiency gain</p>
              </div>
              <TrendingUp className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Decision Accuracy</p>
                <p className="text-2xl font-bold text-purple-600">94.7%</p>
                <p className="text-xs text-muted-foreground">AI-assisted decisions</p>
              </div>
              <Target className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">System Uptime</p>
                <p className="text-2xl font-bold text-green-600">99.97%</p>
                <p className="text-xs text-muted-foreground">Enterprise SLA met</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Strategic Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Strategic Insights & Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
              <h4 className="font-semibold text-blue-900">Market Opportunity Identified</h4>
              <p className="text-blue-800 text-sm mt-1">
                AI analysis suggests expanding into healthcare analytics could generate $1.2M additional revenue
              </p>
              <Button size="sm" className="mt-2" variant="outline">
                View Full Analysis
              </Button>
            </div>
            <div className="p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
              <h4 className="font-semibold text-green-900">Operational Excellence</h4>
              <p className="text-green-800 text-sm mt-1">
                Process automation has reduced manual work by 67%, freeing up 40 hours/week for strategic tasks
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const renderAnalystDashboard = () => (
    <div className="space-y-6">
      {/* Analyst Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Data Processing Rate</p>
                <p className="text-2xl font-bold">847 GB/hr</p>
                <p className="text-xs text-muted-foreground">Real-time analytics</p>
              </div>
              <Database className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Model Accuracy</p>
                <p className="text-2xl font-bold">96.3%</p>
                <p className="text-xs text-muted-foreground">Across all models</p>
              </div>
              <Target className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Insights Generated</p>
                <p className="text-2xl font-bold">1,247</p>
                <p className="text-xs text-muted-foreground">This week</p>
              </div>
              <BarChart3 className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Quality Dashboard */}
      <Card>
        <CardHeader>
          <CardTitle>Data Quality & Pipeline Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <div className="flex items-center gap-3">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <div>
                  <p className="font-medium">Customer Data Pipeline</p>
                  <p className="text-sm text-muted-foreground">Last updated: 2 minutes ago</p>
                </div>
              </div>
              <Badge variant="success">Healthy</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
              <div className="flex items-center gap-3">
                <AlertTriangle className="h-5 w-5 text-yellow-600" />
                <div>
                  <p className="font-medium">Sales Data Pipeline</p>
                  <p className="text-sm text-muted-foreground">Data quality score: 87%</p>
                </div>
              </div>
              <Badge variant="warning">Attention Needed</Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Analytics Tools */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Analytics Tools</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
              <BarChart3 className="h-6 w-6 mb-2" />
              <span className="text-sm">Query Builder</span>
            </Button>
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
              <Database className="h-6 w-6 mb-2" />
              <span className="text-sm">Data Explorer</span>
            </Button>
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
              <TrendingUp className="h-6 w-6 mb-2" />
              <span className="text-sm">Trend Analysis</span>
            </Button>
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
              <Target className="h-6 w-6 mb-2" />
              <span className="text-sm">ML Models</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const renderTechnicalDashboard = () => (
    <div className="space-y-6">
      {/* System Health Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">CPU Usage</p>
                <p className="text-2xl font-bold">23.4%</p>
                <p className="text-xs text-muted-foreground">Across all nodes</p>
              </div>
              <Cpu className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Memory Usage</p>
                <p className="text-2xl font-bold">67.8%</p>
                <p className="text-xs text-muted-foreground">24GB / 32GB</p>
              </div>
              <Activity className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Network I/O</p>
                <p className="text-2xl font-bold">1.2 GB/s</p>
                <p className="text-xs text-muted-foreground">Inbound/Outbound</p>
              </div>
              <Network className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Agents</p>
                <p className="text-2xl font-bold">47</p>
                <p className="text-xs text-muted-foreground">Healthy instances</p>
              </div>
              <Users className="h-8 w-8 text-orange-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            System Alerts & Monitoring
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {alerts.length > 0 ? alerts.map((alert, index) => (
              <div key={index} className={`p-3 rounded-lg border-l-4 ${
                alert.severity === 'critical' ? 'bg-red-50 border-red-500' :
                alert.severity === 'warning' ? 'bg-yellow-50 border-yellow-500' :
                'bg-blue-50 border-blue-500'
              }`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{alert.title}</p>
                    <p className="text-sm text-muted-foreground">{alert.description}</p>
                  </div>
                  <Badge variant={alert.severity === 'critical' ? 'destructive' : 'warning'}>
                    {alert.severity}
                  </Badge>
                </div>
              </div>
            )) : (
              <div className="text-center py-8 text-muted-foreground">
                <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-600" />
                <p>All systems operating normally</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Infrastructure Management */}
      <Card>
        <CardHeader>
          <CardTitle>Infrastructure Management</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
              <Settings className="h-6 w-6 mb-2" />
              <span className="text-sm">Configuration</span>
            </Button>
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
              <Shield className="h-6 w-6 mb-2" />
              <span className="text-sm">Security</span>
            </Button>
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
              <Database className="h-6 w-6 mb-2" />
              <span className="text-sm">Databases</span>
            </Button>
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center">
              <Network className="h-6 w-6 mb-2" />
              <span className="text-sm">Networking</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with Role Selector */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Enterprise Dashboard</h1>
          <p className="text-muted-foreground">
            {userRole === 'executive' ? 'Strategic overview and business insights' :
             userRole === 'analyst' ? 'Data analysis and business intelligence' :
             'System monitoring and technical operations'}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <select 
            value={selectedTimeRange} 
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-2 border rounded-md"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <Badge variant="outline" className="capitalize">
            {userRole} View
          </Badge>
        </div>
      </div>

      {/* Role-specific Dashboard Content */}
      {userRole === 'executive' && renderExecutiveDashboard()}
      {userRole === 'analyst' && renderAnalystDashboard()}
      {userRole === 'technical' && renderTechnicalDashboard()}
    </div>
  )
}