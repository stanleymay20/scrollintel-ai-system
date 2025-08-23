'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { 
  Menu, 
  Bell, 
  Search, 
  MoreVertical,
  TrendingUp, 
  Users, 
  DollarSign,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Smartphone,
  Tablet,
  Monitor,
  Wifi,
  WifiOff,
  RefreshCw,
  Settings,
  User,
  LogOut,
  Home,
  BarChart3,
  Database,
  Shield
} from 'lucide-react'

interface MobileMetric {
  id: string
  title: string
  value: string
  change: string
  trend: 'up' | 'down' | 'stable'
  icon: React.ReactNode
  color: string
}

interface MobileAlert {
  id: string
  title: string
  message: string
  severity: 'info' | 'warning' | 'error' | 'success'
  timestamp: Date
  read: boolean
}

interface MobileResponsiveInterfaceProps {
  userRole: 'executive' | 'analyst' | 'technical'
  className?: string
}

export function MobileResponsiveInterface({ userRole, className }: MobileResponsiveInterfaceProps) {
  const [isOnline, setIsOnline] = useState(true)
  const [deviceType, setDeviceType] = useState<'mobile' | 'tablet' | 'desktop'>('mobile')
  const [notifications, setNotifications] = useState<MobileAlert[]>([])
  const [metrics, setMetrics] = useState<MobileMetric[]>([])
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [lastSync, setLastSync] = useState<Date>(new Date())

  // Sample metrics based on user role
  const getMetricsForRole = (role: string): MobileMetric[] => {
    const baseMetrics = {
      executive: [
        {
          id: '1',
          title: 'Revenue',
          value: '$2.4M',
          change: '+15.3%',
          trend: 'up' as const,
          icon: <DollarSign className="h-5 w-5" />,
          color: 'text-green-600'
        },
        {
          id: '2',
          title: 'Cost Savings',
          value: '$890K',
          change: '+8.2%',
          trend: 'up' as const,
          icon: <TrendingUp className="h-5 w-5" />,
          color: 'text-blue-600'
        },
        {
          id: '3',
          title: 'Decision Accuracy',
          value: '94.7%',
          change: '+2.1%',
          trend: 'up' as const,
          icon: <CheckCircle className="h-5 w-5" />,
          color: 'text-purple-600'
        },
        {
          id: '4',
          title: 'System Uptime',
          value: '99.97%',
          change: 'Stable',
          trend: 'stable' as const,
          icon: <Activity className="h-5 w-5" />,
          color: 'text-green-600'
        }
      ],
      analyst: [
        {
          id: '1',
          title: 'Data Processed',
          value: '847 GB/hr',
          change: '+12%',
          trend: 'up' as const,
          icon: <Database className="h-5 w-5" />,
          color: 'text-blue-600'
        },
        {
          id: '2',
          title: 'Model Accuracy',
          value: '96.3%',
          change: '+1.8%',
          trend: 'up' as const,
          icon: <BarChart3 className="h-5 w-5" />,
          color: 'text-green-600'
        },
        {
          id: '3',
          title: 'Insights Generated',
          value: '1,247',
          change: '+23%',
          trend: 'up' as const,
          icon: <TrendingUp className="h-5 w-5" />,
          color: 'text-purple-600'
        },
        {
          id: '4',
          title: 'Active Pipelines',
          value: '12',
          change: '+2',
          trend: 'up' as const,
          icon: <Activity className="h-5 w-5" />,
          color: 'text-orange-600'
        }
      ],
      technical: [
        {
          id: '1',
          title: 'CPU Usage',
          value: '23.4%',
          change: '-5%',
          trend: 'down' as const,
          icon: <Activity className="h-5 w-5" />,
          color: 'text-blue-600'
        },
        {
          id: '2',
          title: 'Memory Usage',
          value: '67.8%',
          change: '+3%',
          trend: 'up' as const,
          icon: <Database className="h-5 w-5" />,
          color: 'text-yellow-600'
        },
        {
          id: '3',
          title: 'Active Agents',
          value: '47',
          change: '+2',
          trend: 'up' as const,
          icon: <Users className="h-5 w-5" />,
          color: 'text-green-600'
        },
        {
          id: '4',
          title: 'Response Time',
          value: '127ms',
          change: '-15ms',
          trend: 'down' as const,
          icon: <Clock className="h-5 w-5" />,
          color: 'text-purple-600'
        }
      ]
    }
    return baseMetrics[role] || baseMetrics.executive
  }

  // Sample notifications
  const sampleNotifications: MobileAlert[] = [
    {
      id: '1',
      title: 'System Alert',
      message: 'High memory usage detected on server cluster 3',
      severity: 'warning',
      timestamp: new Date(Date.now() - 300000),
      read: false
    },
    {
      id: '2',
      title: 'Revenue Milestone',
      message: 'Monthly revenue target exceeded by 15%',
      severity: 'success',
      timestamp: new Date(Date.now() - 600000),
      read: false
    },
    {
      id: '3',
      title: 'Data Pipeline',
      message: 'Customer data sync completed successfully',
      severity: 'info',
      timestamp: new Date(Date.now() - 900000),
      read: true
    }
  ]

  useEffect(() => {
    // Detect device type
    const updateDeviceType = () => {
      const width = window.innerWidth
      if (width < 768) {
        setDeviceType('mobile')
      } else if (width < 1024) {
        setDeviceType('tablet')
      } else {
        setDeviceType('desktop')
      }
    }

    updateDeviceType()
    window.addEventListener('resize', updateDeviceType)

    // Monitor online status
    const updateOnlineStatus = () => setIsOnline(navigator.onLine)
    window.addEventListener('online', updateOnlineStatus)
    window.addEventListener('offline', updateOnlineStatus)

    // Load initial data
    setMetrics(getMetricsForRole(userRole))
    setNotifications(sampleNotifications)

    return () => {
      window.removeEventListener('resize', updateDeviceType)
      window.removeEventListener('online', updateOnlineStatus)
      window.removeEventListener('offline', updateOnlineStatus)
    }
  }, [userRole])

  const handleSync = async () => {
    try {
      // Simulate data sync
      setLastSync(new Date())
      setMetrics(getMetricsForRole(userRole))
    } catch (error) {
      console.error('Sync failed:', error)
    }
  }

  const getDeviceIcon = () => {
    switch (deviceType) {
      case 'mobile': return <Smartphone className="h-4 w-4" />
      case 'tablet': return <Tablet className="h-4 w-4" />
      default: return <Monitor className="h-4 w-4" />
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error': return 'text-red-600 bg-red-50 border-red-200'
      case 'warning': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'success': return 'text-green-600 bg-green-50 border-green-200'
      default: return 'text-blue-600 bg-blue-50 border-blue-200'
    }
  }

  const unreadCount = notifications.filter(n => !n.read).length

  return (
    <div className={`min-h-screen bg-background ${className}`}>
      {/* Mobile Header */}
      <div className="sticky top-0 z-40 bg-background border-b px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Sheet open={isMenuOpen} onOpenChange={setIsMenuOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="sm">
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-80">
                <SheetHeader>
                  <SheetTitle>ScrollIntel Mobile</SheetTitle>
                </SheetHeader>
                <div className="mt-6 space-y-4">
                  <Button variant="ghost" className="w-full justify-start">
                    <Home className="h-4 w-4 mr-2" />
                    Dashboard
                  </Button>
                  <Button variant="ghost" className="w-full justify-start">
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Analytics
                  </Button>
                  <Button variant="ghost" className="w-full justify-start">
                    <Users className="h-4 w-4 mr-2" />
                    Agents
                  </Button>
                  <Button variant="ghost" className="w-full justify-start">
                    <Database className="h-4 w-4 mr-2" />
                    Data
                  </Button>
                  <Button variant="ghost" className="w-full justify-start">
                    <Settings className="h-4 w-4 mr-2" />
                    Settings
                  </Button>
                  <div className="border-t pt-4">
                    <Button variant="ghost" className="w-full justify-start">
                      <User className="h-4 w-4 mr-2" />
                      Profile
                    </Button>
                    <Button variant="ghost" className="w-full justify-start">
                      <LogOut className="h-4 w-4 mr-2" />
                      Sign Out
                    </Button>
                  </div>
                </div>
              </SheetContent>
            </Sheet>
            <div>
              <h1 className="font-semibold">ScrollIntel</h1>
              <p className="text-xs text-muted-foreground capitalize">{userRole} Dashboard</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="relative">
              <Bell className="h-5 w-5" />
              {unreadCount > 0 && (
                <Badge className="absolute -top-1 -right-1 h-5 w-5 p-0 text-xs">
                  {unreadCount}
                </Badge>
              )}
            </Button>
            <Button variant="ghost" size="sm">
              <Search className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>

      {/* Connection Status */}
      <div className={`px-4 py-2 text-sm ${isOnline ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isOnline ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />}
            <span>{isOnline ? 'Connected' : 'Offline Mode'}</span>
            {getDeviceIcon()}
            <span className="capitalize">{deviceType}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs">Last sync: {lastSync.toLocaleTimeString()}</span>
            <Button variant="ghost" size="sm" onClick={handleSync}>
              <RefreshCw className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-4 space-y-4">
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-3">
          {metrics.map((metric) => (
            <Card key={metric.id} className="p-3">
              <div className="flex items-center justify-between mb-2">
                <div className={metric.color}>
                  {metric.icon}
                </div>
                <Badge variant={metric.trend === 'up' ? 'default' : metric.trend === 'down' ? 'secondary' : 'outline'} className="text-xs">
                  {metric.change}
                </Badge>
              </div>
              <div>
                <p className="text-lg font-bold">{metric.value}</p>
                <p className="text-xs text-muted-foreground">{metric.title}</p>
              </div>
            </Card>
          ))}
        </div>

        {/* Quick Actions */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              <Button variant="outline" className="h-16 flex flex-col items-center justify-center">
                <BarChart3 className="h-6 w-6 mb-1" />
                <span className="text-xs">Analytics</span>
              </Button>
              <Button variant="outline" className="h-16 flex flex-col items-center justify-center">
                <Users className="h-6 w-6 mb-1" />
                <span className="text-xs">Agents</span>
              </Button>
              <Button variant="outline" className="h-16 flex flex-col items-center justify-center">
                <Database className="h-6 w-6 mb-1" />
                <span className="text-xs">Data</span>
              </Button>
              <Button variant="outline" className="h-16 flex flex-col items-center justify-center">
                <Shield className="h-6 w-6 mb-1" />
                <span className="text-xs">Security</span>
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Recent Notifications */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center justify-between">
              Recent Alerts
              <Badge variant="outline">{unreadCount} new</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {notifications.slice(0, 3).map((notification) => (
              <div
                key={notification.id}
                className={`p-3 rounded-lg border ${getSeverityColor(notification.severity)} ${
                  !notification.read ? 'border-l-4' : ''
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-sm">{notification.title}</p>
                    <p className="text-xs mt-1 opacity-80">{notification.message}</p>
                    <p className="text-xs mt-2 opacity-60">
                      {notification.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                  <Button variant="ghost" size="sm">
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
            <Button variant="outline" className="w-full" size="sm">
              View All Notifications
            </Button>
          </CardContent>
        </Card>

        {/* System Status */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">System Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm">All Systems</span>
              <Badge variant="default" className="bg-green-600">
                <CheckCircle className="h-3 w-3 mr-1" />
                Operational
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">API Services</span>
              <Badge variant="default" className="bg-green-600">
                <CheckCircle className="h-3 w-3 mr-1" />
                Healthy
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Data Pipeline</span>
              <Badge variant="outline" className="bg-yellow-50 text-yellow-700">
                <AlertCircle className="h-3 w-3 mr-1" />
                Monitoring
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Navigation (Mobile Only) */}
      {deviceType === 'mobile' && (
        <div className="fixed bottom-0 left-0 right-0 bg-background border-t px-4 py-2">
          <div className="flex items-center justify-around">
            <Button variant="ghost" size="sm" className="flex flex-col items-center">
              <Home className="h-5 w-5" />
              <span className="text-xs mt-1">Home</span>
            </Button>
            <Button variant="ghost" size="sm" className="flex flex-col items-center">
              <BarChart3 className="h-5 w-5" />
              <span className="text-xs mt-1">Charts</span>
            </Button>
            <Button variant="ghost" size="sm" className="flex flex-col items-center">
              <Bell className="h-5 w-5" />
              <span className="text-xs mt-1">Alerts</span>
            </Button>
            <Button variant="ghost" size="sm" className="flex flex-col items-center">
              <Settings className="h-5 w-5" />
              <span className="text-xs mt-1">Settings</span>
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}