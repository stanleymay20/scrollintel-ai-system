'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { RoleBasedDashboard } from './role-based-dashboard'
import { NaturalLanguageInterface } from './natural-language-interface-simple'
import { InteractiveVisualization } from './interactive-visualization'
import { MobileResponsiveInterface } from './mobile-responsive-interface'
import { 
  Monitor, 
  Smartphone, 
  Tablet, 
  User, 
  Settings, 
  BarChart3, 
  MessageSquare,
  Grid,
  Maximize2,
  Minimize2,
  RefreshCw,
  Download,
  Share,
  Bell,
  Search,
  Filter,
  Eye,
  EyeOff
} from 'lucide-react'

interface UserProfile {
  id: string
  name: string
  email: string
  role: 'executive' | 'analyst' | 'technical'
  permissions: string[]
  preferences: UserPreferences
}

interface UserPreferences {
  theme: 'light' | 'dark' | 'auto'
  language: string
  timezone: string
  dashboardLayout: 'compact' | 'comfortable' | 'spacious'
  autoRefresh: boolean
  refreshInterval: number
  notifications: boolean
}

interface EnterpriseUserInterfaceProps {
  user?: UserProfile
  onUserChange?: (user: UserProfile) => void
  className?: string
}

export function EnterpriseUserInterface({ user, onUserChange, className }: EnterpriseUserInterfaceProps) {
  const [currentUser, setCurrentUser] = useState<UserProfile>(user || {
    id: '1',
    name: 'John Executive',
    email: 'john@company.com',
    role: 'executive',
    permissions: ['read', 'write', 'admin'],
    preferences: {
      theme: 'light',
      language: 'en',
      timezone: 'UTC',
      dashboardLayout: 'comfortable',
      autoRefresh: true,
      refreshInterval: 30000,
      notifications: true
    }
  })

  const [activeTab, setActiveTab] = useState('dashboard')
  const [deviceView, setDeviceView] = useState<'desktop' | 'tablet' | 'mobile'>('desktop')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [lastActivity, setLastActivity] = useState<Date>(new Date())

  useEffect(() => {
    // Detect device type based on screen size
    const updateDeviceView = () => {
      const width = window.innerWidth
      if (width < 768) {
        setDeviceView('mobile')
      } else if (width < 1024) {
        setDeviceView('tablet')
      } else {
        setDeviceView('desktop')
      }
    }

    updateDeviceView()
    window.addEventListener('resize', updateDeviceView)

    // Track user activity
    const trackActivity = () => setLastActivity(new Date())
    document.addEventListener('mousedown', trackActivity)
    document.addEventListener('keydown', trackActivity)

    return () => {
      window.removeEventListener('resize', updateDeviceView)
      document.removeEventListener('mousedown', trackActivity)
      document.removeEventListener('keydown', trackActivity)
    }
  }, [])

  const handleRoleChange = (newRole: 'executive' | 'analyst' | 'technical') => {
    const updatedUser = { ...currentUser, role: newRole }
    setCurrentUser(updatedUser)
    if (onUserChange) {
      onUserChange(updatedUser)
    }
  }

  const handlePreferencesChange = (preferences: Partial<UserPreferences>) => {
    const updatedUser = {
      ...currentUser,
      preferences: { ...currentUser.preferences, ...preferences }
    }
    setCurrentUser(updatedUser)
    if (onUserChange) {
      onUserChange(updatedUser)
    }
  }

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }

  const exportDashboard = (format: 'pdf' | 'png' | 'csv') => {
    console.log(`Exporting dashboard as ${format}`)
    // Implementation for dashboard export
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'executive': return 'bg-blue-100 text-blue-800'
      case 'analyst': return 'bg-green-100 text-green-800'
      case 'technical': return 'bg-purple-100 text-purple-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getDeviceIcon = (device: string) => {
    switch (device) {
      case 'mobile': return <Smartphone className="h-4 w-4" />
      case 'tablet': return <Tablet className="h-4 w-4" />
      default: return <Monitor className="h-4 w-4" />
    }
  }

  // Mobile view - use dedicated mobile interface
  if (deviceView === 'mobile') {
    return (
      <MobileResponsiveInterface 
        userRole={currentUser.role}
        className={className}
      />
    )
  }

  return (
    <div className={`min-h-screen bg-background ${isFullscreen ? 'fixed inset-0 z-50' : ''} ${className}`}>
      {/* Header */}
      <div className="sticky top-0 z-40 bg-background border-b">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="text-2xl font-bold">ScrollIntel Enterprise</h1>
              <p className="text-sm text-muted-foreground">
                Advanced AI-Powered Business Intelligence Platform
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Device View Selector */}
            <div className="flex items-center gap-1 p-1 bg-muted rounded-lg">
              <Button
                variant={deviceView === 'desktop' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setDeviceView('desktop')}
              >
                <Monitor className="h-4 w-4" />
              </Button>
              <Button
                variant={deviceView === 'tablet' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setDeviceView('tablet')}
              >
                <Tablet className="h-4 w-4" />
              </Button>
              <Button
                variant={(deviceView as string) === 'mobile' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setDeviceView('mobile')}
              >
                <Smartphone className="h-4 w-4" />
              </Button>
            </div>

            {/* User Info */}
            <div className="flex items-center gap-3">
              <div className="text-right">
                <p className="text-sm font-medium">{currentUser.name}</p>
                <Badge className={getRoleColor(currentUser.role)}>
                  {currentUser.role}
                </Badge>
              </div>
              <Button variant="ghost" size="sm">
                <User className="h-5 w-5" />
              </Button>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm">
                <Bell className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="sm">
                <Search className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="sm" onClick={toggleFullscreen}>
                {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setShowSettings(!showSettings)}>
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="border-t bg-muted/50 px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div>
                  <label className="text-sm font-medium">Role:</label>
                  <select 
                    value={currentUser.role} 
                    onChange={(e) => handleRoleChange(e.target.value as any)}
                    className="ml-2 px-2 py-1 border rounded text-sm"
                  >
                    <option value="executive">Executive</option>
                    <option value="analyst">Analyst</option>
                    <option value="technical">Technical</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">Layout:</label>
                  <select 
                    value={currentUser.preferences.dashboardLayout} 
                    onChange={(e) => handlePreferencesChange({ dashboardLayout: e.target.value as any })}
                    className="ml-2 px-2 py-1 border rounded text-sm"
                  >
                    <option value="compact">Compact</option>
                    <option value="comfortable">Comfortable</option>
                    <option value="spacious">Spacious</option>
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={currentUser.preferences.autoRefresh}
                    onChange={(e) => handlePreferencesChange({ autoRefresh: e.target.checked })}
                    className="rounded"
                  />
                  <label className="text-sm">Auto-refresh</label>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button size="sm" variant="outline" onClick={() => exportDashboard('pdf')}>
                  <Download className="h-4 w-4 mr-1" />
                  Export
                </Button>
                <Button size="sm" variant="outline">
                  <Share className="h-4 w-4 mr-1" />
                  Share
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
          <div className="border-b px-6">
            <TabsList className="grid w-full max-w-md grid-cols-4">
              <TabsTrigger value="dashboard" className="flex items-center gap-2">
                <Grid className="h-4 w-4" />
                Dashboard
              </TabsTrigger>
              <TabsTrigger value="query" className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                Query
              </TabsTrigger>
              <TabsTrigger value="visualizations" className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Charts
              </TabsTrigger>
              <TabsTrigger value="mobile" className="flex items-center gap-2">
                <Smartphone className="h-4 w-4" />
                Mobile
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="p-6">
            <TabsContent value="dashboard" className="mt-0">
              <RoleBasedDashboard 
                userRole={currentUser.role}
                userId={currentUser.id}
                onRoleChange={handleRoleChange}
              />
            </TabsContent>

            <TabsContent value="query" className="mt-0">
              <NaturalLanguageInterface 
                onQuerySubmit={(query) => console.log('Query submitted:', query)}
              />
            </TabsContent>

            <TabsContent value="visualizations" className="mt-0">
              <InteractiveVisualization 
                onVisualizationChange={(viz) => console.log('Visualization changed:', viz)}
              />
            </TabsContent>

            <TabsContent value="mobile" className="mt-0">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Smartphone className="h-5 w-5" />
                    Mobile Interface Preview
                  </CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Preview how the interface appears on mobile devices
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="max-w-sm mx-auto border rounded-lg overflow-hidden">
                    <MobileResponsiveInterface 
                      userRole={currentUser.role}
                      className="scale-75 origin-top"
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </div>
        </Tabs>
      </div>

      {/* Status Bar */}
      <div className="border-t bg-muted/50 px-6 py-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <span>Connected as {currentUser.name}</span>
            <span>Role: {currentUser.role}</span>
            <span className="flex items-center gap-1">
              {getDeviceIcon(deviceView)}
              {deviceView} view
            </span>
          </div>
          <div className="flex items-center gap-4">
            <span>Last activity: {lastActivity.toLocaleTimeString()}</span>
            <span>System status: Operational</span>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Live</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EnterpriseUserInterface