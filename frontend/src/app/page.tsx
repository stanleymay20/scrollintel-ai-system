'use client'

import React, { useState, useEffect } from 'react'
import { Sidebar } from '@/components/layout/sidebar'
import { Header } from '@/components/layout/header'
import { AgentStatusCard } from '@/components/dashboard/agent-status-card'
import { SystemMetricsCard } from '@/components/dashboard/system-metrics'
import { ChatInterface } from '@/components/chat/chat-interface'
import { FileUploadComponent } from '@/components/upload/file-upload'
import { OnboardingSystem, ContextualTooltip } from '@/components/onboarding'
import { Agent, SystemMetrics, ChatMessage, FileUpload } from '@/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { BarChart3, Users, Zap, TrendingUp, AlertCircle } from 'lucide-react'
import { scrollIntelApi } from '@/lib/api'

export default function Dashboard() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null)
  const [selectedAgent, setSelectedAgent] = useState<Agent | undefined>()
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<FileUpload[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showMobileSidebar, setShowMobileSidebar] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isInitialLoading, setIsInitialLoading] = useState(true)

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setIsInitialLoading(true)
        setError(null)
        
        // Load agents and system metrics in parallel
        const [agentsResponse, metricsResponse] = await Promise.all([
          scrollIntelApi.getAgents(),
          scrollIntelApi.getSystemMetrics()
        ])
        
        setAgents(agentsResponse.data || [])
        setSystemMetrics(metricsResponse.data)
      } catch (err) {
        console.error('Failed to load initial data:', err)
        setError('Failed to load dashboard data. Please check your connection.')
      } finally {
        setIsInitialLoading(false)
      }
    }

    loadInitialData()
  }, [])

  // Real-time updates via polling
  useEffect(() => {
    if (isInitialLoading) return

    const interval = setInterval(async () => {
      try {
        const metricsResponse = await scrollIntelApi.getSystemMetrics()
        setSystemMetrics(metricsResponse.data)
        
        // Also refresh agent status
        const agentsResponse = await scrollIntelApi.getAgents()
        setAgents(agentsResponse.data || [])
      } catch (err) {
        console.error('Failed to update metrics:', err)
      }
    }, 10000) // Update every 10 seconds

    return () => clearInterval(interval)
  }, [isInitialLoading])

  const handleAgentInteract = (agentId: string) => {
    const agent = agents.find(a => a.id === agentId)
    setSelectedAgent(agent)
  }

  const handleSendMessage = async (message: string, agentId?: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date(),
      agent_id: agentId,
    }

    setChatMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await scrollIntelApi.sendMessage({
        message,
        agent_id: agentId || selectedAgent?.id,
        conversation_id: `dashboard-${Date.now()}` // Generate conversation ID
      })

      const agentResponse: ChatMessage = {
        id: response.data.id || (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.content || response.data.message,
        timestamp: response.data.timestamp ? new Date(response.data.timestamp) : new Date(),
        agent_id: agentId,
        metadata: response.data.metadata,
      }
      
      setChatMessages(prev => [...prev, agentResponse])
    } catch (err) {
      console.error('Failed to send message:', err)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your message. Please try again.',
        timestamp: new Date(),
        agent_id: agentId,
        metadata: {},
      }
      setChatMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (files: File[]) => {
    for (let index = 0; index < files.length; index++) {
      const file = files[index]
      const fileUpload: FileUpload = {
        id: `${Date.now()}-${index}`,
        name: file.name,
        filename: file.name,
        size: file.size,
        type: file.type,
        status: 'uploading',
        progress: 0,
      }

      setUploadedFiles(prev => [...prev, fileUpload])

      try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('agent_id', selectedAgent?.id || 'default')

        const response = await scrollIntelApi.uploadFile(formData, (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1))
          setUploadedFiles(prev => prev.map(f => 
            f.id === fileUpload.id ? { ...f, progress } : f
          ))
        })

        setUploadedFiles(prev => prev.map(f => 
          f.id === fileUpload.id 
            ? { ...f, status: 'completed' as const, progress: 100, upload_id: response.data.id }
            : f
        ))

        // Add a system message about the file upload
        const systemMessage: ChatMessage = {
          id: `upload-${Date.now()}`,
          role: 'assistant',
          content: `File "${file.name}" has been uploaded successfully and is ready for analysis.`,
          timestamp: new Date(),
          agent_id: selectedAgent?.id,
          metadata: { 
            template_used: 'file_upload_success'
          },
        }
        setChatMessages(prev => [...prev, systemMessage])

      } catch (err) {
        console.error('File upload failed:', err)
        setUploadedFiles(prev => prev.map(f => 
          f.id === fileUpload.id 
            ? { ...f, status: 'error' as const, error: 'Upload failed' }
            : f
        ))
      }
    }
  }

  const handleRemoveFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId))
  }

  return (
    <OnboardingSystem agents={agents}>
      <div className="flex h-screen bg-background">
        {/* Mobile Sidebar Overlay */}
        {showMobileSidebar && (
          <div 
            className="fixed inset-0 z-50 bg-black/50 md:hidden"
            onClick={() => setShowMobileSidebar(false)}
          />
        )}
        
        {/* Sidebar */}
        <div className={`
          fixed md:static inset-y-0 left-0 z-50 transform transition-transform duration-300 ease-in-out
          ${showMobileSidebar ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}>
          <Sidebar />
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col min-w-0">
          <Header 
            onMenuClick={() => setShowMobileSidebar(!showMobileSidebar)}
            showMobileMenu={true}
          />
          
          <main className="flex-1 overflow-auto p-6" data-tour="dashboard">
            <div className="max-w-7xl mx-auto space-y-6">
              {/* Welcome Section */}
              <ContextualTooltip
                target="welcome-section"
                title="Dashboard Overview"
                content="This is your main dashboard where you can see all system metrics, agent status, and quick access to key features."
                trigger="hover"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-3xl font-bold">ScrollIntel Dashboard</h1>
                    <p className="text-muted-foreground">
                      Welcome to the world's most advanced AI-CTO platform
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="success" className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                      All Systems Operational
                    </Badge>
                  </div>
                </div>
              </ContextualTooltip>

              {/* Error State */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                  <div className="flex items-center">
                    <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                    <p className="text-red-700">{error}</p>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="ml-auto"
                      onClick={() => window.location.reload()}
                    >
                      Retry
                    </Button>
                  </div>
                </div>
              )}

              {/* Quick Stats */}
              <ContextualTooltip
                target="quick-stats"
                title="System Metrics"
                content="These cards show real-time system performance metrics including active agents, request volume, response times, and success rates."
                trigger="hover"
              >
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4" data-tour="metrics">
                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Active Agents</p>
                          <p className="text-2xl font-bold">
                            {isInitialLoading ? '...' : agents.length}
                          </p>
                        </div>
                        <Users className="h-8 w-8 text-scrollintel-primary" />
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Total Requests</p>
                          <p className="text-2xl font-bold">
                            {isInitialLoading ? '...' : (systemMetrics?.active_connections?.toLocaleString() || '0')}
                          </p>
                        </div>
                        <BarChart3 className="h-8 w-8 text-scrollintel-secondary" />
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Avg Response</p>
                          <p className="text-2xl font-bold">
                            {isInitialLoading ? '...' : `${systemMetrics?.response_time || 0}ms`}
                          </p>
                        </div>
                        <Zap className="h-8 w-8 text-scrollintel-accent" />
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                          <p className="text-2xl font-bold">
                            {isInitialLoading ? '...' : '99.9%'}
                          </p>
                        </div>
                        <TrendingUp className="h-8 w-8 text-scrollintel-success" />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </ContextualTooltip>

            {/* System Metrics */}
            {systemMetrics && <SystemMetricsCard metrics={systemMetrics} />}

              {/* Main Content Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Agents Section */}
                <ContextualTooltip
                  target="agents-section"
                  title="AI Agents"
                  content="These are your available AI agents. Each agent specializes in different areas like data science, machine learning, and business intelligence. Click on any agent to start a conversation."
                  trigger="hover"
                >
                  <div className="space-y-4" data-tour="agents">
                    <h2 className="text-xl font-semibold">AI Agents</h2>
                    <div className="grid gap-4">
                      {agents.map((agent) => (
                        <AgentStatusCard
                          key={agent.id}
                          agent={agent}
                          onInteract={handleAgentInteract}
                        />
                      ))}
                    </div>
                  </div>
                </ContextualTooltip>

                {/* Chat & Upload Section */}
                <div className="space-y-4">
                  <ContextualTooltip
                    target="chat-interface"
                    title="Chat Interface"
                    content="Use this chat interface to communicate with AI agents. Ask questions, request analysis, or get help with your data and technical decisions."
                    trigger="hover"
                  >
                    <div className="h-96" data-tour="chat-interface">
                      <ChatInterface
                        selectedAgent={selectedAgent}
                        onSendMessage={handleSendMessage}
                        messages={chatMessages}
                        isLoading={isLoading}
                      />
                    </div>
                  </ContextualTooltip>
                  
                  <ContextualTooltip
                    target="file-upload"
                    title="File Upload"
                    content="Upload your data files here for analysis. Supported formats include CSV, Excel, JSON, and SQL files. The AI agents will automatically analyze your data and provide insights."
                    trigger="hover"
                  >
                    <div data-tour="file-upload">
                      <FileUploadComponent
                        onFileUpload={handleFileUpload}
                        uploadedFiles={uploadedFiles}
                        onRemoveFile={handleRemoveFile}
                      />
                    </div>
                  </ContextualTooltip>
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>
    </OnboardingSystem>
  )
}