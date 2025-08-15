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
import { BarChart3, Users, Zap, TrendingUp } from 'lucide-react'

// Mock data for development
const mockAgents: Agent[] = [
  {
    id: 'cto-1',
    name: 'ScrollCTO',
    type: 'CTO',
    status: 'active',
    capabilities: ['Architecture', 'Scaling', 'Tech Stack', 'Cost Analysis'],
    description: 'AI-powered CTO for technical decisions and architecture planning',
    last_active: new Date().toISOString(),
    metrics: {
      requests_handled: 1247,
      avg_response_time: 850,
      success_rate: 98.5,
    },
  },
  {
    id: 'ds-1',
    name: 'ScrollDataScientist',
    type: 'DataScientist',
    status: 'busy',
    capabilities: ['EDA', 'Statistical Analysis', 'Hypothesis Testing', 'Feature Engineering'],
    description: 'Advanced data science and statistical analysis capabilities',
    last_active: new Date().toISOString(),
    metrics: {
      requests_handled: 892,
      avg_response_time: 1200,
      success_rate: 96.8,
    },
  },
  {
    id: 'ml-1',
    name: 'ScrollMLEngineer',
    type: 'MLEngineer',
    status: 'active',
    capabilities: ['Model Training', 'MLOps', 'Deployment', 'Monitoring'],
    description: 'Machine learning engineering and model deployment',
    last_active: new Date().toISOString(),
    metrics: {
      requests_handled: 634,
      avg_response_time: 2100,
      success_rate: 94.2,
    },
  },
  {
    id: 'ai-1',
    name: 'ScrollAIEngineer',
    type: 'AIEngineer',
    status: 'active',
    capabilities: ['RAG', 'LLM Integration', 'Vector Search', 'Embeddings'],
    description: 'AI engineering with memory-enhanced capabilities',
    last_active: new Date().toISOString(),
    metrics: {
      requests_handled: 1456,
      avg_response_time: 950,
      success_rate: 97.3,
    },
  },
]

const mockSystemMetrics: SystemMetrics = {
  cpu_usage: 45.2,
  memory_usage: 67.8,
  active_agents: 12,
  total_requests: 4229,
  avg_response_time: 1125,
  error_rate: 2.1,
}

export default function Dashboard() {
  const [agents, setAgents] = useState<Agent[]>(mockAgents)
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>(mockSystemMetrics)
  const [selectedAgent, setSelectedAgent] = useState<Agent | undefined>()
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<FileUpload[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showMobileSidebar, setShowMobileSidebar] = useState(false)

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemMetrics(prev => ({
        ...prev,
        cpu_usage: Math.max(20, Math.min(90, prev.cpu_usage + (Math.random() - 0.5) * 10)),
        memory_usage: Math.max(30, Math.min(95, prev.memory_usage + (Math.random() - 0.5) * 5)),
        total_requests: prev.total_requests + Math.floor(Math.random() * 5),
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const handleAgentInteract = (agentId: string) => {
    const agent = agents.find(a => a.id === agentId)
    setSelectedAgent(agent)
  }

  const handleSendMessage = async (message: string, agentId?: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
      agent_id: agentId,
    }

    setChatMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    // Simulate API call
    setTimeout(() => {
      const agentResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I understand you want to ${message}. Let me help you with that. This is a simulated response from ${selectedAgent?.name || 'ScrollIntel'}.`,
        timestamp: new Date().toISOString(),
        agent_id: agentId,
        metadata: {
          processing_time: Math.floor(Math.random() * 2000) + 500,
          model_used: 'GPT-4',
        },
      }
      setChatMessages(prev => [...prev, agentResponse])
      setIsLoading(false)
    }, 1500)
  }

  const handleFileUpload = (files: File[]) => {
    files.forEach((file, index) => {
      const fileUpload: FileUpload = {
        id: `${Date.now()}-${index}`,
        filename: file.name,
        size: file.size,
        type: file.type,
        status: 'uploading',
        progress: 0,
      }

      setUploadedFiles(prev => [...prev, fileUpload])

      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadedFiles(prev => prev.map(f => {
          if (f.id === fileUpload.id) {
            const newProgress = Math.min(100, f.progress + Math.random() * 20)
            if (newProgress >= 100) {
              clearInterval(interval)
              return { ...f, progress: 100, status: 'completed' as const }
            }
            return { ...f, progress: newProgress }
          }
          return f
        }))
      }, 200)
    })
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
                          <p className="text-2xl font-bold">{systemMetrics.active_agents}</p>
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
                          <p className="text-2xl font-bold">{systemMetrics.total_requests.toLocaleString()}</p>
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
                          <p className="text-2xl font-bold">{systemMetrics.avg_response_time}ms</p>
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
                          <p className="text-2xl font-bold">96.8%</p>
                        </div>
                        <TrendingUp className="h-8 w-8 text-scrollintel-success" />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </ContextualTooltip>

            {/* System Metrics */}
            <SystemMetricsCard metrics={systemMetrics} />

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