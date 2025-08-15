'use client'

import { AgentStatusCard } from '@/components/dashboard/agent-status-card'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Bot, Play, Pause, Settings } from 'lucide-react'

const mockAgents = [
  {
    id: '1',
    name: 'ScrollAI Engineer',
    status: 'active' as const,
    description: 'Advanced code generation and architecture design',
    capabilities: ['Code Generation', 'Architecture Design', 'Testing', 'Documentation'],
    lastActive: '2 minutes ago'
  },
  {
    id: '2', 
    name: 'ScrollBI Agent',
    status: 'idle' as const,
    description: 'Business intelligence and data analysis',
    capabilities: ['Data Analysis', 'Report Generation', 'Visualization', 'Insights'],
    lastActive: '15 minutes ago'
  },
  {
    id: '3',
    name: 'ScrollML Engineer', 
    status: 'active' as const,
    description: 'Machine learning model development and deployment',
    capabilities: ['Model Training', 'Feature Engineering', 'MLOps', 'Optimization'],
    lastActive: '5 minutes ago'
  },
  {
    id: '4',
    name: 'ScrollQA Engine',
    status: 'error' as const,
    description: 'Quality assurance and testing automation',
    capabilities: ['Test Generation', 'Bug Detection', 'Performance Testing', 'Security Audits'],
    lastActive: '1 hour ago'
  }
]

export default function AgentsPage() {
  return (
    <div className="flex-1 p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">AI Agents</h1>
        <p className="text-muted-foreground">Manage and monitor your ScrollIntel AI agents</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {mockAgents.map((agent) => (
          <Card key={agent.id} className="relative">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Bot className="h-5 w-5" />
                  <CardTitle className="text-lg">{agent.name}</CardTitle>
                </div>
                <Badge 
                  variant={
                    agent.status === 'active' ? 'success' : 
                    agent.status === 'error' ? 'error' : 'secondary'
                  }
                >
                  {agent.status}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">{agent.description}</p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium mb-2">Capabilities</h4>
                  <div className="flex flex-wrap gap-1">
                    {agent.capabilities.map((capability) => (
                      <Badge key={capability} variant="outline" className="text-xs">
                        {capability}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                <div className="text-xs text-muted-foreground">
                  Last active: {agent.lastActive}
                </div>

                <div className="flex gap-2">
                  <Button size="sm" variant="outline">
                    <Play className="h-3 w-3 mr-1" />
                    Start
                  </Button>
                  <Button size="sm" variant="outline">
                    <Pause className="h-3 w-3 mr-1" />
                    Pause
                  </Button>
                  <Button size="sm" variant="outline">
                    <Settings className="h-3 w-3 mr-1" />
                    Config
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}