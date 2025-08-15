'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Agent } from '@/types'
import { Activity, Clock, Zap, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface AgentStatusCardProps {
  agent: Agent
  onInteract?: (agentId: string) => void
}

export function AgentStatusCard({ agent, onInteract }: AgentStatusCardProps) {
  const getStatusColor = (status: Agent['status']) => {
    switch (status) {
      case 'active':
        return 'success'
      case 'busy':
        return 'warning'
      case 'inactive':
        return 'secondary'
      case 'error':
        return 'error'
      default:
        return 'secondary'
    }
  }

  const getStatusIcon = (status: Agent['status']) => {
    switch (status) {
      case 'active':
        return <Activity className="h-4 w-4" />
      case 'busy':
        return <Zap className="h-4 w-4" />
      case 'inactive':
        return <Clock className="h-4 w-4" />
      case 'error':
        return <AlertCircle className="h-4 w-4" />
      default:
        return <Clock className="h-4 w-4" />
    }
  }

  return (
    <Card className={cn(
      "transition-all duration-200 hover:shadow-lg",
      agent.status === 'active' && "ring-2 ring-scrollintel-success/20",
      agent.status === 'busy' && "animate-pulse-glow"
    )}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">{agent.name}</CardTitle>
          <Badge variant={getStatusColor(agent.status)} className="flex items-center gap-1">
            {getStatusIcon(agent.status)}
            {agent.status}
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground">{agent.description}</p>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div>
          <h4 className="text-sm font-medium mb-2">Capabilities</h4>
          <div className="flex flex-wrap gap-1">
            {agent.capabilities.slice(0, 3).map((capability) => (
              <Badge key={capability} variant="outline" className="text-xs">
                {capability}
              </Badge>
            ))}
            {agent.capabilities.length > 3 && (
              <Badge variant="outline" className="text-xs">
                +{agent.capabilities.length - 3} more
              </Badge>
            )}
          </div>
        </div>

        {agent.metrics && (
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="text-center">
              <div className="font-semibold text-scrollintel-primary">
                {agent.metrics.requests_handled}
              </div>
              <div className="text-muted-foreground">Requests</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-scrollintel-secondary">
                {agent.metrics.avg_response_time}ms
              </div>
              <div className="text-muted-foreground">Avg Time</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-scrollintel-success">
                {agent.metrics.success_rate}%
              </div>
              <div className="text-muted-foreground">Success</div>
            </div>
          </div>
        )}

        <Button 
          onClick={() => onInteract?.(agent.id)}
          className="w-full"
          variant="scrollintel"
          disabled={agent.status === 'inactive' || agent.status === 'error'}
        >
          Interact with {agent.name}
        </Button>
      </CardContent>
    </Card>
  )
}