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
  // Ensure agent has required properties with defaults
  const safeAgent = {
    id: agent?.id || 'unknown',
    name: agent?.name || 'Unknown Agent',
    description: agent?.description || 'No description available',
    status: agent?.status || 'inactive' as const,
    capabilities: agent?.capabilities || []
  }

  const getStatusColor = (status: Agent['status']) => {
    switch (status) {
      case 'active':
        return 'default'
      case 'busy':
        return 'secondary'
      case 'inactive':
        return 'outline'
      case 'thinking':
        return 'secondary'
      default:
        return 'outline'
    }
  }

  const getStatusIcon = (status: Agent['status']) => {
    switch (status) {
      case 'active':
        return <Activity className="h-4 w-4" />
      case 'busy':
        return <Zap className="h-4 w-4" />
      case 'thinking':
        return <Zap className="h-4 w-4" />
      case 'inactive':
        return <Clock className="h-4 w-4" />
      default:
        return <Clock className="h-4 w-4" />
    }
  }

  return (
    <Card className={cn(
      "transition-all duration-200 hover:shadow-lg cursor-pointer",
      safeAgent.status === 'active' && "ring-2 ring-green-200",
      safeAgent.status === 'busy' && "ring-2 ring-yellow-200",
      safeAgent.status === 'thinking' && "ring-2 ring-blue-200"
    )}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">{safeAgent.name}</CardTitle>
          <Badge variant={getStatusColor(safeAgent.status)} className="flex items-center gap-1">
            {getStatusIcon(safeAgent.status)}
            {safeAgent.status}
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground">{safeAgent.description}</p>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div>
          <h4 className="text-sm font-medium mb-2">Capabilities</h4>
          <div className="flex flex-wrap gap-1">
            {safeAgent.capabilities.length > 0 ? (
              <>
                {safeAgent.capabilities.slice(0, 3).map((capability, index) => (
                  <Badge key={`${capability}-${index}`} variant="outline" className="text-xs">
                    {capability}
                  </Badge>
                ))}
                {safeAgent.capabilities.length > 3 && (
                  <Badge variant="outline" className="text-xs">
                    +{safeAgent.capabilities.length - 3} more
                  </Badge>
                )}
              </>
            ) : (
              <Badge variant="outline" className="text-xs">
                General AI Assistant
              </Badge>
            )}
          </div>
        </div>

        <Button 
          onClick={() => onInteract?.(safeAgent.id)}
          className="w-full"
          variant="default"
          disabled={safeAgent.status === 'inactive'}
        >
          Chat with {safeAgent.name}
        </Button>
      </CardContent>
    </Card>
  )
}