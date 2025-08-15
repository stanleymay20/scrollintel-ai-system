'use client'

import React from 'react'
import { cn } from '@/lib/utils'
import { Bot, User, Zap, Brain, BarChart3, Code, Database, Lightbulb } from 'lucide-react'

interface AgentAvatarProps {
  agentId: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  status?: 'active' | 'busy' | 'inactive' | 'thinking'
  showStatus?: boolean
  className?: string
  animated?: boolean
}

const agentConfigs = {
  'scroll-cto-agent': {
    name: 'Alex Chen',
    icon: Code,
    colorScheme: 'blue',
    background: 'bg-blue-500',
    statusColor: 'border-blue-400',
    description: 'CTO Advisor'
  },
  'scroll-data-scientist': {
    name: 'Dr. Sarah Kim',
    icon: BarChart3,
    colorScheme: 'purple',
    background: 'bg-purple-500',
    statusColor: 'border-purple-400',
    description: 'Data Scientist'
  },
  'scroll-ml-engineer': {
    name: 'Marcus Rodriguez',
    icon: Brain,
    colorScheme: 'green',
    background: 'bg-green-500',
    statusColor: 'border-green-400',
    description: 'ML Engineer'
  },
  'scroll-bi-agent': {
    name: 'Emma Thompson',
    icon: Database,
    colorScheme: 'orange',
    background: 'bg-orange-500',
    statusColor: 'border-orange-400',
    description: 'BI Specialist'
  },
  'scroll-analyst': {
    name: 'Jordan Park',
    icon: Lightbulb,
    colorScheme: 'teal',
    background: 'bg-teal-500',
    statusColor: 'border-teal-400',
    description: 'Business Analyst'
  }
}

const sizeClasses = {
  sm: 'w-8 h-8',
  md: 'w-12 h-12',
  lg: 'w-16 h-16',
  xl: 'w-24 h-24'
}

const iconSizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8',
  xl: 'w-12 h-12'
}

const statusIndicatorSizes = {
  sm: 'w-2 h-2',
  md: 'w-3 h-3',
  lg: 'w-4 h-4',
  xl: 'w-5 h-5'
}

export function AgentAvatar({
  agentId,
  size = 'md',
  status = 'active',
  showStatus = true,
  className,
  animated = false
}: AgentAvatarProps) {
  const config = agentConfigs[agentId as keyof typeof agentConfigs] || {
    name: 'Agent',
    icon: Bot,
    colorScheme: 'gray',
    background: 'bg-gray-500',
    statusColor: 'border-gray-400',
    description: 'AI Agent'
  }

  const IconComponent = config.icon

  const getStatusColor = () => {
    switch (status) {
      case 'active':
        return 'bg-green-400'
      case 'busy':
        return 'bg-yellow-400'
      case 'thinking':
        return 'bg-blue-400 animate-pulse'
      case 'inactive':
        return 'bg-gray-400'
      default:
        return 'bg-green-400'
    }
  }

  const getStatusBorder = () => {
    switch (status) {
      case 'active':
        return 'border-green-300'
      case 'busy':
        return 'border-yellow-300'
      case 'thinking':
        return 'border-blue-300'
      case 'inactive':
        return 'border-gray-300'
      default:
        return 'border-green-300'
    }
  }

  return (
    <div className={cn('relative inline-block', className)}>
      {/* Main Avatar */}
      <div
        className={cn(
          'rounded-full flex items-center justify-center text-white shadow-lg transition-all duration-200',
          config.background,
          sizeClasses[size],
          animated && 'hover:scale-105',
          status === 'thinking' && 'animate-pulse'
        )}
      >
        <IconComponent className={cn('text-white', iconSizeClasses[size])} />
      </div>

      {/* Status Indicator */}
      {showStatus && (
        <div
          className={cn(
            'absolute -bottom-0.5 -right-0.5 rounded-full border-2 border-white',
            statusIndicatorSizes[size],
            getStatusColor(),
            getStatusBorder()
          )}
        />
      )}

      {/* Thinking Animation Overlay */}
      {status === 'thinking' && (
        <div className="absolute inset-0 rounded-full border-2 border-blue-400 animate-ping opacity-75" />
      )}

      {/* Busy Animation Overlay */}
      {status === 'busy' && (
        <div className="absolute top-0 right-0">
          <Zap className="w-3 h-3 text-yellow-400 animate-bounce" />
        </div>
      )}
    </div>
  )
}

interface AgentAvatarWithInfoProps extends AgentAvatarProps {
  showName?: boolean
  showDescription?: boolean
  layout?: 'horizontal' | 'vertical'
}

export function AgentAvatarWithInfo({
  agentId,
  size = 'md',
  status = 'active',
  showStatus = true,
  showName = true,
  showDescription = false,
  layout = 'horizontal',
  className,
  animated = false
}: AgentAvatarWithInfoProps) {
  const config = agentConfigs[agentId as keyof typeof agentConfigs] || {
    name: 'Agent',
    icon: Bot,
    colorScheme: 'gray',
    background: 'bg-gray-500',
    statusColor: 'border-gray-400',
    description: 'AI Agent'
  }

  return (
    <div
      className={cn(
        'flex items-center gap-3',
        layout === 'vertical' && 'flex-col text-center',
        className
      )}
    >
      <AgentAvatar
        agentId={agentId}
        size={size}
        status={status}
        showStatus={showStatus}
        animated={animated}
      />
      
      {(showName || showDescription) && (
        <div className={cn('flex flex-col', layout === 'vertical' && 'items-center')}>
          {showName && (
            <div className="font-medium text-sm text-foreground">
              {config.name}
            </div>
          )}
          {showDescription && (
            <div className="text-xs text-muted-foreground">
              {config.description}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

interface AgentAvatarGroupProps {
  agentIds: string[]
  size?: 'sm' | 'md' | 'lg'
  maxVisible?: number
  showStatus?: boolean
  className?: string
}

export function AgentAvatarGroup({
  agentIds,
  size = 'md',
  maxVisible = 3,
  showStatus = true,
  className
}: AgentAvatarGroupProps) {
  const visibleAgents = agentIds.slice(0, maxVisible)
  const remainingCount = agentIds.length - maxVisible

  return (
    <div className={cn('flex -space-x-2', className)}>
      {visibleAgents.map((agentId, index) => (
        <div
          key={agentId}
          className="relative"
          style={{ zIndex: visibleAgents.length - index }}
        >
          <AgentAvatar
            agentId={agentId}
            size={size}
            showStatus={showStatus}
            className="border-2 border-white"
          />
        </div>
      ))}
      
      {remainingCount > 0 && (
        <div
          className={cn(
            'rounded-full bg-gray-200 border-2 border-white flex items-center justify-center text-gray-600 font-medium text-xs',
            sizeClasses[size]
          )}
          style={{ zIndex: 0 }}
        >
          +{remainingCount}
        </div>
      )}
    </div>
  )
}

// Animated typing indicator component
export function TypingIndicator({ agentId, message }: { agentId: string; message?: string }) {
  const config = agentConfigs[agentId as keyof typeof agentConfigs] || {
    name: 'Agent',
    icon: Bot,
    colorScheme: 'gray',
    background: 'bg-gray-500',
    statusColor: 'border-gray-400',
    description: 'AI Agent'
  }

  return (
    <div className="flex items-center gap-3 p-3 bg-muted rounded-lg">
      <AgentAvatar
        agentId={agentId}
        size="sm"
        status="thinking"
        animated
      />
      
      <div className="flex items-center gap-2">
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-scrollintel-primary rounded-full animate-bounce" />
          <div 
            className="w-2 h-2 bg-scrollintel-primary rounded-full animate-bounce" 
            style={{ animationDelay: '0.1s' }} 
          />
          <div 
            className="w-2 h-2 bg-scrollintel-primary rounded-full animate-bounce" 
            style={{ animationDelay: '0.2s' }} 
          />
        </div>
        
        <span className="text-sm text-muted-foreground">
          {message || `${config.name} is thinking...`}
        </span>
      </div>
    </div>
  )
}

// Status badge component
export function AgentStatusBadge({ 
  status, 
  className 
}: { 
  status: 'active' | 'busy' | 'inactive' | 'thinking'
  className?: string 
}) {
  const getStatusConfig = () => {
    switch (status) {
      case 'active':
        return {
          label: 'Available',
          color: 'bg-green-100 text-green-800 border-green-200',
          icon: '●'
        }
      case 'busy':
        return {
          label: 'Busy',
          color: 'bg-yellow-100 text-yellow-800 border-yellow-200',
          icon: '●'
        }
      case 'thinking':
        return {
          label: 'Thinking',
          color: 'bg-blue-100 text-blue-800 border-blue-200',
          icon: '●'
        }
      case 'inactive':
        return {
          label: 'Offline',
          color: 'bg-gray-100 text-gray-800 border-gray-200',
          icon: '●'
        }
      default:
        return {
          label: 'Unknown',
          color: 'bg-gray-100 text-gray-800 border-gray-200',
          icon: '●'
        }
    }
  }

  const config = getStatusConfig()

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border',
        config.color,
        className
      )}
    >
      <span className={cn(status === 'thinking' && 'animate-pulse')}>
        {config.icon}
      </span>
      {config.label}
    </span>
  )
}