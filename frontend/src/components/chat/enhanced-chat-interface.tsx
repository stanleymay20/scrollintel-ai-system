'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { AgentAvatar, AgentAvatarWithInfo, TypingIndicator } from '@/components/agents/agent-avatar'
import { ChatMessage, Agent } from '@/types'
import { Send, Bot, User, Clock, Zap, Smile, ThumbsUp, ThumbsDown, Copy, Share } from 'lucide-react'
import { cn } from '@/lib/utils'
import { formatDate } from '@/lib/utils'

interface EnhancedChatInterfaceProps {
  selectedAgent?: Agent
  onSendMessage?: (message: string, agentId?: string) => void
  messages?: ChatMessage[]
  isLoading?: boolean
  onMessageFeedback?: (messageId: string, feedback: 'positive' | 'negative') => void
  onCopyMessage?: (content: string) => void
  onShareConversation?: () => void
  conversationId?: string
  showPersonality?: boolean
  enableStreaming?: boolean
}

interface StreamingMessage {
  id: string
  content: string
  isComplete: boolean
  chunks: string[]
}

export function EnhancedChatInterface({ 
  selectedAgent, 
  onSendMessage, 
  messages = [], 
  isLoading = false,
  onMessageFeedback,
  onCopyMessage,
  onShareConversation,
  conversationId,
  showPersonality = true,
  enableStreaming = true
}: EnhancedChatInterfaceProps) {
  const [inputMessage, setInputMessage] = useState('')
  const [streamingMessage, setStreamingMessage] = useState<StreamingMessage | null>(null)
  const [typingIndicatorMessage, setTypingIndicatorMessage] = useState<string>('')
  const [showTypingIndicator, setShowTypingIndicator] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const scrollToBottom = () => {
    if (messagesEndRef.current?.scrollIntoView) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingMessage, showTypingIndicator])

  // WebSocket connection for real-time streaming
  useEffect(() => {
    if (enableStreaming && conversationId) {
      const wsUrl = `ws://localhost:8000/ws/chat/${conversationId}`
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
      }

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          handleWebSocketMessage(data)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected')
      }

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      return () => {
        if (wsRef.current) {
          wsRef.current.close()
        }
      }
    }
  }, [conversationId, enableStreaming])

  const handleWebSocketMessage = useCallback((data: any) => {
    switch (data.event_type) {
      case 'typing_start':
        setTypingIndicatorMessage(data.data.message)
        setShowTypingIndicator(true)
        break
      
      case 'typing_stop':
        setShowTypingIndicator(false)
        break
      
      case 'response_chunk':
        setStreamingMessage(prev => {
          if (!prev || prev.id !== data.data.stream_id) {
            return {
              id: data.data.stream_id,
              content: data.data.chunk,
              isComplete: false,
              chunks: [data.data.chunk]
            }
          }
          return {
            ...prev,
            content: data.data.partial_response,
            chunks: [...prev.chunks, data.data.chunk]
          }
        })
        break
      
      case 'response_complete':
        setStreamingMessage(prev => prev ? { ...prev, isComplete: true } : null)
        setShowTypingIndicator(false)
        // Clear streaming message after a delay to show completion
        setTimeout(() => setStreamingMessage(null), 1000)
        break
      
      case 'error':
        setShowTypingIndicator(false)
        setStreamingMessage(null)
        console.error('Agent error:', data.data.error)
        break
    }
  }, [])

  const handleSendMessage = () => {
    if (!inputMessage.trim()) return
    
    onSendMessage?.(inputMessage, selectedAgent?.id)
    setInputMessage('')
    inputRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleMessageFeedback = (messageId: string, feedback: 'positive' | 'negative') => {
    onMessageFeedback?.(messageId, feedback)
  }

  const handleCopyMessage = (content: string) => {
    navigator.clipboard.writeText(content)
    onCopyMessage?.(content)
  }

  const getAgentPersonalityGreeting = (agent: Agent) => {
    const greetings = {
      'scroll-cto-agent': "👋 Hi! I'm Alex, your CTO advisor. Ready to architect something amazing?",
      'scroll-data-scientist': "Hello! I'm Dr. Sarah Kim. Let's uncover insights in your data! 📊",
      'scroll-ml-engineer': "Hey! Marcus here. Let's build some intelligent systems! 🤖",
      'scroll-bi-agent': "Hi there! I'm Emma. Let's turn your data into business value! 💼",
      'scroll-analyst': "Hello! I'm Jordan. Ready to analyze and optimize your business? 📈"
    }
    return greetings[agent.id as keyof typeof greetings] || `Hi! I'm ${agent.name}, ready to help!`
  }

  const renderMessage = (message: ChatMessage) => (
    <div
      key={message.id}
      className={cn(
        "flex gap-3 group",
        message.role === 'user' ? "justify-end" : "justify-start"
      )}
    >
      <div className={cn(
        "flex gap-3 max-w-[80%]",
        message.role === 'user' ? "flex-row-reverse" : "flex-row"
      )}>
        {/* Avatar */}
        <div className="flex-shrink-0">
          {message.role === 'user' ? (
            <div className="w-8 h-8 rounded-full bg-scrollintel-primary text-white flex items-center justify-center">
              <User className="h-4 w-4" />
            </div>
          ) : (
            <AgentAvatar
              agentId={message.agent_id || selectedAgent?.id || 'scroll-cto-agent'}
              size="sm"
              status="active"
            />
          )}
        </div>
        
        {/* Message Content */}
        <div className={cn(
          "rounded-lg px-4 py-3 space-y-2 relative",
          message.role === 'user'
            ? "bg-scrollintel-primary text-white"
            : "bg-muted"
        )}>
          {/* Agent Name for Assistant Messages */}
          {message.role === 'assistant' && selectedAgent && showPersonality && (
            <div className="text-xs font-medium text-scrollintel-primary mb-1">
              {selectedAgent.name}
            </div>
          )}
          
          {/* Message Text */}
          <div className="text-sm whitespace-pre-wrap">
            {message.content}
          </div>
          
          {/* Message Metadata */}
          <div className="flex items-center gap-2 text-xs opacity-70">
            <Clock className="h-3 w-3" />
            {formatDate(message.timestamp)}
            {message.metadata?.processing_time && (
              <>
                <Zap className="h-3 w-3" />
                {message.metadata.processing_time}ms
              </>
            )}
          </div>
          
          {/* Message Actions */}
          {message.role === 'assistant' && (
            <div className="flex items-center gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={() => handleCopyMessage(message.content)}
              >
                <Copy className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={() => handleMessageFeedback(message.id, 'positive')}
              >
                <ThumbsUp className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={() => handleMessageFeedback(message.id, 'negative')}
              >
                <ThumbsDown className="h-3 w-3" />
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )

  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-scrollintel-primary" />
            ScrollIntel Chat
          </CardTitle>
          <div className="flex items-center gap-2">
            {selectedAgent && (
              <AgentAvatarWithInfo
                agentId={selectedAgent.id}
                size="sm"
                status={selectedAgent.status}
                showName={true}
                showDescription={false}
                layout="horizontal"
              />
            )}
            {onShareConversation && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onShareConversation}
              >
                <Share className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col min-h-0">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4 min-h-0">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <div className="text-center space-y-4">
                {selectedAgent ? (
                  <AgentAvatarWithInfo
                    agentId={selectedAgent.id}
                    size="lg"
                    status={selectedAgent.status}
                    showName={true}
                    showDescription={true}
                    layout="vertical"
                    animated
                  />
                ) : (
                  <Bot className="h-12 w-12 mx-auto opacity-50" />
                )}
                
                {selectedAgent && showPersonality ? (
                  <div className="space-y-2">
                    <p className="text-sm">{getAgentPersonalityGreeting(selectedAgent)}</p>
                    <p className="text-xs text-muted-foreground">
                      Ask questions, upload files, or request analysis
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <p>Start a conversation with ScrollIntel agents</p>
                    <p className="text-sm">Ask questions, upload files, or request analysis</p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            messages.map(renderMessage)
          )}
          
          {/* Streaming Message */}
          {streamingMessage && (
            <div className="flex gap-3">
              <AgentAvatar
                agentId={selectedAgent?.id || 'scroll-cto-agent'}
                size="sm"
                status="thinking"
                animated
              />
              <div className="bg-muted rounded-lg px-4 py-3 max-w-[80%]">
                <div className="text-sm whitespace-pre-wrap">
                  {streamingMessage.content}
                  {!streamingMessage.isComplete && (
                    <span className="inline-block w-2 h-4 bg-scrollintel-primary animate-pulse ml-1" />
                  )}
                </div>
              </div>
            </div>
          )}
          
          {/* Typing Indicator */}
          {showTypingIndicator && !streamingMessage && (
            <TypingIndicator
              agentId={selectedAgent?.id || 'scroll-cto-agent'}
              message={typingIndicatorMessage}
            />
          )}
          
          {/* Loading Indicator (fallback) */}
          {isLoading && !showTypingIndicator && !streamingMessage && (
            <TypingIndicator
              agentId={selectedAgent?.id || 'scroll-cto-agent'}
            />
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="flex gap-2">
          <Input
            ref={inputRef}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selectedAgent 
                ? `Ask ${selectedAgent.name} anything...` 
                : "Type your message..."
            }
            disabled={isLoading || showTypingIndicator}
            className="flex-1"
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading || showTypingIndicator}
            variant="scrollintel"
            size="icon"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>

        {/* Agent Status Bar */}
        {selectedAgent && (
          <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className={cn(
                "w-2 h-2 rounded-full",
                selectedAgent.status === 'active' && "bg-green-400",
                selectedAgent.status === 'busy' && "bg-yellow-400 animate-pulse",
                selectedAgent.status === 'inactive' && "bg-gray-400"
              )} />
              <span>
                {selectedAgent.status === 'active' && 'Ready to help'}
                {selectedAgent.status === 'busy' && 'Processing...'}
                {selectedAgent.status === 'inactive' && 'Offline'}
              </span>
            </div>
            
            {enableStreaming && (
              <div className="flex items-center gap-1">
                <Zap className="h-3 w-3" />
                <span>Real-time</span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}