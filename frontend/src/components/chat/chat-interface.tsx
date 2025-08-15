'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ChatMessage, Agent } from '@/types'
import { Send, Bot, User, Clock, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'
import { formatDate } from '@/lib/utils'

interface ChatInterfaceProps {
  selectedAgent?: Agent
  onSendMessage?: (message: string, agentId?: string) => void
  messages?: ChatMessage[]
  isLoading?: boolean
}

export function ChatInterface({ 
  selectedAgent, 
  onSendMessage, 
  messages = [], 
  isLoading = false 
}: ChatInterfaceProps) {
  const [inputMessage, setInputMessage] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    if (messagesEndRef.current?.scrollIntoView) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

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

  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-scrollintel-primary" />
            ScrollIntel Chat
          </CardTitle>
          {selectedAgent && (
            <Badge variant="outline" className="flex items-center gap-1">
              <div className={cn(
                "w-2 h-2 rounded-full",
                selectedAgent.status === 'active' && "bg-scrollintel-success",
                selectedAgent.status === 'busy' && "bg-scrollintel-warning",
                selectedAgent.status === 'inactive' && "bg-gray-400"
              )} />
              {selectedAgent.name}
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col min-h-0">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4 min-h-0">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <div className="text-center">
                <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Start a conversation with ScrollIntel agents</p>
                <p className="text-sm">Ask questions, upload files, or request analysis</p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-3",
                  message.role === 'user' ? "justify-end" : "justify-start"
                )}
              >
                <div className={cn(
                  "flex gap-3 max-w-[80%]",
                  message.role === 'user' ? "flex-row-reverse" : "flex-row"
                )}>
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                    message.role === 'user' 
                      ? "bg-scrollintel-primary text-white" 
                      : "bg-scrollintel-secondary text-white"
                  )}>
                    {message.role === 'user' ? (
                      <User className="h-4 w-4" />
                    ) : (
                      <Bot className="h-4 w-4" />
                    )}
                  </div>
                  
                  <div className={cn(
                    "rounded-lg px-4 py-2 space-y-1",
                    message.role === 'user'
                      ? "bg-scrollintel-primary text-white"
                      : "bg-muted"
                  )}>
                    <div className="text-sm whitespace-pre-wrap">
                      {message.content}
                    </div>
                    
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
                  </div>
                </div>
              </div>
            ))
          )}
          
          {isLoading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-scrollintel-secondary text-white flex items-center justify-center">
                <Bot className="h-4 w-4" />
              </div>
              <div className="bg-muted rounded-lg px-4 py-2">
                <div className="flex items-center gap-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-scrollintel-primary rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-scrollintel-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-scrollintel-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                  <span className="text-sm text-muted-foreground">
                    {selectedAgent?.name || 'Agent'} is thinking...
                  </span>
                </div>
              </div>
            </div>
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
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            variant="scrollintel"
            size="icon"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}