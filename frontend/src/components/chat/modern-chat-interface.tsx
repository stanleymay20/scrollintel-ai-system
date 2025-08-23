'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Paperclip, Mic, Square, MoreVertical, Copy, ThumbsUp, ThumbsDown, RotateCcw, Edit3, Trash2, Download, Share2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Progress } from '@/components/ui/progress'
import { MessageProcessor } from './message-processor'
import { ConversationSidebar } from './conversation-sidebar'
import { FileUploadZone } from './file-upload-zone'
import { VoiceInput } from './voice-input'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useConversation } from '@/hooks/useConversation'
import { cn } from '@/lib/utils'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  content_type: 'text' | 'markdown' | 'code' | 'mixed'
  created_at: string
  updated_at?: string
  parent_message_id?: string
  regeneration_count: number
  metadata: {
    model_used?: string
    token_count?: number
    execution_time?: number
    confidence_score?: number
    citations?: Array<{
      text: string
      url: string
      title: string
    }>
  }
  attachments: Array<{
    id: string
    file_name: string
    file_type: string
    file_size: number
    file_url: string
  }>
  reactions: Array<{
    type: string
    user_id: string
  }>
  isStreaming?: boolean
  streamingContent?: string
}

interface Conversation {
  id: string
  title: string
  agent_id?: string
  created_at: string
  updated_at: string
  message_count: number
  last_message_preview?: string
  tags: string[]
  is_archived: boolean
}

export function ModernChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null)
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [attachments, setAttachments] = useState<File[]>([])
  const [showSidebar, setShowSidebar] = useState(true)
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [editingContent, setEditingContent] = useState('')
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  const { socket, isConnected } = useWebSocket()
  const { 
    createConversation, 
    sendMessage, 
    regenerateMessage, 
    editMessage,
    deleteMessage,
    exportConversation,
    shareConversation,
    addReaction
  } = useConversation()

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [inputValue])

  // WebSocket message handling
  useEffect(() => {
    if (!socket) return

    const handleStreamingMessage = (data: any) => {
      if (data.type === 'message_chunk') {
        setMessages(prev => prev.map(msg => 
          msg.id === data.message_id 
            ? { ...msg, streamingContent: (msg.streamingContent || '') + data.content }
            : msg
        ))
      } else if (data.type === 'message_complete') {
        setMessages(prev => prev.map(msg => 
          msg.id === data.message_id 
            ? { ...msg, content: msg.streamingContent || msg.content, isStreaming: false, streamingContent: undefined }
            : msg
        ))
      }
    }

    socket.on('message_stream', handleStreamingMessage)
    
    return () => {
      socket.off('message_stream', handleStreamingMessage)
    }
  }, [socket])

  const handleSendMessage = async () => {
    if (!inputValue.trim() && attachments.length === 0) return
    if (isLoading) return

    const messageContent = inputValue.trim()
    setInputValue('')
    setAttachments([])
    setIsLoading(true)

    try {
      // Create conversation if none exists
      let conversationId = currentConversation?.id
      if (!conversationId) {
        const newConversation = await createConversation({
          title: messageContent.slice(0, 50) + (messageContent.length > 50 ? '...' : ''),
          initial_message: messageContent
        })
        conversationId = newConversation.id
        setCurrentConversation(newConversation)
      }

      // Add user message immediately
      const userMessage: Message = {
        id: `temp-${Date.now()}`,
        role: 'user',
        content: messageContent,
        content_type: 'text',
        created_at: new Date().toISOString(),
        regeneration_count: 0,
        metadata: {},
        attachments: attachments.map(file => ({
          id: `temp-${file.name}`,
          file_name: file.name,
          file_type: file.type,
          file_size: file.size,
          file_url: URL.createObjectURL(file)
        })),
        reactions: []
      }

      setMessages(prev => [...prev, userMessage])

      // Add streaming assistant message placeholder
      const assistantMessage: Message = {
        id: `temp-assistant-${Date.now()}`,
        role: 'assistant',
        content: '',
        content_type: 'markdown',
        created_at: new Date().toISOString(),
        regeneration_count: 0,
        metadata: {},
        attachments: [],
        reactions: [],
        isStreaming: true,
        streamingContent: ''
      }

      setMessages(prev => [...prev, assistantMessage])

      // Send message to backend
      await sendMessage(conversationId, {
        content: messageContent,
        content_type: 'text',
        attachments: attachments.map(file => ({
          file_name: file.name,
          file_type: file.type,
          file_size: file.size,
          file_data: file // This would be processed by the backend
        }))
      })

    } catch (error) {
      console.error('Error sending message:', error)
      // Remove the temporary messages on error
      setMessages(prev => prev.filter(msg => !msg.id.startsWith('temp-')))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleFileUpload = (files: File[]) => {
    setAttachments(prev => [...prev, ...files])
  }

  const handleRemoveAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index))
  }

  const handleVoiceInput = (transcript: string) => {
    setInputValue(prev => prev + (prev ? ' ' : '') + transcript)
  }

  const handleEditMessage = async (messageId: string, newContent: string) => {
    try {
      await editMessage(currentConversation!.id, messageId, newContent)
      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? { ...msg, content: newContent, updated_at: new Date().toISOString() }
          : msg
      ))
      setEditingMessageId(null)
    } catch (error) {
      console.error('Error editing message:', error)
    }
  }

  const handleRegenerateMessage = async (messageId: string) => {
    try {
      const newMessage = await regenerateMessage(currentConversation!.id, messageId)
      setMessages(prev => [...prev, newMessage])
    } catch (error) {
      console.error('Error regenerating message:', error)
    }
  }

  const handleReaction = async (messageId: string, reactionType: string) => {
    try {
      await addReaction(currentConversation!.id, messageId, reactionType)
      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? { 
              ...msg, 
              reactions: [...msg.reactions.filter(r => r.type !== reactionType), { type: reactionType, user_id: 'current_user' }]
            }
          : msg
      ))
    } catch (error) {
      console.error('Error adding reaction:', error)
    }
  }

  const handleExportConversation = async (format: 'markdown' | 'pdf' | 'json' | 'txt') => {
    if (!currentConversation) return
    
    try {
      const exportData = await exportConversation(currentConversation.id, format)
      
      // Create download link
      const blob = new Blob([exportData.content], { type: exportData.mime_type })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = exportData.filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error exporting conversation:', error)
    }
  }

  const handleShareConversation = async () => {
    if (!currentConversation) return
    
    try {
      const shareData = await shareConversation(currentConversation.id, {
        permissions: 'read',
        expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() // 7 days
      })
      
      // Copy share URL to clipboard
      await navigator.clipboard.writeText(window.location.origin + shareData.share_url)
      // Show success toast
    } catch (error) {
      console.error('Error sharing conversation:', error)
    }
  }

  return (
    <TooltipProvider>
      <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
        {/* Conversation Sidebar */}
        {showSidebar && (
          <ConversationSidebar
            conversations={conversations}
            currentConversation={currentConversation}
            onSelectConversation={setCurrentConversation}
            onNewConversation={() => setCurrentConversation(null)}
            onToggleSidebar={() => setShowSidebar(false)}
          />
        )}

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="border-b bg-white dark:bg-gray-800 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {!showSidebar && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowSidebar(true)}
                >
                  ☰
                </Button>
              )}
              <div>
                <h1 className="font-semibold text-lg">
                  {currentConversation?.title || 'New Conversation'}
                </h1>
                {currentConversation?.agent_id && (
                  <Badge variant="secondary" className="text-xs">
                    {currentConversation.agent_id}
                  </Badge>
                )}
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleShareConversation}
                    disabled={!currentConversation}
                  >
                    <Share2 className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Share conversation</TooltipContent>
              </Tooltip>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => handleExportConversation('markdown')}>
                    <Download className="h-4 w-4 mr-2" />
                    Export as Markdown
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleExportConversation('pdf')}>
                    <Download className="h-4 w-4 mr-2" />
                    Export as PDF
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleExportConversation('json')}>
                    <Download className="h-4 w-4 mr-2" />
                    Export as JSON
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className="text-red-600">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete conversation
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          {/* Messages */}
          <ScrollArea className="flex-1 p-4">
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.map((message, index) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-4",
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  )}
                >
                  {message.role !== 'user' && (
                    <Avatar className="h-8 w-8 mt-1">
                      <AvatarImage src="/api/placeholder/32/32" />
                      <AvatarFallback>AI</AvatarFallback>
                    </Avatar>
                  )}

                  <div className={cn(
                    "max-w-[80%] space-y-2",
                    message.role === 'user' ? 'items-end' : 'items-start'
                  )}>
                    <Card className={cn(
                      "p-4",
                      message.role === 'user' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-white dark:bg-gray-800'
                    )}>
                      {editingMessageId === message.id ? (
                        <div className="space-y-2">
                          <Textarea
                            value={editingContent}
                            onChange={(e) => setEditingContent(e.target.value)}
                            className="min-h-[60px]"
                          />
                          <div className="flex justify-end space-x-2">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setEditingMessageId(null)}
                            >
                              Cancel
                            </Button>
                            <Button
                              size="sm"
                              onClick={() => handleEditMessage(message.id, editingContent)}
                            >
                              Save
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <MessageProcessor
                          content={message.isStreaming ? message.streamingContent || '' : message.content}
                          contentType={message.content_type}
                          isStreaming={message.isStreaming}
                        />
                      )}

                      {/* Attachments */}
                      {message.attachments.length > 0 && (
                        <div className="mt-3 space-y-2">
                          {message.attachments.map((attachment) => (
                            <div
                              key={attachment.id}
                              className="flex items-center space-x-2 p-2 bg-gray-100 dark:bg-gray-700 rounded"
                            >
                              <Paperclip className="h-4 w-4" />
                              <span className="text-sm">{attachment.file_name}</span>
                              <span className="text-xs text-gray-500">
                                ({(attachment.file_size / 1024).toFixed(1)} KB)
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Citations */}
                      {message.metadata.citations && message.metadata.citations.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Sources:</div>
                          <div className="space-y-1">
                            {message.metadata.citations.map((citation, idx) => (
                              <a
                                key={idx}
                                href={citation.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400"
                              >
                                {citation.title}
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </Card>

                    {/* Message Actions */}
                    <div className="flex items-center space-x-1 text-xs text-gray-500">
                      <span>{new Date(message.created_at).toLocaleTimeString()}</span>
                      
                      {message.role === 'assistant' && (
                        <>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2"
                            onClick={() => handleReaction(message.id, 'thumbs_up')}
                          >
                            <ThumbsUp className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2"
                            onClick={() => handleReaction(message.id, 'thumbs_down')}
                          >
                            <ThumbsDown className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2"
                            onClick={() => navigator.clipboard.writeText(message.content)}
                          >
                            <Copy className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2"
                            onClick={() => handleRegenerateMessage(message.id)}
                          >
                            <RotateCcw className="h-3 w-3" />
                          </Button>
                        </>
                      )}

                      {message.role === 'user' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2"
                          onClick={() => {
                            setEditingMessageId(message.id)
                            setEditingContent(message.content)
                          }}
                        >
                          <Edit3 className="h-3 w-3" />
                        </Button>
                      )}

                      {/* Metadata */}
                      {message.metadata.model_used && (
                        <Badge variant="outline" className="text-xs">
                          {message.metadata.model_used}
                        </Badge>
                      )}
                      
                      {message.metadata.execution_time && (
                        <span className="text-xs">
                          {message.metadata.execution_time.toFixed(2)}s
                        </span>
                      )}
                    </div>
                  </div>

                  {message.role === 'user' && (
                    <Avatar className="h-8 w-8 mt-1">
                      <AvatarImage src="/api/placeholder/32/32" />
                      <AvatarFallback>You</AvatarFallback>
                    </Avatar>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-center">
                  <div className="flex items-center space-x-2 text-gray-500">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-900"></div>
                    <span>AI is thinking...</span>
                  </div>
                </div>
              )}
            </div>
            <div ref={messagesEndRef} />
          </ScrollArea>

          {/* Input Area */}
          <div className="border-t bg-white dark:bg-gray-800 p-4">
            <div className="max-w-4xl mx-auto">
              {/* Attachments Preview */}
              {attachments.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2">
                  {attachments.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full text-sm"
                    >
                      <Paperclip className="h-3 w-3" />
                      <span>{file.name}</span>
                      <button
                        onClick={() => handleRemoveAttachment(index)}
                        className="text-gray-500 hover:text-red-500"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="flex items-end space-x-2">
                <div className="flex-1 relative">
                  <Textarea
                    ref={textareaRef}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your message..."
                    className="min-h-[44px] max-h-32 resize-none pr-12"
                    disabled={isLoading}
                  />
                  
                  <div className="absolute right-2 bottom-2 flex space-x-1">
                    <FileUploadZone onFilesSelected={handleFileUpload}>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0"
                        disabled={isLoading}
                      >
                        <Paperclip className="h-4 w-4" />
                      </Button>
                    </FileUploadZone>

                    <VoiceInput
                      onTranscript={handleVoiceInput}
                      isRecording={isRecording}
                      onRecordingChange={setIsRecording}
                    >
                      <Button
                        variant="ghost"
                        size="sm"
                        className={cn(
                          "h-8 w-8 p-0",
                          isRecording && "text-red-500"
                        )}
                        disabled={isLoading}
                      >
                        {isRecording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                      </Button>
                    </VoiceInput>
                  </div>
                </div>

                <Button
                  onClick={handleSendMessage}
                  disabled={(!inputValue.trim() && attachments.length === 0) || isLoading}
                  className="h-11"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>

              {/* Connection Status */}
              <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                <div className="flex items-center space-x-2">
                  <div className={cn(
                    "h-2 w-2 rounded-full",
                    isConnected ? "bg-green-500" : "bg-red-500"
                  )} />
                  <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
                </div>
                
                <div className="text-right">
                  <div>Press Enter to send, Shift+Enter for new line</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}