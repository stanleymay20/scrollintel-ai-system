'use client'

import { useState, useCallback } from 'react'
import { api } from '@/lib/api'

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

interface CreateConversationRequest {
  title?: string
  agent_id?: string
  initial_message?: string
}

interface SendMessageRequest {
  content: string
  content_type?: 'text' | 'markdown' | 'code' | 'mixed'
  parent_message_id?: string
  attachments?: Array<{
    file_name: string
    file_type: string
    file_size: number
    file_data: File
  }>
  metadata?: Record<string, any>
}

interface ShareConversationRequest {
  permissions: 'read' | 'comment' | 'edit'
  expires_at?: string
}

export function useConversation() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const createConversation = useCallback(async (request: CreateConversationRequest): Promise<Conversation> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.post('/conversations', request)
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to create conversation'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const getConversations = useCallback(async (params?: {
    limit?: number
    offset?: number
    search?: string
    agent_id?: string
    tags?: string[]
  }): Promise<Conversation[]> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.get('/conversations', { params })
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to fetch conversations'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const getConversation = useCallback(async (conversationId: string): Promise<Conversation> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.get(`/conversations/${conversationId}`)
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to fetch conversation'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const getMessages = useCallback(async (
    conversationId: string,
    params?: {
      limit?: number
      before_message_id?: string
    }
  ): Promise<Message[]> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.get(`/conversations/${conversationId}/messages`, { params })
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to fetch messages'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const sendMessage = useCallback(async (
    conversationId: string,
    request: SendMessageRequest
  ): Promise<Message> => {
    setLoading(true)
    setError(null)
    
    try {
      // Handle file uploads if present
      let formData: FormData | undefined
      if (request.attachments && request.attachments.length > 0) {
        formData = new FormData()
        formData.append('content', request.content)
        formData.append('content_type', request.content_type || 'text')
        
        if (request.parent_message_id) {
          formData.append('parent_message_id', request.parent_message_id)
        }
        
        if (request.metadata) {
          formData.append('metadata', JSON.stringify(request.metadata))
        }
        
        request.attachments.forEach((attachment, index) => {
          formData!.append(`attachments[${index}]`, attachment.file_data)
          formData!.append(`attachment_metadata[${index}]`, JSON.stringify({
            file_name: attachment.file_name,
            file_type: attachment.file_type,
            file_size: attachment.file_size
          }))
        })
      }

      const response = await api.post(
        `/conversations/${conversationId}/messages`,
        formData || request,
        formData ? {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        } : undefined
      )
      
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to send message'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const editMessage = useCallback(async (
    conversationId: string,
    messageId: string,
    newContent: string
  ): Promise<Message> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.put(`/conversations/${conversationId}/messages/${messageId}`, {
        content: newContent
      })
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to edit message'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteMessage = useCallback(async (
    conversationId: string,
    messageId: string
  ): Promise<void> => {
    setLoading(true)
    setError(null)
    
    try {
      await api.delete(`/conversations/${conversationId}/messages/${messageId}`)
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to delete message'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const regenerateMessage = useCallback(async (
    conversationId: string,
    messageId: string
  ): Promise<Message> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.post(`/conversations/${conversationId}/messages/${messageId}/regenerate`)
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to regenerate message'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const addReaction = useCallback(async (
    conversationId: string,
    messageId: string,
    reactionType: string
  ): Promise<void> => {
    setLoading(true)
    setError(null)
    
    try {
      await api.post(`/conversations/${conversationId}/messages/${messageId}/reactions`, {
        reaction_type: reactionType
      })
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to add reaction'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const removeReaction = useCallback(async (
    conversationId: string,
    messageId: string,
    reactionType: string
  ): Promise<void> => {
    setLoading(true)
    setError(null)
    
    try {
      await api.delete(`/conversations/${conversationId}/messages/${messageId}/reactions/${reactionType}`)
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to remove reaction'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const updateConversation = useCallback(async (
    conversationId: string,
    updates: Partial<Pick<Conversation, 'title' | 'tags' | 'is_archived'>>
  ): Promise<Conversation> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.patch(`/conversations/${conversationId}`, updates)
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to update conversation'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteConversation = useCallback(async (conversationId: string): Promise<void> => {
    setLoading(true)
    setError(null)
    
    try {
      await api.delete(`/conversations/${conversationId}`)
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to delete conversation'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const exportConversation = useCallback(async (
    conversationId: string,
    format: 'markdown' | 'pdf' | 'json' | 'txt' = 'markdown'
  ): Promise<{
    format: string
    content: string
    filename: string
    mime_type: string
    size: number
  }> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.post(`/conversations/${conversationId}/export`, {
        format
      })
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to export conversation'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const shareConversation = useCallback(async (
    conversationId: string,
    request: ShareConversationRequest
  ): Promise<{
    share_url: string
    status: string
  }> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.post(`/conversations/${conversationId}/share`, request)
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to share conversation'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const searchConversations = useCallback(async (
    query: string,
    params?: {
      limit?: number
      offset?: number
      agent_id?: string
      tags?: string[]
    }
  ): Promise<Conversation[]> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.get('/conversations/search', {
        params: { query, ...params }
      })
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to search conversations'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  const getConversationAnalytics = useCallback(async (
    conversationId: string
  ): Promise<{
    message_count: number
    total_tokens: number
    average_response_time: number
    models_used: string[]
    created_at: string
    last_activity: string
  }> => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.get(`/conversations/${conversationId}/analytics`)
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to fetch analytics'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    loading,
    error,
    createConversation,
    getConversations,
    getConversation,
    getMessages,
    sendMessage,
    editMessage,
    deleteMessage,
    regenerateMessage,
    addReaction,
    removeReaction,
    updateConversation,
    deleteConversation,
    exportConversation,
    shareConversation,
    searchConversations,
    getConversationAnalytics
  }
}