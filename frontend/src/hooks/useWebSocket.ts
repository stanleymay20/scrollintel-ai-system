'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { io, Socket } from 'socket.io-client'

interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
}

interface StreamingMessage {
  type: 'message_chunk' | 'message_complete' | 'message_error'
  message_id: string
  conversation_id: string
  content?: string
  error?: string
  metadata?: any
}

interface ConnectionStatus {
  connected: boolean
  reconnecting: boolean
  error: string | null
  lastConnected: Date | null
  reconnectAttempts: number
}

export function useWebSocket(url?: string) {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    connected: false,
    reconnecting: false,
    error: null,
    lastConnected: null,
    reconnectAttempts: 0
  })
  
  const socketRef = useRef<Socket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const messageHandlersRef = useRef<Map<string, Set<(data: any) => void>>>(new Map())

  // Get WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    if (url) return url
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = process.env.NODE_ENV === 'development' 
      ? 'localhost:8000' 
      : window.location.host
    
    return `${protocol}//${host}/ws`
  }, [url])

  // Initialize WebSocket connection
  const connect = useCallback(() => {
    if (socketRef.current?.connected) return

    try {
      const wsUrl = getWebSocketUrl()
      console.log('Connecting to WebSocket:', wsUrl)
      
      const newSocket = io(wsUrl, {
        transports: ['websocket', 'polling'],
        timeout: 20000,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        maxReconnectionAttempts: 5,
        forceNew: true
      })

      // Connection event handlers
      newSocket.on('connect', () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setConnectionStatus(prev => ({
          ...prev,
          connected: true,
          reconnecting: false,
          error: null,
          lastConnected: new Date(),
          reconnectAttempts: 0
        }))
        
        // Clear any reconnection timeout
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
          reconnectTimeoutRef.current = null
        }
      })

      newSocket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason)
        setIsConnected(false)
        setConnectionStatus(prev => ({
          ...prev,
          connected: false,
          error: `Disconnected: ${reason}`
        }))
      })

      newSocket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error)
        setConnectionStatus(prev => ({
          ...prev,
          connected: false,
          reconnecting: true,
          error: error.message,
          reconnectAttempts: prev.reconnectAttempts + 1
        }))
      })

      newSocket.on('reconnect', (attemptNumber) => {
        console.log('WebSocket reconnected after', attemptNumber, 'attempts')
        setConnectionStatus(prev => ({
          ...prev,
          reconnecting: false,
          reconnectAttempts: attemptNumber
        }))
      })

      newSocket.on('reconnect_attempt', (attemptNumber) => {
        console.log('WebSocket reconnection attempt:', attemptNumber)
        setConnectionStatus(prev => ({
          ...prev,
          reconnecting: true,
          reconnectAttempts: attemptNumber
        }))
      })

      newSocket.on('reconnect_error', (error) => {
        console.error('WebSocket reconnection error:', error)
        setConnectionStatus(prev => ({
          ...prev,
          error: `Reconnection failed: ${error.message}`
        }))
      })

      newSocket.on('reconnect_failed', () => {
        console.error('WebSocket reconnection failed')
        setConnectionStatus(prev => ({
          ...prev,
          reconnecting: false,
          error: 'Reconnection failed after maximum attempts'
        }))
      })

      // Message streaming handlers
      newSocket.on('message_stream', (data: StreamingMessage) => {
        console.log('Received streaming message:', data)
        
        // Emit to registered handlers
        const handlers = messageHandlersRef.current.get('message_stream')
        if (handlers) {
          handlers.forEach(handler => handler(data))
        }
      })

      // Generic message handler
      newSocket.onAny((eventName, data) => {
        const handlers = messageHandlersRef.current.get(eventName)
        if (handlers) {
          handlers.forEach(handler => handler(data))
        }
      })

      socketRef.current = newSocket
      setSocket(newSocket)

    } catch (error) {
      console.error('Error creating WebSocket connection:', error)
      setConnectionStatus(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Connection failed'
      }))
    }
  }, [getWebSocketUrl])

  // Disconnect WebSocket
  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect()
      socketRef.current = null
      setSocket(null)
      setIsConnected(false)
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }, [])

  // Send message through WebSocket
  const sendMessage = useCallback((event: string, data: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data)
      return true
    } else {
      console.warn('WebSocket not connected, cannot send message')
      return false
    }
  }, [])

  // Subscribe to WebSocket events
  const subscribe = useCallback((event: string, handler: (data: any) => void) => {
    if (!messageHandlersRef.current.has(event)) {
      messageHandlersRef.current.set(event, new Set())
    }
    
    messageHandlersRef.current.get(event)!.add(handler)
    
    // Return unsubscribe function
    return () => {
      const handlers = messageHandlersRef.current.get(event)
      if (handlers) {
        handlers.delete(handler)
        if (handlers.size === 0) {
          messageHandlersRef.current.delete(event)
        }
      }
    }
  }, [])

  // Join conversation room for real-time updates
  const joinConversation = useCallback((conversationId: string) => {
    return sendMessage('join_conversation', { conversation_id: conversationId })
  }, [sendMessage])

  // Leave conversation room
  const leaveConversation = useCallback((conversationId: string) => {
    return sendMessage('leave_conversation', { conversation_id: conversationId })
  }, [sendMessage])

  // Send typing indicator
  const sendTyping = useCallback((conversationId: string, isTyping: boolean) => {
    return sendMessage('typing', { 
      conversation_id: conversationId, 
      is_typing: isTyping 
    })
  }, [sendMessage])

  // Initialize connection on mount
  useEffect(() => {
    connect()
    
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  // Auto-reconnect logic
  useEffect(() => {
    if (!isConnected && !connectionStatus.reconnecting && connectionStatus.reconnectAttempts < 5) {
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log('Attempting to reconnect WebSocket...')
        connect()
      }, Math.min(1000 * Math.pow(2, connectionStatus.reconnectAttempts), 30000))
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
    }
  }, [isConnected, connectionStatus.reconnecting, connectionStatus.reconnectAttempts, connect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    socket,
    isConnected,
    connectionStatus,
    connect,
    disconnect,
    sendMessage,
    subscribe,
    joinConversation,
    leaveConversation,
    sendTyping
  }
}

// Hook for managing conversation-specific WebSocket events
export function useConversationWebSocket(conversationId: string | null) {
  const { socket, isConnected, subscribe, joinConversation, leaveConversation, sendTyping } = useWebSocket()
  const [typingUsers, setTypingUsers] = useState<string[]>([])
  const [streamingMessages, setStreamingMessages] = useState<Map<string, string>>(new Map())

  // Join/leave conversation room when conversation changes
  useEffect(() => {
    if (!conversationId || !isConnected) return

    joinConversation(conversationId)
    
    return () => {
      leaveConversation(conversationId)
    }
  }, [conversationId, isConnected, joinConversation, leaveConversation])

  // Handle typing indicators
  useEffect(() => {
    if (!conversationId) return

    const unsubscribeTyping = subscribe('user_typing', (data: { user_id: string; is_typing: boolean }) => {
      setTypingUsers(prev => {
        if (data.is_typing) {
          return prev.includes(data.user_id) ? prev : [...prev, data.user_id]
        } else {
          return prev.filter(id => id !== data.user_id)
        }
      })
    })

    return unsubscribeTyping
  }, [conversationId, subscribe])

  // Handle streaming messages
  useEffect(() => {
    const unsubscribeStream = subscribe('message_stream', (data: StreamingMessage) => {
      if (data.conversation_id !== conversationId) return

      setStreamingMessages(prev => {
        const newMap = new Map(prev)
        
        if (data.type === 'message_chunk' && data.content) {
          const currentContent = newMap.get(data.message_id) || ''
          newMap.set(data.message_id, currentContent + data.content)
        } else if (data.type === 'message_complete') {
          newMap.delete(data.message_id)
        } else if (data.type === 'message_error') {
          newMap.delete(data.message_id)
        }
        
        return newMap
      })
    })

    return unsubscribeStream
  }, [conversationId, subscribe])

  // Send typing indicator with debouncing
  const handleTyping = useCallback((isTyping: boolean) => {
    if (conversationId && isConnected) {
      sendTyping(conversationId, isTyping)
    }
  }, [conversationId, isConnected, sendTyping])

  return {
    socket,
    isConnected,
    typingUsers,
    streamingMessages,
    handleTyping,
    subscribe
  }
}