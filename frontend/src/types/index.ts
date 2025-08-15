// Core types for ScrollIntel frontend

export interface Agent {
  id: string
  name: string
  description?: string
  status: 'active' | 'busy' | 'inactive' | 'thinking'
  capabilities: string[]
  personality?: {
    traits: string[]
    communication_style: string
    expertise_confidence: number
  }
  avatar_config?: {
    style: string
    color_scheme: string
    accessories: string[]
    background: string
  }
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  agent_id?: string
  metadata?: {
    processing_time?: number
    template_used?: string
    personality_applied?: boolean
    streaming_chunks?: number
    user_feedback?: 'positive' | 'negative'
  }
}

export interface ConversationSummary {
  conversation_id: string
  agent: {
    id: string
    name: string
    avatar_config: any
  }
  start_time: string
  end_time?: string
  total_turns: number
  main_topics: string[]
  user_satisfaction?: number
  key_outcomes: string[]
  follow_up_needed: boolean
}

export interface StreamingEvent {
  event_type: 'typing_start' | 'typing_stop' | 'response_chunk' | 'response_complete' | 'error' | 'agent_status' | 'thinking'
  agent_id: string
  conversation_id: string
  data: any
  timestamp: string
}

export interface UserProfile {
  user_id: string
  preferred_communication_style: string
  expertise_level: 'beginner' | 'intermediate' | 'expert'
  preferred_agents: string[]
  topics_of_interest: string[]
  total_interactions: number
}

export interface SystemStats {
  agents: {
    total: number
    active: number
    personalities_loaded: number
  }
  conversations: {
    active: number
    streaming_enabled: number
  }
  templates: {
    total: number
    agents_with_templates: number
  }
  memory: {
    active_contexts: number
    agent_memories: number
  }
}

// File upload types
export interface FileUpload {
  id: string
  name: string
  filename: string
  size: number
  type: string
  status: 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  url?: string
  metadata?: any
  error_message?: string
  upload_time?: Date
  processing_time?: number
  preview_available?: boolean
  quality_score?: number
}

export interface FileValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
  file_info: {
    name: string
    size: number
    type: string
    extension: string
  }
}

export interface FilePreview {
  columns: Array<{
    name: string
    type: string
    inferred_type: string
  }>
  sample_data: any[]
  total_rows: number
  preview_rows: number
  data_types: Record<string, string>
  statistics?: Record<string, any>
}

export interface FileHistory {
  id: string
  filename: string
  upload_date: Date
  file_size: number
  status: string
  quality_score?: number
  dataset_created?: boolean
}

// Dashboard types
export interface DashboardMetric {
  id: string
  name: string
  value: number | string
  change?: number
  trend?: 'up' | 'down' | 'stable'
  format?: 'number' | 'percentage' | 'currency' | 'duration'
}

export interface ChartData {
  labels: string[]
  datasets: {
    label: string
    data: number[]
    backgroundColor?: string | string[]
    borderColor?: string
    borderWidth?: number
  }[]
}

// API response types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  per_page: number
  pages: number
}

// Utility types
export type LoadingState = 'idle' | 'loading' | 'success' | 'error'

export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: string | null
}