import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('scrollintel_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('scrollintel_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export interface Agent {
  id: string
  name: string
  type: string
  status: 'active' | 'inactive' | 'busy'
  capabilities: string[]
  description: string
  last_active: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  agent_id?: string
}

export interface FileUpload {
  id: string
  filename: string
  size: number
  type: string
  status: 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  preview?: any
}

// Enhanced API functions with all ScrollIntel endpoints
export const scrollIntelApi = {
  // Health check
  health: () => api.get('/health'),

  // System metrics and monitoring
  getSystemMetrics: () => api.get('/api/monitoring/metrics'),
  getMonitoringData: () => api.get('/api/monitoring'),
  getAlerts: () => api.get('/api/monitoring/alerts'),

  // Agents
  getAgents: () => api.get<Agent[]>('/api/agents'),
  getAgent: (id: string) => api.get<Agent>(`/api/agents/${id}`),
  
  // Chat/Messaging - Updated to match backend
  sendMessage: (data: {
    message: string
    agent_id?: string
    conversation_id?: string
  }) => api.post('/api/agents/chat', data),

  // File upload with progress tracking
  uploadFile: (
    formData: FormData, 
    onUploadProgress?: (progressEvent: any) => void
  ) => api.post('/api/files/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress,
  }),

  // File management
  getFiles: () => api.get('/api/files'),
  deleteFile: (id: string) => api.delete(`/api/files/${id}`),

  // Dashboard data
  getDashboardData: () => api.get('/api/dashboard'),
  getDashboards: () => api.get('/api/dashboards'),
  createDashboard: (config: any) => api.post('/api/dashboards', config),
  updateDashboard: (id: string, config: any) => api.put(`/api/dashboards/${id}`, config),
  deleteDashboard: (id: string) => api.delete(`/api/dashboards/${id}`),

  // Analytics
  getAnalytics: (params?: any) => api.get('/api/analytics', { params }),

  // User management
  getCurrentUser: () => api.get('/api/auth/me'),
  updateUser: (data: any) => api.put('/api/auth/me', data),

  // Workspaces
  getWorkspaces: () => api.get('/api/workspaces'),
  createWorkspace: (data: any) => api.post('/api/workspaces', data),

  // API Keys
  getApiKeys: () => api.get('/api/api-keys'),
  createApiKey: (data: any) => api.post('/api/api-keys', data),
  deleteApiKey: (id: string) => api.delete(`/api/api-keys/${id}`),

  // Billing
  getBillingInfo: () => api.get('/api/billing'),
  getUsage: () => api.get('/api/billing/usage'),

  // Visual Generation
  generateImage: (data: any) => api.post('/api/visual-generation/image', data),
  generateVideo: (data: any) => api.post('/api/visual-generation/video', data),

  // Model Factory
  getModels: () => api.get('/api/model-factory/models'),
  deployModel: (data: any) => api.post('/api/model-factory/deploy', data),

  // Prompt Management
  getPrompts: () => api.get('/api/prompts'),
  createPrompt: (data: any) => api.post('/api/prompts', data),
  updatePrompt: (id: string, data: any) => api.put(`/api/prompts/${id}`, data),
  deletePrompt: (id: string) => api.delete(`/api/prompts/${id}`),

  // Orchestration
  getWorkflows: () => api.get('/api/orchestration/workflows'),
  executeWorkflow: (id: string, data?: any) => api.post(`/api/orchestration/workflows/${id}/execute`, data),

  // Ethics Engine
  analyzeEthics: (data: any) => api.post('/api/ethics/analyze', data),
  getEthicsReport: (id: string) => api.get(`/api/ethics/reports/${id}`),

  // CTO Agent specific
  getCTOAnalysis: (data: any) => api.post('/api/agents/cto/analyze', data),
  
  // Data Scientist Agent
  getDataAnalysis: (data: any) => api.post('/api/agents/data-scientist/analyze', data),
  
  // ML Engineer Agent
  getMLAnalysis: (data: any) => api.post('/api/agents/ml-engineer/analyze', data),
  
  // AI Engineer Agent
  getAIAnalysis: (data: any) => api.post('/api/agents/ai-engineer/analyze', data),

  // BI Agent
  getBIAnalysis: (data: any) => api.post('/api/agents/bi/analyze', data),

  // QA Agent
  getQAAnalysis: (data: any) => api.post('/api/agents/qa/analyze', data),

  // Forecast Engine
  getForecast: (data: any) => api.post('/api/forecast/predict', data),

  // Visualization
  createVisualization: (data: any) => api.post('/api/visualization/create', data),
  getVisualizations: () => api.get('/api/visualization'),
}

// Legacy API functions for backward compatibility
export const agentApi = scrollIntelApi
export const fileApi = scrollIntelApi
export const dashboardApi = scrollIntelApi