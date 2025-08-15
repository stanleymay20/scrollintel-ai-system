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

// API functions
export const agentApi = {
  getAgents: () => api.get<Agent[]>('/api/agents'),
  getAgent: (id: string) => api.get<Agent>(`/api/agents/${id}`),
  sendMessage: (agentId: string, message: string) => 
    api.post(`/api/agents/${agentId}/chat`, { message }),
}

export const fileApi = {
  uploadFile: (file: File, onProgress?: (progress: number) => void) => {
    const formData = new FormData()
    formData.append('file', file)
    
    return api.post('/api/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })
  },
  getFiles: () => api.get('/api/files'),
  deleteFile: (id: string) => api.delete(`/api/files/${id}`),
}

export const dashboardApi = {
  getDashboards: () => api.get('/api/dashboards'),
  createDashboard: (config: any) => api.post('/api/dashboards', config),
  updateDashboard: (id: string, config: any) => api.put(`/api/dashboards/${id}`, config),
  deleteDashboard: (id: string) => api.delete(`/api/dashboards/${id}`),
}