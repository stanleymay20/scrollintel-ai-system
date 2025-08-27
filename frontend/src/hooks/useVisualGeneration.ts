import { useState, useCallback, useRef, useEffect } from 'react'
import { scrollIntelApi } from '@/lib/api'

export interface GenerationJob {
  id: string
  type: 'image' | 'video' | 'enhancement' | 'batch'
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress: number
  prompt: string
  result?: {
    content_urls: string[]
    quality_metrics?: any
    generation_time?: number
    cost?: number
    model_used?: string
  }
  error?: string
  metadata?: any
  created_at: string
}

export interface GenerationProgress {
  result_id: string
  status: string
  progress: number
  estimated_completion?: string
  content_urls?: string[]
  error_message?: string
  metadata?: any
}

export function useVisualGeneration() {
  const [jobs, setJobs] = useState<GenerationJob[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const progressCallbacks = useRef<Map<string, (progress: GenerationProgress) => void>>(new Map())

  // Initialize WebSocket connection for real-time updates
  const initializeWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 
      (typeof window !== 'undefined' 
        ? `ws://${window.location.hostname}:8000/ws/visual-generation`
        : 'ws://localhost:8000/ws/visual-generation')

    try {
      const token = localStorage.getItem('scrollintel_token')
      wsRef.current = new WebSocket(`${wsUrl}?token=${token}`)

      wsRef.current.onopen = () => {
        console.log('Visual generation WebSocket connected')
      }

      wsRef.current.onmessage = (event) => {
        try {
          const data: GenerationProgress = JSON.parse(event.data)
          
          // Update job status
          setJobs(prev => prev.map(job => 
            job.id === data.result_id 
              ? {
                  ...job,
                  status: data.status as any,
                  progress: data.progress,
                  result: data.content_urls ? {
                    content_urls: data.content_urls,
                    ...data.metadata
                  } : job.result,
                  error: data.error_message
                }
              : job
          ))

          // Call progress callback if registered
          const callback = progressCallbacks.current.get(data.result_id)
          if (callback) {
            callback(data)
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err)
        }
      }

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setError('Real-time updates unavailable')
      }

      wsRef.current.onclose = () => {
        console.log('WebSocket connection closed')
        // Attempt to reconnect after 5 seconds
        setTimeout(initializeWebSocket, 5000)
      }
    } catch (err) {
      console.error('Failed to initialize WebSocket:', err)
      setError('Failed to connect to real-time updates')
    }
  }, [])

  // Generate image
  const generateImage = useCallback(async (params: {
    prompt: string
    negative_prompt?: string
    resolution?: [number, number]
    num_images?: number
    style?: string
    quality?: string
    seed?: number
    model_preference?: string
  }) => {
    setIsGenerating(true)
    setError(null)

    try {
      const response = await scrollIntelApi.generateImage(params)
      
      if (response.data.success) {
        const newJob: GenerationJob = {
          id: response.data.result_id,
          type: 'image',
          status: 'queued',
          progress: 0,
          prompt: params.prompt,
          created_at: new Date().toISOString()
        }

        setJobs(prev => [...prev, newJob])
        initializeWebSocket()
        
        return response.data
      } else {
        throw new Error(response.data.error_message || 'Generation failed')
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Generation failed'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setIsGenerating(false)
    }
  }, [initializeWebSocket])

  // Generate video
  const generateVideo = useCallback(async (params: {
    prompt: string
    duration?: number
    resolution?: [number, number]
    fps?: number
    style?: string
    quality?: string
    humanoid_generation?: boolean
    physics_simulation?: boolean
    neural_rendering_quality?: string
    temporal_consistency_level?: string
  }) => {
    setIsGenerating(true)
    setError(null)

    try {
      const response = await scrollIntelApi.generateVideo(params)
      
      if (response.data.success) {
        const newJob: GenerationJob = {
          id: response.data.result_id,
          type: 'video',
          status: 'queued',
          progress: 0,
          prompt: params.prompt,
          created_at: new Date().toISOString()
        }

        setJobs(prev => [...prev, newJob])
        initializeWebSocket()
        
        return response.data
      } else {
        throw new Error(response.data.error_message || 'Generation failed')
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Generation failed'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setIsGenerating(false)
    }
  }, [initializeWebSocket])

  // Enhance image
  const enhanceImage = useCallback(async (file: File, enhancementType: string) => {
    setIsGenerating(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await scrollIntelApi.enhanceImage(formData, enhancementType)
      
      if (response.data.success) {
        const newJob: GenerationJob = {
          id: response.data.result_id,
          type: 'enhancement',
          status: 'queued',
          progress: 0,
          prompt: `Enhance ${file.name} (${enhancementType})`,
          created_at: new Date().toISOString()
        }

        setJobs(prev => [...prev, newJob])
        initializeWebSocket()
        
        return response.data
      } else {
        throw new Error(response.data.error_message || 'Enhancement failed')
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Enhancement failed'
      setError(errorMessage)
      throw new Error(errorMessage)
    } finally {
      setIsGenerating(false)
    }
  }, [initializeWebSocket])

  // Cancel generation
  const cancelGeneration = useCallback(async (resultId: string) => {
    try {
      const response = await scrollIntelApi.cancelGeneration(resultId)
      
      if (response.data.success) {
        setJobs(prev => prev.map(job => 
          job.id === resultId 
            ? { ...job, status: 'cancelled' }
            : job
        ))
      }
      
      return response.data
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Cancellation failed'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [])

  // Get generation status
  const getGenerationStatus = useCallback(async (resultId: string) => {
    try {
      const response = await scrollIntelApi.getGenerationStatus(resultId)
      return response.data
    } catch (err: any) {
      console.error('Failed to get generation status:', err)
      return null
    }
  }, [])

  // Register progress callback
  const onProgress = useCallback((resultId: string, callback: (progress: GenerationProgress) => void) => {
    progressCallbacks.current.set(resultId, callback)
    
    // Return cleanup function
    return () => {
      progressCallbacks.current.delete(resultId)
    }
  }, [])

  // Load user generations
  const loadUserGenerations = useCallback(async (params?: any) => {
    try {
      const response = await scrollIntelApi.getUserGenerations(params)
      
      if (response.data.success) {
        const loadedJobs: GenerationJob[] = response.data.generations.map((gen: any) => ({
          id: gen.result_id,
          type: gen.content_type,
          status: gen.status,
          progress: gen.status === 'completed' ? 100 : 0,
          prompt: gen.prompt,
          result: gen.content_urls ? {
            content_urls: gen.content_urls,
            generation_time: gen.generation_time,
            cost: gen.cost,
            model_used: gen.model_used
          } : undefined,
          created_at: gen.created_at
        }))
        
        setJobs(loadedJobs)
      }
    } catch (err: any) {
      console.error('Failed to load user generations:', err)
      setError('Failed to load generation history')
    }
  }, [])

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return {
    jobs,
    isGenerating,
    error,
    generateImage,
    generateVideo,
    enhanceImage,
    cancelGeneration,
    getGenerationStatus,
    onProgress,
    loadUserGenerations,
    clearError: () => setError(null)
  }
}