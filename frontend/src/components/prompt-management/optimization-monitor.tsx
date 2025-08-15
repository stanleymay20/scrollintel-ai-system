'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  ScatterChart,
  Scatter
} from 'recharts'
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  TrendingUp, 
  TrendingDown, 
  Zap,
  Brain,
  Target,
  Clock,
  CheckCircle,
  AlertTriangle,
  BarChart3,
  Activity
} from 'lucide-react'

interface OptimizationMetrics {
  accuracy: number
  relevance: number
  efficiency: number
  coherence: number
  creativity: number
  overall_score: number
}

interface OptimizationGeneration {
  id: string
  generation_number: number
  prompt_content: string
  metrics: OptimizationMetrics
  created_at: string
  is_best: boolean
}

interface OptimizationJob {
  id: string
  name: string
  prompt_id: string
  original_prompt: string
  algorithm: 'genetic' | 'reinforcement_learning' | 'hybrid'
  status: 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_generation: number
  max_generations: number
  best_score: number
  target_score: number
  generations: OptimizationGeneration[]
  config: {
    population_size: number
    mutation_rate: number
    crossover_rate: number
    selection_method: string
  }
  start_time: string
  end_time?: string
  created_by: string
}

interface OptimizationMonitorProps {
  onStartOptimization: (promptId: string, config: any) => void
  onPauseOptimization: (jobId: string) => void
  onStopOptimization: (jobId: string) => void
  onApplyOptimization: (jobId: string, generationId: string) => void
}

export function OptimizationMonitor({
  onStartOptimization,
  onPauseOptimization,
  onStopOptimization,
  onApplyOptimization
}: OptimizationMonitorProps) {
  const [jobs, setJobs] = useState<OptimizationJob[]>([])
  const [selectedJob, setSelectedJob] = useState<OptimizationJob | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Mock data - replace with actual API calls
  useEffect(() => {
    const mockJobs: OptimizationJob[] = [
      {
        id: '1',
        name: 'Content Generation Optimization',
        prompt_id: 'prompt_1',
        original_prompt: 'Generate content about {topic} for {audience}...',
        algorithm: 'genetic',
        status: 'running',
        progress: 65,
        current_generation: 13,
        max_generations: 20,
        best_score: 0.87,
        target_score: 0.90,
        generations: [
          {
            id: 'gen_1',
            generation_number: 1,
            prompt_content: 'Generate content about {topic} for {audience}...',
            metrics: {
              accuracy: 0.75,
              relevance: 0.80,
              efficiency: 0.70,
              coherence: 0.78,
              creativity: 0.65,
              overall_score: 0.74
            },
            created_at: '2024-01-15T10:00:00Z',
            is_best: false
          },
          {
            id: 'gen_13',
            generation_number: 13,
            prompt_content: 'Create compelling, audience-specific content about {topic} that resonates with {audience} by incorporating relevant examples and actionable insights...',
            metrics: {
              accuracy: 0.89,
              relevance: 0.92,
              efficiency: 0.85,
              coherence: 0.88,
              creativity: 0.82,
              overall_score: 0.87
            },
            created_at: '2024-01-15T12:30:00Z',
            is_best: true
          }
        ],
        config: {
          population_size: 50,
          mutation_rate: 0.1,
          crossover_rate: 0.8,
          selection_method: 'tournament'
        },
        start_time: '2024-01-15T10:00:00Z',
        created_by: 'john.doe'
      },
      {
        id: '2',
        name: 'Code Review Enhancement',
        prompt_id: 'prompt_2',
        original_prompt: 'Review the following code...',
        algorithm: 'reinforcement_learning',
        status: 'completed',
        progress: 100,
        current_generation: 25,
        max_generations: 25,
        best_score: 0.94,
        target_score: 0.90,
        generations: [
          {
            id: 'gen_25',
            generation_number: 25,
            prompt_content: 'Conduct a comprehensive code review focusing on security, performance, maintainability, and best practices. Provide specific, actionable feedback with examples...',
            metrics: {
              accuracy: 0.95,
              relevance: 0.96,
              efficiency: 0.92,
              coherence: 0.94,
              creativity: 0.88,
              overall_score: 0.94
            },
            created_at: '2024-01-10T16:45:00Z',
            is_best: true
          }
        ],
        config: {
          population_size: 30,
          mutation_rate: 0.05,
          crossover_rate: 0.7,
          selection_method: 'roulette'
        },
        start_time: '2024-01-10T09:00:00Z',
        end_time: '2024-01-10T17:00:00Z',
        created_by: 'jane.smith'
      }
    ]
    
    setJobs(mockJobs)
    setSelectedJob(mockJobs[0])
    setIsLoading(false)
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Activity className="w-4 h-4 text-green-500 animate-pulse" />
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-500" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-blue-500" />
      case 'failed':
        return <AlertTriangle className="w-4 h-4 text-red-500" />
      default:
        return <Clock className="w-4 h-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-green-100 text-green-800'
      case 'paused':
        return 'bg-yellow-100 text-yellow-800'
      case 'completed':
        return 'bg-blue-100 text-blue-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getAlgorithmIcon = (algorithm: string) => {
    switch (algorithm) {
      case 'genetic':
        return <Zap className="w-4 h-4 text-purple-500" />
      case 'reinforcement_learning':
        return <Brain className="w-4 h-4 text-blue-500" />
      case 'hybrid':
        return <Target className="w-4 h-4 text-green-500" />
      default:
        return <Settings className="w-4 h-4 text-gray-500" />
    }
  }

  const getProgressData = (job: OptimizationJob) => {
    // Generate mock progress data
    const data = []
    for (let i = 1; i <= job.current_generation; i++) {
      data.push({
        generation: i,
        best_score: Math.min(0.95, 0.6 + (i / job.current_generation) * 0.35 + Math.random() * 0.05),
        avg_score: Math.min(0.85, 0.5 + (i / job.current_generation) * 0.3 + Math.random() * 0.05)
      })
    }
    return data
  }

  const getMetricsData = (job: OptimizationJob) => {
    const bestGeneration = job.generations.find(g => g.is_best)
    if (!bestGeneration) return []

    return Object.entries(bestGeneration.metrics)
      .filter(([key]) => key !== 'overall_score')
      .map(([key, value]) => ({
        metric: key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
        value: value * 100,
        target: 85 // Mock target value
      }))
  }

  return (
    <div className="h-full flex">
      {/* Sidebar - Jobs List */}
      <div className="w-80 border-r flex flex-col">
        <div className="p-4 border-b">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Optimization Jobs</h2>
            <Button onClick={() => onStartOptimization('', {})} size="sm">
              <Play className="w-4 h-4 mr-2" />
              New Job
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-auto">
          {jobs.map(job => (
            <div
              key={job.id}
              className={`p-4 border-b cursor-pointer hover:bg-gray-50 ${
                selectedJob?.id === job.id ? 'bg-blue-50 border-blue-200' : ''
              }`}
              onClick={() => setSelectedJob(job)}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium truncate">{job.name}</h3>
                {getStatusIcon(job.status)}
              </div>
              
              <div className="flex items-center space-x-2 mb-2">
                <Badge className={getStatusColor(job.status)}>
                  {job.status}
                </Badge>
                <div className="flex items-center space-x-1">
                  {getAlgorithmIcon(job.algorithm)}
                  <span className="text-xs text-gray-500 capitalize">
                    {job.algorithm.replace('_', ' ')}
                  </span>
                </div>
              </div>
              
              {job.status === 'running' && (
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>Generation {job.current_generation}/{job.max_generations}</span>
                    <span>{job.progress}%</span>
                  </div>
                  <Progress value={job.progress} className="h-1" />
                  <div className="flex justify-between text-xs">
                    <span>Best Score</span>
                    <span className="font-medium">{(job.best_score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )}
              
              {job.status === 'completed' && (
                <div className="text-xs text-gray-500">
                  <div className="flex justify-between">
                    <span>Final Score</span>
                    <span className="font-medium text-green-600">
                      {(job.best_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
              
              <div className="text-xs text-gray-500 mt-2">
                Started {new Date(job.start_time).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {selectedJob ? (
          <>
            {/* Header */}
            <div className="p-6 border-b">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h1 className="text-2xl font-bold">{selectedJob.name}</h1>
                  <div className="flex items-center space-x-2 mt-1">
                    {getAlgorithmIcon(selectedJob.algorithm)}
                    <span className="text-gray-600 capitalize">
                      {selectedJob.algorithm.replace('_', ' ')} Algorithm
                    </span>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {selectedJob.status === 'running' && (
                    <>
                      <Button variant="outline" onClick={() => onPauseOptimization(selectedJob.id)}>
                        <Pause className="w-4 h-4 mr-2" />
                        Pause
                      </Button>
                      <Button variant="destructive" onClick={() => onStopOptimization(selectedJob.id)}>
                        <Square className="w-4 h-4 mr-2" />
                        Stop
                      </Button>
                    </>
                  )}
                  {selectedJob.status === 'completed' && selectedJob.generations.find(g => g.is_best) && (
                    <Button onClick={() => onApplyOptimization(selectedJob.id, selectedJob.generations.find(g => g.is_best)!.id)}>
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Apply Best
                    </Button>
                  )}
                </div>
              </div>

              {/* Status and Key Metrics */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    {getStatusIcon(selectedJob.status)}
                    <span className="font-medium">Status</span>
                  </div>
                  <Badge className={getStatusColor(selectedJob.status)}>
                    {selectedJob.status}
                  </Badge>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <BarChart3 className="w-4 h-4 text-blue-500" />
                    <span className="font-medium">Progress</span>
                  </div>
                  <div className="text-2xl font-bold">
                    {selectedJob.current_generation}/{selectedJob.max_generations}
                  </div>
                  <div className="text-sm text-gray-600">generations</div>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <TrendingUp className="w-4 h-4 text-green-500" />
                    <span className="font-medium">Best Score</span>
                  </div>
                  <div className="text-2xl font-bold">
                    {(selectedJob.best_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">
                    Target: {(selectedJob.target_score * 100).toFixed(0)}%
                  </div>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Clock className="w-4 h-4 text-purple-500" />
                    <span className="font-medium">Runtime</span>
                  </div>
                  <div className="text-2xl font-bold">
                    {selectedJob.end_time 
                      ? Math.round((new Date(selectedJob.end_time).getTime() - new Date(selectedJob.start_time).getTime()) / (1000 * 60 * 60))
                      : Math.round((Date.now() - new Date(selectedJob.start_time).getTime()) / (1000 * 60 * 60))
                    }h
                  </div>
                  <div className="text-sm text-gray-600">elapsed</div>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 p-6 overflow-auto">
              <Tabs defaultValue="progress" className="h-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="progress">Progress</TabsTrigger>
                  <TabsTrigger value="generations">Generations</TabsTrigger>
                  <TabsTrigger value="metrics">Metrics</TabsTrigger>
                  <TabsTrigger value="config">Configuration</TabsTrigger>
                </TabsList>
                
                <TabsContent value="progress" className="space-y-6">
                  {/* Progress Chart */}
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Optimization Progress</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={getProgressData(selectedJob)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="generation" />
                        <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                        <Tooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, '']} />
                        <Line 
                          type="monotone" 
                          dataKey="best_score" 
                          stroke="#8884d8" 
                          strokeWidth={2}
                          name="Best Score"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="avg_score" 
                          stroke="#82ca9d" 
                          strokeWidth={2}
                          name="Average Score"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Current Status */}
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Current Status</h3>
                    <div className="grid grid-cols-2 gap-6">
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium">Overall Progress</span>
                          <span className="text-sm text-gray-600">{selectedJob.progress}%</span>
                        </div>
                        <Progress value={selectedJob.progress} className="h-2" />
                      </div>
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium">Score Progress</span>
                          <span className="text-sm text-gray-600">
                            {((selectedJob.best_score / selectedJob.target_score) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <Progress 
                          value={(selectedJob.best_score / selectedJob.target_score) * 100} 
                          className="h-2" 
                        />
                      </div>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="generations" className="space-y-4">
                  {selectedJob.generations.map(generation => (
                    <div key={generation.id} className="bg-white p-6 rounded-lg border">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          <h4 className="text-lg font-semibold">Generation {generation.generation_number}</h4>
                          {generation.is_best && (
                            <Badge className="bg-yellow-100 text-yellow-800">
                              <TrendingUp className="w-3 h-3 mr-1" />
                              Best
                            </Badge>
                          )}
                        </div>
                        <div className="text-sm text-gray-500">
                          {new Date(generation.created_at).toLocaleString()}
                        </div>
                      </div>
                      
                      <div className="bg-gray-50 p-4 rounded-lg mb-4">
                        <h5 className="font-medium mb-2">Optimized Prompt</h5>
                        <pre className="text-sm whitespace-pre-wrap">{generation.prompt_content}</pre>
                      </div>
                      
                      <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
                        {Object.entries(generation.metrics).map(([key, value]) => (
                          <div key={key} className="text-center">
                            <div className="text-lg font-bold text-blue-600">
                              {(value * 100).toFixed(1)}%
                            </div>
                            <div className="text-xs text-gray-600 capitalize">
                              {key.replace('_', ' ')}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </TabsContent>
                
                <TabsContent value="metrics" className="space-y-6">
                  {/* Metrics Comparison */}
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={getMetricsData(selectedJob)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="metric" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip formatter={(value: any) => [`${value.toFixed(1)}%`, '']} />
                        <Area 
                          type="monotone" 
                          dataKey="value" 
                          stroke="#8884d8" 
                          fill="#8884d8" 
                          fillOpacity={0.6}
                          name="Current Score"
                        />
                        <Area 
                          type="monotone" 
                          dataKey="target" 
                          stroke="#82ca9d" 
                          fill="#82ca9d" 
                          fillOpacity={0.3}
                          name="Target Score"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Detailed Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {getMetricsData(selectedJob).map(metric => (
                      <div key={metric.metric} className="bg-white p-4 rounded-lg border">
                        <div className="flex justify-between items-center mb-2">
                          <h4 className="font-medium">{metric.metric}</h4>
                          <span className="text-lg font-bold">
                            {metric.value.toFixed(1)}%
                          </span>
                        </div>
                        <Progress value={metric.value} className="h-2 mb-2" />
                        <div className="flex justify-between text-sm text-gray-600">
                          <span>Target: {metric.target}%</span>
                          <span className={metric.value >= metric.target ? 'text-green-600' : 'text-red-600'}>
                            {metric.value >= metric.target ? '✓ Met' : '✗ Below target'}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </TabsContent>
                
                <TabsContent value="config" className="space-y-6">
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Algorithm Configuration</h3>
                    <div className="grid grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium mb-2">Algorithm</label>
                        <div className="text-lg capitalize">{selectedJob.algorithm.replace('_', ' ')}</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Population Size</label>
                        <div className="text-lg">{selectedJob.config.population_size}</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Mutation Rate</label>
                        <div className="text-lg">{(selectedJob.config.mutation_rate * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Crossover Rate</label>
                        <div className="text-lg">{(selectedJob.config.crossover_rate * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Selection Method</label>
                        <div className="text-lg capitalize">{selectedJob.config.selection_method}</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Max Generations</label>
                        <div className="text-lg">{selectedJob.max_generations}</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Original Prompt</h3>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <pre className="text-sm whitespace-pre-wrap">{selectedJob.original_prompt}</pre>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-2">No Optimization Job Selected</h2>
              <p className="text-gray-600 mb-4">Select a job from the sidebar to view details</p>
              <Button onClick={() => onStartOptimization('', {})}>
                <Play className="w-4 h-4 mr-2" />
                Start New Optimization
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}