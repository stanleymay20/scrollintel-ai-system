'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { 
  Play, 
  Pause, 
  Square, 
  TrendingUp, 
  TrendingDown, 
  Award,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Target
} from 'lucide-react'

interface ExperimentVariant {
  id: string
  name: string
  prompt_content: string
  traffic_percentage: number
  metrics: {
    conversion_rate: number
    response_time: number
    user_satisfaction: number
    error_rate: number
    total_requests: number
  }
}

interface Experiment {
  id: string
  name: string
  description: string
  status: 'draft' | 'running' | 'paused' | 'completed' | 'cancelled'
  variants: ExperimentVariant[]
  start_date: string
  end_date?: string
  confidence_level: number
  statistical_significance: number
  winner_variant_id?: string
  created_by: string
  created_at: string
}

interface ABTestingDashboardProps {
  onCreateExperiment: () => void
  onEditExperiment: (experiment: Experiment) => void
  onStartExperiment: (experimentId: string) => void
  onPauseExperiment: (experimentId: string) => void
  onStopExperiment: (experimentId: string) => void
  onPromoteWinner: (experimentId: string, variantId: string) => void
}

export function ABTestingDashboard({
  onCreateExperiment,
  onEditExperiment,
  onStartExperiment,
  onPauseExperiment,
  onStopExperiment,
  onPromoteWinner
}: ABTestingDashboardProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Mock data - replace with actual API calls
  useEffect(() => {
    const mockExperiments: Experiment[] = [
      {
        id: '1',
        name: 'Content Generation Optimization',
        description: 'Testing different approaches to content generation prompts',
        status: 'running',
        variants: [
          {
            id: 'v1',
            name: 'Control',
            prompt_content: 'Generate content about {topic}...',
            traffic_percentage: 50,
            metrics: {
              conversion_rate: 0.85,
              response_time: 1200,
              user_satisfaction: 4.2,
              error_rate: 0.02,
              total_requests: 1250
            }
          },
          {
            id: 'v2',
            name: 'Enhanced',
            prompt_content: 'Create engaging content about {topic} with specific focus on {audience}...',
            traffic_percentage: 50,
            metrics: {
              conversion_rate: 0.92,
              response_time: 1350,
              user_satisfaction: 4.6,
              error_rate: 0.015,
              total_requests: 1180
            }
          }
        ],
        start_date: '2024-01-15T10:00:00Z',
        confidence_level: 95,
        statistical_significance: 0.87,
        winner_variant_id: 'v2',
        created_by: 'john.doe',
        created_at: '2024-01-10T09:00:00Z'
      },
      {
        id: '2',
        name: 'Code Review Prompt Test',
        description: 'Comparing different code review prompt structures',
        status: 'completed',
        variants: [
          {
            id: 'v1',
            name: 'Basic',
            prompt_content: 'Review this code...',
            traffic_percentage: 33,
            metrics: {
              conversion_rate: 0.78,
              response_time: 800,
              user_satisfaction: 3.8,
              error_rate: 0.03,
              total_requests: 890
            }
          },
          {
            id: 'v2',
            name: 'Detailed',
            prompt_content: 'Perform a comprehensive code review...',
            traffic_percentage: 33,
            metrics: {
              conversion_rate: 0.82,
              response_time: 1100,
              user_satisfaction: 4.1,
              error_rate: 0.025,
              total_requests: 920
            }
          },
          {
            id: 'v3',
            name: 'Structured',
            prompt_content: 'Review the code following these criteria...',
            traffic_percentage: 34,
            metrics: {
              conversion_rate: 0.89,
              response_time: 950,
              user_satisfaction: 4.4,
              error_rate: 0.018,
              total_requests: 950
            }
          }
        ],
        start_date: '2024-01-01T10:00:00Z',
        end_date: '2024-01-14T18:00:00Z',
        confidence_level: 95,
        statistical_significance: 0.95,
        winner_variant_id: 'v3',
        created_by: 'jane.smith',
        created_at: '2023-12-28T14:00:00Z'
      }
    ]
    
    setExperiments(mockExperiments)
    setSelectedExperiment(mockExperiments[0])
    setIsLoading(false)
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Play className="w-4 h-4 text-green-500" />
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-500" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-blue-500" />
      case 'cancelled':
        return <Square className="w-4 h-4 text-red-500" />
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
      case 'cancelled':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const formatMetric = (value: number, type: string) => {
    switch (type) {
      case 'percentage':
        return `${(value * 100).toFixed(1)}%`
      case 'time':
        return `${value}ms`
      case 'rating':
        return value.toFixed(1)
      default:
        return value.toString()
    }
  }

  const getVariantPerformanceData = (experiment: Experiment) => {
    return experiment.variants.map(variant => ({
      name: variant.name,
      conversion_rate: variant.metrics.conversion_rate * 100,
      response_time: variant.metrics.response_time,
      satisfaction: variant.metrics.user_satisfaction,
      error_rate: variant.metrics.error_rate * 100
    }))
  }

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8']

  return (
    <div className="h-full flex">
      {/* Sidebar - Experiments List */}
      <div className="w-80 border-r flex flex-col">
        <div className="p-4 border-b">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Experiments</h2>
            <Button onClick={onCreateExperiment} size="sm">
              <Play className="w-4 h-4 mr-2" />
              New Test
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-auto">
          {experiments.map(experiment => (
            <div
              key={experiment.id}
              className={`p-4 border-b cursor-pointer hover:bg-gray-50 ${
                selectedExperiment?.id === experiment.id ? 'bg-blue-50 border-blue-200' : ''
              }`}
              onClick={() => setSelectedExperiment(experiment)}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium truncate">{experiment.name}</h3>
                {getStatusIcon(experiment.status)}
              </div>
              
              <div className="flex items-center space-x-2 mb-2">
                <Badge className={getStatusColor(experiment.status)}>
                  {experiment.status}
                </Badge>
                <span className="text-xs text-gray-500">
                  {experiment.variants.length} variants
                </span>
              </div>
              
              <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                {experiment.description}
              </p>
              
              {experiment.status === 'running' && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span>Confidence</span>
                    <span>{experiment.statistical_significance * 100}%</span>
                  </div>
                  <Progress value={experiment.statistical_significance * 100} className="h-1" />
                </div>
              )}
              
              <div className="text-xs text-gray-500 mt-2">
                Started {new Date(experiment.start_date).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {selectedExperiment ? (
          <>
            {/* Header */}
            <div className="p-6 border-b">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h1 className="text-2xl font-bold">{selectedExperiment.name}</h1>
                  <p className="text-gray-600">{selectedExperiment.description}</p>
                </div>
                <div className="flex items-center space-x-2">
                  {selectedExperiment.status === 'draft' && (
                    <Button onClick={() => onStartExperiment(selectedExperiment.id)}>
                      <Play className="w-4 h-4 mr-2" />
                      Start
                    </Button>
                  )}
                  {selectedExperiment.status === 'running' && (
                    <>
                      <Button variant="outline" onClick={() => onPauseExperiment(selectedExperiment.id)}>
                        <Pause className="w-4 h-4 mr-2" />
                        Pause
                      </Button>
                      <Button variant="destructive" onClick={() => onStopExperiment(selectedExperiment.id)}>
                        <Square className="w-4 h-4 mr-2" />
                        Stop
                      </Button>
                    </>
                  )}
                  {selectedExperiment.winner_variant_id && selectedExperiment.status === 'completed' && (
                    <Button onClick={() => onPromoteWinner(selectedExperiment.id, selectedExperiment.winner_variant_id!)}>
                      <Award className="w-4 h-4 mr-2" />
                      Promote Winner
                    </Button>
                  )}
                </div>
              </div>

              {/* Status and Key Metrics */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    {getStatusIcon(selectedExperiment.status)}
                    <span className="font-medium">Status</span>
                  </div>
                  <Badge className={getStatusColor(selectedExperiment.status)}>
                    {selectedExperiment.status}
                  </Badge>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Target className="w-4 h-4 text-blue-500" />
                    <span className="font-medium">Confidence</span>
                  </div>
                  <div className="text-2xl font-bold">
                    {(selectedExperiment.statistical_significance * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Users className="w-4 h-4 text-green-500" />
                    <span className="font-medium">Total Requests</span>
                  </div>
                  <div className="text-2xl font-bold">
                    {selectedExperiment.variants.reduce((sum, v) => sum + v.metrics.total_requests, 0).toLocaleString()}
                  </div>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Award className="w-4 h-4 text-yellow-500" />
                    <span className="font-medium">Winner</span>
                  </div>
                  <div className="text-lg font-semibold">
                    {selectedExperiment.winner_variant_id 
                      ? selectedExperiment.variants.find(v => v.id === selectedExperiment.winner_variant_id)?.name || 'TBD'
                      : 'TBD'
                    }
                  </div>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 p-6 overflow-auto">
              <Tabs defaultValue="overview" className="h-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="variants">Variants</TabsTrigger>
                  <TabsTrigger value="analytics">Analytics</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview" className="space-y-6">
                  {/* Performance Comparison Chart */}
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Performance Comparison</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={getVariantPerformanceData(selectedExperiment)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="conversion_rate" fill="#8884d8" name="Conversion Rate %" />
                        <Bar dataKey="satisfaction" fill="#82ca9d" name="Satisfaction" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Variants Summary */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {selectedExperiment.variants.map((variant, index) => (
                      <div key={variant.id} className="bg-white p-6 rounded-lg border">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="font-semibold">{variant.name}</h4>
                          {selectedExperiment.winner_variant_id === variant.id && (
                            <Badge className="bg-yellow-100 text-yellow-800">
                              <Award className="w-3 h-3 mr-1" />
                              Winner
                            </Badge>
                          )}
                        </div>
                        
                        <div className="space-y-3">
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Conversion Rate</span>
                            <span className="font-medium">
                              {formatMetric(variant.metrics.conversion_rate, 'percentage')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Response Time</span>
                            <span className="font-medium">
                              {formatMetric(variant.metrics.response_time, 'time')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Satisfaction</span>
                            <span className="font-medium">
                              {formatMetric(variant.metrics.user_satisfaction, 'rating')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Error Rate</span>
                            <span className="font-medium">
                              {formatMetric(variant.metrics.error_rate, 'percentage')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-600">Requests</span>
                            <span className="font-medium">
                              {variant.metrics.total_requests.toLocaleString()}
                            </span>
                          </div>
                        </div>
                        
                        <div className="mt-4">
                          <div className="flex justify-between text-sm mb-1">
                            <span>Traffic Split</span>
                            <span>{variant.traffic_percentage}%</span>
                          </div>
                          <Progress value={variant.traffic_percentage} className="h-2" />
                        </div>
                      </div>
                    ))}
                  </div>
                </TabsContent>
                
                <TabsContent value="variants" className="space-y-4">
                  {selectedExperiment.variants.map(variant => (
                    <div key={variant.id} className="bg-white p-6 rounded-lg border">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-lg font-semibold">{variant.name}</h4>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline">{variant.traffic_percentage}% traffic</Badge>
                          {selectedExperiment.winner_variant_id === variant.id && (
                            <Badge className="bg-yellow-100 text-yellow-800">
                              <Award className="w-3 h-3 mr-1" />
                              Winner
                            </Badge>
                          )}
                        </div>
                      </div>
                      
                      <div className="bg-gray-50 p-4 rounded-lg mb-4">
                        <h5 className="font-medium mb-2">Prompt Content</h5>
                        <pre className="text-sm whitespace-pre-wrap">{variant.prompt_content}</pre>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">
                            {formatMetric(variant.metrics.conversion_rate, 'percentage')}
                          </div>
                          <div className="text-sm text-gray-600">Conversion</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">
                            {formatMetric(variant.metrics.response_time, 'time')}
                          </div>
                          <div className="text-sm text-gray-600">Response Time</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-yellow-600">
                            {formatMetric(variant.metrics.user_satisfaction, 'rating')}
                          </div>
                          <div className="text-sm text-gray-600">Satisfaction</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-red-600">
                            {formatMetric(variant.metrics.error_rate, 'percentage')}
                          </div>
                          <div className="text-sm text-gray-600">Error Rate</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-600">
                            {variant.metrics.total_requests.toLocaleString()}
                          </div>
                          <div className="text-sm text-gray-600">Requests</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </TabsContent>
                
                <TabsContent value="analytics" className="space-y-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Traffic Distribution */}
                    <div className="bg-white p-6 rounded-lg border">
                      <h3 className="text-lg font-semibold mb-4">Traffic Distribution</h3>
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie
                            data={selectedExperiment.variants.map((variant, index) => ({
                              name: variant.name,
                              value: variant.traffic_percentage,
                              fill: COLORS[index % COLORS.length]
                            }))}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {selectedExperiment.variants.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Response Time Comparison */}
                    <div className="bg-white p-6 rounded-lg border">
                      <h3 className="text-lg font-semibold mb-4">Response Time Comparison</h3>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={getVariantPerformanceData(selectedExperiment)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="response_time" fill="#ffc658" name="Response Time (ms)" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="settings" className="space-y-6">
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Experiment Settings</h3>
                    <div className="grid grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium mb-2">Confidence Level</label>
                        <div className="text-lg">{selectedExperiment.confidence_level}%</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Created By</label>
                        <div className="text-lg">{selectedExperiment.created_by}</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">Start Date</label>
                        <div className="text-lg">{new Date(selectedExperiment.start_date).toLocaleString()}</div>
                      </div>
                      {selectedExperiment.end_date && (
                        <div>
                          <label className="block text-sm font-medium mb-2">End Date</label>
                          <div className="text-lg">{new Date(selectedExperiment.end_date).toLocaleString()}</div>
                        </div>
                      )}
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-2">No Experiment Selected</h2>
              <p className="text-gray-600 mb-4">Select an experiment from the sidebar to view details</p>
              <Button onClick={onCreateExperiment}>
                <Play className="w-4 h-4 mr-2" />
                Create New Experiment
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}