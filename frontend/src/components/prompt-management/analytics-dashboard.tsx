'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
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
  Cell,
  AreaChart,
  Area,
  ScatterChart,
  Scatter
} from 'recharts'
import { 
  TrendingUp, 
  TrendingDown, 
  Users, 
  Clock, 
  Target,
  Award,
  AlertTriangle,
  Download,
  Filter,
  Calendar,
  BarChart3,
  PieChart as PieChartIcon,
  Activity
} from 'lucide-react'

interface PromptUsageMetrics {
  prompt_id: string
  prompt_name: string
  total_requests: number
  success_rate: number
  avg_response_time: number
  user_satisfaction: number
  error_rate: number
  cost_per_request: number
  last_used: string
}

interface TeamAnalytics {
  user_id: string
  user_name: string
  prompts_created: number
  prompts_used: number
  total_requests: number
  avg_performance: number
  favorite_categories: string[]
}

interface TrendData {
  date: string
  requests: number
  success_rate: number
  avg_response_time: number
  user_satisfaction: number
}

interface CategoryMetrics {
  category: string
  prompt_count: number
  usage_count: number
  avg_performance: number
  top_prompts: string[]
}

interface AnalyticsDashboardProps {
  onExportReport: (filters: any) => void
  onDrillDown: (metric: string, filters: any) => void
}

export function AnalyticsDashboard({ onExportReport, onDrillDown }: AnalyticsDashboardProps) {
  const [promptMetrics, setPromptMetrics] = useState<PromptUsageMetrics[]>([])
  const [teamAnalytics, setTeamAnalytics] = useState<TeamAnalytics[]>([])
  const [trendData, setTrendData] = useState<TrendData[]>([])
  const [categoryMetrics, setCategoryMetrics] = useState<CategoryMetrics[]>([])
  const [selectedTimeRange, setSelectedTimeRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [isLoading, setIsLoading] = useState(true)

  // Mock data - replace with actual API calls
  useEffect(() => {
    const mockPromptMetrics: PromptUsageMetrics[] = [
      {
        prompt_id: '1',
        prompt_name: 'Content Generation',
        total_requests: 1250,
        success_rate: 0.94,
        avg_response_time: 1200,
        user_satisfaction: 4.5,
        error_rate: 0.06,
        cost_per_request: 0.02,
        last_used: '2024-01-20T15:30:00Z'
      },
      {
        prompt_id: '2',
        prompt_name: 'Code Review Assistant',
        total_requests: 890,
        success_rate: 0.89,
        avg_response_time: 800,
        user_satisfaction: 4.2,
        error_rate: 0.11,
        cost_per_request: 0.015,
        last_used: '2024-01-20T14:45:00Z'
      },
      {
        prompt_id: '3',
        prompt_name: 'Email Composer',
        total_requests: 650,
        success_rate: 0.92,
        avg_response_time: 950,
        user_satisfaction: 4.3,
        error_rate: 0.08,
        cost_per_request: 0.018,
        last_used: '2024-01-20T16:15:00Z'
      }
    ]

    const mockTeamAnalytics: TeamAnalytics[] = [
      {
        user_id: '1',
        user_name: 'John Doe',
        prompts_created: 15,
        prompts_used: 45,
        total_requests: 2100,
        avg_performance: 0.91,
        favorite_categories: ['content', 'marketing']
      },
      {
        user_id: '2',
        user_name: 'Jane Smith',
        prompts_created: 22,
        prompts_used: 38,
        total_requests: 1850,
        avg_performance: 0.88,
        favorite_categories: ['development', 'code-review']
      }
    ]

    const mockTrendData: TrendData[] = Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      requests: Math.floor(Math.random() * 200) + 100,
      success_rate: 0.85 + Math.random() * 0.1,
      avg_response_time: 800 + Math.random() * 400,
      user_satisfaction: 4.0 + Math.random() * 0.8
    }))

    const mockCategoryMetrics: CategoryMetrics[] = [
      {
        category: 'content',
        prompt_count: 25,
        usage_count: 1500,
        avg_performance: 0.92,
        top_prompts: ['Content Generation', 'Blog Writer', 'Social Media']
      },
      {
        category: 'development',
        prompt_count: 18,
        usage_count: 1200,
        avg_performance: 0.89,
        top_prompts: ['Code Review', 'Bug Finder', 'Documentation']
      },
      {
        category: 'marketing',
        prompt_count: 12,
        usage_count: 800,
        avg_performance: 0.87,
        top_prompts: ['Email Campaign', 'Ad Copy', 'SEO Content']
      }
    ]

    setPromptMetrics(mockPromptMetrics)
    setTeamAnalytics(mockTeamAnalytics)
    setTrendData(mockTrendData)
    setCategoryMetrics(mockCategoryMetrics)
    setIsLoading(false)
  }, [selectedTimeRange])

  const totalRequests = promptMetrics.reduce((sum, p) => sum + p.total_requests, 0)
  const avgSuccessRate = promptMetrics.reduce((sum, p) => sum + p.success_rate, 0) / promptMetrics.length
  const avgResponseTime = promptMetrics.reduce((sum, p) => sum + p.avg_response_time, 0) / promptMetrics.length
  const avgSatisfaction = promptMetrics.reduce((sum, p) => sum + p.user_satisfaction, 0) / promptMetrics.length

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d']

  const getPerformanceColor = (value: number, type: 'rate' | 'time' | 'satisfaction') => {
    switch (type) {
      case 'rate':
        return value >= 0.9 ? 'text-green-600' : value >= 0.8 ? 'text-yellow-600' : 'text-red-600'
      case 'time':
        return value <= 1000 ? 'text-green-600' : value <= 2000 ? 'text-yellow-600' : 'text-red-600'
      case 'satisfaction':
        return value >= 4.0 ? 'text-green-600' : value >= 3.5 ? 'text-yellow-600' : 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  const getPerformanceIcon = (value: number, type: 'rate' | 'time' | 'satisfaction') => {
    const isGood = type === 'time' ? value <= 1000 : value >= (type === 'satisfaction' ? 4.0 : 0.9)
    return isGood ? <TrendingUp className="w-4 h-4 text-green-500" /> : <TrendingDown className="w-4 h-4 text-red-500" />
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">Prompt Analytics</h1>
          <div className="flex items-center space-x-2">
            <select
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(e.target.value as any)}
              className="px-3 py-2 border rounded-lg"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
              <option value="1y">Last year</option>
            </select>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-3 py-2 border rounded-lg"
            >
              <option value="all">All Categories</option>
              {categoryMetrics.map(cat => (
                <option key={cat.category} value={cat.category}>
                  {cat.category}
                </option>
              ))}
            </select>
            <Button variant="outline" onClick={() => onExportReport({ timeRange: selectedTimeRange, category: selectedCategory })}>
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Total Requests</span>
              <Activity className="w-4 h-4 text-blue-500" />
            </div>
            <div className="text-2xl font-bold">{totalRequests.toLocaleString()}</div>
            <div className="text-sm text-gray-600">+12% from last period</div>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Success Rate</span>
              {getPerformanceIcon(avgSuccessRate, 'rate')}
            </div>
            <div className={`text-2xl font-bold ${getPerformanceColor(avgSuccessRate, 'rate')}`}>
              {(avgSuccessRate * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">+2.3% from last period</div>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Avg Response Time</span>
              {getPerformanceIcon(avgResponseTime, 'time')}
            </div>
            <div className={`text-2xl font-bold ${getPerformanceColor(avgResponseTime, 'time')}`}>
              {Math.round(avgResponseTime)}ms
            </div>
            <div className="text-sm text-gray-600">-5% from last period</div>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">User Satisfaction</span>
              {getPerformanceIcon(avgSatisfaction, 'satisfaction')}
            </div>
            <div className={`text-2xl font-bold ${getPerformanceColor(avgSatisfaction, 'satisfaction')}`}>
              {avgSatisfaction.toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">+0.2 from last period</div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-6 overflow-auto">
        <Tabs defaultValue="overview" className="h-full">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="prompts">Prompt Performance</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="team">Team Analytics</TabsTrigger>
            <TabsTrigger value="categories">Categories</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="space-y-6">
            {/* Usage Trends */}
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Usage Trends</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
                  <YAxis />
                  <Tooltip labelFormatter={(value) => new Date(value).toLocaleDateString()} />
                  <Area 
                    type="monotone" 
                    dataKey="requests" 
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    fillOpacity={0.6}
                    name="Requests"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Top Performing Prompts */}
              <div className="bg-white p-6 rounded-lg border">
                <h3 className="text-lg font-semibold mb-4">Top Performing Prompts</h3>
                <div className="space-y-3">
                  {promptMetrics
                    .sort((a, b) => b.success_rate - a.success_rate)
                    .slice(0, 5)
                    .map((prompt, index) => (
                      <div key={prompt.prompt_id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                            {index + 1}
                          </div>
                          <div>
                            <div className="font-medium">{prompt.prompt_name}</div>
                            <div className="text-sm text-gray-600">
                              {prompt.total_requests} requests
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-bold text-green-600">
                            {(prompt.success_rate * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-600">
                            {prompt.user_satisfaction.toFixed(1)} ★
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>

              {/* Category Distribution */}
              <div className="bg-white p-6 rounded-lg border">
                <h3 className="text-lg font-semibold mb-4">Usage by Category</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={categoryMetrics.map((cat, index) => ({
                        name: cat.category,
                        value: cat.usage_count,
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
                      {categoryMetrics.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="prompts" className="space-y-4">
            <div className="bg-white rounded-lg border overflow-hidden">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold">Prompt Performance Details</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium">Prompt Name</th>
                      <th className="px-4 py-3 text-center font-medium">Requests</th>
                      <th className="px-4 py-3 text-center font-medium">Success Rate</th>
                      <th className="px-4 py-3 text-center font-medium">Avg Response Time</th>
                      <th className="px-4 py-3 text-center font-medium">Satisfaction</th>
                      <th className="px-4 py-3 text-center font-medium">Cost/Request</th>
                      <th className="px-4 py-3 text-center font-medium">Last Used</th>
                    </tr>
                  </thead>
                  <tbody>
                    {promptMetrics.map(prompt => (
                      <tr key={prompt.prompt_id} className="border-b hover:bg-gray-50">
                        <td className="px-4 py-3">
                          <div className="font-medium">{prompt.prompt_name}</div>
                          <div className="text-sm text-gray-600">ID: {prompt.prompt_id}</div>
                        </td>
                        <td className="px-4 py-3 text-center">
                          {prompt.total_requests.toLocaleString()}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span className={`font-medium ${getPerformanceColor(prompt.success_rate, 'rate')}`}>
                            {(prompt.success_rate * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span className={`font-medium ${getPerformanceColor(prompt.avg_response_time, 'time')}`}>
                            {prompt.avg_response_time}ms
                          </span>
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span className={`font-medium ${getPerformanceColor(prompt.user_satisfaction, 'satisfaction')}`}>
                            {prompt.user_satisfaction.toFixed(1)} ★
                          </span>
                        </td>
                        <td className="px-4 py-3 text-center">
                          ${prompt.cost_per_request.toFixed(3)}
                        </td>
                        <td className="px-4 py-3 text-center text-sm text-gray-600">
                          {new Date(prompt.last_used).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="trends" className="space-y-6">
            {/* Multi-metric Trend Chart */}
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Performance Trends</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip labelFormatter={(value) => new Date(value).toLocaleDateString()} />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="success_rate" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    name="Success Rate"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="avg_response_time" 
                    stroke="#82ca9d" 
                    strokeWidth={2}
                    name="Response Time (ms)"
                  />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="user_satisfaction" 
                    stroke="#ffc658" 
                    strokeWidth={2}
                    name="User Satisfaction"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Request Volume Trend */}
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Request Volume Trend</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
                  <YAxis />
                  <Tooltip labelFormatter={(value) => new Date(value).toLocaleDateString()} />
                  <Bar dataKey="requests" fill="#8884d8" name="Requests" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="team" className="space-y-6">
            {/* Team Performance */}
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Team Performance</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {teamAnalytics.map(user => (
                  <div key={user.user_id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold">{user.user_name}</h4>
                      <Badge variant="outline">
                        {(user.avg_performance * 100).toFixed(0)}% avg
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 mb-3">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">
                          {user.prompts_created}
                        </div>
                        <div className="text-sm text-gray-600">Created</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                          {user.prompts_used}
                        </div>
                        <div className="text-sm text-gray-600">Used</div>
                      </div>
                    </div>
                    
                    <div className="text-center mb-3">
                      <div className="text-lg font-bold">
                        {user.total_requests.toLocaleString()}
                      </div>
                      <div className="text-sm text-gray-600">Total Requests</div>
                    </div>
                    
                    <div>
                      <div className="text-sm font-medium mb-1">Favorite Categories</div>
                      <div className="flex flex-wrap gap-1">
                        {user.favorite_categories.map(category => (
                          <Badge key={category} variant="secondary" className="text-xs">
                            {category}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Team Collaboration Chart */}
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Prompts Created vs Used</h3>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={teamAnalytics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="prompts_created" name="Created" />
                  <YAxis dataKey="prompts_used" name="Used" />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value, name) => [value, name === 'prompts_created' ? 'Created' : 'Used']}
                    labelFormatter={(label: any) => label}
                  />
                  <Scatter dataKey="prompts_used" fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="categories" className="space-y-6">
            {/* Category Performance */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {categoryMetrics.map(category => (
                <div key={category.category} className="bg-white p-6 rounded-lg border">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold capitalize">{category.category}</h4>
                    <Badge variant="outline">
                      {category.prompt_count} prompts
                    </Badge>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Usage Count</span>
                      <span className="font-medium">{category.usage_count.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Avg Performance</span>
                      <span className={`font-medium ${getPerformanceColor(category.avg_performance, 'rate')}`}>
                        {(category.avg_performance * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <div className="text-sm font-medium mb-2">Top Prompts</div>
                    <div className="space-y-1">
                      {category.top_prompts.slice(0, 3).map(prompt => (
                        <div key={prompt} className="text-sm text-gray-600 truncate">
                          • {prompt}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Category Comparison Chart */}
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Category Performance Comparison</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={categoryMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="usage_count" fill="#8884d8" name="Usage Count" />
                  <Bar dataKey="prompt_count" fill="#82ca9d" name="Prompt Count" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}