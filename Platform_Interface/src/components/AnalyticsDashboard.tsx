import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import {
  TrendingUp,
  Download,
  Filter,
  RefreshCw,
  Calendar,
  BarChart3,
  Activity,
  Zap,
  Target,
  Users,
  Clock,
  DollarSign
} from 'lucide-react';

const performanceData = [
  { date: '2024-01-01', requests: 1200, latency: 0.8, success: 98.5, cost: 45.2 },
  { date: '2024-01-02', requests: 1350, latency: 0.9, success: 97.8, cost: 52.1 },
  { date: '2024-01-03', requests: 1180, latency: 0.7, success: 99.1, cost: 43.8 },
  { date: '2024-01-04', requests: 1420, latency: 1.1, success: 96.9, cost: 58.3 },
  { date: '2024-01-05', requests: 1580, latency: 1.2, success: 97.5, cost: 64.7 },
  { date: '2024-01-06', requests: 1340, latency: 0.9, success: 98.8, cost: 51.9 },
  { date: '2024-01-07', requests: 1260, latency: 0.8, success: 99.2, cost: 48.5 },
];

const agentPerformance = [
  { agent: 'CTO Agent', accuracy: 94, responseTime: 1.2, satisfaction: 4.8, requests: 1250 },
  { agent: 'Data Scientist', accuracy: 91, responseTime: 2.1, satisfaction: 4.6, requests: 980 },
  { agent: 'ML Engineer', accuracy: 96, responseTime: 1.8, satisfaction: 4.9, requests: 750 },
  { agent: 'BI Analyst', accuracy: 89, responseTime: 1.5, satisfaction: 4.4, requests: 650 },
  { agent: 'DevOps', accuracy: 92, responseTime: 1.0, satisfaction: 4.7, requests: 420 },
];

const usageByDepartment = [
  { name: 'Engineering', value: 35, requests: 2840, color: '#3b82f6' },
  { name: 'Marketing', value: 28, requests: 2280, color: '#10b981' },
  { name: 'Sales', value: 22, requests: 1790, color: '#f59e0b' },
  { name: 'Operations', value: 15, requests: 1220, color: '#ef4444' },
];

const modelMetrics = [
  { metric: 'Accuracy', current: 94.2, target: 95, previous: 92.8 },
  { metric: 'Latency', current: 1.2, target: 1.0, previous: 1.4 },
  { metric: 'Throughput', current: 850, target: 1000, previous: 720 },
  { metric: 'Error Rate', current: 2.1, target: 1.5, previous: 3.2 },
  { metric: 'Cost/Request', current: 0.045, target: 0.040, previous: 0.052 },
];

export function AnalyticsDashboard() {
  const [timeRange, setTimeRange] = useState('7d');
  const [selectedMetric, setSelectedMetric] = useState('requests');

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl mb-2">Analytics Dashboard</h1>
          <p className="text-muted-foreground">Monitor your AI platform performance and usage metrics.</p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1d">Last 24h</SelectItem>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 90 days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Total Requests</p>
                <h3 className="text-2xl">1.2M</h3>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="w-4 h-4 text-green-500" />
                  <span className="text-sm text-green-600">+15.3%</span>
                </div>
              </div>
              <div className="p-3 bg-blue-100 rounded-lg">
                <Activity className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Avg Latency</p>
                <h3 className="text-2xl">1.2s</h3>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="w-4 h-4 text-green-500" />
                  <span className="text-sm text-green-600">-8.7%</span>
                </div>
              </div>
              <div className="p-3 bg-green-100 rounded-lg">
                <Zap className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Success Rate</p>
                <h3 className="text-2xl">98.1%</h3>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="w-4 h-4 text-green-500" />
                  <span className="text-sm text-green-600">+2.4%</span>
                </div>
              </div>
              <div className="p-3 bg-emerald-100 rounded-lg">
                <Target className="w-6 h-6 text-emerald-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Total Cost</p>
                <h3 className="text-2xl">$2,847</h3>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="w-4 h-4 text-red-500" />
                  <span className="text-sm text-red-600">+12.1%</span>
                </div>
              </div>
              <div className="p-3 bg-yellow-100 rounded-lg">
                <DollarSign className="w-6 h-6 text-yellow-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="performance" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="agents">Agent Analytics</TabsTrigger>
          <TabsTrigger value="usage">Usage Patterns</TabsTrigger>
          <TabsTrigger value="models">Model Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Request Volume Over Time
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="requests" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5" />
                  Response Time & Success Rate
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Line yAxisId="left" type="monotone" dataKey="latency" stroke="#f59e0b" strokeWidth={2} />
                      <Line yAxisId="right" type="monotone" dataKey="success" stroke="#10b981" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <DollarSign className="w-5 h-5" />
                Cost Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="cost" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="agents" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Agent Performance Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {agentPerformance.map((agent, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{agent.agent}</span>
                        <Badge variant="secondary">{agent.requests} requests</Badge>
                      </div>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Accuracy: </span>
                          <span className="font-medium">{agent.accuracy}%</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Response: </span>
                          <span className="font-medium">{agent.responseTime}s</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Rating: </span>
                          <span className="font-medium">{agent.satisfaction}/5</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Agent Workload Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={agentPerformance} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="agent" type="category" width={100} />
                      <Tooltip />
                      <Bar dataKey="requests" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="usage" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Usage by Department</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={usageByDepartment}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {usageByDepartment.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-4 mt-4">
                  {usageByDepartment.map((item, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                      <div className="text-sm">
                        <div className="font-medium">{item.name}</div>
                        <div className="text-muted-foreground">{item.requests} requests</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Peak Usage Hours</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-sm text-muted-foreground">
                    Most active hours in your timezone
                  </div>
                  <div className="space-y-3">
                    {[
                      { hour: '9:00 AM - 10:00 AM', usage: 85, requests: 1240 },
                      { hour: '2:00 PM - 3:00 PM', usage: 92, requests: 1350 },
                      { hour: '10:00 AM - 11:00 AM', usage: 78, requests: 1140 },
                      { hour: '3:00 PM - 4:00 PM', usage: 71, requests: 1040 },
                    ].map((slot, index) => (
                      <div key={index} className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>{slot.hour}</span>
                          <span className="font-medium">{slot.requests} requests</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div 
                            className="bg-primary h-2 rounded-full" 
                            style={{ width: `${slot.usage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="models" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {modelMetrics.map((metric, index) => (
              <Card key={index}>
                <CardContent className="p-6">
                  <div className="space-y-2">
                    <h4 className="font-medium">{metric.metric}</h4>
                    <div className="text-2xl">{metric.current}</div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Target: {metric.target}</span>
                      <span className={`${
                        metric.current > metric.previous ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {metric.current > metric.previous ? '+' : ''}{((metric.current - metric.previous) / metric.previous * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}