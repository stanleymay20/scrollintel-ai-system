import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
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
  Cell
} from 'recharts';
import {
  Bot,
  Zap,
  TrendingUp,
  Users,
  Clock,
  CheckCircle,
  AlertCircle,
  Activity,
  Database,
  Cpu
} from 'lucide-react';

const systemMetrics = [
  { name: 'Active Agents', value: '12', change: '+2', trend: 'up', icon: Bot },
  { name: 'API Requests', value: '1.2M', change: '+15%', trend: 'up', icon: Zap },
  { name: 'Active Users', value: '2,847', change: '+8%', trend: 'up', icon: Users },
  { name: 'Avg Response', value: '1.2s', change: '-0.3s', trend: 'down', icon: Clock },
];

const performanceData = [
  { time: '00:00', requests: 120, latency: 0.8, cpu: 45 },
  { time: '04:00', requests: 89, latency: 0.7, cpu: 38 },
  { time: '08:00', requests: 234, latency: 1.1, cpu: 62 },
  { time: '12:00', requests: 378, latency: 1.3, cpu: 78 },
  { time: '16:00', requests: 445, latency: 1.5, cpu: 85 },
  { time: '20:00', requests: 267, latency: 1.0, cpu: 56 },
];

const agentUsageData = [
  { name: 'CTO Agent', value: 35, color: '#3b82f6' },
  { name: 'Data Scientist', value: 28, color: '#10b981' },
  { name: 'ML Engineer', value: 22, color: '#f59e0b' },
  { name: 'BI Analyst', value: 15, color: '#ef4444' },
];

const recentActivities = [
  {
    id: 1,
    type: 'model_training',
    title: 'Customer Segmentation Model',
    agent: 'Sarah Kim',
    status: 'completed',
    time: '2 minutes ago',
    accuracy: '94.2%'
  },
  {
    id: 2,
    type: 'data_analysis',
    title: 'Q4 Revenue Analysis',
    agent: 'Emma Thompson',
    status: 'in_progress',
    time: '15 minutes ago',
    progress: 67
  },
  {
    id: 3,
    type: 'prompt_optimization',
    title: 'Content Generation Prompts',
    agent: 'Alex Chen',
    status: 'completed',
    time: '1 hour ago',
    improvement: '+23%'
  },
];

export function Dashboard() {
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl mb-2">Welcome back, John</h1>
        <p className="text-muted-foreground">Here's what's happening with your AI platform today.</p>
      </div>

      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {systemMetrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <Card key={index}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">{metric.name}</p>
                    <div className="flex items-baseline gap-2">
                      <h3 className="text-2xl">{metric.value}</h3>
                      <Badge 
                        variant={metric.trend === 'up' ? 'default' : 'secondary'}
                        className={metric.trend === 'up' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}
                      >
                        {metric.change}
                      </Badge>
                    </div>
                  </div>
                  <div className="p-3 bg-primary/10 rounded-lg">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Performance Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              System Performance (24h)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Line yAxisId="left" type="monotone" dataKey="requests" stroke="#3b82f6" strokeWidth={2} />
                  <Line yAxisId="right" type="monotone" dataKey="latency" stroke="#10b981" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Agent Usage */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot className="w-5 h-5" />
              Agent Usage Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={agentUsageData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {agentUsageData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-2 mt-4">
              {agentUsageData.map((item, index) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span>{item.name}</span>
                  </div>
                  <span className="font-medium">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activities */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Recent Activities
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {recentActivities.map((activity) => (
              <div key={activity.id} className="flex items-start gap-3 p-3 rounded-lg border border-border">
                <div className="p-2 bg-primary/10 rounded-lg">
                  {activity.status === 'completed' ? (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  ) : (
                    <Clock className="w-4 h-4 text-yellow-500" />
                  )}
                </div>
                <div className="flex-1">
                  <h4 className="font-medium">{activity.title}</h4>
                  <p className="text-sm text-muted-foreground">by {activity.agent}</p>
                  <div className="flex items-center gap-4 mt-2">
                    <Badge variant={activity.status === 'completed' ? 'default' : 'secondary'}>
                      {activity.status === 'completed' ? 'Completed' : 'In Progress'}
                    </Badge>
                    <span className="text-xs text-muted-foreground">{activity.time}</span>
                  </div>
                  {activity.progress && (
                    <Progress value={activity.progress} className="mt-2" />
                  )}
                  {activity.accuracy && (
                    <div className="mt-1 text-sm text-green-600">Accuracy: {activity.accuracy}</div>
                  )}
                  {activity.improvement && (
                    <div className="mt-1 text-sm text-blue-600">Improvement: {activity.improvement}</div>
                  )}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* System Health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="w-5 h-5" />
              System Health
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">CPU Usage</span>
                <span className="text-sm font-medium">78%</span>
              </div>
              <Progress value={78} className="h-2" />
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Memory Usage</span>
                <span className="text-sm font-medium">65%</span>
              </div>
              <Progress value={65} className="h-2" />
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Database Load</span>
                <span className="text-sm font-medium">42%</span>
              </div>
              <Progress value={42} className="h-2" />
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">API Rate Limit</span>
                <span className="text-sm font-medium">85%</span>
              </div>
              <Progress value={85} className="h-2" />
            </div>

            <div className="pt-4 border-t border-border">
              <div className="flex items-center gap-2 text-sm text-green-600">
                <CheckCircle className="w-4 h-4" />
                All systems operational
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}