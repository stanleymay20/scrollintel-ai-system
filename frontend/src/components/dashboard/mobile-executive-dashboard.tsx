"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Users, 
  Activity, 
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  PieChart,
  LineChart,
  RefreshCw,
  Download,
  Share2,
  Filter,
  Calendar,
  Bell
} from 'lucide-react';

interface MetricData {
  label: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  status: 'good' | 'warning' | 'critical';
}

interface ExecutiveSummary {
  title: string;
  overview: string;
  keyFindings: Array<{
    title: string;
    description: string;
    priority: 'critical' | 'high' | 'medium' | 'low';
    impact: 'positive' | 'negative' | 'neutral';
  }>;
  recommendations: Array<{
    title: string;
    description: string;
    timeline: string;
    priority: 'critical' | 'high' | 'medium' | 'low';
  }>;
  nextSteps: string[];
}

interface DashboardData {
  metrics: MetricData[];
  executiveSummary: ExecutiveSummary;
  alerts: Array<{
    id: string;
    type: 'info' | 'warning' | 'error';
    message: string;
    timestamp: string;
  }>;
  lastUpdated: string;
}

const MobileExecutiveDashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      // Simulate API call - replace with actual API endpoint
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockData: DashboardData = {
        metrics: [
          {
            label: 'Revenue',
            value: '$2.4M',
            change: 18.2,
            trend: 'up',
            status: 'good'
          },
          {
            label: 'Active Users',
            value: '15.4K',
            change: 12.5,
            trend: 'up',
            status: 'good'
          },
          {
            label: 'System Uptime',
            value: '99.97%',
            change: 0.1,
            trend: 'up',
            status: 'good'
          },
          {
            label: 'ROI',
            value: '32.5%',
            change: 5.8,
            trend: 'up',
            status: 'good'
          },
          {
            label: 'Error Rate',
            value: '0.02%',
            change: -15.3,
            trend: 'down',
            status: 'good'
          },
          {
            label: 'Response Time',
            value: '245ms',
            change: -8.2,
            trend: 'down',
            status: 'good'
          }
        ],
        executiveSummary: {
          title: 'Q4 2024 Performance Summary',
          overview: 'Strong performance across all key metrics with revenue growth of 18.2% and exceptional system reliability. Strategic initiatives are delivering measurable results with ROI exceeding targets.',
          keyFindings: [
            {
              title: 'Exceptional Revenue Growth',
              description: 'Revenue increased 18.2% quarter-over-quarter, significantly exceeding industry benchmarks.',
              priority: 'high',
              impact: 'positive'
            },
            {
              title: 'Outstanding System Reliability',
              description: 'System uptime of 99.97% demonstrates excellent infrastructure performance.',
              priority: 'medium',
              impact: 'positive'
            },
            {
              title: 'Strong User Engagement',
              description: 'Active user base grew 12.5% with improved retention metrics.',
              priority: 'medium',
              impact: 'positive'
            }
          ],
          recommendations: [
            {
              title: 'Scale High-ROI Initiatives',
              description: 'Increase investment in proven high-return programs to maintain growth momentum.',
              timeline: 'Q1 2025',
              priority: 'high'
            },
            {
              title: 'Expand Market Presence',
              description: 'Leverage strong performance metrics to enter new market segments.',
              timeline: 'Q2 2025',
              priority: 'medium'
            }
          ],
          nextSteps: [
            'Finalize Q1 2025 budget allocation for growth initiatives',
            'Conduct market analysis for expansion opportunities',
            'Implement advanced monitoring for continued reliability'
          ]
        },
        alerts: [
          {
            id: '1',
            type: 'info',
            message: 'Monthly report generation completed successfully',
            timestamp: '2 hours ago'
          },
          {
            id: '2',
            type: 'warning',
            message: 'CPU usage approaching 80% threshold',
            timestamp: '4 hours ago'
          }
        ],
        lastUpdated: new Date().toISOString()
      };
      
      setData(mockData);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'down':
        return <TrendingDown className="h-4 w-4 text-red-500" />;
      default:
        return <Activity className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: 'good' | 'warning' | 'critical') => {
    switch (status) {
      case 'good':
        return 'text-green-600';
      case 'warning':
        return 'text-yellow-600';
      case 'critical':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getPriorityBadgeVariant = (priority: 'critical' | 'high' | 'medium' | 'low') => {
    switch (priority) {
      case 'critical':
        return 'destructive';
      case 'high':
        return 'default';
      case 'medium':
        return 'secondary';
      case 'low':
        return 'outline';
      default:
        return 'outline';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading executive dashboard...</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 mx-auto mb-4 text-red-600" />
          <p className="text-gray-600">Failed to load dashboard data</p>
          <Button onClick={loadDashboardData} className="mt-4">
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm border-b sticky top-0 z-10">
        <div className="px-4 py-3">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-semibold text-gray-900">Executive Dashboard</h1>
              <p className="text-sm text-gray-500">
                Last updated: {new Date(data.lastUpdated).toLocaleTimeString()}
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleRefresh}
                disabled={refreshing}
              >
                <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              </Button>
              <Button variant="ghost" size="sm">
                <Bell className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Alerts */}
      {data.alerts.length > 0 && (
        <div className="px-4 py-2 space-y-2">
          {data.alerts.map((alert) => (
            <Alert key={alert.id} className="py-2">
              <AlertDescription className="text-sm">
                <div className="flex items-center justify-between">
                  <span>{alert.message}</span>
                  <span className="text-xs text-gray-500">{alert.timestamp}</span>
                </div>
              </AlertDescription>
            </Alert>
          ))}
        </div>
      )}

      {/* Main Content */}
      <div className="px-4 py-4">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-4">
            <TabsTrigger value="overview" className="text-xs">Overview</TabsTrigger>
            <TabsTrigger value="metrics" className="text-xs">Metrics</TabsTrigger>
            <TabsTrigger value="insights" className="text-xs">Insights</TabsTrigger>
            <TabsTrigger value="actions" className="text-xs">Actions</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 gap-3">
              {data.metrics.slice(0, 4).map((metric, index) => (
                <Card key={index} className="p-3">
                  <CardContent className="p-0">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-medium text-gray-600">{metric.label}</p>
                      {getTrendIcon(metric.trend)}
                    </div>
                    <div className="space-y-1">
                      <p className={`text-lg font-bold ${getStatusColor(metric.status)}`}>
                        {metric.value}
                      </p>
                      <p className={`text-xs ${metric.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {metric.change >= 0 ? '+' : ''}{metric.change}%
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Executive Summary */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">{data.executiveSummary.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-700 leading-relaxed">
                  {data.executiveSummary.overview}
                </p>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full justify-start text-sm">
                  <Download className="h-4 w-4 mr-2" />
                  Download Full Report
                </Button>
                <Button variant="outline" className="w-full justify-start text-sm">
                  <Share2 className="h-4 w-4 mr-2" />
                  Share Dashboard
                </Button>
                <Button variant="outline" className="w-full justify-start text-sm">
                  <Calendar className="h-4 w-4 mr-2" />
                  Schedule Report
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="space-y-4">
            <div className="space-y-3">
              {data.metrics.map((metric, index) => (
                <Card key={index}>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <h3 className="font-medium text-sm">{metric.label}</h3>
                        {getTrendIcon(metric.trend)}
                      </div>
                      <Badge variant={getPriorityBadgeVariant(metric.status as any)}>
                        {metric.status}
                      </Badge>
                    </div>
                    
                    <div className="flex items-end justify-between mb-2">
                      <span className={`text-2xl font-bold ${getStatusColor(metric.status)}`}>
                        {metric.value}
                      </span>
                      <span className={`text-sm font-medium ${metric.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {metric.change >= 0 ? '+' : ''}{metric.change}%
                      </span>
                    </div>
                    
                    <Progress 
                      value={Math.abs(metric.change)} 
                      className="h-2"
                    />
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Insights Tab */}
          <TabsContent value="insights" className="space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Key Findings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {data.executiveSummary.keyFindings.map((finding, index) => (
                  <div key={index} className="border-l-4 border-blue-500 pl-3 py-2">
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="font-medium text-sm">{finding.title}</h4>
                      <Badge variant={getPriorityBadgeVariant(finding.priority)}>
                        {finding.priority}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-600 leading-relaxed">
                      {finding.description}
                    </p>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Recommendations</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {data.executiveSummary.recommendations.map((rec, index) => (
                  <div key={index} className="border rounded-lg p-3 bg-blue-50">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-sm">{rec.title}</h4>
                      <Badge variant={getPriorityBadgeVariant(rec.priority)}>
                        {rec.priority}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-700 mb-2">{rec.description}</p>
                    <div className="flex items-center text-xs text-gray-500">
                      <Clock className="h-3 w-3 mr-1" />
                      {rec.timeline}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Actions Tab */}
          <TabsContent value="actions" className="space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Next Steps</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {data.executiveSummary.nextSteps.map((step, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
                        <span className="text-xs font-medium text-blue-600">{index + 1}</span>
                      </div>
                      <p className="text-sm text-gray-700 leading-relaxed">{step}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Report Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <Download className="h-4 w-4 mr-2" />
                  Generate Detailed Report
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Share2 className="h-4 w-4 mr-2" />
                  Share with Team
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Calendar className="h-4 w-4 mr-2" />
                  Schedule Next Review
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Filter className="h-4 w-4 mr-2" />
                  Customize Dashboard
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Performance Tracking</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  View Detailed Analytics
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <PieChart className="h-4 w-4 mr-2" />
                  ROI Breakdown
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <LineChart className="h-4 w-4 mr-2" />
                  Trend Analysis
                </Button>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default MobileExecutiveDashboard;