'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  DollarSign, 
  Users, 
  Activity, 
  BarChart3,
  RefreshCw,
  Download,
  Share,
  Filter,
  Calendar,
  ChevronRight,
  ChevronDown
} from 'lucide-react';

interface MetricCard {
  id: string;
  title: string;
  value: string;
  change: string;
  trend: 'up' | 'down' | 'stable';
  category: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
}

interface KeyFinding {
  id: string;
  title: string;
  description: string;
  impact: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  category: string;
}

interface Recommendation {
  id: string;
  title: string;
  description: string;
  timeline: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  implementation_effort: string;
}

interface DashboardData {
  metrics: MetricCard[];
  key_findings: KeyFinding[];
  recommendations: Recommendation[];
  risk_alerts: string[];
  last_updated: string;
  confidence_score: number;
}

const MobileExecutiveDashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockData: DashboardData = {
        metrics: [
          {
            id: '1',
            title: 'Revenue',
            value: '$2.4M',
            change: '+12.5%',
            trend: 'up',
            category: 'financial',
            priority: 'high'
          },
          {
            id: '2',
            title: 'Active Users',
            value: '45.2K',
            change: '+8.3%',
            trend: 'up',
            category: 'engagement',
            priority: 'medium'
          },
          {
            id: '3',
            title: 'System Uptime',
            value: '98.7%',
            change: '-0.3%',
            trend: 'down',
            category: 'performance',
            priority: 'high'
          },
          {
            id: '4',
            title: 'ROI',
            value: '24.8%',
            change: '+3.2%',
            trend: 'up',
            category: 'financial',
            priority: 'high'
          }
        ],
        key_findings: [
          {
            id: '1',
            title: 'Strong Revenue Growth',
            description: 'Revenue growth of 12.5% exceeds quarterly targets',
            impact: 'Positive financial trajectory supporting expansion plans',
            priority: 'high',
            category: 'financial'
          },
          {
            id: '2',
            title: 'System Uptime Below Target',
            description: 'Uptime at 98.7% is below the 99% SLA target',
            impact: 'Service reliability concerns may affect customer satisfaction',
            priority: 'high',
            category: 'performance'
          },
          {
            id: '3',
            title: 'User Engagement Increasing',
            description: 'Active user base growing at 8.3% month-over-month',
            impact: 'Strong user adoption indicates product-market fit',
            priority: 'medium',
            category: 'engagement'
          }
        ],
        recommendations: [
          {
            id: '1',
            title: 'Implement Infrastructure Redundancy',
            description: 'Deploy failover systems to improve uptime reliability',
            timeline: '2-3 months',
            priority: 'high',
            implementation_effort: 'Medium'
          },
          {
            id: '2',
            title: 'Expand Marketing Investment',
            description: 'Leverage strong revenue growth to increase market reach',
            timeline: '1-2 months',
            priority: 'medium',
            implementation_effort: 'Low'
          },
          {
            id: '3',
            title: 'Optimize User Onboarding',
            description: 'Improve new user experience to accelerate growth',
            timeline: '3-4 weeks',
            priority: 'medium',
            implementation_effort: 'Medium'
          }
        ],
        risk_alerts: [
          'System uptime below SLA threshold',
          'Increased customer support tickets related to performance'
        ],
        last_updated: new Date().toISOString(),
        confidence_score: 0.87
      };
      
      setData(mockData);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchDashboardData();
    setRefreshing(false);
  };

  const toggleSection = (sectionId: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId);
    } else {
      newExpanded.add(sectionId);
    }
    setExpandedSections(newExpanded);
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'down': return <TrendingDown className="h-4 w-4 text-red-600" />;
      default: return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'financial': return <DollarSign className="h-5 w-5" />;
      case 'engagement': return <Users className="h-5 w-5" />;
      case 'performance': return <Activity className="h-5 w-5" />;
      default: return <BarChart3 className="h-5 w-5" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-4">
        <div className="max-w-md mx-auto space-y-4">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded mb-4"></div>
            <div className="space-y-3">
              <div className="h-20 bg-gray-200 rounded"></div>
              <div className="h-20 bg-gray-200 rounded"></div>
              <div className="h-20 bg-gray-200 rounded"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-gray-50 p-4 flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Dashboard</h2>
          <p className="text-gray-600 mb-4">Unable to fetch dashboard data</p>
          <Button onClick={fetchDashboardData}>Try Again</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 sticky top-0 z-10">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Executive Dashboard</h1>
            <p className="text-sm text-gray-500">
              Updated {new Date(data.last_updated).toLocaleTimeString()}
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
              <Share className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Confidence Score */}
        <div className="mt-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Analysis Confidence</span>
            <span className="font-medium">{Math.round(data.confidence_score * 100)}%</span>
          </div>
          <Progress value={data.confidence_score * 100} className="h-2 mt-1" />
        </div>
      </div>

      {/* Risk Alerts */}
      {data.risk_alerts.length > 0 && (
        <div className="px-4 py-3 bg-red-50 border-b border-red-200">
          <div className="flex items-start space-x-2">
            <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-sm font-medium text-red-800 mb-1">Risk Alerts</h3>
              {data.risk_alerts.map((alert, index) => (
                <p key={index} className="text-sm text-red-700">{alert}</p>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="px-4 py-4">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-4">
            <TabsTrigger value="overview" className="text-xs">Overview</TabsTrigger>
            <TabsTrigger value="metrics" className="text-xs">Metrics</TabsTrigger>
            <TabsTrigger value="findings" className="text-xs">Findings</TabsTrigger>
            <TabsTrigger value="actions" className="text-xs">Actions</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            {/* Key Metrics Summary */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Key Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {data.metrics.slice(0, 4).map((metric) => (
                  <div key={metric.id} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {getCategoryIcon(metric.category)}
                      <span className="text-sm font-medium">{metric.title}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-semibold">{metric.value}</span>
                      <div className="flex items-center space-x-1">
                        {getTrendIcon(metric.trend)}
                        <span className={`text-xs ${
                          metric.trend === 'up' ? 'text-green-600' : 
                          metric.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                        }`}>
                          {metric.change}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Top Findings */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Top Findings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {data.key_findings.slice(0, 3).map((finding) => (
                  <div key={finding.id} className="border-l-4 border-blue-500 pl-3">
                    <div className="flex items-start justify-between">
                      <h4 className="text-sm font-medium text-gray-900">{finding.title}</h4>
                      <Badge className={`text-xs ${getPriorityColor(finding.priority)}`}>
                        {finding.priority}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">{finding.description}</p>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Top Recommendations */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Priority Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {data.recommendations.slice(0, 3).map((rec) => (
                  <div key={rec.id} className="flex items-start space-x-3">
                    <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <h4 className="text-sm font-medium text-gray-900">{rec.title}</h4>
                      <p className="text-xs text-gray-600 mt-1">{rec.description}</p>
                      <div className="flex items-center space-x-2 mt-2">
                        <Badge variant="outline" className="text-xs">{rec.timeline}</Badge>
                        <Badge className={`text-xs ${getPriorityColor(rec.priority)}`}>
                          {rec.priority}
                        </Badge>
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="space-y-4">
            {data.metrics.map((metric) => (
              <Card key={metric.id}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {getCategoryIcon(metric.category)}
                      <h3 className="text-sm font-medium">{metric.title}</h3>
                    </div>
                    <Badge className={`text-xs ${getPriorityColor(metric.priority)}`}>
                      {metric.priority}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold">{metric.value}</span>
                    <div className="flex items-center space-x-1">
                      {getTrendIcon(metric.trend)}
                      <span className={`text-sm font-medium ${
                        metric.trend === 'up' ? 'text-green-600' : 
                        metric.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {metric.change}
                      </span>
                    </div>
                  </div>
                  <div className="mt-2">
                    <Badge variant="outline" className="text-xs">
                      {metric.category}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          {/* Findings Tab */}
          <TabsContent value="findings" className="space-y-4">
            {data.key_findings.map((finding) => (
              <Card key={finding.id}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="text-sm font-medium text-gray-900 flex-1">{finding.title}</h3>
                    <Badge className={`text-xs ml-2 ${getPriorityColor(finding.priority)}`}>
                      {finding.priority}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{finding.description}</p>
                  
                  <div 
                    className="cursor-pointer"
                    onClick={() => toggleSection(`finding-${finding.id}`)}
                  >
                    <div className="flex items-center justify-between text-sm text-blue-600">
                      <span>View Impact</span>
                      {expandedSections.has(`finding-${finding.id}`) ? 
                        <ChevronDown className="h-4 w-4" /> : 
                        <ChevronRight className="h-4 w-4" />
                      }
                    </div>
                  </div>
                  
                  {expandedSections.has(`finding-${finding.id}`) && (
                    <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                      <h4 className="text-xs font-medium text-gray-700 mb-1">Business Impact</h4>
                      <p className="text-xs text-gray-600">{finding.impact}</p>
                    </div>
                  )}
                  
                  <div className="mt-3">
                    <Badge variant="outline" className="text-xs">
                      {finding.category}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          {/* Actions Tab */}
          <TabsContent value="actions" className="space-y-4">
            {data.recommendations.map((rec) => (
              <Card key={rec.id}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="text-sm font-medium text-gray-900 flex-1">{rec.title}</h3>
                    <Badge className={`text-xs ml-2 ${getPriorityColor(rec.priority)}`}>
                      {rec.priority}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{rec.description}</p>
                  
                  <div className="flex items-center space-x-2 mb-3">
                    <Badge variant="outline" className="text-xs">
                      <Calendar className="h-3 w-3 mr-1" />
                      {rec.timeline}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      Effort: {rec.implementation_effort}
                    </Badge>
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button size="sm" className="flex-1 text-xs">
                      Start Implementation
                    </Button>
                    <Button variant="outline" size="sm" className="text-xs">
                      More Details
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>
        </Tabs>
      </div>

      {/* Bottom Actions */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4">
        <div className="flex space-x-2">
          <Button variant="outline" className="flex-1 text-sm">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
          <Button className="flex-1 text-sm">
            <BarChart3 className="h-4 w-4 mr-2" />
            Full Dashboard
          </Button>
        </div>
      </div>

      {/* Bottom padding to account for fixed actions */}
      <div className="h-20"></div>
    </div>
  );
};

export default MobileExecutiveDashboard;