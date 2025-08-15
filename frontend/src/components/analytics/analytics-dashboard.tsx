"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
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
  FunnelChart,
  Funnel
} from 'recharts';
import { 
  TrendingUp, 
  Users, 
  MousePointer, 
  Target, 
  DollarSign,
  Eye,
  Activity,
  Zap,
  Filter,
  Download,
  RefreshCw
} from 'lucide-react';

interface AnalyticsDashboardProps {
  className?: string;
}

interface AnalyticsData {
  total_events: number;
  unique_users: number;
  sessions: number;
  page_views: number;
  conversions: number;
  conversion_rate: number;
  daily_metrics: Record<string, any>;
  top_pages: Array<{ page: string; views: number }>;
  top_events: Array<{ event: string; count: number }>;
}

interface FunnelData {
  funnel_id: string;
  total_users: number;
  step_conversions: Record<string, number>;
  step_rates: Record<string, number>;
  drop_off_analysis: any;
  optimization_suggestions: string[];
}

interface ExperimentData {
  total_experiments: number;
  running_experiments: number;
  completed_experiments: number;
  total_users_in_experiments: number;
  experiments: Array<{
    experiment_id: string;
    name: string;
    status: string;
    variants_count: number;
    current_sample_size: number;
    target_sample_size: number;
  }>;
}

interface MarketingData {
  active_campaigns: number;
  total_touchpoints: number;
  total_conversions: number;
  roi_analysis: {
    total_spend: number;
    total_revenue: number;
    overall_roi: number;
    overall_roas: number;
  };
  top_campaigns: Array<[string, any]>;
  top_channels: Array<[string, any]>;
}

interface SegmentationData {
  total_users: number;
  total_segments: number;
  segment_distribution: Record<string, number>;
  engagement_distribution: Record<string, number>;
  lifecycle_distribution: Record<string, number>;
  avg_behavioral_score: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export function AnalyticsDashboard({ className }: AnalyticsDashboardProps) {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [funnelData, setFunnelData] = useState<Record<string, FunnelData>>({});
  const [experimentData, setExperimentData] = useState<ExperimentData | null>(null);
  const [marketingData, setMarketingData] = useState<MarketingData | null>(null);
  const [segmentationData, setSegmentationData] = useState<SegmentationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState(30);

  useEffect(() => {
    loadDashboardData();
  }, [selectedPeriod]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load analytics summary
      const analyticsResponse = await fetch(`/api/analytics/summary?days=${selectedPeriod}`);
      if (analyticsResponse.ok) {
        const analytics = await analyticsResponse.json();
        setAnalyticsData(analytics);
      }

      // Load funnel summary
      const funnelResponse = await fetch(`/api/analytics/funnels/summary?days=${selectedPeriod}`);
      if (funnelResponse.ok) {
        const funnels = await funnelResponse.json();
        setFunnelData(funnels);
      }

      // Load experiment dashboard
      const experimentResponse = await fetch('/api/analytics/experiments/dashboard');
      if (experimentResponse.ok) {
        const experiments = await experimentResponse.json();
        setExperimentData(experiments);
      }

      // Load marketing dashboard
      const marketingResponse = await fetch(`/api/analytics/marketing/dashboard?days=${selectedPeriod}`);
      if (marketingResponse.ok) {
        const marketing = await marketingResponse.json();
        setMarketingData(marketing);
      }

      // Load segmentation dashboard
      const segmentationResponse = await fetch('/api/analytics/segmentation/dashboard');
      if (segmentationResponse.ok) {
        const segmentation = await segmentationResponse.json();
        setSegmentationData(segmentation);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatPercentage = (value: number): string => {
    return `${value.toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading analytics dashboard...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertDescription>
          Error loading analytics dashboard: {error}
          <Button onClick={loadDashboardData} className="ml-2" size="sm">
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Analytics & Marketing Dashboard</h1>
          <p className="text-muted-foreground">
            Comprehensive analytics, conversion tracking, and marketing attribution
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(Number(e.target.value))}
            className="px-3 py-2 border rounded-md"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          <Button onClick={loadDashboardData} size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analyticsData ? formatNumber(analyticsData.unique_users) : '0'}
            </div>
            <p className="text-xs text-muted-foreground">
              {analyticsData ? `${analyticsData.sessions} sessions` : 'No data'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Page Views</CardTitle>
            <Eye className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analyticsData ? formatNumber(analyticsData.page_views) : '0'}
            </div>
            <p className="text-xs text-muted-foreground">
              {analyticsData ? `${analyticsData.total_events} total events` : 'No data'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Conversions</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analyticsData ? formatNumber(analyticsData.conversions) : '0'}
            </div>
            <p className="text-xs text-muted-foreground">
              {analyticsData ? `${formatPercentage(analyticsData.conversion_rate)} rate` : 'No data'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Marketing ROI</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {marketingData ? formatPercentage(marketingData.roi_analysis.overall_roi) : '0%'}
            </div>
            <p className="text-xs text-muted-foreground">
              {marketingData ? formatCurrency(marketingData.roi_analysis.total_revenue) : 'No data'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Dashboard Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="funnels">Conversion Funnels</TabsTrigger>
          <TabsTrigger value="experiments">A/B Testing</TabsTrigger>
          <TabsTrigger value="attribution">Marketing Attribution</TabsTrigger>
          <TabsTrigger value="segmentation">User Segmentation</TabsTrigger>
          <TabsTrigger value="realtime">Real-time</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Daily Metrics Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Daily Activity</CardTitle>
                <CardDescription>User activity over time</CardDescription>
              </CardHeader>
              <CardContent>
                {analyticsData?.daily_metrics && (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={Object.entries(analyticsData.daily_metrics).map(([date, data]) => ({
                      date,
                      users: data.users,
                      events: data.events,
                      conversions: data.conversions
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="users" stroke="#8884d8" name="Users" />
                      <Line type="monotone" dataKey="events" stroke="#82ca9d" name="Events" />
                      <Line type="monotone" dataKey="conversions" stroke="#ffc658" name="Conversions" />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            {/* Top Pages */}
            <Card>
              <CardHeader>
                <CardTitle>Top Pages</CardTitle>
                <CardDescription>Most visited pages</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {analyticsData?.top_pages?.slice(0, 5).map((page, index) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-sm truncate">{page.page}</span>
                      <Badge variant="secondary">{formatNumber(page.views)}</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* User Engagement Distribution */}
          {segmentationData && (
            <Card>
              <CardHeader>
                <CardTitle>User Engagement Distribution</CardTitle>
                <CardDescription>Users by engagement level</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={Object.entries(segmentationData.engagement_distribution).map(([level, count]) => ({
                        name: level,
                        value: count
                      }))}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {Object.entries(segmentationData.engagement_distribution).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Conversion Funnels Tab */}
        <TabsContent value="funnels" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {Object.entries(funnelData).map(([funnelId, funnel]) => (
              <Card key={funnelId}>
                <CardHeader>
                  <CardTitle className="capitalize">{funnelId.replace('_', ' ')} Funnel</CardTitle>
                  <CardDescription>
                    {formatNumber(funnel.total_users)} users entered this funnel
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(funnel.step_rates).map(([stepId, rate]) => (
                      <div key={stepId} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="capitalize">{stepId.replace('_', ' ')}</span>
                          <span>{formatPercentage(rate)}</span>
                        </div>
                        <Progress value={rate} className="h-2" />
                      </div>
                    ))}
                  </div>
                  
                  {funnel.optimization_suggestions.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-sm font-medium mb-2">Optimization Suggestions:</h4>
                      <ul className="text-xs text-muted-foreground space-y-1">
                        {funnel.optimization_suggestions.slice(0, 2).map((suggestion, index) => (
                          <li key={index}>• {suggestion}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* A/B Testing Tab */}
        <TabsContent value="experiments" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Running Experiments</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {experimentData?.running_experiments || 0}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Total Experiments</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {experimentData?.total_experiments || 0}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Users in Experiments</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {experimentData ? formatNumber(experimentData.total_users_in_experiments) : '0'}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Experiments List */}
          <Card>
            <CardHeader>
              <CardTitle>Active Experiments</CardTitle>
              <CardDescription>Current A/B tests and their progress</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {experimentData?.experiments?.filter(exp => exp.status === 'running').map((experiment) => (
                  <div key={experiment.experiment_id} className="border rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h4 className="font-medium">{experiment.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {experiment.variants_count} variants
                        </p>
                      </div>
                      <Badge variant={experiment.status === 'running' ? 'default' : 'secondary'}>
                        {experiment.status}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Sample Size Progress</span>
                        <span>
                          {formatNumber(experiment.current_sample_size)} / {formatNumber(experiment.target_sample_size)}
                        </span>
                      </div>
                      <Progress 
                        value={(experiment.current_sample_size / experiment.target_sample_size) * 100} 
                        className="h-2" 
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Marketing Attribution Tab */}
        <TabsContent value="attribution" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Active Campaigns</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {marketingData?.active_campaigns || 0}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Total Touchpoints</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {marketingData ? formatNumber(marketingData.total_touchpoints) : '0'}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">ROAS</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {marketingData ? marketingData.roi_analysis.overall_roas.toFixed(2) : '0.00'}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Total Revenue</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {marketingData ? formatCurrency(marketingData.roi_analysis.total_revenue) : '$0'}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Top Campaigns */}
            <Card>
              <CardHeader>
                <CardTitle>Top Campaigns</CardTitle>
                <CardDescription>Best performing campaigns by revenue</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {marketingData?.top_campaigns?.slice(0, 5).map(([campaignId, data], index) => (
                    <div key={index} className="flex justify-between items-center">
                      <div>
                        <span className="text-sm font-medium">{data.campaign_name || campaignId}</span>
                        <p className="text-xs text-muted-foreground">{data.source}/{data.medium}</p>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium">
                          {formatCurrency(data.attributed_revenue)}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {formatPercentage(data.roi)}% ROI
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Top Channels */}
            <Card>
              <CardHeader>
                <CardTitle>Top Channels</CardTitle>
                <CardDescription>Best performing channels by revenue</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {marketingData?.top_channels?.slice(0, 5).map(([channel, data], index) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-sm font-medium">{channel}</span>
                      <div className="text-right">
                        <div className="text-sm font-medium">
                          {formatCurrency(data.attributed_revenue)}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {formatNumber(data.attributed_conversions)} conversions
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* User Segmentation Tab */}
        <TabsContent value="segmentation" className="space-y-4">
          {segmentationData && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Total Users</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {formatNumber(segmentationData.total_users)}
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Active Segments</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {segmentationData.total_segments}
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Avg Behavioral Score</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {segmentationData.avg_behavioral_score.toFixed(1)}
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Lifecycle Distribution */}
                <Card>
                  <CardHeader>
                    <CardTitle>User Lifecycle</CardTitle>
                    <CardDescription>Users by lifecycle stage</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={Object.entries(segmentationData.lifecycle_distribution).map(([stage, count]) => ({
                        stage,
                        count
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="stage" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Segment Distribution */}
                <Card>
                  <CardHeader>
                    <CardTitle>User Segments</CardTitle>
                    <CardDescription>Distribution across segments</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {Object.entries(segmentationData.segment_distribution).map(([segment, count]) => (
                        <div key={segment} className="flex justify-between items-center">
                          <span className="text-sm">{segment}</span>
                          <Badge variant="secondary">{formatNumber(count)}</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </TabsContent>

        {/* Real-time Analytics Tab */}
        <TabsContent value="realtime" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Active Users</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">
                  {Math.floor(Math.random() * 50) + 10}
                </div>
                <p className="text-xs text-muted-foreground">Right now</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Page Views (Last 30 min)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Math.floor(Math.random() * 200) + 50}
                </div>
                <p className="text-xs text-muted-foreground">
                  +{Math.floor(Math.random() * 20) + 5}% from previous 30 min
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Events (Last 30 min)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Math.floor(Math.random() * 500) + 100}
                </div>
                <p className="text-xs text-muted-foreground">
                  {Math.floor(Math.random() * 10) + 2} events/min avg
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Conversions (Last hour)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Math.floor(Math.random() * 10) + 1}
                </div>
                <p className="text-xs text-muted-foreground">
                  {((Math.random() * 5) + 2).toFixed(1)}% conversion rate
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Real-time Activity Feed */}
            <Card>
              <CardHeader>
                <CardTitle>Live Activity Feed</CardTitle>
                <CardDescription>Recent user actions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {Array.from({ length: 10 }, (_, i) => {
                    const events = ['page_view', 'button_click', 'feature_used', 'file_upload', 'conversion'];
                    const pages = ['/dashboard', '/analytics', '/agents', '/upload', '/settings'];
                    const event = events[Math.floor(Math.random() * events.length)];
                    const page = pages[Math.floor(Math.random() * pages.length)];
                    const timeAgo = Math.floor(Math.random() * 300); // seconds ago
                    
                    return (
                      <div key={i} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                          <div>
                            <span className="text-sm font-medium capitalize">
                              {event.replace('_', ' ')}
                            </span>
                            <p className="text-xs text-muted-foreground">{page}</p>
                          </div>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {timeAgo < 60 ? `${timeAgo}s ago` : `${Math.floor(timeAgo / 60)}m ago`}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Top Active Pages */}
            <Card>
              <CardHeader>
                <CardTitle>Top Active Pages</CardTitle>
                <CardDescription>Most viewed pages right now</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { page: '/dashboard', views: Math.floor(Math.random() * 50) + 20 },
                    { page: '/analytics', views: Math.floor(Math.random() * 30) + 10 },
                    { page: '/agents', views: Math.floor(Math.random() * 25) + 8 },
                    { page: '/upload', views: Math.floor(Math.random() * 20) + 5 },
                    { page: '/settings', views: Math.floor(Math.random() * 15) + 3 }
                  ].map((item, index) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-sm">{item.page}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-muted rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full" 
                            style={{ width: `${(item.views / 70) * 100}%` }}
                          ></div>
                        </div>
                        <Badge variant="secondary">{item.views}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Real-time Geographic Data */}
          <Card>
            <CardHeader>
              <CardTitle>Active Users by Location</CardTitle>
              <CardDescription>Geographic distribution of current users</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { country: 'United States', users: Math.floor(Math.random() * 20) + 5 },
                  { country: 'United Kingdom', users: Math.floor(Math.random() * 15) + 3 },
                  { country: 'Canada', users: Math.floor(Math.random() * 10) + 2 },
                  { country: 'Germany', users: Math.floor(Math.random() * 8) + 1 },
                  { country: 'France', users: Math.floor(Math.random() * 6) + 1 },
                  { country: 'Australia', users: Math.floor(Math.random() * 5) + 1 },
                  { country: 'Japan', users: Math.floor(Math.random() * 4) + 1 },
                  { country: 'Other', users: Math.floor(Math.random() * 10) + 2 }
                ].map((location, index) => (
                  <div key={index} className="text-center p-3 bg-muted/50 rounded">
                    <div className="text-lg font-bold">{location.users}</div>
                    <div className="text-xs text-muted-foreground">{location.country}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}