"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Activity, 
  TrendingUp, 
  Clock, 
  AlertTriangle,
  CheckCircle,
  BarChart3,
  PieChart,
  Download,
  Calendar
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RechartsPieChart,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

interface UsageAnalyticsDashboardProps {
  selectedKeyId?: string | null;
}

interface UsageOverview {
  total_api_keys: number;
  active_api_keys: number;
  total_requests: number;
  successful_requests: number;
  error_rate: number;
  average_response_time: number;
  data_transfer_gb: number;
  top_endpoints: Array<{
    endpoint: string;
    count: number;
    avg_response_time: number;
  }>;
  daily_usage: Array<{
    date: string;
    requests: number;
    successful: number;
    error_rate: number;
    avg_response_time: number;
  }>;
  error_breakdown: Array<{
    status_code: number;
    count: number;
    percentage: number;
  }>;
}

interface QuotaStatus {
  user_id: string;
  current_period: string;
  api_keys: Array<{
    api_key_id: string;
    api_key_name: string;
    period_start: string;
    period_end: string;
    requests: {
      used: number;
      limit: number | null;
      percentage: number;
    };
    data_transfer: {
      used_bytes: number;
      limit_bytes: number | null;
      percentage: number;
    };
    compute_time: {
      used_seconds: number;
      limit_seconds: number | null;
      percentage: number;
    };
    cost: {
      used_usd: number;
      limit_usd: number | null;
      percentage: number;
    };
    is_exceeded: boolean;
    exceeded_at: string | null;
  }>;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export function UsageAnalyticsDashboard({ selectedKeyId }: UsageAnalyticsDashboardProps) {
  const [overview, setOverview] = useState<UsageOverview | null>(null);
  const [quotaStatus, setQuotaStatus] = useState<QuotaStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState('30'); // days

  useEffect(() => {
    fetchAnalytics();
    fetchQuotaStatus();
  }, [selectedKeyId, dateRange]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      let url = '/api/v1/analytics/overview';
      if (selectedKeyId) {
        url = `/api/v1/keys/${selectedKeyId}/usage`;
      }

      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch analytics');
      }

      const data = await response.json();
      setOverview(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const fetchQuotaStatus = async () => {
    try {
      const response = await fetch('/api/v1/analytics/quota', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setQuotaStatus(data);
      }
    } catch (err) {
      console.error('Failed to fetch quota status:', err);
    }
  };

  const exportData = async () => {
    try {
      const response = await fetch('/api/v1/analytics/export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify({
          format: 'csv',
          date_range_days: parseInt(dateRange)
        }),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `api-usage-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error('Failed to export data:', err);
    }
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  const formatBytes = (bytes: number) => {
    if (bytes >= 1024 ** 3) {
      return (bytes / (1024 ** 3)).toFixed(2) + ' GB';
    }
    if (bytes >= 1024 ** 2) {
      return (bytes / (1024 ** 2)).toFixed(2) + ' MB';
    }
    if (bytes >= 1024) {
      return (bytes / 1024).toFixed(2) + ' KB';
    }
    return bytes + ' B';
  };

  const getQuotaColor = (percentage: number) => {
    if (percentage >= 90) return 'text-red-600';
    if (percentage >= 75) return 'text-yellow-600';
    return 'text-green-600';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!overview) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Activity className="h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No Usage Data
          </h3>
          <p className="text-gray-600 text-center">
            Start using your API keys to see analytics here
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-medium">Usage Analytics</h3>
          <p className="text-gray-600">
            {selectedKeyId ? 'API key specific analytics' : 'Overview of all your API keys'}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
          </select>
          <Button variant="outline" size="sm" onClick={exportData}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600">Total Requests</p>
                <p className="text-2xl font-bold">{formatNumber(overview.total_requests)}</p>
              </div>
              <Activity className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold">
                  {((overview.successful_requests / Math.max(overview.total_requests, 1)) * 100).toFixed(1)}%
                </p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600">Avg Response Time</p>
                <p className="text-2xl font-bold">{overview.average_response_time.toFixed(0)}ms</p>
              </div>
              <Clock className="h-8 w-8 text-yellow-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600">Data Transfer</p>
                <p className="text-2xl font-bold">{overview.data_transfer_gb.toFixed(2)} GB</p>
              </div>
              <TrendingUp className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="usage" className="space-y-4">
        <TabsList>
          <TabsTrigger value="usage">Usage Trends</TabsTrigger>
          <TabsTrigger value="endpoints">Top Endpoints</TabsTrigger>
          <TabsTrigger value="errors">Error Analysis</TabsTrigger>
          <TabsTrigger value="quotas">Quotas & Limits</TabsTrigger>
        </TabsList>

        <TabsContent value="usage" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Daily Usage Trend</CardTitle>
              <CardDescription>
                Request volume and response times over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={overview.daily_usage}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="requests" fill="#8884d8" name="Requests" />
                  <Line 
                    yAxisId="right" 
                    type="monotone" 
                    dataKey="avg_response_time" 
                    stroke="#82ca9d" 
                    name="Avg Response Time (ms)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="endpoints" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Top Endpoints</CardTitle>
              <CardDescription>
                Most frequently accessed API endpoints
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {overview.top_endpoints.map((endpoint, index) => (
                  <div key={endpoint.endpoint} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex-1">
                      <code className="text-sm font-mono">{endpoint.endpoint}</code>
                      <p className="text-xs text-gray-600 mt-1">
                        Avg response time: {endpoint.avg_response_time.toFixed(0)}ms
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{formatNumber(endpoint.count)}</p>
                      <p className="text-xs text-gray-600">requests</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="errors" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Error Rate Trend</CardTitle>
                <CardDescription>
                  Error rate over time
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={overview.daily_usage}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="error_rate" 
                      stroke="#ff7300" 
                      fill="#ff7300" 
                      fillOpacity={0.3}
                      name="Error Rate (%)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Error Breakdown</CardTitle>
                <CardDescription>
                  Distribution of HTTP error codes
                </CardDescription>
              </CardHeader>
              <CardContent>
                {overview.error_breakdown.length > 0 ? (
                  <ResponsiveContainer width="100%" height={250}>
                    <RechartsPieChart>
                      <Pie
                        data={overview.error_breakdown}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ status_code, percentage }) => `${status_code} (${percentage.toFixed(1)}%)`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="count"
                      >
                        {overview.error_breakdown.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-64 text-gray-500">
                    <div className="text-center">
                      <CheckCircle className="h-12 w-12 mx-auto mb-2" />
                      <p>No errors in this period</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="quotas" className="space-y-4">
          {quotaStatus && (
            <div className="grid gap-4">
              {quotaStatus.api_keys.map((keyQuota) => (
                <Card key={keyQuota.api_key_id}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <span>{keyQuota.api_key_name}</span>
                      {keyQuota.is_exceeded && (
                        <Badge variant="destructive">Quota Exceeded</Badge>
                      )}
                    </CardTitle>
                    <CardDescription>
                      Period: {new Date(keyQuota.period_start).toLocaleDateString()} - 
                      {keyQuota.period_end ? new Date(keyQuota.period_end).toLocaleDateString() : 'Ongoing'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Requests</p>
                        <p className={`text-lg font-bold ${getQuotaColor(keyQuota.requests.percentage)}`}>
                          {formatNumber(keyQuota.requests.used)}
                          {keyQuota.requests.limit && ` / ${formatNumber(keyQuota.requests.limit)}`}
                        </p>
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${Math.min(keyQuota.requests.percentage, 100)}%` }}
                          ></div>
                        </div>
                      </div>

                      <div>
                        <p className="text-sm font-medium text-gray-600">Data Transfer</p>
                        <p className={`text-lg font-bold ${getQuotaColor(keyQuota.data_transfer.percentage)}`}>
                          {formatBytes(keyQuota.data_transfer.used_bytes)}
                          {keyQuota.data_transfer.limit_bytes && ` / ${formatBytes(keyQuota.data_transfer.limit_bytes)}`}
                        </p>
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                          <div
                            className="bg-green-600 h-2 rounded-full"
                            style={{ width: `${Math.min(keyQuota.data_transfer.percentage, 100)}%` }}
                          ></div>
                        </div>
                      </div>

                      <div>
                        <p className="text-sm font-medium text-gray-600">Compute Time</p>
                        <p className={`text-lg font-bold ${getQuotaColor(keyQuota.compute_time.percentage)}`}>
                          {keyQuota.compute_time.used_seconds.toFixed(1)}s
                          {keyQuota.compute_time.limit_seconds && ` / ${keyQuota.compute_time.limit_seconds.toFixed(1)}s`}
                        </p>
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                          <div
                            className="bg-yellow-600 h-2 rounded-full"
                            style={{ width: `${Math.min(keyQuota.compute_time.percentage, 100)}%` }}
                          ></div>
                        </div>
                      </div>

                      <div>
                        <p className="text-sm font-medium text-gray-600">Cost</p>
                        <p className={`text-lg font-bold ${getQuotaColor(keyQuota.cost.percentage)}`}>
                          ${keyQuota.cost.used_usd.toFixed(2)}
                          {keyQuota.cost.limit_usd && ` / $${keyQuota.cost.limit_usd.toFixed(2)}`}
                        </p>
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                          <div
                            className="bg-purple-600 h-2 rounded-full"
                            style={{ width: `${Math.min(keyQuota.cost.percentage, 100)}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}