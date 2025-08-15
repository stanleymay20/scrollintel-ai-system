"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Activity, 
  Database, 
  Zap, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  BarChart3,
  RefreshCw
} from 'lucide-react';

interface PerformanceMetrics {
  timestamp: string;
  system: {
    cpu_percent: number;
    memory_percent: number;
    memory_available_gb: number;
  };
  response_times: {
    avg_response_time: number;
    p95_response_time: number;
    slow_endpoints_count: number;
  };
  database: {
    avg_query_time: number;
    slow_queries_count: number;
    total_queries: number;
  };
  cache: {
    hit_rate: number;
    total_requests: number;
    local_cache_size: number;
  };
  endpoints: Record<string, {
    avg_response_time: number;
    request_count: number;
    p95_response_time: number;
  }>;
}

interface PerformanceRecommendation {
  endpoint_optimizations: Array<{
    endpoint: string;
    method: string;
    avg_response_time: number;
    suggestions: string[];
  }>;
  database_optimizations: Array<{
    query_hash: string;
    avg_execution_time: number;
    suggestions: string[];
  }>;
  cache_recommendations: Array<{
    type: string;
    message: string;
    priority: string;
  }>;
  system_recommendations: Array<{
    type: string;
    message: string;
    priority: string;
  }>;
}

const PerformanceDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [recommendations, setRecommendations] = useState<PerformanceRecommendation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/performance/dashboard');
      if (!response.ok) throw new Error('Failed to fetch metrics');
      const data = await response.json();
      setMetrics(data.data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  const fetchRecommendations = async () => {
    try {
      const response = await fetch('/api/performance/recommendations');
      if (!response.ok) throw new Error('Failed to fetch recommendations');
      const data = await response.json();
      setRecommendations(data.data);
    } catch (err) {
      console.error('Failed to fetch recommendations:', err);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchMetrics(), fetchRecommendations()]);
      setLoading(false);
    };

    loadData();

    // Auto-refresh every 30 seconds
    const interval = autoRefresh ? setInterval(fetchMetrics, 30000) : null;
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const getHealthStatus = (metrics: PerformanceMetrics) => {
    const issues = [];
    
    if (metrics.system.cpu_percent > 80) issues.push('High CPU');
    if (metrics.system.memory_percent > 85) issues.push('High Memory');
    if (metrics.response_times.avg_response_time > 2) issues.push('Slow Response');
    if (metrics.database.avg_query_time > 1) issues.push('Slow Queries');
    if (metrics.cache.hit_rate < 70) issues.push('Low Cache Hit Rate');

    if (issues.length === 0) return { status: 'healthy', color: 'green', issues: [] };
    if (issues.length <= 2) return { status: 'warning', color: 'yellow', issues };
    return { status: 'critical', color: 'red', issues };
  };

  const formatTime = (seconds: number) => {
    if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
    return `${seconds.toFixed(2)}s`;
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading performance metrics...</span>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Failed to load performance metrics: {error}
        </AlertDescription>
      </Alert>
    );
  }

  const health = getHealthStatus(metrics);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Performance Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time system performance monitoring and optimization
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh: {autoRefresh ? 'On' : 'Off'}
          </Button>
          <Button onClick={fetchMetrics} size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Health Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            {health.status === 'healthy' && <CheckCircle className="h-5 w-5 text-green-500 mr-2" />}
            {health.status === 'warning' && <AlertTriangle className="h-5 w-5 text-yellow-500 mr-2" />}
            {health.status === 'critical' && <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />}
            System Health
            <Badge 
              variant={health.status === 'healthy' ? 'default' : 'destructive'}
              className="ml-2"
            >
              {health.status.toUpperCase()}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {health.issues.length > 0 ? (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Issues detected:</p>
              <div className="flex flex-wrap gap-2">
                {health.issues.map((issue, index) => (
                  <Badge key={index} variant="outline">
                    {issue}
                  </Badge>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-sm text-green-600">All systems operating normally</p>
          )}
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatTime(metrics.response_times.avg_response_time)}
            </div>
            <p className="text-xs text-muted-foreground">
              P95: {formatTime(metrics.response_times.p95_response_time)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.system.cpu_percent.toFixed(1)}%</div>
            <Progress value={metrics.system.cpu_percent} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.system.memory_percent.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              {formatBytes(metrics.system.memory_available_gb * 1024**3)} available
            </p>
            <Progress value={metrics.system.memory_percent} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.cache.hit_rate.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              {metrics.cache.total_requests} total requests
            </p>
            <Progress value={metrics.cache.hit_rate} className="mt-2" />
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics */}
      <Tabs defaultValue="endpoints" className="space-y-4">
        <TabsList>
          <TabsTrigger value="endpoints">Endpoints</TabsTrigger>
          <TabsTrigger value="database">Database</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
        </TabsList>

        <TabsContent value="endpoints" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Endpoint Performance</CardTitle>
              <CardDescription>
                Response time statistics for API endpoints
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(metrics.endpoints).map(([endpoint, stats]) => (
                  <div key={endpoint} className="flex items-center justify-between p-3 border rounded">
                    <div>
                      <p className="font-medium">{endpoint}</p>
                      <p className="text-sm text-muted-foreground">
                        {stats.request_count} requests
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{formatTime(stats.avg_response_time)}</p>
                      <p className="text-sm text-muted-foreground">
                        P95: {formatTime(stats.p95_response_time)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="database" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Database className="h-5 w-5 mr-2" />
                Database Performance
              </CardTitle>
              <CardDescription>
                Query execution statistics and optimization opportunities
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold">{formatTime(metrics.database.avg_query_time)}</p>
                  <p className="text-sm text-muted-foreground">Avg Query Time</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold">{metrics.database.slow_queries_count}</p>
                  <p className="text-sm text-muted-foreground">Slow Queries</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold">{metrics.database.total_queries}</p>
                  <p className="text-sm text-muted-foreground">Total Queries</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-4">
          {recommendations && (
            <>
              {/* Endpoint Optimizations */}
              {recommendations.endpoint_optimizations.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <TrendingUp className="h-5 w-5 mr-2" />
                      Endpoint Optimizations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {recommendations.endpoint_optimizations.slice(0, 5).map((opt, index) => (
                        <div key={index} className="border rounded p-3">
                          <div className="flex items-center justify-between mb-2">
                            <p className="font-medium">{opt.method} {opt.endpoint}</p>
                            <Badge variant="outline">
                              {formatTime(opt.avg_response_time)}
                            </Badge>
                          </div>
                          <div className="space-y-1">
                            {opt.suggestions.map((suggestion, i) => (
                              <p key={i} className="text-sm text-muted-foreground">
                                • {suggestion}
                              </p>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* System Recommendations */}
              {(recommendations.cache_recommendations.length > 0 || 
                recommendations.system_recommendations.length > 0) && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <AlertTriangle className="h-5 w-5 mr-2" />
                      System Recommendations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {[...recommendations.cache_recommendations, ...recommendations.system_recommendations]
                        .map((rec, index) => (
                        <Alert key={index}>
                          <AlertTriangle className="h-4 w-4" />
                          <AlertDescription className="flex items-center justify-between">
                            <span>{rec.message}</span>
                            <Badge variant={rec.priority === 'high' ? 'destructive' : 'secondary'}>
                              {rec.priority}
                            </Badge>
                          </AlertDescription>
                        </Alert>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </TabsContent>
      </Tabs>

      {/* Footer */}
      <div className="text-center text-sm text-muted-foreground">
        Last updated: {new Date(metrics.timestamp).toLocaleString()}
      </div>
    </div>
  );
};

export default PerformanceDashboard;