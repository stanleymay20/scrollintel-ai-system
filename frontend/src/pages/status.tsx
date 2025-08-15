import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';

interface ServiceStatus {
  status: string;
  uptime_percentage: number;
  response_time: number;
  last_check: string;
}

interface Incident {
  id: string;
  service: string;
  title: string;
  description: string;
  started_at: string;
  resolved_at?: string;
  impact: string;
  status: string;
}

interface StatusData {
  overall_status: string;
  last_updated: string;
  services: Record<string, ServiceStatus>;
  incidents: Incident[];
  uptime_stats: {
    '24h': number;
    '7d': number;
    '30d': number;
  };
}

const StatusPage: React.FC = () => {
  const [statusData, setStatusData] = useState<StatusData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatusData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchStatusData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatusData = async () => {
    try {
      const response = await fetch('/api/status');
      if (!response.ok) {
        throw new Error('Failed to fetch status data');
      }
      const data = await response.json();
      setStatusData(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'partial_outage':
        return 'bg-orange-500';
      case 'major_outage':
        return 'bg-red-500';
      case 'maintenance':
        return 'bg-blue-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'operational':
        return 'Operational';
      case 'degraded':
        return 'Degraded Performance';
      case 'partial_outage':
        return 'Partial Outage';
      case 'major_outage':
        return 'Major Outage';
      case 'maintenance':
        return 'Maintenance';
      default:
        return 'Unknown';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'minor':
        return 'bg-yellow-100 text-yellow-800';
      case 'major':
        return 'bg-orange-100 text-orange-800';
      case 'critical':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const calculateDuration = (startDate: string, endDate?: string) => {
    const start = new Date(startDate);
    const end = endDate ? new Date(endDate) : new Date();
    const diffMs = end.getTime() - start.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 60) {
      return `${diffMins} minutes`;
    } else {
      const hours = Math.floor(diffMins / 60);
      const mins = diffMins % 60;
      return `${hours}h ${mins}m`;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading status...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Alert className="max-w-md">
          <AlertDescription>
            Error loading status data: {error}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!statusData) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            ScrollIntel Status
          </h1>
          <p className="text-gray-600">
            Current system status and uptime information
          </p>
        </div>

        {/* Overall Status */}
        <Card className="mb-8">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className={`w-4 h-4 rounded-full ${getStatusColor(statusData.overall_status)}`}></div>
                <div>
                  <h2 className="text-2xl font-semibold text-gray-900">
                    {getStatusText(statusData.overall_status)}
                  </h2>
                  <p className="text-gray-600">
                    All systems operational
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-500">
                  Last updated: {formatDate(statusData.last_updated)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Uptime Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                24 Hour Uptime
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {statusData.uptime_stats['24h'].toFixed(2)}%
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                7 Day Uptime
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {statusData.uptime_stats['7d'].toFixed(2)}%
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                30 Day Uptime
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {statusData.uptime_stats['30d'].toFixed(2)}%
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="services" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="services">Services</TabsTrigger>
            <TabsTrigger value="incidents">Incidents</TabsTrigger>
          </TabsList>

          <TabsContent value="services">
            {/* Services Status */}
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-gray-900">
                Service Status
              </h3>
              <div className="space-y-3">
                {Object.entries(statusData.services).map(([serviceName, service]) => (
                  <Card key={serviceName}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${getStatusColor(service.status)}`}></div>
                          <div>
                            <h4 className="font-medium text-gray-900">
                              {serviceName}
                            </h4>
                            <p className="text-sm text-gray-600">
                              {getStatusText(service.status)}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-6 text-sm text-gray-600">
                          <div>
                            <span className="font-medium">Uptime:</span>{' '}
                            {service.uptime_percentage.toFixed(2)}%
                          </div>
                          <div>
                            <span className="font-medium">Response:</span>{' '}
                            {service.response_time.toFixed(0)}ms
                          </div>
                          <div>
                            <span className="font-medium">Last Check:</span>{' '}
                            {formatDate(service.last_check)}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="incidents">
            {/* Incidents */}
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-gray-900">
                Recent Incidents
              </h3>
              {statusData.incidents.length === 0 ? (
                <Card>
                  <CardContent className="p-6 text-center">
                    <div className="text-green-600 mb-2">
                      <svg className="w-12 h-12 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <h4 className="text-lg font-medium text-gray-900 mb-1">
                      No Recent Incidents
                    </h4>
                    <p className="text-gray-600">
                      All systems have been running smoothly
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-3">
                  {statusData.incidents.map((incident) => (
                    <Card key={incident.id}>
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2">
                              <Badge className={getImpactColor(incident.impact)}>
                                {incident.impact.toUpperCase()}
                              </Badge>
                              <Badge variant="outline">
                                {incident.service}
                              </Badge>
                              {incident.resolved_at && (
                                <Badge className="bg-green-100 text-green-800">
                                  RESOLVED
                                </Badge>
                              )}
                            </div>
                            <h4 className="font-medium text-gray-900 mb-1">
                              {incident.title}
                            </h4>
                            <p className="text-sm text-gray-600 mb-2">
                              {incident.description}
                            </p>
                            <div className="flex items-center space-x-4 text-xs text-gray-500">
                              <span>
                                Started: {formatDate(incident.started_at)}
                              </span>
                              {incident.resolved_at ? (
                                <span>
                                  Resolved: {formatDate(incident.resolved_at)}
                                </span>
                              ) : (
                                <span className="text-orange-600">
                                  Ongoing
                                </span>
                              )}
                              <span>
                                Duration: {calculateDuration(incident.started_at, incident.resolved_at)}
                              </span>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>
            Status page powered by ScrollIntel Monitoring System
          </p>
          <p className="mt-1">
            For support, contact{' '}
            <a href="mailto:support@scrollintel.com" className="text-blue-600 hover:underline">
              support@scrollintel.com
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default StatusPage;