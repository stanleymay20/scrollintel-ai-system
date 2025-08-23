"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  Network, 
  Users, 
  TrendingUp, 
  Target, 
  Globe, 
  Activity,
  RefreshCw,
  Filter,
  Search,
  BarChart3,
  PieChart,
  LineChart
} from 'lucide-react';

interface InfluenceTarget {
  id: string;
  name: string;
  title: string;
  organization: string;
  stakeholderType: string;
  influenceScore: number;
  networkCentrality: number;
  relationshipStatus: string;
  lastInteraction?: string;
}

interface InfluenceNetwork {
  id: string;
  name: string;
  domain: string;
  targets: InfluenceTarget[];
  connections: Record<string, string[]>;
  networkMetrics: {
    density: number;
    centralNodes: number;
    totalConnections: number;
  };
}

interface CampaignMetrics {
  campaignId: string;
  networkReach: number;
  influenceScore: number;
  relationshipQuality: number;
  narrativeAdoption: number;
  mediaCoverage: number;
  sentimentScore: number;
  partnershipConversions: number;
  roi: number;
  successRate: number;
}

interface NetworkStatus {
  activeCampaigns: number;
  networkHealth: {
    status: string;
    score: number;
  };
  influenceMetrics: {
    totalInfluenceScore: number;
    networkReach: number;
  };
  relationshipStatus: {
    activeRelationships: number;
    relationshipHealth: number;
  };
  partnershipStatus: {
    activePartnerships: number;
    partnershipValue: number;
  };
}

export default function InfluenceNetworkDashboard() {
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus | null>(null);
  const [selectedNetwork, setSelectedNetwork] = useState<InfluenceNetwork | null>(null);
  const [campaignMetrics, setCampaignMetrics] = useState<CampaignMetrics[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all');

  // Mock data for demonstration
  const mockNetworkStatus: NetworkStatus = {
    activeCampaigns: 12,
    networkHealth: {
      status: 'healthy',
      score: 0.89
    },
    influenceMetrics: {
      totalInfluenceScore: 0.82,
      networkReach: 15000
    },
    relationshipStatus: {
      activeRelationships: 450,
      relationshipHealth: 0.85
    },
    partnershipStatus: {
      activePartnerships: 25,
      partnershipValue: 50000000
    }
  };

  const mockInfluenceTargets: InfluenceTarget[] = [
    {
      id: '1',
      name: 'Dr. Sarah Johnson',
      title: 'Chief Medical Officer',
      organization: 'Global Health Systems',
      stakeholderType: 'executive',
      influenceScore: 0.92,
      networkCentrality: 0.88,
      relationshipStatus: 'active',
      lastInteraction: '2025-08-20'
    },
    {
      id: '2',
      name: 'Michael Chen',
      title: 'VP of Innovation',
      organization: 'TechCorp Industries',
      stakeholderType: 'executive',
      influenceScore: 0.87,
      networkCentrality: 0.75,
      relationshipStatus: 'developing',
      lastInteraction: '2025-08-18'
    },
    {
      id: '3',
      name: 'Prof. Elena Rodriguez',
      title: 'AI Research Director',
      organization: 'Stanford University',
      stakeholderType: 'thought_leader',
      influenceScore: 0.94,
      networkCentrality: 0.91,
      relationshipStatus: 'strong',
      lastInteraction: '2025-08-21'
    }
  ];

  const mockCampaignMetrics: CampaignMetrics[] = [
    {
      campaignId: 'healthcare-ai-leadership',
      networkReach: 5000,
      influenceScore: 0.78,
      relationshipQuality: 0.82,
      narrativeAdoption: 0.65,
      mediaCoverage: 45,
      sentimentScore: 0.73,
      partnershipConversions: 8,
      roi: 3.2,
      successRate: 0.85
    },
    {
      campaignId: 'global-partnerships',
      networkReach: 8000,
      influenceScore: 0.84,
      relationshipQuality: 0.79,
      narrativeAdoption: 0.71,
      mediaCoverage: 62,
      sentimentScore: 0.81,
      partnershipConversions: 12,
      roi: 4.1,
      successRate: 0.91
    }
  ];

  useEffect(() => {
    // Simulate API call
    const loadData = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setNetworkStatus(mockNetworkStatus);
      setCampaignMetrics(mockCampaignMetrics);
      setIsLoading(false);
    };

    loadData();
  }, []);

  const refreshData = useCallback(async () => {
    setIsLoading(true);
    // Simulate refresh
    await new Promise(resolve => setTimeout(resolve, 500));
    setIsLoading(false);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getRelationshipStatusBadge = (status: string) => {
    const colors = {
      'strong': 'bg-green-100 text-green-800',
      'active': 'bg-blue-100 text-blue-800',
      'developing': 'bg-yellow-100 text-yellow-800',
      'inactive': 'bg-gray-100 text-gray-800'
    };
    return colors[status as keyof typeof colors] || colors.inactive;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span>Loading influence network data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Global Influence Network</h1>
          <p className="text-gray-600">
            Monitor and orchestrate worldwide influence campaigns and relationships
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={refreshData} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button>
            <Network className="h-4 w-4 mr-2" />
            New Campaign
          </Button>
        </div>
      </div>

      {/* Network Status Overview */}
      {networkStatus && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Campaigns</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{networkStatus.activeCampaigns}</div>
              <p className="text-xs text-muted-foreground">
                +2 from last month
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Network Health</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(networkStatus.networkHealth.score * 100).toFixed(0)}%
              </div>
              <p className={`text-xs ${getStatusColor(networkStatus.networkHealth.status)}`}>
                {networkStatus.networkHealth.status}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Network Reach</CardTitle>
              <Globe className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {networkStatus.influenceMetrics.networkReach.toLocaleString()}
              </div>
              <p className="text-xs text-muted-foreground">
                Total influence targets
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Partnership Value</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${(networkStatus.partnershipStatus.partnershipValue / 1000000).toFixed(0)}M
              </div>
              <p className="text-xs text-muted-foreground">
                {networkStatus.partnershipStatus.activePartnerships} active partnerships
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Dashboard Tabs */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="network">Network Graph</TabsTrigger>
          <TabsTrigger value="campaigns">Campaigns</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Relationship Health */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Users className="h-5 w-5 mr-2" />
                  Relationship Health
                </CardTitle>
                <CardDescription>
                  Monitor the health of key relationships
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {networkStatus && (
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Overall Health</span>
                      <span>{(networkStatus.relationshipStatus.relationshipHealth * 100).toFixed(0)}%</span>
                    </div>
                    <Progress value={networkStatus.relationshipStatus.relationshipHealth * 100} />
                  </div>
                )}
                
                <div className="space-y-3">
                  {mockInfluenceTargets.slice(0, 3).map((target) => (
                    <div key={target.id} className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                          <span className="text-xs font-medium text-blue-600">
                            {target.name.split(' ').map(n => n[0]).join('')}
                          </span>
                        </div>
                        <div>
                          <p className="text-sm font-medium">{target.name}</p>
                          <p className="text-xs text-gray-500">{target.organization}</p>
                        </div>
                      </div>
                      <Badge className={getRelationshipStatusBadge(target.relationshipStatus)}>
                        {target.relationshipStatus}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Campaign Performance */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="h-5 w-5 mr-2" />
                  Campaign Performance
                </CardTitle>
                <CardDescription>
                  Track active campaign metrics
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {campaignMetrics.map((campaign) => (
                  <div key={campaign.campaignId} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">
                        {campaign.campaignId.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                      <span className="text-sm text-gray-500">
                        {(campaign.successRate * 100).toFixed(0)}% success
                      </span>
                    </div>
                    <Progress value={campaign.successRate * 100} />
                    <div className="grid grid-cols-3 gap-2 text-xs text-gray-500">
                      <span>Reach: {campaign.networkReach.toLocaleString()}</span>
                      <span>ROI: {campaign.roi.toFixed(1)}x</span>
                      <span>Conversions: {campaign.partnershipConversions}</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="network" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Network className="h-5 w-5 mr-2" />
                Interactive Network Graph
              </CardTitle>
              <CardDescription>
                Explore influence network connections and relationships
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96 bg-gray-50 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <Network className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600">Interactive network visualization</p>
                  <p className="text-sm text-gray-500">
                    Network graph with {mockInfluenceTargets.length} nodes and connections
                  </p>
                </div>
              </div>
              
              {/* Network Controls */}
              <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Button variant="outline" size="sm">
                    <Filter className="h-4 w-4 mr-2" />
                    Filter
                  </Button>
                  <Button variant="outline" size="sm">
                    <Search className="h-4 w-4 mr-2" />
                    Search
                  </Button>
                </div>
                <div className="text-sm text-gray-500">
                  Showing {mockInfluenceTargets.length} influence targets
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="campaigns" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Active Campaigns</CardTitle>
              <CardDescription>
                Monitor and manage influence campaigns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {campaignMetrics.map((campaign) => (
                  <div key={campaign.campaignId} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-medium">
                        {campaign.campaignId.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </h3>
                      <Badge variant="outline">Active</Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500">Network Reach</p>
                        <p className="font-medium">{campaign.networkReach.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Influence Score</p>
                        <p className="font-medium">{(campaign.influenceScore * 100).toFixed(0)}%</p>
                      </div>
                      <div>
                        <p className="text-gray-500">ROI</p>
                        <p className="font-medium">{campaign.roi.toFixed(1)}x</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Success Rate</p>
                        <p className="font-medium">{(campaign.successRate * 100).toFixed(0)}%</p>
                      </div>
                    </div>
                    
                    <div className="mt-3">
                      <Progress value={campaign.successRate * 100} />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <PieChart className="h-5 w-5 mr-2" />
                  Influence Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <PieChart className="h-12 w-12 mx-auto text-gray-400 mb-2" />
                    <p className="text-gray-600">Influence distribution chart</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <LineChart className="h-5 w-5 mr-2" />
                  Campaign Trends
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <LineChart className="h-12 w-12 mx-auto text-gray-400 mb-2" />
                    <p className="text-gray-600">Campaign performance trends</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}