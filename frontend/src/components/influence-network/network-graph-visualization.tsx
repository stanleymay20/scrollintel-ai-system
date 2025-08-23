"use client";

import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Filter, 
  Search,
  Users,
  Target,
  Network
} from 'lucide-react';

interface NetworkNode {
  id: string;
  name: string;
  title: string;
  organization: string;
  stakeholderType: string;
  influenceScore: number;
  networkCentrality: number;
  relationshipStatus: string;
  x?: number;
  y?: number;
}

interface NetworkEdge {
  source: string;
  target: string;
  strength: number;
  type: string;
}

interface NetworkGraphProps {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  width?: number;
  height?: number;
  onNodeClick?: (node: NetworkNode) => void;
  onEdgeClick?: (edge: NetworkEdge) => void;
}

export default function NetworkGraphVisualization({
  nodes = [],
  edges = [],
  width = 800,
  height = 600,
  onNodeClick,
  onEdgeClick
}: NetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });

  // Mock data for demonstration
  const mockNodes: NetworkNode[] = [
    {
      id: '1',
      name: 'Dr. Sarah Johnson',
      title: 'Chief Medical Officer',
      organization: 'Global Health Systems',
      stakeholderType: 'executive',
      influenceScore: 0.92,
      networkCentrality: 0.88,
      relationshipStatus: 'strong',
      x: 400,
      y: 200
    },
    {
      id: '2',
      name: 'Michael Chen',
      title: 'VP of Innovation',
      organization: 'TechCorp Industries',
      stakeholderType: 'executive',
      influenceScore: 0.87,
      networkCentrality: 0.75,
      relationshipStatus: 'active',
      x: 200,
      y: 300
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
      x: 600,
      y: 300
    },
    {
      id: '4',
      name: 'James Wilson',
      title: 'Healthcare Investor',
      organization: 'MedTech Ventures',
      stakeholderType: 'investor',
      influenceScore: 0.83,
      networkCentrality: 0.72,
      relationshipStatus: 'developing',
      x: 300,
      y: 450
    },
    {
      id: '5',
      name: 'Lisa Park',
      title: 'Policy Director',
      organization: 'Health Innovation Council',
      stakeholderType: 'regulator',
      influenceScore: 0.89,
      networkCentrality: 0.85,
      relationshipStatus: 'active',
      x: 500,
      y: 450
    }
  ];

  const mockEdges: NetworkEdge[] = [
    { source: '1', target: '2', strength: 0.8, type: 'collaboration' },
    { source: '1', target: '3', strength: 0.9, type: 'research' },
    { source: '2', target: '4', strength: 0.7, type: 'investment' },
    { source: '3', target: '5', strength: 0.85, type: 'policy' },
    { source: '1', target: '5', strength: 0.75, type: 'advisory' },
    { source: '4', target: '5', strength: 0.6, type: 'funding' }
  ];

  const displayNodes = nodes.length > 0 ? nodes : mockNodes;
  const displayEdges = edges.length > 0 ? edges : mockEdges;

  const getNodeColor = (node: NetworkNode) => {
    const colors = {
      executive: '#3B82F6',
      thought_leader: '#10B981',
      investor: '#F59E0B',
      regulator: '#8B5CF6',
      media: '#EF4444'
    };
    return colors[node.stakeholderType as keyof typeof colors] || '#6B7280';
  };

  const getNodeSize = (node: NetworkNode) => {
    return 8 + (node.influenceScore * 12); // Size based on influence score
  };

  const getEdgeColor = (edge: NetworkEdge) => {
    const colors = {
      collaboration: '#3B82F6',
      research: '#10B981',
      investment: '#F59E0B',
      policy: '#8B5CF6',
      advisory: '#EF4444',
      funding: '#F97316'
    };
    return colors[edge.type as keyof typeof colors] || '#6B7280';
  };

  const getEdgeWidth = (edge: NetworkEdge) => {
    return 1 + (edge.strength * 3); // Width based on relationship strength
  };

  const handleNodeClick = (node: NetworkNode) => {
    setSelectedNode(node);
    onNodeClick?.(node);
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.2, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.3));
  };

  const handleReset = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
    setSelectedNode(null);
  };

  const filteredNodes = displayNodes.filter(node => {
    const matchesSearch = searchQuery === '' || 
      node.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.organization.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesFilter = filterType === 'all' || node.stakeholderType === filterType;
    
    return matchesSearch && matchesFilter;
  });

  const filteredEdges = displayEdges.filter(edge => {
    const sourceVisible = filteredNodes.some(n => n.id === edge.source);
    const targetVisible = filteredNodes.some(n => n.id === edge.target);
    return sourceVisible && targetVisible;
  });

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search nodes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 w-64"
            />
          </div>
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-2 border rounded-md text-sm"
          >
            <option value="all">All Types</option>
            <option value="executive">Executives</option>
            <option value="thought_leader">Thought Leaders</option>
            <option value="investor">Investors</option>
            <option value="regulator">Regulators</option>
            <option value="media">Media</option>
          </select>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm" onClick={handleZoomIn}>
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={handleZoomOut}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
          <span className="text-sm text-gray-500">
            {Math.round(zoomLevel * 100)}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Network Graph */}
        <div className="lg:col-span-3">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Network className="h-5 w-5 mr-2" />
                Influence Network Graph
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative overflow-hidden border rounded-lg" style={{ height: height }}>
                <svg
                  ref={svgRef}
                  width={width}
                  height={height}
                  viewBox={`${-panOffset.x} ${-panOffset.y} ${width / zoomLevel} ${height / zoomLevel}`}
                  className="w-full h-full"
                >
                  {/* Edges */}
                  <g className="edges">
                    {filteredEdges.map((edge, index) => {
                      const sourceNode = filteredNodes.find(n => n.id === edge.source);
                      const targetNode = filteredNodes.find(n => n.id === edge.target);
                      
                      if (!sourceNode || !targetNode) return null;
                      
                      return (
                        <line
                          key={index}
                          x1={sourceNode.x}
                          y1={sourceNode.y}
                          x2={targetNode.x}
                          y2={targetNode.y}
                          stroke={getEdgeColor(edge)}
                          strokeWidth={getEdgeWidth(edge)}
                          opacity={0.6}
                          className="cursor-pointer hover:opacity-100"
                          onClick={() => onEdgeClick?.(edge)}
                        />
                      );
                    })}
                  </g>

                  {/* Nodes */}
                  <g className="nodes">
                    {filteredNodes.map((node) => (
                      <g key={node.id}>
                        <circle
                          cx={node.x}
                          cy={node.y}
                          r={getNodeSize(node)}
                          fill={getNodeColor(node)}
                          stroke={selectedNode?.id === node.id ? '#000' : '#fff'}
                          strokeWidth={selectedNode?.id === node.id ? 3 : 2}
                          className="cursor-pointer hover:opacity-80"
                          onClick={() => handleNodeClick(node)}
                        />
                        <text
                          x={node.x}
                          y={node.y! + getNodeSize(node) + 15}
                          textAnchor="middle"
                          className="text-xs fill-gray-700 pointer-events-none"
                        >
                          {node.name.split(' ')[0]}
                        </text>
                      </g>
                    ))}
                  </g>
                </svg>
              </div>
              
              {/* Legend */}
              <div className="mt-4 flex flex-wrap gap-4 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                  <span>Executive</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span>Thought Leader</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                  <span>Investor</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                  <span>Regulator</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <span>Media</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Node Details Panel */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Users className="h-5 w-5 mr-2" />
                {selectedNode ? 'Node Details' : 'Network Stats'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {selectedNode ? (
                <div className="space-y-4">
                  <div>
                    <h3 className="font-medium">{selectedNode.name}</h3>
                    <p className="text-sm text-gray-600">{selectedNode.title}</p>
                    <p className="text-sm text-gray-500">{selectedNode.organization}</p>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Influence Score</span>
                      <span className="text-sm font-medium">
                        {(selectedNode.influenceScore * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Network Centrality</span>
                      <span className="text-sm font-medium">
                        {(selectedNode.networkCentrality * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Relationship Status</span>
                      <Badge variant="outline" className="text-xs">
                        {selectedNode.relationshipStatus}
                      </Badge>
                    </div>
                  </div>
                  
                  <Button size="sm" className="w-full">
                    View Full Profile
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="text-center">
                    <Target className="h-8 w-8 mx-auto text-gray-400 mb-2" />
                    <p className="text-sm text-gray-600">
                      Click on a node to view details
                    </p>
                  </div>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Total Nodes</span>
                      <span className="font-medium">{filteredNodes.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total Connections</span>
                      <span className="font-medium">{filteredEdges.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Network Density</span>
                      <span className="font-medium">
                        {((filteredEdges.length / (filteredNodes.length * (filteredNodes.length - 1))) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}