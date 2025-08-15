import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

interface Agent {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  status: 'online' | 'offline' | 'busy';
  responseTime: number;
}

interface AgentSelectorProps {
  onAgentSelect: (agent: Agent) => void;
  selectedAgent?: Agent;
}

const CORE_AGENTS: Agent[] = [
  {
    id: 'cto',
    name: 'ScrollCTO',
    description: 'Strategic technology leadership and architecture decisions',
    capabilities: ['Technology Strategy', 'Architecture Design', 'Scaling Decisions', 'Team Structure'],
    status: 'online',
    responseTime: 1.2
  },
  {
    id: 'data-scientist',
    name: 'ScrollDataScientist',
    description: 'Statistical analysis, hypothesis testing, and data insights',
    capabilities: ['EDA', 'Statistical Modeling', 'Feature Engineering', 'Research Design'],
    status: 'online',
    responseTime: 0.8
  },
  {
    id: 'ml-engineer',
    name: 'ScrollMLEngineer',
    description: 'ML pipeline design, model training, and deployment',
    capabilities: ['MLOps', 'Model Training', 'Performance Monitoring', 'Production Deployment'],
    status: 'online',
    responseTime: 1.5
  },
  {
    id: 'ai-engineer',
    name: 'ScrollAIEngineer',
    description: 'AI strategy, LLM integration, and RAG implementation',
    capabilities: ['AI Strategy', 'LLM Integration', 'RAG Systems', 'Vector Operations'],
    status: 'online',
    responseTime: 1.1
  },
  {
    id: 'bi-agent',
    name: 'ScrollBI',
    description: 'Business intelligence, dashboards, and KPI tracking',
    capabilities: ['Dashboard Creation', 'KPI Monitoring', 'Business Insights', 'Alerts'],
    status: 'online',
    responseTime: 0.9
  },
  {
    id: 'qa-agent',
    name: 'ScrollQA',
    description: 'Natural language data querying and SQL generation',
    capabilities: ['Natural Language Queries', 'SQL Generation', 'Data Exploration', 'Query Optimization'],
    status: 'online',
    responseTime: 0.7
  },
  {
    id: 'forecast-agent',
    name: 'ScrollForecast',
    description: 'Time series forecasting and predictive analytics',
    capabilities: ['Time Series Forecasting', 'Trend Analysis', 'Seasonality Detection', 'Model Selection'],
    status: 'online',
    responseTime: 1.3
  }
];

export const AgentSelector: React.FC<AgentSelectorProps> = ({ onAgentSelect, selectedAgent }) => {
  const [agents, setAgents] = useState<Agent[]>(CORE_AGENTS);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Simulate real-time agent status updates
    const interval = setInterval(() => {
      setAgents(prevAgents => 
        prevAgents.map(agent => ({
          ...agent,
          responseTime: Math.round((Math.random() * 0.5 + 0.5) * 100) / 100
        }))
      );
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'busy': return 'bg-yellow-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'online': return 'Online';
      case 'busy': return 'Busy';
      case 'offline': return 'Offline';
      default: return 'Unknown';
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Select AI Agent</h2>
        <Badge variant="outline" className="text-sm">
          {agents.filter(a => a.status === 'online').length} of {agents.length} agents online
        </Badge>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agents.map((agent) => (
          <Card 
            key={agent.id} 
            className={`cursor-pointer transition-all hover:shadow-lg ${
              selectedAgent?.id === agent.id ? 'ring-2 ring-blue-500 bg-blue-50' : ''
            }`}
            onClick={() => onAgentSelect(agent)}
          >
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{agent.name}</CardTitle>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(agent.status)}`} />
                  <span className="text-xs text-gray-500">{getStatusText(agent.status)}</span>
                </div>
              </div>
              <CardDescription className="text-sm">
                {agent.description}
              </CardDescription>
            </CardHeader>
            
            <CardContent>
              <div className="space-y-3">
                <div>
                  <p className="text-xs font-medium text-gray-600 mb-2">Core Capabilities</p>
                  <div className="flex flex-wrap gap-1">
                    {agent.capabilities.slice(0, 3).map((capability, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {capability}
                      </Badge>
                    ))}
                    {agent.capabilities.length > 3 && (
                      <Badge variant="outline" className="text-xs">
                        +{agent.capabilities.length - 3} more
                      </Badge>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>Response Time</span>
                  <span className="font-medium">{agent.responseTime}s avg</span>
                </div>
                
                <Button 
                  size="sm" 
                  className="w-full"
                  disabled={agent.status !== 'online'}
                  variant={selectedAgent?.id === agent.id ? "default" : "outline"}
                >
                  {selectedAgent?.id === agent.id ? 'Selected' : 'Select Agent'}
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      
      {selectedAgent && (
        <Card className="bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-lg text-blue-800">
              {selectedAgent.name} Selected
            </CardTitle>
            <CardDescription>
              Ready to assist with {selectedAgent.description.toLowerCase()}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p className="text-sm font-medium text-blue-700">Full Capabilities:</p>
              <div className="flex flex-wrap gap-1">
                {selectedAgent.capabilities.map((capability, index) => (
                  <Badge key={index} variant="default" className="text-xs bg-blue-100 text-blue-800">
                    {capability}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AgentSelector;