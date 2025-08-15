'use client'

import React from 'react'
import { useOnboarding, OnboardingStep } from './onboarding-provider'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  Bot, 
  Database, 
  BarChart3, 
  Brain, 
  Code, 
  Zap,
  MessageSquare,
  Upload,
  Play
} from 'lucide-react'
import { Agent } from '@/types'

interface AgentTutorialsProps {
  agents: Agent[]
}

export function AgentTutorials({ agents }: AgentTutorialsProps) {
  const { startOnboarding } = useOnboarding()

  const getAgentIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'cto':
        return Brain
      case 'datascientist':
        return Database
      case 'mlengineer':
        return Bot
      case 'aiengineer':
        return Zap
      case 'analyst':
        return BarChart3
      default:
        return Code
    }
  }

  const createAgentTutorial = (agent: Agent): OnboardingStep[] => {
    const Icon = getAgentIcon(agent.type)
    
    const baseSteps: OnboardingStep[] = [
      {
        id: `${agent.id}-intro`,
        title: `Meet ${agent.name}`,
        description: `Learn about ${agent.name}'s capabilities and how to interact with it effectively.`,
        content: (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-scrollintel-primary rounded-lg flex items-center justify-center">
                <Icon className="h-6 w-6 text-white" />
              </div>
              <div>
                <h3 className="font-semibold">{agent.name}</h3>
                <p className="text-sm text-muted-foreground">{agent.description}</p>
              </div>
            </div>
            
            <div className="space-y-2">
              <h4 className="font-medium">Key Capabilities:</h4>
              <div className="flex flex-wrap gap-2">
                {agent.capabilities.map((capability) => (
                  <Badge key={capability} variant="secondary">
                    {capability}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        ),
      },
      {
        id: `${agent.id}-chat`,
        title: 'Start a Conversation',
        description: 'Learn how to chat with this agent and ask questions.',
        target: 'chat-interface',
        content: (
          <div className="space-y-4">
            <div className="bg-muted p-4 rounded-lg">
              <h4 className="font-medium mb-2">Sample Questions:</h4>
              <ul className="space-y-1 text-sm">
                {getAgentSampleQuestions(agent.type).map((question, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <MessageSquare className="h-4 w-4 mt-0.5 text-scrollintel-primary" />
                    <span>"{question}"</span>
                  </li>
                ))}
              </ul>
            </div>
            <p className="text-sm text-muted-foreground">
              Click on the agent card and start typing your question in the chat interface.
            </p>
          </div>
        ),
      },
      {
        id: `${agent.id}-upload`,
        title: 'Upload Data',
        description: 'Learn how to upload files for analysis.',
        target: 'file-upload',
        content: (
          <div className="space-y-4">
            <div className="bg-muted p-4 rounded-lg">
              <h4 className="font-medium mb-2">Supported File Types:</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex items-center gap-2">
                  <Upload className="h-4 w-4 text-scrollintel-primary" />
                  <span>CSV Files</span>
                </div>
                <div className="flex items-center gap-2">
                  <Upload className="h-4 w-4 text-scrollintel-primary" />
                  <span>Excel Files</span>
                </div>
                <div className="flex items-center gap-2">
                  <Upload className="h-4 w-4 text-scrollintel-primary" />
                  <span>JSON Files</span>
                </div>
                <div className="flex items-center gap-2">
                  <Upload className="h-4 w-4 text-scrollintel-primary" />
                  <span>SQL Files</span>
                </div>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              Drag and drop files or click to browse. The agent will automatically analyze your data.
            </p>
          </div>
        ),
      },
      {
        id: `${agent.id}-results`,
        title: 'Understanding Results',
        description: 'Learn how to interpret and export the agent\'s analysis.',
        content: (
          <div className="space-y-4">
            <div className="bg-muted p-4 rounded-lg">
              <h4 className="font-medium mb-2">What to Expect:</h4>
              <ul className="space-y-2 text-sm">
                {getAgentExpectedResults(agent.type).map((result, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <BarChart3 className="h-4 w-4 mt-0.5 text-scrollintel-primary" />
                    <span>{result}</span>
                  </li>
                ))}
              </ul>
            </div>
            <p className="text-sm text-muted-foreground">
              Results can be exported as PDF reports or Excel files for further analysis.
            </p>
          </div>
        ),
      },
    ]

    return baseSteps
  }

  const getAgentSampleQuestions = (type: string): string[] => {
    switch (type.toLowerCase()) {
      case 'cto':
        return [
          'What architecture should I use for a high-traffic web application?',
          'How can I optimize our database performance?',
          'What are the best practices for microservices?',
        ]
      case 'datascientist':
        return [
          'Can you perform exploratory data analysis on my dataset?',
          'What statistical tests should I use for this data?',
          'Help me identify patterns and correlations in my data.',
        ]
      case 'mlengineer':
        return [
          'Build a machine learning model for my dataset.',
          'How can I deploy this model to production?',
          'What metrics should I monitor for model performance?',
        ]
      case 'aiengineer':
        return [
          'Help me implement a RAG system for my documents.',
          'How can I optimize my vector embeddings?',
          'Build a chatbot for my specific use case.',
        ]
      default:
        return [
          'Analyze my business data and provide insights.',
          'Create visualizations for my dataset.',
          'Help me understand trends in my data.',
        ]
    }
  }

  const getAgentExpectedResults = (type: string): string[] => {
    switch (type.toLowerCase()) {
      case 'cto':
        return [
          'Architecture diagrams and recommendations',
          'Technology stack analysis',
          'Performance optimization suggestions',
          'Cost analysis and projections',
        ]
      case 'datascientist':
        return [
          'Statistical analysis reports',
          'Data quality assessments',
          'Correlation matrices and insights',
          'Hypothesis testing results',
        ]
      case 'mlengineer':
        return [
          'Trained machine learning models',
          'Model performance metrics',
          'Deployment configurations',
          'Monitoring dashboards',
        ]
      case 'aiengineer':
        return [
          'AI system implementations',
          'Vector database configurations',
          'RAG pipeline setups',
          'Performance benchmarks',
        ]
      default:
        return [
          'Interactive dashboards',
          'Business intelligence reports',
          'Data visualizations',
          'Key performance indicators',
        ]
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Agent Tutorials</h2>
        <Badge variant="secondary">Interactive Learning</Badge>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agents.map((agent) => {
          const Icon = getAgentIcon(agent.type)
          
          return (
            <Card key={agent.id} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-scrollintel-primary rounded-lg flex items-center justify-center">
                    <Icon className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-base">{agent.name}</CardTitle>
                    <Badge variant="outline" className="text-xs">
                      {agent.type}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  {agent.description}
                </p>
                
                <div className="space-y-2">
                  <h4 className="text-sm font-medium">You'll learn:</h4>
                  <ul className="text-xs text-muted-foreground space-y-1">
                    <li>• How to interact with {agent.name}</li>
                    <li>• Best practices for data upload</li>
                    <li>• Understanding analysis results</li>
                    <li>• Exporting and sharing insights</li>
                  </ul>
                </div>
                
                <Button
                  onClick={() => startOnboarding(createAgentTutorial(agent))}
                  className="w-full"
                  size="sm"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start Tutorial
                </Button>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}