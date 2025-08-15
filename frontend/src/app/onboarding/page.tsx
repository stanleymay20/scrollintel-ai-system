'use client'

import React, { useState } from 'react'
import { AgentTutorials } from '@/components/onboarding/agent-tutorials'
import { SampleDataManager } from '@/components/onboarding/sample-data-manager'
import { OnboardingProvider, useOnboarding } from '@/components/onboarding/onboarding-provider'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  GraduationCap, 
  Database, 
  PlayCircle, 
  BookOpen, 
  Lightbulb,
  CheckCircle,
  ArrowRight
} from 'lucide-react'
import { Agent } from '@/types'

// Mock agents data
const mockAgents: Agent[] = [
  {
    id: 'cto-1',
    name: 'ScrollCTO',
    type: 'CTO',
    status: 'active',
    capabilities: ['Architecture', 'Scaling', 'Tech Stack', 'Cost Analysis'],
    description: 'AI-powered CTO for technical decisions and architecture planning',
    last_active: new Date().toISOString(),
    metrics: {
      requests_handled: 1247,
      avg_response_time: 850,
      success_rate: 98.5,
    },
  },
  {
    id: 'ds-1',
    name: 'ScrollDataScientist',
    type: 'DataScientist',
    status: 'active',
    capabilities: ['EDA', 'Statistical Analysis', 'Hypothesis Testing', 'Feature Engineering'],
    description: 'Advanced data science and statistical analysis capabilities',
    last_active: new Date().toISOString(),
    metrics: {
      requests_handled: 892,
      avg_response_time: 1200,
      success_rate: 96.8,
    },
  },
  {
    id: 'ml-1',
    name: 'ScrollMLEngineer',
    type: 'MLEngineer',
    status: 'active',
    capabilities: ['Model Training', 'MLOps', 'Deployment', 'Monitoring'],
    description: 'Machine learning engineering and model deployment',
    last_active: new Date().toISOString(),
    metrics: {
      requests_handled: 634,
      avg_response_time: 2100,
      success_rate: 94.2,
    },
  },
  {
    id: 'ai-1',
    name: 'ScrollAIEngineer',
    type: 'AIEngineer',
    status: 'active',
    capabilities: ['RAG', 'LLM Integration', 'Vector Search', 'Embeddings'],
    description: 'AI engineering with memory-enhanced capabilities',
    last_active: new Date().toISOString(),
    metrics: {
      requests_handled: 1456,
      avg_response_time: 950,
      success_rate: 97.3,
    },
  },
]

function OnboardingContent() {
  const { startOnboarding } = useOnboarding()
  const [completedSections, setCompletedSections] = useState<string[]>([])

  const quickStartSteps = [
    {
      id: 'welcome',
      title: 'Welcome to ScrollIntel',
      description: 'Get familiar with the platform overview and key features.',
      content: (
        <div className="space-y-4">
          <div className="text-center">
            <GraduationCap className="h-16 w-16 text-scrollintel-primary mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Welcome to ScrollIntel!</h3>
            <p className="text-muted-foreground">
              ScrollIntel is the world's most advanced AI-CTO platform, designed to replace human technical expertise with AI agents.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <PlayCircle className="h-8 w-8 text-scrollintel-primary mx-auto mb-2" />
              <h4 className="font-medium">AI Agents</h4>
              <p className="text-sm text-muted-foreground">Specialized AI agents for different technical roles</p>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <Database className="h-8 w-8 text-scrollintel-primary mx-auto mb-2" />
              <h4 className="font-medium">Data Analysis</h4>
              <p className="text-sm text-muted-foreground">Upload and analyze your data with AI</p>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <Lightbulb className="h-8 w-8 text-scrollintel-primary mx-auto mb-2" />
              <h4 className="font-medium">Insights</h4>
              <p className="text-sm text-muted-foreground">Get actionable insights and recommendations</p>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 'navigation',
      title: 'Platform Navigation',
      description: 'Learn how to navigate the ScrollIntel interface.',
      content: (
        <div className="space-y-4">
          <h3 className="font-semibold">Key Areas of the Platform:</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 bg-muted rounded-lg">
              <CheckCircle className="h-5 w-5 text-scrollintel-primary mt-0.5" />
              <div>
                <h4 className="font-medium">Dashboard</h4>
                <p className="text-sm text-muted-foreground">Your main hub for system metrics and agent status</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-muted rounded-lg">
              <CheckCircle className="h-5 w-5 text-scrollintel-primary mt-0.5" />
              <div>
                <h4 className="font-medium">Chat Interface</h4>
                <p className="text-sm text-muted-foreground">Communicate with AI agents in natural language</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-muted rounded-lg">
              <CheckCircle className="h-5 w-5 text-scrollintel-primary mt-0.5" />
              <div>
                <h4 className="font-medium">File Upload</h4>
                <p className="text-sm text-muted-foreground">Upload data files for analysis and processing</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-muted rounded-lg">
              <CheckCircle className="h-5 w-5 text-scrollintel-primary mt-0.5" />
              <div>
                <h4 className="font-medium">Analytics</h4>
                <p className="text-sm text-muted-foreground">View detailed reports and visualizations</p>
              </div>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 'first-steps',
      title: 'Your First Steps',
      description: 'Recommended actions to get started with ScrollIntel.',
      content: (
        <div className="space-y-4">
          <h3 className="font-semibold">Recommended Getting Started Flow:</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="w-6 h-6 bg-scrollintel-primary rounded-full flex items-center justify-center text-white text-xs font-bold">1</div>
              <div>
                <h4 className="font-medium text-blue-900">Download Sample Data</h4>
                <p className="text-sm text-blue-700">Start with our pre-built datasets to explore features</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="w-6 h-6 bg-scrollintel-primary rounded-full flex items-center justify-center text-white text-xs font-bold">2</div>
              <div>
                <h4 className="font-medium text-blue-900">Try Agent Tutorials</h4>
                <p className="text-sm text-blue-700">Learn how to interact with each AI agent effectively</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="w-6 h-6 bg-scrollintel-primary rounded-full flex items-center justify-center text-white text-xs font-bold">3</div>
              <div>
                <h4 className="font-medium text-blue-900">Upload Your Data</h4>
                <p className="text-sm text-blue-700">Upload your own data files for analysis</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="w-6 h-6 bg-scrollintel-primary rounded-full flex items-center justify-center text-white text-xs font-bold">4</div>
              <div>
                <h4 className="font-medium text-blue-900">Explore Results</h4>
                <p className="text-sm text-blue-700">Review insights, export reports, and take action</p>
              </div>
            </div>
          </div>
        </div>
      ),
    },
  ]

  const handleStartQuickTour = () => {
    startOnboarding(quickStartSteps)
  }

  const markSectionComplete = (sectionId: string) => {
    if (!completedSections.includes(sectionId)) {
      setCompletedSections(prev => [...prev, sectionId])
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-scrollintel-primary to-scrollintel-secondary rounded-full flex items-center justify-center mx-auto mb-4">
            <GraduationCap className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold mb-2">Welcome to ScrollIntel</h1>
          <p className="text-muted-foreground text-lg">
            Get started with our comprehensive onboarding experience
          </p>
          <div className="flex items-center justify-center gap-2 mt-4">
            <Badge variant="secondary">
              {completedSections.length} of 3 sections completed
            </Badge>
          </div>
        </div>

        {/* Quick Start Card */}
        <Card className="mb-8">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <PlayCircle className="h-5 w-5 text-scrollintel-primary" />
                  Quick Start Tour
                </CardTitle>
                <p className="text-muted-foreground mt-1">
                  Take a 5-minute guided tour of the platform
                </p>
              </div>
              <Button onClick={handleStartQuickTour}>
                Start Tour
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </div>
          </CardHeader>
        </Card>

        {/* Main Content Tabs */}
        <Tabs defaultValue="tutorials" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="tutorials" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              Agent Tutorials
              {completedSections.includes('tutorials') && (
                <CheckCircle className="h-4 w-4 text-green-500" />
              )}
            </TabsTrigger>
            <TabsTrigger value="sample-data" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Sample Data
              {completedSections.includes('sample-data') && (
                <CheckCircle className="h-4 w-4 text-green-500" />
              )}
            </TabsTrigger>
            <TabsTrigger value="resources" className="flex items-center gap-2">
              <Lightbulb className="h-4 w-4" />
              Resources
              {completedSections.includes('resources') && (
                <CheckCircle className="h-4 w-4 text-green-500" />
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="tutorials" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-semibold">Interactive Agent Tutorials</h2>
                <p className="text-muted-foreground">
                  Learn how to work with each AI agent through hands-on tutorials
                </p>
              </div>
              <Button
                variant="outline"
                onClick={() => markSectionComplete('tutorials')}
              >
                Mark Complete
              </Button>
            </div>
            <AgentTutorials agents={mockAgents} />
          </TabsContent>

          <TabsContent value="sample-data" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-semibold">Sample Datasets</h2>
                <p className="text-muted-foreground">
                  Download sample data to explore platform capabilities
                </p>
              </div>
              <Button
                variant="outline"
                onClick={() => markSectionComplete('sample-data')}
              >
                Mark Complete
              </Button>
            </div>
            <SampleDataManager />
          </TabsContent>

          <TabsContent value="resources" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-semibold">Learning Resources</h2>
                <p className="text-muted-foreground">
                  Additional resources to help you master ScrollIntel
                </p>
              </div>
              <Button
                variant="outline"
                onClick={() => markSectionComplete('resources')}
              >
                Mark Complete
              </Button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <BookOpen className="h-5 w-5 text-scrollintel-primary" />
                    Documentation
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    Comprehensive guides and API documentation
                  </p>
                  <Button variant="outline" size="sm" className="w-full">
                    View Docs
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <PlayCircle className="h-5 w-5 text-scrollintel-primary" />
                    Video Tutorials
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    Step-by-step video guides for common workflows
                  </p>
                  <Button variant="outline" size="sm" className="w-full">
                    Watch Videos
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Lightbulb className="h-5 w-5 text-scrollintel-primary" />
                    Best Practices
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    Tips and tricks from ScrollIntel experts
                  </p>
                  <Button variant="outline" size="sm" className="w-full">
                    Learn More
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>

        {/* Completion Card */}
        {completedSections.length === 3 && (
          <Card className="mt-8 bg-green-50 border-green-200">
            <CardContent className="p-6 text-center">
              <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-green-900 mb-2">
                Congratulations! 🎉
              </h3>
              <p className="text-green-700 mb-4">
                You've completed the onboarding process. You're ready to start using ScrollIntel!
              </p>
              <Button className="bg-green-600 hover:bg-green-700">
                Go to Dashboard
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

export default function OnboardingPage() {
  return (
    <OnboardingProvider>
      <OnboardingContent />
    </OnboardingProvider>
  )
}