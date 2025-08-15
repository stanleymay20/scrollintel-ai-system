'use client'

import React, { useState, useEffect } from 'react'
import { useOnboarding, OnboardingStep } from './onboarding-provider'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { 
  Rocket, 
  User, 
  Building, 
  Target, 
  Zap, 
  CheckCircle,
  ArrowRight,
  Sparkles
} from 'lucide-react'
import { Agent } from '@/types'

interface WelcomeWizardProps {
  agents: Agent[]
}

interface UserProfile {
  name: string
  role: string
  company: string
  goals: string[]
  experience: string
}

export function WelcomeWizard({ agents }: WelcomeWizardProps) {
  const { startOnboarding } = useOnboarding()
  const [showWizard, setShowWizard] = useState(false)
  const [userProfile, setUserProfile] = useState<UserProfile>({
    name: '',
    role: '',
    company: '',
    goals: [],
    experience: 'beginner'
  })

  useEffect(() => {
    // Check if user has seen the welcome wizard
    const hasSeenWizard = localStorage.getItem('scrollintel-welcome-wizard-completed')
    if (!hasSeenWizard) {
      setShowWizard(true)
    }
  }, [])

  const commonGoals = [
    'Data Analysis & Insights',
    'Machine Learning Models',
    'Business Intelligence',
    'Process Automation',
    'Technical Architecture',
    'Cost Optimization',
    'Performance Monitoring',
    'Predictive Analytics'
  ]

  const experienceLevels = [
    { value: 'beginner', label: 'Beginner', description: 'New to data science and AI' },
    { value: 'intermediate', label: 'Intermediate', description: 'Some experience with data tools' },
    { value: 'advanced', label: 'Advanced', description: 'Experienced with data science and ML' },
    { value: 'expert', label: 'Expert', description: 'Deep expertise in AI/ML systems' }
  ]

  const createPersonalizedOnboarding = (): OnboardingStep[] => {
    const steps: OnboardingStep[] = [
      {
        id: 'welcome',
        title: `Welcome to ScrollIntel, ${userProfile.name}!`,
        description: 'Let\'s get you started with a personalized tour of the platform.',
        content: (
          <div className="space-y-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-scrollintel-primary to-scrollintel-secondary rounded-full flex items-center justify-center mx-auto mb-4">
                <Sparkles className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">You're all set!</h3>
              <p className="text-muted-foreground">
                Based on your profile, we've customized this tour to focus on the features most relevant to you.
              </p>
            </div>
            
            <div className="bg-muted p-4 rounded-lg">
              <h4 className="font-medium mb-2">Your Profile Summary:</h4>
              <div className="space-y-2 text-sm">
                <div><strong>Role:</strong> {userProfile.role}</div>
                <div><strong>Company:</strong> {userProfile.company}</div>
                <div><strong>Experience:</strong> {userProfile.experience}</div>
                <div><strong>Goals:</strong> {userProfile.goals.join(', ')}</div>
              </div>
            </div>
          </div>
        ),
      },
      {
        id: 'dashboard-overview',
        title: 'Dashboard Overview',
        description: 'Your central hub for all AI agents and system metrics.',
        target: 'dashboard',
        content: (
          <div className="space-y-4">
            <p>The dashboard gives you a real-time view of:</p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Active AI agents and their status</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>System performance metrics</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Recent activity and notifications</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Quick access to key features</span>
              </li>
            </ul>
          </div>
        ),
      }
    ]

    // Add goal-specific steps
    if (userProfile.goals.includes('Data Analysis & Insights')) {
      steps.push({
        id: 'data-analysis',
        title: 'Data Analysis Features',
        description: 'Learn how to upload and analyze your data.',
        target: 'file-upload',
        content: (
          <div className="space-y-4">
            <p>Perfect for your data analysis goals! You can:</p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Upload CSV, Excel, and JSON files</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Get automatic exploratory data analysis</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Generate insights and visualizations</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Export results as reports</span>
              </li>
            </ul>
          </div>
        ),
      })
    }

    if (userProfile.goals.includes('Machine Learning Models')) {
      steps.push({
        id: 'ml-features',
        title: 'Machine Learning Capabilities',
        description: 'Discover our ML model building and deployment features.',
        target: 'agents',
        content: (
          <div className="space-y-4">
            <p>Great choice for ML development! Our platform offers:</p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Automated model selection and training</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Model performance evaluation</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>One-click model deployment</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-scrollintel-primary" />
                <span>Real-time monitoring and alerts</span>
              </li>
            </ul>
          </div>
        ),
      })
    }

    // Add experience-level specific guidance
    if (userProfile.experience === 'beginner') {
      steps.push({
        id: 'beginner-tips',
        title: 'Getting Started Tips',
        description: 'Essential tips for new users.',
        content: (
          <div className="space-y-4">
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <h4 className="font-medium text-blue-900 mb-2">💡 Pro Tips for Beginners:</h4>
              <ul className="space-y-2 text-sm text-blue-800">
                <li>• Start with our sample datasets to explore features</li>
                <li>• Use the chat interface to ask questions in plain English</li>
                <li>• Check out agent tutorials for step-by-step guidance</li>
                <li>• Don't hesitate to use the help tooltips throughout the interface</li>
              </ul>
            </div>
          </div>
        ),
      })
    }

    steps.push({
      id: 'next-steps',
      title: 'Ready to Get Started!',
      description: 'Here are your recommended next steps.',
      content: (
        <div className="space-y-4">
          <div className="text-center">
            <Rocket className="h-12 w-12 text-scrollintel-primary mx-auto mb-4" />
            <h3 className="font-semibold mb-2">You're ready to go!</h3>
          </div>
          
          <div className="space-y-3">
            <h4 className="font-medium">Recommended next steps:</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-3 p-3 bg-muted rounded-lg">
                <div className="w-6 h-6 bg-scrollintel-primary rounded-full flex items-center justify-center text-white text-xs font-bold">1</div>
                <span className="text-sm">Try uploading a sample dataset</span>
              </div>
              <div className="flex items-center gap-3 p-3 bg-muted rounded-lg">
                <div className="w-6 h-6 bg-scrollintel-primary rounded-full flex items-center justify-center text-white text-xs font-bold">2</div>
                <span className="text-sm">Chat with an AI agent about your data</span>
              </div>
              <div className="flex items-center gap-3 p-3 bg-muted rounded-lg">
                <div className="w-6 h-6 bg-scrollintel-primary rounded-full flex items-center justify-center text-white text-xs font-bold">3</div>
                <span className="text-sm">Explore the generated insights and reports</span>
              </div>
            </div>
          </div>
        </div>
      ),
    })

    return steps
  }

  const handleStartOnboarding = () => {
    const steps = createPersonalizedOnboarding()
    startOnboarding(steps)
    setShowWizard(false)
    localStorage.setItem('scrollintel-welcome-wizard-completed', 'true')
  }

  const handleSkipWizard = () => {
    setShowWizard(false)
    localStorage.setItem('scrollintel-welcome-wizard-completed', 'true')
  }

  const toggleGoal = (goal: string) => {
    setUserProfile(prev => ({
      ...prev,
      goals: prev.goals.includes(goal)
        ? prev.goals.filter(g => g !== goal)
        : [...prev.goals, goal]
    }))
  }

  if (!showWizard) {
    return (
      <Button
        onClick={() => setShowWizard(true)}
        variant="outline"
        className="fixed bottom-4 right-4 z-40"
      >
        <Rocket className="h-4 w-4 mr-2" />
        Start Tour
      </Button>
    )
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-auto">
        <CardHeader className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-scrollintel-primary to-scrollintel-secondary rounded-full flex items-center justify-center mx-auto mb-4">
            <Zap className="h-8 w-8 text-white" />
          </div>
          <CardTitle className="text-2xl">Welcome to ScrollIntel!</CardTitle>
          <p className="text-muted-foreground">
            Let's personalize your experience. Tell us a bit about yourself to get started.
          </p>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Basic Information */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <User className="h-5 w-5 text-scrollintel-primary" />
              <h3 className="font-semibold">Basic Information</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="name">Your Name</Label>
                <Input
                  id="name"
                  placeholder="Enter your name"
                  value={userProfile.name}
                  onChange={(e) => setUserProfile(prev => ({ ...prev, name: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="role">Your Role</Label>
                <Input
                  id="role"
                  placeholder="e.g., Data Analyst, CTO, Product Manager"
                  value={userProfile.role}
                  onChange={(e) => setUserProfile(prev => ({ ...prev, role: e.target.value }))}
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="company">Company</Label>
              <Input
                id="company"
                placeholder="Your company name"
                value={userProfile.company}
                onChange={(e) => setUserProfile(prev => ({ ...prev, company: e.target.value }))}
              />
            </div>
          </div>

          {/* Goals */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Target className="h-5 w-5 text-scrollintel-primary" />
              <h3 className="font-semibold">What are your main goals?</h3>
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              {commonGoals.map((goal) => (
                <Button
                  key={goal}
                  variant={userProfile.goals.includes(goal) ? "default" : "outline"}
                  size="sm"
                  onClick={() => toggleGoal(goal)}
                  className="justify-start h-auto p-3 text-left"
                >
                  <div className="flex items-center gap-2">
                    {userProfile.goals.includes(goal) && (
                      <CheckCircle className="h-4 w-4" />
                    )}
                    <span className="text-xs">{goal}</span>
                  </div>
                </Button>
              ))}
            </div>
          </div>

          {/* Experience Level */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Building className="h-5 w-5 text-scrollintel-primary" />
              <h3 className="font-semibold">Experience Level</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {experienceLevels.map((level) => (
                <Button
                  key={level.value}
                  variant={userProfile.experience === level.value ? "default" : "outline"}
                  onClick={() => setUserProfile(prev => ({ ...prev, experience: level.value }))}
                  className="h-auto p-4 flex-col items-start"
                >
                  <div className="font-medium">{level.label}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {level.description}
                  </div>
                </Button>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between pt-4 border-t">
            <Button variant="ghost" onClick={handleSkipWizard}>
              Skip for now
            </Button>
            
            <Button 
              onClick={handleStartOnboarding}
              disabled={!userProfile.name || userProfile.goals.length === 0}
            >
              Start Personalized Tour
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}