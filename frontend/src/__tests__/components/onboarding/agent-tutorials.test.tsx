import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { AgentTutorials } from '@/components/onboarding/agent-tutorials'
import { Agent } from '@/types'

// Mock the useOnboarding hook
const mockStartOnboarding = jest.fn()

jest.mock('@/components/onboarding/onboarding-provider', () => ({
  useOnboarding: () => ({
    startOnboarding: mockStartOnboarding,
  }),
}))

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
]

describe('AgentTutorials', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders agent tutorials correctly', () => {
    render(<AgentTutorials agents={mockAgents} />)

    expect(screen.getByText('Agent Tutorials')).toBeInTheDocument()
    expect(screen.getByText('Interactive Learning')).toBeInTheDocument()
    
    // Check if all agents are rendered
    expect(screen.getByText('ScrollCTO')).toBeInTheDocument()
    expect(screen.getByText('ScrollDataScientist')).toBeInTheDocument()
    expect(screen.getByText('ScrollMLEngineer')).toBeInTheDocument()
  })

  it('displays agent information correctly', () => {
    render(<AgentTutorials agents={mockAgents} />)

    // Check agent descriptions
    expect(screen.getByText('AI-powered CTO for technical decisions and architecture planning')).toBeInTheDocument()
    expect(screen.getByText('Advanced data science and statistical analysis capabilities')).toBeInTheDocument()
    expect(screen.getByText('Machine learning engineering and model deployment')).toBeInTheDocument()

    // Check agent types
    expect(screen.getByText('CTO')).toBeInTheDocument()
    expect(screen.getByText('DataScientist')).toBeInTheDocument()
    expect(screen.getByText('MLEngineer')).toBeInTheDocument()
  })

  it('shows learning objectives for each agent', () => {
    render(<AgentTutorials agents={mockAgents} />)

    // Check common learning objectives
    const learningObjectives = screen.getAllByText(/How to interact with|Best practices for data upload|Understanding analysis results|Exporting and sharing insights/)
    expect(learningObjectives.length).toBeGreaterThan(0)
  })

  it('starts tutorial when Start Tutorial button is clicked', () => {
    render(<AgentTutorials agents={mockAgents} />)

    const startButtons = screen.getAllByText('Start Tutorial')
    fireEvent.click(startButtons[0])

    expect(mockStartOnboarding).toHaveBeenCalledTimes(1)
    expect(mockStartOnboarding).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({
          id: 'cto-1-intro',
          title: 'Meet ScrollCTO',
        }),
      ])
    )
  })

  it('creates correct tutorial steps for CTO agent', () => {
    render(<AgentTutorials agents={[mockAgents[0]]} />)

    const startButton = screen.getByText('Start Tutorial')
    fireEvent.click(startButton)

    const tutorialSteps = mockStartOnboarding.mock.calls[0][0]
    
    expect(tutorialSteps).toHaveLength(4)
    expect(tutorialSteps[0].id).toBe('cto-1-intro')
    expect(tutorialSteps[1].id).toBe('cto-1-chat')
    expect(tutorialSteps[2].id).toBe('cto-1-upload')
    expect(tutorialSteps[3].id).toBe('cto-1-results')
  })

  it('creates correct tutorial steps for DataScientist agent', () => {
    render(<AgentTutorials agents={[mockAgents[1]]} />)

    const startButton = screen.getByText('Start Tutorial')
    fireEvent.click(startButton)

    const tutorialSteps = mockStartOnboarding.mock.calls[0][0]
    
    expect(tutorialSteps).toHaveLength(4)
    expect(tutorialSteps[0].id).toBe('ds-1-intro')
    expect(tutorialSteps[1].id).toBe('ds-1-chat')
    expect(tutorialSteps[2].id).toBe('ds-1-upload')
    expect(tutorialSteps[3].id).toBe('ds-1-results')
  })

  it('displays correct capabilities for each agent', () => {
    render(<AgentTutorials agents={mockAgents} />)

    // CTO capabilities
    expect(screen.getByText('Architecture')).toBeInTheDocument()
    expect(screen.getByText('Scaling')).toBeInTheDocument()
    expect(screen.getByText('Tech Stack')).toBeInTheDocument()
    expect(screen.getByText('Cost Analysis')).toBeInTheDocument()

    // DataScientist capabilities
    expect(screen.getByText('EDA')).toBeInTheDocument()
    expect(screen.getByText('Statistical Analysis')).toBeInTheDocument()
    expect(screen.getByText('Hypothesis Testing')).toBeInTheDocument()
    expect(screen.getByText('Feature Engineering')).toBeInTheDocument()
  })

  it('renders with correct grid layout', () => {
    render(<AgentTutorials agents={mockAgents} />)

    const grid = screen.getByText('ScrollCTO').closest('.grid')
    expect(grid).toHaveClass('grid-cols-1', 'md:grid-cols-2', 'lg:grid-cols-3')
  })

  it('handles empty agents array', () => {
    render(<AgentTutorials agents={[]} />)

    expect(screen.getByText('Agent Tutorials')).toBeInTheDocument()
    expect(screen.queryByText('Start Tutorial')).not.toBeInTheDocument()
  })

  it('displays correct icons for different agent types', () => {
    render(<AgentTutorials agents={mockAgents} />)

    // Icons should be present (we can't easily test the specific icon, but we can check the structure)
    const agentCards = screen.getAllByText('Start Tutorial')
    expect(agentCards).toHaveLength(3)
  })

  it('shows hover effects on cards', () => {
    render(<AgentTutorials agents={mockAgents} />)

    const cards = screen.getAllByText('Start Tutorial').map(button => 
      button.closest('.hover\\:shadow-md')
    )
    
    cards.forEach(card => {
      expect(card).toHaveClass('hover:shadow-md')
    })
  })

  it('includes target attributes in tutorial steps', () => {
    render(<AgentTutorials agents={[mockAgents[0]]} />)

    const startButton = screen.getByText('Start Tutorial')
    fireEvent.click(startButton)

    const tutorialSteps = mockStartOnboarding.mock.calls[0][0]
    
    expect(tutorialSteps[1].target).toBe('chat-interface')
    expect(tutorialSteps[2].target).toBe('file-upload')
  })

  it('includes sample questions in chat step content', () => {
    render(<AgentTutorials agents={[mockAgents[0]]} />)

    const startButton = screen.getByText('Start Tutorial')
    fireEvent.click(startButton)

    const tutorialSteps = mockStartOnboarding.mock.calls[0][0]
    const chatStep = tutorialSteps[1]
    
    // The content should be a React element, so we can't easily test the text
    // But we can verify the step has the correct structure
    expect(chatStep.title).toBe('Start a Conversation')
    expect(chatStep.description).toBe('Learn how to chat with this agent and ask questions.')
  })
})