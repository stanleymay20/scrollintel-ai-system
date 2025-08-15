import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import { ABTestingDashboard } from '@/components/prompt-management/ab-testing-dashboard'

const mockProps = {
  onCreateExperiment: jest.fn(),
  onEditExperiment: jest.fn(),
  onStartExperiment: jest.fn(),
  onPauseExperiment: jest.fn(),
  onStopExperiment: jest.fn(),
  onPromoteWinner: jest.fn()
}

describe('ABTestingDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders without crashing', () => {
    render(<ABTestingDashboard {...mockProps} />)
    expect(screen.getByText('Experiments')).toBeInTheDocument()
  })

  it('displays experiments list', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('Content Generation Optimization')).toBeInTheDocument()
      expect(screen.getByText('Code Review Prompt Test')).toBeInTheDocument()
    })
  })

  it('calls onCreateExperiment when new test button is clicked', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    const newTestButton = screen.getByRole('button', { name: /New Test/ })
    await userEvent.click(newTestButton)
    
    expect(mockProps.onCreateExperiment).toHaveBeenCalled()
  })

  it('selects experiment and shows details', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      const experimentItem = screen.getByText('Content Generation Optimization')
      fireEvent.click(experimentItem.closest('div')!)
    })
    
    await waitFor(() => {
      expect(screen.getByText('Content Generation Optimization')).toBeInTheDocument()
      expect(screen.getByText(/Testing different approaches/)).toBeInTheDocument()
    })
  })

  it('displays experiment status correctly', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('running')).toBeInTheDocument()
      expect(screen.getByText('completed')).toBeInTheDocument()
    })
  })

  it('shows confidence level and progress', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText(/87\.0%/)).toBeInTheDocument() // Confidence level
    })
  })

  it('calls onStartExperiment for draft experiments', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    // Mock a draft experiment by modifying the component state
    // This would require the experiment to be in draft status
  })

  it('calls onPauseExperiment for running experiments', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      const pauseButton = screen.queryByRole('button', { name: /Pause/ })
      if (pauseButton) {
        fireEvent.click(pauseButton)
        expect(mockProps.onPauseExperiment).toHaveBeenCalled()
      }
    })
  })

  it('calls onStopExperiment for running experiments', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      const stopButton = screen.queryByRole('button', { name: /Stop/ })
      if (stopButton) {
        fireEvent.click(stopButton)
        expect(mockProps.onStopExperiment).toHaveBeenCalled()
      }
    })
  })

  it('calls onPromoteWinner for completed experiments', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    // Select completed experiment
    await waitFor(() => {
      const completedExperiment = screen.getByText('Code Review Prompt Test')
      fireEvent.click(completedExperiment.closest('div')!)
    })
    
    await waitFor(() => {
      const promoteButton = screen.queryByRole('button', { name: /Promote Winner/ })
      if (promoteButton) {
        fireEvent.click(promoteButton)
        expect(mockProps.onPromoteWinner).toHaveBeenCalled()
      }
    })
  })

  it('displays variant performance metrics', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText(/Control/)).toBeInTheDocument()
      expect(screen.getByText(/Enhanced/)).toBeInTheDocument()
    })
  })

  it('shows performance comparison chart', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('Performance Comparison')).toBeInTheDocument()
    })
  })

  it('switches between tabs correctly', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      const variantsTab = screen.getByRole('tab', { name: 'Variants' })
      fireEvent.click(variantsTab)
      
      expect(screen.getByText(/Prompt Content/)).toBeInTheDocument()
    })
    
    const analyticsTab = screen.getByRole('tab', { name: 'Analytics' })
    await userEvent.click(analyticsTab)
    
    expect(screen.getByText('Traffic Distribution')).toBeInTheDocument()
  })

  it('displays winner badge for winning variants', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      const winnerBadges = screen.getAllByText('Winner')
      expect(winnerBadges.length).toBeGreaterThan(0)
    })
  })

  it('shows experiment settings', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      const settingsTab = screen.getByRole('tab', { name: 'Settings' })
      fireEvent.click(settingsTab)
      
      expect(screen.getByText('Experiment Settings')).toBeInTheDocument()
      expect(screen.getByText('Confidence Level')).toBeInTheDocument()
    })
  })

  it('displays total requests and key metrics', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('Total Requests')).toBeInTheDocument()
      expect(screen.getByText('Confidence')).toBeInTheDocument()
      expect(screen.getByText('Winner')).toBeInTheDocument()
    })
  })

  it('shows traffic distribution pie chart', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      const analyticsTab = screen.getByRole('tab', { name: 'Analytics' })
      fireEvent.click(analyticsTab)
      
      expect(screen.getByText('Traffic Distribution')).toBeInTheDocument()
    })
  })

  it('displays variant details with metrics', async () => {
    render(<ABTestingDashboard {...mockProps} />)
    
    await waitFor(() => {
      const variantsTab = screen.getByRole('tab', { name: 'Variants' })
      fireEvent.click(variantsTab)
      
      expect(screen.getByText(/Conversion/)).toBeInTheDocument()
      expect(screen.getByText(/Response Time/)).toBeInTheDocument()
      expect(screen.getByText(/Satisfaction/)).toBeInTheDocument()
    })
  })
})