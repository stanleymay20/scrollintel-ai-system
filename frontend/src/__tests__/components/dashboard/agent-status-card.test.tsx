import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import '@testing-library/jest-dom'
import { AgentStatusCard } from '@/components/dashboard/agent-status-card'
import { Agent } from '@/types'

const mockAgent: Agent = {
  id: 'test-agent-1',
  name: 'TestAgent',
  type: 'TestType',
  status: 'active',
  capabilities: ['Test Capability 1', 'Test Capability 2', 'Test Capability 3', 'Test Capability 4'],
  description: 'Test agent description',
  last_active: '2024-01-01T00:00:00Z',
  metrics: {
    requests_handled: 100,
    avg_response_time: 500,
    success_rate: 95.5,
  },
}

describe('AgentStatusCard', () => {
  it('renders agent information correctly', () => {
    render(<AgentStatusCard agent={mockAgent} />)
    
    expect(screen.getByText('TestAgent')).toBeInTheDocument()
    expect(screen.getByText('Test agent description')).toBeInTheDocument()
    expect(screen.getByText('active')).toBeInTheDocument()
  })

  it('displays agent capabilities with truncation', () => {
    render(<AgentStatusCard agent={mockAgent} />)
    
    expect(screen.getByText('Test Capability 1')).toBeInTheDocument()
    expect(screen.getByText('Test Capability 2')).toBeInTheDocument()
    expect(screen.getByText('Test Capability 3')).toBeInTheDocument()
    expect(screen.getByText('+1 more')).toBeInTheDocument()
  })

  it('displays agent metrics when available', () => {
    render(<AgentStatusCard agent={mockAgent} />)
    
    expect(screen.getByText('100')).toBeInTheDocument()
    expect(screen.getByText('500ms')).toBeInTheDocument()
    expect(screen.getByText('95.5%')).toBeInTheDocument()
  })

  it('calls onInteract when interact button is clicked', () => {
    const mockOnInteract = jest.fn()
    render(<AgentStatusCard agent={mockAgent} onInteract={mockOnInteract} />)
    
    const interactButton = screen.getByText('Interact with TestAgent')
    fireEvent.click(interactButton)
    
    expect(mockOnInteract).toHaveBeenCalledWith('test-agent-1')
  })

  it('disables interact button for inactive agents', () => {
    const inactiveAgent = { ...mockAgent, status: 'inactive' as const }
    render(<AgentStatusCard agent={inactiveAgent} />)
    
    const interactButton = screen.getByText('Interact with TestAgent')
    expect(interactButton).toBeDisabled()
  })

  it('applies correct styling for different agent statuses', () => {
    const { rerender } = render(<AgentStatusCard agent={mockAgent} />)
    
    // Active agent should have success badge
    expect(screen.getByText('active')).toBeInTheDocument()
    
    // Test busy status
    const busyAgent = { ...mockAgent, status: 'busy' as const }
    rerender(<AgentStatusCard agent={busyAgent} />)
    expect(screen.getByText('busy')).toBeInTheDocument()
    
    // Test error status
    const errorAgent = { ...mockAgent, status: 'error' as const }
    rerender(<AgentStatusCard agent={errorAgent} />)
    expect(screen.getByText('error')).toBeInTheDocument()
  })
})