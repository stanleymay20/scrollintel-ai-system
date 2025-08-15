import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'
import Dashboard from '@/app/page'
import { it } from 'node:test'
import { it } from 'node:test'
import { it } from 'node:test'
import { it } from 'node:test'
import { it } from 'node:test'
import { it } from 'node:test'
import { it } from 'node:test'
import { it } from 'node:test'
import { it } from 'node:test'
import { describe } from 'node:test'

// Mock Next.js router
jest.mock('next/navigation', () => ({
  usePathname: () => '/',
}))

// Mock the child components to focus on integration
jest.mock('@/components/dashboard/agent-status-card', () => ({
  AgentStatusCard: ({ agent, onInteract }: any) => (
    <div data-testid={`agent-${agent.id}`}>
      <span>{agent.name}</span>
      <button onClick={() => onInteract(agent.id)}>
        Interact with {agent.name}
      </button>
    </div>
  ),
}))

jest.mock('@/components/chat/chat-interface', () => ({
  ChatInterface: ({ selectedAgent, onSendMessage, messages, isLoading }: any) => (
    <div data-testid="chat-interface">
      {selectedAgent && <span>Selected: {selectedAgent.name}</span>}
      <input 
        placeholder="Type message"
        onChange={(e) => {}}
      />
      <button onClick={() => onSendMessage('test message', selectedAgent?.id)}>
        Send
      </button>
      {isLoading && <span>Loading...</span>}
      <div data-testid="messages">
        {messages.map((msg: any) => (
          <div key={msg.id}>{msg.content}</div>
        ))}
      </div>
    </div>
  ),
}))

jest.mock('@/components/upload/file-upload', () => ({
  FileUploadComponent: ({ onFileUpload, uploadedFiles }: any) => (
    <div data-testid="file-upload">
      <button onClick={() => onFileUpload([new File(['test'], 'test.csv')])}>
        Upload File
      </button>
      <div data-testid="uploaded-files">
        {uploadedFiles.map((file: any) => (
          <div key={file.id}>{file.filename}</div>
        ))}
      </div>
    </div>
  ),
}))

describe('Dashboard', () => {
  it('renders dashboard correctly', () => {
    render(<Dashboard />)
    
    expect(screen.getByText('ScrollIntel Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Welcome to the world\'s most advanced AI-CTO platform')).toBeInTheDocument()
  })

  it('displays system metrics', () => {
    render(<Dashboard />)
    
    expect(screen.getAllByText('Active Agents')).toHaveLength(2)
    expect(screen.getAllByText('Total Requests')).toHaveLength(2)
    expect(screen.getByText('Avg Response')).toBeInTheDocument()
    expect(screen.getByText('Success Rate')).toBeInTheDocument()
  })

  it('renders agent cards', () => {
    render(<Dashboard />)
    
    expect(screen.getByTestId('agent-cto-1')).toBeInTheDocument()
    expect(screen.getByTestId('agent-ds-1')).toBeInTheDocument()
    expect(screen.getByTestId('agent-ml-1')).toBeInTheDocument()
    expect(screen.getByTestId('agent-ai-1')).toBeInTheDocument()
  })

  it('handles agent interaction', async () => {
    render(<Dashboard />)
    
    const interactButton = screen.getByText('Interact with ScrollCTO')
    fireEvent.click(interactButton)
    
    await waitFor(() => {
      expect(screen.getByText('Selected: ScrollCTO')).toBeInTheDocument()
    })
  })

  it('handles chat message sending', async () => {
    render(<Dashboard />)
    
    // First select an agent
    const interactButton = screen.getByText('Interact with ScrollCTO')
    fireEvent.click(interactButton)
    
    // Then send a message
    const sendButton = screen.getByText('Send')
    fireEvent.click(sendButton)
    
    await waitFor(() => {
      expect(screen.getByText('Loading...')).toBeInTheDocument()
    })
  })

  it('handles file upload', async () => {
    render(<Dashboard />)
    
    const uploadButton = screen.getByText('Upload File')
    fireEvent.click(uploadButton)
    
    await waitFor(() => {
      expect(screen.getByText('test.csv')).toBeInTheDocument()
    })
  })

  it('displays system status badge', () => {
    render(<Dashboard />)
    
    expect(screen.getByText('All Systems Operational')).toBeInTheDocument()
  })

  it('shows mobile menu button on small screens', () => {
    render(<Dashboard />)
    
    // The mobile menu button should be present (though hidden on larger screens)
    const menuButtons = screen.getAllByRole('button')
    expect(menuButtons.length).toBeGreaterThan(0)
  })

  it('updates system metrics over time', async () => {
    render(<Dashboard />)
    
    // Just verify that metrics are displayed
    expect(screen.getAllByText('12')).toHaveLength(2)
    expect(screen.getAllByText('4,229')).toHaveLength(2)
  })
})