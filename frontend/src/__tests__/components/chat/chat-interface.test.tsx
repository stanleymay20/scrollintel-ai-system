import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import { ChatInterface } from '@/components/chat/chat-interface'
import { Agent, ChatMessage } from '@/types'
import { it } from 'node:test'
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

const mockAgent: Agent = {
  id: 'test-agent-1',
  name: 'TestAgent',
  type: 'TestType',
  status: 'active',
  capabilities: ['Test Capability'],
  description: 'Test agent description',
  last_active: '2024-01-01T00:00:00Z',
}

const mockMessages: ChatMessage[] = [
  {
    id: '1',
    role: 'user',
    content: 'Hello, test message',
    timestamp: '2024-01-01T00:00:00Z',
  },
  {
    id: '2',
    role: 'assistant',
    content: 'Hello! How can I help you?',
    timestamp: '2024-01-01T00:01:00Z',
    agent_id: 'test-agent-1',
    metadata: {
      processing_time: 500,
    },
  },
]

describe('ChatInterface', () => {
  it('renders chat interface correctly', () => {
    render(<ChatInterface />)
    
    expect(screen.getByText('ScrollIntel Chat')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument()
  })

  it('displays selected agent information', () => {
    render(<ChatInterface selectedAgent={mockAgent} />)
    
    expect(screen.getByText('TestAgent')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Ask TestAgent anything...')).toBeInTheDocument()
  })

  it('displays chat messages correctly', () => {
    render(<ChatInterface messages={mockMessages} />)
    
    expect(screen.getByText('Hello, test message')).toBeInTheDocument()
    expect(screen.getByText('Hello! How can I help you?')).toBeInTheDocument()
  })

  it('shows empty state when no messages', () => {
    render(<ChatInterface />)
    
    expect(screen.getByText('Start a conversation with ScrollIntel agents')).toBeInTheDocument()
    expect(screen.getByText('Ask questions, upload files, or request analysis')).toBeInTheDocument()
  })

  it('calls onSendMessage when message is sent', async () => {
    const user = userEvent.setup()
    const mockOnSendMessage = jest.fn()
    
    render(
      <ChatInterface 
        selectedAgent={mockAgent}
        onSendMessage={mockOnSendMessage}
      />
    )
    
    const input = screen.getByPlaceholderText('Ask TestAgent anything...')
    const sendButton = screen.getByRole('button')
    
    await user.type(input, 'Test message')
    await user.click(sendButton)
    
    expect(mockOnSendMessage).toHaveBeenCalledWith('Test message', 'test-agent-1')
  })

  it('sends message on Enter key press', async () => {
    const user = userEvent.setup()
    const mockOnSendMessage = jest.fn()
    
    render(
      <ChatInterface 
        selectedAgent={mockAgent}
        onSendMessage={mockOnSendMessage}
      />
    )
    
    const input = screen.getByPlaceholderText('Ask TestAgent anything...')
    
    await user.type(input, 'Test message{enter}')
    
    expect(mockOnSendMessage).toHaveBeenCalledWith('Test message', 'test-agent-1')
  })

  it('clears input after sending message', async () => {
    const user = userEvent.setup()
    const mockOnSendMessage = jest.fn()
    
    render(
      <ChatInterface 
        selectedAgent={mockAgent}
        onSendMessage={mockOnSendMessage}
      />
    )
    
    const input = screen.getByPlaceholderText('Ask TestAgent anything...') as HTMLInputElement
    
    await user.type(input, 'Test message')
    await user.keyboard('{enter}')
    
    expect(input.value).toBe('')
  })

  it('disables input and button when loading', () => {
    render(
      <ChatInterface 
        selectedAgent={mockAgent}
        isLoading={true}
      />
    )
    
    const input = screen.getByPlaceholderText('Ask TestAgent anything...')
    const sendButton = screen.getByRole('button')
    
    expect(input).toBeDisabled()
    expect(sendButton).toBeDisabled()
  })

  it('shows loading indicator when isLoading is true', () => {
    render(
      <ChatInterface 
        selectedAgent={mockAgent}
        isLoading={true}
      />
    )
    
    expect(screen.getByText('TestAgent is thinking...')).toBeInTheDocument()
  })

  it('displays message metadata when available', () => {
    render(<ChatInterface messages={mockMessages} />)
    
    expect(screen.getByText(/500/)).toBeInTheDocument()
    expect(screen.getByText(/ms/)).toBeInTheDocument()
  })
})