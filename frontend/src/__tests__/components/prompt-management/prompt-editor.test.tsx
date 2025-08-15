import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import { PromptEditor } from '@/components/prompt-management/prompt-editor'

const mockPrompt = {
  id: '1',
  name: 'Test Prompt',
  content: 'Generate content about {topic} for {audience}',
  category: 'content',
  tags: ['test', 'content'],
  variables: [
    { name: 'topic', type: 'string', description: 'The topic to write about', required: true },
    { name: 'audience', type: 'string', description: 'Target audience', required: true }
  ],
  created_by: 'test-user',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z'
}

const mockProps = {
  onSave: jest.fn(),
  onTest: jest.fn(),
  onVersionHistory: jest.fn()
}

describe('PromptEditor', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders without crashing', () => {
    render(<PromptEditor {...mockProps} />)
    expect(screen.getByPlaceholderText('Prompt name')).toBeInTheDocument()
  })

  it('loads existing prompt data', () => {
    render(<PromptEditor prompt={mockPrompt} {...mockProps} />)
    
    expect(screen.getByDisplayValue('Test Prompt')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Generate content about {topic} for {audience}')).toBeInTheDocument()
    expect(screen.getByDisplayValue('content')).toBeInTheDocument()
  })

  it('validates prompt content', async () => {
    render(<PromptEditor {...mockProps} />)
    
    const contentTextarea = screen.getByPlaceholderText('Enter your prompt template here...')
    await userEvent.type(contentTextarea, 'Invalid prompt with {unclosed_variable')
    
    await waitFor(() => {
      expect(screen.getByText(/Unclosed variable brackets/)).toBeInTheDocument()
    })
  })

  it('detects undefined variables', async () => {
    render(<PromptEditor {...mockProps} />)
    
    const contentTextarea = screen.getByPlaceholderText('Enter your prompt template here...')
    await userEvent.type(contentTextarea, 'Prompt with {undefined_var}')
    
    await waitFor(() => {
      expect(screen.getByText(/Undefined variable: undefined_var/)).toBeInTheDocument()
    })
  })

  it('allows adding and removing tags', async () => {
    render(<PromptEditor {...mockProps} />)
    
    const tagInput = screen.getByPlaceholderText('Add tag')
    const addButton = screen.getByRole('button', { name: 'Add' })
    
    await userEvent.type(tagInput, 'new-tag')
    await userEvent.click(addButton)
    
    expect(screen.getByText('new-tag ×')).toBeInTheDocument()
    
    // Remove tag
    await userEvent.click(screen.getByText('new-tag ×'))
    expect(screen.queryByText('new-tag ×')).not.toBeInTheDocument()
  })

  it('manages variables correctly', async () => {
    render(<PromptEditor {...mockProps} />)
    
    // Switch to variables tab
    await userEvent.click(screen.getByRole('tab', { name: 'Variables' }))
    
    // Add variable
    await userEvent.click(screen.getByRole('button', { name: 'Add Variable' }))
    
    const nameInputs = screen.getAllByPlaceholderText('variable_name')
    await userEvent.type(nameInputs[0], 'test_var')
    
    expect(screen.getByDisplayValue('test_var')).toBeInTheDocument()
  })

  it('calls onSave when save button is clicked', async () => {
    render(<PromptEditor {...mockProps} />)
    
    const nameInput = screen.getByPlaceholderText('Prompt name')
    await userEvent.type(nameInput, 'New Prompt')
    
    const saveButton = screen.getByRole('button', { name: /Save/ })
    await userEvent.click(saveButton)
    
    expect(mockProps.onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'New Prompt'
      })
    )
  })

  it('calls onTest when test button is clicked', async () => {
    render(<PromptEditor prompt={mockPrompt} {...mockProps} />)
    
    const testButton = screen.getByRole('button', { name: /Test/ })
    await userEvent.click(testButton)
    
    expect(mockProps.onTest).toHaveBeenCalledWith(mockPrompt)
  })

  it('calls onVersionHistory when history button is clicked', async () => {
    render(<PromptEditor prompt={mockPrompt} {...mockProps} />)
    
    const historyButton = screen.getByRole('button', { name: /History/ })
    await userEvent.click(historyButton)
    
    expect(mockProps.onVersionHistory).toHaveBeenCalledWith('1')
  })

  it('shows validation status badge', async () => {
    render(<PromptEditor {...mockProps} />)
    
    // Initially valid (empty content)
    expect(screen.getByText('Valid')).toBeInTheDocument()
    
    // Add invalid content
    const contentTextarea = screen.getByPlaceholderText('Enter your prompt template here...')
    await userEvent.type(contentTextarea, '{invalid')
    
    await waitFor(() => {
      expect(screen.getByText(/1 issues/)).toBeInTheDocument()
    })
  })

  it('handles category and metadata updates', async () => {
    render(<PromptEditor {...mockProps} />)
    
    const categoryInput = screen.getByPlaceholderText('e.g., content-generation')
    await userEvent.type(categoryInput, 'test-category')
    
    expect(screen.getByDisplayValue('test-category')).toBeInTheDocument()
  })

  it('switches between tabs correctly', async () => {
    render(<PromptEditor {...mockProps} />)
    
    // Switch to variables tab
    await userEvent.click(screen.getByRole('tab', { name: 'Variables' }))
    expect(screen.getByText('Variables')).toBeInTheDocument()
    
    // Switch to test tab
    await userEvent.click(screen.getByRole('tab', { name: 'Test' }))
    expect(screen.getByText('Test Output')).toBeInTheDocument()
  })
})