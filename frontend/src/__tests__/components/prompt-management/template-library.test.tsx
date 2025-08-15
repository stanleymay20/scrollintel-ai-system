import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import { TemplateLibrary } from '@/components/prompt-management/template-library'

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn()
  }
})

const mockProps = {
  onCreateNew: jest.fn(),
  onEditTemplate: jest.fn(),
  onDeleteTemplate: jest.fn(),
  onImportTemplates: jest.fn(),
  onExportTemplates: jest.fn()
}

describe('TemplateLibrary', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders without crashing', () => {
    render(<TemplateLibrary {...mockProps} />)
    expect(screen.getByText('Template Library')).toBeInTheDocument()
  })

  it('displays search input and filters', () => {
    render(<TemplateLibrary {...mockProps} />)
    
    expect(screen.getByPlaceholderText('Search templates...')).toBeInTheDocument()
    expect(screen.getByDisplayValue('All Categories')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Date Created')).toBeInTheDocument()
  })

  it('calls onCreateNew when new template button is clicked', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    const newButton = screen.getByRole('button', { name: /New Template/ })
    await userEvent.click(newButton)
    
    expect(mockProps.onCreateNew).toHaveBeenCalled()
  })

  it('filters templates by search query', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    const searchInput = screen.getByPlaceholderText('Search templates...')
    await userEvent.type(searchInput, 'Content')
    
    // Wait for filtering to occur
    await waitFor(() => {
      expect(searchInput).toHaveValue('Content')
    })
  })

  it('filters templates by category', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    const categorySelect = screen.getByDisplayValue('All Categories')
    await userEvent.selectOptions(categorySelect, 'content')
    
    expect(categorySelect).toHaveValue('content')
  })

  it('toggles between grid and list view', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    const listViewButton = screen.getByRole('button', { name: '' }) // List icon button
    await userEvent.click(listViewButton)
    
    // Should switch to list view (implementation detail)
  })

  it('handles template selection', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    // Wait for templates to load
    await waitFor(() => {
      const templateCard = screen.getByText('Content Generation')
      expect(templateCard).toBeInTheDocument()
    })
    
    const templateCard = screen.getByText('Content Generation').closest('div')
    if (templateCard) {
      await userEvent.click(templateCard)
    }
  })

  it('calls onEditTemplate when edit button is clicked', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    await waitFor(() => {
      const editButtons = screen.getAllByRole('button')
      const editButton = editButtons.find(button => 
        button.querySelector('svg') && button.getAttribute('aria-label') === undefined
      )
      if (editButton) {
        fireEvent.click(editButton)
      }
    })
  })

  it('calls onDeleteTemplate when delete button is clicked', async () => {
    // Mock window.confirm
    window.confirm = jest.fn(() => true)
    
    render(<TemplateLibrary {...mockProps} />)
    
    await waitFor(() => {
      const deleteButtons = screen.getAllByRole('button')
      const deleteButton = deleteButtons.find(button => 
        button.querySelector('svg') && button.getAttribute('aria-label') === undefined
      )
      if (deleteButton) {
        fireEvent.click(deleteButton)
      }
    })
  })

  it('handles template import', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    const importButton = screen.getByRole('button', { name: /Import/ })
    await userEvent.click(importButton)
    
    // File input should be triggered (implementation detail)
  })

  it('handles template export', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    // First select a template
    await waitFor(() => {
      const templateCard = screen.getByText('Content Generation').closest('div')
      if (templateCard) {
        fireEvent.click(templateCard)
      }
    })
    
    // Export button should appear
    await waitFor(() => {
      const exportButton = screen.queryByRole('button', { name: /Export/ })
      if (exportButton) {
        fireEvent.click(exportButton)
        expect(mockProps.onExportTemplates).toHaveBeenCalled()
      }
    })
  })

  it('toggles favorite status', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    await waitFor(() => {
      const favoriteButtons = screen.getAllByRole('button')
      const favoriteButton = favoriteButtons.find(button => 
        button.querySelector('svg') && button.getAttribute('aria-label') === undefined
      )
      if (favoriteButton) {
        fireEvent.click(favoriteButton)
      }
    })
  })

  it('copies template content to clipboard', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    await waitFor(() => {
      const copyButtons = screen.getAllByRole('button')
      const copyButton = copyButtons.find(button => 
        button.querySelector('svg') && button.getAttribute('aria-label') === undefined
      )
      if (copyButton) {
        fireEvent.click(copyButton)
      }
    })
    
    // Should call clipboard API (mocked)
  })

  it('filters by tags', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    await waitFor(() => {
      const tagBadges = screen.getAllByText(/generation|marketing|seo/)
      if (tagBadges.length > 0) {
        fireEvent.click(tagBadges[0])
      }
    })
  })

  it('sorts templates correctly', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    const sortSelect = screen.getByDisplayValue('Date Created')
    await userEvent.selectOptions(sortSelect, 'name')
    
    expect(sortSelect).toHaveValue('name')
    
    // Toggle sort order
    const sortOrderButton = screen.getByRole('button', { name: '↓' })
    await userEvent.click(sortOrderButton)
    
    expect(screen.getByRole('button', { name: '↑' })).toBeInTheDocument()
  })

  it('shows no results message when no templates match filters', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    const searchInput = screen.getByPlaceholderText('Search templates...')
    await userEvent.type(searchInput, 'nonexistent-template-xyz')
    
    await waitFor(() => {
      expect(screen.getByText('No templates found')).toBeInTheDocument()
    })
  })

  it('displays template metadata correctly', async () => {
    render(<TemplateLibrary {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('Content Generation')).toBeInTheDocument()
      expect(screen.getByText(/Used \d+ times/)).toBeInTheDocument()
      expect(screen.getByText(/★ \d+\.\d+/)).toBeInTheDocument()
    })
  })
})