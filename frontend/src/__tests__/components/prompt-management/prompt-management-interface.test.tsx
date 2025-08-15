import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import { PromptManagementInterface } from '@/components/prompt-management/prompt-management-interface'

// Mock the child components
jest.mock('@/components/prompt-management/prompt-editor', () => ({
  PromptEditor: ({ onSave, onTest, onVersionHistory }: any) => (
    <div data-testid="prompt-editor">
      <button onClick={() => onSave({ name: 'test' })}>Save</button>
      <button onClick={() => onTest({ name: 'test' })}>Test</button>
      <button onClick={() => onVersionHistory('1')}>History</button>
    </div>
  )
}))

jest.mock('@/components/prompt-management/template-library', () => ({
  TemplateLibrary: ({ onCreateNew, onEditTemplate, onDeleteTemplate }: any) => (
    <div data-testid="template-library">
      <button onClick={onCreateNew}>Create New</button>
      <button onClick={() => onEditTemplate({ id: '1', name: 'test' })}>Edit</button>
      <button onClick={() => onDeleteTemplate('1')}>Delete</button>
    </div>
  )
}))

jest.mock('@/components/prompt-management/ab-testing-dashboard', () => ({
  ABTestingDashboard: ({ onCreateExperiment, onStartExperiment }: any) => (
    <div data-testid="ab-testing-dashboard">
      <button onClick={onCreateExperiment}>Create Experiment</button>
      <button onClick={() => onStartExperiment('1')}>Start Experiment</button>
    </div>
  )
}))

jest.mock('@/components/prompt-management/optimization-monitor', () => ({
  OptimizationMonitor: ({ onStartOptimization, onApplyOptimization }: any) => (
    <div data-testid="optimization-monitor">
      <button onClick={() => onStartOptimization('1', {})}>Start Optimization</button>
      <button onClick={() => onApplyOptimization('1', '1')}>Apply Optimization</button>
    </div>
  )
}))

jest.mock('@/components/prompt-management/analytics-dashboard', () => ({
  AnalyticsDashboard: ({ onExportReport, onDrillDown }: any) => (
    <div data-testid="analytics-dashboard">
      <button onClick={() => onExportReport({})}>Export Report</button>
      <button onClick={() => onDrillDown('metric', {})}>Drill Down</button>
    </div>
  )
}))

// Mock window.confirm
window.confirm = jest.fn(() => true)

// Mock console methods
console.log = jest.fn()
console.error = jest.fn()

describe('PromptManagementInterface', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders without crashing', () => {
    render(<PromptManagementInterface />)
    expect(screen.getByText('Prompt Management')).toBeInTheDocument()
  })

  it('displays main header and navigation', () => {
    render(<PromptManagementInterface />)
    
    expect(screen.getByText('Prompt Management')).toBeInTheDocument()
    expect(screen.getByText('Manage, test, and optimize your AI prompts')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /New Prompt/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Settings/ })).toBeInTheDocument()
  })

  it('shows all navigation tabs', () => {
    render(<PromptManagementInterface />)
    
    expect(screen.getByRole('tab', { name: /Library/ })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /Editor/ })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /A\/B Testing/ })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /Optimization/ })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /Analytics/ })).toBeInTheDocument()
  })

  it('starts with library tab active', () => {
    render(<PromptManagementInterface />)
    
    expect(screen.getByTestId('template-library')).toBeInTheDocument()
  })

  it('switches to editor tab when New Prompt is clicked', async () => {
    render(<PromptManagementInterface />)
    
    const newPromptButton = screen.getByRole('button', { name: /New Prompt/ })
    await userEvent.click(newPromptButton)
    
    expect(screen.getByTestId('prompt-editor')).toBeInTheDocument()
  })

  it('switches tabs correctly', async () => {
    render(<PromptManagementInterface />)
    
    // Switch to A/B Testing tab
    const testingTab = screen.getByRole('tab', { name: /A\/B Testing/ })
    await userEvent.click(testingTab)
    expect(screen.getByTestId('ab-testing-dashboard')).toBeInTheDocument()
    
    // Switch to Optimization tab
    const optimizationTab = screen.getByRole('tab', { name: /Optimization/ })
    await userEvent.click(optimizationTab)
    expect(screen.getByTestId('optimization-monitor')).toBeInTheDocument()
    
    // Switch to Analytics tab
    const analyticsTab = screen.getByRole('tab', { name: /Analytics/ })
    await userEvent.click(analyticsTab)
    expect(screen.getByTestId('analytics-dashboard')).toBeInTheDocument()
  })

  it('handles template library actions', async () => {
    render(<PromptManagementInterface />)
    
    // Create new template
    const createButton = screen.getByText('Create New')
    await userEvent.click(createButton)
    expect(screen.getByTestId('prompt-editor')).toBeInTheDocument()
    
    // Go back to library
    const libraryTab = screen.getByRole('tab', { name: /Library/ })
    await userEvent.click(libraryTab)
    
    // Edit template
    const editButton = screen.getByText('Edit')
    await userEvent.click(editButton)
    expect(screen.getByTestId('prompt-editor')).toBeInTheDocument()
    
    // Delete template
    await userEvent.click(libraryTab)
    const deleteButton = screen.getByText('Delete')
    await userEvent.click(deleteButton)
    expect(window.confirm).toHaveBeenCalled()
  })

  it('handles prompt editor actions', async () => {
    render(<PromptManagementInterface />)
    
    // Switch to editor
    const editorTab = screen.getByRole('tab', { name: /Editor/ })
    await userEvent.click(editorTab)
    
    // Save prompt
    const saveButton = screen.getByText('Save')
    await userEvent.click(saveButton)
    expect(console.log).toHaveBeenCalledWith('Creating prompt:', { name: 'test' })
    
    // Test prompt
    const testButton = screen.getByText('Test')
    await userEvent.click(testButton)
    expect(console.log).toHaveBeenCalledWith('Testing prompt:', { name: 'test' })
    
    // View history
    const historyButton = screen.getByText('History')
    await userEvent.click(historyButton)
    expect(console.log).toHaveBeenCalledWith('Viewing version history for:', '1')
  })

  it('handles A/B testing actions', async () => {
    render(<PromptManagementInterface />)
    
    // Switch to A/B testing
    const testingTab = screen.getByRole('tab', { name: /A\/B Testing/ })
    await userEvent.click(testingTab)
    
    // Create experiment
    const createExperimentButton = screen.getByText('Create Experiment')
    await userEvent.click(createExperimentButton)
    expect(console.log).toHaveBeenCalledWith('Creating new experiment')
    
    // Start experiment
    const startExperimentButton = screen.getByText('Start Experiment')
    await userEvent.click(startExperimentButton)
    expect(console.log).toHaveBeenCalledWith('Starting experiment:', '1')
  })

  it('handles optimization actions', async () => {
    render(<PromptManagementInterface />)
    
    // Switch to optimization
    const optimizationTab = screen.getByRole('tab', { name: /Optimization/ })
    await userEvent.click(optimizationTab)
    
    // Start optimization
    const startOptimizationButton = screen.getByText('Start Optimization')
    await userEvent.click(startOptimizationButton)
    expect(console.log).toHaveBeenCalledWith('Starting optimization:', { promptId: '1', config: {} })
    
    // Apply optimization
    const applyOptimizationButton = screen.getByText('Apply Optimization')
    await userEvent.click(applyOptimizationButton)
    expect(console.log).toHaveBeenCalledWith('Applying optimization:', { jobId: '1', generationId: '1' })
  })

  it('handles analytics actions', async () => {
    render(<PromptManagementInterface />)
    
    // Switch to analytics
    const analyticsTab = screen.getByRole('tab', { name: /Analytics/ })
    await userEvent.click(analyticsTab)
    
    // Export report
    const exportReportButton = screen.getByText('Export Report')
    await userEvent.click(exportReportButton)
    expect(console.log).toHaveBeenCalledWith('Exporting report with filters:', {})
    
    // Drill down
    const drillDownButton = screen.getByText('Drill Down')
    await userEvent.click(drillDownButton)
    expect(console.log).toHaveBeenCalledWith('Drilling down into metric:', { metric: 'metric', filters: {} })
  })

  it('maintains state when switching between tabs', async () => {
    render(<PromptManagementInterface />)
    
    // Create new prompt (should switch to editor)
    const newPromptButton = screen.getByRole('button', { name: /New Prompt/ })
    await userEvent.click(newPromptButton)
    expect(screen.getByTestId('prompt-editor')).toBeInTheDocument()
    
    // Switch to library and back to editor
    const libraryTab = screen.getByRole('tab', { name: /Library/ })
    await userEvent.click(libraryTab)
    
    const editorTab = screen.getByRole('tab', { name: /Editor/ })
    await userEvent.click(editorTab)
    expect(screen.getByTestId('prompt-editor')).toBeInTheDocument()
  })

  it('handles edit template flow correctly', async () => {
    render(<PromptManagementInterface />)
    
    // Edit template from library (should switch to editor with template data)
    const editButton = screen.getByText('Edit')
    await userEvent.click(editButton)
    
    expect(screen.getByTestId('prompt-editor')).toBeInTheDocument()
    // Should be on editor tab
    expect(screen.getByRole('tab', { name: /Editor/ })).toHaveAttribute('data-state', 'active')
  })

  it('returns to library after saving prompt', async () => {
    render(<PromptManagementInterface />)
    
    // Switch to editor
    const editorTab = screen.getByRole('tab', { name: /Editor/ })
    await userEvent.click(editorTab)
    
    // Save prompt
    const saveButton = screen.getByText('Save')
    await userEvent.click(saveButton)
    
    // Should return to library tab
    await waitFor(() => {
      expect(screen.getByTestId('template-library')).toBeInTheDocument()
    })
  })
})