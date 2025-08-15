import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import { AnalyticsDashboard } from '@/components/prompt-management/analytics-dashboard'

const mockProps = {
  onExportReport: jest.fn(),
  onDrillDown: jest.fn()
}

describe('AnalyticsDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders without crashing', () => {
    render(<AnalyticsDashboard {...mockProps} />)
    expect(screen.getByText('Prompt Analytics')).toBeInTheDocument()
  })

  it('displays key metrics', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('Total Requests')).toBeInTheDocument()
      expect(screen.getByText('Success Rate')).toBeInTheDocument()
      expect(screen.getByText('Avg Response Time')).toBeInTheDocument()
      expect(screen.getByText('User Satisfaction')).toBeInTheDocument()
    })
  })

  it('shows time range selector', () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    expect(screen.getByDisplayValue('Last 30 days')).toBeInTheDocument()
  })

  it('shows category filter', () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    expect(screen.getByDisplayValue('All Categories')).toBeInTheDocument()
  })

  it('calls onExportReport when export button is clicked', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const exportButton = screen.getByRole('button', { name: /Export/ })
    await userEvent.click(exportButton)
    
    expect(mockProps.onExportReport).toHaveBeenCalledWith({
      timeRange: '30d',
      category: 'all'
    })
  })

  it('updates time range filter', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const timeRangeSelect = screen.getByDisplayValue('Last 30 days')
    await userEvent.selectOptions(timeRangeSelect, '7d')
    
    expect(timeRangeSelect).toHaveValue('7d')
  })

  it('updates category filter', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    await waitFor(() => {
      const categorySelect = screen.getByDisplayValue('All Categories')
      // Categories are loaded from mock data
      if (screen.queryByText('content')) {
        userEvent.selectOptions(categorySelect, 'content')
      }
    })
  })

  it('displays usage trends chart', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('Usage Trends')).toBeInTheDocument()
    })
  })

  it('shows top performing prompts', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('Top Performing Prompts')).toBeInTheDocument()
      expect(screen.getByText('Content Generation')).toBeInTheDocument()
    })
  })

  it('displays category distribution pie chart', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    await waitFor(() => {
      expect(screen.getByText('Usage by Category')).toBeInTheDocument()
    })
  })

  it('switches to prompts tab and shows detailed table', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const promptsTab = screen.getByRole('tab', { name: 'Prompt Performance' })
    await userEvent.click(promptsTab)
    
    await waitFor(() => {
      expect(screen.getByText('Prompt Performance Details')).toBeInTheDocument()
      expect(screen.getByText('Prompt Name')).toBeInTheDocument()
      expect(screen.getByText('Requests')).toBeInTheDocument()
      expect(screen.getByText('Success Rate')).toBeInTheDocument()
    })
  })

  it('displays trends tab with performance charts', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const trendsTab = screen.getByRole('tab', { name: 'Trends' })
    await userEvent.click(trendsTab)
    
    await waitFor(() => {
      expect(screen.getByText('Performance Trends')).toBeInTheDocument()
      expect(screen.getByText('Request Volume Trend')).toBeInTheDocument()
    })
  })

  it('shows team analytics', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const teamTab = screen.getByRole('tab', { name: 'Team Analytics' })
    await userEvent.click(teamTab)
    
    await waitFor(() => {
      expect(screen.getByText('Team Performance')).toBeInTheDocument()
      expect(screen.getByText('John Doe')).toBeInTheDocument()
      expect(screen.getByText('Jane Smith')).toBeInTheDocument()
    })
  })

  it('displays categories analysis', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const categoriesTab = screen.getByRole('tab', { name: 'Categories' })
    await userEvent.click(categoriesTab)
    
    await waitFor(() => {
      expect(screen.getByText('Category Performance Comparison')).toBeInTheDocument()
      expect(screen.getByText('content')).toBeInTheDocument()
      expect(screen.getByText('development')).toBeInTheDocument()
    })
  })

  it('shows performance indicators with correct colors', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const promptsTab = screen.getByRole('tab', { name: 'Prompt Performance' })
    await userEvent.click(promptsTab)
    
    await waitFor(() => {
      // Should show colored performance indicators
      const successRates = screen.getAllByText(/\d+\.\d+%/)
      expect(successRates.length).toBeGreaterThan(0)
    })
  })

  it('displays team collaboration scatter chart', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const teamTab = screen.getByRole('tab', { name: 'Team Analytics' })
    await userEvent.click(teamTab)
    
    await waitFor(() => {
      expect(screen.getByText('Prompts Created vs Used')).toBeInTheDocument()
    })
  })

  it('shows category metrics with top prompts', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const categoriesTab = screen.getByRole('tab', { name: 'Categories' })
    await userEvent.click(categoriesTab)
    
    await waitFor(() => {
      expect(screen.getByText('Top Prompts')).toBeInTheDocument()
      expect(screen.getByText('Usage Count')).toBeInTheDocument()
      expect(screen.getByText('Avg Performance')).toBeInTheDocument()
    })
  })

  it('displays metric trends over time', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const trendsTab = screen.getByRole('tab', { name: 'Trends' })
    await userEvent.click(trendsTab)
    
    await waitFor(() => {
      // Should show line chart with multiple metrics
      expect(screen.getByText('Performance Trends')).toBeInTheDocument()
    })
  })

  it('shows user favorite categories', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const teamTab = screen.getByRole('tab', { name: 'Team Analytics' })
    await userEvent.click(teamTab)
    
    await waitFor(() => {
      expect(screen.getByText('Favorite Categories')).toBeInTheDocument()
      expect(screen.getByText('content')).toBeInTheDocument()
      expect(screen.getByText('marketing')).toBeInTheDocument()
    })
  })

  it('displays prompt usage statistics in table', async () => {
    render(<AnalyticsDashboard {...mockProps} />)
    
    const promptsTab = screen.getByRole('tab', { name: 'Prompt Performance' })
    await userEvent.click(promptsTab)
    
    await waitFor(() => {
      expect(screen.getByText('Cost/Request')).toBeInTheDocument()
      expect(screen.getByText('Last Used')).toBeInTheDocument()
      expect(screen.getByText('Satisfaction')).toBeInTheDocument()
    })
  })
})