import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChartBuilder } from '@/components/visualization/chart-builder'

// Mock fetch
global.fetch = jest.fn()

const mockData = [
  { category: 'A', sales: 1000, date: '2024-01-01' },
  { category: 'B', sales: 1200, date: '2024-01-02' },
  { category: 'C', sales: 800, date: '2024-01-03' }
]

const mockOnChartCreate = jest.fn()
const mockOnPreview = jest.fn()

describe('ChartBuilder', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    ;(fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        suggestions: [
          {
            type: 'bar',
            title: 'Sales by Category',
            x_axis: 'category',
            y_axis: 'sales',
            confidence: 0.9
          }
        ]
      })
    })
  })

  it('renders chart builder interface', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    expect(screen.getByText('Chart Builder')).toBeInTheDocument()
    expect(screen.getByText('Basic')).toBeInTheDocument()
    expect(screen.getByText('Style')).toBeInTheDocument()
    expect(screen.getByText('Advanced')).toBeInTheDocument()
    expect(screen.getByText('Suggestions')).toBeInTheDocument()
  })

  it('displays chart type options', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Click on chart type selector
    fireEvent.click(screen.getByRole('combobox', { name: /chart type/i }))

    expect(screen.getByText('Line Chart')).toBeInTheDocument()
    expect(screen.getByText('Bar Chart')).toBeInTheDocument()
    expect(screen.getByText('Pie Chart')).toBeInTheDocument()
    expect(screen.getByText('Scatter Plot')).toBeInTheDocument()
    expect(screen.getByText('Area Chart')).toBeInTheDocument()
  })

  it('populates column options from data', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Click on X-axis selector
    fireEvent.click(screen.getByRole('combobox', { name: /x-axis/i }))

    expect(screen.getByText('category')).toBeInTheDocument()
    expect(screen.getByText('sales')).toBeInTheDocument()
    expect(screen.getByText('date')).toBeInTheDocument()
  })

  it('allows chart configuration', async () => {
    const user = userEvent.setup()
    
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Set chart title
    const titleInput = screen.getByLabelText(/chart title/i)
    await user.clear(titleInput)
    await user.type(titleInput, 'Test Chart')

    expect(titleInput).toHaveValue('Test Chart')

    // Select chart type
    fireEvent.click(screen.getByRole('combobox', { name: /chart type/i }))
    fireEvent.click(screen.getByText('Bar Chart'))

    // Select X-axis
    fireEvent.click(screen.getByRole('combobox', { name: /x-axis/i }))
    fireEvent.click(screen.getByText('category'))

    // Select Y-axis
    fireEvent.click(screen.getByRole('combobox', { name: /y-axis/i }))
    fireEvent.click(screen.getByText('sales'))
  })

  it('displays color scheme options', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Switch to Style tab
    fireEvent.click(screen.getByText('Style'))

    // Click on color scheme selector
    fireEvent.click(screen.getByRole('combobox', { name: /color scheme/i }))

    expect(screen.getByText('Default')).toBeInTheDocument()
    expect(screen.getByText('Professional')).toBeInTheDocument()
    expect(screen.getByText('Modern')).toBeInTheDocument()
    expect(screen.getByText('ScrollIntel')).toBeInTheDocument()
  })

  it('allows dimension configuration', async () => {
    const user = userEvent.setup()
    
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Switch to Style tab
    fireEvent.click(screen.getByText('Style'))

    // Find width and height inputs
    const widthInput = screen.getByDisplayValue('800')
    const heightInput = screen.getByDisplayValue('400')

    // Change dimensions
    await user.clear(widthInput)
    await user.type(widthInput, '1000')

    await user.clear(heightInput)
    await user.type(heightInput, '500')

    expect(widthInput).toHaveValue(1000)
    expect(heightInput).toHaveValue(500)
  })

  it('toggles display options', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Switch to Style tab
    fireEvent.click(screen.getByText('Style'))

    // Find display option badges
    const legendBadge = screen.getByText('Legend')
    const gridBadge = screen.getByText('Grid')
    const tooltipBadge = screen.getByText('Tooltip')
    const interactiveBadge = screen.getByText('Interactive')

    // All should be active by default
    expect(legendBadge).toHaveClass('bg-primary')
    expect(gridBadge).toHaveClass('bg-primary')
    expect(tooltipBadge).toHaveClass('bg-primary')
    expect(interactiveBadge).toHaveClass('bg-primary')

    // Click to toggle
    fireEvent.click(legendBadge)
    expect(legendBadge).toHaveClass('border-input')
  })

  it('shows advanced options', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Switch to Advanced tab
    fireEvent.click(screen.getByText('Advanced'))

    expect(screen.getByLabelText(/aggregation/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/group by/i)).toBeInTheDocument()

    // Click on aggregation selector
    fireEvent.click(screen.getByRole('combobox', { name: /aggregation/i }))

    expect(screen.getByText('None')).toBeInTheDocument()
    expect(screen.getByText('Sum')).toBeInTheDocument()
    expect(screen.getByText('Average')).toBeInTheDocument()
    expect(screen.getByText('Count')).toBeInTheDocument()
    expect(screen.getByText('Maximum')).toBeInTheDocument()
    expect(screen.getByText('Minimum')).toBeInTheDocument()
  })

  it('fetches and displays chart suggestions', async () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Switch to Suggestions tab
    fireEvent.click(screen.getByText('Suggestions'))

    // Wait for suggestions to load
    await waitFor(() => {
      expect(screen.getByText('Sales by Category')).toBeInTheDocument()
    })

    expect(screen.getByText('90% match')).toBeInTheDocument()
    expect(screen.getByText('bar chart with category vs sales')).toBeInTheDocument()

    // Verify fetch was called
    expect(fetch).toHaveBeenCalledWith('/api/visualization/charts/suggestions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(mockData)
    })
  })

  it('applies chart suggestion when clicked', async () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Switch to Suggestions tab
    fireEvent.click(screen.getByText('Suggestions'))

    // Wait for suggestions to load
    await waitFor(() => {
      expect(screen.getByText('Sales by Category')).toBeInTheDocument()
    })

    // Click on suggestion
    fireEvent.click(screen.getByText('Sales by Category'))

    // Switch back to Basic tab to verify configuration was applied
    fireEvent.click(screen.getByText('Basic'))

    // Check if title was set
    expect(screen.getByDisplayValue('Sales by Category')).toBeInTheDocument()
  })

  it('calls onPreview when preview button is clicked', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    fireEvent.click(screen.getByText('Preview'))

    expect(mockOnPreview).toHaveBeenCalledWith(
      expect.objectContaining({
        chart_type: 'bar',
        title: '',
        x_axis: '',
        y_axis: '',
        color_scheme: 'default',
        width: 800,
        height: 400,
        interactive: true,
        show_legend: true,
        show_grid: true,
        show_tooltip: true,
        aggregation: '',
        group_by: ''
      })
    )
  })

  it('calls onChartCreate when create button is clicked', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    fireEvent.click(screen.getByText('Create Chart'))

    expect(mockOnChartCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        chart_type: 'bar',
        title: '',
        x_axis: '',
        y_axis: '',
        color_scheme: 'default',
        width: 800,
        height: 400,
        interactive: true,
        show_legend: true,
        show_grid: true,
        show_tooltip: true,
        aggregation: '',
        group_by: ''
      })
    )
  })

  it('handles empty data gracefully', () => {
    render(
      <ChartBuilder
        data={[]}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    expect(screen.getByText('Chart Builder')).toBeInTheDocument()

    // Switch to Suggestions tab
    fireEvent.click(screen.getByText('Suggestions'))

    expect(screen.getByText('No suggestions available. Upload data to get AI-powered recommendations.')).toBeInTheDocument()
  })

  it('handles suggestion fetch error', async () => {
    ;(fetch as jest.Mock).mockRejectedValue(new Error('Network error'))

    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Switch to Suggestions tab
    fireEvent.click(screen.getByText('Suggestions'))

    // Should still render without crashing
    expect(screen.getByText('AI-Powered Chart Suggestions')).toBeInTheDocument()
  })

  it('filters numeric columns for Y-axis', () => {
    const mixedData = [
      { category: 'A', sales: 1000, description: 'Product A', active: true },
      { category: 'B', sales: 1200, description: 'Product B', active: false }
    ]

    render(
      <ChartBuilder
        data={mixedData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Click on Y-axis selector
    fireEvent.click(screen.getByRole('combobox', { name: /y-axis/i }))

    // Should only show numeric columns
    expect(screen.getByText('sales')).toBeInTheDocument()
    // Should not show string columns
    expect(screen.queryByText('description')).not.toBeInTheDocument()
  })

  it('shows categorical columns for grouping', () => {
    render(
      <ChartBuilder
        data={mockData}
        onChartCreate={mockOnChartCreate}
        onPreview={mockOnPreview}
      />
    )

    // Switch to Advanced tab
    fireEvent.click(screen.getByText('Advanced'))

    // Click on Group By selector
    fireEvent.click(screen.getByRole('combobox', { name: /group by/i }))

    expect(screen.getByText('None')).toBeInTheDocument()
    expect(screen.getByText('category')).toBeInTheDocument()
    expect(screen.getByText('date')).toBeInTheDocument()
  })
})