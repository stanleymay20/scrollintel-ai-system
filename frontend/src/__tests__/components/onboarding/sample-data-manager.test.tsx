import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { SampleDataManager } from '@/components/onboarding/sample-data-manager'

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = jest.fn(() => 'mocked-url')
global.URL.revokeObjectURL = jest.fn()

// Mock document.createElement and related methods
const mockAppendChild = jest.fn()
const mockRemoveChild = jest.fn()
const mockClick = jest.fn()

Object.defineProperty(document, 'createElement', {
  value: jest.fn(() => ({
    href: '',
    download: '',
    click: mockClick,
  })),
})

Object.defineProperty(document.body, 'appendChild', {
  value: mockAppendChild,
})

Object.defineProperty(document.body, 'removeChild', {
  value: mockRemoveChild,
})

describe('SampleDataManager', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders sample datasets correctly', () => {
    render(<SampleDataManager />)

    expect(screen.getByText('Sample Datasets')).toBeInTheDocument()
    expect(screen.getByText('4 Datasets Available')).toBeInTheDocument()
    
    // Check if sample datasets are rendered
    expect(screen.getByText('E-commerce Sales Data')).toBeInTheDocument()
    expect(screen.getByText('Customer Analytics Dataset')).toBeInTheDocument()
    expect(screen.getByText('Financial Performance Data')).toBeInTheDocument()
    expect(screen.getByText('Website Analytics Data')).toBeInTheDocument()
  })

  it('displays dataset information correctly', () => {
    render(<SampleDataManager />)

    // Check first dataset details
    expect(screen.getByText('Sample e-commerce transaction data with customer demographics and purchase history.')).toBeInTheDocument()
    expect(screen.getByText('Sales & Marketing')).toBeInTheDocument()
    expect(screen.getByText('CSV')).toBeInTheDocument()
    expect(screen.getByText('2.5 MB')).toBeInTheDocument()
  })

  it('shows dataset preview when Preview button is clicked', async () => {
    render(<SampleDataManager />)

    const previewButtons = screen.getAllByText('Preview')
    fireEvent.click(previewButtons[0])

    await waitFor(() => {
      expect(screen.getByText('E-commerce Sales Data - Preview')).toBeInTheDocument()
      expect(screen.getByText('order_id')).toBeInTheDocument()
      expect(screen.getByText('customer_id')).toBeInTheDocument()
      expect(screen.getByText('ORD-001')).toBeInTheDocument()
    })
  })

  it('closes preview modal when Close button is clicked', async () => {
    render(<SampleDataManager />)

    // Open preview
    const previewButtons = screen.getAllByText('Preview')
    fireEvent.click(previewButtons[0])

    await waitFor(() => {
      expect(screen.getByText('E-commerce Sales Data - Preview')).toBeInTheDocument()
    })

    // Close preview
    fireEvent.click(screen.getByText('Close'))

    await waitFor(() => {
      expect(screen.queryByText('E-commerce Sales Data - Preview')).not.toBeInTheDocument()
    })
  })

  it('closes preview modal when X button is clicked', async () => {
    render(<SampleDataManager />)

    // Open preview
    const previewButtons = screen.getAllByText('Preview')
    fireEvent.click(previewButtons[0])

    await waitFor(() => {
      expect(screen.getByText('E-commerce Sales Data - Preview')).toBeInTheDocument()
    })

    // Close preview using X button
    const closeButton = screen.getByText('×')
    fireEvent.click(closeButton)

    await waitFor(() => {
      expect(screen.queryByText('E-commerce Sales Data - Preview')).not.toBeInTheDocument()
    })
  })

  it('triggers download when Download button is clicked', async () => {
    render(<SampleDataManager />)

    const downloadButtons = screen.getAllByText('Download')
    fireEvent.click(downloadButtons[0])

    await waitFor(() => {
      expect(mockClick).toHaveBeenCalledTimes(1)
      expect(mockAppendChild).toHaveBeenCalledTimes(1)
      expect(mockRemoveChild).toHaveBeenCalledTimes(1)
    })
  })

  it('triggers download from preview modal', async () => {
    render(<SampleDataManager />)

    // Open preview
    const previewButtons = screen.getAllByText('Preview')
    fireEvent.click(previewButtons[0])

    await waitFor(() => {
      expect(screen.getByText('E-commerce Sales Data - Preview')).toBeInTheDocument()
    })

    // Download from modal
    const downloadButton = screen.getByText('Download Dataset')
    fireEvent.click(downloadButton)

    await waitFor(() => {
      expect(mockClick).toHaveBeenCalledTimes(1)
    })
  })

  it('displays correct use cases for each dataset', () => {
    render(<SampleDataManager />)

    // Check use cases for e-commerce dataset
    expect(screen.getByText('• Sales trend analysis')).toBeInTheDocument()
    expect(screen.getByText('• Customer segmentation')).toBeInTheDocument()
    expect(screen.getByText('• Product performance analysis')).toBeInTheDocument()
    expect(screen.getByText('• Revenue forecasting')).toBeInTheDocument()
  })

  it('displays dataset preview information correctly', () => {
    render(<SampleDataManager />)

    // Check preview information
    expect(screen.getByText('10,000 rows')).toBeInTheDocument()
    expect(screen.getByText('7 columns')).toBeInTheDocument()
    expect(screen.getByText('5,000 rows')).toBeInTheDocument()
    expect(screen.getByText('8 columns')).toBeInTheDocument()
  })

  it('shows correct category icons', () => {
    render(<SampleDataManager />)

    // All datasets should have their category badges
    const categoryBadges = screen.getAllByText(/Sales & Marketing|Customer Analytics|Finance|Digital Marketing/)
    expect(categoryBadges.length).toBeGreaterThan(0)
  })

  it('displays sample data in preview table correctly', async () => {
    render(<SampleDataManager />)

    // Open preview for e-commerce dataset
    const previewButtons = screen.getAllByText('Preview')
    fireEvent.click(previewButtons[0])

    await waitFor(() => {
      // Check table headers
      expect(screen.getByText('order_id')).toBeInTheDocument()
      expect(screen.getByText('customer_id')).toBeInTheDocument()
      expect(screen.getByText('product_name')).toBeInTheDocument()
      
      // Check sample data
      expect(screen.getByText('ORD-001')).toBeInTheDocument()
      expect(screen.getByText('CUST-123')).toBeInTheDocument()
      expect(screen.getByText('Wireless Headphones')).toBeInTheDocument()
    })
  })

  it('handles different file formats correctly', () => {
    render(<SampleDataManager />)

    // Check different format badges
    expect(screen.getByText('CSV')).toBeInTheDocument()
    expect(screen.getByText('Excel')).toBeInTheDocument()
    expect(screen.getByText('JSON')).toBeInTheDocument()
  })

  it('displays correct file sizes', () => {
    render(<SampleDataManager />)

    expect(screen.getByText('2.5 MB')).toBeInTheDocument()
    expect(screen.getByText('1.8 MB')).toBeInTheDocument()
    expect(screen.getByText('950 KB')).toBeInTheDocument()
    expect(screen.getByText('3.2 MB')).toBeInTheDocument()
  })
})