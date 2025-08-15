import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ContextualTooltip } from '@/components/onboarding/contextual-tooltip'
import { OnboardingProvider } from '@/components/onboarding/onboarding-provider'

// Mock the useOnboarding hook
const mockOnboarding = {
  state: {
    isActive: false,
    currentStep: 0,
    steps: [],
    isCompleted: false,
    showTooltips: true,
  },
  nextStep: jest.fn(),
  prevStep: jest.fn(),
  skipStep: jest.fn(),
  completeOnboarding: jest.fn(),
  startOnboarding: jest.fn(),
  showTooltip: jest.fn(),
  hideTooltip: jest.fn(),
  toggleTooltips: jest.fn(),
}

jest.mock('@/components/onboarding/onboarding-provider', () => ({
  ...jest.requireActual('@/components/onboarding/onboarding-provider'),
  useOnboarding: () => mockOnboarding,
}))

describe('ContextualTooltip', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders children correctly', () => {
    render(
      <ContextualTooltip
        target="test-target"
        title="Test Title"
        content="Test content"
      >
        <button>Test Button</button>
      </ContextualTooltip>
    )

    expect(screen.getByText('Test Button')).toBeInTheDocument()
    expect(screen.getByText('Test Button')).toHaveAttribute('data-tour', 'test-target')
  })

  it('shows tooltip on hover when trigger is hover', async () => {
    render(
      <ContextualTooltip
        target="test-target"
        title="Test Title"
        content="Test content"
        trigger="hover"
      >
        <button>Test Button</button>
      </ContextualTooltip>
    )

    const button = screen.getByText('Test Button')
    fireEvent.mouseEnter(button)

    await waitFor(() => {
      expect(screen.getByText('Test Title')).toBeInTheDocument()
      expect(screen.getByText('Test content')).toBeInTheDocument()
    })
  })

  it('hides tooltip on mouse leave when trigger is hover', async () => {
    render(
      <ContextualTooltip
        target="test-target"
        title="Test Title"
        content="Test content"
        trigger="hover"
      >
        <button>Test Button</button>
      </ContextualTooltip>
    )

    const button = screen.getByText('Test Button')
    
    // Show tooltip
    fireEvent.mouseEnter(button)
    await waitFor(() => {
      expect(screen.getByText('Test Title')).toBeInTheDocument()
    })

    // Hide tooltip
    fireEvent.mouseLeave(button)
    await waitFor(() => {
      expect(screen.queryByText('Test Title')).not.toBeInTheDocument()
    })
  })

  it('shows help icon when trigger is click', () => {
    render(
      <ContextualTooltip
        target="test-target"
        title="Test Title"
        content="Test content"
        trigger="click"
      >
        <button>Test Button</button>
      </ContextualTooltip>
    )

    // Help icon should be present
    const helpIcon = screen.getByRole('button')
    expect(helpIcon).toBeInTheDocument()
  })

  it('toggles tooltip on click when trigger is click', async () => {
    render(
      <ContextualTooltip
        target="test-target"
        title="Test Title"
        content="Test content"
        trigger="click"
      >
        <button>Test Button</button>
      </ContextualTooltip>
    )

    const container = screen.getByText('Test Button').parentElement
    
    // Click to show tooltip
    fireEvent.click(container!)
    await waitFor(() => {
      expect(screen.getByText('Test Title')).toBeInTheDocument()
    })

    // Click again to hide tooltip
    fireEvent.click(container!)
    await waitFor(() => {
      expect(screen.queryByText('Test Title')).not.toBeInTheDocument()
    })
  })

  it('closes tooltip when close button is clicked', async () => {
    render(
      <ContextualTooltip
        target="test-target"
        title="Test Title"
        content="Test content"
        trigger="hover"
      >
        <button>Test Button</button>
      </ContextualTooltip>
    )

    const button = screen.getByText('Test Button')
    
    // Show tooltip
    fireEvent.mouseEnter(button)
    await waitFor(() => {
      expect(screen.getByText('Test Title')).toBeInTheDocument()
    })

    // Click close button
    const closeButton = screen.getByRole('button', { name: '' }) // X button
    fireEvent.click(closeButton)

    await waitFor(() => {
      expect(screen.queryByText('Test Title')).not.toBeInTheDocument()
    })
  })

  it('closes tooltip when clicking outside', async () => {
    render(
      <div>
        <ContextualTooltip
          target="test-target"
          title="Test Title"
          content="Test content"
          trigger="click"
        >
          <button>Test Button</button>
        </ContextualTooltip>
        <div data-testid="outside">Outside element</div>
      </div>
    )

    const container = screen.getByText('Test Button').parentElement
    
    // Click to show tooltip
    fireEvent.click(container!)
    await waitFor(() => {
      expect(screen.getByText('Test Title')).toBeInTheDocument()
    })

    // Click outside to hide tooltip
    fireEvent.click(screen.getByTestId('outside'))
    await waitFor(() => {
      expect(screen.queryByText('Test Title')).not.toBeInTheDocument()
    })
  })

  it('does not show tooltip when showTooltips is false', () => {
    mockOnboarding.state.showTooltips = false

    render(
      <ContextualTooltip
        target="test-target"
        title="Test Title"
        content="Test content"
        trigger="hover"
      >
        <button>Test Button</button>
      </ContextualTooltip>
    )

    const button = screen.getByText('Test Button')
    fireEvent.mouseEnter(button)

    // Tooltip should not appear
    expect(screen.queryByText('Test Title')).not.toBeInTheDocument()
  })

  it('renders with correct data-tour attribute', () => {
    render(
      <ContextualTooltip
        target="custom-target"
        title="Test Title"
        content="Test content"
      >
        <div>Test Content</div>
      </ContextualTooltip>
    )

    expect(screen.getByText('Test Content')).toHaveAttribute('data-tour', 'custom-target')
  })

  it('applies correct positioning classes', async () => {
    render(
      <ContextualTooltip
        target="test-target"
        title="Test Title"
        content="Test content"
        position="bottom"
        trigger="hover"
      >
        <button>Test Button</button>
      </ContextualTooltip>
    )

    const button = screen.getByText('Test Button')
    fireEvent.mouseEnter(button)

    await waitFor(() => {
      const tooltip = screen.getByText('Test Title').closest('.fixed')
      expect(tooltip).toBeInTheDocument()
    })
  })
})