import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { OnboardingModal } from '@/components/onboarding/onboarding-modal'
import { OnboardingProvider, OnboardingStep } from '@/components/onboarding/onboarding-provider'

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

const testSteps: OnboardingStep[] = [
  {
    id: 'step1',
    title: 'Welcome',
    description: 'Welcome to the platform',
    content: <div>Welcome content</div>,
  },
  {
    id: 'step2',
    title: 'Getting Started',
    description: 'Learn the basics',
    content: <div>Getting started content</div>,
    skippable: false,
  },
]

describe('OnboardingModal', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('does not render when onboarding is not active', () => {
    mockOnboarding.state.isActive = false
    mockOnboarding.state.steps = []

    render(<OnboardingModal />)

    expect(screen.queryByText('Welcome')).not.toBeInTheDocument()
  })

  it('renders correctly when onboarding is active', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 0

    render(<OnboardingModal />)

    expect(screen.getByText('Welcome')).toBeInTheDocument()
    expect(screen.getByText('Welcome to the platform')).toBeInTheDocument()
    expect(screen.getByText('Step 1 of 2')).toBeInTheDocument()
    expect(screen.getByText('Welcome content')).toBeInTheDocument()
  })

  it('shows progress bar with correct percentage', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 0

    render(<OnboardingModal />)

    expect(screen.getByText('50%')).toBeInTheDocument() // Step 1 of 2 = 50%
  })

  it('calls nextStep when Next button is clicked', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 0

    render(<OnboardingModal />)

    fireEvent.click(screen.getByText('Next'))
    expect(mockOnboarding.nextStep).toHaveBeenCalledTimes(1)
  })

  it('calls prevStep when Previous button is clicked', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 1

    render(<OnboardingModal />)

    fireEvent.click(screen.getByText('Previous'))
    expect(mockOnboarding.prevStep).toHaveBeenCalledTimes(1)
  })

  it('does not show Previous button on first step', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 0

    render(<OnboardingModal />)

    expect(screen.queryByText('Previous')).not.toBeInTheDocument()
  })

  it('shows Skip button for skippable steps', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 0 // First step is skippable by default

    render(<OnboardingModal />)

    expect(screen.getByText('Skip')).toBeInTheDocument()
  })

  it('does not show Skip button for non-skippable steps', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 1 // Second step has skippable: false

    render(<OnboardingModal />)

    expect(screen.queryByText('Skip')).not.toBeInTheDocument()
  })

  it('calls skipStep when Skip button is clicked', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 0

    render(<OnboardingModal />)

    fireEvent.click(screen.getByText('Skip'))
    expect(mockOnboarding.skipStep).toHaveBeenCalledTimes(1)
  })

  it('shows Complete button on last step', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 1 // Last step

    render(<OnboardingModal />)

    expect(screen.getByText('Complete')).toBeInTheDocument()
  })

  it('calls completeOnboarding when Complete button is clicked', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 1

    render(<OnboardingModal />)

    fireEvent.click(screen.getByText('Complete'))
    expect(mockOnboarding.completeOnboarding).toHaveBeenCalledTimes(1)
  })

  it('calls completeOnboarding when close button is clicked', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 0

    render(<OnboardingModal />)

    const closeButton = screen.getByRole('button', { name: '' }) // X button
    fireEvent.click(closeButton)
    expect(mockOnboarding.completeOnboarding).toHaveBeenCalledTimes(1)
  })

  it('displays correct step content', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 1

    render(<OnboardingModal />)

    expect(screen.getByText('Getting Started')).toBeInTheDocument()
    expect(screen.getByText('Learn the basics')).toBeInTheDocument()
    expect(screen.getByText('Getting started content')).toBeInTheDocument()
  })

  it('updates progress percentage correctly for different steps', () => {
    mockOnboarding.state.isActive = true
    mockOnboarding.state.steps = testSteps
    mockOnboarding.state.currentStep = 1

    render(<OnboardingModal />)

    expect(screen.getByText('100%')).toBeInTheDocument() // Step 2 of 2 = 100%
  })
})