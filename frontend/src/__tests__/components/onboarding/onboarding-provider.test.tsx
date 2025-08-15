import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { OnboardingProvider, useOnboarding, OnboardingStep } from '@/components/onboarding/onboarding-provider'

// Test component that uses the onboarding context
function TestComponent() {
  const { state, startOnboarding, nextStep, prevStep, completeOnboarding } = useOnboarding()

  const testSteps: OnboardingStep[] = [
    {
      id: 'step1',
      title: 'Step 1',
      description: 'First step',
      content: <div>Step 1 content</div>,
    },
    {
      id: 'step2',
      title: 'Step 2',
      description: 'Second step',
      content: <div>Step 2 content</div>,
    },
  ]

  return (
    <div>
      <div data-testid="is-active">{state.isActive.toString()}</div>
      <div data-testid="current-step">{state.currentStep}</div>
      <div data-testid="is-completed">{state.isCompleted.toString()}</div>
      <div data-testid="steps-length">{state.steps.length}</div>
      
      <button onClick={() => startOnboarding(testSteps)}>Start Onboarding</button>
      <button onClick={nextStep}>Next Step</button>
      <button onClick={prevStep}>Previous Step</button>
      <button onClick={completeOnboarding}>Complete Onboarding</button>
    </div>
  )
}

describe('OnboardingProvider', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear()
  })

  it('provides initial state correctly', () => {
    render(
      <OnboardingProvider>
        <TestComponent />
      </OnboardingProvider>
    )

    expect(screen.getByTestId('is-active')).toHaveTextContent('false')
    expect(screen.getByTestId('current-step')).toHaveTextContent('0')
    expect(screen.getByTestId('is-completed')).toHaveTextContent('false')
    expect(screen.getByTestId('steps-length')).toHaveTextContent('0')
  })

  it('starts onboarding correctly', () => {
    render(
      <OnboardingProvider>
        <TestComponent />
      </OnboardingProvider>
    )

    fireEvent.click(screen.getByText('Start Onboarding'))

    expect(screen.getByTestId('is-active')).toHaveTextContent('true')
    expect(screen.getByTestId('current-step')).toHaveTextContent('0')
    expect(screen.getByTestId('steps-length')).toHaveTextContent('2')
  })

  it('navigates through steps correctly', () => {
    render(
      <OnboardingProvider>
        <TestComponent />
      </OnboardingProvider>
    )

    // Start onboarding
    fireEvent.click(screen.getByText('Start Onboarding'))
    expect(screen.getByTestId('current-step')).toHaveTextContent('0')

    // Go to next step
    fireEvent.click(screen.getByText('Next Step'))
    expect(screen.getByTestId('current-step')).toHaveTextContent('1')

    // Go back to previous step
    fireEvent.click(screen.getByText('Previous Step'))
    expect(screen.getByTestId('current-step')).toHaveTextContent('0')
  })

  it('completes onboarding when reaching the end', () => {
    render(
      <OnboardingProvider>
        <TestComponent />
      </OnboardingProvider>
    )

    // Start onboarding
    fireEvent.click(screen.getByText('Start Onboarding'))
    
    // Go through all steps
    fireEvent.click(screen.getByText('Next Step')) // Step 1
    fireEvent.click(screen.getByText('Next Step')) // Should complete

    expect(screen.getByTestId('is-active')).toHaveTextContent('false')
    expect(screen.getByTestId('is-completed')).toHaveTextContent('true')
  })

  it('completes onboarding manually', () => {
    render(
      <OnboardingProvider>
        <TestComponent />
      </OnboardingProvider>
    )

    fireEvent.click(screen.getByText('Start Onboarding'))
    fireEvent.click(screen.getByText('Complete Onboarding'))

    expect(screen.getByTestId('is-active')).toHaveTextContent('false')
    expect(screen.getByTestId('is-completed')).toHaveTextContent('true')
  })

  it('saves completion state to localStorage', () => {
    render(
      <OnboardingProvider>
        <TestComponent />
      </OnboardingProvider>
    )

    fireEvent.click(screen.getByText('Start Onboarding'))
    fireEvent.click(screen.getByText('Complete Onboarding'))

    expect(localStorage.getItem('scrollintel-onboarding-completed')).toBe('true')
  })

  it('loads completion state from localStorage', () => {
    // Set completion state in localStorage
    localStorage.setItem('scrollintel-onboarding-completed', 'true')

    render(
      <OnboardingProvider>
        <TestComponent />
      </OnboardingProvider>
    )

    expect(screen.getByTestId('is-completed')).toHaveTextContent('true')
  })

  it('prevents going to previous step when at first step', () => {
    render(
      <OnboardingProvider>
        <TestComponent />
      </OnboardingProvider>
    )

    fireEvent.click(screen.getByText('Start Onboarding'))
    expect(screen.getByTestId('current-step')).toHaveTextContent('0')

    // Try to go to previous step (should stay at 0)
    fireEvent.click(screen.getByText('Previous Step'))
    expect(screen.getByTestId('current-step')).toHaveTextContent('0')
  })

  it('throws error when used outside provider', () => {
    // Suppress console.error for this test
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {})

    expect(() => {
      render(<TestComponent />)
    }).toThrow('useOnboarding must be used within an OnboardingProvider')

    consoleSpy.mockRestore()
  })
})