'use client'

import React, { createContext, useContext, useState, useEffect } from 'react'

export interface OnboardingStep {
  id: string
  title: string
  description: string
  target?: string
  content: React.ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
  skippable?: boolean
  action?: () => void
}

export interface OnboardingState {
  isActive: boolean
  currentStep: number
  steps: OnboardingStep[]
  isCompleted: boolean
  showTooltips: boolean
}

interface OnboardingContextType {
  state: OnboardingState
  startOnboarding: (steps: OnboardingStep[]) => void
  nextStep: () => void
  prevStep: () => void
  skipStep: () => void
  completeOnboarding: () => void
  showTooltip: (target: string, content: React.ReactNode) => void
  hideTooltip: () => void
  toggleTooltips: () => void
}

const OnboardingContext = createContext<OnboardingContextType | undefined>(undefined)

export function OnboardingProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<OnboardingState>({
    isActive: false,
    currentStep: 0,
    steps: [],
    isCompleted: false,
    showTooltips: true,
  })

  // Check if user has completed onboarding
  useEffect(() => {
    const completed = localStorage.getItem('scrollintel-onboarding-completed')
    if (completed === 'true') {
      setState(prev => ({ ...prev, isCompleted: true }))
    }
  }, [])

  const startOnboarding = (steps: OnboardingStep[]) => {
    setState(prev => ({
      ...prev,
      isActive: true,
      currentStep: 0,
      steps,
      isCompleted: false,
    }))
  }

  const nextStep = () => {
    setState(prev => {
      const nextStepIndex = prev.currentStep + 1
      if (nextStepIndex >= prev.steps.length) {
        return {
          ...prev,
          isActive: false,
          isCompleted: true,
        }
      }
      return {
        ...prev,
        currentStep: nextStepIndex,
      }
    })
  }

  const prevStep = () => {
    setState(prev => ({
      ...prev,
      currentStep: Math.max(0, prev.currentStep - 1),
    }))
  }

  const skipStep = () => {
    const currentStepData = state.steps[state.currentStep]
    if (currentStepData?.skippable !== false) {
      nextStep()
    }
  }

  const completeOnboarding = () => {
    setState(prev => ({
      ...prev,
      isActive: false,
      isCompleted: true,
    }))
    localStorage.setItem('scrollintel-onboarding-completed', 'true')
  }

  const showTooltip = (target: string, content: React.ReactNode) => {
    // Implementation for contextual tooltips
    console.log('Show tooltip for:', target, content)
  }

  const hideTooltip = () => {
    // Implementation for hiding tooltips
    console.log('Hide tooltip')
  }

  const toggleTooltips = () => {
    setState(prev => ({
      ...prev,
      showTooltips: !prev.showTooltips,
    }))
  }

  const value: OnboardingContextType = {
    state,
    startOnboarding,
    nextStep,
    prevStep,
    skipStep,
    completeOnboarding,
    showTooltip,
    hideTooltip,
    toggleTooltips,
  }

  return (
    <OnboardingContext.Provider value={value}>
      {children}
    </OnboardingContext.Provider>
  )
}

export function useOnboarding() {
  const context = useContext(OnboardingContext)
  if (context === undefined) {
    throw new Error('useOnboarding must be used within an OnboardingProvider')
  }
  return context
}