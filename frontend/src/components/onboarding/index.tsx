'use client'

import React from 'react'
import { OnboardingProvider } from './onboarding-provider'
import { OnboardingModal } from './onboarding-modal'
import { GuidedTour } from './guided-tour'
import { WelcomeWizard } from './welcome-wizard'
import { Agent } from '@/types'

interface OnboardingSystemProps {
  agents: Agent[]
  children: React.ReactNode
}

export function OnboardingSystem({ agents, children }: OnboardingSystemProps) {
  return (
    <OnboardingProvider>
      {children}
      <WelcomeWizard agents={agents} />
      <OnboardingModal />
      <GuidedTour />
    </OnboardingProvider>
  )
}

// Export all components for individual use
export { OnboardingProvider, useOnboarding } from './onboarding-provider'
export { OnboardingModal } from './onboarding-modal'
export { GuidedTour } from './guided-tour'
export { ContextualTooltip } from './contextual-tooltip'
export { AgentTutorials } from './agent-tutorials'
export { SampleDataManager } from './sample-data-manager'
export { WelcomeWizard } from './welcome-wizard'

// Export types
export type { OnboardingStep } from './onboarding-provider'