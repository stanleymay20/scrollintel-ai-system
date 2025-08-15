'use client'

import React from 'react'
import { useOnboarding } from './onboarding-provider'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { X, ChevronLeft, ChevronRight, SkipForward } from 'lucide-react'

export function OnboardingModal() {
  const { state, nextStep, prevStep, skipStep, completeOnboarding } = useOnboarding()

  if (!state.isActive || state.steps.length === 0) {
    return null
  }

  const currentStepData = state.steps[state.currentStep]
  const progress = ((state.currentStep + 1) / state.steps.length) * 100
  const isLastStep = state.currentStep === state.steps.length - 1

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl max-h-[80vh] overflow-auto">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <div className="flex items-center gap-3">
            <CardTitle className="text-xl">{currentStepData.title}</CardTitle>
            <Badge variant="secondary">
              Step {state.currentStep + 1} of {state.steps.length}
            </Badge>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={completeOnboarding}
            className="h-8 w-8"
          >
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>

          {/* Step Description */}
          <p className="text-muted-foreground">{currentStepData.description}</p>

          {/* Step Content */}
          <div className="min-h-[200px]">
            {currentStepData.content}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between pt-4 border-t">
            <div className="flex items-center gap-2">
              {state.currentStep > 0 && (
                <Button variant="outline" onClick={prevStep}>
                  <ChevronLeft className="h-4 w-4 mr-2" />
                  Previous
                </Button>
              )}
            </div>

            <div className="flex items-center gap-2">
              {currentStepData.skippable !== false && !isLastStep && (
                <Button variant="ghost" onClick={skipStep}>
                  <SkipForward className="h-4 w-4 mr-2" />
                  Skip
                </Button>
              )}
              
              <Button onClick={isLastStep ? completeOnboarding : nextStep}>
                {isLastStep ? 'Complete' : 'Next'}
                {!isLastStep && <ChevronRight className="h-4 w-4 ml-2" />}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}