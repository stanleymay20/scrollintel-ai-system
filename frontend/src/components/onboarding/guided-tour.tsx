'use client'

import React, { useEffect, useState } from 'react'
import { useOnboarding } from './onboarding-provider'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { X, ChevronLeft, ChevronRight } from 'lucide-react'

interface GuidedTourProps {
  target?: string
}

export function GuidedTour({ target }: GuidedTourProps) {
  const { state, nextStep, prevStep, completeOnboarding } = useOnboarding()
  const [position, setPosition] = useState({ top: 0, left: 0 })
  const [targetElement, setTargetElement] = useState<HTMLElement | null>(null)

  useEffect(() => {
    if (!state.isActive || !target) return

    const element = document.querySelector(`[data-tour="${target}"]`) as HTMLElement
    if (element) {
      setTargetElement(element)
      
      // Calculate position
      const rect = element.getBoundingClientRect()
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop
      const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
      
      setPosition({
        top: rect.top + scrollTop + rect.height + 10,
        left: rect.left + scrollLeft,
      })

      // Highlight the target element
      element.style.position = 'relative'
      element.style.zIndex = '1000'
      element.style.boxShadow = '0 0 0 4px rgba(59, 130, 246, 0.5)'
      element.style.borderRadius = '8px'
      
      // Scroll into view
      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }

    return () => {
      if (element) {
        element.style.position = ''
        element.style.zIndex = ''
        element.style.boxShadow = ''
        element.style.borderRadius = ''
      }
    }
  }, [state.isActive, target, state.currentStep])

  if (!state.isActive || !target || !targetElement) {
    return null
  }

  const currentStepData = state.steps[state.currentStep]
  const isLastStep = state.currentStep === state.steps.length - 1

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 z-40 bg-black/30" />
      
      {/* Tour Card */}
      <Card 
        className="fixed z-50 w-80 shadow-lg"
        style={{
          top: position.top,
          left: Math.min(position.left, window.innerWidth - 320 - 20),
        }}
      >
        <CardContent className="p-4 space-y-4">
          <div className="flex items-center justify-between">
            <Badge variant="secondary">
              {state.currentStep + 1} of {state.steps.length}
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              onClick={completeOnboarding}
              className="h-6 w-6"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>

          <div>
            <h3 className="font-semibold text-sm mb-2">{currentStepData.title}</h3>
            <p className="text-sm text-muted-foreground">{currentStepData.description}</p>
          </div>

          <div className="flex items-center justify-between">
            {state.currentStep > 0 ? (
              <Button variant="outline" size="sm" onClick={prevStep}>
                <ChevronLeft className="h-3 w-3 mr-1" />
                Back
              </Button>
            ) : (
              <div />
            )}

            <Button size="sm" onClick={isLastStep ? completeOnboarding : nextStep}>
              {isLastStep ? 'Finish' : 'Next'}
              {!isLastStep && <ChevronRight className="h-3 w-3 ml-1" />}
            </Button>
          </div>
        </CardContent>
      </Card>
    </>
  )
}