'use client'

import React, { useState, useRef, useEffect } from 'react'
import { useOnboarding } from './onboarding-provider'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { HelpCircle, X } from 'lucide-react'

interface ContextualTooltipProps {
  target: string
  title: string
  content: string
  position?: 'top' | 'bottom' | 'left' | 'right'
  trigger?: 'hover' | 'click'
  children: React.ReactNode
}

export function ContextualTooltip({
  target,
  title,
  content,
  position = 'top',
  trigger = 'hover',
  children,
}: ContextualTooltipProps) {
  const { state } = useOnboarding()
  const [isVisible, setIsVisible] = useState(false)
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 })
  const triggerRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isVisible && triggerRef.current && tooltipRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect()
      const tooltipRect = tooltipRef.current.getBoundingClientRect()
      
      let top = 0
      let left = 0

      switch (position) {
        case 'top':
          top = triggerRect.top - tooltipRect.height - 8
          left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2
          break
        case 'bottom':
          top = triggerRect.bottom + 8
          left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2
          break
        case 'left':
          top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2
          left = triggerRect.left - tooltipRect.width - 8
          break
        case 'right':
          top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2
          left = triggerRect.right + 8
          break
      }

      // Ensure tooltip stays within viewport
      const padding = 8
      top = Math.max(padding, Math.min(top, window.innerHeight - tooltipRect.height - padding))
      left = Math.max(padding, Math.min(left, window.innerWidth - tooltipRect.width - padding))

      setTooltipPosition({ top, left })
    }
  }, [isVisible, position])

  if (!state.showTooltips) {
    return <div data-tour={target}>{children}</div>
  }

  const handleMouseEnter = () => {
    if (trigger === 'hover') {
      setIsVisible(true)
    }
  }

  const handleMouseLeave = () => {
    if (trigger === 'hover') {
      setIsVisible(false)
    }
  }

  const handleClick = () => {
    if (trigger === 'click') {
      setIsVisible(!isVisible)
    }
  }

  return (
    <>
      <div
        ref={triggerRef}
        data-tour={target}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        className="relative inline-block"
      >
        {children}
        {trigger === 'click' && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute -top-2 -right-2 h-5 w-5 rounded-full bg-scrollintel-primary text-white hover:bg-scrollintel-primary/80"
          >
            <HelpCircle className="h-3 w-3" />
          </Button>
        )}
      </div>

      {isVisible && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsVisible(false)} />
          <Card
            ref={tooltipRef}
            className="fixed z-50 w-64 shadow-lg"
            style={{
              top: tooltipPosition.top,
              left: tooltipPosition.left,
            }}
          >
            <CardContent className="p-3 space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-sm">{title}</h4>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsVisible(false)}
                  className="h-4 w-4"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">{content}</p>
            </CardContent>
          </Card>
        </>
      )}
    </>
  )
}