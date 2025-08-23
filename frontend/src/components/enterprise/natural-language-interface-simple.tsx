'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Send, Sparkles } from 'lucide-react'

interface NaturalLanguageInterfaceProps {
  onQuerySubmit?: (query: string) => void
  className?: string
}

export function NaturalLanguageInterface({ onQuerySubmit, className }: NaturalLanguageInterfaceProps) {
  const [query, setQuery] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)

  const handleSubmitQuery = async () => {
    if (!query.trim()) return
    
    setIsProcessing(true)
    try {
      if (onQuerySubmit) {
        onQuerySubmit(query.trim())
      }
      setQuery('')
    } catch (error) {
      console.error('Query processing failed:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmitQuery()
    }
  }

  return (
    <div className={`space-y-6 ${className}`}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Natural Language Query Interface
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Ask questions in plain English about your business data, system performance, or analytics
          </p>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything about your business data, performance metrics, or insights..."
              className="min-h-[100px] pr-20 resize-none"
              disabled={isProcessing}
            />
            <div className="absolute bottom-3 right-3">
              <Button
                size="sm"
                onClick={handleSubmitQuery}
                disabled={!query.trim() || isProcessing}
              >
                {isProcessing ? (
                  <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}