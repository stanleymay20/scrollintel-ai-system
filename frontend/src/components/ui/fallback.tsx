'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { AlertCircle, RefreshCw, Wifi, WifiOff } from 'lucide-react'

interface FallbackUIProps {
  title?: string
  message?: string
  showRetry?: boolean
  onRetry?: () => void
  type?: 'error' | 'offline' | 'empty' | 'loading'
}

export function FallbackUI({ 
  title = "Something went wrong",
  message = "We're having trouble loading this content.",
  showRetry = true,
  onRetry,
  type = 'error'
}: FallbackUIProps) {
  const getIcon = () => {
    switch (type) {
      case 'offline':
        return <WifiOff className="h-12 w-12 text-muted-foreground" />
      case 'empty':
        return <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground">?</div>
      case 'loading':
        return <div className="h-12 w-12 rounded-full bg-muted animate-pulse" />
      default:
        return <AlertCircle className="h-12 w-12 text-red-500" />
    }
  }

  const getColors = () => {
    switch (type) {
      case 'offline':
        return 'text-orange-600'
      case 'empty':
        return 'text-muted-foreground'
      case 'loading':
        return 'text-blue-600'
      default:
        return 'text-red-600'
    }
  }

  return (
    <div className="flex items-center justify-center p-8">
      <Card className="w-full max-w-md text-center">
        <CardHeader>
          <div className="flex justify-center mb-4">
            {getIcon()}
          </div>
          <CardTitle className={getColors()}>
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">
            {message}
          </p>
          
          {showRetry && (
            <div className="flex gap-2 justify-center">
              <Button 
                onClick={onRetry || (() => window.location.reload())}
                variant="outline"
                className="flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Try Again
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export function OfflineIndicator() {
  const [isOnline, setIsOnline] = React.useState(true)

  React.useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  if (isOnline) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-orange-500 text-white p-2 text-center text-sm">
      <div className="flex items-center justify-center gap-2">
        <WifiOff className="h-4 w-4" />
        You're currently offline. Some features may not work.
      </div>
    </div>
  )
}