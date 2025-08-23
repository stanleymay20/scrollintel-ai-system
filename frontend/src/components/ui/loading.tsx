'use client'

import React from 'react'
import { cn } from '@/lib/utils'
import { Loader2 } from 'lucide-react'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8'
  }

  return (
    <Loader2 
      className={cn(
        'animate-spin text-scrollintel-primary',
        sizeClasses[size],
        className
      )} 
    />
  )
}

interface LoadingStateProps {
  loading?: boolean
  error?: string | null
  children: React.ReactNode
  fallback?: React.ReactNode
  errorFallback?: React.ReactNode
}

export function LoadingState({ 
  loading, 
  error, 
  children, 
  fallback,
  errorFallback 
}: LoadingStateProps) {
  if (error) {
    return errorFallback || (
      <div className="flex items-center justify-center p-8 text-red-600">
        <p>Error: {error}</p>
      </div>
    )
  }

  if (loading) {
    return fallback || (
      <div className="flex items-center justify-center p-8">
        <LoadingSpinner />
      </div>
    )
  }

  return <>{children}</>
}

interface SkeletonProps {
  className?: string
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div 
      className={cn(
        'animate-pulse rounded-md bg-muted',
        className
      )} 
    />
  )
}

export function CardSkeleton() {
  return (
    <div className="p-6 space-y-4">
      <Skeleton className="h-4 w-[250px]" />
      <Skeleton className="h-4 w-[200px]" />
      <Skeleton className="h-4 w-[150px]" />
    </div>
  )
}

export function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="p-6 border rounded-lg">
            <Skeleton className="h-4 w-[100px] mb-2" />
            <Skeleton className="h-8 w-[60px]" />
          </div>
        ))}
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Skeleton className="h-6 w-[120px]" />
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <CardSkeleton key={i} />
            ))}
          </div>
        </div>
        
        <div className="space-y-4">
          <Skeleton className="h-6 w-[100px]" />
          <div className="h-96 border rounded-lg p-4">
            <Skeleton className="h-full w-full" />
          </div>
        </div>
      </div>
    </div>
  )
}