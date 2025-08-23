'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { SystemMetrics } from '@/types'
import { Cpu, MemoryStick, Users, Activity, Clock, AlertTriangle } from 'lucide-react'

interface SystemMetricsProps {
  metrics: SystemMetrics
}

export function SystemMetricsCard({ metrics }: SystemMetricsProps) {
  const getMetricColor = (value: number, thresholds: { warning: number; danger: number }) => {
    if (value >= thresholds.danger) return 'text-red-500'
    if (value >= thresholds.warning) return 'text-yellow-500'
    return 'text-green-500'
  }

  const safeValue = (value: number | undefined, fallback: number = 0): number => {
    return typeof value === 'number' && !isNaN(value) ? value : fallback
  }

  const metricItems = [
    {
      icon: <Cpu className="h-5 w-5" />,
      label: 'CPU Usage',
      value: safeValue(metrics.cpu_usage),
      unit: '%',
      thresholds: { warning: 70, danger: 90 },
      showProgress: true,
    },
    {
      icon: <MemoryStick className="h-5 w-5" />,
      label: 'Memory Usage',
      value: safeValue(metrics.memory_usage),
      unit: '%',
      thresholds: { warning: 80, danger: 95 },
      showProgress: true,
    },
    {
      icon: <Users className="h-5 w-5" />,
      label: 'Active Connections',
      value: safeValue(metrics.active_connections),
      unit: '',
      thresholds: { warning: 50, danger: 100 },
      showProgress: false,
    },
    {
      icon: <Activity className="h-5 w-5" />,
      label: 'Disk Usage',
      value: safeValue(metrics.disk_usage),
      unit: '%',
      thresholds: { warning: 80, danger: 95 },
      showProgress: true,
    },
    {
      icon: <Clock className="h-5 w-5" />,
      label: 'Response Time',
      value: safeValue(metrics.response_time),
      unit: 'ms',
      thresholds: { warning: 1000, danger: 3000 },
      showProgress: false,
    },
    {
      icon: <AlertTriangle className="h-5 w-5" />,
      label: 'Uptime',
      value: safeValue(metrics.uptime, 99.9),
      unit: '%',
      thresholds: { warning: 95, danger: 90 },
      showProgress: false,
    },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-scrollintel-primary" />
          System Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {metricItems.map((item) => (
            <div key={item.label} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {item.icon}
                  <span className="text-sm font-medium">{item.label}</span>
                </div>
                <span className={`text-sm font-semibold ${getMetricColor(item.value, item.thresholds)}`}>
                  {item.value.toLocaleString()}{item.unit}
                </span>
              </div>
              {item.showProgress && (
                <Progress 
                  value={item.value} 
                  className="h-2"
                />
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}