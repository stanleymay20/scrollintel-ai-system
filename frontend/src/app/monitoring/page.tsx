'use client'

import React from 'react'
import { MonitoringDashboard } from '@/components/monitoring/monitoring-dashboard'

export default function MonitoringPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <MonitoringDashboard refreshInterval={30000} />
    </div>
  )
}