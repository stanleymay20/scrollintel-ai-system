'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Database, Plus, RefreshCw, Settings, CheckCircle, AlertCircle, Clock } from 'lucide-react'

const dataSources = [
  {
    id: '1',
    name: 'PostgreSQL Production',
    type: 'Database',
    status: 'connected',
    lastSync: '2 minutes ago',
    records: '2.4M',
    description: 'Main production database with user and transaction data'
  },
  {
    id: '2', 
    name: 'AWS S3 Data Lake',
    type: 'Cloud Storage',
    status: 'connected',
    lastSync: '15 minutes ago',
    records: '850GB',
    description: 'Raw data files and processed analytics datasets'
  },
  {
    id: '3',
    name: 'Salesforce CRM',
    type: 'API',
    status: 'syncing',
    lastSync: '1 hour ago',
    records: '45K',
    description: 'Customer relationship management data'
  },
  {
    id: '4',
    name: 'Google Analytics',
    type: 'API',
    status: 'error',
    lastSync: '6 hours ago',
    records: '1.2M',
    description: 'Website traffic and user behavior analytics'
  },
  {
    id: '5',
    name: 'Stripe Payments',
    type: 'API', 
    status: 'connected',
    lastSync: '5 minutes ago',
    records: '128K',
    description: 'Payment transactions and subscription data'
  }
]

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'connected':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'syncing':
      return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />
    case 'error':
      return <AlertCircle className="h-4 w-4 text-red-500" />
    default:
      return <Clock className="h-4 w-4 text-gray-500" />
  }
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'connected':
      return 'success'
    case 'syncing':
      return 'secondary'
    case 'error':
      return 'error'
    default:
      return 'secondary'
  }
}

export default function DataPage() {
  return (
    <div className="flex-1 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">Data Sources</h1>
          <p className="text-muted-foreground">Manage and monitor your data connections</p>
        </div>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          Add Data Source
        </Button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Sources</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dataSources.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Connected</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dataSources.filter(ds => ds.status === 'connected').length}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Syncing</CardTitle>
            <RefreshCw className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dataSources.filter(ds => ds.status === 'syncing').length}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Errors</CardTitle>
            <AlertCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dataSources.filter(ds => ds.status === 'error').length}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Sources List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {dataSources.map((source) => (
          <Card key={source.id}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  <CardTitle className="text-lg">{source.name}</CardTitle>
                </div>
                <Badge variant={getStatusColor(source.status) as any} className="flex items-center gap-1">
                  {getStatusIcon(source.status)}
                  {source.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">{source.description}</p>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Type:</span>
                    <div className="font-medium">{source.type}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Records:</span>
                    <div className="font-medium">{source.records}</div>
                  </div>
                </div>

                <div className="text-xs text-muted-foreground">
                  Last sync: {source.lastSync}
                </div>

                <div className="flex gap-2">
                  <Button size="sm" variant="outline">
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Sync Now
                  </Button>
                  <Button size="sm" variant="outline">
                    <Settings className="h-3 w-3 mr-1" />
                    Configure
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}