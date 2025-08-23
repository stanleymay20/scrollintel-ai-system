'use client'

import React, { useState, useEffect } from 'react'
import { scrollIntelApi } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  Download, 
  Database, 
  BarChart3, 
  TrendingUp, 
  Users, 
  ShoppingCart,
  DollarSign,
  Activity
} from 'lucide-react'

interface SampleDataset {
  id: string
  name: string
  description: string
  category: string
  size: string
  format: string
  icon: React.ComponentType<any>
  preview: {
    columns: string[]
    rows: number
    sampleData: Record<string, any>[]
  }
  useCases: string[]
  downloadUrl: string
}

export function SampleDataManager() {
  const [sampleDatasets, setSampleDatasets] = useState<SampleDataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<SampleDataset | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Load sample datasets from API
  useEffect(() => {
    const loadSampleDatasets = async () => {
      try {
        setIsLoading(true)
        setError(null)
        
        // Try to fetch from API first
        const response = await scrollIntelApi.getSampleDatasets()
        if (response?.data) {
          setSampleDatasets(response.data)
        } else {
          // Fallback to default datasets if API not available
          setSampleDatasets(getDefaultSampleDatasets())
        }
      } catch (err) {
        console.warn('Failed to load sample datasets from API, using defaults:', err)
        setSampleDatasets(getDefaultSampleDatasets())
      } finally {
        setIsLoading(false)
      }
    }

    loadSampleDatasets()
  }, [])

  const getDefaultSampleDatasets = (): SampleDataset[] => [
    {
      id: 'ecommerce-sales',
      name: 'E-commerce Sales Data',
      description: 'Sample e-commerce transaction data with customer demographics and purchase history.',
      category: 'Sales & Marketing',
      size: '2.5 MB',
      format: 'CSV',
      icon: ShoppingCart,
      preview: {
        columns: ['order_id', 'customer_id', 'product_name', 'category', 'price', 'quantity', 'order_date'],
        rows: 10000,
        sampleData: [
          { order_id: 'ORD-001', customer_id: 'CUST-123', product_name: 'Wireless Headphones', category: 'Electronics', price: 99.99, quantity: 1, order_date: '2024-01-15' },
          { order_id: 'ORD-002', customer_id: 'CUST-456', product_name: 'Running Shoes', category: 'Sports', price: 129.99, quantity: 2, order_date: '2024-01-16' },
          { order_id: 'ORD-003', customer_id: 'CUST-789', product_name: 'Coffee Maker', category: 'Home', price: 79.99, quantity: 1, order_date: '2024-01-17' },
        ]
      },
      useCases: [
        'Sales trend analysis',
        'Customer segmentation',
        'Product performance analysis',
        'Revenue forecasting'
      ],
      downloadUrl: '/api/sample-data/ecommerce-sales.csv'
    },
    {
      id: 'customer-analytics',
      name: 'Customer Analytics Dataset',
      description: 'Customer behavior data including demographics, engagement metrics, and churn indicators.',
      category: 'Customer Analytics',
      size: '1.8 MB',
      format: 'CSV',
      icon: Users,
      preview: {
        columns: ['customer_id', 'age', 'gender', 'location', 'signup_date', 'last_login', 'total_spent', 'churn_risk'],
        rows: 5000,
        sampleData: [
          { customer_id: 'CUST-001', age: 28, gender: 'F', location: 'New York', signup_date: '2023-06-15', last_login: '2024-01-20', total_spent: 1250.50, churn_risk: 'Low' },
          { customer_id: 'CUST-002', age: 35, gender: 'M', location: 'California', signup_date: '2023-03-22', last_login: '2024-01-18', total_spent: 890.25, churn_risk: 'Medium' },
          { customer_id: 'CUST-003', age: 42, gender: 'F', location: 'Texas', signup_date: '2023-01-10', last_login: '2023-12-15', total_spent: 2100.75, churn_risk: 'High' },
        ]
      },
      useCases: [
        'Churn prediction modeling',
        'Customer lifetime value analysis',
        'Demographic segmentation',
        'Engagement pattern analysis'
      ],
      downloadUrl: '/api/sample-data/customer-analytics.csv'
    },
    {
      id: 'financial-metrics',
      name: 'Financial Performance Data',
      description: 'Company financial metrics including revenue, expenses, and key performance indicators.',
      category: 'Finance',
      size: '950 KB',
      format: 'Excel',
      icon: DollarSign,
      preview: {
        columns: ['date', 'revenue', 'expenses', 'profit_margin', 'customer_acquisition_cost', 'monthly_recurring_revenue'],
        rows: 36,
        sampleData: [
          { date: '2024-01', revenue: 125000, expenses: 85000, profit_margin: 0.32, customer_acquisition_cost: 45, monthly_recurring_revenue: 98000 },
          { date: '2024-02', revenue: 132000, expenses: 88000, profit_margin: 0.33, customer_acquisition_cost: 42, monthly_recurring_revenue: 105000 },
          { date: '2024-03', revenue: 145000, expenses: 92000, profit_margin: 0.37, customer_acquisition_cost: 38, monthly_recurring_revenue: 115000 },
        ]
      },
      useCases: [
        'Financial trend analysis',
        'Budget planning and forecasting',
        'KPI dashboard creation',
        'Performance benchmarking'
      ],
      downloadUrl: '/api/sample-data/financial-metrics.xlsx'
    },
    {
      id: 'website-analytics',
      name: 'Website Analytics Data',
      description: 'Web traffic data including page views, user sessions, and conversion metrics.',
      category: 'Digital Marketing',
      size: '3.2 MB',
      format: 'JSON',
      icon: Activity,
      preview: {
        columns: ['date', 'page_views', 'unique_visitors', 'bounce_rate', 'avg_session_duration', 'conversion_rate'],
        rows: 365,
        sampleData: [
          { date: '2024-01-01', page_views: 15420, unique_visitors: 8750, bounce_rate: 0.42, avg_session_duration: 185, conversion_rate: 0.034 },
          { date: '2024-01-02', page_views: 18650, unique_visitors: 10200, bounce_rate: 0.38, avg_session_duration: 210, conversion_rate: 0.041 },
          { date: '2024-01-03', page_views: 12890, unique_visitors: 7650, bounce_rate: 0.45, avg_session_duration: 165, conversion_rate: 0.028 },
        ]
      },
      useCases: [
        'Traffic pattern analysis',
        'Conversion optimization',
        'User behavior insights',
        'Marketing campaign effectiveness'
      ],
      downloadUrl: '/api/sample-data/website-analytics.json'
    }
  ]

  const handleDownload = async (dataset: SampleDataset) => {
    try {
      console.log(`Downloading ${dataset.name}...`)
      
      // Try to download from API first
      try {
        const response = await scrollIntelApi.downloadSampleDataset(dataset.id)
        const blob = new Blob([response.data])
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${dataset.id}.${dataset.format.toLowerCase()}`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
        return
      } catch (apiError) {
        console.warn('API download failed, generating sample data:', apiError)
      }
      
      // Fallback: Create a sample CSV content for demonstration
      const csvContent = generateSampleCSV(dataset)
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${dataset.id}.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Download failed:', error)
      // Could show a toast notification here
    }
  }

  const generateSampleCSV = (dataset: SampleDataset): string => {
    const headers = dataset.preview.columns.join(',')
    const rows = dataset.preview.sampleData.map(row => 
      dataset.preview.columns.map(col => row[col]).join(',')
    ).join('\n')
    return `${headers}\n${rows}`
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Sales & Marketing':
        return TrendingUp
      case 'Customer Analytics':
        return Users
      case 'Finance':
        return DollarSign
      case 'Digital Marketing':
        return Activity
      default:
        return Database
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Sample Datasets</h2>
          <p className="text-muted-foreground">
            Get started quickly with pre-loaded sample data for testing and learning.
          </p>
        </div>
        <Badge variant="secondary">
          {sampleDatasets.length} Datasets Available
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {sampleDatasets.map((dataset) => {
          const Icon = dataset.icon
          const CategoryIcon = getCategoryIcon(dataset.category)
          
          return (
            <Card key={dataset.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-scrollintel-primary rounded-lg flex items-center justify-center">
                      <Icon className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-base">{dataset.name}</CardTitle>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline" className="text-xs">
                          <CategoryIcon className="h-3 w-3 mr-1" />
                          {dataset.category}
                        </Badge>
                        <Badge variant="secondary" className="text-xs">
                          {dataset.format}
                        </Badge>
                        <Badge variant="secondary" className="text-xs">
                          {dataset.size}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  {dataset.description}
                </p>

                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Dataset Preview:</h4>
                  <div className="bg-muted p-3 rounded-lg text-xs">
                    <div className="flex justify-between mb-2">
                      <span>{dataset.preview.rows.toLocaleString()} rows</span>
                      <span>{dataset.preview.columns.length} columns</span>
                    </div>
                    <div className="font-mono text-xs">
                      {dataset.preview.columns.slice(0, 4).join(' | ')}
                      {dataset.preview.columns.length > 4 && ' | ...'}
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Use Cases:</h4>
                  <ul className="text-xs text-muted-foreground space-y-1">
                    {dataset.useCases.map((useCase, index) => (
                      <li key={index}>• {useCase}</li>
                    ))}
                  </ul>
                </div>

                <div className="flex gap-2">
                  <Button
                    onClick={() => setSelectedDataset(dataset)}
                    variant="outline"
                    size="sm"
                    className="flex-1"
                  >
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Preview
                  </Button>
                  <Button
                    onClick={() => handleDownload(dataset)}
                    size="sm"
                    className="flex-1"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Dataset Preview Modal */}
      {selectedDataset && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <Card className="w-full max-w-4xl max-h-[80vh] overflow-auto">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{selectedDataset.name} - Preview</CardTitle>
                <Button
                  variant="ghost"
                  onClick={() => setSelectedDataset(null)}
                >
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      {selectedDataset.preview.columns.map((column) => (
                        <th key={column} className="text-left p-2 font-medium">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selectedDataset.preview.sampleData.map((row, index) => (
                      <tr key={index} className="border-b">
                        {selectedDataset.preview.columns.map((column) => (
                          <td key={column} className="p-2">
                            {row[column]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={() => setSelectedDataset(null)}
                >
                  Close
                </Button>
                <Button onClick={() => handleDownload(selectedDataset)}>
                  <Download className="h-4 w-4 mr-2" />
                  Download Dataset
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}