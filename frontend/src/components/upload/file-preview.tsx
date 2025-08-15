'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { FilePreview } from '@/types'
import { 
  Eye, 
  Download, 
  BarChart3, 
  Table, 
  Info, 
  AlertCircle,
  CheckCircle,
  X,
  FileText,
  Database
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface FilePreviewProps {
  fileId: string
  filename: string
  previewData?: FilePreview
  onClose?: () => void
  onDownload?: () => void
  onCreateDataset?: () => void
  isLoading?: boolean
}

export function FilePreviewComponent({
  fileId,
  filename,
  previewData,
  onClose,
  onDownload,
  onCreateDataset,
  isLoading = false
}: FilePreviewProps) {
  const [activeTab, setActiveTab] = useState('data')

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Loading Preview...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-scrollintel-primary"></div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!previewData) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            Preview Not Available
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Preview data is not available for this file. The file may still be processing or an error occurred.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  const getColumnTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'integer':
      case 'float':
      case 'number':
        return <BarChart3 className="h-4 w-4 text-blue-500" />
      case 'datetime':
      case 'date':
        return <Database className="h-4 w-4 text-purple-500" />
      case 'boolean':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'categorical':
        return <Badge className="h-4 w-4 text-orange-500" />
      default:
        return <FileText className="h-4 w-4 text-gray-500" />
    }
  }

  const getColumnTypeBadge = (type: string) => {
    const colors = {
      'integer': 'bg-blue-100 text-blue-800',
      'float': 'bg-blue-100 text-blue-800',
      'number': 'bg-blue-100 text-blue-800',
      'text': 'bg-gray-100 text-gray-800',
      'datetime': 'bg-purple-100 text-purple-800',
      'date': 'bg-purple-100 text-purple-800',
      'boolean': 'bg-green-100 text-green-800',
      'categorical': 'bg-orange-100 text-orange-800',
      'email': 'bg-pink-100 text-pink-800',
      'url': 'bg-indigo-100 text-indigo-800',
      'phone': 'bg-yellow-100 text-yellow-800'
    }
    
    const colorClass = colors[type.toLowerCase() as keyof typeof colors] || 'bg-gray-100 text-gray-800'
    
    return (
      <Badge variant="outline" className={cn('text-xs', colorClass)}>
        {type}
      </Badge>
    )
  }

  const formatValue = (value: any) => {
    if (value === null || value === undefined || value === '') {
      return <span className="text-muted-foreground italic">null</span>
    }
    
    if (typeof value === 'string' && value.length > 50) {
      return (
        <span title={value}>
          {value.substring(0, 50)}...
        </span>
      )
    }
    
    return String(value)
  }

  return (
    <Card className="w-full max-w-6xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5 text-scrollintel-primary" />
            File Preview: {filename}
          </CardTitle>
          <div className="flex items-center gap-2">
            {onDownload && (
              <Button variant="outline" size="sm" onClick={onDownload}>
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
            )}
            {onCreateDataset && (
              <Button size="sm" onClick={onCreateDataset}>
                <Database className="h-4 w-4 mr-2" />
                Create Dataset
              </Button>
            )}
            {onClose && (
              <Button variant="ghost" size="sm" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="data" className="flex items-center gap-2">
              <Table className="h-4 w-4" />
              Data ({previewData.preview_rows} rows)
            </TabsTrigger>
            <TabsTrigger value="schema" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Schema ({previewData.columns.length} columns)
            </TabsTrigger>
            <TabsTrigger value="stats" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Statistics
            </TabsTrigger>
          </TabsList>

          <TabsContent value="data" className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Showing {previewData.preview_rows} of {previewData.total_rows} rows
              </div>
              {previewData.preview_rows < previewData.total_rows && (
                <Alert className="w-auto">
                  <Info className="h-4 w-4" />
                  <AlertDescription className="text-xs">
                    Preview limited to first {previewData.preview_rows} rows
                  </AlertDescription>
                </Alert>
              )}
            </div>

            <div className="border rounded-lg overflow-hidden">
              <div className="overflow-x-auto max-h-96">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 sticky top-0">
                    <tr>
                      {previewData.columns.map((column, idx) => (
                        <th key={idx} className="px-3 py-2 text-left font-medium border-r last:border-r-0">
                          <div className="flex items-center gap-2">
                            {getColumnTypeIcon(column.inferred_type)}
                            <span className="truncate max-w-32" title={column.name}>
                              {column.name}
                            </span>
                          </div>
                          <div className="mt-1">
                            {getColumnTypeBadge(column.inferred_type)}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.sample_data.map((row, rowIdx) => (
                      <tr key={rowIdx} className="border-t hover:bg-muted/25">
                        {previewData.columns.map((column, colIdx) => (
                          <td key={colIdx} className="px-3 py-2 border-r last:border-r-0 max-w-48">
                            <div className="truncate">
                              {formatValue(row[column.name])}
                            </div>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="schema" className="space-y-4">
            <div className="grid gap-4">
              {previewData.columns.map((column, idx) => (
                <Card key={idx} className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getColumnTypeIcon(column.inferred_type)}
                      <h4 className="font-medium">{column.name}</h4>
                    </div>
                    <div className="flex items-center gap-2">
                      {getColumnTypeBadge(column.inferred_type)}
                      <Badge variant="outline" className="text-xs">
                        {column.type}
                      </Badge>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Data Type:</span>
                      <div className="font-medium">{column.type}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Inferred Type:</span>
                      <div className="font-medium">{column.inferred_type}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Column Index:</span>
                      <div className="font-medium">{idx + 1}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Sample Values:</span>
                      <div className="font-medium text-xs">
                        Available in data tab
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="stats" className="space-y-4">
            {previewData.statistics ? (
              <div className="grid gap-4">
                {Object.entries(previewData.statistics).map(([column, stats]) => (
                  <Card key={column} className="p-4">
                    <h4 className="font-medium mb-3 flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      {column}
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                      {Object.entries(stats as Record<string, number>).map(([stat, value]) => (
                        <div key={stat}>
                          <span className="text-muted-foreground capitalize">{stat}:</span>
                          <div className="font-medium">
                            {typeof value === 'number' ? value.toFixed(2) : String(value)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  Statistical analysis is only available for numeric columns. No numeric columns were detected in this file.
                </AlertDescription>
              </Alert>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}