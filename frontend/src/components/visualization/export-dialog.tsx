'use client'

import React, { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Download, 
  FileText, 
  FileSpreadsheet, 
  Image, 
  File,
  Settings,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react'

interface ExportDialogProps {
  trigger?: React.ReactNode
  chartIds?: string[]
  dashboardId?: string
  onExport?: (exportData: any) => void
}

const exportFormats = [
  { 
    value: 'pdf', 
    label: 'PDF Report', 
    icon: FileText, 
    description: 'Professional report with charts and data',
    features: ['Print-friendly', 'Charts as images', 'Data tables', 'Custom branding']
  },
  { 
    value: 'excel', 
    label: 'Excel Workbook', 
    icon: FileSpreadsheet, 
    description: 'Interactive spreadsheet with charts and data',
    features: ['Multiple sheets', 'Interactive charts', 'Raw data', 'Formulas']
  },
  { 
    value: 'csv', 
    label: 'CSV Data', 
    icon: File, 
    description: 'Raw data in comma-separated format',
    features: ['Raw data only', 'Universal format', 'Lightweight', 'Easy import']
  },
  { 
    value: 'png', 
    label: 'PNG Images', 
    icon: Image, 
    description: 'High-quality chart images',
    features: ['High resolution', 'Transparent background', 'Web-ready', 'Print quality']
  },
  { 
    value: 'json', 
    label: 'JSON Data', 
    icon: File, 
    description: 'Structured data with metadata',
    features: ['Complete data', 'Chart configs', 'Metadata', 'API-friendly']
  }
]

export function ExportDialog({ trigger, chartIds, dashboardId, onExport }: ExportDialogProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedFormat, setSelectedFormat] = useState('pdf')
  const [exportConfig, setExportConfig] = useState({
    custom_title: '',
    custom_description: '',
    include_data: true,
    include_metadata: true,
    page_size: 'A4',
    orientation: 'portrait',
    charts_per_page: 2,
    include_summary: true,
    include_filters: true
  })
  const [isExporting, setIsExporting] = useState(false)
  const [exportProgress, setExportProgress] = useState(0)
  const [exportResult, setExportResult] = useState<any>(null)

  const handleExport = async () => {
    setIsExporting(true)
    setExportProgress(0)

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setExportProgress(prev => Math.min(prev + 10, 90))
      }, 200)

      const exportRequest = {
        format: selectedFormat,
        chart_ids: chartIds,
        dashboard_id: dashboardId,
        ...exportConfig
      }

      const response = await fetch('/api/visualization/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exportRequest)
      })

      clearInterval(progressInterval)
      setExportProgress(100)

      if (response.ok) {
        const result = await response.json()
        setExportResult(result)
        onExport?.(result)
      } else {
        throw new Error('Export failed')
      }
    } catch (error) {
      console.error('Export error:', error)
      setExportResult({ success: false, error: 'Export failed' })
    } finally {
      setIsExporting(false)
    }
  }

  const handleDownload = (downloadUrl: string, filename: string) => {
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const resetDialog = () => {
    setExportResult(null)
    setExportProgress(0)
    setIsExporting(false)
  }

  const selectedFormatInfo = exportFormats.find(f => f.value === selectedFormat)

  return (
    <Dialog open={isOpen} onOpenChange={(open) => {
      setIsOpen(open)
      if (!open) resetDialog()
    }}>
      <DialogTrigger asChild>
        {trigger || (
          <Button variant="outline" className="flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export
          </Button>
        )}
      </DialogTrigger>
      
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Export Visualization
          </DialogTitle>
        </DialogHeader>

        {!exportResult ? (
          <div className="space-y-6">
            {/* Format Selection */}
            <div className="space-y-4">
              <Label className="text-base font-medium">Export Format</Label>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {exportFormats.map((format) => {
                  const Icon = format.icon
                  return (
                    <Card
                      key={format.value}
                      className={`cursor-pointer transition-all ${
                        selectedFormat === format.value 
                          ? 'ring-2 ring-blue-500 bg-blue-50' 
                          : 'hover:bg-gray-50'
                      }`}
                      onClick={() => setSelectedFormat(format.value)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start gap-3">
                          <Icon className="h-6 w-6 text-blue-600 mt-1" />
                          <div className="flex-1">
                            <h3 className="font-medium">{format.label}</h3>
                            <p className="text-sm text-gray-600 mt-1">{format.description}</p>
                            <div className="flex flex-wrap gap-1 mt-2">
                              {format.features.slice(0, 2).map((feature) => (
                                <Badge key={feature} variant="outline" className="text-xs">
                                  {feature}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </div>

            {/* Export Configuration */}
            <div className="space-y-4">
              <Label className="text-base font-medium">Export Configuration</Label>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="title">Custom Title</Label>
                  <Input
                    id="title"
                    value={exportConfig.custom_title}
                    onChange={(e) => setExportConfig(prev => ({ ...prev, custom_title: e.target.value }))}
                    placeholder="Enter custom title"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={exportConfig.custom_description}
                    onChange={(e) => setExportConfig(prev => ({ ...prev, custom_description: e.target.value }))}
                    placeholder="Enter description"
                    rows={3}
                  />
                </div>
              </div>

              {/* Format-specific options */}
              {selectedFormat === 'pdf' && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <Label>Page Size</Label>
                    <Select
                      value={exportConfig.page_size}
                      onValueChange={(value) => setExportConfig(prev => ({ ...prev, page_size: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="A4">A4</SelectItem>
                        <SelectItem value="Letter">Letter</SelectItem>
                        <SelectItem value="Legal">Legal</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Orientation</Label>
                    <Select
                      value={exportConfig.orientation}
                      onValueChange={(value) => setExportConfig(prev => ({ ...prev, orientation: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="portrait">Portrait</SelectItem>
                        <SelectItem value="landscape">Landscape</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Charts per Page</Label>
                    <Select
                      value={exportConfig.charts_per_page.toString()}
                      onValueChange={(value) => setExportConfig(prev => ({ ...prev, charts_per_page: parseInt(value) }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">1</SelectItem>
                        <SelectItem value="2">2</SelectItem>
                        <SelectItem value="3">3</SelectItem>
                        <SelectItem value="4">4</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {/* Include options */}
              <div className="space-y-3">
                <Label>Include in Export</Label>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="include-data"
                      checked={exportConfig.include_data}
                      onCheckedChange={(checked) => 
                        setExportConfig(prev => ({ ...prev, include_data: !!checked }))
                      }
                    />
                    <Label htmlFor="include-data">Raw Data</Label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="include-metadata"
                      checked={exportConfig.include_metadata}
                      onCheckedChange={(checked) => 
                        setExportConfig(prev => ({ ...prev, include_metadata: !!checked }))
                      }
                    />
                    <Label htmlFor="include-metadata">Metadata</Label>
                  </div>

                  {selectedFormat === 'pdf' && (
                    <>
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="include-summary"
                          checked={exportConfig.include_summary}
                          onCheckedChange={(checked) => 
                            setExportConfig(prev => ({ ...prev, include_summary: !!checked }))
                          }
                        />
                        <Label htmlFor="include-summary">Summary</Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="include-filters"
                          checked={exportConfig.include_filters}
                          onCheckedChange={(checked) => 
                            setExportConfig(prev => ({ ...prev, include_filters: !!checked }))
                          }
                        />
                        <Label htmlFor="include-filters">Applied Filters</Label>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Export Progress */}
            {isExporting && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Exporting...</span>
                </div>
                <Progress value={exportProgress} className="w-full" />
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex justify-between">
              <Button variant="outline" onClick={() => setIsOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleExport} disabled={isExporting}>
                {isExporting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Export {selectedFormatInfo?.label}
                  </>
                )}
              </Button>
            </div>
          </div>
        ) : (
          /* Export Result */
          <div className="space-y-6">
            {exportResult.success ? (
              <div className="text-center space-y-4">
                <CheckCircle className="h-16 w-16 text-green-500 mx-auto" />
                <div>
                  <h3 className="text-lg font-medium">Export Successful!</h3>
                  <p className="text-gray-600">Your visualization has been exported successfully.</p>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Format:</span>
                      <span className="ml-2 font-medium">{selectedFormatInfo?.label}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Size:</span>
                      <span className="ml-2 font-medium">
                        {exportResult.size ? `${(exportResult.size / 1024).toFixed(1)} KB` : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                <Button
                  onClick={() => handleDownload(exportResult.download_url, exportResult.export_id)}
                  className="w-full"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download File
                </Button>
              </div>
            ) : (
              <div className="text-center space-y-4">
                <AlertCircle className="h-16 w-16 text-red-500 mx-auto" />
                <div>
                  <h3 className="text-lg font-medium">Export Failed</h3>
                  <p className="text-gray-600">{exportResult.error || 'An error occurred during export.'}</p>
                </div>
                <Button onClick={resetDialog} variant="outline">
                  Try Again
                </Button>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}