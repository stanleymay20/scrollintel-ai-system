'use client'

import React, { useState, useCallback } from 'react'
import { FileUploadComponent } from '@/components/upload/file-upload'
import { FilePreviewComponent } from '@/components/upload/file-preview'
import { FileManagerComponent } from '@/components/upload/file-manager'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { FileUpload, FileHistory, FilePreview } from '@/types'
import { 
  Upload, 
  FileText, 
  Database, 
  Eye, 
  Download,
  Trash2,
  RefreshCw,
  CheckCircle,
  AlertCircle
} from 'lucide-react'

export default function FileUploadDemoPage() {
  const [uploadedFiles, setUploadedFiles] = useState<FileUpload[]>([])
  const [selectedFilePreview, setSelectedFilePreview] = useState<{
    fileId: string
    filename: string
    previewData?: FilePreview
  } | null>(null)
  const [fileHistory, setFileHistory] = useState<FileHistory[]>([
    {
      id: 'hist-1',
      filename: 'sales_data_2023.csv',
      upload_date: new Date('2024-01-15T10:30:00'),
      file_size: 2048576,
      status: 'completed',
      quality_score: 92,
      dataset_created: true
    },
    {
      id: 'hist-2',
      filename: 'customer_feedback.json',
      upload_date: new Date('2024-01-14T15:45:00'),
      file_size: 1536000,
      status: 'completed',
      quality_score: 87,
      dataset_created: false
    },
    {
      id: 'hist-3',
      filename: 'inventory_schema.sql',
      upload_date: new Date('2024-01-13T09:15:00'),
      file_size: 512000,
      status: 'completed',
      quality_score: 95,
      dataset_created: true
    }
  ])
  const [isLoading, setIsLoading] = useState(false)

  // Simulate file upload process
  const handleFileUpload = useCallback(async (files: File[]) => {
    for (const file of files) {
      const fileId = `upload-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      
      // Create initial file upload record
      const newFile: FileUpload = {
        id: fileId,
        name: file.name,
        filename: file.name,
        size: file.size,
        type: file.type,
        status: 'uploading',
        progress: 0,
        upload_time: new Date(),
        preview_available: false
      }
      
      setUploadedFiles(prev => [...prev, newFile])
      
      // Simulate upload progress
      const updateProgress = (progress: number, status: FileUpload['status'] = 'uploading') => {
        setUploadedFiles(prev => prev.map(f => 
          f.id === fileId 
            ? { ...f, progress, status }
            : f
        ))
      }
      
      // Simulate upload phases
      try {
        // Upload phase
        for (let i = 0; i <= 100; i += 10) {
          await new Promise(resolve => setTimeout(resolve, 100))
          updateProgress(i)
        }
        
        // Processing phase
        updateProgress(0, 'processing')
        for (let i = 0; i <= 100; i += 20) {
          await new Promise(resolve => setTimeout(resolve, 200))
          updateProgress(i, 'processing')
        }
        
        // Complete
        const processingTime = Math.random() * 3 + 0.5 // 0.5-3.5 seconds
        const qualityScore = Math.floor(Math.random() * 30) + 70 // 70-100
        
        setUploadedFiles(prev => prev.map(f => 
          f.id === fileId 
            ? { 
                ...f, 
                progress: 100, 
                status: 'completed',
                processing_time: processingTime,
                quality_score: qualityScore,
                preview_available: true
              }
            : f
        ))
        
        // Add to history
        const historyItem: FileHistory = {
          id: `hist-${fileId}`,
          filename: file.name,
          upload_date: new Date(),
          file_size: file.size,
          status: 'completed',
          quality_score: qualityScore,
          dataset_created: false
        }
        setFileHistory(prev => [historyItem, ...prev])
        
      } catch (error) {
        // Simulate error (10% chance)
        if (Math.random() < 0.1) {
          setUploadedFiles(prev => prev.map(f => 
            f.id === fileId 
              ? { 
                  ...f, 
                  status: 'error',
                  error_message: 'Failed to process file: Invalid format detected'
                }
              : f
          ))
        }
      }
    }
  }, [])

  const handleRemoveFile = useCallback((fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId))
  }, [])

  const handlePreviewFile = useCallback((fileId: string) => {
    const file = uploadedFiles.find(f => f.id === fileId)
    if (!file) return

    // Generate mock preview data
    const mockPreviewData: FilePreview = {
      columns: [
        { name: 'id', type: 'int64', inferred_type: 'integer' },
        { name: 'name', type: 'object', inferred_type: 'text' },
        { name: 'email', type: 'object', inferred_type: 'email' },
        { name: 'score', type: 'float64', inferred_type: 'float' },
        { name: 'category', type: 'object', inferred_type: 'categorical' },
        { name: 'created_at', type: 'object', inferred_type: 'datetime' }
      ],
      sample_data: [
        { id: 1, name: 'Alice Johnson', email: 'alice@example.com', score: 85.5, category: 'Premium', created_at: '2024-01-15T10:30:00Z' },
        { id: 2, name: 'Bob Smith', email: 'bob@example.com', score: 92.0, category: 'Standard', created_at: '2024-01-15T11:15:00Z' },
        { id: 3, name: 'Charlie Brown', email: 'charlie@example.com', score: 78.5, category: 'Premium', created_at: '2024-01-15T12:00:00Z' },
        { id: 4, name: 'Diana Prince', email: 'diana@example.com', score: 88.0, category: 'Enterprise', created_at: '2024-01-15T13:45:00Z' },
        { id: 5, name: 'Eve Wilson', email: 'eve@example.com', score: 95.5, category: 'Premium', created_at: '2024-01-15T14:30:00Z' }
      ],
      total_rows: 1250,
      preview_rows: 5,
      data_types: {
        'id': 'int64',
        'name': 'object',
        'email': 'object',
        'score': 'float64',
        'category': 'object',
        'created_at': 'object'
      },
      statistics: {
        'score': {
          'count': 1250,
          'mean': 87.3,
          'std': 12.4,
          'min': 45.0,
          'max': 100.0,
          '25%': 78.5,
          '50%': 88.0,
          '75%': 95.2
        }
      }
    }

    setSelectedFilePreview({
      fileId,
      filename: file.filename,
      previewData: mockPreviewData
    })
  }, [uploadedFiles])

  const handleDownloadFile = useCallback((fileId: string) => {
    const file = uploadedFiles.find(f => f.id === fileId)
    if (!file) return

    // Simulate file download
    const blob = new Blob(['Sample file content'], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = file.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [uploadedFiles])

  const handleCreateDataset = useCallback((fileId: string) => {
    const file = uploadedFiles.find(f => f.id === fileId)
    if (!file) return

    // Simulate dataset creation
    alert(`Dataset creation initiated for ${file.filename}. You will be notified when complete.`)
    
    // Update history to show dataset was created
    setFileHistory(prev => prev.map(item => 
      item.filename === file.filename 
        ? { ...item, dataset_created: true }
        : item
    ))
  }, [uploadedFiles])

  const handleRefresh = useCallback(() => {
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
    }, 1000)
  }, [])

  return (
    <div className="container mx-auto py-8 space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-scrollintel-primary">
          Advanced File Upload Demo
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Experience our advanced file upload system with drag-and-drop support, 
          real-time progress tracking, file validation, preview capabilities, and comprehensive file management.
        </p>
      </div>

      <div className="grid gap-8">
        {/* File Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              File Upload & Management
            </CardTitle>
          </CardHeader>
          <CardContent>
            <FileUploadComponent
              onFileUpload={handleFileUpload}
              uploadedFiles={uploadedFiles}
              onRemoveFile={handleRemoveFile}
              onPreviewFile={handlePreviewFile}
              onDownloadFile={handleDownloadFile}
              fileHistory={fileHistory}
              maxFiles={10}
              maxSize={100 * 1024 * 1024} // 100MB
              acceptedTypes={['.csv', '.xlsx', '.json', '.sql', '.txt']}
              enablePreview={true}
              enableHistory={true}
              enableValidation={true}
            />
          </CardContent>
        </Card>

        {/* File Manager Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              File Manager
            </CardTitle>
          </CardHeader>
          <CardContent>
            <FileManagerComponent
              files={uploadedFiles}
              fileHistory={fileHistory}
              onPreviewFile={handlePreviewFile}
              onDownloadFile={handleDownloadFile}
              onDeleteFile={handleRemoveFile}
              onCreateDataset={handleCreateDataset}
              onRefresh={handleRefresh}
              isLoading={isLoading}
            />
          </CardContent>
        </Card>

        {/* File Preview Section */}
        {selectedFilePreview && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                File Preview
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FilePreviewComponent
                fileId={selectedFilePreview.fileId}
                filename={selectedFilePreview.filename}
                previewData={selectedFilePreview.previewData}
                onClose={() => setSelectedFilePreview(null)}
                onDownload={() => handleDownloadFile(selectedFilePreview.fileId)}
                onCreateDataset={() => handleCreateDataset(selectedFilePreview.fileId)}
                isLoading={false}
              />
            </CardContent>
          </Card>
        )}

        {/* Features Overview */}
        <Card>
          <CardHeader>
            <CardTitle>Key Features</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Upload className="h-5 w-5 text-blue-500" />
                  <h3 className="font-semibold">Drag & Drop Upload</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Intuitive drag-and-drop interface with visual feedback and file validation
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <RefreshCw className="h-5 w-5 text-green-500" />
                  <h3 className="font-semibold">Progress Tracking</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Real-time progress bars for upload and processing with detailed status updates
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-purple-500" />
                  <h3 className="font-semibold">File Validation</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Comprehensive validation with format detection and security checks
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Eye className="h-5 w-5 text-orange-500" />
                  <h3 className="font-semibold">Data Preview</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Interactive data preview with schema analysis and statistics
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Database className="h-5 w-5 text-red-500" />
                  <h3 className="font-semibold">File Management</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Complete file management with search, filtering, and bulk operations
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-indigo-500" />
                  <h3 className="font-semibold">Multiple Formats</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Support for CSV, Excel, JSON, SQL, and more with auto-detection
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Usage Instructions */}
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Try it out:</strong> Drag and drop files into the upload area above, or click to select files. 
            Watch the real-time progress tracking, explore the file preview functionality, and test the file management features. 
            This demo simulates the complete file upload workflow including validation, processing, and quality analysis.
          </AlertDescription>
        </Alert>
      </div>
    </div>
  )
}