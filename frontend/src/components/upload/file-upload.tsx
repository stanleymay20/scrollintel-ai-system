'use client'

import React, { useCallback, useState, useRef, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { FileUpload, FileValidationResult, FilePreview, FileHistory } from '@/types'
import { 
  Upload, 
  File, 
  Image, 
  FileText, 
  Database, 
  X, 
  CheckCircle, 
  AlertCircle,
  Loader2,
  Eye,
  Download,
  History,
  Star,
  AlertTriangle,
  FileCheck,
  Clock,
  Trash2,
  RefreshCw
} from 'lucide-react'
import { cn, formatBytes } from '@/lib/utils'

interface FileUploadProps {
  onFileUpload?: (files: File[]) => void
  uploadedFiles?: FileUpload[]
  onRemoveFile?: (fileId: string) => void
  onPreviewFile?: (fileId: string) => void
  onDownloadFile?: (fileId: string) => void
  fileHistory?: FileHistory[]
  maxFiles?: number
  maxSize?: number
  acceptedTypes?: string[]
  enablePreview?: boolean
  enableHistory?: boolean
  enableValidation?: boolean
}

export function FileUploadComponent({ 
  onFileUpload,
  uploadedFiles = [],
  onRemoveFile,
  onPreviewFile,
  onDownloadFile,
  fileHistory = [],
  maxFiles = 10,
  maxSize = 100 * 1024 * 1024, // 100MB
  acceptedTypes = ['.csv', '.xlsx', '.json', '.sql', '.txt', '.pdf'],
  enablePreview = true,
  enableHistory = true,
  enableValidation = true
}: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [validationResults, setValidationResults] = useState<Record<string, FileValidationResult>>({})
  const [previewData, setPreviewData] = useState<Record<string, FilePreview>>({})
  const [activeTab, setActiveTab] = useState('upload')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // File validation function
  const validateFile = useCallback((file: File): FileValidationResult => {
    const errors: string[] = []
    const warnings: string[] = []

    // Size validation
    if (file.size > maxSize) {
      errors.push(`File size (${formatBytes(file.size)}) exceeds maximum allowed size (${formatBytes(maxSize)})`)
    }

    if (file.size < 100) {
      warnings.push('File is very small and may be empty')
    }

    // Type validation
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!acceptedTypes.includes(fileExtension)) {
      errors.push(`File type ${fileExtension} is not supported. Supported types: ${acceptedTypes.join(', ')}`)
    }

    // Name validation
    if (file.name.length > 255) {
      errors.push('Filename is too long (max 255 characters)')
    }

    const dangerousChars = /[<>:"|?*\x00-\x1f]/
    if (dangerousChars.test(file.name)) {
      errors.push('Filename contains invalid characters')
    }

    // Duplicate check
    const isDuplicate = uploadedFiles.some(uploaded => 
      uploaded.filename === file.name && uploaded.size === file.size
    )
    if (isDuplicate) {
      warnings.push('A file with the same name and size has already been uploaded')
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      file_info: {
        name: file.name,
        size: file.size,
        type: file.type,
        extension: fileExtension
      }
    }
  }, [maxSize, acceptedTypes, uploadedFiles])

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Validate accepted files
    const validatedFiles: File[] = []
    const newValidationResults: Record<string, FileValidationResult> = {}

    acceptedFiles.forEach(file => {
      const validation = validateFile(file)
      const fileKey = `${file.name}-${file.size}-${file.lastModified}`
      newValidationResults[fileKey] = validation

      if (validation.valid) {
        validatedFiles.push(file)
      }
    })

    // Handle rejected files
    rejectedFiles.forEach(({ file, errors }) => {
      const fileKey = `${file.name}-${file.size}-${file.lastModified}`
      newValidationResults[fileKey] = {
        valid: false,
        errors: errors.map((e: any) => e.message),
        warnings: [],
        file_info: {
          name: file.name,
          size: file.size,
          type: file.type,
          extension: '.' + file.name.split('.').pop()?.toLowerCase()
        }
      }
    })

    setValidationResults(prev => ({ ...prev, ...newValidationResults }))

    if (validatedFiles.length > 0) {
      onFileUpload?.(validatedFiles)
    }
  }, [onFileUpload, validateFile])

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    maxFiles,
    maxSize,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json'],
      'application/sql': ['.sql'],
      'text/plain': ['.txt', '.sql'],
      'application/pdf': ['.pdf'],
    },
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    noClick: true, // We'll handle clicks manually
  })

  const getFileIcon = (type: string, filename?: string) => {
    if (type.includes('image')) return <Image className="h-5 w-5 text-blue-500" />
    if (type.includes('csv') || type.includes('excel') || filename?.endsWith('.xlsx')) return <Database className="h-5 w-5 text-green-500" />
    if (type.includes('json')) return <FileText className="h-5 w-5 text-yellow-500" />
    if (type.includes('sql') || filename?.endsWith('.sql')) return <Database className="h-5 w-5 text-purple-500" />
    if (type.includes('pdf')) return <File className="h-5 w-5 text-red-500" />
    return <File className="h-5 w-5 text-gray-500" />
  }

  const getStatusIcon = (status: FileUpload['status']) => {
    switch (status) {
      case 'uploading':
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
      case 'processing':
        return <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return null
    }
  }

  const getStatusColor = (status: FileUpload['status']) => {
    switch (status) {
      case 'uploading':
        return 'default'
      case 'processing':
        return 'secondary'
      case 'completed':
        return 'default'
      case 'error':
        return 'destructive'
      default:
        return 'secondary'
    }
  }

  const getQualityBadge = (score?: number) => {
    if (!score) return null
    
    if (score >= 90) return <Badge variant="default" className="bg-green-100 text-green-800">Excellent</Badge>
    if (score >= 75) return <Badge variant="default" className="bg-blue-100 text-blue-800">Good</Badge>
    if (score >= 60) return <Badge variant="default" className="bg-yellow-100 text-yellow-800">Fair</Badge>
    return <Badge variant="destructive">Poor</Badge>
  }

  const formatUploadTime = (date?: Date) => {
    if (!date) return ''
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)

    if (days > 0) return `${days}d ago`
    if (hours > 0) return `${hours}h ago`
    if (minutes > 0) return `${minutes}m ago`
    return 'Just now'
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5 text-scrollintel-primary" />
          Advanced File Upload
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="files" className="flex items-center gap-2">
              <FileCheck className="h-4 w-4" />
              Files ({uploadedFiles.length})
            </TabsTrigger>
            {enableHistory && (
              <TabsTrigger value="history" className="flex items-center gap-2">
                <History className="h-4 w-4" />
                History
              </TabsTrigger>
            )}
          </TabsList>

          <TabsContent value="upload" className="space-y-4">
            {/* Validation Results */}
            {enableValidation && Object.keys(validationResults).length > 0 && (
              <div className="space-y-2">
                {Object.entries(validationResults).map(([key, result]) => (
                  <Alert key={key} variant={result.valid ? "default" : "destructive"}>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-medium">{result.file_info.name}</div>
                      {result.errors.map((error, idx) => (
                        <div key={idx} className="text-sm text-red-600">{error}</div>
                      ))}
                      {result.warnings.map((warning, idx) => (
                        <div key={idx} className="text-sm text-yellow-600">{warning}</div>
                      ))}
                    </AlertDescription>
                  </Alert>
                ))}
              </div>
            )}

            {/* Drop Zone */}
            <div
              {...getRootProps()}
              className={cn(
                "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200",
                isDragActive || dragActive
                  ? "border-scrollintel-primary bg-scrollintel-primary/5 scale-105"
                  : "border-muted-foreground/25 hover:border-scrollintel-primary/50 hover:bg-muted/50"
              )}
              onClick={open}
            >
              <input {...getInputProps()} ref={fileInputRef} />
              <div className={cn(
                "transition-transform duration-200",
                isDragActive && "scale-110"
              )}>
                <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              </div>
              
              {isDragActive ? (
                <div className="animate-pulse">
                  <p className="text-lg font-medium text-scrollintel-primary">
                    Drop files here
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Release to upload
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-lg font-medium mb-2">
                    Drag & drop files here, or click to select
                  </p>
                  <p className="text-sm text-muted-foreground mb-4">
                    Supports: {acceptedTypes.join(', ')}
                  </p>
                  <p className="text-xs text-muted-foreground mb-4">
                    Max {maxFiles} files, {formatBytes(maxSize)} each
                  </p>
                  <Button variant="outline" size="sm">
                    <Upload className="h-4 w-4 mr-2" />
                    Choose Files
                  </Button>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="files" className="space-y-4">
            {uploadedFiles.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <File className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No files uploaded yet</p>
                <Button variant="outline" size="sm" className="mt-2" onClick={() => setActiveTab('upload')}>
                  Upload Files
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                {uploadedFiles.map((file) => (
                  <Card key={file.id} className="p-4">
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-1">
                        {getFileIcon(file.type, file.filename)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <p className="text-sm font-medium truncate">
                              {file.filename}
                            </p>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <span>{formatBytes(file.size)}</span>
                              {file.upload_time && (
                                <>
                                  <span>•</span>
                                  <span>{formatUploadTime(file.upload_time)}</span>
                                </>
                              )}
                              {file.processing_time && (
                                <>
                                  <span>•</span>
                                  <span>{file.processing_time.toFixed(1)}s</span>
                                </>
                              )}
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-2">
                            {getQualityBadge(file.quality_score)}
                            <Badge variant={getStatusColor(file.status)} className="flex items-center gap-1">
                              {getStatusIcon(file.status)}
                              {file.status}
                            </Badge>
                          </div>
                        </div>
                        
                        {/* Progress Bar */}
                        {(file.status === 'uploading' || file.status === 'processing') && (
                          <div className="mb-2">
                            <div className="flex items-center justify-between text-xs mb-1">
                              <span className="text-muted-foreground">
                                {file.status === 'uploading' ? 'Uploading...' : 'Processing...'}
                              </span>
                              <span className="font-medium">{file.progress}%</span>
                            </div>
                            <Progress value={file.progress} className="h-2" />
                          </div>
                        )}
                        
                        {/* Error Message */}
                        {file.status === 'error' && file.error_message && (
                          <Alert variant="destructive" className="mb-2">
                            <AlertCircle className="h-4 w-4" />
                            <AlertDescription className="text-xs">
                              {file.error_message}
                            </AlertDescription>
                          </Alert>
                        )}
                        
                        {/* Action Buttons */}
                        <div className="flex items-center gap-2">
                          {enablePreview && file.status === 'completed' && file.preview_available && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => onPreviewFile?.(file.id)}
                            >
                              <Eye className="h-3 w-3 mr-1" />
                              Preview
                            </Button>
                          )}
                          
                          {file.status === 'completed' && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => onDownloadFile?.(file.id)}
                            >
                              <Download className="h-3 w-3 mr-1" />
                              Download
                            </Button>
                          )}
                          
                          {file.status === 'error' && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                // Retry upload logic would go here
                              }}
                            >
                              <RefreshCw className="h-3 w-3 mr-1" />
                              Retry
                            </Button>
                          )}
                          
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => onRemoveFile?.(file.id)}
                            className="text-red-600 hover:text-red-700 hover:bg-red-50"
                          >
                            <Trash2 className="h-3 w-3 mr-1" />
                            Remove
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {enableHistory && (
            <TabsContent value="history" className="space-y-4">
              {fileHistory.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No upload history</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {fileHistory.map((item) => (
                    <div key={item.id} className="flex items-center gap-3 p-3 border rounded-lg hover:bg-muted/50">
                      <div className="flex-shrink-0">
                        {getFileIcon('', item.filename)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium truncate">{item.filename}</p>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <span>{formatBytes(item.file_size)}</span>
                              <span>•</span>
                              <span>{formatUploadTime(item.upload_date)}</span>
                              {item.dataset_created && (
                                <>
                                  <span>•</span>
                                  <Badge variant="outline" className="text-xs">Dataset Created</Badge>
                                </>
                              )}
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-2">
                            {getQualityBadge(item.quality_score)}
                            <Badge variant="outline">{item.status}</Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </TabsContent>
          )}
        </Tabs>
      </CardContent>
    </Card>
  )
}