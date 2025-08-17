'use client'

import React, { useRef, useState, useCallback } from 'react'
import { Upload, File, Image, FileText, Code, Archive, X, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { cn } from '@/lib/utils'

interface FileUploadZoneProps {
  onFilesSelected: (files: File[]) => void
  maxFiles?: number
  maxFileSize?: number // in MB
  acceptedTypes?: string[]
  children?: React.ReactNode
}

interface FileWithPreview extends File {
  id: string
  preview?: string
  uploadProgress?: number
  error?: string
}

export function FileUploadZone({
  onFilesSelected,
  maxFiles = 10,
  maxFileSize = 50, // 50MB default
  acceptedTypes = [
    'image/*',
    'text/*',
    'application/pdf',
    'application/json',
    'application/javascript',
    'application/typescript',
    '.md',
    '.py',
    '.js',
    '.ts',
    '.jsx',
    '.tsx',
    '.css',
    '.html',
    '.xml',
    '.yaml',
    '.yml',
    '.csv',
    '.txt'
  ],
  children
}: FileUploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [files, setFiles] = useState<FileWithPreview[]>([])
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const [errors, setErrors] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) return <Image className="h-4 w-4" />
    if (file.type.startsWith('text/') || file.name.endsWith('.md')) return <FileText className="h-4 w-4" />
    if (file.type.includes('javascript') || file.type.includes('typescript') || 
        file.name.match(/\.(js|ts|jsx|tsx|py|css|html)$/)) return <Code className="h-4 w-4" />
    if (file.type.includes('zip') || file.type.includes('archive')) return <Archive className="h-4 w-4" />
    return <File className="h-4 w-4" />
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxFileSize * 1024 * 1024) {
      return `File size exceeds ${maxFileSize}MB limit`
    }

    // Check file type
    const isAccepted = acceptedTypes.some(type => {
      if (type.startsWith('.')) {
        return file.name.toLowerCase().endsWith(type.toLowerCase())
      }
      if (type.includes('*')) {
        const baseType = type.split('/')[0]
        return file.type.startsWith(baseType)
      }
      return file.type === type
    })

    if (!isAccepted) {
      return 'File type not supported'
    }

    return null
  }

  const processFiles = useCallback(async (fileList: FileList) => {
    const newFiles: FileWithPreview[] = []
    const newErrors: string[] = []

    // Check total file count
    if (files.length + fileList.length > maxFiles) {
      newErrors.push(`Maximum ${maxFiles} files allowed`)
      setErrors(prev => [...prev, ...newErrors])
      return
    }

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i]
      const fileId = `${file.name}-${Date.now()}-${i}`
      
      const error = validateFile(file)
      if (error) {
        newErrors.push(`${file.name}: ${error}`)
        continue
      }

      const fileWithPreview: FileWithPreview = Object.assign(file, {
        id: fileId,
        uploadProgress: 0
      })

      // Generate preview for images
      if (file.type.startsWith('image/')) {
        try {
          const preview = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader()
            reader.onload = () => resolve(reader.result as string)
            reader.onerror = reject
            reader.readAsDataURL(file)
          })
          fileWithPreview.preview = preview
        } catch (error) {
          console.error('Error generating preview:', error)
        }
      }

      newFiles.push(fileWithPreview)
    }

    if (newErrors.length > 0) {
      setErrors(prev => [...prev, ...newErrors])
    }

    if (newFiles.length > 0) {
      setFiles(prev => [...prev, ...newFiles])
      onFilesSelected(newFiles)
    }
  }, [files.length, maxFiles, maxFileSize, acceptedTypes, onFilesSelected])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const droppedFiles = e.dataTransfer.files
    if (droppedFiles.length > 0) {
      processFiles(droppedFiles)
    }
  }, [processFiles])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files
    if (selectedFiles && selectedFiles.length > 0) {
      processFiles(selectedFiles)
    }
    // Reset input value to allow selecting the same file again
    e.target.value = ''
  }, [processFiles])

  const removeFile = (fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId))
  }

  const clearErrors = () => {
    setErrors([])
  }

  const openFileDialog = () => {
    fileInputRef.current?.click()
  }

  if (children) {
    return (
      <>
        <div onClick={openFileDialog}>
          {children}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={acceptedTypes.join(',')}
          onChange={handleFileSelect}
          className="hidden"
        />
      </>
    )
  }

  return (
    <div className="space-y-4">
      {/* Upload Zone */}
      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer",
          isDragOver
            ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
            : "border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={openFileDialog}
      >
        <Upload className="h-8 w-8 mx-auto mb-4 text-gray-400" />
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
          <span className="font-medium">Click to upload</span> or drag and drop
        </p>
        <p className="text-xs text-gray-500">
          Up to {maxFiles} files, max {maxFileSize}MB each
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Supports images, documents, code files, and more
        </p>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={acceptedTypes.join(',')}
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Error Messages */}
      {errors.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <div className="space-y-1">
              {errors.map((error, index) => (
                <div key={index}>{error}</div>
              ))}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearErrors}
              className="mt-2 h-6 px-2"
            >
              Dismiss
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* File List */}
      {files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">Selected Files ({files.length})</h4>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setFiles([])}
              className="h-6 px-2 text-xs"
            >
              Clear all
            </Button>
          </div>
          
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {files.map((file) => (
              <div
                key={file.id}
                className="flex items-center space-x-3 p-2 bg-gray-50 dark:bg-gray-800 rounded-lg"
              >
                {/* File Preview/Icon */}
                <div className="flex-shrink-0">
                  {file.preview ? (
                    <img
                      src={file.preview}
                      alt={file.name}
                      className="h-8 w-8 object-cover rounded"
                    />
                  ) : (
                    <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded flex items-center justify-center">
                      {getFileIcon(file)}
                    </div>
                  )}
                </div>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{file.name}</p>
                  <div className="flex items-center space-x-2 text-xs text-gray-500">
                    <span>{formatFileSize(file.size)}</span>
                    <Badge variant="secondary" className="text-xs">
                      {file.type || 'Unknown'}
                    </Badge>
                  </div>
                  
                  {/* Upload Progress */}
                  {file.uploadProgress !== undefined && file.uploadProgress < 100 && (
                    <Progress value={file.uploadProgress} className="h-1 mt-1" />
                  )}
                  
                  {/* Error */}
                  {file.error && (
                    <p className="text-xs text-red-500 mt-1">{file.error}</p>
                  )}
                </div>

                {/* Remove Button */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeFile(file.id)}
                  className="h-6 w-6 p-0 text-gray-400 hover:text-red-500"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}