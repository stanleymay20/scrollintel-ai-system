'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { FileUpload, FileHistory } from '@/types'
import { 
  Search,
  Filter,
  Download,
  Trash2,
  Eye,
  Database,
  FileText,
  Calendar,
  SortAsc,
  SortDesc,
  RefreshCw,
  Archive,
  Star,
  MoreHorizontal
} from 'lucide-react'
import { cn, formatBytes } from '@/lib/utils'

interface FileManagerProps {
  files: FileUpload[]
  fileHistory: FileHistory[]
  onPreviewFile?: (fileId: string) => void
  onDownloadFile?: (fileId: string) => void
  onDeleteFile?: (fileId: string) => void
  onCreateDataset?: (fileId: string) => void
  onRefresh?: () => void
  isLoading?: boolean
}

type SortField = 'name' | 'size' | 'date' | 'status' | 'quality'
type SortDirection = 'asc' | 'desc'
type FilterStatus = 'all' | 'completed' | 'processing' | 'error' | 'uploading'

export function FileManagerComponent({
  files,
  fileHistory,
  onPreviewFile,
  onDownloadFile,
  onDeleteFile,
  onCreateDataset,
  onRefresh,
  isLoading = false
}: FileManagerProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<FilterStatus>('all')
  const [sortField, setSortField] = useState<SortField>('date')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list')

  // Filter and sort files
  const filteredAndSortedFiles = React.useMemo(() => {
    let filtered = files.filter(file => {
      const matchesSearch = file.filename.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesStatus = statusFilter === 'all' || file.status === statusFilter
      return matchesSearch && matchesStatus
    })

    filtered.sort((a, b) => {
      let aValue: any, bValue: any

      switch (sortField) {
        case 'name':
          aValue = a.filename.toLowerCase()
          bValue = b.filename.toLowerCase()
          break
        case 'size':
          aValue = a.size
          bValue = b.size
          break
        case 'date':
          aValue = a.upload_time ? new Date(a.upload_time).getTime() : 0
          bValue = b.upload_time ? new Date(b.upload_time).getTime() : 0
          break
        case 'status':
          aValue = a.status
          bValue = b.status
          break
        case 'quality':
          aValue = a.quality_score || 0
          bValue = b.quality_score || 0
          break
        default:
          return 0
      }

      if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1
      if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1
      return 0
    })

    return filtered
  }, [files, searchTerm, statusFilter, sortField, sortDirection])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const handleSelectFile = (fileId: string) => {
    const newSelected = new Set(selectedFiles)
    if (newSelected.has(fileId)) {
      newSelected.delete(fileId)
    } else {
      newSelected.add(fileId)
    }
    setSelectedFiles(newSelected)
  }

  const handleSelectAll = () => {
    if (selectedFiles.size === filteredAndSortedFiles.length) {
      setSelectedFiles(new Set())
    } else {
      setSelectedFiles(new Set(filteredAndSortedFiles.map(f => f.id)))
    }
  }

  const handleBulkDelete = () => {
    selectedFiles.forEach(fileId => {
      onDeleteFile?.(fileId)
    })
    setSelectedFiles(new Set())
  }

  const getStatusIcon = (status: FileUpload['status']) => {
    switch (status) {
      case 'uploading':
        return <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />
      case 'processing':
        return <RefreshCw className="h-4 w-4 animate-spin text-yellow-500" />
      case 'completed':
        return <Database className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertDescription className="h-4 w-4 text-red-500" />
      default:
        return <FileText className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status: FileUpload['status']) => {
    const variants = {
      uploading: 'default',
      processing: 'secondary',
      completed: 'default',
      error: 'destructive'
    }
    
    return (
      <Badge variant={variants[status] as any} className="flex items-center gap-1">
        {getStatusIcon(status)}
        {status}
      </Badge>
    )
  }

  const getQualityBadge = (score?: number) => {
    if (!score) return null
    
    if (score >= 90) return <Badge className="bg-green-100 text-green-800">Excellent</Badge>
    if (score >= 75) return <Badge className="bg-blue-100 text-blue-800">Good</Badge>
    if (score >= 60) return <Badge className="bg-yellow-100 text-yellow-800">Fair</Badge>
    return <Badge variant="destructive">Poor</Badge>
  }

  const formatDate = (date?: Date) => {
    if (!date) return 'Unknown'
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(new Date(date))
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Archive className="h-5 w-5 text-scrollintel-primary" />
            File Manager ({filteredAndSortedFiles.length} files)
          </CardTitle>
          <div className="flex items-center gap-2">
            {onRefresh && (
              <Button variant="outline" size="sm" onClick={onRefresh} disabled={isLoading}>
                <RefreshCw className={cn("h-4 w-4 mr-2", isLoading && "animate-spin")} />
                Refresh
              </Button>
            )}
            {selectedFiles.size > 0 && (
              <Button variant="destructive" size="sm" onClick={handleBulkDelete}>
                <Trash2 className="h-4 w-4 mr-2" />
                Delete Selected ({selectedFiles.size})
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Filters and Search */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>
          
          <Select value={statusFilter} onValueChange={(value) => setStatusFilter(value as FilterStatus)}>
            <SelectTrigger className="w-48">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="processing">Processing</SelectItem>
              <SelectItem value="uploading">Uploading</SelectItem>
              <SelectItem value="error">Error</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Sort Controls */}
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Sort by:</span>
          {(['name', 'size', 'date', 'status', 'quality'] as SortField[]).map((field) => (
            <Button
              key={field}
              variant={sortField === field ? "default" : "ghost"}
              size="sm"
              onClick={() => handleSort(field)}
              className="flex items-center gap-1"
            >
              {field.charAt(0).toUpperCase() + field.slice(1)}
              {sortField === field && (
                sortDirection === 'asc' ? <SortAsc className="h-3 w-3" /> : <SortDesc className="h-3 w-3" />
              )}
            </Button>
          ))}
        </div>

        {/* Files List */}
        {filteredAndSortedFiles.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No files found</p>
            {searchTerm || statusFilter !== 'all' ? (
              <Button 
                variant="outline" 
                size="sm" 
                className="mt-2"
                onClick={() => {
                  setSearchTerm('')
                  setStatusFilter('all')
                }}
              >
                Clear Filters
              </Button>
            ) : null}
          </div>
        ) : (
          <div className="space-y-2">
            {/* Select All Header */}
            <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
              <input
                type="checkbox"
                checked={selectedFiles.size === filteredAndSortedFiles.length && filteredAndSortedFiles.length > 0}
                onChange={handleSelectAll}
                className="rounded"
              />
              <span className="text-sm font-medium">
                {selectedFiles.size > 0 ? `${selectedFiles.size} selected` : 'Select all'}
              </span>
            </div>

            {/* Files */}
            {filteredAndSortedFiles.map((file) => (
              <Card key={file.id} className={cn(
                "p-4 transition-colors",
                selectedFiles.has(file.id) && "bg-scrollintel-primary/5 border-scrollintel-primary"
              )}>
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={selectedFiles.has(file.id)}
                    onChange={() => handleSelectFile(file.id)}
                    className="mt-1 rounded"
                  />
                  
                  <div className="flex-shrink-0 mt-1">
                    <FileText className="h-5 w-5 text-muted-foreground" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-medium truncate">{file.filename}</h4>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <span>{formatBytes(file.size)}</span>
                          <span>•</span>
                          <span>{formatDate(file.upload_time)}</span>
                          {file.processing_time && (
                            <>
                              <span>•</span>
                              <span>{file.processing_time.toFixed(1)}s processing</span>
                            </>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {getQualityBadge(file.quality_score)}
                        {getStatusBadge(file.status)}
                      </div>
                    </div>
                    
                    {/* Progress Bar for Active Files */}
                    {(file.status === 'uploading' || file.status === 'processing') && (
                      <div className="mb-3">
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span>{file.status === 'uploading' ? 'Uploading...' : 'Processing...'}</span>
                          <span>{file.progress}%</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div 
                            className="bg-scrollintel-primary h-2 rounded-full transition-all duration-300"
                            style={{ width: `${file.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                    
                    {/* Error Message */}
                    {file.status === 'error' && file.error_message && (
                      <Alert variant="destructive" className="mb-3">
                        <AlertDescription className="text-sm">
                          {file.error_message}
                        </AlertDescription>
                      </Alert>
                    )}
                    
                    {/* Action Buttons */}
                    <div className="flex items-center gap-2">
                      {file.status === 'completed' && file.preview_available && onPreviewFile && (
                        <Button variant="outline" size="sm" onClick={() => onPreviewFile(file.id)}>
                          <Eye className="h-3 w-3 mr-1" />
                          Preview
                        </Button>
                      )}
                      
                      {file.status === 'completed' && onDownloadFile && (
                        <Button variant="outline" size="sm" onClick={() => onDownloadFile(file.id)}>
                          <Download className="h-3 w-3 mr-1" />
                          Download
                        </Button>
                      )}
                      
                      {file.status === 'completed' && onCreateDataset && (
                        <Button size="sm" onClick={() => onCreateDataset(file.id)}>
                          <Database className="h-3 w-3 mr-1" />
                          Create Dataset
                        </Button>
                      )}
                      
                      {onDeleteFile && (
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          onClick={() => onDeleteFile(file.id)}
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                        >
                          <Trash2 className="h-3 w-3 mr-1" />
                          Delete
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}