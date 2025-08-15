'use client'

import { FileUploadComponent } from '@/components/upload/file-upload'

export default function UploadPage() {
  return (
    <div className="flex-1 p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">File Upload</h1>
        <p className="text-muted-foreground">Upload files for processing by ScrollIntel AI</p>
      </div>
      
      <div className="max-w-4xl">
        <FileUploadComponent onFileUpload={(files) => console.log('Files uploaded:', files)} />
      </div>
    </div>
  )
}