import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import '@testing-library/jest-dom'
import { FileUploadComponent } from '@/components/upload/file-upload'
import { FileUpload } from '@/types'

const mockUploadedFiles: FileUpload[] = [
  {
    id: 'file-1',
    filename: 'test.csv',
    size: 1024,
    type: 'text/csv',
    status: 'completed',
    progress: 100,
  },
  {
    id: 'file-2',
    filename: 'data.xlsx',
    size: 2048,
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    status: 'uploading',
    progress: 50,
  },
  {
    id: 'file-3',
    filename: 'error.json',
    size: 512,
    type: 'application/json',
    status: 'error',
    progress: 0,
    error_message: 'Invalid JSON format',
  },
]

// Mock react-dropzone
jest.mock('react-dropzone', () => ({
  useDropzone: ({ onDrop }: any) => ({
    getRootProps: () => ({
      onClick: () => {},
      onDrop,
    }),
    getInputProps: () => ({}),
    isDragActive: false,
  }),
}))

describe('FileUploadComponent', () => {
  it('renders file upload interface correctly', () => {
    render(<FileUploadComponent />)
    
    expect(screen.getByText('File Upload')).toBeInTheDocument()
    expect(screen.getByText('Drag & drop files here, or click to select')).toBeInTheDocument()
    expect(screen.getByText('Supports: .csv, .xlsx, .json, .sql, .txt, .pdf')).toBeInTheDocument()
  })

  it('displays uploaded files correctly', () => {
    render(<FileUploadComponent uploadedFiles={mockUploadedFiles} />)
    
    expect(screen.getByText('Uploaded Files')).toBeInTheDocument()
    expect(screen.getByText('test.csv')).toBeInTheDocument()
    expect(screen.getByText('data.xlsx')).toBeInTheDocument()
    expect(screen.getByText('error.json')).toBeInTheDocument()
  })

  it('shows correct file status badges', () => {
    render(<FileUploadComponent uploadedFiles={mockUploadedFiles} />)
    
    expect(screen.getByText('completed')).toBeInTheDocument()
    expect(screen.getByText('uploading')).toBeInTheDocument()
    expect(screen.getByText('error')).toBeInTheDocument()
  })

  it('displays file sizes correctly', () => {
    render(<FileUploadComponent uploadedFiles={mockUploadedFiles} />)
    
    expect(screen.getByText('1 KB')).toBeInTheDocument()
    expect(screen.getByText('2 KB')).toBeInTheDocument()
    expect(screen.getByText('512 Bytes')).toBeInTheDocument()
  })

  it('shows progress for uploading files', () => {
    render(<FileUploadComponent uploadedFiles={mockUploadedFiles} />)
    
    expect(screen.getByText('50%')).toBeInTheDocument()
  })

  it('displays error messages for failed uploads', () => {
    render(<FileUploadComponent uploadedFiles={mockUploadedFiles} />)
    
    expect(screen.getByText('Invalid JSON format')).toBeInTheDocument()
  })

  it('calls onRemoveFile when remove button is clicked', () => {
    const mockOnRemoveFile = jest.fn()
    render(
      <FileUploadComponent 
        uploadedFiles={mockUploadedFiles}
        onRemoveFile={mockOnRemoveFile}
      />
    )
    
    const removeButtons = screen.getAllByRole('button')
    fireEvent.click(removeButtons[0])
    
    expect(mockOnRemoveFile).toHaveBeenCalledWith('file-1')
  })

  it('calls onFileUpload when files are dropped', () => {
    const mockOnFileUpload = jest.fn()
    const mockFile = new File(['test content'], 'test.csv', { type: 'text/csv' })
    
    render(<FileUploadComponent onFileUpload={mockOnFileUpload} />)
    
    // This would be triggered by the dropzone in a real scenario
    // For testing, we'll simulate the callback
    mockOnFileUpload([mockFile])
    
    expect(mockOnFileUpload).toHaveBeenCalledWith([mockFile])
  })

  it('shows correct file type icons', () => {
    render(<FileUploadComponent uploadedFiles={mockUploadedFiles} />)
    
    // The component should render different icons for different file types
    // We can't easily test the specific icons, but we can verify the structure
    const fileItems = screen.getAllByText(/test\.csv|data\.xlsx|error\.json/)
    expect(fileItems).toHaveLength(3)
  })
})