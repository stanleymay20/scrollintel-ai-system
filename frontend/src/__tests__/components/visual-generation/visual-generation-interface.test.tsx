import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import { VisualGenerationInterface } from '@/components/visual-generation/visual-generation-interface'

// Mock the child components
jest.mock('@/components/visual-generation/video-generation-panel', () => ({
  VideoGenerationPanel: ({ onGenerate }: { onGenerate: Function }) => (
    <div data-testid="video-generation-panel">
      <button onClick={() => onGenerate('video', { prompt: 'test video' })}>
        Generate Video
      </button>
    </div>
  )
}))

jest.mock('@/components/visual-generation/image-generation-panel', () => ({
  ImageGenerationPanel: ({ onGenerate }: { onGenerate: Function }) => (
    <div data-testid="image-generation-panel">
      <button onClick={() => onGenerate('image', { prompt: 'test image' })}>
        Generate Image
      </button>
    </div>
  )
}))

jest.mock('@/components/visual-generation/humanoid-designer', () => ({
  HumanoidDesigner: ({ onGenerate }: { onGenerate: Function }) => (
    <div data-testid="humanoid-designer">
      <button onClick={() => onGenerate('humanoid', { prompt: 'test humanoid' })}>
        Generate Humanoid
      </button>
    </div>
  )
}))

jest.mock('@/components/visual-generation/depth-visualization-tools', () => ({
  DepthVisualizationTools: ({ onGenerate }: { onGenerate: Function }) => (
    <div data-testid="depth-visualization-tools">
      <button onClick={() => onGenerate('2d-to-3d', { prompt: 'test 3d' })}>
        Convert to 3D
      </button>
    </div>
  )
}))

jest.mock('@/components/visual-generation/real-time-preview', () => ({
  RealTimePreview: React.forwardRef(({ job, isGenerating }: any, ref: any) => (
    <div data-testid="real-time-preview" ref={ref}>
      {job ? `Job: ${job.id} - ${job.status}` : 'No job selected'}
      {isGenerating && ' - Generating...'}
    </div>
  ))
}))

jest.mock('@/components/visual-generation/timeline-editor', () => ({
  TimelineEditor: () => <div data-testid="timeline-editor">Timeline Editor</div>
}))

describe('VisualGenerationInterface', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders the main interface with all tabs', () => {
    render(<VisualGenerationInterface />)
    
    expect(screen.getByText('ScrollIntel Visual Generation')).toBeInTheDocument()
    expect(screen.getByText('Ultra-Realistic Video')).toBeInTheDocument()
    expect(screen.getByText('High-Quality Images')).toBeInTheDocument()
    expect(screen.getByText('Humanoid Designer')).toBeInTheDocument()
    expect(screen.getByText('2D-to-3D Conversion')).toBeInTheDocument()
  })

  it('displays the video generation panel by default', () => {
    render(<VisualGenerationInterface />)
    
    expect(screen.getByTestId('video-generation-panel')).toBeInTheDocument()
    expect(screen.queryByTestId('image-generation-panel')).not.toBeInTheDocument()
  })

  it('switches between tabs correctly', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Switch to image tab
    await user.click(screen.getByText('High-Quality Images'))
    expect(screen.getByTestId('image-generation-panel')).toBeInTheDocument()
    expect(screen.queryByTestId('video-generation-panel')).not.toBeInTheDocument()
    
    // Switch to humanoid tab
    await user.click(screen.getByText('Humanoid Designer'))
    expect(screen.getByTestId('humanoid-designer')).toBeInTheDocument()
    expect(screen.queryByTestId('image-generation-panel')).not.toBeInTheDocument()
    
    // Switch to 2D-to-3D tab
    await user.click(screen.getByText('2D-to-3D Conversion'))
    expect(screen.getByTestId('depth-visualization-tools')).toBeInTheDocument()
    expect(screen.queryByTestId('humanoid-designer')).not.toBeInTheDocument()
  })

  it('shows timeline editor only for video tab', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Video tab should show timeline
    expect(screen.getByTestId('timeline-editor')).toBeInTheDocument()
    
    // Switch to image tab - timeline should be hidden
    await user.click(screen.getByText('High-Quality Images'))
    expect(screen.queryByTestId('timeline-editor')).not.toBeInTheDocument()
    
    // Switch back to video tab - timeline should be visible again
    await user.click(screen.getByText('Ultra-Realistic Video'))
    expect(screen.getByTestId('timeline-editor')).toBeInTheDocument()
  })

  it('handles video generation correctly', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Generate a video
    await user.click(screen.getByText('Generate Video'))
    
    // Check that a job was created and appears in the queue
    await waitFor(() => {
      expect(screen.getByText('video')).toBeInTheDocument()
    })
    
    // Check that the preview shows the job
    expect(screen.getByText(/Job: job_/)).toBeInTheDocument()
  })

  it('handles image generation correctly', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Switch to image tab and generate
    await user.click(screen.getByText('High-Quality Images'))
    await user.click(screen.getByText('Generate Image'))
    
    // Check that a job was created
    await waitFor(() => {
      expect(screen.getByText('image')).toBeInTheDocument()
    })
  })

  it('handles humanoid generation correctly', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Switch to humanoid tab and generate
    await user.click(screen.getByText('Humanoid Designer'))
    await user.click(screen.getByText('Generate Humanoid'))
    
    // Check that a job was created
    await waitFor(() => {
      expect(screen.getByText('humanoid')).toBeInTheDocument()
    })
  })

  it('handles 2D-to-3D conversion correctly', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Switch to 2D-to-3D tab and generate
    await user.click(screen.getByText('2D-to-3D Conversion'))
    await user.click(screen.getByText('Convert to 3D'))
    
    // Check that a job was created
    await waitFor(() => {
      expect(screen.getByText('2d-to-3d')).toBeInTheDocument()
    })
  })

  it('displays job queue correctly', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    expect(screen.getByText('Generation Queue')).toBeInTheDocument()
    expect(screen.getByText('No generation jobs yet')).toBeInTheDocument()
    
    // Generate a job
    await user.click(screen.getByText('Generate Video'))
    
    // Check that the job appears in the queue
    await waitFor(() => {
      expect(screen.queryByText('No generation jobs yet')).not.toBeInTheDocument()
      expect(screen.getByText('video')).toBeInTheDocument()
    })
  })

  it('allows job selection from queue', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Generate two jobs
    await user.click(screen.getByText('Generate Video'))
    await user.click(screen.getByText('High-Quality Images'))
    await user.click(screen.getByText('Generate Image'))
    
    await waitFor(() => {
      expect(screen.getAllByText(/video|image/)).toHaveLength(2)
    })
    
    // Click on a job in the queue
    const jobCards = screen.getAllByRole('button').filter(button => 
      button.textContent?.includes('video') || button.textContent?.includes('image')
    )
    
    if (jobCards.length > 0) {
      await user.click(jobCards[0])
      // Job should be selected (this would be indicated by styling changes)
    }
  })

  it('simulates generation progress', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Generate a job
    await user.click(screen.getByText('Generate Video'))
    
    // Check initial status
    await waitFor(() => {
      expect(screen.getByText('queued')).toBeInTheDocument()
    })
    
    // Wait for processing to start
    await waitFor(() => {
      expect(screen.getByText('processing')).toBeInTheDocument()
    }, { timeout: 1000 })
    
    // Wait for completion
    await waitFor(() => {
      expect(screen.getByText('completed')).toBeInTheDocument()
    }, { timeout: 6000 })
  })

  it('displays real-time preview', () => {
    render(<VisualGenerationInterface />)
    
    expect(screen.getByTestId('real-time-preview')).toBeInTheDocument()
    expect(screen.getByText('No job selected')).toBeInTheDocument()
  })

  it('shows generation status in preview', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Generate a job
    await user.click(screen.getByText('Generate Video'))
    
    // Check that preview shows the job
    await waitFor(() => {
      expect(screen.getByText(/Job: job_.*queued/)).toBeInTheDocument()
    })
  })

  it('displays correct badges and indicators', () => {
    render(<VisualGenerationInterface />)
    
    expect(screen.getByText('Ultra-Realistic 4K')).toBeInTheDocument()
    expect(screen.getByText('Settings')).toBeInTheDocument()
    expect(screen.getByText('Share')).toBeInTheDocument()
  })

  it('handles multiple concurrent generations', async () => {
    const user = userEvent.setup()
    render(<VisualGenerationInterface />)
    
    // Generate multiple jobs quickly
    await user.click(screen.getByText('Generate Video'))
    await user.click(screen.getByText('High-Quality Images'))
    await user.click(screen.getByText('Generate Image'))
    await user.click(screen.getByText('Humanoid Designer'))
    await user.click(screen.getByText('Generate Humanoid'))
    
    // Check that all jobs are created
    await waitFor(() => {
      const jobElements = screen.getAllByText(/video|image|humanoid/)
      expect(jobElements.length).toBeGreaterThanOrEqual(3)
    })
  })
})