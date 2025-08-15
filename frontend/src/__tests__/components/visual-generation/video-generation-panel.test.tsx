import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import { VideoGenerationPanel } from '@/components/visual-generation/video-generation-panel'

describe('VideoGenerationPanel', () => {
  const mockOnGenerate = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders the video generation panel with all elements', () => {
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    expect(screen.getByText('Ultra-Realistic Video Generation')).toBeInTheDocument()
    expect(screen.getByText('Generate photorealistic videos with breakthrough AI technology')).toBeInTheDocument()
    expect(screen.getByText('10x Faster Than Competitors')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Describe your ultra-realistic video scene in detail...')).toBeInTheDocument()
  })

  it('handles prompt input correctly', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    const promptInput = screen.getByPlaceholderText('Describe your ultra-realistic video scene in detail...')
    await user.type(promptInput, 'A beautiful sunset over mountains')
    
    expect(promptInput).toHaveValue('A beautiful sunset over mountains')
  })

  it('displays all tabs correctly', () => {
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    expect(screen.getByText('Basic Settings')).toBeInTheDocument()
    expect(screen.getByText('Advanced')).toBeInTheDocument()
    expect(screen.getByText('Professional')).toBeInTheDocument()
  })

  it('shows basic settings by default', () => {
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    expect(screen.getByText('Duration: 5 seconds')).toBeInTheDocument()
    expect(screen.getByText('Resolution')).toBeInTheDocument()
    expect(screen.getByText('Frame Rate')).toBeInTheDocument()
  })

  it('allows duration adjustment', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    const durationSlider = screen.getByRole('slider')
    fireEvent.change(durationSlider, { target: { value: '10' } })
    
    await waitFor(() => {
      expect(screen.getByText('Duration: 10 seconds')).toBeInTheDocument()
    })
  })

  it('allows resolution selection', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    const resolutionSelect = screen.getByDisplayValue('4K Ultra HD (3840x2160)')
    await user.click(resolutionSelect)
    
    expect(screen.getByText('1080p (1920x1080)')).toBeInTheDocument()
    expect(screen.getByText('8K (7680x4320)')).toBeInTheDocument()
  })

  it('allows frame rate selection', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    const fpsSelect = screen.getByDisplayValue('60 FPS (Ultra Smooth)')
    await user.click(fpsSelect)
    
    expect(screen.getByText('24 FPS (Cinematic)')).toBeInTheDocument()
    expect(screen.getByText('30 FPS (Standard)')).toBeInTheDocument()
    expect(screen.getByText('120 FPS (Professional)')).toBeInTheDocument()
  })

  it('switches to advanced tab and shows advanced settings', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    await user.click(screen.getByText('Advanced'))
    
    expect(screen.getByText('Visual Style')).toBeInTheDocument()
    expect(screen.getByText('Motion Intensity: 50%')).toBeInTheDocument()
    expect(screen.getByText('Camera Movement')).toBeInTheDocument()
  })

  it('allows style selection in advanced tab', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    await user.click(screen.getByText('Advanced'))
    
    const styleSelect = screen.getByDisplayValue('Photorealistic')
    await user.click(styleSelect)
    
    expect(screen.getByText('Cinematic')).toBeInTheDocument()
    expect(screen.getByText('Documentary')).toBeInTheDocument()
    expect(screen.getByText('Broadcast Quality')).toBeInTheDocument()
    expect(screen.getByText('Film Grade Production')).toBeInTheDocument()
  })

  it('allows motion intensity adjustment', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    await user.click(screen.getByText('Advanced'))
    
    const motionSlider = screen.getAllByRole('slider')[1] // Second slider is motion intensity
    fireEvent.change(motionSlider, { target: { value: '75' } })
    
    await waitFor(() => {
      expect(screen.getByText('Motion Intensity: 75%')).toBeInTheDocument()
    })
  })

  it('allows camera movement selection', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    await user.click(screen.getByText('Advanced'))
    
    const cameraSelect = screen.getByDisplayValue('Static')
    await user.click(cameraSelect)
    
    expect(screen.getByText('Pan')).toBeInTheDocument()
    expect(screen.getByText('Tilt')).toBeInTheDocument()
    expect(screen.getByText('Zoom')).toBeInTheDocument()
    expect(screen.getByText('Dolly')).toBeInTheDocument()
    expect(screen.getByText('Handheld')).toBeInTheDocument()
    expect(screen.getByText('Cinematic Movement')).toBeInTheDocument()
  })

  it('switches to professional tab and shows professional features', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    await user.click(screen.getByText('Professional'))
    
    expect(screen.getByText('Source Image (Optional)')).toBeInTheDocument()
    expect(screen.getByText('Upload reference image')).toBeInTheDocument()
    expect(screen.getByText('Temporal Consistency')).toBeInTheDocument()
    expect(screen.getByText('Neural Rendering')).toBeInTheDocument()
  })

  it('handles file upload in professional tab', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    await user.click(screen.getByText('Professional'))
    
    const fileInput = screen.getByLabelText('Upload reference image')
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
    
    await user.upload(fileInput, file)
    
    expect(screen.getByText('test.jpg')).toBeInTheDocument()
  })

  it('shows professional feature cards', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    await user.click(screen.getByText('Professional'))
    
    expect(screen.getByText('Zero artifacts guaranteed')).toBeInTheDocument()
    expect(screen.getByText('Proprietary algorithms')).toBeInTheDocument()
  })

  it('calls onGenerate with correct parameters when generate button is clicked', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    // Fill in some values
    const promptInput = screen.getByPlaceholderText('Describe your ultra-realistic video scene in detail...')
    await user.type(promptInput, 'Test video prompt')
    
    // Click generate button
    const generateButton = screen.getByText('Generate Ultra-Realistic Video')
    await user.click(generateButton)
    
    expect(mockOnGenerate).toHaveBeenCalledWith('video', {
      prompt: 'Test video prompt',
      duration: 5,
      resolution: '4k',
      fps: 60,
      style: 'photorealistic',
      motionIntensity: 50,
      cameraMovement: 'none',
      sourceImage: null
    })
  })

  it('includes uploaded file in generation parameters', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    await user.click(screen.getByText('Professional'))
    
    // Upload a file
    const fileInput = screen.getByLabelText('Upload reference image')
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
    await user.upload(fileInput, file)
    
    // Fill prompt and generate
    const promptInput = screen.getByPlaceholderText('Describe your ultra-realistic video scene in detail...')
    await user.type(promptInput, 'Test with image')
    
    const generateButton = screen.getByText('Generate Ultra-Realistic Video')
    await user.click(generateButton)
    
    expect(mockOnGenerate).toHaveBeenCalledWith('video', expect.objectContaining({
      prompt: 'Test with image',
      sourceImage: file
    }))
  })

  it('shows enhance prompt and style suggestions buttons', () => {
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    expect(screen.getByText('Enhance Prompt')).toBeInTheDocument()
    expect(screen.getByText('Style Suggestions')).toBeInTheDocument()
  })

  it('displays all duration range indicators', () => {
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    expect(screen.getByText('1s')).toBeInTheDocument()
    expect(screen.getByText('10 minutes (600s)')).toBeInTheDocument()
  })

  it('shows recommended and premium badges correctly', () => {
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    expect(screen.getByText('Recommended')).toBeInTheDocument()
    expect(screen.getByText('Premium')).toBeInTheDocument()
  })

  it('maintains state when switching between tabs', async () => {
    const user = userEvent.setup()
    render(<VideoGenerationPanel onGenerate={mockOnGenerate} />)
    
    // Set a prompt
    const promptInput = screen.getByPlaceholderText('Describe your ultra-realistic video scene in detail...')
    await user.type(promptInput, 'Test prompt')
    
    // Switch tabs
    await user.click(screen.getByText('Advanced'))
    await user.click(screen.getByText('Basic Settings'))
    
    // Prompt should still be there
    expect(promptInput).toHaveValue('Test prompt')
  })
})