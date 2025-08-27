'use client'

import React, { useState, useRef, useCallback, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { VideoGenerationPanel } from './video-generation-panel'
import { ImageGenerationPanel } from './image-generation-panel'
import { HumanoidDesigner } from './humanoid-designer'
import { TimelineEditor } from './timeline-editor'
import { DepthVisualizationTools } from './depth-visualization-tools'
import { RealTimePreview } from './real-time-preview'
import { useVisualGeneration, GenerationJob } from '@/hooks/useVisualGeneration'
import { Play, Pause, Square, Download, Share2, Settings, AlertCircle, X } from 'lucide-react'

export function VisualGenerationInterface() {
  const [activeTab, setActiveTab] = useState('video')
  const [selectedJob, setSelectedJob] = useState<GenerationJob | null>(null)
  const previewRef = useRef<HTMLDivElement>(null)

  const {
    jobs,
    isGenerating,
    error,
    generateImage,
    generateVideo,
    enhanceImage,
    cancelGeneration,
    onProgress,
    loadUserGenerations,
    clearError
  } = useVisualGeneration()

  // Load user's generation history on mount
  useEffect(() => {
    loadUserGenerations({ limit: 20 })
  }, [loadUserGenerations])

  const handleGenerate = useCallback(async (type: string, params: any) => {
    try {
      let result
      
      switch (type) {
        case 'image':
          result = await generateImage(params)
          break
        case 'video':
          result = await generateVideo(params)
          break
        case 'humanoid':
          // Humanoid generation uses video generation with special parameters
          result = await generateVideo({
            ...params,
            humanoid_generation: true,
            neural_rendering_quality: 'photorealistic_plus',
            temporal_consistency_level: 'ultra_high'
          })
          break
        case '2d-to-3d':
          // 2D-to-3D conversion (enhancement)
          if (params.file) {
            result = await enhanceImage(params.file, '2d_to_3d')
          }
          break
        default:
          throw new Error(`Unknown generation type: ${type}`)
      }

      // Set the newly created job as selected
      if (result?.result_id) {
        const newJob = jobs.find(job => job.id === result.result_id)
        if (newJob) {
          setSelectedJob(newJob)
        }
      }
    } catch (error: any) {
      console.error('Generation failed:', error)
      // Error is handled by the hook
    }
  }, [generateImage, generateVideo, enhanceImage, jobs])

  const handleCancelGeneration = useCallback(async (jobId: string) => {
    try {
      await cancelGeneration(jobId)
    } catch (error: any) {
      console.error('Cancellation failed:', error)
    }
  }, [cancelGeneration])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500'
      case 'processing': return 'bg-blue-500'
      case 'failed': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <div className="flex items-center space-x-4">
          <h1 className="text-2xl font-bold text-white">ScrollIntel Visual Generation</h1>
          <Badge variant="outline" className="text-purple-400 border-purple-400">
            Ultra-Realistic 4K
          </Badge>
          {isGenerating && (
            <Badge variant="outline" className="text-blue-400 border-blue-400 animate-pulse">
              Generating...
            </Badge>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </Button>
          <Button variant="outline" size="sm">
            <Share2 className="w-4 h-4 mr-2" />
            Share
          </Button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert className="mx-4 mt-4 border-red-500 bg-red-500/10">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="flex items-center justify-between">
            <span>{error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearError}
              className="h-auto p-1 text-red-400 hover:text-red-300"
            >
              <X className="h-4 w-4" />
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <div className="flex-1 flex">
        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Generation Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
            <TabsList className="grid w-full grid-cols-4 bg-slate-800 border-slate-700">
              <TabsTrigger value="video" className="data-[state=active]:bg-purple-600">
                Ultra-Realistic Video
              </TabsTrigger>
              <TabsTrigger value="image" className="data-[state=active]:bg-purple-600">
                High-Quality Images
              </TabsTrigger>
              <TabsTrigger value="humanoid" className="data-[state=active]:bg-purple-600">
                Humanoid Designer
              </TabsTrigger>
              <TabsTrigger value="2d-to-3d" className="data-[state=active]:bg-purple-600">
                2D-to-3D Conversion
              </TabsTrigger>
            </TabsList>

            <div className="flex-1 flex">
              {/* Generation Panels */}
              <div className="flex-1 p-4">
                <TabsContent value="video" className="h-full">
                  <VideoGenerationPanel onGenerate={handleGenerate} />
                </TabsContent>
                <TabsContent value="image" className="h-full">
                  <ImageGenerationPanel onGenerate={handleGenerate} />
                </TabsContent>
                <TabsContent value="humanoid" className="h-full">
                  <HumanoidDesigner onGenerate={handleGenerate} />
                </TabsContent>
                <TabsContent value="2d-to-3d" className="h-full">
                  <DepthVisualizationTools onGenerate={handleGenerate} />
                </TabsContent>
              </div>

              {/* Real-time Preview */}
              <div className="w-1/3 p-4 border-l border-slate-700">
                <RealTimePreview 
                  job={selectedJob}
                  isGenerating={isGenerating}
                  ref={previewRef}
                />
              </div>
            </div>
          </Tabs>

          {/* Timeline Editor */}
          {activeTab === 'video' && (
            <div className="h-48 border-t border-slate-700">
              <TimelineEditor />
            </div>
          )}
        </div>

        {/* Job Queue Sidebar */}
        <div className="w-80 border-l border-slate-700 bg-slate-800/50">
          <div className="p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Generation Queue</h3>
            <div className="space-y-3">
              {jobs.map((job) => (
                <Card 
                  key={job.id} 
                  className={`cursor-pointer transition-all ${
                    selectedJob?.id === job.id ? 'ring-2 ring-purple-500' : ''
                  } bg-slate-700 border-slate-600`}
                  onClick={() => setSelectedJob(job)}
                >
                  <CardContent className="p-3">
                    <div className="flex items-center justify-between mb-2">
                      <Badge variant="outline" className="text-xs">
                        {job.type}
                      </Badge>
                      <div className={`w-2 h-2 rounded-full ${getStatusColor(job.status)}`} />
                    </div>
                    <p className="text-sm text-slate-300 truncate mb-2">
                      {job.prompt || 'No prompt'}
                    </p>
                    {job.status === 'processing' && (
                      <Progress value={job.progress} className="h-1" />
                    )}
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs text-slate-400 capitalize">
                        {job.status}
                      </span>
                      {job.status === 'completed' && job.result?.content_urls && (
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          className="h-6 px-2"
                          onClick={() => window.open(job.result!.content_urls[0], '_blank')}
                        >
                          <Download className="w-3 h-3" />
                        </Button>
                      )}
                      {(job.status === 'processing' || job.status === 'queued') && (
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          className="h-6 px-2 text-red-400 hover:text-red-300"
                          onClick={() => handleCancelGeneration(job.id)}
                        >
                          <X className="w-3 h-3" />
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
              {jobs.length === 0 && (
                <div className="text-center text-slate-400 py-8">
                  No generation jobs yet
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}