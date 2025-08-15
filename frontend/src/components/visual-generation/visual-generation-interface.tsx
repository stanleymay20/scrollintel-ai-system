'use client'

import React, { useState, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { VideoGenerationPanel } from './video-generation-panel'
import { ImageGenerationPanel } from './image-generation-panel'
import { HumanoidDesigner } from './humanoid-designer'
import { TimelineEditor } from './timeline-editor'
import { DepthVisualizationTools } from './depth-visualization-tools'
import { RealTimePreview } from './real-time-preview'
import { Play, Pause, Square, Download, Share2, Settings } from 'lucide-react'

interface GenerationJob {
  id: string
  type: 'image' | 'video' | 'humanoid' | '2d-to-3d'
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: number
  prompt: string
  result?: string
  metadata?: any
}

export function VisualGenerationInterface() {
  const [activeTab, setActiveTab] = useState('video')
  const [jobs, setJobs] = useState<GenerationJob[]>([])
  const [selectedJob, setSelectedJob] = useState<GenerationJob | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const previewRef = useRef<HTMLDivElement>(null)

  const handleGenerate = useCallback(async (type: string, params: any) => {
    const newJob: GenerationJob = {
      id: `job_${Date.now()}`,
      type: type as any,
      status: 'queued',
      progress: 0,
      prompt: params.prompt || '',
    }

    setJobs(prev => [...prev, newJob])
    setSelectedJob(newJob)
    setIsGenerating(true)

    try {
      // Simulate generation process
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 500))
        setJobs(prev => prev.map(job => 
          job.id === newJob.id 
            ? { ...job, progress: i, status: i === 100 ? 'completed' : 'processing' }
            : job
        ))
      }
    } catch (error) {
      setJobs(prev => prev.map(job => 
        job.id === newJob.id 
          ? { ...job, status: 'failed' }
          : job
      ))
    } finally {
      setIsGenerating(false)
    }
  }, [])

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
                      {job.status === 'completed' && (
                        <Button size="sm" variant="ghost" className="h-6 px-2">
                          <Download className="w-3 h-3" />
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