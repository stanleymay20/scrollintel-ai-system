'use client'

import React, { forwardRef, useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Play, Pause, Square, Download, Maximize2, Volume2, VolumeX } from 'lucide-react'

interface GenerationJob {
  id: string
  type: 'image' | 'video' | 'humanoid' | '2d-to-3d'
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: number
  prompt: string
  result?: string
  metadata?: any
}

interface RealTimePreviewProps {
  job: GenerationJob | null
  isGenerating: boolean
}

export const RealTimePreview = forwardRef<HTMLDivElement, RealTimePreviewProps>(
  ({ job, isGenerating }, ref) => {
    const [isPlaying, setIsPlaying] = useState(false)
    const [isMuted, setIsMuted] = useState(false)
    const [currentFrame, setCurrentFrame] = useState(0)
    const [totalFrames, setTotalFrames] = useState(100)
    const [previewQuality, setPreviewQuality] = useState('4K')

    useEffect(() => {
      if (job?.type === 'video' && job.status === 'processing') {
        // Simulate frame-by-frame preview updates
        const interval = setInterval(() => {
          setCurrentFrame(prev => Math.min(prev + 1, totalFrames))
        }, 100)
        return () => clearInterval(interval)
      }
    }, [job, totalFrames])

    const getPreviewContent = () => {
      if (!job) {
        return (
          <div className="h-full flex items-center justify-center text-slate-400">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-slate-700 rounded-lg flex items-center justify-center">
                <Play className="w-8 h-8" />
              </div>
              <p>Select a generation job to preview</p>
            </div>
          </div>
        )
      }

      if (job.status === 'processing') {
        return (
          <div className="h-full flex flex-col">
            {/* Real-time Preview Area */}
            <div className="flex-1 bg-slate-900 rounded-lg relative overflow-hidden">
              {job.type === 'video' && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-32 h-32 mx-auto mb-4 bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg animate-pulse flex items-center justify-center">
                      <Play className="w-16 h-16 text-white" />
                    </div>
                    <p className="text-white font-medium">Generating 4K Video</p>
                    <p className="text-slate-400 text-sm">Frame {currentFrame} of {totalFrames}</p>
                    <Badge variant="outline" className="mt-2 text-green-400 border-green-400">
                      Real-time Preview
                    </Badge>
                  </div>
                </div>
              )}
              
              {job.type === 'image' && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-32 h-32 mx-auto mb-4 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg animate-pulse flex items-center justify-center">
                      <div className="w-16 h-16 bg-white/20 rounded"></div>
                    </div>
                    <p className="text-white font-medium">Generating High-Quality Image</p>
                    <p className="text-slate-400 text-sm">Resolution: 1024x1024</p>
                  </div>
                </div>
              )}

              {job.type === 'humanoid' && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-32 h-32 mx-auto mb-4 bg-gradient-to-br from-gold-600 to-amber-600 rounded-lg animate-pulse flex items-center justify-center">
                      <div className="w-12 h-16 bg-white/20 rounded-full"></div>
                    </div>
                    <p className="text-white font-medium">Creating Ultra-Realistic Human</p>
                    <p className="text-slate-400 text-sm">Biometric Accuracy: 99%</p>
                  </div>
                </div>
              )}

              {job.type === '2d-to-3d' && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-32 h-32 mx-auto mb-4 bg-gradient-to-br from-cyan-600 to-blue-600 rounded-lg animate-pulse flex items-center justify-center">
                      <div className="w-16 h-16 border-2 border-white/20 rounded transform rotate-45"></div>
                    </div>
                    <p className="text-white font-medium">Converting to 3D</p>
                    <p className="text-slate-400 text-sm">Sub-pixel precision depth mapping</p>
                  </div>
                </div>
              )}

              {/* Progress Overlay */}
              <div className="absolute bottom-4 left-4 right-4">
                <div className="bg-black/50 backdrop-blur-sm rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white text-sm font-medium">
                      {job.progress}% Complete
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {previewQuality}
                    </Badge>
                  </div>
                  <Progress value={job.progress} className="h-2" />
                </div>
              </div>
            </div>

            {/* Generation Stats */}
            <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
              <div className="bg-slate-700 rounded p-2">
                <div className="text-slate-400">Processing Time</div>
                <div className="text-white font-medium">
                  {Math.floor((Date.now() - parseInt(job.id.split('_')[1])) / 1000)}s
                </div>
              </div>
              <div className="bg-slate-700 rounded p-2">
                <div className="text-slate-400">Quality</div>
                <div className="text-white font-medium">Ultra HD</div>
              </div>
            </div>
          </div>
        )
      }

      if (job.status === 'completed') {
        return (
          <div className="h-full flex flex-col">
            {/* Completed Preview */}
            <div className="flex-1 bg-slate-900 rounded-lg relative overflow-hidden">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="w-32 h-32 mx-auto mb-4 bg-gradient-to-br from-green-600 to-emerald-600 rounded-lg flex items-center justify-center">
                    <Download className="w-16 h-16 text-white" />
                  </div>
                  <p className="text-white font-medium">Generation Complete!</p>
                  <p className="text-slate-400 text-sm">{job.type} ready for download</p>
                  <Badge variant="outline" className="mt-2 text-green-400 border-green-400">
                    100% Quality
                  </Badge>
                </div>
              </div>

              {/* Video Controls (if video) */}
              {job.type === 'video' && (
                <div className="absolute bottom-4 left-4 right-4">
                  <div className="bg-black/50 backdrop-blur-sm rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setIsPlaying(!isPlaying)}
                        >
                          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setIsMuted(!isMuted)}
                        >
                          {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                        </Button>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button size="sm" variant="ghost">
                          <Maximize2 className="w-4 h-4" />
                        </Button>
                        <Button size="sm" variant="ghost">
                          <Download className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Completion Stats */}
            <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
              <div className="bg-slate-700 rounded p-2">
                <div className="text-slate-400">Total Time</div>
                <div className="text-white font-medium">45s</div>
              </div>
              <div className="bg-slate-700 rounded p-2">
                <div className="text-slate-400">Quality Score</div>
                <div className="text-white font-medium">98%</div>
              </div>
              <div className="bg-slate-700 rounded p-2">
                <div className="text-slate-400">File Size</div>
                <div className="text-white font-medium">2.4 GB</div>
              </div>
            </div>
          </div>
        )
      }

      if (job.status === 'failed') {
        return (
          <div className="h-full flex items-center justify-center text-slate-400">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-red-600 rounded-lg flex items-center justify-center">
                <Square className="w-8 h-8 text-white" />
              </div>
              <p className="text-red-400 font-medium">Generation Failed</p>
              <p className="text-slate-400 text-sm">Please try again with different settings</p>
            </div>
          </div>
        )
      }

      return (
        <div className="h-full flex items-center justify-center text-slate-400">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 bg-slate-700 rounded-lg flex items-center justify-center">
              <Play className="w-8 h-8" />
            </div>
            <p>Queued for processing...</p>
          </div>
        </div>
      )
    }

    return (
      <Card className="h-full bg-slate-800 border-slate-700" ref={ref}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white text-lg">Real-Time Preview</CardTitle>
            {job && (
              <Badge 
                variant="outline" 
                className={`text-xs ${
                  job.status === 'completed' ? 'text-green-400 border-green-400' :
                  job.status === 'processing' ? 'text-blue-400 border-blue-400' :
                  job.status === 'failed' ? 'text-red-400 border-red-400' :
                  'text-gray-400 border-gray-400'
                }`}
              >
                {job.status}
              </Badge>
            )}
          </div>
          {job && (
            <p className="text-slate-400 text-sm truncate">
              {job.prompt || `${job.type} generation`}
            </p>
          )}
        </CardHeader>
        <CardContent className="flex-1 p-4">
          {getPreviewContent()}
        </CardContent>
      </Card>
    )
  }
)