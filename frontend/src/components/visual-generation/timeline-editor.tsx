'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { 
  Play, 
  Pause, 
  Square, 
  SkipBack, 
  SkipForward, 
  Scissors, 
  Copy, 
  Trash2,
  Volume2,
  Eye,
  Lock,
  Unlock
} from 'lucide-react'

interface TimelineClip {
  id: string
  type: 'video' | 'audio' | 'image' | 'text'
  name: string
  startTime: number
  duration: number
  track: number
  color: string
  locked: boolean
  visible: boolean
}

export function TimelineEditor() {
  const [clips, setClips] = useState<TimelineClip[]>([
    {
      id: '1',
      type: 'video',
      name: 'Main Video',
      startTime: 0,
      duration: 10,
      track: 0,
      color: 'bg-blue-600',
      locked: false,
      visible: true
    },
    {
      id: '2',
      type: 'audio',
      name: 'Background Music',
      startTime: 0,
      duration: 15,
      track: 1,
      color: 'bg-green-600',
      locked: false,
      visible: true
    },
    {
      id: '3',
      type: 'image',
      name: 'Overlay Image',
      startTime: 5,
      duration: 3,
      track: 2,
      color: 'bg-purple-600',
      locked: false,
      visible: true
    }
  ])
  
  const [currentTime, setCurrentTime] = useState(0)
  const [totalDuration, setTotalDuration] = useState(20)
  const [isPlaying, setIsPlaying] = useState(false)
  const [selectedClip, setSelectedClip] = useState<string | null>(null)
  const [zoom, setZoom] = useState([1])
  const timelineRef = useRef<HTMLDivElement>(null)

  const pixelsPerSecond = 40 * zoom[0]

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentTime(prev => {
          if (prev >= totalDuration) {
            setIsPlaying(false)
            return 0
          }
          return prev + 0.1
        })
      }, 100)
    }
    return () => clearInterval(interval)
  }, [isPlaying, totalDuration])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const frames = Math.floor((seconds % 1) * 30)
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`
  }

  const handleClipClick = (clipId: string) => {
    setSelectedClip(clipId === selectedClip ? null : clipId)
  }

  const handleClipDrag = (clipId: string, newStartTime: number) => {
    setClips(prev => prev.map(clip => 
      clip.id === clipId 
        ? { ...clip, startTime: Math.max(0, newStartTime) }
        : clip
    ))
  }

  const toggleClipLock = (clipId: string) => {
    setClips(prev => prev.map(clip => 
      clip.id === clipId 
        ? { ...clip, locked: !clip.locked }
        : clip
    ))
  }

  const toggleClipVisibility = (clipId: string) => {
    setClips(prev => prev.map(clip => 
      clip.id === clipId 
        ? { ...clip, visible: !clip.visible }
        : clip
    ))
  }

  const duplicateClip = (clipId: string) => {
    const clipToDuplicate = clips.find(c => c.id === clipId)
    if (clipToDuplicate) {
      const newClip: TimelineClip = {
        ...clipToDuplicate,
        id: Date.now().toString(),
        name: `${clipToDuplicate.name} Copy`,
        startTime: clipToDuplicate.startTime + clipToDuplicate.duration
      }
      setClips(prev => [...prev, newClip])
    }
  }

  const deleteClip = (clipId: string) => {
    setClips(prev => prev.filter(c => c.id !== clipId))
    if (selectedClip === clipId) {
      setSelectedClip(null)
    }
  }

  const getTrackName = (trackIndex: number) => {
    switch (trackIndex) {
      case 0: return 'Video Track'
      case 1: return 'Audio Track'
      case 2: return 'Overlay Track'
      default: return `Track ${trackIndex + 1}`
    }
  }

  return (
    <div className="h-full bg-slate-800 border-t border-slate-700 flex flex-col">
      {/* Timeline Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <div className="flex items-center space-x-4">
          <h3 className="text-white font-semibold">Professional Timeline Editor</h3>
          <Badge variant="outline" className="text-purple-400 border-purple-400">
            Frame-by-Frame Precision
          </Badge>
        </div>
        
        {/* Playback Controls */}
        <div className="flex items-center space-x-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setCurrentTime(0)}
          >
            <SkipBack className="w-4 h-4" />
          </Button>
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
            onClick={() => setIsPlaying(false)}
          >
            <Square className="w-4 h-4" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setCurrentTime(totalDuration)}
          >
            <SkipForward className="w-4 h-4" />
          </Button>
          
          <div className="text-white text-sm font-mono ml-4">
            {formatTime(currentTime)} / {formatTime(totalDuration)}
          </div>
        </div>

        {/* Timeline Tools */}
        <div className="flex items-center space-x-2">
          <Button size="sm" variant="ghost" disabled={!selectedClip}>
            <Scissors className="w-4 h-4" />
          </Button>
          <Button 
            size="sm" 
            variant="ghost" 
            disabled={!selectedClip}
            onClick={() => selectedClip && duplicateClip(selectedClip)}
          >
            <Copy className="w-4 h-4" />
          </Button>
          <Button 
            size="sm" 
            variant="ghost" 
            disabled={!selectedClip}
            onClick={() => selectedClip && deleteClip(selectedClip)}
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Timeline Content */}
      <div className="flex-1 flex">
        {/* Track Headers */}
        <div className="w-48 bg-slate-900 border-r border-slate-700">
          <div className="h-12 border-b border-slate-700 flex items-center px-4">
            <span className="text-white text-sm font-medium">Tracks</span>
          </div>
          {[0, 1, 2].map(trackIndex => (
            <div key={trackIndex} className="h-16 border-b border-slate-700 flex items-center px-4">
              <div className="flex items-center justify-between w-full">
                <span className="text-white text-sm">{getTrackName(trackIndex)}</span>
                <div className="flex items-center space-x-1">
                  <Button size="sm" variant="ghost" className="w-6 h-6 p-0">
                    <Volume2 className="w-3 h-3" />
                  </Button>
                  <Button size="sm" variant="ghost" className="w-6 h-6 p-0">
                    <Eye className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Timeline Area */}
        <div className="flex-1 overflow-x-auto">
          <div className="relative" style={{ width: `${totalDuration * pixelsPerSecond}px` }}>
            {/* Time Ruler */}
            <div className="h-12 border-b border-slate-700 bg-slate-900 relative">
              {Array.from({ length: Math.ceil(totalDuration) + 1 }, (_, i) => (
                <div
                  key={i}
                  className="absolute top-0 h-full border-l border-slate-600"
                  style={{ left: `${i * pixelsPerSecond}px` }}
                >
                  <span className="text-xs text-slate-400 ml-1 mt-1 block">
                    {formatTime(i)}
                  </span>
                </div>
              ))}
            </div>

            {/* Playhead */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-20 pointer-events-none"
              style={{ left: `${currentTime * pixelsPerSecond}px` }}
            >
              <div className="w-3 h-3 bg-red-500 -ml-1.5 -mt-1.5 rounded-full"></div>
            </div>

            {/* Tracks */}
            {[0, 1, 2].map(trackIndex => (
              <div key={trackIndex} className="h-16 border-b border-slate-700 relative">
                {clips
                  .filter(clip => clip.track === trackIndex)
                  .map(clip => (
                    <div
                      key={clip.id}
                      className={`absolute top-2 bottom-2 rounded cursor-pointer transition-all ${
                        clip.color
                      } ${
                        selectedClip === clip.id ? 'ring-2 ring-white' : ''
                      } ${
                        clip.locked ? 'opacity-75' : ''
                      } ${
                        !clip.visible ? 'opacity-50' : ''
                      }`}
                      style={{
                        left: `${clip.startTime * pixelsPerSecond}px`,
                        width: `${clip.duration * pixelsPerSecond}px`
                      }}
                      onClick={() => handleClipClick(clip.id)}
                    >
                      <div className="h-full flex items-center justify-between px-2">
                        <div className="flex items-center space-x-1">
                          {clip.locked && <Lock className="w-3 h-3 text-white" />}
                          {!clip.visible && <Eye className="w-3 h-3 text-white opacity-50" />}
                          <span className="text-white text-xs font-medium truncate">
                            {clip.name}
                          </span>
                        </div>
                        
                        {selectedClip === clip.id && (
                          <div className="flex items-center space-x-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="w-4 h-4 p-0"
                              onClick={(e) => {
                                e.stopPropagation()
                                toggleClipLock(clip.id)
                              }}
                            >
                              {clip.locked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="w-4 h-4 p-0"
                              onClick={(e) => {
                                e.stopPropagation()
                                toggleClipVisibility(clip.id)
                              }}
                            >
                              <Eye className={`w-3 h-3 ${clip.visible ? '' : 'opacity-50'}`} />
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Timeline Footer */}
      <div className="h-12 border-t border-slate-700 flex items-center justify-between px-4">
        <div className="flex items-center space-x-4">
          <span className="text-slate-400 text-sm">Zoom:</span>
          <div className="w-32">
            <Slider
              value={zoom}
              onValueChange={setZoom}
              max={3}
              min={0.5}
              step={0.1}
              className="w-full"
            />
          </div>
          <span className="text-slate-400 text-sm">{Math.round(zoom[0] * 100)}%</span>
        </div>

        {selectedClip && (
          <div className="flex items-center space-x-4">
            <span className="text-slate-400 text-sm">
              Selected: {clips.find(c => c.id === selectedClip)?.name}
            </span>
            <Badge variant="outline" className="text-xs">
              {clips.find(c => c.id === selectedClip)?.duration}s
            </Badge>
          </div>
        )}

        <div className="text-slate-400 text-sm">
          {clips.length} clips • {totalDuration}s total
        </div>
      </div>
    </div>
  )
}