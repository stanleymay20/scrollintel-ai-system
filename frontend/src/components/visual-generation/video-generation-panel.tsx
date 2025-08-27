'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Upload, Wand2, Zap, Crown, Sparkles } from 'lucide-react'

interface VideoGenerationPanelProps {
  onGenerate: (type: string, params: any) => void
}

export function VideoGenerationPanel({ onGenerate }: VideoGenerationPanelProps) {
  const [prompt, setPrompt] = useState('')
  const [duration, setDuration] = useState([5])
  const [resolution, setResolution] = useState('4k')
  const [fps, setFps] = useState('60')
  const [style, setStyle] = useState('photorealistic')
  const [motionIntensity, setMotionIntensity] = useState([50])
  const [cameraMovement, setCameraMovement] = useState('none')
  const [sourceImage, setSourceImage] = useState<File | null>(null)

  const handleGenerate = () => {
    // Convert resolution string to array
    let resolutionArray: [number, number]
    switch (resolution) {
      case '4k':
        resolutionArray = [3840, 2160]
        break
      case '1080p':
        resolutionArray = [1920, 1080]
        break
      case '720p':
        resolutionArray = [1280, 720]
        break
      default:
        resolutionArray = [1920, 1080]
    }

    onGenerate('video', {
      prompt,
      duration: duration[0],
      resolution: resolutionArray,
      fps: parseInt(fps),
      style,
      quality: 'ultra_high',
      humanoid_generation: style === 'humanoid' || prompt.toLowerCase().includes('person') || prompt.toLowerCase().includes('human'),
      physics_simulation: true,
      neural_rendering_quality: 'photorealistic_plus',
      temporal_consistency_level: 'ultra_high'
    })
  }

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSourceImage(file)
    }
  }

  return (
    <div className="h-full flex flex-col space-y-6">
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-white flex items-center">
                <Crown className="w-5 h-5 mr-2 text-yellow-500" />
                Ultra-Realistic Video Generation
              </CardTitle>
              <CardDescription className="text-slate-400">
                Generate photorealistic videos with breakthrough AI technology
              </CardDescription>
            </div>
            <Badge variant="outline" className="text-purple-400 border-purple-400">
              10x Faster Than Competitors
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Prompt Input */}
          <div className="space-y-2">
            <Label htmlFor="prompt" className="text-white">
              Prompt
              <Badge variant="outline" className="ml-2 text-xs">
                AI-Enhanced
              </Badge>
            </Label>
            <Textarea
              id="prompt"
              placeholder="Describe your ultra-realistic video scene in detail..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="min-h-[100px] bg-slate-700 border-slate-600 text-white placeholder:text-slate-400"
            />
            <div className="flex space-x-2">
              <Button variant="outline" size="sm">
                <Wand2 className="w-4 h-4 mr-2" />
                Enhance Prompt
              </Button>
              <Button variant="outline" size="sm">
                <Sparkles className="w-4 h-4 mr-2" />
                Style Suggestions
              </Button>
            </div>
          </div>

          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-3 bg-slate-700">
              <TabsTrigger value="basic">Basic Settings</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
              <TabsTrigger value="professional">Professional</TabsTrigger>
            </TabsList>

            <TabsContent value="basic" className="space-y-4">
              {/* Duration */}
              <div className="space-y-2">
                <Label className="text-white">Duration: {duration[0]} seconds</Label>
                <Slider
                  value={duration}
                  onValueChange={setDuration}
                  max={600}
                  min={1}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-400">
                  <span>1s</span>
                  <span>10 minutes (600s)</span>
                </div>
              </div>

              {/* Resolution */}
              <div className="space-y-2">
                <Label className="text-white">Resolution</Label>
                <Select value={resolution} onValueChange={setResolution}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="1080p">1080p (1920x1080)</SelectItem>
                    <SelectItem value="4k">
                      4K Ultra HD (3840x2160) 
                      <Badge variant="outline" className="ml-2 text-xs">Recommended</Badge>
                    </SelectItem>
                    <SelectItem value="8k">8K (7680x4320)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Frame Rate */}
              <div className="space-y-2">
                <Label className="text-white">Frame Rate</Label>
                <Select value={fps} onValueChange={setFps}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="24">24 FPS (Cinematic)</SelectItem>
                    <SelectItem value="30">30 FPS (Standard)</SelectItem>
                    <SelectItem value="60">
                      60 FPS (Ultra Smooth)
                      <Badge variant="outline" className="ml-2 text-xs">Premium</Badge>
                    </SelectItem>
                    <SelectItem value="120">120 FPS (Professional)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </TabsContent>

            <TabsContent value="advanced" className="space-y-4">
              {/* Style */}
              <div className="space-y-2">
                <Label className="text-white">Visual Style</Label>
                <Select value={style} onValueChange={setStyle}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="photorealistic">
                      Photorealistic
                      <Badge variant="outline" className="ml-2 text-xs">Best Quality</Badge>
                    </SelectItem>
                    <SelectItem value="cinematic">Cinematic</SelectItem>
                    <SelectItem value="documentary">Documentary</SelectItem>
                    <SelectItem value="broadcast">Broadcast Quality</SelectItem>
                    <SelectItem value="film-grade">Film Grade Production</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Motion Intensity */}
              <div className="space-y-2">
                <Label className="text-white">Motion Intensity: {motionIntensity[0]}%</Label>
                <Slider
                  value={motionIntensity}
                  onValueChange={setMotionIntensity}
                  max={100}
                  min={0}
                  step={5}
                  className="w-full"
                />
              </div>

              {/* Camera Movement */}
              <div className="space-y-2">
                <Label className="text-white">Camera Movement</Label>
                <Select value={cameraMovement} onValueChange={setCameraMovement}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="none">Static</SelectItem>
                    <SelectItem value="pan">Pan</SelectItem>
                    <SelectItem value="tilt">Tilt</SelectItem>
                    <SelectItem value="zoom">Zoom</SelectItem>
                    <SelectItem value="dolly">Dolly</SelectItem>
                    <SelectItem value="handheld">Handheld</SelectItem>
                    <SelectItem value="cinematic">Cinematic Movement</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </TabsContent>

            <TabsContent value="professional" className="space-y-4">
              {/* Source Image Upload */}
              <div className="space-y-2">
                <Label className="text-white">Source Image (Optional)</Label>
                <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                    id="source-image"
                  />
                  <label htmlFor="source-image" className="cursor-pointer">
                    <Upload className="w-8 h-8 mx-auto mb-2 text-slate-400" />
                    <p className="text-slate-400">
                      {sourceImage ? sourceImage.name : 'Upload reference image'}
                    </p>
                  </label>
                </div>
              </div>

              {/* Professional Features */}
              <div className="grid grid-cols-2 gap-4">
                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Zap className="w-4 h-4 text-yellow-500" />
                      <span className="text-white text-sm">Temporal Consistency</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">Zero artifacts guaranteed</p>
                  </CardContent>
                </Card>
                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Crown className="w-4 h-4 text-purple-500" />
                      <span className="text-white text-sm">Neural Rendering</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">Proprietary algorithms</p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>

          {/* Generate Button */}
          <Button 
            onClick={handleGenerate}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
            size="lg"
          >
            <Zap className="w-5 h-5 mr-2" />
            Generate Ultra-Realistic Video
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}