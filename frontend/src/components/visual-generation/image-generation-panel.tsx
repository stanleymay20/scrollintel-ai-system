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
import { Image, Palette, Sparkles, Wand2, Grid3X3 } from 'lucide-react'

interface ImageGenerationPanelProps {
  onGenerate: (type: string, params: any) => void
}

export function ImageGenerationPanel({ onGenerate }: ImageGenerationPanelProps) {
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [style, setStyle] = useState('photorealistic')
  const [resolution, setResolution] = useState('1024x1024')
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [numImages, setNumImages] = useState([1])
  const [guidanceScale, setGuidanceScale] = useState([7.5])
  const [steps, setSteps] = useState([50])
  const [seed, setSeed] = useState('')

  const handleGenerate = () => {
    const [width, height] = resolution.split('x').map(Number)
    
    onGenerate('image', {
      prompt,
      negative_prompt: negativePrompt || undefined,
      style,
      resolution: [width, height],
      num_images: numImages[0],
      quality: 'high',
      seed: seed ? parseInt(seed) : undefined,
      guidance_scale: guidanceScale[0],
      steps: steps[0]
    })
  }

  return (
    <div className="h-full flex flex-col space-y-6">
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-white flex items-center">
                <Image className="w-5 h-5 mr-2 text-blue-500" />
                High-Quality Image Generation
              </CardTitle>
              <CardDescription className="text-slate-400">
                Create stunning images with multiple AI models
              </CardDescription>
            </div>
            <Badge variant="outline" className="text-blue-400 border-blue-400">
              Multi-Model Ensemble
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
              placeholder="Describe the image you want to generate..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="min-h-[80px] bg-slate-700 border-slate-600 text-white placeholder:text-slate-400"
            />
          </div>

          {/* Negative Prompt */}
          <div className="space-y-2">
            <Label htmlFor="negative-prompt" className="text-white">
              Negative Prompt (Optional)
            </Label>
            <Textarea
              id="negative-prompt"
              placeholder="What you don't want in the image..."
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              className="min-h-[60px] bg-slate-700 border-slate-600 text-white placeholder:text-slate-400"
            />
          </div>

          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-3 bg-slate-700">
              <TabsTrigger value="basic">Basic Settings</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
              <TabsTrigger value="models">Model Selection</TabsTrigger>
            </TabsList>

            <TabsContent value="basic" className="space-y-4">
              {/* Style */}
              <div className="space-y-2">
                <Label className="text-white">Style</Label>
                <Select value={style} onValueChange={setStyle}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="photorealistic">Photorealistic</SelectItem>
                    <SelectItem value="artistic">Artistic</SelectItem>
                    <SelectItem value="cartoon">Cartoon</SelectItem>
                    <SelectItem value="sketch">Sketch</SelectItem>
                    <SelectItem value="abstract">Abstract</SelectItem>
                    <SelectItem value="oil-painting">Oil Painting</SelectItem>
                    <SelectItem value="watercolor">Watercolor</SelectItem>
                    <SelectItem value="digital-art">Digital Art</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Resolution */}
              <div className="space-y-2">
                <Label className="text-white">Resolution</Label>
                <Select value={resolution} onValueChange={setResolution}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="512x512">512x512 (Fast)</SelectItem>
                    <SelectItem value="1024x1024">
                      1024x1024 (Standard)
                      <Badge variant="outline" className="ml-2 text-xs">Recommended</Badge>
                    </SelectItem>
                    <SelectItem value="2048x2048">2048x2048 (High Quality)</SelectItem>
                    <SelectItem value="4096x4096">4096x4096 (Ultra HD)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Aspect Ratio */}
              <div className="space-y-2">
                <Label className="text-white">Aspect Ratio</Label>
                <div className="grid grid-cols-4 gap-2">
                  {['1:1', '16:9', '9:16', '4:3', '3:4', '21:9', '3:2', '2:3'].map((ratio) => (
                    <Button
                      key={ratio}
                      variant={aspectRatio === ratio ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setAspectRatio(ratio)}
                      className="text-xs"
                    >
                      {ratio}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Number of Images */}
              <div className="space-y-2">
                <Label className="text-white">Number of Images: {numImages[0]}</Label>
                <Slider
                  value={numImages}
                  onValueChange={setNumImages}
                  max={4}
                  min={1}
                  step={1}
                  className="w-full"
                />
              </div>
            </TabsContent>

            <TabsContent value="advanced" className="space-y-4">
              {/* Guidance Scale */}
              <div className="space-y-2">
                <Label className="text-white">Guidance Scale: {guidanceScale[0]}</Label>
                <Slider
                  value={guidanceScale}
                  onValueChange={setGuidanceScale}
                  max={20}
                  min={1}
                  step={0.5}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Creative</span>
                  <span>Precise</span>
                </div>
              </div>

              {/* Steps */}
              <div className="space-y-2">
                <Label className="text-white">Inference Steps: {steps[0]}</Label>
                <Slider
                  value={steps}
                  onValueChange={setSteps}
                  max={150}
                  min={10}
                  step={5}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Fast</span>
                  <span>High Quality</span>
                </div>
              </div>

              {/* Seed */}
              <div className="space-y-2">
                <Label className="text-white">Seed (Optional)</Label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    placeholder="Random seed for reproducibility"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                    className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder:text-slate-400"
                  />
                  <Button
                    variant="outline"
                    onClick={() => setSeed(Math.floor(Math.random() * 1000000).toString())}
                  >
                    Random
                  </Button>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="models" className="space-y-4">
              <div className="grid grid-cols-1 gap-4">
                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Palette className="w-4 h-4 text-purple-500" />
                        <span className="text-white font-medium">Stable Diffusion XL</span>
                      </div>
                      <Badge variant="outline" className="text-green-400 border-green-400">
                        Available
                      </Badge>
                    </div>
                    <p className="text-xs text-slate-400">
                      High-quality image generation with excellent prompt adherence
                    </p>
                  </CardContent>
                </Card>

                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Sparkles className="w-4 h-4 text-blue-500" />
                        <span className="text-white font-medium">DALL-E 3</span>
                      </div>
                      <Badge variant="outline" className="text-green-400 border-green-400">
                        Available
                      </Badge>
                    </div>
                    <p className="text-xs text-slate-400">
                      Advanced understanding of complex prompts and natural language
                    </p>
                  </CardContent>
                </Card>

                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Grid3X3 className="w-4 h-4 text-orange-500" />
                        <span className="text-white font-medium">Midjourney</span>
                      </div>
                      <Badge variant="outline" className="text-green-400 border-green-400">
                        Available
                      </Badge>
                    </div>
                    <p className="text-xs text-slate-400">
                      Artistic and creative image generation with unique aesthetics
                    </p>
                  </CardContent>
                </Card>
              </div>

              <div className="p-4 bg-slate-700 rounded-lg">
                <h4 className="text-white font-medium mb-2">Model Selection Strategy</h4>
                <p className="text-sm text-slate-400">
                  Our intelligent model selector will automatically choose the best model 
                  based on your prompt and style preferences for optimal results.
                </p>
              </div>
            </TabsContent>
          </Tabs>

          {/* Action Buttons */}
          <div className="flex space-x-2">
            <Button variant="outline" className="flex-1">
              <Wand2 className="w-4 h-4 mr-2" />
              Enhance Prompt
            </Button>
            <Button variant="outline" className="flex-1">
              <Sparkles className="w-4 h-4 mr-2" />
              Style Transfer
            </Button>
          </div>

          {/* Generate Button */}
          <Button 
            onClick={handleGenerate}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
            size="lg"
          >
            <Image className="w-5 h-5 mr-2" />
            Generate High-Quality Images
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}