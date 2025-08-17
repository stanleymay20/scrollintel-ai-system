'use client'

import React, { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Upload, Layers, Eye, Box, Zap, Crown } from 'lucide-react'

interface DepthVisualizationToolsProps {
  onGenerate: (type: string, params: any) => void
}

export function DepthVisualizationTools({ onGenerate }: DepthVisualizationToolsProps) {
  const [sourceImage, setSourceImage] = useState<File | null>(null)
  const [sourceVideo, setSourceVideo] = useState<File | null>(null)
  const [conversionType, setConversionType] = useState('image-to-3d')
  const [depthPrecision, setDepthPrecision] = useState([99])
  const [geometryQuality, setGeometryQuality] = useState([95])
  const [parallaxIntensity, setParallaxIntensity] = useState([80])
  const [temporalConsistency, setTemporalConsistency] = useState([99])
  const [cameraMovementRealism, setCameraMovementRealism] = useState([99])
  const [edgePreservation, setEdgePreservation] = useState(true)
  const [vrCompatible, setVrCompatible] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (file.type.startsWith('image/')) {
        setSourceImage(file)
        setSourceVideo(null)
        setConversionType('image-to-3d')
      } else if (file.type.startsWith('video/')) {
        setSourceVideo(file)
        setSourceImage(null)
        setConversionType('video-to-3d')
      }
    }
  }

  const handleGenerate = () => {
    onGenerate('2d-to-3d', {
      sourceImage,
      sourceVideo,
      conversionType,
      depthPrecision: depthPrecision[0],
      geometryQuality: geometryQuality[0],
      parallaxIntensity: parallaxIntensity[0],
      temporalConsistency: temporalConsistency[0],
      cameraMovementRealism: cameraMovementRealism[0],
      edgePreservation,
      vrCompatible
    })
  }

  return (
    <div className="h-full flex flex-col space-y-6">
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-white flex items-center">
                <Crown className="w-5 h-5 mr-2 text-cyan-500" />
                Advanced 2D-to-3D Conversion
              </CardTitle>
              <CardDescription className="text-slate-400">
                Transform flat content into immersive 3D experiences with perfect depth
              </CardDescription>
            </div>
            <Badge variant="outline" className="text-cyan-400 border-cyan-400">
              Sub-Pixel Precision
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* File Upload */}
          <div className="space-y-2">
            <Label className="text-white">Source Media</Label>
            <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,video/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <div className="space-y-4">
                {sourceImage || sourceVideo ? (
                  <div className="space-y-2">
                    <div className="w-16 h-16 mx-auto bg-slate-700 rounded-lg flex items-center justify-center">
                      {sourceImage ? (
                        <img 
                          src={URL.createObjectURL(sourceImage)} 
                          alt="Source" 
                          className="w-full h-full object-cover rounded-lg"
                        />
                      ) : (
                        <Box className="w-8 h-8 text-slate-400" />
                      )}
                    </div>
                    <p className="text-white font-medium">
                      {sourceImage?.name || sourceVideo?.name}
                    </p>
                    <p className="text-xs text-slate-400">
                      {sourceImage ? 'Image' : 'Video'} • Ready for 3D conversion
                    </p>
                  </div>
                ) : (
                  <>
                    <Upload className="w-12 h-12 mx-auto text-slate-400" />
                    <div>
                      <p className="text-white font-medium">Upload Image or Video</p>
                      <p className="text-sm text-slate-400">
                        Supports JPG, PNG, MP4, MOV formats
                      </p>
                    </div>
                  </>
                )}
                <Button 
                  variant="outline" 
                  onClick={() => fileInputRef.current?.click()}
                >
                  {sourceImage || sourceVideo ? 'Change File' : 'Choose File'}
                </Button>
              </div>
            </div>
          </div>

          <Tabs defaultValue="depth" className="w-full">
            <TabsList className="grid w-full grid-cols-3 bg-slate-700">
              <TabsTrigger value="depth">Depth Analysis</TabsTrigger>
              <TabsTrigger value="geometry">3D Geometry</TabsTrigger>
              <TabsTrigger value="output">Output Settings</TabsTrigger>
            </TabsList>

            <TabsContent value="depth" className="space-y-4">
              {/* Depth Precision */}
              <div className="space-y-2">
                <Label className="text-white">
                  Depth Precision: {depthPrecision[0]}%
                  <Badge variant="outline" className="ml-2 text-xs">
                    Sub-Pixel Accuracy
                  </Badge>
                </Label>
                <Slider
                  value={depthPrecision}
                  onValueChange={setDepthPrecision}
                  max={99}
                  min={80}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-400">
                  Controls the accuracy of depth map generation
                </p>
              </div>

              {/* Edge Preservation */}
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-white">Perfect Edge Detection</Label>
                  <p className="text-xs text-slate-400">Maintains sharp object boundaries</p>
                </div>
                <Button
                  variant={edgePreservation ? "default" : "outline"}
                  size="sm"
                  onClick={() => setEdgePreservation(!edgePreservation)}
                >
                  {edgePreservation ? "Enabled" : "Disabled"}
                </Button>
              </div>

              {/* Depth Visualization Preview */}
              <Card className="bg-slate-700 border-slate-600">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Layers className="w-4 h-4 text-cyan-500" />
                    <span className="text-white font-medium">Multi-Scale Depth Analysis</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="h-16 bg-gradient-to-r from-black to-white rounded opacity-50"></div>
                    <div className="h-16 bg-gradient-to-r from-blue-900 to-blue-300 rounded opacity-50"></div>
                    <div className="h-16 bg-gradient-to-r from-purple-900 to-purple-300 rounded opacity-50"></div>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">
                    Depth maps will be generated at multiple scales for optimal accuracy
                  </p>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="geometry" className="space-y-4">
              {/* Geometry Quality */}
              <div className="space-y-2">
                <Label className="text-white">
                  3D Geometry Quality: {geometryQuality[0]}%
                  <Badge variant="outline" className="ml-2 text-xs">
                    Ultra High
                  </Badge>
                </Label>
                <Slider
                  value={geometryQuality}
                  onValueChange={setGeometryQuality}
                  max={100}
                  min={70}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-400">
                  Controls mesh density and geometric reconstruction quality
                </p>
              </div>

              {/* Parallax Intensity */}
              <div className="space-y-2">
                <Label className="text-white">
                  Parallax Intensity: {parallaxIntensity[0]}%
                </Label>
                <Slider
                  value={parallaxIntensity}
                  onValueChange={setParallaxIntensity}
                  max={100}
                  min={0}
                  step={5}
                  className="w-full"
                />
                <p className="text-xs text-slate-400">
                  Controls the strength of 3D depth effect
                </p>
              </div>

              {/* Camera Movement Realism */}
              <div className="space-y-2">
                <Label className="text-white">
                  Camera Movement Realism: {cameraMovementRealism[0]}%
                  <Badge variant="outline" className="ml-2 text-xs">
                    Revolutionary
                  </Badge>
                </Label>
                <Slider
                  value={cameraMovementRealism}
                  onValueChange={setCameraMovementRealism}
                  max={99}
                  min={70}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-400">
                  Generates realistic camera movement and perspective shifts
                </p>
              </div>

              {/* Temporal Consistency (for videos) */}
              {sourceVideo && (
                <div className="space-y-2">
                  <Label className="text-white">
                    Temporal Depth Consistency: {temporalConsistency[0]}%
                    <Badge variant="outline" className="ml-2 text-xs">
                      Zero Flicker
                    </Badge>
                  </Label>
                  <Slider
                    value={temporalConsistency}
                    onValueChange={setTemporalConsistency}
                    max={99}
                    min={85}
                    step={1}
                    className="w-full"
                  />
                  <p className="text-xs text-slate-400">
                    Maintains consistent depth across video frames
                  </p>
                </div>
              )}
            </TabsContent>

            <TabsContent value="output" className="space-y-4">
              {/* Output Format */}
              <div className="space-y-2">
                <Label className="text-white">Output Format</Label>
                <Select defaultValue="stereoscopic">
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="stereoscopic">Stereoscopic 3D</SelectItem>
                    <SelectItem value="depth-map">Depth Map + Original</SelectItem>
                    <SelectItem value="anaglyph">Anaglyph 3D</SelectItem>
                    <SelectItem value="side-by-side">Side-by-Side 3D</SelectItem>
                    <SelectItem value="vr-180">VR 180°</SelectItem>
                    <SelectItem value="vr-360">VR 360°</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* VR Compatibility */}
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-white">VR/AR Compatible</Label>
                  <p className="text-xs text-slate-400">Optimized for immersive viewing</p>
                </div>
                <Button
                  variant={vrCompatible ? "default" : "outline"}
                  size="sm"
                  onClick={() => setVrCompatible(!vrCompatible)}
                >
                  {vrCompatible ? "Enabled" : "Disabled"}
                </Button>
              </div>

              {/* Advanced Features */}
              <div className="grid grid-cols-1 gap-4">
                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Eye className="w-4 h-4 text-green-500" />
                      <span className="text-white font-medium">Foreground/Background Separation</span>
                    </div>
                    <p className="text-xs text-slate-400">
                      Intelligent object segmentation for accurate depth layering
                    </p>
                  </CardContent>
                </Card>

                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Zap className="w-4 h-4 text-yellow-500" />
                      <span className="text-white font-medium">Real-time Preview</span>
                    </div>
                    <p className="text-xs text-slate-400">
                      Live 3D preview during conversion process
                    </p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>

          {/* Generate Button */}
          <Button 
            onClick={handleGenerate}
            disabled={!sourceImage && !sourceVideo}
            className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white disabled:opacity-50"
            size="lg"
          >
            <Box className="w-5 h-5 mr-2" />
            Convert to Ultra-Realistic 3D
          </Button>

          {/* Technical Info */}
          <div className="p-4 bg-slate-700 rounded-lg">
            <h4 className="text-white font-medium mb-2">Advanced 3D Conversion Technology</h4>
            <ul className="text-xs text-slate-400 space-y-1">
              <li>• Sub-pixel precision depth estimation</li>
              <li>• Perfect geometric reconstruction</li>
              <li>• Zero-artifact temporal consistency</li>
              <li>• 99% camera movement accuracy</li>
              <li>• Professional VR/AR compatibility</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}