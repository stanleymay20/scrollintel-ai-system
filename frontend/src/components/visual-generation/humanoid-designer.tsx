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
import { User, Brain, Heart, Eye, Smile, Crown } from 'lucide-react'

interface HumanoidDesignerProps {
  onGenerate: (type: string, params: any) => void
}

export function HumanoidDesigner({ onGenerate }: HumanoidDesignerProps) {
  const [prompt, setPrompt] = useState('')
  const [age, setAge] = useState([25])
  const [gender, setGender] = useState('any')
  const [ethnicity, setEthnicity] = useState('any')
  const [bodyType, setBodyType] = useState('average')
  const [facialExpression, setFacialExpression] = useState('neutral')
  const [emotionalAuthenticity, setEmotionalAuthenticity] = useState([99])
  const [anatomicalAccuracy, setAnatomicalAccuracy] = useState([99])
  const [skinDetail, setSkinDetail] = useState([95])
  const [microExpressions, setMicroExpressions] = useState([98])
  const [eyeContact, setEyeContact] = useState(true)
  const [naturalBehavior, setNaturalBehavior] = useState(true)

  const handleGenerate = () => {
    onGenerate('humanoid', {
      prompt,
      age: age[0],
      gender,
      ethnicity,
      bodyType,
      facialExpression,
      emotionalAuthenticity: emotionalAuthenticity[0],
      anatomicalAccuracy: anatomicalAccuracy[0],
      skinDetail: skinDetail[0],
      microExpressions: microExpressions[0],
      eyeContact,
      naturalBehavior
    })
  }

  return (
    <div className="h-full flex flex-col space-y-6">
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-white flex items-center">
                <Crown className="w-5 h-5 mr-2 text-gold-500" />
                Advanced Humanoid Character Designer
              </CardTitle>
              <CardDescription className="text-slate-400">
                Create ultra-realistic digital humans with perfect biometric accuracy
              </CardDescription>
            </div>
            <Badge variant="outline" className="text-gold-400 border-gold-400">
              99% Human Accuracy
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Character Description */}
          <div className="space-y-2">
            <Label htmlFor="character-prompt" className="text-white">
              Character Description
              <Badge variant="outline" className="ml-2 text-xs">
                Biometric AI
              </Badge>
            </Label>
            <Textarea
              id="character-prompt"
              placeholder="Describe the humanoid character you want to create..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="min-h-[100px] bg-slate-700 border-slate-600 text-white placeholder:text-slate-400"
            />
          </div>

          <Tabs defaultValue="physical" className="w-full">
            <TabsList className="grid w-full grid-cols-4 bg-slate-700">
              <TabsTrigger value="physical">Physical</TabsTrigger>
              <TabsTrigger value="facial">Facial</TabsTrigger>
              <TabsTrigger value="biometric">Biometric</TabsTrigger>
              <TabsTrigger value="behavior">Behavior</TabsTrigger>
            </TabsList>

            <TabsContent value="physical" className="space-y-4">
              {/* Age */}
              <div className="space-y-2">
                <Label className="text-white">Age: {age[0]} years</Label>
                <Slider
                  value={age}
                  onValueChange={setAge}
                  max={80}
                  min={18}
                  step={1}
                  className="w-full"
                />
              </div>

              {/* Gender */}
              <div className="space-y-2">
                <Label className="text-white">Gender</Label>
                <Select value={gender} onValueChange={setGender}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="any">Any</SelectItem>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                    <SelectItem value="non-binary">Non-binary</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Ethnicity */}
              <div className="space-y-2">
                <Label className="text-white">Ethnicity</Label>
                <Select value={ethnicity} onValueChange={setEthnicity}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="any">Any</SelectItem>
                    <SelectItem value="caucasian">Caucasian</SelectItem>
                    <SelectItem value="african">African</SelectItem>
                    <SelectItem value="asian">Asian</SelectItem>
                    <SelectItem value="hispanic">Hispanic</SelectItem>
                    <SelectItem value="middle-eastern">Middle Eastern</SelectItem>
                    <SelectItem value="mixed">Mixed</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Body Type */}
              <div className="space-y-2">
                <Label className="text-white">Body Type</Label>
                <Select value={bodyType} onValueChange={setBodyType}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="slim">Slim</SelectItem>
                    <SelectItem value="average">Average</SelectItem>
                    <SelectItem value="athletic">Athletic</SelectItem>
                    <SelectItem value="muscular">Muscular</SelectItem>
                    <SelectItem value="curvy">Curvy</SelectItem>
                    <SelectItem value="plus-size">Plus Size</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </TabsContent>

            <TabsContent value="facial" className="space-y-4">
              {/* Facial Expression */}
              <div className="space-y-2">
                <Label className="text-white">Primary Expression</Label>
                <Select value={facialExpression} onValueChange={setFacialExpression}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="neutral">Neutral</SelectItem>
                    <SelectItem value="happy">Happy</SelectItem>
                    <SelectItem value="serious">Serious</SelectItem>
                    <SelectItem value="contemplative">Contemplative</SelectItem>
                    <SelectItem value="confident">Confident</SelectItem>
                    <SelectItem value="friendly">Friendly</SelectItem>
                    <SelectItem value="professional">Professional</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Micro-expressions */}
              <div className="space-y-2">
                <Label className="text-white">
                  Micro-expression Accuracy: {microExpressions[0]}%
                  <Badge variant="outline" className="ml-2 text-xs">
                    Revolutionary
                  </Badge>
                </Label>
                <Slider
                  value={microExpressions}
                  onValueChange={setMicroExpressions}
                  max={99}
                  min={70}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-400">
                  Controls subtle facial movements and emotional authenticity
                </p>
              </div>

              {/* Skin Detail Level */}
              <div className="space-y-2">
                <Label className="text-white">
                  Skin Detail Level: {skinDetail[0]}%
                  <Badge variant="outline" className="ml-2 text-xs">
                    Pore-Level Detail
                  </Badge>
                </Label>
                <Slider
                  value={skinDetail}
                  onValueChange={setSkinDetail}
                  max={100}
                  min={80}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-400">
                  Includes pores, hair follicles, and subsurface scattering
                </p>
              </div>
            </TabsContent>

            <TabsContent value="biometric" className="space-y-4">
              {/* Anatomical Accuracy */}
              <div className="space-y-2">
                <Label className="text-white">
                  Anatomical Accuracy: {anatomicalAccuracy[0]}%
                  <Badge variant="outline" className="ml-2 text-xs">
                    Medical Grade
                  </Badge>
                </Label>
                <Slider
                  value={anatomicalAccuracy}
                  onValueChange={setAnatomicalAccuracy}
                  max={99}
                  min={90}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-400">
                  Perfect proportions and biomechanically accurate structure
                </p>
              </div>

              {/* Emotional Authenticity */}
              <div className="space-y-2">
                <Label className="text-white">
                  Emotional Authenticity: {emotionalAuthenticity[0]}%
                  <Badge variant="outline" className="ml-2 text-xs">
                    AI Breakthrough
                  </Badge>
                </Label>
                <Slider
                  value={emotionalAuthenticity}
                  onValueChange={setEmotionalAuthenticity}
                  max={99}
                  min={85}
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-400">
                  Natural emotional responses and genuine expressions
                </p>
              </div>

              {/* Biometric Features */}
              <div className="grid grid-cols-2 gap-4">
                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Eye className="w-4 h-4 text-blue-500" />
                      <span className="text-white text-sm">Eye Tracking</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">Realistic gaze patterns</p>
                  </CardContent>
                </Card>
                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Heart className="w-4 h-4 text-red-500" />
                      <span className="text-white text-sm">Pulse Simulation</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">Subtle life-like movements</p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="behavior" className="space-y-4">
              {/* Behavioral Settings */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label className="text-white">Natural Eye Contact</Label>
                    <p className="text-xs text-slate-400">Realistic blinking and gaze patterns</p>
                  </div>
                  <Button
                    variant={eyeContact ? "default" : "outline"}
                    size="sm"
                    onClick={() => setEyeContact(!eyeContact)}
                  >
                    {eyeContact ? "Enabled" : "Disabled"}
                  </Button>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label className="text-white">Natural Behavior Cues</Label>
                    <p className="text-xs text-slate-400">Subtle human behavioral patterns</p>
                  </div>
                  <Button
                    variant={naturalBehavior ? "default" : "outline"}
                    size="sm"
                    onClick={() => setNaturalBehavior(!naturalBehavior)}
                  >
                    {naturalBehavior ? "Enabled" : "Disabled"}
                  </Button>
                </div>
              </div>

              {/* Advanced Behavioral Features */}
              <div className="grid grid-cols-1 gap-4">
                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Brain className="w-4 h-4 text-purple-500" />
                      <span className="text-white font-medium">Cognitive Modeling</span>
                    </div>
                    <p className="text-xs text-slate-400">
                      Advanced AI models personality traits and decision-making patterns
                    </p>
                  </CardContent>
                </Card>

                <Card className="bg-slate-700 border-slate-600">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Smile className="w-4 h-4 text-yellow-500" />
                      <span className="text-white font-medium">Emotional Intelligence</span>
                    </div>
                    <p className="text-xs text-slate-400">
                      Context-aware emotional responses and social awareness
                    </p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>

          {/* Generate Button */}
          <Button 
            onClick={handleGenerate}
            className="w-full bg-gradient-to-r from-gold-600 to-amber-600 hover:from-gold-700 hover:to-amber-700 text-white"
            size="lg"
          >
            <User className="w-5 h-5 mr-2" />
            Generate Ultra-Realistic Humanoid
          </Button>

          {/* Disclaimer */}
          <div className="p-4 bg-slate-700 rounded-lg">
            <p className="text-xs text-slate-400">
              <strong>Note:</strong> Our humanoid generation technology creates digital humans 
              that are indistinguishable from real people. Please use responsibly and in 
              accordance with ethical guidelines.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}