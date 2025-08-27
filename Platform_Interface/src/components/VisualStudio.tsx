import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { ImageWithFallback } from './figma/ImageWithFallback';
import {
  Image,
  Video,
  Upload,
  Download,
  Play,
  Pause,
  RotateCcw,
  Zap,
  Settings,
  Eye,
  Grid,
  List,
  Filter,
  Search,
  Clock,
  CheckCircle,
  XCircle,
  Loader
} from 'lucide-react';

interface GenerationJob {
  id: string;
  type: 'image' | 'video';
  prompt: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  output?: string;
  createdAt: string;
  model: string;
  duration?: number;
}

const mockJobs: GenerationJob[] = [
  {
    id: '1',
    type: 'image',
    prompt: 'A modern office workspace with natural lighting',
    status: 'completed',
    progress: 100,
    output: 'https://images.unsplash.com/photo-1497366216548-37526070297c?w=400&h=300&fit=crop',
    createdAt: '2 minutes ago',
    model: 'DALL-E 3'
  },
  {
    id: '2',
    type: 'video',
    prompt: 'Product showcase animation for tech startup',
    status: 'processing',
    progress: 67,
    createdAt: '5 minutes ago',
    model: 'Runway Gen-2',
    duration: 10
  },
  {
    id: '3',
    type: 'image',
    prompt: 'Abstract data visualization background',
    status: 'completed',
    progress: 100,
    output: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&h=300&fit=crop',
    createdAt: '15 minutes ago',
    model: 'Midjourney'
  },
  {
    id: '4',
    type: 'image',
    prompt: 'Professional headshot for AI agent avatar',
    status: 'failed',
    progress: 0,
    createdAt: '30 minutes ago',
    model: 'Stable Diffusion'
  }
];

export function VisualStudio() {
  const [activeTab, setActiveTab] = useState('generate');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedModel, setSelectedModel] = useState('dall-e-3');
  const [imageSettings, setImageSettings] = useState({
    quality: 'standard',
    size: '1024x1024',
    style: 'vivid'
  });

  const handleGenerate = () => {
    if (!prompt.trim()) return;
    setIsGenerating(true);
    // Simulate generation
    setTimeout(() => {
      setIsGenerating(false);
      setPrompt('');
    }, 3000);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing': return <Loader className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      default: return <Clock className="w-4 h-4 text-yellow-500" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl mb-2">Visual Generation Studio</h1>
          <p className="text-muted-foreground">Create stunning images and videos using AI-powered models.</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline">
            <Upload className="w-4 h-4 mr-2" />
            Upload Reference
          </Button>
          <Button variant="outline">
            <Settings className="w-4 h-4 mr-2" />
            Model Settings
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="generate">Generate</TabsTrigger>
          <TabsTrigger value="gallery">Gallery</TabsTrigger>
          <TabsTrigger value="batch">Batch Processing</TabsTrigger>
        </TabsList>

        <TabsContent value="generate" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Generation Panel */}
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="w-5 h-5" />
                    AI Generation
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="prompt">Prompt</Label>
                    <Textarea
                      id="prompt"
                      placeholder="Describe what you want to generate..."
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      className="min-h-[100px]"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="model">AI Model</Label>
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="dall-e-3">DALL-E 3</SelectItem>
                          <SelectItem value="midjourney">Midjourney</SelectItem>
                          <SelectItem value="stable-diffusion">Stable Diffusion</SelectItem>
                          <SelectItem value="runway">Runway Gen-2</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="type">Content Type</Label>
                      <Select defaultValue="image">
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="image">Image</SelectItem>
                          <SelectItem value="video">Video</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <Button 
                    onClick={handleGenerate} 
                    disabled={!prompt.trim() || isGenerating}
                    className="w-full"
                  >
                    {isGenerating ? (
                      <>
                        <Loader className="w-4 h-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4 mr-2" />
                        Generate
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Preview Area */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Eye className="w-5 h-5" />
                    Preview
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="aspect-video bg-muted rounded-lg flex items-center justify-center border-2 border-dashed">
                    {isGenerating ? (
                      <div className="text-center">
                        <Loader className="w-8 h-8 animate-spin mx-auto mb-2" />
                        <p className="text-sm text-muted-foreground">Generating your content...</p>
                        <Progress value={33} className="w-48 mt-2" />
                      </div>
                    ) : (
                      <div className="text-center">
                        <Image className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
                        <p className="text-sm text-muted-foreground">Your generated content will appear here</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Settings Panel */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Image Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>Quality</Label>
                    <Select value={imageSettings.quality} onValueChange={(value) => 
                      setImageSettings(prev => ({ ...prev, quality: value }))
                    }>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="standard">Standard</SelectItem>
                        <SelectItem value="hd">HD</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label>Size</Label>
                    <Select value={imageSettings.size} onValueChange={(value) => 
                      setImageSettings(prev => ({ ...prev, size: value }))
                    }>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1024x1024">1024x1024</SelectItem>
                        <SelectItem value="1792x1024">1792x1024</SelectItem>
                        <SelectItem value="1024x1792">1024x1792</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label>Style</Label>
                    <Select value={imageSettings.style} onValueChange={(value) => 
                      setImageSettings(prev => ({ ...prev, style: value }))
                    }>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="vivid">Vivid</SelectItem>
                        <SelectItem value="natural">Natural</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Advanced Settings</Label>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="creativity" className="text-sm">Creativity</Label>
                        <Switch id="creativity" />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-sm">Guidance Scale</Label>
                        <Slider defaultValue={[7.5]} max={20} min={1} step={0.5} />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-sm">Steps</Label>
                        <Slider defaultValue={[50]} max={100} min={10} step={10} />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Generations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {mockJobs.slice(0, 3).map((job) => (
                      <div key={job.id} className="flex items-center gap-3 p-2 rounded border">
                        <div className="w-12 h-12 bg-muted rounded overflow-hidden">
                          {job.output && (
                            <ImageWithFallback
                              src={job.output}
                              alt={job.prompt}
                              className="w-full h-full object-cover"
                            />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{job.prompt}</p>
                          <div className="flex items-center gap-2 mt-1">
                            {getStatusIcon(job.status)}
                            <span className="text-xs text-muted-foreground">{job.createdAt}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="gallery" className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input placeholder="Search generations..." className="pl-10 w-64" />
              </div>
              <Button variant="outline">
                <Filter className="w-4 h-4 mr-2" />
                Filter
              </Button>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant={viewMode === 'grid' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('grid')}
              >
                <Grid className="w-4 h-4" />
              </Button>
              <Button
                variant={viewMode === 'list' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('list')}
              >
                <List className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {mockJobs.map((job) => (
                <Card key={job.id} className="overflow-hidden">
                  <div className="aspect-square bg-muted relative">
                    {job.output ? (
                      <ImageWithFallback
                        src={job.output}
                        alt={job.prompt}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        {job.type === 'image' ? (
                          <Image className="w-8 h-8 text-muted-foreground" />
                        ) : (
                          <Video className="w-8 h-8 text-muted-foreground" />
                        )}
                      </div>
                    )}
                    <div className="absolute top-2 right-2">
                      <Badge variant={job.status === 'completed' ? 'default' : 'secondary'}>
                        {job.status}
                      </Badge>
                    </div>
                    {job.status === 'processing' && (
                      <div className="absolute bottom-0 left-0 right-0 p-2 bg-black/50">
                        <Progress value={job.progress} className="h-1" />
                      </div>
                    )}
                  </div>
                  <CardContent className="p-4">
                    <p className="text-sm font-medium mb-2 line-clamp-2">{job.prompt}</p>
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{job.model}</span>
                      <span>{job.createdAt}</span>
                    </div>
                    <div className="flex gap-2 mt-3">
                      <Button size="sm" variant="outline" className="flex-1">
                        <Download className="w-3 h-3 mr-1" />
                        Download
                      </Button>
                      <Button size="sm" variant="outline">
                        <RotateCcw className="w-3 h-3" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {mockJobs.map((job) => (
                <Card key={job.id}>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-16 bg-muted rounded overflow-hidden">
                        {job.output ? (
                          <ImageWithFallback
                            src={job.output}
                            alt={job.prompt}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            {job.type === 'image' ? (
                              <Image className="w-6 h-6 text-muted-foreground" />
                            ) : (
                              <Video className="w-6 h-6 text-muted-foreground" />
                            )}
                          </div>
                        )}
                      </div>
                      <div className="flex-1">
                        <h4 className="font-medium mb-1">{job.prompt}</h4>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span>{job.model}</span>
                          <span>{job.createdAt}</span>
                          <Badge variant={job.status === 'completed' ? 'default' : 'secondary'}>
                            {job.status}
                          </Badge>
                        </div>
                        {job.status === 'processing' && (
                          <Progress value={job.progress} className="mt-2" />
                        )}
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <Download className="w-4 h-4" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <RotateCcw className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="batch" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Batch Processing</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg mb-2">Upload Batch File</h3>
                <p className="text-muted-foreground mb-4">
                  Upload a CSV file with prompts to generate multiple images or videos at once.
                </p>
                <Button>
                  <Upload className="w-4 h-4 mr-2" />
                  Choose File
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}