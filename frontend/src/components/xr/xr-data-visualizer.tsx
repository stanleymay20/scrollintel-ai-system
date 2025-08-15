'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Headphones, 
  Maximize, 
  RotateCcw, 
  Settings, 
  Users, 
  Download,
  Play,
  Pause,
  Volume2,
  VolumeX
} from 'lucide-react'

// XR Data Types
interface XRDataPoint {
  id: string
  position: [number, number, number]
  value: number
  label: string
  color: string
  metadata: Record<string, any>
}

interface XRVisualizationConfig {
  type: 'scatter3d' | 'network3d' | 'surface3d' | 'volume3d'
  data: XRDataPoint[]
  interactions: {
    hover: boolean
    select: boolean
    manipulate: boolean
  }
  animation: {
    enabled: boolean
    duration: number
    loop: boolean
  }
  collaboration: {
    enabled: boolean
    maxUsers: number
    voiceChat: boolean
  }
}

interface XRSession {
  sessionId: string
  users: Array<{
    id: string
    name: string
    avatar: string
    position: [number, number, number]
    isActive: boolean
  }>
  isHost: boolean
  voiceEnabled: boolean
}

// WebXR Hook
const useWebXR = () => {
  const [isSupported, setIsSupported] = useState(false)
  const [isSessionActive, setIsSessionActive] = useState(false)
  const [sessionType, setSessionType] = useState<'immersive-vr' | 'immersive-ar' | null>(null)

  useEffect(() => {
    // Check WebXR support
    if ('xr' in navigator) {
      const xr = (navigator as any).xr
      
      Promise.all([
        xr.isSessionSupported('immersive-vr'),
        xr.isSessionSupported('immersive-ar')
      ]).then(([vrSupported, arSupported]) => {
        setIsSupported(vrSupported || arSupported)
      }).catch(() => {
        setIsSupported(false)
      })
    }
  }, [])

  const startXRSession = useCallback(async (type: 'immersive-vr' | 'immersive-ar') => {
    if (!isSupported) return false

    try {
      const xr = (navigator as any).xr
      const session = await xr.requestSession(type, {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hand-tracking', 'eye-tracking']
      })

      setIsSessionActive(true)
      setSessionType(type)

      session.addEventListener('end', () => {
        setIsSessionActive(false)
        setSessionType(null)
      })

      return session
    } catch (error) {
      console.error('Failed to start XR session:', error)
      return false
    }
  }, [isSupported])

  const endXRSession = useCallback(() => {
    setIsSessionActive(false)
    setSessionType(null)
  }, [])

  return {
    isSupported,
    isSessionActive,
    sessionType,
    startXRSession,
    endXRSession
  }
}

// 3D Scene Component
const XR3DScene: React.FC<{
  config: XRVisualizationConfig
  onDataPointSelect?: (point: XRDataPoint) => void
}> = ({ config, onDataPointSelect }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<any>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!canvasRef.current) return

    // Initialize 3D scene (using Three.js-like API)
    const initScene = async () => {
      try {
        // In a real implementation, you would use Three.js or similar
        // This is a placeholder for the 3D scene initialization
        
        const canvas = canvasRef.current!
        const context = canvas.getContext('webgl2') || canvas.getContext('webgl')
        
        if (!context) {
          console.error('WebGL not supported')
          return
        }

        // Create scene, camera, renderer
        const scene = {
          canvas,
          context,
          dataPoints: config.data,
          camera: {
            position: [0, 0, 5],
            rotation: [0, 0, 0]
          },
          controls: {
            enabled: true,
            sensitivity: 1.0
          }
        }

        sceneRef.current = scene
        
        // Render initial scene
        renderScene(scene)
        
        setIsLoading(false)
      } catch (error) {
        console.error('Failed to initialize 3D scene:', error)
        setIsLoading(false)
      }
    }

    initScene()
  }, [config])

  const renderScene = (scene: any) => {
    if (!scene || !scene.context) return

    const { context, dataPoints } = scene
    
    // Clear canvas
    context.clearColor(0.1, 0.1, 0.2, 1.0)
    context.clear(context.COLOR_BUFFER_BIT | context.DEPTH_BUFFER_BIT)
    
    // Render data points (simplified)
    dataPoints.forEach((point: XRDataPoint, index: number) => {
      // In a real implementation, render 3D objects here
      console.log(`Rendering point ${point.id} at position`, point.position)
    })
  }

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!sceneRef.current || !onDataPointSelect) return

    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Ray casting to detect clicked data point (simplified)
    const clickedPoint = config.data.find((point, index) => {
      // In a real implementation, perform proper ray casting
      return Math.random() > 0.8 // Placeholder for hit detection
    })

    if (clickedPoint) {
      onDataPointSelect(clickedPoint)
    }
  }

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer"
        width={800}
        height={600}
        onClick={handleCanvasClick}
        style={{ background: 'linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%)' }}
      />
      
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="text-white">Loading 3D Scene...</div>
        </div>
      )}
      
      {/* 3D Controls Overlay */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <Button
          size="sm"
          variant="secondary"
          onClick={() => {
            // Reset camera position
            if (sceneRef.current) {
              sceneRef.current.camera.position = [0, 0, 5]
              sceneRef.current.camera.rotation = [0, 0, 0]
              renderScene(sceneRef.current)
            }
          }}
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
        
        <Button
          size="sm"
          variant="secondary"
          onClick={() => {
            // Toggle fullscreen
            if (canvasRef.current) {
              if (document.fullscreenElement) {
                document.exitFullscreen()
              } else {
                canvasRef.current.requestFullscreen()
              }
            }
          }}
        >
          <Maximize className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

// Collaboration Panel
const CollaborationPanel: React.FC<{
  session: XRSession | null
  onJoinSession: () => void
  onLeaveSession: () => void
  onToggleVoice: () => void
}> = ({ session, onJoinSession, onLeaveSession, onToggleVoice }) => {
  if (!session) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Collaboration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Button onClick={onJoinSession} className="w-full">
            Join Collaborative Session
          </Button>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5" />
          Collaborative Session
          <Badge variant="secondary">{session.users.length} users</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Active Users */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Active Users</h4>
          {session.users.map((user) => (
            <div key={user.id} className="flex items-center gap-2 p-2 rounded-lg bg-muted">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-sm">
                {user.name.charAt(0)}
              </div>
              <span className="text-sm">{user.name}</span>
              {user.id === 'current-user' && (
                <Badge variant="outline" className="ml-auto">You</Badge>
              )}
              {session.isHost && user.id === 'current-user' && (
                <Badge variant="default" className="ml-auto">Host</Badge>
              )}
              <div className={`w-2 h-2 rounded-full ${user.isActive ? 'bg-green-500' : 'bg-gray-400'}`} />
            </div>
          ))}
        </div>

        {/* Voice Controls */}
        <div className="flex items-center justify-between">
          <span className="text-sm">Voice Chat</span>
          <Button
            size="sm"
            variant={session.voiceEnabled ? "default" : "outline"}
            onClick={onToggleVoice}
          >
            {session.voiceEnabled ? (
              <Volume2 className="h-4 w-4" />
            ) : (
              <VolumeX className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Session Controls */}
        <div className="flex gap-2">
          <Button variant="outline" onClick={onLeaveSession} className="flex-1">
            Leave Session
          </Button>
          {session.isHost && (
            <Button variant="destructive" className="flex-1">
              End Session
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// Main XR Data Visualizer Component
export const XRDataVisualizer: React.FC<{
  data: XRDataPoint[]
  title?: string
  onExport?: (format: string) => void
}> = ({ data, title = "XR Data Visualization", onExport }) => {
  const { isSupported, isSessionActive, sessionType, startXRSession, endXRSession } = useWebXR()
  
  const [config, setConfig] = useState<XRVisualizationConfig>({
    type: 'scatter3d',
    data,
    interactions: {
      hover: true,
      select: true,
      manipulate: true
    },
    animation: {
      enabled: false,
      duration: 2000,
      loop: false
    },
    collaboration: {
      enabled: false,
      maxUsers: 8,
      voiceChat: false
    }
  })

  const [selectedPoint, setSelectedPoint] = useState<XRDataPoint | null>(null)
  const [xrSession, setXrSession] = useState<XRSession | null>(null)

  // Update config when data changes
  useEffect(() => {
    setConfig(prev => ({ ...prev, data }))
  }, [data])

  const handleStartVR = async () => {
    const session = await startXRSession('immersive-vr')
    if (session) {
      console.log('VR session started')
    }
  }

  const handleStartAR = async () => {
    const session = await startXRSession('immersive-ar')
    if (session) {
      console.log('AR session started')
    }
  }

  const handleJoinCollaboration = () => {
    // Simulate joining a collaborative session
    setXrSession({
      sessionId: 'session-123',
      users: [
        {
          id: 'current-user',
          name: 'You',
          avatar: '',
          position: [0, 0, 0],
          isActive: true
        },
        {
          id: 'user-2',
          name: 'Alice Johnson',
          avatar: '',
          position: [2, 0, 1],
          isActive: true
        },
        {
          id: 'user-3',
          name: 'Bob Smith',
          avatar: '',
          position: [-1, 1, 2],
          isActive: false
        }
      ],
      isHost: true,
      voiceEnabled: false
    })
  }

  const handleLeaveCollaboration = () => {
    setXrSession(null)
  }

  const handleToggleVoice = () => {
    if (xrSession) {
      setXrSession({
        ...xrSession,
        voiceEnabled: !xrSession.voiceEnabled
      })
    }
  }

  const handleExport = (format: string) => {
    if (onExport) {
      onExport(format)
    } else {
      // Default export behavior
      console.log(`Exporting visualization as ${format}`)
    }
  }

  return (
    <div className="w-full h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <Headphones className="h-6 w-6" />
          <h2 className="text-xl font-semibold">{title}</h2>
          {isSessionActive && (
            <Badge variant="default" className="ml-2">
              {sessionType === 'immersive-vr' ? 'VR Active' : 'AR Active'}
            </Badge>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {/* XR Controls */}
          {isSupported && !isSessionActive && (
            <>
              <Button variant="outline" onClick={handleStartVR}>
                Enter VR
              </Button>
              <Button variant="outline" onClick={handleStartAR}>
                Enter AR
              </Button>
            </>
          )}
          
          {isSessionActive && (
            <Button variant="destructive" onClick={endXRSession}>
              Exit XR
            </Button>
          )}
          
          {/* Export Options */}
          <Button
            variant="outline"
            onClick={() => handleExport('glb')}
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          
          <Button variant="outline">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* 3D Visualization */}
        <div className="flex-1 p-4">
          <Card className="h-full">
            <CardContent className="p-0 h-full">
              <XR3DScene
                config={config}
                onDataPointSelect={setSelectedPoint}
              />
            </CardContent>
          </Card>
        </div>

        {/* Side Panel */}
        <div className="w-80 p-4 space-y-4 border-l">
          {/* Visualization Controls */}
          <Card>
            <CardHeader>
              <CardTitle>Visualization</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs value={config.type} onValueChange={(value) => 
                setConfig(prev => ({ ...prev, type: value as any }))
              }>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="scatter3d">Scatter</TabsTrigger>
                  <TabsTrigger value="network3d">Network</TabsTrigger>
                </TabsList>
                <TabsList className="grid w-full grid-cols-2 mt-2">
                  <TabsTrigger value="surface3d">Surface</TabsTrigger>
                  <TabsTrigger value="volume3d">Volume</TabsTrigger>
                </TabsList>
              </Tabs>

              <div className="mt-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Animation</span>
                  <Button
                    size="sm"
                    variant={config.animation.enabled ? "default" : "outline"}
                    onClick={() => setConfig(prev => ({
                      ...prev,
                      animation: { ...prev.animation, enabled: !prev.animation.enabled }
                    }))}
                  >
                    {config.animation.enabled ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  </Button>
                </div>

                <div className="text-sm text-muted-foreground">
                  Data Points: {config.data.length}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Selected Point Details */}
          {selectedPoint && (
            <Card>
              <CardHeader>
                <CardTitle>Selected Point</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div>
                    <span className="text-sm font-medium">Label:</span>
                    <span className="text-sm ml-2">{selectedPoint.label}</span>
                  </div>
                  <div>
                    <span className="text-sm font-medium">Value:</span>
                    <span className="text-sm ml-2">{selectedPoint.value}</span>
                  </div>
                  <div>
                    <span className="text-sm font-medium">Position:</span>
                    <span className="text-sm ml-2">
                      ({selectedPoint.position.map(p => p.toFixed(2)).join(', ')})
                    </span>
                  </div>
                  {Object.entries(selectedPoint.metadata).map(([key, value]) => (
                    <div key={key}>
                      <span className="text-sm font-medium">{key}:</span>
                      <span className="text-sm ml-2">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Collaboration Panel */}
          <CollaborationPanel
            session={xrSession}
            onJoinSession={handleJoinCollaboration}
            onLeaveSession={handleLeaveCollaboration}
            onToggleVoice={handleToggleVoice}
          />

          {/* WebXR Status */}
          {!isSupported && (
            <Card>
              <CardContent className="p-4">
                <div className="text-sm text-muted-foreground text-center">
                  WebXR not supported in this browser.
                  <br />
                  Try Chrome or Firefox with WebXR enabled.
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

export default XRDataVisualizer