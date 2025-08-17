'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Mic, MicOff, Square, Volume2, VolumeX } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { cn } from '@/lib/utils'

interface VoiceInputProps {
  onTranscript: (transcript: string) => void
  isRecording: boolean
  onRecordingChange: (recording: boolean) => void
  children?: React.ReactNode
  language?: string
  continuous?: boolean
  interimResults?: boolean
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
  resultIndex: number
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string
  message: string
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onstart: ((this: SpeechRecognition, ev: Event) => any) | null
  onend: ((this: SpeechRecognition, ev: Event) => any) | null
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null
  onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition
    webkitSpeechRecognition: new () => SpeechRecognition
  }
}

export function VoiceInput({
  onTranscript,
  isRecording,
  onRecordingChange,
  children,
  language = 'en-US',
  continuous = true,
  interimResults = true
}: VoiceInputProps) {
  const [isSupported, setIsSupported] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [transcript, setTranscript] = useState('')
  const [interimTranscript, setInterimTranscript] = useState('')
  const [audioLevel, setAudioLevel] = useState(0)
  const [isListening, setIsListening] = useState(false)
  
  const recognitionRef = useRef<SpeechRecognition | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const microphoneRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  // Check for speech recognition support
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    setIsSupported(!!SpeechRecognition)
    
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition()
    }
  }, [])

  // Setup audio level monitoring
  const setupAudioMonitoring = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      audioContextRef.current = audioContext
      
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      analyserRef.current = analyser
      
      const microphone = audioContext.createMediaStreamSource(stream)
      microphoneRef.current = microphone
      microphone.connect(analyser)
      
      const dataArray = new Uint8Array(analyser.frequencyBinCount)
      
      const updateAudioLevel = () => {
        if (!analyserRef.current || !isRecording) return
        
        analyser.getByteFrequencyData(dataArray)
        const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length
        setAudioLevel(Math.min(100, (average / 128) * 100))
        
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel)
      }
      
      updateAudioLevel()
    } catch (error) {
      console.error('Error setting up audio monitoring:', error)
      setError('Microphone access denied')
    }
  }, [isRecording])

  // Cleanup audio monitoring
  const cleanupAudioMonitoring = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
    
    if (microphoneRef.current) {
      microphoneRef.current.disconnect()
      microphoneRef.current = null
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    
    setAudioLevel(0)
  }, [])

  // Setup speech recognition
  useEffect(() => {
    if (!recognitionRef.current) return

    const recognition = recognitionRef.current
    recognition.continuous = continuous
    recognition.interimResults = interimResults
    recognition.lang = language

    recognition.onstart = () => {
      setIsListening(true)
      setError(null)
      setupAudioMonitoring()
    }

    recognition.onend = () => {
      setIsListening(false)
      cleanupAudioMonitoring()
      if (isRecording) {
        // Restart if we're still supposed to be recording
        try {
          recognition.start()
        } catch (error) {
          console.error('Error restarting recognition:', error)
        }
      }
    }

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let finalTranscript = ''
      let interimTranscript = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          finalTranscript += result[0].transcript
        } else {
          interimTranscript += result[0].transcript
        }
      }

      setTranscript(prev => prev + finalTranscript)
      setInterimTranscript(interimTranscript)

      if (finalTranscript) {
        onTranscript(finalTranscript)
      }
    }

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.error('Speech recognition error:', event.error)
      
      let errorMessage = 'Speech recognition error'
      switch (event.error) {
        case 'no-speech':
          errorMessage = 'No speech detected'
          break
        case 'audio-capture':
          errorMessage = 'Microphone not available'
          break
        case 'not-allowed':
          errorMessage = 'Microphone access denied'
          break
        case 'network':
          errorMessage = 'Network error'
          break
        case 'service-not-allowed':
          errorMessage = 'Speech service not allowed'
          break
        default:
          errorMessage = `Speech recognition error: ${event.error}`
      }
      
      setError(errorMessage)
      setIsListening(false)
      onRecordingChange(false)
      cleanupAudioMonitoring()
    }

    return () => {
      if (recognition) {
        recognition.onstart = null
        recognition.onend = null
        recognition.onresult = null
        recognition.onerror = null
      }
    }
  }, [language, continuous, interimResults, isRecording, onTranscript, onRecordingChange, setupAudioMonitoring, cleanupAudioMonitoring])

  // Handle recording state changes
  useEffect(() => {
    if (!recognitionRef.current) return

    const recognition = recognitionRef.current

    if (isRecording && !isListening) {
      try {
        recognition.start()
      } catch (error) {
        console.error('Error starting recognition:', error)
        setError('Failed to start speech recognition')
      }
    } else if (!isRecording && isListening) {
      try {
        recognition.stop()
      } catch (error) {
        console.error('Error stopping recognition:', error)
      }
    }
  }, [isRecording, isListening])

  const toggleRecording = () => {
    if (!isSupported) {
      setError('Speech recognition not supported in this browser')
      return
    }

    setError(null)
    onRecordingChange(!isRecording)
    
    if (!isRecording) {
      setTranscript('')
      setInterimTranscript('')
    }
  }

  const clearError = () => {
    setError(null)
  }

  if (children) {
    return (
      <>
        <div onClick={toggleRecording}>
          {children}
        </div>
        {error && (
          <Alert variant="destructive" className="mt-2">
            <AlertDescription>
              {error}
              <Button variant="ghost" size="sm" onClick={clearError} className="ml-2">
                Dismiss
              </Button>
            </AlertDescription>
          </Alert>
        )}
      </>
    )
  }

  if (!isSupported) {
    return (
      <Alert>
        <MicOff className="h-4 w-4" />
        <AlertDescription>
          Speech recognition is not supported in this browser.
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="space-y-4">
      {/* Voice Input Controls */}
      <div className="flex items-center justify-center space-x-4">
        <Button
          onClick={toggleRecording}
          variant={isRecording ? "destructive" : "default"}
          size="lg"
          className={cn(
            "relative",
            isRecording && "animate-pulse"
          )}
        >
          {isRecording ? (
            <Square className="h-5 w-5 mr-2" />
          ) : (
            <Mic className="h-5 w-5 mr-2" />
          )}
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </Button>

        {isRecording && (
          <div className="flex items-center space-x-2">
            <Volume2 className="h-4 w-4 text-gray-500" />
            <div className="w-20">
              <Progress value={audioLevel} className="h-2" />
            </div>
            <Badge variant="secondary" className="text-xs">
              {Math.round(audioLevel)}%
            </Badge>
          </div>
        )}
      </div>

      {/* Status */}
      {isListening && (
        <div className="text-center">
          <div className="inline-flex items-center space-x-2 text-sm text-green-600 dark:text-green-400">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span>Listening...</span>
          </div>
        </div>
      )}

      {/* Live Transcript */}
      {(transcript || interimTranscript) && (
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="text-sm">
            <span className="text-gray-900 dark:text-gray-100">{transcript}</span>
            <span className="text-gray-500 dark:text-gray-400 italic">
              {interimTranscript}
            </span>
            {isRecording && (
              <span className="inline-block w-1 h-4 bg-blue-500 animate-pulse ml-1" />
            )}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <MicOff className="h-4 w-4" />
          <AlertDescription>
            {error}
            <Button variant="ghost" size="sm" onClick={clearError} className="ml-2">
              Dismiss
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Language Selection */}
      <div className="flex items-center justify-center space-x-2 text-xs text-gray-500">
        <span>Language:</span>
        <Badge variant="outline">{language}</Badge>
      </div>
    </div>
  )
}