'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { 
  Send, 
  Mic, 
  MicOff, 
  Sparkles, 
  Clock, 
  TrendingUp, 
  BarChart3,
  Database,
  Users,
  DollarSign,
  Lightbulb,
  Search,
  Filter,
  Download,
  Share
} from 'lucide-react'

interface QuerySuggestion {
  id: string
  text: string
  category: 'analytics' | 'business' | 'technical' | 'insights'
  icon: React.ReactNode
}

interface QueryResult {
  id: string
  query: string
  response: string
  data?: any
  visualizations?: any[]
  timestamp: Date
  processingTime: number
  confidence: number
}

interface NaturalLanguageInterfaceProps {
  onQuerySubmit?: (query: string) => void
  className?: string
}

export function NaturalLanguageInterface({ onQuerySubmit, className }: NaturalLanguageInterfaceProps) {
  const [query, setQuery] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [queryHistory, setQueryHistory] = useState<QueryResult[]>([])
  const [selectedResult, setSelectedResult] = useState<QueryResult | null>(null)
  const [suggestions, setSuggestions] = useState<QuerySuggestion[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const recognitionRef = useRef<any>(null)

  // Sample query suggestions
  const defaultSuggestions: QuerySuggestion[] = [
    {
      id: '1',
      text: 'Show me the revenue trends for the last quarter',
      category: 'business',
      icon: <TrendingUp className="h-4 w-4" />
    },
    {
      id: '2',
      text: 'What are the top performing agents this month?',
      category: 'analytics',
      icon: <BarChart3 className="h-4 w-4" />
    },
    {
      id: '3',
      text: 'How is our system performance compared to last week?',
      category: 'technical',
      icon: <Database className="h-4 w-4" />
    },
    {
      id: '4',
      text: 'Identify cost optimization opportunities',
      category: 'insights',
      icon: <Lightbulb className="h-4 w-4" />
    },
    {
      id: '5',
      text: 'Show customer satisfaction metrics by region',
      category: 'business',
      icon: <Users className="h-4 w-4" />
    },
    {
      id: '6',
      text: 'What is the ROI of our AI implementations?',
      category: 'business',
      icon: <DollarSign className="h-4 w-4" />
    }
  ]

  useEffect(() => {
    setSuggestions(defaultSuggestions)
    
    // Initialize speech recognition if available
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = false
      recognitionRef.current.interimResults = false
      recognitionRef.current.lang = 'en-US'

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript
        setQuery(transcript)
        setIsListening(false)
      }

      recognitionRef.current.onerror = () => {
        setIsListening(false)
      }

      recognitionRef.current.onend = () => {
        setIsListening(false)
      }
    }
  }, [])

  const handleSubmitQuery = async () => {
    if (!query.trim() || isProcessing) return

    setIsProcessing(true)
    const startTime = Date.now()

    try {
      // Simulate API call to natural language processing service
      const response = await fetch('/api/natural-language/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() })
      })

      const result = await response.json()
      const processingTime = Date.now() - startTime

      const queryResult: QueryResult = {
        id: Date.now().toString(),
        query: query.trim(),
        response: result.response || generateMockResponse(query),
        data: result.data,
        visualizations: result.visualizations,
        timestamp: new Date(),
        processingTime,
        confidence: result.confidence || 0.95
      }

      setQueryHistory(prev => [queryResult, ...prev])
      setSelectedResult(queryResult)
      setQuery('')
      
      if (onQuerySubmit) {
        onQuerySubmit(query.trim())
      }
    } catch (error) {
      console.error('Query processing failed:', error)
      // Add error handling
    } finally {
      setIsProcessing(false)
    }
  }

  const generateMockResponse = (query: string): string => {
    const lowerQuery = query.toLowerCase()
    
    if (lowerQuery.includes('revenue') || lowerQuery.includes('sales')) {
      return "Based on the latest data, revenue has increased by 15.3% compared to last quarter, with total revenue reaching $2.4M. The growth is primarily driven by enterprise clients (+23%) and our new AI services (+45%). Key contributing factors include improved customer retention (94.2%) and successful expansion into healthcare analytics."
    }
    
    if (lowerQuery.includes('performance') || lowerQuery.includes('system')) {
      return "System performance metrics show excellent health across all infrastructure components. Average response time is 127ms (15% improvement), uptime is 99.97%, and we're processing 847GB/hour of data. CPU utilization is optimal at 23.4%, and all 47 active agents are operating within normal parameters."
    }
    
    if (lowerQuery.includes('agents') || lowerQuery.includes('top')) {
      return "Top performing agents this month: 1) CTO Agent (94.7% accuracy, 1,247 decisions), 2) Data Scientist Agent (96.3% model accuracy, 89 insights), 3) BI Agent (847GB processed, 156 reports generated). Overall agent efficiency has improved by 12% with the new orchestration system."
    }
    
    if (lowerQuery.includes('cost') || lowerQuery.includes('optimization')) {
      return "Cost optimization analysis reveals several opportunities: 1) Consolidating underutilized cloud resources could save $45K/month, 2) Automating manual processes could reduce operational costs by $67K/month, 3) Optimizing data storage strategies could save $23K/month. Total potential savings: $135K/month."
    }
    
    return "I've analyzed your query and found relevant insights. The data shows positive trends across key metrics with opportunities for optimization in several areas. Would you like me to dive deeper into any specific aspect?"
  }

  const handleSuggestionClick = (suggestion: QuerySuggestion) => {
    setQuery(suggestion.text)
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  const toggleVoiceInput = () => {
    if (!recognitionRef.current) return

    if (isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
    } else {
      recognitionRef.current.start()
      setIsListening(true)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmitQuery()
    }
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'business': return 'bg-blue-100 text-blue-800'
      case 'analytics': return 'bg-green-100 text-green-800'
      case 'technical': return 'bg-purple-100 text-purple-800'
      case 'insights': return 'bg-orange-100 text-orange-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Query Input Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Natural Language Query Interface
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Ask questions in plain English about your business data, system performance, or analytics
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="relative">
            <Textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything about your business data, performance metrics, or insights..."
              className="min-h-[100px] pr-20 resize-none"
              disabled={isProcessing}
            />
            <div className="absolute bottom-3 right-3 flex items-center gap-2">
              {recognitionRef.current && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={toggleVoiceInput}
                  className={isListening ? 'text-red-600' : ''}
                >
                  {isListening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                </Button>
              )}
              <Button
                size="sm"
                onClick={handleSubmitQuery}
                disabled={!query.trim() || isProcessing}
              >
                {isProcessing ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>

          {/* Query Suggestions */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Suggested Queries</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {suggestions.map((suggestion) => (
                <Button
                  key={suggestion.id}
                  variant="outline"
                  size="sm"
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="justify-start h-auto p-3 text-left"
                >
                  <div className="flex items-start gap-2">
                    {suggestion.icon}
                    <div className="flex-1">
                      <p className="text-sm">{suggestion.text}</p>
                      <Badge variant="secondary" className={`text-xs mt-1 ${getCategoryColor(suggestion.category)}`}>
                        {suggestion.category}
                      </Badge>
                    </div>
                  </div>
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Query Results */}
      {queryHistory.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Results List */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Query History
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 max-h-96 overflow-y-auto">
              {queryHistory.map((result) => (
                <div
                  key={result.id}
                  onClick={() => setSelectedResult(result)}
                  className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedResult?.id === result.id 
                      ? 'border-primary bg-primary/5' 
                      : 'border-border hover:bg-muted/50'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-sm font-medium line-clamp-2">{result.query}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline" className="text-xs">
                          {result.processingTime}ms
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {Math.round(result.confidence * 100)}% confidence
                        </Badge>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {result.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Selected Result Details */}
          {selectedResult && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Search className="h-5 w-5" />
                    Query Result
                  </span>
                  <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline">
                      <Share className="h-4 w-4" />
                    </Button>
                    <Button size="sm" variant="outline">
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-medium text-sm text-muted-foreground mb-2">Query</h4>
                  <p className="text-sm bg-muted p-3 rounded-lg">{selectedResult.query}</p>
                </div>
                
                <div>
                  <h4 className="font-medium text-sm text-muted-foreground mb-2">Response</h4>
                  <div className="text-sm bg-background p-4 rounded-lg border">
                    {selectedResult.response}
                  </div>
                </div>

                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>Processing Time: {selectedResult.processingTime}ms</span>
                  <span>Confidence: {Math.round(selectedResult.confidence * 100)}%</span>
                  <span>Generated: {selectedResult.timestamp.toLocaleString()}</span>
                </div>

                {selectedResult.visualizations && selectedResult.visualizations.length > 0 && (
                  <div>
                    <h4 className="font-medium text-sm text-muted-foreground mb-2">Visualizations</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {selectedResult.visualizations.map((viz, index) => (
                        <div key={index} className="p-3 bg-muted rounded-lg text-center">
                          <BarChart3 className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                          <p className="text-xs">{viz.title || `Chart ${index + 1}`}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Empty State */}
      {queryHistory.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Sparkles className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-lg font-medium mb-2">Ready to Answer Your Questions</h3>
            <p className="text-muted-foreground mb-4">
              Ask me anything about your business data, system performance, or analytics in plain English
            </p>
            <Button onClick={() => textareaRef.current?.focus()}>
              Start Asking Questions
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}