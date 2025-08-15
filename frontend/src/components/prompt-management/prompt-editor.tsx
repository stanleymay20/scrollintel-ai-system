'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Save, Play, History, Settings, AlertCircle, CheckCircle } from 'lucide-react'

interface PromptVariable {
  name: string
  type: string
  description: string
  required: boolean
}

interface PromptTemplate {
  id?: string
  name: string
  content: string
  category: string
  tags: string[]
  variables: PromptVariable[]
  created_by?: string
  created_at?: string
  updated_at?: string
}

interface ValidationError {
  line: number
  message: string
  type: 'error' | 'warning'
}

interface PromptEditorProps {
  prompt?: PromptTemplate
  onSave: (prompt: PromptTemplate) => void
  onTest: (prompt: PromptTemplate) => void
  onVersionHistory: (promptId: string) => void
}

export function PromptEditor({ prompt, onSave, onTest, onVersionHistory }: PromptEditorProps) {
  const [currentPrompt, setCurrentPrompt] = useState<PromptTemplate>(
    prompt || {
      name: '',
      content: '',
      category: '',
      tags: [],
      variables: []
    }
  )
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([])
  const [isValidating, setIsValidating] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [newTag, setNewTag] = useState('')
  const [testOutput, setTestOutput] = useState('')

  // Syntax highlighting and validation
  useEffect(() => {
    validatePrompt(currentPrompt.content)
  }, [currentPrompt.content])

  const validatePrompt = async (content: string) => {
    setIsValidating(true)
    const errors: ValidationError[] = []
    
    // Basic validation rules
    const lines = content.split('\n')
    lines.forEach((line, index) => {
      // Check for unclosed variables
      const openBraces = (line.match(/\{/g) || []).length
      const closeBraces = (line.match(/\}/g) || []).length
      if (openBraces !== closeBraces) {
        errors.push({
          line: index + 1,
          message: 'Unclosed variable brackets',
          type: 'error'
        })
      }
      
      // Check for undefined variables
      const variableMatches = line.match(/\{([^}]+)\}/g)
      if (variableMatches) {
        variableMatches.forEach(match => {
          const varName = match.slice(1, -1)
          if (!currentPrompt.variables.some(v => v.name === varName)) {
            errors.push({
              line: index + 1,
              message: `Undefined variable: ${varName}`,
              type: 'warning'
            })
          }
        })
      }
    })
    
    setValidationErrors(errors)
    setIsValidating(false)
  }

  const handleSave = async () => {
    setIsSaving(true)
    try {
      await onSave(currentPrompt)
    } finally {
      setIsSaving(false)
    }
  }

  const handleTest = async () => {
    setTestOutput('Testing prompt...')
    try {
      await onTest(currentPrompt)
      setTestOutput('Test completed successfully')
    } catch (error) {
      setTestOutput(`Test failed: ${error}`)
    }
  }

  const addTag = () => {
    if (newTag && !currentPrompt.tags.includes(newTag)) {
      setCurrentPrompt(prev => ({
        ...prev,
        tags: [...prev.tags, newTag]
      }))
      setNewTag('')
    }
  }

  const removeTag = (tagToRemove: string) => {
    setCurrentPrompt(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }))
  }

  const addVariable = () => {
    setCurrentPrompt(prev => ({
      ...prev,
      variables: [...prev.variables, {
        name: '',
        type: 'string',
        description: '',
        required: true
      }]
    }))
  }

  const updateVariable = (index: number, field: keyof PromptVariable, value: any) => {
    setCurrentPrompt(prev => ({
      ...prev,
      variables: prev.variables.map((variable, i) => 
        i === index ? { ...variable, [field]: value } : variable
      )
    }))
  }

  const removeVariable = (index: number) => {
    setCurrentPrompt(prev => ({
      ...prev,
      variables: prev.variables.filter((_, i) => i !== index)
    }))
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-4">
          <input
            type="text"
            placeholder="Prompt name"
            value={currentPrompt.name}
            onChange={(e) => setCurrentPrompt(prev => ({ ...prev, name: e.target.value }))}
            className="text-lg font-semibold bg-transparent border-none outline-none"
          />
          <Badge variant={validationErrors.length > 0 ? "destructive" : "default"}>
            {validationErrors.length === 0 ? (
              <><CheckCircle className="w-3 h-3 mr-1" /> Valid</>
            ) : (
              <><AlertCircle className="w-3 h-3 mr-1" /> {validationErrors.length} issues</>
            )}
          </Badge>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm" onClick={handleTest}>
            <Play className="w-4 h-4 mr-2" />
            Test
          </Button>
          {currentPrompt.id && (
            <Button variant="outline" size="sm" onClick={() => onVersionHistory(currentPrompt.id!)}>
              <History className="w-4 h-4 mr-2" />
              History
            </Button>
          )}
          <Button onClick={handleSave} disabled={isSaving}>
            <Save className="w-4 h-4 mr-2" />
            {isSaving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Editor */}
        <div className="flex-1 flex flex-col">
          <Tabs defaultValue="editor" className="flex-1 flex flex-col">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="editor">Editor</TabsTrigger>
              <TabsTrigger value="variables">Variables</TabsTrigger>
              <TabsTrigger value="test">Test</TabsTrigger>
            </TabsList>
            
            <TabsContent value="editor" className="flex-1 flex flex-col">
              <div className="flex-1 p-4">
                <Textarea
                  placeholder="Enter your prompt template here..."
                  value={currentPrompt.content}
                  onChange={(e) => setCurrentPrompt(prev => ({ ...prev, content: e.target.value }))}
                  className="h-full resize-none font-mono"
                />
              </div>
              
              {/* Validation Errors */}
              {validationErrors.length > 0 && (
                <div className="p-4 border-t">
                  <h4 className="font-semibold mb-2">Validation Issues</h4>
                  <div className="space-y-2">
                    {validationErrors.map((error, index) => (
                      <Alert key={index} variant={error.type === 'error' ? 'destructive' : 'default'}>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>
                          Line {error.line}: {error.message}
                        </AlertDescription>
                      </Alert>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="variables" className="flex-1 p-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">Variables</h3>
                  <Button onClick={addVariable} size="sm">Add Variable</Button>
                </div>
                
                <div className="space-y-4">
                  {currentPrompt.variables.map((variable, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium mb-1">Name</label>
                          <input
                            type="text"
                            value={variable.name}
                            onChange={(e) => updateVariable(index, 'name', e.target.value)}
                            className="w-full p-2 border rounded"
                            placeholder="variable_name"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-1">Type</label>
                          <select
                            value={variable.type}
                            onChange={(e) => updateVariable(index, 'type', e.target.value)}
                            className="w-full p-2 border rounded"
                          >
                            <option value="string">String</option>
                            <option value="number">Number</option>
                            <option value="boolean">Boolean</option>
                            <option value="array">Array</option>
                          </select>
                        </div>
                        <div className="col-span-2">
                          <label className="block text-sm font-medium mb-1">Description</label>
                          <input
                            type="text"
                            value={variable.description}
                            onChange={(e) => updateVariable(index, 'description', e.target.value)}
                            className="w-full p-2 border rounded"
                            placeholder="Variable description"
                          />
                        </div>
                        <div className="flex items-center justify-between col-span-2">
                          <label className="flex items-center">
                            <input
                              type="checkbox"
                              checked={variable.required}
                              onChange={(e) => updateVariable(index, 'required', e.target.checked)}
                              className="mr-2"
                            />
                            Required
                          </label>
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => removeVariable(index)}
                          >
                            Remove
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="test" className="flex-1 p-4">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Test Output</h3>
                <div className="p-4 bg-gray-50 rounded-lg min-h-32">
                  <pre className="whitespace-pre-wrap">{testOutput || 'No test output yet'}</pre>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>

        {/* Sidebar */}
        <div className="w-80 border-l p-4 space-y-6">
          {/* Metadata */}
          <div>
            <h3 className="font-semibold mb-2">Metadata</h3>
            <div className="space-y-2">
              <div>
                <label className="block text-sm font-medium mb-1">Category</label>
                <input
                  type="text"
                  value={currentPrompt.category}
                  onChange={(e) => setCurrentPrompt(prev => ({ ...prev, category: e.target.value }))}
                  className="w-full p-2 border rounded"
                  placeholder="e.g., content-generation"
                />
              </div>
            </div>
          </div>

          {/* Tags */}
          <div>
            <h3 className="font-semibold mb-2">Tags</h3>
            <div className="flex flex-wrap gap-2 mb-2">
              {currentPrompt.tags.map(tag => (
                <Badge key={tag} variant="secondary" className="cursor-pointer" onClick={() => removeTag(tag)}>
                  {tag} ×
                </Badge>
              ))}
            </div>
            <div className="flex space-x-2">
              <input
                type="text"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addTag()}
                className="flex-1 p-2 border rounded"
                placeholder="Add tag"
              />
              <Button onClick={addTag} size="sm">Add</Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}