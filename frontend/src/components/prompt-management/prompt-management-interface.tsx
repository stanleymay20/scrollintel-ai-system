'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PromptEditor } from './prompt-editor'
import { TemplateLibrary } from './template-library'
import { ABTestingDashboard } from './ab-testing-dashboard'
import { OptimizationMonitor } from './optimization-monitor'
import { AnalyticsDashboard } from './analytics-dashboard'
import { 
  Edit, 
  Library, 
  FlaskConical, 
  Zap, 
  BarChart3,
  Plus,
  Settings
} from 'lucide-react'

interface PromptTemplate {
  id?: string
  name: string
  content: string
  category: string
  tags: string[]
  variables: any[]
  created_by?: string
  created_at?: string
  updated_at?: string
}

interface Experiment {
  id: string
  name: string
  description: string
  status: string
  variants: any[]
  start_date: string
  end_date?: string
  confidence_level: number
  statistical_significance: number
  winner_variant_id?: string
  created_by: string
  created_at: string
}

export function PromptManagementInterface() {
  const [activeTab, setActiveTab] = useState('library')
  const [currentPrompt, setCurrentPrompt] = useState<PromptTemplate | undefined>()
  const [isEditorOpen, setIsEditorOpen] = useState(false)

  // Template Library handlers
  const handleCreateNewTemplate = () => {
    setCurrentPrompt(undefined)
    setIsEditorOpen(true)
    setActiveTab('editor')
  }

  const handleEditTemplate = (template: PromptTemplate) => {
    setCurrentPrompt(template)
    setIsEditorOpen(true)
    setActiveTab('editor')
  }

  const handleDeleteTemplate = async (templateId: string) => {
    if (confirm('Are you sure you want to delete this template?')) {
      try {
        // API call to delete template
        console.log('Deleting template:', templateId)
        // Refresh template list
      } catch (error) {
        console.error('Failed to delete template:', error)
      }
    }
  }

  const handleImportTemplates = async (file: File) => {
    try {
      const text = await file.text()
      const templates = JSON.parse(text)
      console.log('Importing templates:', templates)
      // API call to import templates
    } catch (error) {
      console.error('Failed to import templates:', error)
    }
  }

  const handleExportTemplates = async (templateIds: string[]) => {
    try {
      // API call to get templates
      const templates: any[] = [] // Mock data
      const blob = new Blob([JSON.stringify(templates, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'prompt-templates.json'
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to export templates:', error)
    }
  }

  // Prompt Editor handlers
  const handleSavePrompt = async (prompt: PromptTemplate) => {
    try {
      if (prompt.id) {
        // Update existing prompt
        console.log('Updating prompt:', prompt)
      } else {
        // Create new prompt
        console.log('Creating prompt:', prompt)
      }
      setIsEditorOpen(false)
      setActiveTab('library')
    } catch (error) {
      console.error('Failed to save prompt:', error)
    }
  }

  const handleTestPrompt = async (prompt: PromptTemplate) => {
    try {
      console.log('Testing prompt:', prompt)
      // API call to test prompt
    } catch (error) {
      console.error('Failed to test prompt:', error)
    }
  }

  const handleVersionHistory = (promptId: string) => {
    console.log('Viewing version history for:', promptId)
    // Navigate to version history view
  }

  // A/B Testing handlers
  const handleCreateExperiment = () => {
    console.log('Creating new experiment')
    // Navigate to experiment creation
  }

  const handleEditExperiment = (experiment: Experiment) => {
    console.log('Editing experiment:', experiment)
    // Navigate to experiment editor
  }

  const handleStartExperiment = async (experimentId: string) => {
    try {
      console.log('Starting experiment:', experimentId)
      // API call to start experiment
    } catch (error) {
      console.error('Failed to start experiment:', error)
    }
  }

  const handlePauseExperiment = async (experimentId: string) => {
    try {
      console.log('Pausing experiment:', experimentId)
      // API call to pause experiment
    } catch (error) {
      console.error('Failed to pause experiment:', error)
    }
  }

  const handleStopExperiment = async (experimentId: string) => {
    try {
      console.log('Stopping experiment:', experimentId)
      // API call to stop experiment
    } catch (error) {
      console.error('Failed to stop experiment:', error)
    }
  }

  const handlePromoteWinner = async (experimentId: string, variantId: string) => {
    try {
      console.log('Promoting winner:', { experimentId, variantId })
      // API call to promote winner
    } catch (error) {
      console.error('Failed to promote winner:', error)
    }
  }

  // Optimization handlers
  const handleStartOptimization = async (promptId: string, config: any) => {
    try {
      console.log('Starting optimization:', { promptId, config })
      // API call to start optimization
    } catch (error) {
      console.error('Failed to start optimization:', error)
    }
  }

  const handlePauseOptimization = async (jobId: string) => {
    try {
      console.log('Pausing optimization:', jobId)
      // API call to pause optimization
    } catch (error) {
      console.error('Failed to pause optimization:', error)
    }
  }

  const handleStopOptimization = async (jobId: string) => {
    try {
      console.log('Stopping optimization:', jobId)
      // API call to stop optimization
    } catch (error) {
      console.error('Failed to stop optimization:', error)
    }
  }

  const handleApplyOptimization = async (jobId: string, generationId: string) => {
    try {
      console.log('Applying optimization:', { jobId, generationId })
      // API call to apply optimization
    } catch (error) {
      console.error('Failed to apply optimization:', error)
    }
  }

  // Analytics handlers
  const handleExportReport = async (filters: any) => {
    try {
      console.log('Exporting report with filters:', filters)
      // API call to generate and download report
    } catch (error) {
      console.error('Failed to export report:', error)
    }
  }

  const handleDrillDown = (metric: string, filters: any) => {
    console.log('Drilling down into metric:', { metric, filters })
    // Navigate to detailed view
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Prompt Management</h1>
            <p className="text-gray-600">Manage, test, and optimize your AI prompts</p>
          </div>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </Button>
            <Button onClick={handleCreateNewTemplate}>
              <Plus className="w-4 h-4 mr-2" />
              New Prompt
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          <div className="bg-white border-b px-6">
            <TabsList className="grid w-full max-w-2xl grid-cols-5">
              <TabsTrigger value="library" className="flex items-center space-x-2">
                <Library className="w-4 h-4" />
                <span>Library</span>
              </TabsTrigger>
              <TabsTrigger value="editor" className="flex items-center space-x-2">
                <Edit className="w-4 h-4" />
                <span>Editor</span>
              </TabsTrigger>
              <TabsTrigger value="testing" className="flex items-center space-x-2">
                <FlaskConical className="w-4 h-4" />
                <span>A/B Testing</span>
              </TabsTrigger>
              <TabsTrigger value="optimization" className="flex items-center space-x-2">
                <Zap className="w-4 h-4" />
                <span>Optimization</span>
              </TabsTrigger>
              <TabsTrigger value="analytics" className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4" />
                <span>Analytics</span>
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex-1 overflow-hidden">
            <TabsContent value="library" className="h-full m-0">
              <TemplateLibrary
                onCreateNew={handleCreateNewTemplate}
                onEditTemplate={handleEditTemplate}
                onDeleteTemplate={handleDeleteTemplate}
                onImportTemplates={handleImportTemplates}
                onExportTemplates={handleExportTemplates}
              />
            </TabsContent>

            <TabsContent value="editor" className="h-full m-0">
              <PromptEditor
                prompt={currentPrompt}
                onSave={handleSavePrompt}
                onTest={handleTestPrompt}
                onVersionHistory={handleVersionHistory}
              />
            </TabsContent>

            <TabsContent value="testing" className="h-full m-0">
              <ABTestingDashboard
                onCreateExperiment={handleCreateExperiment}
                onEditExperiment={handleEditExperiment}
                onStartExperiment={handleStartExperiment}
                onPauseExperiment={handlePauseExperiment}
                onStopExperiment={handleStopExperiment}
                onPromoteWinner={handlePromoteWinner}
              />
            </TabsContent>

            <TabsContent value="optimization" className="h-full m-0">
              <OptimizationMonitor
                onStartOptimization={handleStartOptimization}
                onPauseOptimization={handlePauseOptimization}
                onStopOptimization={handleStopOptimization}
                onApplyOptimization={handleApplyOptimization}
              />
            </TabsContent>

            <TabsContent value="analytics" className="h-full m-0">
              <AnalyticsDashboard
                onExportReport={handleExportReport}
                onDrillDown={handleDrillDown}
              />
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </div>
  )
}