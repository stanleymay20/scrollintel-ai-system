'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Search, 
  Filter, 
  Plus, 
  Star, 
  Copy, 
  Edit, 
  Trash2, 
  Download,
  Upload,
  Grid,
  List
} from 'lucide-react'

interface PromptTemplate {
  id: string
  name: string
  content: string
  category: string
  tags: string[]
  variables: any[]
  created_by: string
  created_at: string
  updated_at: string
  usage_count: number
  rating: number
  is_favorite: boolean
}

interface TemplateLibraryProps {
  onCreateNew: () => void
  onEditTemplate: (template: PromptTemplate) => void
  onDeleteTemplate: (templateId: string) => void
  onImportTemplates: (file: File) => void
  onExportTemplates: (templateIds: string[]) => void
}

export function TemplateLibrary({ 
  onCreateNew, 
  onEditTemplate, 
  onDeleteTemplate,
  onImportTemplates,
  onExportTemplates
}: TemplateLibraryProps) {
  const [templates, setTemplates] = useState<PromptTemplate[]>([])
  const [filteredTemplates, setFilteredTemplates] = useState<PromptTemplate[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [sortBy, setSortBy] = useState<'name' | 'created_at' | 'usage_count' | 'rating'>('created_at')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [selectedTemplates, setSelectedTemplates] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)

  // Mock data - replace with actual API calls
  useEffect(() => {
    const mockTemplates: PromptTemplate[] = [
      {
        id: '1',
        name: 'Content Generation',
        content: 'Generate {content_type} about {topic} for {audience}...',
        category: 'content',
        tags: ['generation', 'marketing', 'seo'],
        variables: [
          { name: 'content_type', type: 'string', required: true },
          { name: 'topic', type: 'string', required: true },
          { name: 'audience', type: 'string', required: true }
        ],
        created_by: 'john.doe',
        created_at: '2024-01-15T10:00:00Z',
        updated_at: '2024-01-20T15:30:00Z',
        usage_count: 45,
        rating: 4.5,
        is_favorite: true
      },
      {
        id: '2',
        name: 'Code Review Assistant',
        content: 'Review the following {language} code and provide feedback...',
        category: 'development',
        tags: ['code-review', 'development', 'quality'],
        variables: [
          { name: 'language', type: 'string', required: true },
          { name: 'code', type: 'string', required: true }
        ],
        created_by: 'jane.smith',
        created_at: '2024-01-10T09:00:00Z',
        updated_at: '2024-01-18T11:45:00Z',
        usage_count: 32,
        rating: 4.2,
        is_favorite: false
      }
    ]
    
    setTemplates(mockTemplates)
    setFilteredTemplates(mockTemplates)
    setIsLoading(false)
  }, [])

  // Filter and search logic
  useEffect(() => {
    let filtered = templates

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(template =>
        template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        template.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    }

    // Category filter
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(template => template.category === selectedCategory)
    }

    // Tags filter
    if (selectedTags.length > 0) {
      filtered = filtered.filter(template =>
        selectedTags.every(tag => template.tags.includes(tag))
      )
    }

    // Sort
    filtered.sort((a, b) => {
      const aValue = a[sortBy]
      const bValue = b[sortBy]
      const comparison = aValue < bValue ? -1 : aValue > bValue ? 1 : 0
      return sortOrder === 'asc' ? comparison : -comparison
    })

    setFilteredTemplates(filtered)
  }, [templates, searchQuery, selectedCategory, selectedTags, sortBy, sortOrder])

  const categories = ['all', ...Array.from(new Set(templates.map(t => t.category)))]
  const allTags = Array.from(new Set(templates.flatMap(t => t.tags)))

  const toggleTemplateSelection = (templateId: string) => {
    setSelectedTemplates(prev =>
      prev.includes(templateId)
        ? prev.filter(id => id !== templateId)
        : [...prev, templateId]
    )
  }

  const toggleFavorite = async (templateId: string) => {
    setTemplates(prev =>
      prev.map(template =>
        template.id === templateId
          ? { ...template, is_favorite: !template.is_favorite }
          : template
      )
    )
  }

  const copyTemplate = async (template: PromptTemplate) => {
    try {
      await navigator.clipboard.writeText(template.content)
      // Show success toast
    } catch (error) {
      console.error('Failed to copy template:', error)
    }
  }

  const handleFileImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      onImportTemplates(file)
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">Template Library</h1>
          <div className="flex items-center space-x-2">
            <input
              type="file"
              accept=".json"
              onChange={handleFileImport}
              className="hidden"
              id="import-templates"
            />
            <Button variant="outline" size="sm" onClick={() => document.getElementById('import-templates')?.click()}>
              <Upload className="w-4 h-4 mr-2" />
              Import
            </Button>
            {selectedTemplates.length > 0 && (
              <Button variant="outline" size="sm" onClick={() => onExportTemplates(selectedTemplates)}>
                <Download className="w-4 h-4 mr-2" />
                Export ({selectedTemplates.length})
              </Button>
            )}
            <Button onClick={onCreateNew}>
              <Plus className="w-4 h-4 mr-2" />
              New Template
            </Button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded-lg"
            />
          </div>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border rounded-lg"
          >
            {categories.map(category => (
              <option key={category} value={category}>
                {category === 'all' ? 'All Categories' : category}
              </option>
            ))}
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border rounded-lg"
          >
            <option value="created_at">Date Created</option>
            <option value="name">Name</option>
            <option value="usage_count">Usage Count</option>
            <option value="rating">Rating</option>
          </select>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </Button>

          <div className="flex border rounded-lg">
            <Button
              variant={viewMode === 'grid' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('grid')}
            >
              <Grid className="w-4 h-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('list')}
            >
              <List className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Tag Filters */}
        {allTags.length > 0 && (
          <div className="mt-4">
            <div className="flex flex-wrap gap-2">
              {allTags.map(tag => (
                <Badge
                  key={tag}
                  variant={selectedTags.includes(tag) ? 'default' : 'outline'}
                  className="cursor-pointer"
                  onClick={() => setSelectedTags(prev =>
                    prev.includes(tag)
                      ? prev.filter(t => t !== tag)
                      : [...prev, tag]
                  )}
                >
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 p-4 overflow-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-lg">Loading templates...</div>
          </div>
        ) : filteredTemplates.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <div className="text-lg mb-2">No templates found</div>
            <div className="text-sm">Try adjusting your search or filters</div>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredTemplates.map(template => (
              <div
                key={template.id}
                className={`p-4 border rounded-lg hover:shadow-md transition-shadow cursor-pointer ${
                  selectedTemplates.includes(template.id) ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => toggleTemplateSelection(template.id)}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold truncate">{template.name}</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      toggleFavorite(template.id)
                    }}
                  >
                    <Star className={`w-4 h-4 ${template.is_favorite ? 'fill-yellow-400 text-yellow-400' : ''}`} />
                  </Button>
                </div>
                
                <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                  {template.content.substring(0, 100)}...
                </p>
                
                <div className="flex flex-wrap gap-1 mb-3">
                  {template.tags.slice(0, 3).map(tag => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                  {template.tags.length > 3 && (
                    <Badge variant="secondary" className="text-xs">
                      +{template.tags.length - 3}
                    </Badge>
                  )}
                </div>
                
                <div className="flex items-center justify-between text-xs text-gray-500 mb-3">
                  <span>Used {template.usage_count} times</span>
                  <span>★ {template.rating}</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">
                    by {template.created_by}
                  </span>
                  <div className="flex space-x-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        copyTemplate(template)
                      }}
                    >
                      <Copy className="w-3 h-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        onEditTemplate(template)
                      }}
                    >
                      <Edit className="w-3 h-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        onDeleteTemplate(template.id)
                      }}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {filteredTemplates.map(template => (
              <div
                key={template.id}
                className={`p-4 border rounded-lg hover:shadow-sm transition-shadow cursor-pointer ${
                  selectedTemplates.includes(template.id) ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => toggleTemplateSelection(template.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <h3 className="font-semibold">{template.name}</h3>
                      <Badge variant="outline">{template.category}</Badge>
                      {template.is_favorite && (
                        <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                      )}
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      {template.content.substring(0, 150)}...
                    </p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                      <span>Used {template.usage_count} times</span>
                      <span>★ {template.rating}</span>
                      <span>by {template.created_by}</span>
                      <span>{new Date(template.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div className="flex space-x-1 ml-4">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        copyTemplate(template)
                      }}
                    >
                      <Copy className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        onEditTemplate(template)
                      }}
                    >
                      <Edit className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        onDeleteTemplate(template.id)
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}