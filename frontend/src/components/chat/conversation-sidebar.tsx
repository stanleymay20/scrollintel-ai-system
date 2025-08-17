'use client'

import React, { useState, useEffect } from 'react'
import { Search, Plus, MoreVertical, Archive, Trash2, Edit3, MessageSquare, Calendar, Tag, X, Filter } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { cn } from '@/lib/utils'

interface Conversation {
  id: string
  title: string
  agent_id?: string
  created_at: string
  updated_at: string
  message_count: number
  last_message_preview?: string
  tags: string[]
  is_archived: boolean
}

interface ConversationSidebarProps {
  conversations: Conversation[]
  currentConversation: Conversation | null
  onSelectConversation: (conversation: Conversation) => void
  onNewConversation: () => void
  onToggleSidebar: () => void
}

export function ConversationSidebar({
  conversations,
  currentConversation,
  onSelectConversation,
  onNewConversation,
  onToggleSidebar
}: ConversationSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [filteredConversations, setFilteredConversations] = useState(conversations)
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [selectedAgent, setSelectedAgent] = useState<string>('')
  const [sortBy, setSortBy] = useState<'updated' | 'created' | 'title'>('updated')
  const [showArchived, setShowArchived] = useState(false)
  const [editingTitle, setEditingTitle] = useState<string | null>(null)
  const [newTitle, setNewTitle] = useState('')

  // Get unique tags and agents
  const allTags = Array.from(new Set(conversations.flatMap(c => c.tags)))
  const allAgents = Array.from(new Set(conversations.map(c => c.agent_id).filter(Boolean)))

  // Filter and sort conversations
  useEffect(() => {
    let filtered = conversations.filter(conversation => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase()
        const matchesTitle = conversation.title.toLowerCase().includes(query)
        const matchesPreview = conversation.last_message_preview?.toLowerCase().includes(query)
        if (!matchesTitle && !matchesPreview) return false
      }

      // Tag filter
      if (selectedTags.length > 0) {
        const hasSelectedTag = selectedTags.some(tag => conversation.tags.includes(tag))
        if (!hasSelectedTag) return false
      }

      // Agent filter
      if (selectedAgent && conversation.agent_id !== selectedAgent) {
        return false
      }

      // Archive filter
      if (!showArchived && conversation.is_archived) {
        return false
      }

      return true
    })

    // Sort conversations
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'updated':
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
        case 'created':
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        case 'title':
          return a.title.localeCompare(b.title)
        default:
          return 0
      }
    })

    setFilteredConversations(filtered)
  }, [conversations, searchQuery, selectedTags, selectedAgent, sortBy, showArchived])

  const handleEditTitle = async (conversationId: string, title: string) => {
    try {
      // API call to update conversation title
      await fetch(`/api/conversations/${conversationId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
      })
      
      setEditingTitle(null)
      // Refresh conversations list
    } catch (error) {
      console.error('Error updating conversation title:', error)
    }
  }

  const handleDeleteConversation = async (conversationId: string) => {
    if (!confirm('Are you sure you want to delete this conversation?')) return
    
    try {
      await fetch(`/api/conversations/${conversationId}`, {
        method: 'DELETE'
      })
      // Refresh conversations list
    } catch (error) {
      console.error('Error deleting conversation:', error)
    }
  }

  const handleArchiveConversation = async (conversationId: string, archive: boolean) => {
    try {
      await fetch(`/api/conversations/${conversationId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_archived: archive })
      })
      // Refresh conversations list
    } catch (error) {
      console.error('Error archiving conversation:', error)
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60)

    if (diffInHours < 1) {
      return 'Just now'
    } else if (diffInHours < 24) {
      return `${Math.floor(diffInHours)}h ago`
    } else if (diffInHours < 24 * 7) {
      return `${Math.floor(diffInHours / 24)}d ago`
    } else {
      return date.toLocaleDateString()
    }
  }

  const clearFilters = () => {
    setSearchQuery('')
    setSelectedTags([])
    setSelectedAgent('')
    setSortBy('updated')
    setShowArchived(false)
  }

  const hasActiveFilters = searchQuery || selectedTags.length > 0 || selectedAgent || showArchived

  return (
    <div className="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Conversations</h2>
          <div className="flex items-center space-x-1">
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm">
                  <Filter className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Filter Conversations</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Sort by</label>
                    <Select value={sortBy} onValueChange={(value: any) => setSortBy(value)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="updated">Last updated</SelectItem>
                        <SelectItem value="created">Date created</SelectItem>
                        <SelectItem value="title">Title</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {allAgents.length > 0 && (
                    <div>
                      <label className="text-sm font-medium mb-2 block">Agent</label>
                      <Select value={selectedAgent} onValueChange={setSelectedAgent}>
                        <SelectTrigger>
                          <SelectValue placeholder="All agents" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="">All agents</SelectItem>
                          {allAgents.map(agent => (
                            <SelectItem key={agent} value={agent!}>
                              {agent}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {allTags.length > 0 && (
                    <div>
                      <label className="text-sm font-medium mb-2 block">Tags</label>
                      <div className="space-y-2 max-h-32 overflow-y-auto">
                        {allTags.map(tag => (
                          <div key={tag} className="flex items-center space-x-2">
                            <Checkbox
                              id={tag}
                              checked={selectedTags.includes(tag)}
                              onCheckedChange={(checked) => {
                                if (checked) {
                                  setSelectedTags(prev => [...prev, tag])
                                } else {
                                  setSelectedTags(prev => prev.filter(t => t !== tag))
                                }
                              }}
                            />
                            <label htmlFor={tag} className="text-sm">
                              {tag}
                            </label>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="archived"
                      checked={showArchived}
                      onCheckedChange={(checked) => setShowArchived(checked === true)}
                    />
                    <label htmlFor="archived" className="text-sm">
                      Show archived conversations
                    </label>
                  </div>

                  {hasActiveFilters && (
                    <Button variant="outline" onClick={clearFilters} className="w-full">
                      Clear all filters
                    </Button>
                  )}
                </div>
              </DialogContent>
            </Dialog>
            
            <Button variant="ghost" size="sm" onClick={onToggleSidebar}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <Button onClick={onNewConversation} className="w-full mb-4">
          <Plus className="h-4 w-4 mr-2" />
          New Conversation
        </Button>

        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>

        {hasActiveFilters && (
          <div className="mt-2 flex flex-wrap gap-1">
            {selectedTags.map(tag => (
              <Badge
                key={tag}
                variant="secondary"
                className="text-xs cursor-pointer"
                onClick={() => setSelectedTags(prev => prev.filter(t => t !== tag))}
              >
                {tag} ×
              </Badge>
            ))}
            {selectedAgent && (
              <Badge
                variant="secondary"
                className="text-xs cursor-pointer"
                onClick={() => setSelectedAgent('')}
              >
                {selectedAgent} ×
              </Badge>
            )}
            {showArchived && (
              <Badge
                variant="secondary"
                className="text-xs cursor-pointer"
                onClick={() => setShowArchived(false)}
              >
                Archived ×
              </Badge>
            )}
          </div>
        )}
      </div>

      {/* Conversations List */}
      <ScrollArea className="flex-1">
        <div className="p-2">
          {filteredConversations.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-sm">
                {hasActiveFilters ? 'No conversations match your filters' : 'No conversations yet'}
              </p>
              {hasActiveFilters && (
                <Button variant="link" onClick={clearFilters} className="text-xs mt-2">
                  Clear filters
                </Button>
              )}
            </div>
          ) : (
            <div className="space-y-1">
              {filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={cn(
                    "group relative p-3 rounded-lg cursor-pointer transition-colors",
                    "hover:bg-gray-100 dark:hover:bg-gray-700",
                    currentConversation?.id === conversation.id && "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800",
                    conversation.is_archived && "opacity-60"
                  )}
                  onClick={() => onSelectConversation(conversation)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      {editingTitle === conversation.id ? (
                        <Input
                          value={newTitle}
                          onChange={(e) => setNewTitle(e.target.value)}
                          onBlur={() => {
                            if (newTitle.trim()) {
                              handleEditTitle(conversation.id, newTitle.trim())
                            } else {
                              setEditingTitle(null)
                            }
                          }}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              if (newTitle.trim()) {
                                handleEditTitle(conversation.id, newTitle.trim())
                              } else {
                                setEditingTitle(null)
                              }
                            }
                          }}
                          className="h-6 text-sm"
                          autoFocus
                        />
                      ) : (
                        <h3 className="font-medium text-sm truncate mb-1">
                          {conversation.title}
                        </h3>
                      )}
                      
                      {conversation.last_message_preview && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 line-clamp-2 mb-2">
                          {conversation.last_message_preview}
                        </p>
                      )}

                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <div className="flex items-center space-x-2">
                          <Calendar className="h-3 w-3" />
                          <span>{formatDate(conversation.updated_at)}</span>
                          <span>•</span>
                          <span>{conversation.message_count} messages</span>
                        </div>
                      </div>

                      {/* Tags and Agent */}
                      <div className="flex items-center justify-between mt-2">
                        <div className="flex flex-wrap gap-1">
                          {conversation.agent_id && (
                            <Badge variant="outline" className="text-xs">
                              {conversation.agent_id}
                            </Badge>
                          )}
                          {conversation.tags.slice(0, 2).map(tag => (
                            <Badge key={tag} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                          {conversation.tags.length > 2 && (
                            <Badge variant="secondary" className="text-xs">
                              +{conversation.tags.length - 2}
                            </Badge>
                          )}
                        </div>

                        {conversation.is_archived && (
                          <Archive className="h-3 w-3 text-gray-400" />
                        )}
                      </div>
                    </div>

                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="opacity-0 group-hover:opacity-100 h-6 w-6 p-0"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <MoreVertical className="h-3 w-3" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation()
                            setEditingTitle(conversation.id)
                            setNewTitle(conversation.title)
                          }}
                        >
                          <Edit3 className="h-4 w-4 mr-2" />
                          Rename
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation()
                            handleArchiveConversation(conversation.id, !conversation.is_archived)
                          }}
                        >
                          <Archive className="h-4 w-4 mr-2" />
                          {conversation.is_archived ? 'Unarchive' : 'Archive'}
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteConversation(conversation.id)
                          }}
                          className="text-red-600"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="text-xs text-gray-500 text-center">
          {filteredConversations.length} of {conversations.length} conversations
        </div>
      </div>
    </div>
  )
}