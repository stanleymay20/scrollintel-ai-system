"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Folder, 
  FolderPlus, 
  Users, 
  Settings, 
  Globe,
  Lock,
  Building2,
  Search,
  MoreHorizontal,
  AlertCircle,
  CheckCircle,
  UserPlus
} from 'lucide-react';

interface Workspace {
  id: string;
  name: string;
  description?: string;
  organization_id: string;
  owner_id: string;
  visibility: string;
  created_at: string;
}

interface WorkspaceManagerProps {
  organizationId: string;
}

export default function WorkspaceManager({ organizationId }: WorkspaceManagerProps) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Create workspace state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [createData, setCreateData] = useState({
    name: '',
    description: '',
    visibility: 'private'
  });
  const [creating, setCreating] = useState(false);
  
  // Member management state
  const [selectedWorkspace, setSelectedWorkspace] = useState<string | null>(null);
  const [showMembersModal, setShowMembersModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [members, setMembers] = useState<any[]>([]);
  const [loadingMembers, setLoadingMembers] = useState(false);

  useEffect(() => {
    fetchWorkspaces();
  }, [organizationId]);

  const fetchWorkspaces = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/v1/user-management/workspaces?organization_id=${organizationId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('session_token')}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch workspaces');
      }

      const workspaceData = await response.json();
      setWorkspaces(workspaceData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workspaces');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateWorkspace = async () => {
    try {
      setCreating(true);
      setError(null);
      setSuccess(null);

      const response = await fetch(`/api/v1/user-management/organizations/${organizationId}/workspaces`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('session_token')}`
        },
        body: JSON.stringify(createData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create workspace');
      }

      setSuccess('Workspace created successfully');
      setShowCreateForm(false);
      setCreateData({ name: '', description: '', visibility: 'private' });
      fetchWorkspaces(); // Refresh the workspace list
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workspace');
    } finally {
      setCreating(false);
    }
  };

  const handleViewMembers = async (workspaceId: string) => {
    setSelectedWorkspace(workspaceId);
    setShowMembersModal(true);
    await fetchWorkspaceMembers(workspaceId);
  };

  const handleWorkspaceSettings = (workspaceId: string) => {
    setSelectedWorkspace(workspaceId);
    setShowSettingsModal(true);
  };

  const fetchWorkspaceMembers = async (workspaceId: string) => {
    try {
      setLoadingMembers(true);
      const response = await fetch(`/api/v1/user-management/workspaces/${workspaceId}/members`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('session_token')}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch workspace members');
      }

      const membersData = await response.json();
      setMembers(membersData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workspace members');
    } finally {
      setLoadingMembers(false);
    }
  };

  const getVisibilityIcon = (visibility: string) => {
    switch (visibility) {
      case 'public':
        return <Globe className="w-4 h-4" />;
      case 'organization':
        return <Building2 className="w-4 h-4" />;
      case 'private':
      default:
        return <Lock className="w-4 h-4" />;
    }
  };

  const getVisibilityBadgeVariant = (visibility: string) => {
    switch (visibility) {
      case 'public':
        return 'default';
      case 'organization':
        return 'secondary';
      case 'private':
      default:
        return 'outline';
    }
  };

  const filteredWorkspaces = workspaces.filter(workspace =>
    workspace.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (workspace.description?.toLowerCase().includes(searchTerm.toLowerCase()) ?? false)
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Workspaces</h2>
          <p className="text-gray-600">Organize your projects and collaborate with your team</p>
        </div>
        <Button onClick={() => setShowCreateForm(true)}>
          <FolderPlus className="w-4 h-4 mr-2" />
          Create Workspace
        </Button>
      </div>

      {/* Create Workspace Form */}
      {showCreateForm && (
        <Card>
          <CardHeader>
            <CardTitle>Create New Workspace</CardTitle>
            <CardDescription>
              Create a workspace to organize your projects and collaborate with your team
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="workspace-name">Workspace Name</Label>
              <Input
                id="workspace-name"
                value={createData.name}
                onChange={(e) => setCreateData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="My Workspace"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="workspace-description">Description (Optional)</Label>
              <Textarea
                id="workspace-description"
                value={createData.description}
                onChange={(e) => setCreateData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Brief description of this workspace"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="workspace-visibility">Visibility</Label>
              <Select value={createData.visibility} onValueChange={(value) => setCreateData(prev => ({ ...prev, visibility: value }))}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="private">
                    <div className="flex items-center">
                      <Lock className="w-4 h-4 mr-2" />
                      Private - Only invited members
                    </div>
                  </SelectItem>
                  <SelectItem value="organization">
                    <div className="flex items-center">
                      <Building2 className="w-4 h-4 mr-2" />
                      Organization - All organization members
                    </div>
                  </SelectItem>
                  <SelectItem value="public">
                    <div className="flex items-center">
                      <Globe className="w-4 h-4 mr-2" />
                      Public - Anyone with the link
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowCreateForm(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateWorkspace} disabled={creating || !createData.name.trim()}>
                {creating ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Creating...
                  </>
                ) : (
                  <>
                    <FolderPlus className="w-4 h-4 mr-2" />
                    Create Workspace
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Workspace List */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Your Workspaces</CardTitle>
              <CardDescription>
                {workspaces.length} workspace{workspaces.length !== 1 ? 's' : ''} available
              </CardDescription>
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search workspaces..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 w-64"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredWorkspaces.map((workspace) => (
              <Card key={workspace.id} className="hover:shadow-md transition-shadow cursor-pointer">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-2">
                      <Folder className="w-5 h-5 text-blue-600" />
                      <CardTitle className="text-lg">{workspace.name}</CardTitle>
                    </div>
                    <Button variant="ghost" size="sm">
                      <MoreHorizontal className="w-4 h-4" />
                    </Button>
                  </div>
                  {workspace.description && (
                    <CardDescription className="text-sm">
                      {workspace.description}
                    </CardDescription>
                  )}
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex items-center justify-between">
                    <Badge variant={getVisibilityBadgeVariant(workspace.visibility)} className="text-xs">
                      {getVisibilityIcon(workspace.visibility)}
                      <span className="ml-1 capitalize">{workspace.visibility}</span>
                    </Badge>
                    <div className="flex items-center space-x-2">
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleViewMembers(workspace.id)}
                      >
                        <Users className="w-4 h-4 mr-1" />
                        Members
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleWorkspaceSettings(workspace.id)}
                      >
                        <Settings className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-gray-500">
                    Created {new Date(workspace.created_at).toLocaleDateString()}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          {filteredWorkspaces.length === 0 && (
            <div className="text-center py-8">
              <Folder className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No workspaces found</h3>
              <p className="text-gray-600">
                {searchTerm 
                  ? 'Try adjusting your search criteria'
                  : 'Create your first workspace to get started'
                }
              </p>
              {!searchTerm && (
                <Button className="mt-4" onClick={() => setShowCreateForm(true)}>
                  <FolderPlus className="w-4 h-4 mr-2" />
                  Create Workspace
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Members Management Modal */}
      {showMembersModal && selectedWorkspace && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Workspace Members</h3>
              <Button variant="ghost" onClick={() => setShowMembersModal(false)}>
                ×
              </Button>
            </div>
            
            {loadingMembers ? (
              <div className="flex items-center justify-center p-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-600">
                    {members.length} member{members.length !== 1 ? 's' : ''}
                  </p>
                  <Button size="sm">
                    <UserPlus className="w-4 h-4 mr-2" />
                    Add Member
                  </Button>
                </div>
                
                <div className="space-y-2">
                  {members.map((member) => (
                    <div key={member.user_id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                          <span className="text-sm font-medium text-blue-600">
                            {member.full_name?.charAt(0) || member.email.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <div>
                          <p className="font-medium">{member.full_name || member.email}</p>
                          <p className="text-sm text-gray-500">{member.email}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline" className="capitalize">
                          {member.role}
                        </Badge>
                        <Button variant="ghost" size="sm">
                          <MoreHorizontal className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
                
                {members.length === 0 && (
                  <div className="text-center py-8">
                    <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No members yet</h3>
                    <p className="text-gray-600">Add members to start collaborating</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Workspace Settings Modal */}
      {showSettingsModal && selectedWorkspace && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-lg">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Workspace Settings</h3>
              <Button variant="ghost" onClick={() => setShowSettingsModal(false)}>
                ×
              </Button>
            </div>
            
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Workspace settings and configuration options will be available here.
              </p>
              
              <div className="flex justify-end">
                <Button variant="outline" onClick={() => setShowSettingsModal(false)}>
                  Close
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}