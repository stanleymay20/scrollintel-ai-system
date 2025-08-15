"use client";

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent } from '@/components/ui/card';
import { Info } from 'lucide-react';

interface APIKey {
  id: string;
  name: string;
  description?: string;
  key_display: string;
  permissions: string[];
  rate_limit_per_minute: number;
  rate_limit_per_hour: number;
  rate_limit_per_day: number;
  quota_requests_per_month?: number;
  is_active: boolean;
  last_used?: string;
  expires_at?: string;
  created_at: string;
  updated_at: string;
}

interface APIKeyEditDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  apiKey: APIKey;
  onUpdateKey: (keyId: string, updateData: any) => void;
}

const PERMISSION_OPTIONS = [
  { id: 'agents:read', label: 'Read Agents', description: 'View agent information and capabilities' },
  { id: 'agents:execute', label: 'Execute Agents', description: 'Run agent tasks and get responses' },
  { id: 'data:read', label: 'Read Data', description: 'Access uploaded datasets and files' },
  { id: 'data:write', label: 'Write Data', description: 'Upload and modify datasets' },
  { id: 'models:read', label: 'Read Models', description: 'View ML models and their metadata' },
  { id: 'models:execute', label: 'Execute Models', description: 'Run predictions and inference' },
  { id: 'dashboards:read', label: 'Read Dashboards', description: 'View dashboard configurations' },
  { id: 'dashboards:write', label: 'Write Dashboards', description: 'Create and modify dashboards' },
];

export function APIKeyEditDialog({ open, onOpenChange, apiKey, onUpdateKey }: APIKeyEditDialogProps) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    permissions: [] as string[],
    rate_limit_per_minute: 60,
    rate_limit_per_hour: 1000,
    rate_limit_per_day: 10000,
    is_active: true,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (apiKey) {
      setFormData({
        name: apiKey.name,
        description: apiKey.description || '',
        permissions: apiKey.permissions,
        rate_limit_per_minute: apiKey.rate_limit_per_minute,
        rate_limit_per_hour: apiKey.rate_limit_per_hour,
        rate_limit_per_day: apiKey.rate_limit_per_day,
        is_active: apiKey.is_active,
      });
    }
  }, [apiKey]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      setError('API key name is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await onUpdateKey(apiKey.id, formData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update API key');
    } finally {
      setLoading(false);
    }
  };

  const handlePermissionToggle = (permissionId: string) => {
    setFormData(prev => ({
      ...prev,
      permissions: prev.permissions.includes(permissionId)
        ? prev.permissions.filter(p => p !== permissionId)
        : [...prev.permissions, permissionId]
    }));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit API Key</DialogTitle>
          <DialogDescription>
            Update the settings for your API key. Note that some changes may take a few minutes to take effect.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="basic">Basic Info</TabsTrigger>
              <TabsTrigger value="permissions">Permissions</TabsTrigger>
              <TabsTrigger value="limits">Rate Limits</TabsTrigger>
            </TabsList>

            <TabsContent value="basic" className="space-y-4">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="name">API Key Name *</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="e.g., Production API Key"
                    required
                  />
                </div>

                <div>
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={formData.description}
                    onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    placeholder="Optional description of what this key will be used for"
                    rows={3}
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    id="is_active"
                    checked={formData.is_active}
                    onCheckedChange={(checked) => setFormData(prev => ({ ...prev, is_active: checked }))}
                  />
                  <Label htmlFor="is_active">Active</Label>
                  <p className="text-sm text-gray-600">
                    Inactive keys cannot be used for API requests
                  </p>
                </div>

                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    <div className="space-y-2">
                      <p><strong>Key ID:</strong> {apiKey.id}</p>
                      <p><strong>Created:</strong> {new Date(apiKey.created_at).toLocaleDateString()}</p>
                      <p><strong>Last Used:</strong> {apiKey.last_used ? new Date(apiKey.last_used).toLocaleDateString() : 'Never'}</p>
                      {apiKey.expires_at && (
                        <p><strong>Expires:</strong> {new Date(apiKey.expires_at).toLocaleDateString()}</p>
                      )}
                    </div>
                  </AlertDescription>
                </Alert>
              </div>
            </TabsContent>

            <TabsContent value="permissions" className="space-y-4">
              <div>
                <Label className="text-base font-medium">API Permissions</Label>
                <p className="text-sm text-gray-600 mb-4">
                  Select which API endpoints this key can access
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {PERMISSION_OPTIONS.map((permission) => (
                    <Card
                      key={permission.id}
                      className={`cursor-pointer transition-colors ${
                        formData.permissions.includes(permission.id)
                          ? 'border-blue-500 bg-blue-50'
                          : 'hover:border-gray-300'
                      }`}
                      onClick={() => handlePermissionToggle(permission.id)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start space-x-3">
                          <input
                            type="checkbox"
                            checked={formData.permissions.includes(permission.id)}
                            onChange={() => handlePermissionToggle(permission.id)}
                            className="mt-1"
                          />
                          <div className="flex-1">
                            <h4 className="font-medium">{permission.label}</h4>
                            <p className="text-sm text-gray-600">{permission.description}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                {formData.permissions.length === 0 && (
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                      No permissions selected. This API key will have very limited access.
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </TabsContent>

            <TabsContent value="limits" className="space-y-4">
              <div>
                <Label className="text-base font-medium">Rate Limits</Label>
                <p className="text-sm text-gray-600 mb-4">
                  Configure the rate limits for this API key
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label htmlFor="rate_minute">Requests Per Minute</Label>
                    <Input
                      id="rate_minute"
                      type="number"
                      min="1"
                      max="10000"
                      value={formData.rate_limit_per_minute}
                      onChange={(e) => setFormData(prev => ({ 
                        ...prev, 
                        rate_limit_per_minute: parseInt(e.target.value) || 1 
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="rate_hour">Requests Per Hour</Label>
                    <Input
                      id="rate_hour"
                      type="number"
                      min="1"
                      max="100000"
                      value={formData.rate_limit_per_hour}
                      onChange={(e) => setFormData(prev => ({ 
                        ...prev, 
                        rate_limit_per_hour: parseInt(e.target.value) || 1 
                      }))}
                    />
                  </div>
                  <div>
                    <Label htmlFor="rate_day">Requests Per Day</Label>
                    <Input
                      id="rate_day"
                      type="number"
                      min="1"
                      max="1000000"
                      value={formData.rate_limit_per_day}
                      onChange={(e) => setFormData(prev => ({ 
                        ...prev, 
                        rate_limit_per_day: parseInt(e.target.value) || 1 
                      }))}
                    />
                  </div>
                </div>

                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    Rate limit changes take effect immediately. Be careful not to set limits too low 
                    for production applications.
                  </AlertDescription>
                </Alert>
              </div>
            </TabsContent>
          </Tabs>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={loading}>
              {loading ? 'Updating...' : 'Update API Key'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}