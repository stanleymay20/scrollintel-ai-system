"use client";

import React, { useState } from 'react';
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
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  Info, 
  Plus, 
  X,
  Shield,
  Zap,
  Clock,
  DollarSign
} from 'lucide-react';

interface APIKeyCreateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreateKey: (keyData: any) => void;
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

const RATE_LIMIT_PRESETS = {
  development: {
    name: 'Development',
    description: 'For testing and development',
    per_minute: 10,
    per_hour: 100,
    per_day: 1000,
    icon: <Shield className="h-4 w-4" />
  },
  production: {
    name: 'Production',
    description: 'For production applications',
    per_minute: 60,
    per_hour: 1000,
    per_day: 10000,
    icon: <Zap className="h-4 w-4" />
  },
  high_volume: {
    name: 'High Volume',
    description: 'For high-traffic applications',
    per_minute: 200,
    per_hour: 5000,
    per_day: 50000,
    icon: <Clock className="h-4 w-4" />
  },
  custom: {
    name: 'Custom',
    description: 'Set your own limits',
    per_minute: 60,
    per_hour: 1000,
    per_day: 10000,
    icon: <DollarSign className="h-4 w-4" />
  }
};

export function APIKeyCreateDialog({ open, onOpenChange, onCreateKey }: APIKeyCreateDialogProps) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    permissions: [] as string[],
    rate_limit_per_minute: 60,
    rate_limit_per_hour: 1000,
    rate_limit_per_day: 10000,
    quota_requests_per_month: null as number | null,
    expires_in_days: null as number | null,
  });
  const [selectedPreset, setSelectedPreset] = useState<keyof typeof RATE_LIMIT_PRESETS>('production');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      setError('API key name is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await onCreateKey(formData);
      // Reset form
      setFormData({
        name: '',
        description: '',
        permissions: [],
        rate_limit_per_minute: 60,
        rate_limit_per_hour: 1000,
        rate_limit_per_day: 10000,
        quota_requests_per_month: null,
        expires_in_days: null,
      });
      setSelectedPreset('production');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create API key');
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

  const handlePresetChange = (preset: keyof typeof RATE_LIMIT_PRESETS) => {
    setSelectedPreset(preset);
    if (preset !== 'custom') {
      const presetConfig = RATE_LIMIT_PRESETS[preset];
      setFormData(prev => ({
        ...prev,
        rate_limit_per_minute: presetConfig.per_minute,
        rate_limit_per_hour: presetConfig.per_hour,
        rate_limit_per_day: presetConfig.per_day,
      }));
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create New API Key</DialogTitle>
          <DialogDescription>
            Create a new API key for programmatic access to ScrollIntel services.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic">Basic Info</TabsTrigger>
              <TabsTrigger value="permissions">Permissions</TabsTrigger>
              <TabsTrigger value="limits">Rate Limits</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
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
                  <p className="text-sm text-gray-600 mt-1">
                    A descriptive name to help you identify this key
                  </p>
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
                <Label className="text-base font-medium">Rate Limit Preset</Label>
                <p className="text-sm text-gray-600 mb-4">
                  Choose a preset or customize your rate limits
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  {Object.entries(RATE_LIMIT_PRESETS).map(([key, preset]) => (
                    <Card
                      key={key}
                      className={`cursor-pointer transition-colors ${
                        selectedPreset === key
                          ? 'border-blue-500 bg-blue-50'
                          : 'hover:border-gray-300'
                      }`}
                      onClick={() => handlePresetChange(key as keyof typeof RATE_LIMIT_PRESETS)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start space-x-3">
                          <div className="mt-1">{preset.icon}</div>
                          <div className="flex-1">
                            <h4 className="font-medium flex items-center">
                              {preset.name}
                              {selectedPreset === key && (
                                <Badge variant="default" className="ml-2">Selected</Badge>
                              )}
                            </h4>
                            <p className="text-sm text-gray-600 mb-2">{preset.description}</p>
                            {key !== 'custom' && (
                              <div className="text-xs text-gray-500">
                                {preset.per_minute}/min • {preset.per_hour}/hour • {preset.per_day}/day
                              </div>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                {selectedPreset === 'custom' && (
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="rate_minute">Per Minute</Label>
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
                      <Label htmlFor="rate_hour">Per Hour</Label>
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
                      <Label htmlFor="rate_day">Per Day</Label>
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
                )}
              </div>
            </TabsContent>

            <TabsContent value="advanced" className="space-y-4">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="quota">Monthly Request Quota</Label>
                  <Input
                    id="quota"
                    type="number"
                    min="1"
                    value={formData.quota_requests_per_month || ''}
                    onChange={(e) => setFormData(prev => ({ 
                      ...prev, 
                      quota_requests_per_month: e.target.value ? parseInt(e.target.value) : null 
                    }))}
                    placeholder="Leave empty for unlimited"
                  />
                  <p className="text-sm text-gray-600 mt-1">
                    Maximum number of requests per month (optional)
                  </p>
                </div>

                <div>
                  <Label htmlFor="expires">Expires In (Days)</Label>
                  <Input
                    id="expires"
                    type="number"
                    min="1"
                    max="365"
                    value={formData.expires_in_days || ''}
                    onChange={(e) => setFormData(prev => ({ 
                      ...prev, 
                      expires_in_days: e.target.value ? parseInt(e.target.value) : null 
                    }))}
                    placeholder="Leave empty for no expiration"
                  />
                  <p className="text-sm text-gray-600 mt-1">
                    Number of days until this key expires (optional)
                  </p>
                </div>

                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Security Note:</strong> API keys provide full access to your account 
                    within the granted permissions. Store them securely and never share them publicly.
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
              {loading ? 'Creating...' : 'Create API Key'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}