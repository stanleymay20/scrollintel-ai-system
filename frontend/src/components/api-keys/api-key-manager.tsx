"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Plus, 
  Key, 
  Eye, 
  EyeOff, 
  Copy, 
  Edit, 
  Trash2, 
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';
import { APIKeyCreateDialog } from './api-key-create-dialog';
import { APIKeyEditDialog } from './api-key-edit-dialog';
import { UsageAnalyticsDashboard } from './usage-analytics-dashboard';

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

interface RateLimitStatus {
  allowed: boolean;
  minute: {
    limit: number;
    used: number;
    remaining: number;
    reset_at: string;
  };
  hour: {
    limit: number;
    used: number;
    remaining: number;
    reset_at: string;
  };
  day: {
    limit: number;
    used: number;
    remaining: number;
    reset_at: string;
  };
}

export function APIKeyManager() {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [editingKey, setEditingKey] = useState<APIKey | null>(null);
  const [selectedKeyId, setSelectedKeyId] = useState<string | null>(null);
  const [rateLimitStatus, setRateLimitStatus] = useState<Record<string, RateLimitStatus>>({});
  const [newKeyData, setNewKeyData] = useState<{ key: string; api_key: APIKey } | null>(null);

  useEffect(() => {
    fetchAPIKeys();
  }, []);

  const fetchAPIKeys = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/keys/', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch API keys');
      }

      const keys = await response.json();
      setApiKeys(keys);
      
      // Fetch rate limit status for each key
      for (const key of keys) {
        fetchRateLimitStatus(key.id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const fetchRateLimitStatus = async (keyId: string) => {
    try {
      const response = await fetch(`/api/v1/keys/${keyId}/rate-limit`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });

      if (response.ok) {
        const status = await response.json();
        setRateLimitStatus(prev => ({ ...prev, [keyId]: status }));
      }
    } catch (err) {
      console.error('Failed to fetch rate limit status:', err);
    }
  };

  const handleCreateKey = async (keyData: any) => {
    try {
      const response = await fetch('/api/v1/keys/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(keyData),
      });

      if (!response.ok) {
        throw new Error('Failed to create API key');
      }

      const result = await response.json();
      setNewKeyData(result);
      setApiKeys(prev => [...prev, result.api_key]);
      setShowCreateDialog(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create API key');
    }
  };

  const handleUpdateKey = async (keyId: string, updateData: any) => {
    try {
      const response = await fetch(`/api/v1/keys/${keyId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(updateData),
      });

      if (!response.ok) {
        throw new Error('Failed to update API key');
      }

      const updatedKey = await response.json();
      setApiKeys(prev => prev.map(key => key.id === keyId ? updatedKey : key));
      setEditingKey(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update API key');
    }
  };

  const handleDeleteKey = async (keyId: string) => {
    if (!confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`/api/v1/keys/${keyId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to delete API key');
      }

      setApiKeys(prev => prev.filter(key => key.id !== keyId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete API key');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // You could add a toast notification here
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusBadge = (key: APIKey) => {
    if (!key.is_active) {
      return <Badge variant="secondary">Inactive</Badge>;
    }
    
    if (key.expires_at && new Date(key.expires_at) < new Date()) {
      return <Badge variant="destructive">Expired</Badge>;
    }

    const status = rateLimitStatus[key.id];
    if (status && !status.allowed) {
      return <Badge variant="destructive">Rate Limited</Badge>;
    }

    return <Badge variant="default">Active</Badge>;
  };

  const getRateLimitProgress = (used: number, limit: number) => {
    const percentage = (used / limit) * 100;
    return Math.min(percentage, 100);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {newKeyData && (
        <Alert>
          <Key className="h-4 w-4" />
          <AlertDescription>
            <div className="space-y-2">
              <p className="font-medium">API Key Created Successfully!</p>
              <p className="text-sm">
                Please copy your API key now. You won't be able to see it again.
              </p>
              <div className="flex items-center space-x-2 p-2 bg-gray-100 rounded font-mono text-sm">
                <span className="flex-1">{newKeyData.key}</span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => copyToClipboard(newKeyData.key)}
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
              <Button
                size="sm"
                onClick={() => setNewKeyData(null)}
              >
                I've saved my key
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}

      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">API Key Management</h2>
          <p className="text-gray-600">
            Manage your API keys for programmatic access to ScrollIntel
          </p>
        </div>
        <Button onClick={() => setShowCreateDialog(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Create API Key
        </Button>
      </div>

      <Tabs defaultValue="keys" className="space-y-4">
        <TabsList>
          <TabsTrigger value="keys">API Keys</TabsTrigger>
          <TabsTrigger value="analytics">Usage Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="keys" className="space-y-4">
          {apiKeys.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Key className="h-12 w-12 text-gray-400 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  No API Keys
                </h3>
                <p className="text-gray-600 text-center mb-4">
                  Create your first API key to start using the ScrollIntel API
                </p>
                <Button onClick={() => setShowCreateDialog(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Create API Key
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4">
              {apiKeys.map((key) => (
                <Card key={key.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center space-x-2">
                          <span>{key.name}</span>
                          {getStatusBadge(key)}
                        </CardTitle>
                        <CardDescription>{key.description}</CardDescription>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setSelectedKeyId(key.id)}
                        >
                          <Activity className="h-4 w-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setEditingKey(key)}
                        >
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleDeleteKey(key.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <label className="text-sm font-medium text-gray-700">
                          API Key
                        </label>
                        <div className="flex items-center space-x-2 mt-1">
                          <code className="flex-1 p-2 bg-gray-100 rounded text-sm font-mono">
                            {key.key_display}
                          </code>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => copyToClipboard(key.key_display)}
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <label className="text-sm font-medium text-gray-700">
                            Created
                          </label>
                          <p className="text-sm text-gray-600">
                            {formatDate(key.created_at)}
                          </p>
                        </div>
                        <div>
                          <label className="text-sm font-medium text-gray-700">
                            Last Used
                          </label>
                          <p className="text-sm text-gray-600">
                            {key.last_used ? formatDate(key.last_used) : 'Never'}
                          </p>
                        </div>
                        <div>
                          <label className="text-sm font-medium text-gray-700">
                            Expires
                          </label>
                          <p className="text-sm text-gray-600">
                            {key.expires_at ? formatDate(key.expires_at) : 'Never'}
                          </p>
                        </div>
                        <div>
                          <label className="text-sm font-medium text-gray-700">
                            Monthly Quota
                          </label>
                          <p className="text-sm text-gray-600">
                            {key.quota_requests_per_month?.toLocaleString() || 'Unlimited'}
                          </p>
                        </div>
                      </div>

                      {rateLimitStatus[key.id] && (
                        <div>
                          <label className="text-sm font-medium text-gray-700 mb-2 block">
                            Rate Limits
                          </label>
                          <div className="grid grid-cols-3 gap-4">
                            <div>
                              <div className="flex justify-between text-xs text-gray-600 mb-1">
                                <span>Per Minute</span>
                                <span>
                                  {rateLimitStatus[key.id].minute.used} / {rateLimitStatus[key.id].minute.limit}
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-blue-600 h-2 rounded-full"
                                  style={{
                                    width: `${getRateLimitProgress(
                                      rateLimitStatus[key.id].minute.used,
                                      rateLimitStatus[key.id].minute.limit
                                    )}%`
                                  }}
                                ></div>
                              </div>
                            </div>
                            <div>
                              <div className="flex justify-between text-xs text-gray-600 mb-1">
                                <span>Per Hour</span>
                                <span>
                                  {rateLimitStatus[key.id].hour.used} / {rateLimitStatus[key.id].hour.limit}
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-blue-600 h-2 rounded-full"
                                  style={{
                                    width: `${getRateLimitProgress(
                                      rateLimitStatus[key.id].hour.used,
                                      rateLimitStatus[key.id].hour.limit
                                    )}%`
                                  }}
                                ></div>
                              </div>
                            </div>
                            <div>
                              <div className="flex justify-between text-xs text-gray-600 mb-1">
                                <span>Per Day</span>
                                <span>
                                  {rateLimitStatus[key.id].day.used} / {rateLimitStatus[key.id].day.limit}
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-blue-600 h-2 rounded-full"
                                  style={{
                                    width: `${getRateLimitProgress(
                                      rateLimitStatus[key.id].day.used,
                                      rateLimitStatus[key.id].day.limit
                                    )}%`
                                  }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="analytics">
          <UsageAnalyticsDashboard selectedKeyId={selectedKeyId} />
        </TabsContent>
      </Tabs>

      <APIKeyCreateDialog
        open={showCreateDialog}
        onOpenChange={setShowCreateDialog}
        onCreateKey={handleCreateKey}
      />

      {editingKey && (
        <APIKeyEditDialog
          open={!!editingKey}
          onOpenChange={(open) => !open && setEditingKey(null)}
          apiKey={editingKey}
          onUpdateKey={handleUpdateKey}
        />
      )}
    </div>
  );
}