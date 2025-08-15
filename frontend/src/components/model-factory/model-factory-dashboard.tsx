"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Loader2, 
  Plus, 
  Settings, 
  CheckCircle, 
  Rocket, 
  Database,
  TrendingUp,
  Clock,
  Target
} from 'lucide-react';
import { ModelFactoryInterface } from './model-factory-interface';
import { ModelValidation } from './model-validation';
import { ModelDeployment } from './model-deployment';
import { api } from '@/lib/api';

interface CustomModel {
  id: string;
  name: string;
  algorithm: string;
  dataset_id: string;
  metrics: Record<string, number>;
  is_deployed: boolean;
  created_at: string;
  training_duration_seconds: number;
}

export function ModelFactoryDashboard() {
  const [models, setModels] = useState<CustomModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<CustomModel | null>(null);
  const [activeTab, setActiveTab] = useState('create');

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await api.get('/models');
      setModels(response.data.models || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const handleModelCreated = (model: any) => {
    // Refresh models list
    loadModels();
    // Switch to models tab
    setActiveTab('models');
  };

  const handleModelSelect = (model: CustomModel, action: 'validate' | 'deploy') => {
    setSelectedModel(model);
    setActiveTab(action);
  };

  const formatMetrics = (metrics: Record<string, number>) => {
    return Object.entries(metrics)
      .slice(0, 3) // Show only first 3 metrics
      .map(([key, value]) => (
        <Badge key={key} variant="secondary" className="text-xs">
          {key}: {typeof value === 'number' ? value.toFixed(3) : value}
        </Badge>
      ));
  };

  const getModelStats = () => {
    const totalModels = models.length;
    const deployedModels = models.filter(m => m.is_deployed).length;
    const avgTrainingTime = models.length > 0 
      ? models.reduce((sum, m) => sum + (m.training_duration_seconds || 0), 0) / models.length
      : 0;

    return { totalModels, deployedModels, avgTrainingTime };
  };

  const stats = getModelStats();

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading model factory...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">ScrollModelFactory</h1>
          <p className="text-muted-foreground">
            Create, validate, and deploy custom ML models with UI-driven configuration
          </p>
        </div>
        <Button onClick={loadModels} variant="outline">
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="flex items-center p-6">
            <Database className="h-8 w-8 text-blue-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-muted-foreground">Total Models</p>
              <p className="text-2xl font-bold">{stats.totalModels}</p>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="flex items-center p-6">
            <Rocket className="h-8 w-8 text-green-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-muted-foreground">Deployed</p>
              <p className="text-2xl font-bold">{stats.deployedModels}</p>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="flex items-center p-6">
            <Clock className="h-8 w-8 text-orange-500" />
            <div className="ml-4">
              <p className="text-sm font-medium text-muted-foreground">Avg Training Time</p>
              <p className="text-2xl font-bold">{stats.avgTrainingTime.toFixed(1)}s</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="create">Create Model</TabsTrigger>
          <TabsTrigger value="models">My Models</TabsTrigger>
          <TabsTrigger value="validate">Validate</TabsTrigger>
          <TabsTrigger value="deploy">Deploy</TabsTrigger>
        </TabsList>

        <TabsContent value="create" className="space-y-6">
          <ModelFactoryInterface onModelCreated={handleModelCreated} />
        </TabsContent>

        <TabsContent value="models" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Custom Models
              </CardTitle>
              <CardDescription>
                Manage your custom trained models
              </CardDescription>
            </CardHeader>
            <CardContent>
              {models.length === 0 ? (
                <div className="text-center py-8">
                  <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">No custom models created yet</p>
                  <Button 
                    onClick={() => setActiveTab('create')} 
                    className="mt-4"
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Create Your First Model
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  {models.map((model) => (
                    <Card key={model.id} className="border">
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <h4 className="font-semibold">{model.name}</h4>
                              {model.is_deployed && (
                                <Badge variant="default">Deployed</Badge>
                              )}
                            </div>
                            
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <Target className="h-4 w-4" />
                              <span>{model.algorithm.replace('_', ' ')}</span>
                              <span>•</span>
                              <Clock className="h-4 w-4" />
                              <span>{model.training_duration_seconds?.toFixed(1)}s</span>
                            </div>
                            
                            <div className="flex flex-wrap gap-1">
                              {formatMetrics(model.metrics)}
                            </div>
                            
                            <p className="text-xs text-muted-foreground">
                              Created {new Date(model.created_at).toLocaleDateString()}
                            </p>
                          </div>
                          
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleModelSelect(model, 'validate')}
                            >
                              <CheckCircle className="mr-1 h-3 w-3" />
                              Validate
                            </Button>
                            
                            <Button
                              size="sm"
                              variant={model.is_deployed ? "secondary" : "default"}
                              onClick={() => handleModelSelect(model, 'deploy')}
                            >
                              <Rocket className="mr-1 h-3 w-3" />
                              {model.is_deployed ? 'Manage' : 'Deploy'}
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="validate" className="space-y-6">
          {selectedModel ? (
            <ModelValidation
              modelId={selectedModel.id}
              modelName={selectedModel.name}
              onValidationComplete={(result) => {
                console.log('Validation completed:', result);
              }}
            />
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <CheckCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">Select a model from the Models tab to validate</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="deploy" className="space-y-6">
          {selectedModel ? (
            <ModelDeployment
              modelId={selectedModel.id}
              modelName={selectedModel.name}
              onDeploymentComplete={(result) => {
                console.log('Deployment completed:', result);
                // Refresh models to update deployment status
                loadModels();
              }}
            />
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <Rocket className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">Select a model from the Models tab to deploy</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}