"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Play, Settings, Database, Target, Cpu, CheckCircle, AlertCircle } from 'lucide-react';
import { api } from '@/lib/api';

interface ModelTemplate {
  name: string;
  description: string;
  recommended_algorithms: string[];
  default_parameters: {
    test_size: number;
    validation_strategy: string;
    scoring_metric: string;
  };
  preprocessing: string[];
  evaluation_metrics: string[];
}

interface Algorithm {
  name: string;
  supports_classification: boolean;
  supports_regression: boolean;
  default_params: Record<string, any>;
  tunable_params: string[];
}

interface Dataset {
  id: string;
  name: string;
  columns: string[];
  row_count: number;
}

interface ModelFactoryInterfaceProps {
  onModelCreated?: (model: any) => void;
}

export function ModelFactoryInterface({ onModelCreated }: ModelFactoryInterfaceProps) {
  const [templates, setTemplates] = useState<Record<string, ModelTemplate>>({});
  const [algorithms, setAlgorithms] = useState<Record<string, Algorithm>>({});
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form state
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [modelName, setModelName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [validationStrategy, setValidationStrategy] = useState('train_test_split');
  const [hyperparameterTuning, setHyperparameterTuning] = useState(false);
  const [customParams, setCustomParams] = useState<Record<string, any>>({});

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load templates, algorithms, and datasets in parallel
      const [templatesRes, algorithmsRes, datasetsRes] = await Promise.all([
        api.get('/model-factory/templates'),
        api.get('/model-factory/algorithms'),
        api.get('/datasets')
      ]);

      setTemplates(templatesRes.data.templates);
      setAlgorithms(algorithmsRes.data.algorithms);
      setDatasets(datasetsRes.data.datasets || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const handleTemplateChange = (templateKey: string) => {
    setSelectedTemplate(templateKey);
    const template = templates[templateKey];
    if (template) {
      // Set recommended algorithm
      if (template.recommended_algorithms.length > 0) {
        setSelectedAlgorithm(template.recommended_algorithms[0]);
      }
      // Set default validation strategy
      setValidationStrategy(template.default_parameters.validation_strategy);
    }
  };

  const handleDatasetChange = (datasetId: string) => {
    setSelectedDataset(datasetId);
    const dataset = datasets.find(d => d.id === datasetId);
    if (dataset) {
      // Reset column selections
      setTargetColumn('');
      setFeatureColumns([]);
    }
  };

  const handleFeatureColumnToggle = (column: string, checked: boolean) => {
    if (checked) {
      setFeatureColumns(prev => [...prev, column]);
    } else {
      setFeatureColumns(prev => prev.filter(col => col !== column));
    }
  };

  const handleCreateModel = async () => {
    try {
      setCreating(true);
      setError(null);
      setSuccess(null);

      if (!modelName || !selectedDataset || !selectedAlgorithm || !targetColumn) {
        throw new Error('Please fill in all required fields');
      }

      const requestData = {
        model_name: modelName,
        dataset_id: selectedDataset,
        algorithm: selectedAlgorithm,
        template: selectedTemplate || null,
        target_column: targetColumn,
        feature_columns: featureColumns.length > 0 ? featureColumns : null,
        custom_params: customParams,
        validation_strategy: validationStrategy,
        hyperparameter_tuning: hyperparameterTuning
      };

      const response = await api.post('/model-factory/models', requestData);
      
      setSuccess(`Model "${modelName}" created successfully!`);
      onModelCreated?.(response.data);
      
      // Reset form
      resetForm();
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create model');
    } finally {
      setCreating(false);
    }
  };

  const resetForm = () => {
    setModelName('');
    setSelectedTemplate('');
    setSelectedAlgorithm('');
    setSelectedDataset('');
    setTargetColumn('');
    setFeatureColumns([]);
    setValidationStrategy('train_test_split');
    setHyperparameterTuning(false);
    setCustomParams({});
  };

  const selectedDatasetObj = datasets.find(d => d.id === selectedDataset);
  const availableColumns = selectedDatasetObj?.columns || [];
  const availableFeatureColumns = availableColumns.filter(col => col !== targetColumn);

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
          <h2 className="text-2xl font-bold">ScrollModelFactory</h2>
          <p className="text-muted-foreground">Create custom ML models with UI-driven configuration</p>
        </div>
        <Button onClick={resetForm} variant="outline">
          Reset Form
        </Button>
      </div>

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

      <Tabs defaultValue="configure" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="configure">Configure Model</TabsTrigger>
          <TabsTrigger value="parameters">Parameters</TabsTrigger>
          <TabsTrigger value="review">Review & Create</TabsTrigger>
        </TabsList>

        <TabsContent value="configure" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Model Template Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Model Template
                </CardTitle>
                <CardDescription>
                  Choose a pre-configured template for common use cases
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Select value={selectedTemplate} onValueChange={handleTemplateChange}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a template (optional)" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(templates).map(([key, template]) => (
                      <SelectItem key={key} value={key}>
                        {template.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {selectedTemplate && templates[selectedTemplate] && (
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      {templates[selectedTemplate].description}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {templates[selectedTemplate].recommended_algorithms.map(alg => (
                        <Badge key={alg} variant="secondary" className="text-xs">
                          {alg.replace('_', ' ')}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Algorithm Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="h-5 w-5" />
                  Algorithm
                </CardTitle>
                <CardDescription>
                  Select the machine learning algorithm
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Select value={selectedAlgorithm} onValueChange={setSelectedAlgorithm}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select an algorithm" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(algorithms).map(([key, algorithm]) => (
                      <SelectItem key={key} value={key}>
                        {algorithm.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {selectedAlgorithm && algorithms[selectedAlgorithm] && (
                  <div className="space-y-2">
                    <div className="flex gap-2">
                      {algorithms[selectedAlgorithm].supports_classification && (
                        <Badge variant="outline">Classification</Badge>
                      )}
                      {algorithms[selectedAlgorithm].supports_regression && (
                        <Badge variant="outline">Regression</Badge>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Dataset Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Dataset
                </CardTitle>
                <CardDescription>
                  Choose the dataset for training
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Select value={selectedDataset} onValueChange={handleDatasetChange}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.map(dataset => (
                      <SelectItem key={dataset.id} value={dataset.id}>
                        {dataset.name} ({dataset.row_count} rows)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {selectedDatasetObj && (
                  <div className="text-sm text-muted-foreground">
                    {selectedDatasetObj.columns.length} columns available
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Model Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Model Configuration</CardTitle>
                <CardDescription>
                  Basic model settings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="model-name">Model Name</Label>
                  <Input
                    id="model-name"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    placeholder="Enter model name"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="validation-strategy">Validation Strategy</Label>
                  <Select value={validationStrategy} onValueChange={setValidationStrategy}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="train_test_split">Train/Test Split</SelectItem>
                      <SelectItem value="cross_validation">Cross Validation</SelectItem>
                      <SelectItem value="stratified_split">Stratified Split</SelectItem>
                      <SelectItem value="time_series_split">Time Series Split</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="hyperparameter-tuning"
                    checked={hyperparameterTuning}
                    onCheckedChange={(checked: boolean) => setHyperparameterTuning(checked)}
                  />
                  <Label htmlFor="hyperparameter-tuning">
                    Enable hyperparameter tuning
                  </Label>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Column Selection */}
          {selectedDatasetObj && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Column Selection
                </CardTitle>
                <CardDescription>
                  Select target and feature columns
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="target-column">Target Column</Label>
                    <Select value={targetColumn} onValueChange={setTargetColumn}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select target column" />
                      </SelectTrigger>
                      <SelectContent>
                        {availableColumns.map(column => (
                          <SelectItem key={column} value={column}>
                            {column}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Feature Columns (optional)</Label>
                    <div className="max-h-40 overflow-y-auto space-y-2 border rounded p-2">
                      {availableFeatureColumns.map(column => (
                        <div key={column} className="flex items-center space-x-2">
                          <Checkbox
                            id={`feature-${column}`}
                            checked={featureColumns.includes(column)}
                            onCheckedChange={(checked: boolean) => 
                              handleFeatureColumnToggle(column, checked)
                            }
                          />
                          <Label htmlFor={`feature-${column}`} className="text-sm">
                            {column}
                          </Label>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Leave empty to use all columns except target
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="parameters" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Custom Parameters</CardTitle>
              <CardDescription>
                Fine-tune algorithm parameters (optional)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {selectedAlgorithm && algorithms[selectedAlgorithm] ? (
                <div className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    Tunable parameters for {algorithms[selectedAlgorithm].name}:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {algorithms[selectedAlgorithm].tunable_params.map(param => (
                      <Badge key={param} variant="outline">
                        {param}
                      </Badge>
                    ))}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Custom parameter configuration will be available in a future update.
                    For now, default parameters will be used with optional hyperparameter tuning.
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">Select an algorithm to see available parameters</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="review" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Review Configuration</CardTitle>
              <CardDescription>
                Review your model configuration before creating
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="font-medium">Model Name</Label>
                  <p className="text-sm">{modelName || 'Not specified'}</p>
                </div>
                <div className="space-y-2">
                  <Label className="font-medium">Template</Label>
                  <p className="text-sm">
                    {selectedTemplate ? templates[selectedTemplate]?.name : 'None selected'}
                  </p>
                </div>
                <div className="space-y-2">
                  <Label className="font-medium">Algorithm</Label>
                  <p className="text-sm">
                    {selectedAlgorithm ? algorithms[selectedAlgorithm]?.name : 'Not selected'}
                  </p>
                </div>
                <div className="space-y-2">
                  <Label className="font-medium">Dataset</Label>
                  <p className="text-sm">
                    {selectedDatasetObj?.name || 'Not selected'}
                  </p>
                </div>
                <div className="space-y-2">
                  <Label className="font-medium">Target Column</Label>
                  <p className="text-sm">{targetColumn || 'Not selected'}</p>
                </div>
                <div className="space-y-2">
                  <Label className="font-medium">Feature Columns</Label>
                  <p className="text-sm">
                    {featureColumns.length > 0 
                      ? `${featureColumns.length} selected` 
                      : 'All except target'
                    }
                  </p>
                </div>
                <div className="space-y-2">
                  <Label className="font-medium">Validation Strategy</Label>
                  <p className="text-sm">{validationStrategy.replace('_', ' ')}</p>
                </div>
                <div className="space-y-2">
                  <Label className="font-medium">Hyperparameter Tuning</Label>
                  <p className="text-sm">{hyperparameterTuning ? 'Enabled' : 'Disabled'}</p>
                </div>
              </div>

              <div className="pt-4">
                <Button 
                  onClick={handleCreateModel} 
                  disabled={creating || !modelName || !selectedAlgorithm || !selectedDataset || !targetColumn}
                  className="w-full"
                >
                  {creating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating Model...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Create Model
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}