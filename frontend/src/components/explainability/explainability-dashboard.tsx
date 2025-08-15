"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  Brain, 
  Eye, 
  Target, 
  AlertTriangle, 
  CheckCircle, 
  Upload,
  Download,
  RefreshCw,
  Info
} from 'lucide-react';

interface ExplainabilityDashboardProps {
  className?: string;
}

interface ExplanationResult {
  status: string;
  shap_values?: number[][];
  base_values?: number[];
  feature_names?: string[];
  visualization?: {
    type: string;
    figure: string;
  };
  explanations?: Array<[string, number]>;
  counterfactuals?: Array<{
    feature: string;
    original_value: number;
    modified_value: number;
    change: number;
  }>;
}

interface BiasResult {
  status: string;
  bias_metrics?: Record<string, {
    group_rates: Record<string, number>;
    demographic_parity_difference: number;
    bias_detected: boolean;
  }>;
  overall_bias_detected?: boolean;
}

const ExplainabilityDashboard: React.FC<ExplainabilityDashboardProps> = ({ className }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [engineStatus, setEngineStatus] = useState<any>(null);
  const [shapResult, setShapResult] = useState<ExplanationResult | null>(null);
  const [limeResult, setLimeResult] = useState<ExplanationResult | null>(null);
  const [featureImportance, setFeatureImportance] = useState<any>(null);
  const [biasResult, setBiasResult] = useState<BiasResult | null>(null);
  const [counterfactuals, setCounterfactuals] = useState<ExplanationResult | null>(null);
  const [attentionResult, setAttentionResult] = useState<any>(null);
  const [transformerSetup, setTransformerSetup] = useState<any>(null);
  const [inputText, setInputText] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchEngineStatus();
  }, []);

  const fetchEngineStatus = async () => {
    try {
      const response = await fetch('/api/explainx/status');
      const data = await response.json();
      setEngineStatus(data);
    } catch (error) {
      console.error('Failed to fetch engine status:', error);
    }
  };

  const handleFileUpload = async (file: File, type: 'model' | 'data') => {
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const endpoint = type === 'model' ? '/api/explainx/upload-model' : '/api/explainx/upload-data';
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      if (result.status === 'success') {
        await fetchEngineStatus();
      }
    } catch (error) {
      console.error(`Failed to upload ${type}:`, error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateShapExplanations = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/shap/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: [[0.1, 0.2, 0.3, 0.4, 0.5]], // Sample data
          explanation_type: 'waterfall'
        }),
      });

      const result = await response.json();
      setShapResult(result);
    } catch (error) {
      console.error('Failed to generate SHAP explanations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateLimeExplanations = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/lime/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instance: [0.1, 0.2, 0.3, 0.4, 0.5], // Sample instance
          num_features: 5
        }),
      });

      const result = await response.json();
      setLimeResult(result);
    } catch (error) {
      console.error('Failed to generate LIME explanations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeFeatureImportance = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/feature-importance?method=shap');
      const result = await response.json();
      setFeatureImportance(result);
    } catch (error) {
      console.error('Failed to analyze feature importance:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const detectBias = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/bias/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          protected_features: ['feature_0', 'feature_1'],
          predictions: [0.1, 0.8, 0.3, 0.9, 0.2],
          labels: [0, 1, 0, 1, 0]
        }),
      });

      const result = await response.json();
      setBiasResult(result);
    } catch (error) {
      console.error('Failed to detect bias:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateCounterfactuals = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/counterfactual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instance: [0.1, 0.2, 0.3, 0.4, 0.5]
        }),
      });

      const result = await response.json();
      setCounterfactuals(result);
    } catch (error) {
      console.error('Failed to generate counterfactuals:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const setupTransformerModel = async (modelName: string) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/transformer/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName
        }),
      });

      const result = await response.json();
      setTransformerSetup(result);
      await fetchEngineStatus(); // Refresh status
    } catch (error) {
      console.error('Failed to setup transformer model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateAttentionVisualization = async () => {
    if (!inputText.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/attention/visualize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          layer: -1,
          head: -1
        }),
      });

      const result = await response.json();
      setAttentionResult(result);
    } catch (error) {
      console.error('Failed to generate attention visualization:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateMultiHeadAttention = async () => {
    if (!inputText.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/attention/multi-head', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          layer: -1
        }),
      });

      const result = await response.json();
      setAttentionResult(result);
    } catch (error) {
      console.error('Failed to generate multi-head attention:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateLayerWiseAttention = async () => {
    if (!inputText.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('/api/explainx/attention/layer-wise', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText
        }),
      });

      const result = await response.json();
      setAttentionResult(result);
    } catch (error) {
      console.error('Failed to generate layer-wise attention:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const renderFeatureImportanceChart = () => {
    if (!featureImportance?.data?.ranking) return null;

    const chartData = featureImportance.data.ranking.slice(0, 10).map(([feature, score]: [string, number]) => ({
      feature,
      importance: score
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="importance" fill="#3b82f6" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderBiasMetrics = () => {
    if (!biasResult?.bias_metrics) return null;

    return (
      <div className="space-y-4">
        {Object.entries(biasResult.bias_metrics).map(([feature, metrics]) => (
          <Card key={feature}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {metrics.bias_detected ? (
                  <AlertTriangle className="h-5 w-5 text-red-500" />
                ) : (
                  <CheckCircle className="h-5 w-5 text-green-500" />
                )}
                {feature}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div>
                  <span className="font-medium">Demographic Parity Difference: </span>
                  <Badge variant={metrics.demographic_parity_difference > 0.1 ? "destructive" : "default"}>
                    {metrics.demographic_parity_difference.toFixed(3)}
                  </Badge>
                </div>
                <div>
                  <span className="font-medium">Group Rates: </span>
                  {Object.entries(metrics.group_rates).map(([group, rate]) => (
                    <Badge key={group} variant="outline" className="ml-1">
                      {group}: {rate.toFixed(3)}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">ExplainX Dashboard</h1>
          <p className="text-muted-foreground">
            Comprehensive model interpretability and explainable AI
          </p>
        </div>
        <Button onClick={fetchEngineStatus} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh Status
        </Button>
      </div>

      {/* Engine Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Engine Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          {engineStatus ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <span className="font-medium">Status: </span>
                <Badge variant={engineStatus.status === 'active' ? 'default' : 'destructive'}>
                  {engineStatus.status}
                </Badge>
              </div>
              <div>
                <span className="font-medium">SHAP Ready: </span>
                <Badge variant={engineStatus.shap_ready ? 'default' : 'secondary'}>
                  {engineStatus.shap_ready ? 'Yes' : 'No'}
                </Badge>
              </div>
              <div>
                <span className="font-medium">LIME Ready: </span>
                <Badge variant={engineStatus.lime_ready ? 'default' : 'secondary'}>
                  {engineStatus.lime_ready ? 'Yes' : 'No'}
                </Badge>
              </div>
              <div>
                <span className="font-medium">Model Loaded: </span>
                <Badge variant={engineStatus.model_loaded ? 'default' : 'secondary'}>
                  {engineStatus.model_loaded ? 'Yes' : 'No'}
                </Badge>
              </div>
            </div>
          ) : (
            <div>Loading engine status...</div>
          )}
        </CardContent>
      </Card>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="shap">SHAP</TabsTrigger>
          <TabsTrigger value="lime">LIME</TabsTrigger>
          <TabsTrigger value="attention">Attention</TabsTrigger>
          <TabsTrigger value="bias">Bias Detection</TabsTrigger>
          <TabsTrigger value="counterfactual">Counterfactuals</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* File Upload */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Upload Model & Data
                </CardTitle>
                <CardDescription>
                  Upload your trained model and training data for explanation
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Model File</label>
                  <input
                    type="file"
                    accept=".pkl,.joblib"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], 'model')}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Training Data</label>
                  <input
                    type="file"
                    accept=".csv,.xlsx"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], 'data')}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Feature Importance */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Feature Importance
                </CardTitle>
                <CardDescription>
                  Analyze global feature importance using SHAP
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button 
                  onClick={analyzeFeatureImportance} 
                  disabled={isLoading || !engineStatus?.shap_ready}
                  className="w-full"
                >
                  {isLoading ? 'Analyzing...' : 'Analyze Feature Importance'}
                </Button>
                {featureImportance && (
                  <div className="mt-4">
                    {renderFeatureImportanceChart()}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>
                Generate explanations and analyze your model
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Button 
                  onClick={generateShapExplanations}
                  disabled={isLoading || !engineStatus?.shap_ready}
                  variant="outline"
                >
                  Generate SHAP
                </Button>
                <Button 
                  onClick={generateLimeExplanations}
                  disabled={isLoading || !engineStatus?.lime_ready}
                  variant="outline"
                >
                  Generate LIME
                </Button>
                <Button 
                  onClick={detectBias}
                  disabled={isLoading || !engineStatus?.model_loaded}
                  variant="outline"
                >
                  Detect Bias
                </Button>
                <Button 
                  onClick={generateCounterfactuals}
                  disabled={isLoading || !engineStatus?.model_loaded}
                  variant="outline"
                >
                  Counterfactuals
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="shap" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                SHAP Explanations
              </CardTitle>
              <CardDescription>
                SHapley Additive exPlanations for model interpretability
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={generateShapExplanations}
                disabled={isLoading || !engineStatus?.shap_ready}
                className="mb-4"
              >
                {isLoading ? 'Generating...' : 'Generate SHAP Explanations'}
              </Button>
              
              {shapResult && shapResult.status === 'success' && (
                <div className="space-y-4">
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                      SHAP values show how each feature contributes to the model's prediction.
                      Positive values push the prediction higher, negative values push it lower.
                    </AlertDescription>
                  </Alert>
                  
                  {shapResult.visualization && (
                    <div className="border rounded-lg p-4">
                      <h3 className="font-medium mb-2">Visualization</h3>
                      <div dangerouslySetInnerHTML={{ 
                        __html: `<div id="shap-plot">${shapResult.visualization.figure}</div>` 
                      }} />
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="lime" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                LIME Explanations
              </CardTitle>
              <CardDescription>
                Local Interpretable Model-agnostic Explanations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={generateLimeExplanations}
                disabled={isLoading || !engineStatus?.lime_ready}
                className="mb-4"
              >
                {isLoading ? 'Generating...' : 'Generate LIME Explanations'}
              </Button>
              
              {limeResult && limeResult.status === 'success' && (
                <div className="space-y-4">
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                      LIME explains individual predictions by learning a local interpretable model.
                    </AlertDescription>
                  </Alert>
                  
                  {limeResult.explanations && (
                    <div className="space-y-2">
                      <h3 className="font-medium">Feature Contributions</h3>
                      {limeResult.explanations.map(([feature, contribution], index) => (
                        <div key={index} className="flex justify-between items-center p-2 border rounded">
                          <span>{feature}</span>
                          <Badge variant={contribution > 0 ? 'default' : 'destructive'}>
                            {contribution.toFixed(3)}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="bias" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                Bias Detection
              </CardTitle>
              <CardDescription>
                Detect potential bias in model predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={detectBias}
                disabled={isLoading || !engineStatus?.model_loaded}
                className="mb-4"
              >
                {isLoading ? 'Analyzing...' : 'Detect Bias'}
              </Button>
              
              {biasResult && biasResult.status === 'success' && (
                <div className="space-y-4">
                  <Alert variant={biasResult.overall_bias_detected ? 'destructive' : 'default'}>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      {biasResult.overall_bias_detected 
                        ? 'Potential bias detected in model predictions'
                        : 'No significant bias detected'
                      }
                    </AlertDescription>
                  </Alert>
                  
                  {renderBiasMetrics()}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="attention" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Attention Visualization
              </CardTitle>
              <CardDescription>
                Visualize attention patterns in transformer models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Transformer Model Setup */}
              <div className="space-y-2">
                <label className="block text-sm font-medium">Transformer Model</label>
                <div className="flex gap-2">
                  <select 
                    className="flex-1 p-2 border rounded"
                    onChange={(e) => e.target.value && setupTransformerModel(e.target.value)}
                  >
                    <option value="">Select a model...</option>
                    <option value="distilbert-base-uncased">DistilBERT Base</option>
                    <option value="bert-base-uncased">BERT Base</option>
                    <option value="roberta-base">RoBERTa Base</option>
                  </select>
                  <Badge variant={engineStatus?.transformer_ready ? 'default' : 'secondary'}>
                    {engineStatus?.transformer_ready ? 'Ready' : 'Not Setup'}
                  </Badge>
                </div>
              </div>

              {/* Text Input */}
              <div className="space-y-2">
                <label className="block text-sm font-medium">Input Text</label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Enter text to analyze attention patterns..."
                  className="w-full p-2 border rounded h-24 resize-none"
                />
              </div>

              {/* Action Buttons */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <Button 
                  onClick={generateAttentionVisualization}
                  disabled={isLoading || !engineStatus?.transformer_ready || !inputText.trim()}
                  variant="outline"
                >
                  Single Head
                </Button>
                <Button 
                  onClick={generateMultiHeadAttention}
                  disabled={isLoading || !engineStatus?.transformer_ready || !inputText.trim()}
                  variant="outline"
                >
                  Multi-Head
                </Button>
                <Button 
                  onClick={generateLayerWiseAttention}
                  disabled={isLoading || !engineStatus?.transformer_ready || !inputText.trim()}
                  variant="outline"
                >
                  Layer-Wise
                </Button>
              </div>

              {/* Results Display */}
              {attentionResult && attentionResult.status === 'success' && (
                <div className="space-y-4">
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                      Attention weights show how much each token attends to other tokens.
                      Darker colors indicate stronger attention.
                    </AlertDescription>
                  </Alert>

                  {/* Tokens Display */}
                  {attentionResult.tokens && (
                    <div className="space-y-2">
                      <h3 className="font-medium">Tokens</h3>
                      <div className="flex flex-wrap gap-1">
                        {attentionResult.tokens.map((token: string, index: number) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {token}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Visualization */}
                  {attentionResult.visualization && (
                    <div className="border rounded-lg p-4">
                      <h3 className="font-medium mb-2">Attention Heatmap</h3>
                      <div dangerouslySetInnerHTML={{ 
                        __html: `<div id="attention-plot">${attentionResult.visualization.figure}</div>` 
                      }} />
                    </div>
                  )}

                  {/* Multi-head Analysis */}
                  {attentionResult.head_analyses && (
                    <div className="space-y-2">
                      <h3 className="font-medium">Head Analysis</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {attentionResult.head_analyses.slice(0, 4).map((head: any, index: number) => (
                          <div key={index} className="p-3 border rounded">
                            <div className="font-medium">Head {head.head_index}</div>
                            <div className="text-sm space-y-1">
                              <div>Entropy: {head.attention_entropy?.toFixed(3)}</div>
                              <div>Focus: {head.focus_metrics?.attention_concentration?.toFixed(3)}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Layer Analysis */}
                  {attentionResult.layer_analyses && (
                    <div className="space-y-2">
                      <h3 className="font-medium">Layer Analysis</h3>
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {attentionResult.layer_analyses.map((layer: any, index: number) => (
                          <div key={index} className="p-2 border rounded text-sm">
                            <div className="font-medium">Layer {layer.layer_index}</div>
                            <div className="text-muted-foreground">
                              Entropy: {layer.average_attention_entropy?.toFixed(3)} | 
                              Diversity: {layer.attention_diversity?.toFixed(3)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Statistics */}
                  {attentionResult.statistics && (
                    <div className="space-y-2">
                      <h3 className="font-medium">Statistics</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="font-medium">Layers</div>
                          <div>{attentionResult.statistics.num_layers}</div>
                        </div>
                        <div>
                          <div className="font-medium">Heads</div>
                          <div>{attentionResult.statistics.num_heads}</div>
                        </div>
                        <div>
                          <div className="font-medium">Sequence Length</div>
                          <div>{attentionResult.statistics.sequence_length}</div>
                        </div>
                        <div>
                          <div className="font-medium">Total Heads</div>
                          <div>{attentionResult.total_heads}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Transformers Not Available Warning */}
              {!engineStatus?.transformers_available && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Transformer attention visualization requires the transformers library. 
                    Install with: pip install transformers torch
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="counterfactual" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <RefreshCw className="h-5 w-5" />
                Counterfactual Explanations
              </CardTitle>
              <CardDescription>
                What would need to change for a different prediction?
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={generateCounterfactuals}
                disabled={isLoading || !engineStatus?.model_loaded}
                className="mb-4"
              >
                {isLoading ? 'Generating...' : 'Generate Counterfactuals'}
              </Button>
              
              {counterfactuals && counterfactuals.status === 'success' && (
                <div className="space-y-4">
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                      Counterfactuals show what changes would lead to different predictions.
                    </AlertDescription>
                  </Alert>
                  
                  {counterfactuals.counterfactuals && (
                    <div className="space-y-2">
                      <h3 className="font-medium">Potential Changes</h3>
                      {counterfactuals.counterfactuals.map((cf, index) => (
                        <div key={index} className="p-3 border rounded-lg">
                          <div className="font-medium">{cf.feature}</div>
                          <div className="text-sm text-muted-foreground">
                            Change from {cf.original_value} to {cf.modified_value}
                          </div>
                          <Badge variant="outline">
                            Impact: {cf.change.toFixed(3)}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {isLoading && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="p-6">
            <div className="flex items-center space-x-4">
              <RefreshCw className="h-6 w-6 animate-spin" />
              <div>Processing...</div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default ExplainabilityDashboard;